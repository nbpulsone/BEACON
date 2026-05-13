import torch
import os 
import heapq
import random

from torch.utils import data
from transformers import AutoTokenizer

from .augment import Augmenter
from domain_agnostic.exceptions import ModelNotFoundError
from domain_agnostic.ditto import DittoModel
from apex import amp

# map lm name to huggingface's pre-trained model names
lm_mp = {'roberta': 'roberta-base',
         'distilbert': 'distilbert-base-uncased'}

def get_tokenizer(lm):
    if lm in lm_mp:
        return AutoTokenizer.from_pretrained(lm_mp[lm])
    else:
        return AutoTokenizer.from_pretrained(lm)


class DittoDataset(data.Dataset):
    """EM dataset"""

    def __init__(self,
                 path,
                 max_len=256,
                 size=None,
                 lm='roberta',
                 da=None,
                 tgt_path=None,
                 ckpt_path=None,
                 sample_method=None,
                 hp=None):
        self.tokenizer = get_tokenizer(lm)
        self.pairs = []
        self.labels = []
        self.max_len = max_len
        self.size = size
        self.hp = hp
        
        # lines associated with this dataset
        if isinstance(path, list):
            lines = path
        else:
            lines = list(open(path))

        # lines associated with target dataset
        if tgt_path is not None and ckpt_path is not None:
            if isinstance(tgt_path, list):
                tgt_lines = tgt_path
            else:
                tgt_lines = list(open(tgt_path))

            # use distribution aware method to subsample the data
            # TODO: Maybe try reweighting here instead
            lines = self.distribution_aware_samlping(lines, tgt_lines, ckpt_path, hp)

        for line in lines:
            s1, s2, label = line.strip().split('\t')
            self.pairs.append((s1, s2))
            self.labels.append(int(label))

        self.pairs = self.pairs[:size]
        self.labels = self.labels[:size]
        self.da = da
        if da is not None:
            self.augmenter = Augmenter()
        else:
            self.augmenter = None


    def __len__(self):
        """Return the size of the dataset."""
        return len(self.pairs)

    def __getitem__(self, idx):
        """Return a tokenized item of the dataset.

        Args:
            idx (int): the index of the item

        Returns:
            List of int: token ID's of the two entities
            List of int: token ID's of the two entities augmented (if da is set)
            int: the label of the pair (0: unmatch, 1: match)
        """
        left = self.pairs[idx][0]
        right = self.pairs[idx][1]

        # left + right
        x = self.tokenizer.encode(text=left,
                                  text_pair=right,
                                  max_length=self.max_len,
                                  truncation=True)

        # augment if da is set
        if self.da is not None:
            combined = self.augmenter.augment_sent(left + ' [SEP] ' + right, self.da)
            left, right = combined.split(' [SEP] ')
            x_aug = self.tokenizer.encode(text=left,
                                      text_pair=right,
                                      max_length=self.max_len,
                                      truncation=True)
            return x, x_aug, self.labels[idx]
        else:
            return x, self.labels[idx]

    # TODO/TEST: perform distribution aware sampling method
    def distribution_aware_samlping(self, lines, tgt_lines, ckpt_path, hp):
        # load model checkpoint
        if not os.path.exists(ckpt_path):
            raise ModelNotFoundError(ckpt_path)
        if hp.use_gpu:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            device = 'cpu'
        model = DittoModel(device=device, lm=hp.lm)
        saved_state = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
        model.load_state_dict(saved_state['model'])
        model = model.to(device)
        if hp.fp16 and 'cuda' in device:
            model = amp.initialize(model, opt_level='O2')

        # create embeddings for dataset and target
        print(f"Generating train embeddings...\n")
        embeds = self.encode_lines_to_embeddings(lines, model)
        print(f"...done!\n")
        print(f"Generating target/validation embeddings...\n")
        tgt_embeds = self.encode_lines_to_embeddings(tgt_lines, model)
        print(f"...done!\n")

        # invoke distribution fitting method and return resulting lines
        if hp.sample_method == 'tvdf':
            selected_indices = get_tvdf_samples(embeds, tgt_embeds, device=device) 
        elif hp.sample_method == 'cs_centroid':
            selected_indices = get_closest_centroid_samples(embeds, tgt_embeds, device=device)
            # debug Nearest to Centroid
            selected = set(selected_indices)
            orig_labels = [int(line.strip().split('\t')[-1]) for line in lines]
            kept_labels = [orig_labels[i] for i in selected]
            print("original positive rate:", sum(orig_labels) / len(orig_labels))
            print("kept positive rate:", sum(kept_labels) / len(kept_labels))
        elif hp.sample_method == 'cs':
            selected_indices = get_closest_samples(embeds, tgt_embeds, device=device)
        else:
            selected_indices = select_random_samples(embeds, tgt_embeds, device=device) 
        return [lines[i] for i in selected_indices]

    @staticmethod
    def pad(batch):
        """Merge a list of dataset items into a train/test batch
        Args:
            batch (list of tuple): a list of dataset items

        Returns:
            LongTensor: x1 of shape (batch_size, seq_len)
            LongTensor: x2 of shape (batch_size, seq_len).
                        Elements of x1 and x2 are padded to the same length
            LongTensor: a batch of labels, (batch_size,)
        """
        if len(batch[0]) == 3:
            x1, x2, y = zip(*batch)

            maxlen = max([len(x) for x in x1+x2])
            x1 = [xi + [0]*(maxlen - len(xi)) for xi in x1]
            x2 = [xi + [0]*(maxlen - len(xi)) for xi in x2]
            return torch.LongTensor(x1), \
                   torch.LongTensor(x2), \
                   torch.LongTensor(y)
        else:
            x12, y = zip(*batch)
            maxlen = max([len(x) for x in x12])
            x12 = [xi + [0]*(maxlen - len(xi)) for xi in x12]
            return torch.LongTensor(x12), \
                   torch.LongTensor(y)

    """
    Generate embeddings using the give model state
    """
    def encode_lines_to_embeddings(self, lines, model, batch_size=64):
        model.eval()
        all_embeds = []

        # tokenize all lines first
        tokenized = []
        for line in lines:
            s1, s2, _ = line.strip().split('\t')
            x = self.tokenizer.encode(
                text=s1,
                text_pair=s2,
                max_length=self.max_len,
                truncation=True
            )
            tokenized.append(x)

        with torch.no_grad():
            for i in range(0, len(tokenized), batch_size):
                batch = tokenized[i:i+batch_size]
                maxlen = max(len(x) for x in batch)
                batch = [x + [0] * (maxlen - len(x)) for x in batch]
                x_tensor = torch.LongTensor(batch).to(model.device)

                embeds = model(x_tensor, embed=True)   # shape: (batch, hidden_dim)
                all_embeds.append(embeds.cpu())

        return torch.cat(all_embeds, dim=0)


"""calculate the distance between embeddings using cosine distance"""
def cosine_distance(x, y, dim=1, eps=1e-8):
    return 1.0 - torch.nn.functional.cosine_similarity(x, y, dim=dim, eps=eps)

"""
Do TVDF for single domain, unbudgeted scenario
"""
def get_tvdf_samples(embeds, tgt_embeds, frac=0.7, device='cpu'):
    # ensure embedding data is on GPU (and normalized)
    embed_data = torch.nn.functional.normalize(
        embeds.to(device=device, dtype=torch.float32), p=2, dim=1
    )
    target_embed_data = torch.nn.functional.normalize(
        tgt_embeds.to(device=device, dtype=torch.float32), p=2, dim=1
    )

    # calculate target centroid --> what we want to fit to in our downsampling
    target_mean = target_embed_data.mean(dim=0)
    target_mean = torch.nn.functional.normalize(target_mean.unsqueeze(0), p=2, dim=1).squeeze(0)

    n_pairs = embed_data.size(0)
    n_keep = max(1, int(n_pairs * frac))

    # tracks our selected samples
    selected = list(range(n_pairs))

    while len(selected) > n_keep:
        best_remove_pos = None # best removal candidate
        best_dist = None # distance between centroid w/o best removal candidate and target

        # test each of our current selected
        for pos in range(len(selected)):
            trial_idx = selected[:pos] + selected[pos+1:]
            trial_mean = embed_data[trial_idx].mean(dim=0)
            trial_mean = torch.nn.functional.normalize(trial_mean.unsqueeze(0), p=2, dim=1).squeeze(0)
            dist = cosine_distance(
                trial_mean.unsqueeze(0),
                target_mean.unsqueeze(0),
                dim=1
            ).item()

            # if it moves the centroid the current furthest away, track it
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_remove_pos = pos

        # remove the current best removal candidate
        selected.pop(best_remove_pos)

    return selected

"""
Nearest to Centroid: Select samples that are closest to the target centroid
"""
def get_closest_centroid_samples(embeds, tgt_embeds, frac=0.7, device='cpu'):
    # ensure embedding data is on GPU (and normalized)
    embed_data = torch.nn.functional.normalize(
        embeds.to(device=device, dtype=torch.float32), p=2, dim=1
    )
    target_embed_data = torch.nn.functional.normalize(
        tgt_embeds.to(device=device, dtype=torch.float32), p=2, dim=1
    )

    # calculate target centroid --> what we want to fit to in our downsampling
    target_mean = target_embed_data.mean(dim=0)
    target_mean = torch.nn.functional.normalize(target_mean.unsqueeze(0), p=2, dim=1).squeeze(0)

    # calculate the distances between the embeddings and the target centroid
    dists = cosine_distance(
        embed_data,
        target_mean.unsqueeze(0).expand_as(embed_data),
        dim=1
    )

    n = embed_data.size(0)
    n_keep = max(1, int(n * frac))

    # debug Nearest to centroid
    print("~~~NC diagnostics~~~")
    print("train size:", embeds.size(0))
    print("target size:", tgt_embeds.size(0))
    print("n_keep:", max(1, int(embeds.size(0) * frac)))
    print("distance min/mean/max:",
        dists.min().item(),
        dists.mean().item(),
        dists.max().item())

    # keep the samples that are closest to the target centroid
    keep = torch.topk(dists, k=n_keep, largest=False).indices
    return keep.tolist()

"""
Select samples that are closest to *any* point in the target embedding space (not just centroid)
"""
def get_closest_samples(embeds, tgt_embeds, frac=0.7, device='cpu', chunk_size=1024):
    # ensure embedding data is on GPU (and normalized)
    embed_data = torch.nn.functional.normalize(
        embeds.to(device=device, dtype=torch.float32), p=2, dim=1
    )
    target_embed_data = torch.nn.functional.normalize(
        tgt_embeds.to(device=device, dtype=torch.float32), p=2, dim=1
    )

    n = embed_data.size(0)
    n_keep = max(1, int(n * frac))

    closest_dists = []
    # run cdist to calculate distances between embed <-> target embeds in chunks
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunk = embed_data[start:end]

        sim_mat = torch.matmul(chunk, target_embed_data.T)   # cosine similarity if normalized
        dmat = 1.0 - sim_mat                           # cosine distance
        chunk_min = dmat.min(dim=1).values
        closest_dists.append(chunk_min)

    # return the samples with the closest distances
    closest_dists = torch.cat(closest_dists, dim=0)
    keep = torch.topk(closest_dists, k=n_keep, largest=False).indices
    return keep.tolist()

"""
Baseline: select the samples randomly
"""
def select_random_samples(embeds, tgt_embeds, frac=0.7, device='cpu'):
    n = embeds.size(0)
    n_keep = max(1, int(n * frac))
    return random.sample(range(n), n_keep)
