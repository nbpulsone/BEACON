import torch

from torch.utils import data
from transformers import AutoTokenizer

from .augment import Augmenter

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
                 singleton=False,
                 tokenizer=None):
        if tokenizer is None:
            self.tokenizer = get_tokenizer(lm)
        else:
            self.tokenizer = tokenizer
        self.pairs = []
        self.labels = []
        self.max_len = max_len
        self.size = size
        self.singleton = singleton

        if isinstance(path, list):
            lines = path
        else:
            lines = open(path)

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
        elif self.singleton:
            # encode singletons as well 
            x_left = self.tokenizer.encode(text=left, max_length=self.max_len, truncation=True)
            x_right = self.tokenizer.encode(text=right, max_length=self.max_len, truncation=True)
            return x, x_left, x_right, self.labels[idx]
        else:
            return x, self.labels[idx]


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
        elif len(batch[0]) == 4:
            x12, x_left, x_right, y = zip(*batch)
            maxlen = max([len(x) for x in x12])
            x12 = [xi + [0]*(maxlen - len(xi)) for xi in x12]
            x_left = [xi + [0]*(maxlen - len(xi)) for xi in x_left]
            x_right = [xi + [0]*(maxlen - len(xi)) for xi in x_right]
            return torch.LongTensor(x12), \
                   torch.LongTensor(x_left), \
                   torch.LongTensor(x_right), \
                   torch.LongTensor(y)
        else:
            x12, y = zip(*batch)
            maxlen = max([len(x) for x in x12])
            x12 = [xi + [0]*(maxlen - len(xi)) for xi in x12]
            return torch.LongTensor(x12), \
                   torch.LongTensor(y)

class EmbedDittoDataset(data.Dataset):
    """EM dataset"""

    def __init__(self,
                 path,
                 max_len=256,
                 size=None,
                 lm='roberta',
                 tokenizer=None):
        if tokenizer is None:
            self.tokenizer = get_tokenizer(lm)
        else:
            self.tokenizer = tokenizer
        self.singletons = []
        self.labels = []
        self.max_len = max_len
        self.size = size

        if isinstance(path, list):
            lines = path
        else:
            lines = open(path)

        for line in lines:
            self.singletons.append(line.strip())
            self.labels.append(0)

        self.singletons = self.singletons[:size]
        self.labels = self.labels[:size]

    def __len__(self):
        """Return the size of the dataset."""
        return len(self.singletons)

    def __getitem__(self, idx):
        """Return a tokenized item of the dataset.

        Args:
            idx (int): the index of the item

        Returns:
            List of int: token ID's of the two entities
            List of int: token ID's of the two entities augmented (if da is set)
            int: the label of the pair (0: unmatch, 1: match)
        """
        # left + right
        return self.tokenizer.encode(text=self.singletons[idx],
                                  max_length=self.max_len,
                                  truncation=True), self.labels[idx]


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
