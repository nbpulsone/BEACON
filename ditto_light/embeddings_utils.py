import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import sklearn.metrics as metrics
import argparse
from torch.utils import data
from transformers import AutoModel, AdamW
from apex import amp
import copy

lm_mp = {'roberta': 'roberta-base',
         'distilbert': 'distilbert-base-uncased'}

class EmbedDittoModel(nn.Module):
    """A baseline model for EM."""

    def __init__(self, device='cuda', lm='roberta', alpha_aug=0.8, shared_model=None):
        super().__init__()
        if shared_model is not None: 
            self.bert = copy.deepcopy(shared_model) # TODO: optimize by removing deep copy here
        elif lm in lm_mp:
            self.bert = AutoModel.from_pretrained(lm_mp[lm])
        else:
            self.bert = AutoModel.from_pretrained(lm)

        self.device = device
        self.alpha_aug = alpha_aug

        # linear layer
        hidden_size = self.bert.config.hidden_size
        self.fc = torch.nn.Linear(hidden_size, 2)


    def forward(self, x1, x2=None):
        """Encode the left, right, and the concatenation of left+right.

        Args:
            x1 (LongTensor): a batch of ID's
            x2 (LongTensor, optional): a batch of ID's (augmented)

        Returns:
            Tensor: binary prediction
        """
        x1 = x1.to(self.device) # (batch_size, seq_len)
        if x2 is not None:
            # MixDA
            x2 = x2.to(self.device) # (batch_size, seq_len)
            enc = self.bert(torch.cat((x1, x2)))[0][:, 0, :]
            batch_size = len(x1)
            enc1 = enc[:batch_size] # (batch_size, emb_size)
            enc2 = enc[batch_size:] # (batch_size, emb_size)

            aug_lam = np.random.beta(self.alpha_aug, self.alpha_aug)
            enc = enc1 * aug_lam + enc2 * (1.0 - aug_lam)
        else:
            enc = self.bert(x1)[0][:, 0, :]
        return enc # TEMP
        #return self.fc(enc) # .squeeze() # .sigmoid()

def embed_evaluate(model, iterator, task, embed_path, threshold=None):
    """Evaluate a model on a validation/test dataset

    Args:
        model (DMModel): the EM model
        iterator (Iterator): the valid/test dataset iterator
        threshold (float, optional): the threshold on the 0-class

    Returns:
        float: the F1 score
        float (optional): if threshold is not provided, the threshold
            value that gives the optimal F1
    """
    all_p = []
    all_y = []
    all_probs = []
    avg_vectors = []
    all_vectors = torch.tensor([])
    total_sz = 0
    with torch.no_grad():
        for batch in iterator:
            x, y = batch
            logits = model(x)
            
            # TEMP: append avg vector to list
            sz = logits.shape[0]
            #print(f"\n\nLOGITS: {[torch.mean(logits, axis = 0).shape, sz]}\n\n")
            avg_vectors.append([torch.mean(logits, axis=0), sz])
            all_vectors = torch.cat((all_vectors, logits.cpu()), 0)
            #print(f"\n\nLOGITS: {logits}\n\n")
            total_sz += sz

            probs = logits.softmax(dim=1)[:, 1]
            all_probs += probs.cpu().numpy().tolist()
            all_y += y.cpu().numpy().tolist()

    # save the vectors
    avg_vec = torch.zeros(768)
    for x in avg_vectors:
      avg_vec += ((float(x[1])/total_sz) * x[0].cpu())
    torch.save(avg_vec, os.path.join(embed_path, task + ".pt"))
    torch.save(all_vectors, os.path.join(embed_path, "all_" + task + ".pt"))
    print(f"{task} VECTORS GENERATED: {all_vectors.shape}")
    return 


def train_embeddings(trainset, task, embed_path, batch_size=32, lm='distilbert', fp16=True, shared_model=None):
    """Train and evaluate the model

    Args:
        trainset (DittoDataset): the training set
        validset (DittoDataset): the validation set
        testset (DittoDataset): the test set
        run_tag (str): the tag of the run
        hp (Namespace): Hyper-parameters (e.g., batch_size,
                        learning rate, fp16)

    Returns:
        None
    """
    padder = trainset.pad
    # create the DataLoaders
    train_iter = data.DataLoader(dataset=trainset,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 num_workers=0,
                                 collate_fn=padder)

    # initialize model, optimizer, and LR scheduler
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = EmbedDittoModel(device=device,
                       lm=lm, shared_model=shared_model)
    model = model.cuda()
    optimizer = AdamW(model.parameters())

    if fp16:
        model, optimizer = amp.initialize(model, optimizer, opt_level='O2')

    # TEMP: save training vectors
    print("COUNTING TRAIN VECTORS...")
    embed_evaluate(model, train_iter, task, embed_path)
    print("...DONE!")
