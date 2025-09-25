import os
import argparse
import json
import sys
import torch
import numpy as np
import random

sys.path.insert(0, "Snippext_public")

from ditto_light.dataset import EmbedDittoDataset
from ditto_light.summarize import Summarizer
from ditto_light.knowledge import *
from ditto_light.embeddings_utils import train_embeddings

def get_singletons(data, domains, ds_idx):
    new_data = {}

    # get the singletons from each domain
    for dom in domains:
        new_data[dom] = []
        for i in range(len(data[dom][ds_idx])):
            line = data[dom][ds_idx][i]
            left, right, _ = line.split("\t")
            new_data[dom].append(left+'\n')

            # account for new line at end of file
            if i == len(data[dom][ds_idx]) - 1:
                new_data[dom].append(right)
            else:
                new_data[dom].append(right+'\n')

    return new_data

def generate_embeddings(data, domains, embed_path, ds='train', tmp_file_path="updated_data", lm='distilbert', max_len=64, batch_size=32, fp16=True, tokenizer=None, shared_model=None):
    # create embed path and temp file paths if needed
    os.makedirs(embed_path, exist_ok=True)
    os.makedirs(tmp_file_path, exist_ok=True)

    # get the singletons
    ds_idx = ['train', 'val', 'test'].index(ds)
    singleton_data = get_singletons(data, domains, ds_idx)

    # save data to file
    for dom in domains:
        trainset = os.path.join(tmp_file_path, dom + "_train.txt")
        with open(trainset, "w") as f:
            f.writelines(singleton_data[dom])
        validset = os.path.join(tmp_file_path, dom + "_val.txt")
        with open(validset, "w") as f:
            f.writelines(data[dom][1])
        testset = os.path.join(tmp_file_path, dom + "_test.txt")
        with open(testset, "w") as f:
            f.writelines(data[dom][2])

        # load task configuration
        #configs = json.load(open('configs.json'))
        #configs = {conf['name'] : conf for conf in configs}
        #config = configs[dom]

        #trainset = config['trainset']
        #validset = config['validset']
        #testset = config['testset']

        # load train/dev/test sets
        train_dataset = EmbedDittoDataset(trainset,
                                   lm=lm,
                                   max_len=max_len,
                                   size=None,
                                   tokenizer=tokenizer)

        # train and evaluate the model
        train_embeddings(train_dataset, dom, embed_path, batch_size, lm, fp16, shared_model=shared_model)
