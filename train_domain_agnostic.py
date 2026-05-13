import os
import argparse
import json
import sys
import torch
import numpy as np
import random

sys.path.insert(0, "Snippext_public")

from domain_agnostic.dataset import DittoDataset
from domain_agnostic.summarize import Summarizer
from domain_agnostic.knowledge import *
from domain_agnostic.ditto import train

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="Structured/Beer")
    parser.add_argument("--run_id", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--n_epochs", type=int, default=20)
    parser.add_argument("--finetuning", dest="finetuning", action="store_true")
    parser.add_argument("--save_model", dest="save_model", action="store_true")
    parser.add_argument("--logdir", type=str, default="checkpoints/")
    parser.add_argument("--lm", type=str, default='distilbert')
    parser.add_argument("--fp16", dest="fp16", action="store_true")
    parser.add_argument("--da", type=str, default=None)
    parser.add_argument("--alpha_aug", type=float, default=0.8)
    parser.add_argument("--dk", type=str, default=None)
    parser.add_argument("--summarize", dest="summarize", action="store_true")
    parser.add_argument("--size", type=int, default=None)
    parser.add_argument("--use_gpu", dest="use_gpu", action="store_true")
    parser.add_argument("--outfile", type=str, default=None)
    parser.add_argument("--sample_method", type=str, default='random')

    hp = parser.parse_args()

    # set seeds
    seed = hp.run_id
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # only a single task for baseline
    task = hp.task

    # create the tag of the run
    run_tag = '%s_lm=%s_da=%s_dk=%s_su=%s_size=%s_id=%d' % (task, hp.lm, hp.da,
            hp.dk, hp.summarize, str(hp.size), hp.run_id)
    run_tag = run_tag.replace('/', '_')

    # load task configuration
    configs = json.load(open('configs.json'))
    configs = {conf['name'] : conf for conf in configs}
    config = configs[task]

    trainset = config['trainset']
    validset = config['validset']
    testset = config['testset']

    # summarize the sequences up to the max sequence length
    if hp.summarize:
        summarizer = Summarizer(config, lm=hp.lm)
        trainset = summarizer.transform_file(trainset, max_len=hp.max_len)
        validset = summarizer.transform_file(validset, max_len=hp.max_len)
        testset = summarizer.transform_file(testset, max_len=hp.max_len)

    if hp.dk is not None:
        if hp.dk == 'product':
            injector = ProductDKInjector(config, hp.dk)
        else:
            injector = GeneralDKInjector(config, hp.dk)

        trainset = injector.transform_file(trainset)
        validset = injector.transform_file(validset)
        testset = injector.transform_file(testset)

    # load train/dev/test sets
    train_dataset = DittoDataset(trainset,
                                   lm=hp.lm,
                                   max_len=hp.max_len,
                                   size=hp.size,
                                   da=hp.da,
                                   hp=hp)
    valid_dataset = DittoDataset(validset, lm=hp.lm)
    test_dataset = DittoDataset(testset, lm=hp.lm)

    # create checkpoint path for this model/run
    hp.ckpt_dir = os.path.join(hp.logdir, hp.task, hp.sample_method, f"run_{hp.run_id}")
    hp.base_ckpt_path = os.path.join(hp.ckpt_dir, "base_model.pt")
    hp.df_ckpt_path = os.path.join(hp.ckpt_dir, "df_model.pt")

    # train and evaluate the model
    print(f"\n\n~~~TRAINING BASE MODEL~~~\n\n")
    hp.ckpt_path = hp.base_ckpt_path
    base_run_tag = run_tag + "_base"
    original_f1 = train(train_dataset,
          valid_dataset,
          test_dataset,
          base_run_tag, hp)
    
    # regenerate the training set to fit to the target distribution and retrain
    # use the trained model's checkpoint to generate informative embeddings
    train_dataset = DittoDataset(trainset,
                                   lm=hp.lm,
                                   max_len=hp.max_len,
                                   size=hp.size,
                                   da=hp.da,
                                   tgt_path=validset,
                                   ckpt_path = hp.base_ckpt_path,
                                   hp=hp)
    print(f"\n\n~~~TRAINING DF MODEL~~~\n\n")
    hp.ckpt_path = hp.df_ckpt_path
    df_run_tag = run_tag + "_df"
    dist_aware_f1 = train(train_dataset,
          valid_dataset,
          test_dataset,
          df_run_tag, hp)
    
    # compare results in output file
    if hp.outfile is not None:
        with open(hp.outfile, 'a') as f:
            f.write(f"~~~ {hp.sample_method} --> {hp.task} ~~~\n")
            f.write(f'Original F1: {original_f1}\n')
            f.write(f'Distribution-Aware F1: {dist_aware_f1}\n\n')
