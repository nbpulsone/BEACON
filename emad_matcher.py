import torch
import torch.nn as nn
import os
import numpy as np
import random
import json
import jsonlines
import csv
import re
import time
import argparse
import sys
import sklearn
import traceback

from torch.utils import data
import torch.nn.functional as F
from tqdm import tqdm
from apex import amp
from scipy.special import softmax

from ditto_light.ditto import evaluate, DittoModel
from ditto_light.exceptions import ModelNotFoundError
from ditto_light.dataset import DittoDataset
from ditto_light.summarize import Summarizer
from ditto_light.knowledge import *
from ditto_light.process_budget import process_budget
from ensemble_evaluators import tune_threshold_ensemble, predict_ensemble
from transformers import AutoTokenizer, AutoModel

#TODO ADD PARALLEL SUPPORT W/ JOB_ID

# map lm name to huggingface's pre-trained model names
lm_mp = {'roberta': 'roberta-base',
         'distilbert': 'distilbert-base-uncased'}

def get_tokenizer(lm):
    if lm in lm_mp:
        return AutoTokenizer.from_pretrained(lm_mp[lm])
    else:
        return AutoTokenizer.from_pretrained(lm)


def set_seed(seed: int):
    """
    Helper function for reproducible behavior to set the seed in ``random``, ``numpy``, ``torch``
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_str(ent1, ent2, summarizer=None, max_len=256, dk_injector=None):
    """Serialize a pair of data entries

    Args:
        ent1 (Dictionary): the 1st data entry
        ent2 (Dictionary): the 2nd data entry
        summarizer (Summarizer, optional): the summarization module
        max_len (int, optional): the max sequence length
        dk_injector (DKInjector, optional): the domain-knowledge injector

    Returns:
        string: the serialized version
    """
    content = ''
    for ent in [ent1, ent2]:
        if isinstance(ent, str):
            content += ent
        else:
            for attr in ent.keys():
                content += 'COL %s VAL %s ' % (attr, ent[attr])
        content += '\t'

    content += '0'

    if summarizer is not None:
        content = summarizer.transform(content, max_len=max_len)

    new_ent1, new_ent2, _ = content.split('\t')
    if dk_injector is not None:
        new_ent1 = dk_injector.transform(new_ent1)
        new_ent2 = dk_injector.transform(new_ent2)

    return new_ent1 + '\t' + new_ent2 + '\t0'


def build_pooled_valid(config, domain_data, tasks, data_dir, N_MIN=100):
    """
    if the current domain's valid set is very small, build a pooled valid file for threshold tuning
    by concatenating all domains' validation lines
    """
    val_path = config['validset']
    try:
        n_val = sum(1 for _ in open(val_path))
        print(f"\nEmpty file: {val_path}\n")
    except FileNotFoundError:
        n_val = 0
        print(f"\nFile not found: {val_path}\n")

    if n_val >= N_MIN:
        return config, None  # use domain-specific valid as-is

    # build pooled validation file
    pooled_path = os.path.join(data_dir or ".", f"pooled_valid_{config['name']}.txt")
    with open(pooled_path, "w") as fout:
        for dom in tasks:
            fout.writelines(domain_data[dom][1])  # each domain's validation lines

    cfg_pooled = dict(config)
    cfg_pooled['validset'] = pooled_path
    return cfg_pooled, pooled_path


def classify(sentence_pairs, model,
             lm='distilbert',
             max_len=256,
             threshold=None,
             late_fusion_only=False,
             tokenizer=None):
    """Apply the MRPC model.

    Args:
        sentence_pairs (list of str): the sequence pairs
        model (MultiTaskNet): the model in pytorch
        max_len (int, optional): the max sequence length
        threshold (float, optional): the threshold of the 0's class

    Returns:
        list of float: the scores of the pairs
    """
    inputs = sentence_pairs
    # print('max_len =', max_len)
    dataset = DittoDataset(inputs,
                           max_len=max_len,
                           lm=lm,
                           singleton=late_fusion_only,
                           tokenizer=tokenizer)
    # print(dataset[0])
    iterator = data.DataLoader(dataset=dataset,
                               batch_size=len(dataset),
                               shuffle=False,
                               num_workers=0,
                               collate_fn=DittoDataset.pad)

    # prediction
    all_probs = []
    all_logits = []
    with torch.no_grad():
        # print('Classification')
        if late_fusion_only:
            for batch in iterator:
                x_pair, x_left, x_right, y = batch
                enc_left, enc_right = model.forward_only_singletons(x_left, x_right)
                enc_left = F.normalize(enc_left, dim=1)
                enc_right = F.normalize(enc_right, dim=1)
                cosine_sim = F.cosine_similarity(enc_left, enc_right)
                all_probs += cosine_sim.detach().cpu().numpy().tolist()
        else:
            for i, batch in enumerate(iterator):
                x, _ = batch
                logits = model(x)
                probs = logits.softmax(dim=1)[:, 1]
                all_probs += probs.cpu().numpy().tolist()
                all_logits += logits.cpu().numpy().tolist()

    if threshold is None:
        threshold = 0.5

    pred = [1 if p > threshold else 0 for p in all_probs]
    if late_fusion_only:
        #print(f"PRED: {pred}")
        #print(f"ALL PROBS: {all_probs}")
        return pred, all_probs
    else:
        return pred, all_logits


def predict(input_path, output_path, config,
            model,
            batch_size=1024,
            summarizer=None,
            lm='distilbert',
            max_len=256,
            dk_injector=None,
            threshold=None,
            late_fusion_only=False,
            tokenizer=None):
    """Run the model over the input file containing the candidate entry pairs

    Args:
        input_path (str): the input file path
        output_path (str): the output file path
        config (Dictionary): task configuration
        model (DittoModel): the model for prediction
        batch_size (int): the batch size
        summarizer (Summarizer, optional): the summarization module
        max_len (int, optional): the max sequence length
        dk_injector (DKInjector, optional): the domain-knowledge injector
        threshold (float, optional): the threshold of the 0's class

    Returns:
        None
    """
    pairs = []

    def process_batch(rows, pairs, writer, late_fusion_only, tokenizer=None):
        predictions, class_output = classify(pairs, model, lm=lm,
                                       max_len=max_len,
                                       threshold=threshold,
                                       late_fusion_only=late_fusion_only,
                                       tokenizer=tokenizer)
        # try:
        #     predictions, logits = classify(pairs, model, lm=lm,
        #                                    max_len=max_len,
        #                                    threshold=threshold)
        # except:
        #     # ignore the whole batch
        #     return
        if late_fusion_only:
            all_probs = class_output
            for row, pred, score in zip(rows, predictions, all_probs):
                output = {'left': row[0], 'right': row[1],
                        'match': pred,
                        'match_confidence': score}
                writer.write(output)
        else:
            scores = softmax(class_output, axis=1)
            for row, pred, score in zip(rows, predictions, scores):
                output = {'left': row[0], 'right': row[1],
                    'match': pred,
                    'match_confidence': score[int(pred)]}
                writer.write(output)

    # input_path can also be train/valid/test.txt
    # convert to jsonlines
    if '.txt' in input_path:
        with jsonlines.open(input_path + '.jsonl', mode='w') as writer:
            for line in open(input_path):
                writer.write(line.split('\t')[:2])
        input_path += '.jsonl'

    # batch processing
    start_time = time.time()
    with jsonlines.open(input_path) as reader,\
         jsonlines.open(output_path, mode='w') as writer:
        pairs = []
        rows = []
        for idx, row in tqdm(enumerate(reader)):
            pairs.append(to_str(row[0], row[1], summarizer, max_len, dk_injector))
            rows.append(row)
            if len(pairs) == batch_size:
                process_batch(rows, pairs, writer, late_fusion_only, tokenizer=tokenizer)
                pairs.clear()
                rows.clear()

        if len(pairs) > 0:
            process_batch(rows, pairs, writer, late_fusion_only, tokenizer=tokenizer)

    run_time = time.time() - start_time
    run_tag = '%s_lm=%s_dk=%s_su=%s' % (config['name'], lm, str(dk_injector != None), str(summarizer != None))
    os.system('echo %s %f >> log.txt' % (run_tag, run_time))


def tune_threshold(config, model, hp, tokenizer=None, data_dir=None):
    """Tune the prediction threshold for a given model on a validation set"""
    #validset = os.path.join(data_dir, config['validset']) if data_dir is not None else config['validset']
    validset = config['validset']

    # summarize the sequences up to the max sequence length
    set_seed(123)
    summarizer = injector = None
    if hp.summarize:
        summarizer = Summarizer(config, lm=hp.lm, tokenizer=tokenizer)
        validset = summarizer.transform_file(validset, max_len=hp.max_len, overwrite=True)

    if hp.dk is not None:
        if hp.dk == 'product':
            injector = ProductDKInjector(config, hp.dk)
        else:
            injector = GeneralDKInjector(config, hp.dk)

        validset = injector.transform_file(validset)

    # load dev sets
    valid_dataset = DittoDataset(validset,
                                 max_len=hp.max_len,
                                 lm=hp.lm,
                                 singleton=hp.late_fusion_only,
                                 tokenizer=tokenizer)

    # print(valid_dataset[0])

    valid_iter = data.DataLoader(dataset=valid_dataset,
                                 batch_size=64,
                                 shuffle=False,
                                 num_workers=0,
                                 collate_fn=DittoDataset.pad)

    # acc, prec, recall, f1, v_loss, th = eval_classifier(model, valid_iter,
    #                                                     get_threshold=True)
    f1, th = evaluate(model, valid_iter, threshold=None, late_fusion_only=hp.late_fusion_only)

    # verify F1
    set_seed(123)
    tmpfile = os.path.join(data_dir, "tmp.jsonl") if data_dir is not None else "tmp.jsonl"
    predict(validset, tmpfile, config, model,
            summarizer=summarizer,
            max_len=hp.max_len,
            lm=hp.lm,
            dk_injector=injector,
            threshold=th,
            late_fusion_only=hp.late_fusion_only,
            tokenizer=tokenizer)

    predicts = []
    with jsonlines.open(tmpfile, mode="r") as reader:
        for line in reader:
            predicts.append(int(line['match']))
    os.system(f"rm {tmpfile}")

    labels = []
    with open(validset) as fin:
        for line in fin:
            labels.append(int(line.split('\t')[-1]))

    real_f1 = sklearn.metrics.f1_score(labels, predicts)
    print("load_f1 =", f1)
    print("real_f1 =", real_f1)

    return th



def load_model(task, path, method, lm, use_gpu, fp16=True, pairwise_encoder=True, singleton_encoder=False, shared_model=None):
    """Load a model for a specific task.

    Args:
        task (str): the task name
        path (str): the path of the checkpoint directory
        lm (str): the language model
        use_gpu (boolean): whether to use gpu
        fp16 (boolean, optional): whether to use fp16

    Returns:
        Dictionary: the task config
        MultiTaskNet: the model
    """

    # load models
    checkpoint = os.path.join(path, task, f'{method}_model.pt')
    print(f"CP PATH: {checkpoint}")
    if not os.path.exists(checkpoint):
        raise ModelNotFoundError(checkpoint)

    configs = json.load(open('configs.json'))
    configs = {conf['name'] : conf for conf in configs}
    config = configs[task]

    if use_gpu:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = 'cpu'

    # determine network architecture from saved state dictionary
    saved_state = torch.load(checkpoint, map_location=lambda storage, loc: storage)
    arch = saved_state['arch']
    
    singleton_encoder = False
    if arch == 'singleton-encoder' or arch == 'dual-encoder':
        singleton_encoder = True
    pairwise_encoder = True
    if not (arch == 'pairwise-encoder' or arch == 'dual-encoder'):
        pairwise_encoder = False
    print(f"\n{task} ARCHITECTURE IS {arch}, PAIRWISE: {pairwise_encoder}, SINGLETON: {singleton_encoder}\n")

    # load the model weights
    model = DittoModel(device=device, lm=lm, use_pairwise_encoder=pairwise_encoder, use_singleton_encoder=singleton_encoder, shared_model=shared_model)
    model.load_state_dict(saved_state['model'])
    model = model.to(device)

    if fp16 and 'cuda' in device:
        model = amp.initialize(model, opt_level='O2')

    return config, model, saved_state


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default='Structured/Beer')
    parser.add_argument("--result_file", type=str, default='results.txt')
    parser.add_argument("--out_dir", type=str, default='./output')
    parser.add_argument("--lm", type=str, default='distilbert')
    parser.add_argument("--use_gpu", dest="use_gpu", action="store_true")
    parser.add_argument("--fp16", dest="fp16", action="store_true")
    parser.add_argument("--checkpoint_path", type=str, default='checkpoints/')
    parser.add_argument("--dk", type=str, default=None)
    parser.add_argument("--summarize", dest="summarize", action="store_true")
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--method", type=str, default="spec")
    parser.add_argument("--budget", type=int, default=1000)
    parser.add_argument("--late_fusion_only", dest="late_fusion_only", action="store_true")
    parser.add_argument("--job_id", type=int, default=None)
    parser.add_argument("--run_id", type=int, default=0)
    parser.add_argument("--ensemble_type", type=str, default="soft_vote")
    hp = parser.parse_args()

    # set seeds
    seed = hp.run_id
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # get the tokenizer once for the provided lm (optimization)
    tokenizer = get_tokenizer(hp.lm)

    # load PLM model (to be copied for each model)
    if hp.lm in lm_mp: # provided lm is in the map
        shared_model = AutoModel.from_pretrained(lm_mp[hp.lm])
    else:
        shared_model = AutoModel.from_pretrained(hp.lm)
    set_seed(123)

    # do this just to get domain names and write val and test data
    data_dir = f'updated_data_{hp.job_id}' if hp.job_id is not None else None
    domain_data, tasks = process_budget(hp.task, None, 1000, tokenizer, output_dir=data_dir)
    chkpt_path = hp.checkpoint_path + f'_{hp.job_id}' if hp.job_id is not None else hp.checkpoint_path

    # create output directory if it doesnt already exist
    os.makedirs(hp.out_dir, exist_ok=True)

    methods = (hp.method).split('-')
    select_best_ensemble = "vote" not in hp.ensemble_type
    for task in tasks:
        # SINGLE MODEL EVLAUTATION
        if len(methods) == 1 or select_best_ensemble:
            print(f"\nSINGLE MODEL EVALUATION\n")
            # load the models
            config, model, saved_state = load_model(task, chkpt_path, hp.method,
                            hp.lm, hp.use_gpu, hp.fp16, pairwise_encoder=not hp.late_fusion_only, singleton_encoder=hp.late_fusion_only, shared_model=shared_model)

            # write out test and val data
            train_fpath =  os.path.join(data_dir, config['trainset']) if data_dir is not None else config['trainset']
            with open(train_fpath, 'w') as f:
                f.writelines(domain_data[task][0])
            val_fpath = os.path.join(data_dir, config['validset']) if data_dir is not None else config['validset']
            with open(val_fpath, 'w') as f:
                f.writelines(domain_data[task][1])
            test_fpath = os.path.join(data_dir, config['testset']) if data_dir is not None else config['testset']
            with open(test_fpath, 'w') as f:
                f.writelines(domain_data[task][2])
            config['trainset'] = train_fpath
            config['validset'] = val_fpath
            config['testset'] = test_fpath

            # summarize, dk
            summarizer = dk_injector = None
            if hp.summarize:
                summarizer = Summarizer(config, hp.lm, tokenizer=tokenizer)
            if hp.dk is not None:
                if 'product' in hp.dk:
                    dk_injector = ProductDKInjector(config, hp.dk)
                else:
                    dk_injector = GeneralDKInjector(config, hp.dk)

            # tune threshold
            #cfg_for_tuning, _ = build_pooled_valid(config, domain_data, tasks, data_dir, N_MIN=60)
            #threshold = tune_threshold(cfg_for_tuning, model, hp, tokenizer=tokenizer, data_dir=data_dir)
            threshold = tune_threshold(config, model, hp, tokenizer=tokenizer, data_dir=data_dir)

            # run prediction
            outfile = os.path.join(hp.out_dir, f"{task}_{hp.method}_{hp.budget}_{hp.job_id}.txt") # output prediction results in same directory
            predict(test_fpath, outfile, config, model,
                    summarizer=summarizer,
                    max_len=hp.max_len,
                    lm=hp.lm,
                    dk_injector=dk_injector,
                    threshold=threshold,
                    late_fusion_only=hp.late_fusion_only,
                    tokenizer=tokenizer)
        # ENSEMBLE EVALUATION
        else:
            print(f"VOTING ENSEMBLE MODEL EVALUATION\n")
            # load the weights for each model
            loaded = []
            weights = []
            for m in methods:
                cfg_i, model_i, state_i = load_model(task, chkpt_path, m, hp.lm, hp.use_gpu, hp.fp16,
                                                    pairwise_encoder=not hp.late_fusion_only,
                                                    singleton_encoder=hp.late_fusion_only,
                                                    shared_model=shared_model)
                model_i.eval()
                loaded.append(model_i)
                weights.append(state_i.get('val_f1', 1.0))  # same order as `models`
            if "weighted" not in hp.ensemble_type:
                weights = None
            config = cfg_i  # same across methods for a given task

            # write out test and val data
            train_fpath =  os.path.join(data_dir, config['trainset']) if data_dir is not None else config['trainset']
            with open(train_fpath, 'w') as f:
                f.writelines(domain_data[task][0])
            val_fpath = os.path.join(data_dir, config['validset']) if data_dir is not None else config['validset']
            with open(val_fpath, 'w') as f:
                f.writelines(domain_data[task][1])
            test_fpath = os.path.join(data_dir, config['testset']) if data_dir is not None else config['testset']
            with open(test_fpath, 'w') as f:
                f.writelines(domain_data[task][2])
            config['trainset'] = train_fpath
            config['validset'] = val_fpath
            config['testset'] = test_fpath

            # summarize, dk
            summarizer = dk_injector = None
            if hp.summarize:
                summarizer = Summarizer(config, hp.lm, tokenizer=tokenizer)
            if hp.dk is not None:
                if 'product' in hp.dk:
                    dk_injector = ProductDKInjector(config, hp.dk)
                else:
                    dk_injector = GeneralDKInjector(config, hp.dk)

            # ensemble threshold
            cfg_for_tuning, _ = build_pooled_valid(config, domain_data, tasks, data_dir, N_MIN=60)
            #threshold = tune_threshold_ensemble(cfg_for_tuning, loaded, hp, tokenizer=tokenizer, data_dir=data_dir,
            #                        ensemble_type=hp.ensemble_type, weights=weights)
            threshold = tune_threshold_ensemble(cfg_for_tuning, loaded, hp, tokenizer=tokenizer, data_dir=data_dir,
                                    ensemble_type=hp.ensemble_type, weights=weights)

            # ensemble prediction + write file
            outfile = os.path.join(hp.out_dir, f"{task}_{hp.method}_{hp.budget}_{hp.job_id}.txt")
            predict_ensemble(test_fpath, outfile, config, loaded,
                 summarizer=summarizer, max_len=hp.max_len, lm=hp.lm,
                 dk_injector=dk_injector, threshold=threshold,
                 late_fusion_only=hp.late_fusion_only, tokenizer=tokenizer,
                 ensemble_type=hp.ensemble_type, weights=weights)

        # calculate and report f1, precision, recall to result file
        predicts = []
        with jsonlines.open(outfile, mode="r") as reader:
            for line in reader:
                predicts.append(int(line['match']))

        labels = []
        with open(test_fpath, mode="r") as fin:
            for line in fin:
                labels.append(int(line.split('\t')[-1]))

        real_f1 = sklearn.metrics.f1_score(labels, predicts)
        real_precision = sklearn.metrics.precision_score(labels, predicts)
        real_recall = sklearn.metrics.recall_score(labels, predicts)

        print(task + " TEST EVAULATION F1: " + str(real_f1))
        if hp.result_file is not None:
            with open(hp.result_file, 'a') as f:
                f.write(task + " F1: " + str(real_f1) + "\n")
                f.write(task + " Precision: " + str(real_precision) + "\n")
                f.write(task + " Recall: " + str(real_recall) + "\n")
