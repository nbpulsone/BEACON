import torch
import os
import numpy as np
import jsonlines
import time

from torch.utils import data
import torch.nn.functional as F
from tqdm import tqdm
from ditto_light.dataset import DittoDataset
from ditto_light.summarize import Summarizer
from ditto_light.knowledge import *
import random
import sklearn.metrics as skm

"""
Functions that enable ensemble evaluation of EMAD models
Derived from ditto matcher.py, generated with help from GPT-4o
"""

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

def _batch_probs_from_model(model, batch, late_fusion_only=False):
    if late_fusion_only:
        # singleton cs loss
        _, x_left, x_right, y = batch
        enc_left, enc_right = model.forward_only_singletons(x_left, x_right)
        enc_left = F.normalize(enc_left, dim=1)
        enc_right = F.normalize(enc_right, dim=1)
        cos = F.cosine_similarity(enc_left, enc_right)
        return cos.detach().cpu().numpy()  # shape: (N,)
    else:
        # pairwise crossentropy loss
        x = batch[0]  
        logits = model(x)
        probs = logits.softmax(dim=1)[:, 1]
        return probs.detach().cpu().numpy()


def evaluate_ensemble(models, iterator, threshold=None, late_fusion_only=False,
                      ensemble_type="soft_vote", weights=None):
    """Evaluate an ensemble by averaging per-model probabilities.
       - If threshold is None: return (f1, best_th) like ditto.evaluate
       - Else: return f1 at the given threshold
    """

    all_y = []
    all_probs = []
    with torch.no_grad():
        for batch in iterator:
            # gather labels
            y = batch[-1]
            all_y += y.cpu().numpy().tolist()

            # gather per-model probs for this batch
            pm = []
            for m in models:
                pm.append(_batch_probs_from_model(m, batch, late_fusion_only=late_fusion_only))
            Pr = np.stack(pm, axis=1)#.mean(axis=1)

            # uniform or weighted soft-voting ensemble scheme
            if ensemble_type == "soft_vote_weighted" and weights is not None and len(weights) == Pr.shape[1]:
                w = np.asarray(weights, dtype=float)
                w_sum = w.sum()
                if w_sum > 0:
                    #print("\nWEIGHTED SOFT ENSEMBLE\n")
                    w = w / w_sum
                    agg = (Pr * w[None, :]).sum(axis=1)
                else:
                    #print("\nw SOFT ENSEMBLE\n")
                    agg = Pr.mean(axis=1)
            else:
                #print("\nSOFT ENSEMBLE\n")
                agg = Pr.mean(axis=1)
            all_probs += agg.tolist()

    if threshold is not None:
        pred = [1 if p > threshold else 0 for p in all_probs]
        return skm.f1_score(all_y, pred)

    # same threshold sweep from ditto.evaluate
    if late_fusion_only:
        best_th, rng = 0.5, np.linspace(0.0, 1.0, 101)
    else:
        best_th, rng = 0.5, np.arange(0.0, 1.0, 0.05)
 
    best_f1 = 0.0
    for th in rng:
        pred = [1 if p > th else 0 for p in all_probs]
        f1 = skm.f1_score(all_y, pred)
        if f1 > best_f1:
            best_f1, best_th = f1, th
    return best_f1, best_th


def tune_threshold_ensemble(config, models, hp, tokenizer=None, data_dir=None,
                            ensemble_type="soft_vote", weights=None):
    """ tune threshold for model(s) """
    #validset = os.path.join(data_dir, config['validset']) if data_dir is not None else config['validset']
    validset = config['validset']
    set_seed(123)

    # respect summarize/DK
    summarizer = injector = None
    if hp.summarize:
        summarizer = Summarizer(config, lm=hp.lm, tokenizer=tokenizer)
        validset = summarizer.transform_file(validset, max_len=hp.max_len, overwrite=True)
    if hp.dk is not None:
        injector = ProductDKInjector(config, hp.dk) if hp.dk == 'product' else GeneralDKInjector(config, hp.dk)
        validset = injector.transform_file(validset)

    # init valid set
    valid_dataset = DittoDataset(validset, max_len=hp.max_len, lm=hp.lm,
                                 singleton=hp.late_fusion_only, tokenizer=tokenizer)
    valid_iter = data.DataLoader(dataset=valid_dataset, batch_size=64, shuffle=False,
                                 num_workers=0, collate_fn=DittoDataset.pad)

    # call evaluate, get the threshold
    f1, th = evaluate_ensemble(models, valid_iter, threshold=None,
                           late_fusion_only=hp.late_fusion_only,
                           ensemble_type=ensemble_type, weights=weights)

    return th


def predict_ensemble(input_path, output_path, config, models, batch_size=1024,
                     summarizer=None, lm='distilbert', max_len=256,
                     dk_injector=None, threshold=None,
                     late_fusion_only=False, tokenizer=None,
                     ensemble_type="soft_vote", weights=None):
    """ use model(s) to make predictions on the input path, store results in output path """
    #pairs = []
    def serialize(row):
        return to_str(row[0], row[1], summarizer, max_len, dk_injector)

    # support .txt -> .jsonl conversion
    if '.txt' in input_path:
        with jsonlines.open(input_path + '.jsonl', mode='w') as writer:
            for line in open(input_path):
                writer.write(line.split('\t')[:2])
        input_path += '.jsonl'

    start_time = time.time()
    with jsonlines.open(input_path) as reader, jsonlines.open(output_path, mode='w') as writer:
        rows, serialized = [], []
        for idx, row in tqdm(enumerate(reader)):
            rows.append(row)
            serialized.append(serialize(row))
            if len(serialized) == batch_size:
                # one DittoDataset for the batch, shared by all models
                ds = DittoDataset(serialized, max_len=max_len, lm=lm,
                                  singleton=late_fusion_only, tokenizer=tokenizer)
                it = data.DataLoader(dataset=ds, batch_size=len(ds), shuffle=False,
                                     num_workers=0, collate_fn=DittoDataset.pad)
                for batch in it:
                    pm = []
                    for m in models:
                        pm.append(_batch_probs_from_model(m, batch, late_fusion_only=late_fusion_only))
                    #probs = np.stack(pm, axis=1).mean(axis=1)
                    # uniform or weighted soft voting ensemble scheme
                    Pr = np.stack(pm, axis=1)  # shape: (N, M)
                    if ensemble_type == "soft_vote_weighted" and weights is not None and len(weights) == Pr.shape[1]:
                        w = np.asarray(weights, dtype=float)
                        s = w.sum()
                        if s > 0:
                            w = w / s
                            probs = (Pr * w[None, :]).sum(axis=1)
                        else:
                            probs = Pr.mean(axis=1)  # fallback if all-zero weights
                    else:
                        probs = Pr.mean(axis=1)
                preds = (probs > (0.5 if threshold is None else threshold)).astype(int)
                if late_fusion_only:
                    for row, pred, p in zip(rows, preds.tolist(), probs.tolist()):
                        writer.write({'left': row[0], 'right': row[1], 'match': int(pred), 'match_confidence': float(p)})
                else:
                    # output schema: confidence == prob of predicted class
                    for row, pred, p in zip(rows, preds.tolist(), probs.tolist()):
                        conf = p if pred == 1 else (1.0 - p)
                        writer.write({'left': row[0], 'right': row[1], 'match': int(pred), 'match_confidence': float(conf)})
                rows.clear(); serialized.clear()

        if serialized:
            ds = DittoDataset(serialized, max_len=max_len, lm=lm,
                              singleton=late_fusion_only, tokenizer=tokenizer)
            it = data.DataLoader(dataset=ds, batch_size=len(ds), shuffle=False,
                                 num_workers=0, collate_fn=DittoDataset.pad)
            for batch in it:
                pm = []
                for m in models:
                    pm.append(_batch_probs_from_model(m, batch, late_fusion_only=late_fusion_only))
                #probs = np.stack(pm, axis=1).mean(axis=1)
                # uniform or weighted soft voting ensemble scheme
                Pr = np.stack(pm, axis=1)  # shape: (N, M)
                if ensemble_type == "soft_vote_weighted" and weights is not None and len(weights) == Pr.shape[1]:
                    w = np.asarray(weights, dtype=float)
                    s = w.sum()
                    if s > 0:
                        w = w / s
                        probs = (Pr * w[None, :]).sum(axis=1)
                    else:
                        probs = Pr.mean(axis=1)  # fallback if all-zero weights
                else:
                    probs = Pr.mean(axis=1)
            preds = (probs > (0.5 if threshold is None else threshold)).astype(int)
            if late_fusion_only:
                for row, pred, p in zip(rows, preds.tolist(), probs.tolist()):
                    writer.write({'left': row[0], 'right': row[1], 'match': int(pred), 'match_confidence': float(p)})
            else:
                for row, pred, p in zip(rows, preds.tolist(), probs.tolist()):
                    conf = p if pred == 1 else (1.0 - p)
                    writer.write({'left': row[0], 'right': row[1], 'match': int(pred), 'match_confidence': float(conf)})

    run_time = time.time() - start_time
    run_tag = '%s_lm=%s_dk=%s_su=%s' % (config['name'], lm, str(dk_injector != None), str(summarizer != None))
    os.system('echo %s %f >> log.txt' % (run_tag, run_time))
