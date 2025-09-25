import heapq
import torch
import numpy as np
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

"""
The following functions were modified GPT-40 output
They implement farthest first selection of out-of-domain samples 
to complement the in-domain samples
"""
def to_unit_pairs(X, device="cuda"):
    # (gpu) convert pairs to average of singletons
    X = torch.as_tensor(X, device=device, dtype=torch.float64)
    X = torch.nn.functional.normalize(X, p=2, dim=1)
    pairs3d = X.view(-1, 2, X.size(1))     # (num_pairs, 2, dim)
    pairs = pairs3d.mean(dim=1)            # (num_pairs, dim)
    pairs = torch.nn.functional.normalize(pairs, p=2, dim=1)
    return pairs


""" K-CENTER GREEDY """
@torch.no_grad()
def kcenter_greedy_pairs(
    centers0: torch.Tensor,   # (C, D) normalized in-domain pairs (s0)
    cands: torch.Tensor,      # (N, D) normalized out-of-domain pairs (pool U)
    k: int,
    batch: int = 8192,
    device: str = "cuda"
) -> List[int]:
    """
    farthest first traversal (2-approx to k-center) with cosine distance d = 1 - dot,
    assuming inputs are already row-normalized
    """

    # insure correct dimension
    assert centers0.ndim == 2 and cands.ndim == 2
    assert centers0.size(1) == cands.size(1)
    C = centers0.size(0); N = cands.size(0)

    # short-circuit
    if N == 0 or k <= 0:
        return []

    centers0 = centers0.to(device=device, dtype=torch.float64)
    cands = cands.to(device=device, dtype=torch.float64)

    # initialize min distance to nearest existing center for all out-of-domaind data
    min_dist = torch.full((N,), float("inf"), dtype=torch.float32, device=device)
    for start in range(0, C, batch):
        end = min(start + batch, C)
        sims = cands @ centers0[start:end].T
        dists = 1.0 - sims
        chunk_min, _ = dists.min(dim=1)
        min_dist = torch.minimum(min_dist, chunk_min)

    # select k farthest first out-of-domains samples to add to in-domain
    selected: List[int] = []
    steps = min(k, N)
    for _ in range(steps):
        idx = int(torch.argmax(min_dist).item())
        selected.append(idx)

        c = cands[idx:idx+1]       
        sims = (cands @ c.T).squeeze(1)
        dists = 1.0 - sims
        min_dist = torch.minimum(min_dist, dists)
        min_dist[idx] = -1.0 

    return selected


def select_best_pairs_kcg(domain_idx, train_embed_data, domains, k=100, device= "cuda"):
    # build pair sets (gpu)
    train_pair_data = [to_unit_pairs(x, device=device) for x in train_embed_data]
    domain_pairs = train_pair_data[domain_idx]

    # gather all out-of-domain pairs with back-pointers
    cand_blocks, backptr = [], []
    for j, pairs in enumerate(train_pair_data):
        if j == domain_idx or pairs.size(0) == 0:
            continue
        cand_blocks.append(pairs)
        backptr.extend([(domains[j], i) for i in range(pairs.size(0))])

    if not cand_blocks:
        return []

    # out of domain candidate pairs as embeddings
    cands = torch.cat(cand_blocks, dim=0)

    # return the samples selected by k-center greedy approx
    sel = kcenter_greedy_pairs(domain_pairs, cands, k=k, batch=8192, device=device)
    return [backptr[i] for i in sel]
