import os
import shutil
import random
from sklearn.metrics.pairwise import cosine_similarity, rbf_kernel
import heapq
import torch
import numpy as np
import time
import json
import nltk
from ditto_light.generate_embeddings import generate_embeddings
from ditto_light.augment import augment_entity_matching_dataset
from sklearn.neighbors import KernelDensity
import torch
from torch.nn.functional import cosine_similarity as torch_cosine_similarity
from sklearn.decomposition import PCA
from scipy.stats import iqr

import torch, heapq, math
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


def to_unit_pairs(X, device="cuda"):
    X = torch.as_tensor(X, device=device, dtype=torch.float64)
    X = torch.nn.functional.normalize(X, p=2, dim=1)
    pairs3d = X.view(-1, 2, X.size(1))     # (num_pairs, 2, dim)
    pairs = pairs3d.mean(dim=1)            # (num_pairs, dim)
    pairs = torch.nn.functional.normalize(pairs, p=2, dim=1)
    return pairs


def train_domain_classifier(
    train_pairs,
    target_pairs,
    val_frac=0.25,
    device: str = "cuda"
):
    """
    Train a tiny linear domain classifier on pair embeddings for domain `domain_idx`.
    Returns w, b as torch tensors.

    train_embed_data: list of [N,dim] arrays (per domain, singleton embeddings)
    target_embed_data: list (1 entry = global pool, else per-domain)
    """

    X = np.vstack([
        train_pairs.detach().cpu().numpy(),
        target_pairs.detach().cpu().numpy()
    ])
    y = np.array([0]*len(train_pairs) + [1]*len(target_pairs))

    # handle tiny data safely
    if len(np.unique(y)) < 2 or len(y) < 20:
        # can’t validate; treat as no separation
        return None, None, 0.5

    Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=val_frac, stratify=y)

    clf = LogisticRegression(max_iter=300, solver="lbfgs")
    clf.fit(Xtr, ytr)
    auc = roc_auc_score(yva, clf.predict_proba(Xva)[:,1])

    w = torch.tensor(clf.coef_[0], dtype=torch.float64)
    b = torch.tensor(clf.intercept_[0], dtype=torch.float64)
    return w, b, float(auc)


@torch.no_grad()
def select_best_pairs_dr(
    domain_idx: int,
    train_embed_data: Sequence,          # list[array-like] of singletons for each domain (len divisible by 2)
    target_embed_data: Sequence,         # list[array-like] (1 item = shared target pool, else per-domain)
    domains: Sequence[str],
    k: int = 100,
    device: str = "cuda",

    # --- importance-weight (beta) options ---
    beta_mode: str = "logistic",         # "logistic" | "precomputed" | "none"
    beta_model: Optional[Dict[str, torch.Tensor]] = None,   # for "logistic": {"w": (D,), "b": ()}
    beta_func: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,  # for "precomputed": returns beta for (N,D)

    beta_clip: float = 10.0,             # cap beta to [1/beta_clip, beta_clip]
    beta_shrink: float = 0.2,            # shrink towards 1: beta <- (1-beta_shrink)*beta + beta_shrink*1.0
    score_mode: str = "product",         # "product" (beta * gain) or "sum" (gain + c*log(beta))
    sum_c: float = 0.25,                 # only used if score_mode="sum"

    # --- selection-time diagnostics/guards ---
    meff_floor_ratio: float = 0.30,      # require m_eff / m >= this threshold for the SELECTED set
    sample_size_ceiling: int = 200,
    return_details: bool = False
) -> List[Tuple[str, int]]:
    """
    Doubly-robust flavored sample selection:
      - Build pair embeddings (normalize singletons -> average adjacent -> renormalize).
      - Compute centroid-shift gain for each candidate (cosine similarity improvement to target centroid).
      - Estimate importance weights beta(x) (logistic domain classifier or user-provided).
      - Stabilize beta via clipping/shrinkage; score = beta * gain (or sum mode).
      - Pick top-k while monitoring effective sample size of chosen betas (m_eff = ||beta||_1^2 / ||beta||_2^2).

    Returns: list of (domain_name, pair_index_in_that_domain).
    """

    # -------- helpers --------
    def sigmoid(z: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(z)

    def compute_beta_logistic(X_pairs: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        # p(test|x) from a linear domain classifier on pair embeddings
        # beta = p / (1 - p) with eps for numerical safety
        p = sigmoid(X_pairs @ w + b)
        eps = 1e-6
        p = torch.clamp(p, eps, 1 - eps)
        return p / (1 - p)

    def stabilize_beta(beta: torch.Tensor) -> torch.Tensor:
        if beta_clip is not None and beta_clip > 0:
            inv = 1.0 / beta_clip
            beta = torch.clamp(beta, min=inv, max=beta_clip)
        if beta_shrink > 0:
            beta = (1.0 - beta_shrink) * beta + beta_shrink * 1.0
        return beta

    def effective_sample_size(beta: torch.Tensor) -> float:
        l1 = beta.abs().sum().item()
        l2_sq = (beta ** 2).sum().item()
        return (l1 * l1) / l2_sq if l2_sq > 0 else 0.0

    # -------- build pair sets --------
    train_pair_data  = [to_unit_pairs(torch.as_tensor(x, device=device), device=device) for x in train_embed_data]
    target_src = target_embed_data[0] if len(target_embed_data) == 1 else target_embed_data[domain_idx]
    target_pairs = to_unit_pairs(torch.as_tensor(target_src, device=device), device=device)

    domain_pairs = train_pair_data[domain_idx]
    if domain_pairs.numel() == 0:
        return []

    # checks to make sure we use the classifier only if it is good enough and we have very few samples 
    if domain_pairs.size(0) > sample_size_ceiling:
        print(f"CLASSIFIER NOT USED (LARGE DATASET)")
        beta_mode = "none"

    if beta_mode == "logistic" and beta_model is None:
        w, b, auc = train_domain_classifier(train_pair_data[domain_idx], target_pairs)
        # handle degenerate / tiny data case
        if w is None or b is None: # no model (not enough data)
            print(f"CLASSIFIER NOT USED (NO MODEL)")
            beta_mode = "none"
            beta_model = None
        elif auc < 0.62: # model not accurate enough
            print(f"CLASSIFIER NOT USED (PERFORMANCE)")
            beta_mode = "none"               
            beta_model = None
        else:
            print(f"CLASSIFIER USED")
            beta_model = {"w": w, "b": b}   

    orig_sum  = domain_pairs.sum(dim=0)                      # (D,)
    n_pairs   = domain_pairs.size(0)
    orig_mean = orig_sum / n_pairs                           # (D,)
    target_mean = target_pairs.mean(dim=0)                   # (D,)

    sim_orig = torch.nn.functional.cosine_similarity(
        orig_mean.unsqueeze(0), target_mean.unsqueeze(0)
    ).item()

    # -------- prepare beta estimator --------
    def betas_for_pairs(X_pairs: torch.Tensor, beta_model: Optional[Dict[str, torch.Tensor]] = None) -> torch.Tensor:
        if beta_mode == "none":
            return torch.ones(X_pairs.size(0), device=device, dtype=torch.float64)
        elif beta_mode == "precomputed":
            assert beta_func is not None, "beta_func must be provided when beta_mode='precomputed'"
            beta = beta_func(X_pairs)  # expect torch tensor on device
            return beta.to(device=device, dtype=torch.float64)
        elif beta_mode == "logistic":
            w = beta_model["w"].to(device=device, dtype=torch.float64)
            b = beta_model["b"].to(device=device, dtype=torch.float64)
            return compute_beta_logistic(X_pairs, w, b)
        else:
            raise ValueError(f"Unknown beta_mode={beta_mode}")

    # -------- score candidates domain-by-domain --------
    heap = []  # store (-score, j, i_pair, beta_i)
    for j, other_pairs in enumerate(train_pair_data):
        if j == domain_idx or other_pairs.size(0) == 0:
            continue

        # centroid-shift gain if we add one pair from domain j
        new_means = (orig_sum.unsqueeze(0) + other_pairs) / (n_pairs + 1)   # (num_pairs_j, D)
        sim_new   = torch.nn.functional.cosine_similarity(
            new_means, target_mean.unsqueeze(0).expand_as(new_means)
        )                                                                    # (num_pairs_j,)
        gains = sim_new - sim_orig                                           # (num_pairs_j,)

        # importance weights for those pairs
        beta_raw = betas_for_pairs(other_pairs, beta_model)
        beta_stb = stabilize_beta(beta_raw)

        # TODO: test non-hybrid score
        # hybrid score
        if score_mode == "product":
            score = beta_stb * gains
        elif score_mode == "sum":
            score = gains + (sum_c * torch.log(beta_stb))
        else:
            raise ValueError(f"Unknown score_mode={score_mode}")

        # push to heap on CPU
        score_cpu = score.detach().to("cpu")
        beta_cpu  = beta_stb.detach().to("cpu")
        for i_pair in range(score_cpu.numel()):
            heapq.heappush(heap, (-float(score_cpu[i_pair].item()), j, int(i_pair), float(beta_cpu[i_pair].item())))

    if not heap:
        return []

    # -------- take top-k with an m_eff guard on the SELECTED set --------
    selected: List[Tuple[str, int]] = []
    selected_betas = []

    while heap and len(selected) < k:
        neg_score, j, i_pair, beta_i = heapq.heappop(heap)
        trial_betas = torch.tensor(selected_betas + [beta_i], dtype=torch.float64)
        meff = effective_sample_size(trial_betas)
        if (meff / len(trial_betas)) >= meff_floor_ratio:
            selected.append((domains[j], i_pair))
            selected_betas.append(beta_i)
        # else: skip this overly spiky candidate; try next

    if return_details:
        details = {
            "sim_orig": sim_orig,
            "target_mean_norm": float(target_mean.norm().item()),
            "selected_meff": effective_sample_size(torch.tensor(selected_betas, dtype=torch.float64)) if selected_betas else 0.0,
            "selected_k": len(selected),
            "meff_floor_ratio": meff_floor_ratio,
            "beta_clip": beta_clip,
            "beta_shrink": beta_shrink,
            "score_mode": score_mode,
        }
        return selected, details

    return selected


""" K-CENTER GREEDY
Select k out-of-domain samples that, when added to the existing in-domain set,
minimize the maximum distance of any out-of-domain point to the nearest selected/in-domain point
 """
@torch.no_grad()
def kcenter_greedy_pairs(
    centers0: torch.Tensor,   # (C, D) normalized in-domain pairs (s0)
    cands: torch.Tensor,      # (N, D) normalized out-of-domain pairs (pool U)
    k: int,
    batch: int = 8192,
    device: str = "cuda"
) -> List[int]:
    """
    Farthest-first traversal (2-approx to k-center) with cosine distance d = 1 - dot,
    assuming inputs are already row-normalized.
    Returns indices in [0, N).
    """
    assert centers0.ndim == 2 and cands.ndim == 2
    assert centers0.size(1) == cands.size(1)
    C = centers0.size(0); N = cands.size(0)

    # Short-circuit
    if N == 0 or k <= 0:
        return []

    centers0 = centers0.to(device=device, dtype=torch.float64)
    cands = cands.to(device=device, dtype=torch.float64)

    # Initialize min distance to nearest existing center
    min_dist = torch.full((N,), float("inf"), dtype=torch.float32, device=device)
    for start in range(0, C, batch):
        end = min(start + batch, C)
        sims = cands @ centers0[start:end].T          # (N, b)
        dists = 1.0 - sims
        chunk_min, _ = dists.min(dim=1)
        min_dist = torch.minimum(min_dist, chunk_min)

    selected: List[int] = []
    steps = min(k, N)
    for _ in range(steps):
        idx = int(torch.argmax(min_dist).item())
        selected.append(idx)

        c = cands[idx:idx+1]                          # (1, D)
        sims = (cands @ c.T).squeeze(1)               # (N,)
        dists = 1.0 - sims
        min_dist = torch.minimum(min_dist, dists)
        min_dist[idx] = -1.0                          # exclude reselect

    return selected


def select_best_pairs_kcg(domain_idx, train_embed_data, domains, k=100, device= "cuda"):
    # Build pair sets (you already have this)
    train_pair_data = [to_unit_pairs(x, device=device) for x in train_embed_data]

    domain_pairs = train_pair_data[domain_idx]        # (C, D) existing centers (s0)
    # gather all out-of-domain pairs with back-pointers
    cand_blocks, backptr = [], []
    for j, pairs in enumerate(train_pair_data):
        if j == domain_idx or pairs.size(0) == 0:
            continue
        cand_blocks.append(pairs)
        backptr.extend([(domains[j], i) for i in range(pairs.size(0))])

    if not cand_blocks:
        return []

    cands = torch.cat(cand_blocks, dim=0)            # (N, D)

    sel = kcenter_greedy_pairs(domain_pairs, cands, k=k, batch=8192, device=device)
    return [backptr[i] for i in sel]







"""OPTIMAL TRANSPORT-FLAVORED SELECTION"""
@torch.no_grad()
def _kcenter_on_pool(
    cands: torch.Tensor, k: int, device: str = "cuda"
) -> List[int]:
    """
    Farthest-first on the candidate pool itself (no external centers).
    Seed with the first item (highest OT score in U_top).
    """
    cands = cands.to(device=device, dtype=torch.float64)
    N = cands.size(0)
    if N == 0 or k <= 0:
        return []
    selected = [0]  # seed with top-1 by OT score (already at index 0 of U_top)
    # init min squared distance to the seed
    sims = (cands @ cands[0:1].T).squeeze(1)      # (N,)
    min_d2 = 2.0 * (1.0 - sims)                   # squared Euclid on unit sphere
    min_d2[0] = -1.0
    for _ in range(1, min(k, N)):
        idx = int(torch.argmax(min_d2).item())
        selected.append(idx)
        sims = (cands @ cands[idx:idx+1].T).squeeze(1)
        d2   = 2.0 * (1.0 - sims)
        min_d2 = torch.minimum(min_d2, d2)
        min_d2[idx] = -1.0
    return selected


@torch.no_grad()
def select_best_pairs_ot(
    domain_idx: int,
    train_embed_data: Sequence,        # list of singletons arrays/tensors per domain
    domains: Sequence[str],
    k: int = 100,
    device: str = "cuda",
    # --- OT knobs ---
    eps: float = 0.05,                 # Sinkhorn temperature (softmin sharpness)
    m_factor: int = 8,                 # top-M = m_factor * k before KCG
    batch_U: int = 8192,               # batch size over candidates
    # --- Target weighting ---
    target_weights: Optional[torch.Tensor] = None,   # shape (|T|,); will be normalized
    target_labels:  Optional[torch.Tensor] = None,   # shape (|T|,), ints; optional
    class_weights:  Optional[Dict[int, float]] = None,  # e.g., {1:1.0, 0:0.5}
    kcenter_batch: int = 8192,
    diversify: bool = True
) -> List[Tuple[str, int]]:
    """
    OT-guided selection:
      1) Score each OOD candidate u by soft OT coverage of T: score(u) = sum_t softmin(-C/eps)[u,t] * q_t
      2) Take top-M scores; run K-center greedy vs T to get diverse final k.
      3) Return backpointers as (donor_domain, pair_idx_in_that_domain).

    S starts as T (your in-domain pairs). U are all other domains' pairs.
    """

    # ---------- Build pair sets ----------
    train_pair_data = [to_unit_pairs(x, device=device) for x in train_embed_data]
    T_pairs = train_pair_data[domain_idx]                  # (|T|, D)
    if T_pairs.size(0) == 0:
        return []

    # gather U (all OOD pairs) with back-pointers
    U_blocks, backptr = [], []
    for j, P in enumerate(train_pair_data):
        if j == domain_idx or P.size(0) == 0:
            continue
        U_blocks.append(P)
        backptr.extend([(domains[j], i) for i in range(P.size(0))])
    if not U_blocks:
        return []
    U_pairs = torch.cat(U_blocks, dim=0)                   # (|U|, D)
    N_u, N_t = U_pairs.size(0), T_pairs.size(0)

    # ---------- Target weights q on T ----------
    if target_weights is not None:
        q = torch.as_tensor(target_weights, device=device, dtype=torch.float64)
        if q.numel() != N_t:
            raise ValueError("target_weights length must equal |T|")
    else:
        q = torch.ones(N_t, device=device, dtype=torch.float64)

    if target_labels is not None and class_weights is not None:
        y = torch.as_tensor(target_labels, device=device)
        if y.numel() != N_t:
            raise ValueError("target_labels length must equal |T|")
        q = q.clone()
        for cls, w in class_weights.items():
            q[y == cls] *= float(w)

    q = q / (q.sum() + 1e-12)  # normalize to 1

    # ---------- Optional drift guard: precompute min-dist-to-T for all U ----------
    # Use squared Euclid on unit vectors via 2*(1-cos)
    # We'll also reuse these sims in batches for scores.
    min_d2_to_T = torch.full((N_u,), float("inf"), dtype=torch.float64, device=device)
    # We'll compute scores in the same loop, but need a work buffer:
    scores = torch.empty(N_u, dtype=torch.float64, device=device)

    # Pre-compute T transposed and q for matmuls
    Tt = T_pairs.T.contiguous()  # (D, |T|)
    qv = q.contiguous()          # (|T|,)

    # ---------- Batched scoring: softmin over T ----------
    # score(u) = softmax(-C/eps) @ q, with C = 2*(1 - u.t), t in T
    # To keep memory bounded, process U in chunks
    for s in range(0, N_u, batch_U):
        e = min(s + batch_U, N_u)
        U_chunk = U_pairs[s:e]                        # (B,D)
        sims = U_chunk @ Tt                           # (B, |T|)
        d2 = 2.0 * (1.0 - sims)                       # (B, |T|)
        # update min distance to T for drift guard
        chunk_min_d2, _ = d2.min(dim=1)               # (B,)
        min_d2_to_T[s:e] = torch.minimum(min_d2_to_T[s:e], chunk_min_d2)

        # softmin kernel over T
        logits = -d2 / float(eps)
        # subtract row max for numerical stability
        logits = logits - logits.max(dim=1, keepdim=True).values
        K = torch.exp(logits)
        K = K / (K.sum(dim=1, keepdim=True) + 1e-12)  # (B, |T|)

        scores[s:e] = K @ qv                           # weighted coverage of T

    idx_all = torch.arange(N_u, device=device)
        
    # If caller wants everything, return everyone once, OT-sorted
    if k >= idx_all.numel():
        order = torch.argsort(scores, descending=True)
        return [backptr[int(i)] for i in order.tolist()]

    # OT-only mode: pick global top-k by OT score (no diversity step)
    if not diversify:
        topk_idx = torch.topk(scores, k=min(k, scores.numel()), largest=True).indices
        final_idx = idx_all[topk_idx]
        return [backptr[int(i)] for i in final_idx.tolist()]

    # ---------- Top-M then K-center for diversity ----------
    if idx_all.numel() == 0:
        return []

    M = int(max(k, m_factor * k))
    topM = torch.topk(scores, k=min(M, scores.numel()), largest=True).indices
    sel_pool_idx = idx_all[topM]                        # indices into U_pairs
    U_top = U_pairs[sel_pool_idx]                       # (M', D)

    # diversity within the OT top-M
    sel_rel = _kcenter_on_pool(U_top, k=k, device=device)
    final_idx = sel_pool_idx[torch.as_tensor(sel_rel, device=device)]

    # NEW: if diversity returned < k (e.g., tiny M), pad with next-best by score (uniques)
    if final_idx.numel() < k:
        chosen = set(final_idx.tolist())
        # take remaining from sel_pool_idx ordered by scores_masked (which corresponds to idx_all)
        # get global ordering over idx_all by score
        order = torch.argsort(scores, descending=True)
        for gi in order.tolist():
            cand = int(idx_all[gi].item())
            if cand not in chosen:
                final_idx = torch.cat([final_idx, torch.tensor([cand], device=device)])
                chosen.add(cand)
                if final_idx.numel() == k:
                    break

    # map back
    out = [backptr[int(i)] for i in final_idx.tolist()]
    return out


def select_best_pairs_kcg_label_aware(domain_idx, train_embed_data, matches_embed_data, nonmatches_embed_data, domains, k=1000, device= "cuda", prop_matches_to_select=0.5, in_domain_labels=True, out_domain_labels=True):
    print("\n\nSELECTING LABEL-AWARE KCG SAMPLES\n\n")
    # calculate positives and negatives needed
    label_pair_selections = []
    k_pos = int(k * prop_matches_to_select)
    k_neg = k - k_pos

    def dedup_two_lists(neg_list, pos_list, k_neg, k_pos):
        # helper: de-dupe negatives and positives selected if necessary (ID only case)
        seen = set()
        neg_u = []
        for x in neg_list:
            if x not in seen:
                neg_u.append(x)
                seen.add(x)
            if len(neg_u) == k_neg:
                break

        pos_u = []
        for x in pos_list:
            if x not in seen:
                pos_u.append(x)
                seen.add(x)
            if len(pos_u) == k_pos:
                break

        return neg_u, pos_u

    # no in + no out
    if not in_domain_labels and not out_domain_labels:
        label_pair_selections = select_best_pairs_kcg(domain_idx, train_embed_data, domains, k=k, device=device)
    # in + no out
    elif in_domain_labels and not out_domain_labels:
        # create temporary data arrays to accomodate this scenario
        tmp_neg = list(train_embed_data); tmp_neg[domain_idx] = nonmatches_embed_data[domain_idx]
        tmp_pos = list(train_embed_data); tmp_pos[domain_idx] = matches_embed_data[domain_idx]
        buffer = max(100, int(0.25 * k)) # add small buffer to conver cases when the same sample is selected twice
        
        # add samples comparing NEGATIVE in-domain distribution using KCG 
        print(f"[ID] Selecting {k_neg + buffer} label-aware KCG non-matches...")
        neg_raw = select_best_pairs_kcg(domain_idx, tmp_neg, domains, k=k_neg + buffer, device=device)

        # add samples comparing POSITIVE in-domain distribution using KCG 
        print(f"[ID] Selecting {k_pos + buffer} label-aware KCG matches...")
        pos_raw = select_best_pairs_kcg(domain_idx, tmp_pos, domains, k=k_pos + buffer, device=device)

        # de-dupe the selected samples
        neg_selected, pos_selected = dedup_two_lists(neg_raw, pos_raw, k_neg, k_pos)
        label_pair_selections = neg_selected + pos_selected
    # not in + out
    elif not in_domain_labels and out_domain_labels:
        # create temporary data arrays to accomodate this scenario
        tmp_neg = list(nonmatches_embed_data); tmp_neg[domain_idx] = train_embed_data[domain_idx]
        tmp_pos = list(matches_embed_data);    tmp_pos[domain_idx] = train_embed_data[domain_idx]

        # add NEGATIVE samples using KCG, comparing to general in-domain distribution
        print(f"[OOD] Selecting {k_neg} label-aware KCG non-matches...")
        label_pair_selections.append(select_best_pairs_kcg(domain_idx, tmp_neg, domains, k=k_neg, device=device))

        # add POSITIVE samples using KCG, comparing to general in-domain distribution
        print(f"[OOD] Selecting {k_pos} label-aware KCG matches...")
        label_pair_selections.append(select_best_pairs_kcg(domain_idx, tmp_pos, domains, k=k_pos, device=device))        
    # in + out
    else:
        # add NEGATIVE samples using KCG
        print(f"[ID/OOD] Selecting {k_neg} label-aware KCG non-matches...")
        label_pair_selections.append(select_best_pairs_kcg(domain_idx, nonmatches_embed_data, domains, k=k_neg, device=device))

        # add POSITIVE samples using KCG
        print(f"[ID/OOD] Selecting {k_pos} label-aware KCG matches...")
        label_pair_selections.append(select_best_pairs_kcg(domain_idx, matches_embed_data, domains, k=k_pos, device=device))

    return label_pair_selections
