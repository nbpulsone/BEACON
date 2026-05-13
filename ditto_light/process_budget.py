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
from ditto_light.selection_methods import select_best_pairs_dr, select_best_pairs_kcg, select_best_pairs_ot, select_best_pairs_kcg_label_aware
from ditto_light.training_samples_selection import TopKSelection
from sklearn.cluster import KMeans
from collections import Counter

nltk.download('stopwords')

""" Read the data from a directory, find its domains and store the train, val, test data """
def read_domain_files(domains, fpath):
    # dictionary to contain all the data per domain
    data = {}

    # go through each file in each domain and add the data
    for dom in domains:
        # init domain
        data[dom] = [[],[],[]]

        # train data
        trainset = os.path.join(fpath, dom + "_train.txt")
        with open(trainset, "r") as f:
            data[dom][0] = f.readlines()
        # validation data
        valset = os.path.join(fpath, dom + "_val.txt")
        with open(valset, "r") as f:
            data[dom][1] = f.readlines()
        # test data
        testset = os.path.join(fpath, dom + "_test.txt")
        with open(testset, "r") as f:
            data[dom][2] = f.readlines()
        # debug
        print(f"Read samples for {dom}: {[len(data[dom][i]) for i in range(len(data[dom]))]}")
    return data

# Oversample domain specific data to get the budget (spec)
def get_specific_samples(data, domains, budget):
    # over sample to get the correct number of samples
    result_data = {}
    for dom in domains:
        result_data[dom] = []
        if len(data[dom][0]) >= budget:
            # for now just random sample
            #TODO: eventually use sota active learning methods here
            result_data[dom] = random.sample(data[dom][0], budget)
        else:
            result_data[dom] = data[dom][0] * (budget // len(data[dom][0])) + data[dom][0][:budget % len(data[dom][0])]

    return result_data

# Choose random samples for each domain (gen)
def get_random_samples(data, domains, budget):
    # gather all the data
    all_data = []
    for dom in domains:
        all_data.extend(data[dom][0])

    # default to using all the data
    if budget == None or budget < 0 or budget > len(all_data):
        target = len(all_data)
    else:
        target = budget

    # take a random sample
    result_data = {}
    for dom in domains:
        result_data[dom] = random.sample(all_data, target)

        # oversample to fill budget if needed
        if len(result_data[dom]) < budget:
            result_data[dom] = result_data[dom] * (budget // len(result_data[dom])) + result_data[dom][:budget % len(result_data[dom])]

    return result_data  


# Choose the nearest neighbors samples to complement the domain specific data 
def get_nn_samples(data, domains, budget, embed_fpath, EMBED_DIM=768, single_domain=None):
    # get the embeddings
    embed_data = []
    if isinstance(embed_fpath, str):
        for dom in domains:
            embed_file = os.path.join(embed_fpath, "all_" + dom + ".pt")
            embed_data.append(torch.load(embed_file).numpy())
    else:
        embed_data = embed_fpath

    # (singleton) normalize the vectors
    for i in range(len(embed_data)):
        norms = np.linalg.norm(embed_data[i], axis=1, keepdims=True)
        embed_data[i] = embed_data[i] / np.clip(norms, a_min=1e-10, a_max=None)

    # convert form singleton -> pairwise by averaging adacent vectors (since the order is maintained and thus they form matches)
    for i in range(len(embed_data)):
        n = len(embed_data[i])
        res = np.zeros((n//2, EMBED_DIM))
        for j in range(0, n, 2):
            res[j//2] = (embed_data[i][j] + embed_data[i][j+1])/2
        
        # NEW: L2-normalize each pair vector
        pair_norms = np.linalg.norm(res, axis=1, keepdims=True)
        pairs = res / np.clip(pair_norms, a_min=1e-10, a_max=None)
        embed_data[i] = pairs
        #embed_data[i] = res

    # get the average embedding vectors
    embed_avgs = []
    for i in range(len(embed_data)):
        avg = np.mean(embed_data[i], axis=0)

        # NEW: L2-normalize the centroid so Euclidean distance ≡ cosine ranking
        c_norm = np.linalg.norm(avg)
        if c_norm > 0:
            avg = avg / c_norm
            
        embed_avgs.append(avg)
    
    # get nearest neighbor data for each domain
    result_data = {}
    domains_to_run = [dom for dom in domains if dom == single_domain or single_domain is None] # optimize when only one domain's data is needed
    for dom in domains_to_run:
        # start with domaind specific data
        result_data[dom] = data[dom][0].copy()
        i = domains.index(dom)

        # if already have enough data --> skip
        if len(result_data[dom]) >= budget:
            result_data[dom] = random.sample(result_data[dom], budget)
            continue

        # accumulate all other data as the search space for this category
        all_other_data = []
        # all_other_data = np.empty((0, EMBED_DIM))
        domain_indices = []
        for j, vectors in enumerate(embed_data):
            if i == j:
                continue
            #all_other_data = np.concatenate((all_other_data, vectors), axis=0)
            all_other_data.append(vectors)
            domain_indices.extend([(j, idx) for idx in range(len(vectors))])
        all_other_data = np.vstack(all_other_data)

        # computer costine distance to centroid of target domain
        target_centroid = embed_avgs[i]
        distances = np.linalg.norm(all_other_data - target_centroid, axis=1)
        closest_indices = np.argsort(distances)[:budget - len(result_data[dom])]

        #print(f"\n\n\n{dom} NN BEST SAMPLES: [", end='')
        for idx in closest_indices:
            donor_dom_idx, pair_idx = domain_indices[idx]
            donor_dom = domains[donor_dom_idx]
            #print(f"({donor_dom}, {pair_idx}), ", end='')
            result_data[dom].append(data[donor_dom][0][pair_idx])
        #print(']\n\n\n')
    
        # oversample result data if needed
        if len(result_data[dom]) < budget:
            full = result_data[dom]
            result_data[dom] = full * (budget // len(full)) + full[:budget % len(full)]

        # log
        print(f"{dom} added nearest neighbors to reach {len(result_data[dom])} samples")
    return result_data 


# ~~~ HELPER FUNCTIONS FOR TT ~~~
def group_singleton_embeddings_to_pairwise(singleton_embeddings):
    return np.mean(singleton_embeddings.reshape(-1, 2, singleton_embeddings.shape[1]), axis=1)

def get_pair_vector(embed_list, pair_idx):
    """Extract the two singleton embeddings that make up a pair."""
    return embed_list[2 * pair_idx: 2 * pair_idx + 2]

def compute_gain(current_train, target_mean, candidate_pair):
    orig_mean = np.mean(current_train, axis=0)
    new_train = np.vstack([current_train, candidate_pair])
    new_mean = np.mean(new_train, axis=0)

    sim_orig = cosine_similarity(orig_mean.reshape(1, -1), target_mean.reshape(1, -1))[0, 0]
    sim_new = cosine_similarity(new_mean.reshape(1, -1), target_mean.reshape(1, -1))[0, 0]

    return sim_new - sim_orig

def select_best_pairs(domain_idx, train_embed_data, target_embed_data, domains, k=100):
    print("\n\nSELECTING CS\n\n")
    # get the training embeddings for this domain
    domain_train = train_embed_data[domain_idx]

    # check if only a single domain is included in the target data (optimization)
    if len(target_embed_data) == 1: 
        domain_target = target_embed_data[0]#.numpy()
    else:
        domain_target = target_embed_data[domain_idx]
    target_mean = np.mean(domain_target, axis=0)

    # create a heap to organize "max gain" samples
    heap = []
    for j, other_train in enumerate(train_embed_data):
        if j == domain_idx:
            continue
        other_train = other_train#.numpy()
        num_pairs = len(other_train) // 2
        for i in range(num_pairs):
            pair_vec = get_pair_vector(other_train, i)
            gain = compute_gain(domain_train, target_mean, np.array(pair_vec))
            #if gain > 0:
            heapq.heappush(heap, (-gain, j, i))  # max-heap => negative gain

    #k = max(k, len(heap))
    # get top-k highest gain pairs
    top_k = heapq.nsmallest(k, heap)
    return [(domains[j], i) for _, j, i in top_k]

#TODO: test normalize functions
def select_best_pairs_gpu(domain_idx, train_embed_data, target_embed_data, domains, k=100, device='cuda'):
    print("\n\nSELECTING CS GPU\n\n")
    # move embeddings to GPU
    train_embed_data = [torch.tensor(x, device=device, dtype=torch.float64) for x in train_embed_data]
    target_embed_data = [torch.tensor(x, device=device, dtype=torch.float64) for x in target_embed_data]

    # get current domain embeddings and target mean
    domain_train = train_embed_data[domain_idx]
    domain_train = torch.nn.functional.normalize(domain_train, p=2, dim=1)
    orig_mean = domain_train.mean(dim=0)

    domain_target = target_embed_data[0] if len(target_embed_data) == 1 else target_embed_data[domain_idx]
    domain_target = torch.nn.functional.normalize(domain_target, p=2, dim=1)
    target_mean = domain_target.mean(dim=0)

    # compute gain using cosine sim
    sim_orig = torch_cosine_similarity(orig_mean.unsqueeze(0), target_mean.unsqueeze(0)).item()

    heap = []
    for j, other_train in enumerate(train_embed_data):
        if j == domain_idx:
            continue

        # normalize the data from this domain
        other_train  = torch.nn.functional.normalize(other_train, p=2, dim=1)
    
        # reshape into (num_pairs, 2, dim)
        pairs = other_train.view(-1, 2, other_train.size(1))  # shape: (num_pairs, 2, dim)

        # sum the two unit rows in each pair (treat as adding two samples)
        pair_sums = pairs.sum(dim=1)  # shape: (num_pairs, dim)

        # calculate the new centroids for each pair to get gain
        new_means = (domain_train.sum(dim=0) + pair_sums) / (domain_train.size(0) + 2)
        sim_new = torch_cosine_similarity(new_means, target_mean.unsqueeze(0).expand_as(new_means))
        gains = sim_new - sim_orig  # shape: (num_pairs,)

        # push to heap (on CPU)
        for i in range(len(gains)):
            heapq.heappush(heap, (-gains[i].item(), j, i))

    top_k = heapq.nsmallest(k, heap)
    #print(f"\n\n\nBEST SAMPLE INDICES: {[(domains[j], i) for _, j, i in top_k]}\n\n\n")
    #print(f"\n\n\nGAINS: {[x for x, j, i in top_k]}\n\n\n")
    return [(domains[j], i) for _, j, i in top_k]


# TODO: test CS dist. fitting method with changes made
def select_best_pairs_gpu2(domain_idx, train_embed_data, target_embed_data, domains, k=100, device='cuda'):
    print("\n\nSELECTING CS GPU (centroid shift with pairwise pipeline)\n\n")
    # move embeddings to GPU
    train_embed_data  = [torch.tensor(x, device=device, dtype=torch.float64) for x in train_embed_data]
    target_embed_data = [torch.tensor(x, device=device, dtype=torch.float64) for x in target_embed_data]

    # --- helper: singleton -> adjacent-pair average -> row-normalize each pair ---
    def to_unit_pairs(X: torch.Tensor) -> torch.Tensor:
        # row-normalize singletons
        X = torch.nn.functional.normalize(X, p=2, dim=1)
        # reshape to pairs and average adjacent singletons
        pairs3d = X.view(-1, 2, X.size(1))  # shape: (num_pairs, 2, dim)
        pairs = pairs3d.mean(dim=1)         # shape: (num_pairs, dim)
        # row-normalize each pair vector
        pairs = torch.nn.functional.normalize(pairs, p=2, dim=1)
        return pairs

    # --- build unit pair sets (train + target) ---
    train_pair_data  = [to_unit_pairs(x) for x in train_embed_data]
    # if target has a single blob, reuse it; else take per-domain
    target_pair_data = [to_unit_pairs(target_embed_data[0] if len(target_embed_data) == 1 else target_embed_data[i])
                        for i in range(len(train_embed_data))]

    # current domain: orig mean and target mean (both from unit pairs)
    domain_pairs = train_pair_data[domain_idx]
    orig_sum = domain_pairs.sum(dim=0)
    orig_mean = orig_sum / domain_pairs.size(0)

    target_pairs = target_pair_data[domain_idx]
    target_mean = target_pairs.mean(dim=0)
    # NOTE: no need to normalize target_mean for cosine; cosine_similarity divides by norms internally.

    # baseline similarity (constant for this domain)
    sim_orig = torch.nn.functional.cosine_similarity(
        orig_mean.unsqueeze(0), target_mean.unsqueeze(0)
    ).item()

    # --- rank pairs from other domains by centroid-shift gain ---
    heap = []
    n_pairs = domain_pairs.size(0)
    for j, other_pairs in enumerate(train_pair_data):
        if j == domain_idx or other_pairs.size(0) == 0:
            continue

        # add each candidate PAIR (as one sample in pair-space) to the centroid
        new_means = (orig_sum.unsqueeze(0) + other_pairs) / (n_pairs + 1)  # shape: (num_pairs_j, dim)

        # cosine to target centroid; gain = sim_new - sim_orig (sim_orig constant)
        sim_new = torch.nn.functional.cosine_similarity(
            new_means, target_mean.unsqueeze(0).expand_as(new_means)
        )
        gains = sim_new - sim_orig  # shape: (num_pairs_j,)

        # push to heap (on CPU); negate for nsmallest-as-max
        for i_pair in range(other_pairs.size(0)):
            heapq.heappush(heap, (-gains[i_pair].item(), j, i_pair))

    if not heap:
        return []

    top_k = heapq.nsmallest(k, heap)

    # debug
    #print(f"\n\n\nBEST SAMPLE INDICES: {[(domains[j], i) for _, j, i in top_k]}\n\n\n")
    #print(f"\n\n\nGAINS: {[x for x, j, i in top_k]}\n\n\n")

    return [(domains[j], i) for _, j, i in top_k]


def select_best_pairs_gpu2_label_aware(domain_idx, train_embed_data, matches_embed_data, nonmatches_embed_data, target_embed_data, target_matches_embed_data, target_nonmatches_embed_data, domains, k=1000, device='cuda', prop_matches_to_select=0.5, in_domain_labels=True, out_domain_labels=True):
    print("\n\nSELECTING LABEL-AWARE CS GPU (centroid shift with pairwise pipeline)\n\n")
    
    # get number of samples to select from each label
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

    # index target differently if using only one domain
    tgt_domain_idx = domain_idx if len(target_embed_data) > 1 else 0

    # not in + not out
    if not in_domain_labels and not out_domain_labels:
        label_pair_selections = select_best_pairs_gpu2(domain_idx, train_embed_data, target_embed_data, domains, k=k)
    # in + not out
    elif in_domain_labels and not out_domain_labels:
        # create temporary data arrays to accomodate this scenario
        tmp_neg = list(train_embed_data); tmp_neg[domain_idx] = nonmatches_embed_data[domain_idx]
        tmp_pos = list(train_embed_data); tmp_pos[domain_idx] = matches_embed_data[domain_idx]
        tgt_tmp_neg = list(target_embed_data); tgt_tmp_neg[tgt_domain_idx] = target_nonmatches_embed_data[tgt_domain_idx]
        tgt_tmp_pos = list(target_embed_data); tgt_tmp_pos[tgt_domain_idx] = target_matches_embed_data[tgt_domain_idx]
        buffer = max(50, int(0.1 * k)) # add small buffer to conver cases when the same sample is selected twice

        # add samples comparing NEGATIVE in-domain distribution using TVDF
        print(f"[ID] Selecting {k_neg + buffer} label-aware TVDF non-matches...")
        neg_raw = select_best_pairs_gpu2(domain_idx, tmp_neg, tgt_tmp_neg, domains, k=k_neg)

        # add samples comparing POSITIVE in-domain distribution using TVDF
        print(f"[ID] Selecting {k_pos + buffer} label-aware TVDF matches...") 
        pos_raw = select_best_pairs_gpu2(domain_idx, tmp_pos, tgt_tmp_pos, domains, k=k_pos)

        # de-dupe the selected samples
        neg_selected, pos_selected = dedup_two_lists(neg_raw, pos_raw, k_neg, k_pos)
        label_pair_selections = neg_selected + pos_selected
    # not in + out
    elif not in_domain_labels and out_domain_labels:
        # create temporary data arrays to accomodate this scenario
        tmp_neg = list(nonmatches_embed_data); tmp_neg[domain_idx] = train_embed_data[domain_idx]
        tmp_pos = list(matches_embed_data);    tmp_pos[domain_idx] = train_embed_data[domain_idx]
        tgt_tmp_neg = list(target_nonmatches_embed_data); tgt_tmp_neg[tgt_domain_idx] = target_embed_data[tgt_domain_idx]
        tgt_tmp_pos = list(target_matches_embed_data); tgt_tmp_pos[tgt_domain_idx] = target_embed_data[tgt_domain_idx]

        # add NEGATIVE samples using TVDF, comparing to general in-domain distribution
        print(f"[OOD] Selecting {k_neg} label-aware TVDF non-matches...")
        label_pair_selections.append(select_best_pairs_gpu2(domain_idx, tmp_neg, tgt_tmp_neg, domains, k=k_neg))

        # add POSITIVE samples using TVDF, comparing to general in-domain distribution
        print(f"[OOD] Selecting {k_pos} label-aware TVDF matches...")
        label_pair_selections.append(select_best_pairs_gpu2(domain_idx, tmp_pos, tgt_tmp_pos, domains, k=k_pos))        
    # in + out
    else:
        # add NEGATIVE samples using TVDF
        print(f"Selecting {k_neg} label-aware TVDF non-matches...")
        label_pair_selections.append(select_best_pairs_gpu2(domain_idx, nonmatches_embed_data, target_nonmatches_embed_data, domains, k=k_neg))

        # add POSITIVE samples using TVDF
        print(f"Selecting {k_pos} label-aware TVDF matches...")
        label_pair_selections.append(select_best_pairs_gpu2(domain_idx, matches_embed_data, target_matches_embed_data, domains, k=k_pos))

    return label_pair_selections


def select_best_pca_pairs_gpu2(domain_idx, train_embed_data, target_embed_data, domains, k=100, device='cuda', pca_n=3):
    print("\n\nSELECTING PCA COVERAGE GPU (pairwise pipeline)\n\n")

    # move embeddings to device
    train_embed_data  = [torch.as_tensor(x, device=device, dtype=torch.float32) for x in train_embed_data]
    target_embed_data = [torch.as_tensor(x, device=device, dtype=torch.float32) for x in target_embed_data]

    def to_unit_pairs(X: torch.Tensor) -> torch.Tensor:
        # row-normalize singletons
        X = torch.nn.functional.normalize(X, p=2, dim=1)
        # reshape to pairs and average adjacent singletons
        pairs3d = X.view(-1, 2, X.size(1))  # shape: (num_pairs, 2, dim)
        pairs = pairs3d.mean(dim=1)         # shape: (num_pairs, dim)
        # row-normalize each pair vector
        pairs = torch.nn.functional.normalize(pairs, p=2, dim=1)
        return pairs

    # build pair embeddings
    train_pair_data = [to_unit_pairs(x) for x in train_embed_data]
    target_pair_data = [
        to_unit_pairs(target_embed_data[0] if len(target_embed_data) == 1 else target_embed_data[i])
        for i in range(len(train_embed_data))
    ]

    domain_pairs = train_pair_data[domain_idx]
    target_pairs = target_pair_data[domain_idx]

    if domain_pairs.size(0) == 0 or target_pairs.size(0) == 0:
        return []

    # fit PCA ONCE on target pairs using torch
    target_mean = target_pairs.mean(dim=0, keepdim=True)
    target_centered = target_pairs - target_mean
    max_rank = min(target_centered.size(0), target_centered.size(1))
    r = min(pca_n, max_rank)
    if r == 0:
        return []

    # V[:, :r] are principal directions
    _, _, V = torch.pca_lowrank(target_centered, q=r, center=False)
    pcs = V[:, :r]   # shape: (dim, r)

    # project source + target into target PCA space
    target_pairs_pca = (target_pairs - target_mean) @ pcs   # (Nt, r)
    source_pairs_pca = (domain_pairs - target_mean) @ pcs   # (Ns, r)

    # baseline coverage: target -> source nearest-neighbor distance
    dists = torch.cdist(target_pairs_pca, source_pairs_pca)   # (Nt, Ns)
    min_dists = dists.min(dim=1).values                       # (Nt,)
    base_coverage = min_dists.mean()

    heap = []

    for j, other_pairs in enumerate(train_pair_data):
        if j == domain_idx or other_pairs.size(0) == 0:
            continue

        # convert this domain's embeddings into the target-based PCA space
        other_pairs_pca = (other_pairs - target_mean) @ pcs   # (No, r)

        # go through each pair in this point and see if/how much it improves target coverage
        for i_pair in range(other_pairs_pca.size(0)):
            z = other_pairs_pca[i_pair].unsqueeze(0)                 # (1, r)
            d_x = torch.cdist(target_pairs_pca, z).squeeze(1)        # (Nt,)
            new_min = torch.minimum(min_dists, d_x)
            new_coverage = new_min.mean()
            gain = (base_coverage - new_coverage).item()

            heapq.heappush(heap, (-gain, j, i_pair))

    if not heap:
        return []

    top_k = heapq.nsmallest(k, heap)
    return [(domains[j], i) for _, j, i in top_k]

def select_best_raw_coverage_pairs_gpu2(domain_idx, train_embed_data, target_embed_data, domains, k=100, device='cuda', batch_size=2048):
    print("\n\nSELECTING RAW EMBEDDING COVERAGE GPU (pairwise pipeline)\n\n")
    with torch.no_grad():
        train_embed_data = [
            torch.as_tensor(x, device=device, dtype=torch.float32)
            for x in train_embed_data
        ]
        target_embed_data = [
            torch.as_tensor(x, device=device, dtype=torch.float32)
            for x in target_embed_data
        ]

        def to_unit_pairs(X: torch.Tensor) -> torch.Tensor:
            X = torch.nn.functional.normalize(X, p=2, dim=1)
            pairs3d = X.view(-1, 2, X.size(1))
            pairs = pairs3d.mean(dim=1)
            pairs = torch.nn.functional.normalize(pairs, p=2, dim=1)
            return pairs

        # Build pair embeddings in original 768-d space
        train_pair_data = [to_unit_pairs(x) for x in train_embed_data]
        target_pair_data = [
            to_unit_pairs(target_embed_data[0] if len(target_embed_data) == 1 else target_embed_data[i])
            for i in range(len(train_embed_data))
        ]

        domain_pairs = train_pair_data[domain_idx]
        target_pairs = target_pair_data[domain_idx]

        if domain_pairs.size(0) == 0 or target_pairs.size(0) == 0:
            return []

        # Baseline coverage: target -> current in-domain source nearest-neighbor distance
        dists = torch.cdist(target_pairs, domain_pairs)   # (Nt, Ns)
        min_dists = dists.min(dim=1).values               # (Nt,)
        base_coverage = min_dists.mean()

        heap = []

        for j, other_pairs in enumerate(train_pair_data):
            if j == domain_idx or other_pairs.size(0) == 0:
                continue

            # Process candidate pairs in batches to avoid huge cdist matrices
            for start in range(0, other_pairs.size(0), batch_size):
                end = min(start + batch_size, other_pairs.size(0))
                cand_batch = other_pairs[start:end]       # (B, 768)

                # Distance from every target pair to every candidate in this batch
                batch_dists = torch.cdist(target_pairs, cand_batch)  # (Nt, B)

                # Coverage after adding each candidate independently
                new_min = torch.minimum(min_dists.unsqueeze(1), batch_dists)
                new_coverage = new_min.mean(dim=0)                  # (B,)

                gains = base_coverage - new_coverage                # (B,)

                for local_idx, gain in enumerate(gains):
                    i_pair = start + local_idx
                    heapq.heappush(heap, (-gain.item(), j, i_pair))

        if not heap:
            return []

        top_k = heapq.nsmallest(k, heap)
        return [(domains[j], i) for _, j, i in top_k]

def select_best_pairs_gpu(domain_idx, train_embed_data, target_embed_data, domains, k=100, device='cuda'):
    print("\n\nSELECTING CS GPU\n\n")
    # move embeddings to GPU
    train_embed_data = [torch.tensor(x, device=device, dtype=torch.float64) for x in train_embed_data]
    target_embed_data = [torch.tensor(x, device=device, dtype=torch.float64) for x in target_embed_data]

    # get current domain embeddings and target mean
    domain_train = train_embed_data[domain_idx]
    domain_train = torch.nn.functional.normalize(domain_train, p=2, dim=1)
    orig_mean = domain_train.mean(dim=0)

    domain_target = target_embed_data[0] if len(target_embed_data) == 1 else target_embed_data[domain_idx]
    domain_target = torch.nn.functional.normalize(domain_target, p=2, dim=1)
    target_mean = domain_target.mean(dim=0)

    # compute gain using cosine sim
    sim_orig = torch_cosine_similarity(orig_mean.unsqueeze(0), target_mean.unsqueeze(0)).item()

    heap = []
    for j, other_train in enumerate(train_embed_data):
        if j == domain_idx:
            continue

        # normalize the data from this domain
        other_train  = torch.nn.functional.normalize(other_train, p=2, dim=1)
    
        # reshape into (num_pairs, 2, dim)
        pairs = other_train.view(-1, 2, other_train.size(1))  # shape: (num_pairs, 2, dim)

        # sum the two unit rows in each pair (treat as adding two samples)
        pair_sums = pairs.sum(dim=1)  # shape: (num_pairs, dim)

        # calculate the new centroids for each pair to get gain
        new_means = (domain_train.sum(dim=0) + pair_sums) / (domain_train.size(0) + 2)
        sim_new = torch_cosine_similarity(new_means, target_mean.unsqueeze(0).expand_as(new_means))
        gains = sim_new - sim_orig  # shape: (num_pairs,)

        # push to heap (on CPU)
        for i in range(len(gains)):
            heapq.heappush(heap, (-gains[i].item(), j, i))

    top_k = heapq.nsmallest(k, heap)
    #print(f"\n\n\nBEST SAMPLE INDICES: {[(domains[j], i) for _, j, i in top_k]}\n\n\n")
    #print(f"\n\n\nGAINS: {[x for x, j, i in top_k]}\n\n\n")
    return [(domains[j], i) for _, j, i in top_k]


def coral_loss(mu_src, Cov_src, mu_tgt, Cov_tgt, l=0.01):
    """
    mu_s: (d,) or (b,d)
    C_s:  (d,d) or (b,d,d)
    mu_t: (d,)
    C_t:  (d,d)
    returns: scalar if unbatched, else (b,)
    """
    # mean term
    mean_term = ((mu_src - mu_tgt)**2).sum(dim=-1)

    # cov term (Frobenius)
    # handle (b,d,d) - (d,d) broadcasting
    cov_diff = Cov_src - Cov_tgt
    cov_term = (cov_diff * cov_diff).sum(dim=(-2, -1))

    return mean_term + l * cov_term


def select_best_var_pairs_gpu2(domain_idx, train_embed_data, target_embed_data, domains, k=100, device='cuda'):
    print("\n\nSELECTING CS GPU (centroid shift with pairwise pipeline)\n\n")
    # move embeddings to GPU
    train_embed_data  = [torch.tensor(x, device=device, dtype=torch.float32) for x in train_embed_data]
    target_embed_data = [torch.tensor(x, device=device, dtype=torch.float32) for x in target_embed_data]

    # helper: singleton -> adjacent-pair average -> row-normalize each pair
    def to_unit_pairs(X):
        # row-normalize singletons
        X = torch.nn.functional.normalize(X, p=2, dim=1)
        # reshape to pairs and average adjacent singletons
        pairs3d = X.view(-1, 2, X.size(1))  # shape: (num_pairs, 2, dim)
        pairs = pairs3d.mean(dim=1)         # shape: (num_pairs, dim)
        # row-normalize each pair vector
        pairs = torch.nn.functional.normalize(pairs, p=2, dim=1)
        return pairs

    # build unit pair sets (train + target)
    train_pair_data  = [to_unit_pairs(x) for x in train_embed_data]
    # if target has a single blob, reuse it; else take per-domain
    target_pair_data = [to_unit_pairs(target_embed_data[0] if len(target_embed_data) == 1 else target_embed_data[i])
                        for i in range(len(train_embed_data))]

    # isolate domain and target pairs, return nothing if either are empty
    domain_pairs = train_pair_data[domain_idx]
    n_dom_pairs = domain_pairs.size(0)
    target_pairs = target_pair_data[domain_idx]
    n_tgt_pairs = target_pairs.size(0)
    if n_dom_pairs == 0 or n_tgt_pairs == 0:
        return []

    # current domain: original mean and variance (from unit pairs)
    orig_sum = domain_pairs.sum(dim=0)
    orig_sum_sq = (domain_pairs * domain_pairs).sum(dim=0)
    orig_mean = orig_sum / n_dom_pairs
    orig_var = orig_sum_sq / n_dom_pairs - orig_mean * orig_mean
    #orig_S = domain_pairs.T @ domain_pairs # use for updates

    # current domain: target mean and variance (from unit pairs)
    target_sum = target_pairs.sum(dim=0)
    target_sum_sq = (target_pairs * target_pairs).sum(dim=0)
    target_mean = target_sum / n_tgt_pairs
    target_var = target_sum_sq / n_tgt_pairs - target_mean * target_mean
    
    def coral_diag_loss(mean_src, var_src, mean_tgt, var_tgt, l=0.01):
        mean_term = ((mean_src - mean_tgt)**2).mean(dim=-1)
        var_term  = ((var_src - var_tgt)**2).mean(dim=-1)
        return mean_term + l * var_term

    orig_loss = coral_diag_loss(orig_mean, orig_var, target_mean, target_var)

    # --- rank pairs from other domains by centroid-shift gain ---
    heap = []
    for j, other_pairs in enumerate(train_pair_data):
        if j == domain_idx or other_pairs.size(0) == 0:
            continue

        x = other_pairs                     # (m,d)
        x2 = x * x # (m,d)

        # new means and variances
        new_sum  = orig_sum.unsqueeze(0) + x          # (m,d)
        new_sum_sq = orig_sum_sq.unsqueeze(0) + x2
        new_mean = new_sum / (n_dom_pairs + 1)
        new_var = new_sum_sq / (n_dom_pairs + 1) - new_mean * new_mean

        new_loss = coral_diag_loss(new_mean, new_var, target_mean, target_var)

        # gain = decrease in distance
        gains = orig_loss - new_loss

        for i_pair in range(gains.numel()):
            heapq.heappush(heap, (-gains[i_pair].item(), j, i_pair))

    if not heap:
        return []

    top_k = heapq.nsmallest(k, heap)

    # debug
    #print(f"\n\n\nBEST SAMPLE INDICES: {[(domains[j], i) for _, j, i in top_k]}\n\n\n")
    #print(f"\n\n\nGAINS: {[x for x, j, i in top_k]}\n\n\n")

    return [(domains[j], i) for _, j, i in top_k]


def select_best_med_pairs_gpu2(domain_idx, train_embed_data, target_embed_data, domains, k=100, device='cuda'):
    print("\n\nSELECTING MEDOID SAMPLE GPU (medoid shift with pairwise pipeline)\n\n")
    # move embeddings to GPU
    train_embed_data  = [torch.tensor(x, device=device, dtype=torch.float64) for x in train_embed_data]
    target_embed_data = [torch.tensor(x, device=device, dtype=torch.float64) for x in target_embed_data]

    # --- helper: singleton -> adjacent-pair average -> row-normalize each pair ---
    def to_unit_pairs(X: torch.Tensor) -> torch.Tensor:
        # row-normalize singletons
        X = torch.nn.functional.normalize(X, p=2, dim=1)
        # reshape to pairs and average adjacent singletons
        pairs3d = X.view(-1, 2, X.size(1))  # shape: (num_pairs, 2, dim)
        pairs = pairs3d.mean(dim=1)         # shape: (num_pairs, dim)
        # row-normalize each pair vector
        pairs = torch.nn.functional.normalize(pairs, p=2, dim=1)
        return pairs

    def calculate_medoid(vectors):
        """
        Calculates the medoid of a collection of vectors (N, D).
        """
        # 1. Compute pairwise distances (N, N)
        dist_matrix = torch.cdist(vectors, vectors, p=2) # Euclidean distance
        
        # 2. Sum distances for each vector (N,)
        total_distances = dist_matrix.sum(dim=1)
        
        # 3. Find index of the vector with the minimum total distance
        medoid_index = torch.argmin(total_distances)
        
        return vectors[medoid_index]

    # --- build unit pair sets (train + target) ---
    train_pair_data  = [to_unit_pairs(x) for x in train_embed_data]
    # if target has a single blob, reuse it; else take per-domain
    target_pair_data = [to_unit_pairs(target_embed_data[0] if len(target_embed_data) == 1 else target_embed_data[i])
                        for i in range(len(train_embed_data))]

    # current domain: orig mean and target mean (both from unit pairs)
    domain_pairs = train_pair_data[domain_idx]
    #orig_sum = domain_pairs.sum(dim=0)
    #orig_mean = orig_sum / domain_pairs.size(0)
    orig_med = calculate_medoid(domain_pairs)

    target_pairs = target_pair_data[domain_idx]
    #target_mean = target_pairs.mean(dim=0)
    target_med = calculate_medoid(target_pairs)
    # NOTE: no need to normalize target_mean for cosine; cosine_similarity divides by norms internally.
    # Direction from train medoid to val medoid
    direction = target_med - orig_med                    # (D,)
    dir_norm = direction.norm(p=2)
    direction_unit = direction / dir_norm                # (D,)

    # --- rank pairs from other domains by directional score ---
    # s(x) = cos(x - m_train, m_val - m_train)
    heap = []
    for j, other_pairs in enumerate(train_pair_data):
        if j == domain_idx or other_pairs.numel() == 0:
            continue

        # Center candidates relative to orig_med
        centered = other_pairs - orig_med.unsqueeze(0)   # (N_other, D)

        # Vectorized cosine similarity:
        # cosine_similarity(A, b) expects b broadcastable to A
        gains = torch_cosine_similarity(centered, direction_unit.unsqueeze(0), dim=1)  # (N_other,)

        # push to heap (CPU scalars). Negate to use nsmallest for max-k.
        # If you want to avoid pushing *all* pairs, you can do a local top-k per domain first.
        gains_cpu = gains.detach().to("cpu")
        for i_pair in range(gains_cpu.numel()):
            heapq.heappush(heap, (-float(gains_cpu[i_pair]), j, i_pair))

    if not heap:
        return []

    top_k = heapq.nsmallest(k, heap)
    return [(domains[j], i) for _, j, i in top_k]


""" KDE 1D METHOD (SLOW) """
def compute_gain_kde_1d(current_train, candidate_pair, orig_log_densities, kde_models):
    # compute the probability of the training centroid with this new sample
    new_train = np.vstack([current_train, candidate_pair])
    new_mean = np.mean(new_train, axis=0)
    new_log_densities = np.array([kde.score_samples([[new_mean[i]]])[0] for i, kde in enumerate(kde_models)])

    # compute the probability "gain" of this new sample
    return np.mean(new_log_densities - orig_log_densities)

def select_best_pairs_kde_1d(domain_index, train_embed_data, target_embed_data, domains, k=100, bandwidth=0.5):
    # train and target dataset embeddings for this domain
    domain_train = train_embed_data[domain_index]
    domain_target_singleton = target_embed_data[domain_index]
    domain_target = group_singleton_embeddings_to_pairwise(domain_target_singleton)

    # get 1D KDE for each dimension on pairwise target data 
    kde_models = []
    for dim in range(domain_target.shape[1]): # 768 for distilbert
        dim_data = domain_target[:, dim].reshape(-1, 1)
        kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(dim_data)
        kde_models.append(kde)
    assert len(kde_models) == 768

    # compute probability of this training data's centroid
    orig_mean = np.mean(domain_train, axis=0)
    orig_log_densities = np.array([kde.score_samples([[orig_mean[i]]])[0] for i, kde in enumerate(kde_models)])

    # rank candidate samples based on KDE gain
    heap = []
    for j, other_train in enumerate(train_embed_data):
        if j == domain_index:
            continue
        
        # loop through the pairs of another category, rank them based on KDE scores
        num_pairs = len(other_train) // 2
        for i in range(num_pairs):
            candidate_pair = get_pair_vector(other_train, i)
            gain = compute_gain_kde_1d(domain_train, candidate_pair, orig_log_densities, kde_models)
            heapq.heappush(heap, (-gain, j, i))

    # return top-k closest samples to target dataset
    if k is None:
        k = len(heap)
    top_k = heapq.nsmallest(k, heap)
    return [(domains[j], i) for _, j, i in top_k]

""" KDE-2D METHOD (BLOB) """
def _pairwise_avg(singletons):
    # (2m, d) -> (m, d)
    return np.mean(singletons.reshape(-1, 2, singletons.shape[1]), axis=1)

def _silverman_bandwidth_2d(X2):
    # one scalar bandwidth for sklearn.neighbors.KernelDensity
    n = len(X2)
    if n < 5:
        return 1.0
    s = np.std(X2, axis=0, ddof=1)
    s_robust = np.minimum(s, iqr(X2, axis=0) / 1.34)
    sigma = np.mean(s_robust)
    # Silverman/Scott-ish scalar rule for 2D
    return 1.06 * sigma * n ** (-1/5)

def select_best_pairs_kde_2d(domain_idx, train_embed_data, target_embed_data, domains, k=100, reducer='pca', pca_n=20, bandwidth=None):
    print("\n\nSELECTING KDE 2D\n\n")
    """
    Fast 'blob' selector:
      1) make pairwise embeddings for target domain (2m,d) -> (m,d)
      2) reduce to 2D (PCA over target only; apply to everyone)
      3) fit 2-D KDE on target2d
      4) score each candidate pair by KDE log-density in 2D
    """
    # pairwise target for this domain
    tgt_single = target_embed_data[domain_idx]
    tgt_pairs = _pairwise_avg(tgt_single)  # (m, d)

    # dimension reduction trained on target, applied to all
    if reducer == 'pca':
        # pre-whiten with PCA to pca_n first, then to 2
        if pca_n and pca_n > 2 and pca_n < tgt_pairs.shape[1]:
            pca_hi = PCA(n_components=pca_n, svd_solver='randomized').fit(tgt_pairs)
            Zt = pca_hi.transform(tgt_pairs)
            pca2 = PCA(n_components=2, svd_solver='full').fit(Zt)
            tgt2d = pca2.transform(Zt)
            def to2d(X):
                return pca2.transform(pca_hi.transform(X))
        else:
            pca2 = PCA(n_components=2, svd_solver='randomized').fit(tgt_pairs)
            tgt2d = pca2.transform(tgt_pairs)
            def to2d(X): return pca2.transform(X)
    else:
        raise ValueError("Unrecognized reducer for KDE calcualtion")

    # automative bandwidth calculation via silverman bandwidth
    if bandwidth is None:
        bandwidth = _silverman_bandwidth_2d(tgt2d)
        bandwidth = max(bandwidth, 1e-3)

    # 2-D KDE on reduced target data
    kde2 = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(tgt2d)

    # score candidates
    heap = []
    for j, other_train in enumerate(train_embed_data):
        if j == domain_idx:
            continue
        #num_pairs = len(other_train) // 2
        # form pairwise vectors for this donor domain
        cand_pairs = np.mean(other_train.reshape(-1, 2, other_train.shape[1]), axis=1)
        cand2d = to2d(cand_pairs)  # (num_pairs, 2)
        logp = kde2.score_samples(cand2d)  # higher = more inside the blob
        # push to heap
        for i, lp in enumerate(logp):
            heapq.heappush(heap, (-lp, j, i))  # max-heap by log-density

    top_k = heapq.nsmallest(k, heap)
    return [(domains[j], i) for _, j, i in top_k]

    
""" MMD METHOD """
def compute_mmd(X, Y, gamma=1.0):
    """
    Compute MMD between X and Y using RBF kernel.
    X: (n, d)
    Y: (m, d)
    """
    K_xx = rbf_kernel(X, X, gamma=gamma)
    K_yy = rbf_kernel(Y, Y, gamma=gamma)
    K_xy = rbf_kernel(X, Y, gamma=gamma)

    mmd = K_xx.mean() + K_yy.mean() - 2 * K_xy.mean()
    return mmd

def compute_mmd_gain(train_data, target_data, candidate_pair, orig_mmd, gamma=1.0):
    # return change in MMD when candidate is added to train_data
    new_train = np.vstack([train_data, candidate_pair])
    new_mmd = compute_mmd(new_train, target_data, gamma=gamma)
    return orig_mmd - new_mmd  # lower mmd => higher gain

def select_best_pairs_mmd(domain_idx, train_embed_data, target_embed_data, domains, k=100, gamma=None):
    # train and target dataset embeddings for this category
    domain_train = train_embed_data[domain_idx]
    domain_target_singleton = target_embed_data[domain_idx]
    domain_target = group_singleton_embeddings_to_pairwise(domain_target_singleton)

    # default gamma value to 1/EMBED_DIM
    if gamma == None:
        gamma = 1.0 / (domain_target.shape[1])

    # compute mmd with training data by itself
    orig_mmd = compute_mmd(domain_train, domain_target, gamma=gamma)

    heap = []
    for j, other_train in enumerate(train_embed_data):
        if j == domain_idx:
            continue

        # loop through all the pairs of another category
        num_pairs = len(other_train) // 2
        for i in range(num_pairs):
            candidate_pair = get_pair_vector(other_train, i)
            gain = compute_mmd_gain(domain_train, domain_target, candidate_pair, orig_mmd, gamma=gamma)
            heapq.heappush(heap, (-gain, j, i))  # max-heap => negative gain

    # get top-k samples closest to target dataset
    top_k = heapq.nsmallest(k, heap)
    return [(domains[j], i) for _, j, i in top_k]

# Use embeddings to find the nearest neighbors to fill the budget (plm nn)
def get_distribution_aware_samples(data, domains, budget, train_embed_path, target_embed_path, single_domain=None, metric='cs'):
    # TODO: add distribution-aware oversampling functionality?
        # idea: auto set to # all samples, run best selection, then oversample as I do here
    # for now: if budget excedes the number of all samples, just oversample and return
    #if budget >= sum([len(data[dom][0]) for dom in domains]):
        #for dom in domains:
            #result_data[dom] = data[dom][0] * (budget // len(data[dom][0])) + data[dom][0][:budget % len(data[dom][0])]
        #return result_data

    # get embedding data for trainset
    train_embed_data = []
    if isinstance(train_embed_path, str):
        for dom in domains:
            embed_file = os.path.join(train_embed_path, "all_" + dom + ".pt")
            train_embed_data.append(torch.load(embed_file).numpy())
    else:
        train_embed_data = train_embed_path

    # normalize train embeddings (cosine sim only)
    """
    if metric == 'cs':
        for i in range(len(train_embed_data)):
            norms = np.linalg.norm(train_embed_data[i], axis=1, keepdims=True)
            normalized_vectors = train_embed_data[i] / norms
            train_embed_data[i] = normalized_vectors
    """

    # get embedding data for target set 
    target_embed_data = []
    if isinstance(target_embed_path, str):
        for dom in domains:
            embed_file = os.path.join(target_embed_path, "all_" + dom + ".pt")
            target_embed_data.append(torch.load(embed_file).numpy())
    else:
        target_embed_data = target_embed_path

    # normalize target embeddings (cosine sim only)
    """
    if metric == 'cs':
        for i in range(len(target_embed_data)):
            norms = np.linalg.norm(target_embed_data[i], axis=1, keepdims=True)
            normalized_vectors = target_embed_data[i] / norms
            target_embed_data[i] = normalized_vectors
    """
    # create new training data
    result_data = {}

    domains_to_run = [dom for dom in domains if dom == single_domain or single_domain is None] # optimize when only one domain's data is needed
    for idx, dom in enumerate(domains_to_run):
        #original_lines = data[dom][0].copy()  # start with original
        # just sample for the original data
        if len(data[dom][0]) >= budget:
            result_data[dom] = random.sample(data[dom][0], budget)
        # get enough samples to augment to the budget
        else:
            result_data[dom] = data[dom][0].copy()
            TARGET = budget - len(result_data[dom])
            domain_idx = domains.index(dom) if len(train_embed_data) > 1 else 0
            print(f"Augmenting {dom} with {TARGET} samples")

            #use correct selection technique based on the provided metric
            if metric == 'cs':
                best_pairs = select_best_pairs_gpu2(domain_idx, train_embed_data, target_embed_data, domains, k=TARGET)
            if metric == 'cs2': # second-order: mean + variance
                best_pairs = select_best_var_pairs_gpu2(domain_idx, train_embed_data, target_embed_data, domains, k=TARGET)
            if metric == 'med': # medoid shift
                best_pairs = select_best_med_pairs_gpu2(domain_idx, train_embed_data, target_embed_data, domains, k=TARGET)
            if metric == 'pca':
                #best_pairs = select_best_pca_pairs_gpu2(domain_idx, train_embed_data, target_embed_data, domains, k=TARGET)
                best_pairs = select_best_raw_coverage_pairs_gpu2(domain_idx, train_embed_data, target_embed_data, domains, k=TARGET)
            elif metric == 'kde':
                # TODO: Optimize this somehow (can optimize for non-dynamic, but how to do so for dynamic?)
                best_pairs = select_best_pairs_kde_1d(domain_idx, train_embed_data, target_embed_data, domains, k=TARGET)
            elif metric == 'mmd':
                # TODO: Optimize this somehow (can optimize for non-dynamic, but how to do so for dynamic?)
                # idea: use subset of target data if its large
                # idea: use linear-time approximation of mmd 
                best_pairs = select_best_pairs_mmd(domain_idx, train_embed_data, target_embed_data, domains, k=TARGET)
            elif metric == 'kde2d':
                best_pairs = select_best_pairs_kde_2d(domain_idx, train_embed_data, target_embed_data, domains, k=TARGET)
            elif metric == 'dr':
                best_pairs, info = select_best_pairs_dr(
                    domain_idx=domain_idx,
                    train_embed_data=train_embed_data,
                    target_embed_data=target_embed_data,
                    domains=domains,
                    k=TARGET,
                    device="cuda",
                    beta_mode="logistic",
                    beta_model=None,
                    beta_clip=5,
                    beta_shrink=0.1,
                    return_details=True
                )
                #print('DOUBLY ROBUST INFO: ')
                #print(info)
            else:
                best_pairs = select_best_pairs(domain_idx, train_embed_data, target_embed_data, domains, k=TARGET)

            # collect the corresponding samples
            # TEMP: collect donor dom list
            #donor_dom_list = []
            for donor_dom, pair_idx in best_pairs:
                donor_line = data[donor_dom][0][pair_idx]
                result_data[dom].append(donor_line)
                #donor_dom_list.append(donor_dom)
            print(f"Augmented {dom} with {len(best_pairs)} best-matching pairs from other categories. Total pairs: {len(result_data[dom])}")

            # TEMP: write selected donor doms to file
            """
            donor_dom_counts = Counter(donor_dom_list)
            with open(f"{dom}_selected_tv.txt", "a") as f:
                f.write(f"~~~start~~~\n")
                for dm, ct in donor_dom_counts.items():
                    f.write(f"{dm}: {ct}\n")
                f.write(f"~~~end~~~\n")"""

            # oversample to fill the rest of the budget if needed
            if len(result_data[dom]) < budget:
                result_data[dom] = result_data[dom] * (budget // len(result_data[dom])) + result_data[dom][:budget % len(result_data[dom])]
                print(f"Oversampled {dom} to fill the budget. Total pairs: {len(result_data[dom])}")

    return result_data


def get_k_center_greedy_samples(data, domains, budget, train_embed_path, single_domain=None, use_ot=False):
    # get embedding data for trainset
    train_embed_data = []
    if isinstance(train_embed_path, str):
        for dom in domains:
            embed_file = os.path.join(train_embed_path, "all_" + dom + ".pt")
            train_embed_data.append(torch.load(embed_file).numpy())
    else:
        train_embed_data = train_embed_path

    # create new training data
    result_data = {}

    domains_to_run = [dom for dom in domains if dom == single_domain or single_domain is None] # optimize when only one domain's data is needed
    for idx, dom in enumerate(domains_to_run):
        #original_lines = data[dom][0].copy()  # start with original
        # just sample for the original data
        if len(data[dom][0]) >= budget:
            result_data[dom] = random.sample(data[dom][0], budget)
        # get enough samples to augment to the budget
        else:
            result_data[dom] = data[dom][0].copy()
            TARGET = budget - len(result_data[dom])
            print(f"Augmenting {dom} with {TARGET} samples")

            # get k-center greedy samples for this domain 
            # from out-of-domain pool of other samples
            domain_idx = domains.index(dom) if len(train_embed_data) > 1 else 0
            if use_ot:
                print("...using OT method")
                best_pairs = select_best_pairs_ot(domain_idx, train_embed_data, domains=domains, k=TARGET, diversify=False)
            else:
                print("...using KCG method")
                best_pairs = select_best_pairs_kcg(domain_idx, train_embed_data, domains, k=TARGET)

            # collect the corresponding samples
            # TEMP: collect donor-doms
            #donor_dom_list = []
            for donor_dom, pair_idx in best_pairs:
                donor_line = data[donor_dom][0][pair_idx]
                result_data[dom].append(donor_line)
                #donor_dom_list.append(donor_dom)
            print(f"Augmented {dom} with {len(best_pairs)} best-matching pairs from other categories. Total pairs: {len(result_data[dom])}")

            # TEMP: save donor dom list to file
            """donor_dom_counts = Counter(donor_dom_list)
            with open(f"{dom}_selected_kcg.txt", "a") as f:
                f.write(f"~~~start~~~\n")
                for dm, ct in donor_dom_counts.items():
                    f.write(f"{dm}: {ct}\n")
                f.write(f"~~~end~~~\n")"""

            # oversample to fill the rest of the budget if needed
            if len(result_data[dom]) < budget:
                result_data[dom] = result_data[dom] * (budget // len(result_data[dom])) + result_data[dom][:budget % len(result_data[dom])]
                print(f"Oversampled {dom} to fill the budget. Total pairs: {len(result_data[dom])}")

    return result_data


def get_label_aware_samples(data, domains, budget, train_embed_path, target_embed_path=None, method='kcg', single_domain=None, in_domain_labels=True, out_domain_labels=True):
    # get embedding data for trainset
    train_embed_data = []
    if isinstance(train_embed_path, str):
        for dom in domains:
            embed_file = os.path.join(train_embed_path, "all_" + dom + ".pt")
            train_embed_data.append(torch.load(embed_file).numpy())
    else:
        train_embed_data = train_embed_path

    def _labels_from_lines(lines):
        # robust to tabs/spaces; label is last token
        return np.array([int(ln.strip().split()[-1]) for ln in lines], dtype=int)

    def _singleton_mask_from_pair_mask(pair_mask):
        # each pair corresponds to 2 singleton rows (adjacent)
        return np.repeat(pair_mask, 2)

    # label-aware partition: lines + singleton embeddings
    match_data = {}
    nonmatch_data = {}
    matches_embed_data = []
    nonmatches_embed_data = []

    for i, dom in enumerate(domains):
        train_lines = data[dom][0]
        y = _labels_from_lines(train_lines)
        pos_mask = (y == 1)
        neg_mask = (y == 0)

        # split lines (pair-level)
        match_data[dom] = [ln for ln, m in zip(train_lines, pos_mask) if m]
        nonmatch_data[dom] = [ln for ln, m in zip(train_lines, neg_mask) if m]

        # split embeddings (singleton-level)
        E = train_embed_data[i] # (2*num_pairs, D)
        pos_single_mask = _singleton_mask_from_pair_mask(pos_mask)
        neg_single_mask = _singleton_mask_from_pair_mask(neg_mask)

        matches_embed_data.append(E[pos_single_mask])
        nonmatches_embed_data.append(E[neg_single_mask])

    # SANITY CHECK: did we partition embeddings correctly?
    for i, dom in enumerate(domains):
        assert len(matches_embed_data[i]) == 2 * len(match_data[dom])
        assert len(nonmatches_embed_data[i]) == 2 * len(nonmatch_data[dom])

    # for target/validation data (TVDF)
    target_embed_data = []
    if target_embed_path is not None:
        if isinstance(target_embed_path, str):
            for dom in domains:
                embed_file = os.path.join(target_embed_path, "all_" + dom + ".pt")
                target_embed_data.append(torch.load(embed_file).numpy())
        else:
            target_embed_data = target_embed_path
        
        #label-aware partition: lines + singleton embeddings for val data (TVDF)
        target_matches_embed_data = []
        target_nonmatches_embed_data = []

        for i in range(len(target_embed_data)):
            # determine the domain based on what target embeddings are available
            if len(target_embed_data) == 1:
                dom = domains[i] if single_domain is None else single_domain
            else:
                dom = domains[i]

            # get validation data and isolate matches and nonmatches
            val_lines = data[dom][1]
            yv = _labels_from_lines(val_lines)
            pos_mask = (yv == 1)
            neg_mask = (yv == 0)

            Ev = target_embed_data[i]  # expected (2*num_val_pairs, D)
            pos_single_mask = _singleton_mask_from_pair_mask(pos_mask)
            neg_single_mask = _singleton_mask_from_pair_mask(neg_mask)

             # optional: fail fast with a clear message
            assert Ev.shape[0] == pos_single_mask.shape[0], (
                f"Target embed/label mismatch for dom={dom}: "
                f"Ev rows={Ev.shape[0]} vs mask={pos_single_mask.shape[0]} "
                f"(#val_pairs={len(val_lines)})"
            )

            target_matches_embed_data.append(Ev[pos_single_mask])
            target_nonmatches_embed_data.append(Ev[neg_single_mask])

    # create new training data
    result_data = {}

    domains_to_run = [dom for dom in domains if dom == single_domain or single_domain is None] # optimize when only one domain's data is needed
    for idx, dom in enumerate(domains_to_run):
        #original_lines = data[dom][0].copy()  # start with original
        # just sample for the original data
        if len(data[dom][0]) >= budget:
            result_data[dom] = random.sample(data[dom][0], budget)
        # get enough samples to augment to the budget
        else:
            result_data[dom] = data[dom][0].copy()
            TARGET = budget - len(result_data[dom])
            print(f"Augmenting {dom} with {TARGET} samples")

            # get k-center greedy samples for this domain 
            # from out-of-domain pool of other samples
            domain_idx = domains.index(dom) if len(train_embed_data) > 1 else 0
            if method.startswith('kcg'):
                print("...using KCG method")
                # perform label aware KCG
                best_pairs = select_best_pairs_kcg_label_aware(domain_idx, train_embed_data, matches_embed_data, nonmatches_embed_data, domains, k=TARGET, in_domain_labels=in_domain_labels, out_domain_labels=out_domain_labels)
            elif method.startswith('tv'):
                print("...using TVDF method")
                # perform label aware TVDF
                best_pairs = select_best_pairs_gpu2_label_aware(domain_idx, train_embed_data, matches_embed_data, nonmatches_embed_data, target_embed_data, target_matches_embed_data, target_nonmatches_embed_data, domains, k=TARGET, in_domain_labels=in_domain_labels, out_domain_labels=out_domain_labels)
            else:
                print(f"METHOD {method} NOT SUPPORTED")
                exit(1)

            # collect the corresponding samples
            # TEMP: collect donor-doms
            #donor_dom_list = []
            if out_domain_labels:
                for match_lbl in [0, 1]:
                    data_source = match_data if match_lbl == 1 else nonmatch_data
                    for donor_dom, pair_idx in best_pairs[match_lbl]:
                        donor_line = data_source[donor_dom][pair_idx]
                        result_data[dom].append(donor_line)
                        #donor_dom_list.append(donor_dom)
                print(f"Augmented {dom} with {len(best_pairs[0]) + len(best_pairs[1])} best-matching pairs from other categories. Total pairs: {len(result_data[dom])}")
            else:
                for donor_dom, pair_idx in best_pairs:
                    donor_line = data[donor_dom][0][pair_idx]
                    result_data[dom].append(donor_line)
                print(f"Augmented {dom} with {len(best_pairs)} best-matching pairs from other categories. Total pairs: {len(result_data[dom])}")

            # oversample to fill the rest of the budget if needed
            if len(result_data[dom]) < budget:
                result_data[dom] = result_data[dom] * (budget // len(result_data[dom])) + result_data[dom][:budget % len(result_data[dom])]
                print(f"Oversampled {dom} to fill the budget. Total pairs: {len(result_data[dom])}")

    return result_data


def get_augmented_samples(data, domains, budget):
    result_data = {}
    for dom in domains:
        if len(data[dom][0]) >= budget:
            print(f"DONT HAVE TO AUGMENT {dom} TO {budget}")
            result_data[dom] = random.sample(data[dom][0], budget)
        else:
            result_data[dom] = augment_entity_matching_dataset(data, dom, budget)
            print(f"AUGMENTED {dom} TO {len(result_data[dom])}")


    # return augmented data for each domain
    return result_data


def distribute_global_budget(data, global_budget):
    print(f"~~~GLOBAL BUDGET REDISTRIBUTION~~~")
    
    # quick exit (e.g. global budget too big)
    sizes = {dom:len(data[dom][0]) for dom in data}
    num_total_samples = sum([sizes[dom] for dom in data])
    if global_budget >= num_total_samples:
        for dom in domains:
            print(f"{dom}: {sizes[dom]} --> {sizes[dom]} (took all)")
        return data

    # distribute samples proportionally
    allocation_distribution = {dom:0 for dom in data}
    total_num_allocated = 0
    for dom in data:
        num_dom_samples = len(data[dom][0])
        num_allocated_samples = int(global_budget * (num_dom_samples / num_total_samples))
        if num_allocated_samples > num_dom_samples:
            num_allocated_samples = num_dom_samples
        allocation_distribution[dom] = num_allocated_samples
        total_num_allocated += num_allocated_samples

    # randomly redistribute leftovers
    leftovers = global_budget - total_num_allocated
    print(f"... {leftovers} leftover samples to distribute")
    while leftovers > 0:
        dom = random.choice(list(data.keys()))
        if allocation_distribution[dom] < sizes[dom]:
            allocation_distribution[dom] += 1
            leftovers -= 1

    # randomly select data according to global budget distribution
    for dom in data:
        num_dom_samples = len(data[dom][0])
        data[dom][0] = random.sample(data[dom][0], allocation_distribution[dom])
        print(f"{dom}: {num_dom_samples} --> {allocation_distribution[dom]}")

    return data


import math, random
from collections import defaultdict

def distribute_global_budget_fair(data, global_budget, alpha=1.0, min_per_domain=0):
    """
    Fairness allocation: smaller domains get more.
    Allocation weights w_i ∝ (size_i)^(-alpha).
      alpha=0    -> equal per-domain
      alpha=0.5  -> 1/sqrt(size)
      alpha=1.0  -> 1/size (inverse)
    Caps at each domain's available sample count.
    Uses deterministic largest-fraction rounding, with overflow redistribution.
    """

    print(f"~~~GLOBAL BUDGET REDISTRIBUTION (INVERSE, alpha={alpha})~~~")

    # 1) Gather sizes and caps
    domains = list(data.keys())
    sizes = {dom: len(data[dom][0]) for dom in domains}
    caps = {dom: sizes[dom] for dom in domains}

    # Quick exit: if global_budget >= total available, just take all
    total_available = sum(sizes.values())
    if global_budget >= total_available:
        for dom in domains:
            print(f"{dom}: {sizes[dom]} --> {sizes[dom]} (took all)")
        return data

    # 2) Compute raw weights (avoid zero with tiny epsilon)
    eps = 1e-9
    weights = {dom: (sizes[dom] + eps) ** (-alpha) for dom in domains}

    # If all weights ~0 (degenerate), fall back to equal
    weight_sum = sum(weights.values())
    if weight_sum <= 0:
        weights = {dom: 1.0 for dom in domains}
        weight_sum = len(domains)

    # 3) Optional floor per domain (guarantee a minimum, but don’t exceed caps)
    floors = {dom: min(min_per_domain, caps[dom]) for dom in domains}
    floor_total = sum(floors.values())
    if floor_total > global_budget:
        # If the floor is larger than the budget, shrink floors fairly
        # by giving 1 to as many domains as budget allows (preferring small domains).
        sorted_by_size = sorted(domains, key=lambda d: sizes[d])
        floors = {dom: 0 for dom in domains}
        for dom in sorted_by_size[:global_budget]:
            floors[dom] = 1
        floor_total = global_budget

    # 4) Distribute the *remaining* budget proportionally to weights
    remaining = global_budget - floor_total
    if remaining < 0:
        remaining = 0

    # Ideal fractional allocations
    ideal = {dom: floors[dom] + remaining * (weights[dom] / weight_sum) for dom in domains}

    # 5) Integer rounding via floor + largest fractional parts, with caps
    alloc = {dom: min(int(math.floor(ideal[dom])), caps[dom]) for dom in domains}
    used = sum(alloc.values())
    leftover = global_budget - used

    # Assign leftover units to the largest fractional remainders where cap not hit
    remainders = sorted(
        [(dom, (ideal[dom] - math.floor(ideal[dom]))) for dom in domains],
        key=lambda x: x[1],
        reverse=True
    )
    i = 0
    while leftover > 0 and i < len(remainders):
        dom, _ = remainders[i]
        if alloc[dom] < caps[dom]:
            alloc[dom] += 1
            leftover -= 1
        i += 1

    # If still leftover (because many were capped), do a second pass among any with spare capacity
    if leftover > 0:
        available = [dom for dom in domains if alloc[dom] < caps[dom]]
        j = 0
        while leftover > 0 and available:
            dom = available[j % len(available)]
            if alloc[dom] < caps[dom]:
                alloc[dom] += 1
                leftover -= 1
            j += 1

    # 6) Sample per-domain according to final allocation
    for dom in domains:
        n0 = sizes[dom]
        n_take = alloc[dom]
        if n_take < n0:
            data[dom][0] = random.sample(data[dom][0], n_take)
        # else keep all
        print(f"{dom}: {n0} --> {n_take}")

    return data

def distribute_global_budget_central(data, budget, embed_fpath, EMBED_DIM=768, method='nearest'):
    print("Distributing centralized embeddings...")
    # direction of centrality (nearest or furthest)
    dir = 1 if method == 'nearest' else -1

    # get domains
    domains = list(data.keys())

    # get the embeddings
    embed_data = []
    if isinstance(embed_fpath, str):
        for dom in domains:
            embed_file = os.path.join(embed_fpath, "all_" + dom + ".pt")
            embed_data.append(torch.load(embed_file).numpy())
    else:
        embed_data = embed_fpath

    # (singleton) normalize the vectors
    for i in range(len(embed_data)):
        norms = np.linalg.norm(embed_data[i], axis=1, keepdims=True)
        embed_data[i] = embed_data[i] / np.clip(norms, a_min=1e-10, a_max=None)

    # convert form singleton -> pairwise by averaging adacent vectors (since the order is maintained and thus they form matches)
    all_embeds = []
    for i in range(len(embed_data)):
        n = len(embed_data[i])
        res = np.zeros((n//2, EMBED_DIM))
        for j in range(0, n, 2):
            res[j//2] = (embed_data[i][j] + embed_data[i][j+1])/2
        
        # NEW: L2-normalize each pair vector
        pair_norms = np.linalg.norm(res, axis=1, keepdims=True)
        pairs = res / np.clip(pair_norms, a_min=1e-10, a_max=None)
        embed_data[i] = pairs
        all_embeds.extend(pairs)
        #embed_data[i] = res

    # calculate overall mean
    centroid = np.mean(all_embeds, axis=0)

    # create a queue to order samples by closest to center
    sample_q = []
    for i, dom_embeds in enumerate(embed_data):
        for j, vec in enumerate(dom_embeds):
            sim = np.dot(centroid, vec) * dir
            heapq.heappush(sample_q, (sim, i, j))
            
    # select top BUDGET samples from the queue and replace corresponding list in "data"
    top_k = heapq.nlargest(budget, sample_q)
    new_data = {dom: [] for dom in domains}
    for _, i, j in top_k:
        new_data[domains[i]].append(data[domains[i]][0][j])
    
    # report new sample distribution
    print(f"...done. GLOBAL SAMPLE CENTRALIZED REDISTRIBUTION: ")
    for dom in domains:
        print(f"{dom}: {len(data[dom][0])} --> {len(new_data[dom])}")
        data[dom][0] = new_data[dom]

    return data

def distribute_global_budget_kmeans(data, budget, embed_fpath, EMBED_DIM=768, var_threshold=0.95, max_pca_dim=200):
    print("Distributing centralized embeddings (K-Means representative selection)...")
    # get domains
    domains = list(data.keys())

    # get the embeddings
    embed_data = []
    if isinstance(embed_fpath, str):
        for dom in domains:
            embed_file = os.path.join(embed_fpath, "all_" + dom + ".pt")
            embed_data.append(torch.load(embed_file).numpy())
    else:
        embed_data = embed_fpath

    # (singleton) normalize the vectors
    for i in range(len(embed_data)):
        norms = np.linalg.norm(embed_data[i], axis=1, keepdims=True)
        embed_data[i] = embed_data[i] / np.clip(norms, a_min=1e-10, a_max=None)

    # convert form singleton -> pairwise by averaging adacent vectors (since the order is maintained and thus they form matches)

    # --- Step 2. Convert singletons → pairwise vectors ---
    all_embeds = []
    pair_domain_indices = []  # track (domain_idx, pair_idx)
    for i in range(len(embed_data)):
        n = len(embed_data[i])
        res = np.zeros((n // 2, EMBED_DIM))
        for j in range(0, n, 2):
            res[j // 2] = (embed_data[i][j] + embed_data[i][j + 1]) / 2
        pair_norms = np.linalg.norm(res, axis=1, keepdims=True)
        pairs = res / np.clip(pair_norms, a_min=1e-10, a_max=None)
        embed_data[i] = pairs
        all_embeds.append(pairs)
        pair_domain_indices.extend([(i, k) for k in range(len(pairs))])

    all_embeds = np.vstack(all_embeds)
    num_domains = len(domains)

    # PCA to setup for K-means
    print("Running PCA with automatic variance threshold...")
    pca_full = PCA(n_components=min(EMBED_DIM, max_pca_dim))
    pca_full.fit(all_embeds)
    cumulative_var = np.cumsum(pca_full.explained_variance_ratio_)
    PCA_DIM = np.searchsorted(cumulative_var, var_threshold) + 1
    PCA_DIM = min(PCA_DIM, max_pca_dim)
    print(f"Selected {PCA_DIM} PCA components to retain ~{var_threshold*100:.1f}% variance.")

    pca = PCA(n_components=PCA_DIM)
    reduced = pca.fit_transform(all_embeds)

    # --- Step 4. Compute domain centroids for K-Means init ---
    init_centroids = []
    for i in range(num_domains):
        dom_centroid = np.mean(pca.transform(embed_data[i]), axis=0)
        init_centroids.append(dom_centroid)
    init_centroids = np.vstack(init_centroids)

    # --- Step 5. Run K-Means ---
    print(f"Running K-Means with k={num_domains} and PCA init centroids...")
    kmeans = KMeans(n_clusters=num_domains, init=init_centroids, n_init=1, max_iter=300, random_state=42)
    cluster_labels = kmeans.fit_predict(reduced)
    cluster_centers = kmeans.cluster_centers_

    # --- Step 6. Select representative samples per cluster ---
    #per_cluster_budget = max(1, budget // num_domains)
    selected_indices = set()
    new_data = {dom: [] for dom in domains}

    for cluster_id in range(num_domains):
        cluster_points = np.where(cluster_labels == cluster_id)[0]
        per_cluster_budget = int(budget * len(cluster_points) / len(all_embeds))
        if len(cluster_points) == 0:
            continue

        dists = np.linalg.norm(reduced[cluster_points] - cluster_centers[cluster_id], axis=1)
        nearest_idxs = cluster_points[np.argsort(dists)[:per_cluster_budget]]

        for global_idx in nearest_idxs:
            if global_idx in selected_indices:
                continue
            selected_indices.add(global_idx)
            dom_idx, pair_idx = pair_domain_indices[global_idx]
            dom_name = domains[dom_idx]
            if pair_idx < len(data[dom_name][0]):
                new_data[dom_name].append(data[dom_name][0][pair_idx])

    # --- Step 7. Fill remaining budget (if needed) ---
    total_selected = sum(len(v) for v in new_data.values())
    leftovers = budget - total_selected
    if leftovers > 0:
        print(f"Filling remaining {leftovers} samples randomly...")
        all_candidates = [(d, line) for d in domains for line in data[d][0]]
        remaining = np.random.choice(len(all_candidates), size=min(leftovers, len(all_candidates)), replace=False)
        for idx in remaining:
            dom, line = all_candidates[idx]
            new_data[dom].append(line)

    # --- Step 8. Log and return ---
    print(f"...done. GLOBAL SAMPLE K-MEANS REDISTRIBUTION:")
    for dom in domains:
        print(f"{dom}: {len(data[dom][0])} → {len(new_data[dom])}")
        data[dom][0] = new_data[dom]

    return data



def collect_universal_trainset(data, domains):
    print(f"BUDGET == GLOBAL BUDGET =====> Universl trainset")
    # collect all the training data
    all_train = []
    for dom in domains:
        all_train.extend(data[dom][0])
    #random.shuffle(all_train)

    # distribute to each domains training data
    updated_data = {}
    for dom in domains:
        updated_data[dom] = all_train.copy()

    return updated_data

def get_battleship_samples(data, domains, budget):
    # get seed from linux environment (default to 42 if unassigned)
    try:
        seed = int(os.getenv('SEED', '42'))
    except ValueError:
        print(f"Warning: Environment variable $SEED is set to a non-integer value: {os.getenv('SEED')}. Using default value 42 instead.")
        seed = 42

    # battleship method
    result_data = {}
    for dom in domains:
        if len(data[dom][0]) >= budget:
            # for now just random sample
            result_data[dom] = random.sample(data[dom][0], budget)
        else:
            # start with all in domain data
            result_data[dom] = data[dom][0].copy()

            # use battleship active learning (one iteration) to select the best out-of-domains samples to label
            samples_needed = budget - len(result_data[dom])
            selected_ood = TopKSelection.active_select_from_domain_dict(
                data_dict=data,
                domain=dom,      
                budget=samples_needed,  
                seed=seed
            )

            # add selected out-of-domain to updated trainset
            result_data[dom].extend(selected_ood)
    return result_data


def process_budget(task, method, budget, tokenizer, shared_model=None, lm='distilbert', single_domain=None, output_dir=None, global_budget=None, used_data=None):
    # load task configuration
    print(f"Processing {task} with budget {budget} and method {method}")
    configs = json.load(open('configs.json'))
    configs = {conf['name'] : conf for conf in configs}
    config = configs[task]
    domains = config['domains']
    
    # get the destination to write the data to
    updated_data_dest = config['updated_data_path']
    if output_dir is not None:
        updated_data_dest = os.path.join(output_dir, updated_data_dest)

    # read domain files for the domain
    if used_data is None:
        data = read_domain_files(domains, config['data_path'])
        if global_budget is not None:
            #data = distribute_global_budget(data, global_budget)
            #data = distribute_global_budget_fair(data, global_budget)

            # embedding-based
            # get training embeddings
            train_embed_path = os.path.join(output_dir, config['train_embed_path']) if output_dir is not None else config['train_embed_path']
            if not os.path.exists(train_embed_path):
                os.makedirs(train_embed_path)
            singleton_fpath = os.path.join(output_dir, 'singleton_data') if output_dir is not None else 'singleton_data'
            generate_embeddings(data, domains, train_embed_path, lm = lm, tokenizer = tokenizer, shared_model=shared_model, tmp_file_path=singleton_fpath)
            #distribute_global_budget_central(data, global_budget, train_embed_path, method='furthest')
            distribute_global_budget_kmeans(data, global_budget, train_embed_path)
    else:
        data = used_data

    # parse the method and budget, get samples accordingly    
    if method is None:
        # just used to read files and get domains
        return data, domains
    elif global_budget is not None and global_budget == budget:
        updated_data = collect_universal_trainset(data, domains)
    elif method == "spec":
        # domain specific model
        updated_data = get_specific_samples(data, domains, budget)
    elif method == "gen":
        # general model
        updated_data = get_random_samples(data, domains, budget)
    elif type(method) == str and method.startswith("nn"):
        # nearest neighbor model
        # get the train embeddings
        train_embed_path = os.path.join(output_dir, config['train_embed_path']) if output_dir is not None else config['train_embed_path']
        if not os.path.exists(train_embed_path):
            os.makedirs(train_embed_path)
        singleton_fpath = os.path.join(output_dir, 'singleton_data') if output_dir is not None else 'singleton_data'
        generate_embeddings(data, domains, train_embed_path, lm = lm, tokenizer = tokenizer, shared_model=shared_model, tmp_file_path=singleton_fpath)
        
        # calculate the nearest neighbor samples
        updated_data = get_nn_samples(data, domains, budget, train_embed_path, single_domain=single_domain)
    elif type(method) == str and (method.startswith("tv") or method.startswith("tt") or method.startswith("tm")):
        # train/target distribution fitting model
        # generate the embeddings for train and target data
        train_embed_path = os.path.join(output_dir, config['train_embed_path']) if output_dir is not None else config['train_embed_path']
        if not os.path.exists(train_embed_path):
            os.makedirs(train_embed_path)

        # TV use validset, TT use testset, TM minimizes any shift from current train centroid
        if method.startswith('tv'):
            target_embed_path = os.path.join(output_dir, config['val_embed_path']) if output_dir is not None else config['val_embed_path']
        elif method.startswith('tt'): 
            target_embed_path = os.path.join(output_dir, config['test_embed_path']) if output_dir is not None else config['test_embed_path']
        else:
            target_embed_path = os.path.join(output_dir, config['train_embed_path']) if output_dir is not None else config['train_embed_path']
        if not os.path.exists(target_embed_path):
            os.makedirs(target_embed_path)

        # generate embeddings for the needed datasets
        print(f"TARGET EMBED PATH IS {target_embed_path}")
        singleton_fpath = os.path.join(output_dir, 'singleton_data') if output_dir is not None else 'singleton_data'
        generate_embeddings(data, domains, train_embed_path, lm = lm, tokenizer = tokenizer, shared_model=shared_model, tmp_file_path=singleton_fpath)
        if train_embed_path != target_embed_path:
            ds_key = 'val' if method.startswith('tv') else 'test'
            generate_embeddings(data, domains, target_embed_path, ds=ds_key, lm = lm, tokenizer = tokenizer, shared_model=shared_model, tmp_file_path=singleton_fpath)
        
        # get the distribution aware samples
        method_parts = method.split("_")
        metric = method_parts[-1]
        if len(method_parts) == 1 or metric == "tv" or metric == "tt" or metric == "tm":
            metric = "cs"
        print(f"TT: Metric used is {metric}")
        if metric.startswith('la'):
            in_domain_labels = 'i' in metric
            out_domain_labels = 'o' in metric
            updated_data = get_label_aware_samples(data, domains, budget, train_embed_path, target_embed_path=target_embed_path, method=method, single_domain=single_domain, in_domain_labels=in_domain_labels, out_domain_labels=out_domain_labels)
        else:
            updated_data = get_distribution_aware_samples(data, domains, budget, train_embed_path, target_embed_path, single_domain=single_domain, metric=metric)
    elif type(method) == str and method.startswith("kcg"):
        # get the train embeddings
        train_embed_path = os.path.join(output_dir, config['train_embed_path']) if output_dir is not None else config['train_embed_path']
        if not os.path.exists(train_embed_path):
            os.makedirs(train_embed_path)
        singleton_fpath = os.path.join(output_dir, 'singleton_data') if output_dir is not None else 'singleton_data'
        generate_embeddings(data, domains, train_embed_path, lm = lm, tokenizer = tokenizer, shared_model=shared_model, tmp_file_path=singleton_fpath)
        use_ot = method.endswith("_ot")
        metric = method.split('_')[-1]
        label_aware = metric.startswith("la")
        if label_aware:
            in_domain_labels = 'i' in metric
            out_domain_labels = 'o' in metric
            updated_data = get_label_aware_samples(data, domains, budget, train_embed_path, method=method, single_domain=single_domain, in_domain_labels=in_domain_labels, out_domain_labels=out_domain_labels)
        else:
            updated_data = get_k_center_greedy_samples(data, domains, budget, train_embed_path, single_domain=single_domain, use_ot=use_ot)
    elif method == "aug":
        updated_data = get_augmented_samples(data, domains, budget)
    elif method == "battleship":
        updated_data = get_battleship_samples(data, domains, budget)
    else:
        print(f"COULD NOT PARSE THE METHOD, NO DATA WAS GENEREATED")
        return None, domains

    # copy files over to the destination specified by the config
    if os.path.exists(updated_data_dest):
        shutil.rmtree(updated_data_dest)
    os.makedirs(updated_data_dest)
    domains_to_write = domains if single_domain is None else [single_domain]
    for domain in domains_to_write:
        updated_trainset = os.path.join(updated_data_dest, domain + "_train.txt")
        with open(updated_trainset, "w") as ftrain:
            ftrain.writelines(updated_data[domain])
        updated_validset = os.path.join(updated_data_dest, domain + "_val.txt")
        with open(updated_validset, "w") as fval:
            fval.writelines(data[domain][1])
        updated_testset = os.path.join(updated_data_dest, domain + "_test.txt")
        with open(updated_testset, "w") as ftest:
            ftest.writelines(data[domain][2])

    # return the data dictionary and the list of domains
    return data, domains


# main program (for testing)
if __name__ == '__main__':
    task = 'wdc_category'
    budget = 20000
    method = 'aug'
    data, domains = process_budget(task, method, budget)
