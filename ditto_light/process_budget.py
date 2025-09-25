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
from ditto_light.selection_methods import select_best_pairs_kcg

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
        #print(f"Read samples for {dom}: {[len(data[dom][i]) for i in range(len(data[dom]))]}")
    return data

# Oversample domain specific data to get the budget (spec)
def get_specific_samples(data, domains, budget):
    # over sample to get the correct number of samples
    result_data = {}
    for dom in domains:
        result_data[dom] = []
        if len(data[dom][0]) >= budget:
            # random sample
            result_data[dom] = random.sample(data[dom][0], budget)
        else:
            # oversample if needed
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

        for idx in closest_indices:
            donor_dom_idx, pair_idx = domain_indices[idx]
            donor_dom = domains[donor_dom_idx]
            result_data[dom].append(data[donor_dom][0][pair_idx])
    
        # oversample result data if needed
        if len(result_data[dom]) < budget:
            full = result_data[dom]
            result_data[dom] = full * (budget // len(full)) + full[:budget % len(full)]

        # log
        print(f"{dom} added nearest neighbors to reach {len(result_data[dom])} samples")
    return result_data 


# ~~~ HELPER FUNCTIONS FOR TT ~~~
def group_singleton_embeddings_to_pairwise(singleton_embeddings):
    # convert group of adjacent singleton embeddings to average of pairs
    return np.mean(singleton_embeddings.reshape(-1, 2, singleton_embeddings.shape[1]), axis=1)

def get_pair_vector(embed_list, pair_idx):
    # extract the two singleton embeddings that make up a pair
    return embed_list[2 * pair_idx: 2 * pair_idx + 2]

def compute_gain(current_train, target_mean, candidate_pair):
    # compute the centroid gain towards the targe mean from adding candidate pair to current train
    orig_mean = np.mean(current_train, axis=0)
    new_train = np.vstack([current_train, candidate_pair])
    new_mean = np.mean(new_train, axis=0)

    sim_orig = cosine_similarity(orig_mean.reshape(1, -1), target_mean.reshape(1, -1))[0, 0]
    sim_new = cosine_similarity(new_mean.reshape(1, -1), target_mean.reshape(1, -1))[0, 0]

    return sim_new - sim_orig

def select_best_pairs(domain_idx, train_embed_data, target_embed_data, domains, k=100):
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

def select_best_pairs_gpu2(domain_idx, train_embed_data, target_embed_data, domains, k=100, device='cuda'):
    # select best pairs but using GPU for optimization
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
    # NOTE: no need to normalize target_mean for cosine; F.cosine_similarity divides by norms internally.

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
            heapq.heappush(heap, (-lp, j, i))  

    top_k = heapq.nsmallest(k, heap)
    return [(domains[j], i) for _, j, i in top_k]

    
""" MMD METHOD (ALSO SLOW) """
def compute_mmd(X, Y, gamma=1.0):
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
    # get embedding data for trainset
    train_embed_data = []
    if isinstance(train_embed_path, str):
        for dom in domains:
            embed_file = os.path.join(train_embed_path, "all_" + dom + ".pt")
            train_embed_data.append(torch.load(embed_file).numpy())
    else:
        train_embed_data = train_embed_path

    # get embedding data for target set 
    target_embed_data = []
    if isinstance(target_embed_path, str):
        for dom in domains:
            embed_file = os.path.join(target_embed_path, "all_" + dom + ".pt")
            target_embed_data.append(torch.load(embed_file).numpy())
    else:
        target_embed_data = target_embed_path

    # create new training data
    result_data = {}

    domains_to_run = [dom for dom in domains if dom == single_domain or single_domain is None] # optimize when only one domain's data is needed
    for idx, dom in enumerate(domains_to_run):
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
            elif metric == 'kde':
                best_pairs = select_best_pairs_kde_1d(domain_idx, train_embed_data, target_embed_data, domains, k=TARGET)
            elif metric == 'mmd':
                best_pairs = select_best_pairs_mmd(domain_idx, train_embed_data, target_embed_data, domains, k=TARGET)
            elif metric == 'kde2d':
                best_pairs = select_best_pairs_kde_2d(domain_idx, train_embed_data, target_embed_data, domains, k=TARGET)
            else:
                best_pairs = select_best_pairs(domain_idx, train_embed_data, target_embed_data, domains, k=TARGET)

            # collect the corresponding samples
            for donor_dom, pair_idx in best_pairs:
                donor_line = data[donor_dom][0][pair_idx]
                result_data[dom].append(donor_line)
            print(f"Augmented {dom} with {len(best_pairs)} best-matching pairs from other categories. Total pairs: {len(result_data[dom])}")

            # oversample to fill the rest of the budget if needed
            if len(result_data[dom]) < budget:
                result_data[dom] = result_data[dom] * (budget // len(result_data[dom])) + result_data[dom][:budget % len(result_data[dom])]
                print(f"Oversampled {dom} to fill the budget. Total pairs: {len(result_data[dom])}")

    return result_data


def get_k_center_greedy_samples(data, domains, budget, train_embed_path, single_domain=None):
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
        # just sample for the original data
        if len(data[dom][0]) >= budget:
            result_data[dom] = random.sample(data[dom][0], budget)
        # get enough samples to augment to the budget
        else:
            result_data[dom] = data[dom][0].copy()
            TARGET = budget - len(result_data[dom])
            print(f"Augmenting {dom} with {TARGET} samples using KCG method")

            # get k-center greedy samples for this domain 
            # from out-of-domain pool of other samples
            domain_idx = domains.index(dom) if len(train_embed_data) > 1 else 0
            best_pairs = select_best_pairs_kcg(domain_idx, train_embed_data, domains, k=TARGET)

            # collect the corresponding samples
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
    # use LLM-generated synonym table to generate augments
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


def process_budget(task, method, budget, tokenizer, shared_model=None, lm='distilbert', single_domain=None, output_dir=None):
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
    data = read_domain_files(domains, config['data_path'])
    
    # parse the method and budget, get samples accordingly
    if method is None:
        # just used to read files and get domains
        return data, domains
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
        updated_data = get_distribution_aware_samples(data, domains, budget, train_embed_path, target_embed_path, single_domain=single_domain, metric=metric)
    elif type(method) == str and method.startswith("kcg"):
        # get the train embeddings
        train_embed_path = os.path.join(output_dir, config['train_embed_path']) if output_dir is not None else config['train_embed_path']
        if not os.path.exists(train_embed_path):
            os.makedirs(train_embed_path)
        singleton_fpath = os.path.join(output_dir, 'singleton_data') if output_dir is not None else 'singleton_data'
        generate_embeddings(data, domains, train_embed_path, lm = lm, tokenizer = tokenizer, shared_model=shared_model, tmp_file_path=singleton_fpath)
        updated_data = get_k_center_greedy_samples(data, domains, budget, train_embed_path, single_domain=single_domain)
    elif method == "aug":
        updated_data = get_augmented_samples(data, domains, budget)
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
    task = 'wdc_category_20cc'
    budget = 20000
    method = 'aug'
    data, domains = process_budget(task, method, budget)
