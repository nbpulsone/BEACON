import argparse
import json
import sys
import torch
import numpy as np
import random

sys.path.insert(0, "Snippext_public")

from ditto_light.dataset import DittoDataset
from ditto_light.summarize import Summarizer
from ditto_light.knowledge import *
from ditto_light.ditto import train
from ditto_light.process_budget import process_budget

from transformers import AutoTokenizer, AutoModel

# map lm name to huggingface's pre-trained model names
lm_mp = {'roberta': 'roberta-base',
         'distilbert': 'distilbert-base-uncased'}

def get_tokenizer(lm):
    if lm in lm_mp:
        return AutoTokenizer.from_pretrained(lm_mp[lm])
    else:
        return AutoTokenizer.from_pretrained(lm)
    
#TODO handle singletons w generate_embeddings better
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="wdc_category")
    parser.add_argument("--budget", type=int, default=1000)
    parser.add_argument("--method", type=str, default="nn-tv-aug")
    parser.add_argument("--run_id", type=int, default=0)
    parser.add_argument("--job_id", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--n_epochs", type=int, default=20)
    parser.add_argument("--finetuning", dest="finetuning", action="store_true")
    parser.add_argument("--save_model", dest="save_model", action="store_true")
    parser.add_argument("--logdir", type=str, default="checkpoints")
    parser.add_argument("--lm", type=str, default='distilbert')
    parser.add_argument("--fp16", dest="fp16", action="store_true")
    parser.add_argument("--da", type=str, default=None)
    parser.add_argument("--alpha_aug", type=float, default=0.8)
    parser.add_argument("--dk", type=str, default=None)
    parser.add_argument("--summarize", dest="summarize", action="store_true")
    parser.add_argument("--size", type=int, default=None)
    parser.add_argument("--dynamic_sampling", action="store_true")
    parser.add_argument("--resample_freq", type=int, default=1)
    parser.add_argument("--alpha_cosine", type=float, default=0.1)
    parser.add_argument("--late_fusion_only", action="store_true")
    parser.add_argument("--cosine_threshold", type=float, default=None)
    parser.add_argument("--keep_lower_budgets", dest="keep_lower_budgets", action="store_true")
    parser.add_argument("--ensemble_type", type=str, default="soft_vote") # options: weighted_soft_vote, soft_vote, hard_vote, select_best
    parser.add_argument("--ensemble_output_file", type=str, default=None)
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

    # set output folder if requested
    if hp.job_id is not None:
        data_dir = f'updated_data_{hp.job_id}'
        if not os.path.exists(data_dir):
            os.mkdir(data_dir)
    else:
        data_dir = None
    hp.logdir = hp.logdir + f'_{hp.job_id}' if hp.job_id is not None else hp.logdir

    # parse the budget information and process it
    general_task = hp.task
    methods = (hp.method).split('-')
    budget = hp.budget

    # get the model and tokenizer once for the provided lm (optimization)
    if hp.lm in lm_mp: # provided lm is in the map
        shared_model = AutoModel.from_pretrained(lm_mp[hp.lm])
    else:
        shared_model = AutoModel.from_pretrained(hp.lm)
    tokenizer = get_tokenizer(hp.lm)

    # do this just to get domain names
    # potential optimization: pre-generate embeddings around here
    _, domains = process_budget(general_task, None, budget, tokenizer, output_dir=data_dir)
    
    # use dynamics sampling only if its initially set or using 'NN,' 'TV,' or 'KCG' methods
    use_dynamic_sampling = hp.dynamic_sampling

    # save default method for model save/ naming purposes
    select_best_ensemble = "vote" not in hp.ensemble_type

    # if voting, save the method raw as is
    ensemble_method = hp.method if select_best_ensemble else None

    # run training and evaluation for all domains
    for task in domains:
        hp.task = task
        print(f"\n=====TRAINING: {hp.task}, ENSEMBLE METHODS: {[m for m in methods]}=====\n")

        # init everything for this domain
        # create the tag of the run
        run_tag = '%s_lm=%s_da=%s_dk=%s_su=%s_size=%s_id=%d' % (task, hp.lm, hp.da,
                hp.dk, hp.summarize, str(hp.size), hp.run_id)
        run_tag = run_tag.replace('/', '_')

        # load task configuration
        configs = json.load(open('configs.json'))
        configs = {conf['name'] : conf for conf in configs}
        config = configs[task]

        # training, validation, and testing set for this domain
        trainset = os.path.join(data_dir, config['trainset']) if data_dir is not None else config['trainset']
        validset = os.path.join(data_dir, config['validset']) if data_dir is not None else config['validset']
        testset = os.path.join(data_dir, config['testset']) if data_dir is not None else config['testset']
        config['trainset'] = trainset
        config['validset'] = validset
        config['testset'] = testset

        # go through each method, save the one with the highest f1 score
        best_val_f1 = 0.0
        best_test_f1 = 0.0
        best_method = None
        for method in methods:
            print(f"~~~{hp.task}: {method}~~~")
            hp.method = method
            hp.dynamic_sampling = use_dynamic_sampling and (method.startswith('nn') or method.startswith('tv') or method.startswith('tt') or method.startswith('tm') or method.startswith('kcg'))
            print(f"{method} USE DYNAMIC SAMPLING: {hp.dynamic_sampling}")

            # get the data for the given method
            domain_data, _ = process_budget(general_task, method, budget, tokenizer, shared_model=shared_model, lm=hp.lm, single_domain=hp.task, output_dir=data_dir)

            # summarize the sequences up to the max sequence length
            print(f"Summarizing the data...")
            if hp.summarize:
                # TODO: test summarization as a way to boost raw performance
                summarizer = Summarizer(config, lm=hp.lm, tokenizer=tokenizer)
                trainset = summarizer.transform_file(config['trainset'], max_len=hp.max_len, overwrite=True)
                validset = summarizer.transform_file(config['validset'], max_len=hp.max_len)
                testset = summarizer.transform_file(config['testset'], max_len=hp.max_len)
            print(f"...done")

            # load train set with new data
            train_dataset = DittoDataset(trainset,
                                        lm=hp.lm,
                                        max_len=hp.max_len,
                                        size=hp.size,
                                        da=hp.da,
                                        singleton=hp.dynamic_sampling or hp.late_fusion_only,
                                        tokenizer=tokenizer)
            
            # load dev/test sets
            valid_dataset = DittoDataset(validset, lm=hp.lm, singleton=hp.dynamic_sampling or hp.late_fusion_only, tokenizer=tokenizer)
            test_dataset = DittoDataset(testset, lm=hp.lm, singleton=hp.dynamic_sampling or hp.late_fusion_only, tokenizer=tokenizer)

            # train and evaluate the model! 
            val_f1, test_f1 = train(train_dataset,
                           valid_dataset,
                           test_dataset,
                           run_tag, hp, domain_data, tokenizer=tokenizer, shared_model=shared_model, init_dev_f1=best_val_f1, init_test_f1=best_test_f1, ensemble_method=ensemble_method)
            
            # update the best val f1 so only the best model is saved (ensemble)
            if select_best_ensemble and val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_test_f1 = test_f1
                best_method = method
            
        print(f"\nTRAINING COMPLETED: {task}\n")
        if select_best_ensemble and hp.ensemble_output_file is not None:
            with open(hp.ensemble_output_file, "a") as f:
                f.write(f"{task},{budget},{best_method}\n")
     