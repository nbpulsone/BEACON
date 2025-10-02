# BEACON: Budget-Aware Entity Matching Across Domains
Code repository for the paper _Budget-Aware Entity Matching Across Domains_.

Entity Matching (EM) is the task of determining whether two data records refer to the same real-world object. Recent approaches employing Pretrained Language Models (PLMs) and Large Language Models (LLMs) often require large amounts of labeled data, and do not consider important distinctions between EM in different domains (e.g. product categories). We present BEACON, a distribution-aware framework for low-resource EM across domains, leveraging a dynamic training loop and a novel resampling mechanism.

This work builds on the [DITTO](https://github.com/megagonlabs/ditto/tree/master) pipeline for fine-tuning a PLM for EM. 
<img width="1779" height="1080" alt="beacon_pipeline" src="https://github.com/user-attachments/assets/1b7be7e2-3c7d-4df3-b13a-3c198b67b4f7" />

## Requirements
1. Python 3.7.13
2. For list of dependencies, see [requirements.txt](requirements.txt)

## Datasets
We derive the datasets for our experiments from [WDC](https://webdatacommons.org/largescaleproductcorpus/wdc-products/#toc5).
See the [data](data) folder for details.

## Quick Start
To train and evaluate a model for EM across domains, use the following command:
`./train_eval_em_across_domains <model> <dataset>`

For example: 
`./train_eval_em_across_domains spec wdc_category_50cc_half_seen`

The available models to run are: 
1. spec (domain-specific model)
2. gen (general model)
3. nn (nearest-neighbors model)
4. tv (train-validation distribution fitting model)
5. kcg (k-center greedy model)
6. kcg-tv (BEACON model)
Other models ensembles can be generated using hypens (e.g. spec-gen-nn)

See [configs.json](configs.json) for the available datasets and domain configurations to run.
