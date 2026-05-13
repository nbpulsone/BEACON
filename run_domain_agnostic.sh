#!/bin/bash
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem=16g
#SBATCH -J "DITTO_test"
#SBATCH -t 24:00:00
#SBATCH --gres=gpu:1
#SBATCH -C "A100|V100|P100|A30|A100-80G"

#SBATCH -o ./logs/ditto_%j.out
#SBATCH -e ./logs/ditto_%j.err

module load python/3.7.13/jz4yxoc
source ../../ditto/myenv/bin/activate

tasks=(
    "wdc_all_small"
    #"wdc_cameras_small"
    #"wdc_computers_small"
    #"wdc_shoes_small"
    #"wdc_watches_small"
    #"wdc_all_medium"
    #"wdc_cameras_medium"
    #"wdc_computers_medium"
    #"wdc_shoes_medium"
    #"wdc_watches_medium"
    #"wdc_all_large"
    #"wdc_cameras_large"
    #"wdc_computers_large"
    #"wdc_shoes_large"
    #"wdc_watches_large"
    #"wdc_all_xlarge"
    #"wdc_cameras_xlarge"
    #"wdc_computers_xlarge"
    #"wdc_shoes_xlarge"
    #"wdc_watches_xlarge"
    #"Dirty/DBLP-ACM"
    #"Dirty/DBLP-GoogleScholar"
    #"Dirty/iTunes-Amazon"
    #"Dirty/Walmart-Amazon"
    "Structured/Amazon-Google"
    "Structured/Beer"
    "Structured/DBLP-ACM"
    "Structured/DBLP-GoogleScholar"
    "Structured/Fodors-Zagats"
    "Structured/iTunes-Amazon"
    "Structured/Walmart-Amazon"
    #"Textual/Abt-Buy"
    #"Textual/Company"
    #"wdc_all_title_small"
    #"wdc_all_title_medium"
    #"wdc_all_title_large"
    #"wdc_all_title_xlarge"
    #"wdc_shoes_title_small"
    #"wdc_shoes_title_medium"
    #"wdc_shoes_title_large"
    #"wdc_shoes_title_xlarge"
    #"wdc_computers_title_small"
    #"wdc_computers_title_medium"
    #"wdc_computers_title_large"
    #"wdc_computers_title_xlarge"
    #"wdc_watches_title_small"
    #"wdc_watches_title_medium"
    #"wdc_watches_title_large"
    #"wdc_watches_title_xlarge"
    #"wdc_cameras_title_small"
    #"wdc_cameras_title_medium"
    #"wdc_cameras_title_large"
    #"wdc_cameras_title_xlarge"
)

# Run the training script 
for task in "${tasks[@]}"; do
    # get random baseline
    CUDA_VISIBLE_DEVICES=0 python train_domain_agnostic.py \
    --task "${task}" \
    --batch_size 32 \
    --max_len 64 \
    --lr 3e-5 \
    --n_epochs 5 \
    --finetuning \
    --lm distilbert \
    --fp16 \
    --save_model \
    --use_gpu \
    --outfile="dist_results.txt" \
    --sample_method="random" 

    # do closest to target centroid
    CUDA_VISIBLE_DEVICES=0 python train_domain_agnostic.py \
    --task "${task}" \
    --batch_size 32 \
    --max_len 64 \
    --lr 3e-5 \
    --n_epochs 5 \
    --finetuning \
    --lm distilbert \
    --fp16 \
    --save_model \
    --use_gpu \
    --outfile="dist_results.txt" \
    --sample_method="cs_centroid"

    # do closest to any point in the target space
    CUDA_VISIBLE_DEVICES=0 python train_domain_agnostic.py \
    --task "${task}" \
    --batch_size 32 \
    --max_len 64 \
    --lr 3e-5 \
    --n_epochs 5 \
    --finetuning \
    --lm distilbert \
    --fp16 \
    --save_model \
    --use_gpu \
    --outfile="dist_results.txt" \
    --sample_method="cs"

    # do tvdf sampling
    # do closest to any point in the target space
    CUDA_VISIBLE_DEVICES=0 python train_domain_agnostic.py \
    --task "${task}" \
    --batch_size 32 \
    --max_len 64 \
    --lr 3e-5 \
    --n_epochs 5 \
    --finetuning \
    --lm distilbert \
    --fp16 \
    --save_model \
    --use_gpu \
    --outfile="dist_results.txt" \
    --sample_method="tvdf"
done
