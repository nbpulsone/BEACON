#!/bin/bash
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem=16g
#SBATCH -J "DITTO_DYNAMIC_test"
#SBATCH -t 24:00:00
#SBATCH --gres=gpu:1

#SBATCH -o ./logs/ditto_dynamic_2500_%j.out
#SBATCH -e ./logs/ditto_dynamic_2500_%j.err

module load python/3.7.13/jz4yxoc
source ./myenv/bin/activate

# List of task names for the 13 categories
categories=(
    "wdc_automotive"
    "wdc_camera_and_photo"
    "wdc_cell_phones_and_accessories"
    "wdc_clothing"
    "wdc_computers_and_accessories"
    "wdc_home_and_garden"
    "wdc_jewelry"
    "wdc_movies_and_tv"
    "wdc_musical_instruments"
    "wdc_office_products"
    "wdc_other_electronics"
    "wdc_sports_and_outdoors"
    "wdc_tools_and_home_improvement"
)

result_file="./ensemble_test.txt"

out_dir="./output"

CUDA_VISIBLE_DEVICES=0 python emad_matcher.py \
    --task "wdc_category_80cc_fully_seen" \
    --result_file "$result_file" \
    --out_dir "$out_dir" \
    --lm roberta \
    --max_len 64 \
    --use_gpu \
    --fp16 \
    --checkpoint_path checkpoints \
    --method "kcg-tv" \
    --job_id 1578670 \
    --run_id 82 \
    --budget 10000 \
    --summarize

