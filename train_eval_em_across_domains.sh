#!/bin/bash
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem=32g
#SBATCH -p short
#SBATCH -t 24:00:00
#SBATCH --gres=gpu:1
#SBATCH -C "A100|V100|P100|A30|A100-80G"

#module load python/3.7.13/jz4yxoc
SEED=${SEED:-42}

# edit this for more methods
method="$1"

# see config
DATASET_TO_RUN="$2"

# List of budgets to loop over
#budgets=(1000 2000 3000 4000 5000 6000 7000 8000 9000 10000)
budgets=(1000)

#result file
result_file="${method}_test_${SEED}_${DATASET_TO_RUN}_${SLURM_JOB_ID}.txt"
out_dir="./output_${method}_${SEED}_${SLURM_JOB_ID}"

# Loop over each method
echo "Started at: $(date)" >> "$result_file"

first_budget=true

for budget in "${budgets[@]}"; do
    echo "===================" >> "$result_file"
    echo "STARTING BUDGET: $budget" >> "$result_file"
    echo "===================" >> "$result_file"

    # Conditionally add --keep_lower_budgets
    if $first_budget; then
        echo "NO KEEP FLAG!!!:"
        keep_flag=""
        first_budget=false  # flip after first budget
    else
        echo "USING KEEP FLAG!!!:"
        keep_flag="--keep_lower_budgets"
    fi

    # Loop over each budget
    echo "$method : $budget"
    echo "~~~ $method $budget ~~~" >> "$result_file"
    result_file="${method}_test_${SEED}_${DATASET_TO_RUN}_${SLURM_JOB_ID}.txt"

    # Run the training script 
    CUDA_VISIBLE_DEVICES=0 python train_ensemble_em_across_domains.py \
    --task "$DATASET_TO_RUN" \
    --budget "$budget" \
    --method "$method" \
    --job_id "$SLURM_JOB_ID" \
    --batch_size 32 \
    --max_len 64 \
    --lr 3e-5 \
    --n_epochs 5 \
    --run_id "$SEED" \
    --finetuning \
    --lm roberta \
    --fp16 \
    --save_model \
    --dynamic_sampling \
    --summarize \
    --ensemble_type "soft_vote" \
    $keep_flag

    echo "EVALUATING $method, $budget budget"
    CUDA_VISIBLE_DEVICES=0 python emad_matcher.py \
    --task "$DATASET_TO_RUN" \
    --result_file "$result_file" \
    --out_dir "$out_dir" \
    --lm roberta \
    --max_len 64 \
    --use_gpu \
    --fp16 \
    --checkpoint_path checkpoints \
    --method "$method" \
    --budget "$budget" \
    --job_id "$SLURM_JOB_ID" \
    --run_id "$SEED" \
    --summarize \
    --ensemble_type "soft_vote" \

done

rm -rf $out_dir
rm -rf "checkpoints_${SLURM_JOB_ID}"
rm -rf "updated_data_${SLURM_JOB_ID}"

echo "Ended at: $(date)" >> "$result_file"
# TODO: Make new program to generate csv of results and maybe graph