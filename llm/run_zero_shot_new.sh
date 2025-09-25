#!/bin/bash
#SBATCH -J zs_em
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem=32g
#SBATCH -t 24:00:00
#SBATCH -o ./logs/zs_em_%j.out
#SBATCH -e ./logs/zs_em_%j.err
#SBATCH --gres=gpu:1
#SBATCH -C "H100|A100|H200|L40S|A30|V100|A100-80G"

module purge
module load cuda/12.8.0/4fdo42o
module load python/3.9.18/5ydd7yq
source /home/nbpulsone/ditto/llm/llmenv/bin/activate

#debug
python - <<'PY'
import torch, os
print("Torch:", torch.__version__, "| CUDA:", torch.version.cuda,
      "| GPU?", torch.cuda.is_available(),
      "| Name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
try:
    import bitsandbytes as bnb
    from bitsandbytes.cuda_setup import main as bnbm
    print("bitsandbytes check:\n", bnbm.main_check())
except Exception as e:
    print("bitsandbytes import error:", e)
PY

models_to_run=(
    #"meta-llama/Meta-Llama-3.1-8B-Instruct"
    "NECOUDBFM/Jellyfish-8B"
)

input_dir_to_run=(
    "wdc_20cc"
    "wdc_20cc_halfseen"
    "wdc_20cc_seen"
    "wdc_50cc"
    "wdc_50cc_halfseen"
    "wdc_50cc_seen"
    "wdc_80cc"
    "wdc_80cc_halfseen"
    "wdc_80cc_seen"
)

# ensure hugging face key available
if [[ -z "${HUGGINGFACE_HUB_TOKEN:-}" ]]; then
    if [[ -f "$HOME/.hf" ]]; then
      export HUGGINGFACE_HUB_TOKEN="$(cat "$HOME/.hf")"
    else
      echo "ERROR: HUGGINGFACE_HUB_TOKEN not set and \$HOME/.hf not found." >&2
      exit 1
    fi
fi
for model in "${models_to_run[@]}"; do
    for test in "${input_dir_to_run[@]}"; do
        echo "Testing $model on $test..."
        python zero_shot_new.py \
        --backend hf \
        --hf_model $model \
        --hf_4bit \
        --input_dir $test \
        --output "./results_${test}.txt"
    done
done
