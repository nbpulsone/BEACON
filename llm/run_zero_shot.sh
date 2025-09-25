#!/bin/bash
#SBATCH -J zs_em
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 2
#SBATCH --mem=32G
#SBATCH -t 24:00:00
#SBATCH -o ./logs/zs_em_%j.out
#SBATCH -e ./logs/zs_em_%j.err
#SBATCH -C "H100|A100|H200|L40S|A30|V100|A100-80G"

# --- config you can override at submit-time with --export=ALL,VAR=... ---
INPUT_DIR=${INPUT_DIR:-"$(realpath -m ./test)"}   # DITTO-format test file
OUTPUT_FILE=${OUTPUT_FILE:-"$(realpath -m ./results.txt)"}
MODEL=${MODEL:-gpt-4o-mini}                       # or gpt-4o
LIMIT=${LIMIT:-0}                                 # 0 = all pairs
SLEEP=${SLEEP:-0.02}                              # small pause to avoid rate limits
BACKEND=${BACKEND:-hf}
HF_4BIT=${HF_4BIT:-1}
# ------------------------------------------------------------------------

mkdir -p logs

# --- lightweight venv; idempotent ---
module load python/3.9.18/5ydd7yq
source llmenv/bin/activate

# --- run with unbuffered output so tail -f shows progress immediately ---
export PYTHONUNBUFFERED=1

echo "Starting zero-shot EM:"
echo "  Input: $INPUT_DIR"
echo "  Output: $OUTPUT_FILE"
echo "  Limit: $LIMIT"
echo "  Sleep: $SLEEP"
echo "  JobID: $SLURM_JOB_ID"

# --- prepare environment either for huggingface or openai ---
if [[ "${BACKEND:-openai}" == "hf" ]]; then
  if [[ -z "${HUGGINGFACE_HUB_TOKEN:-}" ]]; then
    if [[ -f "$HOME/.hf" ]]; then
      export HUGGINGFACE_HUB_TOKEN="$(cat "$HOME/.hf")"
    else
      echo "ERROR: HUGGINGFACE_HUB_TOKEN not set and \$HOME/.hf not found." >&2
      exit 1
    fi
  fi
  python -u zero_shot_new.py \
    --backend hf \
    --hf_model "${HF_MODEL:-meta-llama/Meta-Llama-3.1-8B-Instruct}" \
    ${HF_4BIT:+--hf_4bit} \
    --input_dir "${INPUT_DIR:-./test}" \
    --output "${OUTPUT_FILE:-./results.txt}"
else
  
# --- OpenAI key: prefer loading from a protected file (~/.openai_key) ---
  # Create ~/.openai_key with the single line: sk-... (chmod 600)
  if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    if [[ -f "$HOME/.openai_key" ]]; then
      export OPENAI_API_KEY="$(cat "$HOME/.openai_key")"
    else
      echo "ERROR: OPENAI_API_KEY not set and \$HOME/.openai_key not found." >&2
      exit 1
    fi
  fi
  pip -q install openai
  python -u zero_shot_new.py \
    --backend openai \
    --model "${MODEL:-gpt-4o-mini}" \
    --input_dir "${INPUT_DIR:-./test}" \
    --output "${OUTPUT_FILE:-./results.txt}"
fi


echo "Done."
# export OPENAI_API_KEY=sk-...
# run with sbatch --export=ALL,INPUT_DIR=./test,OUTPUT_FILE=./results.txt,MODEL=gpt-4o-mini run_zero_shot.sh