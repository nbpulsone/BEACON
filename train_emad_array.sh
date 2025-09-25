m="${1:-spec}"
d="${2:-wdc_category}"
echo "$m"
echo "$d"
for s in 22 42 62 82; do
  sbatch --export=ALL,SEED=$s,METHOD="$m",DATASET_TO_RUN="$d" \
         -J em_seed_$s \
         -o logs/em_${m}_${s}_%j.out \
         -e logs/em_${m}_${s}_%j.err \
         train_eval_ensemble_em_across_domains.sh
done