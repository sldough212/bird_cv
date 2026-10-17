#!/bin/bash
# Submit the frozen-split train -> evaluate chain with afterok dependencies.
# Run from anywhere: bash slurm_scripts/submit.sh
# Override CONFIG or RUN_ID in the environment. Outputs land in
# {path_to_base_output}/train/{yolo,videomae}/$RUN_ID and
# {path_to_base_output}/evaluation/{windows,sweep,test}/$RUN_ID.
set -euo pipefail

cd "$(dirname "$0")/.."
export CONFIG=${CONFIG:-bird_cv/pipelines/configs/config.toml}
RUN_ID=${RUN_ID:-$(date +%Y%m%d_%H%M%S)}
echo "RUN_ID=$RUN_ID CONFIG=$CONFIG"

yolo=$(sbatch --parsable slurm_scripts/run_train_yolo.sh --run-id "$RUN_ID")
mae=$(sbatch --parsable slurm_scripts/run_train_mae.sh --run-id "$RUN_ID")
echo "train_yolo=$yolo train_mae=$mae"

BASE=$(sed -n 's/^path_to_base_output *= *"\(.*\)".*/\1/p' "$CONFIG")
windows=$(sbatch --parsable --dependency=afterok:$mae \
    --gres=gpu:1 --partition=mb-l40s,mb-a30,mb-h100,mb-a6000,mb-r6000 \
    slurm_scripts/run_evaluate.sh windows --run-id "$RUN_ID" \
    --mae-model "$BASE/train/videomae/$RUN_ID/videomae/best")
echo "evaluate_windows=$windows"

# End-to-end evaluation needs predictions from the inference pipeline (#39)
# under {path_to_base_output}/predictions/{val_sweep/gap{g}_conf{c},test}/,
# made with $BASE/train/yolo/$RUN_ID/best.pt and
# $BASE/train/videomae/$RUN_ID/videomae/best.
if [ -f slurm_scripts/run_infer_array.sh ]; then
    # TODO(#39): submit val-sweep inference here once its CLI is merged, then
    #   sweep=$(sbatch --parsable --dependency=afterok:$infer_val slurm_scripts/run_evaluate.sh sweep --run-id "$RUN_ID")
    # and test inference with best_setting.json, then
    #   sbatch --dependency=afterok:$infer_test slurm_scripts/run_evaluate.sh test --run-id "$RUN_ID"
    echo "run_infer_array.sh found but not wired yet; submit inference + evaluation by hand."
else
    echo "No inference job yet (#39); after predictions exist run:"
fi
echo "  sbatch slurm_scripts/run_evaluate.sh sweep --run-id $RUN_ID"
echo "  sbatch slurm_scripts/run_evaluate.sh test --run-id $RUN_ID"
