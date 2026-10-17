#!/bin/bash
# Submit the frozen-split train -> evaluate chain with afterok dependencies.
#
#   bash slurm_scripts/submit.sh [--smoke]
#
# --smoke (or SMOKE=1) runs 2 tuning trials and 2 epochs per model, for a
# first cluster run; it is a smoke test, never report its scores.
# Override CONFIG or RUN_ID in the environment. Output paths come from the
# config ([train] output, path_to_base_output) and are printed below.
set -euo pipefail

cd "$(dirname "$0")/.."
export CONFIG=${CONFIG:-bird_cv/pipelines/configs/config.toml}
RUN_ID=${RUN_ID:-$(date +%Y%m%d_%H%M%S)}
SMOKE=${SMOKE:-0}
[ "${1:-}" = "--smoke" ] && SMOKE=1
TRAIN_ARGS=(--run-id "$RUN_ID")
[ "$SMOKE" = "1" ] && TRAIN_ARGS+=(--smoke)

# Resolve paths with the same code the jobs use.
if command -v module >/dev/null 2>&1; then
    module load gcc/14.2.0 python/3.12.0
fi
source .venv/bin/activate
if ! PATHS=$(python3 bird_cv/pipelines/train.py "$CONFIG" paths --run-id "$RUN_ID"); then
    echo "Could not resolve output paths from $CONFIG" >&2
    exit 1
fi
eval "$PATHS"
echo "RUN_ID=$RUN_ID CONFIG=$CONFIG SMOKE=$SMOKE"
echo "$PATHS"

yolo=$(sbatch --parsable slurm_scripts/run_train_yolo.sh "${TRAIN_ARGS[@]}")
mae=$(sbatch --parsable slurm_scripts/run_train_mae.sh "${TRAIN_ARGS[@]}")
echo "train_yolo=$yolo train_mae=$mae"

windows=$(sbatch --parsable --dependency=afterok:$mae \
    --gres=gpu:1 --partition=mb-l40s,mb-a30,mb-h100,mb-a6000,mb-r6000 \
    slurm_scripts/run_evaluate.sh windows --run-id "$RUN_ID" --mae-model "$MAE_BEST")
echo "evaluate_windows=$windows"

# End-to-end evaluation needs predictions from the inference pipeline (#39)
# under {path_to_base_output}/predictions/{val_sweep/gap{g}_conf{c},test}/,
# made with $YOLO_BEST and $MAE_BEST.
# TODO(#39): once its CLI is merged, submit val-sweep inference with
# --dependency=afterok:$yolo:$mae, then
#   sbatch --dependency=afterok:$infer_val slurm_scripts/run_evaluate.sh sweep --run-id "$RUN_ID"
# then test inference with best_setting.json, then
#   sbatch --dependency=afterok:$infer_test slurm_scripts/run_evaluate.sh test --run-id "$RUN_ID"
echo "Inference (#39) is not chained yet; after predictions exist run:"
echo "  sbatch slurm_scripts/run_evaluate.sh sweep --run-id $RUN_ID"
echo "  sbatch slurm_scripts/run_evaluate.sh test --run-id $RUN_ID"
