#!/bin/bash
# Submit the frozen-split train -> evaluate chain with afterok dependencies.
#
#   bash slurm_scripts/submit.sh [--smoke]
#
# Trains every [train] role (ROLES, printed by `train.py CONFIG paths`) with
# the model its config table names, then scores each role's model on val/test.
# --smoke (or SMOKE=1) runs 2 tuning trials and 2 epochs per role, for a
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

GPU=(--gres=gpu:1 --partition=mb-l40s,mb-a30,mb-h100,mb-a6000,mb-r6000)
TRAIN_JOBS=()
for role in $ROLES; do
    job=$(sbatch --parsable slurm_scripts/run_train.sh "$role" "${TRAIN_ARGS[@]}")
    best_var="${role^^}_BEST"
    eval_job=$(sbatch --parsable --dependency=afterok:$job "${GPU[@]}" \
        slurm_scripts/run_evaluate.sh model --role "$role" \
        --checkpoint "${!best_var}" --run-id "${RUN_ID}_${role}")
    echo "train_${role}=$job evaluate_${role}=$eval_job"
    TRAIN_JOBS+=("$job")
done

# End-to-end evaluation needs predictions from the inference pipeline under
# {path_to_base_output}/predictions/{val_sweep/gap{g}_conf{c},test}/, made
# with $DETECTOR_BEST and $CLASSIFIER_BEST.
# TODO(integration): run_infer_array.sh reads models and gap-fill settings
# from the config only; once it can take them per run, submit val-sweep
# inference with --dependency=afterok:$(IFS=:; echo "${TRAIN_JOBS[*]}"), then
#   sbatch --dependency=afterok:$infer_val slurm_scripts/run_evaluate.sh sweep --run-id "$RUN_ID"
# then test inference with best_setting.json, then
#   sbatch --dependency=afterok:$infer_test slurm_scripts/run_evaluate.sh test --run-id "$RUN_ID"
echo "Inference is not chained yet; after predictions exist run:"
echo "  sbatch slurm_scripts/run_evaluate.sh sweep --run-id $RUN_ID"
echo "  sbatch slurm_scripts/run_evaluate.sh test --run-id $RUN_ID"
