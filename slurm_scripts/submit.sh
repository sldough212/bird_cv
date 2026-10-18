#!/bin/bash
# Submit the frozen-split train -> evaluate chain with afterok dependencies.
#
#   bash slurm_scripts/submit.sh [--smoke] [--prepared ID] [PREPARE_ARGS...]
#
# Without --prepared (or PREPARED=ID), runs prepare here first (PREPARE_ARGS go
# to prepare.py, e.g. --detect-export F --classify-export F or --hostname IP)
# and captures the prepared-data id it prints; every job gets that id.
# Trains every [models] role (ROLES, printed by `train.py CONFIG paths`) with
# the model its config table names, then scores each role's model on val/test.
# --smoke (or SMOKE=1) runs 2 tuning trials and 2 epochs per role, for a
# first cluster run; it is a smoke test, never report its scores.
# Override CONFIG or RUN_ID in the environment. Output paths come from
# bird_cv/core/layout.py under path_to_base_output and are printed below.
set -euo pipefail

cd "$(dirname "$0")/.."
export CONFIG=${CONFIG:-bird_cv/pipelines/configs/config.toml}
RUN_ID=${RUN_ID:-$(date +%Y%m%d_%H%M%S)}
SMOKE=${SMOKE:-0}
PREPARED=${PREPARED:-}
PREPARE_ARGS=()
while [ $# -gt 0 ]; do
    case "$1" in
        --smoke) SMOKE=1 ;;
        --prepared) PREPARED=$2; shift ;;
        *) PREPARE_ARGS+=("$1") ;;
    esac
    shift
done

if command -v module >/dev/null 2>&1; then
    module load gcc/14.2.0 python/3.12.0
fi
source .venv/bin/activate

if [ -z "$PREPARED" ]; then
    PREPARE_OUT=$(python3 bird_cv/pipelines/prepare.py "$CONFIG" "${PREPARE_ARGS[@]}")
    PREPARED=$(echo "$PREPARE_OUT" | sed -n 's/^PREPARED=//p' | tail -n 1)
    if [ -z "$PREPARED" ]; then
        echo "prepare.py printed no PREPARED= id" >&2
        exit 1
    fi
fi
TRAIN_ARGS=(--run-id "$RUN_ID" --prepared "$PREPARED")
[ "$SMOKE" = "1" ] && TRAIN_ARGS+=(--smoke)

# Resolve paths with the same code the jobs use.
if ! PATHS=$(python3 bird_cv/pipelines/train.py "$CONFIG" paths --run-id "$RUN_ID"); then
    echo "Could not resolve output paths from $CONFIG" >&2
    exit 1
fi
eval "$PATHS"
echo "RUN_ID=$RUN_ID CONFIG=$CONFIG SMOKE=$SMOKE PREPARED=$PREPARED"
echo "$PATHS"

GPU=(--gres=gpu:1 --partition=mb-l40s,mb-a30,mb-h100,mb-a6000,mb-r6000)
TRAIN_JOBS=()
for role in $ROLES; do
    job=$(sbatch --parsable slurm_scripts/run_train.sh "$role" "${TRAIN_ARGS[@]}")
    eval_job=$(sbatch --parsable --dependency=afterok:$job "${GPU[@]}" \
        slurm_scripts/run_evaluate.sh model --role "$role" \
        --run-id "$RUN_ID" --prepared "$PREPARED")
    echo "train_${role}=$job evaluate_${role}=$eval_job"
    TRAIN_JOBS+=("$job")
done

# End-to-end evaluation reads predictions from the inference runs that
# bird_cv/core/layout.py places under infer/$RUN_ID/{val_sweep/gap{g}_conf{c},test}/,
# made with this run's checkpoints (infer.py --train-run-id $RUN_ID).
# TODO(integration, #48 part 3): submit val-sweep inference with
# --dependency=afterok:$(IFS=:; echo "${TRAIN_JOBS[*]}"), then
#   sbatch --dependency=afterok:$infer_val slurm_scripts/run_evaluate.sh sweep --run-id "$RUN_ID" --prepared "$PREPARED"
# then test inference with best_setting.json, then
#   sbatch --dependency=afterok:$infer_test slurm_scripts/run_evaluate.sh test --run-id "$RUN_ID" --prepared "$PREPARED"
echo "Inference is not chained yet (waiting on #48 part 3). Once val-sweep and"
echo "test predictions exist under infer/$RUN_ID/, run:"
echo "  sbatch slurm_scripts/run_evaluate.sh sweep --run-id $RUN_ID --prepared $PREPARED"
echo "  sbatch slurm_scripts/run_evaluate.sh test --run-id $RUN_ID --prepared $PREPARED"
