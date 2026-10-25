#!/bin/bash
# Submit the whole pipeline as afterok-chained SLURM jobs:
# prepare -> build datasets -> train each [models] role (+ model metrics)
# -> val-sweep inference array -> evaluate sweep -> test inference at the
# chosen setting -> evaluate test.
#
#   bash slurm_scripts/submit.sh [--smoke] [--prepared ID] [PREPARE_ARGS...]
#
# PREPARE_ARGS go to prepare.py (e.g. --detect-export F --classify-export F).
# The prepare job writes its id to PREPARED_FILE, which later jobs read via
# --run-id; --prepared ID (or PREPARED=ID) skips prepare and writes ID there.
# --smoke (or SMOKE=1): 2 tuning trials and 2 epochs per role, and 2 videos
# per inference run; a smoke test, never report its scores.
# Override CONFIG or RUN_ID in the environment.
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
if ! PATHS=$(python3 bird_cv/pipelines/train.py "$CONFIG" paths --run-id "$RUN_ID"); then
    echo "Could not resolve output paths from $CONFIG" >&2
    exit 1
fi
eval "$PATHS"
echo "RUN_ID=$RUN_ID CONFIG=$CONFIG SMOKE=$SMOKE"
echo "$PATHS"

[ -n "$ROLES" ] || { echo "No [models] roles in $CONFIG" >&2; exit 1; }
if [ -n "$PREPARED" ]; then
    mkdir -p "$(dirname "$PREPARED_FILE")"
    echo "$PREPARED" > "$PREPARED_FILE"
fi
RUN=(--run-id "$RUN_ID")
INFER=("${RUN[@]}" --train-run-id "$RUN_ID")
TRAIN=("${RUN[@]}")
if [ "$SMOKE" = "1" ]; then
    TRAIN+=(--smoke)
    INFER+=(--limit 2)
fi
GPU=(--gres=gpu:1 --partition=mb-l40s,mb-a30,mb-h100,mb-a6000)
export RUN_LOGS="$BASE/slurm_logs/$RUN_ID"
LOGS=$RUN_LOGS
mkdir -p "$LOGS"
log() { echo --job-name="$1" --output="$LOGS/$1-%j.out"; }
alog() { echo --job-name="$1" --output="$LOGS/$1-%A_%a.out"; }
DEP=(--kill-on-invalid-dep=yes --dependency)
JOBS=()

after=()
if [ -z "$PREPARED" ]; then
    prepare=$(sbatch --parsable $(log prepare) slurm_scripts/run_prepare.sh --run-id "$RUN_ID" \
        ${PREPARE_ARGS[@]+"${PREPARE_ARGS[@]}"})
    echo "prepare=$prepare"
    JOBS+=("$prepare")
    after=("${DEP[@]}" "afterok:$prepare")
fi
build=$(sbatch --parsable $(log build_datasets) ${after[@]+"${after[@]}"} slurm_scripts/run_build_datasets.sh "${RUN[@]}")
echo "build_datasets=$build"
JOBS+=("$build")

TRAIN_JOBS=()
for role in $ROLES; do
    job=$(sbatch --parsable $(log "train_$role") "${DEP[@]}" "afterok:$build" \
        slurm_scripts/run_train.sh "$role" "${TRAIN[@]}")
    eval_job=$(sbatch --parsable $(log "evaluate_$role") "${DEP[@]}" "afterok:$job" "${GPU[@]}" \
        slurm_scripts/run_evaluate.sh model --role "$role" "${RUN[@]}")
    echo "train_${role}=$job evaluate_${role}=$eval_job"
    TRAIN_JOBS+=("$job")
    JOBS+=("$job" "$eval_job")
done
trained=$(IFS=:; echo "${TRAIN_JOBS[*]}")

sweep=$(sbatch --parsable $(alog infer_sweep) "${DEP[@]}" "afterok:$trained" \
    --array=0-$((N_SWEEP - 1)) \
    slurm_scripts/run_infer_sweep.sh "${INFER[@]}")
eval_sweep=$(sbatch --parsable $(log evaluate_sweep) "${DEP[@]}" "afterok:$sweep" \
    slurm_scripts/run_evaluate.sh sweep "${RUN[@]}")
infer_test=$(sbatch --parsable $(alog infer_test) "${DEP[@]}" "afterok:$eval_sweep" \
    slurm_scripts/run_infer_array.sh --split test "${INFER[@]}")
eval_test=$(sbatch --parsable $(log evaluate_test) "${DEP[@]}" "afterok:$infer_test" \
    slurm_scripts/run_evaluate.sh test "${RUN[@]}")
JOBS+=("$sweep" "$eval_sweep" "$infer_test" "$eval_test")
echo "infer_sweep=$sweep ($N_SWEEP settings) evaluate_sweep=$eval_sweep"
echo "infer_test=$infer_test evaluate_test=$eval_test"
echo "Logs: $LOGS"
echo "Prepared id: $PREPARED_FILE"
echo "Predictions: $INFER_DIR/{val_sweep,test}  Sweep: $SWEEP_DIR  Test: $TEST_DIR"
echo "Cancel: scancel ${JOBS[*]}"
