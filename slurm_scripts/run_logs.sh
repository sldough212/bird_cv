# Sourced by job scripts submitted on their own: send the rest of the job's
# output to {path_to_base_output}/slurm_logs/{run id}/{job name}-{job id}.out.
# submit.sh already points --output there and exports RUN_LOGS, so this is
# skipped for its jobs.
#   source slurm_scripts/run_logs.sh RUN_ID
if [ -z "${RUN_LOGS:-}" ]; then
    eval "$(python3 bird_cv/pipelines/train.py "$CONFIG" paths --run-id "$1" | grep '^BASE=')"
    RUN_LOGS="$BASE/slurm_logs/$1"
    mkdir -p "$RUN_LOGS"
    log="$RUN_LOGS/$SLURM_JOB_NAME-${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}${SLURM_ARRAY_TASK_ID:+_$SLURM_ARRAY_TASK_ID}.out"
    echo "Logging to $log"
    exec >"$log" 2>&1
fi
