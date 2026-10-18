#!/bin/bash
# Shebang indicating this is a bash script.
# Do NOT put a comment after the shebang, this will cause an error.
#SBATCH --account=passerinagenome
#SBATCH --time=12:00:00
#SBATCH --partition=mb-l40s
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --array=0-3

# End-to-end inference, videos split across array tasks; every task writes
# into the same run directory (infer/infer_<array job id>, see
# bird_cv/core/layout.py). Models come from the config; extra arguments go to
# infer.py, e.g.:
#   sbatch slurm_scripts/run_infer_array.sh --camera H7,I22 --limit 4
#   sbatch slurm_scripts/run_infer_array.sh --train-run-id RUN_ID

echo "SLURM_JOB_ID:" $SLURM_JOB_ID "TASK:" $SLURM_ARRAY_TASK_ID
start=$(date +'%D %T')
echo "Start:" $start
module purge
module load gcc/14.2.0 python/3.12.0

cd $SLURM_SUBMIT_DIR

source .venv/bin/activate

export PYTHONUNBUFFERED=1

python3 bird_cv/pipelines/infer.py "${CONFIG:-bird_cv/pipelines/configs/config.toml}" \
    --run-id "infer_${SLURM_ARRAY_JOB_ID}" \
    --task-index "${SLURM_ARRAY_TASK_ID}" \
    --num-tasks "${SLURM_ARRAY_TASK_COUNT}" \
    "$@"

end=$(date +'%D %T')
echo "End:" $end
