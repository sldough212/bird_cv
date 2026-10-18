#!/bin/bash
# Val-sweep inference: array task ID runs share ID % T of the val videos at the
# (ID / T)-th [evaluate] (gap fill, det_conf) setting, T = SWEEP_TASKS
# (default 4). submit.sh sets --array=0-$((N_SWEEP * T - 1)); extra arguments
# go to infer.py, e.g.
#   sbatch --array=0-35 slurm_scripts/run_infer_sweep.sh --run-id RUN_ID --train-run-id RUN_ID
#SBATCH --account=passerinagenome
#SBATCH --time=12:00:00
#SBATCH --partition=mb-l40s
#SBATCH --gres=gpu:1
#SBATCH --mem=64G

echo "SLURM_JOB_ID:" $SLURM_JOB_ID "TASK:" $SLURM_ARRAY_TASK_ID
start=$(date +'%D %T')
echo "Start:" $start
module purge
module load gcc/14.2.0 python/3.12.0

cd $SLURM_SUBMIT_DIR

source .venv/bin/activate

export PYTHONUNBUFFERED=1

T=${SWEEP_TASKS:-4}
python3 bird_cv/pipelines/infer.py "${CONFIG:-bird_cv/pipelines/configs/config.toml}" \
    --split val --sweep-index $((SLURM_ARRAY_TASK_ID / T)) \
    --task-index $((SLURM_ARRAY_TASK_ID % T)) --num-tasks "$T" "$@"

end=$(date +'%D %T')
echo "End:" $end
