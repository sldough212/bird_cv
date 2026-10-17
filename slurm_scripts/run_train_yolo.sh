#!/bin/bash
# Train (Ray-tune) YOLO from yolo11n.pt and select on val presence F1.
# Extra arguments go to train.py, e.g. --run-id ID, or for a CV fold:
#   sbatch slurm_scripts/run_train_yolo.sh --fold 0 --hyperparameters RUN_DIR/best_hyperparameters.json
#SBATCH --account=passerinagenome
#SBATCH --time=48:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --partition=mb-l40s,mb-a30,mb-h100,mb-a6000,mb-r6000
#SBATCH --gres=gpu:1

echo "SLURM_JOB_ID:" $SLURM_JOB_ID
start=$(date +'%D %T')
echo "Start:" $start
module purge
module load gcc/14.2.0 python/3.12.0

cd $SLURM_SUBMIT_DIR

source .venv/bin/activate

export PYTHONUNBUFFERED=1

CONFIG=${CONFIG:-bird_cv/pipelines/configs/config.toml}
python3 bird_cv/pipelines/train.py $CONFIG yolo "$@"
end=$(date +'%D %T')
echo "End:" $end
