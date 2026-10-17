#!/bin/bash
# Evaluate predictions: first argument is the mode (sweep, test, cv, windows),
# the rest go to evaluate.py, e.g.
#   sbatch slurm_scripts/run_evaluate.sh test --run-id ID
#   sbatch --gres=gpu:1 --partition=mb-l40s slurm_scripts/run_evaluate.sh windows --mae-model DIR
#SBATCH --account=passerinagenome
#SBATCH --time=6:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4

echo "SLURM_JOB_ID:" $SLURM_JOB_ID
start=$(date +'%D %T')
echo "Start:" $start
module purge
module load gcc/14.2.0 python/3.12.0

cd $SLURM_SUBMIT_DIR

source .venv/bin/activate

export PYTHONUNBUFFERED=1

CONFIG=${CONFIG:-bird_cv/pipelines/configs/config.toml}
python3 bird_cv/pipelines/evaluate.py $CONFIG "$@"
end=$(date +'%D %T')
echo "End:" $end
