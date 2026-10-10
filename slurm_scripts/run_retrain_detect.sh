#!/bin/bash
#SBATCH --account=passerinagenome
#SBATCH --time=6:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --partition=mb-l40s,mb-a30,mb-h100,mb-a6000,mb-r6000
#SBATCH --gres=gpu:1

echo "SLURM_JOB_ID:" $SLURM_JOB_ID        # Can access Slurm related Environment variables.
start=$(date +'%D %T')                    # Can call bash commands.
echo "Start:" $start
module purge
module load gcc/14.2.0 python/3.12.0      # Load the modules you require for your environment.

cd $SLURM_SUBMIT_DIR

source .venv/bin/activate

export PYTHONUNBUFFERED=1

python3 bird_cv/pipelines/retrain_detect.py bird_cv/pipelines/configs/config.toml
sleep 1m
end=$(date +'%D %T')
echo "End:" $end

# sbatch run.sh
# squeue --user pdoughe1 