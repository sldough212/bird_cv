#!/bin/bash                               
# Shebang indicating this is a bash script.
# Do NOT put a comment after the shebang, this will cause an error.
#SBATCH --account=passerinagenome          # Use #SBATCH to define Slurm related values.
#SBATCH --time=10:00:00                      # Must define an account and wall-time.
#SBATCH --mem=64G

echo "SLURM_JOB_ID:" $SLURM_JOB_ID        # Can access Slurm related Environment variables.
start=$(date +'%D %T')                    # Can call bash commands.
echo "Start:" $start
module purge
module load gcc/14.2.0 python/3.12.0      # Load the modules you require for your environment.

cd $SLURM_SUBMIT_DIR

source .venv/bin/activate

# python3 bird_cv/detection/train_yolo.py                        # Call your scripts/commands.
python3 bird_cv/tasks/run_crop_yolo.py
sleep 1m
end=$(date +'%D %T')
echo "End:" $end

# sbatch run.sh
# squeue --user pdoughe1 