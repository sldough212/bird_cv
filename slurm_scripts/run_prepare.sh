#!/bin/bash
#SBATCH --account=passerinagenome
#SBATCH --time=2:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

echo "SLURM_JOB_ID:" $SLURM_JOB_ID
start=$(date +'%D %T')
echo "Start:" $start
module purge
module load gcc/14.2.0 python/3.12.0

cd $SLURM_SUBMIT_DIR

source .venv/bin/activate

export PYTHONUNBUFFERED=1

NODE_IP=$(hostname -I | awk '{print $1}')

# Exports both Label Studio projects, then snapshot -> ground truth -> split
# -> summary -> sampling. Pass --detect-export/--classify-export to reuse files.
python3 bird_cv/pipelines/prepare.py bird_cv/pipelines/configs/config.toml --hostname $NODE_IP "$@"
end=$(date +'%D %T')
echo "End:" $end

# sbatch slurm_scripts/run_prepare.sh
