#!/bin/bash
# Build every [models] role's training data (frozen split and CV folds) from
# prepared data; arguments go to build_datasets.py, e.g.
#   sbatch slurm_scripts/run_build_datasets.sh --prepared PREPARED_ID
#   sbatch slurm_scripts/run_build_datasets.sh --prepared PREPARED_ID --role detector --fold 0
# CPU only: frames are decoded and written as JPEGs, no model runs.
#SBATCH --account=passerinagenome
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8

echo "SLURM_JOB_ID:" $SLURM_JOB_ID
start=$(date +'%D %T')
echo "Start:" $start
module purge
module load gcc/14.2.0 python/3.12.0

cd $SLURM_SUBMIT_DIR

source .venv/bin/activate

export PYTHONUNBUFFERED=1

CONFIG=${CONFIG:-bird_cv/pipelines/configs/config.toml}
python3 bird_cv/pipelines/build_datasets.py $CONFIG \
    --processes "${SLURM_CPUS_PER_TASK:-4}" "$@"
end=$(date +'%D %T')
echo "End:" $end
