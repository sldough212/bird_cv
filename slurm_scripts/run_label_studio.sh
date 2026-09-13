#!/bin/bash                               
# Shebang indicating this is a bash script.
# Do NOT put a comment after the shebang, this will cause an error.
#SBATCH --account=passerinagenome
#SBATCH --time=2:00:00
#SBATCH --mem=16G

echo "SLURM_JOB_ID:" $SLURM_JOB_ID        # Can access Slurm related Environment variables.
start=$(date +'%D %T')                    # Can call bash commands.
echo "Start:" $start
module purge
module load gcc/14.2.0 python/3.12.0      # Load the modules you require for your environment.

source .venv/bin/activate

NODE_IP=$(hostname -I | tr ' ' '\n' | grep '^10\.' | head -n1)

echo "=============================="
echo "Label Studio starting on node: $(hostname)"
echo "Node IP: $NODE_IP"
echo "Port: 8080"
echo "Job ID: $SLURM_JOB_ID"
echo "=============================="
echo ""
echo "To tunnel, run this on your local machine:"
echo "ssh -L 8080:$NODE_IP:8080 pdoughe1@medicinebow.arcc.uwyo.edu"
echo ""

export LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true
export LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=/

label-studio start --port=8080 --no-browser
