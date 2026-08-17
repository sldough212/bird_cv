#!/bin/bash                               
# Shebang indicating this is a bash script.
# Do NOT put a comment after the shebang, this will cause an error.
#SBATCH --account=passerinagenome
#SBATCH --time=2:00:00
#SBATCH --mem=16G

source .venv/bin/activate

NODE_IP=$(hostname -I | awk '{print $1}')

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

label-studio start --port=8080 --no-browser
