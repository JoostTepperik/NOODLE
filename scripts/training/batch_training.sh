#!/bin/bash

#SBATCH --job-name=torsion_train
#SBATCH --partition=genoa         # or 'gpu' if you need GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24       # 24 CPUs for data loading
#SBATCH --time=24:00:00          # 30 minutes (adjust as needed)
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

# Activate conda environment
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "=========================================="

# Load modules if needed
# module load cuda/11.8
# module load python/3.10

# Activate conda environment if needed
# source activate your_env

# Run with unbuffered output (-u flag)
python -u train.py \
    --data /home/jtepperik/thesis/energy_model/scripts/data_processing/data/training_variable_context_3/training_data.h5 \
    --output_dir outputs/energy_loss_c3 \
    --max_context 3 \
    --n_bins 36 \
    --hidden_dim 768 \
    --n_layers 3 \
    --batch_size 2048 \
    --n_epochs 20 \


echo "=========================================="
echo "End time: $(date)"