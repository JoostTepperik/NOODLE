#!/bin/bash

#SBATCH --job-name=torsion_train
#SBATCH --partition=genoa         # or 'gpu' if you need GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24       # 24 CPUs for data loading
#SBATCH --time=00:30:00          # 30 minutes (adjust as needed)
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

# Load modules (if needed)
module purge
module load 2023
module load Python/3.10.8-GCCcore-12.2.0  # Or your Python module

# Activate conda environment
source ~/.bashrc

# Go to training directory
cd $HOME/thesis/energy_model/scripts/training

# Create log directory
mkdir -p ../../logs

# Run training
python train_jax.py \
    --data /home/jtepperik/thesis/energy_model/data/training_5k/training_data.h5 \
    --batch_size 1024 \
    --hidden_dim 512 \
    --n_layers 4 \
    --epochs 15
    --output_dir ../../outputs/von_mises_jaxvv6

echo "Training complete!"