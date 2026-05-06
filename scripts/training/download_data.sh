#!/bin/bash

#SBATCH --job-name=data_download
#SBATCH --partition=genoa         # or 'gpu' if you need GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24       # 24 CPUs for data loading
#SBATCH --time=00:45:00          # 45 minutes (adjust as needed)
#SBATCH --output=logs/data_download_%j.out
#SBATCH --error=logs/data_download_%j.err

module purge
module load 2023
module load Python/3.10.8-GCCcore-12.2.0  # Or your Python module

# Activate conda environment
source ~/.bashrc

# Go to scriptstory
cd $HOME/thesis/energy_model/scripts

# Create log directory
mkdir -p ../../logs

# Donload and process data
python process_data.py \
     --output_dir ../data/5000_structures \
     --n_structures 5000 \
     --use_diverse


echo "dataset created!"