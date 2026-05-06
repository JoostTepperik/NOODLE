#!/bin/bash

#SBATCH --job-name=torsion_train
#SBATCH --partition=rome         # or 'gpu' if you need GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16      # 24 CPUs for data loading
#SBATCH --time=01:30:00          # 30 minutes (adjust as needed)
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err


mkdir -p logs

python create_context_windows.py \
    --input /home/jtepperik/thesis/energy_model/data/pdb_redo/pdbredo_all_residues.h5 \
    --output data/training_variable_context_21/training_data.h5 \
    --max_context 21 \
    --n_workers 16