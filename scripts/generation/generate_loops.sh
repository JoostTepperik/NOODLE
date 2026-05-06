#!/bin/bash

#SBATCH --job-name=kic_1M_gen
#SBATCH --partition=genoa         # CPU partition on Snellius
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24       # Use all 24 CPUs for multiprocessing
#SBATCH --time=12:00:00          # 12 hours (adjust if it finishes faster)
#SBATCH --output=logs/kic_gen_%j.out
#SBATCH --error=logs/kic_gen_%j.err

# Create logs directory if it doesn't exist
mkdir -p logs

# Optional: Load your conda environment here if you don't have it loading automatically
# source /home/jtepperik/miniconda3/etc/profile.d/conda.sh
# conda activate <your_env_name>

echo "Starting 1-Million Sample KIC Generation on 24 cores..."
echo "Date: $(date)"

# Run the optimized KIC pipeline
python test_cdr3_pipeline.py \
    --n-cpus 24 \
    --n-samples 50000 \
    --n-structures 25 \
    --max-init-clash 50000 \
    --max-init-intra 100 \
    --max-init-energy 16 \
    --temperature 0.1 \
    --filter-order energy,intra,fw

echo "Generation complete!"
echo "Date: $(date)"