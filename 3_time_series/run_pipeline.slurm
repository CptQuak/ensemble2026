#!/bin/bash -l
#SBATCH -J timeseries_predictions  # Name of the job
#SBATCH -N 1                       # Number of nodes
#SBATCH --ntasks-per-node=16       # Number of CPU cores
#SBATCH --gpus=1                   # Number of GPUs (max 1 before the weekend, max 3 during)
#SBATCH --time=12:00:00            # Maximum execution time (HH:MM:SS)
#SBATCH -A tutorial                # Account
#SBATCH -p tutorial                # Partition
#SBATCH --mem=64GB                 # Memory allocation
#SBATCH --output=pipeline_%J.out   # Standard output and error log
# Load the required Python module
module add GCCcore/13.2.0 Python/3.11.5
# Ensure uv caches to SCRATCH instead of HOME
export PATH="$SCRATCH/bin:$PATH"
export UV_CACHE_DIR=$SCRATCH/task3/.uv_cache
# Run your prediction pipeline
uv run main.py --data_path data/data.csv --optimize