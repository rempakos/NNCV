#!/bin/bash

# Load environment variables from .env
if [ -f .env ]; then
    source .env
fi

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --gpus=1
#SBATCH --partition=gpu_mig
#SBATCH --time=3:00:00

# Pull container from dockerhub
apptainer pull container.sif docker://cclaes/5lsm0:v1

# Set the Hugging Face token as an environment variable
export HF_TOKEN=$HF_TOKEN

# Use the huggingface-cli package inside the container to download the data
mkdir -p data
apptainer exec container.sif \
    huggingface-cli download TimJaspersUe/5LSM0 --local-dir ./data --repo-type dataset
