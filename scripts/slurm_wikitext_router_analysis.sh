#!/bin/bash
#SBATCH -A amowli_lab_gpu
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:A30:2
#SBATCH --time=00:58:00
#SBATCH --mem=100GB

source ./venv.sh

python scripts/wikitext_router_analysis.py --model-name olmoe --hf-cache-dir /pub/afeeney/huggingface/cache/