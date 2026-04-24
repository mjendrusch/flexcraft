#!/bin/bash
#SBATCH --job-name=abscibind
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64gb
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:full:1
#SBATCH --output=logs/abscibind_%j.out
#SBATCH --error=logs/abscibind_%j.err
# ── User-defined paths ────────────────────────────────────────────────────────
CONDA_ENV="flexcraft"           # e.g. flexcraft
AF_PARAMS="params/af/params"       # e.g. /path/to/BinderDesign/params/af
PROJECT_DIR="/home/hgf_dkfz/hgf_dsb0249/workspaces/haicwork/hgf_dsb0249-BinderDesign/flexcraft"   # absolute path on cluster
# ──────────────────────────────────────────────────────────────────────────────
source ~/.bashrc
conda activate "$CONDA_ENV"
module purge
cd "$PROJECT_DIR"

export CUDA_VISIBLE_DEVICES=0

python tests/test_abscibind.py --verbose --max_designs 5 --n_recycle 4
