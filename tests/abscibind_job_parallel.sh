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

# ───Info─────────────────────────────────────────────────────
# Script for running concurrent abscibind runs on multiple gpus
# configured for juwels cluster
# ────────────────────────────────────────────────────────

# ── User-defined paths ────────────────────────────────────────────────────────
CONDA_PATH="programmes/miniforge3"  # path to miniforge or ... relative to project dir
CONDA_ENV="flexcraft"               # e.g. flexcraft
AF_PARAMS="params/af/params"        # e.g. /path/to/BinderDesign/params/af
PROJECT_NAME="hai_1252"             # project id on cluster
REPO_NAME="flexcraft"
# ──────────────────────────────────────────────────────────────────────────────
jutil env activate -p "$PROJECT_NAME"
PROJECT_DIR="$PROJECT"   # absolute path on cluster

source ~/.bashrc
cd "$PROJECT_DIR"
source "$CONDA_PATH/bin/activate"
conda activate "$CONDA_ENV"
module purge
export CUDA_VISIBLE_DEVICES=0
cd "$PROJECT_DIR/$REPO_NAME"



COUNT=0
for targets in "IL36R,CD40,ACVR2B" "C5,TSLP" "IL17A,FXI" "HER2,TNFRSF9"; do
    CUDA_VISIBLE_DEVICES=$COUNT python tests/test_abscibind.py --af_model "model_1_ptm" --verbose --targets $targets &
    COUNT=$((COUNT + 1))
done
wait