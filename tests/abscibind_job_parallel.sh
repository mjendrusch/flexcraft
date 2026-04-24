#!/bin/bash -x
#SBATCH --job-name=abscibind
#SBATCH --time=02:00:00
#SBATCH --account=hai_1252
# budget account where contingent is taken from
#SBATCH --nodes=1
#SBATCH --ntasks=4
# can be omitted if --nodes and --ntasks-per-node
# are given
#SBATCH --ntasks-per-node=4
# if keyword omitted: Max. 96 tasks per node
# (SMT enabled, see comment below)
#SBATCH --output=logs/abscibind_%j.out
#SBATCH --error=logs/abscibind_%j.err
#SBATCH --partition=develbooster
#SBATCH --gres=gpu:4
# For gpus and and booster partition

# *** start of job script ***
# Note: The current working directory at this point is
# the directory where sbatch was executed.



# ───Info─────────────────────────────────────────────────────
# Script for running concurrent abscibind runs on multiple gpus
# configured for juwels cluster
# ────────────────────────────────────────────────────────

# ── User-defined paths ────────────────────────────────────────────────────────
CONDA_PATH="miniforge3"  # path to miniforge or ... relative to project dir
CONDA_ENV="flexcraft"               # e.g. flexcraft
AF_PARAMS="params/af/params"        # e.g. /path/to/BinderDesign/params/af
PROJECT_NAME="hai_1252"             # project id on cluster
REPO_NAME="flexcraft"
# ──────────────────────────────────────────────────────────────────────────────
jutil env activate -p "$PROJECT_NAME"
PROJECT_DIR="$PROJECT"   # absolute path on cluster

source ~/.bashrc
cd "$PROJECT_DIR/toulouse1"
source "$CONDA_PATH/bin/activate"
conda activate "$CONDA_ENV"
module purge
export CUDA_VISIBLE_DEVICES=0

cd "$REPO_NAME"
pwd

COUNT=0
for targets in "IL36R CD40 ACVR2B" "C5 TSLP" "IL17A FXI" "HER2 TNFRSF9"; do
    echo "$targets"
    CUDA_VISIBLE_DEVICES=$COUNT python tests/test_abscibind.py --verbose --targets $targets --n_recycle 4 &
    COUNT=$((COUNT + 1))
done
wait