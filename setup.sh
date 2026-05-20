#! /usr/bin/bash
# get miniforge
wget "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh

# clone repo
git clone https://github.com/NTBiotech/flexcraft.git flexcraft
cd flexcraft
git checkout adaptOrigin1

# Prepare conda env
mamba create -n flexcraft python=3.10
conda activate flexcraft
pip install -e .

# Install af params
mkdir params
cd params
mkdir af
cd af
mkdir params
wget https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar | tar -x -C params
cd ../..

# load test data for abscibind
cd tests
python <<EOF
from abscibind import load_data
load_data("../../o1_iptm_scoring")
EOF