# flexcraft

Very, very experimental attempt at a library for plugging together single-script protein structure generation, design and AF2 filtering.
This aspires to become a library that helps write tools like BindCraft, but for arbitrary protein design tasks.
At the moment, it is still very far away from that state and is very much **work in progress**, the API **will** change.

The only reason this is a public repository is because I want to experiment with it in Google Colab.

## Installation
Installation is not polished yet, might require some fiddling with packages.
```
conda create -n flexcraft python=3.10
conda activate flexcraft
wget https://zenodo.org/records/14711580/files/salad_params.tar.gz
tar -xzf salad_params.tar.gz
git clone https://github.com/mjendrusch/flexcraft.git
cd flexcraft
pip install -e .
mkdir pmpnn_params
cd pmpnn_params
for noise in 05 10 20 30; do
    wget https://github.com/sokrypton/ColabDesign/raw/refs/heads/main/colabdesign/mpnn/weights/v_48_0${noise}.pkl
done
cd ..
mkdir solmpnn_params
cd solmpnn_params
for noise in 05 10 20 30; do
    wget https://github.com/sokrypton/ColabDesign/raw/refs/heads/main/colabdesign/mpnn/weights/v_48_0${noise}.pkl
done
cd ..
mkdir params
curl -fsSL https://storage.googleapis.com/alphafold/alphafold_params_2022-03-02.tar | tar x -C params
cd ..
```

## Usage
I would not suggest using this repository at this stage.
But if you really want to, there are some example scripts in the `scripts/` directory.
Once I figure out what I want the API to actually look like, I will add proper usage documentation.
