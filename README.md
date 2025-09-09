# flexcraft

Experimental attempt at a library for plugging together single-script protein structure generation, design and AF2 filtering.
This aspires to become a library that helps write tools like BindCraft, but for arbitrary protein design tasks.
At the moment, it is still very much **work in progress**, the API **will** change.

However, it is now in a somewhat usable state. I would not recommend using it for production yet, but it will get there.

## Installation
```
# set up environment
conda create -n flexcraft python=3.10
conda activate flexcraft
pip install git+https://github.com/mjendrusch/flexcraft.git
# download and extract model parameters
wget https://zenodo.org/records/14711580/files/salad_params.tar.gz
tar -xzf salad_params.tar.gz
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

## How to use flexcraft?
For now, flexcraft is rather sparsely documented. Have a look at `scripts/` for examples on how to use it.
In general, flexcraft is built on top of the `DesignData` class in `flexcraft.data`, which can be used as
input for all the models wrapped in flexcraft.

E.g. for running AlphaFold predictions (see the script at the bottom of `flexcraft.structure.af._model`):
```python
# given a DesignData object
af_data: DesignData = ...
# make an AlphaFold input with initial guess and compute plddt and pae:
features = AFInput.from_data(af_data).add_guess(af_data)
result: AFResult = predictor(params, key(), features)
plddt = result.plddt.mean()
pae = result.pae.mean()
```

In the scripts folder, you will find the following example scripts.
- `simple_salad.py`: generate structures using salad with random secondary structure
  conditioning, design sequences with ProteinMPNN and predict structures with AlphaFold 2
- `test_hal.py`: generate structures using AF2 hallucination.
