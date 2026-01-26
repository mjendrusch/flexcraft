# flexcraft

`flexcraft` is a library for combining protein generative models, sequence design and structure predictors
to write end-to-end, single-script protein design pipelines à la BindCraft.
Currently, `flexcraft` mostly relies on [salad](github.com/mjendrusch/salad) for backbone generation,
ProteinMPNN-variants for sequence design and AF2 for structure prediction and filtering.
It also provides access to PyRosetta for additional physics-based filtering.

`flexcraft` is currently in the early stages of development. Documentation is sparse and APIs might change,
however, `flexcraft` already provides a number of end-to-end protein design [pipelines](#pipelines) that are ready to use.

## Getting started
### Installation
To install `flexcraft` and `salad`, simply clone this repository and follow the steps: 

```bash
# clone repository
git clone https://github.com/mjendrusch/flexcraft.git
# set up environment
conda create -n flexcraft python=3.10
conda activate flexcraft
# install flexcraft, salad & dependencies
cd flexcraft
pip install -e .
# download model parameters
bash download_params.sh
# optionally install PyRosetta
pip install pyrosetta-installer
python -c 'import pyrosetta_installer; pyrosetta_installer.install_pyrosetta()'
```

### Getting your 1st protein
Check if `flexcraft` is working properly on your machine / GPU by having it design a protein *de novo*:
```bash
python -m flexcraft.pipelines.denovo --num_aa "100" --num_designs 10 --out_path test_outputs/
```
If your machine has more than 1 GPU, consider using `CUDA_VISIBLE_DEVICES=0` to restrict the script to
the first GPU on the machine.

This will create a directory `test_output` with the following contents:
- `attempts/*.pdb`: output designs from `salad` prior to sequence design
- `fail/*.pdb`: designs that failed AF2 filters after ProteinMPNN sequence design
- `success/*.pdb`: designs that successfully passed AF2 filters after ProteinMPNN sequence design
- `all.csv`: AF2 scores and sequences for all ProteinMPNN-designed sequences tested
- `success.csv`: AF2 scores and sequences for all designs passing AF2 filters

This directory structure is common to all [pipelines](#pipelines) currently implemented in `flexcraft`.

## Pipelines
`flexcraft` implements customizable pipelines for a number of protein design tasks:
- motif scaffolding:
  - [`flexcraft.pipelines.graft.simple`](docs/pipeline_motif.md):
    pipeline for simple motif scaffolding tasks with small motifs,
    no symmetry requirements and no large inter-motif distances
  - [`flexcraft.pipelines.graft.general`](docs/pipeline_motif.md):
    pipeline for more involved motif scaffolding tasks, with shape
    or symmetry requirements, large motifs (>50 residues)
- binder design:
  - [`flexcraft.pipelines.binders`](docs/pipeline_binders.md)
- unconditional *de novo* design:
  - [`flexcraft.pipelines.denovo`](docs/pipeline_denovo.md)

## How to write `flexcraft` pipelines
For now, flexcraft is rather sparsely documented. Have a look at `scripts/` and `flexcraft/pipelines`
for examples on how to write pipelines.
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
