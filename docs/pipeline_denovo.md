# Unconditional *de novo* design pipeline

`flexcraft.pipelines.denovo` provides an unconditional *de novo* protein design
pipeline combining `salad` backbone generation, `ProteinMPNN` sequence design
with temperature annealing, `AlphaFold2` self-consistency filtering (`sc_lddt`),
and optional `PyRosetta` `FastRelax` refinement.

## Getting started

Design 10 *de novo* 100-residue monomers:

```bash
uv run --no-sync python -m flexcraft.pipelines.denovo \
    --num_aa "100" \
    --num_designs 10 \
    --out_path test_runs/denovo-100-1/
```

If `PyRosetta` is not installed in `.venv`, disable Rosetta relaxation with
`--relax False`:

```bash
uv run python -m flexcraft.pipelines.denovo \
    --num_aa "100" \
    --num_designs 10 \
    --relax False \
    --out_path test_runs/denovo-100-1/
```

## Output structure

`--out_path` is populated with:
- `attempts/design_<attempt>.pdb`: raw `salad` backbone trajectories
- `success/design_<attempt>_<seq_id>.pdb`: `AlphaFold2` predictions passing `sc_lddt > 0.7` (with per-residue `sc_lddt` stored in the B-factor column)
- `relaxed/design_<attempt>_<seq_id>.pdb`: Rosetta `FastRelax`-refined structures (when `--relax True`)
- `all.csv`: sequences and metrics (`sc_rmsd`, `relax_rmsd`, `lddt`, `plddt`, `sc_lddt`, `success`) for all evaluated sequences
- `success.csv`: subset of `all.csv` passing the `sc_lddt > 0.7` filter

## Options

### Parameter paths
Defaults match the layout created by `bash download_params.sh`:
- `--salad_params`: `salad` checkpoint path. Default: `"params/salad/default_vp-200k.jax"`.
- `--pmpnn_params`: `ProteinMPNN` checkpoint path. Default: `"params/pmpnn/v_48_030.pkl"`.
- `--af2_params`: directory containing AlphaFold2 `params/`. Default: `"params/af/"`.
- `--af2_model`: AlphaFold2 model name. Default: `"model_1_ptm"`.

### Design & filtering settings
- `--num_aa`: chain length specification parsed by `salad.inference.parse_num_aa` (e.g. `"100"` for a 100-aa monomer, `"80:80"` for a heterodimer, or `"c60"` for a cyclic chain). Default: `"100"`.
- `--out_path`: output directory path. Default: `"outputs/"`.
- `--num_designs`: number of `salad` backbones to generate. Default: `100`.
- `--num_samples`: maximum number of `ProteinMPNN` sequences to sample per backbone (temperature linearly annealed from `1.0` down across samples; stops early on the first passing sequence per backbone). Default: `10`.
- `--center`: center `ProteinMPNN` logits across positions (`"True"` or `"False"`). Default: `"True"`.
- `--relax`: run `PyRosetta` `FastRelax` on passing designs (`"True"` or `"False"`). Lazily imports `pyrosetta` only when `"True"`. Default: `"True"`.
- `--seed`: random seed for `Keygen`. Default: `42`.
