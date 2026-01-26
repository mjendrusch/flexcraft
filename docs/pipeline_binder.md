# `salad` binder pipeline

`flexcraft.pipelines.binder` provides a single-script binder design pipeline
using `salad` for backbone generation, ProteinMPNN for sequence design
and `BindCraft`-type filters for design selection.

## Getting started

To design binders to one of the example target in `examples/binder_targets/pd1_hniu.pdb`,
you can run the following command:
```bash
python -m flexcraft.pipelines.binder \
    --target examples/binder_targets/pd1_hniu.pdb \
    --out_path test_runs/binder-pd1-80-1/ \
    --num_aa 80 --num_designs 10 --use_cycler True \
    --hotspots "A56,A123" \
    --h_bias 0.3 --radius 14.0 \
    --fix_template True --relax_cutoff 0.0
```

This will run 80-amino-acid binder design for human PD-1 until 10 designs
have passed all filters.

Here, `--hotspots` specifies a set of hotspot residues for binder targeting.
As the target is a mostly-beta structure, we introduce a non-zero `--h_bias`
to ensure that the model generates a balanced mix of alpha and beta binders.
As the target site is a flat surface we increase `--radius` to 12.0
Angstroms, to ensure that the binder designs have sufficient space.
Finally, we set `--fix_template True` and `--relax_cutoff 0.0`,
as we do not want the target to change conformation.

## Options
### parameter & dependency paths
All default paths point to the default parameters in the directory
structure created after running `download_params.sh`.
- `--salad_params`:
    path to the `salad` checkpoint used for design.
- `--pmpnn_params`:
    path to the ProteinMPNN checkpoint used for design.
- `--af2_params`:
    path to a directory containing a subdirectory named `params`, which
    contains the AF2 checkpoints.
- `--alphaball_path`:
    path to the DAlphaBall executable.
### input and output files
- `--target`:
    path to the target PDB file.
- `--out_path`:
    path to the output directory. Directory will be created if it does
    not exist.
- `--write_cycles`:
    write AF2-cycler intermediate files? "True" or "False".
    "True" by default.
- `--write_failed`:
    write AF2 predictions of failed designs? "True" or "False".
    Default: "True".
- `--dry_run`:
    only run and write `salad` outputs? "True" of "False".
    Default: "False".
### target & binder settings
- `--hotspots`:
    comma-separated list of hotspot residues to target. E.g. "A25,A76-100".
    residue chains and indices should be specified as they are in the target
    PDB file.
- `--coldspots`:
    comma-separated list of coldspot residues which should not be
    in contact with designed binders.
- `--num_aa`:
    number of amino acids in the designed binder.
    We recommend `--num_aa 80` for campaigns where the experimental
    endpoint involves display, as designs can be ordered as an oligo pool.
- `--[h|e|l]_bias`: bias the binder for **h**elix, sh**e**et or **l**oop.
- `--radius`:
    radius in Angstroms at which designs are initialized from any hotspot.
    Default: 10.0. Consider tuning this for new targets.
- `--clash_radius`:
    distance to the nearest target CA in Angstroms at which design
    initialization is rejected as likely clashing.
    Default: 10.0. Consider tuning this together with `--radius` for
    new targets.
- `--coldspot_radius`:
    distance to the nearest coldspot residue in Angstroms at which
    design initialization is rejected as being off-target.
    Default: 15.0.
- `--visualize_centers`:
    write random binder initialization positions with current settings
    to `out_path/target_centers.pdb`.
    use this when tuning `--hotspots` and related options.
### campaign size settings
- `--num_designs`: number of requested designs passing all filters. Default: 48.
    Consider generating more designs than required, then selecting the best
    designs by ipAE.
- `--num_sequences`: number of ProteinMPNN sequences per backbone. Default: 10.
- `--num_success`: number of successful designs to look for per backbone. 
    Default: 1.
### design process settings
- `--prev_threshold`:
    threshold diffusion time above which self-conditioning will be used.
    Default: 0.9.
- `--clash_lr`:
    clash potential learning rate. Higher values encourage less compact,
    more helical structures with fewer potential clashes.
    Default: 5e-3. Reasonable range: 0.0 - 7e-2.
- `--compact_lr`:
    compactness potential learning rate. Higher values encourage more compact
    structures but also increase potential for clashes.
    Therefore, larger `compact_lr` necessitates larget `clash_lr`.
    Default: 1e-4. Reasonable range: 0.0 - 2e-3.
- `--contact_lr`:
    binder-target contact potential.
    Default: 1e-2. Reasonable range: 0.0 - 5e-2.
- `--relax_cutoff`:
    diffusion time threshold below which binders are not fixed to their
    initialization position allowing for interface refinement.
    Default: 3.0.
- `--use_cycler`:
    use AF2-cycler-like protocol for binders? "True" or "False".
    Default: "False".
- `--num_cycles`:
    number of AF2-cycler iterations. Default: 10.
- `--fix_template`:
    use full target sequence for target template during AF2-cycler
    to exactly fix target positions? "True" or "False".
    Default: "False". This allows for target flexibility during design.
- `--ipae_shortcut_threshold`:
    maximum normalized ipAE threshold to consider Rosetta relaxation.
    Default: 0.35
- `--seed`: `jax` starting random seed
### binder design with target mutations
- `--allow_target_mutations`:
    number of target interface residues allowed to be redesigned by ProteinMPNN. Can take a range for random sampling, e.g. "0-3".
    Default: 0.
- `--redesign_radius`: 
    target residue distance for redesign in Angstrom (if target mutations are allowed). Default: 10.0 Angstroms. 