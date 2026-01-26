# simple motif scaffolding pipeline

`flexcraft.pipelines.graft.simple` provides a single-script motif grafting
pipeline using `salad` for backbone generation, ProteinMPNN for sequence 
design and AF2 for structure prediction and design selection.

## Getting started

To design scaffolds for a part of PD-1 at
`examples/binder_targets/pd1_hniu.pdb`, you can run the following command:
```bash
python -m flexcraft.pipelines.graft.simple \
    --motif_path examples/binder_targets/pd1_hniu.pdb \
    --out_path test_runs/motif-pd1-1/ \
    --assembly "50:A111-117:50:A121-127:50" \
    --num_designs 10 --use_motif_aa all \
    --config default_vp --salad_params params/salad/default_vp-200k.jax \
    --timescale "cosine(t)"
```

This will run motif scaffolding for two surface-exposed beta strands in PD-1.
Here, `--assembly` specifies how we want to scaffold the motif:
The two strands `A111-117` and `A121-127` are scaffolded with 50 residues
on each side and 50 residues in between the strands.
The amino acid sequence of the motif is preserved (`--use_motif_aa all`)
and a variance-preserving (vp) diffusion model is used for `salad`.

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
### input and output files
- `--motif_path`:
    path to the motif file PDB file.
- `--out_path`:
    path to the output directory. Directory will be created if it does
    not exist.
- `--write_failed`:
    write AF2 predictions of failed designs? "True" or "False".
    Default: "False".
- `--salad_only`:
    only run and write `salad` outputs? "True" of "False".
    Default: "False".
### motif settings
- `--assembly`:
    assembly string specifying how motifs should be scaffolded.
    Assembly strings can contain multiple ":"-separated **chains**,
    each containing one or more ","-separated **segments**.
    - motif segments:
        contain motif information, relative to the input motif.pdb file.
        E.g. `A20-30` specifies that amino acids `20-30` from chain `A`
        should be placed in this segment.
        Motif segments may also specify motif group information;
        motifs belonging to the same group are scaffolded together,
        while motifs belonging to different groups are scaffolded
        separately. By default, all motif segments are in group `0`.
        To specify a different motif group, use a motif segment
        like `A20-30@3`, which places the motif in the group following
        the `@` separator.
    - non-motif segments:
        contain a length specification for a fully designable segment.
        E.g. `50` for 50 designable amino acids, or `10-30` for a random
        number between 10 and 30 amino acids.
    - examples:
        - `50,A20-30,10-20,A40-50,50`: a single chain with two parts
            of the same motif with a variable number of amino acids
            between them.
        - `50,A1-20,30:50,B1-20,30`: two chains scaffolding a motif
            that is itself the interface between chains A and B
            in the motif PDB file.
        - `50,A1-20@0,50,A1-20@1,50`: two copies of the same motif
            in different motif groups.
        - `20,A1-20,20:B`: scaffold the motif on chain A, taking along
            all of chain B. This is useful if a motif should be scaffolded
            in such a way, that the scaffold is compatible with interacting
            with chain B.
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
- `--[h|e|l]_bias`: bias the design for **h**elix, sh**e**et or **l**oop.
- `--align_threshold`: threshold diffusion time below which to use
    structural alignment for motif scaffolding.
- `--dmap_threshold`: threshold diffusion time below which to use
    distance map input for motif scaffolding.
- `--template_motif`:
    apply a template to the motif during structure prediction?
    "True" or "False". Default: False.
- `--use_motif_aa`:
    provide motif sequence information to the model?
    One of "none", "all", "within_motif".
    - `none`: provides no sequence information.
    - `all`: default. Provides full sequence information.
    - `within_motif`: Only provides buried residue sequence information.
- `--mask_motif_plddt`:
    should motif pLDDT be excluded from the pLDDT average for filtering?
    "True" or "False". Default: "True".
- `--timescale`:
    diffusion timescale. For variance-preserving checkpoints (vp),
    should be `cosine(t)`. For variance-expanding checkpoints (ve),
    should be `ve(t)`.
- `--seed`: `jax` starting random seed
