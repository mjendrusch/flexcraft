import os
import random
from collections import defaultdict
import numpy as np
import jax.numpy as jnp

from flexcraft.pipelines.graft.common import (
    setup_directories, model_step, get_assembly_lengths, sample_data,
    make_simple_assembly, get_motif_paths)
from flexcraft.pipelines.graft.options import MODEL_OPTIONS, SAMPLER_OPTIONS
from flexcraft.utils import *
from flexcraft.files.csv import ScoreCSV
from flexcraft.sequence.mpnn import make_pmpnn
from flexcraft.sequence.sample import *
from flexcraft.structure.af import *
from flexcraft.structure.metrics import RMSD
from flexcraft.utils import *

import salad.inference as si

def parse_center_file(path):
    with open(path, "rt") as f:
        center = []
        center_group = []
        center_time = []
        for line in f:
            length, group, x, y, z, ct = line.strip().split(",")
            length, group = int(length), int(group)
            x, y, z = float(x), float(y), float(z)
            ct = float(ct)
            center += [[x, y, z]] * length
            center_group += [group] * length
            center_time += [ct] * length
    center = np.array(center, dtype=np.float32)
    center -= center.mean(axis=0)
    center_group = np.array(center_group, dtype=np.int32)
    center_time = np.array(center_time, dtype=np.int32)
    return center, center_group, center_time

def parse_pseudo_center_file(path):
    with open(path, "rt") as f:
        center = []
        center_group = []
        center_time = []
        first_line = True
        for line in f:
            if line.startswith(("center", "group")) and first_line:
                # skip header
                first_line = False
                continue
            fields = line.strip().split(",")
            if len(fields) < 5:
                group, x, y, z = fields
                ct = "0.0"
            else:
                group, x, y, z, ct = fields
            group = int(group)
            x, y, z = float(x), float(y), float(z)
            ct = float(ct)
            center += [[x, y, z]]
            center_group += [group]
            center_time += [ct]
            first_line = False
    center = np.array(center, dtype=np.float32)
    center_group = np.array(center_group, dtype=np.int32)
    center_time = np.array(center_time, dtype=np.int32)
    return dict(center=center, center_group=center_group, center_time=center_time)

opt = parse_options(
    "Simple motif grafting with salad.",
    **MODEL_OPTIONS,
    **SAMPLER_OPTIONS,
    out_path="outputs/",
    motif_path="motifs/",
    assembly="20-50,A1-20@0,20-50",
    symmetry="none",
    centers="none",
    center_radius=12.0,
    buried_contacts=6,
    compact_lr=0.0, # 1e-4 - 1e-3
    clash_lr=0.0, # 1e-3 - 7e-2
    compact_by="chain", # &0 &1
    use_motif_dssp="False",
    use_motif_aa="none",
    center_to_chain="False",
    scaffold_to_glycine="False",
    timescale="ve(t, sigma_max=80.0)",
    seed=42
)

# set up output directories
setup_directories(opt.out_path)

# get motif PDB files
motif_paths = get_motif_paths(
    opt.motif_path, opt.assembly)

# set up random key
key = Keygen(opt.seed)

# model setup
task = defaultdict(
    lambda: None, mode=["grad"], 
    potentials=dict(compact=opt.compact_lr, clash=opt.clash_lr),
    use_motif_dssp=opt.use_motif_dssp == "True",
    use_motif_aa=opt.use_motif_aa == "True",
    center_to_chain=opt.center_to_chain == "True",
    compact_by=opt.compact_by,
    buried_contacts=opt.buried_contacts)
pseudo_centers = None
if opt.centers != "none":
    task["center"] = True
    pseudo_centers = parse_pseudo_center_file(opt.centers)
if opt.symmetry != "none":
    task["sym"] = int(opt.symmetry)
salad_config, salad_params = si.make_salad_model(
    opt.salad_config, opt.salad_params)
salad_config.center_radius = opt.center_radius
salad_sampler = si.Sampler(model_step(salad_config, task),
                           prev_threshold=opt.prev_threshold,
                           out_steps=[400],
                           timescale=opt.timescale)

for motif, assembly in motif_paths:
    # base name of the motif path without the final file ending
    # other "." in the file name are preserved
    motif_name = ".".join(os.path.basename(motif).split(".")[:-1])
    # get settings dictionary for a motif and simple assembly specification
    settings = make_simple_assembly(motif, assembly)
    # get the maximum length of designs with that specification
    _, max_length = get_assembly_lengths(
        settings["segments"], settings["assembly"])
    for idx in range(opt.num_designs):
        data, init_prev = sample_data(
            salad_config, settings["segments"], settings["assembly"],
            pos_init=True, pseudo_centers=pseudo_centers)
        has_motif = data["has_motif"]
        aa_condition = data["motif_aa"]
        data = pad_dict(data, max_length)
        init_prev = pad_dict(init_prev, max_length)
        data = pad_dict(data, max_length)
        init_prev = pad_dict(init_prev, max_length)
        steps = salad_sampler(salad_params, key(), data, init_prev)
        for ids, design in enumerate(steps):
            design = data_from_protein(si.data.to_protein(design))
            if opt.scaffold_to_glycine == "True":
                # fix all non-relaxed motif amino acids
                non_glycine = has_motif * (design.aa == aa_condition) > 0
                # make all generated and relaxed motif amino acids glycines
                design = design.update(
                    aa=jnp.where(non_glycine, aa_condition, aas.AF2_CODE.index("G")))
            design.save_pdb(f"{opt.out_path}/attempts/{motif_name}_design_{idx}_{ids}.pdb")
