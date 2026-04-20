import os
import shutil
import gemmi
from copy import deepcopy
import random

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

from Bio.SeqUtils.IsoelectricPoint import IsoelectricPoint as IP

import salad.inference as si
from salad.modules.utils.dssp import assign_dssp
from salad.inference.symmetry import Screw
from salad.modules.utils.geometry import index_mean, index_align, positions_to_ncacocb

from flexcraft.utils import *
from flexcraft.sequence.mpnn import make_pmpnn
from flexcraft.sequence.sample import *
from flexcraft.structure.af import *
from flexcraft.structure.metrics import RMSD
from flexcraft.data.data import DesignData
from flexcraft.files.csv import ScoreCSV
from flexcraft.files.pdb import PDBFile
from flexcraft.protocols import af_cycler
from flexcraft.protocols.bindcraft_filter import BindCraftProperties

from flexcraft.pipelines.utils import bias_dssp

from flexcraft.pipelines.bind.common import (
    setup_dirs, config_from_opt, parse_target, parse_hotspots,
    sample_centers_spaced, save_centers, setup_files, binder_step
)
from flexcraft.pipelines.bind.config import BINDER_OPT, PARAM_PATHS, DESIGN_OPT

import pyrosetta as pr

DESIGN_OPT["num_sequences"] = 2
opt = parse_options(
    "Use salad to generate large protein complexes.",
    in_path="inputs/",
    out_path="output/",
    centermost_k="none",
    use_guess="False",
    target_chains="A",
    **PARAM_PATHS, **BINDER_OPT, **DESIGN_OPT
)
target_chains = np.array([
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ".index(c)
    for c in opt.target_chains.strip().split(",")], dtype=np.int32)

# set up output directories
setup_dirs(opt.out_path)

# set up RNG
key = Keygen(opt.seed)

# set up model configuration
config, salad_params = si.make_salad_model(
    "default_ve_scaled", opt.salad_params)
config = config_from_opt(config, opt)

# get parent paths
base_path = opt.in_path
parent_paths = os.listdir(base_path)
# TODO: refactor this
example_target = PDBFile(path = base_path + "/" + parent_paths[0]).to_data()
is_target = (example_target.chain_index[:, None] == target_chains).any(axis=-1)
config.target_size = is_target.astype(np.int32).sum()

# make sampler
salad_sampler = si.Sampler(
    binder_step(config),
    prev_threshold=opt.prev_threshold,
    out_steps=400,
    timescale="ve(t, sigma_max=80.0)")

# set up ProteinMPNN
pmpnn = jax.jit(make_pmpnn(opt.pmpnn_params, eps=0.05))
# set up logit transform:
transform = lambda center, do_center, T: transform_logits((
    toggle_transform(
        center_logits(center), use=do_center),
    scale_by_temperature(T),
    forbid("C", aas.PMPNN_CODE),
    norm_logits
))

# set up AlphaFold2
af2_params = get_model_haiku_params(
    model_name="model_1_ptm",
    data_dir=opt.af2_params, fuse=True)
af2_config = model_config("model_1_ptm")
af2_config.model.global_config.use_dgram = False
af2 = jax.jit(make_predict(make_af2(af2_config), num_recycle=4))
cycler = af_cycler(jax.jit(make_predict(make_af2(af2_config), num_recycle=0)),
                pmpnn, confidence=None, fix_template=opt.fix_template == "True",
                fix_all=opt.fix_all == "True")
filter = BindCraftProperties(
    opt.out_path, key, opt.af2_params,
    use_guess=opt.use_guess == "True",
    filter=opt.bindcraft_success_filter,
    ipae_shortcut_threshold=opt.ipae_shortcut_threshold)


# set up pyrosetta
pr.init(f'-ignore_unrecognized_res -ignore_zero_occupancy -mute all -holes:dalphaball {opt.alphaball_path} '
        f'-corrections::beta_nov16 true -relax:default_repeats 1')

# set up output files
success, all_designs = setup_files(opt.out_path, prefix_keys=["parent", "parent_rmsd"])

# sample structures
success_count = 0
_attempt = 0
while success_count < opt.num_designs:
    attempt = _attempt
    _attempt += 1
    attempt_success_count = 0
    parent_index = attempt % len(parent_paths)
    parent_path = opt.in_path + "/" + parent_paths[parent_index]
    parent_name = ".".join(parent_paths[parent_index].split(".")[:-1])
    print(parent_name)
    # load starting PDB file
    parent = PDBFile(path=parent_path).to_data()
    # define target mask
    is_target = (parent.chain_index[:, None] == target_chains).any(axis=-1)
    # define starting positions for diffusion
    # TODO: implement simpler DesignData to salad conversion
    init_pos = positions_to_ncacocb(parent["atom_positions"])
    init_pos = jnp.concatenate((
        init_pos,
        jnp.zeros((init_pos.shape[0], config.augment_size, 3), dtype=jnp.float32)), axis=1)
    init_pos = init_pos.at[:, 5:].set(init_pos[:, 1:2])
    # center
    init_pos -= init_pos[:, 1].mean(axis=0)
    # set up salad input
    init_data, init_prev = si.data.from_config(
        config, num_aa=init_pos.shape[0],
        chain_index=parent.chain_index,
        residue_index=parent.residue_index,
        cyclic_mask=jnp.zeros_like(parent.residue_index, dtype=jnp.bool_),
        init_pos=init_pos)
    init_data["is_target"] = is_target
    init_data["aa_condition"] = jnp.where(is_target, parent["aa"], 20)
    target_ca = init_data["pos"][is_target][:, 1]
    # compute binder center
    pos_center = init_data["pos"][~is_target][:, 1].mean(axis=0)
    init_data["center"] = pos_center
    # partial diffusion
    start_steps = random.choice([100, 200, 300, 350])
    design = salad_sampler(salad_params, key(), init_data, init_prev, start_steps=start_steps)
    design = data_from_protein(si.data.to_protein(design))
    parent_rmsd = RMSD()(design, parent)
    if opt.write_attempts == "True":
        design.save_pdb(f"{opt.out_path}/attempts/design_{attempt}.pdb")
    # salad only to check if designed structures are reasonable
    if opt.salad_only == "True":
        success_count += 1
        continue
    design_info = dict(parent=parent_name, parent_rmsd=parent_rmsd, attempt=attempt)
    design_info["n_target_mutations"] = 0 
    init_info = design_info

    # identify contact residues
    ca = design["atom_positions"][:, 1]
    dist_matrix = np.linalg.norm(ca[is_target, None] - ca[None, ~is_target], axis=-1)

    # Create a mask for target residues at the interface (e.g., within 10A of any binder residue)
    is_interface_target = (dist_matrix < opt.redesign_radius).any(axis=1) 
    # Create the base input for MPNN: binders are masked (20), target is fixed.
    target_aa = jnp.where(is_target, aas.translate(init_data["aa_condition"], aas.AF2_CODE, aas.PMPNN_CODE), 20)

    # determine the desired number of target mutations for this specific design: 
    target_mut = opt.allow_target_mutations.split("-")
    if len(target_mut) == 1:
        desired_mutations = int(target_mut[0])
    elif len(target_mut) == 2: 
        lower_bound = int(target_mut[0])
        upper_bound = int(target_mut[1])
        desired_mutations = np.random.randint(lower_bound, upper_bound + 1)
    else:
        raise ValueError("Incorrect format for allowed_target_mut, "
                        "this must be an int passed as a string (e.g. '0') "
                        "or a range separated by '-', e.g. '0-3'.")
    
    # If target mutations are allowed, overwrite parts of the target_aa array.
    if desired_mutations > 0:
        interface_indices = jnp.where(is_interface_target)[0]
        # Randomly choose a subset of these unique local residues to mutate
        num_to_mutate = min(len(interface_indices), desired_mutations)
        
        if num_to_mutate > 0:
            # log number to output csv
            design_info["n_target_mutations"] = num_to_mutate 
            selected_indices = np.random.choice(interface_indices, size=num_to_mutate, replace=False)
            # Mask these chosen residues in the MPNN input array by setting them to 20
            target_aa = target_aa.at[selected_indices].set(20)

    # get center for PMPNN
    logit_center = pmpnn(key(), design.update(aa=target_aa))["logits"][~is_target].mean(axis=0)
    num_sequences = opt.num_sequences
    for idx in range(num_sequences):
        if attempt_success_count >= opt.num_success:
            break
        # ensure that no overwriting occurs
        design_info = {k: v for k, v in init_info.items()}
        temperature = random.choice((0.01, 0.1, 0.2))#, 0.3, 0.5))
        do_center_logits = random.choice((True, False))
        logit_transform = transform(logit_center, do_center_logits, temperature)
        pmpnn_sampler = sample(pmpnn, logit_transform=logit_transform)
        design_info.update(
            seq_id=idx, T=temperature,
            center=do_center_logits)
        pmpnn_input = design.update(aa=target_aa)
        pmpnn_result, _ = pmpnn_sampler(key(), pmpnn_input)
        pmpnn_result = pmpnn_input.update(
            aa=aas.translate(pmpnn_result["aa"], aas.PMPNN_CODE, aas.AF2_CODE))
        
        # write target and binder seq to output csv
        target_seq = pmpnn_result[is_target].to_sequence_string()
        binder_seq = pmpnn_result[~is_target].to_sequence_string()
        design_info.update(target_seq=target_seq, binder_seq=binder_seq)

        # optionally run monomer filter
        if opt.monomer_filter == "True":
            pmpnn_binder = pmpnn_result[-opt.num_aa:]
            monomer_result: AFResult = af2(af2_params, key(), AFInput.from_data(pmpnn_binder))
            plddt = monomer_result.plddt.mean()
            sc_rmsd = RMSD()(monomer_result.to_data(), pmpnn_binder)
            if not (plddt > 0.8 and sc_rmsd < 2.0):
                design_info["success"] = 0
                design_info["failure_reason"] = "bad_monomer"
                all_designs.write_line(design_info)
                continue

        # run bindcraft filter
        af2_result, properties = filter(f"design_{attempt}_{idx}", pmpnn_result, is_target=is_target)
        design_info.update(properties)
        design_info["failure_reason"] = properties["reason"]
        design_info["success"] = int(design_info["success"])

        if not properties["success"]:
            design_info["failure_reason"] = properties["reason"]
            all_designs.write_line(design_info)
            continue
        
        design_info.update(success=1, failure_reason="none")
        all_designs.write_line(design_info)
        success.write_line(design_info)
        shutil.copyfile(f"{opt.out_path}/relaxed/design_{attempt}_{idx}_model_1.pdb",
                        f"{opt.out_path}/success/design_{attempt}_{idx}.pdb")
        success_count += 1
        attempt_success_count += 1
