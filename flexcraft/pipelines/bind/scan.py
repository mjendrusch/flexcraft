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
from salad.modules.utils.geometry import index_mean, index_align

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
    sample_centers_spaced, save_centers, setup_files_scan, binder_step
)
from flexcraft.pipelines.bind.config import BINDER_OPT, PARAM_PATHS, DESIGN_OPT

import pyrosetta as pr

DESIGN_OPT.update(num_designs=5)
opt = parse_options(
    "Use salad to generate large protein complexes.",
    out_path="output/",
    target_path="targets/",
    max_attempts=200,
    target_offset=0,
    target_step=1,
    randomize="False",
    **PARAM_PATHS, **BINDER_OPT, **DESIGN_OPT
)

# set up output directories
setup_dirs(opt.out_path)

# set up RNG
seed = opt.seed
if opt.randomize == "True":
    seed = random.randint(0, 1_000_000)
key = Keygen(seed)

# set up model configuration
config, salad_params = si.make_salad_model(
    "default_ve_scaled", opt.salad_params)
config = config_from_opt(config, opt)

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

# set up pyrosetta
pr.init(f'-ignore_unrecognized_res -ignore_zero_occupancy -mute all -holes:dalphaball {opt.alphaball_path} '
        f'-corrections::beta_nov16 true -relax:default_repeats 1')

# set up output files
success, stats = setup_files_scan(opt.out_path)

targets = os.listdir(opt.target_path)

# sample structures
for target in targets[opt.target_offset::opt.target_step]:
    target: str
    target_name = target.split(".")[0]
    print(f"Processing target {target_name} ...")
    full_path = f"{opt.target_path}/{target}"
    init_data, init_prev, target_center, is_target, target_size = parse_target(
        config, full_path, opt.num_aa)
    if init_data["plddt"][:, 1].mean() < 0.7:
        print(init_data["plddt"])
        print(f"skipping low-pLDDT target {target_name}")
        continue
    config = config_from_opt(config, opt)
    hotpos, coldpos, hotspot_mask, coldspot_mask = parse_hotspots(
        init_data, "auto", "plddt")
    target_ca = init_data["pos"][is_target][:, 1]
    pos_centers = sample_centers_spaced(
        target_ca, hotpos,
        radius_min=opt.radius,
        clash_radius=opt.clash_radius,
        min_distance=opt.min_center_distance,
        coldspot_radius=opt.coldspot_radius,
        coldpos=coldpos)
    if pos_centers is None:
        print(f"skipping target {target_name}: no centers found")
        continue
    save_centers(full_path, f"{opt.out_path}/centers_{target_name}.pdb", target_center, pos_centers)
    # make sampler
    salad_sampler = si.Sampler(binder_step(config),
                            prev_threshold=opt.prev_threshold,
                            out_steps=400,
                            timescale="ve(t, sigma_max=80.0)")

    success_count = 0
    _attempt = 0
    while success_count < opt.num_designs and _attempt < opt.max_attempts:
        print(f"Processing target {target_name} ...")
        attempt = _attempt
        _attempt += 1
        attempt_success_count = 0
        pos_center = pos_centers[np.random.randint(0, pos_centers.shape[0])]
        xyz = dict(x=pos_center[0], y=pos_center[1], z=pos_center[2])
        init_data["center"] = pos_center
        init_data["pos"] = jnp.where(is_target[:, None, None], init_data["pos"], pos_center)
        # bias secondary structure
        init_data = bias_dssp(init_data, L=opt.l_bias, H=opt.h_bias, E=opt.e_bias, where=~is_target)
        design = salad_sampler(salad_params, key(), init_data, init_prev)
        design = data_from_protein(si.data.to_protein(design))
        # design.save_pdb(f"{opt.out_path}/attempts/design_{attempt}.pdb")
        LHE = design.p_dssp
        design_info = dict(target=target_name, attempt=attempt)
        design_info.update(**xyz)
        design_info.update(**LHE)
        init_info = design_info
        # optionally apply af2cycler
        if opt.use_cycler == "True":
            cycled = design
            for idc in range(opt.num_cycles):
                cycled, predicted = cycler(af2_params, key, cycled, cycle_mask=~is_target)
            predicted = predicted.align_to(design, weight=is_target)
            cycled = cycled.align_to(design, weight=is_target)
            p_target = predicted["atom_positions"][is_target, 1]
            p_binder = predicted["atom_positions"][~is_target, 1]
            contact = (np.linalg.norm(p_binder[:, None] - p_target[None, :], axis=-1) < 8.0).any(axis=1).astype(np.int32).sum()
            center_cycled = predicted["atom_positions"][~is_target, 1].mean(axis=0)
            center_design = design["atom_positions"][~is_target, 1].mean(axis=0)
            cycle_drift = np.linalg.norm(center_cycled - center_design)
            center_drift = np.linalg.norm(center_design - pos_center)
            design_info.update(cycle_drift=cycle_drift, center_drift=center_drift, num_contacts=contact)
            if opt.write_cycles == "True":
                predicted.save_pdb(f"{opt.out_path}/cycles/design_{attempt}_{idc}.pdb")
            # FIXME: stop cycling from launching the protein into space
            if contact > 10:
                design = cycled

        # Create the base input for MPNN: binders are masked (20), target is fixed.
        target_aa = jnp.where(is_target, aas.translate(init_data["aa_condition"], aas.AF2_CODE, aas.PMPNN_CODE), 20)

        # get center for PMPNN
        logit_center = pmpnn(key(), design.update(aa=target_aa))["logits"][target_size:].mean(axis=0)
        num_sequences = opt.num_sequences
        for idx in range(num_sequences):
            if attempt_success_count >= opt.num_success:
                break
            # ensure that no overwriting occurs
            design_info = {k: v for k, v in init_info.items()}
            temperature = random.choice((0.01, 0.1, 0.2))#, 0.3, 0.5))
            design_info.update(T=temperature)
            do_center_logits = True#random.choice((True, False))
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
            pmpnn_binder = pmpnn_result[-opt.num_aa:]
            monomer_result: AFResult = af2(af2_params, key(), AFInput.from_data(pmpnn_binder))
            plddt = monomer_result.plddt.mean()
            sc_rmsd = RMSD()(monomer_result.to_data(), pmpnn_binder)
            design_info.update(plddt=plddt, sc_rmsd=sc_rmsd)
            print(binder_seq, f"plddt: {plddt:.2f}", f"scRMSD: {sc_rmsd:.2f}")
            if plddt < 0.8 or sc_rmsd > 2.0:
                design_info.update(success_monomer=0, success_binder=0)
                stats.write_line(design_info)
                continue
            design_info.update(success_monomer=1)

            # run bindcraft filter
            af2_result: AFResult = af2(
                af2_params, key(),
                AFInput.from_data(pmpnn_result)
                .add_guess(pmpnn_result)
                .add_template(pmpnn_result, where=init_data["is_target"]))
            design_info["i_pAE"] = af2_result.ipae.mean()
            print(binder_seq, f"ipAE: {af2_result.ipae.mean():.3f}")
            reason = "none"
            if design_info["i_pAE"] >= 0.35:
                reason = "ipAE"
            design_info["failure_reason"] = reason
            design_info["success_binder"] = int(reason == "none")

            if not design_info["success_binder"]:
                stats.write_line(design_info)
                continue
            
            success.write_line(design_info)
            stats.write_line(design_info)
            af2_result.save_pdb(f"{opt.out_path}/success/binder_{target_name}_{attempt}_{idx}.pdb")
            success_count += 1
            attempt_success_count += 1
