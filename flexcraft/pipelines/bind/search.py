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
    sample_centers_spaced, save_centers, setup_files, binder_step
)
from flexcraft.pipelines.bind.config import BINDER_OPT, PARAM_PATHS, DESIGN_OPT
from flexcraft.protocols.search import beam_search, binder_fitness, genetic_search, genetic_binder_fitness, salad_proposal

# import pyrosetta as pr

opt = parse_options(
    "Use salad to generate large protein complexes.",
    out_path="output/",
    target="target.pdb",
    centermost_k="none",
    use_guess="False",
    target_chains="all",
    **PARAM_PATHS, **BINDER_OPT, **DESIGN_OPT
)

# set up output directories
setup_dirs(opt.out_path)

# set up RNG
key = Keygen(opt.seed)

# set up model configuration
config, salad_params = si.make_salad_model(
    "default_ve_scaled", opt.salad_params)
init_data, init_prev, target_center, is_target, target_size = parse_target(
    config, opt.target, opt.num_aa, target_chains=opt.target_chains)
config = config_from_opt(config, opt)
hotpos, coldpos, hotspot_mask, coldspot_mask = parse_hotspots(
    init_data, opt.hotspots, opt.coldspots)
target_ca = init_data["pos"][is_target][:, 1]
pos_centers = sample_centers_spaced(
    target_ca, hotpos,
    radius_min=opt.radius,
    clash_radius=opt.clash_radius,
    min_distance=opt.min_center_distance,
    centermost_k=None if opt.centermost_k == "none" else int(opt.centermost_k),
    coldspot_radius=opt.coldspot_radius,
    coldpos=coldpos)
# optionally write a PDB file with all valid center positions
save_centers(opt.target, f"{opt.out_path}/target_centers.pdb",
                target_center, pos_centers)
if opt.centers_only == "True":
    exit(0)

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
#search = beam_search(salad_sampler, binder_fitness(pmpnn, transform, af2, af2_params, key))
proposal = salad_proposal(config, salad_params, pmpnn, key, binder_step(config),
                          out_steps=400, prev_threshold=opt.prev_threshold,
                          timescale="ve(t, sigma_max=80.0)")
fitness = genetic_binder_fitness(af2, af2_params, key)

search = genetic_search(proposal, fitness, init=lambda x: x, steps=10)


# set up pyrosetta
# pr.init(f'-ignore_unrecognized_res -ignore_zero_occupancy -mute all -holes:dalphaball {opt.alphaball_path} '
#         f'-corrections::beta_nov16 true -relax:default_repeats 1')

# set up output files
success, all_designs = setup_files(opt.out_path)

# sample structures
success_count = 0
_attempt = 0
while success_count < opt.num_designs:
    attempt = _attempt
    _attempt += 1
    pos_center = pos_centers[np.random.randint(0, pos_centers.shape[0])]
    init_data["center"] = pos_center
    init_data["pos"] = jnp.where(is_target[:, None, None], init_data["pos"], pos_center)
    # bias secondary structure
    init_data = bias_dssp(init_data, L=opt.l_bias, H=opt.h_bias, E=opt.e_bias, where=~is_target)
    # successes, instances = search(
    #     salad_params, key, init_data, init_prev,
    #     log_to=f"{opt.out_path}/cycles/design_{attempt}")
    successes, instances = search(init_data, log_to=f"{opt.out_path}/cycles/design_{attempt}")
    for i, (score, success, data) in enumerate(instances):
        result: DesignData = data["prediction"]
        result.save_pdb(f"{opt.out_path}/attempts/design_{attempt}_{i}.pdb")
        print(result.to_sequence_string().split(":")[-1], score, success)
    for i, (score, success, data) in enumerate(successes):
        prediction: DesignData = data["prediction"]
        prediction.save_pdb(f"{opt.out_path}/success/design_{attempt}_{i}.pdb")
    success_count += len(successes)
