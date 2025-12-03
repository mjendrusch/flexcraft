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

from salad.aflib.common.protein import from_pdb_string
from salad.modules.utils.geometry import positions_to_ncacocb

import pyrosetta as pr

# salad model step
def model_step(config):
    config = deepcopy(config)
    config.eval = True
    @hk.transform
    def step(data, prev):
        noise = si.StructureDiffusionNoise(config)
        predict = si.StructureDiffusionPredict(config)
        # extract relevant data for potentials
        pos = data["pos"]
        base_pos = pos
        resi = data["residue_index"]
        chain = data["chain_index"]
        center = data["center"]
        t = data["t_pos"][0]
        binder_center = pos[config.target_size:, 1].mean(axis=0)
        center = jnp.where(t > 2.0, center, binder_center)
        pos = jnp.where((is_target)[:, None, None] > 0, pos, pos - binder_center[None, None, :] + center[None, None, :])
        # compute compact step to ensure monomer globularity
        ca = pos[:, 1]
        mean_pos = index_mean(ca, chain, jnp.ones_like(chain, dtype=jnp.bool_)[:, None])
        compact_step = config.compact_lr * (mean_pos[:, None] - pos)
        # move clashing residues further away from each other
        clash_step = config.clash_lr * si.contacts.clash_step(pos, resi, chain, threshold=8.0)
        # contact step
        def contact_update(x, d0=4.0, r0=8.0):
            contact_mask = is_target[:, None] * (~is_target)[None, :]
            def contact_loss(ca):
                eps = 1e-6
                ca = Vec3Array.from_array(ca)
                rij = (ca[:, None] - ca[None, :]).norm()
                sij = (1 - ((rij - d0) / r0) ** 6 + eps) / (1 - ((rij - d0) / r0) ** 12 + eps)
                return (sij * contact_mask).mean()
            return jax.grad(contact_loss, argnums=(0,))(x)[0][:, None]
        contact_step = config.contact_lr * contact_update(ca)
        # apply compact and clash steps scaled by noise standard deviation
        pos = pos + t * (compact_step + clash_step + contact_step)
        #pos = jnp.where((chain == chain.max())[:, None, None], pos + t * (compact_step + clash_step), pos)
        data["pos"] = pos
        # apply noise
        data.update(noise(data))
        # FIXME: center the entire thing
        # data["pos_noised"] -= data["pos_noised"][:, 1].mean(axis=0)
        out, prev = predict(data, prev)
        N = out["pos"].shape[0]
        align_index = jnp.zeros((N,), dtype=jnp.int32)
        align_mask = jnp.array(is_target, dtype=jnp.float32)
        align_weight = None
        out["pos"] = index_align(
            out["pos"], base_pos,
            index=align_index,
            mask=align_mask,
            weight=align_weight)
        out["pos"] = jnp.where(
            t > config.relax_cutoff,
            out["pos"].at[:config.target_size, :5].set(base_pos[:config.target_size, :5]), out["pos"])
        return out, prev
    return step.apply

def parse_target(c, path, num_aa):
    with open(path, "rt") as f:
        structure = from_pdb_string(f.read())
    resi = structure.residue_index
    chain = structure.chain_index
    is_target = np.zeros(resi.shape[0] + num_aa, dtype=np.bool_)
    is_target[:resi.shape[0]] = True
    atom14 = atom37_to_atom14(
        structure.aatype,
        Vec3Array.from_array(structure.atom_positions),
        structure.atom_mask)[0].to_array()
    aatype_provided = structure.aatype
    aa_condition = np.concatenate((aatype_provided, np.array(num_aa * [20], dtype=np.int32)), axis=0)
    target_ncacocb = positions_to_ncacocb(atom14)
    # center
    target_center = target_ncacocb[:, 1].mean(axis=0)
    target_ncacocb -= target_center
    target = np.concatenate((
        target_ncacocb,
        target_ncacocb[:, 1:2] + np.zeros((target_ncacocb.shape[0], c.augment_size, 3), dtype=jnp.float32)),
        axis=1)
    binder = np.zeros((num_aa, 5 + c.augment_size, 3), dtype=np.float32)
    init_pos = np.concatenate((target, binder), axis=0)
    resi = np.concatenate((resi, np.arange(num_aa, dtype=np.int32)), axis=0)
    chain = np.concatenate((chain, np.array(num_aa * [chain.max() + 1], dtype=np.int32)), axis=0)
    num_aa = init_pos.shape[0]
    is_cyclic = False
    cyclic_mask = np.zeros_like(chain, dtype=np.bool_)
    return num_aa, init_pos, target_center, aa_condition, resi, chain, is_target, is_cyclic, cyclic_mask

opt = parse_options(
    "Use salad to generate large protein complexes.",
    salad_params="../../../../params/flexcraft_params/salad_params/default_ve_scaled-200k.jax", 
    pmpnn_params="../../../../params/flexcraft_params/pmpnn_params/v_48_030.pkl",
    af2_params="../../../../params/flexcraft_params/af2_params",
    alphaball_path="../../../../params/flexcraft_params/DAlphaBall.gcc", # get this file from BindCraft GitHub and make executable before running
    out_path="/path-to/output/",
    target="target.pdb",
    scaffold="none", # TODO: make this work for Ab design
    scaffold_relax_cutoff=-1.0,
    hotspots="none",
    coldspots="none",
    num_aa=50,
    set_rosetta_intf="A_B", # which chains to set the interface between for scoring the interface with Rosetta # A_B for monomeric target and a single binder, may be expanded in the future
    bindcraft_success_filter="default", # default is just bindcraft filter, can be expanded by adding filters to bindcraft_filter.py function
    num_designs=48,
    num_sequences=10,
    num_success=1,
    radius=12.0,
    coldspot_radius=15.0,
    clash_radius=10.0,
    clash_lr=5e-3,
    compact_lr=1e-4,
    contact_lr=1e-2,
    dry_run="False",
    relax_cutoff=3.0, # t threshold to allow target protein to move in final steps of denoising for interface refinement
    prev_threshold=0.9,
    use_cycler="False",
    fix_template="False",
    af2_cycler_repeats=10, # how many iterations the af2_cycler should do
    visualize_centers="False",
    ipae_shortcut_threshold=0.35, # which ipae threshold to use for Rosetta relaxation
    allow_target_mutations="0", # number of target interface residues allowed to be redesigned by ProteinMPNN. Set to 0 to leave target sequence as is. Can take a range for random sampling, e.g. "0-3"
    redesign_radius=10, # target residue distance for redesign in Angstrom (if target mutations are allowed). 
    save_af2cycler_pdb="True", # whether to save all af2cycler pdb files
    save_fail_pdb="True", # whether to save pdb files of failed designs
    seed=37,
)


# set up output directories
os.makedirs(f"{opt.out_path}/attempts/", exist_ok=True)
os.makedirs(f"{opt.out_path}/success/", exist_ok=True)
os.makedirs(f"{opt.out_path}/fail/", exist_ok=True)
os.makedirs(f"{opt.out_path}/relaxed/", exist_ok=True)
os.makedirs(f"{opt.out_path}/cycles/", exist_ok=True)

# set up RNG
key = Keygen(opt.seed)

# set up model configuration
config, salad_params = si.make_salad_model(
    "default_ve_scaled", opt.salad_params)
num_aa, init_pos, target_center, aa_condition, resi, chain, is_target, is_cyclic, cyclic_mask = parse_target(
        config, opt.target, opt.num_aa)
if opt.hotspots != "none":
    hotspots = np.array([
        ["ABCDEFGHIJKLMNOPQRSTUVW".index(h[0]), int(h[1:])]
        for h in opt.hotspots.strip().split(",")], dtype=np.int32)
else:
    hotspots = np.array([(c, r) for r, c in zip(resi, chain)])
use_coldspots = False
if opt.coldspots != "none":
    use_coldspots = True
    coldspots = np.array([
        ["ABCDEFGHIJKLMNOPQRSTUVW".index(h[0]), int(h[1:])]
        for h in opt.coldspots.strip().split(",")], dtype=np.int32)
    coldspot_mask = ((coldspots[:, 0] == chain[:, None]) * (coldspots[:, 1] == resi[:, None])).any(axis=1)
hotspot_mask = ((hotspots[:, 0] == chain[:, None]) * (hotspots[:, 1] == resi[:, None])).any(axis=1)
hotpos = init_pos[:, 1][hotspot_mask]
base_chain = chain
target_size = int(is_target.sum())
target_ca = init_pos[is_target][:, 1]
sphere = np.random.randn(hotpos.shape[0], 1000, 3)
sphere /= np.linalg.norm(sphere, axis=-1, keepdims=True)
sphere *= opt.radius
sphere += hotpos[:, None]
sphere = sphere.reshape(-1, 3)
good = (np.linalg.norm(sphere[:, None, :] - target_ca[None, :, :], axis=-1) > opt.clash_radius).all(axis=1)
if use_coldspots:
    coldspot_ca = init_pos[coldspot_mask][:, 1]
    good = good * (np.linalg.norm(sphere[:, None, :] - coldspot_ca[None, :, :], axis=-1) > opt.coldspot_radius).all(axis=1)
pos_centers = sphere[good]
# optionally write a PDB file with all valid center positions
if opt.visualize_centers == "True":
    target = gemmi.read_structure(opt.target)
    model = target[0]
    chain = gemmi.Chain("B")
    for c_pos in pos_centers:
        c_pos = c_pos + target_center
        res = gemmi.Residue()
        atom = gemmi.Atom()
        atom.name = "O"
        atom.pos = gemmi.Position(*c_pos)
        res.name = "HOH"
        res.add_atom(atom)
        chain.add_residue(res)
    model.add_chain(chain)
    target.write_pdb(f"{opt.out_path}/target_centers.pdb")
    exit(0)

config.clash_lr = opt.clash_lr
config.compact_lr = opt.compact_lr
config.contact_lr = opt.contact_lr
config.radius = opt.radius
config.relax_cutoff = opt.relax_cutoff
config.target_size = target_size
monomer_num_aa = opt.num_aa
init_data, init_prev = si.data.from_config(
    config, num_aa=num_aa, chain_index=chain,
    residue_index=resi, cyclic_mask=cyclic_mask,
    init_pos=init_pos)

# make sampler
salad_sampler = si.Sampler(model_step(config),
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
                   pmpnn, confidence=None, fix_template=opt.fix_template == "True")
filter = BindCraftProperties(opt.out_path, key, opt.af2_params, set_int=opt.set_rosetta_intf, filter=opt.bindcraft_success_filter, ipae_shortcut_threshold=opt.ipae_shortcut_threshold)


# set up pyrosetta
pr.init(f'-ignore_unrecognized_res -ignore_zero_occupancy -mute all -holes:dalphaball {opt.alphaball_path} -corrections::beta_nov16 true -relax:default_repeats 1')

# set up output files
score_keys = (
    "attempt", "seq_id", "target_seq","binder_seq","n_target_mutations","success", "failure_reason"
)
score_keys = list(score_keys) + [
    f"{i}_{name}"
    for i in ("Average", "1", "2")
    for name in [
        "i_pAE",
        "pLDDT",
        "pTM",
        "i_pTM",
        "Surface_Hydrophobicity",
        "ShapeComplementarity",
        "ShapeComplementarity",
        "dSASA",
        "n_InterfaceResidues", 
        "n_InterfaceHbonds", 
        "n_InterfaceUnsatHbonds", 
        "RMSD",
    ]
]
success = ScoreCSV(
    f"{opt.out_path}/success.csv", score_keys, default="none")
all_designs = ScoreCSV(
    f"{opt.out_path}/all.csv", score_keys, default="none")

# sample structures
success_count = 0
_attempt = 0
while success_count < opt.num_designs:
    attempt = _attempt
    _attempt += 1
    attempt_success_count = 0
    pos_center = pos_centers[np.random.randint(0, pos_centers.shape[0])]
    init_data["center"] = pos_center
    init_data["pos"] = jnp.where(is_target[:, None, None], init_data["pos"], pos_center)
    init_data["aa_condition"] = aa_condition # FIXME
    design = salad_sampler(salad_params, key(), init_data, init_prev)
    if isinstance(design, list):
        designs = design
        for idd, design in enumerate(designs):
            design = data_from_protein(
                si.data.to_protein(design))
            design.save_pdb(f"{opt.out_path}/attempts/design_{attempt}_{idd}.pdb")
    else:
        design = data_from_protein(
            si.data.to_protein(design))
        design.save_pdb(f"{opt.out_path}/attempts/design_{attempt}.pdb")
    # dry run to check if designed structures are reasonable
    if opt.dry_run == "True":
        continue
    design_info = dict(attempt=attempt)
    design_info["n_target_mutations"] = 0 
    init_info = design_info
    # optionally apply af2cycler
    if opt.use_cycler == "True":
        cycled = design
        for idc in range(opt.af2_cycler_repeats):
            cycled, predicted = cycler(af2_params, key, cycled, cycle_mask=~is_target)
            if opt.save_af2cycler_pdb == "True":
                predicted.save_pdb(f"{opt.out_path}/cycles/design_{attempt}_{idc}.pdb")
        design = cycled

    # identify contact residues
    ca = design["atom_positions"][:, 1]
    dist_matrix = np.linalg.norm(ca[is_target, None] - ca[None, ~is_target], axis=-1)
    
    # Create a mask for target residues at the interface (e.g., within 10A of any binder residue)
    is_interface_target = (dist_matrix < opt.redesign_radius).any(axis=1) 
    # Create the base input for MPNN: binders are masked (20), target is fixed.
    target_aa = jnp.where(is_target, aas.translate(aa_condition, aas.AF2_CODE, aas.PMPNN_CODE), 20)

    # determine the desired number of target mutations for this specific design: 
    target_mut = opt.allow_target_mutations.split("-")
    if len(target_mut) == 1:
        desired_mutations = int(target_mut)
    elif len(target_mut) == 2: 
        lower_bound = int(target_mut[0])
        upper_bound = int(target_mut[1])
        desired_mutations = np.random.randint(lower_bound, upper_bound + 1)
    else:
        raise ValueError("Incorrect format for allowed_target_mut, this must be an int passed as a string (e.g. '0') or a range separated by '-', e.g. '0-3'.")
    
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
    logit_center = pmpnn(key(), design.update(aa=target_aa))["logits"][target_size:].mean(axis=0)
    num_sequences = opt.num_sequences
    for idx in range(num_sequences):
        if attempt_success_count >= opt.num_success:
            break
        # ensure that no overwriting occurs
        design_info = {k: v for k, v in init_info.items()}
        temperature = random.choice((0.01, 0.1, 0.2))#, 0.3, 0.5))
        do_center_logits = True
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
        af2_aa_order = "ARNDCQEGHILKMFPSTWYV" # standard af2 code
        seq_indices = np.array(pmpnn_result["aa"]).flatten()
        target_all_indices = seq_indices[is_target] # extract target sequence
        target_seq = "".join([af2_aa_order[i] if i < 20 else "X" for i in target_all_indices])

        binder_all_indices = seq_indices[~is_target] # extract binder sequences
        binder_seq = "".join([af2_aa_order[i] if i < 20 else "X" for i in binder_all_indices])

        design_info.update(target_seq=target_seq, binder_seq=binder_seq)

        # run bindcraft filter
        af2_result, properties = filter(f"design_{attempt}_{idx}", pmpnn_result, is_target=is_target)
        design_info.update(properties)
        design_info["success"] = int(design_info["success"])

        if not properties["success"]:
            design_info["failure_reason"] = "designability"
            all_designs.write_line(design_info)
            if opt.save_fail_pdb == "True": #made optional to save disk space
                af2_result.save_pdb(f"{opt.out_path}/fail/design_{attempt}_{idx}.pdb")
            continue
        
        design_info.update(success=1, failure_reason="none")
        all_designs.write_line(design_info)
        success.write_line(design_info)
        shutil.copyfile(f"{opt.out_path}/relaxed/design_{attempt}_{idx}_model_1.pdb",
                        f"{opt.out_path}/success/design_{attempt}_{idx}.pdb")
        success_count += 1
        attempt_success_count += 1
