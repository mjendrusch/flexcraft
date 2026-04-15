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

# salad model step for binder design
def binder_step(config):
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
        is_target = data["is_target"]
        t = data["t_pos"][0]
        # FIXME: non-constant version
        # binder_center = index_mean(pos[:, 1], is_target, is_target)[0]
        binder_center = pos[config.target_size:, 1].mean(axis=0)
        center = jnp.where(t > config.fix_center_threshold, center, binder_center)
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
        # center positions prior to noising
        pos = pos - index_mean(pos[:, 1], data["batch_index"], data["mask"][:, None])[:, None]
        data["pos"] = pos
        # apply noise
        data.update(noise(data))
        out, prev = predict(data, prev)
        N = out["pos"].shape[0]
        align_index = jnp.zeros((N,), dtype=jnp.int32)
        # NOTE: do not code while sleep deprived.
        # you will write bugs.
        #
        # align output to target, using only target
        # amino acids for alignment.
        align_mask = data["mask"]
        align_weight = jnp.array(is_target, dtype=jnp.float32)
        out["pos"] = index_align(
            out["pos"], base_pos,
            index=align_index,
            mask=align_mask,
            weight=align_weight)
        # FIXME: non-constant version
        # out["pos"] = jnp.where(
        #     (t > config.relax_cutoff) * is_target,
        #     out["pos"].at[:, :5].set(base_pos[:, :5]), out["pos"]
        # )
        out["pos"] = jnp.where(
            t > config.relax_cutoff,
            out["pos"].at[:config.target_size, :5].set(base_pos[:config.target_size, :5]), out["pos"])
        return out, prev
    return step.apply

def parse_target(c, path, num_aa, target_chains=None):
    with open(path, "rt") as f:
        structure = from_pdb_string(f.read(), convert_chains=False)
    resi = structure.residue_index
    chain_names = structure.chain_index
    aatype = structure.aatype
    atom_positions = structure.atom_positions
    atom_mask = structure.atom_mask
    if target_chains is not None:
        selected_chains = np.array([c for c in target_chains])
        selection = (chain_names[:, None] == selected_chains).any(axis=1)
        resi = resi[selection]
        chain_names = chain_names[selection]
        aatype = aatype[selection]
        atom_positions = atom_positions[selection]
        atom_mask = atom_mask[selection]
    unique_chains = list(np.unique(chain_names))
    chain = np.array([unique_chains.index(c) for c in chain_names], np.int32)
    target_size = resi.shape[0]
    is_target = np.zeros(resi.shape[0] + num_aa, dtype=np.bool_)
    is_target[:resi.shape[0]] = True
    atom14 = atom37_to_atom14(
        aatype,
        Vec3Array.from_array(atom_positions),
        atom_mask)[0].to_array()
    aatype_provided = aatype
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
    c.target_size = target_size

    init_data, init_prev = si.data.from_config(
        c, num_aa=num_aa, chain_index=chain,
        residue_index=resi, cyclic_mask=cyclic_mask,
        init_pos=init_pos)
    init_data["is_target"] = is_target
    init_data["aa_condition"] = aa_condition
    init_data["plddt"] = structure.b_factors
    return init_data, init_prev, target_center, is_target, target_size

def parse_hotspots(init_data, hotspot_string="none", coldspot_string="none"):
    is_target = init_data["is_target"]
    resi = init_data["residue_index"][is_target]
    chain = init_data["chain_index"][is_target]
    init_pos = init_data["pos"][is_target]
    if hotspot_string not in ("none", "auto"):
        hotspots = np.array([
            ["ABCDEFGHIJKLMNOPQRSTUVW".index(h[0]), int(h[1:])]
            for h in hotspot_string.strip().split(",")], dtype=np.int32)
    else:
        hotspots = np.array([(c, r) for r, c in zip(resi, chain)])
    use_coldspots = False
    coldspot_mask = None
    coldpos = None
    if coldspot_string not in ("none", "plddt"):
        use_coldspots = True
        coldspots = np.array([
            ["ABCDEFGHIJKLMNOPQRSTUVW".index(h[0]), int(h[1:])]
            for h in coldspot_string.strip().split(",")], dtype=np.int32)
        coldspot_mask = ((coldspots[:, 0] == chain[:, None]) * (coldspots[:, 1] == resi[:, None])).any(axis=1)
        coldpos = init_pos[coldspot_mask][:, 1]
    elif coldspot_string == "plddt":
        coldspot_mask = auto_coldspots(init_data["plddt"][:, 1])
        coldpos = init_pos[coldspot_mask][:, 1]
    hotspot_mask = ((hotspots[:, 0] == chain[:, None]) * (hotspots[:, 1] == resi[:, None])).any(axis=1)
    if hotspot_string == "auto":
        hotspot_mask = auto_hotspots(init_data["aa_condition"][is_target])
    hotpos = init_pos[:, 1][hotspot_mask]
    return hotpos, coldpos, hotspot_mask, coldspot_mask

hydrophobic_aas = np.array([aas.AF2_CODE.index(c) for c in "LIFYWH"])
def auto_hotspots(target_aa):
    is_hydrophobic = (target_aa[:, None] == hydrophobic_aas).any(axis=-1)
    hotspot_mask = is_hydrophobic
    if not hotspot_mask.any():
        hotspot_mask = np.ones_like(target_aa, dtype=np.bool_)
    return hotspot_mask

def auto_coldspots(plddt):
    coldspot_mask = plddt < 0.5
    return coldspot_mask

def sample_centers_basic(target_ca, hotpos, radius=10.0,
                         clash_radius=None, coldspot_radius=15.0,
                         coldpos=None, num_samples=1000):
    if clash_radius is None:
        clash_radius = radius
    sphere = np.random.randn(hotpos.shape[0], num_samples, 3)
    sphere /= np.linalg.norm(sphere, axis=-1, keepdims=True)
    sphere *= radius
    sphere += hotpos[:, None]
    sphere = sphere.reshape(-1, 3)
    good = (np.linalg.norm(sphere[:, None, :] - target_ca[None, :, :], axis=-1) > clash_radius).all(axis=1)
    if coldpos is not None:
        #coldpos = init_pos[coldspot_mask][:, 1]
        good = good * (np.linalg.norm(sphere[:, None, :] - coldpos[None, :, :], axis=-1) > coldspot_radius).all(axis=1)
    pos_centers = sphere[good]
    return pos_centers

def sample_centers_spaced(target_ca, hotpos, radius_min=10.0, radius_max=None,
                          centermost_k=None,
                          num_samples=100, num_iter=20, min_distance=2.0, **kwargs):
    points = None
    for i in range(num_iter):
        if radius_max is None:
            radius_max = radius_min
        radius = np.random.rand() * (radius_max - radius_min) + radius_min
        add_points = sample_centers_basic(
            target_ca, hotpos, radius=radius, num_samples=num_samples, **kwargs)
        if add_points.shape[0] == 0:
            continue
        if points is None:
            points = add_points
        else:
            points = np.concatenate((points, add_points), axis=0)
        if len(points) == 1:
            continue
        point_dist = np.linalg.norm(points[:, None] - points[None, :], axis=-1) > min_distance
        point_indices = [0]
        while point_dist[point_indices[-1]].any():
            next_point = np.argmax(point_dist[point_indices[-1]], axis=0)
            point_dist *= point_dist[point_indices[-1]][None, :]
            point_indices.append(next_point)
        points = points[np.array(point_indices)]
    if centermost_k is not None and points is not None:
        distance_to_hotspot = np.linalg.norm(points[:, None] - hotpos[None, :], axis=-1)
        hotspot_assignment = np.argmin(distance_to_hotspot, axis=1)
        new_points = []
        for i in np.unique(hotspot_assignment):
            hotspot_points = points[hotspot_assignment == i]
            distance_to_assigned = np.linalg.norm(hotspot_points - hotpos[i], axis=-1)
            selected = np.argsort(distance_to_assigned, axis=0)[:centermost_k]
            hotspot_points = hotspot_points[selected]
            new_points.append(hotspot_points)
        points = np.concatenate(new_points, axis=0)
    return points

# def most_hydrophobic(target_ca, centers, target_aa, radius=15.0):
#     is_hydrophobic = (target_aa[:, None] == hydrophobic_aas).any(axis=-1)
#     dist = np.linalg.norm(centers[:, None] - target_ca[None, :], axis=-1)
#     # TODO

def save_centers(target_path, out_path, target_center, pos_centers):
    target = gemmi.read_structure(target_path)
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
    target.write_pdb(out_path)

def config_from_opt(config, opt):
    config.clash_lr = opt.clash_lr
    config.compact_lr = opt.compact_lr
    config.contact_lr = opt.contact_lr
    config.radius = opt.radius
    config.relax_cutoff = opt.relax_cutoff
    config.fix_center_threshold = opt.fix_center_threshold
    return config

def setup_dirs(out_path):
    os.makedirs(f"{out_path}/attempts/", exist_ok=True)
    os.makedirs(f"{out_path}/success/", exist_ok=True)
    os.makedirs(f"{out_path}/fail/", exist_ok=True)
    os.makedirs(f"{out_path}/relaxed/", exist_ok=True)
    os.makedirs(f"{out_path}/cycles/", exist_ok=True)

def setup_files(out_path, prefix_keys=None, suffix_keys=None):
    if prefix_keys is None:
        prefix_keys = []
    if suffix_keys is None:
        suffix_keys = []
    score_keys = prefix_keys + [
        "attempt", "seq_id", "target_seq", "binder_seq",
        "n_target_mutations", "success", "failure_reason"
    ]
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
            "dSASA",
            "n_InterfaceResidues", 
            "n_InterfaceHbonds", 
            "n_InterfaceUnsatHbonds", 
            "RMSD",
        ]
    ]
    score_keys += suffix_keys
    success = ScoreCSV(
        f"{out_path}/success.csv", score_keys, default="none")
    all_designs = ScoreCSV(
        f"{out_path}/all.csv", score_keys, default="none")
    return success, all_designs

def setup_files_scan(out_path):
    score_keys = (
        "target", "attempt", "seq_id", "target_seq", "binder_seq",
        "i_pAE", "plddt", "sc_rmsd", "L", "H", "E", "x", "y", "z",
        "cycle_drift", "center_drift", "num_contacts",
        "success_monomer", "success_binder"
    )
    success = ScoreCSV(
        f"{out_path}/success.csv", score_keys, default="none")
    stats = ScoreCSV(
        f"{out_path}/all.csv", score_keys, default="none")
    return success, stats
