import os
import shutil
import gemmi
from copy import deepcopy
import json
import random

import haiku as hk
import jax
import jax.numpy as jnp

import salad.inference as si
from salad.modules.utils.dssp import assign_dssp
from salad.inference.symmetry import Screw
from salad.modules.utils.geometry import index_mean, index_align
from salad.aflib.model.all_atom_multimer import atom37_to_atom14

from flexcraft.utils import *
from flexcraft.sequence.sample import *
from flexcraft.structure.af import *

from salad.aflib.common.protein import from_pdb_string, Protein
from salad.modules.utils.geometry import positions_to_ncacocb

# define constants
CHAIN_NAMES = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
SYMS = {"Screw": Screw}

def _symmetry_bundle(pos, bundle):
    index = jnp.argsort(bundle, axis=0, stable=True)
    pos = pos[index]
    return pos

def _symmetry_unbundle(pos, bundle):
    index = jnp.argsort(bundle, axis=0, stable=True)
    return jnp.zeros_like(pos).at[index].set(pos)

def model_step(config, task):
    """Generalized model step for motif grafting tasks."""
    config = deepcopy(config)
    config.eval = True
    symmetrizer = None
    if task["sym"] is not None:
        num_repeats = task["sym"]
        symmetrizer = Screw(num_repeats, 360 / num_repeats, symmetry_axis=2, radius_axis=1)
    @hk.transform
    def step(data, prev):
        # set up salad models
        noise = si.StructureDiffusionNoise(config)
        predict = si.StructureDiffusionPredict(config)
        # get diffusion time
        t = data["t_pos"][0]
        # get motif information
        motif, has_motif, group = data["motif"], data["has_motif"], data["motif_group"]
        use_motif = data["use_motif"]
        # compact indices
        compact_indices = dict(chain=data["chain_index"])
        # optionally center non-motifs in each chain
        if task["center_to_chain"]:
            pos = data["pos"]
            ca = data["pos"][:, 1]
            chain_center = index_mean(ca, data["chain_index"], data["mask"][:, None])
            motif_center = index_mean(ca, data["chain_index"], (data["mask"] * has_motif)[:, None], apply_mask=False)
            non_motif_center = index_mean(ca, data["chain_index"], (data["mask"] * (~has_motif))[:, None])
            non_motif_update = (motif_center - non_motif_center)[:, None]
            pos = jnp.where(((~has_motif) * (t > 40.0))[:, None, None], pos + non_motif_update, pos)
            data["pos"] = pos
        # optionally symmetrize
        if "center" in task:
            pos = data["pos"]
            ca = pos[:, 1]
            center = data["center"] * config.center_radius
            center -= (center * data["mask"][:, None]).sum(axis=0) / jnp.maximum(1, data["mask"].sum(axis=0))
            center_group = data["center_group"]
            compact_indices["center"] = center_group
            center_time = data["center_time"]
            group_center = index_mean(ca, center_group, data["mask"][:, None])
            pos = jnp.where(((t > center_time) * (center_group != -1))[:, None, None],
                            pos - group_center[:, None] + center[:, None], pos)
        if symmetrizer is not None:
            data["pos"] = pos
            # manually pin / unpin stuff above
            pos = data["pos"]
            # set up compact index
            repeat_index = jnp.repeat(
                jnp.arange(symmetrizer.count, dtype=np.int32)[:, None],
                pos.shape[0] // symmetrizer.count, axis=1)
            compact_indices["repeat"] = repeat_index.reshape(-1)
            # reorder connected chains for symmetrization
            pos = _symmetry_bundle(pos, data["bundle"])
            representative = symmetrizer.couple_pos(pos, do_radius=False)
            pos = symmetrizer.replicate_pos(representative, do_radius=False)
            # revert order of chains
            pos = _symmetry_unbundle(pos, data["bundle"])
            data["pos"] = pos

        # compute distance map
        cb = Vec3Array.from_array(positions_to_ncacocb(motif)[:, 4])
        has_motif *= use_motif
        resi = data["residue_index"]
        chain = data["chain_index"]
        dmap = (cb[:, None] - cb[None, :]).norm()
        dmap_mask = has_motif[:, None] * has_motif[None, :]
        dmap_mask *= group[:, None] == group[None, :]
        dmap_mask = dmap_mask > 0
        resi_dist = jnp.where(
            chain[:, None] == chain[None, :],
            abs(resi[:, None] - resi[None, :]),
            jnp.inf)
        data["dmap"] = dmap
        data["dmap_mask"] = jnp.where(t < 10.0, dmap_mask, 0.0) # FIXME
        # apply amino acid conditioning
        if task["use_motif_aa"] != "none":
            aatype = data["motif_aa"]
            aa_condition = make_aa_condition(aatype, dmap, dmap_mask, resi_dist, task)
            data["aa_condition"] = aa_condition
        # apply dssp conditioning
        if task["use_motif_dssp"]:
            data["dssp_condition"] = data["motif_dssp"]
        # optionally apply gradients
        if task["potentials"] is not None:
            pos = data["pos"]
            data["compact_index"] = compact_indices[task["compact_by"]]
            pos += t * gradient_package(task["potentials"])(pos, data)
            data["pos"] = pos
        if "grad" in task["mode"]:
            pos = data["pos"]
            aln_motif = index_align(motif, pos, group, has_motif)
            mca = aln_motif[:, 1]
            def restraint_indexed(ca):
                val = ((ca - mca) ** 2).sum(axis=-1)
                return (val * has_motif).sum() / jnp.maximum(1, has_motif.sum())
            pos -= jnp.where(t >= 1.0, t * jax.grad(restraint_indexed, argnums=(0,))(pos[:, 1])[0][:, None], 0.0)
            data["pos"] = pos
        # apply noise
        # FIXME: use index mean here
        data["pos"] = data["pos"] - data["pos"][:, 1].mean(axis=0)
        data.update(noise(data))
        # predict structure
        out, prev = predict(data, prev)
        # optionally graft motif explicitly
        if "align" in task["mode"]:
            # center positions
            out["pos"] = out["pos"] - index_mean(out["pos"][:, 1], data["batch_index"], data["mask"][:, None])[:, None]
            motif_aligned = index_align(motif, out["pos"], group, has_motif)
            out["pos"] = out["pos"].at[:, :4].set(
                jnp.where(has_motif[:, None, None],
                          motif_aligned[:, :4],
                          out["pos"][:, :4]))
        return out, prev
    return step.apply

def gradient_package(c):
    def update(pos, data):
        result = jnp.zeros_like(pos)
        if "clash" in c:
            resi = data["residue_index"]
            chain = data["chain_index"]
            result += c["clash"] * si.contacts.clash_step(pos, resi, chain, threshold=8.0)
        if "compact" in c:
            compact_index = data["chain_index"]
            if "compact_index" in data:
                compact_index = data["compact_index"]
            mask = data["mask"]
            result += c["compact"] * si.contacts.compact_step(pos, compact_index, mask)
        # TODO: do we need further potentials?
        # TODO: per-segment control of potentials?
        return result
    return update

def prepare_segments(data):
    pdbs = dict()
    segments = dict()
    pdb_names = list()
    for pdb_name, pdb_path in data["pdbs"].items():
        with open(pdb_path) as f:
            pdbs[pdb_name] = from_pdb_string(f.read())
        pdb_names.append(pdb_name)
    for name, segment_defn in data["segments"].items():
        if "pdb" in segment_defn:
            segment = extract_motif(pdbs, pdb_names, segment_defn)
        else:
            segment = segment_defn
        segments[name] = segment
    return segments

def extract_motif(pdbs, pdb_names: list, segment_defn):
    pdb: Protein = pdbs[segment_defn["pdb"]]
    positions, mask14 = atom37_to_atom14(
        pdb.aatype, Vec3Array.from_array(pdb.atom_positions), pdb.atom_mask)
    positions = positions.to_array()
    dssp, _, _ = assign_dssp(
        positions, np.zeros((positions.shape[0],), dtype=np.int32),
        pdb.atom_mask.any(axis=1))
    pdb_index = pdb_names.index(segment_defn["pdb"])
    selection = segment_defn["selection"]
    chain_name = selection[0]
    chain = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".index(chain_name)
    residues = selection[1:]
    selector = pdb.chain_index == chain
    if residues:
        residue_list = []
        for chunk in residues.split(","):
            if "-" in chunk:
                start, end = [int(c) for c in chunk.split("-")]
                residue_list += list(range(start, end + 1))
            else:
                residue_list.append(int(chunk))
        residue_list = np.array(residue_list, dtype=np.int32)
        selector = selector * (pdb.residue_index[:, None] == residue_list[None, :]).any(axis=1)
    group = 0
    if "group" in segment_defn:
        group = segment_defn["group"]
    length = selector.astype(np.int32).sum()
    center = np.zeros((length, 3), dtype=np.float32)
    center_group = np.full((length,), -1, dtype=np.int32)
    center_time = np.full((length,), 1_000, dtype=np.float32)
    use_center = np.zeros((length,), dtype=np.bool_)
    if "center" in segment_defn:
        center = segment_defn["center"]
        center_group = -1
        center_time = 1_000
        if "center_group" in segment_defn:
            center_group = segment_defn["center_group"]
        if "center_time" in segment_defn:
            center_time = segment_defn["center_time"]
        center = np.repeat(np.array(
            center, dtype=np.float32)[None], axis=0)
        center_group = np.full((length,), center_group, dtype=np.int32)
        center_time = np.full((length,), center_time, dtype=np.float32)
        use_center = np.ones((length,), dtype=np.bool_)
    # convert motif to atom14
    aatype = pdb.aatype[selector]
    all_atom_pos = positions[selector]
    all_atom_mask = mask14[selector]
    motif = positions[selector]
    motif_dssp = dssp[selector]
    return dict(
        pdb_index = selector.astype(np.int32).sum() * [pdb_index],
        motif_chain_index = pdb.chain_index[selector],
        has_motif = all_atom_mask.any(axis=1),
        motif = motif,
        motif_mask = all_atom_mask,
        motif_aa = aatype,
        motif_dssp = motif_dssp,
        motif_group = np.full_like(pdb.chain_index[selector], group),
        center = center, center_group = center_group, center_time = center_time,
        use_center = use_center,
        bundle=np.full((length,), segment_defn["bundle"], dtype=np.int32)
    )

def sample_data(salad_config, segments, assembly,
                use_motif=True, pos_init=False,
                pseudo_centers=None):
    chunks = []
    motif_centers = []
    for chain in assembly:
        chain_centers = []
        for segment_name in chain:
            segment = segments[segment_name]
            if "max_length" in segment:
                continue
            else:
                chain_centers.append(segment["motif"][:, 1].mean(axis=0))
        motif_centers.append(chain_centers)
    absolute_segment_index = 0
    for idc, chain in enumerate(assembly):
        offset = 1
        chain_center = np.stack(motif_centers[idc], axis=0).mean(axis=0)
        for ids, segment_name in enumerate(chain):
            segment = deepcopy(segments[segment_name])
            if "max_length" in segment:
                length = np.random.randint(segment["min_length"], segment["max_length"] + 1)
                segment = dict(
                    pdb_index = np.full((length,), -1, dtype=np.int32),
                    motif_chain_index = np.full((length,), -1, dtype=np.int32),
                    has_motif = np.zeros((length,), dtype=np.bool_),
                    motif = np.zeros((length, 14, 3), dtype=np.float32),
                    motif_mask = np.zeros((length, 14), dtype=np.bool_),
                    motif_aa = np.full((length,), 20, dtype=np.int32),
                    motif_group = np.full((length,), -1, dtype=np.int32),
                    motif_dssp = np.full((length,), 3, dtype=np.int32),
                    bundle = np.full((length,), segment["bundle"], dtype=np.int32))
                if pseudo_centers is not None:
                    segment["center"] = np.zeros((length, 3), dtype=np.float32) + pseudo_centers["center"][absolute_segment_index]
                    segment["center_group"] = np.full((length,), pseudo_centers["center_group"][absolute_segment_index], dtype=np.int32)
                    segment["center_time"] = np.full((length,), pseudo_centers["center_time"][absolute_segment_index], dtype=np.float32)
                    segment["use_center"] = np.ones((length,), dtype=np.bool_)
                else:
                    segment["center"] = np.zeros((length, 3), dtype=np.float32) + chain_center
                    segment["center_group"] = np.full((length,), -1, dtype=np.int32)
                    segment["center_time"] = np.full((length,), 1_000, dtype=np.float32)
                    segment["use_center"] = np.zeros((length,), dtype=np.bool_)
                pos = np.zeros((length, 5 + salad_config.augment_size, 3), dtype=np.float32)
                pos += chain_center
            else:
                length = segment["motif"].shape[0]
                if pseudo_centers is not None:
                    segment["center"] = np.zeros((length, 3), dtype=np.float32) + pseudo_centers["center"][absolute_segment_index]
                    segment["center_group"] = np.full((length,), pseudo_centers["center_group"][absolute_segment_index], dtype=np.int32)
                    segment["center_time"] = np.full((length,), pseudo_centers["center_time"][absolute_segment_index], dtype=np.float32)
                    segment["use_center"] = np.ones((length,), dtype=np.bool_)
                pos = np.concatenate((
                    segment["motif"][:, :4],
                    np.repeat(segment["motif"][:, 1, None], 1 + salad_config.augment_size, axis=1)),
                      axis=1)
            if not pos_init:
                pos = np.zeros_like(pos)
            # shared chain metadata for salad inputs
            metadata = dict(
                residue_index = offset + np.arange(length, dtype=np.int32),
                chain_index = np.full((length,), idc, dtype=np.int32),
                batch_index = np.zeros((length,), dtype=np.int32),
                pos = pos,
                seq = np.full((length,), 20, dtype=np.int32),
                aa_gt = np.full((length,), 20, dtype=np.int32),
                t_pos = np.ones((length,), dtype=np.float32),
                t_seq = np.ones((length,), dtype=np.float32),
                mask = np.ones((length,), dtype=np.bool_),
                cyclic_mask = np.zeros((length,), dtype=np.bool_),
                use_motif = np.full((length,), use_motif, dtype=np.bool_))
            segment.update(metadata)
            chunks.append(segment)
            offset += length
            absolute_segment_index += 1
    data = {k: np.concatenate([c[k] for c in chunks], axis=0) for k in chunks[0]}
    # center stuff on the motifs FIXME
    data["pos"] -= data["pos"][:, 1].mean(axis=0)
    init_prev = {
        "pos": np.zeros_like(data["pos"]),
        "local": np.zeros((data["pos"].shape[0], salad_config.local_size), dtype=np.float32)
    }
    return data, init_prev

def make_simple_assembly(pdb_path, assembly_string):
    assembly_string = assembly_string.strip()
    assembly_string = assembly_string.replace(" ", "")
    chains = assembly_string.split(":")
    assembly = []
    segments = dict()
    segment_index = 0
    for chain in chains:
        chain_segments = chain.split(",")
        chain_content = []
        for seg in chain_segments:
            seg: str
            segment_name = f"segment_{segment_index}"
            if seg[0] in CHAIN_NAMES:
                pdb_chain = seg[0]
                seg = seg[1:]
                group = 0
                symmetry_bundle = 0
                if "&" in seg:
                    seg, symmetry_bundle = seg.split("&")
                    symmetry_bundle = int(symmetry_bundle)
                if "@" in seg:
                    seg, group = seg.split("@")
                    group = int(group)
                selection = f"{pdb_chain}{seg}"
                segments[segment_name] = dict(
                    pdb="motif", selection=selection, group=group, bundle=symmetry_bundle)
            else:
                symmetry_bundle = 0
                if "&" in seg:
                    seg, symmetry_bundle = seg.split("&")
                    symmetry_bundle = int(symmetry_bundle)
                if "-" in seg:
                    min_length, max_length = list(map(int, seg.split("-")))
                else:
                    min_length = max_length = int(seg)
                segments[segment_name] = dict(
                    min_length=min_length, max_length=max_length, bundle=symmetry_bundle)
            chain_content.append(segment_name)
            segment_index += 1
        assembly.append(chain_content)
    data = dict(pdbs=dict(motif=pdb_path), segments=segments, assembly=assembly)
    data["segments"] = prepare_segments(data)
    return data

def make_ghosts(pdb_path, ghost_string):
    data = make_simple_assembly(pdb_path, ghost_string)
    ghost = []
    ghost_group = []
    for name, segment in data["segments"].items():
        ghost.append(segment["motif"])
        ghost_group.append(segment["motif_group"])
    result = dict()
    result["ghost"] = np.concatenate(ghost, axis=0)
    result["ghost_group"] = np.concatenate(ghost_group, axis=0)
    return result

def make_aa_condition(aatype, dmap, dmap_mask, resi_dist, task):
    if task["use_motif_aa"] == "all":
        aa_condition = aatype
    elif task["use_motif_aa"] == "within_motif":
        contact_long_range = jnp.where(dmap_mask * (resi_dist > 4), dmap, jnp.inf) < 8.0
        num_contacts = contact_long_range.astype(jnp.int32).sum(axis=1)
        buried = num_contacts >= task["buried_contacts"]
        aa_condition = jnp.where(buried, aatype, 20)
    else:
        aa_condition = jnp.full_like(aatype, 20)
    return aa_condition

def get_assembly_lengths(segments, assembly):
    min_length = 0
    max_length = 0
    for chain in assembly:
        for segment_name in chain:
            segment = segments[segment_name]
            if "max_length" in segment:
                segment_max_length = segment["max_length"]
                segment_min_length = segment["min_length"]
            else:
                segment_max_length = segment["motif"].shape[0]
                segment_min_length = segment_max_length
            max_length += segment_max_length
            min_length += segment_min_length
    return min_length, max_length

def get_motif_paths(path: str, assembly = None):
    if path.endswith(".pdb"):
        return [(path, assembly)]
    if path.endswith(".jsonl"):
        return _motif_paths_from_jsonl(path)
    return _motif_paths_from_directory(path, assembly)

def _motif_paths_from_directory(path: str, assembly: str):
    result = []
    for name in os.listdir(path):
        if name.endswith(".pdb"):
            full_path = f"{path}/{name}"
            result.append((full_path, assembly))
    return result

def _motif_paths_from_jsonl(path):
    result = []
    with open(path) as f:
        for line in f:
            data = json.loads(line.strip())
            result.append((data["path"], data["assembly"]))
    return result

def setup_directories(out_path):
    os.makedirs(f"{out_path}/attempts/", exist_ok=True)
    os.makedirs(f"{out_path}/success/", exist_ok=True)
    os.makedirs(f"{out_path}/fail/", exist_ok=True)
    os.makedirs(f"{out_path}/relaxed/", exist_ok=True)
