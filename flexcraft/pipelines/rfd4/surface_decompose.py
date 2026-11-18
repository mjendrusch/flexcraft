"""Surface patch decomposition and RFD4/AF3Like motif scaffolding pipeline.

Decomposes target protein structures into solvent-exposed, non-interface surface
sidechain patches, designs monomeric scaffolds around each patch using RFD4
unindexed sidechain coordinate conditioning (``C_CRD``), pre-filters by
Hungarian + C2-symmetric Kabsch sidechain matching, and validates folding
self-consistency with ``AF3Like`` (defaulting to ``protenix-v2``).

Example:
    python -m flexcraft.pipelines.rfd4.surface_decompose \\
        --in_path target.pdb --out_path runs/surface_decompose \\
        --num_aa 72 --num_designs 20
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import math
import pathlib
import time
from typing import Any, Sequence

import gemmi
import jax
import jax.numpy as jnp
import numpy as np
from rfd4fx.encoding import CANONICAL_AA3_TO_1, encoding_tables
from rfd4fx.flow_matching import RFD4Sampler, THINKING_HIGH
from rfd4fx.model import RFD4
from rfd4fx.rfd4_input import RFD4Input, is_surface
from scipy.optimize import linear_sum_assignment

from flexcraft.files.csv import ScoreCSV
from flexcraft.pipelines.rfd4.common import (
    BACKBONE_ATOMS,
    C2_SYMMETRY_SWAPS,
    JIT_VMAP_BACKBONE_RMSD,
    WATERS,
    add_forbidden_args,
    add_gen_only_args,
    build_af3_predictor,
    coupled_kabsch_sidechain_rmsd,
    discover_pdb_paths,
    extract_atom24_samples,
    extract_residue_atom_dicts,
    normalize_blank_chain_ids,
    parse_forbidden_aas,
    prepare_af3_protein_spec,
    prepare_output_dir,
    save_aligned_prediction_pdb,
    write_gemmi_pdb,
)
from flexcraft.structure.common._utils import align_up

logger = logging.getLogger("surface_decompose")

SCORE_COLUMNS: tuple[str, ...] = (
    "target",
    "target_chain",
    "attempt",
    "center_residue",
    "covered_residues",
    "sequence",
    "rfd4_motif_rmsd",
    "sc_rmsd",
    "motif_rmsd",
    "mean_plddt",
    "motif_plddt",
    "mean_pae",
    "ptm",
    "failure",
)


@dataclasses.dataclass
class SanitizedResidue:
    """Single sanitized canonical polymer residue from target preprocessing."""

    res_token: str
    chain_id: str
    seqid_num: int
    icode: str
    res_name: str
    aa_idx: int
    polymer_seq_idx: int
    atom_coords: dict[str, np.ndarray]
    pos37_sc: np.ndarray
    mask37_sc: np.ndarray
    sc_atom_names: tuple[str, ...]
    has_cb: bool
    is_interface: bool = False
    is_surface: bool = False


@dataclasses.dataclass
class PatchSpec:
    """Host descriptor for one valid seed residue's exposed sidechain patch."""

    seed_token: str
    member_tokens: tuple[str, ...]
    valid_seed_tokens: tuple[str, ...]
    member_residues: tuple[SanitizedResidue, ...]
    pos_raw: np.ndarray
    pos_centered: np.ndarray
    mask37: np.ndarray
    aa_idx: np.ndarray
    polymer_seq_idx: np.ndarray
    c_crd_offset: np.ndarray


@dataclasses.dataclass
class ChainContext:
    """Per-chain surface decomposition state and diagnostic filter counts."""

    target_name: str
    chain_id: str
    residues: list[SanitizedResidue]
    valid_seeds: list[str]
    patch_members: dict[str, tuple[SanitizedResidue, ...]] = dataclasses.field(
        default_factory=dict
    )
    patches: dict[str, PatchSpec] = dataclasses.field(default_factory=dict)
    n_polymer: int = 0
    n_non_gly_cb: int = 0
    n_interface_excluded: int = 0
    n_surface: int = 0
    n_valid_seeds: int = 0


@dataclasses.dataclass
class TargetPlan:
    """Sanitized target structure and per-chain patch decomposition contexts."""

    target_name: str
    pdb_path: pathlib.Path
    sanitized_structure: gemmi.Structure
    chain_residues: dict[str, list[SanitizedResidue]] = dataclasses.field(
        default_factory=dict
    )
    chains: list[ChainContext] = dataclasses.field(default_factory=list)


def build_parser() -> argparse.ArgumentParser:
    """Construct the CLI argument parser for ``surface_decompose``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--in_path",
        type=pathlib.Path,
        required=True,
        help="input .pdb file or directory of .pdb files",
    )
    parser.add_argument(
        "--out_path",
        type=pathlib.Path,
        required=True,
        help="output root directory for attempts/, success/, seeds/, and scores.csv",
    )
    parser.add_argument(
        "--num_aa",
        type=int,
        required=True,
        help="total length of each designed RFD4 scaffold chain (>= 1)",
    )
    parser.add_argument(
        "--num_designs",
        type=int,
        default=100,
        help="maximum attempts per isolated protein chain (>= 0)",
    )
    parser.add_argument(
        "--max_seed_attempts",
        type=int,
        default=10,
        help="maximum draws as center_residue before retiring an uncovered seed (>= 1)",
    )
    parser.add_argument(
        "--cutoff",
        type=float,
        default=5.0,
        help="spherical sidechain-to-seed heavy-atom cutoff in Angstroms (> 0.0)",
    )
    parser.add_argument(
        "--min_num_patch",
        type=int,
        default=5,
        help="minimum surface residues required in a patch for seed eligibility (>= 3)",
    )
    parser.add_argument(
        "--interface_cutoff",
        type=float,
        default=8.0,
        help="inter-chain CB-CB cutoff in Angstroms excluding interface residues from seeds (>= 0.0)",
    )
    parser.add_argument(
        "--f_rfd4_rmsd",
        type=float,
        default=1.0,
        help="pre-fold RFD4 motif sidechain Kabsch RMSD upper bound (<) in Angstroms",
    )
    parser.add_argument(
        "--f_sc_rmsd",
        type=float,
        default=2.0,
        help="post-fold 4-atom backbone scRMSD upper bound (<) in Angstroms",
    )
    parser.add_argument(
        "--f_motif_rmsd",
        type=float,
        default=1.0,
        help="post-fold predictor motif sidechain Kabsch RMSD upper bound (<) in Angstroms",
    )
    parser.add_argument(
        "--f_plddt",
        type=float,
        default=0.85,
        help="post-fold sample-mean normalized pLDDT ([0, 1]) lower bound (>)",
    )
    parser.add_argument(
        "--f_pae",
        type=float,
        default=3.0,
        help="post-fold sample-mean unnormalized PAE upper bound (<) in Angstroms",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="protenix-v2",
        help="AF3Like structure predictor identifier ('protenix-*', 'joltz', 'esmfold2_*', 'af2_*')",
    )
    parser.add_argument(
        "--cache",
        type=pathlib.Path,
        default=None,
        help="optional parameter cache directory passed to AF3Like",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=4,
        help="number of structure predictor diffusion samples per design (>= 1)",
    )
    parser.add_argument(
        "--num_recycle",
        type=int,
        default=4,
        help="number of structure predictor recycling iterations (>= 0)",
    )
    add_forbidden_args(
        parser,
        help_text=(
            "single-letter amino acid codes to reject on non-motif scaffold positions "
            "post-RFD4 (default: 'C'; pass '' or 'none' to allow all)"
        ),
    )
    add_gen_only_args(parser)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="allow writing outputs into an existing non-empty --out_path directory",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse and validate CLI flags for ``surface_decompose``."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.num_aa < 1:
        parser.error(f"--num_aa must be >= 1, got {args.num_aa}")
    if args.num_designs < 0:
        parser.error(f"--num_designs must be >= 0, got {args.num_designs}")
    if args.max_seed_attempts < 1:
        parser.error(f"--max_seed_attempts must be >= 1, got {args.max_seed_attempts}")
    if args.min_num_patch < 3:
        parser.error(f"--min_num_patch must be >= 3, got {args.min_num_patch}")
    if args.num_samples < 1:
        parser.error(f"--num_samples must be >= 1, got {args.num_samples}")
    if args.num_recycle < 0:
        parser.error(f"--num_recycle must be >= 0, got {args.num_recycle}")

    for flag, val in (
        ("--cutoff", args.cutoff),
        ("--f_rfd4_rmsd", args.f_rfd4_rmsd),
        ("--f_sc_rmsd", args.f_sc_rmsd),
        ("--f_motif_rmsd", args.f_motif_rmsd),
        ("--f_pae", args.f_pae),
    ):
        if not (math.isfinite(val) and val > 0.0):
            parser.error(f"{flag} must be finite and > 0.0, got {val}")

    if not (math.isfinite(args.interface_cutoff) and args.interface_cutoff >= 0.0):
        parser.error(f"--interface_cutoff must be finite and >= 0.0, got {args.interface_cutoff}")
    if not (math.isfinite(args.f_plddt) and 0.0 <= args.f_plddt <= 1.0):
        parser.error(f"--f_plddt must be finite and in [0.0, 1.0], got {args.f_plddt}")
    if not args.model or not args.model.strip():
        parser.error("--model must be a non-empty string")

    try:
        parse_forbidden_aas(args.forbidden)
    except ValueError as exc:
        parser.error(str(exc))

    return args


def _format_score(val: float | None, *, precision: int = 4) -> str:
    return "none" if val is None else f"{val:.{precision}f}"


def sanitize_target_structure(pdb_path: pathlib.Path, interface_cutoff: float) -> TargetPlan:
    """Sanitize model 0 of ``pdb_path`` and mark cross-chain CB interface residues."""
    st = gemmi.read_structure(str(pdb_path))
    if len(st) == 0:
        raise ValueError(f"PDB structure has no models: {pdb_path}")

    for model_idx in range(len(st) - 1, 0, -1):
        del st[model_idx]

    st.setup_entities()
    st.assign_subchains(force=False)
    model = st[0]
    normalize_blank_chain_ids(model)

    _, token_to_idx, token_atoms, _ = encoding_tables()
    chain_residues: dict[str, list[SanitizedResidue]] = {}
    empty_chain_indices: list[int] = []

    for chain_idx, chain in enumerate(model):
        poly = chain.get_polymer()
        if poly.check_polymer_type() not in (
            gemmi.PolymerType.PeptideL,
            gemmi.PolymerType.PeptideD,
        ):
            empty_chain_indices.append(chain_idx)
            continue

        for res in poly:
            if res.name.strip() == "MSE":
                res.name = "MET"
                res.het_flag = "A"
                for atom in res:
                    if atom.name.strip() == "SE":
                        atom.name = "SD"
                        atom.element = gemmi.Element("S")

        chain_id = chain.name
        retained_residues: list[SanitizedResidue] = []
        keep_res_indices: set[int] = set()
        polymer_seq_idx = 0
        saw_skipped_residue = False

        for res_idx, res in enumerate(chain):
            res_name = res.name.strip()
            if res_name in WATERS or res.het_flag == "H" or res_name not in CANONICAL_AA3_TO_1:
                if retained_residues:
                    saw_skipped_residue = True
                continue

            seen_atoms: set[str] = set()
            keep_atom_indices: set[int] = set()
            atom_coords: dict[str, np.ndarray] = {}
            for atom_idx, atom in enumerate(res):
                if atom.element.name in {"H", "D"}:
                    continue
                atom_name = atom.name.strip()
                if not atom_name or atom_name in seen_atoms:
                    continue
                coord = np.array([atom.pos.x, atom.pos.y, atom.pos.z], dtype=np.float64)
                if not np.isfinite(coord).all():
                    continue
                seen_atoms.add(atom_name)
                atom.altloc = "\x00"
                keep_atom_indices.add(atom_idx)
                atom_coords[atom_name] = coord

            if not keep_atom_indices:
                if retained_residues:
                    saw_skipped_residue = True
                continue

            for atom_idx in range(len(res) - 1, -1, -1):
                if atom_idx not in keep_atom_indices:
                    del res[atom_idx]

            if retained_residues:
                prev_res = retained_residues[-1]
                has_peptide_bond = (
                    not saw_skipped_residue
                    and "C" in prev_res.atom_coords
                    and "N" in atom_coords
                    and float(np.linalg.norm(prev_res.atom_coords["C"] - atom_coords["N"])) < 2.0
                )
                polymer_seq_idx += 1 if has_peptide_bond else 2
            saw_skipped_residue = False

            slots = token_atoms[res_name]
            pos37_sc = np.full((37, 3), np.nan, dtype=np.float32)
            mask37_sc = np.zeros(37, dtype=bool)
            sc_names: list[str] = []
            for slot_idx, atom_name in enumerate(slots):
                if not atom_name or atom_name in BACKBONE_ATOMS:
                    continue
                if atom_name in atom_coords:
                    pos37_sc[slot_idx] = atom_coords[atom_name].astype(np.float32)
                    mask37_sc[slot_idx] = True
                    sc_names.append(atom_name)

            seqid_num = int(res.seqid.num)
            icode = res.seqid.icode.strip()
            retained_residues.append(
                SanitizedResidue(
                    res_token=f"{chain_id}{seqid_num}{icode}",
                    chain_id=chain_id,
                    seqid_num=seqid_num,
                    icode=icode,
                    res_name=res_name,
                    aa_idx=token_to_idx[res_name],
                    polymer_seq_idx=polymer_seq_idx,
                    atom_coords=atom_coords,
                    pos37_sc=pos37_sc,
                    mask37_sc=mask37_sc,
                    sc_atom_names=tuple(sc_names),
                    has_cb=("CB" in atom_coords),
                )
            )
            keep_res_indices.add(res_idx)

        if not retained_residues:
            empty_chain_indices.append(chain_idx)
            continue

        for res_idx in range(len(chain) - 1, -1, -1):
            if res_idx not in keep_res_indices:
                del chain[res_idx]

        chain_residues.setdefault(chain_id, []).extend(retained_residues)

    for chain_idx in sorted(empty_chain_indices, reverse=True):
        del model[chain_idx]

    chain_ids = list(chain_residues.keys())
    if len(chain_ids) > 1 and interface_cutoff > 0.0:
        cb_by_chain: dict[str, tuple[list[SanitizedResidue], np.ndarray]] = {}
        for cid in chain_ids:
            cb_res = [r for r in chain_residues[cid] if r.res_name != "GLY" and r.has_cb]
            cb_xyz = (
                np.stack([r.atom_coords["CB"] for r in cb_res], axis=0)
                if cb_res
                else np.zeros((0, 3), dtype=np.float64)
            )
            cb_by_chain[cid] = (cb_res, cb_xyz)

        for cid in chain_ids:
            cb_res, cb_xyz = cb_by_chain[cid]
            if not cb_res:
                continue
            other_arrays = [
                cb_by_chain[ocid][1]
                for ocid in chain_ids
                if ocid != cid and len(cb_by_chain[ocid][1]) > 0
            ]
            if not other_arrays:
                continue
            other_cb = np.concatenate(other_arrays, axis=0)
            dists = np.linalg.norm(cb_xyz[:, None, :] - other_cb[None, :, :], axis=-1)
            for idx, is_iface in enumerate((dists < interface_cutoff).any(axis=1)):
                if is_iface:
                    cb_res[idx].is_interface = True

    return TargetPlan(
        target_name=pdb_path.stem,
        pdb_path=pdb_path,
        sanitized_structure=st,
        chain_residues=chain_residues,
    )


def precompute_chain_patches(
    target_name: str,
    chain_id: str,
    residues: list[SanitizedResidue],
    *,
    cutoff: float,
    min_num_patch: int,
) -> ChainContext:
    """Run isolated-chain Shrake-Rupley SASA test and precompute sidechain patches."""
    occluder_pts = [
        r.atom_coords[an]
        for r in residues
        for an in BACKBONE_ATOMS
        if an in r.atom_coords
    ]
    occluders = (
        np.stack(occluder_pts, axis=0).astype(np.float64)
        if occluder_pts
        else np.zeros((0, 3), dtype=np.float64)
    )

    surface_candidates = [
        r for r in residues if r.res_name != "GLY" and r.has_cb and len(r.sc_atom_names) >= 1
    ]
    if surface_candidates:
        cb_coords = np.stack(
            [r.atom_coords["CB"] for r in surface_candidates], axis=0
        ).astype(np.float64)
        exposed = is_surface(cb_coords, occluders, radius=4.0, n_points=64, threshold=0.2)
        for idx, flag in enumerate(exposed):
            surface_candidates[idx].is_surface = bool(flag)

    exposed_candidates = [r for r in surface_candidates if r.is_surface]
    heavy_blocks = [
        np.stack(list(r.atom_coords.values()), axis=0).astype(np.float64)
        for r in exposed_candidates
    ]
    sc_blocks = [
        np.stack([r.atom_coords[an] for an in r.sc_atom_names], axis=0).astype(np.float64)
        for r in exposed_candidates
    ]

    valid_seeds: list[str] = []
    patch_members: dict[str, tuple[SanitizedResidue, ...]] = {}
    for seed_idx, seed_res in enumerate(exposed_candidates):
        s_heavy = heavy_blocks[seed_idx]
        members = [
            cand_res
            for cand_idx, cand_res in enumerate(exposed_candidates)
            if float(np.linalg.norm(s_heavy[:, None, :] - sc_blocks[cand_idx][None, :, :], axis=-1).min())
            < cutoff
        ]
        if (not seed_res.is_interface) and len(members) >= min_num_patch:
            valid_seeds.append(seed_res.res_token)
            patch_members[seed_res.res_token] = tuple(members)

    return ChainContext(
        target_name=target_name,
        chain_id=chain_id,
        residues=residues,
        valid_seeds=valid_seeds,
        patch_members=patch_members,
        patches={},
        n_polymer=len(residues),
        n_non_gly_cb=len(surface_candidates),
        n_interface_excluded=sum(1 for r in exposed_candidates if r.is_interface),
        n_surface=len(exposed_candidates),
        n_valid_seeds=len(valid_seeds),
    )


def build_patch_specs(plans: list[TargetPlan]) -> int:
    """Populate ``PatchSpec`` descriptors across all valid seeds and return ``M_max``."""
    m_max = 0
    for plan in plans:
        for ctx in plan.chains:
            valid_seed_set = set(ctx.valid_seeds)
            for seed_token in ctx.valid_seeds:
                members = ctx.patch_members[seed_token]
                pos_raw = np.stack([r.pos37_sc for r in members], axis=0).astype(np.float32)
                mask37 = np.stack([r.mask37_sc for r in members], axis=0).astype(bool)
                c_crd_offset = pos_raw[mask37].mean(axis=0).astype(np.float64)
                pos_centered = np.where(
                    mask37[..., None],
                    pos_raw - c_crd_offset.astype(np.float32),
                    np.nan,
                ).astype(np.float32)
                ctx.patches[seed_token] = PatchSpec(
                    seed_token=seed_token,
                    member_tokens=tuple(r.res_token for r in members),
                    valid_seed_tokens=tuple(
                        r.res_token for r in members if r.res_token in valid_seed_set
                    ),
                    member_residues=members,
                    pos_raw=pos_raw,
                    pos_centered=pos_centered,
                    mask37=mask37,
                    aa_idx=np.array([r.aa_idx for r in members], dtype=np.int32),
                    polymer_seq_idx=np.array(
                        [r.polymer_seq_idx for r in members], dtype=np.int32
                    ),
                    c_crd_offset=c_crd_offset,
                )
                m_max = max(m_max, len(members))
    return m_max


def describe_zero_seed_stage(ctx: ChainContext) -> str:
    """Return human-readable explanation of why ``ctx`` yielded zero seeds."""
    if ctx.n_polymer == 0:
        return "sanitization (0 retained canonical residues)"
    if ctx.n_non_gly_cb == 0:
        return "non-GLY CB filter (0 eligible residues)"
    if ctx.n_surface == 0:
        return "Shrake-Rupley surface test (0 exposed residues)"
    if ctx.n_surface - ctx.n_interface_excluded == 0:
        return "interface exclusion (all exposed residues on inter-chain interface)"
    return "min_num_patch filter (no non-interface surface patch met min_num_patch)"


def prepare_target_plans(
    pdb_paths: Sequence[pathlib.Path],
    *,
    interface_cutoff: float,
    cutoff: float,
    min_num_patch: int,
    log: logging.Logger | None = None,
) -> tuple[list[TargetPlan], int]:
    """Sanitize PDBs, compute surface patches/specs, and return ``(plans, M_max)``."""
    active_log = logger if log is None else log
    plans: list[TargetPlan] = []
    for pdb_path in pdb_paths:
        plan = sanitize_target_structure(pdb_path, interface_cutoff)
        for chain_id, residues in plan.chain_residues.items():
            ctx = precompute_chain_patches(
                plan.target_name,
                chain_id,
                residues,
                cutoff=cutoff,
                min_num_patch=min_num_patch,
            )
            active_log.info(
                "Target %s chain %s: n_polymer=%d n_non_gly_cb=%d "
                "n_interface_excluded=%d n_surface=%d n_valid_seeds=%d",
                plan.target_name,
                chain_id,
                ctx.n_polymer,
                ctx.n_non_gly_cb,
                ctx.n_interface_excluded,
                ctx.n_surface,
                ctx.n_valid_seeds,
            )
            if ctx.n_valid_seeds == 0:
                active_log.warning(
                    "Target %s chain %s has 0 valid seeds (emptied at %s); skipping.",
                    plan.target_name,
                    chain_id,
                    describe_zero_seed_stage(ctx),
                )
            plan.chains.append(ctx)
        plans.append(plan)
    return plans, build_patch_specs(plans)


def compile_patch_batch(
    base_rfd4_input: RFD4Input,
    patch: PatchSpec,
    n_pad: int,
):
    """Compile an unindexed ``C_CRD`` motif batch padded to ``n_pad``."""
    return (
        base_rfd4_input.add_unindexed_motif(
            patch.pos_raw,
            patch.aa_idx,
            patch.mask37,
            residue_index=patch.polymer_seq_idx,
            chain_index=np.zeros(len(patch.member_residues), np.float32),
            fixed=True,
            ordered=True,
            target_chain="A",
        )
        .pad_to_nearest(n_pad)
        .to_batch()
    )


def has_forbidden_scaffold_aa(
    sequence: str,
    col_ind: np.ndarray | Sequence[int],
    forbidden_aas: set[str] | frozenset[str],
) -> bool:
    """Return True if any non-motif scaffold position in ``sequence`` contains a forbidden AA."""
    if not forbidden_aas:
        return False
    motif_set = {int(idx) for idx in col_ind}
    return any(aa in forbidden_aas for idx, aa in enumerate(sequence) if idx not in motif_set)


def match_rfd4_patch(
    patch: PatchSpec,
    aa_pred_idx: np.ndarray,
    coors_ang: np.ndarray,
    atom_mask: np.ndarray,
) -> tuple[np.ndarray | None, float | None, np.ndarray | None, np.ndarray | None]:
    """Assign RFD4 scaffold residues 1-to-1 to ``patch`` in ``C_CRD`` frame and fit Kabsch."""
    aa_pred = np.asarray(aa_pred_idx, dtype=np.int32).reshape(-1)
    coords = np.asarray(coors_ang, dtype=np.float64)
    mask = np.asarray(atom_mask, dtype=bool)
    num_aa = aa_pred.shape[0]
    m_count = len(patch.member_residues)

    if num_aa < m_count:
        return None, None, None, None

    req_vals, req_counts = np.unique(patch.aa_idx, return_counts=True)
    for aa_val, req_cnt in zip(req_vals, req_counts):
        if int((aa_pred == aa_val).sum()) < int(req_cnt):
            return None, None, None, None

    _, _, token_atoms, atom_slot_map = encoding_tables()
    cost = np.full((m_count, num_aa), np.inf, dtype=np.float64)

    for m_idx, res in enumerate(patch.member_residues):
        slot_map = atom_slot_map[res.res_name]
        slots_id = np.array([slot_map[an] for an in res.sc_atom_names], dtype=np.int32)
        q_centered = patch.pos_centered[m_idx, slots_id].astype(np.float64)

        swaps = C2_SYMMETRY_SWAPS.get(res.res_name, ())
        if swaps:
            swap_dict = {src: dst for a, b in swaps for src, dst in ((a, b), (b, a))}
            slots_sw = np.array(
                [slot_map[swap_dict.get(an, an)] for an in res.sc_atom_names], dtype=np.int32
            )
        else:
            slots_sw = None

        for j_raw in np.flatnonzero(aa_pred == res.aa_idx):
            j_idx = int(j_raw)
            if not bool(mask[j_idx, slots_id].all()):
                continue
            pts_id = coords[j_idx, slots_id]
            if not (np.isfinite(pts_id).all() and (np.abs(pts_id).sum(axis=-1) > 0.0).all()):
                continue
            err = float(np.sum((pts_id - q_centered) ** 2))
            if slots_sw is not None and bool(mask[j_idx, slots_sw].all()):
                pts_sw = coords[j_idx, slots_sw]
                if np.isfinite(pts_sw).all() and (np.abs(pts_sw).sum(axis=-1) > 0.0).all():
                    err = min(err, float(np.sum((pts_sw - q_centered) ** 2)))
            cost[m_idx, j_idx] = err

        if not np.isfinite(cost[m_idx]).any():
            return None, None, None, None

    try:
        row_ind, col_ind = linear_sum_assignment(cost)
    except ValueError:
        return None, None, None, None

    if len(row_ind) != m_count or not np.isfinite(cost[row_ind, col_ind]).all():
        return None, None, None, None

    ordered_cols = np.empty(m_count, dtype=np.int32)
    ordered_cols[row_ind] = col_ind

    pred_atom_dicts: list[dict[str, np.ndarray]] = []
    for m_idx, res in enumerate(patch.member_residues):
        j_idx = int(ordered_cols[m_idx])
        slots = token_atoms[res.res_name]
        atom_dict: dict[str, np.ndarray] = {}
        for slot_idx, atom_name in enumerate(slots):
            if not atom_name or not bool(mask[j_idx, slot_idx]):
                continue
            pt = coords[j_idx, slot_idx]
            if np.isfinite(pt).all() and float(np.abs(pt).sum()) > 0.0:
                atom_dict[atom_name] = pt
        pred_atom_dicts.append(atom_dict)

    rmsd, rot, trans = coupled_kabsch_sidechain_rmsd(pred_atom_dicts, patch.member_residues)
    return ordered_cols, float(rmsd), rot, trans


def write_attempt_pdb(
    rfd4_result: Any,
    out_file: pathlib.Path,
    *,
    col_ind: np.ndarray | None,
    rot: np.ndarray | None,
    trans: np.ndarray | None,
    c_crd_offset: np.ndarray,
) -> None:
    """Write target-aligned RFD4 attempt PDB with motif B-factor annotations."""
    if col_ind is None:
        rfd4_result.save_pdb(out_file, offset=c_crd_offset, highlight_res_ids=())
    else:
        assert rot is not None and trans is not None
        rfd4_result.save_pdb(
            out_file,
            R=rot,
            t=trans,
            highlight_res_ids=np.asarray(col_ind, dtype=np.int64) + 1,
        )


def write_seeds_pdb(
    sanitized_structure: gemmi.Structure,
    out_file: pathlib.Path,
    *,
    valid_seed_sets: dict[str, set[str]],
    seed_pass_counts: dict[str, dict[str, int]],
) -> None:
    """Write sanitized target PDB with B-factor ``-1.0`` (non-seed) or ``min(N, 999)``."""

    def _seed_b_factor(cid: str, res: gemmi.Residue) -> float:
        valid_set = valid_seed_sets.get(cid, set())
        counts = seed_pass_counts.get(cid, {})
        res_token = f"{cid}{int(res.seqid.num)}{res.seqid.icode.strip()}"
        if res_token in valid_set:
            return float(min(int(counts.get(res_token, 0)), 999))
        return -1.0

    write_gemmi_pdb(sanitized_structure, out_file, residue_b_factor_fn=_seed_b_factor)


def write_zero_design_seeds(plans: Sequence[TargetPlan], seeds_dir: pathlib.Path) -> None:
    """Export baseline ``seeds/<target>.pdb`` files when no designs are sampled."""
    for plan in plans:
        write_seeds_pdb(
            plan.sanitized_structure,
            seeds_dir / f"{plan.target_name}.pdb",
            valid_seed_sets={ctx.chain_id: set(ctx.valid_seeds) for ctx in plan.chains},
            seed_pass_counts={
                ctx.chain_id: {tok: 0 for tok in ctx.valid_seeds} for ctx in plan.chains
            },
        )


def evaluate_decompose_prediction(
    pred_result: Any,
    rfd4_coors37_ang: np.ndarray,
    rfd4_mask37: np.ndarray,
    patch: PatchSpec,
    col_ind: np.ndarray,
    *,
    num_aa: int,
    f_sc_rmsd: float = 2.0,
    f_motif_rmsd: float = 1.0,
    f_plddt: float = 0.85,
    f_pae: float = 3.0,
) -> tuple[dict[str, float | str | bool], int, np.ndarray, np.ndarray, np.ndarray]:
    """Score an ``AF3Like`` prediction on 4-atom backbone scRMSD, motif sidechains, and confidence."""
    total_tokens = int(np.asarray(pred_result.token_pad_mask).shape[0])
    motif_mask = np.zeros(total_tokens, dtype=bool)
    motif_mask[np.asarray(col_ind, dtype=np.int32)] = True

    mean_plddt_arr = np.asarray(jnp.atleast_1d(pred_result.mean_plddt), dtype=np.float64)
    motif_plddt_arr = np.asarray(
        jnp.atleast_1d(pred_result.index_plddt(motif_mask)), dtype=np.float64
    )
    mean_pae_arr = np.asarray(
        jnp.atleast_1d(pred_result.max_pae * pred_result.mean_pae), dtype=np.float64
    )
    ptm_arr = np.asarray(jnp.atleast_1d(pred_result.ptm), dtype=np.float64)

    pred_atom24, pred_mask24 = extract_atom24_samples(pred_result)
    pred_atom24_np = np.asarray(pred_atom24[:, :num_aa], dtype=np.float32)
    pred_mask24_np = np.asarray(pred_mask24[:num_aa], dtype=bool)
    num_samples = int(pred_atom24_np.shape[0])

    # In RFD4 atom37, protein backbone slots are N=0, CA=1, C=2, O=4:
    rfd4_bb4 = np.asarray(rfd4_coors37_ang[:num_aa, [0, 1, 2, 4], :], dtype=np.float32)
    rfd4_coords24 = np.zeros((num_aa, 24, 3), dtype=np.float32)
    rfd4_coords24[:, :4, :] = rfd4_bb4
    backbone_complete = bool(rfd4_mask37[:num_aa, [0, 1, 2, 4]].all()) and bool(
        pred_mask24_np[:, :4].all()
    )

    if backbone_complete:
        sc_rmsd_arr = np.asarray(
            JIT_VMAP_BACKBONE_RMSD(
                jnp.asarray(pred_atom24_np),
                jnp.asarray(rfd4_coords24),
                jnp.ones((num_aa,), dtype=jnp.bool_),
            ),
            dtype=np.float64,
        )
    else:
        sc_rmsd_arr = np.full((num_samples,), np.inf, dtype=np.float64)

    motif_rmsd_arr = np.full((num_samples,), np.inf, dtype=np.float64)
    rots: list[np.ndarray] = []
    trans_list: list[np.ndarray] = []
    for s_idx in range(num_samples):
        res_atom_dicts = extract_residue_atom_dicts(
            pred_result, sample_index=s_idx, num_residues=num_aa
        )
        matched_dicts = [res_atom_dicts[int(j)] for j in col_ind]
        if any(
            any(an not in matched_dicts[m_idx] for an in res.sc_atom_names)
            for m_idx, res in enumerate(patch.member_residues)
        ):
            rots.append(np.eye(3, dtype=np.float64))
            trans_list.append(np.zeros(3, dtype=np.float64))
        else:
            m_rmsd, r_s, t_s = coupled_kabsch_sidechain_rmsd(
                matched_dicts, patch.member_residues
            )
            motif_rmsd_arr[s_idx] = m_rmsd
            rots.append(r_s)
            trans_list.append(t_s)

    sc_rmsd = float(sc_rmsd_arr.mean())
    motif_rmsd = float(motif_rmsd_arr.mean())
    mean_plddt = float(mean_plddt_arr.mean())
    motif_plddt = float(motif_plddt_arr.mean())
    mean_pae = float(mean_pae_arr.mean())
    ptm = float(ptm_arr.mean())

    finite_motif = np.where(np.isfinite(motif_rmsd_arr), motif_rmsd_arr, np.inf)
    best_sample_idx = int(np.argmin(finite_motif)) if np.isfinite(finite_motif).any() else 0

    plddt_raw = np.asarray(pred_result.plddt, dtype=np.float64)
    best_res_plddt = (
        plddt_raw[:num_aa] if plddt_raw.ndim == 1 else plddt_raw[best_sample_idx, :num_aa]
    )

    failures: list[str] = []
    if not (np.isfinite(sc_rmsd) and sc_rmsd < f_sc_rmsd):
        failures.append("f_sc_rmsd")
    if not (np.isfinite(motif_rmsd) and motif_rmsd < f_motif_rmsd):
        failures.append("f_motif_rmsd")
    if not (mean_plddt > f_plddt):
        failures.append("f_plddt")
    if not (np.isfinite(mean_pae) and mean_pae < f_pae):
        failures.append("f_pae")
    failure_str = ";".join(failures) if failures else "none"

    summary: dict[str, float | str | bool] = {
        "sc_rmsd": sc_rmsd,
        "motif_rmsd": motif_rmsd,
        "mean_plddt": mean_plddt,
        "motif_plddt": motif_plddt,
        "mean_pae": mean_pae,
        "ptm": ptm,
        "failure": failure_str,
        "accepted": not failures,
    }
    return (
        summary,
        best_sample_idx,
        rots[best_sample_idx],
        trans_list[best_sample_idx],
        best_res_plddt,
    )


def main(argv: Sequence[str] | None = None) -> None:
    """Run surface patch decomposition, RFD4 unindexed scaffolding, and AF3Like filtering."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args = parse_args(argv)

    pdb_paths = discover_pdb_paths(args.in_path)
    subdirs = ("attempts", "seeds") if args.gen_only else ("attempts", "success", "seeds")
    prepare_output_dir(args.out_path, overwrite=args.overwrite, subdirs=subdirs)

    scores_csv = (
        None
        if args.gen_only
        else ScoreCSV(args.out_path / "scores.csv", keys=list(SCORE_COLUMNS), default="none")
    )

    plans, m_max = prepare_target_plans(
        pdb_paths,
        interface_cutoff=args.interface_cutoff,
        cutoff=args.cutoff,
        min_num_patch=args.min_num_patch,
        log=logger,
    )

    if m_max == 0 or args.num_designs == 0:
        logger.info(
            "Skipping model initialization (M_max=%d, num_designs=%d); writing seed heatmaps only.",
            m_max,
            args.num_designs,
        )
        write_zero_design_seeds(plans, args.out_path / "seeds")
        return

    n_pad = align_up(args.num_aa + max(m_max, 1), 64)
    base_rfd4_input = RFD4Input.from_chains(args.num_aa)
    jax_model = RFD4.from_pretrained()
    sampler = RFD4Sampler(jax_model.step(), jax_model)

    predictor = (
        None
        if args.gen_only
        else build_af3_predictor(
            args.model,
            cache=args.cache,
            num_recycle=args.num_recycle,
            num_samples=args.num_samples,
        )
    )

    forbidden_set = frozenset(parse_forbidden_aas(args.forbidden))
    base_key = jax.random.key(args.seed)

    for target_idx, plan in enumerate(plans):
        valid_seed_sets: dict[str, set[str]] = {}
        seed_pass_counts: dict[str, dict[str, int]] = {}

        for chain_idx, ctx in enumerate(plan.chains):
            valid_seed_sets[ctx.chain_id] = set(ctx.valid_seeds)
            seed_pass_counts[ctx.chain_id] = {tok: 0 for tok in ctx.valid_seeds}
            if not ctx.valid_seeds:
                continue

            covered: set[str] = set()
            seed_attempts: dict[str, int] = {tok: 0 for tok in ctx.valid_seeds}
            active_chain_batches: dict[str, Any] = {}
            chain_rng = np.random.default_rng(
                np.random.SeedSequence([args.seed & 0xFFFFFFFF, target_idx, chain_idx])
            )

            n_attempts = 0
            n_f_match = 0
            n_f_rfd4_rmsd = 0
            n_f_forbidden = 0
            n_accepted = 0

            for attempt in range(args.num_designs):
                active_seeds = [
                    tok
                    for tok in ctx.valid_seeds
                    if tok not in covered and seed_attempts[tok] < args.max_seed_attempts
                ]
                if not active_seeds:
                    break

                n_attempts += 1
                center_residue = active_seeds[int(chain_rng.integers(0, len(active_seeds)))]
                seed_attempts[center_residue] += 1
                patch = ctx.patches[center_residue]

                if center_residue not in active_chain_batches:
                    active_chain_batches[center_residue] = compile_patch_batch(
                        base_rfd4_input,
                        patch,
                        n_pad,
                    )
                batch = active_chain_batches[center_residue]

                attempt_key = jax.random.fold_in(
                    jax.random.fold_in(jax.random.fold_in(base_key, target_idx), chain_idx),
                    attempt,
                )
                rfd4_key, pred_key = jax.random.split(attempt_key)

                t0 = time.perf_counter()
                rfd4_res = sampler.sample(
                    batch,
                    key=rfd4_key,
                    recycle_schedule=THINKING_HIGH,
                )
                t_rfd4_s = time.perf_counter() - t0

                aa_pred_idx = rfd4_res.aa_idx(num_aa=args.num_aa)
                sequence = rfd4_res.sequence(num_aa=args.num_aa)
                coors_ang, atom_mask = rfd4_res.to_atom37(num_aa=args.num_aa)
                col_ind, rfd4_motif_rmsd, r_rfd4, t_rfd4 = match_rfd4_patch(
                    patch, aa_pred_idx, coors_ang, atom_mask
                )

                pdb_filename = f"{plan.target_name}_{center_residue}_{attempt}.pdb"
                write_attempt_pdb(
                    rfd4_res,
                    args.out_path / "attempts" / pdb_filename,
                    col_ind=col_ind,
                    rot=r_rfd4,
                    trans=t_rfd4,
                    c_crd_offset=patch.c_crd_offset,
                )

                if args.gen_only:
                    logger.info(
                        "%s %s attempt=%d seed=%s t_rfd4_s=%.3f (gen_only)",
                        plan.target_name,
                        ctx.chain_id,
                        attempt,
                        center_residue,
                        t_rfd4_s,
                    )
                    continue

                assert scores_csv is not None and predictor is not None
                covered_str = ";".join(patch.member_tokens)
                base_row: dict[str, Any] = {
                    "target": plan.target_name,
                    "target_chain": ctx.chain_id,
                    "attempt": attempt,
                    "center_residue": center_residue,
                    "covered_residues": covered_str,
                    "sequence": sequence,
                }

                if col_ind is None:
                    n_f_match += 1
                    scores_csv.write_line({**base_row, "failure": "f_match"})
                    logger.info(
                        "%s %s attempt=%d seed=%s t_rfd4_s=%.3f failure=f_match",
                        plan.target_name,
                        ctx.chain_id,
                        attempt,
                        center_residue,
                        t_rfd4_s,
                    )
                    continue

                prefold_failures: list[str] = []
                if rfd4_motif_rmsd is None or not (rfd4_motif_rmsd < args.f_rfd4_rmsd):
                    n_f_rfd4_rmsd += 1
                    prefold_failures.append("f_rfd4_rmsd")
                if has_forbidden_scaffold_aa(sequence, col_ind, forbidden_set):
                    n_f_forbidden += 1
                    prefold_failures.append("f_forbidden")
                if prefold_failures:
                    failure_str = ";".join(prefold_failures)
                    scores_csv.write_line(
                        {
                            **base_row,
                            "rfd4_motif_rmsd": _format_score(rfd4_motif_rmsd),
                            "failure": failure_str,
                        }
                    )
                    logger.info(
                        "%s %s attempt=%d seed=%s t_rfd4_s=%.3f rfd4_motif_rmsd=%s failure=%s",
                        plan.target_name,
                        ctx.chain_id,
                        attempt,
                        center_residue,
                        t_rfd4_s,
                        _format_score(rfd4_motif_rmsd),
                        failure_str,
                    )
                    continue

                spec = prepare_af3_protein_spec(
                    [sequence],
                    model=args.model,
                    max_total_length=args.num_aa,
                )
                t1 = time.perf_counter()
                prediction = predictor(pred_key, spec)
                t_pred_s = time.perf_counter() - t1

                summary, best_idx, r_best, t_best, best_plddt = evaluate_decompose_prediction(
                    prediction.result,
                    coors_ang,
                    atom_mask,
                    patch,
                    col_ind,
                    num_aa=args.num_aa,
                    f_sc_rmsd=args.f_sc_rmsd,
                    f_motif_rmsd=args.f_motif_rmsd,
                    f_plddt=args.f_plddt,
                    f_pae=args.f_pae,
                )

                if bool(summary["accepted"]):
                    n_accepted += 1
                    save_aligned_prediction_pdb(
                        prediction,
                        args.out_path / "success" / pdb_filename,
                        sample_index=best_idx,
                        rot=r_best,
                        trans=t_best,
                        res_plddt=best_plddt,
                    )
                    for tok in patch.valid_seed_tokens:
                        covered.add(tok)
                        seed_pass_counts[ctx.chain_id][tok] += 1

                scores_csv.write_line(
                    {
                        **base_row,
                        "rfd4_motif_rmsd": _format_score(rfd4_motif_rmsd),
                        "sc_rmsd": _format_score(float(summary["sc_rmsd"])),
                        "motif_rmsd": _format_score(float(summary["motif_rmsd"])),
                        "mean_plddt": _format_score(float(summary["mean_plddt"])),
                        "motif_plddt": _format_score(float(summary["motif_plddt"])),
                        "mean_pae": _format_score(float(summary["mean_pae"])),
                        "ptm": _format_score(float(summary["ptm"])),
                        "failure": str(summary["failure"]),
                    }
                )
                logger.info(
                    "%s %s attempt=%d seed=%s t_rfd4_s=%.3f t_pred_s=%.3f "
                    "sc_rmsd=%.4f motif_rmsd=%.4f mean_plddt=%.4f mean_pae=%.4f failure=%s",
                    plan.target_name,
                    ctx.chain_id,
                    attempt,
                    center_residue,
                    t_rfd4_s,
                    t_pred_s,
                    float(summary["sc_rmsd"]),
                    float(summary["motif_rmsd"]),
                    float(summary["mean_plddt"]),
                    float(summary["mean_pae"]),
                    str(summary["failure"]),
                )

            active_chain_batches.clear()
            logger.info(
                "Chain summary %s %s: n_attempts=%d n_f_match=%d "
                "n_f_rfd4_rmsd=%d n_f_forbidden=%d n_accepted=%d covered=%d/%d",
                plan.target_name,
                ctx.chain_id,
                n_attempts,
                n_f_match,
                n_f_rfd4_rmsd,
                n_f_forbidden,
                n_accepted,
                len(covered),
                len(ctx.valid_seeds),
            )

        write_seeds_pdb(
            plan.sanitized_structure,
            args.out_path / "seeds" / f"{plan.target_name}.pdb",
            valid_seed_sets=valid_seed_sets,
            seed_pass_counts=seed_pass_counts,
        )


if __name__ == "__main__":
    main()
