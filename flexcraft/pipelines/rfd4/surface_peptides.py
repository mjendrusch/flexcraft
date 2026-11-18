"""Graft one or more flexible surface-exposed peptides onto RFD4 scaffolds with AF3Like filtering.

Adapts ``flexcraft.pipelines.graft.surface_peptides`` to ``rfd4fx`` using
sequence (``C_SEQ``) and stochastic peptide-sidechain RASA (``C_RAS = 3``)
conditioning via ``RFD4Input``, optional cyclic ``Screw`` symmetrization inside
``step`` with standard ``RFD4Sampler``, and MSA-free, template-free ``AF3Like``
structure prediction filtered on non-peptide scaffold ``scRMSD`` and ``pLDDT``.

Example:
    python -m flexcraft.pipelines.rfd4.surface_peptides \\
        --peptides MERRYCHRISTMAS --assembly "20-50,@0,20-50" \\
        --p_exposed 0.9 --num_designs 5
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import pathlib
from typing import Sequence
import warnings

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from rfd4fx.flow_matching import RFD4Sampler, THINKING_HIGH, force_zero_com, vf_to_score
from rfd4fx.model import RFD4
from rfd4fx.nn.transformer import compute_flow_emit_masks
from rfd4fx.rfd4_input import RFD4Input
from salad.inference.symmetry import Screw
from scipy.optimize import linear_sum_assignment

import flexcraft.sequence.aa_codes as aas
from flexcraft.pipelines.order.common import gs_pad
from flexcraft.pipelines.rfd4.common import (
    BACKBONE_RMSD,
    JIT_VMAP_BACKBONE_RMSD,
    add_forbidden_args,
    add_gen_only_args,
    clamp_conditioned_aa,
    extract_atom24_samples,
    extract_chain_sequences,
    forbid_amino_acids,
    kabsch_transform,
    parse_bool,
    parse_forbidden_aas,
    prepare_af3_protein_spec,
    setup_rfd4_output_dirs,
    sidechain_atom37_mask,
)
from flexcraft.structure.af3_like import AF3Like
from flexcraft.structure.common._utils import align_up

SURFACE_SCORE_KEYS: tuple[str, ...] = (
    "attempt",
    "sequence",
    "plddt",
    "max_plddt",
    "motif_plddt",
    "sc_rmsd",
    "min_sc_rmsd",
    "mean_pae",
    "mean_ptm",
    "exposed_fraction",
    "success",
)

_CANONICAL_AA = frozenset(aas.AF2_CODE)
prepare_surface_af3_spec = prepare_af3_protein_spec


@dataclass(frozen=True)
class SegmentSpec:
    """Single scaffold length range or grafted peptide reference inside a chain."""

    is_peptide: bool
    min_length: int
    max_length: int
    peptide_index: int | None = None
    sequence: str = ""


@dataclass(frozen=True)
class AssemblySpec:
    """Validated multi-chain or single-chain surface-peptide assembly specification."""

    chains: tuple[tuple[SegmentSpec, ...], ...]
    budgets: tuple[int, ...] | None
    symmetrize: bool
    sym_count: int
    min_total_length: int
    max_total_length: int
    rfd4_pad_to: int


def build_parser() -> argparse.ArgumentParser:
    """Construct the CLI argument parser for the RFD4 surface-peptides pipeline."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--assembly",
        default="20-50,@0,20-50",
        help="assembly string ('chain1:chain2', segments 'min-max', 'N', or '@k' for peptide k)",
    )
    parser.add_argument(
        "--assembly_budget",
        default="none",
        help="colon-separated per-chain residue budgets (or 'none' to disable budget padding)",
    )
    parser.add_argument(
        "--peptide_length",
        type=int,
        default=0,
        help="target length for GGS-linker peptide padding (0 disables padding)",
    )
    parser.add_argument(
        "--peptides",
        default="MERRYCHRISTMAS",
        help="comma-separated list of 1-letter peptide sequences to graft",
    )
    parser.add_argument(
        "--p_exposed",
        type=float,
        default=0.9,
        help="per-residue probability in [0, 1] of surface-exposed RASA conditioning on peptide sidechains",
    )
    parser.add_argument(
        "--symmetrize",
        type=parse_bool,
        nargs="?",
        const=True,
        default=False,
        metavar="{True,False}",
        help="enable cyclic Screw symmetrization",
    )
    parser.add_argument(
        "--no-symmetrize",
        dest="symmetrize",
        action="store_false",
        help="disable cyclic Screw symmetrization",
    )
    parser.add_argument(
        "--sym_count",
        type=int,
        default=1,
        help="fold count k >= 1 for Screw cyclic symmetry inside step",
    )
    parser.add_argument(
        "--model",
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
        "--num_designs",
        type=int,
        default=5,
        help="number of passing designs to generate before stopping",
    )
    parser.add_argument(
        "--max_attempts",
        type=int,
        default=200,
        help="maximum number of design attempts before stopping",
    )
    parser.add_argument(
        "--out_path",
        type=pathlib.Path,
        default=pathlib.Path("runs/rfd4-surface-peptides"),
    )
    parser.add_argument(
        "--f_sc_rmsd",
        type=float,
        default=2.0,
        help="maximum sample-mean non-peptide scaffold backbone scRMSD (in Angstroms)",
    )
    parser.add_argument(
        "--f_plddt",
        type=float,
        default=0.8,
        help="minimum sample-mean non-peptide scaffold pLDDT (in [0, 1])",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=4,
        help="number of structure predictor diffusion samples per design",
    )
    parser.add_argument(
        "--num_recycle",
        type=int,
        default=3,
        help="number of structure predictor recycling iterations",
    )
    parser.add_argument(
        "--force_loop",
        type=parse_bool,
        nargs="?",
        const=True,
        default=False,
        metavar="{True,False}",
        help="condition grafted peptide regions to loop secondary structure (C_SEC='L')",
    )
    parser.add_argument(
        "--no-force_loop",
        dest="force_loop",
        action="store_false",
        help="leave grafted peptide secondary structure unconditioned (default)",
    )
    add_forbidden_args(parser)
    add_gen_only_args(parser)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def parse_peptides(peptides_str: str, peptide_length: int = 0) -> list[str]:
    """Parse comma-separated peptide sequences and optionally pad with GGS linkers."""
    if peptide_length < 0:
        raise ValueError(f"--peptide_length must be >= 0, got {peptide_length}.")
    raw_items = [item.strip().upper() for item in peptides_str.split(",")]
    if not raw_items or any(not item for item in raw_items):
        raise ValueError("--peptides must contain one or more non-empty peptide sequences.")
    peptides: list[str] = []
    for pep in raw_items:
        invalid = sorted(set(pep) - _CANONICAL_AA)
        if invalid:
            raise ValueError(
                f"Peptide {pep!r} contains non-canonical amino acid characters: {invalid!r}."
            )
        peptides.append(gs_pad(pep, num_aa=peptide_length) if peptide_length > 0 else pep)
    return peptides


def _parse_segment(token: str, peptides: Sequence[str]) -> SegmentSpec:
    """Parse a single comma-delimited segment token (``@k``, ``N``, or ``min-max``)."""
    text = token.strip()
    if not text:
        raise ValueError("Empty segment in --assembly.")
    if text.startswith("@"):
        idx_str = text[1:]
        if not idx_str.isdigit():
            raise ValueError(
                f"Invalid peptide segment {text!r}; expected non-negative '@k' index."
            )
        pep_idx = int(idx_str)
        if pep_idx >= len(peptides):
            raise ValueError(
                f"Peptide index '@{pep_idx}' out of range for {len(peptides)} peptide(s)."
            )
        seq = peptides[pep_idx]
        return SegmentSpec(
            is_peptide=True,
            min_length=len(seq),
            max_length=len(seq),
            peptide_index=pep_idx,
            sequence=seq,
        )
    if "-" in text:
        parts = text.split("-")
        if len(parts) != 2 or not parts[0].isdigit() or not parts[1].isdigit():
            raise ValueError(f"Malformed scaffold length range {text!r} in --assembly.")
        min_len, max_len = int(parts[0]), int(parts[1])
    else:
        if not text.isdigit():
            raise ValueError(f"Malformed scaffold length {text!r} in --assembly.")
        min_len = max_len = int(text)
    if min_len > max_len:
        raise ValueError(f"Scaffold min_length > max_length in segment {text!r}.")
    return SegmentSpec(is_peptide=False, min_length=min_len, max_length=max_len)


def parse_assembly_spec(
    assembly_str: str,
    peptides: Sequence[str],
    *,
    assembly_budget: str | None = "none",
    symmetrize: bool = False,
    sym_count: int = 1,
) -> AssemblySpec:
    """Parse and validate assembly segments, per-chain budgets, and cyclic symmetry."""
    if sym_count < 1:
        raise ValueError(f"--sym_count must be >= 1, got {sym_count}.")
    if symmetrize and sym_count < 2:
        raise ValueError("--symmetrize requires --sym_count >= 2.")
    if not symmetrize and sym_count > 1:
        warnings.warn(
            f"--sym_count={sym_count} has no effect when --symmetrize is False.",
            UserWarning,
            stacklevel=2,
        )

    chain_strs = [c.strip() for c in assembly_str.strip().split(":")]
    if not chain_strs or any(not c for c in chain_strs):
        raise ValueError("--assembly must specify at least one non-empty chain.")
    if len(chain_strs) > 26:
        raise ValueError(f"--assembly supports at most 26 chains (A..Z), got {len(chain_strs)}.")

    chains = tuple(
        tuple(_parse_segment(seg, peptides) for seg in chain_str.split(","))
        for chain_str in chain_strs
    )

    budgets: tuple[int, ...] | None = None
    if assembly_budget is not None and assembly_budget.strip().lower() != "none":
        budget_parts = [b.strip() for b in assembly_budget.strip().split(":")]
        if len(budget_parts) != len(chains):
            raise ValueError(
                f"--assembly_budget has {len(budget_parts)} entries, expected {len(chains)}."
            )
        parsed_budgets: list[int] = []
        for part in budget_parts:
            if not part.isdigit() or int(part) < 1:
                raise ValueError(f"Invalid chain budget {part!r}; expected positive integer.")
            parsed_budgets.append(int(part))
        budgets = tuple(parsed_budgets)

    min_total = 0
    max_total = 0
    min_scaffold_total = 0
    for c_idx, segs in enumerate(chains):
        c_min = sum(s.min_length for s in segs)
        c_max = sum(s.max_length for s in segs)
        c_pep = sum(s.min_length for s in segs if s.is_peptide)
        if budgets is not None:
            if c_min > budgets[c_idx]:
                raise ValueError(
                    f"Chain {c_idx} minimum length ({c_min}) exceeds --assembly_budget ({budgets[c_idx]})."
                )
            min_total += budgets[c_idx]
            max_total += budgets[c_idx]
            min_scaffold_total += budgets[c_idx] - c_pep
        else:
            if c_min < 1:
                raise ValueError(
                    f"Chain {c_idx} in --assembly can have length 0; minimum must be >= 1."
                )
            min_total += c_min
            max_total += c_max
            min_scaffold_total += c_min - c_pep

    if min_scaffold_total < 1:
        raise ValueError("--assembly must contain at least 1 guaranteed designable scaffold residue.")

    if symmetrize:
        if len(chains) > 1:
            if len(chains) % sym_count != 0:
                raise ValueError(
                    f"Multi-chain --assembly has {len(chains)} chains, not divisible by --sym_count={sym_count}."
                )
            chains_per_unit = len(chains) // sym_count
            if chains != chains[:chains_per_unit] * sym_count:
                raise ValueError(
                    "All symmetric multi-chain units in --assembly must have identical segment specs."
                )
            if budgets is not None and budgets != budgets[:chains_per_unit] * sym_count:
                raise ValueError("All symmetric multi-chain units in --assembly_budget must match.")
        else:
            segs0 = chains[0]
            if budgets is not None and budgets[0] % sym_count != 0:
                raise ValueError(
                    f"Single-chain symmetric --assembly_budget ({budgets[0]}) must be divisible by --sym_count={sym_count}."
                )
            if len(segs0) % sym_count == 0:
                if segs0 != segs0[: len(segs0) // sym_count] * sym_count:
                    raise ValueError(
                        "Single-chain symmetric --assembly segments must repeat identically sym_count times."
                    )
            elif len(segs0) == 1 and not segs0[0].is_peptide:
                u_min = max(1, (segs0[0].min_length + sym_count - 1) // sym_count)
                u_max = (
                    (budgets[0] // sym_count)
                    if budgets is not None
                    else (segs0[0].max_length // sym_count)
                )
                if u_min > u_max:
                    raise ValueError(
                        f"Single-chain segment {segs0[0].min_length}-{segs0[0].max_length} has no length divisible by --sym_count={sym_count}."
                    )
                min_total = sym_count * u_min if budgets is None else budgets[0]
                max_total = sym_count * u_max
            else:
                raise ValueError(
                    f"Single-chain symmetric --assembly has {len(segs0)} segments, not divisible by --sym_count={sym_count}."
                )

    if symmetrize:
        unit_max = max_total // sym_count
        multiple = 64 if sym_count % 2 == 1 else 32
        rfd4_pad_to = sym_count * align_up(unit_max, multiple)
    else:
        rfd4_pad_to = align_up(max_total, 64)

    return AssemblySpec(
        chains=chains,
        budgets=budgets,
        symmetrize=symmetrize,
        sym_count=sym_count,
        min_total_length=min_total,
        max_total_length=max_total,
        rfd4_pad_to=rfd4_pad_to,
    )


def _sample_chain_unit(
    segments: Sequence[SegmentSpec],
    budget: int | None,
    p_exposed: float,
    rng: np.random.Generator,
) -> tuple[str, np.ndarray, np.ndarray]:
    """Sample one chain (or single-chain repeat unit), N/C budget pad, and exposure mask."""
    parts: list[str] = []
    pep_flags: list[np.ndarray] = []
    min_after = [
        sum(s.min_length for s in segments[idx + 1 :]) for idx in range(len(segments))
    ]
    used = 0
    for idx, seg in enumerate(segments):
        if seg.is_peptide:
            parts.append(seg.sequence)
            pep_flags.append(np.ones(len(seg.sequence), dtype=bool))
            used += len(seg.sequence)
        else:
            max_len = (
                min(seg.max_length, budget - used - min_after[idx])
                if budget is not None
                else seg.max_length
            )
            length = int(rng.integers(seg.min_length, max_len + 1))
            parts.append("X" * length)
            pep_flags.append(np.zeros(length, dtype=bool))
            used += length

    seq = "".join(parts)
    is_peptide = np.concatenate(pep_flags) if pep_flags else np.zeros(0, dtype=bool)

    if budget is not None and len(seq) < budget:
        remaining = budget - len(seq)
        left_pad = int(rng.integers(0, remaining + 1))
        right_pad = remaining - left_pad
        seq = ("X" * left_pad) + seq + ("X" * right_pad)
        is_peptide = np.concatenate(
            [np.zeros(left_pad, dtype=bool), is_peptide, np.zeros(right_pad, dtype=bool)]
        )

    exposed = is_peptide & (rng.random(len(seq)) < p_exposed)
    return seq, is_peptide, exposed


def sample_attempt_chains(
    spec: AssemblySpec,
    p_exposed: float,
    rng: np.random.Generator,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Sample per-chain sequences, ``is_peptide`` mask, and ``exposed_mask`` for one attempt."""
    if not 0.0 <= p_exposed <= 1.0:
        raise ValueError(f"--p_exposed must be in [0, 1], got {p_exposed}.")

    if not spec.symmetrize or len(spec.chains) > 1:
        repeats = spec.sym_count if spec.symmetrize else 1
        unit_count = len(spec.chains) // repeats
        unit_samples = [
            _sample_chain_unit(
                spec.chains[idx],
                spec.budgets[idx] if spec.budgets is not None else None,
                p_exposed,
                rng,
            )
            for idx in range(unit_count)
        ]
        unit_seqs = [item[0] for item in unit_samples]
        unit_pep = np.concatenate([item[1] for item in unit_samples])
        unit_exp = np.concatenate([item[2] for item in unit_samples])
        return (
            unit_seqs * repeats,
            np.tile(unit_pep, repeats),
            np.tile(unit_exp, repeats),
        )

    segs0 = spec.chains[0]
    unit_budget = (spec.budgets[0] // spec.sym_count) if spec.budgets is not None else None
    if len(segs0) % spec.sym_count == 0:
        unit_segs = segs0[: len(segs0) // spec.sym_count]
    else:
        u_min = max(1, (segs0[0].min_length + spec.sym_count - 1) // spec.sym_count)
        u_max = unit_budget if unit_budget is not None else (segs0[0].max_length // spec.sym_count)
        unit_segs = (SegmentSpec(is_peptide=False, min_length=u_min, max_length=u_max),)

    u_seq, u_pep, u_exp = _sample_chain_unit(unit_segs, unit_budget, p_exposed, rng)
    return (
        [u_seq * spec.sym_count],
        np.tile(u_pep, spec.sym_count),
        np.tile(u_exp, spec.sym_count),
    )


def build_surface_rfd4_input(
    chain_sequences: Sequence[str],
    exposed_mask: np.ndarray,
    *,
    is_peptide: np.ndarray | None = None,
    force_loop: bool = False,
    forbidden: str = "C",
    pad_to: int,
    np_seed: int = 0,
) -> RFD4Input:
    """Compile sequence-, optional loop-DSSP-, forbidden-AA-, and RASA-conditioned ``RFD4Input``."""
    chain_dicts = [{"kind": "aa", "sequence": seq} for seq in chain_sequences]
    inp = RFD4Input.from_chains(*chain_dicts, np_seed=np_seed, pad_to=pad_to)
    full_seq = "".join(chain_sequences)
    pep_mask = (
        np.asarray(is_peptide, dtype=bool)
        if is_peptide is not None
        else np.array([ch != "X" for ch in full_seq], dtype=bool)
    )
    if force_loop:
        dssp = "".join("L" if flag else "X" for flag in pep_mask)
        inp = inp.add_dssp(dssp, start=0)
    inp = forbid_amino_acids(inp, forbidden, designable_mask=~pep_mask)
    where_atom37 = sidechain_atom37_mask(full_seq, np.asarray(exposed_mask, dtype=bool))
    return inp.add_rasa(3, aa=None, where=where_atom37, start=0)


def _unit_block_indices(
    n_padded: int,
    sym_count: int,
    padding_mask: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Map contiguous ``[0 : sym_count * unit_len]`` tokens to/from ``sym_count`` fixed blocks."""
    unit_pad = n_padded // sym_count
    unit_len = jnp.maximum(jnp.sum(padding_mask[0].astype(jnp.int32)) // sym_count, 1)

    unit_idx = jnp.arange(sym_count, dtype=jnp.int32)[:, None]
    pos_idx = jnp.arange(unit_pad, dtype=jnp.int32)[None, :]
    valid_block = jnp.broadcast_to(pos_idx < unit_len, (sym_count, unit_pad)).reshape(-1)
    gather_idx = jnp.minimum(unit_idx * unit_len + pos_idx, n_padded - 1).reshape(-1)

    flat_idx = jnp.arange(n_padded, dtype=jnp.int32)
    scatter_idx = jnp.minimum(
        (flat_idx // unit_len) * unit_pad + (flat_idx % unit_len),
        n_padded - 1,
    )
    return gather_idx, valid_block, scatter_idx


def make_surface_peptide_step(
    jax_model: RFD4,
    *,
    symmetrize: bool = False,
    sym_count: int = 1,
):
    """Create the RFD4 denoising step callable with optional cyclic ``Screw`` symmetry.

    ``local_latents`` are never symmetrized or modified.
    """
    base_step = jax_model.step()
    if not (symmetrize and sym_count > 1):
        def step(model: RFD4, key, data):
            out = base_step(model, key, data)
            aa_pred = clamp_conditioned_aa(out.aa_pred, data)
            return eqx.tree_at(lambda b: b.aa_pred, out, aa_pred, is_leaf=lambda x: x is None)

        return step

    screw = Screw(
        sym_count,
        360.0 / sym_count,
        radius=0.0,
        translation=0.0,
        symmetry_axis=2,
    )

    def _symmetrize_ca(ca_b_n_3: jax.Array, emit_bb: jax.Array, padding_mask: jax.Array) -> jax.Array:
        n_padded = ca_b_n_3.shape[1]
        gather_idx, valid_block, scatter_idx = _unit_block_indices(n_padded, sym_count, padding_mask)
        block_ca = jnp.where(valid_block[:, None], ca_b_n_3[0, gather_idx], 0.0)
        expanded = jnp.broadcast_to(block_ca[:, None, :], (n_padded, 2, 3))
        rep = screw.select_pos(expanded, do_radius=False)
        sym_block = screw.replicate_pos(rep, do_radius=False)[:, 0, :]
        return force_zero_com(sym_block[scatter_idx][None], emit_bb)

    def _symmetrize_aa(aa_b_n: jax.Array, padding_mask: jax.Array) -> jax.Array:
        n_padded = aa_b_n.shape[1]
        gather_idx, valid_block, scatter_idx = _unit_block_indices(n_padded, sym_count, padding_mask)
        block_aa = jnp.where(valid_block, aa_b_n[0, gather_idx], 0)
        sym_block = screw.replicate_features(screw.select_features(block_aa))
        return jnp.where(padding_mask[0], sym_block[scatter_idx], 0)[None]

    def symmetric_step(model: RFD4, key, data):
        emit_bb = compute_flow_emit_masks(data)["bb_ca"]
        x_t_sym = _symmetrize_ca(data.x_t_bb_ca, emit_bb, data.padding_mask)
        data_in = eqx.tree_at(lambda b: b.x_t_bb_ca, data, x_t_sym)
        out = base_step(model, key, data_in)

        x1_sym = _symmetrize_ca(out.x_pred, emit_bb, data.padding_mask)
        denom = jnp.maximum(1.0 - data.t_bb_ca[:, None, None], 1e-5)
        v_bb = (x1_sym - x_t_sym) / denom
        s_bb = vf_to_score(x_t_sym, v_bb, data.t_bb_ca)

        v_pred = {**out.v_pred, "bb_ca": v_bb}
        s_pred = {**out.s_pred, "bb_ca": s_bb}
        aa_pred = _symmetrize_aa(clamp_conditioned_aa(out.aa_pred, data_in), data.padding_mask)
        aa_pred = clamp_conditioned_aa(aa_pred, data_in)
        return eqx.tree_at(
            lambda b: (b.aa_pred, b.x_pred, b.v_pred, b.s_pred),
            out,
            (aa_pred, x1_sym, v_pred, s_pred),
            is_leaf=lambda x: x is None,
        )

    return symmetric_step


def _multichain_symmetric_rmsd(
    pred_atom24: np.ndarray,
    rfd4_coords24: np.ndarray,
    scaffold_mask: np.ndarray,
    sym_count: int,
) -> np.ndarray:
    """Compute subunit-permutation-invariant scaffold backbone RMSD across multi-chain units."""
    num_samples, real_len = pred_atom24.shape[:2]
    unit_len = real_len // sym_count
    unit_scaffold = scaffold_mask[:unit_len]
    rfd4_units = rfd4_coords24.reshape(sym_count, unit_len, 24, 3)
    rfd4_scaf_bb = rfd4_units[:, unit_scaffold, :4, :].reshape(sym_count, -1, 3).astype(np.float64)
    scaffold_jnp = jnp.asarray(scaffold_mask, dtype=jnp.bool_)

    best_per_sample = np.full((num_samples,), np.inf, dtype=np.float64)
    for s_idx in range(num_samples):
        pred_units = pred_atom24[s_idx].reshape(sym_count, unit_len, 24, 3)
        pred_scaf_bb = (
            pred_units[:, unit_scaffold, :4, :].reshape(sym_count, -1, 3).astype(np.float64)
        )
        candidate_refs: list[np.ndarray] = []
        seen_perms: set[tuple[int, ...]] = set()
        for anchor in range(sym_count):
            rot, trans = kabsch_transform(pred_scaf_bb[anchor], rfd4_scaf_bb[0])
            aligned = pred_scaf_bb @ rot + trans
            diff = aligned[:, None, :, :] - rfd4_scaf_bb[None, :, :, :]
            cost = np.sum(diff * diff, axis=(-2, -1))
            _, perm_arr = linear_sum_assignment(cost)
            perm = tuple(int(p) for p in perm_arr)
            if perm not in seen_perms:
                seen_perms.add(perm)
                candidate_refs.append(rfd4_units[list(perm)].reshape(real_len, 24, 3))

        pred_single = jnp.asarray(pred_atom24[s_idx], dtype=jnp.float32)
        rmsd_vals = [
            float(BACKBONE_RMSD(pred_single, jnp.asarray(ref, dtype=jnp.float32), mask=scaffold_jnp))
            for ref in candidate_refs
        ]
        best_per_sample[s_idx] = min(rmsd_vals)
    return best_per_sample


def evaluate_surface_peptide_prediction(
    pred_result,
    rfd4_coords24: jax.Array | np.ndarray,
    rfd4_mask24: jax.Array | np.ndarray,
    is_peptide: jax.Array | np.ndarray,
    *,
    num_chains: int = 1,
    symmetrize: bool = False,
    sym_count: int = 1,
    f_sc_rmsd: float = 2.0,
    f_plddt: float = 0.8,
) -> tuple[dict[str, float | bool], int]:
    """Score an ``AF3Like`` prediction on non-peptide scaffold residues and select best sample."""
    is_pep_np = np.asarray(is_peptide, dtype=bool)
    real_len = int(is_pep_np.shape[0])
    total_tokens = int(np.asarray(pred_result.token_pad_mask).shape[0])

    scaffold_full = np.zeros(total_tokens, dtype=bool)
    scaffold_full[:real_len] = ~is_pep_np
    peptide_full = np.zeros(total_tokens, dtype=bool)
    peptide_full[:real_len] = is_pep_np

    scaffold_plddt = np.asarray(
        jnp.atleast_1d(pred_result.index_plddt(scaffold_full)), dtype=np.float64
    )
    motif_plddt = (
        np.asarray(jnp.atleast_1d(pred_result.index_plddt(peptide_full)), dtype=np.float64)
        if bool(is_pep_np.any())
        else np.zeros_like(scaffold_plddt)
    )
    scaffold_pae = np.asarray(
        jnp.atleast_1d(pred_result.max_pae * pred_result.index_pae(scaffold_full)),
        dtype=np.float64,
    )
    scaffold_ptm = np.asarray(
        jnp.atleast_1d(pred_result.index_ptm(scaffold_full)), dtype=np.float64
    )

    pred_atom24, pred_mask24 = extract_atom24_samples(pred_result)
    pred_atom24_np = np.asarray(pred_atom24[:, :real_len], dtype=np.float32)
    pred_mask24_np = np.asarray(pred_mask24[:real_len], dtype=bool)
    rfd4_coords_np = np.asarray(rfd4_coords24[:real_len], dtype=np.float32)
    rfd4_mask_np = np.asarray(rfd4_mask24[:real_len], dtype=bool)
    scaffold_np = ~is_pep_np

    num_samples = int(pred_atom24_np.shape[0])
    backbone_complete = (
        bool(scaffold_np.any())
        and bool(rfd4_mask_np[scaffold_np, :4].all())
        and bool(pred_mask24_np[scaffold_np, :4].all())
    )
    if not backbone_complete:
        rmsd_arr = np.full((num_samples,), np.inf, dtype=np.float64)
    elif symmetrize and sym_count >= 2 and num_chains > 1:
        rmsd_arr = _multichain_symmetric_rmsd(
            pred_atom24_np, rfd4_coords_np, scaffold_np, sym_count
        )
    else:
        rmsd_arr = np.asarray(
            JIT_VMAP_BACKBONE_RMSD(
                jnp.asarray(pred_atom24_np),
                jnp.asarray(rfd4_coords_np),
                jnp.asarray(scaffold_np),
            ),
            dtype=np.float64,
        )

    plddt_mean = float(scaffold_plddt.mean())
    sc_rmsd_mean = float(rmsd_arr.mean())
    finite_rmsd = np.where(np.isfinite(rmsd_arr), rmsd_arr, np.inf)
    best_sample_idx = int(np.argmin(finite_rmsd)) if np.isfinite(finite_rmsd).any() else 0
    is_success = bool(
        np.isfinite(sc_rmsd_mean) and (sc_rmsd_mean < f_sc_rmsd) and (plddt_mean > f_plddt)
    )

    summary: dict[str, float | bool] = {
        "plddt": plddt_mean,
        "max_plddt": float(scaffold_plddt.max()),
        "motif_plddt": float(motif_plddt.mean()),
        "sc_rmsd": sc_rmsd_mean,
        "min_sc_rmsd": float(rmsd_arr.min()),
        "mean_pae": float(scaffold_pae.mean()),
        "mean_ptm": float(scaffold_ptm.mean()),
        "success": is_success,
    }
    return summary, best_sample_idx


def main(argv: Sequence[str] | None = None) -> None:
    """Run surface-peptide grafting with RFD4 and optional MSA-free AF3Like filtering."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.num_designs < 1:
        parser.error("--num_designs must be >= 1")
    if args.max_attempts < 1:
        parser.error("--max_attempts must be >= 1")
    if args.num_samples < 1:
        parser.error("--num_samples must be >= 1")
    if args.num_recycle < 0:
        parser.error("--num_recycle must be >= 0")
    if not 0.0 <= args.p_exposed <= 1.0:
        parser.error("--p_exposed must be in [0, 1]")

    try:
        parse_forbidden_aas(args.forbidden)
        peptides = parse_peptides(args.peptides, args.peptide_length)
        assembly_spec = parse_assembly_spec(
            args.assembly,
            peptides,
            assembly_budget=args.assembly_budget,
            symmetrize=args.symmetrize,
            sym_count=args.sym_count,
        )
    except ValueError as exc:
        parser.error(str(exc))

    attempts_dir, predictions_dir, scores_csv = setup_rfd4_output_dirs(
        args.out_path,
        gen_only=args.gen_only,
        score_keys=SURFACE_SCORE_KEYS,
    )

    jax_model = RFD4.from_pretrained()
    step_fn = make_surface_peptide_step(
        jax_model,
        symmetrize=assembly_spec.symmetrize,
        sym_count=assembly_spec.sym_count,
    )
    sampler = RFD4Sampler(step_fn, jax_model)

    predictor = None
    if not args.gen_only:
        af3_kwargs = {"cache": str(args.cache)} if args.cache is not None else {}
        predictor = AF3Like(model=args.model, **af3_kwargs).predictor(
            num_recycle=args.num_recycle,
            num_samples=args.num_samples,
        )

    variable_length = assembly_spec.min_total_length < assembly_spec.max_total_length
    base_key = jax.random.key(args.seed)
    success_count = 0
    attempt = 0

    while success_count < args.num_designs and attempt < args.max_attempts:
        rng = np.random.default_rng(
            np.random.SeedSequence([args.seed & 0xFFFFFFFF, attempt])
        )
        np_seed = int(rng.integers(0, 2**32))
        chain_cond_seqs, is_peptide, exposed_mask = sample_attempt_chains(
            assembly_spec,
            args.p_exposed,
            rng,
        )
        inp = build_surface_rfd4_input(
            chain_cond_seqs,
            exposed_mask,
            is_peptide=is_peptide,
            force_loop=args.force_loop,
            forbidden=args.forbidden,
            pad_to=assembly_spec.rfd4_pad_to,
            np_seed=np_seed,
        )
        batch = inp.to_batch()

        attempt_key = jax.random.fold_in(base_key, attempt)
        rfd4_key, pred_key = jax.random.split(attempt_key)

        rfd4_result = sampler.sample(
            batch,
            key=rfd4_key,
            recycle_schedule=THINKING_HIGH,
        )
        attempt_pdb = attempts_dir / f"design_{attempt}.pdb"
        rfd4_result.save_pdb(attempt_pdb)

        if args.gen_only:
            success_count += 1
            print(
                f"design_{attempt}.pdb written (gen_only, {success_count}/{args.num_designs})",
                flush=True,
            )
            attempt += 1
            continue

        assert predictor is not None and predictions_dir is not None and scores_csv is not None
        rfd4_atom24 = rfd4_result.to_atom24()
        chain_lengths = [len(seq) for seq in chain_cond_seqs]
        designed_chains = extract_chain_sequences(
            rfd4_result,
            chain_lengths,
            rfd4_restype=rfd4_atom24.restype,
        )
        spec = prepare_surface_af3_spec(
            designed_chains,
            model=args.model,
            max_total_length=assembly_spec.max_total_length,
            variable_length=variable_length,
            symmetrize=assembly_spec.symmetrize,
            sym_count=assembly_spec.sym_count,
        )
        prediction = predictor(pred_key, spec)
        summary, best_sample_idx = evaluate_surface_peptide_prediction(
            prediction.result,
            rfd4_atom24.coords,
            rfd4_atom24.mask,
            is_peptide,
            num_chains=len(designed_chains),
            symmetrize=assembly_spec.symmetrize,
            sym_count=assembly_spec.sym_count,
            f_sc_rmsd=args.f_sc_rmsd,
            f_plddt=args.f_plddt,
        )

        prediction_pdb = predictions_dir / f"design_{attempt}.pdb"
        prediction.save_pdb(prediction_pdb, sample_index=best_sample_idx)

        exposed_fraction = (
            float(exposed_mask[is_peptide].mean()) if bool(is_peptide.any()) else 0.0
        )
        row: dict[str, object] = {
            "attempt": attempt,
            "sequence": ":".join(designed_chains),
            "exposed_fraction": exposed_fraction,
            **summary,
        }
        scores_csv.write_line(row)

        if summary["success"]:
            success_count += 1
        print(
            f"design_{attempt}.pdb written (sc_rmsd={summary['sc_rmsd']:.2f}, "
            f"plddt={summary['plddt']:.3f}, success={summary['success']}, "
            f"{success_count}/{args.num_designs})",
            flush=True,
        )
        attempt += 1


if __name__ == "__main__":
    main()
