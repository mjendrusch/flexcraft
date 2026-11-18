"""RFD4 input conditioning and sequence decoding helpers."""

from __future__ import annotations

from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as np
from rfd4fx.encoding import UNIFIED_ATOM37_ENCODING
from rfd4fx.flow_matching import RFD4Result
from rfd4fx.rfd4_input import RFD4Input
from rfd4fx.spec import sequence_to_names

import flexcraft.sequence.aa_codes as aas
from flexcraft.pipelines.rfd4.common.cli import parse_forbidden_aas
from flexcraft.pipelines.rfd4.common.geometry import BACKBONE_ATOMS

_RFD4_AA_VOCAB = "X" + aas.AF2_CODE + "X"


def forbid_amino_acids(
    rfd4_input: RFD4Input,
    forbidden: str,
    *,
    designable_mask: np.ndarray | Sequence[bool] | None = None,
) -> RFD4Input:
    """Set ``C_SEQ_PROB_<AA> = 0.0`` on designable residues via ``RFD4Input.forbid``."""
    codes = parse_forbidden_aas(forbidden)
    if not codes or not hasattr(rfd4_input, "forbid"):
        return rfd4_input
    where = True if designable_mask is None else np.asarray(designable_mask, dtype=bool)
    return rfd4_input.forbid("".join(codes), where=where)


def decode_rfd4_tokens(token_indices: np.ndarray | jax.Array) -> str:
    """Decode 1D RFD4/AtomWorks token indices (1..20 = AF2_CODE) to a 1-letter AA string."""
    indices = np.asarray(token_indices, dtype=np.int32).reshape(-1)
    return "".join(
        _RFD4_AA_VOCAB[int(idx)] if 0 <= int(idx) < len(_RFD4_AA_VOCAB) else "X"
        for idx in indices
    )


def extract_chain_sequences(
    rfd4_result: RFD4Result,
    chain_lengths: Sequence[int],
    *,
    rfd4_restype: np.ndarray | jax.Array | None = None,
) -> list[str]:
    """Extract designed 1-letter protein sequences per chain from an ``RFD4Result``."""
    total_len = sum(chain_lengths)
    if hasattr(rfd4_result, "sequence"):
        full_seq = rfd4_result.sequence()[:total_len]
    elif rfd4_restype is not None:
        full_seq = decode_rfd4_tokens(np.asarray(rfd4_restype)[:total_len])
    else:
        full_seq = decode_rfd4_tokens(np.asarray(rfd4_result.to_atom24().restype)[:total_len])

    if len(full_seq) != total_len:
        raise ValueError(f"Decoded sequence length {len(full_seq)} != expected {total_len}.")

    sequences: list[str] = []
    offset = 0
    for length in chain_lengths:
        sequences.append(full_seq[offset : offset + length])
        offset += length
    return sequences


def extract_binder_sequence(
    rfd4_result: RFD4Result,
    binder_length: int,
    *,
    rfd4_restype: np.ndarray | jax.Array | None = None,
) -> str:
    """Extract the 1-letter amino acid sequence of the designed binder chain."""
    if hasattr(rfd4_result, "sequence"):
        return rfd4_result.sequence(num_aa=binder_length)
    if rfd4_restype is None:
        rfd4_restype = rfd4_result.to_atom24().restype
    return decode_rfd4_tokens(np.asarray(rfd4_restype)[:binder_length])


def extract_target_sequences(rfd4_input: RFD4Input, binder_length: int) -> list[str]:
    """Extract 1-letter protein sequences per target chain in first-appearance order."""
    padding_mask = np.asarray(rfd4_input.padding_mask[0], dtype=bool)
    real_indices = np.flatnonzero(padding_mask)
    target_indices = real_indices[real_indices >= binder_length]
    if target_indices.size == 0:
        raise ValueError("RFD4Input contains no target tokens after binder_length.")
    if rfd4_input.cond_seq is None:
        raise ValueError("RFD4Input.cond_seq is unset; cannot decode target sequences.")

    cond_seq = np.asarray(rfd4_input.cond_seq[0], dtype=np.int32)[target_indices]
    if rfd4_input.chain_iid is None:
        chain_iids = np.ones(target_indices.shape, dtype=np.float32)
    else:
        chain_iids = np.asarray(rfd4_input.chain_iid[0], dtype=np.float32)[target_indices]

    ordered_chains: list[float] = []
    for chain_id in chain_iids:
        val = float(chain_id)
        if val not in ordered_chains:
            ordered_chains.append(val)

    target_sequences: list[str] = []
    for chain_id in ordered_chains:
        chain_mask = chain_iids == chain_id
        seq = decode_rfd4_tokens(cond_seq[chain_mask])
        if not seq:
            raise ValueError(f"Decoded empty target sequence for chain_iid={chain_id}.")
        target_sequences.append(seq)
    return target_sequences


def sidechain_atom37_mask(
    sequence: str,
    residue_mask: np.ndarray | Sequence[bool],
    *,
    kind: str = "aa",
) -> np.ndarray:
    """Build ``(L, 37)`` bool mask strictly over sidechain heavy atoms of selected residues.

    Residues without sidechain heavy atoms (such as ``GLY`` or unknown ``'X'``)
    have all ``False`` entries so backbone atoms are never conditioned.
    """
    mask_1d = np.asarray(residue_mask, dtype=bool)
    if mask_1d.shape != (len(sequence),):
        raise ValueError(
            f"residue_mask shape {mask_1d.shape} does not match sequence length {len(sequence)}."
        )
    where_atom37 = np.zeros((len(sequence), 37), dtype=bool)
    for idx in np.flatnonzero(mask_1d):
        ch = sequence[int(idx)]
        if ch == "X":
            continue
        res_name = sequence_to_names(ch, kind)[0]
        atom_names = UNIFIED_ATOM37_ENCODING.token_atoms.get(res_name, ())
        for slot_idx, raw_name in enumerate(atom_names):
            atom_name = str(raw_name).strip()
            if atom_name and atom_name not in BACKBONE_ATOMS:
                where_atom37[int(idx), slot_idx] = True
    return where_atom37


def clamp_conditioned_aa(aa_pred: jax.Array, data) -> jax.Array:
    """Preserve conditioned amino acid identities on active tokens."""
    if data.cond_seq is None:
        return aa_pred
    has_cond = (data.cond_seq != 0) & data.padding_mask
    return jnp.where(has_cond, data.cond_seq, aa_pred)
