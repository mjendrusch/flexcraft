"""AF3Like spec/predictor setup, residue atom extraction, and Gemmi PDB export helpers."""

from __future__ import annotations

import pathlib
from typing import Any, Callable, Sequence

import gemmi
import jax
import numpy as np
from rfd4fx.pdb_io import save_gemmi_pdb
from salad.aflib.common.residue_constants import atom_types as _AF_ATOM37_NAMES

from flexcraft.structure.af3_like import AF3Like, AF3LikeSpec
from flexcraft.structure.common._utils import align_up


def build_af3_predictor(
    model: str = "protenix-v2",
    *,
    cache: pathlib.Path | str | None = None,
    num_recycle: int = 4,
    num_samples: int = 4,
):
    """Instantiate an eager ``AF3Like`` predictor."""
    af3_kwargs = {"cache": str(cache)} if cache is not None else {}
    return AF3Like(model=model, **af3_kwargs).predictor(
        num_recycle=num_recycle,
        num_samples=num_samples,
    )


def prepare_af3_protein_spec(
    chain_sequences: Sequence[str],
    *,
    model: str = "protenix-v2",
    max_total_length: int,
    variable_length: bool = False,
    symmetrize: bool = False,
    sym_count: int = 1,
    num_residue_atoms: int = 24,
) -> AF3LikeSpec:
    """Build an MSA-free, template-free protein ``AF3LikeSpec`` with static XLA padding.

    Uses ``num_repeats=sym_count`` only for multi-chain symmetric homooligomers
    (``len(chain_sequences) > 1 and len(chain_sequences) == sym_count``), never
    for single-chain symmetric designs.
    """
    num_chains = len(chain_sequences)
    is_multichain_homooligomer = (
        symmetrize and sym_count > 1 and num_chains > 1 and num_chains == sym_count
    )
    if is_multichain_homooligomer:
        spec = AF3LikeSpec().add_protein(
            chain_sequences[0],
            use_msa=False,
            num_repeats=sym_count,
        )
    else:
        spec = AF3LikeSpec().add_protein(*chain_sequences, use_msa=False)

    needs_token_pad = (
        variable_length
        or (model.startswith("esmfold2") and num_chains > 8)
        or (model.startswith("joltz") and num_chains > 1 and not is_multichain_homooligomer)
    )
    if needs_token_pad:
        pad_tokens = align_up(max_total_length + max(0, 2 * num_chains - 16), 32)
        return spec.pad_to(
            pad_tokens,
            num_residue_atoms=num_residue_atoms,
            num_msa=num_chains + 1,
        )
    if num_residue_atoms > 0:
        return spec.pad_atoms(num_residue_atoms=num_residue_atoms)
    return spec


def extract_atom24_samples(pred_result) -> tuple[jax.Array, jax.Array]:
    """Return ``(atom24_coords_4d, atom24_mask_2d)`` promoting single-sample results to ``(1, L, 24, 3)``."""
    pred_atom24, pred_mask24 = pred_result.atom24_samples
    if pred_result.is_single_sample:
        pred_atom24 = pred_atom24[None]
    return pred_atom24, pred_mask24


def extract_residue_atom_dicts(
    pred_result: Any,
    *,
    sample_index: int = 0,
    num_residues: int | None = None,
) -> list[dict[str, np.ndarray]]:
    """Extract per-residue ``{atom_name: xyz}`` dictionaries from any ``AF3Like`` or ``AFResult``."""
    data = getattr(pred_result, "data", None)
    if not isinstance(data, dict) or "features" not in data:
        unpadded = pred_result.unpad() if hasattr(pred_result, "unpad") else pred_result
        prot = unpadded.to_data().to_protein()
        total = int(prot.atom_positions.shape[0]) if num_residues is None else int(num_residues)
        out: list[dict[str, np.ndarray]] = []
        for res_idx in range(total):
            res_dict: dict[str, np.ndarray] = {}
            if res_idx < prot.atom_positions.shape[0]:
                for slot_idx, atom_name in enumerate(_AF_ATOM37_NAMES):
                    if prot.atom_mask[res_idx, slot_idx] > 0:
                        res_dict[atom_name] = np.asarray(
                            prot.atom_positions[res_idx, slot_idx], dtype=np.float64
                        )
            out.append(res_dict)
        return out

    features = data["features"]
    samples = np.asarray(data["samples"], dtype=np.float64)
    if samples.ndim == 4:
        atom_coords = samples[sample_index, 0]
    elif samples.ndim == 3:
        atom_coords = samples[0 if pred_result.is_single_sample else sample_index]
    else:
        atom_coords = samples

    atom_valid = np.asarray(pred_result.atom_pad_mask, dtype=bool).reshape(-1)
    atom_to_tok = np.asarray(pred_result.atom_to_token)
    tok_for_atom = np.argmax(atom_to_tok, axis=-1)
    valid_tokens = int(np.asarray(pred_result.token_pad_mask, dtype=bool).sum())
    target_res = valid_tokens if num_residues is None else int(num_residues)

    chars = np.asarray(features["ref_atom_name_chars"])
    if chars.ndim == 4:
        chars = chars[0]

    out = [{} for _ in range(target_res)]
    valid_atom_indices = np.flatnonzero(atom_valid)
    if valid_atom_indices.size == 0:
        return out

    valid_chars = chars[valid_atom_indices]
    has_name = valid_chars.max(axis=(-1, -2)) > 0
    char_codes = np.argmax(valid_chars, axis=-1) + 32

    for k_idx, atom_idx in enumerate(valid_atom_indices):
        if not bool(has_name[k_idx]):
            continue
        tok = int(tok_for_atom[atom_idx])
        if 0 <= tok < target_res:
            atom_name = "".join(chr(int(c)) for c in char_codes[k_idx]).strip()
            if atom_name:
                out[tok][atom_name] = atom_coords[int(atom_idx)]
    return out


def normalize_blank_chain_ids(model: gemmi.Model) -> None:
    """Normalize blank chain names in-place to ``'_'``, ``'_1'``, ``'_2'``, ..."""
    existing = {chain.name.strip() for chain in model if chain.name.strip() != ""}
    for chain in model:
        if chain.name.strip() == "":
            if "_" not in existing:
                new_name = "_"
            else:
                suffix = 1
                while f"_{suffix}" in existing:
                    suffix += 1
                new_name = f"_{suffix}"
            chain.name = new_name
            existing.add(new_name)


def write_gemmi_pdb(
    structure: gemmi.Structure,
    out_file: pathlib.Path,
    *,
    residue_b_factor_fn: Callable[[str, gemmi.Residue], float] | None = None,
) -> None:
    """Write a ``gemmi.Structure`` to PDB, optionally setting per-residue B-factors."""
    if residue_b_factor_fn is not None:
        for chain in structure[0]:
            chain_id = chain.name
            for res in chain:
                b_val = float(residue_b_factor_fn(chain_id, res))
                for atom in res:
                    atom.b_iso = b_val
    save_gemmi_pdb(structure, out_file)


def save_aligned_prediction_pdb(
    prediction: Any,
    out_path: pathlib.Path,
    *,
    sample_index: int = 0,
    rot: np.ndarray | None = None,
    trans: np.ndarray | None = None,
    res_plddt: np.ndarray | Sequence[float] | None = None,
) -> None:
    """Save an ``AF3Like`` prediction to PDB with optional rigid alignment and per-residue B-factors."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    prediction.save_pdb(out_path, sample_index=sample_index)
    if rot is None and trans is None and res_plddt is None:
        return

    st = gemmi.read_structure(str(out_path))
    rot_arr = np.eye(3, dtype=np.float64) if rot is None else np.asarray(rot, dtype=np.float64)
    trans_arr = (
        np.zeros(3, dtype=np.float64) if trans is None else np.asarray(trans, dtype=np.float64)
    )
    plddt_arr = None if res_plddt is None else np.asarray(res_plddt, dtype=np.float64).reshape(-1)

    res_idx = 0
    for chain in st[0]:
        for res in chain:
            b_val = (
                float(plddt_arr[res_idx])
                if plddt_arr is not None and res_idx < plddt_arr.shape[0]
                else None
            )
            for atom in res:
                if rot is not None or trans is not None:
                    xyz = (
                        np.array([atom.pos.x, atom.pos.y, atom.pos.z], dtype=np.float64) @ rot_arr
                        + trans_arr
                    )
                    atom.pos = gemmi.Position(float(xyz[0]), float(xyz[1]), float(xyz[2]))
                if b_val is not None:
                    atom.b_iso = b_val
            res_idx += 1
    save_gemmi_pdb(st, out_path)
