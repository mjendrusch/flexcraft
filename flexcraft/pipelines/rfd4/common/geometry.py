"""Kabsch superposition, backbone RMSD, and C2-symmetric sidechain RMSD helpers."""

from __future__ import annotations

from typing import Any, Sequence

import jax
import numpy as np

from flexcraft.structure.metrics import RMSD

BACKBONE_ATOMS: frozenset[str] = frozenset({"N", "CA", "C", "O", "OXT"})
WATERS: frozenset[str] = frozenset({"HOH", "DOD", "WAT"})
C2_SYMMETRY_SWAPS: dict[str, tuple[tuple[str, str], ...]] = {
    "ASP": (("OD1", "OD2"),),
    "GLU": (("OE1", "OE2"),),
    "PHE": (("CD1", "CD2"), ("CE1", "CE2")),
    "TYR": (("CD1", "CD2"), ("CE1", "CE2")),
    "ARG": (("NH1", "NH2"),),
}

BACKBONE_RMSD = RMSD(atoms=("N", "CA", "C", "O"))
JIT_VMAP_BACKBONE_RMSD = jax.jit(
    jax.vmap(
        lambda pred_pos, ref_pos, mask: BACKBONE_RMSD(pred_pos, ref_pos, mask=mask),
        in_axes=(0, None, None),
    )
)


def kabsch_transform(mobile: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute proper 3D rotation ``R`` and translation ``t`` aligning ``mobile`` onto ``target``."""
    mob_mean = mobile.mean(axis=0)
    tgt_mean = target.mean(axis=0)
    u_mat, _, vt_mat = np.linalg.svd((mobile - mob_mean).T @ (target - tgt_mean))
    det_sign = 1.0 if np.linalg.det(u_mat @ vt_mat) >= 0.0 else -1.0
    rot = u_mat @ np.diag([1.0, 1.0, det_sign]) @ vt_mat
    return rot, tgt_mean - mob_mean @ rot


def kabsch_rmsd(
    mobile: np.ndarray, target: np.ndarray
) -> tuple[float, np.ndarray, np.ndarray]:
    """Compute reflection-free Kabsch RMSD and transform ``mobile @ R + t -> target``."""
    mob_arr = np.asarray(mobile, dtype=np.float64)
    tgt_arr = np.asarray(target, dtype=np.float64)
    if (
        mob_arr.ndim != 2
        or tgt_arr.ndim != 2
        or mob_arr.shape != tgt_arr.shape
        or mob_arr.shape[1] != 3
        or mob_arr.shape[0] == 0
    ):
        raise ValueError(
            f"kabsch_rmsd requires matching non-empty (N, 3) arrays, got {mob_arr.shape} and {tgt_arr.shape}."
        )
    if not (np.isfinite(mob_arr).all() and np.isfinite(tgt_arr).all()):
        raise ValueError("kabsch_rmsd requires finite coordinates; received NaN or Inf.")
    rot, trans = kabsch_transform(mob_arr, tgt_arr)
    diff = mob_arr @ rot + trans - tgt_arr
    rmsd = float(np.sqrt(np.mean(np.sum(diff * diff, axis=-1))))
    return rmsd, rot, trans


def coupled_kabsch_sidechain_rmsd(
    pred_atom_dicts: Sequence[dict[str, np.ndarray]],
    motif_residues: Sequence[Any],
) -> tuple[float, np.ndarray, np.ndarray]:
    """Jointly optimize rigid-body Kabsch fit and C2-symmetric sidechain atom swaps."""
    if len(pred_atom_dicts) != len(motif_residues):
        raise ValueError(
            f"Length mismatch: {len(pred_atom_dicts)} predicted vs {len(motif_residues)} motif residues."
        )
    if not motif_residues:
        raise ValueError("coupled_kabsch_sidechain_rmsd requires >= 1 residue.")

    m_count = len(motif_residues)
    p_id_list: list[np.ndarray] = []
    p_sw_list: list[np.ndarray] = []
    q_list: list[np.ndarray] = []
    sym_indices: list[int] = []

    for m_idx, (pred, res) in enumerate(zip(pred_atom_dicts, motif_residues)):
        for atom_name in res.sc_atom_names:
            if atom_name not in pred:
                label = getattr(res, "res_token", res.res_name)
                raise ValueError(f"Missing sidechain atom {atom_name!r} for {label} ({res.res_name}).")
        q_arr = np.stack([res.atom_coords[an] for an in res.sc_atom_names], axis=0).astype(np.float64)
        p_id = np.stack([pred[an] for an in res.sc_atom_names], axis=0).astype(np.float64)
        q_list.append(q_arr)
        p_id_list.append(p_id)

        swaps = C2_SYMMETRY_SWAPS.get(res.res_name, ())
        if swaps:
            swap_map = {src: dst for a, b in swaps for src, dst in ((a, b), (b, a))}
            if all(swap_map.get(an, an) in pred for an in res.sc_atom_names):
                sym_indices.append(m_idx)
                p_sw = np.stack(
                    [pred[swap_map.get(an, an)] for an in res.sc_atom_names], axis=0
                ).astype(np.float64)
                p_sw_list.append(p_sw)
            else:
                p_sw_list.append(p_id)
        else:
            p_sw_list.append(p_id)

    target_all = np.concatenate(q_list, axis=0)
    m_sym = len(sym_indices)

    if m_sym <= 8:
        best_rmsd = float("inf")
        best_rot = np.eye(3, dtype=np.float64)
        best_trans = np.zeros(3, dtype=np.float64)
        for mask_bits in range(1 << m_sym):
            flipped = {sym_indices[k] for k in range(m_sym) if (mask_bits >> k) & 1}
            mobile_all = np.concatenate(
                [p_sw_list[m] if m in flipped else p_id_list[m] for m in range(m_count)],
                axis=0,
            )
            rmsd, rot, trans = kabsch_rmsd(mobile_all, target_all)
            if rmsd < best_rmsd:
                best_rmsd, best_rot, best_trans = rmsd, rot, trans
        return best_rmsd, best_rot, best_trans

    inv_p: list[np.ndarray] = []
    inv_q: list[np.ndarray] = []
    for pred, res in zip(pred_atom_dicts, motif_residues):
        swapped_names = {an for pair in C2_SYMMETRY_SWAPS.get(res.res_name, ()) for an in pair}
        for atom_name in res.sc_atom_names:
            if atom_name not in swapped_names:
                inv_p.append(np.asarray(pred[atom_name], dtype=np.float64))
                inv_q.append(np.asarray(res.atom_coords[atom_name], dtype=np.float64))

    need_ca_fallback = len(inv_p) < 3
    if not need_ca_fallback:
        inv_p_arr = np.stack(inv_p, axis=0)
        inv_q_arr = np.stack(inv_q, axis=0)
        s_p = np.linalg.svd(inv_p_arr - inv_p_arr.mean(axis=0), compute_uv=False)
        s_q = np.linalg.svd(inv_q_arr - inv_q_arr.mean(axis=0), compute_uv=False)
        if s_p[1] < 1e-6 or s_q[1] < 1e-6:
            need_ca_fallback = True

    if need_ca_fallback:
        for pred, res in zip(pred_atom_dicts, motif_residues):
            if "CA" in pred and "CA" in res.atom_coords:
                inv_p.append(np.asarray(pred["CA"], dtype=np.float64))
                inv_q.append(np.asarray(res.atom_coords["CA"], dtype=np.float64))

    _, rot, trans = kabsch_rmsd(np.stack(inv_p, axis=0), np.stack(inv_q, axis=0))
    flips = [False] * m_count
    rmsd = float("inf")
    for iteration in range(5):
        new_flips = [False] * m_count
        for m_idx in sym_indices:
            err_id = float(np.sum((p_id_list[m_idx] @ rot + trans - q_list[m_idx]) ** 2))
            err_sw = float(np.sum((p_sw_list[m_idx] @ rot + trans - q_list[m_idx]) ** 2))
            new_flips[m_idx] = err_sw < err_id
        mobile_all = np.concatenate(
            [p_sw_list[m] if new_flips[m] else p_id_list[m] for m in range(m_count)],
            axis=0,
        )
        rmsd, rot, trans = kabsch_rmsd(mobile_all, target_all)
        if iteration > 0 and new_flips == flips:
            break
        flips = new_flips
    return rmsd, rot, trans
