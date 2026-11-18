"""LDDT-derived metrics."""

from typing import Any

import jax
import jax.numpy as jnp

from flexcraft.data.data import DesignData


def _pairwise_distances(coords: jax.Array, non_self: jax.Array) -> jax.Array:
    """Compute pairwise Euclidean distances with gradient-safe diagonal handling."""
    diff = coords[:, None, :] - coords[None, :, :]
    safe_diff = jnp.where(non_self[..., None], diff, 1.0)
    dist = jnp.linalg.norm(safe_diff, axis=-1)
    return jnp.where(non_self, dist, 0.0)


def position_lddt_ca(
    x: jax.Array,
    target: jax.Array,
    local_distance: float = 15.0,
    thresholds=(0.5, 1.0, 2.0, 4.0),
    mask=None,
) -> jax.Array:
    """Compute per-residue LDDT-CA between predicted coordinates ``x`` and ``target``.

    Args:
        x: Predicted coordinates of shape ``(N, M, 3)`` or ``(N, 3)``.
        target: Reference coordinates of shape ``(N, M, 3)`` or ``(N, 3)``.
        local_distance: Inclusion cutoff in Angstroms.
        thresholds: Distance difference thresholds in Angstroms.
        mask: Optional 1D ``(N,)`` residue mask or 2D ``(N, N)`` pair mask.

    Returns:
        Per-residue LDDT-CA array of shape ``(N,)`` in ``[0.0, 1.0]``.
    """
    if x.ndim == 3:
        x = x[:, 1]
    if target.ndim == 3:
        target = target[:, 1]

    num_residues = x.shape[0]
    non_self = ~jnp.eye(num_residues, dtype=jnp.bool_)
    valid_residue = jnp.ones((num_residues,), dtype=jnp.bool_)
    if mask is None:
        pair_mask = non_self
    else:
        mask_bool = jnp.asarray(mask) > 0
        if mask_bool.ndim == 1:
            valid_residue = mask_bool
            pair_mask = mask_bool[:, None] & mask_bool[None, :] & non_self
        else:
            pair_mask = mask_bool & non_self

    x_dist = _pairwise_distances(x, non_self)
    target_dist = _pairwise_distances(target, non_self)
    local_mask = (target_dist <= local_distance) & pair_mask
    local_count = jnp.maximum(1, local_mask.astype(jnp.int32).sum(axis=1))
    distance_difference = jnp.abs(x_dist - target_dist)
    test_thresholds = jnp.asarray(thresholds, dtype=jnp.float32)
    test = (distance_difference[..., None] < test_thresholds).astype(jnp.float32)
    per_residue = jnp.where(local_mask, test.mean(axis=-1), 0.0).sum(axis=1) / local_count
    return jnp.where(valid_residue, per_residue, 0.0)


def lddt(x: Any, target: Any) -> jax.Array:
    """Compute per-residue LDDT-CA for ``DesignData``, ``Result``, or coordinate arrays."""
    valid_mask = None
    batch_pair_mask = None
    if hasattr(x, "to_data"):
        x = x.to_data()
    if hasattr(target, "to_data"):
        target = target.to_data()

    if isinstance(x, DesignData):
        if "mask" in x:
            x_mask = jnp.asarray(x["mask"], dtype=jnp.bool_)
            valid_mask = x_mask if valid_mask is None else (valid_mask & x_mask)
        if "batch_index" in x:
            batch_index = x["batch_index"]
            same_batch = batch_index[:, None] == batch_index[None, :]
            batch_pair_mask = same_batch if batch_pair_mask is None else (batch_pair_mask & same_batch)
        x = x["atom_positions"]

    if isinstance(target, DesignData):
        if "mask" in target:
            target_mask = jnp.asarray(target["mask"], dtype=jnp.bool_)
            valid_mask = target_mask if valid_mask is None else (valid_mask & target_mask)
        if "batch_index" in target:
            batch_index = target["batch_index"]
            same_batch = batch_index[:, None] == batch_index[None, :]
            batch_pair_mask = same_batch if batch_pair_mask is None else (batch_pair_mask & same_batch)
        target = target["atom_positions"]

    num_residues = x.shape[0]
    pair_mask = ~jnp.eye(num_residues, dtype=jnp.bool_)
    if valid_mask is not None:
        pair_mask = pair_mask & valid_mask[:, None] & valid_mask[None, :]
    if batch_pair_mask is not None:
        pair_mask = pair_mask & batch_pair_mask

    result = position_lddt_ca(x, target, mask=pair_mask)
    if valid_mask is not None:
        result = jnp.where(valid_mask, result, 0.0)
    return result


def sc_lddt(x: Any, target: Any) -> jax.Array:
    """Compute self-consistent LDDT (``LDDT * pLDDT``) for ``DesignData`` or ``Result`` objects."""
    valid_mask = None
    if isinstance(x, DesignData):
        plddt = x["plddt"]
        if "mask" in x:
            valid_mask = jnp.asarray(x["mask"], dtype=jnp.bool_)
        x_for_lddt = x
    elif isinstance(x, dict):
        plddt = x["plddt"]
        if "mask" in x:
            valid_mask = jnp.asarray(x["mask"], dtype=jnp.bool_)
        x_for_lddt = DesignData.from_dict(x) if "atom_positions" in x else x["atom_positions"]
    elif hasattr(x, "to_data") and hasattr(x, "plddt"):
        x_data = x.to_data()
        plddt = x_data["plddt"] if "plddt" in x_data else x.plddt
        valid_mask = jnp.asarray(x_data["mask"], dtype=jnp.bool_) if "mask" in x_data else None
        token_mask = getattr(x, "token_pad_mask", None)
        if token_mask is not None:
            token_mask = jnp.asarray(token_mask, dtype=jnp.bool_)
            valid_mask = token_mask if valid_mask is None else (valid_mask & token_mask)
        x_for_lddt = x_data
    elif hasattr(x, "plddt") and (hasattr(x, "atom24") or hasattr(x, "atom14")):
        positions, atom_mask = x.atom24 if hasattr(x, "atom24") else x.atom14
        plddt = x.plddt
        valid_mask = jnp.asarray(atom_mask, dtype=jnp.bool_).any(axis=-1)
        token_mask = getattr(x, "token_pad_mask", None)
        if token_mask is not None:
            valid_mask = valid_mask & jnp.asarray(token_mask, dtype=jnp.bool_)
        x_for_lddt = DesignData(
            data=dict(atom_positions=positions, atom_mask=atom_mask, mask=valid_mask)
        )
    else:
        raise NotImplementedError(
            "Argument 0 of sc_lddt requires a DesignData, dict, or Result object."
        )

    if plddt.ndim == 2:
        plddt = plddt.mean(axis=0)
    comparison_lddt = lddt(x_for_lddt, target)
    result = plddt * comparison_lddt
    if valid_mask is not None:
        result = jnp.where(valid_mask, result, 0.0)
    return result
