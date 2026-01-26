"""LDDT-derived metrics."""

from typing import Any

import jax
import jax.numpy as jnp

from flexcraft.data.data import DesignData

def position_lddt_ca(x: jax.Array, target: jax.Array,
                     local_distance=15.0,
                     thresholds=(0.5, 1.0, 2.0, 4.0),
                     mask=None) -> jax.Array:
    """Compute LDDT-CA given target coordinates `target` and predicted coordinates `x`.
    
    Args:
        x: predicted coordinates. float32 Array of shape ``(N, M, 3)``, where
           ``x[:, 1]`` should yield the CA coordinates of an amino acid residue.
        target: target coordinates. float32 Array of shape ``(N, M, 3)``.
        local_distance: distance threshold for LDDT in Angstroms. Default: 15.0.
        thresholds: LDDT error thresholds in Angstroms to be averaged over.
            Default: (0.5, 1.0, 2.0, 4.0).
        mask: optional mask of positions to consider for LDDT. bool Array of shape ``(N)``.
    Returns:
        float32 Array of shape `(x.shape[0],)`: , containing LDDT between 0.0 and 1.0
            for each residue.
    """
    if x.ndim == 3:
        x = x[:, 1]
    if target.ndim == 3:
        target = target[:, 1]
    if mask is None:
        mask = 1
    x_dist = jnp.linalg.norm(x[:, None] - x[None, :], axis=-1)
    target_dist = jnp.linalg.norm(target[:, None] - target[None, :], axis=-1)
    local_mask = (target_dist <= local_distance) * mask > 0
    local_count = jnp.maximum(1, local_mask.astype(jnp.int32).sum(axis=1))
    distance_difference = abs(x_dist - target_dist)
    test_thresholds = jnp.array(thresholds, dtype=jnp.float32)
    test = (distance_difference[..., None] < test_thresholds).astype(jnp.float32)
    lddt = jnp.where(local_mask, test.mean(axis=-1), 0).sum(axis=1) / local_count
    return lddt

def lddt(x: Any, target: jax.Array | DesignData) -> jax.Array:
    """Compute LDDT-CA for DesignData or compatible objects.
    
    Args:
        x: predicted structure.
            `DesignData` object with N residues, any object convertible to
            `DesignData` using a `to_data` method (e.g. `AFResult`) or
            float32 ``Array`` of shape ``N, M, 3``.
        target: reference structure.
            `DesignData` compatible object with N residues,
            or float32 position `Array` of shape ``(N, M, 3)``,
            where target[:, 1] contains reference CA positions.
    
    Returns:
        float32 Array of shape (N,): residue pLDDT
            Array containing pLDDT values for each residue in the
            range [0, 1].
    """
    mask = 1
    if hasattr(x, "to_data"):
        x = x.to_data()
    if isinstance(x, DesignData):
        if "batch_index" in x:
            batch_index = x["batch_index"]
            mask *= batch_index[:, None] == batch_index[None, :]
        x = x["atom_positions"]
    if isinstance(target, DesignData):
        if "batch_index" in target:
            batch_index = target["batch_index"]
            mask *= batch_index[:, None] == batch_index[None, :]
        target = target["atom_positions"]
    mask = mask > 0
    return position_lddt_ca(x, target, mask=mask)

def sc_lddt(x: Any, target: jax.Array | DesignData) -> jax.Array:
    """Compute self-consistent LDDT (LDDT * pLDDT) for DesignData or compatible objects.
    
    Args:
        x: predicted structure.
            `DesignData` object with N residues, any object convertible to
            `DesignData` using a `to_data` method (e.g. `AFResult`) or
            float32 ``Array`` of shape ``N, M, 3``.
        target: reference structure.
            `DesignData` compatible object with N residues,
            or float32 position `Array` of shape ``(N, M, 3)``,
            where target[:, 1] contains reference CA positions.

    Returns:
        float32 Array of shape (N,): residue scLDDT
            Array containing self-consistent LDDT values for each
            residue in the range [0, 1].
    """
    if isinstance(x, dict):
        positions = x["atom_positions"]
        plddt = x["plddt"]
    elif hasattr(x, "plddt") and hasattr(x, "atom14"):
        positions, _ = x.atom14
        plddt = x.plddt
    else:
        raise NotImplementedError(
            "Argument 0 of sc_lddt requires a dictionary or PredictorResult input.")
    comparison_lddt = lddt(positions, target)
    return plddt * comparison_lddt
