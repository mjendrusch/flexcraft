from typing import Any

import jax
import jax.numpy as jnp

from flexcraft.data.data import DesignData

def position_lddt_ca(x: jax.Array, target: jax.Array,
                     local_distance=15.0,
                     thresholds=(0.5, 1.0, 2.0, 4.0)) -> jax.Array:
    if x.ndim == 3:
        x = x[:, 1]
    if target.ndim == 3:
        target = target[:, 1]
    x_dist = jnp.linalg.norm(x[:, None] - x[None, :], axis=-1)
    target_dist = jnp.linalg.norm(target[:, None] - target[None, :], axis=-1)
    local_mask = target_dist <= local_distance
    local_count = jnp.maximum(1, local_mask.astype(jnp.int32).sum(axis=1))
    distance_difference = abs(x_dist - target_dist)
    test_thresholds = jnp.array(thresholds, dtype=jnp.float32)
    test = (distance_difference[..., None] < test_thresholds).astype(jnp.float32)
    lddt = jnp.where(local_mask, test.mean(axis=-1), 0).sum(axis=1) / local_count
    return lddt

def lddt(x: Any, target: jax.Array | DesignData) -> jax.Array:
    if hasattr(x, "to_data"):
        x = x.to_data()
    if isinstance(x, DesignData):
        x = x["atom_positions"]
    if isinstance(target, DesignData):
        target = target["atom_positions"]
    return position_lddt_ca(x, target)

def sc_lddt(x: Any, target: jax.Array | DesignData) -> jax.Array:
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
