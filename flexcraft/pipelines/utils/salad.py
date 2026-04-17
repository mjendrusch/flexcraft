import jax
import jax.numpy as jnp

def bias_dssp(init_data, L=0.0, H=0.0, E=0.0, where=True):
    if not isinstance(where, jax.Array):
        where = jnp.full((init_data["pos"].shape[0],), where, dtype=jnp.bool_)
    if L > 0.0 or H > 0.0 or E > 0.0:
        init_data["dssp_mean"] = (where)[:, None] * jnp.array([
            L, H, E], dtype=jnp.float32)
        init_data["dssp_mean_mask"] = where
    return init_data
