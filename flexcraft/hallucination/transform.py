import jax
import jax.numpy as jnp

def logp_to_simplex(x, scale=1.0):
    return jax.nn.softmax(x * scale, axis=-1)

def project_to_simplex(x, scale=1.0):
    shape = x.shape
    x = x.reshape(-1, x.shape[-1])
    size = x.shape[1]
    x_sorted = jnp.sort(x, axis=1)[:, ::-1]
    scale = jnp.ones(len(x)) * scale
    cssv = jnp.cumsum(x_sorted, axis=1) - scale[:, None]
    ind = jnp.arange(size) + 1
    cond = x_sorted - cssv / ind > 0
    rho = jnp.count_nonzero(cond, axis=1)
    theta = cssv[jnp.arange(len(x)), rho - 1] / rho
    return jnp.maximum(x - theta[:, None], 0)

def forbid(x: jax.Array, indices: jax.Array = None,
           log_forbid = -6, simplex = True):
    if simplex:
        result = x.at[:, indices].set(0.0)
        result /= jnp.maximum(result.sum(axis=1), 1e-3)
    else:
        result = x.at[:, indices].set(log_forbid)
    return result
