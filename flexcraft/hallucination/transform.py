"""Simplex and logit transformations for sequence hallucination."""

import jax
import jax.numpy as jnp


def logp_to_simplex(x: jax.Array, scale: float = 1.0) -> jax.Array:
    """Map logits ``x`` onto the probability simplex via temperature-scaled softmax."""
    return jax.nn.softmax(x * scale, axis=-1)


def project_to_simplex(x: jax.Array, scale: float = 1.0) -> jax.Array:
    """Euclidean projection onto the scaled probability simplex along the last axis."""
    shape = x.shape
    flat = x.reshape(-1, shape[-1])
    size = flat.shape[1]
    x_sorted = jnp.sort(flat, axis=1)[:, ::-1]
    scales = jnp.ones(flat.shape[0], dtype=flat.dtype) * scale
    cssv = jnp.cumsum(x_sorted, axis=1) - scales[:, None]
    ind = jnp.arange(size, dtype=flat.dtype) + 1.0
    cond = x_sorted - cssv / ind > 0
    rho = jnp.maximum(1, jnp.count_nonzero(cond, axis=1))
    theta = cssv[jnp.arange(flat.shape[0]), rho - 1] / rho
    return jnp.maximum(flat - theta[:, None], 0.0).reshape(shape)


def forbid(
    x: jax.Array,
    indices: jax.Array | None = None,
    log_forbid: float = -6.0,
    simplex: bool = True,
) -> jax.Array:
    """Zero (and renormalize) or clamp forbidden amino acid channels."""
    if simplex:
        result = x.at[..., indices].set(0.0)
        result = result / jnp.maximum(result.sum(axis=-1, keepdims=True), 1e-3)
    else:
        result = x.at[..., indices].set(log_forbid)
    return result
