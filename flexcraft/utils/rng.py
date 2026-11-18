"""Utilities for JAX pseudo-random key generation."""

import jax
import jax.numpy as jnp
import numpy as np


class Keygen:
    """Stateful PRNG key splitter for JAX."""

    def __init__(self, key: jnp.ndarray | int):
        if isinstance(key, (int, np.integer)):
            self.key = jax.random.key(int(key))
        else:
            self.key = key

    def __call__(self, num_keys: int = 1) -> jnp.ndarray:
        """Split and return ``num_keys`` fresh PRNG key(s)."""
        if num_keys < 1:
            raise ValueError(f"num_keys must be >= 1, got {num_keys}")
        keys = jax.random.split(self.key, num_keys + 1)
        self.key = keys[0]
        if num_keys == 1:
            return keys[1]
        return keys[1:]
