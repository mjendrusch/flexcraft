import jax
import jax.numpy as jnp

class Keygen:
    """Seeded random number generator utility for jax."""
    def __init__(self, key: jnp.ndarray | int):
        self.key = jax.random.key(key) if isinstance(key, int) else key

    def __call__(self, num_keys=1) -> jnp.ndarray:
        self.key, out_key = jax.random.split(self.key, num_keys + 1)
        return out_key
