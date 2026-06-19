from typing import Any

import jax
import jax.numpy as jnp

from flexcraft.data.data import DesignData

def distogram_contacts(result: Any, target: DesignData,
                       contact_distance = 8.0) -> jax.Array:
    is_contact = target.contacts(
        contact_distance=contact_distance, atom_index = 4)
    entropy = result.contact_entropy(
        contact_distance = contact_distance)
    result = is_contact * entropy
    if is_contact.ndim == 3:
        result = result.mean(axis=0)
    return result

# TODO: radius of gyration, etc.
