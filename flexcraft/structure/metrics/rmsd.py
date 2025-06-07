from typing import List
from dataclasses import dataclass

import numpy as np

import jax
import jax.numpy as jnp

from salad.modules.utils.geometry import index_kabsch, apply_alignment
from flexcraft.data.data import DesignData

@dataclass
class RMSD:
    atoms: List[str]
    def __call__(self, x, y, index=None, mask=None):
        order = ["N", "CA", "C", "O", "CB"]
        atom_index = np.array([
            order.index(a) for a in self.atoms], dtype=np.int32)
        if isinstance(x, DesignData):
            x = x["atom_positions"]
        if isinstance(y, DesignData):
            y = y["atom_positions"]
        x = x[:, atom_index].reshape(-1, 3)
        y = y[:, atom_index].reshape(-1, 3)
        if mask is None:
            mask = np.ones(x.shape[:1], dtype=np.bool_)
        if index is None:
            index = np.zeros(x.shape[:1], dtype=np.int32)
        params = index_kabsch(
            x, y, index, mask)
        x_p = apply_alignment(x, params)
        return jnp.sqrt(jnp.where(
            mask[:, None], (y - x_p) ** 2, 0).mean())
