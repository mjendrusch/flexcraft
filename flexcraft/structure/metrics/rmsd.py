"RMSD-derived metrics."

from typing import Sequence
from dataclasses import dataclass, field

import numpy as np

import jax
import jax.numpy as jnp

from salad.modules.utils.geometry import index_align, index_kabsch, apply_alignment
from flexcraft.data.data import DesignData

def _rmsd_default_atoms():
    return ["CA"]

@dataclass
class RMSD:
    """Root mean square deviation (RMSD) metric.
    Specify the list of atom names to take RMSD over these atoms, e.g. ["N", "CA", "C"].
    Computes RMSD-CA by default, aligning over all selected atoms.
    """
    atoms: Sequence[str] = field(default_factory=_rmsd_default_atoms)
    def __call__(self, x, y, index=None, mask=None):
        """Compute the RMSD between two inputs x and y.
        
        Args:
            x, y: atom5+ format positions, or DesignData objects.
            index: optional partition of the input objects into groups 
                which should be aligned separately.
            mask: masked positions excluded from RMSD computation.

        Returns:
            Root mean square deviation (RMSD) over selected atoms.
        """
        order = ["N", "CA", "C", "O", "CB"]
        atom_index = np.array([
            order.index(a) for a in self.atoms], dtype=np.int32)
        # get positions from design data objects, otherwise
        # assume that the input is atom positions
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

        # FIXME: is this right?
        params = index_kabsch(
            x, y, index, mask)
        x_p = apply_alignment(x, params)#index_align(x, y, index, mask)
        return jnp.sqrt(jnp.where(
            mask[:, None], (y - x_p) ** 2, 0).mean())
