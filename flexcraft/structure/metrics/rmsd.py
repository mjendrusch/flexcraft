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

def _data_to_pos(x):
    if hasattr(x, "to_data"):
        x = x.to_data()
        assert isinstance(x, DesignData)
    if isinstance(x, DesignData):
        x = x["atom_positions"]
    return x

@dataclass
class RMSD:
    """Root mean square deviation (RMSD) metric.
    Specify the list of atom names to take RMSD over these atoms, e.g. ["N", "CA", "C"].
    Computes RMSD-CA by default, aligning over all selected atoms.
    """
    atoms: Sequence[str] = field(default_factory=_rmsd_default_atoms)
    def __call__(self, x, y, index=None, mask=None, eval_mask=None, weight=None):
        """Compute the RMSD between two inputs x and y.
        
        Args:
            x, y: atom5+ format positions, or DesignData objects.
            index: optional partition of the input objects into groups 
                which should be aligned separately.
            mask: masked positions excluded from alignment and RMSD computation.
            eval_mask: masked positions excluded from RMSD computation only.
            weight: optional per-residue weight for alignment.

        Returns:
            Root mean square deviation (RMSD) over selected atoms.
        """
        order = ["N", "CA", "C", "O", "CB"]
        atom_index = np.array([
            order.index(a) for a in self.atoms], dtype=np.int32)
        # get positions from design data objects, otherwise
        # assume that the input is atom positions
        x = _data_to_pos(x)
        y = _data_to_pos(y)
        if mask is None:
            mask = np.ones(x.shape[:1], dtype=np.bool_)
        if eval_mask is None:
            eval_mask = mask
        if index is None:
            index = np.zeros(x.shape[:1], dtype=np.int32)
        mask = np.broadcast_to(mask[:, None], (x.shape[0], atom_index.shape[0])).reshape(-1)
        eval_mask = np.broadcast_to(eval_mask[:, None], (x.shape[0], atom_index.shape[0])).reshape(-1)
        index = np.broadcast_to(index[:, None], (x.shape[0], atom_index.shape[0])).reshape(-1)
        x = x[:, atom_index].reshape(-1, 3)
        y = y[:, atom_index].reshape(-1, 3)

        params = index_kabsch(
            x, y, index, mask, weight=weight)
        x_p = apply_alignment(x[:, None], params)[:, 0]
        result = jnp.sqrt(
            jnp.where(eval_mask > 0, ((y - x_p) ** 2).sum(axis=-1), 0).sum()
            / jnp.maximum(1, eval_mask.astype(jnp.int32).sum()))
        return result
