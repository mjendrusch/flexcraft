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
                Included in RMSD if eval_mask!=None.
            eval_mask: masked positions excluded from RMSD computation only.
            weight: optional per-residue weight for alignment.

        Returns:
            Root mean square deviation (RMSD) over selected atoms.
        """
        order = ["N", "CA", "C", "O", "CB"]
        # map the atoms to evaluate to index in the aa (maps to first 5 slots in atom14 format (DesignData))
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
        # broadcast shapes to no. of atoms in protein x no of eval atoms in aa, then flatten
        # converts from per residue to per atom
        mask = np.broadcast_to(mask[:, None], (x.shape[0], atom_index.shape[0])).reshape(-1)
        eval_mask = np.broadcast_to(eval_mask[:, None], (x.shape[0], atom_index.shape[0])).reshape(-1)
        index = np.broadcast_to(index[:, None], (x.shape[0], atom_index.shape[0])).reshape(-1)
        x = x[:, atom_index].reshape(-1, 3)
        y = y[:, atom_index].reshape(-1, 3)

        # kabsch alignment to find ideal euclidean transformation for minimal RMSD
        # calculates separate alignment per group in index
        params = index_kabsch(
            x, y, index, mask, weight=weight)
        # apply the transformation to align x to y
        x_p = apply_alignment(x[:, None], params)[:, 0]
        # calculate the actual RMSD
        result = jnp.sqrt(
            jnp.where(eval_mask > 0, ((y - x_p) ** 2).sum(axis=-1), 0).sum()
            / jnp.maximum(1, eval_mask.astype(jnp.int32).sum()))
        return result

def _convert_input(x):
    '''Helper to convert input to atom positions (jnp array).'''
    if hasattr(x, "to_data"):
        x = x.to_data()
        assert isinstance(x, DesignData)
    if isinstance(x, DesignData):
        x = x["atom_positions"]
    return x

class LRMSD(RMSD):
    """
    Ligand root mean square deviation (LRMSD) metric.
    Adapted from: Levine, S. et al., 2026. Origin-1: a generative AI platform for de novo antibody design against novel epitopes. https://doi.org/10.64898/2026.01.14.699389
    By default, computes LRMSD for CA atoms.
    """
    def __call__(self,
                 x,y,
                 is_target:jnp.ndarray,
                 mask=None,
                 index = None
                 ):
        '''
        Calculate the lRMSD.
        Args:
            x, y: atom5+ format positions, or DesignData objects.
            is_target: jnp.ndarray[bool] of same length as x and y, specifiying the target
            index: optional partition of the input objects into groups 
                which should be aligned separately.
            mask: masked positions excluded from alignment and RMSD computation.
                Included in RMSD if eval_mask!=None.
            

        '''
        # convert positions to jnp array
        x = _convert_input(x)
        y = _convert_input(y)
        assert len(x) == len(y), ValueError("Predicted sequence has different length from template!")
        assert len(x) == len(is_target), ValueError("is_target and x have mismatching length!")
        # receptor excluded from RMSD
        eval_mask = is_target
        # ligand excluded from alignment
        weight = ~is_target
        return super().__call__(x, y, index, mask, eval_mask, weight)