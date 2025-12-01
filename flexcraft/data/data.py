"""This module provides the DesignData class, which is a wrapped dictionary containing
protein structure and sequence for passing information between different protein design
models."""

import os
from copy import copy
from typing import Any, List, cast, Iterable, Dict

import numpy as np

from salad.aflib.common.protein import Protein, to_pdb
from salad.modules.utils.dssp import assign_dssp
from salad.aflib.model.all_atom_multimer import atom14_to_atom37, get_atom37_mask

import tree
import jax
import jax.numpy as jnp
import chex
from chex import Array

class Block:
    def __init__(self, data, name, weight=1.0):
        self.data = data
        self.name = name
        self.weight = weight
        # TODO: develop this into full-on design blocks
        self.chain_designator = 0
        self.batch_designator = 0
        if isinstance(data, int):
            self.length = data
        elif isinstance(data, DesignData):
            self.length = data.size
        else:
            self.length = len(data)

    def __len__(self):
        return self.length

# class Assembly:
#     def __init__(self, children):
#         self.children = children

#     def __add__(self, other):
#         for sublist

@chex.dataclass(mappable_dataclass=False)
class DesignData:
    data: dict
    @classmethod
    def from_blocks(cls, blocks: List[Block]) -> "DesignData":
        data = dict() # TODO
        return DesignData(data=data)

    @classmethod
    def from_length(cls, length):
        """Initialize an empty DesignData object for a given protein length.
        
        Args:
            length (int): number of residues in the protein to be designed.
        
        Returns:
            output: DesignData object with `length` residues.
        """
        return DesignData.from_dict(dict(
            aa=jnp.full((length,), 20, dtype=jnp.int32),
            atom_positions=jnp.zeros((length, 14, 3), dtype=jnp.float32),
            atom_mask=jnp.zeros((length, 14), dtype=jnp.bool_),
            residue_index=jnp.arange(length, dtype=jnp.int32),
            chain_index=jnp.zeros((length,), dtype=jnp.int32),
            batch_index=jnp.zeros((length,), dtype=jnp.int32),
            tie_index=jnp.arange(length, dtype=jnp.int32),
            tie_weights=jnp.ones((length,), dtype=jnp.float32),
            mask=jnp.ones((length,), dtype=jnp.bool_)
        ))

    @classmethod
    def from_dict(cls, data: dict) -> "DesignData":
        """Convert a dictionary into a DesignData object."""
        return DesignData(data=data)

    def to_dict(self) -> dict:
        """Convert a DesignData object into a dictionary."""
        return self.data

    def cpu(self) -> "DesignData":
        """Copy a design data object to CPU."""
        return DesignData(
            data=tree.map_structure(np.array, self.data))

    @property
    def batch_index(self) -> Array:
        """The batch index of residues in a DesignData object."""
        return self.data["batch_index"]

    @property
    def chain_index(self) -> Array:
        """The chain index of a DesignData object."""
        return self.data["chain_index"]

    @property
    def residue_index(self) -> Array:
        """The residue index of a DesignData object."""
        return self.data["residue_index"]

    @property
    def size(self) -> int:
        """The number of residues in a DesignData object."""
        assert len(self.chain_index.shape) > 0
        return self.chain_index.shape[0]

    @property
    def dssp(self) -> jax.Array:
        """The secondary structure of a DesignData object.
        Secondary structure (loop, helix, strand) states are encoded as integers.
        (loop = 0, helix = 1, strand = 2)
        """
        if "dssp" in self.data:
            return self["dssp"]
        self.assign_dssp()
        return self.data["dssp"]

    @property
    def p_dssp(self) -> Dict[str, float]:
        data = self.assign_dssp()
        L, H, E = jax.nn.one_hot(data.dssp, 3).mean(axis=0).T
        return dict(L=L, H=H, E=E)

    def assign_dssp(self) -> "DesignData":
        result = self
        dssp, *_ = assign_dssp(
            self.data["atom_positions"][:, :4],
            self.data["batch_index"],
            self.data["atom_mask"][:, :4].all(axis=1))
        self.data["dssp"] = dssp
        return result

    @property
    def aa(self) -> jax.Array | None:
        """The amino acid sequence of a DesignData object.
        If the DesignData object has no associated sequence,
        return a fully-masked sequence.
        """
        if "aa" in self.data:
            return self.data["aa"]
        elif "aa_profile" in self.data:
            return jnp.argmax(self.data["aa_profile"])
        elif "residue_index" in self.data:
            return jnp.full_like(self.data["residue_index"], 20)
        return None

    @property
    def aa_profile(self) -> Array | None:
        """The 20-state amino acid profile of a DesignData object.
        If no amino acid sequence or profile is found, return None."""
        if "aa_profile" in self.data:
            return self.data["aa_profile"]
        elif "aa" in self.data:
            return jax.nn.one_hot(self.data["aa"], 20, axis=-1)
        return None

    def __getitem__(self, item_index):
        """Retrieve a field or slice from a DesignData object.
        
        Args:
            item_index (str | index): if `item_index` is a string, retrieve the field
                with the name `item_index` from the underlying dictionary.
                Otherwise, slice the underlying dictionary using `item_index`.

        Returns:
            The value of self.dict[item_index], if item_index is a string.
            Otherwise, returns a DataDict object with all arrays in the underlying
            dictionary indexed by `item_index`.
        """
        if isinstance(item_index, str):
            return self.data[item_index]
        return self.index(item_index)
    
    def __setitem__(self, item_index, value):
        if isinstance(item_index, str):
            self.data[item_index] = value
        else:
            tmp = self.update_index(item_index, value)
            self.data.update(tmp.data)

    def __iter__(self):
        return iter(self.data)
    
    def keys(self):
        return self.data.keys()
    
    def values(self):
        return self.data.values()
    
    def items(self):
        return self.data.items()

    def __len__(self):
        return len(self.data)

    def index(self, index):
        """Index into a design data object using a single index or a list of indices.
        
        Args:
            index: Index or list of indices for slicing a DesignData object.
        
        Returns:
            DesignData object sliced by the input index. If a list of indices was passed,
            instead contains the concatenation of DesignData objects resulting from slicing
            with each of the indices in the input list.
        """
        if not isinstance(index, (list, tuple)):
            index = [index]

        return DesignData(
            data={k: jnp.concatenate([v[c] for c in index], axis=0)
                  for k, v in self.data.items()})

    def update_index(self, index, value=None, **kwargs):
        """Update the entries of a DesignData object at a given index.
        
        Args:
            index: Index for slicing a DesignData object.
            value: Optional DesignData or dict object to replace the input DesignData at `index`.
            kwargs: Keyword arguments to replace in the output DesignData at `index`.
        
        Returns:
            DesignData object with data at `index` replaced by the contents of `value` or `kwargs`.
            When `value` and `kwargs` are both passed, the content of `kwargs` overrides the content of `value`.
        """
        # if not isinstance(index, (list, tuple)):
        #     index = [index]

        result = self.copy()
        # contig_index = jnp.concatenate([
        #     jnp.arange(c.start, c.stop, dtype=jnp.int32)
        #     for c in index], axis=-1)
        # print("CI_SHAPE", contig_index.shape, index[0].start, index[0].stop, index[0])
        # FIXME
        if value is not None:
            result.data = {k: v.at[index].set(value[k]) for k, v in result.data.items() if k in value}
        for k, v in kwargs.items():
            # print(f"V_SHAPE {k}", v.shape)
            result.data[k] = result.data[k].at[index].set(v)
        return result

    def drop_aa(self, where = True) -> "DesignData":
        """Drop amino acid sequence information from a DesignData object.
        This is a useful shorthand for preparing DesignData input for a sequence design model.
        
        Args:
            where: Boolean or boolean array which specifies at which indices in the
                   DesignData object amino acid sequence information should be dropped.
        
        Returns:
            DesignData object where amino acid sequence information was dropped at the
            indices specified by `where`. If `where` is True, all amino acid information
            will be dropped.
        """
        if self.aa is not None:
            return self.update(
                aa=jnp.where(where, jnp.full_like(self.aa, 20), self.aa))
        return self.copy()

    def copy(self) -> "DesignData":
        return DesignData(data=copy(self.data))

    def update(self, **kwargs) -> "DesignData":
        """Update entries in a DesignData object.
        
        Args:
            kwargs: keyword arguments specifying which fields should be updated.

        Returns:
            DesignData with fields specified in the keys of `kwargs` changed to the
            values of `kwargs`.
        """
        result = self.copy()
        for key, value in kwargs.items():
            result.data[key] = value
        return result

    @classmethod
    def concatenate(cls: "type[DesignData]",
                    items: List["DesignData"],
                    sep_chains=True,
                    sep_batch=False) -> "DesignData":
        """Concatenate multiple DesignData objects, optionally updating chain and batch indices.
        
        Args:
            items: list of DesignData objects to concatenate.
            sep_chains: optional boolean specifying if concatenated DesignData should have separate
                chain indices. Default: True.
            sep_batch: optional boolean specifying if concatenated DesignData should have separate
                batch indices. Default: False
        """
        result = DesignData(
            data=concatenate_dict([
                item.data for item in items], axis=0))
        if sep_chains:
            result.data["chain_index"] = _sep_groups([c.chain_index for c in items])
        if sep_batch and "batch_index" in items[0].data:
            result.data["batch_index"] = _sep_groups([c["batch_index"] for c in items])
        return result

    def split(self, index=None) -> List["DesignData"]:
        """Split along an index. If no index is provided, split by batch index."""
        if index is None:
            index = self.batch_index
        index_values = np.sort(np.unique(index))
        result = []
        for value in index_values:
            result.append(self[index == value])
        return result

    def untie(self) -> "DesignData":
        """Drop tie-index information from a DesignData object."""
        return self.update(tie_index=jnp.arange(self["tie_index"].shape[0]),
                           tie_weights=jnp.ones_like(self["tie_index"], dtype=jnp.float32))

    def tie(self, tie_blocks) -> "DesignData":
        """Add a tie index to a DesignData object based on a list of tie blocks.
        
        Args:
            tie_blocks: list of block objects.
        """
        unique_ties = dict()
        offset = 0
        tie_index = []
        tie_weights = []
        for item in tie_blocks:
            length = len(item)
            name = item.name
            if name not in unique_ties:
                unique_ties[name] = jnp.arange(offset, offset + length)
                offset += length
            tie_index.append(unique_ties[name])
            tie_weights.append(item.weight * np.ones((length,), dtype=np.float32))
        tie_index = jnp.concatenate(tie_index, axis=0)
        tie_weights = jnp.concatenate(tie_weights, axis=0)
        return self.update(tie_index=tie_index, tie_weights=tie_weights)

    def __add__(self, other: "DesignData") -> "DesignData":
        """Concatenation of chain fragments combines them into a single chain
        with consecutive residue indices."""
        result = type(self).concatenate([self, other], sep_chains=False)
        result = result.update(
            chain_index=jnp.zeros_like(result.chain_index),
            residue_index=jnp.arange(result.residue_index.shape[0], dtype=jnp.int32))
        return result

    def __truediv__(self, other: "DesignData") -> "DesignData":
        """Complex of chains with consecutive chain indices."""
        return type(self).concatenate([self, other], sep_chains=True)

    def to_protein(self, b_factors=None):
        """Turn a DesignData object into an AlphaFold2 Protein object."""
        data = tree.map_structure(lambda x: np.array(x), self.data)
        atom37 = atom14_to_atom37(data["atom_positions"], data["aa"])
        has_atom = data["mask"]
        # FIXME: drop masked atoms
        mask37 = has_atom[:, None] * get_atom37_mask(data["aa"])#atom14_to_atom37(data["atom_mask"], data["aa"])
        if b_factors is None:
            b_factors = np.ones_like(mask37, dtype=np.float32)
            if "plddt" in data:
                b_factors = b_factors * data["plddt"][:, None]
        return Protein(
            aatype=data["aa"],
            atom_positions=atom37,
            atom_mask=mask37,
            residue_index=data["residue_index"],
            chain_index=data["chain_index"],
            b_factors=b_factors)

    def to_pdb_string(self) -> str:
        """Return a PDB string corresponding to this DesignData object.
        
        Returns:
            PDB string for this DesignData object.
        """
        return to_pdb(self.to_protein())

    def save_pdb(self, path):
        """Save a DesignData object to a PDB file.
        
        Args:
            path: path to the output PDB file.
        """
        directory = os.path.dirname(path)
        if directory and not os.path.isdir(directory):
            os.makedirs(directory)
        with open(path, "wt") as f:
            f.write(to_pdb(self.to_protein()))


def concatenate_dict(items: List[dict], axis=0) -> dict:
    """Concatenate multiple dictionaries of arrays by concatenating
    arrays for each key along an axis.
    
    Args:
        items: list of dictionaries of arrays.
        axis: axis along which to concatenate arrays for each key.

    Returns:
        Dictionary containing concatenated arrays for each key in the
        list of input dictionaries.
    """
    return {k: jnp.concatenate([
        item[k] for item in items], axis=axis)
        for k in items[0]}

def _sep_groups(indices):
    result = []
    offset = 0
    for item in indices:
        result.append(item + offset)
        offset += result[-1].max() + 1
    result = jnp.concatenate(result, axis=0)
    return result
