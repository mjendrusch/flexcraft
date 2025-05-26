import os
from copy import copy
from typing import Any, List, cast, Iterable

import numpy as np

from salad.aflib.common.protein import Protein, to_pdb
from salad.aflib.model.all_atom_multimer import atom14_to_atom37, get_atom37_mask

import tree
import jax
import jax.numpy as jnp
import chex
from chex import Array

@chex.dataclass(mappable_dataclass=False)
class DesignData:
    data: dict
    @classmethod
    def from_blocks(cls, blocks: Any) -> "DesignData":
        data = dict() # TODO
        return DesignData(data=data)
    
    @classmethod
    def from_dict(cls, data: dict) -> "DesignData":
        return DesignData(data=data)

    def to_dict(self) -> dict:
        return self.data

    def cpu(self) -> "DesignData":
        return DesignData(
            data=tree.map_structure(np.array, self.data))

    @property
    def chain_index(self) -> Array:
        return self.data["chain_index"]

    @property
    def residue_index(self) -> Array:
        return self.data["residue_index"]

    @property
    def size(self) -> int:
        assert len(self.chain_index.shape) > 0
        return self.chain_index.shape[0]

    @property
    def aa(self) -> jax.Array | None:
        if "aa" in self.data:
            return self.data["aa"]
        elif "aa_profile" in self.data:
            return jnp.argmax(self.data["aa_profile"])
        elif "residue_index" in self.data:
            return jnp.full_like(self.data["residue_index"], 20)
        return None

    @property
    def aa_profile(self) -> Array | None:
        if "aa_profile" in self.data:
            return self.data["aa_profile"]
        elif "aa" in self.data:
            return jax.nn.one_hot(self.data["aa"], 20, axis=-1)
        return None

    def __getitem__(self, item_index):
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
        if not isinstance(index, (list, tuple)):
            index = [index]

        return DesignData(
            data={k: jnp.concatenate([v[c] for c in index], axis=0)
                  for k, v in self.data.items()})

    def update_index(self, index, value=None, **kwargs):
        if not isinstance(index, (list, tuple)):
            index = []

        result = self.copy()
        contig_index = jnp.concatenate([
            jnp.arange(c.start, c.stop, dtype=jnp.int32)
            for c in index], axis=-1)
        if value is not None:
            result.data = {k: v.at[contig_index].set(value) for k, v in result.data.items()}
        for k, v in kwargs.items():
            result.data[k] = v
        return result

    def drop_aa(self) -> "DesignData":
        if self.aa is not None:
            return self.update(aa=jnp.full_like(self.aa, 20))
        return self.copy()

    def copy(self) -> "DesignData":
        return DesignData(data=copy(self.data))

    def update(self, **kwargs) -> "DesignData":
        result = self.copy()
        for key, value in kwargs.items():
            result.data[key] = value
        return result

    @classmethod
    def concatenate(cls: "type[DesignData]",
                    items: Iterable["DesignData"],
                    sep_chains=True) -> "DesignData":
        result = DesignData(
            data=concatenate_dict([
                item.data for item in items], axis=0))
        if sep_chains:
            chains = []
            offset = 0
            for item in items:
                chains.append(item.chain_index + offset)
                offset += chains[-1].max()
            result.data["chain_index"] = jnp.concatenate(chains, axis=0)
        return result

    def __add__(self, other: "DesignData") -> "DesignData":
        return type(self).concatenate([self, other], sep_chains=False)

    def __truediv__(self, other: "DesignData") -> "DesignData":
        return type(self).concatenate([self, other], sep_chains=True)

    def to_protein(self, b_factors=None):
        data = tree.map_structure(lambda x: np.array(x), self.data)
        atom37 = atom14_to_atom37(data["atom_positions"], data["aa"])
        mask37 = get_atom37_mask(data["aa"])#atom14_to_atom37(data["atom_mask"], data["aa"])
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

    def save_pdb(self, path):
        directory = os.path.dirname(path)
        if directory and not os.path.isdir(directory):
            os.makedirs(directory)
        with open(path, "wt") as f:
            f.write(to_pdb(self.to_protein()))


def concatenate_dict(items: List[dict], axis=0) -> dict:
    return {k: jnp.concatenate([
        item[k] for item in items], axis=axis)
        for k in items[0]}

# # testing stuff
# def make_step(blocks):
#     def inner(data):
#         sliced = data.slice(blocks.index)
#         pos = sliced["pos"]
#         # symmetrize
#         sliced.update(pos=pos)

# def test():
#     index = slice(50, 100)
#     contigs = [slice(10, 20), slice(30, 40), slice(50, 60)]
#     x: DesignData = DesignData(dict(aa=jnp.zeros((200,), dtype=jnp.int32)))
#     # x.update(aa=x.aa.at[index].set(20))
#     sym_slice = x[contigs]["pos"]
#     # do sym op
#     sym_slices = symmetrize(sym_slice)
#     x = x.update_index(index, sym_slices)
#     if x.aa is not None:
#         x[contigs] = x[x.data["motif"] == 0].drop_aa()

# # def _check_data(data: dict) -> bool:
