"""DesignData container for passing protein structure and sequence across models."""

import os
from copy import copy
from typing import Dict, List

import chex
from chex import Array
import jax
import jax.numpy as jnp
import numpy as np
from salad.aflib.common.protein import Protein, to_pdb
from salad.aflib.model.all_atom_multimer import (
    atom14_to_atom37,
    atom37_to_atom14,
    get_atom14_mask,
    get_atom37_mask,
)
from salad.aflib.model.geometry import Vec3Array
from salad.modules.utils.dssp import assign_dssp
from salad.modules.utils.geometry import index_align, positions_to_ncacocb
import tree

import flexcraft.sequence.aa_codes as aas


_PER_RESIDUE_KEYS = frozenset({
    "aa",
    "aa_profile",
    "aa_one_hot",
    "logits",
    "atom_positions",
    "atom_mask",
    "residue_index",
    "chain_index",
    "batch_index",
    "tie_index",
    "tie_weights",
    "mask",
    "plddt",
    "dssp",
    "residue_type",
    "local",
    "ent",
    "var_ent",
    "order",
})

_PAIR_KEYS = frozenset({
    "pae",
    "pae_logits",
    "distogram",
    "distance",
    "pair",
    "offset",
    "contacts",
})


def _has_tracer(data: dict) -> bool:
    return any(isinstance(value, jax.core.Tracer) for value in data.values())


def _is_sample_batched_per_residue(
    key: str,
    value,
    num_residues: int,
    single_sample: bool,
) -> bool:
    """Return True when ``value`` has a leading sample axis before ``num_residues``."""
    if single_sample or not hasattr(value, "shape"):
        return False
    if key == "atom_positions":
        return value.ndim == 4 and value.shape[1] == num_residues
    if key == "atom_mask":
        return value.ndim == 3 and value.shape[1] == num_residues
    if key in ("dssp", "plddt"):
        return value.ndim == 2 and value.shape[1] == num_residues
    return False


def _is_sample_batched_pair(
    key: str,
    value,
    num_residues: int,
    single_sample: bool,
) -> bool:
    """Return True when a residue-pair tensor has a leading sample axis."""
    if single_sample or not hasattr(value, "shape") or value.ndim < 3:
        return False
    if value.shape[1] != num_residues or value.shape[2] != num_residues:
        return False
    if key in ("pae", "distance", "contacts", "offset"):
        return value.ndim == 3
    if key in ("pae_logits", "distogram", "pair"):
        return value.ndim == 4
    return value.shape[0] != num_residues


def _ensure_atom24(data: dict) -> dict:
    """Zero-pad 14-slot atom_positions and atom_mask tensors to 24 atom slots."""
    if not isinstance(data, dict):
        return data
    pos = data.get("atom_positions")
    atom_mask = data.get("atom_mask")
    needs_pos_pad = (
        isinstance(pos, (np.ndarray, jax.Array))
        and pos.ndim >= 3
        and pos.shape[-2] == 14
        and pos.shape[-1] == 3
    )
    needs_mask_pad = (
        isinstance(atom_mask, (np.ndarray, jax.Array))
        and atom_mask.ndim >= 2
        and atom_mask.shape[-1] == 14
    )
    if not (needs_pos_pad or needs_mask_pad):
        return data

    out = copy(data)
    if needs_pos_pad:
        pad_fn = np.pad if isinstance(pos, np.ndarray) else jnp.pad
        pad_width = [(0, 0)] * (pos.ndim - 2) + [(0, 10), (0, 0)]
        out["atom_positions"] = pad_fn(pos, pad_width)
    if needs_mask_pad:
        pad_fn = np.pad if isinstance(atom_mask, np.ndarray) else jnp.pad
        pad_width = [(0, 0)] * (atom_mask.ndim - 1) + [(0, 10)]
        out["atom_mask"] = pad_fn(atom_mask, pad_width)
    return out


class Block:
    def __init__(self, data, name, weight=1.0):
        self.data = data
        self.name = name
        self.weight = weight
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


@chex.dataclass(mappable_dataclass=False)
class DesignData:
    data: dict

    def __post_init__(self):
        if isinstance(self.data, dict):
            self.data = _ensure_atom24(self.data)

    @classmethod
    def from_blocks(cls, blocks: List[Block]) -> "DesignData":
        data = dict()
        return DesignData(data=data)

    @classmethod
    def from_length(cls, length: int) -> "DesignData":
        """Initialize an empty canonical atom24 DesignData object for ``length`` residues."""
        return DesignData.from_dict(dict(
            aa=jnp.full((length,), 20, dtype=jnp.int32),
            atom_positions=jnp.zeros((length, 24, 3), dtype=jnp.float32),
            atom_mask=jnp.zeros((length, 24), dtype=jnp.bool_),
            residue_index=jnp.arange(length, dtype=jnp.int32),
            chain_index=jnp.zeros((length,), dtype=jnp.int32),
            batch_index=jnp.zeros((length,), dtype=jnp.int32),
            tie_index=jnp.arange(length, dtype=jnp.int32),
            tie_weights=jnp.ones((length,), dtype=jnp.float32),
            mask=jnp.ones((length,), dtype=jnp.bool_),
        ))

    @classmethod
    def from_sequence(cls, sequence: str) -> "DesignData":
        result = []
        for subseq in sequence.split(":"):
            subchain = DesignData.from_length(len(subseq))
            subchain = subchain.update(aa=aas.encode(subseq, aas.AF2_CODE))
            result.append(subchain)
        return DesignData.concatenate(result, sep_chains=True).untie()

    @classmethod
    def from_dict(cls, data: dict) -> "DesignData":
        """Convert a dictionary into a canonical atom24 DesignData object."""
        return DesignData(data=_ensure_atom24(data))

    @classmethod
    def from_protein(cls, protein: Protein) -> "DesignData":
        """Convert an AlphaFold / aflib ``Protein`` object into canonical atom24 ``DesignData``."""
        positions, atom_mask = atom37_to_atom14(
            protein.aatype,
            Vec3Array.from_array(protein.atom_positions),
            protein.atom_mask,
        )
        return cls.from_dict(dict(
            aa=protein.aatype,
            atom_positions=positions.to_array(),
            atom_mask=atom_mask,
            residue_index=protein.residue_index,
            chain_index=protein.chain_index,
            batch_index=np.zeros_like(protein.chain_index),
            tie_index=np.arange(protein.aatype.shape[0], dtype=np.int32),
            tie_weights=np.ones((protein.aatype.shape[0],), dtype=np.float32),
            mask=atom_mask.any(axis=1),
            plddt=protein.b_factors[:, 1],
        ))

    @classmethod
    def from_salad(cls, data: dict) -> "DesignData":
        """Convert a salad dictionary into canonical atom24 ``DesignData``."""
        atom_mask = get_atom14_mask(data["aatype"])
        return cls.from_dict(dict(
            aa=data["aatype"],
            atom_positions=data["atom_pos"],
            atom_mask=atom_mask,
            residue_index=data["residue_index"],
            chain_index=data["chain_index"],
            batch_index=data["batch_index"],
            tie_index=np.arange(data["aatype"].shape[0], dtype=np.int32),
            tie_weights=np.ones((data["aatype"].shape[0],), dtype=np.float32),
            mask=data["mask"] * atom_mask.any(axis=1),
        ))

    def unpad(self) -> "DesignData":
        """Strip padded residue slots on host using ``self.data['mask']``."""
        if _has_tracer(self.data):
            raise RuntimeError(
                "DesignData.unpad() cannot be called inside jax.jit; "
                "use DesignData['mask'] inside traced functions."
            )
        if "mask" not in self.data or self.data["mask"] is None:
            return self.copy()

        valid = np.asarray(self.data["mask"], dtype=bool)
        num_residues = valid.shape[0]
        single_sample = self.is_single_sample
        unpadded = {}

        for key, value in self.data.items():
            if value is None or not hasattr(value, "shape") or value.ndim == 0:
                unpadded[key] = value
                continue

            is_pair = key in _PAIR_KEYS or (
                key not in _PER_RESIDUE_KEYS
                and value.ndim >= 2
                and value.shape[0] == num_residues
                and value.shape[1] == num_residues
            )
            if is_pair:
                if _is_sample_batched_pair(key, value, num_residues, single_sample):
                    unpadded[key] = value[:, valid][:, :, valid]
                elif value.shape[0] == num_residues and value.shape[1] == num_residues:
                    unpadded[key] = value[np.ix_(valid, valid)]
                else:
                    unpadded[key] = value
            elif _is_sample_batched_per_residue(key, value, num_residues, single_sample):
                unpadded[key] = value[:, valid]
            elif value.shape[0] == num_residues:
                unpadded[key] = value[valid]
            else:
                unpadded[key] = value

        if "tie_index" in unpadded and unpadded["tie_index"].shape[0] > 0:
            raw_tie = unpadded["tie_index"]
            _, remapped = np.unique(np.asarray(raw_tie), return_inverse=True)
            remapped = remapped.astype(np.int32)
            unpadded["tie_index"] = (
                jnp.asarray(remapped) if isinstance(raw_tie, jax.Array) else remapped
            )

        return DesignData(data=unpadded)

    def to_salad(self, augment_size: int = 15) -> dict:
        """Convert to salad dictionary format using the 14-atom slice."""
        atom14 = self["atom_positions"][..., :14, :]
        ncacocb = positions_to_ncacocb(atom14)
        converted_pos = jnp.concatenate((
            ncacocb,
            ncacocb[:, 1:2] + jnp.zeros((ncacocb.shape[0], augment_size, 3), dtype=jnp.float32),
        ), axis=1)
        return dict(
            pos=converted_pos,
            atom_pos=atom14,
            aatype=self["aa"],
            residue_index=self.residue_index,
            chain_index=self.chain_index,
            batch_index=self.batch_index,
            mask=self["mask"],
        )

    def to_dict(self) -> dict:
        """Convert a DesignData object into a dictionary."""
        return self.data

    def cpu(self) -> "DesignData":
        """Copy a DesignData object to CPU numpy arrays."""
        return DesignData(data=tree.map_structure(np.array, self.data))

    @property
    def is_single_sample(self) -> bool:
        """Return True when ``atom_positions`` contains a single sample ``(L, A, 3)``."""
        if "atom_positions" not in self.data:
            return True
        return self.data["atom_positions"].ndim == 3

    @property
    def protein(self) -> "DesignData":
        """Return the protein-only portion of a DesignData object."""
        if "residue_type" in self.data:
            return self[self.data["residue_type"] == 0]
        return self

    @property
    def batch_index(self) -> Array:
        """Return residue batch indices."""
        return self.data["batch_index"]

    @property
    def chain_index(self) -> Array:
        """Return residue chain indices."""
        return self.data["chain_index"]

    @property
    def residue_index(self) -> Array:
        """Return residue indices."""
        return self.data["residue_index"]

    @property
    def size(self) -> int:
        """Return the number of residues in this DesignData object."""
        assert len(self.chain_index.shape) > 0
        return self.chain_index.shape[0]

    @property
    def dssp(self) -> jax.Array:
        """Return integer secondary structure states ``(loop=0, helix=1, strand=2)``."""
        if "dssp" in self.data:
            return self["dssp"]
        self.assign_dssp()
        return self.data["dssp"]

    @property
    def p_dssp(self) -> Dict[str, float]:
        data = self.assign_dssp()
        one_hot = jax.nn.one_hot(data.dssp, 3)
        if one_hot.ndim == 3:
            one_hot = one_hot.mean(axis=0)
        if "mask" in data.data:
            valid = jnp.asarray(data.data["mask"], dtype=jnp.float32)
            probs = (one_hot * valid[:, None]).sum(axis=0) / jnp.maximum(1.0, valid.sum())
        else:
            probs = one_hot.mean(axis=0)
        L, H, E = probs[0], probs[1], probs[2]
        return dict(L=L, H=H, E=E)

    def assign_dssp(self) -> "DesignData":
        raw_atom_mask = (
            self.data["atom_mask"][0]
            if self.data["atom_mask"].ndim == 3
            else self.data["atom_mask"]
        )
        bb_mask = raw_atom_mask[:, :4].all(axis=1)
        if "mask" in self.data:
            bb_mask = bb_mask & jnp.asarray(self.data["mask"], dtype=jnp.bool_)
        if self.is_single_sample:
            dssp, *_ = assign_dssp(
                self.data["atom_positions"][:, :4],
                self.data["batch_index"],
                bb_mask,
            )
        else:
            dssp, *_ = jax.vmap(assign_dssp, in_axes=(0, None, None), out_axes=0)(
                self.data["atom_positions"][:, :, :4],
                self.data["batch_index"],
                bb_mask,
            )
        self.data["dssp"] = dssp
        return self

    def distance_map(self, atom_index: int = 4):
        pos = self.atom5[..., atom_index, :]
        return jnp.linalg.norm(pos[..., :, None, :] - pos[..., None, :, :], axis=-1)

    def contacts(self, contact_distance: float = 8.0, atom_index: int = 4) -> jax.Array:
        dist = self.distance_map(atom_index=atom_index)
        return dist < contact_distance

    @property
    def aa(self) -> jax.Array | None:
        """Return integer amino acid sequence (or masked ``20`` if unset)."""
        if "aa" in self.data:
            return self.data["aa"]
        if "aa_profile" in self.data:
            return jnp.argmax(self.data["aa_profile"], axis=-1)
        if "residue_index" in self.data:
            return jnp.full_like(self.data["residue_index"], 20)
        return None

    @property
    def aa_profile(self) -> Array | None:
        """Return 20-state amino acid profile if sequence or profile is present."""
        if "aa_profile" in self.data:
            return self.data["aa_profile"]
        if "aa" in self.data:
            return jax.nn.one_hot(self.data["aa"], 20, axis=-1)
        return None

    @property
    def atom5(self) -> Array | None:
        return positions_to_ncacocb(self["atom_positions"][..., :14, :])

    @property
    def cb(self) -> Array:
        return self.atom5[..., 4, :]

    @property
    def ca(self) -> Array:
        return self.atom5[..., 1, :]

    def __getitem__(self, item_index):
        """Retrieve a named field or slice all residue fields."""
        if isinstance(item_index, str):
            return self.data[item_index]
        return self.index(item_index)

    def __setitem__(self, item_index, value):
        if isinstance(item_index, str):
            self.data[item_index] = value
            if item_index in ("atom_positions", "atom_mask"):
                self.data = _ensure_atom24(self.data)
        else:
            tmp = self.update_index(item_index, value)
            self.data.update(tmp.data)

    def __iter__(self):
        return iter(self.data)

    def get(self, key: str, default=None):
        return self.data.get(key, default)

    def keys(self):
        return self.data.keys()

    def values(self):
        return self.data.values()

    def items(self):
        return self.data.items()

    def __len__(self):
        return len(self.data)

    def index(self, index) -> "DesignData":
        """Slice a DesignData object using a single index or list of indices."""
        if not isinstance(index, (list, tuple)):
            index = [index]

        single_sample = self.is_single_sample
        if single_sample:
            num_residues = (
                self.data["atom_positions"].shape[0]
                if "atom_positions" in self.data
                else next(iter(self.data.values())).shape[0]
            )
        else:
            num_residues = self.data["atom_positions"].shape[1]

        sliced_data = {}
        for key, value in self.data.items():
            is_pair = key in _PAIR_KEYS or (
                key not in _PER_RESIDUE_KEYS
                and hasattr(value, "shape")
                and value.ndim >= 2
                and value.shape[0] == num_residues
                and value.shape[1] == num_residues
            )
            if is_pair:
                if _is_sample_batched_pair(key, value, num_residues, single_sample):
                    rows = jnp.concatenate([value[:, c] for c in index], axis=1)
                    sliced_data[key] = jnp.concatenate([rows[:, :, c] for c in index], axis=2)
                elif value.shape[0] == num_residues and value.shape[1] == num_residues:
                    rows = jnp.concatenate([value[c] for c in index], axis=0)
                    sliced_data[key] = jnp.concatenate([rows[:, c] for c in index], axis=1)
                else:
                    sliced_data[key] = jnp.concatenate([value[c] for c in index], axis=0)
            elif _is_sample_batched_per_residue(key, value, num_residues, single_sample):
                sliced_data[key] = jnp.concatenate([value[:, c] for c in index], axis=1)
            else:
                sliced_data[key] = jnp.concatenate([value[c] for c in index], axis=0)

        return DesignData(data=sliced_data)

    def update_index(self, index, value=None, **kwargs) -> "DesignData":
        """Update entries at ``index`` while preserving all other keys in ``self.data``."""
        result = self.copy()
        single_sample = self.is_single_sample
        if single_sample:
            num_residues = (
                self.data["atom_positions"].shape[0]
                if "atom_positions" in self.data
                else next(iter(self.data.values())).shape[0]
            )
        else:
            num_residues = self.data["atom_positions"].shape[1]
        if value is not None:
            if isinstance(value, DesignData):
                value = value.data
            else:
                value = _ensure_atom24(value)
            for key, current in list(result.data.items()):
                if key in value:
                    arr = jnp.asarray(current)
                    if _is_sample_batched_per_residue(key, arr, num_residues, single_sample):
                        result.data[key] = arr.at[:, index].set(value[key])
                    else:
                        result.data[key] = arr.at[index].set(value[key])
        if kwargs:
            kwargs = _ensure_atom24(kwargs)
            for key, replacement in kwargs.items():
                arr = jnp.asarray(result.data[key])
                if _is_sample_batched_per_residue(key, arr, num_residues, single_sample):
                    result.data[key] = arr.at[:, index].set(replacement)
                else:
                    result.data[key] = arr.at[index].set(replacement)
        return result

    def drop_aa(self, where=True) -> "DesignData":
        """Replace amino acid identities at ``where`` with the masked token ``20``."""
        if self.aa is not None:
            return self.update(
                aa=jnp.where(where, jnp.full_like(self.aa, 20), self.aa)
            )
        return self.copy()

    def copy(self) -> "DesignData":
        return DesignData(data=copy(self.data))

    def update(self, **kwargs) -> "DesignData":
        """Return a shallow copy with updated fields."""
        result = self.copy()
        for key, value in kwargs.items():
            result.data[key] = value
        result.data = _ensure_atom24(result.data)
        return result

    @classmethod
    def concatenate(
        cls: "type[DesignData]",
        items: List["DesignData"],
        sep_chains: bool = True,
        sep_batch: bool = False,
        preserve_tie: bool = False,
    ) -> "DesignData":
        """Concatenate multiple DesignData objects along the residue axis."""
        result = DesignData(
            data=concatenate_dict([item.data for item in items], axis=0)
        )
        if sep_chains:
            result.data["chain_index"] = _sep_groups([c.chain_index for c in items])
        if sep_batch and "batch_index" in items[0].data:
            result.data["batch_index"] = _sep_groups([c["batch_index"] for c in items])
        if not preserve_tie:
            result = result.untie()
        return result

    def split(self, index=None) -> List["DesignData"]:
        """Split along ``index`` (defaulting to ``batch_index``)."""
        if index is None:
            index = self.batch_index
        index_values = np.sort(np.unique(index))
        return [self[index == value] for value in index_values]

    def untie(self) -> "DesignData":
        """Reset tie indices to independent residues."""
        return self.update(
            tie_index=jnp.arange(self["residue_index"].shape[0], dtype=jnp.int32),
            tie_weights=jnp.ones_like(self["residue_index"], dtype=jnp.float32),
        )

    def tie(self, tie_blocks) -> "DesignData":
        """Assign tie indices and weights from a list of ``Block`` objects."""
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
        """Combine fragments into a single chain with consecutive residue indices."""
        result = type(self).concatenate([self, other], sep_chains=False)
        return result.update(
            chain_index=jnp.zeros_like(result.chain_index),
            residue_index=jnp.arange(result.residue_index.shape[0], dtype=jnp.int32),
        )

    def __truediv__(self, other: "DesignData") -> "DesignData":
        """Combine into a multi-chain complex with distinct chain indices."""
        return type(self).concatenate([self, other], sep_chains=True)

    def align_to(
        self,
        other: "DesignData | Array",
        mask: Array | None = None,
        weight: Array | None = None,
    ) -> "DesignData":
        """Align coordinates to another ``DesignData`` object or coordinate array."""
        if isinstance(other, DesignData):
            other_data = other
            other = other_data["atom_positions"]
            if mask is None:
                mask = other_data["atom_mask"].any(axis=1)
                if "mask" in other_data.data:
                    mask = mask & jnp.asarray(other_data["mask"], dtype=jnp.bool_)
        if mask is None:
            mask = self["atom_mask"].any(axis=1)
        if "mask" in self.data:
            mask = jnp.asarray(mask, dtype=jnp.bool_) & jnp.asarray(self["mask"], dtype=jnp.bool_)
        return self.update(
            atom_positions=jnp.where(
                self["atom_mask"][..., None],
                index_align(
                    self["atom_positions"],
                    other,
                    jnp.zeros_like(mask, dtype=jnp.int32),
                    mask,
                    weight=weight,
                ),
                0.0,
            )
        )

    def to_protein(self, b_factors=None) -> Protein:
        """Convert to an AlphaFold ``Protein`` object (unpadding on host and slicing ``atom14``)."""
        if "mask" in self.data and not _has_tracer(self.data):
            if b_factors is not None:
                valid = np.asarray(self.data["mask"], dtype=bool)
                if not valid.all() and len(b_factors) == valid.shape[0]:
                    b_factors = np.asarray(b_factors)[valid]
            self = self.unpad()

        data = tree.map_structure(lambda x: np.array(x), self.data)
        atom14 = data["atom_positions"][:, :14]
        aa = np.clip(data["aa"], 0, 20)
        atom37 = atom14_to_atom37(atom14, aa)
        has_atom = data["mask"]
        mask37 = has_atom[:, None] * get_atom37_mask(aa)
        if b_factors is None:
            b_factors = np.ones_like(mask37, dtype=np.float32)
            if "plddt" in data:
                b_factors = b_factors * data["plddt"][:, None]
        return Protein(
            aatype=aa,
            atom_positions=atom37,
            atom_mask=mask37,
            residue_index=data["residue_index"],
            chain_index=data["chain_index"],
            b_factors=b_factors,
        )

    def to_sequence_string(self, code=aas.AF2_CODE) -> str:
        """Return ``':'``-separated amino acid sequence string across valid chains."""
        if "mask" in self.data and not _has_tracer(self.data):
            self = self.unpad()
        chains = np.unique(np.asarray(self.chain_index))
        chains = chains[chains >= 0]
        result = []
        for chain_id in chains:
            selector = self.chain_index == chain_id
            result.append(aas.decode(self.aa[selector], code=code))
        return ":".join(result)

    def to_pdb_string(self) -> str:
        """Return a PDB-formatted string for this DesignData object."""
        return to_pdb(self.to_protein())

    def save_pdb(self, path: str):
        """Save this DesignData object to ``path`` in PDB format (unpadding on host)."""
        if "mask" in self.data and not _has_tracer(self.data):
            self = self.unpad()
        directory = os.path.dirname(path)
        if directory and not os.path.isdir(directory):
            os.makedirs(directory)
        with open(path, "wt") as handle:
            handle.write(to_pdb(self.to_protein()))


def concatenate_dict(items: List[dict], axis: int = 0, drop_non_common: bool = True) -> dict:
    """Concatenate a list of array dictionaries along ``axis``."""
    return {
        key: jnp.concatenate([item[key] for item in items], axis=axis)
        for key in items[0]
        if drop_non_common and _all_have(items, key)
    }


def _all_have(items, key) -> bool:
    return all(key in item for item in items)


def _sep_groups(indices):
    result = []
    offset = 0
    for item in indices:
        result.append(item + offset)
        offset += result[-1].max() + 1
    return jnp.concatenate(result, axis=0)
