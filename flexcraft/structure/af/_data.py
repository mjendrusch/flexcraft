from copy import copy
from typing import Any, Optional

import numpy as np

import jax
import jax.numpy as jnp
import chex
from chex import Array
import tree

from colabdesign.af.alphafold.model.geometry import Vec3Array
from colabdesign.af.alphafold.model.all_atom_multimer import atom14_to_atom37, atom37_to_atom14
from colabdesign.af.prep import prep_input_features
import colabdesign.af.inputs as cd_inputs

import salad.aflib.common.protein as af_protein

from flexcraft.data.data import DesignData

@chex.dataclass(mappable_dataclass=False)
class AFInput:
    prev_init: bool = False
    pos_init: bool = False
    data: dict = None
    def __getitem__(self, name):
        return self.data[name]
    def __setitem__(self, name, value):
        self.data[name] = value
        return self.data
    def __iter__(self):
        return iter(self.data)
    def keys(self):
        return self.data.keys()
    
    def values(self):
        return self.data.values()
    
    def items(self):
        return self.data.items()
    @staticmethod
    def from_data(data: Any) -> "AFInput":
        return AFInput(
            prev_init=False,
            pos_init=False,
            data=_af_input_from_data(data))

    def add_guess(self,
                  data: DesignData | None = None,
                  pos: Array | None = None) -> "AFInput":
        if data is not None:
            positions = data["atom_positions"]
        elif pos is not None:
            positions = pos
        result = copy(self)
        result.prev_init = True
        atom_positions = atom14_to_atom37(
            positions, self.data["aatype"])
        result.data = copy(result.data)
        result.data["prev"] = copy(result.data["prev"])
        result.data["prev"]["prev_pos"] = atom_positions
        return result
    
    def add_pos(self,
                data: DesignData | None = None,
                pos: Array | None = None):
        if data is not None:
            positions = data["atom_positions"]
        elif pos is not None:
            positions = pos
        result = copy(self)
        result.pos_init = True
        result.data = copy(result.data)
        result.data["initial_atom_pos"] = atom14_to_atom37(
            positions, self.data["aatype"])
        return result

    def update_templates(self, data: Any) -> "AFInput":
        result = AFInput(
            prev_init=self.prev_init, pos_init=self.pos_init,
            data={k: v for k, v in self.data.items()})
        result["templates"] = data
        return result

    def update_sequence(self, sequence: Array) -> "AFInput":
        result = AFInput(
            prev_init=self.prev_init, pos_init=self.pos_init,
            data={k: v for k, v in self.data.items()})
        if len(sequence.shape) == 1:
            sequence = jax.nn.one_hot(sequence, 20, axis=-1)
        _update_sequence(result["data"], sequence)
        return result

    def to_data(self) -> Any:
        return ... # TODO

def _af_input_from_sequence(sequence):
    L = sequence.shape[0]
    result = prep_input_features(L=L)
    result = _update_sequence(result, sequence)
    residue_index = jnp.arange(L, dtype=jnp.int32)
    chain_index = jnp.zeros_like(residue_index)
    result["residue_index"] = residue_index
    result["asym_id"] = result["sym_id"] = chain_index
    result["entity_id"] = jnp.zeros_like(chain_index)
    prev = {'prev_msa_first_row': jnp.zeros([L, 256]),
            'prev_pair': jnp.zeros([L, L, 128]),
            'prev_pos': jnp.zeros([L, 37, 3])}
    result["prev"] = prev
    result["mask_template_interchain"] = False
    result["use_dropout"] = False
    return result

def _af_input_from_data(data: DesignData):
    seq_one_hot = jax.nn.one_hot(data.aa, 20, axis=-1)
    if "aa_one_hot" in data:
        seq_one_hot: Array = data["aa_one_hot"]
    L = seq_one_hot.shape[0]
    result = _af_input_from_sequence(seq_one_hot)
    result["residue_index"] = data["residue_index"]
    result["asym_id"] = result["sym_id"] = data["chain_index"]
    result["entity_id"] = tie_blocks(data)
    result["mask_template_interchain"] = False
    result["use_dropout"] = False
    if "tie_index" in data.data:
        result["tie_index"] = data["tie_index"]
        result["tie_weights"] = data["tie_weights"]
    return result

def tie_blocks(data):
    if "tie_blocks" in data:
        return data["tie_blocks"]
    tie_index = data["tie_index"]
    store = np.full_like(tie_index, -1)
    store[tie_index] = data["chain_index"]
    return store[tie_index]

def _update_sequence(features, one_hot):
    features = copy(features)
    cd_inputs.update_seq(one_hot[None], features)
    cd_inputs.update_aatype(jnp.argmax(one_hot, axis=-1), features)
    return features

@chex.dataclass(mappable_dataclass=False)
class AFResult:
    inputs: dict = None
    result: dict = None
    def _mean_of_binned(self, name, has_edges=True) -> jnp.ndarray:
        logits = self.result[name]["logits"]
        if has_edges:
            bin_edges = self.result[name]["bin_edges"]
            bin_step = bin_edges[1] - bin_edges[0]
            bin_edges = jnp.concatenate((bin_edges[:1] - bin_step, bin_edges, bin_edges[-1:] + bin_step), axis=0)
            bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2
        else:
            bin_centers = jnp.arange(logits.shape[-1]) / logits.shape[-1]
            bin_centers += 1 / logits.shape[-1] / 2
        return (bin_centers * jax.nn.softmax(logits)).sum(axis=-1)

    @property
    def atom14(self):
        atom37 = self.result["structure_module"]['final_atom_positions']
        mask37 = self.result["structure_module"]['final_atom_mask']
        atom14, mask14 = atom37_to_atom14(
            self.inputs["aatype"], Vec3Array.from_array(atom37), mask37)
        return atom14.to_array(), mask14

    @property
    def atom4(self) -> jnp.ndarray:
        atom14, _ = self.atom14
        return atom14[:, :4]

    @property
    def plddt(self):
        return self._mean_of_binned("predicted_lddt", has_edges=False)

    @property
    def pae(self):
        return self._mean_of_binned("predicted_aligned_error", has_edges=False)
    
    @property
    def ipae(self):
        pae = self.pae
        chain = self.inputs["asym_id"]
        other_chain = chain[:, None] != chain[None, :]
        return (pae * other_chain).sum() / jnp.maximum(1, other_chain.sum())

    @property
    def distance(self):
        return self._mean_of_binned("distogram")

    def to_data(self) -> DesignData:
        atom14, mask14 = self.atom14
        return DesignData(data=dict(
            atom_positions=atom14,
            atom_mask=mask14,
            aa=self.inputs["aatype"],
            # aa_one_hot=self.inputs["aa_one_hot"],
            mask=mask14.any(axis=1),
            residue_index=self.inputs["residue_index"],
            chain_index=self.inputs["asym_id"],
            plddt=self.plddt,
            pae=self.pae,
            distogram=self.distance
        ))

    def contact_probability(self, contact_distance=10.0) -> jnp.ndarray:
        distogram = jax.nn.softmax(self.result["distogram"]["logits"], axis=-1)
        bin_edges: jnp.ndarray = self.result["distogram"]["bin_edges"]
        bin_step = bin_edges[1] - bin_edges[0]
        bin_edges = jnp.concatenate((
            bin_edges[:1] - bin_step, bin_edges, bin_edges[-1:] + bin_step), axis=0)    
        edge_mask = bin_edges[1:] < contact_distance
        return (edge_mask * distogram).sum(axis=-1)
    
    def contact_entropy(self, contact_distance=14.0) -> jnp.ndarray:
        distogram = jax.nn.log_softmax(self.result["distogram"]["logits"], axis=-1)
        bin_edges: jnp.ndarray = self.result["distogram"]["bin_edges"]
        bin_step = bin_edges[1] - bin_edges[0]
        bin_edges = jnp.concatenate((
            bin_edges[:1] - bin_step, bin_edges, bin_edges[-1:] + bin_step), axis=0)    
        edge_mask = bin_edges[1:] < contact_distance
        distogram_clipped = jax.nn.softmax(distogram - 1e9 * (1 - edge_mask), axis=-1)
        distogram_clipped = jnp.where(edge_mask, distogram_clipped, 0)
        return -(distogram_clipped * distogram).sum(axis=-1)

    def save_pdb(self, path):
        _save_af_pdb(path, self)
        return path

def _save_af_pdb(path: str, result: AFResult):
    plddt = result.plddt
    protein = af_protein.from_prediction(
        tree.map_structure(lambda x: np.array(x), result.inputs),
        tree.map_structure(lambda x: np.array(x), result.result),
        b_factors=jnp.broadcast_to(plddt[:, None], (plddt.shape[0], 37)),
        remove_leading_feature_dimension=False)
    pdb_string = af_protein.to_pdb(protein)
    with open(path, "wt") as f:
        f.write(pdb_string)