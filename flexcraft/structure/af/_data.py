"""This module implements AlphaFold2 input (AFInput) and result (AFResult) dataclasses."""

from copy import copy
from typing import Any, Optional

import numpy as np

import jax
import jax.numpy as jnp
import chex
from chex import Array
import tree

from colabdesign.af.alphafold.model.geometry import Vec3Array
from colabdesign.af.alphafold.model.modules import pseudo_beta_fn
from colabdesign.af.alphafold.model.all_atom_multimer import atom14_to_atom37, atom37_to_atom14
from colabdesign.af.prep import prep_input_features
import colabdesign.af.inputs as cd_inputs

import salad.aflib.common.protein as af_protein

from flexcraft.data.data import DesignData

@chex.dataclass(mappable_dataclass=False)
class AFInput:
    """AlphaFold 2 input dataclass."""
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
        """Convert a dictionary or DesignData object to AFInput."""
        return AFInput(
            prev_init=False,
            pos_init=False,
            data=_af_input_from_data(data))

    @staticmethod
    def from_sequence(sequence: Any) -> "AFInput":
        return AFInput(
            prev_init=False,
            pos_init=False,
            data=_af_input_from_sequence(sequence)
        )

    def add_guess(self,
                  data: DesignData | None = None,
                  pos: Array | None = None) -> "AFInput":
        """Add initial guess information to an AFInput using a
        DesignData object `data` or coordinate array `pos`."""
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
        """Add structure module initial position information to
        an AFInput using a Design data object `data` or a coordinate array `pos`."""
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

    def block_diagonal(self, num_sequences=2):
        """Expand an AFInput's multiple sequence alignment features to
        block diagonal form, with a total number of sequences `num_sequences`."""
        chain = self.data["asym_id"]
        chain = jnp.unique(chain, size=num_sequences, return_inverse=True)[1]
        gap_seq = jax.nn.one_hot(jnp.full_like(chain, 21), 22)
        select = jax.nn.one_hot(chain, num_sequences, axis=0) > 0
        gap_feat = jnp.zeros_like(
            self.data["msa_feat"][0]).at[..., 0:22].set(gap_seq).at[..., 25:47].set(gap_seq)
        msa_feat = self.data["msa_feat"]
        msa_feat = jnp.where(select[..., None], msa_feat, gap_feat[None])
        result = copy(self)
        result.data = {k: v for k, v in result.data.items()}
        result.data["msa_feat"] = msa_feat
        result.data["msa_mask"] = jnp.ones((num_sequences, chain.shape[0]), dtype=jnp.bool_)
        result.data["msa_row_mask"] = jnp.ones((num_sequences,), dtype=jnp.bool_)
        return result

    def add_template(self,
                     data: DesignData | None = None,
                     pos: Array | None = None,
                     pos_mask: Array | None = None,
                     aa: Array | None = None,
                     where: Array = True):
        """Add template information to an AFInput, using a DesignData object,
        or a set of coordinates `pos`, atom mask `pos_mask` and integer-encoded
        amino acid sequence `aa`.
        `where` limits the template to a subset of residues defined by a boolean mask.
        """
        where = jnp.array(where, dtype=jnp.bool_)
        if data is not None:
            positions = data["atom_positions"]
            position_mask = data["atom_mask"]
            aa = data["aa"]
            where = where * data["atom_mask"].any(axis=1)
        else:
            if pos is not None:
                positions = self.data["atom_positions"]
                position_mask = pos_mask
            if aa is None:
                aa = self.data["aatype"]
            else:
                aa = jnp.broadcast_to(aa, self.data["aatype"].shape)
        if self.data["template_mask"].all():
            self.data.update(_add_template(self.data))
        positions = atom14_to_atom37(positions, aa)
        position_mask = atom14_to_atom37(position_mask, aa)
        atom_where = position_mask * where[:, None]
        atom_where = atom_where.at[..., 5:].set(0)
        pseudo_beta, pseudo_where = pseudo_beta_fn(aa, positions, atom_where)
        af_input = _init_template(self.data)
        update = dict(
            template_mask = af_input["template_mask"].at[-1].set(1),
            template_aatype = af_input["template_aatype"].at[-1].set(jnp.where(where, aa, 20)),
            template_all_atom_positions = af_input["template_all_atom_positions"].at[-1].set(positions),
            template_all_atom_mask = af_input["template_all_atom_mask"].at[-1].set(atom_where),
            template_pseudo_beta = af_input["template_pseudo_beta"].at[-1].set(pseudo_beta),
            template_pseudo_beta_mask = af_input["template_pseudo_beta_mask"].at[-1].set(where),
        )
        result = copy(self)
        result.pos_init = True
        result.data = copy(result.data)
        result.data.update(update)
        return result

    def update_templates(self, data: Any) -> "AFInput":
        result = AFInput(
            prev_init=self.prev_init, pos_init=self.pos_init,
            data={k: v for k, v in self.data.items()})
        result["templates"] = data
        return result

    def update_sequence(self, sequence: Array) -> "AFInput":
        """Modify the underlying sequence of an AFInput to `sequence`."""
        result = AFInput(
            prev_init=self.prev_init, pos_init=self.pos_init,
            data={k: v for k, v in self.data.items()})
        if len(sequence.shape) == 1:
            sequence = jax.nn.one_hot(sequence, 20, axis=-1)
        _update_sequence(result["data"], sequence)
        return result

    def to_data(self) -> Any:
        return ... # TODO

def _init_template(data):
    return {
        key: jnp.asarray(data[key])
        for key in (
            'template_aatype',
            'template_all_atom_mask',
            'template_all_atom_positions',
            'template_mask',
            'template_pseudo_beta',
            'template_pseudo_beta_mask'
        )
    }

def _add_template(data):
    return {
        key: _extend_array(data[key], axis=0)
        for key in (
            'template_aatype',
            'template_all_atom_mask',
            'template_all_atom_positions',
            'template_mask',
            'template_pseudo_beta',
            'template_pseudo_beta_mask'
        )
    }

def _extend_array(data: Array, axis=0):
    return jnp.concatenate((data, jnp.zeros_like(data[:1])), axis=axis)

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
    result["residue_index"] = _chain_residue_index(
        data["residue_index"], data["chain_index"])
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
    tie_index = jnp.unique(tie_index, return_inverse=True,
                           size=tie_index.shape[0])[1]
    store = jnp.full_like(tie_index, -1)
    store = store.at[tie_index].set(data["chain_index"])
    return store[tie_index]

def _update_sequence(features, one_hot):
    features = copy(features)
    cd_inputs.update_seq(one_hot[None], features)
    cd_inputs.update_aatype(jnp.argmax(one_hot, axis=-1), features)
    return features

def _chain_residue_index(residue_index, chain_index):
    consecutive_index = jnp.arange(residue_index.shape[0], dtype=jnp.int32)
    chain_break_index = jnp.concatenate((
        jnp.zeros((1,), dtype=jnp.int32),
        chain_index[1:] != chain_index[:-1]), axis=0)
    offset = jnp.cumsum(chain_break_index, axis=0)
    return consecutive_index + 50 * offset

@chex.dataclass(mappable_dataclass=False)
class AFResult:
    """AlphaFold result dataclass."""
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
        """Atom14 format atom coordinates."""
        atom37 = self.result["structure_module"]['final_atom_positions']
        mask37 = self.result["structure_module"]['final_atom_mask']
        atom14, mask14 = atom37_to_atom14(
            self.inputs["aatype"], Vec3Array.from_array(atom37), mask37)
        return atom14.to_array(), mask14

    @property
    def atom4(self) -> jnp.ndarray:
        """N, CA, C and O atom coordinates."""
        atom14, _ = self.atom14
        return atom14[:, :4]

    @property
    def plddt(self):
        """Per-residue pLDDT."""
        return self._mean_of_binned("predicted_lddt", has_edges=False)

    @property
    def pae(self):
        """Per-residue pair predicted aligned error."""
        return self._mean_of_binned("predicted_aligned_error", has_edges=False)
    
    @property
    def ipae(self):
        """Mean interface pAE."""
        pae = self.pae
        chain = self.inputs["asym_id"]
        other_chain = chain[:, None] != chain[None, :]
        return (pae * other_chain).sum() / jnp.maximum(1, other_chain.sum())

    @property
    def distance(self):
        """Distogram predicted distance of pairs of amino acid residues."""
        return self._mean_of_binned("distogram")

    def to_data(self) -> DesignData:
        """Convert an AFResult to DesignData."""
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
        """Compute the distogram predicted contact probability for each
        pair of amino acids.
        
        Args:
            contact_distance: Contact distance cutoff in Angstroms. Default: 10.0.
        """
        distogram = jax.nn.softmax(self.result["distogram"]["logits"], axis=-1)
        bin_edges: jnp.ndarray = self.result["distogram"]["bin_edges"]
        bin_step = bin_edges[1] - bin_edges[0]
        bin_edges = jnp.concatenate((
            bin_edges[:1] - bin_step, bin_edges, bin_edges[-1:] + bin_step), axis=0)    
        edge_mask = bin_edges[1:] < contact_distance
        return (edge_mask * distogram).sum(axis=-1)
    
    def contact_entropy(self, contact_distance=14.0) -> jnp.ndarray:
        """Compute the distogram contact entropy for each pair of amino acids.
        This is the metric used for optimization in BoltzDesign-1, Cho et al. 2025 (10.1101/2025.04.06.647261).
        """
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
        """Save an AFResult in PDB format at `path`."""
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