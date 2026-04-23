"""This module implements AlphaFold2 input (AFInput) and result (AFResult) dataclasses."""

from copy import copy
import os
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
    prev_init: bool = False # wether data is recycled
    pos_init: bool = False # wether atom-positions are initialized
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
        """Convert a dictionary or DesignData object to AFInput (lossless)."""
        return AFInput(
            prev_init=False,
            pos_init=False,
            data=_af_input_from_data(data))

    @staticmethod
    def from_sequence(sequence: Any) -> "AFInput":
        '''Initialize AFInput object from one-hot encoded amino-acid-sequence.'''
        return AFInput(
            prev_init=False,
            pos_init=False,
            data=_af_input_from_sequence(sequence)
        )

    @staticmethod
    def from_chains(*chains) -> "AFInput":
        '''
        Initialize AFInput from multiple chains.

        Args:
            *chains: iterable of dict-like containing:
                "kind": kind of chain (only "protein" supported)
                and at least one of:
                "sequence": list of str, encoding amino-acids
                "length" : int, length of protein without aa-identity specification
        
        Notes:
            - The chains are first encoded as DesignData and the converted to AFInput.
        '''
        chain_info = []
        for chain in chains:
            kind = chain["kind"]
            if chain["kind"] != "protein":
                raise NotImplementedError(
                    f"AF2 can only handle protein chains, chain['kind']=='{kind}' is not supported.")
            if "sequence" in chain:
                chain_info.append(DesignData.from_sequence(chain["sequence"]))
            elif "length" in chain:
                chain_info.append(DesignData.from_length(chain["length"]))
            else:
                raise NotImplementedError(
                    f"Chain info needs either a sequence or a length.")
        result = DesignData.concatenate(chain_info, sep_chains=True)
        return AFInput.from_data(result)

    def add_guess(self,
                  data: DesignData | None = None,
                  pos: Array | None = None) -> "AFInput":
        """Add initial guess information as "prev" key to an AFInput using a
        DesignData object `data` or coordinate array `pos`."""
        if data is not None:
            positions = data["atom_positions"]
        elif pos is not None:
            positions = pos
        result = copy(self)
        result.prev_init = True
        # convert to af2 format
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
                     where: Array = True,
                     template_sidechains: Array = False,
                     mask_interchain: Array = False):
        """Add template information to an AFInput, using a DesignData object,
        or a set of coordinates `pos`, atom mask `pos_mask` and integer-encoded
        amino acid sequence `aa`.
        `where` limits the template to a subset of residues defined by a boolean mask.

        Returns:
            Result: Copy of self with added templates.

        Notes:
            - Templates are saved under 'template_*' keys.
            - add_templates can be called multiple times.
            - Templates after the first are concatenated in the last axis.
            - The original DesignData is also modified in the process, but not fully.
        """
        where = jnp.array(where, dtype=jnp.bool_)
        template_sidechains = jnp.array(template_sidechains, dtype=jnp.bool_)
        if data is not None:
            positions = data["atom_positions"]
            position_mask = data["atom_mask"]
            aa = data["aa"]
            where = where * data["atom_mask"].any(axis=1)
            template_sidechains = template_sidechains * data["atom_mask"].any(axis=1)
        else:
            if pos is not None:
                positions = self.data["atom_positions"]
                position_mask = pos_mask
                where = where * pos_mask.any(axis=1)
                template_sidechains = template_sidechains * pos_mask.any(axis=1)
            if aa is None:
                aa = self.data["aatype"]
            else:
                aa = jnp.broadcast_to(aa, self.data["aatype"].shape)
        if self.data["template_mask"].all():
            self.data.update(_add_template(self.data))
        positions = atom14_to_atom37(positions, aa)
        position_mask = atom14_to_atom37(position_mask, aa)
        atom_where = position_mask * where[:, None]
        atom_where = jnp.where(
            (template_sidechains[:, None] * atom_where),
            atom_where, atom_where.at[..., 5:].set(0))
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
        if mask_interchain:
            result.data["mask_template_interchain"] = mask_interchain
        return result

    def update_templates(self, data: Any) -> "AFInput":
        '''Replace templates feature of existing AFInput with data.'''
        result = AFInput(
            prev_init=self.prev_init, pos_init=self.pos_init,
            data={k: v for k, v in self.data.items()})
        result["templates"] = data
        return result

    def update_sequence(self, sequence: Array) -> "AFInput":
        """
        Modify the underlying sequence of an AFInput to `sequence`.

        Args:
            sequence: array of ints, either one-hot or int encoding of amino-acids
        """
        # result = AFInput(
        #     prev_init=self.prev_init, pos_init=self.pos_init,
        #     data={k: v for k, v in self.data.items()})
        if len(sequence.shape) == 1:
            sequence = jax.nn.one_hot(sequence, 20, axis=-1)
        result = AFInput(
            prev_init=self.prev_init, pos_init=self.pos_init,
            data=_update_sequence(self.data, sequence))
        return result

    def to_data(self) -> Any:
        raise NotImplementedError("This function is not implemented, yet!")
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
    result = _prep_input_features(L=L)
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


# FIXME: is this causing sharding issues? unlikely
def _prep_input_features(L, N=1, T=1, eN=1):
  '''
  given [L]ength, [N]umber of sequences and number of [T]emplates
  return dictionary of blank features
  '''
  inputs = {'aatype': jnp.zeros(L,int),
            'target_feat': jnp.zeros((L,20)),
            'msa_feat': jnp.zeros((N,L,49)),
            # 23 = one_hot -> (20, UNK, GAP, MASK)
            # 1  = has deletion
            # 1  = deletion_value
            # 23 = profile
            # 1  = deletion_mean_value
  
            'seq_mask': jnp.ones(L),
            'msa_mask': jnp.ones((N,L)),
            'msa_row_mask': jnp.ones(N),
            'atom14_atom_exists': jnp.zeros((L,14)),
            'atom37_atom_exists': jnp.zeros((L,37)),
            'residx_atom14_to_atom37': jnp.zeros((L,14),int),
            'residx_atom37_to_atom14': jnp.zeros((L,37),int),            
            'residue_index': jnp.arange(L),
            'extra_deletion_value': jnp.zeros((eN,L)),
            'extra_has_deletion': jnp.zeros((eN,L)),
            'extra_msa': jnp.zeros((eN,L),int),
            'extra_msa_mask': jnp.zeros((eN,L)),
            'extra_msa_row_mask': jnp.zeros(eN),

            # for template inputs
            'template_aatype': jnp.zeros((T,L),int),
            'template_all_atom_mask': jnp.zeros((T,L,37)),
            'template_all_atom_positions': jnp.zeros((T,L,37,3)),
            'template_mask': jnp.zeros(T),
            'template_pseudo_beta': jnp.zeros((T,L,3)),
            'template_pseudo_beta_mask': jnp.zeros((T,L)),

            # for alphafold-multimer
            'asym_id': jnp.zeros(L),
            'sym_id': jnp.zeros(L),
            'entity_id': jnp.zeros(L),
            'all_atom_positions': jnp.zeros((N,37,3))}
  return inputs

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
    residue_index_offset = jnp.concatenate((
        jnp.zeros((1,), dtype=jnp.int32),
        residue_index[1:] - residue_index[:-1]), axis=0)
    #residue_index_has_break = residue_index_breaks > 1
    chain_index_breaks = jnp.concatenate((
        jnp.zeros((1,), dtype=jnp.int32),
        chain_index[1:] != chain_index[:-1]), axis=0)
    residue_index_offset = jnp.where(chain_index_breaks > 0, chain_index_breaks * 50, residue_index_offset)
    output_index = jnp.cumsum(residue_index_offset, axis=0)
    return output_index
    # chain_break_index = jnp.concatenate((
    #     jnp.zeros((1,), dtype=jnp.int32),
    #     chain_index[1:] != chain_index[:-1]), axis=0)
    # offset = jnp.cumsum(chain_break_index, axis=0)
    # return consecutive_index + 50 * offset

@chex.dataclass(mappable_dataclass=False)
class AFResult:
    """AlphaFold result dataclass."""
    inputs: dict = None # AFInput data
    result: dict = None # raw AF2 output

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
    def residue_index(self):
        return self.inputs["residue_index"]
    
    @property
    def chain_index(self):
        return self.inputs["asym_id"]

    @property
    def local(self):
        """Local residue features."""
        return self.result["representations"]["msa_first_row"]
    
    @property
    def pair(self):
        """Residue pair features."""
        return self.result["representations"]["pair"]

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

    def ptm_matrix(self, L=None):
        """Predicted template modeling score (TM score)."""
        pae = self.pae * 32
        if L is None:
            L = pae.shape[0]
        d0 = 1.24 * (L - 15) ** (1 / 3) - 1.8
        d0 = jnp.where(L < 27, 1, d0)
        return (1 / (1 + (pae / d0) ** 2))
    
    def bptm_matrix(self, L=None):
        logits = self.result["predicted_aligned_error"]["logits"]
        probabilities = jax.nn.softmax(logits, axis=-1)
        bin_centers = jnp.arange(logits.shape[-1]) / logits.shape[-1]
        bin_centers += 1 / logits.shape[-1] / 2
        bin_centers *= 32
        if L is None:
            L = logits.shape[0]
        d0 = 1.24 * (L - 15) ** (1 / 3) - 1.8
        d0 = jnp.where(L < 27, 1, d0)
        return (probabilities * (1 / (1 + (bin_centers / d0) ** 2))).sum(axis=-1)

    def target_ptm(self, is_target, bptm=True):
        target_size = is_target.astype(jnp.int32).sum()
        if bptm:
            matrix = self.bptm_matrix(target_size)
        else:
            matrix = self.ptm_matrix(target_size)
        binder_target_mask = (~is_target[:, None]) * is_target[None, :]
        matrix = jnp.where(binder_target_mask, matrix, 0.0)
        ptm_mean = matrix.sum(axis=1) / jnp.maximum(1, binder_target_mask.sum(axis=1))
        ptm_max = ptm_mean.max(axis=0)
        return ptm_max

    @property
    def bptm(self):
        return self.bptm_matrix().mean(axis=1).max(axis=0)

    # FIXME: returns a constant value. Value should change
    @property
    def ptm(self):
        return self.ptm_matrix().mean(axis=1).max(axis=0)

    @property
    def iptm(self):
        """Interface predicted TM score."""
        ptm = self.ptm_matrix()
        chain = self.inputs["asym_id"]
        other_chain = chain[:, None] != chain[None, :]
        result = (other_chain * ptm).sum(axis=1) / jnp.maximum(other_chain.sum(axis=1), 1)
        result = result.max(axis=0)
        return result
    
    @property
    def ibptm(self):
        """Interface predicted TM score."""
        ptm = self.bptm_matrix()
        chain = self.inputs["asym_id"]
        other_chain = chain[:, None] != chain[None, :]
        result = (other_chain * ptm).sum(axis=1) / jnp.maximum(other_chain.sum(axis=1), 1)
        result = result.max(axis=0)
        return result

    def ipsae(self, pae_cutoff=0.5):
        mask = self.pae < pae_cutoff
        chain = self.inputs["asym_id"]
        other_chain = chain[:, None] != chain[None, :]
        mask *= other_chain
        L = mask.sum(axis=1)[:, None]
        ptm = mask * self.ptm_matrix(L)
        return ptm.mean(axis=1).max(axis=0)

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
            batch_index=jnp.zeros_like(self.inputs["residue_index"]),
            plddt=self.plddt,
            pae=self.pae,
            distogram=self.distance
        )).untie()

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

    def contact_score(self, contact_distance=14.0, min_resi_distance=10, num_contacts=25):
        entropy = self.contact_entropy(contact_distance=contact_distance)
        resi_dist = abs(self.residue_index[:, None] - self.residue_index[None, :])
        other_chain = self.chain_index[:, None] != self.chain_index[None, :]
        entropy = jnp.where((resi_dist >= min_resi_distance) + other_chain > 0, entropy, 1e6)
        contact_score = entropy.sort(axis=1)[:, :num_contacts].mean(axis=-1).mean()
        return contact_score

    def chain_contact_score(self, target_chain, source_chain=0, contact_distance=14.0,
                            min_resi_distance=10, num_contacts=25):
        entropy = self.contact_entropy(contact_distance=contact_distance)
        resi_dist = abs(self.residue_index[:, None] - self.residue_index[None, :])
        other_chain = self.chain_index[:, None] != self.chain_index[None, :]
        entropy = jnp.where((resi_dist >= min_resi_distance) + other_chain > 0, entropy, 1e6)
        selector = self.chain_index == target_chain
        entropy = jnp.where(selector[None, :], entropy, 1e6)
        binder_selector = ~selector
        if source_chain is not None:
            if target_chain == source_chain:
                binder_selector = selector
            else:
                binder_selector = binder_selector * (self.chain_index == source_chain)
        contact_score = (entropy.sort(axis=1)[:, :num_contacts].mean(axis=-1) * binder_selector).sum()
        contact_score /= jnp.maximum(1, binder_selector.sum())
        return contact_score

    def intra_contact_score(self, chain, contact_distance=14.0,
                           min_resi_distance=10, num_contacts=25):
        return self.chain_contact_score(
            chain, chain, contact_distance=contact_distance,
            min_resi_distance=min_resi_distance, num_contacts=num_contacts)

    def save_pdb(self, path):
        """Save an AFResult in PDB format at `path`."""
        directory = os.path.dirname(path)
        if directory and not os.path.isdir(directory):
            os.makedirs(directory)
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