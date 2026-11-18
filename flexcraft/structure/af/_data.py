"""AlphaFold2 input (AFInput) and result (AFResult) dataclasses."""

from copy import copy
import os
import warnings
from typing import Any, Optional

import numpy as np

import jax
import jax.numpy as jnp
import chex
from chex import Array
import tree

from flexcraft.structure._colab_af.alphafold.model.geometry import Vec3Array
from flexcraft.structure._colab_af.alphafold.model.modules import pseudo_beta_fn
from flexcraft.structure._colab_af.alphafold.model.all_atom_multimer import atom14_to_atom37, atom37_to_atom14
from flexcraft.structure._colab_af.prep import prep_input_features
import flexcraft.structure._colab_af.inputs as cd_inputs
import flexcraft.sequence.aa_codes as aas

import salad.aflib.common.protein as af_protein

from flexcraft.data.data import DesignData

# Features with residue axis at position 0: (L,) or (L, ...)
_RESIDUE_AXIS0_FEATURES = {
    "aatype", "target_feat", "seq_mask", "atom14_atom_exists",
    "atom37_atom_exists", "residx_atom14_to_atom37",
    "residx_atom37_to_atom14", "residue_index", "asym_id",
    "sym_id", "entity_id", "tie_index", "tie_weights",
    "initial_atom_pos",
}
# MSA features: (N_msa, L, ...)
_MSA_FEATURES = {"msa_feat", "msa_mask"}
# Extra MSA features: (N_extra, L, ...)
_EXTRA_MSA_FEATURES = {
    "extra_msa", "extra_msa_mask",
    "extra_deletion_value", "extra_has_deletion",
}
# Template features: (T, L, ...)
_TEMPLATE_FEATURES = {
    "template_aatype", "template_all_atom_mask",
    "template_all_atom_positions", "template_pseudo_beta",
    "template_pseudo_beta_mask",
}

# AFResult input keys known to have shape (L, ...) but NOT (L, L)
_INPUT_RESIDUE_KEYS = {
    "aatype", "target_feat", "seq_mask", "atom14_atom_exists",
    "atom37_atom_exists", "residx_atom14_to_atom37",
    "residx_atom37_to_atom14", "residue_index", "asym_id",
    "sym_id", "entity_id", "tie_index", "tie_weights",
    "initial_atom_pos",
}
# AFResult result keys with (L, ...) shapes
_RESULT_RESIDUE_KEYS = {
    "final_atom_positions", "final_atom_mask", "act",
    "msa_first_row", "single",
}
# Keys with (L, L, ...) pair shapes
_RESULT_PAIR_KEYS = {
    "pair",
}
# Dict-result sub-keys with (L, L, ...) pair shapes inside logits
_RESULT_PAIR_LOGIT_KEYS = {
    "distogram", "predicted_aligned_error",
}
# Dict-result sub-keys with (L, ...) residue shapes inside logits
_RESULT_RESIDUE_LOGIT_KEYS = {
    "predicted_lddt",
}


def _pad_feature(value, name, target_residues, num_msa=None, num_templates=None):
    """Pad a single AF2 feature array to target sizes."""
    shape = value.shape
    if len(shape) == 0:
        return value

    # Template features: (T, L, ...) — pad both template axis and residue axis
    if name in _TEMPLATE_FEATURES:
        pad_widths = [(0, 0)] * len(shape)
        current_residues = shape[1]
        if target_residues > current_residues:
            pad_widths[1] = (0, target_residues - current_residues)
        if num_templates is not None and num_templates > shape[0]:
            pad_widths[0] = (0, num_templates - shape[0])
        if any(p != (0, 0) for p in pad_widths):
            return jnp.pad(value, pad_widths)
        return value

    # template_mask: (T,) — pad template axis only
    if name == "template_mask":
        if num_templates is not None and num_templates > shape[0]:
            return jnp.pad(value, [(0, num_templates - shape[0])])
        return value

    # MSA features: (N_msa, L, ...) — pad both MSA and residue axes
    if name in _MSA_FEATURES:
        pad_widths = [(0, 0)] * len(shape)
        current_residues = shape[1]
        if target_residues > current_residues:
            pad_widths[1] = (0, target_residues - current_residues)
        if num_msa is not None and num_msa > shape[0]:
            pad_widths[0] = (0, num_msa - shape[0])
        if any(p != (0, 0) for p in pad_widths):
            return jnp.pad(value, pad_widths)
        return value

    # MSA row mask: (N_msa,) — pad MSA axis only
    if name == "msa_row_mask":
        if num_msa is not None and num_msa > shape[0]:
            return jnp.pad(value, [(0, num_msa - shape[0])])
        return value

    # Extra MSA features: (N_extra, L, ...)
    if name in _EXTRA_MSA_FEATURES:
        pad_widths = [(0, 0)] * len(shape)
        current_residues = shape[1]
        if target_residues > current_residues:
            pad_widths[1] = (0, target_residues - current_residues)
        if num_msa is not None and num_msa > shape[0]:
            pad_widths[0] = (0, num_msa - shape[0])
        if any(p != (0, 0) for p in pad_widths):
            return jnp.pad(value, pad_widths)
        return value

    # Extra MSA row mask: (N_extra,)
    if name == "extra_msa_row_mask":
        if num_msa is not None and num_msa > shape[0]:
            return jnp.pad(value, [(0, num_msa - shape[0])])
        return value

    # Residue-axis-0 features: (L, ...) or (L,)
    if name in _RESIDUE_AXIS0_FEATURES:
        current_residues = shape[0]
        if target_residues > current_residues:
            pad_widths = [(0, target_residues - current_residues)] + [(0, 0)] * (len(shape) - 1)
            padded = jnp.pad(value, pad_widths)
            if name in ("asym_id", "sym_id", "entity_id"):
                padded = padded.at[current_residues:].set(-1)
            return padded
        return value

    return value


def _pad_prev(prev, current_residues, target_residues):
    """Pad prev dict entries."""
    if target_residues <= current_residues:
        return prev
    result = {}
    diff = target_residues - current_residues
    for key, value in prev.items():
        if key == "prev_msa_first_row":
            result[key] = jnp.pad(value, [(0, diff), (0, 0)])
        elif key == "prev_pair":
            result[key] = jnp.pad(value, [(0, diff), (0, diff), (0, 0)])
        elif key == "prev_pos":
            result[key] = jnp.pad(value, [(0, diff), (0, 0), (0, 0)])
        elif key == "prev_dgram":
            result[key] = jnp.pad(value, [(0, diff), (0, diff), (0, 0)])
        else:
            result[key] = value
    return result


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

    @property
    def chain_index(self):
        return self.data["asym_id"]

    @property
    def residue_index(self):
        return self.data["residue_index"]

    @staticmethod
    def from_data(data: Any) -> "AFInput":
        """Convert a dictionary or DesignData object to AFInput."""
        return AFInput(
            prev_init=False, pos_init=False,
            data=_af_input_from_data(data))

    @staticmethod
    def from_sequence(sequence: Any) -> "AFInput":
        if isinstance(sequence, str):
            sequence = jax.nn.one_hot(aas.encode(sequence, aas.AF2_CODE), 20)
        return AFInput(
            prev_init=False, pos_init=False,
            data=_af_input_from_sequence(sequence))

    @staticmethod
    def from_chains(*chains) -> "AFInput":
        chain_info = []
        for chain in chains:
            kind = chain["kind"]
            if chain["kind"] != "protein":
                raise NotImplementedError(
                    f"AF2 can only handle protein chains, '{kind}' is not supported.")
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
        """Add initial guess. Accepts (L, 24, 3) or (L, 14, 3)."""
        if data is not None:
            positions = data["atom_positions"]
        elif pos is not None:
            positions = pos
        else:
            raise ValueError("Either data or pos must be provided.")
        if positions.shape[-2] > 14:
            positions = positions[:, :14]
        result = copy(self)
        result.prev_init = True
        atom_positions = atom14_to_atom37(positions, self.data["aatype"])
        result.data = copy(result.data)
        result.data["prev"] = copy(result.data["prev"])
        result.data["prev"]["prev_pos"] = atom_positions
        return result

    def add_pos(self,
                data: DesignData | None = None,
                pos: Array | None = None):
        """Add structure module initial position. Accepts (L, 24, 3) or (L, 14, 3)."""
        if data is not None:
            positions = data["atom_positions"]
        elif pos is not None:
            positions = pos
        else:
            raise ValueError("Either data or pos must be provided.")
        if positions.shape[-2] > 14:
            positions = positions[:, :14]
        result = copy(self)
        result.pos_init = True
        result.data = copy(result.data)
        result.data["initial_atom_pos"] = atom14_to_atom37(
            positions, self.data["aatype"])
        return result

    def block_diagonal(self, num_sequences=2):
        """Expand MSA features to block diagonal form."""
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
        """Add template information. Accepts atom24 or atom14 format."""
        where = jnp.array(where, dtype=jnp.bool_)
        template_sidechains = jnp.array(template_sidechains, dtype=jnp.bool_)
        if data is not None:
            positions = data["atom_positions"]
            if positions.shape[-2] > 14:
                positions = positions[:, :14]
            position_mask = data["atom_mask"]
            if position_mask.shape[-1] > 14:
                position_mask = position_mask[:, :14]
            aa = data["aa"]
            where = where * position_mask.any(axis=1)
            template_sidechains = template_sidechains * position_mask.any(axis=1)
        else:
            if pos is not None:
                positions = pos
                if positions.shape[-2] > 14:
                    positions = positions[:, :14]
                if pos_mask is not None:
                    position_mask = pos_mask
                    if position_mask.shape[-1] > 14:
                        position_mask = position_mask[:, :14]
                else:
                    position_mask = jnp.ones(positions.shape[:-1], dtype=jnp.float32)
                where = where * position_mask.any(axis=1)
                template_sidechains = template_sidechains * position_mask.any(axis=1)
            else:
                raise ValueError("Either data or pos must be provided.")
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
        result = AFInput(
            prev_init=self.prev_init, pos_init=self.pos_init,
            data={k: v for k, v in self.data.items()})
        result["templates"] = data
        return result

    def update_sequence(self, sequence: Array) -> "AFInput":
        """Modify the underlying sequence of an AFInput to `sequence`."""
        if len(sequence.shape) == 1:
            sequence = jax.nn.one_hot(sequence, 20, axis=-1)
        L_padded = self.data["aatype"].shape[0]
        L_seq = sequence.shape[0]
        if L_seq < L_padded:
            # Zero-pad sequence to match padded length
            sequence = jnp.pad(sequence, [(0, L_padded - L_seq), (0, 0)])
        # Always mask by seq_mask so padded positions stay zeroed
        seq_mask = self.data["seq_mask"]
        sequence = sequence * seq_mask[:, None]
        result = AFInput(
            prev_init=self.prev_init, pos_init=self.pos_init,
            data=_update_sequence(self.data, sequence))
        return result

    def pad_to(self, size: int | None,
               num_residue_atoms: int = 24, *,
               num_msa: int | None = None,
               num_templates: int | None = None) -> "AFInput":
        """Pad AFInput to target residue count with monotonic right-padding.

        Args:
            size: Target number of residues. Warns and no-ops if less than current.
            num_residue_atoms: Accepted for API compatibility, ignored.
            num_msa: Target MSA depth.
            num_templates: Target template count.
        """
        if self.data is None:
            return self
        current_residues = self.data["aatype"].shape[0]

        if size is not None and size < current_residues:
            warnings.warn(
                f"pad_to({size}) < current residue count ({current_residues}), "
                f"keeping current size.")
            size = current_residues

        target_residues = size if size is not None else current_residues

        if num_msa is not None:
            current_msa = self.data["msa_feat"].shape[0]
            if num_msa < current_msa:
                warnings.warn(
                    f"num_msa={num_msa} < current MSA depth ({current_msa}), "
                    f"keeping current depth.")
                num_msa = None

        if num_templates is not None:
            current_templates = self.data["template_mask"].shape[0]
            if num_templates < current_templates:
                warnings.warn(
                    f"num_templates={num_templates} < current template count "
                    f"({current_templates}), keeping current count.")
                num_templates = None

        if (target_residues == current_residues
                and num_msa is None and num_templates is None):
            return self

        result = copy(self)
        result.data = {}
        for key, value in self.data.items():
            if key == "prev":
                result.data[key] = _pad_prev(value, current_residues, target_residues)
            elif isinstance(value, (jnp.ndarray, np.ndarray)):
                result.data[key] = _pad_feature(
                    jnp.asarray(value), key, target_residues,
                    num_msa=num_msa, num_templates=num_templates)
            else:
                result.data[key] = value
        return result

    def pad_to_nearest_multiple(self, multiple: int,
                                num_residue_atoms: int = 24, *,
                                num_msa: int | None = None,
                                num_templates: int | None = None) -> "AFInput":
        """Pad to the nearest multiple of `multiple`."""
        if self.data is None:
            return self
        current_residues = self.data["aatype"].shape[0]
        target = ((current_residues + multiple - 1) // multiple) * multiple
        return self.pad_to(target, num_residue_atoms=num_residue_atoms,
                           num_msa=num_msa, num_templates=num_templates)

    def pad_atoms(self, num_residue_atoms: int = 24) -> "AFInput":
        """Validate atom slot request for AF2 (fixed 14/37 atom slots per residue)."""
        if num_residue_atoms <= 0:
            raise ValueError(
                f"num_residue_atoms must be positive, got {num_residue_atoms}."
            )
        if num_residue_atoms < 14:
            warnings.warn(
                f"num_residue_atoms={num_residue_atoms} < 14; AF2 uses fixed "
                "14/37 atom slots per residue.",
                UserWarning,
                stacklevel=2,
            )
        return self

    def to_data(self) -> Any:
        return ... # TODO

def _init_template(data):
    return {
        key: jnp.asarray(data[key])
        for key in (
            'template_aatype', 'template_all_atom_mask',
            'template_all_atom_positions', 'template_mask',
            'template_pseudo_beta', 'template_pseudo_beta_mask'
        )
    }

def _add_template(data):
    return {
        key: _extend_array(data[key], axis=0)
        for key in (
            'template_aatype', 'template_all_atom_mask',
            'template_all_atom_positions', 'template_mask',
            'template_pseudo_beta', 'template_pseudo_beta_mask'
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


def _prep_input_features(L, N=1, T=1, eN=1):
  inputs = {'aatype': jnp.zeros(L,int),
            'target_feat': jnp.zeros((L,20)),
            'msa_feat': jnp.zeros((N,L,49)),
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
            'template_aatype': jnp.zeros((T,L),int),
            'template_all_atom_mask': jnp.zeros((T,L,37)),
            'template_all_atom_positions': jnp.zeros((T,L,37,3)),
            'template_mask': jnp.zeros(T),
            'template_pseudo_beta': jnp.zeros((T,L,3)),
            'template_pseudo_beta_mask': jnp.zeros((T,L)),
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
    residue_index_offset = jnp.concatenate((
        jnp.zeros((1,), dtype=jnp.int32),
        residue_index[1:] - residue_index[:-1]), axis=0)
    chain_index_breaks = jnp.concatenate((
        jnp.zeros((1,), dtype=jnp.int32),
        chain_index[1:] != chain_index[:-1]), axis=0)
    residue_index_offset = jnp.where(
        chain_index_breaks > 0, chain_index_breaks * 50, residue_index_offset)
    return jnp.cumsum(residue_index_offset, axis=0)


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
            bin_edges = jnp.concatenate(
                (bin_edges[:1] - bin_step, bin_edges, bin_edges[-1:] + bin_step), axis=0)
            bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2
        else:
            bin_centers = jnp.arange(logits.shape[-1]) / logits.shape[-1]
            bin_centers += 1 / logits.shape[-1] / 2
        return (bin_centers * jax.nn.softmax(logits)).sum(axis=-1)

    @property
    def admits_samples(self):
        return False

    @property
    def is_single_sample(self):
        return True

    @property
    def residue_index(self):
        return self.inputs["residue_index"]

    @property
    def chain_index(self):
        return self.inputs["asym_id"]

    @property
    def token_pad_mask(self):
        """1D (L,) bool mask: True for valid (non-padded) tokens."""
        return jnp.asarray(self.inputs["seq_mask"]) > 0

    @property
    def atom_pad_mask(self):
        """Same as token_pad_mask for AF2 (fixed atom slots per residue)."""
        return self.token_pad_mask

    @property
    def _masked_chain_index(self):
        """Chain index with padded tokens set to -1."""
        raw_chain = self.inputs.get("chain_index", self.inputs["asym_id"])
        return jnp.where(self.token_pad_mask, raw_chain, -1)

    @property
    def local(self):
        return self.result["representations"]["msa_first_row"]

    @property
    def pair(self):
        return self.result["representations"]["pair"]

    @property
    def atom14(self):
        """Atom14 format atom coordinates. Returns (coords, mask)."""
        atom37 = self.result["structure_module"]['final_atom_positions']
        mask37 = self.result["structure_module"]['final_atom_mask']
        atom14, mask14 = atom37_to_atom14(
            self.inputs["aatype"], Vec3Array.from_array(atom37), mask37)
        return atom14.to_array(), mask14

    @property
    def atom24(self):
        """Atom24 format coordinates zero-padded from atom14."""
        coords14, mask14 = self.atom14
        coords24 = jnp.pad(coords14, [(0, 0), (0, 10), (0, 0)])
        mask24 = jnp.pad(mask14, [(0, 0), (0, 10)])
        return coords24, mask24

    @property
    def atom24_samples(self):
        """Single-sample alias returning atom24 (coords, mask) for AF3Like parity."""
        return self.atom24

    @property
    def atom4(self) -> jnp.ndarray:
        atom14, _ = self.atom14
        return atom14[:, :4]

    @property
    def plddt(self):
        return self._mean_of_binned("predicted_lddt", has_edges=False)

    def index_plddt(self, mask=None):
        """Masked mean pLDDT over active residues."""
        valid = self.token_pad_mask
        if mask is not None:
            valid = valid & jnp.asarray(mask, dtype=jnp.bool_)
        return (self.plddt * valid).sum(axis=-1) / jnp.maximum(1, valid.sum())

    @property
    def mean_plddt(self):
        """Masked mean pLDDT."""
        return self.index_plddt()

    @property
    def pae(self):
        return self._mean_of_binned("predicted_aligned_error", has_edges=False)

    def index_pae(self, mask=None):
        """Masked mean pAE over an optional residue subset."""
        valid = self.token_pad_mask
        if mask is not None:
            valid = valid & jnp.asarray(mask, dtype=jnp.bool_)
        pair_mask = valid[:, None] & valid[None, :]
        return (self.pae * pair_mask).sum(axis=(-1, -2)) / jnp.maximum(1, pair_mask.sum())

    @property
    def mean_pae(self):
        """Masked mean pAE."""
        return self.index_pae()

    @property
    def pae_logits(self):
        return self.result["predicted_aligned_error"]["logits"]

    @property
    def pae_bin_edges(self):
        pae_dict = self.result["predicted_aligned_error"]
        if "breaks" in pae_dict:
            breaks = pae_dict["breaks"]
            step = breaks[1] - breaks[0]
            return jnp.concatenate(
                [breaks, breaks[-1:] + step, breaks[-1:] + 2 * step], axis=0
            )
        if "bin_edges" in pae_dict:
            edges = pae_dict["bin_edges"]
            step = edges[1] - edges[0]
            return jnp.concatenate(
                [edges[:1] - step, edges, edges[-1:] + step], axis=0
            )
        return jnp.linspace(0.0, 32.0, self.pae_logits.shape[-1] + 1)

    @property
    def pae_bin_centers(self):
        return 0.5 * (self.pae_bin_edges[:-1] + self.pae_bin_edges[1:])

    @property
    def max_pae(self):
        return self.pae_bin_edges[-1]

    # --- Confidence metrics delegated to flexcraft.structure.metrics ---

    def ptm_matrix(self, L=None):
        """pTM matrix from pae_logits (bin-level expectation)."""
        from flexcraft.structure.metrics.confidence import ptm_matrix_from_logits
        return ptm_matrix_from_logits(
            self.pae_logits,
            num_residues=L,
            token_mask=self.token_pad_mask,
            max_pae=self.max_pae,
            bin_centers=self.pae_bin_centers,
        )

    def bptm_matrix(self, L=None):
        """Alias for ptm_matrix."""
        return self.ptm_matrix(L=L)

    def index_ptm(self, mask=None):
        """Complex or subcomplex masked pTM."""
        from flexcraft.structure.metrics.confidence import compute_ptm
        return compute_ptm(
            self.pae_logits,
            mask=mask,
            token_mask=self.token_pad_mask,
            max_pae=self.max_pae,
            bin_centers=self.pae_bin_centers,
        )

    @property
    def ptm(self):
        """Complex-wide masked pTM."""
        return self.index_ptm()

    @property
    def bptm(self):
        return self.ptm

    def index_iptm(self, chain_index=None):
        """Interface predicted TM score (masked)."""
        from flexcraft.structure.metrics.confidence import compute_iptm
        ci = (
            self._masked_chain_index
            if chain_index is None
            else jnp.where(self.token_pad_mask, chain_index, -1)
        )
        return compute_iptm(
            self.pae_logits,
            ci,
            token_mask=self.token_pad_mask,
            max_pae=self.max_pae,
            bin_centers=self.pae_bin_centers,
        )

    @property
    def iptm(self):
        """Interface predicted TM score (masked)."""
        return self.index_iptm()

    @property
    def ibptm(self):
        return self.iptm

    def index_ipae(self, chain_index=None):
        """Mean interface pAE (masked)."""
        from flexcraft.structure.metrics.confidence import compute_ipae
        ci = self._masked_chain_index if chain_index is None else chain_index
        return compute_ipae(
            self.pae,
            ci,
            token_mask=self.token_pad_mask,
        )

    @property
    def ipae(self):
        """Mean interface pAE (masked)."""
        return self.index_ipae()

    def index_min_ipae(self, chain_index=None):
        """Minimum interface pAE (masked)."""
        from flexcraft.structure.metrics.confidence import compute_min_ipae
        ci = self._masked_chain_index if chain_index is None else chain_index
        return compute_min_ipae(
            self.pae,
            ci,
            token_mask=self.token_pad_mask,
        )

    @property
    def min_ipae(self):
        """Minimum interface pAE (masked)."""
        return self.index_min_ipae()

    def chain_selector(self, target_chain, source_chain=None):
        """Select (source, target) block using AND logic."""
        from flexcraft.structure.metrics.confidence import chain_pair_mask
        return chain_pair_mask(
            self._masked_chain_index, target_chain, source_chain,
            token_mask=self.token_pad_mask)

    def chain_pae(self, target_chain, source_chain=None):
        """Masked chain-pair pAE."""
        from flexcraft.structure.metrics.confidence import compute_chain_pae
        return compute_chain_pae(
            self.pae, self._masked_chain_index,
            target_chain, source_chain,
            token_mask=self.token_pad_mask)

    def chain_ptm(self, target_chain, source_chain=None):
        """Masked chain-pair pTM."""
        from flexcraft.structure.metrics.confidence import compute_chain_ptm
        return compute_chain_ptm(
            self.pae_logits, self._masked_chain_index,
            target_chain, source_chain,
            token_mask=self.token_pad_mask,
            max_pae=self.max_pae,
            bin_centers=self.pae_bin_centers,
        )

    def index_ipsae(self, chain_index=None, raw_pae_threshold=15.0, pae_cutoff=None):
        """Interface pSAE (masked)."""
        from flexcraft.structure.metrics.confidence import compute_ipsae
        ci = self._masked_chain_index if chain_index is None else chain_index
        threshold = (
            pae_cutoff * self.max_pae
            if pae_cutoff is not None
            else raw_pae_threshold
        )
        return compute_ipsae(
            self.pae_logits,
            self.pae,
            ci,
            token_mask=self.token_pad_mask,
            raw_pae_threshold=threshold,
            max_pae=self.max_pae,
            bin_centers=self.pae_bin_centers,
            normalized_pae=True,
        )

    def ipsae(self, chain_index=None, raw_pae_threshold=15.0, pae_cutoff=None):
        """Interface pSAE (masked)."""
        return self.index_ipsae(
            chain_index=chain_index,
            raw_pae_threshold=raw_pae_threshold,
            pae_cutoff=pae_cutoff,
        )

    def target_ptm(self, is_target, bptm=True):
        token_mask = self.token_pad_mask
        is_target_valid = jnp.asarray(is_target, dtype=jnp.bool_) & token_mask
        target_size = is_target_valid.astype(jnp.int32).sum()
        matrix = self.ptm_matrix(target_size)
        binder_target_mask = (~is_target_valid[:, None] & token_mask[:, None]) * is_target_valid[None, :]
        matrix = jnp.where(binder_target_mask, matrix, 0.0)
        ptm_mean = matrix.sum(axis=1) / jnp.maximum(1, binder_target_mask.sum(axis=1))
        return ptm_mean.max(axis=0)

    def index_ptm_score(self, chain_index=None):
        """Log-sum-exp interface pTM score (masked)."""
        from flexcraft.structure.metrics.confidence import compute_ptm_score
        ci = self._masked_chain_index if chain_index is None else chain_index
        return compute_ptm_score(
            self.pae_logits,
            ci,
            token_mask=self.token_pad_mask,
            max_pae=self.max_pae,
            bin_centers=self.pae_bin_centers,
        )

    @property
    def ptm_score(self):
        """Log-sum-exp interface pTM score (masked)."""
        return self.index_ptm_score()

    # --- Distogram properties ---

    @property
    def distance(self):
        return self._mean_of_binned("distogram")

    @property
    def distogram_mean(self):
        return self.distance

    @property
    def distogram(self):
        return jax.nn.softmax(self.result["distogram"]["logits"], axis=-1)

    @property
    def log_distogram(self):
        return jax.nn.log_softmax(self.result["distogram"]["logits"], axis=-1)

    @property
    def distogram_bin_edges(self):
        bin_edges = self.result["distogram"]["bin_edges"]
        bin_step = bin_edges[1] - bin_edges[0]
        return jnp.concatenate(
            (bin_edges[:1] - bin_step, bin_edges, bin_edges[-1:] + bin_step), axis=0)

    @property
    def distogram_bin_centers(self):
        return (self.distogram_bin_edges[1:] + self.distogram_bin_edges[:-1]) / 2

    # --- Output conversion ---

    def to_data(self, return_samples=False) -> DesignData:
        """Convert to DesignData. Emits atom24 (L, 24, 3) with padding masks."""
        coords24, mask24 = self.atom24
        _, mask14 = self.atom14
        token_mask = self.token_pad_mask
        residue_mask = mask14.any(axis=1) * token_mask
        raw_chain = self.inputs.get("chain_index", self.inputs["asym_id"])
        masked_chain = jnp.where(token_mask, raw_chain, -1)
        return DesignData(data=dict(
            atom_positions=coords24,
            atom_mask=mask24,
            aa=self.inputs["aatype"],
            mask=residue_mask,
            residue_index=self.inputs["residue_index"],
            chain_index=masked_chain,
            batch_index=jnp.zeros_like(self.inputs["residue_index"]),
            plddt=self.plddt,
            pae=self.pae,
            distogram=self.distance
        )).untie()

    def unpad(self) -> "AFResult":
        """Remove padding. Host-only (raises if traced)."""
        mask = self.token_pad_mask
        if isinstance(mask, jax.core.Tracer):
            raise RuntimeError(
                "AFResult.unpad() cannot be called inside jax.jit.")
        n_valid = int(mask.sum())
        L = mask.shape[0]
        if n_valid == L:
            return self

        # Slice inputs using explicit key-based logic
        new_inputs = {}
        for key, value in self.inputs.items():
            if key == "prev":
                new_prev = {}
                for pk, pv in value.items():
                    if pk in ("prev_msa_first_row", "prev_pos"):
                        new_prev[pk] = pv[:n_valid]
                    elif pk in ("prev_pair", "prev_dgram"):
                        new_prev[pk] = pv[:n_valid, :n_valid]
                    else:
                        new_prev[pk] = pv
                new_inputs[key] = new_prev
            elif key in _RESIDUE_AXIS0_FEATURES:
                new_inputs[key] = value[:n_valid] if isinstance(value, (jnp.ndarray, np.ndarray, jax.Array)) else value
            elif key in _MSA_FEATURES or key in _EXTRA_MSA_FEATURES:
                new_inputs[key] = value[:, :n_valid] if isinstance(value, (jnp.ndarray, np.ndarray, jax.Array)) else value
            elif key in _TEMPLATE_FEATURES:
                new_inputs[key] = value[:, :n_valid] if isinstance(value, (jnp.ndarray, np.ndarray, jax.Array)) else value
            else:
                new_inputs[key] = value

        # Slice results
        new_result = {}
        for key, value in self.result.items():
            if key == "representations":
                new_reps = {}
                for rk, rv in value.items():
                    if rk in ("msa_first_row", "single"):
                        new_reps[rk] = rv[:n_valid]
                    elif rk == "pair":
                        new_reps[rk] = rv[:n_valid, :n_valid]
                    else:
                        new_reps[rk] = rv
                new_result[key] = new_reps
            elif key == "structure_module":
                new_sm = {}
                for sk, sv in value.items():
                    if isinstance(sv, (jnp.ndarray, np.ndarray, jax.Array)):
                        if sk in ("final_atom_positions",):
                            new_sm[sk] = sv[:n_valid]
                        elif sk in ("final_atom_mask",):
                            new_sm[sk] = sv[:n_valid]
                        elif sk == "act":
                            new_sm[sk] = sv[:n_valid]
                        else:
                            new_sm[sk] = sv
                    else:
                        new_sm[sk] = sv
                new_result[key] = new_sm
            elif key in _RESULT_PAIR_LOGIT_KEYS:
                # These are dicts with "logits" (L, L, B) and maybe "bin_edges"
                new_dict = {}
                for dk, dv in value.items():
                    if dk == "logits" and isinstance(dv, (jnp.ndarray, jax.Array)):
                        new_dict[dk] = dv[:n_valid, :n_valid]
                    else:
                        new_dict[dk] = dv
                new_result[key] = new_dict
            elif key in _RESULT_RESIDUE_LOGIT_KEYS:
                # These are dicts with "logits" (L, B)
                new_dict = {}
                for dk, dv in value.items():
                    if dk == "logits" and isinstance(dv, (jnp.ndarray, jax.Array)):
                        new_dict[dk] = dv[:n_valid]
                    else:
                        new_dict[dk] = dv
                new_result[key] = new_dict
            elif key == "prev":
                new_prev = {}
                for pk, pv in value.items():
                    if pk in ("prev_msa_first_row", "prev_pos"):
                        new_prev[pk] = pv[:n_valid]
                    elif pk in ("prev_pair", "prev_dgram"):
                        new_prev[pk] = pv[:n_valid, :n_valid]
                    else:
                        new_prev[pk] = pv
                new_result[key] = new_prev
            else:
                new_result[key] = value

        return AFResult(inputs=new_inputs, result=new_result)

    # --- Contact scores (masked) ---

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

    def contact_score(self, contact_distance=14.0, min_resi_distance=10, num_contacts=25):
        entropy = self.contact_entropy(contact_distance=contact_distance)
        mask = self.token_pad_mask
        resi_dist = abs(self.residue_index[:, None] - self.residue_index[None, :])
        chain = self._masked_chain_index
        other_chain = chain[:, None] != chain[None, :]
        # Mask out padded residues
        valid_pair = mask[:, None] * mask[None, :]
        entropy = jnp.where(
            ((resi_dist >= min_resi_distance) | other_chain) & valid_pair,
            entropy, 1e6)
        contact_score = entropy.sort(axis=1)[:, :num_contacts].mean(axis=-1)
        return (contact_score * mask).sum() / jnp.maximum(1, mask.sum())

    def chain_contact_score(self, target_chain, source_chain=0, contact_distance=14.0,
                            min_resi_distance=10, num_contacts=25):
        entropy = self.contact_entropy(contact_distance=contact_distance)
        mask = self.token_pad_mask
        resi_dist = abs(self.residue_index[:, None] - self.residue_index[None, :])
        chain = self._masked_chain_index
        other_chain = chain[:, None] != chain[None, :]
        valid_pair = mask[:, None] * mask[None, :]
        entropy = jnp.where(
            ((resi_dist >= min_resi_distance) | other_chain) & valid_pair,
            entropy, 1e6)
        selector = chain == target_chain
        entropy = jnp.where(selector[None, :], entropy, 1e6)
        binder_selector = ~selector * mask
        if source_chain is not None:
            if target_chain == source_chain:
                binder_selector = selector * mask
            else:
                binder_selector = binder_selector * (chain == source_chain)
        contact_score = (entropy.sort(axis=1)[:, :num_contacts].mean(axis=-1) * binder_selector).sum()
        contact_score /= jnp.maximum(1, binder_selector.sum())
        return contact_score

    def intra_contact_score(self, chain, contact_distance=14.0,
                           min_resi_distance=10, num_contacts=25):
        return self.chain_contact_score(
            chain, chain, contact_distance=contact_distance,
            min_resi_distance=min_resi_distance, num_contacts=num_contacts)

    def index_contact_score(self, target_index, source_index, contact_distance=14.0,
                            min_resi_distance=10, num_contacts=25):
        entropy = self.contact_entropy(contact_distance=contact_distance)
        mask = self.token_pad_mask
        resi_dist = abs(self.residue_index[:, None] - self.residue_index[None, :])
        chain = self._masked_chain_index
        other_chain = chain[:, None] != chain[None, :]
        valid_pair = mask[:, None] * mask[None, :]
        entropy = jnp.where(
            ((resi_dist >= min_resi_distance) | other_chain) & valid_pair,
            entropy, 1e6)
        entropy = jnp.where(target_index[None, :], entropy, 1e6)
        sorted_entropy = entropy.sort(axis=1)[:, :num_contacts]
        is_valid_entropy = sorted_entropy < 1e5
        mean_entropy = (sorted_entropy * is_valid_entropy).sum(axis=-1) / jnp.maximum(1, is_valid_entropy.sum(axis=-1))
        contact_score = (mean_entropy * source_index * mask).sum()
        contact_score /= jnp.maximum(1, (source_index * mask).sum())
        return contact_score

    # --- I/O ---

    def save(self, path):
        np.savez_compressed(
            path,
            **{f"inputs_{k}": np.array(v) for k, v in self.inputs.items()},
            **{f"result_{k}": np.array(v) for k, v in self.result.items()})

    def save_pdb(self, path):
        """Save in PDB format. Unpads if needed."""
        result = self
        if not isinstance(self.token_pad_mask, jax.core.Tracer):
            result = self.unpad()
        directory = os.path.dirname(path)
        if directory and not os.path.isdir(directory):
            os.makedirs(directory)
        _save_af_pdb(path, result)
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
