"""Protenix result and prediction dataclasses."""

from typing import Any
from dataclasses import dataclass, fields, is_dataclass, replace

import numpy as np

import jax
import jax.numpy as jnp

from flexcraft.structure.common._result import AF3LikeResult
from flexcraft.structure.protenix._data import ProtenixWriter


# Feature-key -> axis schema for unpadding.
# Each tuple maps positional axes to named dimension kinds:
#   None = batch/leave alone, "token" = token dim, "atom" = atom dim
_PROTENIX_FEATURE_SCHEMA = {
    # Token-axis features
    "token_index": ("token",),
    "residue_index": ("token",),
    "asym_id": ("token",),
    "entity_id": ("token",),
    "sym_id": ("token",),
    "restype": ("token", None),
    "profile": ("token", None),
    "deletion_mean": ("token",),
    "token_pad_mask": ("token",),
    "atom_rep_atom_idx": ("token",),
    "has_frame": ("token",),
    "frame_atom_index": ("token", None),
    "token_bonds": ("token", "token"),
    "distogram_rep_atom_mask": ("atom",),
    # Atom-axis features
    "is_protein": ("atom",),
    "is_dna": ("atom",),
    "is_rna": ("atom",),
    "is_ligand": ("atom",),
    "atom_to_token_idx": ("atom",),
    "atom_to_tokatom_idx": ("atom",),
    "atom_pad_mask": ("atom",),
    "ref_pos": ("atom", None),
    "ref_mask": ("atom",),
    "ref_space_uid": ("atom",),
    "ref_charge": ("atom",),
    "ref_element": ("atom", None),
    "ref_atom_name_chars": ("atom", None, None),
    "mol_id": ("atom",),
    "mol_atom_index": ("atom",),
    "entity_mol_id": ("atom",),
    "pae_rep_atom_mask": ("atom",),
    "plddt_m_rep_atom_mask": ("atom",),
    "modified_res_mask": ("atom",),
    "bond_mask": ("atom", "atom"),
    # MSA features: (N_msa, L) or (N_msa, L, ...)
    "msa": (None, "token"),
    "msa_mask": (None, "token"),
    "has_deletion": (None, "token"),
    "deletion_value": (None, "token"),
    # Template features: (T, L, ...) or (T,)
    "template_aatype": (None, "token"),
    "template_distogram": (None, "token", "token", None),
    "template_pseudo_beta_mask": lambda arr: (
        (None, "token", "token") if arr.ndim == 3 else (None, "token")
    ),
    "template_unit_vector": (None, "token", "token", None),
    "template_backbone_frame_mask": lambda arr: (
        (None, "token", "token") if arr.ndim == 3 else (None, "token")
    ),
    "template_mask": (None,),
}


class ProtenixResult(AF3LikeResult):
    data: dict

    _plddt_logits_is_atom_level = True

    @property
    def residue_index(self):
        return self.data["features"]["residue_index"]

    @property
    def chain_index(self):
        return self.data["features"]["asym_id"]

    @property
    def restype(self):
        res_type_one_hot = self.data["features"]["restype"]
        return jnp.argmax(res_type_one_hot, axis=-1)

    @property
    def mol_type(self):
        return jnp.argmax(jnp.stack((
            self.data["features"]["is_protein"],
            self.data["features"]["is_dna"],
            self.data["features"]["is_rna"],
            self.data["features"]["is_ligand"],
        ), axis=-1), axis=-1)[self.data["features"]["atom_rep_atom_idx"]]

    @property
    def token_pad_mask(self):
        """1D (L,) bool mask: True for valid tokens."""
        mask = self.data["features"].get("token_pad_mask", None)
        if mask is not None:
            return jnp.asarray(mask) > 0
        return jnp.ones(self.residue_index.shape[0], dtype=jnp.bool_)

    @property
    def atom_pad_mask(self):
        """1D (A,) bool mask: True for valid atoms."""
        mask = self.data["features"].get("atom_pad_mask", None)
        if mask is not None:
            return jnp.asarray(mask) > 0
        num_atoms = self.data["features"]["atom_to_token_idx"].shape[-1]
        return jnp.ones(num_atoms, dtype=jnp.bool_)

    @property
    def atom_to_token(self):
        """Map atoms to tokens, masking padded atoms so they don't map to token 0."""
        num_tokens = self.residue_index.shape[0]
        atom_to_token_idx = self.data["features"]["atom_to_token_idx"]
        one_hot = jax.nn.one_hot(atom_to_token_idx, num_tokens, axis=1)
        return one_hot * self.atom_pad_mask[..., None]

    @property
    def plddt_logits(self):
        index = self.data["features"]["atom_rep_atom_idx"]
        if self.is_single_sample:
            return self.data["confidence"].plddt_logits[0, index]
        return self.data["confidence"].plddt_logits[:, 0, index]

    @property
    def plddt(self):
        return (jax.nn.softmax(self.plddt_logits, axis=-1) * (jnp.arange(50) + 0.5) / 50).sum(axis=-1)

    @property
    def pae(self):
        PAE_BINS = jnp.arange(start=0.25, stop=32.0, step=0.5)
        pae = (jax.nn.softmax(self.pae_logits, axis=-1) * PAE_BINS).sum(axis=-1)
        if self.is_single_sample:
            return jnp.fill_diagonal(pae, 0.0, inplace=False) / 32
        return jax.vmap(lambda x: jnp.fill_diagonal(x, 0.0, inplace=False))(pae) / 32

    def _feature_schema(self) -> dict:
        return _PROTENIX_FEATURE_SCHEMA

    def _unpad_confidence(self, confidence, n_valid_tokens, n_valid_atoms,
                          n_total_tokens, n_total_atoms):
        """Slice confidence fields by token/atom axes."""
        if confidence is None:
            return None

        def _slice_field(name, val):
            if val is None or not hasattr(val, "shape"):
                return val
            if name == "plddt":
                if val.shape[-1] == n_total_atoms and n_total_atoms != n_total_tokens:
                    return val[..., :n_valid_atoms]
                return val[..., :n_valid_tokens]
            if name == "plddt_logits":
                # Atom-level pLDDT logits for Protenix
                if val.shape[-2] == n_total_atoms and n_total_atoms != n_total_tokens:
                    return val[..., :n_valid_atoms, :]
                return val[..., :n_valid_tokens, :]
            if name in ("pae", "pde"):
                return val[..., :n_valid_tokens, :n_valid_tokens]
            if name in ("pae_logits", "pde_logits"):
                return val[..., :n_valid_tokens, :n_valid_tokens, :]
            if name == "resolved_logits":
                if val.shape[-2] == n_total_atoms and n_total_atoms != n_total_tokens:
                    return val[..., :n_valid_atoms, :]
                return val[..., :n_valid_tokens, :]
            return val

        if isinstance(confidence, dict):
            return {k: _slice_field(k, v) for k, v in confidence.items()}
        if is_dataclass(confidence):
            updates = {
                f.name: _slice_field(f.name, getattr(confidence, f.name))
                for f in fields(confidence)
            }
            return replace(confidence, **updates)
        return confidence

    def _unpad_state(self, state, n_valid_tokens):
        """Slice trunk embedding state."""
        if state is None:
            return state
        if hasattr(state, "shape") and state.ndim == 4:
            return state[:, :n_valid_tokens, :n_valid_tokens, :]
        if is_dataclass(state) and hasattr(state, "s") and hasattr(state, "z"):
            return replace(
                state,
                s=state.s[:n_valid_tokens],
                z=state.z[:n_valid_tokens, :n_valid_tokens],
            )
        return state

    def _unpad_embedding(self, embedding, n_valid_tokens):
        """Slice initial embedding."""
        if embedding is None or not is_dataclass(embedding):
            return embedding
        updates = {}
        for f in fields(embedding):
            val = getattr(embedding, f.name)
            if val is None or not hasattr(val, "shape"):
                continue
            if f.name in ("s_inputs", "s_init") and val.ndim >= 1:
                updates[f.name] = val[:n_valid_tokens]
            elif f.name in ("z_init", "relative_position_encoding") and val.ndim >= 2:
                updates[f.name] = val[:n_valid_tokens, :n_valid_tokens]
        return replace(embedding, **updates) if updates else embedding

    def unpad(self) -> "ProtenixResult":
        """Remove padding. Host-only (raises if traced)."""
        token_mask = self.token_pad_mask
        atom_mask = self.atom_pad_mask
        if isinstance(token_mask, jax.core.Tracer) or isinstance(
            atom_mask, jax.core.Tracer
        ):
            raise RuntimeError(
                "ProtenixResult.unpad() cannot be called inside jax.jit.")

        n_valid_tokens = int(np.asarray(token_mask, dtype=bool).sum())
        n_valid_atoms = int(np.asarray(atom_mask, dtype=bool).sum())
        n_total_tokens = int(token_mask.shape[0])
        n_total_atoms = int(atom_mask.shape[0])
        if n_valid_tokens == n_total_tokens and n_valid_atoms == n_total_atoms:
            return self

        unpadded_data = dict(self.data)

        # Slice samples: (..., A, 3)
        samples = unpadded_data.get("samples")
        if samples is not None and isinstance(samples, (jnp.ndarray, jax.Array)):
            if self.is_single_sample:
                unpadded_data["samples"] = samples[:, :n_valid_atoms, :]
            else:
                unpadded_data["samples"] = samples[:, :, :n_valid_atoms, :]

        # Slice distogram: (L, L, bins)
        if "distogram" in unpadded_data and unpadded_data["distogram"] is not None:
            unpadded_data["distogram"] = unpadded_data["distogram"][
                :n_valid_tokens, :n_valid_tokens, :]

        # Slice features using schema
        if "features" in unpadded_data and unpadded_data["features"] is not None:
            unpadded_data["features"] = _unpad_features_by_schema(
                unpadded_data["features"],
                n_valid_tokens, n_valid_atoms,
                n_total_tokens, n_total_atoms,
                self._feature_schema())

        # Slice confidence
        if "confidence" in unpadded_data:
            unpadded_data["confidence"] = self._unpad_confidence(
                unpadded_data["confidence"],
                n_valid_tokens, n_valid_atoms,
                n_total_tokens, n_total_atoms)

        # Slice state (TrunkEmbedding)
        if "state" in unpadded_data:
            unpadded_data["state"] = self._unpad_state(
                unpadded_data["state"], n_valid_tokens)

        # Slice embedding (InitialEmbedding)
        if "embedding" in unpadded_data:
            unpadded_data["embedding"] = self._unpad_embedding(
                unpadded_data["embedding"], n_valid_tokens)

        return ProtenixResult(data=unpadded_data)


def _unpad_features_by_schema(features, n_valid_tokens, n_valid_atoms,
                               n_total_tokens, n_total_atoms, schema):
    """Slice feature tensors using explicit schema to avoid shape collisions."""
    axis_limits = {"token": n_valid_tokens, "atom": n_valid_atoms}
    result = {}
    for key, value in features.items():
        if value is None or key not in schema:
            result[key] = value
            continue
        if not hasattr(value, "shape") or len(value.shape) == 0:
            result[key] = value
            continue
        axes = schema[key]
        if callable(axes):
            axes = axes(value)
        slices = []
        needs_slice = False
        for axis_idx, axis_name in enumerate(axes):
            if axis_idx >= len(value.shape):
                break
            if axis_name is not None and axis_name in axis_limits:
                limit = int(axis_limits[axis_name])
                if limit < value.shape[axis_idx]:
                    needs_slice = True
                slices.append(slice(0, limit))
            else:
                slices.append(slice(None))
        # Extend with full slices for remaining axes
        while len(slices) < len(value.shape):
            slices.append(slice(None))
        result[key] = value[tuple(slices)] if needs_slice else value
    return result


@dataclass
class ProtenixPrediction:
    data: Any
    writer: ProtenixWriter = None

    @property
    def result(self):
        return ProtenixResult(data=self.data)

    def prep_data(self, sample_index=0):
        """Extract coordinates and pLDDT for saving."""
        is_multisample = len(self.data["samples"].shape) == 4
        plddt = (jax.nn.softmax(self.data["confidence"].plddt_logits, axis=-1)
                 * (jnp.arange(50) + 0.5)).sum(axis=-1) / 50
        coords = self.data["samples"]
        if is_multisample:
            plddt = plddt[sample_index, 0]
            coords = coords[sample_index, 0]
        else:
            plddt = plddt[0]
            coords = coords[0]
        return np.array(coords), np.array(plddt)

    def save_pdb(self, path, sample_index=0):
        """Save as PDB. Unpads before writing."""
        result = self.result
        if not isinstance(result.token_pad_mask, jax.core.Tracer):
            result = result.unpad()
        # Create an unpadded prediction for prep_data
        unpadded_pred = ProtenixPrediction(data=result.data, writer=self.writer)
        if self.writer is not None:
            coords, plddt = unpadded_pred.prep_data(sample_index=sample_index)
            self.writer.save_pdb(path, coords, plddt=plddt)

    def save_cif(self, path, sample_index=0):
        """Save as CIF. Unpads before writing."""
        result = self.result
        if not isinstance(result.token_pad_mask, jax.core.Tracer):
            result = result.unpad()
        unpadded_pred = ProtenixPrediction(data=result.data, writer=self.writer)
        if self.writer is not None:
            coords, plddt = unpadded_pred.prep_data(sample_index=sample_index)
            self.writer.save_cif(path, coords, plddt=plddt)

    def save(self, path):
        self.result.save(path)
