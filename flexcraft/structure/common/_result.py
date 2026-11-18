from dataclasses import fields, is_dataclass, replace
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from flexcraft.data.data import DesignData
from flexcraft.structure.common._utils import (
    atom_array_to_atomX,
    broadcast_array_to_atomX,
    get_contact_atom,
    unpad_features_by_schema,
)
from flexcraft.structure.metrics import (
    bin_pair_distances,
    chain_pair_mask,
    compute_chain_pae,
    compute_chain_ptm,
    compute_ipae,
    compute_ipsae,
    compute_iptm,
    compute_min_ipae,
    compute_ptm,
    compute_ptm_score,
    distogram_expected_distance,
    inter_chain_mask,
    pae_bin_centers,
    ptm_matrix_from_logits,
)

_DEFAULT_UNPAD_SCHEMA = {
    "token_index": (None, "token"),
    "residue_index": (None, "token"),
    "asym_id": (None, "token"),
    "entity_id": (None, "token"),
    "sym_id": (None, "token"),
    "mol_type": (None, "token"),
    "res_type": (None, "token", None),
    "profile": (None, "token", None),
    "deletion_mean": (None, "token"),
    "token_pad_mask": (None, "token"),
    "token_resolved_mask": (None, "token"),
    "token_disto_mask": (None, "token"),
    "token_attention_mask": (None, "token"),
    "input_ids": (None, "token"),
    "disto_center": (None, "token", None),
    "disto_coords_ensemble": lambda arr: (
        (None, None, "token", None)
        if arr.ndim == 4
        else (None, "token", None)
    ),
    "disto_target": lambda arr: (
        (None, "token", "token", None, None)
        if arr.ndim == 5
        else (None, "token", "token", None)
    ),
    "distogram_atom_idx": (None, "token"),
    "center_atom_idx": (None, "token"),
    "pocket_feature": lambda arr: (
        (None, "token", None) if arr.ndim == 3 else (None, "token")
    ),
    "frames_idx": lambda arr: (
        (None, None, "token", None)
        if arr.ndim == 4
        else (None, "token", None)
    ),
    "disto_cond": (None, "token", "token"),
    "disto_cond_mask": (None, "token", "token"),
    "token_bonds": lambda arr: (
        (None, "token", "token", None)
        if arr.ndim == 4
        else (None, "token", "token")
    ),
    "type_bonds": (None, "token", "token"),
    "contact_conditioning": (None, "token", "token", None),
    "contact_threshold": (None, "token", "token"),
    "msa": (None, "msa", "token", None),
    "msa_mask": (None, "msa", "token"),
    "msa_attention_mask": (None, "msa", "token"),
    "msa_paired": (None, "msa", "token"),
    "has_deletion": (None, "msa", "token"),
    "deletion_value": (None, "msa", "token"),
    "template_frame_rot": (None, "template", "token", None, None),
    "template_frame_t": (None, "template", "token", None),
    "template_ca": (None, "template", "token", None),
    "template_cb": (None, "template", "token", None),
    "template_mask": (None, "template", "token"),
    "template_mask_cb": (None, "template", "token"),
    "template_mask_frame": (None, "template", "token"),
    "template_restype": (None, "template", "token", None),
    "visibility_ids": (None, "template", "token"),
    "atom_pad_mask": (None, "atom"),
    "atom_attention_mask": (None, "atom"),
    "atom_resolved_mask": lambda arr: (
        (None, None, "atom") if arr.ndim == 3 else (None, "atom")
    ),
    "atom_to_token": lambda arr: (
        (None, "atom", "token") if arr.ndim == 3 else (None, "atom")
    ),
    "token_to_rep_atom": (None, "token", "atom"),
    "token_to_center_atom": (None, "token", "atom"),
    "r_set_to_rep_atom": (None, "token", "atom"),
    "ref_pos": (None, "atom", None),
    "ref_space_uid": (None, "atom"),
    "ref_charge": (None, "atom"),
    "ref_element": lambda arr: (
        (None, "atom", None) if arr.ndim == 3 else (None, "atom")
    ),
    "ref_atom_name_chars": lambda arr: (
        (None, "atom", None, None) if arr.ndim == 4 else (None, "atom", None)
    ),
    "ref_chirality": (None, "atom"),
    "is_resolved": (None, "atom"),
    "gt_coords": lambda arr: (
        (None, None, "atom", None) if arr.ndim == 4 else (None, "atom", None)
    ),
    "coords": lambda arr: (
        (None, None, "atom", None) if arr.ndim == 4 else (None, "atom", None)
    ),
}


class AF3LikeResult(eqx.Module):
    data: dict

    _plddt_logits_is_atom_level = False

    @property
    def token_pad_mask(self) -> jax.Array:
        features = self.data["features"]
        if "token_pad_mask" in features and features["token_pad_mask"] is not None:
            mask = features["token_pad_mask"]
            return (mask[0] if mask.ndim == 2 else mask) > 0
        if (
            "token_attention_mask" in features
            and features["token_attention_mask"] is not None
        ):
            mask = features["token_attention_mask"]
            return (mask[0] if mask.ndim == 2 else mask).astype(jnp.bool_)
        return jnp.ones((self.residue_index.shape[0],), dtype=jnp.bool_)

    @property
    def atom_pad_mask(self) -> jax.Array:
        features = self.data["features"]
        if "atom_pad_mask" in features and features["atom_pad_mask"] is not None:
            mask = features["atom_pad_mask"]
            return (mask[0] if mask.ndim == 2 else mask) > 0
        if (
            "atom_attention_mask" in features
            and features["atom_attention_mask"] is not None
        ):
            mask = features["atom_attention_mask"]
            return (mask[0] if mask.ndim == 2 else mask).astype(jnp.bool_)
        return self.atom_to_token.sum(axis=-1) > 0

    @property
    def log_distogram(self):
        logits = self.data["distogram"]
        return jax.nn.log_softmax(logits, axis=-1)

    @property
    def distogram(self):
        return jnp.exp(self.log_distogram)

    @property
    def distogram_bin_edges(self):
        num_bins = self.data["distogram"].shape[-1]
        return jnp.linspace(2.0, 22.0, num_bins + 1)

    @property
    def distogram_bin_centers(self):
        edges = self.distogram_bin_edges
        return 0.5 * (edges[1:] + edges[:-1])

    @property
    def distogram_mean(self):
        return distogram_expected_distance(
            self.distogram, self.distogram_bin_centers
        )

    @property
    def distance(self) -> jax.Array:
        return self.distogram_mean

    @property
    def sample_distogram(self):
        sample24, mask24 = self.atom24_samples
        if self.is_single_sample:
            cb = jnp.where(mask24[:, 4, None], sample24[:, 4], sample24[:, 1])[
                None
            ]
        else:
            cb = jnp.where(
                mask24[None, :, 4, None],
                sample24[:, :, 4],
                sample24[:, :, 1],
            )
        dist = jnp.linalg.norm(
            cb[..., :, None, :] - cb[..., None, :, :], axis=-1
        )
        binned = bin_pair_distances(dist, self.distogram_bin_edges)
        num_bins = self.data["distogram"].shape[-1]
        return jax.nn.one_hot(binned, num_bins).mean(axis=0)

    def sample_distogram_nll(self):
        gt = self.sample_distogram
        predicted = self.log_distogram
        nll = -(gt * predicted).sum(axis=-1)
        pair_mask = self.token_pad_mask[:, None] & self.token_pad_mask[None, :]
        return (nll * pair_mask).sum() / jnp.maximum(1, pair_mask.sum())

    @property
    def residue_index(self) -> jax.Array:
        return self.data["features"]["residue_index"][0]

    @property
    def chain_index(self) -> jax.Array:
        return self.data["features"]["asym_id"][0]

    @chain_index.setter
    def chain_index(self, value: jax.Array):
        self.data["features"]["asym_id"] = value[None]

    def set_chain_index(self, value: jax.Array):
        self.data["features"]["asym_id"] = value[None]
        return self.chain_index

    def contact_probability(self, contact_distance=10.0) -> jax.Array:
        """Compute predicted contact probability for each token pair."""
        edge_mask = self.distogram_bin_edges[1:] < contact_distance
        return (edge_mask * self.distogram).sum(axis=-1)

    def contact_entropy(self, contact_distance=14.0) -> jax.Array:
        """Compute distogram contact entropy for each token pair."""
        edge_mask = self.distogram_bin_edges[1:] < contact_distance
        distogram_clipped = jax.nn.softmax(
            self.log_distogram - 1e9 * (1 - edge_mask), axis=-1
        )
        distogram_clipped = jnp.where(edge_mask, distogram_clipped, 0)
        return -(distogram_clipped * self.log_distogram).sum(axis=-1)

    def contact_score(
        self, contact_distance=14.0, min_resi_distance=10, num_contacts=25
    ):
        token_mask = self.token_pad_mask
        return self.index_contact_score(
            token_mask,
            token_mask,
            contact_distance=contact_distance,
            min_resi_distance=min_resi_distance,
            num_contacts=num_contacts,
        )

    def chain_contact_score(
        self,
        target_chain,
        source_chain=0,
        contact_distance=14.0,
        min_resi_distance=10,
        num_contacts=25,
    ):
        token_mask = self.token_pad_mask
        chain = jnp.where(token_mask, self.chain_index, -1)
        target_arr = jnp.atleast_1d(jnp.asarray(target_chain, dtype=jnp.int32))
        selector = (chain[:, None] == target_arr[None, :]).any(axis=1) & token_mask
        if source_chain is None:
            binder_selector = (~selector) & token_mask
        else:
            source_arr = jnp.atleast_1d(
                jnp.asarray(source_chain, dtype=jnp.int32)
            )
            binder_selector = (
                chain[:, None] == source_arr[None, :]
            ).any(axis=1) & token_mask
        return self.index_contact_score(
            selector,
            binder_selector,
            contact_distance=contact_distance,
            min_resi_distance=min_resi_distance,
            num_contacts=num_contacts,
        )

    def index_contact_score(
        self,
        target_index,
        source_index,
        contact_distance=14.0,
        min_resi_distance=10,
        num_contacts=25,
    ):
        token_mask = self.token_pad_mask
        target_mask = jnp.asarray(target_index, dtype=jnp.bool_) & token_mask
        source_mask = jnp.asarray(source_index, dtype=jnp.bool_) & token_mask
        entropy = self.contact_entropy(contact_distance=contact_distance)
        resi_dist = jnp.abs(
            self.residue_index[:, None] - self.residue_index[None, :]
        )
        other_chain = inter_chain_mask(self.chain_index, token_mask=token_mask)
        pair_valid = (
            token_mask[:, None]
            & target_mask[None, :]
            & ((resi_dist >= min_resi_distance) | other_chain)
        )
        entropy = jnp.where(pair_valid, entropy, 1e6)
        sorted_entropy = jnp.sort(entropy, axis=1)[:, :num_contacts]
        is_valid_entropy = sorted_entropy < 1e5
        mean_entropy = (sorted_entropy * is_valid_entropy).sum(
            axis=-1
        ) / jnp.maximum(1, is_valid_entropy.sum(axis=-1))
        return (mean_entropy * source_mask).sum() / jnp.maximum(
            1, source_mask.sum()
        )

    def intra_contact_score(
        self,
        chain,
        contact_distance=14.0,
        min_resi_distance=10,
        num_contacts=25,
    ):
        return self.chain_contact_score(
            chain,
            chain,
            contact_distance=contact_distance,
            min_resi_distance=min_resi_distance,
            num_contacts=num_contacts,
        )

    def chain_contact_frequency(
        self, target_chain=0, binder_chain=1, contact_distance=8.0
    ) -> jax.Array:
        """Per-residue contact frequency on `target_chain` against `binder_chain`."""
        atom24, _ = self.atom24_samples
        if self.is_single_sample:
            atom24 = atom24[None]
        contact_atom = jax.vmap(get_contact_atom, (0, None), 0)(
            atom24, self.mol_type
        )
        pair_mask = chain_pair_mask(
            self.chain_index,
            binder_chain,
            source_chain=target_chain,
            token_mask=self.token_pad_mask,
        )

        def per_sample(coords):
            dist = jnp.linalg.norm(
                coords[:, None, :] - coords[None, :, :], axis=-1
            )
            return ((dist < contact_distance) & pair_mask).any(axis=1)

        contacts = jax.vmap(per_sample, in_axes=0, out_axes=0)(contact_atom)
        return contacts.mean(axis=0) * self.token_pad_mask

    @property
    def atom_to_token(self):
        return self.data["features"]["atom_to_token"][0]

    @property
    def atom24_samples(self):
        sample = self.data["samples"]
        return self._transform_sampled(sample, num_atoms=24)

    @property
    def atom24(self):
        if self.is_single_sample:
            return self.atom24_samples
        sample24, mask24 = self.atom24_samples
        pair_mask = self.token_pad_mask[:, None] & self.token_pad_mask[None, :]
        sample_pae = (self.pae * pair_mask).sum(axis=(-1, -2)) / jnp.maximum(
            1, pair_mask.sum()
        )
        best_pae = jnp.argmin(sample_pae, axis=0)
        return sample24[best_pae], mask24

    @property
    def atom14(self):
        atom24, mask24 = self.atom24
        return atom24[:, :14], mask24[:, :14]

    @property
    def atom4(self):
        atom24, _ = self.atom24
        return atom24[:, :4]

    @property
    def cb_samples(self):
        atom24, _ = self.atom24_samples
        return jax.vmap(get_contact_atom, (0, None), 0)(atom24, self.mol_type)

    @property
    def cb(self):
        atom24, _ = self.atom24
        return get_contact_atom(atom24, self.mol_type)

    @property
    def mol_type(self):
        return self.data["features"]["mol_type"][0]

    def _transform_sampled(self, sampled_property, num_atoms=24):
        if self.is_single_sample:
            sampled_property = sampled_property[0]
        else:
            sampled_property = sampled_property[:, 0]
            sampled_property = jnp.moveaxis(sampled_property, 0, -1)
        sample24, mask24 = atom_array_to_atomX(
            sampled_property, self.atom_to_token, num_atoms=num_atoms
        )
        if not self.is_single_sample:
            sample24 = jnp.moveaxis(sample24, -1, 0)
        return sample24, mask24

    def _broadcast_sampled(self, sampled_property, num_atoms=24):
        if self.is_single_sample:
            sampled_property = sampled_property[0]
        else:
            sampled_property = sampled_property[:, 0]
            sampled_property = jnp.moveaxis(sampled_property, 0, -1)
        sample24, mask24 = broadcast_array_to_atomX(
            sampled_property, self.atom_to_token, num_atoms=num_atoms
        )
        if not self.is_single_sample:
            sample24 = jnp.moveaxis(sample24, -1, 0)
        return sample24, mask24

    @property
    def is_single_sample(self):
        return len(self.data["samples"].shape) == 3

    @property
    def admits_samples(self):
        return True

    @property
    def plddt_logits(self):
        plddt_logits = self.data["confidence"].plddt_logits
        if self.is_single_sample:
            return plddt_logits[0]
        return plddt_logits[:, 0]

    @property
    def plddt(self):
        plddt = self.data["confidence"].plddt
        if self.is_single_sample:
            return plddt[0]
        return plddt[:, 0]

    def index_plddt(self, mask=None) -> jax.Array:
        valid = self.token_pad_mask
        if mask is not None:
            valid = valid & jnp.asarray(mask, dtype=jnp.bool_)
        return (self.plddt * valid).sum(axis=-1) / jnp.maximum(1, valid.sum())

    @property
    def mean_plddt(self) -> jax.Array:
        return self.index_plddt()

    @property
    def max_pae(self) -> float:
        return 32.0

    @property
    def pae_logits(self):
        if self.is_single_sample:
            return self.data["confidence"].pae_logits[0]
        return self.data["confidence"].pae_logits[:, 0]

    @property
    def pae_bin_edges(self) -> jax.Array:
        return jnp.linspace(0.0, self.max_pae, self.pae_logits.shape[-1] + 1)

    @property
    def pae_bin_centers(self) -> jax.Array:
        edges = self.pae_bin_edges
        return 0.5 * (edges[:-1] + edges[1:])

    @property
    def pae(self):
        if self.is_single_sample:
            return (
                jnp.fill_diagonal(
                    self.data["confidence"].pae[0], 0.0, inplace=False
                )
                / self.max_pae
            )
        return (
            jax.vmap(lambda x: jnp.fill_diagonal(x, 0.0, inplace=False))(
                self.data["confidence"].pae[:, 0]
            )
            / self.max_pae
        )

    def index_pae(self, mask=None) -> jax.Array:
        valid = self.token_pad_mask
        if mask is not None:
            valid = valid & jnp.asarray(mask, dtype=jnp.bool_)
        pair_mask = valid[:, None] & valid[None, :]
        return (self.pae * pair_mask).sum(axis=(-1, -2)) / jnp.maximum(
            1, pair_mask.sum()
        )

    @property
    def mean_pae(self) -> jax.Array:
        return self.index_pae()

    def index_ipae(self, chain_index=None) -> jax.Array:
        ci = self.chain_index if chain_index is None else chain_index
        return compute_ipae(self.pae, ci, token_mask=self.token_pad_mask)

    @property
    def ipae(self) -> jax.Array:
        return self.index_ipae()

    def index_min_ipae(self, chain_index=None) -> jax.Array:
        ci = self.chain_index if chain_index is None else chain_index
        return compute_min_ipae(self.pae, ci, token_mask=self.token_pad_mask)

    @property
    def min_ipae(self) -> jax.Array:
        return self.index_min_ipae()

    def chain_selector(self, target_chain, source_chain=None):
        return chain_pair_mask(
            self.chain_index,
            target_chain,
            source_chain=source_chain,
            token_mask=self.token_pad_mask,
        )

    def chain_pae(self, target_chain, source_chain=None) -> jax.Array:
        return compute_chain_pae(
            self.pae,
            self.chain_index,
            target_chain,
            source_chain=source_chain,
            token_mask=self.token_pad_mask,
        )

    def index_ptm(self, mask=None) -> jax.Array:
        return compute_ptm(
            self.pae_logits,
            mask=mask,
            token_mask=self.token_pad_mask,
            max_pae=self.max_pae,
            bin_centers=self.pae_bin_centers,
        )

    @property
    def ptm(self) -> jax.Array:
        return self.index_ptm()

    def index_iptm(self, chain_index=None) -> jax.Array:
        ci = self.chain_index if chain_index is None else chain_index
        return compute_iptm(
            self.pae_logits,
            ci,
            token_mask=self.token_pad_mask,
            max_pae=self.max_pae,
            bin_centers=self.pae_bin_centers,
        )

    @property
    def iptm(self) -> jax.Array:
        return self.index_iptm()

    def index_ipsae(
        self,
        chain_index=None,
        raw_pae_threshold: float = 15.0,
        pae_cutoff: float | None = None,
    ) -> jax.Array:
        ci = self.chain_index if chain_index is None else chain_index
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

    def ipsae(
        self,
        chain_index=None,
        raw_pae_threshold: float = 15.0,
        pae_cutoff: float | None = None,
    ) -> jax.Array:
        return self.index_ipsae(
            chain_index=chain_index,
            raw_pae_threshold=raw_pae_threshold,
            pae_cutoff=pae_cutoff,
        )

    def index_ptm_score(self, chain_index=None) -> jax.Array:
        ci = self.chain_index if chain_index is None else chain_index
        return compute_ptm_score(
            self.pae_logits,
            ci,
            token_mask=self.token_pad_mask,
            max_pae=self.max_pae,
            bin_centers=self.pae_bin_centers,
        )

    @property
    def ptm_score(self) -> jax.Array:
        return self.index_ptm_score()

    def chain_ptm(self, target_chain, source_chain=None) -> jax.Array:
        return compute_chain_ptm(
            self.pae_logits,
            self.chain_index,
            target_chain,
            source_chain=source_chain,
            token_mask=self.token_pad_mask,
            max_pae=self.max_pae,
            bin_centers=self.pae_bin_centers,
        )

    def ptm_matrix(self, L=None) -> jax.Array:
        return ptm_matrix_from_logits(
            self.pae_logits,
            num_residues=L,
            token_mask=self.token_pad_mask,
            max_pae=self.max_pae,
            bin_centers=self.pae_bin_centers,
        )

    @property
    def restype(self):
        res_type_one_hot = self.data["features"]["res_type"][0]
        res_type = jnp.argmax(res_type_one_hot, axis=-1)
        num_types = res_type_one_hot.shape[-1]
        res_type = res_type - 2
        res_type = jnp.where(res_type < 0, res_type + num_types, res_type)
        return res_type

    def to_data(self, return_samples=False) -> DesignData:
        """Convert an AF3LikeResult to static-shaped, mask-carrying DesignData."""
        if return_samples:
            atom24, mask24 = self.atom24_samples
        else:
            atom24, mask24 = self.atom24
        token_mask = self.token_pad_mask
        return DesignData(
            data=dict(
                atom_positions=atom24,
                atom_mask=mask24 & token_mask[:, None],
                aa=self.restype,
                mask=mask24.any(axis=-1) & token_mask,
                residue_index=self.residue_index,
                chain_index=jnp.where(token_mask, self.chain_index, -1),
                batch_index=jnp.zeros_like(self.residue_index),
                plddt=(
                    self.plddt.mean(axis=0)
                    if len(self.plddt.shape) == 2
                    else self.plddt
                ),
            )
        ).untie()

    def _feature_schema(self) -> dict:
        return _DEFAULT_UNPAD_SCHEMA

    def _unpad_confidence(
        self,
        confidence,
        n_valid_tokens: int,
        n_valid_atoms: int,
        n_total_tokens: int,
        n_total_atoms: int,
    ):
        if confidence is None:
            return None

        def _slice_field(name: str, val: Any):
            if val is None or not hasattr(val, "shape"):
                return val
            if name == "plddt":
                if (
                    self._plddt_logits_is_atom_level
                    and val.shape[-1] == n_total_atoms
                    and n_total_atoms != n_total_tokens
                ):
                    return val[..., :n_valid_atoms]
                return val[..., :n_valid_tokens]
            if name == "plddt_logits":
                if self._plddt_logits_is_atom_level or (
                    val.shape[-2] == n_total_atoms
                    and n_total_atoms != n_total_tokens
                ):
                    return val[..., :n_valid_atoms, :]
                return val[..., :n_valid_tokens, :]
            if name in ("pae", "pde"):
                return val[..., :n_valid_tokens, :n_valid_tokens]
            if name in ("pae_logits", "pde_logits"):
                return val[..., :n_valid_tokens, :n_valid_tokens, :]
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

    def _unpad_state(self, state, n_valid_tokens: int):
        if state is None:
            return None
        if hasattr(state, "shape") and state.ndim == 4:
            return state[:, :n_valid_tokens, :n_valid_tokens, :]
        if is_dataclass(state) and hasattr(state, "s") and hasattr(state, "z"):
            return replace(
                state,
                s=state.s[:, :n_valid_tokens],
                z=state.z[:, :n_valid_tokens, :n_valid_tokens],
            )
        return state

    def _unpad_embedding(self, embedding, n_valid_tokens: int):
        if embedding is None or not is_dataclass(embedding):
            return embedding
        updates = {}
        for f in fields(embedding):
            val = getattr(embedding, f.name)
            if val is None or not hasattr(val, "shape"):
                continue
            if f.name in ("s_inputs", "s_init") and val.ndim >= 2:
                updates[f.name] = val[:, :n_valid_tokens]
            elif (
                f.name in ("z_init", "relative_position_encoding")
                and val.ndim >= 3
            ):
                updates[f.name] = val[:, :n_valid_tokens, :n_valid_tokens]
        return replace(embedding, **updates) if updates else embedding

    def unpad(self) -> "AF3LikeResult":
        """Host-only: slice padded token and atom dimensions back to valid counts."""
        token_mask = self.token_pad_mask
        atom_mask = self.atom_pad_mask
        if isinstance(token_mask, jax.core.Tracer) or isinstance(
            atom_mask, jax.core.Tracer
        ):
            raise RuntimeError(
                f"{type(self).__name__}.unpad() cannot be called inside jax.jit; "
                "use token_pad_mask / atom_pad_mask inside traced functions."
            )
        n_valid_tokens = int(np.asarray(token_mask, dtype=bool).sum())
        n_valid_atoms = int(np.asarray(atom_mask, dtype=bool).sum())
        n_total_tokens = int(token_mask.shape[0])
        n_total_atoms = int(atom_mask.shape[0])
        if n_valid_tokens == n_total_tokens and n_valid_atoms == n_total_atoms:
            return self

        samples = self.data["samples"]
        if self.is_single_sample:
            unpadded_samples = samples[:, :n_valid_atoms, :]
        else:
            unpadded_samples = samples[:, :, :n_valid_atoms, :]

        unpadded_data = dict(self.data)
        unpadded_data["samples"] = unpadded_samples
        if "distogram" in unpadded_data and unpadded_data["distogram"] is not None:
            unpadded_data["distogram"] = unpadded_data["distogram"][
                :n_valid_tokens, :n_valid_tokens, :
            ]
        if "features" in unpadded_data and unpadded_data["features"] is not None:
            unpadded_data["features"] = unpad_features_by_schema(
                unpadded_data["features"],
                {"token": n_valid_tokens, "atom": n_valid_atoms},
                self._feature_schema(),
            )
        if "confidence" in unpadded_data:
            unpadded_data["confidence"] = self._unpad_confidence(
                unpadded_data["confidence"],
                n_valid_tokens,
                n_valid_atoms,
                n_total_tokens,
                n_total_atoms,
            )
        if "state" in unpadded_data:
            unpadded_data["state"] = self._unpad_state(
                unpadded_data["state"], n_valid_tokens
            )
        if "embedding" in unpadded_data:
            unpadded_data["embedding"] = self._unpad_embedding(
                unpadded_data["embedding"], n_valid_tokens
            )
        return type(self)(data=unpadded_data)

    def save(self, path: str):
        np.savez_compressed(
            path, **{k: np.array(v) for k, v in self.data.items()}
        )
