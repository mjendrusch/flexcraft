from typing import Any
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import equinox as eqx
from flexcraft.data.data import DesignData
from flexcraft.structure.boltz._utils import *
from flexcraft.structure.boltz._data import Joltz2Writer

class JoltzResult(eqx.Module):
    data: dict

    @property
    def log_distogram(self):
        logits = self.data["distogram"]
        return jax.nn.log_softmax(logits, axis=-1)

    @property
    def distogram(self):
        return jnp.exp(self.log_distogram)

    @property
    def distogram_bin_edges(self):
        return jnp.linspace(2.0, 22.0, 65)

    @property
    def sample_distogram(self):
        if self.is_single_sample:
            cb, _ = self.atom24_samples[:, 4]
            ca, mask = self.atom24_samples[:, 1]
            cb = jnp.where(mask[:, 4, None], cb, ca)
            cb = cb[None]
        else:
            cb, _ = self.atom24_samples[..., 4, :]
            ca, mask = self.atom24_samples[..., 1, :]
            cb = jnp.where(mask[None, :, 4, None], cb, ca)
        dist = jnp.linalg.norm(cb[..., :, None, :] - cb[..., None, :, :], axis=-1)
        bin_centers = (self.distogram_bin_edges[:-1] + self.distogram_bin_edges[1:]) / 2
        dist = jnp.argmax(dist[..., None] - bin_centers, axis=-1)
        dist = jax.nn.one_hot(dist, 64).mean(axis=0)
        return dist

    def sample_distogram_nll(self):
        gt = self.sample_distogram
        predicted = self.log_distogram
        return -(gt * predicted).sum(axis=-1).mean()

    @property
    def residue_index(self) -> jax.Array:
        return self.data["features"]["residue_index"][0]
    
    @property
    def chain_index(self) -> jax.Array:
        return self.data["features"]["asym_id"][0]

    @chain_index.setter
    def set_chain_index(self, value: jax.Array):
        self.data["features"]["asym_id"] = value[None]
        return self.chain_index

    def contact_probability(self, contact_distance=10.0) -> jax.Array:
        """Compute the distogram predicted contact probability for each
        pair of amino acids.
        
        Args:
            contact_distance: Contact distance cutoff in Angstroms. Default: 10.0.
        """
        edge_mask = self.distogram_bin_edges[1:] < contact_distance
        return (edge_mask * self.distogram).sum(axis=-1)
    
    def contact_entropy(self, contact_distance=14.0) -> jax.Array:
        """Compute the distogram contact entropy for each pair of amino acids.
        This is the metric used for optimization in BoltzDesign-1, Cho et al. 2025 (10.1101/2025.04.06.647261).
        """
        edge_mask = self.distogram_bin_edges[1:] < contact_distance
        distogram_clipped = jax.nn.softmax(self.log_distogram - 1e9 * (1 - edge_mask), axis=-1)
        distogram_clipped = jnp.where(edge_mask, distogram_clipped, 0)
        return -(distogram_clipped * self.log_distogram).sum(axis=-1)

    def contact_score(self, contact_distance=14.0, min_resi_distance=10, num_contacts=25):
        entropy = self.contact_entropy(contact_distance=contact_distance)
        resi_dist = abs(self.residue_index[:, None] - self.residue_index[None, :])
        other_chain = self.chain_index[:, None] != self.chain_index[None, :]
        entropy = jnp.where((resi_dist >= min_resi_distance) + other_chain > 0, entropy, 1e6)
        contact_score = entropy.sort(axis=1)[:, :num_contacts].mean(axis=-1).mean()
        return contact_score

    def chain_contact_score(self, target_chain, source_chain=0, contact_distance=14.0,
                            min_resi_distance=10, num_contacts=25):
        if isinstance(target_chain, int):
            target_chain = [target_chain]
        target_chain = jnp.array(target_chain, dtype=jnp.int32)
        selector = (self.chain_index[:, None] == target_chain).any(axis=1)
        if source_chain is None:
            binder_selector = ~selector
        else:
            if isinstance(source_chain, int):
                source_chain = [source_chain]
            source_chain = jnp.array(source_chain, dtype=jnp.int32)
            binder_selector = (self.chain_index[:, None] == source_chain).any(axis=1)
        return self.index_contact_score(
            selector, binder_selector, contact_distance=contact_distance,
            min_resi_distance=min_resi_distance, num_contacts=num_contacts)

    def index_contact_score(self, target_index, source_index, contact_distance=14.0,
                            min_resi_distance=10, num_contacts=25):
        entropy = self.contact_entropy(contact_distance=contact_distance)
        resi_dist = abs(self.residue_index[:, None] - self.residue_index[None, :])
        other_chain = self.chain_index[:, None] != self.chain_index[None, :]
        entropy = jnp.where((resi_dist >= min_resi_distance) + other_chain > 0, entropy, 1e6)
        entropy = jnp.where(target_index[None, :], entropy, 1e6)
        # NOTE: Ensure that only valid entries are averaged
        sorted_entropy = entropy.sort(axis=1)[:, :num_contacts]
        is_valid_entropy = sorted_entropy < 1e5
        mean_entropy = (sorted_entropy * is_valid_entropy).sum(axis=-1) / jnp.maximum(1, is_valid_entropy.sum(axis=-1))
        contact_score = (mean_entropy * source_index).sum()
        contact_score /= jnp.maximum(1, source_index.sum())
        return contact_score

    def intra_contact_score(self, chain, contact_distance=14.0,
                           min_resi_distance=10, num_contacts=25):
        return self.chain_contact_score(
            chain, chain, contact_distance=contact_distance,
            min_resi_distance=min_resi_distance, num_contacts=num_contacts)

    @property
    def atom_to_token(self):
        return self.data["features"]["atom_to_token"][0]

    @property
    def atom24_samples(self):
        sample = self.data["samples"]
        sample24, mask24 = self._transform_sampled(sample, num_atoms=24)
        return sample24, mask24

    @property
    def atom24(self):
        if self.is_single_sample:
            return self.atom24_samples
        sample24, mask24 = self.atom24_samples
        pae = self.pae.mean(axis=(-1, -2))
        best_pae = jnp.argmin(pae, axis=0)
        return sample24[best_pae], mask24

    @property
    def atom14(self):
        atom24, mask24 = self.atom24
        return atom24[:, :14], mask24[:, :14]

    @property
    def atom4(self):
        atom24, mask24 = self.atom24
        return atom24[:, :4]

    @property
    def cb_samples(self):
        atom24, mask24 = self.atom24_samples
        return jax.vmap(get_contact_atom, (0, None), 0)(atom24, self.data["features"]["mol_type"][0])

    @property
    def cb(self):
        atom24, mask24 = self.atom24
        return get_contact_atom(atom24, self.data["features"]["mol_type"][0])

    def _transform_sampled(self, sampled_property, num_atoms=24):
        if self.is_single_sample:
            sampled_property = sampled_property[0]
        else:
            sampled_property = sampled_property[:, 0]
            sampled_property = jnp.moveaxis(sampled_property, 0, -1)
        sample24, mask24 = atom_array_to_atomX(
            sampled_property, self.atom_to_token, num_atoms=num_atoms)
        if not self.is_single_sample:
            sample24 = jnp.moveaxis(sample24, -1, 0)
        return sample24, mask24

    # TODO
    def _broadcast_sampled(self, sampled_property, num_atoms=24):
        if self.is_single_sample:
            sampled_property = sampled_property[0]
        else:
            sampled_property = sampled_property[:, 0]
            sampled_property = jnp.moveaxis(sampled_property, 0, -1)
        sample24, mask24 = broadcast_array_to_atomX(
            sampled_property, self.atom_to_token, num_atoms=num_atoms)
        if not self.is_single_sample:
            sample24 = jnp.moveaxis(sample24, -1, 0)
        return sample24, mask24

    @property
    def is_single_sample(self):
        return len(self.data["samples"].shape) == 3

    @property
    def plddt_logits(self):
        plddt_logits = self.data["confidence"].plddt_logits
        if self.is_single_sample:
            plddt_logits = plddt_logits[0]
        else:
            plddt_logits = plddt_logits[:, 0]
        return plddt_logits
        #plddt24, mask24 = self._transform_sampled(self.data["confidence"].plddt_logits, num_atoms=24)
        #return plddt24, mask24

    @property
    def plddt(self):
        plddt = self.data["confidence"].plddt
        if self.is_single_sample:
            plddt = plddt[0]
        else:
            plddt = plddt[:, 0]
        return plddt
        # plddt24, mask24 = self._transform_sampled(self.data["confidence"].plddt, num_atoms=24)
        # plddt = (plddt24 * mask24).sum(axis=-1) / jnp.maximum(1, mask24.sum(axis=-1))
        # return plddt
    
    @property
    def pae_logits(self):
        if self.is_single_sample:
            return self.data["confidence"].pae_logits[0]
        return self.data["confidence"].pae_logits[:, 0]
    
    @property
    def pae(self):
        if self.is_single_sample:
            return jnp.fill_diagonal(self.data["confidence"].pae[0], 0.0, inplace=False) / 32
        return jax.vmap(lambda x: jnp.fill_diagonal(x, 0.0, inplace=False))(self.data["confidence"].pae[:, 0]) / 32
    
    @property
    def ipae(self):
        pae = self.pae
        chain = self.chain_index
        other_chain = chain[:, None] != chain[None, :]
        return (pae * other_chain).sum() / jnp.maximum(1, other_chain.sum())

    def chain_selector(self, target_chain, source_chain=None):
        if source_chain is None:
            source_chain = target_chain
        chain = self.chain_index
        selector = (
            (chain == target_chain)[None, :]
            + (chain == source_chain)[:, None] > 0
        )
        return selector

    def chain_pae(self, target_chain, source_chain=None):
        selector = self.chain_selector(target_chain, source_chain)
        return (self.pae * selector).sum(axis=(-1, -2)) / jnp.maximum(1, selector.sum())

    @property
    def ptm(self):
        return self.ptm_matrix().mean(axis=-1).max()

    @property
    def iptm(self):
        return self.index_iptm(self.chain_index)
        # ptm = self.ptm_matrix() # FIXME: adjust L?
        # chain = self.chain_index
        # other_chain = chain[:, None] != chain[None, :]
        # return ((ptm * other_chain).sum(-1) / jnp.maximum(1, other_chain.sum(-1))).max()

    def ipsae(self, chain_index=None, raw_pae_threshold=15.0):
        raw_pae = self.pae * 32
        chain = chain_index
        if chain is None:
            chain = self.chain_index
        mask = chain[:, None] != chain[None, :]
        mask *= raw_pae < raw_pae_threshold
        mask_count = jnp.maximum(1, mask.astype(jnp.int32).sum(axis=-1))
        ptm = self.ptm_matrix(L = mask_count)
        ipsae = (ptm * mask / mask_count[..., None]).sum(axis=-1).max()
        return ipsae

    @property
    def ptm_score(self):
        return self.index_ptm_score(self.chain_index)

    def index_iptm(self, chain_index=None):
        if chain_index is None:
            chain_index = self.chain_index
        ptm = self.ptm_matrix()
        other_chain = chain_index[:, None] != chain_index[None, :]
        return ((ptm * other_chain).sum(-1) / jnp.maximum(1, other_chain.sum(-1))).max()

    def index_ptm_score(self, chain_index=None):
        if chain_index is None:
            chain_index = self.chain_index
        pae_logits = self.pae_logits
        num_aa = pae_logits.shape[0]
        if not self.is_single_sample:
            num_aa = pae_logits.shape[1]
        num_aa = max(num_aa, 19)

        d0 = 1.24 * (num_aa - 15) ** (1.0 / 3) - 1.8
        bin_centers = (jnp.arange(64) / 64 + 1 / 128) * 32

        scale = 1.0 / (1 + bin_centers ** 2 / d0 ** 2)
        score = jax.nn.logsumexp(a=pae_logits, b=scale, axis=-1)
        if not self.is_single_sample:
            score = score.mean(axis=0)
        other_chain = chain_index[:, None] != chain_index[None, :]
        score = (score * other_chain).sum() / jnp.maximum(1, other_chain.sum())
        return score


    def chain_ptm(self, target_chain, source_chain=None):
        ptm = self.ptm_matrix()
        selector = self.chain_selector(target_chain, source_chain)
        mean_ptm = (ptm * selector).sum(axis=-1) / jnp.maximum(1, selector.sum(axis=-1))
        max_ptm = mean_ptm.max()
        return max_ptm

    def ptm_matrix(self, L=None):
        logits = self.pae_logits
        probabilities = jax.nn.softmax(logits, axis=-1)
        # if not self.is_single_sample:
        #     probabilities = probabilities.mean(axis=0)
        bin_centers = jnp.arange(probabilities.shape[-1]) / probabilities.shape[-1]
        bin_centers += 1 / probabilities.shape[-1] / 2
        bin_centers *= 32
        if L is None:
            L = probabilities.shape[0]
        d0 = 1.24 * (jnp.maximum(27, L) - 15) ** (1 / 3) - 1.8
        d0 = jnp.where(L < 27, 1, d0)
        # if L was an array, broadcast it
        if isinstance(L, jax.Array) and len(L.shape) >= 1:
            d0 = d0[..., None, None]
        return (probabilities * (1 / (1 + (bin_centers / d0) ** 2))).sum(axis=-1)

    @property
    def restype(self):
        # get one-hot residue type
        res_type_one_hot = self.data["features"]["res_type"][0]
        res_type = jnp.argmax(res_type_one_hot, axis=-1)
        # flip order, such that 20 amino acids are first
        # and padding / gap tokens are the last two tokens
        num_types = res_type_one_hot.shape[-1]
        res_type = res_type - 2
        res_type = jnp.where(res_type < 0, res_type + num_types, res_type)
        return res_type

    def to_data(self, return_samples=False) -> DesignData:
        """Convert an AFResult to DesignData."""
        if return_samples:
            atom24, mask24 = self.atom24_samples
        else:
            atom24, mask24 = self.atom24
        return DesignData(data=dict(
            atom_positions=atom24,
            atom_mask=mask24,
            aa=self.restype,
            mask=mask24.any(axis=1),
            residue_index=self.residue_index,
            chain_index=self.chain_index,
            batch_index=jnp.zeros_like(self.residue_index),
            plddt=self.plddt.mean(axis=0) if len(self.plddt.shape) == 2 else self.plddt,
        )).untie()


@dataclass
class JoltzPrediction:
    data: Any
    writer: Joltz2Writer
    @property
    def result(self):
        return JoltzResult(data=self.data)

    def save_pdb(self, path, sample_index=0):
        is_multisample = len(self.data["samples"].shape) == 4
        if is_multisample:
            self.writer.save_pdb(path, self.data["samples"][sample_index],
                                 plddt=self.data["confidence"].plddt[sample_index])
        else:
            self.writer.save_pdb(path, self.data["samples"],
                                 plddt=self.data["confidence"].plddt)

    def save_cif(self, path, sample_index=0):
        self.writer.save_cif(path, self.data["samples"][sample_index][None],
                             plddt=self.data["confidence"].plddt[sample_index][None])
