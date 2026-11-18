"""Distogram and contact divergence losses for hallucination."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp

if TYPE_CHECKING:
    from flexcraft.data.data import DesignData
    from flexcraft.structure.boltz._result import JoltzResult


def _valid_token_mask(
    length: int,
    *,
    mask: jax.Array | None = None,
    chain_index: jax.Array | None = None,
    objects: tuple[Any, ...] = (),
) -> jax.Array:
    valid = jnp.ones((length,), dtype=jnp.bool_)
    for obj in objects:
        if obj is None:
            continue
        tok_mask = getattr(obj, "token_pad_mask", None)
        if tok_mask is not None:
            valid = valid & jnp.asarray(tok_mask, dtype=jnp.bool_)
        elif isinstance(obj, dict) and "mask" in obj:
            valid = valid & jnp.asarray(obj["mask"], dtype=jnp.bool_)
        elif hasattr(obj, "data") and isinstance(getattr(obj, "data", None), dict) and "mask" in obj.data:
            valid = valid & jnp.asarray(obj.data["mask"], dtype=jnp.bool_)

        obj_chain = getattr(obj, "chain_index", None)
        if obj_chain is not None:
            chain_arr = jnp.asarray(obj_chain)
            if chain_arr.ndim == 1 and chain_arr.shape[0] == length and chain_arr.dtype != jnp.bool_:
                valid = valid & (chain_arr >= 0)

    if mask is not None:
        mask_arr = jnp.asarray(mask, dtype=jnp.bool_)
        if mask_arr.ndim == 1:
            valid = valid & mask_arr

    if chain_index is not None:
        chain_arr = jnp.asarray(chain_index)
        if chain_arr.dtype != jnp.bool_:
            valid = valid & (chain_arr >= 0)

    return valid


def _valid_pair_mask(
    shape_2d: tuple[int, int],
    *,
    mask: jax.Array | None = None,
    chain_index: jax.Array | None = None,
    objects: tuple[Any, ...] = (),
) -> jax.Array:
    valid = _valid_token_mask(
        shape_2d[0], mask=mask, chain_index=chain_index, objects=objects
    )
    pair_mask = valid[:, None] & valid[None, :]
    if mask is not None:
        mask_arr = jnp.asarray(mask, dtype=jnp.bool_)
        if mask_arr.ndim == 2:
            pair_mask = pair_mask & mask_arr
    return pair_mask


def _to_probs_and_log_probs(arr: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Convert either probabilities or log-probabilities/logits into ``(P, log P)``."""
    arr = jnp.asarray(arr, dtype=jnp.float32)
    row_sums = arr.sum(axis=-1)
    active = jnp.abs(row_sums) > 1e-6
    looks_like_probs = (
        (jnp.min(arr) >= -1e-6)
        & jnp.any(active)
        & jnp.all(jnp.where(active, jnp.abs(row_sums - 1.0) < 1e-3, True))
    )
    log_from_logits = jax.nn.log_softmax(arr, axis=-1)
    prob_from_logits = jax.nn.softmax(arr, axis=-1)
    clamped = jnp.maximum(arr, 0.0)
    safe_probs = jnp.where(clamped > 0, clamped, 1.0)
    log_from_probs = jnp.where(clamped > 0, jnp.log(safe_probs), -1e30)
    probs = jnp.where(looks_like_probs, clamped, prob_from_logits)
    log_probs = jnp.where(looks_like_probs, log_from_probs, log_from_logits)
    return probs, log_probs


def distogram_contacts(
    result: Any,
    target: DesignData,
    contact_distance: float = 8.0,
) -> jax.Array:
    is_contact = target.contacts(contact_distance=contact_distance, atom_index=4)
    pair_mask = _valid_pair_mask(
        is_contact.shape[-2:], objects=(result, target)
    )
    is_contact = is_contact & pair_mask
    entropy = result.contact_entropy(contact_distance=contact_distance)
    out = jnp.where(pair_mask, is_contact * entropy, 0.0)
    if is_contact.ndim == 3:
        out = out.mean(axis=0)
    return out


def residue_helix_contacts(result: JoltzResult) -> jax.Array:
    resi = result.residue_index
    chain = result.chain_index
    valid = _valid_token_mask(resi.shape[0], chain_index=chain, objects=(result,))
    pair_mask = valid[:, None] & valid[None, :]
    helix_mask = (
        (resi[:, None] - resi[None, :] == 3)
        & (chain[:, None] == chain[None, :])
        & pair_mask
    )
    prob = result.contact_probability(contact_distance=6.0)
    per_residue = -jnp.log((prob * helix_mask).sum(axis=-1) + 1e-8)
    return jnp.where(valid, per_residue, 0.0)


def helix_contacts(result: JoltzResult) -> jax.Array:
    valid = _valid_token_mask(
        result.residue_index.shape[0],
        chain_index=result.chain_index,
        objects=(result,),
    )
    per_residue = residue_helix_contacts(result)
    return per_residue.sum() / jnp.maximum(1.0, valid.astype(jnp.float32).sum())


def distogram_jsd(
    x: Any,
    y: Any,
    *,
    mask: jax.Array | None = None,
    chain_index: jax.Array | None = None,
) -> jax.Array:
    """Compute pairwise Jensen-Shannon divergence ``0.5 * KL(P || M) + 0.5 * KL(Q || M)``."""
    x_arr = x.log_distogram if hasattr(x, "log_distogram") else x
    y_arr = y.log_distogram if hasattr(y, "log_distogram") else y
    p, log_p = _to_probs_and_log_probs(x_arr)
    q, log_q = _to_probs_and_log_probs(y_arr)
    log_m = jnp.logaddexp(log_p, log_q) - jnp.log(2.0)
    kl_pm = jnp.where(p > 0, p * (log_p - log_m), 0.0).sum(axis=-1)
    kl_qm = jnp.where(q > 0, q * (log_q - log_m), 0.0).sum(axis=-1)
    jsd = 0.5 * (kl_pm + kl_qm)
    if jsd.ndim >= 2:
        pair_mask = _valid_pair_mask(
            jsd.shape[-2:], mask=mask, chain_index=chain_index, objects=(x, y)
        )
        jsd = jnp.where(pair_mask, jsd, 0.0)
    elif mask is not None or chain_index is not None:
        valid = _valid_token_mask(
            jsd.shape[0], mask=mask, chain_index=chain_index, objects=(x, y)
        )
        jsd = jnp.where(valid, jsd, 0.0)
    return jsd


def _mask_residue_distance(
    jsd: jax.Array,
    residue_index: jax.Array | None,
    min_residue_distance: int,
) -> jax.Array:
    mask = jnp.ones(jsd.shape[-2:], dtype=jnp.bool_)
    if residue_index is not None:
        distance = jnp.abs(residue_index[:, None] - residue_index[None, :])
        mask = distance >= min_residue_distance
    return mask


def mean_distogram_jsd(
    x: Any,
    y: Any,
    residue_index: jax.Array | None = None,
    min_residue_distance: int = 0,
    *,
    mask: jax.Array | None = None,
    chain_index: jax.Array | None = None,
) -> jax.Array:
    jsd = distogram_jsd(x, y, mask=mask, chain_index=chain_index)
    pair_mask = _mask_residue_distance(jsd, residue_index, min_residue_distance)
    pair_mask = pair_mask & _valid_pair_mask(
        jsd.shape[-2:], mask=mask, chain_index=chain_index, objects=(x, y)
    )
    masked_jsd = jnp.where(pair_mask, jsd, 0.0)
    return masked_jsd.sum() / jnp.maximum(1.0, pair_mask.astype(jnp.float32).sum())


def max_distogram_jsd(
    x: Any,
    y: Any,
    residue_index: jax.Array | None = None,
    min_residue_distance: int = 0,
    *,
    mask: jax.Array | None = None,
    chain_index: jax.Array | None = None,
) -> jax.Array:
    jsd = distogram_jsd(x, y, mask=mask, chain_index=chain_index)
    pair_mask = _mask_residue_distance(jsd, residue_index, min_residue_distance)
    pair_mask = pair_mask & _valid_pair_mask(
        jsd.shape[-2:], mask=mask, chain_index=chain_index, objects=(x, y)
    )
    masked_jsd = jnp.where(pair_mask, jsd, 0.0)
    valid_rows = pair_mask.any(axis=-1)
    row_max = jnp.where(valid_rows, masked_jsd.max(axis=-1), 0.0)
    return row_max.sum() / jnp.maximum(1.0, valid_rows.astype(jnp.float32).sum())


def topk_distogram_jsd(
    x: Any,
    y: Any,
    k: int = 10,
    residue_index: jax.Array | None = None,
    min_residue_distance: int = 0,
    *,
    mask: jax.Array | None = None,
    chain_index: jax.Array | None = None,
) -> jax.Array:
    jsd = distogram_jsd(x, y, mask=mask, chain_index=chain_index)
    pair_mask = _mask_residue_distance(jsd, residue_index, min_residue_distance)
    pair_mask = pair_mask & _valid_pair_mask(
        jsd.shape[-2:], mask=mask, chain_index=chain_index, objects=(x, y)
    )
    masked_jsd = jnp.where(pair_mask, jsd, -1.0)
    k_eff = min(k, masked_jsd.shape[-1])
    sorted_vals = jnp.sort(masked_jsd, axis=-1)[..., -k_eff:]
    is_valid = sorted_vals >= 0.0
    top_vals = jnp.where(is_valid, sorted_vals, 0.0).sum(axis=-1) / jnp.maximum(
        1.0, is_valid.astype(jnp.float32).sum(axis=-1)
    )
    valid_rows = pair_mask.any(axis=-1)
    return jnp.where(valid_rows, top_vals, 0.0).sum() / jnp.maximum(
        1.0, valid_rows.astype(jnp.float32).sum()
    )


def differential_contact_entropy(
    x: JoltzResult,
    y: JoltzResult,
    x_slice=slice(None),
    y_slice=slice(None),
    k: int = 10,
    contact_distance: float = 10.0,
) -> jax.Array:
    x_entropy = x.contact_entropy(contact_distance=contact_distance)[x_slice, x_slice]
    y_entropy = y.contact_entropy(contact_distance=contact_distance)[y_slice, y_slice]
    delta = x_entropy - y_entropy
    sort_index = jnp.argsort(delta, axis=1)
    index = jnp.arange(sort_index.shape[0])[:, None]
    x_targets = sort_index[:, :k]
    y_targets = sort_index[:, -k:]
    return (x_entropy[index, x_targets].mean() + y_entropy[index, y_targets].mean()) / 2


def contact_jsd(
    x: JoltzResult,
    y: JoltzResult,
    contact_distance: float = 10.0,
    *,
    mask: jax.Array | None = None,
    chain_index: jax.Array | None = None,
) -> jax.Array:
    """Compute true Jensen-Shannon divergence between contact-conditioned distograms."""
    edge_mask = x.distogram_bin_edges[1:] < contact_distance
    x_logits_masked = jnp.where(edge_mask, x.log_distogram, -1e9)
    y_logits_masked = jnp.where(edge_mask, y.log_distogram, -1e9)
    p = jnp.where(edge_mask, jax.nn.softmax(x_logits_masked, axis=-1), 0.0)
    q = jnp.where(edge_mask, jax.nn.softmax(y_logits_masked, axis=-1), 0.0)
    p = p / jnp.maximum(p.sum(axis=-1, keepdims=True), 1e-12)
    q = q / jnp.maximum(q.sum(axis=-1, keepdims=True), 1e-12)
    m = 0.5 * (p + q)

    p_safe = jnp.where(p > 0, p, 1.0)
    q_safe = jnp.where(q > 0, q, 1.0)
    m_for_p = jnp.where(p > 0, m, 1.0)
    m_for_q = jnp.where(q > 0, m, 1.0)
    kl_pm = jnp.where(p > 0, p * (jnp.log(p_safe) - jnp.log(m_for_p)), 0.0).sum(axis=-1)
    kl_qm = jnp.where(q > 0, q * (jnp.log(q_safe) - jnp.log(m_for_q)), 0.0).sum(axis=-1)
    jsd = 0.5 * (kl_pm + kl_qm)

    pair_mask = _valid_pair_mask(
        jsd.shape[-2:], mask=mask, chain_index=chain_index, objects=(x, y)
    )
    return jnp.where(pair_mask, jsd, 0.0)
