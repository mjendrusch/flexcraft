"""Pure-JAX masked confidence and distogram metrics.

This module provides the pure-JAX implementation backing ``<model>Result``
containers (``AFResult``, ``AF3LikeResult``, ``JoltzResult``, ``ProtenixResult``,
and ``ESMFold2Result``). Callers holding a ``<model>Result`` instance should use
``<model>Result.index_*``, ``<model>Result.chain_*``, and property methods
rather than calling ``compute_*`` functions directly on tensors extracted from a
``<model>Result``.
"""

import jax
import jax.numpy as jnp


def _normalize_chain_index(chain_index: jax.Array) -> jax.Array:
    """Convert chain indices (including boolean masks) to integer JAX arrays."""
    chain_arr = jnp.asarray(chain_index)
    if chain_arr.dtype == jnp.bool_:
        return chain_arr.astype(jnp.int32)
    return chain_arr


def _in_chains(chain_index: jax.Array, chains) -> jax.Array:
    """Return a boolean residue mask matching a scalar or 1D collection of chain IDs."""
    chains_arr = jnp.asarray(chains)
    if chains_arr.dtype == jnp.bool_:
        chains_arr = chains_arr.astype(jnp.int32)
    else:
        chains_arr = chains_arr.astype(chain_index.dtype)
    if chains_arr.ndim == 0:
        return chain_index == chains_arr
    return (chain_index[:, None] == chains_arr[None, :]).any(axis=-1)


def _resolve_pae_bin_centers(
    num_bins: int,
    max_pae: float,
    bin_centers: jax.Array | None,
) -> jax.Array:
    """Return explicit float32 PAE bin centers or uniform defaults."""
    if bin_centers is None:
        return pae_bin_centers(num_bins=num_bins, max_pae=max_pae)
    return jnp.asarray(bin_centers, dtype=jnp.float32)


def d0_from_length(
    num_residues,
    *,
    min_length: float = 19.0,
    ipsae: bool = False,
) -> jax.Array:
    """Compute the TM-score distance scale d0 from an active residue count.

    Args:
        num_residues: Scalar or array of active residue counts.
        min_length: Minimum effective length when ``ipsae=False``.
        ipsae: If True, apply the Dunbrack (2025) ipSAE floor (``d0 = 1.0`` for ``L < 27``).

    Returns:
        Distance scale ``d0`` in Angstroms.
    """
    length = jnp.asarray(num_residues, dtype=jnp.float32)
    d0_standard = 1.24 * (jnp.maximum(length, min_length) - 15.0) ** (1.0 / 3.0) - 1.8
    d0_ipsae_raw = 1.24 * (jnp.maximum(27.0, length) - 15.0) ** (1.0 / 3.0) - 1.8
    d0_ipsae = jnp.where(length < 27.0, 1.0, d0_ipsae_raw)
    if isinstance(ipsae, bool):
        return d0_ipsae if ipsae else d0_standard
    return jnp.where(ipsae, d0_ipsae, d0_standard)


def pae_bin_centers(num_bins: int = 64, max_pae: float = 32.0) -> jax.Array:
    """Return uniform PAE histogram bin centers in Angstroms."""
    return (jnp.arange(num_bins, dtype=jnp.float32) + 0.5) * (max_pae / num_bins)


def ptm_matrix_from_logits(
    pae_logits: jax.Array,
    num_residues=None,
    *,
    token_mask: jax.Array | None = None,
    max_pae: float = 32.0,
    bin_centers: jax.Array | None = None,
    ipsae: bool = False,
) -> jax.Array:
    """Compute expected pairwise TM-score matrix from PAE logits.

    Args:
        pae_logits: Logits of shape ``(L, L, B)`` or ``(S, L, L, B)``.
        num_residues: Optional effective length scalar or per-row array.
        token_mask: Optional boolean mask of shape ``(L,)`` marking valid tokens.
        max_pae: Maximum PAE range in Angstroms.
        bin_centers: Optional explicit PAE bin centers in Angstroms ``(B,)``.
        ipsae: Whether to use ipSAE ``d0`` clamping.

    Returns:
        Expected TM-score matrix of shape ``(L, L)`` or ``(S, L, L)``.
    """
    valid = None
    if token_mask is not None:
        valid = jnp.asarray(token_mask, dtype=jnp.bool_)
    if num_residues is None:
        if valid is not None:
            num_residues = valid.astype(jnp.float32).sum()
        else:
            num_residues = pae_logits.shape[-2]

    d0 = d0_from_length(num_residues, ipsae=ipsae)
    if d0.ndim >= 1:
        if d0.ndim == pae_logits.ndim - 1 and d0.shape[-1] == 1:
            d0 = d0[..., None]
        elif (
            pae_logits.ndim == 4
            and d0.ndim == 2
            and d0.shape[-1] == 1
            and d0.shape[0] == pae_logits.shape[-2]
            and pae_logits.shape[-2] > 1
        ):
            d0 = d0[None, :, :, None]
        else:
            d0 = d0[..., None, None]

    centers = _resolve_pae_bin_centers(pae_logits.shape[-1], max_pae, bin_centers)
    probabilities = jax.nn.softmax(pae_logits, axis=-1)
    tm_mat = (probabilities * (1.0 / (1.0 + (centers / d0) ** 2))).sum(axis=-1)
    if valid is not None:
        pair_mask = valid[:, None] & valid[None, :]
        tm_mat = jnp.where(pair_mask, tm_mat, 0.0)
    return tm_mat


def chain_pair_mask(
    chain_index: jax.Array,
    target_chain,
    source_chain=None,
    *,
    token_mask: jax.Array | None = None,
) -> jax.Array:
    """Build a 2D ``(source_chain, target_chain)`` boolean pair mask excluding padding."""
    if source_chain is None:
        source_chain = target_chain
    chain_arr = _normalize_chain_index(chain_index)
    valid = chain_arr >= 0
    if token_mask is not None:
        valid = valid & jnp.asarray(token_mask, dtype=jnp.bool_)
    is_source = _in_chains(chain_arr, source_chain) & valid
    is_target = _in_chains(chain_arr, target_chain) & valid
    return is_source[:, None] & is_target[None, :]


def inter_chain_mask(
    chain_index: jax.Array,
    *,
    token_mask: jax.Array | None = None,
) -> jax.Array:
    """Build a 2D inter-chain boolean pair mask excluding padded tokens."""
    chain_arr = _normalize_chain_index(chain_index)
    valid = chain_arr >= 0
    if token_mask is not None:
        valid = valid & jnp.asarray(token_mask, dtype=jnp.bool_)
    return (chain_arr[:, None] != chain_arr[None, :]) & valid[:, None] & valid[None, :]


def compute_ptm(
    pae_logits: jax.Array,
    *,
    mask: jax.Array | None = None,
    token_mask: jax.Array | None = None,
    max_pae: float = 32.0,
    bin_centers: jax.Array | None = None,
) -> jax.Array:
    """Compute complex-wide or subcomplex masked pTM from PAE logits per sample."""
    num_tokens = pae_logits.shape[-2]
    if token_mask is None:
        valid = jnp.ones((num_tokens,), dtype=jnp.bool_)
    else:
        valid = jnp.asarray(token_mask, dtype=jnp.bool_)
    if mask is not None:
        valid = valid & jnp.asarray(mask, dtype=jnp.bool_)
    pair_mask = valid[:, None] & valid[None, :]
    tm_mat = ptm_matrix_from_logits(
        pae_logits,
        num_residues=valid.astype(jnp.float32).sum(),
        token_mask=valid,
        max_pae=max_pae,
        bin_centers=bin_centers,
        ipsae=False,
    )
    denom = jnp.maximum(1.0, pair_mask.astype(jnp.float32).sum(axis=-1))
    per_query = jnp.where(pair_mask, tm_mat, 0.0).sum(axis=-1) / denom
    per_query = jnp.where(valid, per_query, 0.0)
    return per_query.max(axis=-1)


def compute_iptm(
    pae_logits: jax.Array,
    chain_index: jax.Array,
    *,
    token_mask: jax.Array | None = None,
    max_pae: float = 32.0,
    bin_centers: jax.Array | None = None,
) -> jax.Array:
    """Compute masked interface pTM (ipTM) across distinct chains per sample."""
    pair_mask = inter_chain_mask(chain_index, token_mask=token_mask)
    tm_mat = ptm_matrix_from_logits(
        pae_logits,
        token_mask=token_mask,
        max_pae=max_pae,
        bin_centers=bin_centers,
        ipsae=False,
    )
    row_counts = pair_mask.astype(jnp.float32).sum(axis=-1)
    per_query = jnp.where(pair_mask, tm_mat, 0.0).sum(axis=-1) / jnp.maximum(1.0, row_counts)
    per_query = jnp.where(row_counts > 0, per_query, 0.0)
    return per_query.max(axis=-1)


def compute_chain_ptm(
    pae_logits: jax.Array,
    chain_index: jax.Array,
    target_chain,
    source_chain=None,
    *,
    token_mask: jax.Array | None = None,
    max_pae: float = 32.0,
    bin_centers: jax.Array | None = None,
) -> jax.Array:
    """Compute masked pTM over a ``(source_chain, target_chain)`` block per sample."""
    pair_mask = chain_pair_mask(
        chain_index, target_chain, source_chain=source_chain, token_mask=token_mask
    )
    tm_mat = ptm_matrix_from_logits(
        pae_logits,
        token_mask=token_mask,
        max_pae=max_pae,
        bin_centers=bin_centers,
        ipsae=False,
    )
    row_counts = pair_mask.astype(jnp.float32).sum(axis=-1)
    per_query = jnp.where(pair_mask, tm_mat, 0.0).sum(axis=-1) / jnp.maximum(1.0, row_counts)
    per_query = jnp.where(row_counts > 0, per_query, 0.0)
    return per_query.max(axis=-1)


def compute_ipae(
    pae: jax.Array,
    chain_index: jax.Array,
    *,
    token_mask: jax.Array | None = None,
) -> jax.Array:
    """Compute masked mean inter-chain PAE per sample."""
    pair_mask = inter_chain_mask(chain_index, token_mask=token_mask)
    denom = jnp.maximum(1.0, pair_mask.astype(jnp.float32).sum())
    return jnp.where(pair_mask, pae, 0.0).sum(axis=(-1, -2)) / denom


def compute_min_ipae(
    pae: jax.Array,
    chain_index: jax.Array,
    *,
    token_mask: jax.Array | None = None,
) -> jax.Array:
    """Compute masked minimum inter-chain PAE per sample (``inf`` if no interface pairs)."""
    pair_mask = inter_chain_mask(chain_index, token_mask=token_mask)
    inf = jnp.asarray(jnp.inf, dtype=jnp.result_type(pae, jnp.float32))
    min_val = jnp.where(pair_mask, pae, inf).min(axis=(-1, -2))
    return jnp.where(pair_mask.any(), min_val, inf)


def compute_chain_pae(
    pae: jax.Array,
    chain_index: jax.Array,
    target_chain,
    source_chain=None,
    *,
    token_mask: jax.Array | None = None,
) -> jax.Array:
    """Compute masked mean PAE over a ``(source_chain, target_chain)`` block per sample."""
    pair_mask = chain_pair_mask(
        chain_index, target_chain, source_chain=source_chain, token_mask=token_mask
    )
    denom = jnp.maximum(1.0, pair_mask.astype(jnp.float32).sum())
    return jnp.where(pair_mask, pae, 0.0).sum(axis=(-1, -2)) / denom


def compute_ipsae(
    pae_logits: jax.Array,
    pae: jax.Array,
    chain_index: jax.Array,
    *,
    token_mask: jax.Array | None = None,
    raw_pae_threshold: float = 15.0,
    max_pae: float = 32.0,
    bin_centers: jax.Array | None = None,
    normalized_pae: bool | None = None,
) -> jax.Array:
    """Compute masked ipSAE (Dunbrack 2025) from PAE logits and PAE matrix per sample."""
    inter_mask = inter_chain_mask(chain_index, token_mask=token_mask)
    num_tokens = pae.shape[-1]
    off_diag = ~jnp.eye(num_tokens, dtype=jnp.bool_)
    centers = _resolve_pae_bin_centers(pae_logits.shape[-1], max_pae, bin_centers)

    if normalized_pae is True:
        raw_pae = pae * max_pae
    elif normalized_pae is False:
        raw_pae = pae
    elif normalized_pae is None:
        if token_mask is not None:
            valid_tokens = jnp.asarray(token_mask, dtype=jnp.bool_)
            valid_off_diag = valid_tokens[:, None] & valid_tokens[None, :] & off_diag
        else:
            valid_off_diag = off_diag
        expected_raw = (jax.nn.softmax(pae_logits, axis=-1) * centers).sum(axis=-1)
        pae_scaled = pae * max_pae
        err_raw = jnp.where(valid_off_diag, jnp.abs(pae - expected_raw), 0.0).sum(
            axis=(-1, -2), keepdims=True
        )
        err_scaled = jnp.where(valid_off_diag, jnp.abs(pae_scaled - expected_raw), 0.0).sum(
            axis=(-1, -2), keepdims=True
        )
        raw_pae = jnp.where(err_scaled < err_raw, pae_scaled, pae)
    else:
        raw_pae = jnp.where(normalized_pae, pae * max_pae, pae)

    raw_pae = jnp.where(off_diag, raw_pae, 0.0)
    pair_mask = inter_mask & (raw_pae < raw_pae_threshold)
    row_counts = pair_mask.astype(jnp.float32).sum(axis=-1)
    denom = jnp.maximum(1.0, row_counts)
    tm_mat = ptm_matrix_from_logits(
        pae_logits,
        num_residues=denom,
        token_mask=token_mask,
        max_pae=max_pae,
        bin_centers=centers,
        ipsae=True,
    )
    per_query = jnp.where(pair_mask, tm_mat, 0.0).sum(axis=-1) / denom
    per_query = jnp.where(row_counts > 0, per_query, 0.0)
    return per_query.max(axis=-1)


def compute_ptm_score(
    pae_logits: jax.Array,
    chain_index: jax.Array,
    *,
    token_mask: jax.Array | None = None,
    max_pae: float = 32.0,
    bin_centers: jax.Array | None = None,
) -> jax.Array:
    """Compute masked log-expected inter-chain TM energy score per sample."""
    if token_mask is not None:
        num_residues = jnp.asarray(token_mask, dtype=jnp.float32).sum()
    else:
        num_residues = pae_logits.shape[-2]
    d0 = d0_from_length(num_residues, ipsae=False)
    centers = _resolve_pae_bin_centers(pae_logits.shape[-1], max_pae, bin_centers)
    scale = 1.0 / (1.0 + (centers / d0) ** 2)
    score = jax.nn.logsumexp(a=pae_logits, b=scale, axis=-1)
    pair_mask = inter_chain_mask(chain_index, token_mask=token_mask)
    denom = jnp.maximum(1.0, pair_mask.astype(jnp.float32).sum())
    return jnp.where(pair_mask, score, 0.0).sum(axis=(-1, -2)) / denom


def distogram_expected_distance(
    distogram_probs: jax.Array,
    bin_centers: jax.Array,
) -> jax.Array:
    """Compute expected pairwise distances from distogram probabilities and bin centers."""
    return (distogram_probs * bin_centers).sum(axis=-1)


def bin_pair_distances(
    distances: jax.Array,
    bin_edges: jax.Array,
) -> jax.Array:
    """Assign pairwise distances to the nearest distogram bin center."""
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    return jnp.argmin(jnp.abs(distances[..., None] - bin_centers), axis=-1)
