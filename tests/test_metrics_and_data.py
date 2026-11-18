"""Unit and @jax.jit tests for confidence/geometry metrics, DesignData, MPNN, RNG, and hallucination."""

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from flexcraft.data.data import DesignData
from flexcraft.hallucination.losses.distogram import (
    contact_jsd,
    distogram_jsd,
    max_distogram_jsd,
    mean_distogram_jsd,
    topk_distogram_jsd,
)
from flexcraft.hallucination.opt import jit_loss_update, simplex_agpm, simplex_apgm_step
from flexcraft.sequence.mpnn import ProteinFeatures, ProteinMPNN, _verify_solmpnn_weights
import flexcraft.structure.metrics as metrics_pkg
import flexcraft.structure.metrics.confidence as confidence_mod
from flexcraft.structure.metrics import (
    RMSD,
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
    d0_from_length,
    distogram_expected_distance,
    inter_chain_mask,
    lddt,
    pae_bin_centers,
    position_lddt_ca,
    ptm_matrix_from_logits,
    sc_lddt,
)
from flexcraft.utils.rng import Keygen


def _make_synthetic_pae(num_valid: int = 12, num_padded: int = 16, num_bins: int = 64):
    """Build synthetic unpadded and padded PAE logits, PAE matrices, chain indices, and masks."""
    key = jax.random.key(42)
    k1, k2 = jax.random.split(key)
    logits_unpad = jax.random.normal(k1, (num_valid, num_valid, num_bins), dtype=jnp.float32)
    centers = pae_bin_centers(num_bins=num_bins, max_pae=32.0)
    pae_raw_unpad = (jax.nn.softmax(logits_unpad, axis=-1) * centers).sum(axis=-1)
    pae_raw_unpad = jnp.fill_diagonal(pae_raw_unpad, 0.0, inplace=False)
    chain_unpad = jnp.array([0] * (num_valid // 2) + [1] * (num_valid - num_valid // 2), dtype=jnp.int32)

    pad_len = num_padded - num_valid
    noise_pad = jax.random.normal(k2, (num_padded, num_padded, num_bins), dtype=jnp.float32) * 5.0
    logits_pad = noise_pad.at[:num_valid, :num_valid].set(logits_unpad)
    pae_raw_pad = jnp.full((num_padded, num_padded), 29.0, dtype=jnp.float32)
    pae_raw_pad = pae_raw_pad.at[:num_valid, :num_valid].set(pae_raw_unpad)
    chain_pad = jnp.concatenate([chain_unpad, jnp.full((pad_len,), -1, dtype=jnp.int32)])
    token_mask = jnp.arange(num_padded) < num_valid
    return logits_unpad, pae_raw_unpad, chain_unpad, logits_pad, pae_raw_pad, chain_pad, token_mask


def test_confidence_docstring_and_exports():
    """Verify confidence.py docstring and compute_min_ipae re-export."""
    assert "compute_min_ipae" in metrics_pkg.__all__
    doc = confidence_mod.__doc__ or ""
    assert "AFResult" in doc and "AF3LikeResult" in doc


def test_confidence_d0_and_pae_bin_centers_jit():
    """Verify d0_from_length (L < 27 vs L >= 27, ipsae True/False) and pae_bin_centers under @jax.jit."""
    d0_std_fn = jax.jit(lambda n: d0_from_length(n, ipsae=False))
    d0_ipsae_fn = jax.jit(lambda n: d0_from_length(n, ipsae=True))

    assert jnp.allclose(d0_ipsae_fn(10), 1.0, atol=1e-6)
    assert jnp.allclose(d0_ipsae_fn(26), 1.0, atol=1e-6)
    assert float(d0_std_fn(10)) < 1.0

    for length in (27, 40, 120):
        expected = 1.24 * (float(length) - 15.0) ** (1.0 / 3.0) - 1.8
        assert jnp.allclose(d0_std_fn(length), expected, atol=1e-5)
        assert jnp.allclose(d0_ipsae_fn(length), expected, atol=1e-5)

    arr_out = d0_ipsae_fn(jnp.array([12.0, 26.0, 27.0, 64.0]))
    assert arr_out.shape == (4,)
    assert jnp.allclose(arr_out[:2], 1.0)
    assert float(arr_out[2]) > 1.0

    centers = jax.jit(lambda: pae_bin_centers(64, 32.0))()
    assert centers.shape == (64,)
    assert jnp.allclose(centers[0], 0.25)
    assert jnp.allclose(centers[-1], 31.75)


def test_chain_and_inter_chain_masks_jit():
    """Verify AND-based chain_pair_mask and inter_chain_mask exclude asym_id == -1 and ~token_mask."""
    chain_index = jnp.array([0, 0, 1, 1, -1, 0], dtype=jnp.int32)
    token_mask = jnp.array([True, True, True, True, True, False])

    cp_mask = jax.jit(lambda c, m: chain_pair_mask(c, target_chain=1, source_chain=0, token_mask=m))(
        chain_index, token_mask
    )
    expected_cp = jnp.zeros((6, 6), dtype=jnp.bool_).at[0:2, 2:4].set(True)
    assert jnp.array_equal(cp_mask, expected_cp)

    intra_0 = jax.jit(lambda c, m: chain_pair_mask(c, target_chain=0, token_mask=m))(
        chain_index, token_mask
    )
    expected_intra_0 = jnp.zeros((6, 6), dtype=jnp.bool_).at[0:2, 0:2].set(True)
    assert jnp.array_equal(intra_0, expected_intra_0)

    multi_cp = jax.jit(lambda c, m: chain_pair_mask(c, target_chain=jnp.array([0, 1]), source_chain=0, token_mask=m))(
        chain_index, token_mask
    )
    expected_multi = jnp.zeros((6, 6), dtype=jnp.bool_).at[0:2, 0:4].set(True)
    assert jnp.array_equal(multi_cp, expected_multi)

    ic_mask = jax.jit(lambda c, m: inter_chain_mask(c, token_mask=m))(chain_index, token_mask)
    expected_ic = (
        jnp.zeros((6, 6), dtype=jnp.bool_)
        .at[0:2, 2:4].set(True)
        .at[2:4, 0:2].set(True)
    )
    assert jnp.array_equal(ic_mask, expected_ic)


def test_all_confidence_metrics_padding_invariance_jit():
    """Verify padding invariance of all confidence metrics (including compute_min_ipae & subcomplex compute_ptm) under @jax.jit."""
    (
        logits_unpad,
        pae_raw_unpad,
        chain_unpad,
        logits_pad,
        pae_raw_pad,
        chain_pad,
        token_mask,
    ) = _make_synthetic_pae(num_valid=12, num_padded=18)

    # 1. ptm_matrix_from_logits
    mat_unpad = jax.jit(ptm_matrix_from_logits)(logits_unpad)
    mat_pad = jax.jit(lambda l, m: ptm_matrix_from_logits(l, token_mask=m))(logits_pad, token_mask)
    assert jnp.allclose(mat_unpad, mat_pad[:12, :12], atol=1e-6)
    assert jnp.all(mat_pad[12:, :] == 0.0)
    assert jnp.all(mat_pad[:, 12:] == 0.0)

    # 2. compute_ptm (full complex + subcomplex mask using d0(valid.sum()))
    ptm_unpad = jax.jit(compute_ptm)(logits_unpad)
    ptm_pad = jax.jit(lambda l, m: compute_ptm(l, token_mask=m))(logits_pad, token_mask)
    assert ptm_unpad.shape == () and ptm_pad.shape == ()
    assert jnp.allclose(ptm_unpad, ptm_pad, atol=1e-6)

    # Subcomplex pTM on a 30-residue chain inside a 50-residue complex (so d0(30) != d0(50))
    big_logits = jax.random.normal(jax.random.key(77), (50, 50, 64), dtype=jnp.float32)
    big_logits_pad = jnp.pad(big_logits, ((0, 10), (0, 10), (0, 0)), constant_values=4.0)
    sub_mask_50 = jnp.arange(50) < 30
    sub_mask_60 = jnp.arange(60) < 30
    tok_mask_60 = jnp.arange(60) < 50
    sub_ptm_slice = jax.jit(compute_ptm)(big_logits[:30, :30])
    sub_ptm_masked = jax.jit(lambda l, sm: compute_ptm(l, mask=sm))(big_logits, sub_mask_50)
    sub_ptm_padded = jax.jit(lambda l, sm, tm: compute_ptm(l, mask=sm, token_mask=tm))(
        big_logits_pad, sub_mask_60, tok_mask_60
    )
    assert jnp.allclose(sub_ptm_slice, sub_ptm_masked, atol=1e-6)
    assert jnp.allclose(sub_ptm_slice, sub_ptm_padded, atol=1e-6)

    # 3. compute_iptm
    iptm_unpad = jax.jit(compute_iptm)(logits_unpad, chain_unpad)
    iptm_pad = jax.jit(lambda l, c, m: compute_iptm(l, c, token_mask=m))(logits_pad, chain_pad, token_mask)
    assert jnp.allclose(iptm_unpad, iptm_pad, atol=1e-6)

    # 4. compute_chain_ptm
    cptm_unpad = jax.jit(lambda l, c: compute_chain_ptm(l, c, 1, 0))(logits_unpad, chain_unpad)
    cptm_pad = jax.jit(lambda l, c, m: compute_chain_ptm(l, c, 1, 0, token_mask=m))(
        logits_pad, chain_pad, token_mask
    )
    assert jnp.allclose(cptm_unpad, cptm_pad, atol=1e-6)

    # 5. compute_ipae & compute_min_ipae
    ipae_unpad = jax.jit(compute_ipae)(pae_raw_unpad, chain_unpad)
    ipae_pad = jax.jit(lambda p, c, m: compute_ipae(p, c, token_mask=m))(pae_raw_pad, chain_pad, token_mask)
    assert jnp.allclose(ipae_unpad, ipae_pad, atol=1e-6)

    min_ipae_unpad = jax.jit(compute_min_ipae)(pae_raw_unpad, chain_unpad)
    min_ipae_pad = jax.jit(lambda p, c, m: compute_min_ipae(p, c, token_mask=m))(
        pae_raw_pad, chain_pad, token_mask
    )
    assert min_ipae_unpad.shape == () and min_ipae_pad.shape == ()
    assert jnp.allclose(min_ipae_unpad, min_ipae_pad, atol=1e-6)
    assert float(min_ipae_unpad) <= float(ipae_unpad)

    # Single-chain (no inter-chain pairs) returns +inf
    mono_chain = jnp.zeros_like(chain_pad)
    min_ipae_mono = jax.jit(lambda p, c, m: compute_min_ipae(p, c, token_mask=m))(
        pae_raw_pad, mono_chain, token_mask
    )
    assert min_ipae_mono.shape == () and jnp.isinf(min_ipae_mono)

    # 6. compute_chain_pae
    cpae_unpad = jax.jit(lambda p, c: compute_chain_pae(p, c, 1, 0))(pae_raw_unpad, chain_unpad)
    cpae_pad = jax.jit(lambda p, c, m: compute_chain_pae(p, c, 1, 0, token_mask=m))(
        pae_raw_pad, chain_pad, token_mask
    )
    assert jnp.allclose(cpae_unpad, cpae_pad, atol=1e-6)

    # 7. compute_ipsae (normalized_pae=None, True, False + rigid complex)
    ipsae_raw_unpad = jax.jit(compute_ipsae)(logits_unpad, pae_raw_unpad, chain_unpad)
    ipsae_norm_unpad = jax.jit(compute_ipsae)(logits_unpad, pae_raw_unpad / 32.0, chain_unpad)
    ipsae_explicit_true = jax.jit(lambda l, p, c: compute_ipsae(l, p, c, normalized_pae=True))(
        logits_unpad, pae_raw_unpad / 32.0, chain_unpad
    )
    ipsae_explicit_false = jax.jit(lambda l, p, c: compute_ipsae(l, p, c, normalized_pae=False))(
        logits_unpad, pae_raw_unpad, chain_unpad
    )
    ipsae_raw_pad = jax.jit(lambda l, p, c, m: compute_ipsae(l, p, c, token_mask=m))(
        logits_pad, pae_raw_pad, chain_pad, token_mask
    )
    ipsae_norm_pad = jax.jit(lambda l, p, c, m: compute_ipsae(l, p, c, token_mask=m))(
        logits_pad, pae_raw_pad / 32.0, chain_pad, token_mask
    )
    assert jnp.allclose(ipsae_raw_unpad, ipsae_norm_unpad, atol=1e-6)
    assert jnp.allclose(ipsae_raw_unpad, ipsae_explicit_true, atol=1e-6)
    assert jnp.allclose(ipsae_raw_unpad, ipsae_explicit_false, atol=1e-6)
    assert jnp.allclose(ipsae_raw_unpad, ipsae_raw_pad, atol=1e-6)
    assert jnp.allclose(ipsae_raw_unpad, ipsae_norm_pad, atol=1e-6)

    rigid_logits = jnp.full((8, 8, 64), -20.0, dtype=jnp.float32).at[:, :, 0].set(10.0)
    centers = pae_bin_centers(64, 32.0)
    rigid_pae_raw = jnp.fill_diagonal(
        (jax.nn.softmax(rigid_logits, axis=-1) * centers).sum(axis=-1), 0.0, inplace=False
    )
    assert float(rigid_pae_raw.max()) < 1.0
    rigid_chains = jnp.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=jnp.int32)
    rigid_ipsae_from_raw = jax.jit(lambda l, p, c: compute_ipsae(l, p, c, raw_pae_threshold=0.5))(
        rigid_logits, rigid_pae_raw, rigid_chains
    )
    rigid_ipsae_from_norm = jax.jit(lambda l, p, c: compute_ipsae(l, p, c, raw_pae_threshold=0.5))(
        rigid_logits, rigid_pae_raw / 32.0, rigid_chains
    )
    assert float(rigid_ipsae_from_raw) > 0.8
    assert jnp.allclose(rigid_ipsae_from_raw, rigid_ipsae_from_norm, atol=1e-6)

    # 8. compute_ptm_score
    score_unpad = jax.jit(compute_ptm_score)(logits_unpad, chain_unpad)
    score_pad = jax.jit(lambda l, c, m: compute_ptm_score(l, c, token_mask=m))(
        logits_pad, chain_pad, token_mask
    )
    assert jnp.allclose(score_unpad, score_pad, atol=1e-6)


def test_confidence_multisample_shapes_vmap_equivalence_and_custom_bin_centers_jit():
    """Verify () vs (S,) shapes, exact jax.vmap equivalence under @jax.jit, and custom bin_centers across all metrics."""
    num_samples = 4
    num_valid = 12
    num_padded = 16
    num_bins = 64
    centers_default = pae_bin_centers(num_bins, 32.0)
    # Non-uniform custom bin centers (strictly smaller -> strictly higher TM scores)
    centers_custom = centers_default * 0.6

    key = jax.random.key(2026)
    k_valid, k_pad = jax.random.split(key)
    logits_valid = jax.random.normal(k_valid, (num_samples, num_valid, num_valid, num_bins), dtype=jnp.float32)
    # Give each sample a distinct PAE regime so per-sample row_counts & d0 vary in compute_ipsae
    logits_valid = logits_valid.at[0, :, :, 0].add(10.0)  # ~0.25 A (all inter pairs < 15 A)
    logits_valid = logits_valid.at[1, :6, 6:, 8].add(6.0) # ~4.25 A on half, ~16 A on rest
    logits_valid = logits_valid.at[2, :, :, 24].add(5.0)  # ~12.25 A (mixed pairs < 15 A)
    logits_valid = logits_valid.at[3, :, :, 50].add(10.0) # ~25.25 A (0 pairs < 15 A -> ipsae == 0)

    logits_pad = (jax.random.normal(k_pad, (num_samples, num_padded, num_padded, num_bins), dtype=jnp.float32) * 3.0)
    logits_pad = logits_pad.at[:, :num_valid, :num_valid, :].set(logits_valid)

    pae_raw_pad = (jax.nn.softmax(logits_pad, axis=-1) * centers_default).sum(axis=-1)
    pae_raw_pad = jax.vmap(lambda m: jnp.fill_diagonal(m, 0.0, inplace=False))(pae_raw_pad)
    pae_raw_pad = pae_raw_pad.at[:, num_valid:, :].set(30.0).at[:, :, num_valid:].set(30.0)
    pae_norm_pad = pae_raw_pad / 32.0

    chain_pad = jnp.array([0] * 6 + [1] * 6 + [-1] * 4, dtype=jnp.int32)
    token_mask = jnp.arange(num_padded) < num_valid
    sub_mask = jnp.arange(num_padded) < 6

    # Check exact (S,) == jax.vmap(single_sample) equivalence under @jax.jit across all 8 metrics
    checks = [
        (
            "compute_ptm",
            jax.jit(lambda l: compute_ptm(l, token_mask=token_mask)),
            logits_pad,
        ),
        (
            "compute_ptm_subcomplex",
            jax.jit(lambda l: compute_ptm(l, mask=sub_mask, token_mask=token_mask)),
            logits_pad,
        ),
        (
            "compute_iptm",
            jax.jit(lambda l: compute_iptm(l, chain_pad, token_mask=token_mask)),
            logits_pad,
        ),
        (
            "compute_chain_ptm",
            jax.jit(lambda l: compute_chain_ptm(l, chain_pad, 1, 0, token_mask=token_mask)),
            logits_pad,
        ),
        (
            "compute_ipae",
            jax.jit(lambda p: compute_ipae(p, chain_pad, token_mask=token_mask)),
            pae_raw_pad,
        ),
        (
            "compute_min_ipae",
            jax.jit(lambda p: compute_min_ipae(p, chain_pad, token_mask=token_mask)),
            pae_raw_pad,
        ),
        (
            "compute_chain_pae",
            jax.jit(lambda p: compute_chain_pae(p, chain_pad, 1, 0, token_mask=token_mask)),
            pae_raw_pad,
        ),
        (
            "compute_ptm_score",
            jax.jit(lambda l: compute_ptm_score(l, chain_pad, token_mask=token_mask)),
            logits_pad,
        ),
    ]
    for name, fn, batched_arg in checks:
        out_single = fn(batched_arg[0])
        out_batched = fn(batched_arg)
        out_vmapped = jax.jit(jax.vmap(fn))(batched_arg)
        assert out_single.shape == (), f"{name} single shape {out_single.shape} != ()"
        assert out_batched.shape == (num_samples,), f"{name} batched shape {out_batched.shape} != ({num_samples},)"
        assert jnp.allclose(out_batched, out_vmapped, atol=1e-6), f"{name} batched != vmap: {out_batched} vs {out_vmapped}"

    # Multi-sample compute_ipsae across normalized_pae=None, True, False
    for norm_flag, pae_in in (
        (None, pae_raw_pad),
        (None, pae_norm_pad),
        (True, pae_norm_pad),
        (False, pae_raw_pad),
    ):
        ipsae_fn = jax.jit(
            lambda l, p, nf=norm_flag: compute_ipsae(
                l, p, chain_pad, token_mask=token_mask, normalized_pae=nf
            )
        )
        s0 = ipsae_fn(logits_pad[0], pae_in[0])
        batched = ipsae_fn(logits_pad, pae_in)
        vmapped = jax.jit(jax.vmap(ipsae_fn))(logits_pad, pae_in)
        assert s0.shape == ()
        assert batched.shape == (num_samples,)
        assert jnp.allclose(batched, vmapped, atol=1e-6), f"ipsae ({norm_flag}) batched != vmap"
        assert float(batched[0]) > 0.7 and jnp.allclose(batched[3], 0.0, atol=1e-6)

    # Multi-sample compute_min_ipae with no inter-chain pairs returns (S,) filled with +inf
    mono_chain = jnp.where(token_mask, 0, -1)
    min_ipae_mono_batched = jax.jit(lambda p: compute_min_ipae(p, mono_chain, token_mask=token_mask))(pae_raw_pad)
    assert min_ipae_mono_batched.shape == (num_samples,)
    assert jnp.all(jnp.isinf(min_ipae_mono_batched))

    # Custom bin_centers verification across ptm_matrix_from_logits, compute_ptm, compute_iptm,
    # compute_chain_ptm, compute_ipsae, and compute_ptm_score (both () and (S,))
    mat_def = jax.jit(lambda l: ptm_matrix_from_logits(l, token_mask=token_mask))(logits_pad)
    mat_cus = jax.jit(lambda l, bc: ptm_matrix_from_logits(l, token_mask=token_mask, bin_centers=bc))(
        logits_pad, centers_custom
    )
    assert mat_cus.shape == (num_samples, num_padded, num_padded)
    assert float(mat_cus[:, :num_valid, :num_valid].mean()) > float(mat_def[:, :num_valid, :num_valid].mean())

    for custom_fn, default_fn in (
        (
            lambda l, bc: compute_ptm(l, token_mask=token_mask, bin_centers=bc),
            lambda l: compute_ptm(l, token_mask=token_mask),
        ),
        (
            lambda l, bc: compute_iptm(l, chain_pad, token_mask=token_mask, bin_centers=bc),
            lambda l: compute_iptm(l, chain_pad, token_mask=token_mask),
        ),
        (
            lambda l, bc: compute_chain_ptm(l, chain_pad, 1, 0, token_mask=token_mask, bin_centers=bc),
            lambda l: compute_chain_ptm(l, chain_pad, 1, 0, token_mask=token_mask),
        ),
        (
            lambda l, bc: compute_ptm_score(l, chain_pad, token_mask=token_mask, bin_centers=bc),
            lambda l: compute_ptm_score(l, chain_pad, token_mask=token_mask),
        ),
    ):
        val_def_m = jax.jit(default_fn)(logits_pad)
        val_cus_m = jax.jit(custom_fn)(logits_pad, centers_custom)
        val_cus_s = jax.jit(custom_fn)(logits_pad[0], centers_custom)
        assert val_cus_s.shape == () and val_cus_m.shape == (num_samples,)
        assert jnp.allclose(val_cus_m[0], val_cus_s, atol=1e-6)
        assert jnp.all(val_cus_m > val_def_m)

    ipsae_def = jax.jit(
        lambda l, p: compute_ipsae(l, p, chain_pad, token_mask=token_mask, normalized_pae=True)
    )(logits_pad[:3], pae_norm_pad[:3])
    ipsae_cus = jax.jit(
        lambda l, p, bc: compute_ipsae(
            l, p, chain_pad, token_mask=token_mask, bin_centers=bc, normalized_pae=True
        )
    )(logits_pad[:3], pae_norm_pad[:3], centers_custom)
    assert ipsae_cus.shape == (3,)
    assert jnp.all(ipsae_cus > ipsae_def)


def test_distogram_expected_distance_and_bin_pair_distances_jit():
    """Verify distogram_expected_distance and bin_pair_distances under @jax.jit."""
    edges = jnp.linspace(2.0, 22.0, 65)
    centers = 0.5 * (edges[:-1] + edges[1:])
    probs = jax.nn.one_hot(jnp.array([[0, 10], [20, 63]]), 64, dtype=jnp.float32)

    expected_dist = jax.jit(distogram_expected_distance)(probs, centers)
    assert jnp.allclose(expected_dist, centers[jnp.array([[0, 10], [20, 63]])], atol=1e-6)

    binned = jax.jit(bin_pair_distances)(expected_dist, edges)
    assert jnp.array_equal(binned, jnp.array([[0, 10], [20, 63]]))


def test_rmsd_and_lddt_and_sc_lddt_under_jit():
    """Verify RMSD, position_lddt_ca, lddt, and sc_lddt inside @jax.jit with padding invariance."""
    key = jax.random.key(7)
    coords_a = jax.random.normal(key, (10, 24, 3), dtype=jnp.float32) * 5.0
    coords_b = coords_a + jnp.array([3.0, -2.0, 1.5], dtype=jnp.float32)

    data_a = DesignData.from_length(10).update(
        atom_positions=coords_a,
        atom_mask=jnp.ones((10, 24), dtype=jnp.bool_),
        plddt=jnp.full((10,), 0.85, dtype=jnp.float32),
    )
    data_b = DesignData.from_length(10).update(
        atom_positions=coords_b,
        atom_mask=jnp.ones((10, 24), dtype=jnp.bool_),
        plddt=jnp.full((10,), 0.85, dtype=jnp.float32),
    )

    garbage = jnp.full((4, 24, 3), 999.0, dtype=jnp.float32)
    mask_pad = jnp.arange(14) < 10
    data_a_pad = DesignData.from_length(14).update(
        atom_positions=jnp.concatenate([coords_a, garbage], axis=0),
        atom_mask=jnp.ones((14, 24), dtype=jnp.bool_) & mask_pad[:, None],
        mask=mask_pad,
        plddt=jnp.concatenate([jnp.full((10,), 0.85, dtype=jnp.float32), jnp.ones((4,), dtype=jnp.float32)]),
    )
    data_b_pad = DesignData.from_length(14).update(
        atom_positions=jnp.concatenate([coords_b, -garbage], axis=0),
        atom_mask=jnp.ones((14, 24), dtype=jnp.bool_) & mask_pad[:, None],
        mask=mask_pad,
        plddt=jnp.concatenate([jnp.full((10,), 0.85, dtype=jnp.float32), jnp.ones((4,), dtype=jnp.float32)]),
    )

    rmsd_metric = RMSD(("N", "CA", "C", "O"))
    rms_unpad = jax.jit(rmsd_metric)(data_a, data_b)
    rms_pad = jax.jit(rmsd_metric)(data_a_pad, data_b_pad)
    assert jnp.allclose(rms_unpad, 0.0, atol=1e-4)
    assert jnp.allclose(rms_pad, 0.0, atol=1e-4)

    coords_c = coords_a.at[:, 1, 0].add(1.2)
    data_c = data_a.update(atom_positions=coords_c)
    data_c_pad = data_a_pad.update(atom_positions=jnp.concatenate([coords_c, -garbage], axis=0))

    rms_ac_unpad = jax.jit(rmsd_metric)(data_a, data_c)
    rms_ac_pad = jax.jit(rmsd_metric)(data_a_pad, data_c_pad)
    assert jnp.allclose(rms_ac_unpad, rms_ac_pad, atol=1e-5)

    single_valid_mask = jnp.arange(10) == 0
    single_lddt = jax.jit(lambda x, y, m: position_lddt_ca(x, y, mask=m))(
        coords_a[:, 1], coords_c[:, 1], single_valid_mask
    )
    assert jnp.allclose(single_lddt, 0.0, atol=1e-6)

    grad_coords = jax.jit(jax.grad(lambda x: position_lddt_ca(x, coords_c[:, 1]).sum()))(coords_a[:, 1])
    assert jnp.all(jnp.isfinite(grad_coords))

    lddt_unpad = jax.jit(lddt)(data_a, data_c)
    lddt_pad = jax.jit(lddt)(data_a_pad, data_c_pad)
    assert jnp.allclose(lddt_unpad, lddt_pad[:10], atol=1e-6)
    assert jnp.all(lddt_pad[10:] == 0.0)

    sc_unpad = jax.jit(sc_lddt)(data_a, data_c)
    sc_pad = jax.jit(sc_lddt)(data_a_pad, data_c_pad)
    assert jnp.allclose(sc_unpad, sc_pad[:10], atol=1e-6)
    assert jnp.all(sc_pad[10:] == 0.0)


def test_design_data_atom24_unpad_and_jit_guard():
    """Verify DesignData canonical atom24 (L, 24, 3), S == L sample-batched slicing, host .unpad(), and @jax.jit guard."""
    d_len = DesignData.from_length(8)
    assert d_len["atom_positions"].shape == (8, 24, 3)
    assert d_len["atom_mask"].shape == (8, 24)

    d14 = DesignData.from_dict(
        dict(
            aa=jnp.zeros((6,), dtype=jnp.int32),
            atom_positions=jnp.ones((6, 14, 3), dtype=jnp.float32),
            atom_mask=jnp.ones((6, 14), dtype=jnp.bool_),
            residue_index=jnp.arange(6, dtype=jnp.int32),
            chain_index=jnp.zeros((6,), dtype=jnp.int32),
            batch_index=jnp.zeros((6,), dtype=jnp.int32),
            mask=jnp.array([True, True, True, True, False, False]),
            pae=jnp.arange(36, dtype=jnp.float32).reshape(6, 6),
        )
    )
    assert d14["atom_positions"].shape == (6, 24, 3)
    assert d14["atom_mask"].shape == (6, 24)
    assert jnp.all(d14["atom_positions"][:, 14:] == 0.0)
    assert not jnp.any(d14["atom_mask"][:, 14:])

    unpadded = d14.unpad()
    assert unpadded.size == 4
    assert unpadded["atom_positions"].shape == (4, 24, 3)
    assert unpadded["pae"].shape == (4, 4)

    d_batched = DesignData.from_dict(
        dict(
            aa=jnp.zeros((6,), dtype=jnp.int32),
            atom_positions=jnp.ones((6, 6, 24, 3), dtype=jnp.float32),
            atom_mask=jnp.ones((6, 6, 24), dtype=jnp.bool_),
            plddt=jnp.ones((6, 6), dtype=jnp.float32),
            pae=jnp.ones((6, 6, 6), dtype=jnp.float32),
            residue_index=jnp.arange(6, dtype=jnp.int32),
            chain_index=jnp.zeros((6,), dtype=jnp.int32),
            batch_index=jnp.zeros((6,), dtype=jnp.int32),
            mask=jnp.array([True, True, True, True, False, False]),
        )
    )
    assert not d_batched.is_single_sample
    unpadded_batched = d_batched.unpad()
    assert unpadded_batched["atom_positions"].shape == (6, 4, 24, 3)
    assert unpadded_batched["plddt"].shape == (6, 4)
    assert unpadded_batched["pae"].shape == (6, 4, 4)

    sliced_batched = d_batched[:4]
    assert sliced_batched["atom_positions"].shape == (6, 4, 24, 3)
    assert sliced_batched["plddt"].shape == (6, 4)

    with pytest.raises(RuntimeError, match="cannot be called inside jax.jit"):
        jax.jit(lambda x: x.unpad())(d14)


def test_protein_mpnn_edge_idx_padding_invariance_decode_and_non_mutation(tmp_path):
    """Verify ProteinMPNN _get_edge_idx padding invariance, encode/decode under @jax.jit, non-mutation, and solmpnn guard."""
    key = jax.random.key(11)
    coords_valid = jax.random.normal(key, (8, 24, 3), dtype=jnp.float32) * 4.0
    coords_padded = jnp.concatenate([coords_valid, coords_valid[:4]], axis=0)

    data_unpad = DesignData.from_length(8).update(
        atom_positions=coords_valid,
        atom_mask=jnp.ones((8, 24), dtype=jnp.bool_),
        aa=jnp.arange(8, dtype=jnp.int32) % 20,
    )
    mask_pad = jnp.arange(12) < 8
    data_pad = DesignData.from_length(12).update(
        atom_positions=coords_padded,
        atom_mask=jnp.ones((12, 24), dtype=jnp.bool_) & mask_pad[:, None],
        mask=mask_pad,
        aa=jnp.concatenate([data_unpad["aa"], jnp.full((4,), 20, dtype=jnp.int32)]),
    )

    @hk.transform
    def _edge_idx_fn(ca, mask, batch):
        feat = ProteinFeatures(edge_features=16, node_features=16, top_k=6, augment_eps=0.0)
        return feat._get_edge_idx(ca, mask, batch)

    params = _edge_idx_fn.init(key, coords_valid[:, 1], data_unpad["mask"], data_unpad["batch_index"])
    idx_unpad = jax.jit(_edge_idx_fn.apply)(
        params, key, coords_valid[:, 1], data_unpad["mask"], data_unpad["batch_index"]
    )
    idx_pad = jax.jit(_edge_idx_fn.apply)(
        params, key, coords_padded[:, 1], data_pad["mask"], data_pad["batch_index"]
    )
    assert jnp.all(idx_pad[:8] < 8)
    assert jnp.array_equal(idx_unpad, idx_pad[:8])

    @hk.transform
    def _mpnn_forward_and_decode(d):
        model = ProteinMPNN(
            num_letters=21,
            node_features=16,
            edge_features=16,
            hidden_dim=16,
            num_encoder_layers=1,
            num_decoder_layers=1,
            k_neighbors=6,
            augment_eps=0.0,
            dropout=0.0,
        )
        out = model(d)
        encoded = model.encode(d)
        encoded["order"] = jnp.arange(d["aa"].shape[0], dtype=jnp.int32)
        decoded_logits = model.decode(encoded)
        return out, decoded_logits

    orig_keys = set(data_pad.data.keys())
    orig_aa = np.array(data_pad["aa"], copy=True)
    mpnn_params = _mpnn_forward_and_decode.init(key, data_pad)
    (out, decoded_logits) = jax.jit(_mpnn_forward_and_decode.apply)(mpnn_params, key, data_pad)
    assert out["logits"].shape == (12, 21)
    assert decoded_logits.shape == (12, 21)
    assert jnp.all(jnp.isfinite(decoded_logits))
    assert set(data_pad.data.keys()) == orig_keys
    assert np.array_equal(np.asarray(data_pad["aa"]), orig_aa)

    with pytest.raises(RuntimeError, match="points to vanilla ProteinMPNN"):
        _verify_solmpnn_weights("params/pmpnn/v_48_020.pkl")

    pmpnn_dir = tmp_path / "pmpnn"
    solmpnn_dir = tmp_path / "solmpnn"
    pmpnn_dir.mkdir()
    solmpnn_dir.mkdir()
    dup_bytes = b"identical-checkpoint-bytes"
    (pmpnn_dir / "v_48_020.pkl").write_bytes(dup_bytes)
    (solmpnn_dir / "v_48_020.pkl").write_bytes(dup_bytes)
    with pytest.raises(RuntimeError, match="identical to vanilla ProteinMPNN"):
        _verify_solmpnn_weights(str(solmpnn_dir / "v_48_020.pkl"))


def test_keygen_splitting_and_validation():
    """Verify Keygen(num_keys) returns 1 key by default, N keys when requested, and rejects < 1."""
    kg = Keygen(0)
    k1 = kg()
    k2 = kg(1)
    assert k1.shape == k2.shape
    assert not jnp.array_equal(k1, k2)

    multi = kg(4)
    assert multi.shape[0] == 4
    for i in range(4):
        for j in range(i + 1, 4):
            assert not jnp.array_equal(multi[i], multi[j])

    with pytest.raises(ValueError, match="num_keys must be >= 1"):
        kg(0)


def test_hallucination_jvp_and_distogram_jsd_jit():
    """Verify simplex_apgm_step softmax_grad JVP and distogram/contact JSD properties under @jax.jit."""
    target = jax.nn.softmax(jnp.arange(20, dtype=jnp.float32))[None, :]

    def _toy_loss(x, key=None, context=None, params=None):
        loss = jnp.sum((x - target) ** 2)
        return loss, {"loss": loss}

    loss_update = jit_loss_update(_toy_loss)
    x0 = jnp.zeros((4, 20), dtype=jnp.float32)
    step_fn = simplex_apgm_step(
        loss_update,
        key=Keygen(0),
        lr=0.5,
        momentum=0.1,
        opt_logits=True,
        softmax_grad=True,
    )
    state, _, (loss0, _) = step_fn((x0, x0))
    state, _, (loss1, _) = step_fn(state)
    assert float(loss1) < float(loss0)

    best, best_val, _ = simplex_agpm(
        x0,
        loss_update,
        key=Keygen(1),
        num_steps=5,
        lr=0.5,
        opt_logits=True,
        softmax_grad=True,
    )
    assert best.shape == (4, 20)
    assert jnp.allclose(best.sum(axis=-1), 1.0, atol=1e-5)
    assert jnp.isfinite(best_val)

    key = jax.random.key(99)
    k1, k2 = jax.random.split(key)
    logits_p = jax.random.normal(k1, (6, 6, 32), dtype=jnp.float32)
    logits_q = jax.random.normal(k2, (6, 6, 32), dtype=jnp.float32)
    probs_p = jax.nn.softmax(logits_p, axis=-1)
    probs_q = jax.nn.softmax(logits_q, axis=-1)

    jsd_self = jax.jit(distogram_jsd)(logits_p, logits_p)
    assert jnp.allclose(jsd_self, 0.0, atol=1e-6)

    jsd_pq_logits = jax.jit(distogram_jsd)(logits_p, logits_q)
    jsd_qp_logits = jax.jit(distogram_jsd)(logits_q, logits_p)
    jsd_pq_probs = jax.jit(distogram_jsd)(probs_p, probs_q)
    assert jnp.allclose(jsd_pq_logits, jsd_qp_logits, atol=1e-6)
    assert jnp.allclose(jsd_pq_logits, jsd_pq_probs, atol=1e-5)
    assert jnp.all((jsd_pq_logits >= -1e-6) & (jsd_pq_logits <= jnp.log(2.0) + 1e-5))

    logits_p_pad = jnp.pad(logits_p, ((0, 3), (0, 3), (0, 0)), constant_values=5.0)
    logits_q_pad = jnp.pad(logits_q, ((0, 3), (0, 3), (0, 0)), constant_values=-5.0)
    mask_pad = jnp.arange(9) < 6
    resi_unpad = jnp.arange(6, dtype=jnp.int32)
    resi_pad = jnp.arange(9, dtype=jnp.int32)

    for fn in (mean_distogram_jsd, max_distogram_jsd, topk_distogram_jsd):
        val_unpad = jax.jit(lambda a, b, r: fn(a, b, residue_index=r, min_residue_distance=2))(
            logits_p, logits_q, resi_unpad
        )
        val_pad = jax.jit(lambda a, b, r, m: fn(a, b, residue_index=r, min_residue_distance=2, mask=m))(
            logits_p_pad, logits_q_pad, resi_pad, mask_pad
        )
        assert jnp.allclose(val_unpad, val_pad, atol=1e-6)

    class _DummyDistoResult:
        def __init__(self, log_disto, token_mask):
            self.log_distogram = jax.nn.log_softmax(log_disto, axis=-1)
            self.distogram_bin_edges = jnp.linspace(2.0, 22.0, log_disto.shape[-1] + 1)
            self.token_pad_mask = token_mask

    res_p = _DummyDistoResult(logits_p_pad, mask_pad)
    res_q = _DummyDistoResult(logits_q_pad, mask_pad)
    cjsd = contact_jsd(res_p, res_q, contact_distance=10.0)
    cjsd_rev = contact_jsd(res_q, res_p, contact_distance=10.0)
    assert jnp.allclose(cjsd, cjsd_rev, atol=1e-6)
    assert jnp.all(cjsd[6:, :] == 0.0)
    assert jnp.all(cjsd[:, 6:] == 0.0)
    assert jnp.all((cjsd >= -1e-6) & (cjsd <= jnp.log(2.0) + 1e-5))
