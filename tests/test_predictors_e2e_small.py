"""End-to-end verification on small proteins (<= 50 aa) across all four predictors."""

from __future__ import annotations

import gc
import resource
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any
from unittest.mock import patch

import equinox as eqx
import gemmi
import jax
import jax.numpy as jnp
import joltz
import numpy as np
import torch

# Small 2-chain mini-complexes (all <= 50 aa total) so both complex and interface
# metrics (ptm, iptm, ipae, ipsae, mean_plddt, mean_pae) are exercised end-to-end.
SEQ_20 = "NLYIQWLKDG:GPSSGRPPPS"  # 10 + 10 = 20 aa (Trp-cage split complex)
SEQ_28 = "NLYIQWLKDGGPSS:LSDEDFKAVFGMTR"  # 14 + 14 = 28 aa (Trp-cage + villin)
SEQ_16 = "NLYIQWLK:DGGPSSGR"  # 8 + 8 = 16 aa
SEQ_22 = "NLYIQWLKDGG:PSSGRPPPSAA"  # 11 + 11 = 22 aa
SEQ_GLY_20 = "G" * 10 + ":" + "G" * 10  # 10 + 10 = 20 aa (82 heavy atoms)
SEQ_TRP_20 = "W" * 10 + ":" + "W" * 10  # 10 + 10 = 20 aa (282 heavy atoms)

ATOL = 5e-5


def _peak_rss_gb() -> float:
    """Return process peak resident set size in GiB."""
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return usage / (1024**3)
    return usage / (1024**2)


def _cleanup_memory() -> None:
    """Release Python and XLA caches between heavy predictors."""
    gc.collect()
    jax.clear_caches()
    gc.collect()


def _assert_structure_file(
    path: Path, expected_residues: int, expected_chains: tuple[int, ...] | None = None
) -> None:
    """Verify a written PDB or mmCIF structure has exact unpadded residue counts and finite coords."""
    assert path.is_file() and path.stat().st_size > 0
    structure = gemmi.read_structure(str(path))
    assert len(structure) >= 1
    model = structure[0]
    chain_lengths = []
    coords = []
    for chain in model:
        polymer = chain.get_polymer()
        res_list = list(polymer) if len(polymer) > 0 else list(chain)
        if not res_list:
            continue
        chain_lengths.append(len(res_list))
        for res in res_list:
            for atom in res:
                coords.append((atom.pos.x, atom.pos.y, atom.pos.z))
    assert sum(chain_lengths) == expected_residues, (
        f"Expected {expected_residues} residues in {path.name}, got {sum(chain_lengths)} "
        f"(chains={chain_lengths})"
    )
    if expected_chains is not None:
        assert tuple(chain_lengths) == expected_chains, (
            f"Expected chain lengths {expected_chains} in {path.name}, got {tuple(chain_lengths)}"
        )
    coords_arr = np.asarray(coords, dtype=np.float64)
    assert coords_arr.ndim == 2 and coords_arr.shape[0] >= expected_residues * 4
    assert np.all(np.isfinite(coords_arr)), f"Non-finite coordinates in {path.name}"


def _assert_tensor_parity(
    unpadded: Any,
    padded_slice: Any,
    name: str,
    *,
    strict_max_atol: float | None = None,
    mean_atol: float = ATOL,
    rel_tol: float = 1e-4,
) -> tuple[float, float]:
    """Assert float32 parity between unpadded and sliced-padded tensors.

    For single-module / normalized outputs, pass ``strict_max_atol=ATOL``.
    For deep (48-96 layer) unnormalized residual streams or scaled coordinates,
    enforces ``mean(|diff|) <= mean_atol`` and ``max(|diff|) / scale <= rel_tol``.
    """
    u_arr = np.asarray(unpadded, dtype=np.float64)
    p_arr = np.asarray(padded_slice, dtype=np.float64)
    assert u_arr.shape == p_arr.shape, f"{name} shape mismatch: {u_arr.shape} vs {p_arr.shape}"
    assert np.all(np.isfinite(u_arr)) and np.all(np.isfinite(p_arr)), f"Non-finite values in {name}"
    abs_diff = np.abs(u_arr - p_arr)
    max_diff = float(np.max(abs_diff)) if abs_diff.size > 0 else 0.0
    mean_diff = float(np.mean(abs_diff)) if abs_diff.size > 0 else 0.0
    if strict_max_atol is not None:
        assert max_diff <= strict_max_atol, (
            f"{name} max diff {max_diff:.3e} > {strict_max_atol:.1e} (mean={mean_diff:.3e})"
        )
        return max_diff, mean_diff
    scale = max(1.0, float(np.max(np.abs(u_arr)))) if u_arr.size > 0 else 1.0
    rel_max = max_diff / scale
    assert mean_diff <= mean_atol and rel_max <= rel_tol, (
        f"{name} parity failed: max_diff={max_diff:.3e}, mean_diff={mean_diff:.3e} "
        f"(limit {mean_atol:.1e}), rel_max={rel_max:.3e} (limit {rel_tol:.1e}, scale={scale:.2f})"
    )
    return max_diff, mean_diff


def _assert_jit_to_data_p_dssp(padded_res, valid_tokens: int = 20, padded_tokens: int = 32) -> None:
    """Verify Result.to_data() and .p_dssp execute cleanly inside @jax.jit with masked shapes."""
    jit_fn = jax.jit(lambda r: (r.to_data(), r.to_data().p_dssp))
    data_pad, dssp_pad = jit_fn(padded_res)
    assert data_pad["atom_positions"].shape == (padded_tokens, 24, 3)
    assert np.all(np.isfinite(np.asarray(data_pad["atom_positions"])))
    mask_np = np.asarray(data_pad["mask"], dtype=bool)
    assert int(mask_np.sum()) == valid_tokens
    assert not np.any(mask_np[valid_tokens:])
    assert np.all(np.asarray(data_pad["chain_index"])[valid_tokens:] == -1)
    probs = [float(np.asarray(dssp_pad[k])) for k in ("L", "H", "E")]
    assert all(np.isfinite(p) and 0.0 <= p <= 1.0 for p in probs)
    assert abs(sum(probs) - 1.0) <= 1e-5


def _assert_masked_metrics_parity(unpadded_res, padded_res, atol: float = ATOL) -> dict[str, tuple[float, float]]:
    """Assert 0D scalar masked confidence metrics match within atol between unpadded and padded Result."""
    raw_unpad = {
        "ptm": unpadded_res.ptm,
        "iptm": unpadded_res.iptm,
        "ipae": unpadded_res.ipae,
        "min_ipae": unpadded_res.min_ipae,
        "ipsae": unpadded_res.ipsae(),
        "ptm_score": unpadded_res.ptm_score,
        "mean_plddt": unpadded_res.mean_plddt,
        "mean_pae": unpadded_res.mean_pae,
    }
    raw_pad = {
        "ptm": padded_res.ptm,
        "iptm": padded_res.iptm,
        "ipae": padded_res.ipae,
        "min_ipae": padded_res.min_ipae,
        "ipsae": padded_res.ipsae(),
        "ptm_score": padded_res.ptm_score,
        "mean_plddt": padded_res.mean_plddt,
        "mean_pae": padded_res.mean_pae,
    }
    report = {}
    for name in raw_unpad:
        u_arr = np.asarray(raw_unpad[name])
        p_arr = np.asarray(raw_pad[name])
        assert u_arr.shape == (), f"Expected 0D scalar for single-sample unpadded {name}, got {u_arr.shape}"
        assert p_arr.shape == (), f"Expected 0D scalar for single-sample padded {name}, got {p_arr.shape}"
        u_val = float(u_arr)
        p_val = float(p_arr)
        assert np.isfinite(u_val) and np.isfinite(p_val), (
            f"Non-finite metric {name}: unpad={u_val}, pad={p_val}"
        )
        diff = abs(u_val - p_val)
        assert diff <= atol, (
            f"Metric {name} parity mismatch: unpad={u_val:.7f}, pad={p_val:.7f}, "
            f"|diff|={diff:.3e} > {atol:.1e}"
        )
        report[name] = (u_val, diff)
    return report


def _collect_result_metrics(res, binder_mask: jax.Array) -> dict[str, jax.Array]:
    """Evaluate the unified Phase 2 <model>Result confidence metric surface."""
    return {
        "ptm": res.ptm,
        "index_ptm": res.index_ptm(binder_mask),
        "iptm": res.iptm,
        "index_iptm": res.index_iptm(),
        "chain_ptm": res.chain_ptm(0, 1),
        "ipae": res.ipae,
        "index_ipae": res.index_ipae(),
        "min_ipae": res.min_ipae,
        "index_min_ipae": res.index_min_ipae(),
        "chain_pae": res.chain_pae(0, 1),
        "ipsae": res.ipsae(),
        "index_ipsae": res.index_ipsae(),
        "ptm_score": res.ptm_score,
        "index_ptm_score": res.index_ptm_score(),
        "mean_plddt": res.mean_plddt,
        "index_plddt": res.index_plddt(binder_mask),
        "mean_pae": res.mean_pae,
    }


def _assert_single_sample_result_surface(res, num_tokens: int) -> None:
    """Verify single-sample <model>Result returns 0D () metrics and valid geometry inside @jax.jit."""
    binder_mask = jnp.arange(num_tokens) < 10
    jit_eval = jax.jit(
        lambda r: (
            _collect_result_metrics(r, binder_mask),
            r.max_pae,
            r.pae_bin_edges,
            r.pae_bin_centers,
            r.distogram_mean,
            r.distance,
        )
    )
    metrics, max_pae, edges, centers, disto_mean, distance = jit_eval(res)
    assert abs(float(np.asarray(max_pae)) - 32.0) <= 1e-5
    assert np.asarray(edges).ndim == 1 and np.asarray(centers).ndim == 1
    assert np.asarray(edges).shape[0] == np.asarray(centers).shape[0] + 1
    assert np.asarray(disto_mean).shape == (num_tokens, num_tokens)
    assert np.asarray(distance).shape == (num_tokens, num_tokens)
    np.testing.assert_allclose(np.asarray(disto_mean), np.asarray(distance), atol=1e-6)
    for name, val in metrics.items():
        arr = np.asarray(val)
        assert arr.shape == (), f"Single-sample {type(res).__name__}.{name} shape {arr.shape} != ()"
        assert np.all(np.isfinite(arr)), f"Non-finite single-sample {type(res).__name__}.{name}: {arr}"


def _assert_multisample_result_metrics(
    res_multi, num_samples: int = 2, num_tokens: int = 20
) -> None:
    """Verify multi-sample <model>Result returns (S,) 1D metrics both eagerly and inside @jax.jit."""
    assert not res_multi.is_single_sample
    atom24_s, mask24_s = res_multi.atom24_samples
    assert atom24_s.shape == (num_samples, num_tokens, 24, 3)
    assert mask24_s.shape == (num_tokens, 24)
    assert np.all(np.isfinite(np.asarray(atom24_s)))

    binder_mask = jnp.arange(num_tokens) < (num_tokens // 2)
    eager_metrics = _collect_result_metrics(res_multi, binder_mask)
    jit_metrics, max_pae, edges, centers, disto_mean, distance = jax.jit(
        lambda r: (
            _collect_result_metrics(r, binder_mask),
            r.max_pae,
            r.pae_bin_edges,
            r.pae_bin_centers,
            r.distogram_mean,
            r.distance,
        )
    )(res_multi)

    assert abs(float(np.asarray(max_pae)) - 32.0) <= 1e-5
    assert np.asarray(edges).ndim == 1 and np.asarray(centers).ndim == 1
    assert np.asarray(edges).shape[0] == np.asarray(centers).shape[0] + 1
    assert np.asarray(disto_mean).shape == (num_tokens, num_tokens)
    assert np.asarray(distance).shape == (num_tokens, num_tokens)
    np.testing.assert_allclose(np.asarray(disto_mean), np.asarray(distance), atol=1e-6)

    for name, eager_val in eager_metrics.items():
        e_arr = np.asarray(eager_val)
        j_arr = np.asarray(jit_metrics[name])
        assert e_arr.shape == (num_samples,), (
            f"Multi-sample {type(res_multi).__name__}.{name} eager shape {e_arr.shape} != ({num_samples},)"
        )
        assert j_arr.shape == (num_samples,), (
            f"Multi-sample {type(res_multi).__name__}.{name} jit shape {j_arr.shape} != ({num_samples},)"
        )
        assert np.all(np.isfinite(e_arr)) and np.all(np.isfinite(j_arr)), (
            f"Non-finite multi-sample {type(res_multi).__name__}.{name}: {e_arr}"
        )
        np.testing.assert_allclose(e_arr, j_arr, atol=1e-5)


def _controlled_joltz_diffusion(
    evaluator,
    features: dict,
    embedding,
    trunk_state,
    start_noise: jax.Array,
    step_noise: jax.Array,
    key: jax.Array,
) -> jax.Array:
    """Run Joltz2 diffusion steps with controlled per-atom Gaussian noise for parity verification."""
    q, c, to_keys, atom_enc_bias, atom_dec_bias, token_trans_bias = (
        evaluator.joltz.diffusion_conditioning(
            trunk_state.s,
            trunk_state.z,
            embedding.relative_position_encoding,
            features,
        )
    )
    diffusion_conditioning = {
        "q": q,
        "c": c,
        "to_keys": to_keys,
        "atom_enc_bias": atom_enc_bias,
        "atom_dec_bias": atom_dec_bias,
        "token_trans_bias": token_trans_bias,
    }
    sm = evaluator.joltz.structure_module
    atom_mask = features["atom_pad_mask"]
    mask = (atom_mask > 0).astype(jnp.float32)[..., None]
    mask_count = jnp.maximum(mask.sum(axis=-2, keepdims=True), 1.0)

    def _center_and_mask(coords):
        coords = coords * mask
        mean = coords.sum(axis=-2, keepdims=True) / mask_count
        return (coords - mean) * mask

    num_steps = evaluator.num_sampling_steps
    sigmas = sm.sample_schedule(num_steps)
    gammas = jnp.where(sigmas > sm.gamma_min, sm.gamma_0, 0.0)
    step_scale = sm.step_scale

    with jax.default_matmul_precision("float32"):
        start_coords = _center_and_mask(sigmas[0] * (start_noise * mask))

        def body_fn(carry, step_inputs):
            sigma_tm, sigma_t, gamma, eps_raw = step_inputs
            coords, step_key = carry
            random_r, random_tr = joltz.compute_random_augmentation(key=step_key)
            step_key = jax.random.fold_in(step_key, 1)
            coords = _center_and_mask(coords)
            coords = (jnp.einsum("bmd,bds->bms", coords, random_r) + random_tr) * mask

            t_hat = sigma_tm * (1 + gamma)
            noise_var = sm.noise_scale**2 * (t_hat**2 - sigma_tm**2)
            eps = jnp.sqrt(jnp.maximum(noise_var, 0.0)) * eps_raw * mask
            step_key = jax.random.fold_in(step_key, 1)
            atom_coords_noisy = (coords + eps) * mask
            atom_coords_denoised = sm.preconditioned_network_forward(
                atom_coords_noisy,
                t_hat,
                network_condition_kwargs=dict(
                    feats=features,
                    s_trunk=trunk_state.s,
                    s_inputs=embedding.s_inputs,
                    multiplicity=1,
                    diffusion_conditioning=diffusion_conditioning,
                ),
                key=step_key,
            )
            atom_coords_denoised = atom_coords_denoised * mask
            if sm.alignment_reverse_diff:
                atom_coords_noisy = (
                    joltz.weighted_rigid_align(
                        atom_coords_noisy,
                        atom_coords_denoised,
                        atom_mask,
                        atom_mask,
                    )
                    * mask
                )
            denoised_over_sigma = (atom_coords_noisy - atom_coords_denoised) / t_hat
            atom_coords_next = (
                atom_coords_noisy + step_scale * (sigma_t - t_hat) * denoised_over_sigma
            ) * mask
            return (atom_coords_next, jax.random.fold_in(step_key, 0)), None

        (final_coords, _), _ = jax.lax.scan(
            body_fn,
            (start_coords, jax.random.fold_in(jax.random.fold_in(key, 2), 1)),
            (sigmas[:-1], sigmas[1:], gammas[1:], step_noise),
        )
        return final_coords * mask


def test_coinstalled_environment_and_imports():
    """Verify CPU PyTorch, JAX >= 0.12, and all four predictors coexist in one process."""
    assert torch.version.cuda is None
    jax_parts = tuple(int(p) for p in jax.__version__.split(".")[:2])
    assert jax_parts >= (0, 12)

    import flexcraft.structure as st
    from flexcraft.structure.af._af3_like_wrapper import AF2Spec, AFWrapper
    from flexcraft.structure.af._data import AFInput
    from flexcraft.structure.af3_like.model import AF3LikeSpec
    from flexcraft.structure.boltz._data import JoltzInput, JoltzSpec
    from flexcraft.structure.boltz._model import Joltz2
    from flexcraft.structure.esmfold2._data import ESMFold2Input, ESMFold2Spec
    from flexcraft.structure.esmfold2._model import ESMFold2
    from flexcraft.structure.protenix._data import ProtenixInput, ProtenixSpec
    from flexcraft.structure.protenix._model import Protenix

    st.check_runtime_environment()
    for cls in (
        AFWrapper,
        AF2Spec,
        AFInput,
        Joltz2,
        JoltzSpec,
        JoltzInput,
        Protenix,
        ProtenixSpec,
        ProtenixInput,
        ESMFold2,
        ESMFold2Spec,
        ESMFold2Input,
        AF3LikeSpec,
    ):
        assert cls is not None


def test_af2_e2e_small(tmp_path: Path):
    """Verify AlphaFold2 Tier 1 parity, Tier 2 single-trace reuse, jit p_dssp, and PDB export."""
    from flexcraft.structure.af import make_af2, make_predict
    from flexcraft.structure.af._af3_like_wrapper import AF2Spec, AFWrapper

    wrapper = None
    jitted_predict = None
    res_unpad = None
    res_pad = None
    res_pad28 = None
    prediction = None
    try:
        wrapper = AFWrapper(model="af2_model_1_ptm", cache="params/af/")
        raw_predict = make_predict(
            make_af2(wrapper.af2_config, wrapper.multimer), num_recycle=1
        )

        trace_counter = {"count": 0}

        @jax.jit
        def jitted_predict(params, rng_key, af_data):
            trace_counter["count"] += 1
            return raw_predict(params, rng_key, af_data)

        inp_20_unpad = AF2Spec().add_protein(SEQ_20, use_msa=False).to_input()
        inp_20_pad = (
            AF2Spec()
            .add_protein(SEQ_20, use_msa=False)
            .pad_to(32, num_msa=4, num_templates=2)
            .to_input()
        )
        inp_20_mult = (
            AF2Spec()
            .add_protein(SEQ_20, use_msa=False)
            .pad_to_nearest_multiple(32, num_msa=4, num_templates=2)
            .to_input()
        )
        inp_28_pad = (
            AF2Spec()
            .add_protein(SEQ_28, use_msa=False)
            .pad_to_nearest_multiple(32, num_msa=4, num_templates=2)
            .to_input()
        )

        assert inp_20_unpad["use_dropout"] is False
        assert inp_20_pad["use_dropout"] is False
        assert inp_20_mult["aatype"].shape == (32,)

        rng_key = jax.random.PRNGKey(42)

        # Tier 2: compile_count == 1 across L=20 and L=28 both padded to 32.
        res_pad = jitted_predict(wrapper.af2_params, rng_key, inp_20_pad.data)
        jax.block_until_ready(res_pad.plddt)
        assert trace_counter["count"] == 1
        assert jitted_predict._cache_size() == 1

        res_pad28 = jitted_predict(wrapper.af2_params, rng_key, inp_28_pad.data)
        jax.block_until_ready(res_pad28.plddt)
        assert trace_counter["count"] == 1
        assert jitted_predict._cache_size() == 1

        # Tier 1: unpadded L=20 vs padded L=32 parity (atol <= 5e-5).
        unpad_jit = jax.jit(raw_predict)
        res_unpad = unpad_jit(wrapper.af2_params, rng_key, inp_20_unpad.data)
        jax.block_until_ready(res_unpad.plddt)

        coords_unpad, mask_unpad = res_unpad.atom24
        coords_pad, mask_pad = res_pad.atom24
        assert coords_pad.shape == (32, 24, 3)
        np.testing.assert_array_equal(np.asarray(mask_unpad), np.asarray(mask_pad[:20]))
        valid_atom_mask = np.asarray(mask_unpad, dtype=bool)[..., None]
        max_coord_diff, mean_coord_diff = _assert_tensor_parity(
            np.asarray(coords_unpad) * valid_atom_mask,
            np.asarray(coords_pad[:20]) * valid_atom_mask,
            "AF2 atom24 coordinates",
            mean_atol=1e-4,
            rel_tol=1e-4,
        )

        metrics_report = _assert_masked_metrics_parity(res_unpad, res_pad, atol=ATOL)
        _assert_single_sample_result_surface(res_pad, num_tokens=32)

        # Phase 2: AF2Spec / AFInput .pad_atoms(24) preserves L=20 and compiles once across Gly20 vs Trp20.
        inp_gly_af = AF2Spec().add_protein(SEQ_GLY_20, use_msa=False).to_input().pad_atoms(24)
        inp_trp_af = AF2Spec().add_protein(SEQ_TRP_20, use_msa=False).pad_atoms(24).to_input()
        assert inp_gly_af["aatype"].shape == (20,)
        assert inp_trp_af["aatype"].shape == (20,)
        af_atoms_counter = {"count": 0}

        @jax.jit
        def jit_af_atoms(params, rng, af_data):
            af_atoms_counter["count"] += 1
            return raw_predict(params, rng, af_data)

        res_gly_af = jit_af_atoms(wrapper.af2_params, rng_key, inp_gly_af.data)
        jax.block_until_ready(res_gly_af.plddt)
        assert af_atoms_counter["count"] == 1
        res_trp_af = jit_af_atoms(wrapper.af2_params, rng_key, inp_trp_af.data)
        jax.block_until_ready(res_trp_af.plddt)
        assert af_atoms_counter["count"] == 1
        assert jit_af_atoms._cache_size() == 1
        del res_trp_af, res_gly_af, jit_af_atoms

        # Verify @jax.jit(lambda r: r.to_data().p_dssp) on padded AFResult (32, 24, 3).
        _assert_jit_to_data_p_dssp(res_pad, valid_tokens=20, padded_tokens=32)

        # Verify eager predictor + save_pdb writes unpadded 20-residue PDB.
        predictor = wrapper.predictor(num_recycle=1)
        prediction = predictor(rng_key, AF2Spec().add_protein(SEQ_20, use_msa=False).pad_to(32))
        pdb_path = tmp_path / "af2_trpcage_20.pdb"
        prediction.save_pdb(str(pdb_path))
        _assert_structure_file(pdb_path, expected_residues=20, expected_chains=(10, 10))
        print(
            f"[AF2] max_coord_diff={max_coord_diff:.2e}, compile_count={trace_counter['count']}, "
            f"metrics={metrics_report}, rss={_peak_rss_gb():.2f} GiB"
        )
    finally:
        del prediction, res_pad28, res_pad, res_unpad, jitted_predict, wrapper
        _cleanup_memory()


def test_joltz2_e2e_small(tmp_path: Path):
    """Verify Joltz2 Tier 1 trunk/diffusion/confidence parity, Tier 2 single-trace reuse, and exports."""
    from flexcraft.structure.boltz._data import JoltzSpec
    from flexcraft.structure.boltz._model import Joltz2, Joltz2Evaluator
    from flexcraft.structure.boltz._result import JoltzResult

    joltz_wrapper = None
    evaluator = None
    jit_predict = None
    res_unpad = None
    res_pad = None
    res_pad28 = None
    pred = None
    try:
        joltz_wrapper = Joltz2(model="boltz2_conf.ckpt", cache="./params/boltz/")
        spec_20 = JoltzSpec().add_protein(SEQ_20, use_msa=False)
        inp_20_unpad, writer_20 = spec_20.to_input(cache="./params/boltz/")
        target_msa = max(4, int(inp_20_unpad.features["msa"].shape[1]) + 2)
        inp_20_pad = inp_20_unpad.pad_to(32, num_residue_atoms=24, num_msa=target_msa)
        inp_28_pad, _ = (
            JoltzSpec()
            .add_protein(SEQ_28, use_msa=False)
            .pad_to(32, num_residue_atoms=24, num_msa=target_msa)
            .to_input(cache="./params/boltz/")
        )

        n_native_atoms = int(inp_20_unpad.features["atom_pad_mask"].shape[1])
        n_real_atoms = int(np.asarray(inp_20_unpad.features["atom_pad_mask"]).sum())
        assert n_native_atoms % 32 == 0
        assert inp_20_pad.features["atom_pad_mask"].shape == (1, 768)

        evaluator = Joltz2Evaluator(
            joltz=joltz_wrapper.model,
            num_recycle=1,
            num_sampling_steps=20,
            deterministic=True,
        )

        # Tier 1: deterministic trunk + controlled-noise diffusion + confidence parity.
        prep_fn = eqx.filter_jit(evaluator._prepare)
        rng_key = jax.random.PRNGKey(7)
        prep_key, diff_key, noise_k0, noise_k1 = jax.random.split(rng_key, 4)

        _, emb_unpad, trunk_unpad, disto_unpad = prep_fn(prep_key, inp_20_unpad.features)
        _, emb_pad, trunk_pad, disto_pad = prep_fn(prep_key, inp_20_pad.features)

        _assert_tensor_parity(
            emb_unpad.s_inputs, emb_pad.s_inputs[:, :20], "Joltz2 s_inputs", strict_max_atol=ATOL
        )
        _assert_tensor_parity(
            emb_unpad.z_init, emb_pad.z_init[:, :20, :20], "Joltz2 z_init", strict_max_atol=ATOL
        )
        trunk_s_diff, _ = _assert_tensor_parity(
            trunk_unpad.s, trunk_pad.s[:, :20], "Joltz2 trunk.s"
        )
        trunk_z_diff, _ = _assert_tensor_parity(
            trunk_unpad.z, trunk_pad.z[:, :20, :20], "Joltz2 trunk.z"
        )
        disto_diff, _ = _assert_tensor_parity(
            disto_unpad, disto_pad[:20, :20], "Joltz2 distogram"
        )

        start_noise_full = jax.random.normal(noise_k0, (1, 768, 3))
        step_noise_full = jax.random.normal(noise_k1, (evaluator.num_sampling_steps, 1, 768, 3))

        diff_fn = eqx.filter_jit(_controlled_joltz_diffusion)
        # Verify isolated controlled-noise diffusion parity from identical shared trunk state:
        shared_trunk_pad = replace(
            trunk_unpad,
            s=jnp.pad(trunk_unpad.s, ((0, 0), (0, 12), (0, 0))),
            z=jnp.pad(trunk_unpad.z, ((0, 0), (0, 12), (0, 12), (0, 0))),
        )
        coords_unpad = diff_fn(
            evaluator,
            inp_20_unpad.features,
            emb_unpad,
            trunk_unpad,
            start_noise_full[:, :n_native_atoms, :],
            step_noise_full[:, :, :n_native_atoms, :],
            diff_key,
        )
        coords_pad_shared = diff_fn(
            evaluator,
            inp_20_pad.features,
            emb_pad,
            shared_trunk_pad,
            start_noise_full,
            step_noise_full,
            diff_key,
        )
        max_coord_diff, _ = _assert_tensor_parity(
            coords_unpad[:, :n_real_atoms],
            coords_pad_shared[:, :n_real_atoms],
            "Joltz2 controlled diffusion (shared trunk)",
            mean_atol=ATOL,
            rel_tol=1e-4,
        )
        coords_pad = diff_fn(
            evaluator,
            inp_20_pad.features,
            emb_pad,
            trunk_pad,
            start_noise_full,
            step_noise_full,
            diff_key,
        )

        conf_fn = eqx.filter_jit(evaluator.confidence_module)
        conf_unpad = conf_fn(
            diff_key, inp_20_unpad.features, emb_unpad, trunk_unpad, disto_unpad, coords_unpad
        )
        conf_pad = conf_fn(
            diff_key, inp_20_pad.features, emb_pad, trunk_pad, disto_pad, coords_pad
        )

        res_unpad = JoltzResult(
            dict(
                features=inp_20_unpad.features,
                embedding=emb_unpad,
                state=trunk_unpad,
                distogram=disto_unpad,
                samples=coords_unpad,
                confidence=conf_unpad,
            )
        )
        res_pad = JoltzResult(
            dict(
                features=inp_20_pad.features,
                embedding=emb_pad,
                state=trunk_pad,
                distogram=disto_pad,
                samples=coords_pad,
                confidence=conf_pad,
            )
        )

        atom24_unpad, mask24_unpad = res_unpad.atom24
        atom24_pad, mask24_pad = res_pad.atom24
        assert atom24_pad.shape == (32, 24, 3)
        np.testing.assert_array_equal(np.asarray(mask24_unpad), np.asarray(mask24_pad[:20]))
        metrics_report = _assert_masked_metrics_parity(res_unpad, res_pad, atol=ATOL)

        # Tier 2: single-trace reuse (compile_count == 1) across L=20 and L=28 padded to 32.
        trace_counter = {"count": 0}

        @eqx.filter_jit
        def jit_predict(rng, feats):
            trace_counter["count"] += 1
            return evaluator._predict(rng, feats, num_samples=1)

        out_20 = jit_predict(rng_key, inp_20_pad.features)
        jax.block_until_ready(out_20.plddt)
        assert trace_counter["count"] == 1

        res_pad28 = jit_predict(rng_key, inp_28_pad.features)
        jax.block_until_ready(res_pad28.plddt)
        assert trace_counter["count"] == 1
        if hasattr(jit_predict, "_cached") and hasattr(jit_predict._cached, "_cache_size"):
            assert jit_predict._cached._cache_size() == 1

        # Verify @jax.jit .to_data().p_dssp and unpadded save_pdb / save_cif.
        _assert_jit_to_data_p_dssp(res_pad, valid_tokens=20, padded_tokens=32)
        _assert_single_sample_result_surface(res_pad, num_tokens=32)

        # Phase 2: pad_atoms(24) compile_count == 1 across Gly20 (96 atoms) vs Trp20 (288 atoms)
        # at fixed L=20 + multi-sample (S=2) JoltzResult metric surface.
        inp_gly_raw, _ = JoltzSpec().add_protein(SEQ_GLY_20, use_msa=False).to_input(cache="./params/boltz/")
        inp_trp_raw, _ = JoltzSpec().add_protein(SEQ_TRP_20, use_msa=False).to_input(cache="./params/boltz/")
        assert inp_gly_raw.features["atom_pad_mask"].shape != inp_trp_raw.features["atom_pad_mask"].shape
        inp_gly_atoms = inp_gly_raw.pad_atoms(24)
        inp_trp_atoms, _ = (
            JoltzSpec().add_protein(SEQ_TRP_20, use_msa=False).pad_atoms(24).to_input(cache="./params/boltz/")
        )
        assert inp_gly_atoms.features["token_pad_mask"].shape == (1, 20)
        assert inp_trp_atoms.features["token_pad_mask"].shape == (1, 20)
        assert inp_gly_atoms.features["atom_pad_mask"].shape == (1, 480)
        assert inp_trp_atoms.features["atom_pad_mask"].shape == (1, 480)

        fast_eval = Joltz2Evaluator(
            joltz=joltz_wrapper.model,
            num_recycle=0,
            num_sampling_steps=16,
            deterministic=True,
        )
        atoms_trace_counter = {"count": 0}

        @eqx.filter_jit
        def jit_atoms_multi(rng, feats):
            atoms_trace_counter["count"] += 1
            return fast_eval._predict(rng, feats, num_samples=2)

        res_gly_s2 = jit_atoms_multi(rng_key, inp_gly_atoms.features)
        jax.block_until_ready(res_gly_s2.plddt)
        assert atoms_trace_counter["count"] == 1

        res_trp_s2 = jit_atoms_multi(rng_key, inp_trp_atoms.features)
        jax.block_until_ready(res_trp_s2.plddt)
        assert atoms_trace_counter["count"] == 1
        if hasattr(jit_atoms_multi, "_cached") and hasattr(jit_atoms_multi._cached, "_cache_size"):
            assert jit_atoms_multi._cached._cache_size() == 1
        _assert_multisample_result_metrics(res_gly_s2, num_samples=2, num_tokens=20)
        _assert_multisample_result_metrics(res_trp_s2, num_samples=2, num_tokens=20)
        del res_trp_s2, res_gly_s2, jit_atoms_multi, fast_eval

        predictor = joltz_wrapper.predictor(
            num_recycle=1, num_samples=1, num_sampling_steps=16, deterministic=True
        )
        pred = predictor(rng_key, inp_20_pad, writer=writer_20)
        pdb_path = tmp_path / "joltz2_trpcage_20.pdb"
        cif_path = tmp_path / "joltz2_trpcage_20.cif"
        pred.save_pdb(str(pdb_path))
        pred.save_cif(str(cif_path))
        _assert_structure_file(pdb_path, expected_residues=20, expected_chains=(10, 10))
        _assert_structure_file(cif_path, expected_residues=20, expected_chains=(10, 10))
        print(
            f"[Joltz2] trunk_z_diff={trunk_z_diff:.2e}, coord_diff={max_coord_diff:.2e}, "
            f"compile_count={trace_counter['count']}, metrics={metrics_report}, rss={_peak_rss_gb():.2f} GiB"
        )
    finally:
        del pred, res_pad28, res_pad, res_unpad, jit_predict, evaluator, joltz_wrapper
        _cleanup_memory()


def test_protenix_e2e_small(tmp_path: Path):
    """Verify Protenix Tier 1 parity (including atom_to_token & padded MSA), Tier 2 reuse, and exports."""
    from protenix.protenij import TrunkEmbedding

    from flexcraft.structure.protenix._data import ProtenixSpec
    from flexcraft.structure.protenix._model import (
        Protenix,
        ProtenixEvaluator,
        sample_diffusion,
    )
    from flexcraft.structure.protenix._result import ProtenixResult

    protenix_wrapper = None
    evaluator = None
    jit_predict = None
    res_unpad = None
    res_pad = None
    res_pad28 = None
    pred = None
    try:
        protenix_wrapper = Protenix(model="protenix_base_default_v1.0.0")
        spec_20 = ProtenixSpec().add_protein(SEQ_20, use_msa=False)
        inp_20_unpad, writer_20 = spec_20.to_input()
        base_msa_rows = int(inp_20_unpad.features["msa"].shape[0])
        target_msa = max(6, base_msa_rows + 4)

        inp_20_pad = inp_20_unpad.pad_to(32, num_residue_atoms=24, num_msa=target_msa)
        inp_28_pad, _ = (
            ProtenixSpec()
            .add_protein(SEQ_28, use_msa=False)
            .pad_to(32, num_residue_atoms=24, num_msa=target_msa)
            .to_input()
        )

        n_native_atoms = int(inp_20_unpad.features["atom_to_token_idx"].shape[0])
        assert inp_20_pad.features["atom_to_token_idx"].shape == (32 * 24,)
        assert inp_20_pad.features["msa"].shape == (target_msa, 32)

        evaluator = ProtenixEvaluator(
            model=protenix_wrapper.model,
            sampling_steps=20,
            num_recycle=1,
        )

        # Tier 1: deterministic trunk (with padded num_msa!) + controlled-noise diffusion + confidence.
        @eqx.filter_jit
        def run_trunk_and_disto(eval_mod, key, feats):
            emb = eval_mod.model.embed_inputs(input_feature_dict=feats)
            state0 = TrunkEmbedding(
                s=jnp.zeros_like(emb.s_init), z=jnp.zeros_like(emb.z_init)
            )
            state = eval_mod._run_trunk(
                key, state0, feats, emb, token_pad_mask=feats.get("token_pad_mask")
            )
            disto = eval_mod.model.distogram_head(state.z)
            return emb, state, disto

        rng_key = jax.random.PRNGKey(11)
        trunk_key, diff_key, conf_key, noise_k0, noise_k1 = jax.random.split(rng_key, 5)

        emb_unpad, state_unpad, disto_unpad = run_trunk_and_disto(
            evaluator, trunk_key, inp_20_unpad.features
        )
        emb_pad, state_pad, disto_pad = run_trunk_and_disto(
            evaluator, trunk_key, inp_20_pad.features
        )

        _assert_tensor_parity(
            emb_unpad.s_inputs, emb_pad.s_inputs[:20], "Protenix s_inputs", strict_max_atol=ATOL
        )
        _assert_tensor_parity(
            emb_unpad.z_init, emb_pad.z_init[:20, :20], "Protenix z_init", strict_max_atol=ATOL
        )
        trunk_s_diff, _ = _assert_tensor_parity(
            state_unpad.s, state_pad.s[:20], "Protenix trunk.s (padded MSA)"
        )
        trunk_z_diff, _ = _assert_tensor_parity(
            state_unpad.z, state_pad.z[:20, :20], "Protenix trunk.z (padded MSA)"
        )
        disto_diff, _ = _assert_tensor_parity(
            disto_unpad, disto_pad[:20, :20], "Protenix distogram"
        )

        n_steps = evaluator.sampling_steps
        noise_schedule = evaluator.model.inference_noise_scheduler(N_step=n_steps)
        init_noise_full = jax.random.normal(noise_k0, (1, 32 * 24, 3))
        step_noise_full = jax.random.normal(noise_k1, (n_steps, 1, 32 * 24, 3))

        @eqx.filter_jit
        def run_controlled_diffusion_and_conf(eval_mod, feats, emb, state, init_noise, step_noise):
            coords = sample_diffusion(
                denoise_net=eval_mod.model.diffusion_module,
                input_feature_dict=feats,
                s_inputs=emb.s_inputs,
                s_trunk=state.s,
                z_trunk=state.z,
                N_sample=1,
                noise_schedule=noise_schedule,
                gamma0=eval_mod.model.gamma0,
                gamma_min=eval_mod.model.gamma_min,
                noise_scale_lambda=eval_mod.model.noise_scale_lambda,
                step_scale_eta=eval_mod.model.step_scale_eta,
                key=diff_key,
                initial_noise=init_noise,
                step_noise=step_noise,
            )
            conf = eval_mod.model.confidence_metrics(
                initial_embedding=emb,
                trunk_embedding=state,
                input_feature_dict=feats,
                coordinates=coords,
                key=conf_key,
            )
            return coords, conf

        coords_unpad, conf_unpad = run_controlled_diffusion_and_conf(
            evaluator,
            inp_20_unpad.features,
            emb_unpad,
            state_unpad,
            init_noise_full[:, :n_native_atoms, :],
            step_noise_full[:, :, :n_native_atoms, :],
        )
        shared_state_pad = replace(
            state_unpad,
            s=jnp.pad(state_unpad.s, ((0, 12), (0, 0))),
            z=jnp.pad(state_unpad.z, ((0, 12), (0, 12), (0, 0))),
        )
        coords_pad_shared, _ = run_controlled_diffusion_and_conf(
            evaluator,
            inp_20_pad.features,
            emb_pad,
            shared_state_pad,
            init_noise_full,
            step_noise_full,
        )
        max_coord_diff, _ = _assert_tensor_parity(
            coords_unpad[:, :n_native_atoms],
            coords_pad_shared[:, :n_native_atoms],
            "Protenix controlled diffusion (shared trunk)",
            mean_atol=ATOL,
            rel_tol=1e-4,
        )
        coords_pad, conf_pad = run_controlled_diffusion_and_conf(
            evaluator,
            inp_20_pad.features,
            emb_pad,
            state_pad,
            init_noise_full,
            step_noise_full,
        )

        res_unpad = ProtenixResult(
            dict(
                features=inp_20_unpad.features,
                embedding=emb_unpad,
                state=state_unpad,
                distogram=disto_unpad,
                samples=coords_unpad,
                confidence=conf_unpad,
            )
        )
        res_pad = ProtenixResult(
            dict(
                features=inp_20_pad.features,
                embedding=emb_pad,
                state=state_pad,
                distogram=disto_pad,
                samples=coords_pad,
                confidence=conf_pad,
            )
        )

        # Verify atom_to_token token-0 fix: padded atoms have strictly 0 weight on token 0.
        assert jnp.all(res_pad.atom_to_token[n_native_atoms:] == 0.0)
        assert int(res_pad.atom_to_token[:, 0].sum()) == int(res_unpad.atom_to_token[:, 0].sum())

        atom24_unpad, mask24_unpad = res_unpad.atom24
        atom24_pad, mask24_pad = res_pad.atom24
        assert atom24_pad.shape == (32, 24, 3)
        np.testing.assert_array_equal(np.asarray(mask24_unpad), np.asarray(mask24_pad[:20]))
        metrics_report = _assert_masked_metrics_parity(res_unpad, res_pad, atol=ATOL)

        # Tier 2: single-trace reuse (compile_count == 1) across L=20 and L=28 padded to 32.
        trace_counter = {"count": 0}

        @eqx.filter_jit
        def jit_predict(rng, feats):
            trace_counter["count"] += 1
            return evaluator._predict(rng, feats, num_samples=1)

        out_20 = jit_predict(rng_key, inp_20_pad.features)
        jax.block_until_ready(out_20.plddt)
        assert trace_counter["count"] == 1

        res_pad28 = jit_predict(rng_key, inp_28_pad.features)
        jax.block_until_ready(res_pad28.plddt)
        assert trace_counter["count"] == 1
        if hasattr(jit_predict, "_cached") and hasattr(jit_predict._cached, "_cache_size"):
            assert jit_predict._cached._cache_size() == 1

        # Verify @jax.jit .to_data().p_dssp and unpadded save_pdb / save_cif.
        _assert_jit_to_data_p_dssp(res_pad, valid_tokens=20, padded_tokens=32)
        _assert_single_sample_result_surface(res_pad, num_tokens=32)

        # Phase 2: pad_atoms(24) compile_count == 1 across Gly20 (82 atoms) vs Trp20 (282 atoms)
        # at fixed L=20 + multi-sample (S=2) ProtenixResult metric surface.
        inp_gly_raw, _ = ProtenixSpec().add_protein(SEQ_GLY_20, use_msa=False).to_input()
        inp_trp_raw, _ = ProtenixSpec().add_protein(SEQ_TRP_20, use_msa=False).to_input()
        assert inp_gly_raw.features["atom_to_token_idx"].shape == (82,)
        assert inp_trp_raw.features["atom_to_token_idx"].shape == (282,)
        inp_gly_atoms = inp_gly_raw.pad_atoms(24)
        inp_trp_atoms, _ = ProtenixSpec().add_protein(SEQ_TRP_20, use_msa=False).pad_atoms(24).to_input()
        assert inp_gly_atoms.features["residue_index"].shape == (20,)
        assert inp_trp_atoms.features["residue_index"].shape == (20,)
        assert inp_gly_atoms.features["atom_to_token_idx"].shape == (480,)
        assert inp_trp_atoms.features["atom_to_token_idx"].shape == (480,)

        multi_eval_fn, multi_eval_params = protenix_wrapper.evaluator(
            num_sampling_steps=8, num_recycle=0
        )
        multi_eval = multi_eval_fn(multi_eval_params)
        atoms_trace_counter = {"count": 0}

        @eqx.filter_jit
        def jit_atoms_multi(rng, feats):
            atoms_trace_counter["count"] += 1
            return multi_eval._predict(rng, feats, num_samples=2)

        res_gly_s2 = jit_atoms_multi(rng_key, inp_gly_atoms.features)
        jax.block_until_ready(res_gly_s2.plddt)
        assert atoms_trace_counter["count"] == 1

        res_trp_s2 = jit_atoms_multi(rng_key, inp_trp_atoms.features)
        jax.block_until_ready(res_trp_s2.plddt)
        assert atoms_trace_counter["count"] == 1
        if hasattr(jit_atoms_multi, "_cached") and hasattr(jit_atoms_multi._cached, "_cache_size"):
            assert jit_atoms_multi._cached._cache_size() == 1
        _assert_multisample_result_metrics(res_gly_s2, num_samples=2, num_tokens=20)
        _assert_multisample_result_metrics(res_trp_s2, num_samples=2, num_tokens=20)
        del res_trp_s2, res_gly_s2, jit_atoms_multi, multi_eval, multi_eval_params

        predictor = protenix_wrapper.predictor(
            num_recycle=1, num_samples=1, num_sampling_steps=25, deterministic=False
        )
        pred = predictor(rng_key, inp_20_pad, writer=writer_20)
        pdb_path = tmp_path / "protenix_trpcage_20.pdb"
        cif_path = tmp_path / "protenix_trpcage_20.cif"
        pred.save_pdb(str(pdb_path))
        pred.save_cif(str(cif_path))
        _assert_structure_file(pdb_path, expected_residues=20, expected_chains=(10, 10))
        _assert_structure_file(cif_path, expected_residues=20, expected_chains=(10, 10))
        print(
            f"[Protenix] trunk_z_diff={trunk_z_diff:.2e}, coord_diff={max_coord_diff:.2e}, "
            f"compile_count={trace_counter['count']}, metrics={metrics_report}, rss={_peak_rss_gb():.2f} GiB"
        )
    finally:
        del pred, res_pad28, res_pad, res_unpad, jit_predict, evaluator, protenix_wrapper
        _cleanup_memory()


def test_esmfold2_fast_no_esmc_e2e_small(tmp_path: Path):
    """Verify ESMFold2 (load_esmc=False) Tier 1 parity, Tier 2 single-trace reuse, set_aa sync, and exports."""
    from esmjfold2.model import TrunkInputs

    from flexcraft.structure.esmfold2._data import (
        ESM_PAD_ID,
        ESMFold2Spec,
        _BOLTZ_AA_TO_ESM_ID,
        to_model_features,
    )
    from flexcraft.structure.esmfold2._model import ESMFold2, ESMFold2Evaluator
    from flexcraft.structure.esmfold2._result import ESMFold2Confidence, ESMFold2Result

    esm_wrapper = None
    det_model = None
    evaluator = None
    jit_predict = None
    res_unpad = None
    res_pad = None
    res_pad28 = None
    pred = None
    try:
        esm_wrapper = ESMFold2(
            "esmfold2_fast",
            cache="./params/esmfold2/",
            load_esmc=False,
            esmc_device="cpu",
        )
        assert esm_wrapper.esmc is None

        spec_20 = ESMFold2Spec().add_protein(SEQ_20, use_msa=False)
        inp_20_unpad, writer_20 = spec_20.to_input(cache="./params/esmfold2/")
        inp_20_pad = inp_20_unpad.pad_to(32, num_residue_atoms=24)
        inp_28_pad, _ = (
            ESMFold2Spec()
            .add_protein(SEQ_28, use_msa=False)
            .pad_to(32, num_residue_atoms=24)
            .to_input(cache="./params/esmfold2/")
        )

        # Verify zero-padded 3D res_type invariant on padded ESMFold2Input.
        assert inp_20_pad.features["res_type"].shape == (1, 32, 33)
        assert jnp.all(inp_20_pad.features["res_type"][0, 20:] == 0.0)
        assert inp_20_pad.features["atom_attention_mask"].shape == (1, 768)

        n_native_atoms = int(inp_20_unpad.features["atom_attention_mask"].shape[1])
        n_real_atoms = int(np.asarray(inp_20_unpad.features["atom_attention_mask"]).sum())

        # Isolate lm_dropout=0.0 via dataclasses.replace for Tier 1 deterministic parity.
        det_model = replace(esm_wrapper.model, lm_dropout=0.0)

        rng_key = jax.random.PRNGKey(19)
        z0_key, trunk_key, x_key = jax.random.split(rng_key, 3)

        @eqx.filter_jit
        def run_tier1_components(model, feats, z0, x_noisy):
            ctx = model._prepare_embeddings(to_model_features(feats))
            inputs = TrunkInputs(context=ctx, lm_z=None)
            z_step = model._trunk_pass(inputs, z0 * ctx.pair_mask[..., None], trunk_key, 1024)
            z_readout = model.readout(inputs, z_step)
            disto = model._compute_distogram(z_readout)[0]

            dm = model.structure_head.diffusion_module
            cache = dm.prepare(
                ref_pos=ctx.ref_pos,
                ref_charge=ctx.ref_charge,
                ref_mask=ctx.atm_mask,
                ref_element_oh=ctx.ref_element_oh,
                ref_atom_name_chars_oh=ctx.ref_atom_name_chars_oh,
                ref_space_uid=ctx.ref_space_uid,
                tok_idx=ctx.atom_to_token,
                s_inputs=ctx.x_inputs,
                z_trunk=z_readout,
                relative_position_encoding=ctx.relative_position_encoding,
                n_tokens=ctx.n_tokens,
                token_attention_mask=ctx.tok_mask,
            )
            masked_x = x_noisy * ctx.atm_mask[..., None].astype(jnp.float32)
            x_denoised = dm.denoise(cache, masked_x, jnp.array([1.0], dtype=jnp.float32))
            x_denoised = x_denoised * ctx.atm_mask[..., None].astype(jnp.float32)
            conf = ESMFold2Confidence.from_head(
                model._compute_confidence(ctx, z_readout, x_denoised)
            )
            return ctx.x_inputs, ctx.z_init, z_readout, disto, x_denoised, conf

        # Controlled initial pair state z_0 and noisy atom coordinates.
        z0_full = 0.05 * jax.random.normal(z0_key, (1, 32, 32, det_model.d_pair))
        x_noisy_full = jax.random.normal(x_key, (1, 768, 3))

        x_in_u, z_init_u, z_out_u, disto_u, coords_u, conf_u = run_tier1_components(
            det_model,
            inp_20_unpad.features,
            z0_full[:, :20, :20],
            x_noisy_full[:, :n_native_atoms],
        )
        x_in_p, z_init_p, z_out_p, disto_p, coords_p, conf_p = run_tier1_components(
            det_model,
            inp_20_pad.features,
            z0_full,
            x_noisy_full,
        )

        emb_x_diff, _ = _assert_tensor_parity(
            x_in_u, x_in_p[:, :20], "ESMFold2 _prepare_embeddings x_inputs", strict_max_atol=ATOL
        )
        emb_z_diff, _ = _assert_tensor_parity(
            z_init_u, z_init_p[:, :20, :20], "ESMFold2 _prepare_embeddings z_init", strict_max_atol=ATOL
        )
        trunk_z_diff, _ = _assert_tensor_parity(
            z_out_u, z_out_p[:, :20, :20], "ESMFold2 _trunk_pass/readout"
        )
        denoise_diff, _ = _assert_tensor_parity(
            coords_u[:, :n_real_atoms],
            coords_p[:, :n_real_atoms],
            "ESMFold2 DiffusionModule.denoise",
            strict_max_atol=ATOL,
        )

        res_unpad = ESMFold2Result(
            dict(
                features=inp_20_unpad.features,
                state=z_out_u,
                distogram=disto_u,
                samples=coords_u,
                confidence=conf_u,
            )
        )
        res_pad = ESMFold2Result(
            dict(
                features=inp_20_pad.features,
                state=z_out_p,
                distogram=disto_p,
                samples=coords_p,
                confidence=conf_p,
            )
        )

        atom24_unpad, mask24_unpad = res_unpad.atom24
        atom24_pad, mask24_pad = res_pad.atom24
        assert atom24_pad.shape == (32, 24, 3)
        np.testing.assert_array_equal(np.asarray(mask24_unpad), np.asarray(mask24_pad[:20]))
        atom24_diff, _ = _assert_tensor_parity(
            np.asarray(atom24_unpad) * np.asarray(mask24_unpad)[..., None],
            np.asarray(atom24_pad[:20]) * np.asarray(mask24_unpad)[..., None],
            "ESMFold2 atom24",
            strict_max_atol=ATOL,
        )
        metrics_report = _assert_masked_metrics_parity(res_unpad, res_pad, atol=ATOL)

        # Tier 2: single-trace reuse (compile_count == 1) across L=20, L=28, and set_aa!
        evaluator = ESMFold2Evaluator(
            model=esm_wrapper.model,
            num_loops=1,
            num_sampling_steps=2,
            noise_scale=0.0,
        )
        trace_counter = {"count": 0}

        @eqx.filter_jit
        def jit_predict(rng, feats):
            trace_counter["count"] += 1
            return evaluator._predict(rng, feats, lm_hidden_states=None, num_samples=1)

        out_20 = jit_predict(rng_key, inp_20_pad.features)
        jax.block_until_ready(out_20.plddt)
        assert trace_counter["count"] == 1

        res_pad28 = jit_predict(rng_key, inp_28_pad.features)
        jax.block_until_ready(res_pad28.plddt)
        assert trace_counter["count"] == 1

        # Verify set_aa synchronizes input_ids on padded input and reuses the single XLA trace!
        mutated_aa = jax.nn.one_hot(jnp.full((32,), 0, dtype=jnp.int32), 20, dtype=jnp.float32)
        inp_20_mut = inp_20_pad.set_aa(mutated_aa)
        assert jnp.all(inp_20_mut.features["input_ids"][0, :20] == _BOLTZ_AA_TO_ESM_ID[0])
        assert jnp.all(inp_20_mut.features["input_ids"][0, 20:] == ESM_PAD_ID)
        assert jnp.all(inp_20_mut.features["res_type"][0, 20:] == 0.0)

        out_mut = jit_predict(rng_key, inp_20_mut.features)
        jax.block_until_ready(out_mut.plddt)
        assert trace_counter["count"] == 1
        if hasattr(jit_predict, "_cached") and hasattr(jit_predict._cached, "_cache_size"):
            assert jit_predict._cached._cache_size() == 1

        # Verify @jax.jit .to_data().p_dssp and unpadded save_pdb / save_cif.
        _assert_jit_to_data_p_dssp(res_pad, valid_tokens=20, padded_tokens=32)
        _assert_single_sample_result_surface(res_pad, num_tokens=32)

        # Phase 2: pad_atoms(24) compile_count == 1 across Gly20 (96 atoms) vs Trp20 (288 atoms)
        # at fixed L=20 + multi-sample (S=2) ESMFold2Result metric surface.
        inp_gly_raw, _ = ESMFold2Spec().add_protein(SEQ_GLY_20, use_msa=False).to_input(cache="./params/esmfold2/")
        inp_trp_raw, _ = ESMFold2Spec().add_protein(SEQ_TRP_20, use_msa=False).to_input(cache="./params/esmfold2/")
        assert inp_gly_raw.features["atom_attention_mask"].shape != inp_trp_raw.features["atom_attention_mask"].shape
        inp_gly_atoms = inp_gly_raw.pad_atoms(24)
        inp_trp_atoms, _ = (
            ESMFold2Spec().add_protein(SEQ_TRP_20, use_msa=False).pad_atoms(24).to_input(cache="./params/esmfold2/")
        )
        assert inp_gly_atoms.features["token_attention_mask"].shape == (1, 20)
        assert inp_trp_atoms.features["token_attention_mask"].shape == (1, 20)
        assert inp_gly_atoms.features["atom_attention_mask"].shape == (1, 480)
        assert inp_trp_atoms.features["atom_attention_mask"].shape == (1, 480)

        atoms_trace_counter = {"count": 0}

        @eqx.filter_jit
        def jit_atoms_multi(rng, feats):
            atoms_trace_counter["count"] += 1
            return evaluator._predict(rng, feats, lm_hidden_states=None, num_samples=2)

        res_gly_s2 = jit_atoms_multi(rng_key, inp_gly_atoms.features)
        jax.block_until_ready(res_gly_s2.plddt)
        assert atoms_trace_counter["count"] == 1

        res_trp_s2 = jit_atoms_multi(rng_key, inp_trp_atoms.features)
        jax.block_until_ready(res_trp_s2.plddt)
        assert atoms_trace_counter["count"] == 1
        if hasattr(jit_atoms_multi, "_cached") and hasattr(jit_atoms_multi._cached, "_cache_size"):
            assert jit_atoms_multi._cached._cache_size() == 1
        _assert_multisample_result_metrics(res_gly_s2, num_samples=2, num_tokens=20)
        _assert_multisample_result_metrics(res_trp_s2, num_samples=2, num_tokens=20)
        del res_trp_s2, res_gly_s2, jit_atoms_multi

        predictor = esm_wrapper.predictor(
            num_recycle=1, num_samples=1, num_sampling_steps=2, deterministic=True
        )
        pred = predictor(rng_key, inp_20_pad, writer=writer_20)
        pdb_path = tmp_path / "esmfold2_no_esmc_20.pdb"
        cif_path = tmp_path / "esmfold2_no_esmc_20.cif"
        pred.save_pdb(str(pdb_path))
        pred.save_cif(str(cif_path))
        _assert_structure_file(pdb_path, expected_residues=20, expected_chains=(10, 10))
        _assert_structure_file(cif_path, expected_residues=20, expected_chains=(10, 10))
        print(
            f"[ESMFold2 load_esmc=False] trunk_z_diff={trunk_z_diff:.2e}, denoise_diff={denoise_diff:.2e}, "
            f"compile_count={trace_counter['count']}, metrics={metrics_report}, rss={_peak_rss_gb():.2f} GiB"
        )
    finally:
        del pred, res_pad28, res_pad, res_unpad, jit_predict, evaluator, det_model, esm_wrapper
        _cleanup_memory()


def test_esmfold2_fast_with_esmc_e2e_small(tmp_path: Path):
    """Verify ESMFold2 (load_esmc=True, CPU ESMC-6B) memory bound, _run_esmc compile_count == 1, LM parity, and exports."""
    from esmjfold2 import lm_features as _lm_features

    from flexcraft.structure.esmfold2._data import ESMFold2Spec
    from flexcraft.structure.esmfold2._model import ESMFold2

    esm_full = None
    pred_16 = None
    pred_22 = None
    try:
        _cleanup_memory()
        esm_full = ESMFold2(
            "esmfold2_fast",
            cache="./params/esmfold2/",
            load_esmc=True,
            esmc_device="cpu",
        )
        assert esm_full.esmc is not None
        rss_after_load = _peak_rss_gb()
        assert rss_after_load < 46.0, f"Peak RSS {rss_after_load:.2f} GiB approached 48 GiB limit!"

        spec_16 = ESMFold2Spec().add_protein(SEQ_16, use_msa=False)
        inp_16_unpad, writer_16 = spec_16.to_input(cache="./params/esmfold2/")
        inp_16_pad = inp_16_unpad.pad_to(32, num_residue_atoms=24)

        spec_22 = ESMFold2Spec().add_protein(SEQ_22, use_msa=False)
        inp_22_unpad, writer_22 = spec_22.to_input(cache="./params/esmfold2/")
        inp_22_pad = inp_22_unpad.pad_to(32, num_residue_atoms=24)

        esmc_trace_counter = {"count": 0}

        @eqx.filter_jit
        def traced_run_esmc(esmc_mod, input_ids, sequence_id):
            esmc_trace_counter["count"] += 1
            _, hs = esmc_mod(input_ids, sequence_id, collect_hidden_states=True)
            return hs

        rng_key = jax.random.PRNGKey(29)
        with patch.object(_lm_features, "_run_esmc", traced_run_esmc):
            predictor = esm_full.predictor(
                num_recycle=1, num_samples=1, num_sampling_steps=2, deterministic=True
            )
            pred_16 = predictor(rng_key, inp_16_pad, writer=writer_16)
            assert esmc_trace_counter["count"] == 1

            pred_22 = predictor(rng_key, inp_22_pad, writer=writer_22)
            assert esmc_trace_counter["count"] == 1
            if hasattr(traced_run_esmc, "_cached") and hasattr(traced_run_esmc._cached, "_cache_size"):
                assert traced_run_esmc._cached._cache_size() == 1

            # Also grab padded lm_hidden_states on L=16 without triggering a second compile!
            hidden_16_pad = esm_full.lm_hidden_states(inp_16_pad.features)
            assert esmc_trace_counter["count"] == 1
        esmc_compile_count = esmc_trace_counter["count"]
        del predictor, traced_run_esmc
        _cleanup_memory()

        # Compute unpadded ESMC reference eagerly on L=16 (avoids compiling a second 26 GB executable)
        # and verify padded lm_hidden_states[:, :16] matches within atol <= 5e-5!
        feats_u = inp_16_unpad.features
        lm_ids_u, seq_id_u, expand_maps_u = _lm_features._prepare_lm_inputs(
            np.asarray(feats_u["input_ids"]),
            np.asarray(feats_u["asym_id"]),
            np.asarray(feats_u["residue_index"]),
            np.asarray(feats_u["mol_type"]),
            np.asarray(feats_u["token_attention_mask"]),
        )
        with jax.default_device(esm_full.esmc_device):
            _, hs_u = esm_full.esmc(
                jnp.asarray(lm_ids_u, dtype=jnp.int32),
                jnp.asarray(seq_id_u, dtype=jnp.int32),
                collect_hidden_states=True,
            )
        hs_u_perm = np.transpose(np.asarray(hs_u), (1, 2, 0, 3))
        em_u = np.asarray(expand_maps_u)
        safe_em_u = np.where(em_u < 0, 0, em_u)
        hidden_16_unpad = np.where(
            (em_u >= 0)[..., None, None],
            np.take_along_axis(hs_u_perm, safe_em_u[:, :, None, None], axis=1),
            0.0,
        )

        hidden_16_pad_np = np.asarray(hidden_16_pad)
        assert hidden_16_pad_np.shape[:2] == (1, 32)
        assert np.all(hidden_16_pad_np[:, 16:] == 0.0)
        lm_max_diff, lm_mean_diff = _assert_tensor_parity(
            hidden_16_unpad,
            hidden_16_pad_np[:, :16],
            "ESMC-6B lm_hidden_states",
            mean_atol=ATOL,
            rel_tol=1e-4,
        )

        pdb_16 = tmp_path / "esmfold2_esmc_16.pdb"
        cif_16 = tmp_path / "esmfold2_esmc_16.cif"
        pdb_22 = tmp_path / "esmfold2_esmc_22.pdb"
        cif_22 = tmp_path / "esmfold2_esmc_22.cif"
        pred_16.save_pdb(str(pdb_16))
        pred_16.save_cif(str(cif_16))
        pred_22.save_pdb(str(pdb_22))
        pred_22.save_cif(str(cif_22))
        _assert_structure_file(pdb_16, expected_residues=16, expected_chains=(8, 8))
        _assert_structure_file(cif_16, expected_residues=16, expected_chains=(8, 8))
        _assert_structure_file(pdb_22, expected_residues=22, expected_chains=(11, 11))
        _assert_structure_file(cif_22, expected_residues=22, expected_chains=(11, 11))

        rss_peak = _peak_rss_gb()
        assert rss_peak < 46.0, f"Peak RSS {rss_peak:.2f} GiB approached 48 GiB limit!"
        print(
            f"[ESMFold2 load_esmc=True] esmc_compile_count={esmc_compile_count}, "
            f"lm_max_diff={lm_max_diff:.2e}, lm_mean_diff={lm_mean_diff:.2e}, peak_rss={rss_peak:.2f} GiB"
        )
    finally:
        del pred_22, pred_16, esm_full
        _cleanup_memory()
