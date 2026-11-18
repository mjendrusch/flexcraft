"""Protenix model wrapper with padding-aware MSA masking and recycling."""

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from protenix.backend import load_model
from protenix.protenij import ConfidenceMetrics, InitialEmbedding, TrunkEmbedding
from protenix.protenij import Protenix as _Protenix

try:
    from protenix.protenij import mask_atom_values as _protenij_mask_values
except ImportError:
    try:
        from protenix.atom_padding import mask_atom_values as _protenij_mask_values
    except ImportError:
        _protenij_mask_values = None

try:
    from protenix.protenij import center_atom_coordinates as _protenij_center_atoms
except ImportError:
    try:
        from protenix.atom_padding import (
            center_atom_coordinates as _protenij_center_atoms,
        )
    except ImportError:
        _protenij_center_atoms = None

from flexcraft.structure.common._data import InputSpec
from flexcraft.structure.protenix._data import ProtenixInput, ProtenixSpec
from flexcraft.structure.protenix._result import ProtenixPrediction, ProtenixResult


def _mask_token_values(
    values: jax.Array, mask: jax.Array | None
) -> jax.Array:
    """Zero padded token or atom positions along axis -2."""
    if mask is None:
        return values
    if _protenij_mask_values is not None:
        return _protenij_mask_values(values, mask)
    return values * (jnp.asarray(mask) > 0).astype(values.dtype)[..., None]


def _center_atom_coordinates(
    coords: jax.Array, atom_pad_mask: jax.Array | None
) -> jax.Array:
    """Center atom coordinates over valid atoms and zero padded atoms."""
    if atom_pad_mask is None:
        return coords - jnp.mean(coords, axis=-2, keepdims=True)
    if _protenij_center_atoms is not None:
        return _protenij_center_atoms(coords, atom_pad_mask)
    mask = (jnp.asarray(atom_pad_mask) > 0).astype(coords.dtype)[..., None]
    denom = jnp.maximum(jnp.sum(mask, axis=-2, keepdims=True), 1.0)
    mean = jnp.sum(coords * mask, axis=-2, keepdims=True) / denom
    return (coords - mean) * mask


# Adapted from escalante-bio/protenij under Apache 2.0 license, clamping
# `t_hat**2 - c_tau_last**2` at 0.0 to avoid negative `-ulp/2` FMA rounding
# inside `jnp.sqrt` when `gamma == 0.0` (`t_hat == c_tau_last`).
def sample_diffusion(
    *,
    denoise_net,
    input_feature_dict: dict,
    s_inputs,
    s_trunk,
    z_trunk,
    noise_schedule,
    N_sample: int = 1,
    gamma0: float = 0.8,
    gamma_min: float = 1.0,
    noise_scale_lambda: float = 1.003,
    step_scale_eta: float = 1.5,
    key,
    initial_noise=None,
    step_noise=None,
):
    """Sample diffusion coordinates with atom-mask centering and FMA-safe sqrt."""
    n_atom = input_feature_dict["atom_to_token_idx"].shape[-1]
    batch_shape = s_inputs.shape[:-2]
    atom_pad_mask = input_feature_dict.get("atom_pad_mask", None)

    if initial_noise is None:
        init_eps = jax.random.normal(
            key=key, shape=(*batch_shape, N_sample, n_atom, 3)
        )
    else:
        init_eps = initial_noise
    x_l = noise_schedule[0] * init_eps
    x_l = _mask_token_values(x_l, atom_pad_mask)

    @jax.checkpoint
    def body_function(carry, step_inputs):
        x_curr, step_key = carry
        c_tau_last, c_tau, step_eps = step_inputs
        x_curr = _center_atom_coordinates(x_curr, atom_pad_mask)

        gamma = jax.lax.select(c_tau > gamma_min, gamma0, 0.0)
        t_hat = c_tau_last * (gamma + 1)

        delta_noise_level = jnp.sqrt(
            jnp.maximum(t_hat**2 - c_tau_last**2, 0.0)
        )
        step_key = jax.random.fold_in(step_key, 1)
        if step_eps is None:
            step_eps = jax.random.normal(key=step_key, shape=x_curr.shape)
        x_noisy = x_curr + noise_scale_lambda * delta_noise_level * step_eps
        x_noisy = _mask_token_values(x_noisy, atom_pad_mask)

        t_hat_batched = jnp.tile(
            t_hat.reshape((1,) * (len(batch_shape) + 1)),
            (*batch_shape, N_sample),
        )

        x_denoised = denoise_net(
            x_noisy=x_noisy,
            t_hat_noise_level=t_hat_batched,
            input_feature_dict=input_feature_dict,
            s_inputs=s_inputs,
            s_trunk=s_trunk,
            z_trunk=z_trunk,
        )

        delta = (x_noisy - x_denoised) / t_hat_batched[..., None, None]
        dt = c_tau - t_hat_batched
        x_next = x_noisy + step_scale_eta * dt[..., None, None] * delta
        x_next = _mask_token_values(x_next, atom_pad_mask)
        return (x_next, step_key), None

    (x_l, _), _ = jax.lax.scan(
        body_function,
        init=(x_l, key),
        xs=(noise_schedule[:-1], noise_schedule[1:], step_noise),
    )
    return _mask_token_values(x_l, atom_pad_mask)


class _MaskedOPM(eqx.Module):
    """Mask padded MSA rows in Protenix OuterProductMean with exact norm parity."""

    opm: Any
    msa_mask: jax.Array

    def __getattr__(self, name: str):
        return getattr(self.opm, name)

    def __call__(self, m: jax.Array, token_mask: jax.Array | None = None) -> jax.Array:
        m = _mask_token_values(m, token_mask)
        mask = self.msa_mask[: m.shape[-3]].astype(jnp.float32)
        if token_mask is not None:
            tok_f = (jnp.asarray(token_mask) > 0).astype(jnp.float32)
            mask = mask * jnp.broadcast_to(tok_f, m.shape[:-1])
        ln = self.opm.layer_norm(m)
        mask_3d = mask[..., None]
        a = mask_3d * self.opm.linear_1(ln)
        b = mask_3d * self.opm.linear_2(ln)
        a = jnp.swapaxes(a, -2, -3)
        b = jnp.swapaxes(b, -2, -3)
        outer = jnp.einsum("...bac,...dae->...bdce", a, b)
        outer = outer.reshape(outer.shape[:-2] + (-1,))
        outer = self.opm.linear_out(outer)
        norm = (
            jnp.einsum("...abc,...adc->...bdc", mask_3d, mask_3d)
            + self.opm.eps
        )
        return outer / norm


class _MaskedMSAModule(eqx.Module):
    """Prioritize valid MSA rows and mask padded rows across MSAModule blocks."""

    inner: Any

    def __getattr__(self, name: str):
        return getattr(self.inner, name)

    def _subsample_limit(self, n_msa: int) -> int:
        cfg = getattr(self.inner, "msa_configs", None)
        training = bool(getattr(self.inner, "training", False))
        if cfg is not None:
            cutoff_key = "train_cutoff" if training else "test_cutoff"
            val = (
                cfg.get(cutoff_key)
                if hasattr(cfg, "get")
                else getattr(cfg, cutoff_key, None)
            )
            if isinstance(val, int) and val > 0:
                return min(val, n_msa)
            for key in ("sample_cutoff", "test_cutoff", "n_sample", "max_msa_size"):
                val = cfg.get(key) if hasattr(cfg, "get") else getattr(cfg, key, None)
                if isinstance(val, int) and val > 0:
                    return min(val, n_msa)
        for attr in (
            "test_cutoff",
            "sample_cutoff",
            "max_msa_size",
            "num_subsampled_msa",
            "n_sample",
        ):
            val = getattr(self.inner, attr, None)
            if isinstance(val, int) and val > 0:
                return min(val, n_msa)
        return n_msa

    def __call__(self, features, z, s_inputs, pair_mask=None, key=None):
        msa_mask = features.get("msa_mask", None)
        if (
            msa_mask is None
            or "msa" not in features
            or jnp.asarray(msa_mask).ndim < 2
        ):
            return self.inner(
                features, z, s_inputs, pair_mask=pair_mask, key=key
            )

        if getattr(self.inner, "stacked_block_params", None) is None:
            return z

        msa_mask_arr = jnp.asarray(msa_mask, dtype=jnp.float32)
        row_valid = (msa_mask_arr > 0).any(axis=-1)
        n_msa = int(row_valid.shape[0])
        row_indices = jnp.arange(n_msa, dtype=jnp.int32)

        cutoff = self._subsample_limit(n_msa)
        if n_msa > cutoff:
            if key is not None:
                key, sample_key = jax.random.split(key)
                rand_tie = jax.random.uniform(sample_key, (n_msa,))
            else:
                rand_tie = row_indices.astype(jnp.float32) / max(1, n_msa)
            priority = jnp.where(
                row_valid,
                rand_tie,
                2.0 + row_indices.astype(jnp.float32) / max(1, n_msa),
            )
            priority = priority.at[0].set(-1.0)
            order = jnp.argsort(priority)[:cutoff]
        else:
            priority = jnp.where(row_valid, row_indices, n_msa + row_indices)
            priority = priority.at[0].set(-1)
            order = jnp.argsort(priority)

        masked_features = dict(features)
        for msa_key in ("msa", "has_deletion", "deletion_value"):
            if (
                msa_key in masked_features
                and masked_features[msa_key] is not None
            ):
                val = jnp.asarray(masked_features[msa_key])
                if val.ndim >= 1 and val.shape[0] == n_msa:
                    masked_features[msa_key] = val[order]

        token_mask = masked_features.get("token_pad_mask", None)
        if token_mask is None and pair_mask is not None:
            token_mask = jnp.any(pair_mask, axis=-1)

        reordered_valid = row_valid[order]
        reordered_msa_mask = (
            (msa_mask_arr[order] > 0).astype(jnp.float32)
            * reordered_valid[:, None].astype(jnp.float32)
        )
        if token_mask is not None:
            tok_f = (jnp.asarray(token_mask) > 0).astype(jnp.float32)
            reordered_msa_mask = reordered_msa_mask * tok_f[None, :]
        masked_features["msa_mask"] = reordered_msa_mask

        keys_ordered = getattr(
            self.inner,
            "input_feature_keys_ordered",
            ("msa", "has_deletion", "deletion_value"),
        )
        input_feature_dims = getattr(self.inner, "input_feature", {})
        msa_parts = []
        msa_shape_2d = jnp.asarray(masked_features["msa"]).shape[:2]
        for feat_key in keys_ordered:
            feat_dim = (
                input_feature_dims.get(feat_key)
                if hasattr(input_feature_dims, "get")
                else None
            )
            if not isinstance(feat_dim, int):
                feat_dim = 32 if feat_key == "msa" else 1
            if feat_key in masked_features and masked_features[feat_key] is not None:
                arr = jnp.asarray(masked_features[feat_key])
                if feat_dim > 1 and arr.ndim == 2:
                    arr = jax.nn.one_hot(arr, feat_dim, dtype=s_inputs.dtype)
                elif feat_dim == 1 and arr.ndim == 2:
                    arr = arr[..., None].astype(s_inputs.dtype)
                else:
                    arr = arr.astype(s_inputs.dtype)
            else:
                arr = jnp.zeros((*msa_shape_2d, feat_dim), dtype=s_inputs.dtype)
            msa_parts.append(arr)

        msa_feat = jnp.concatenate(msa_parts, axis=-1)
        msa_sample = self.inner.linear_no_bias_m(
            msa_feat
        ) + self.inner.linear_no_bias_s(s_inputs)
        msa_sample = _mask_token_values(msa_sample, token_mask)
        msa_sample = msa_sample * reordered_valid[:, None, None].astype(
            msa_sample.dtype
        )

        def scan_body(carry, block_params):
            m_curr, z_curr, step_key = carry
            if step_key is not None:
                step_key, block_key = jax.random.split(step_key)
            else:
                block_key = None
            block = eqx.combine(self.inner.block_static, block_params)
            block = eqx.tree_at(
                lambda b: b.outer_product_mean_msa,
                block,
                _MaskedOPM(
                    opm=block.outer_product_mean_msa,
                    msa_mask=reordered_msa_mask,
                ),
            )
            m_curr, z_curr = block(
                m_curr, z_curr, pair_mask, key=block_key
            )
            m_curr = m_curr * reordered_valid[:, None, None].astype(
                m_curr.dtype
            )
            return (m_curr, z_curr, step_key), None

        (_, z_out, _), _ = jax.lax.scan(
            scan_body,
            (msa_sample, z, key),
            self.inner.stacked_block_params,
        )
        return z_out


class Protenix:
    def __init__(self, model="protenix_base_default_v1.0.0", cache=None, **kwargs):
        self.model = load_model(model)

    def predictor(self, num_recycle=2, num_samples=1,
                  num_sampling_steps=25, deterministic=False):
        evaluator = ProtenixEvaluator(
            model=self.model,
            num_recycle=num_recycle,
            sampling_steps=num_sampling_steps)
        if deterministic:
            evaluator = eqx.tree_at(
                lambda m: (m.model.gamma0, m.model.step_scale_eta, m.model.noise_scale_lambda, m.model.N_steps),
                evaluator,
                (0.0, 1.0, 0.0, num_sampling_steps))
        jit_predict = eqx.filter_jit(evaluator._predict)

        def _predict(key, spec, writer=None):
            if isinstance(spec, ProtenixInput):
                input_data = spec
            elif isinstance(spec, (ProtenixSpec, InputSpec)):
                input_data, spec_writer = spec.to_input()
                if writer is None:
                    writer = spec_writer
            else:
                input_data, spec_writer = spec.to_input()
                if writer is None:
                    writer = spec_writer
            prediction = jit_predict(key, input_data.features, num_samples=num_samples)
            if not isinstance(prediction.token_pad_mask, jax.core.Tracer):
                prediction = prediction.unpad()
            return ProtenixPrediction(
                data=prediction.data,
                writer=writer)
        return _predict

    def evaluator(self, num_sampling_steps=25, sample_parallel=False,
                 num_recycle=4, stop_recycle_gradient=True):
        evaluator = ProtenixEvaluator(
            model=self.model, sampling_steps=num_sampling_steps,
            sample_parallel=sample_parallel,
            num_recycle=num_recycle,
            stop_recycle_gradient=stop_recycle_gradient)
        evaluator = eqx.tree_at(
            lambda m: (m.model.gamma0, m.model.step_scale_eta, m.model.noise_scale_lambda, m.model.N_steps),
            evaluator,
            (0.0, 1.0, 1.0, num_sampling_steps))
        evaluator_params, evaluator_static = eqx.partition(evaluator, eqx.is_array)
        def _evaluator(params):
            return eqx.combine(params, evaluator_static)
        return _evaluator, evaluator_params


class ProtenixEvaluator(eqx.Module):
    model: _Protenix
    sampling_steps: int = 25
    num_recycle: int = 4
    sample_parallel: bool = False
    stop_recycle_gradient: bool = True

    def _predict(self, key, features, recycling_state: TrunkEmbedding | None = None, num_samples=1):
        """Run full prediction: embedding -> trunk -> distogram -> diffusion -> confidence."""
        keys = jax.random.split(key, 10)
        embedding = self.model.embed_inputs(input_feature_dict=features)
        state = recycling_state
        if state is None:
            state = TrunkEmbedding(
                s=jnp.zeros_like(embedding.s_init),
                z=jnp.zeros_like(embedding.z_init))

        token_pad_mask = features.get("token_pad_mask", None)
        atom_pad_mask = features.get("atom_pad_mask", None)

        state = self._run_trunk(keys[1], state, features, embedding,
                                token_pad_mask=token_pad_mask)

        log_distogram: jax.Array = self.model.distogram_head(state.z)

        (coords, confidence) = self.structure_module(
            keys[2], state, features, embedding,
            num_samples=num_samples,
            sampling_steps=self.sampling_steps,
            parallel=self.sample_parallel,
            token_pad_mask=token_pad_mask,
            atom_pad_mask=atom_pad_mask)
        return ProtenixResult(dict(
            features=features, embedding=embedding, state=state,
            distogram=log_distogram, samples=coords, confidence=confidence
        ))

    def predict(self, key, protenix_input: ProtenixInput,
                recycling_state: TrunkEmbedding | None = None,
                num_samples=1):
        features = protenix_input.features
        return self._predict(
            key, features, recycling_state=recycling_state,
            num_samples=num_samples)

    def __call__(self, key, protenix_input: ProtenixInput,
                 recycling_state = None, num_samples = 1):
        return self.predict(key, protenix_input,
                            recycling_state = recycling_state,
                            num_samples = num_samples)

    def embedding(self, key, features) -> InitialEmbedding:
        embedding = self.model.embed_inputs(input_feature_dict=features)
        return embedding

    def _get_msa_module(self, features):
        """Return msa_module wrapped via eqx.tree_at to mask padded MSA rows."""
        msa_mask = features.get("msa_mask", None)
        if msa_mask is None or jnp.asarray(msa_mask).ndim < 2:
            return self.model.msa_module
        if isinstance(self.model.msa_module, _MaskedMSAModule):
            return self.model.msa_module
        wrapped_model = eqx.tree_at(
            lambda m: m.msa_module,
            self.model,
            _MaskedMSAModule(inner=self.model.msa_module),
        )
        return wrapped_model.msa_module

    def _run_trunk(self, key, state: TrunkEmbedding, features,
                   embedding: InitialEmbedding,
                   token_pad_mask=None) -> TrunkEmbedding:
        """Run trunk with recycling, delegating to upstream protenij methods.

        num_recycle=R runs R+1 total trunk passes (R recycling + 1 final).
        num_recycle=0 runs 1 pass. This matches Joltz2/AF2 convention.
        """
        pair_mask = None
        if token_pad_mask is not None:
            pair_mask = token_pad_mask[:, None] * token_pad_mask[None, :]

        msa_module = self._get_msa_module(features)

        def body(i, trunk_state):
            state, key = trunk_state
            state: TrunkEmbedding
            if self.stop_recycle_gradient:
                state = jax.tree.map(jax.lax.stop_gradient, state)
            s, z = state.s, state.z
            z = embedding.z_init + self.model.linear_no_bias_z_cycle(
                self.model.layernorm_z_cycle(z)
            )
            if self.model.template_embedder.n_blocks > 0:
                z = z + self.model.template_embedder(features, z, pair_mask=pair_mask, key=key)
            z = msa_module(
                features,
                z,
                embedding.s_inputs,
                pair_mask=pair_mask,
                key=key,
            )
            s = embedding.s_init + self.model.linear_no_bias_s(self.model.layernorm_s(s))
            s, z = self.model.pairformer_stack(
                s, z, pair_mask=pair_mask, key=jax.random.fold_in(key, 1)
            )
            return (TrunkEmbedding(s=s, z=z), jax.random.fold_in(key, 1))

        if self.num_recycle > 0:
            state, key = jax.lax.fori_loop(0, self.num_recycle, body, (state, key))
        if self.stop_recycle_gradient:
            state = jax.tree.map(jax.lax.stop_gradient, state)
        return body(0, (state, key))[0]

    def structure_module(self, key, state, features, embedding,
                         num_samples=1, sampling_steps=25, parallel=False,
                         token_pad_mask=None, atom_pad_mask=None):
        """Run diffusion and confidence with FMA-safe noise variance clamping."""
        noise_schedule = self.model.inference_noise_scheduler(
            N_step=sampling_steps
        )

        def single_sample(carry, subkey):
            sample: jax.Array = sample_diffusion(
                denoise_net=self.model.diffusion_module,
                input_feature_dict=features,
                s_inputs=embedding.s_inputs,
                s_trunk=state.s,
                z_trunk=state.z,
                N_sample=1,
                noise_schedule=noise_schedule,
                gamma0=self.model.gamma0,
                gamma_min=self.model.gamma_min,
                noise_scale_lambda=self.model.noise_scale_lambda,
                step_scale_eta=self.model.step_scale_eta,
                key=subkey,
            )

            confidence: ConfidenceMetrics = self.model.confidence_metrics(
                initial_embedding=embedding,
                trunk_embedding=state,
                input_feature_dict=features,
                coordinates=sample,
                key=subkey,
            )
            return carry, (sample, confidence)

        if num_samples > 1:
            if parallel:
                vmap_sample = jax.vmap(single_sample, in_axes=(None, 0), out_axes=(None, (0, 0)))
                _, (samples, confidence) = vmap_sample(None, jax.random.split(key, num_samples))
            else:
                _, (samples, confidence) = jax.lax.scan(single_sample, None, jax.random.split(key, num_samples))
        else:
            _, (samples, confidence) = single_sample(None, key)
        return samples, confidence
