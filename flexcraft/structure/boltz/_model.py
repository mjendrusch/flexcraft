# adapted from mosaic ((c) 2025 escalante under MIT license)
import gc
from dataclasses import asdict
from pathlib import Path
from typing import Any

import boltz.main as boltz_main
import equinox as eqx
import jax
import joltz
import numpy as np
from boltz.model.models.boltz2 import Boltz2
from jax import numpy as jnp
from joltz import TrunkState

import flexcraft.sequence.aa_codes as aas
from flexcraft.data.data import DesignData
from flexcraft.structure.boltz._data import Joltz2Writer, JoltzInput, JoltzSpec
from flexcraft.structure.boltz._result import JoltzPrediction, JoltzResult
from flexcraft.structure.common._data import InputSpec
from flexcraft.structure.common._utils import atomX_to_atom_array
from flexcraft.utils.rng import Keygen

_MSA_ROW_KEYS = (
    "msa",
    "msa_mask",
    "msa_paired",
    "has_deletion",
    "deletion_value",
)


class _MaskedMSAModule(eqx.Module):
    """Ensure valid MSA rows precede padded rows before `[:1024]` subsampling."""

    msa_module: Any
    num_subsampled_msa: int = eqx.field(static=True, default=1024)

    def __getattr__(self, name: str):
        return getattr(self.msa_module, name)

    def _prioritize_feats(self, feats: dict, key: Any) -> dict:
        if (
            "msa" not in feats
            or "msa_mask" not in feats
            or feats["msa"] is None
            or feats["msa_mask"] is None
        ):
            return feats
        msa_mask = feats["msa_mask"]
        num_rows = msa_mask.shape[1]
        valid_row = (msa_mask[0] > 0).any(axis=-1)
        row_indices = jnp.arange(num_rows, dtype=jnp.int32)
        if num_rows > self.num_subsampled_msa:
            if key is not None:
                rand_tie = jax.random.uniform(
                    jax.random.fold_in(key, 991), (num_rows,)
                )
            else:
                rand_tie = row_indices.astype(jnp.float32) / jnp.maximum(
                    1, num_rows
                )
            priority = jnp.where(
                valid_row,
                rand_tie,
                2.0
                + row_indices.astype(jnp.float32) / jnp.maximum(1, num_rows),
            )
            priority = priority.at[0].set(-1.0)
            order = jnp.argsort(priority)[: self.num_subsampled_msa]
        else:
            order = jnp.argsort(
                jnp.where(valid_row, row_indices, num_rows + row_indices)
            )
        reordered = dict(feats)
        for name in _MSA_ROW_KEYS:
            if name in reordered and reordered[name] is not None:
                reordered[name] = reordered[name][:, order]
        return reordered

    def __call__(self, *args, **kwargs):
        key = kwargs.get("key")
        if key is None:
            for arg in args:
                if isinstance(arg, jax.Array) and (
                    jax.dtypes.issubdtype(arg.dtype, jax.dtypes.prng_key)
                    or (arg.dtype == jnp.uint32 and arg.shape == (2,))
                ):
                    key = arg
                    break
        if "feats" in kwargs and isinstance(kwargs["feats"], dict):
            new_kwargs = dict(kwargs)
            new_kwargs["feats"] = self._prioritize_feats(kwargs["feats"], key)
            return self.msa_module(*args, **new_kwargs)
        new_args = list(args)
        for idx, arg in enumerate(new_args):
            if isinstance(arg, dict) and "msa" in arg:
                new_args[idx] = self._prioritize_feats(arg, key)
                break
        return self.msa_module(*new_args, **kwargs)


def _wrap_joltz_msa_module(model: Any) -> Any:
    if not hasattr(model, "msa_module") or model.msa_module is None:
        return model
    if isinstance(model.msa_module, _MaskedMSAModule):
        return model
    max_rows = int(getattr(model.msa_module, "num_subsampled_msa", 1024))
    wrapped = _MaskedMSAModule(
        msa_module=model.msa_module,
        num_subsampled_msa=max_rows,
    )
    return eqx.tree_at(lambda m: m.msa_module, model, wrapped)


def load_boltz2(model="boltz2_conf.ckpt", cache=Path("./params/boltz/")):
    if isinstance(cache, str):
        cache = Path(cache)
    if not cache.exists():
        print(f"Downloading Boltz checkpoint to {cache}")
        cache.mkdir(parents=True, exist_ok=True)
        boltz_main.download_boltz2(cache)

    torch_model = Boltz2.load_from_checkpoint(
        cache / model,
        strict=True,
        map_location="cpu",
        predict_args={
            "recycling_steps": 0,
            "sampling_steps": 25,
            "diffusion_samples": 1,
        },
        diffusion_process_args=asdict(boltz_main.Boltz2DiffusionParams()),
        msa_args=asdict(
            boltz_main.MSAModuleArgs(
                subsample_msa=True,
                num_subsampled_msa=1024,
                use_paired_feature=True,
            )
        ),
        pairformer_args=asdict(boltz_main.PairformerArgsV2()),
    ).eval()

    jax_model = joltz.from_torch(torch_model)
    del torch_model
    gc.collect()
    jax_model = _wrap_joltz_msa_module(jax_model)
    model_params, model_static = eqx.partition(jax_model, eqx.is_inexact_array)
    return eqx.combine(jax.device_put(model_params), model_static)


def _spec_to_joltz_spec(spec: InputSpec) -> JoltzSpec:
    if isinstance(spec, JoltzSpec):
        return spec
    joltz_spec = JoltzSpec(
        *spec.chains,
        templates=spec.templates,
        constraints=spec.constraints,
    )
    joltz_spec.temporaries = list(spec.temporaries)
    joltz_spec.precomputed_msas = list(spec.precomputed_msas)
    joltz_spec._pad_spec = getattr(spec, "_pad_spec", None)
    return joltz_spec


class Joltz2:
    def __init__(self, model="boltz2_conf.ckpt", cache="./params/boltz/", **kwargs):
        self.cache = cache
        self.model = load_boltz2(model=model, cache=cache)
        self.model_jit = eqx.filter_jit(self.model)

    def evaluator(
        self,
        num_recycle=4,
        num_sampling_steps=25,
        deterministic=False,
        stop_recycle_gradient=True,
    ):
        evaluator = Joltz2Evaluator(
            joltz=self.model,
            num_recycle=num_recycle,
            num_sampling_steps=num_sampling_steps,
            deterministic=deterministic,
            stop_recycle_gradient=stop_recycle_gradient,
        )
        evaluator_params, evaluator_static = eqx.partition(
            evaluator, eqx.is_array
        )

        def _evaluator(params):
            return eqx.combine(params, evaluator_static)

        return _evaluator, evaluator_params

    def predictor(
        self,
        num_recycle=2,
        num_samples=1,
        num_sampling_steps=25,
        deterministic=False,
    ):
        jit_predict = eqx.filter_jit(
            Joltz2Evaluator(
                joltz=self.model,
                num_recycle=num_recycle,
                num_sampling_steps=num_sampling_steps,
                deterministic=deterministic,
            )._predict
        )

        def _predict(
            key,
            joltz_spec: JoltzSpec | InputSpec | JoltzInput,
            writer: Joltz2Writer | None = None,
        ):
            if isinstance(joltz_spec, JoltzInput):
                joltz_input = joltz_spec
                joltz_writer = writer
            else:
                spec = _spec_to_joltz_spec(joltz_spec)
                joltz_input, spec_writer = spec.to_input(cache=self.cache)
                joltz_writer = writer if writer is not None else spec_writer
            prediction = jit_predict(
                key, joltz_input.features, num_samples=num_samples
            )
            prediction = prediction.unpad()
            return JoltzPrediction(data=prediction.data, writer=joltz_writer)

        return _predict


class Joltz2Evaluator(eqx.Module):
    joltz: Any
    num_recycle: int = 3
    num_sampling_steps: int = 25
    deterministic: bool = False
    stop_recycle_gradient: bool = True

    def initial_embedding(self, features):
        return self.joltz.embed_inputs(features)

    def trunk(self, base_key, features, embedding):
        joltz_model = _wrap_joltz_msa_module(self.joltz)
        state = TrunkState(
            s=jnp.zeros_like(embedding.s_init),
            z=jnp.zeros_like(embedding.z_init),
        )
        key = base_key

        if self.num_recycle > 0:

            def body_fn(carry, _):
                trunk_state, step_key = carry
                trunk_state, step_key = joltz_model.trunk_iteration(
                    trunk_state,
                    embedding,
                    features,
                    key=step_key,
                    deterministic=self.deterministic,
                )
                if self.stop_recycle_gradient:
                    trunk_state = jax.tree.map(
                        jax.lax.stop_gradient, trunk_state
                    )
                return (trunk_state, step_key), None

            scan_fn = jax.remat(body_fn) if self.stop_recycle_gradient else body_fn
            (state, key), _ = jax.lax.scan(
                scan_fn,
                (state, key),
                None,
                length=self.num_recycle,
            )

        final_state, _ = joltz_model.trunk_iteration(
            state,
            embedding,
            features,
            key=key,
            deterministic=self.deterministic,
        )
        return final_state

    def structure_module(
        self,
        key,
        features,
        embedding,
        trunk_state,
        positions=None,
        start_steps=10,
        edit_fn=None,
        edit_settings=None,
    ):
        q, c, to_keys, atom_enc_bias, atom_dec_bias, token_trans_bias = (
            self.joltz.diffusion_conditioning(
                trunk_state.s,
                trunk_state.z,
                embedding.relative_position_encoding,
                features,
            )
        )
        with jax.default_matmul_precision("float32"):
            return jax.lax.stop_gradient(
                self._partial_sample(
                    atom_coords=positions,
                    s_trunk=trunk_state.s,
                    s_inputs=embedding.s_inputs,
                    feats=features,
                    num_sampling_steps=self.num_sampling_steps,
                    start_steps=start_steps,
                    edit_fn=edit_fn,
                    edit_settings=edit_settings,
                    atom_mask=features["atom_pad_mask"],
                    multiplicity=1,
                    diffusion_conditioning={
                        "q": q,
                        "c": c,
                        "to_keys": to_keys,
                        "atom_enc_bias": atom_enc_bias,
                        "atom_dec_bias": atom_dec_bias,
                        "token_trans_bias": token_trans_bias,
                    },
                    key=jax.random.fold_in(key, 2),
                )
            )

    def _partial_sample(
        self,
        atom_coords,
        atom_mask,
        num_sampling_steps,
        *,
        key,
        start_steps=0,
        edit_fn=None,
        edit_settings=None,
        feats=None,
        **network_condition_kwargs,
    ):
        sm = self.joltz.structure_module
        shape = (*atom_mask.shape, 3)
        mask = (atom_mask > 0).astype(jnp.float32)[..., None]
        mask_count = jnp.maximum(mask.sum(axis=-2, keepdims=True), 1.0)

        def _center_and_mask(coords):
            coords = coords * mask
            mean = coords.sum(axis=-2, keepdims=True) / mask_count
            return (coords - mean) * mask

        sigmas = sm.sample_schedule(num_sampling_steps)
        gammas = jnp.where(sigmas > sm.gamma_min, sm.gamma_0, 0.0)
        step_scale = sm.step_scale

        @jax.checkpoint
        def sample_body_function(carry, step_inputs):
            sigma_tm, sigma_t, gamma = step_inputs
            coords, step_key = carry
            random_r, random_tr = joltz.compute_random_augmentation(
                key=step_key
            )
            step_key = jax.random.fold_in(step_key, 1)
            coords = _center_and_mask(coords)
            coords = (
                jnp.einsum("bmd,bds->bms", coords, random_r) + random_tr
            ) * mask

            t_hat = sigma_tm * (1 + gamma)
            noise_var = sm.noise_scale**2 * (t_hat**2 - sigma_tm**2)
            eps = (
                jnp.sqrt(jnp.maximum(noise_var, 0.0))
                * jax.random.normal(shape=shape, key=step_key)
                * mask
            )
            step_key = jax.random.fold_in(step_key, 1)
            atom_coords_noisy = (coords + eps) * mask
            atom_coords_denoised = sm.preconditioned_network_forward(
                atom_coords_noisy,
                t_hat,
                network_condition_kwargs=dict(
                    feats=feats,
                    **network_condition_kwargs,
                ),
                key=step_key,
            )
            atom_coords_denoised = atom_coords_denoised * mask
            if edit_fn is not None:
                atom_coords_denoised = (
                    edit_fn(
                        step_key,
                        JoltzInput(features=feats),
                        atom_coords_denoised,
                        edit_settings=edit_settings,
                    )
                    * mask
                )

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

            denoised_over_sigma = (
                atom_coords_noisy - atom_coords_denoised
            ) / t_hat
            atom_coords_next = (
                atom_coords_noisy
                + step_scale * (sigma_t - t_hat) * denoised_over_sigma
            ) * mask

            return (atom_coords_next, jax.random.fold_in(step_key, 0)), None

        start_noise = jax.random.normal(shape=shape, key=key) * mask
        start_coords = sigmas[0] * start_noise
        if atom_coords is not None:
            start_coords = atom_coords + sigmas[start_steps + 1] * start_noise
        start_coords = _center_and_mask(start_coords)

        (final_coords, _), _ = jax.lax.scan(
            sample_body_function,
            (start_coords, jax.random.fold_in(key, 1)),
            (
                sigmas[:-1][start_steps:],
                sigmas[1:][start_steps:],
                gammas[1:][start_steps:],
            ),
        )
        return final_coords * mask

    def confidence_module(
        self, key, features, embedding, trunk_state, distogram, positions
    ):
        return self.joltz.confidence_module(
            s_inputs=embedding.s_inputs,
            s=trunk_state.s,
            z=trunk_state.z,
            x_pred=positions,
            feats=features,
            pred_distogram_logits=distogram[None],
            key=jax.random.fold_in(key, 5),
            deterministic=self.deterministic,
        )

    def _prepare(self, key, features):
        embedding = self.initial_embedding(features)
        trunk_state = self.trunk(key, features, embedding)
        distogram = self.joltz.distogram_module(trunk_state.z)[0, :, :, 0, :]
        return features, embedding, trunk_state, distogram

    def prepare(self, key, joltz_input: JoltzInput):
        return self._prepare(key, joltz_input.features)

    def _predict(
        self, key, features, positions=None, start_steps=0, num_samples=1
    ):
        key, subkey = jax.random.split(key)
        features, embedding, trunk_state, distogram = self._prepare(
            subkey, features
        )

        def body(state, sample_key):
            sample = self.structure_module(
                sample_key,
                features,
                embedding,
                trunk_state,
                positions=positions,
                start_steps=start_steps,
            )
            confidence = self.confidence_module(
                sample_key,
                features,
                embedding,
                trunk_state,
                distogram,
                sample,
            )
            return state, (sample, confidence)

        if num_samples > 1:
            _, (samples, confidence) = jax.lax.scan(
                body, None, jax.random.split(key, num_samples)
            )
        else:
            _, (samples, confidence) = body(None, key)
        return JoltzResult(
            dict(
                features=features,
                embedding=embedding,
                state=trunk_state,
                distogram=distogram,
                samples=samples,
                confidence=confidence,
            )
        )

    def predict(self, key, joltz_input: JoltzInput, num_samples=1):
        return self._predict(key, joltz_input.features, num_samples=num_samples)

    def __call__(self, key, joltz_input: JoltzInput, num_samples=1):
        return self.predict(key, joltz_input, num_samples=num_samples)

    def score(
        self,
        key,
        joltz_input: JoltzInput,
        positions: jax.Array = None,
        data: DesignData = None,
    ):
        k1, k2 = jax.random.split(key, 2)
        if data is not None:
            sequence = data.aa
            if jnp.issubdtype(sequence.dtype, jnp.integer):
                sequence = jax.nn.one_hot(sequence, 20)
            joltz_input = joltz_input.set_aa(sequence)
            positions = data["atom_positions"]
        if positions is None:
            raise ValueError(
                "Either `positions` or `data` must be provided to score()."
            )
        features, embedding, trunk_state, distogram = self.prepare(
            k1, joltz_input
        )
        atom_to_token = features["atom_to_token"]
        if len(positions.shape) == 3 and not (
            positions.shape[0] == 1
            and positions.shape[1] == atom_to_token.shape[1]
        ):
            positions = atomX_to_atom_array(positions, atom_to_token[0])
            positions = positions[None]
        elif len(positions.shape) == 4:
            positions = atomX_to_atom_array(
                jnp.moveaxis(positions, 0, -1), atom_to_token[0]
            )
            positions = jnp.moveaxis(positions, -1, 0)
            positions = positions[:, None]
        if positions.shape[-2] < atom_to_token.shape[1]:
            raise ValueError("Positions need to cover all atoms.")
        confidence = self.confidence_module(
            k2, features, embedding, trunk_state, distogram, positions
        )
        return JoltzResult(
            dict(
                features=features,
                embedding=embedding,
                state=trunk_state,
                distogram=distogram,
                samples=positions,
                confidence=confidence,
            )
        )
