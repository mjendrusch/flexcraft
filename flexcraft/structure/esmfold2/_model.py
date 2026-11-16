import gc
from pathlib import Path

import jax

import equinox as eqx
import esmjfold2
from esmjfold2 import ESMFold2 as _ESMFold2
from esmjfold2 import compute_lm_hidden_states, load_model, save_model

from flexcraft.structure.esmfold2._data import (
    ESMFold2Input, ESMFold2Spec, to_model_features)
from flexcraft.structure.esmfold2._result import (
    ESMFold2Confidence, ESMFold2Prediction, ESMFold2Result)

CHECKPOINTS = {
    "esmfold2_fast": "biohub/ESMFold2-Fast",
    "esmfold2_full": "biohub/ESMFold2",
}

class ESMFold2:
    def __init__(self, model="esmfold2_fast", cache="./params/esmfold2/",
                 load_esmc=True, esmc_device="cpu"):
        """ESMFold2 structure predictor.

        Args:
            model (str): checkpoint name (esmfold2_fast, esmfold2_full) or a
                HuggingFace repo id, which has to contain a "/".
            cache (str): directory for converted JAX weights.
            load_esmc (bool): load ESMC-6B for language model conditioning.
                Without it, predictions run on the sequence one-hot alone.
            esmc_device (str): device holding the ESMC weights, as "cpu", "gpu"
                or "gpu:1". ESMC-6B is about 24 GB in fp32 and stays resident
                for the whole run, so it only fits beside the trunk on a large
                card. Put it on "gpu:1" to give it a card of its own.
        """
        self.repo = _resolve_checkpoint(model)
        self.cache = cache
        self.esmc_device = _resolve_device(esmc_device)
        self.model, self.esmc = load_esmfold2(
            self.repo, cache=cache, load_esmc=load_esmc,
            esmc_device=self.esmc_device)

    def lm_hidden_states(self, features):
        """Run ESMC over the tokenized input, or return None without ESMC."""
        if self.esmc is None:
            return None
        # run on esmc_device, then hand the features back to the trunk's device
        with jax.default_device(self.esmc_device):
            hidden_states = compute_lm_hidden_states(
                self.esmc,
                features["input_ids"],
                features["asym_id"],
                features["residue_index"],
                features["mol_type"],
                features["token_attention_mask"])
        return jax.device_put(hidden_states, jax.devices()[0])

    def predictor(self, num_recycle=None, num_samples=1, num_sampling_steps=14,
                  deterministic=False, msa_max_depth=1024):
        jit_predict = eqx.filter_jit(ESMFold2Evaluator(
            model=self.model,
            num_loops=num_recycle,
            num_sampling_steps=num_sampling_steps,
            noise_scale=0.0 if deterministic else None,
            msa_max_depth=msa_max_depth)._predict)
        def _predict(key, esmfold_spec: ESMFold2Spec):
            input, writer = esmfold_spec.to_input()
            lm_hidden_states = self.lm_hidden_states(input.features)
            prediction = jit_predict(key, input.features,
                                     lm_hidden_states=lm_hidden_states,
                                     num_samples=num_samples)
            return ESMFold2Prediction(data=prediction.data, writer=writer)
        return _predict

    def input_predictor(self, num_recycle=None, num_samples=1, num_sampling_steps=14,
                        deterministic=False, msa_max_depth=1024):
        # like predictor, but runs on an ESMFold2Input directly (e.g. after
        # set_aa) instead of building it from an ESMFold2Spec.
        jit_predict = eqx.filter_jit(ESMFold2Evaluator(
            model=self.model,
            num_loops=num_recycle,
            num_sampling_steps=num_sampling_steps,
            noise_scale=0.0 if deterministic else None,
            msa_max_depth=msa_max_depth)._predict)
        def _predict(key, esmfold_input: ESMFold2Input, esmfold_writer=None,
                     lm_hidden_states=None):
            if lm_hidden_states is None:
                lm_hidden_states = self.lm_hidden_states(esmfold_input.features)
            prediction = jit_predict(key, esmfold_input.features,
                                     lm_hidden_states=lm_hidden_states,
                                     num_samples=num_samples)
            return ESMFold2Prediction(data=prediction.data, writer=esmfold_writer)
        return _predict

    def evaluator(self, num_recycle=None, num_sampling_steps=14,
                  deterministic=False, msa_max_depth=1024):
        evaluator = ESMFold2Evaluator(
            model=self.model,
            num_loops=num_recycle,
            num_sampling_steps=num_sampling_steps,
            noise_scale=0.0 if deterministic else None,
            msa_max_depth=msa_max_depth)
        evaluator_params, evaluator_static = eqx.partition(evaluator, eqx.is_array)
        def _evaluator(params):
            return eqx.combine(params, evaluator_static)
        return _evaluator, evaluator_params

class ESMFold2Evaluator(eqx.Module):
    model: _ESMFold2
    num_loops: int | None = None
    num_sampling_steps: int = 14
    msa_max_depth: int | None = 1024
    noise_scale: float | None = None
    step_scale: float | None = None
    sample_parallel: bool = False

    def _predict(self, key, features, lm_hidden_states=None, num_samples=1):
        keys = jax.random.split(key, 3)
        # run embedding
        context = self.embedding(keys[0], features)
        lm_pair = self.language_model(lm_hidden_states)
        # run trunk
        z: jax.Array = self.trunk(keys[1], context, lm_pair)
        # run distogram
        log_distogram: jax.Array = self.model._compute_distogram(z)[0]
        # run diffusion
        (coords, confidence) = self.structure_module(
            keys[2], context, z,
            num_samples=num_samples,
            parallel=self.sample_parallel)
        return ESMFold2Result(dict(
            features=features, state=z,
            distogram=log_distogram, samples=coords, confidence=confidence
        ))

    def predict(self, key, esmfold_input: ESMFold2Input,
                lm_hidden_states=None, num_samples=1):
        features = esmfold_input.features
        return self._predict(
            key, features, lm_hidden_states=lm_hidden_states,
            num_samples=num_samples)

    def __call__(self, key, esmfold_input: ESMFold2Input,
                 lm_hidden_states=None, num_samples=1):
        return self.predict(key, esmfold_input,
                            lm_hidden_states=lm_hidden_states,
                            num_samples=num_samples)

    def embedding(self, key, features):
        return self.model._prepare_embeddings(to_model_features(features))

    def language_model(self, lm_hidden_states):
        if lm_hidden_states is None:
            return None
        return self.model.language_model(lm_hidden_states)

    def trunk(self, key, context, lm_pair) -> jax.Array:
        # None defers to the checkpoint's own refinement loop count
        num_loops = self.model.num_loops if self.num_loops is None else self.num_loops
        return self.model._run_trunk(
            context, lm_pair, key,
            num_loops=num_loops,
            msa_max_depth=self.msa_max_depth)

    def structure_module(self, key, context, z, num_samples=1, parallel=False):
        def single_sample(carry, key):
            sample: jax.Array = self.model._sample_structure(
                context, z, key,
                num_sampling_steps=self.num_sampling_steps,
                noise_scale=self.noise_scale,
                step_scale=self.step_scale)

            confidence = ESMFold2Confidence.from_head(
                self.model._compute_confidence(context, z, sample))
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

def load_esmfold2(repo, cache="./params/esmfold2/", load_esmc=True, esmc_device=None):
    """Load converted JAX weights, converting from the torch checkpoint once.

    ESMC is built inside a default_device context rather than moved afterwards:
    at 24 GB it does not fit on the trunk's card alongside the trunk, so it must
    never be materialized there in the first place.
    """
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    model_path = cache / repo.split("/")[-1]
    esmc_path = cache / "ESMC-6B"
    if esmc_device is None:
        esmc_device = jax.devices()[0]
    model = _load_cached(model_path)
    esmc = None
    if load_esmc:
        with jax.default_device(esmc_device):
            esmc = _load_cached(esmc_path)
    if model is not None and (esmc is not None or not load_esmc):
        return model, esmc

    import torch
    from esm.models.esmfold2 import EsmFold2Model
    torch_model = EsmFold2Model.from_pretrained(
        repo, load_esmc=load_esmc, device="cpu",
        dtype=torch.float32, esmc_precision="fp32").eval()
    if model is None:
        model = esmjfold2.from_torch(torch_model)
        save_model(model, str(model_path))
    if load_esmc and esmc is None:
        with jax.default_device(esmc_device):
            esmc = esmjfold2.from_torch(torch_model.esmc)
        save_model(esmc, str(esmc_path))
    del torch_model
    gc.collect()
    return model, esmc

def _resolve_checkpoint(model: str) -> str:
    """Map a checkpoint name to its HuggingFace repo, or pass a repo id through."""
    if "/" in model:
        return model
    if model not in CHECKPOINTS:
        raise ValueError(
            f"Unknown ESMFold2 checkpoint '{model}'. Use one of "
            f"{sorted(CHECKPOINTS)}, or a HuggingFace repo id like "
            f"'biohub/ESMFold2-Fast'.")
    return CHECKPOINTS[model]

def _resolve_device(spec: str):
    """Parse a device spec of the form "cpu", "gpu" or "gpu:1"."""
    kind, _, index = spec.partition(":")
    return jax.devices(kind)[int(index) if index else 0]

def _load_cached(path):
    # save_model writes the .eqx first, so a run that died mid-save leaves a
    # partial cache behind; treat it as absent and convert again
    if not all(Path(f"{path}.{suffix}").exists()
               for suffix in ("eqx", "skeleton.pkl")):
        return None
    return load_model(str(path))
