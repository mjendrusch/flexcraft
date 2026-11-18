import gc
from dataclasses import fields, is_dataclass, replace
from pathlib import Path
from typing import Any

import equinox as eqx
import esmjfold2
import jax
import jax.numpy as jnp
import numpy as np
from esmjfold2 import ESMFold2 as _ESMFold2
from esmjfold2 import compute_lm_hidden_states, load_model, save_model

from flexcraft.structure.common._data import InputSpec
from flexcraft.structure.common._utils import align_up
from flexcraft.structure.esmfold2._data import (
    ESM_PAD_ID,
    ESMFold2Input,
    ESMFold2Spec,
    ESMFold2Writer,
    to_model_features,
)
from flexcraft.structure.esmfold2._result import (
    ESMFold2Confidence,
    ESMFold2Prediction,
    ESMFold2Result,
)

CHECKPOINTS = {
    "esmfold2_fast": "biohub/ESMFold2-Fast",
    "esmfold2_full": "biohub/ESMFold2",
}


def _spec_to_esmfold2_spec(spec: InputSpec) -> ESMFold2Spec:
    if isinstance(spec, ESMFold2Spec):
        return spec
    esm_spec = ESMFold2Spec(
        *spec.chains,
        templates=spec.templates,
        constraints=spec.constraints,
    )
    esm_spec.temporaries = list(spec.temporaries)
    esm_spec.precomputed_msas = list(spec.precomputed_msas)
    esm_spec._pad_spec = getattr(spec, "_pad_spec", None)
    return esm_spec


def _extract_tok_and_lm_indices(
    entry: Any, prot_tok_indices: np.ndarray, n_tokens: int
) -> tuple[np.ndarray, np.ndarray]:
    """Resolve `(tok_idx, lm_idx)` from a single batch item's `expand_maps` entry."""
    if entry is None:
        empty = np.empty((0,), dtype=np.intp)
        return empty, empty
    if isinstance(entry, dict):
        tok_idx = np.fromiter(entry.keys(), dtype=np.intp)
        lm_idx = np.fromiter(entry.values(), dtype=np.intp)
        return tok_idx, lm_idx
    if isinstance(entry, tuple) and len(entry) == 2:
        first = np.asarray(entry[0], dtype=np.intp)
        second = np.asarray(entry[1], dtype=np.intp)
        if np.array_equal(first, prot_tok_indices):
            return first, second
        if np.array_equal(second, prot_tok_indices):
            return second, first
        return first, second
    arr = np.asarray(entry, dtype=np.intp)
    if arr.size == 0:
        empty = np.empty((0,), dtype=np.intp)
        return empty, empty
    if arr.ndim == 2 and arr.shape[1] == 2:
        if np.array_equal(arr[:, 0], prot_tok_indices):
            return arr[:, 0], arr[:, 1]
        if np.array_equal(arr[:, 1], prot_tok_indices):
            return arr[:, 1], arr[:, 0]
        return arr[:, 0], arr[:, 1]
    if arr.ndim == 1:
        if arr.shape[0] == len(prot_tok_indices) and (
            len(prot_tok_indices) != n_tokens or np.all(arr >= 1)
        ):
            return prot_tok_indices, arr
        if arr.shape[0] == n_tokens:
            if np.any(arr < 0):
                valid = arr >= 0
                return np.nonzero(valid)[0], arr[valid]
            return prot_tok_indices, arr[prot_tok_indices]
    raise TypeError(f"Unsupported expand_maps entry format: {type(entry)!r}")


def _scatter_lm_hidden_states(
    raw_hidden: jax.Array,
    expand_maps: Any,
    features: dict,
    n_tokens: int,
    s_padded: int,
) -> jax.Array:
    """Scatter ESMC outputs `(B, S_padded, ...)` onto complex token positions `(B, L, ...)`."""
    batch_size = int(features["input_ids"].shape[0])
    if (
        raw_hidden.ndim == 4
        and raw_hidden.shape[0] != batch_size
        and raw_hidden.shape[1] == batch_size
        and raw_hidden.shape[2] == s_padded
    ):
        raw_hidden = jnp.moveaxis(raw_hidden, 0, 2)
    hidden = jnp.zeros(
        (batch_size, n_tokens, *raw_hidden.shape[2:]), dtype=raw_hidden.dtype
    )
    mol_type_np = np.asarray(features["mol_type"])
    tok_mask_np = np.asarray(features["token_attention_mask"], dtype=bool)
    for b in range(batch_size):
        prot_tok_indices = np.nonzero((mol_type_np[b] == 0) & tok_mask_np[b])[0]
        tok_idx, lm_idx = _extract_tok_and_lm_indices(
            expand_maps[b], prot_tok_indices, n_tokens
        )
        if len(tok_idx) > 0:
            hidden = hidden.at[b, tok_idx].set(raw_hidden[b, lm_idx])
    return hidden


def _prioritize_context_msa_rows(
    context: Any, key: jax.Array, msa_max_depth: int | None
) -> tuple[Any, jax.Array]:
    """Compact valid MSA rows ahead of padded rows before `_run_trunk` subsampling."""
    if msa_max_depth is None or not is_dataclass(context):
        return context, key
    mask_field = None
    mask_val = None
    for f in fields(context):
        if f.name in ("msa_attention_mask", "msa_mask"):
            val = getattr(context, f.name)
            if val is not None and hasattr(val, "shape") and val.ndim == 3:
                mask_field = f.name
                mask_val = val
                break
    if mask_field is None or mask_val.shape[1] <= msa_max_depth:
        return context, key

    num_rows = int(mask_val.shape[1])
    key, subkey = jax.random.split(key)
    rest_count = num_rows - 1
    valid_rest = (mask_val[0, 1:] > 0).any(axis=-1)
    rand_tie = jax.random.uniform(subkey, (rest_count,))
    rest_indices = jnp.arange(rest_count, dtype=jnp.float32)
    priority = jnp.where(
        valid_rest,
        rand_tie,
        2.0 + rest_indices / jnp.maximum(1.0, float(rest_count)),
    )
    chosen_rest = jnp.argsort(priority)[: msa_max_depth - 1] + 1
    chosen_idx = jnp.concatenate(
        [jnp.zeros((1,), dtype=chosen_rest.dtype), chosen_rest], axis=0
    )

    updates = {}
    for f in fields(context):
        val = getattr(context, f.name)
        if val is None:
            continue
        if f.name in ("n_msa", "num_msa") and isinstance(val, int):
            updates[f.name] = int(msa_max_depth)
        elif (
            hasattr(val, "shape")
            and val.ndim >= 3
            and val.shape[1] == num_rows
            and (
                f.name.startswith("msa")
                or f.name in ("has_deletion", "deletion_value")
            )
        ):
            updates[f.name] = val[:, chosen_idx]
    return (replace(context, **updates) if updates else context), key


class ESMFold2:
    def __init__(
        self,
        model="esmfold2_fast",
        cache="./params/esmfold2/",
        load_esmc=True,
        esmc_device="cpu",
        **kwargs,
    ):
        """ESMFold2 structure predictor."""
        self.repo = _resolve_checkpoint(model)
        self.cache = cache
        self.esmc_device = _resolve_device(esmc_device)
        self.model, self.esmc = load_esmfold2(
            self.repo,
            cache=cache,
            load_esmc=load_esmc,
            esmc_device=self.esmc_device,
        )

    def lm_hidden_states(self, features):
        """Run ESMC over bucket-padded tokenized input, or return None without ESMC."""
        if self.esmc is None:
            return None
        from esmjfold2 import lm_features as _lm_features

        prepared = _lm_features._prepare_lm_inputs(
            features["input_ids"],
            features["asym_id"],
            features["residue_index"],
            features["mol_type"],
            features["token_attention_mask"],
        )
        if prepared is None:
            with jax.default_device(self.esmc_device):
                hidden_states = compute_lm_hidden_states(
                    self.esmc,
                    features["input_ids"],
                    features["asym_id"],
                    features["residue_index"],
                    features["mol_type"],
                    features["token_attention_mask"],
                )
            return jax.device_put(hidden_states, jax.devices()[0])

        lm_input_ids, sequence_id, expand_maps = prepared
        if lm_input_ids is None or lm_input_ids.shape[1] == 0:
            with jax.default_device(self.esmc_device):
                hidden_states = compute_lm_hidden_states(
                    self.esmc,
                    features["input_ids"],
                    features["asym_id"],
                    features["residue_index"],
                    features["mol_type"],
                    features["token_attention_mask"],
                )
            return jax.device_put(hidden_states, jax.devices()[0])

        n_tokens = int(features["res_type"].shape[1])
        s_valid = int(lm_input_ids.shape[1])
        s_padded = align_up(max(s_valid, n_tokens + 16), 32)
        pad_len = s_padded - s_valid
        lm_ids_np = np.asarray(lm_input_ids)
        seq_id_np = np.asarray(sequence_id)
        if pad_len > 0:
            lm_ids_np = np.pad(
                lm_ids_np, ((0, 0), (0, pad_len)), constant_values=ESM_PAD_ID
            )
            seq_id_np = np.pad(
                seq_id_np, ((0, 0), (0, pad_len)), constant_values=-1
            )

        with jax.default_device(self.esmc_device):
            raw_hidden = _lm_features._run_esmc(
                self.esmc,
                jnp.asarray(lm_ids_np, dtype=jnp.int32),
                jnp.asarray(seq_id_np, dtype=jnp.int32),
            )
            hidden_states = _scatter_lm_hidden_states(
                raw_hidden, expand_maps, features, n_tokens, s_padded
            )
        return jax.device_put(hidden_states, jax.devices()[0])

    def predictor(
        self,
        num_recycle=None,
        num_samples=1,
        num_sampling_steps=14,
        deterministic=False,
        msa_max_depth=1024,
    ):
        jit_predict = eqx.filter_jit(
            ESMFold2Evaluator(
                model=self.model,
                num_loops=num_recycle,
                num_sampling_steps=num_sampling_steps,
                noise_scale=0.0 if deterministic else None,
                msa_max_depth=msa_max_depth,
            )._predict
        )

        def _predict(
            key,
            spec_or_input: ESMFold2Spec | InputSpec | ESMFold2Input,
            writer: ESMFold2Writer | None = None,
            lm_hidden_states=None,
            *,
            esmfold_writer: ESMFold2Writer | None = None,
        ):
            if writer is None:
                writer = esmfold_writer
            if isinstance(spec_or_input, ESMFold2Input):
                esm_input = spec_or_input
            else:
                spec = _spec_to_esmfold2_spec(spec_or_input)
                esm_input, spec_writer = spec.to_input(cache=self.cache)
                if writer is None:
                    writer = spec_writer
            if lm_hidden_states is None:
                lm_hidden_states = self.lm_hidden_states(esm_input.features)
            prediction = jit_predict(
                key,
                esm_input.features,
                lm_hidden_states=lm_hidden_states,
                num_samples=num_samples,
            )
            prediction = prediction.unpad()
            return ESMFold2Prediction(data=prediction.data, writer=writer)

        return _predict

    def input_predictor(
        self,
        num_recycle=None,
        num_samples=1,
        num_sampling_steps=14,
        deterministic=False,
        msa_max_depth=1024,
    ):
        return self.predictor(
            num_recycle=num_recycle,
            num_samples=num_samples,
            num_sampling_steps=num_sampling_steps,
            deterministic=deterministic,
            msa_max_depth=msa_max_depth,
        )

    def evaluator(
        self,
        num_recycle=None,
        num_sampling_steps=14,
        deterministic=False,
        msa_max_depth=1024,
    ):
        evaluator = ESMFold2Evaluator(
            model=self.model,
            num_loops=num_recycle,
            num_sampling_steps=num_sampling_steps,
            noise_scale=0.0 if deterministic else None,
            msa_max_depth=msa_max_depth,
        )
        evaluator_params, evaluator_static = eqx.partition(
            evaluator, eqx.is_array
        )

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
        context = self.embedding(keys[0], features)
        lm_pair = self.language_model(lm_hidden_states)
        z: jax.Array = self.trunk(keys[1], context, lm_pair)
        log_distogram: jax.Array = self.model._compute_distogram(z)[0]
        coords, confidence = self.structure_module(
            keys[2],
            context,
            z,
            num_samples=num_samples,
            parallel=self.sample_parallel,
        )
        return ESMFold2Result(
            dict(
                features=features,
                state=z,
                distogram=log_distogram,
                samples=coords,
                confidence=confidence,
            )
        )

    def predict(
        self, key, esmfold_input: ESMFold2Input, lm_hidden_states=None, num_samples=1
    ):
        return self._predict(
            key,
            esmfold_input.features,
            lm_hidden_states=lm_hidden_states,
            num_samples=num_samples,
        )

    def __call__(
        self, key, esmfold_input: ESMFold2Input, lm_hidden_states=None, num_samples=1
    ):
        return self.predict(
            key,
            esmfold_input,
            lm_hidden_states=lm_hidden_states,
            num_samples=num_samples,
        )

    def embedding(self, key, features):
        return self.model._prepare_embeddings(to_model_features(features))

    def language_model(self, lm_hidden_states):
        if lm_hidden_states is None:
            return None
        return self.model.language_model(lm_hidden_states)

    def trunk(self, key, context, lm_pair) -> jax.Array:
        num_loops = (
            self.model.num_loops if self.num_loops is None else self.num_loops
        )
        context, key = _prioritize_context_msa_rows(
            context, key, self.msa_max_depth
        )
        return self.model._run_trunk(
            context,
            lm_pair,
            key,
            num_loops=num_loops,
            msa_max_depth=self.msa_max_depth,
        )

    def structure_module(
        self, key, context, z, num_samples=1, parallel=False
    ):
        def single_sample(carry, sample_key):
            sample: jax.Array = self.model._sample_structure(
                context,
                z,
                sample_key,
                num_sampling_steps=self.num_sampling_steps,
                noise_scale=self.noise_scale,
                step_scale=self.step_scale,
            )
            confidence = ESMFold2Confidence.from_head(
                self.model._compute_confidence(context, z, sample)
            )
            return carry, (sample, confidence)

        if num_samples > 1:
            if parallel:
                vmap_sample = jax.vmap(
                    single_sample, in_axes=(None, 0), out_axes=(None, (0, 0))
                )
                _, (samples, confidence) = vmap_sample(
                    None, jax.random.split(key, num_samples)
                )
            else:
                _, (samples, confidence) = jax.lax.scan(
                    single_sample, None, jax.random.split(key, num_samples)
                )
        else:
            _, (samples, confidence) = single_sample(None, key)
        return samples, confidence


def _convert_esmc_low_memory(torch_esmc):
    """Convert PyTorch ESMC-6B block-by-block and leaf-by-leaf to cap peak RAM."""
    if not hasattr(torch_esmc, "embed") or not hasattr(
        torch_esmc, "transformer"
    ):
        return esmjfold2.from_torch(torch_esmc)

    from esmjfold2.esmc import ESMC, TransformerStack

    embed = esmjfold2.from_torch(torch_esmc.embed)
    torch_esmc.embed = None
    norm = esmjfold2.from_torch(torch_esmc.transformer.norm)
    torch_esmc.transformer.norm = None

    torch_blocks = torch_esmc.transformer.blocks
    n_blocks = len(torch_blocks)
    jax_blocks = []
    for idx in range(n_blocks):
        jax_blocks.append(esmjfold2.from_torch(torch_blocks[idx]))
        torch_blocks[idx] = None
        if (idx + 1) % 8 == 0:
            gc.collect()

    torch_esmc.transformer = None
    for mod_name in list(getattr(torch_esmc, "_modules", ())):
        torch_esmc._modules[mod_name] = None
    for param_name in list(getattr(torch_esmc, "_parameters", ())):
        torch_esmc._parameters[param_name] = None
    for buf_name in list(getattr(torch_esmc, "_buffers", ())):
        torch_esmc._buffers[buf_name] = None
    del torch_blocks, torch_esmc
    gc.collect()

    params0, static = eqx.partition(jax_blocks[0], eqx.is_inexact_array)
    leaves0, treedef = jax.tree.flatten(params0)
    num_leaves = len(leaves0)
    del params0, leaves0

    leaves_per_block = []
    for idx in range(n_blocks):
        block_leaves = jax.tree.leaves(
            eqx.filter(jax_blocks[idx], eqx.is_inexact_array)
        )
        leaves_per_block.append(block_leaves)
        jax_blocks[idx] = None
    del jax_blocks
    gc.collect()

    stacked_leaves = []
    for leaf_idx in range(num_leaves):
        col = [leaves_per_block[b][leaf_idx] for b in range(n_blocks)]
        for b in range(n_blocks):
            leaves_per_block[b][leaf_idx] = None
        stacked_leaves.append(jnp.stack(col, axis=0))
        del col
    del leaves_per_block
    gc.collect()

    stacked_params = jax.tree.unflatten(treedef, stacked_leaves)
    del stacked_leaves
    gc.collect()

    return ESMC(
        embed=embed,
        transformer=TransformerStack(
            block_params=stacked_params,
            block_static=static,
            norm=norm,
        ),
    )


def load_esmfold2(
    repo, cache="./params/esmfold2/", load_esmc=True, esmc_device=None
):
    """Load converted JAX weights, converting sequentially from PyTorch if needed."""
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

    need_model = model is None
    need_esmc = bool(load_esmc and esmc is None)
    if not need_model and not need_esmc:
        return model, esmc

    import torch
    from esm.models.esmfold2 import EsmFold2Model

    torch_model = EsmFold2Model.from_pretrained(
        repo,
        load_esmc=need_esmc,
        device="cpu",
        dtype=torch.float32,
        esmc_precision="fp32",
    ).eval()
    torch_esmc = torch_model.esmc if need_esmc else None
    if need_esmc:
        torch_model.esmc = None
    if need_model:
        model = esmjfold2.from_torch(torch_model)
        save_model(model, str(model_path))
    del torch_model
    gc.collect()

    if need_esmc:
        with jax.default_device(esmc_device):
            esmc = _convert_esmc_low_memory(torch_esmc)
        del torch_esmc
        gc.collect()
        save_model(esmc, str(esmc_path))
    return model, esmc


def _resolve_checkpoint(model: str) -> str:
    """Map a checkpoint name to its HuggingFace repo, or pass a repo id through."""
    if "/" in model:
        return model
    if model not in CHECKPOINTS:
        raise ValueError(
            f"Unknown ESMFold2 checkpoint '{model}'. Use one of "
            f"{sorted(CHECKPOINTS)}, or a HuggingFace repo id like "
            f"'biohub/ESMFold2-Fast'."
        )
    return CHECKPOINTS[model]


def _resolve_device(spec: str):
    """Parse a device spec of the form 'cpu', 'gpu', or 'gpu:1'."""
    kind, _, index = spec.partition(":")
    return jax.devices(kind)[int(index) if index else 0]


def _load_cached(path):
    if not all(
        Path(f"{path}.{suffix}").exists() for suffix in ("eqx", "skeleton.pkl")
    ):
        return None
    return load_model(str(path))
