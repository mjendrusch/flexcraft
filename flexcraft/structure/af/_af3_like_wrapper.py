
from typing import Any
from dataclasses import dataclass
from copy import copy

import numpy as np

import jax
import jax.numpy as jnp

import warnings

from flexcraft.files import PDBFile

from flexcraft.data.data import DesignData
from flexcraft.structure.common._data import InputSpec
from flexcraft.structure.af import AFInput, AFResult, make_af2, get_model_haiku_params, model_config, make_predict

_CHAIN_NAMES = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _apply_pad_spec(inp, pad_spec):
    """Apply a _pad_spec (dict, tuple, or list) to a model input."""
    if pad_spec is None:
        return inp
    if isinstance(pad_spec, dict):
        method = pad_spec.get("method") or pad_spec.get("mode") or pad_spec.get("kind")
        if method == "pad_atoms":
            return inp.pad_atoms(pad_spec.get("num_residue_atoms", 24))
        if pad_spec.get("_multiple") is not None or method == "pad_to_nearest_multiple":
            mult = pad_spec.get("_multiple", pad_spec.get("multiple"))
            return inp.pad_to_nearest_multiple(
                mult,
                pad_spec.get("num_residue_atoms", 24),
                num_msa=pad_spec.get("num_msa"),
                num_templates=pad_spec.get("num_templates"))
        return inp.pad_to(
            pad_spec.get("size"),
            pad_spec.get("num_residue_atoms", 24),
            num_msa=pad_spec.get("num_msa"),
            num_templates=pad_spec.get("num_templates"))
    if isinstance(pad_spec, tuple) and len(pad_spec) == 3:
        method_name, args, kwargs = pad_spec
        return getattr(inp, method_name)(*args, **kwargs)
    if isinstance(pad_spec, list):
        for op in pad_spec:
            inp = _apply_pad_spec(inp, op)
        return inp
    raise TypeError(f"Unsupported _pad_spec format: {pad_spec!r}")


class AF2Spec(InputSpec):
    def to_input(self, **kwargs):
        for chain in self.chains:
            if chain["kind"] != "protein":
                warnings.warn(f"Skipping invalid chain type '{chain['kind']}' for AF2.")
        if self.constraints:
            warnings.warn(f"Skipping provided constraints for AF2.")
        af_input = AFInput.from_chains(*[c for c in self.chains if c["kind"] == "protein"])
        for template_info in self.templates:
            # templates are dicts with "pdb"/"cif" keys, not bare strings
            path = template_info.get("pdb") or template_info.get("cif") or template_info.get("template")
            if isinstance(path, str) and path.endswith((".pdb", ".pdb1")):
                tdata: DesignData = PDBFile(path).to_data()
            elif isinstance(path, str) and path.endswith(".cif"):
                raise NotImplementedError("Under construction.")
            else:
                raise ValueError(f"Invalid template path: {path}")
            if "template_chains" in template_info:
                chains = [_CHAIN_NAMES.index(c) for c in template_info["template_chains"]]
            else:
                chains = np.arange(len(_CHAIN_NAMES))
            pos = tdata.data["atom_positions"]
            aa = af_input.data["aatype"]
            where = jnp.isin(af_input.chain_index, chains)
            af_input = af_input.add_template(pos=pos, aa=aa, where=where)
        # Apply padding spec if set (use _apply_pad_spec or inherited method)
        pad_spec = getattr(self, "_pad_spec", None)
        if pad_spec is not None:
            af_input = _apply_pad_spec(af_input, pad_spec)
        return af_input

class AFEvaluator:
    def __init__(self, model, params):
        self.model = model
        self.params = params

    def predict(self, key, input: AFInput, num_samples=1):
        return self.model(self.params, key, input)

    def __call__(self, key, input: AFInput, num_samples=1):
        return self.predict(key, input, num_samples=num_samples)

@dataclass
class AF2Prediction:
    data: AFResult
    writer: Any = None
    @property
    def result(self):
        return self.data

    def save_pdb(self, path, sample_index=0):
        """Save AF2 prediction as a PDB file. Unpads automatically."""
        result = self.data
        if not isinstance(result.token_pad_mask, jax.core.Tracer):
            result = result.unpad()
        result.save_pdb(path)

    def save_cif(self, path, sample_index=0):
        raise NotImplementedError("AF2 results do not save as CIF at the moment.")

    def save(self, path):
        """Save AF2 features as a compressed npz archive for debugging."""
        self.result.save(path)

class AFWrapper:
    def __init__(self, model: str = "af2_model_1_ptm", cache: str = "params/af/", **kwargs):
        self.model_name = model[4:]
        self.af2_params = get_model_haiku_params(
                model_name=self.model_name,
                data_dir=cache, fuse=True)
        self.af2_config = model_config(self.model_name)
        self.multimer = "multimer" in self.model_name
        self.af2_config.model.global_config.use_dgram = False
        self.af2_config.model.global_config.bfloat16 = kwargs.get("bfloat16", False)

    def evaluator(self, num_recycle=4, **kwargs):
        af2 = jax.jit(make_predict(make_af2(self.af2_config, self.multimer), num_recycle=num_recycle))
        def _wrapper(params):
            return AFEvaluator(af2, params)
        return _wrapper, self.af2_params

    def predictor(self, num_recycle=4, **kwargs):
        evaluator, params = self.evaluator(num_recycle=num_recycle, **kwargs)
        def _wrapper(key, spec, **kwargs):
            # Accept AF2Spec, InputSpec, or AFInput directly
            if isinstance(spec, AFInput):
                af_input = spec
            elif isinstance(spec, InputSpec):
                af_input = spec.to_input()
            else:
                af_input = spec.to_input()
            result = evaluator(params)(key, af_input, num_samples=1)
            # Unpad on host
            if not isinstance(result.token_pad_mask, jax.core.Tracer):
                result = result.unpad()
            return AF2Prediction(result)
        return _wrapper
