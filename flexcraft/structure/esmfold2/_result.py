from typing import Any
from dataclasses import dataclass

import numpy as np

import jax
import equinox as eqx

from flexcraft.structure.common._result import AF3LikeResult
from flexcraft.structure.esmfold2._data import ESMFold2Writer

class ESMFold2Confidence(eqx.Module):
    """The subset of the confidence head output we keep across samples."""
    plddt: jax.Array
    plddt_logits: jax.Array
    pae: jax.Array
    pae_logits: jax.Array
    ptm: jax.Array
    iptm: jax.Array

    @classmethod
    def from_head(cls, confidence: dict) -> "ESMFold2Confidence":
        return cls(**{name: confidence[name] for name in
                      ("plddt", "plddt_logits", "pae", "pae_logits", "ptm", "iptm")})

class ESMFold2Result(AF3LikeResult):
    data: dict

    @property
    def atom_to_token(self):
        features = self.data["features"]
        num_tokens = features["residue_index"].shape[1]
        one_hot = jax.nn.one_hot(features["atom_to_token"][0], num_tokens, axis=1)
        # padding atoms carry index 0, so without the mask token 0 absorbs them
        # and atom_array_to_atomX mis-packs every token after it. The model drops
        # them the same way, see ESMFold2._prepare_embeddings.
        return one_hot * features["atom_attention_mask"][0][:, None]

    @property
    def plddt_logits(self):
        # the head scores atoms, so pick out each token's representative atom
        index = self.data["features"]["distogram_atom_idx"][0]
        plddt_logits = self.data["confidence"].plddt_logits
        if self.is_single_sample:
            return plddt_logits[0, index]
        return plddt_logits[:, 0, index]

@dataclass
class ESMFold2Prediction:
    data: Any
    writer: ESMFold2Writer
    @property
    def result(self):
        return ESMFold2Result(data=self.data)

    def prep_data(self, sample_index=0):
        is_multisample = len(self.data["samples"].shape) == 4
        coords = self.data["samples"]
        plddt = self.data["confidence"].plddt
        if is_multisample:
            coords = coords[sample_index, 0]
            plddt = plddt[sample_index, 0]
        else:
            coords = coords[0]
            plddt = plddt[0]
        return np.array(coords), np.array(plddt)

    def save_pdb(self, path, sample_index=0):
        coords, plddt = self.prep_data(sample_index=sample_index)
        self.writer.save_pdb(path, coords, plddt=plddt)

    def save_cif(self, path, sample_index=0):
        coords, plddt = self.prep_data(sample_index=sample_index)
        self.writer.save_cif(path, coords, plddt=plddt)

    def save(self, path):
        self.result.save(path)
