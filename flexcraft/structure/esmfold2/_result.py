from dataclasses import dataclass
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from flexcraft.structure.common._result import AF3LikeResult
from flexcraft.structure.esmfold2._data import (
    ESMFOLD2_FEATURE_SCHEMA,
    ESMFold2Writer,
)


class ESMFold2Confidence(eqx.Module):
    """Subset of confidence head outputs retained across samples."""

    plddt: jax.Array
    plddt_logits: jax.Array
    pae: jax.Array
    pae_logits: jax.Array
    ptm: jax.Array
    iptm: jax.Array

    @classmethod
    def from_head(cls, confidence: dict) -> "ESMFold2Confidence":
        return cls(
            **{
                name: confidence[name]
                for name in (
                    "plddt",
                    "plddt_logits",
                    "pae",
                    "pae_logits",
                    "ptm",
                    "iptm",
                )
            }
        )


class ESMFold2Result(AF3LikeResult):
    data: dict

    _plddt_logits_is_atom_level = True

    @property
    def token_pad_mask(self) -> jax.Array:
        return self.data["features"]["token_attention_mask"][0].astype(
            jnp.bool_
        )

    @property
    def atom_pad_mask(self) -> jax.Array:
        return self.data["features"]["atom_attention_mask"][0].astype(
            jnp.bool_
        )

    @property
    def atom_to_token(self):
        features = self.data["features"]
        num_tokens = features["residue_index"].shape[1]
        one_hot = jax.nn.one_hot(
            features["atom_to_token"][0], num_tokens, axis=1
        )
        return one_hot * self.atom_pad_mask[:, None]

    @property
    def plddt_logits(self):
        index = self.data["features"]["distogram_atom_idx"][0]
        plddt_logits = self.data["confidence"].plddt_logits
        if self.is_single_sample:
            return plddt_logits[0, index]
        return plddt_logits[:, 0, index]

    def _feature_schema(self) -> dict:
        return ESMFOLD2_FEATURE_SCHEMA

    def unpad(self) -> "ESMFold2Result":
        return super().unpad()


@dataclass
class ESMFold2Prediction:
    data: Any
    writer: ESMFold2Writer

    @property
    def result(self) -> ESMFold2Result:
        return ESMFold2Result(data=self.data)

    def prep_data(self, sample_index=0):
        unpadded = self.result.unpad()
        is_multisample = len(unpadded.data["samples"].shape) == 4
        coords = unpadded.data["samples"]
        plddt = unpadded.data["confidence"].plddt
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
