from dataclasses import dataclass
from typing import Any

from flexcraft.structure.boltz._data import (
    Joltz2Writer,
    get_boltz_feature_schema,
)
from flexcraft.structure.common._result import AF3LikeResult


class JoltzResult(AF3LikeResult):
    data: dict

    def _feature_schema(self) -> dict:
        features = self.data.get("features", {})
        current_tokens = (
            int(features["res_type"].shape[1]) if "res_type" in features else None
        )
        return get_boltz_feature_schema(current_tokens)


@dataclass
class JoltzPrediction:
    data: Any
    writer: Joltz2Writer

    @property
    def result(self) -> JoltzResult:
        return JoltzResult(data=self.data)

    def _prep_sample(self, sample_index: int = 0):
        unpadded = self.result.unpad()
        samples = unpadded.data["samples"]
        plddt = unpadded.data["confidence"].plddt
        if len(samples.shape) == 4:
            return samples[sample_index], plddt[sample_index]
        return samples, plddt

    def save_pdb(self, path, sample_index=0):
        coords, plddt = self._prep_sample(sample_index=sample_index)
        self.writer.save_pdb(path, coords, plddt=plddt)

    def save_cif(self, path, sample_index=0):
        coords, plddt = self._prep_sample(sample_index=sample_index)
        self.writer.save_cif(path, coords, plddt=plddt)

    def save(self, path):
        self.result.save(path)
