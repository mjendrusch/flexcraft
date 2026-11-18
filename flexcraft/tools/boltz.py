"""Deprecated legacy Boltz CLI wrapper."""

import json
import os
import uuid
import warnings

import numpy as np
import yaml

from flexcraft.data.data import DesignData
from flexcraft.files.pdb import PDBFile
import flexcraft.sequence.aa_codes as aas

_DEPRECATION_MESSAGE = (
    "flexcraft.tools is deprecated and will be removed in a future release."
)

warnings.warn(_DEPRECATION_MESSAGE, DeprecationWarning, stacklevel=2)


class BoltzPredictor:
    def __init__(self, path, tmpdir="tmp/", gpu=8):
        warnings.warn(_DEPRECATION_MESSAGE, DeprecationWarning, stacklevel=2)
        self.path = path
        self.tmpdir = tmpdir
        self.gpu = gpu

    def __call__(self, data: DesignData, homomer=False):
        warnings.warn(_DEPRECATION_MESSAGE, DeprecationWarning, stacklevel=2)
        input_id = str(uuid.uuid4())
        boltz_input = BoltzYAML(
            data,
            path=f"{self.tmpdir}/predict_{input_id}.yaml",
            homomer=homomer,
        )
        output = BoltzResult(
            boltz_input.basename(),
            boltz_input.basename(),
            path=f"{self.tmpdir}/output_{input_id}/",
        )
        os.system(
            f"bash {self.path}/scripts/run_boltz.sh {boltz_input.path} "
            f"{output.path} {self.gpu} &> {self.tmpdir}/log_{input_id}"
        )
        result = output.load()
        boltz_input.remove()
        output.remove()
        return result


class BoltzYAML:
    def __init__(self, data: DesignData | None = None, path=None, homomer=False):
        warnings.warn(_DEPRECATION_MESSAGE, DeprecationWarning, stacklevel=2)
        self.path = path
        self.data = data
        self.homomer = homomer
        if self.data is not None:
            self.write_data()

    def remove(self):
        return self.data

    def basename(self):
        return os.path.basename(self.path).split(".")[0]

    def write_data(self):
        chain_names = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        data = self.data.unpad().cpu()
        chain_index = data["chain_index"]
        chains = np.unique(chain_index)
        chains = chains[chains >= 0]
        result = dict(version=1, sequences=[])
        if self.homomer:
            c = chains[0]
            chain_data = data[data["chain_index"] == c]
            chain_name = [chain_names[i] for i in range(len(chains))]
            sequence = aas.decode(chain_data.aa, aas.AF2_CODE)
            msa = "empty"
            result["sequences"].append(
                dict(protein=dict(id=chain_name, sequence=sequence, msa=msa))
            )
        else:
            for c in chains:
                chain_data = data[data["chain_index"] == c]
                chain_name = chain_names[c]
                sequence = aas.decode(chain_data.aa, aas.AF2_CODE)
                msa = "empty"
                result["sequences"].append(
                    dict(protein=dict(id=chain_name, sequence=sequence, msa=msa))
                )
        result_yaml = yaml.dump(result)
        tmpdir = os.path.dirname(self.path)
        if tmpdir and not os.path.isdir(tmpdir):
            os.makedirs(tmpdir)
        with open(self.path, "wt") as handle:
            handle.write(result_yaml)


class BoltzResult:
    def __init__(self, base_name, input_name, path):
        warnings.warn(_DEPRECATION_MESSAGE, DeprecationWarning, stacklevel=2)
        self.path = path
        self.base_name = base_name
        self.input_name = input_name
        self.data = None
        self.chain_iptm = None

    def load(self):
        path = f"{self.path}/boltz_results_{self.base_name}/predictions/{self.input_name}/"
        result = dict()
        for name in os.listdir(path):
            full_path = f"{path}/{name}"
            base = name.split(".")[0]
            model = int(base.split("_")[-1])
            if model not in result:
                result[model] = dict(structure=..., confidence=...)
            if full_path.endswith(".pdb"):
                result[model]["structure"] = PDBFile(path=full_path).to_data()
            elif full_path.endswith(".json"):
                with open(full_path, "rt") as handle:
                    result[model]["confidence"] = json.load(handle)
        self.data = result[0]
        return self

    @property
    def plddt(self):
        return self.data["confidence"]["complex_plddt"]

    @property
    def ptm(self):
        return self.data["confidence"]["ptm"]

    @property
    def iptm(self):
        return self.data["confidence"]["iptm"]

    @property
    def pair_ptm(self):
        if self.chain_iptm is None:
            chain_iptm = self.data["confidence"]["pair_chains_iptm"]
            indices = sorted([int(key) for key in chain_iptm.keys()])
            num_chains = max(indices) + 1
            result = np.zeros((num_chains, num_chains), dtype=np.float32)
            for i in indices:
                for j in indices:
                    result[i, j] = chain_iptm[str(i)][str(j)]
            self.chain_iptm = result
        return self.chain_iptm

    def remove(self):
        return self
