
import os

import numpy as np
import yaml
import json
import uuid

from flexcraft.data.data import DesignData
from flexcraft.files.pdb import PDBFile
import flexcraft.sequence.aa_codes as aas

class BoltzPredictor:
    def __init__(self, path, tmpdir="tmp/", gpu=8):
        self.path = path
        self.tmpdir = tmpdir
        self.gpu = gpu

    def __call__(self, data: DesignData, homomer=False):
        input_id = str(uuid.uuid4())
        input = BoltzYAML(data,
                          path=f"{self.tmpdir}/predict_{input_id}.yaml",
                          homomer=homomer)
        output = BoltzResult(
            input.basename(), input.basename(), path=f"{self.tmpdir}/output_{input_id}/")
        os.system(f"bash {self.path}/scripts/run_boltz.sh {input.path} {output.path} {self.gpu} &> {self.tmpdir}/log_{input_id}")
        result = output.load()
        input.remove()
        output.remove()
        return result

class BoltzYAML:
    def __init__(self, data: DesignData | None = None,
                 path=None, homomer=False):
        self.path = path
        self.data = data
        self.homomer = homomer
        if self.data is not None:
            self.write_data()

    def remove(self):
        #os.remove(self.path)
        return self.data

    def basename(self):
        return os.path.basename(self.path).split(".")[0]

    def write_data(self):
        chain_names = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        data = self.data.cpu()
        chain_index = data["chain_index"]
        chains = np.unique(chain_index)
        result = dict(version=1, sequences=[])
        if self.homomer:
            c = chains[0]
            chain_data = data[data["chain_index"] == c]
            chain_name = [chain_names[i] for i in range(len(chains))]
            sequence = aas.decode(chain_data.aa, aas.AF2_CODE)
            msa = "empty"
            result["sequences"].append(dict(
                protein=dict(id=chain_name, sequence=sequence, msa=msa)))
        else:
            for c in chains:
                chain_data = data[data["chain_index"] == c]
                chain_name = chain_names[c]
                sequence = aas.decode(chain_data.aa, aas.AF2_CODE)
                msa = "empty"
                result["sequences"].append(dict(
                    protein=dict(id=chain_name, sequence=sequence, msa=msa)))
        result_yaml = yaml.dump(result)
        tmpdir = os.path.dirname(self.path)
        if tmpdir and not os.path.isdir(tmpdir):
            os.makedirs(tmpdir)
        with open(self.path, "wt") as f:
            f.write(result_yaml)

class BoltzResult:
    def __init__(self, base_name, input_name, path):
        self.path = path
        self.base_name = base_name
        self.input_name = input_name
        self.data = None
        self.chain_iptm = None

    def load(self):
        base = os.listdir(self.path)[0]
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
                with open(full_path, "rt") as f:
                    result[model]["confidence"] = json.load(f)
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
            N = max(indices) + 1
            result = np.zeros((N, N), dtype=np.float32)
            for i in indices:
                for j in indices:
                    result[i, j] = chain_iptm[str(i)][str(j)]
            self.chain_iptm = result
        return self.chain_iptm

    def remove(self):
        #os.system(f"rm -r {self.path}/boltz_results_{self.base_name}/")
        return self
