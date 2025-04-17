from copy import copy
import chex

import numpy as np

import jax
import jax.numpy as jnp

from colabdesign.af.alphafold.model import model
from colabdesign.af.alphafold.model.geometry import Vec3Array
from colabdesign.af.alphafold.model.all_atom_multimer import atom14_to_atom37
from colabdesign.af.prep import prep_input_features
import colabdesign.af.inputs as cd_inputs
from colabdesign.af.alphafold.model.config import model_config
from colabdesign.af.alphafold.model.data import get_model_haiku_params
from alphafold.common.protein import to_pdb, from_prediction

def make_af2(config, use_multimer=False):
    def inner(params, key, data):
        return model.RunModel(
            config, None,
            recycle_mode=None,
            use_multimer=use_multimer).apply(params, key, data)
    return inner

def update_sequence(features, one_hot):
    features = copy(features)
    cd_inputs.update_seq(one_hot[None], features)
    cd_inputs.update_aatype(jnp.argmax(one_hot, axis=-1), features)
    return features

def tie_blocks(data):
    if "tie_blocks" in data:
        return data["tie_blocks"]
    tie_index = data["tie_index"]
    store = np.full_like(tie_index, -1)
    store[tie_index] = data["chain_index"]
    return store[tie_index]

def make_af_data(data):
    L = data["aa"].shape[0]
    result = prep_input_features(L=L)
    seq_one_hot = jax.nn.one_hot(data["aa"], 21, axis=-1)
    result = update_sequence(result, seq_one_hot)
    result["residue_index"] = data["residue_index"]
    result["asym_id"] = result["sym_id"] = data["chain_index"]
    result["entity_id"] = tie_blocks(data)
    prev = {'prev_msa_first_row': np.zeros([L, 256]),
            'prev_pair': np.zeros([L, L, 128]),
            'prev_pos': np.zeros([L, 37, 3])}
    result["prev"] = prev
    result["mask_template_interchain"] = False
    result["use_dropout"] = False
    return result

def add_guess(features, data):
    atom_positions = atom14_to_atom37(data["atom_positions"], data["aa"])
    features = copy(features)
    features["prev"] = copy(features["prev"])
    features["prev"]["prev_pos"] = atom_positions
    return features

def make_predict(model, num_recycle=4):
    def inner(params, key, af_data):
        def body(prev, subkey):
            results = model(params, subkey, af_data)
            prev = results["prev"]
            return prev, results
        prev = af_data["prev"]
        if num_recycle > 0:
            prev, _ = jax.lax.scan(body, prev, jax.random.split(key, num_recycle - 1))
        af_data["prev"] = prev
        return model(params, key, af_data)
    return inner


@chex.dataclass
class AFResult:
    result: dict
    def _mean_of_binned(self, name, has_edges=True) -> jnp.ndarray:
        logits = self.result[name]["logits"]
        if has_edges:
            bin_edges = self.result[name]["bin_edges"]
            bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2
        else:
            bin_centers = jnp.arange(logits.shape[-1]) / logits.shape[-1]
            bin_centers += 1 / logits.shape[-1] / 2
        return (bin_centers * jax.nn.softmax(logits)).sum(axis=-1)

    @property
    def plddt(self):
        return self._mean_of_binned("predicted_lddt", has_edges=False)

    @property
    def pae(self):
        return self._mean_of_binned("predicted_aligned_error", has_edges=False)

    @property
    def distance(self):
        return self._mean_of_binned("distogram")

    def save_pdb(self, path):
        save_af_pdb(path, self)

def save_af_pdb(path: str, result: AFResult):
    plddt = result.plddt
    protein = from_prediction(
        features,
        result.result,
        b_factors=jnp.broadcast_to(plddt[:, None], (plddt.shape[0], 37)),
        remove_leading_feature_dimension=False)
    pdb_string = to_pdb(protein)
    with open(path, "wt") as f:
        f.write(pdb_string)

def rec_print_dict(x, indent=0):
    if isinstance(x, dict):
        for k, v in x.items():
            print(f"{'  ' * indent}{k}: {rec_print_dict(v, indent + 1)}")
    elif isinstance(x, list):
        for k, v in enumerate(x):
            print(f"{'  ' * indent}{k}: {rec_print_dict(v, indent + 1)}")
    else:
        if hasattr(x, "shape"):
            print(x.shape)
        else:
            print(type(x))


if __name__ == "__main__":
    from flexcraft.utils.options import parse_options
    from flexcraft.utils.rng import Keygen
    from flexcraft.utils import Keygen, parse_options, load_pdb, strip_aa, tie_homomer
    from flexcraft.sequence.sample import *
    from flexcraft.sequence.mpnn import make_pmpnn
    from flexcraft.sequence.aa_codes import reindex_aatype, decode_sequence, AF2_AA_CODE, PMPNN_AA_CODE
    # from colabdesign.af.alphafold.common.protein import from_prediction, to_pdb
    opt = parse_options(
        "predict structures with AlphaFold",
        param_path="params/",
        pmpnn_path="../prosesame/v_48_030.pkl",
        model_name="model_1_ptm",
        pdb_path="pdb_path.pdb",
        out_path="out",
        center="False",
        temperature=0.1,
        samples=10,
        seed=42
    )
    params = get_model_haiku_params(
        model_name=opt.model_name,
        data_dir=opt.param_path, fuse=True)
    config = model_config(opt.model_name)
    config.model.global_config.use_dgram = False
    predictor = jax.jit(make_predict(make_af2(config), num_recycle=2))
    key = Keygen(opt.seed)
    data = load_pdb(opt.pdb_path)
    data = strip_aa(data)
    num_aa = data["aa"].shape[0]
    pmpnn = jax.jit(make_pmpnn(opt.pmpnn_path, eps=0.05))
    center = pmpnn(key(), data)["logits"].mean(axis=0)
    transform = transform_logits([
        toggle_transform(
            center_logits(center=center), use=opt.center == "True"),
        scale_by_temperature(temperature=opt.temperature),
        #forbid("C", PMPNN_AA_CODE),
        norm_logits
    ])
    sampler = sample(pmpnn, logit_transform=transform)

    with open(f"{opt.out_path}_scores.csv", "wt") as f:
        f.write("design,sequence,plddt,pae\n")
        for idx in range(opt.samples):
            data = strip_aa(data)
            result, log_p = sampler(key(), data)
            data["aa"] = reindex_aatype(result["aa"], PMPNN_AA_CODE, AF2_AA_CODE)
            score = -pmpnn(key(), data)["logits"][jnp.arange(num_aa), data["aa"]].mean()
            print(f"Designed sequence {idx}:")
            sequence = decode_sequence(data["aa"], AF2_AA_CODE)
            print(sequence)
            features = make_af_data(data)
            result = AFResult(result=predictor(params, key(), features))
            plddt = result.plddt.mean()
            pae = result.pae.mean()
            print(f"Predicted structure with PMPNN-score: {score:.2f} pLDDT: {plddt:.2f} and pAE: {pae:.2f}")
            result.save_pdb(opt.out_path + f"_{idx}.pdb")
            f.write(f"{idx},{sequence},{plddt:.2f},{pae:.2f}\n")
            f.flush()
