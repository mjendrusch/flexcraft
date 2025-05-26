from copy import copy
import chex

import numpy as np

import tree
import jax
import jax.numpy as jnp

from colabdesign.af.alphafold.model import model
from colabdesign.af.alphafold.model.geometry import Vec3Array
from colabdesign.af.alphafold.model.all_atom_multimer import atom14_to_atom37, atom37_to_atom14
from colabdesign.af.prep import prep_input_features
import colabdesign.af.inputs as cd_inputs
from colabdesign.af.alphafold.model.config import model_config
from colabdesign.af.alphafold.model.data import get_model_haiku_params
from salad.aflib.common.protein import to_pdb, from_prediction
import flexcraft.sequence.aa_codes as aas


def make_af2(config, use_multimer=False):
    def inner(params, key, data):
        if isinstance(data, DesignData):
            data = AFInput.from_data(data)
        if isinstance(data, AFInputs):
            data = data.data
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

def af_data_from_sequence(sequence):
    L = sequence.shape[0]
    result = prep_input_features(L=L)
    result = update_sequence(result, sequence)
    residue_index = jnp.arange(L, dtype=jnp.int32)
    chain_index = jnp.zeros_like(residue_index)
    result["residue_index"] = residue_index
    result["asym_id"] = result["sym_id"] = chain_index
    result["entity_id"] = jnp.zeros_like(chain_index)
    prev = {'prev_msa_first_row': jnp.zeros([L, 256]),
            'prev_pair': jnp.zeros([L, L, 128]),
            'prev_pos': jnp.zeros([L, 37, 3])}
    result["prev"] = prev
    result["mask_template_interchain"] = False
    result["use_dropout"] = False
    return result

def make_af_data(data):
    seq_one_hot = jax.nn.one_hot(data["aa"], 20, axis=-1)
    if "aa_one_hot" in data:
        seq_one_hot = data["aa_one_hot"]
    result = af_data_from_sequence(seq_one_hot)
    # L = data["aa"].shape[0]
    # result = prep_input_features(L=L)
    # result = update_sequence(result, seq_one_hot)
    result["residue_index"] = data["residue_index"]
    result["asym_id"] = result["sym_id"] = data["chain_index"]
    result["entity_id"] = tie_blocks(data)
    return result

def rename_af_chain(data, to_index=0, from_index=None):
    data = copy(data)
    chain = data["chain_index"]
    if from_index is not None:
        data["chain_index"] = jnp.where(
            chain == from_index, to_index, chain)
    else:
        data["chain_index"] = jnp.full_like(chain, to_index)
    return data

def soft_sequence(
        data: jnp.ndarray, # (N, 20)
        temperature=1.0,
        soft=0.0,
        hard=0.0):
    # tempered softmax
    softmax = jax.nn.softmax(data / temperature, axis=-1)
    # straight-through estimator one-hot
    one_hot = jax.nn.one_hot(jnp.argmax(data, axis=-1), 20)
    one_hot = jax.lax.stop_gradient(one_hot - softmax) + softmax
    # output
    result = soft * softmax + (1 - soft) * data
    result = hard * one_hot + (1 - hard) * result
    return result

def forbid_sequence(
        data: jnp.ndarray, value: float = 0.0) -> jnp.ndarray:
    return data.at[aas.AF2_CODE.index("C")].set(value)

def combined_sequence(
        target, # (N_target, 20)
        binder, # (N_binder, 20)
        **kwargs):
    return jnp.concatenate([
        target, soft_sequence(binder, **kwargs)
    ], axis=0)

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
            prev, _ = jax.lax.scan(
                jax.remat(body), prev,
                jax.random.split(key, num_recycle - 1))
        af_data["prev"] = prev
        return model(params, key, af_data)
    return inner


@chex.dataclass
class AFResult:
    inputs: dict
    result: dict
    def _mean_of_binned(self, name, has_edges=True) -> jnp.ndarray:
        logits = self.result[name]["logits"]
        if has_edges:
            bin_edges = self.result[name]["bin_edges"]
            bin_step = bin_edges[1] - bin_edges[0]
            bin_edges = jnp.concatenate((bin_edges[:1] - bin_step, bin_edges, bin_edges[-1:] + bin_step), axis=0)
            bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2
        else:
            bin_centers = jnp.arange(logits.shape[-1]) / logits.shape[-1]
            bin_centers += 1 / logits.shape[-1] / 2
        return (bin_centers * jax.nn.softmax(logits)).sum(axis=-1)

    @property
    def atom14(self):
        atom37 = self.result["structure_module"]['final_atom_positions']
        mask37 = self.result["structure_module"]['final_atom_mask']
        atom14, mask14 = atom37_to_atom14(self.result["aatype"], Vec3Array.from_array(atom37), mask37)
        return atom14.to_array(), mask14

    @property
    def atom4(self) -> jnp.ndarray:
        atom14, _ = self.atom14
        return atom14[:, :4]

    @property
    def plddt(self):
        return self._mean_of_binned("predicted_lddt", has_edges=False)

    @property
    def pae(self):
        return self._mean_of_binned("predicted_aligned_error", has_edges=False)
    
    @property
    def ipae(self):
        pae = self.pae
        chain = self.inputs["asym_id"]
        other_chain = chain[:, None] != chain[None, :]
        return (pae * other_chain).sum() / jnp.maximum(1, other_chain.sum())

    @property
    def distance(self):
        return self._mean_of_binned("distogram")

    def contact_probability(self, contact_distance=10.0) -> jnp.ndarray:
        distogram = jax.nn.softmax(self.result["distogram"]["logits"], axis=-1)
        bin_edges: jnp.ndarray = self.result["distogram"]["bin_edges"]
        bin_step = bin_edges[1] - bin_edges[0]
        bin_edges = jnp.concatenate((
            bin_edges[:1] - bin_step, bin_edges, bin_edges[-1:] + bin_step), axis=0)    
        edge_mask = bin_edges[1:] < contact_distance
        return (edge_mask * distogram).sum(axis=-1)
    
    def contact_entropy(self, contact_distance=14.0) -> jnp.ndarray:
        distogram = jax.nn.log_softmax(self.result["distogram"]["logits"], axis=-1)
        bin_edges: jnp.ndarray = self.result["distogram"]["bin_edges"]
        bin_step = bin_edges[1] - bin_edges[0]
        bin_edges = jnp.concatenate((
            bin_edges[:1] - bin_step, bin_edges, bin_edges[-1:] + bin_step), axis=0)    
        edge_mask = bin_edges[1:] < contact_distance
        distogram_clipped = jax.nn.softmax(distogram - 1e9 * (1 - edge_mask), axis=-1)
        distogram_clipped = jnp.where(edge_mask, distogram_clipped, 0)
        return -(distogram_clipped * distogram).sum(axis=-1)

    def save_pdb(self, path):
        _save_af_pdb(path, self)
        return path

def _save_af_pdb(path: str, result: AFResult):
    plddt = result.plddt
    protein = from_prediction(
        tree.map_structure(lambda x: np.array(x), result.inputs),
        tree.map_structure(lambda x: np.array(x), result.result),
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
    import os
    from flexcraft.utils.options import parse_options
    from flexcraft.utils.rng import Keygen
    from flexcraft.utils import Keygen, parse_options, load_pdb, strip_aa, tie_homomer
    from flexcraft.sequence.sample import *
    from flexcraft.sequence.mpnn import make_pmpnn
    # from colabdesign.af.alphafold.common.protein import from_prediction, to_pdb
    opt = parse_options(
        "predict structures with AlphaFold",
        param_path="params/",
        pmpnn_path="../prosesame/v_48_030.pkl",
        model_name="model_1_ptm",
        pdb_path="pdb_files/",
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
    pmpnn = jax.jit(make_pmpnn(opt.pmpnn_path, eps=0.05))

    os.makedirs(f"{opt.out_path}/predictions/", exist_ok=True)
    with open(f"{opt.out_path}/scores.csv", "wt") as f, \
         open(f"{opt.out_path}/succ.csv", "wt") as f_succ:
        f.write("name,design,sequence,plddt,pae\n")
        f_succ.write("name,sequence,plddt,pae\n")
        for name in os.listdir(opt.pdb_path):
            data = load_pdb(f"{opt.pdb_path}/{name}")
            data = strip_aa(data)
            num_aa = data["aa"].shape[0]
            center = pmpnn(key(), data)["logits"].mean(axis=0)
            transform = transform_logits([
                toggle_transform(
                    center_logits(center=center), use=opt.center == "True"),
                scale_by_temperature(temperature=opt.temperature),
                #forbid("C", aas.PMPNN_CODE),
                norm_logits
            ])
            sampler = sample(pmpnn, logit_transform=transform)
            os.makedirs(f"{opt.out_path}/predictions/{name}/", exist_ok=True)
            best_seq = ""
            best_plddt = 0.0
            best_pae = 1.0
            prev_aa = data["aa"]
            for idx in range(opt.samples):
                # data = strip_aa(data)
                result, log_p = sampler(key(), data)
                data["aa"] = aas.translate(result["aa"], aas.PMPNN_CODE, aas.AF2_CODE)
                score = -pmpnn(key(), data)["logits"][jnp.arange(num_aa), data["aa"]].mean()
                print(f"Designed sequence {idx}:")
                sequence = aas.decode(data["aa"], aas.AF2_CODE)
                print(sequence)
                features = make_af_data(data)
                result = AFResult(inputs=features,
                                  result=predictor(params, key(), features))
                plddt = result.plddt.mean()
                # redesign worst:
                lddt = result.plddt
                bad_plddt = 0.7
                if plddt > 0.9 or (lddt > bad_plddt).all():
                    data = strip_aa(data)
                    prev_aa = data["aa"]
                else:
                    data["aa"] = aas.translate(data["aa"], aas.AF2_CODE, aas.PMPNN_CODE)
                    data["aa"] = jnp.where(lddt <= bad_plddt, 20, data["aa"])
                pae = result.pae.mean()
                if plddt > best_plddt:
                    best_seq = sequence
                    best_plddt = plddt
                    best_pae = pae
                print(f"Predicted structure with PMPNN-score: {score:.2f} pLDDT: {plddt:.2f} and pAE: {pae:.2f}")
                result.save_pdb(f"{opt.out_path}/predictions/{name}/design_{idx}.pdb")
                f.write(f"{name},{idx},{sequence},{plddt:.2f},{pae:.2f}\n")
                f.flush()
            f_succ.write(f"{name},{best_seq},{best_plddt},{best_pae}\n")
            f_succ.flush()
