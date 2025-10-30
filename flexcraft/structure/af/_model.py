"""This model wraps ColabDesign's AF2 to consume AFInputs and return AFResults."""

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
from flexcraft.data.data import DesignData
from flexcraft.structure.af._data import AFInput, AFResult


def make_af2(config, use_multimer=False):
    """Construct an AlphaFold model given a configuration."""
    def inner(params, key, data):
        config.model.num_recycle = None
        return model.RunModel(
            config, None,
            recycle_mode=None,
            use_multimer=use_multimer).apply(params, key, data)
    return inner

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

def make_predict(model, num_recycle=4):
    """Wrap AlphaFold 2 to use DesignData or AFInput inputs and return AFResult."""
    def inner(params, key, data):
        def body(prev, subkey):
            results = model(params, subkey, data)
            prev = results["prev"]
            return prev, results
        if isinstance(data, DesignData):
            data = AFInput.from_data(data)
        if isinstance(data, AFInput):
            data = data.data
        prev = data["prev"]
        if num_recycle - 1 > 0:
            prev, _ = jax.lax.scan(
                jax.remat(body), prev,
                jax.random.split(key, num_recycle - 1))
        data["prev"] = prev
        result = model(params, key, data)
        result = AFResult(inputs=data, result=result)
        return result
    return inner


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
            data = DesignData.from_dict(
                load_pdb(f"{opt.pdb_path}/{name}"))
            data = data.drop_aa()
            num_aa = data["aa"].shape[0]
            center = pmpnn(key(), data)["logits"].mean(axis=0)
            transform = transform_logits([
                toggle_transform(
                    center_logits(center=center), use=opt.center == "True"),
                scale_by_temperature(temperature=opt.temperature),
                forbid("C", aas.PMPNN_CODE),
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
                pmpnn_data = data.update(aa=result["aa"])
                score = -pmpnn(key(), data)["logits"][jnp.arange(num_aa), data["aa"]].mean()
                af_data = data.update(
                    aa=aas.translate(result["aa"], aas.PMPNN_CODE, aas.AF2_CODE))
                print(f"Designed sequence {idx}:")
                sequence = aas.decode(af_data.aa, aas.AF2_CODE)
                print(sequence)
                features = AFInput.from_data(af_data).add_guess(af_data)
                result: AFResult = predictor(params, key(), features)
                plddt = result.plddt.mean()
                pae = result.pae.mean()
                print(f"Predicted structure with PMPNN-score: {score:.2f} pLDDT: {plddt:.2f} and pAE: {pae:.2f}")
                result.save_pdb(f"{opt.out_path}/predictions/{name}/design_{idx}.pdb")
                f.write(f"{name},{idx},{sequence},{plddt:.2f},{pae:.2f}\n")
                f.flush()
            f_succ.write(f"{name},{best_seq},{best_plddt},{best_pae}\n")
            f_succ.flush()
