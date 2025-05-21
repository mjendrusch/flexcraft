
import os
import time
from copy import deepcopy

import jax
import jax.numpy as jnp

import haiku as hk

import salad.inference as si

from flexcraft.sequence.mpnn import make_pmpnn
from flexcraft.sequence.sample import *
from flexcraft.sequence.aa_codes import PMPNN_AA_CODE, AF2_AA_CODE, decode_sequence
from flexcraft.structure.af import *
from flexcraft.utils import Keygen, parse_options, strip_aa, data_from_protein

# here, we implement a denoising step
def model_step(config):
    # get the configuration and set eval to true
    # this turns off any model components that
    # are only used during training.
    config = deepcopy(config)
    config.eval = True
    # salad is built on top of haiku, which means
    # that any functions using salad modules need
    # to be hk.transform-ed before use.
    @hk.transform
    def step(data, prev):
        # instantiate a noise generator
        noise = si.StructureDiffusionNoise(config)
        # and a denoising model
        predict = si.StructureDiffusionPredict(config)
        # we can edit the structure before noise
        # is applied here:
        ...
        # apply noise
        data.update(noise(data))
        # and edit the noised structure here:
        ...
        # run model
        out, prev = predict(data, prev)
        # or the output of the model here:
        ...
        return out, prev
    # return the pure apply function generated
    # by haiku from our step
    return step.apply


opt = parse_options(
    "Protein sequence design with custom protein MPNN",
    num_aa="100",
    out_path="outputs/",
    salad_params="./default_vp.jax",
    pmpnn_params="../prosesame/v_48_030.pkl",
    af2_params="params/",
    af2_model="model_1_ptm",
    temperature=0.1,
    center="True",
    num_designs=100,
    num_samples=10,
    seed=42
)
os.makedirs(opt.out_path, exist_ok=True)

key = Keygen(opt.seed)
pmpnn = jax.jit(make_pmpnn(opt.pmpnn_params, eps=0.05))

# make salad model
salad_config, salad_params = si.make_salad_model("default_vp", opt.salad_params)

# initialize salad data and prev from the num_aa specification
num_aa, resi, chain, is_cyclic, cyclic_mask = si.parse_num_aa(opt.num_aa)
salad_data, init_prev = si.data.from_config(
    salad_config,
    num_aa=num_aa,
    residue_index=resi,
    chain_index=chain,
    cyclic_mask=cyclic_mask)

salad_step = model_step(salad_config)

# make AF2 model
af2_params = get_model_haiku_params(
    model_name=opt.af2_model,
    data_dir=opt.af2_params, fuse=True)
af2_config = model_config(opt.af2_model)
af2_config.model.global_config.use_dgram = False
af2 = jax.jit(make_predict(make_af2(af2_config), num_recycle=2))

# build a sampler object for sampling
sampler = si.Sampler(salad_step, out_steps=400)
# run a loop with num_design steps
print("Starting design...")
for idx in range(opt.num_designs):
    # generate a structure in each step
    start = time.time()
    design = sampler(salad_params, key(), salad_data, init_prev)
    print(f"Design {idx} in {time.time() - start:.2f} s")
    # convert it to ProteinMPNN input
    pmpnn_data = data_from_protein(si.data.to_protein(design))
    pmpnn_data = strip_aa(pmpnn_data)
    step_0_logits = pmpnn(key(), pmpnn_data)["logits"]
    center = step_0_logits.mean(axis=0)
    # ensure one TRP/W residue
    p_W = jax.nn.softmax(step_0_logits - center)[:, PMPNN_AA_CODE.index("W")]
    best_W = jnp.argmax(p_W, axis=0)
    # set up logit transform
    transform = transform_logits([
        toggle_transform(
            center_logits(center=center), use=opt.center == "True"),
        scale_by_temperature(temperature=opt.temperature),
        forbid("C", PMPNN_AA_CODE),
        norm_logits
    ])
    pmpnn_sampler = sample(pmpnn, logit_transform=transform)
    for ids in range(opt.num_samples):
        pmpnn_data = strip_aa(pmpnn_data)
        # ensure one TRP/W residue (at the most probable position)
        pmpnn_data["aa"][best_W] = PMPNN_AA_CODE.index("W")
        # sample all other residues
        result, log_p = pmpnn_sampler(key(), pmpnn_data)
        pmpnn_data["aa"] = reindex_aatype(result["aa"], PMPNN_AA_CODE, AF2_AA_CODE)
        sequence = decode_sequence(pmpnn_data["aa"], AF2_AA_CODE)
        af_data = make_af_data(pmpnn_data)
        result = AFResult(inputs=af_data,
                          result=af2(af2_params, key(), af_data))
        plddt = result.plddt.mean()
        print(idx, ids, sequence, plddt)
        if plddt > 0.8:
            result.save_pdb(f"{opt.out_path}/result_{idx}.pdb")
            break
