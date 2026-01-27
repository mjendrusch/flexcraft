
import os
import time
from copy import deepcopy

import jax
import jax.numpy as jnp

import haiku as hk

import salad.inference as si

import flexcraft.sequence.aa_codes as aas
from flexcraft.sequence.mpnn import make_pmpnn
from flexcraft.sequence.sample import *
from flexcraft.structure.af import *
from flexcraft.utils import Keygen, parse_options, data_from_protein
from flexcraft.files.pdb import PDBFile
from flexcraft.tools.boltz import BoltzYAML

from flexcraft.rosetta.relax import fastrelax
import pyrosetta as pr
from flexcraft.structure.metrics import RMSD, sc_lddt, lddt

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
    salad_params="params/salad/default_vp-200k.jax",
    pmpnn_params="params/pmpnn/v_48_030.pkl",
    af2_params="params/af/",
    af2_model="model_1_ptm",
    temperature=0.1,
    center="True",
    num_designs=100,
    num_samples=10,
    seed=42
)
os.makedirs(opt.out_path, exist_ok=True)

# initialize pyrosetta
pr.init(f'-ignore_unrecognized_res -ignore_zero_occupancy -mute all -corrections::beta_nov16 true -relax:default_repeats 1')

key = Keygen(opt.seed)
# set up ProteinMPNN
pmpnn = jax.jit(make_pmpnn(opt.pmpnn_params, eps=0.05))
# set up logit transform
transform = lambda center, T: transform_logits([
    toggle_transform(
        center_logits(center=center), use=opt.center == "True"),
    scale_by_temperature(temperature=T),
    forbid("C", aas.PMPNN_CODE),
    norm_logits
])
# set up sampler
pmpnn_sampler = lambda center, T: sample(
    pmpnn, logit_transform=transform(center, T))

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
    salad_data["dssp_condition"], dssp_string = si.random_dssp(
        num_aa, p=0.0, return_string=True)
    print(dssp_string)
    design = sampler(salad_params, key(), salad_data, init_prev)
    print(f"Design {idx} in {time.time() - start:.2f} s")
    # convert it to ProteinMPNN input
    pmpnn_data = data_from_protein(
        si.data.to_protein(design)).drop_aa()
    step_0_logits = pmpnn(key(), pmpnn_data)["logits"]
    center = step_0_logits.mean(axis=0)
    # ensure one TRP/W residue
    p_W = jax.nn.softmax(step_0_logits - center)[:, aas.PMPNN_CODE.index("W")]
    best_W = jnp.argmax(p_W, axis=0)
    #pmpnn_sampler = sample(pmpnn, logit_transform=transform(center, opt.temperature))
    for ids in range(opt.num_samples):
        # anneal temperature
        T = 1 - ids / opt.num_samples
        # prepare input
        pmpnn_data = pmpnn_data.drop_aa()
        # ensure one TRP/W residue (at the most probable position)
        pmpnn_data["aa"] = pmpnn_data["aa"].at[best_W].set(
            aas.PMPNN_CODE.index("W"))
        # sample all other residues
        result, log_p = pmpnn_sampler(center, T)(key(), pmpnn_data)
        # ProteinMPNN has a different amino acid order than AF2
        # translate the amino acid order
        pmpnn_data["aa"] = aas.translate(result["aa"], aas.PMPNN_CODE, aas.AF2_CODE)
        sequence = aas.decode(pmpnn_data["aa"], aas.AF2_CODE)
        af_data = AFInput.from_data(pmpnn_data).add_guess(pmpnn_data).add_pos(pmpnn_data)
        result = af2(af2_params, key(), af_data)
        plddt = result.plddt.mean()
        lddt_mean = lddt(result, pmpnn_data).mean()
        residue_sc_lddt = sc_lddt(result, pmpnn_data)
        sc_lddt_mean = residue_sc_lddt.mean()
        sc_rmsd = RMSD(atoms=["CA"])(result.to_data(), pmpnn_data)
        print(idx, ids, sequence, plddt, lddt_mean, sc_lddt_mean, sc_rmsd)
        # BoltzYAML(result.to_data(), path="tmp/boltz.yaml")
        if sc_lddt_mean > 0.7:#plddt > 0.8 and sc_rmsd < 2.0:
            pdb = PDBFile(
                result.to_data().update(plddt=residue_sc_lddt),
                path=f"{opt.out_path}/result_{idx}.pdb")
            relaxed = fastrelax(pdb, f"{os.path.dirname(opt.out_path + '/')}_relaxed/result_{idx}.pdb")
            relax_rmsd = RMSD(atoms=["CA"])(pdb.to_data(), relaxed.to_data(), mask=None)
            print(f"relaxed RMSD {relax_rmsd:.2f} A")
            break
