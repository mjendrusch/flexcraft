
import os
import time
from copy import deepcopy

import jax
import jax.numpy as jnp

import haiku as hk

import salad.inference as si
from salad.modules.utils.geometry import positions_to_ncacocb

import flexcraft.sequence.aa_codes as aas
from flexcraft.sequence.mpnn import make_pmpnn
from flexcraft.sequence.sample import *
from flexcraft.structure.af import *
from flexcraft.utils import Keygen, parse_options, data_from_protein
from flexcraft.files.pdb import PDBFile
from flexcraft.tools.boltz import BoltzYAML

from flexcraft.rosetta.relax import fastrelax
from flexcraft.files.csv import ScoreCSV
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
        c = config
        t = data["t_pos"][0]
        # instantiate a noise generator
        noise = si.StructureDiffusionNoise(config)
        # and a denoising model
        predict = si.StructureDiffusionPredict(config)
        # compute step in the direction of the target CA positions
        # and broadcast to entire amino acid
        target_ca = data["target_ca"]
        ca = data["pos"][:, 1]
        ca_update = (target_ca - ca)[:, None, :]
        fix_mask = (t > c.fix_threshold) * (data["dssp_condition"] < 3)
        if c.ca_mode == "gradient":
            data["pos"] = data["pos"] + t * c.ca_lr * ca_update * fix_mask[:, None, None]
        elif c.ca_mode == "fix":
            data["pos"] = jnp.where(
                fix_mask[:, None, None], data["pos"] + ca_update, data["pos"])
        # apply noise
        data.update(noise(data))
        # run model
        out, prev = predict(data, prev)
        return out, prev
    # return the pure apply function generated
    # by haiku from our step
    return step.apply


opt = parse_options(
    "Protein sequence design with custom protein MPNN",
    in_path="inputs/",
    out_path="outputs/",
    fx_path="./",
    salad_params="params/salad/default_vp-200k.jax",
    pmpnn_params="params/pmpnn/v_48_030.pkl",
    af2_params="params/af/",
    af2_model="model_1_ptm",
    ca_mode="none",
    ca_lr=0.1,
    fix_threshold=0.0,
    temperature=0.1,
    center="True",
    num_designs=100,
    num_samples=10,
    seed=42
)
os.makedirs(f"{opt.out_path}/attempts", exist_ok=True)
os.makedirs(f"{opt.out_path}/success", exist_ok=True)
os.makedirs(f"{opt.out_path}/relaxed", exist_ok=True)

# set up output files
score_keys = (
    "name", "mode",
    "sequence", "target_rmsd", "sc_rmsd", "lddt", "plddt", "sc_lddt"
)

success = ScoreCSV(
    f"{opt.out_path}/success.csv", score_keys, default="none")

# initialize pyrosetta
# pr.init(f'-ignore_unrecognized_res -ignore_zero_occupancy -mute all -corrections::beta_nov16 true -relax:default_repeats 1')

key = Keygen(opt.seed)
# set up ProteinMPNN
pmpnn = jax.jit(make_pmpnn(f"{opt.fx_path}/{opt.pmpnn_params}", eps=0.05))
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
salad_config, salad_params = si.make_salad_model("default_vp", f"{opt.fx_path}/{opt.salad_params}")
salad_config.ca_mode = opt.ca_mode
salad_config.ca_lr = opt.ca_lr
salad_config.fix_threshold = opt.fix_threshold

# initialize salad data and prev from the num_aa specification
salad_step = model_step(salad_config)

# make AF2 model
af2_params = get_model_haiku_params(
    model_name=opt.af2_model,
    data_dir=f"{opt.fx_path}/{opt.af2_params}", fuse=True)
af2_config = model_config(opt.af2_model)
af2_config.model.global_config.use_dgram = False
af2 = jax.jit(make_predict(make_af2(af2_config), num_recycle=2))

# build a sampler object for sampling
sampler = si.Sampler(salad_step, start_steps=300, out_steps=400)
# run a loop with num_design steps
print("Starting design...")
file_names = os.listdir(opt.in_path)
for name in file_names:
    if not name.endswith(".pdb"):
        continue
    base_name = ".".join(name.split(".")[:-1])
    full_path = f"{opt.in_path}/{name}"
    start_structure = PDBFile(path=full_path).to_data()
    raw_dssp = start_structure.dssp
    raw_dssp = jnp.where(raw_dssp == 0, 3, raw_dssp)
    ncacocb = positions_to_ncacocb(start_structure["atom_positions"])
    # center
    center = ncacocb[:, 1].mean(axis=0)
    ncacocb -= center
    raw_init_pos = np.concatenate((
        ncacocb,
        ncacocb[:, 1:2] + np.zeros((ncacocb.shape[0], salad_config.augment_size, 3), dtype=jnp.float32)),
        axis=1)
    for mode in ("flip", "reverse", "revflip"):
        init_pos = raw_init_pos
        if mode == "flip":
            init_pos = -raw_init_pos
            dssp = raw_dssp
        elif mode == "reverse":
            init_pos = raw_init_pos[::-1]
            dssp = raw_dssp[::-1]
        elif mode == "revflip":
            init_pos = -raw_init_pos[::-1]
            dssp = raw_dssp[::-1]
        salad_data, init_prev = si.data.from_config(
            salad_config,
            num_aa=start_structure.aa.shape[0],
            residue_index=start_structure["residue_index"],
            chain_index=start_structure["chain_index"],
            init_pos=init_pos)
        salad_data["target_ca"] = init_pos[:, 1]
        salad_data["dssp_condition"] = dssp
        done = False
        for idx in range(50):
            if done:
                break
            design = sampler(salad_params, key(), salad_data, init_prev)
            design = data_from_protein(si.data.to_protein(design))
            # design.save_pdb(f"{opt.out_path}/attempts/{base_name}_{mode}_{idx}.pdb")
            # convert it to ProteinMPNN input
            pmpnn_data = design.drop_aa()
            step_0_logits = pmpnn(key(), pmpnn_data)["logits"]
            center = step_0_logits.mean(axis=0)
            for ids in range(2):
                if done:
                    break
                # anneal temperature
                T = 0.1
                # design_info["seq_id"] = ids
                # design_info["T"] = T
                # design_info["center"] = True
                # prepare input
                pmpnn_data = pmpnn_data.drop_aa()
                # ensure one TRP/W residue (at the most probable position)
                # pmpnn_data["aa"] = pmpnn_data["aa"].at[best_W].set(
                #     aas.PMPNN_CODE.index("W"))
                # sample all other residues
                result, log_p = pmpnn_sampler(center, T)(key(), pmpnn_data)
                # ProteinMPNN has a different amino acid order than AF2
                # translate the amino acid order
                pmpnn_data["aa"] = aas.translate(result["aa"], aas.PMPNN_CODE, aas.AF2_CODE)
                sequence = pmpnn_data.to_sequence_string(code=aas.AF2_CODE)
                # design_info["sequence"] = sequence
                af_data = AFInput.from_data(pmpnn_data).add_guess(pmpnn_data).add_pos(pmpnn_data)
                result = af2(af2_params, key(), af_data)
                plddt = result.plddt.mean()
                lddt_mean = lddt(result, pmpnn_data).mean()
                residue_sc_lddt = sc_lddt(result, pmpnn_data)
                sc_lddt_mean = residue_sc_lddt.mean()
                sc_rmsd = RMSD(atoms=["CA"])(result.to_data(), pmpnn_data)
                target_rmsd = RMSD(atoms=["CA"])(result.to_data(), init_pos)
                print(idx, ids, sequence, plddt, lddt_mean, sc_lddt_mean, sc_rmsd)
                if plddt >= 0.8 and sc_rmsd < 2.0:
                    pdb = PDBFile(
                        result.to_data().update(plddt=residue_sc_lddt),
                        path=f"{opt.out_path}/success/{base_name}_{mode}_{idx}_{ids}.pdb")
                    success.write_line(dict(
                        name=base_name, mode=mode, sequence=sequence,
                        target_rmsd=target_rmsd, sc_rmsd=sc_rmsd,
                        lddt=lddt_mean, plddt=plddt, sc_lddt=sc_lddt_mean
                    ))
                    done = True
