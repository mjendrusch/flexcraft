
import os
import time
from copy import deepcopy, copy

import random
import jax
import jax.numpy as jnp
import numpy as np

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

from flexcraft.rosetta.relax import fastrelax, pair_energies
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
    salad_params="params/salad/default_vp_scaled-200k.jax",
    pmpnn_params="params/pmpnn/v_48_030.pkl",
    af2_params="params/af/",
    af2_model="model_1_ptm",
    num_blocks=10,
    num_designs=100,
    num_samples=20,
    std_buffer=3.0,
    strict="False",
    seed=42
)

def cloud_std_default(num_aa):
    minval = num_aa ** 0.4
    return minval + np.random.rand() * opt.std_buffer

os.makedirs(f"{opt.out_path}/predictions/", exist_ok=True)
os.makedirs(f"{opt.out_path}/relaxed/", exist_ok=True)
os.makedirs(f"{opt.out_path}/data/", exist_ok=True)

# initialize pyrosetta
pr.init(f'-ignore_unrecognized_res -ignore_zero_occupancy -mute all -corrections::beta_nov16 true -relax:default_repeats 1')

key = Keygen(opt.seed)
# set up ProteinMPNN
pmpnn = jax.jit(make_pmpnn(opt.pmpnn_params, eps=0.05))
# set up logit transform
transform = lambda center, T, do_center: transform_logits([
    toggle_transform(
        center_logits(center=center), use=do_center),
    scale_by_temperature(temperature=T),
    forbid("C", aas.PMPNN_CODE),
    norm_logits
])
# set up sampler
pmpnn_sampler = lambda center, T, do_center: sample(
    pmpnn, logit_transform=transform(center, T, do_center))

# make salad model
salad_config, salad_params = si.make_salad_model("default_vp_scaled", opt.salad_params)

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
resampler = copy(sampler)
resampler.start_steps = 300
# run a loop with num_design steps
print("Starting design...")

for blk in range(opt.num_blocks):
    dataset = dict(
        pos=[],
        aa=[],
        num_contacts=[],
        local=[],
        # pair=[],
        # E_pair=[],
        dssp=[],
        sc_lddt=[],
        # best_aa=[],
    )
    for idx in range(opt.num_designs):
        # generate a structure in each step
        start = time.time()
        p_keep_loop = 1.0#random.choice([1.0, 0.5, 0.0])
        salad_data["dssp_condition"], dssp_string = si.random_dssp(
            num_aa, p=0.0, p_keep_loop=p_keep_loop, return_string=True)
        print("DSSP target:", dssp_string)
        attempt_done = False
        # look for best sequence
        best_val = 0.0
        best_item = None
        best_result = None
        for attempt in range(3):
            if attempt_done:
                break
            salad_data["cloud_std"] = cloud_std_default(num_aa)
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
            for ids in range(opt.num_samples):
                data_item = dict()
                # sample settings
                T = random.choice([0.01, 0.1, 0.3, 0.5])
                do_center = random.choice([True, False])
                # prepare input
                pmpnn_data = pmpnn_data.drop_aa()
                # sample all other residues
                result, log_p = pmpnn_sampler(center, T, do_center)(key(), pmpnn_data)
                # ProteinMPNN has a different amino acid order than AF2
                # translate the amino acid order
                pmpnn_data["aa"] = aas.translate(result["aa"], aas.PMPNN_CODE, aas.AF2_CODE)
                sequence = aas.decode(pmpnn_data["aa"], aas.AF2_CODE)
                af_data = AFInput.from_data(pmpnn_data)
                result: AFResult = af2(af2_params, key(), af_data)
                # properties to save for training dataset
                result_data: DesignData = result.to_data()
                pos = result_data["atom_positions"]
                cb = positions_to_ncacocb(pos)[:, -1]
                dist = np.linalg.norm(cb[:, None] - cb[None, :], axis=-1)
                resi = result_data["residue_index"]
                contact_mask = abs(resi[:, None] - resi[None, :]) >= 10
                contacts = (dist < 8.0) * contact_mask > 0
                # number of amino acid contacts
                num_contacts = contacts.astype(np.int32).sum(axis=1)
                # AF2 amino acid features
                local_features = np.array(result.local)
                pair_features = np.array(result.pair)
                # 3-state secondary structure
                dssp = np.array(result_data.dssp)
                plddt = np.array(result.plddt)
                mean_plddt = plddt.mean()
                lddt_mean = lddt(result, pmpnn_data).mean()
                # scLDDT
                residue_sc_lddt = sc_lddt(result, pmpnn_data)
                sc_lddt_mean = residue_sc_lddt.mean()
                sc_rmsd = RMSD(atoms=["CA"])(result.to_data(), pmpnn_data)
                print(idx, ids, sequence, plddt.mean(), lddt_mean, sc_lddt_mean, sc_rmsd)
                data_item["aa"] = np.array(pmpnn_data["aa"])
                data_item["pos"] = np.array(pos)
                data_item["num_contacts"] = num_contacts
                data_item["local"] = local_features
                # data_item["pair"] = pair_features
                data_item["dssp"] = dssp
                data_item["sc_lddt"] = np.array(residue_sc_lddt)
                if sc_lddt_mean > best_val or best_item is None:
                    best_val = sc_lddt_mean
                    best_item = data_item
                    best_result = result
                success = sc_lddt_mean > 0.7
                if opt.strict == "True":
                    success = (mean_plddt > 0.8) * (sc_rmsd < 2.0)
                if success:
                    pdb = PDBFile(
                        result.to_data().update(plddt=residue_sc_lddt),
                        path=f"{opt.out_path}/predictions/result_{blk}_{idx}.pdb")
                    for k, v in data_item.items():
                        dataset[k].append(v)
                    attempt_done = True
                    break
        if not attempt_done:
            for k, v in best_item.items():
                dataset[k].append(v)
            pdb = PDBFile(
                best_result.to_data().update(plddt=best_item["sc_lddt"]),
                path=f"{opt.out_path}/predictions/result_{blk}_{idx}.pdb")
            # relaxed, pose = fastrelax(pdb, f"{os.path.dirname(opt.out_path + '/')}/relaxed/result_{idx}_{ids}.pdb", return_pose=True)
            # energies = pair_energies(relaxed)
            # dataset["E_pair"].append(np.array(energies["total_score"]))
        # dataset["best_aa"].append(opt.num_samples * [np.array(best_aa)])
    dataset = {k: np.stack(v, axis=0) for k, v in dataset.items()}
    np.savez_compressed(f"{opt.out_path}/data/block_{blk}.npz", **dataset)
