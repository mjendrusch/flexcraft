import os
import random
import time
from copy import copy, deepcopy

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

import salad.inference as si
from salad.modules.utils.geometry import positions_to_ncacocb

import flexcraft.sequence.aa_codes as aas
from flexcraft.data.data import DesignData
from flexcraft.files.pdb import PDBFile
from flexcraft.sequence.mpnn import make_pmpnn
from flexcraft.sequence.sample import (
    center_logits,
    forbid,
    norm_logits,
    sample,
    scale_by_temperature,
    toggle_transform,
    transform_logits,
)
from flexcraft.structure.af import (
    AFInput,
    AFResult,
    get_model_haiku_params,
    make_af2,
    make_predict,
    model_config,
)
from flexcraft.structure.metrics import RMSD, lddt, sc_lddt
from flexcraft.utils import Keygen, data_from_protein, parse_options


def model_step(config):
    """Build a pure Haiku apply function for one `salad` denoising step."""
    config = deepcopy(config)
    config.eval = True

    @hk.transform
    def step(data, prev):
        noise = si.StructureDiffusionNoise(config)
        predict = si.StructureDiffusionPredict(config)
        data.update(noise(data))
        out, prev = predict(data, prev)
        return out, prev

    return step.apply


def main():
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
        seed=42,
    )

    def cloud_std_default(n_aa):
        minval = n_aa**0.4
        return minval + np.random.rand() * opt.std_buffer

    os.makedirs(f"{opt.out_path}/predictions/", exist_ok=True)
    os.makedirs(f"{opt.out_path}/relaxed/", exist_ok=True)
    os.makedirs(f"{opt.out_path}/data/", exist_ok=True)

    key = Keygen(opt.seed)
    pmpnn = jax.jit(make_pmpnn(opt.pmpnn_params, eps=0.05))
    transform = lambda center, T, do_center: transform_logits(
        [
            toggle_transform(center_logits(center=center), use=do_center),
            scale_by_temperature(temperature=T),
            forbid("C", aas.PMPNN_CODE),
            norm_logits,
        ]
    )
    pmpnn_sampler = lambda center, T, do_center: sample(
        pmpnn, logit_transform=transform(center, T, do_center)
    )

    salad_config, salad_params = si.make_salad_model(
        "default_vp_scaled", opt.salad_params
    )
    num_aa, resi, chain, is_cyclic, cyclic_mask = si.parse_num_aa(opt.num_aa)
    salad_data, init_prev = si.data.from_config(
        salad_config,
        num_aa=num_aa,
        residue_index=resi,
        chain_index=chain,
        cyclic_mask=cyclic_mask,
    )
    salad_step = model_step(salad_config)

    af2_params = get_model_haiku_params(
        model_name=opt.af2_model, data_dir=opt.af2_params, fuse=True
    )
    af2_config = model_config(opt.af2_model)
    af2_config.model.global_config.use_dgram = False
    af2 = jax.jit(make_predict(make_af2(af2_config), num_recycle=2))

    sampler = si.Sampler(salad_step, out_steps=400)
    resampler = copy(sampler)
    resampler.start_steps = 300
    print("Starting design...")

    for blk in range(opt.num_blocks):
        dataset = dict(
            pos=[],
            aa=[],
            num_contacts=[],
            local=[],
            dssp=[],
            sc_lddt=[],
        )
        for idx in range(opt.num_designs):
            start = time.time()
            salad_data["dssp_condition"], dssp_string = si.random_dssp(
                num_aa, p=0.0, p_keep_loop=1.0, return_string=True
            )
            print("DSSP target:", dssp_string)
            attempt_done = False
            best_val = 0.0
            best_item = None
            best_result = None
            for _ in range(3):
                if attempt_done:
                    break
                salad_data["cloud_std"] = cloud_std_default(num_aa)
                design = sampler(salad_params, key(), salad_data, init_prev)
                print(f"Design {idx} in {time.time() - start:.2f} s")
                pmpnn_data = data_from_protein(si.data.to_protein(design)).drop_aa()
                step_0_logits = pmpnn(key(), pmpnn_data)["logits"]
                center = step_0_logits.mean(axis=0)
                for ids in range(opt.num_samples):
                    data_item = dict()
                    T = random.choice([0.01, 0.1, 0.3, 0.5])
                    do_center = random.choice([True, False])
                    pmpnn_data = pmpnn_data.drop_aa()
                    sampled, _ = pmpnn_sampler(center, T, do_center)(key(), pmpnn_data)
                    pmpnn_data["aa"] = aas.translate(
                        sampled["aa"], aas.PMPNN_CODE, aas.AF2_CODE
                    )
                    sequence = aas.decode(pmpnn_data["aa"], aas.AF2_CODE)
                    af_data = AFInput.from_data(pmpnn_data)
                    result: AFResult = af2(af2_params, key(), af_data).unpad()
                    result_data: DesignData = result.to_data().unpad()
                    pos = result_data["atom_positions"]
                    cb = positions_to_ncacocb(pos)[:, -1]
                    dist = np.linalg.norm(cb[:, None] - cb[None, :], axis=-1)
                    resi_idx = result_data["residue_index"]
                    contact_mask = abs(resi_idx[:, None] - resi_idx[None, :]) >= 10
                    contacts = (dist < 8.0) * contact_mask > 0
                    num_contacts = contacts.astype(np.int32).sum(axis=1)
                    local_features = np.array(result.local)
                    dssp = np.array(result_data.dssp)
                    mean_plddt = float(result.mean_plddt)
                    lddt_mean = float(lddt(result, pmpnn_data).mean())
                    residue_sc_lddt = sc_lddt(result, pmpnn_data)
                    sc_lddt_mean = float(residue_sc_lddt.mean())
                    sc_rmsd = float(RMSD(atoms=["CA"])(result_data, pmpnn_data))
                    print(
                        idx,
                        ids,
                        sequence,
                        mean_plddt,
                        lddt_mean,
                        sc_lddt_mean,
                        sc_rmsd,
                    )
                    data_item["aa"] = np.array(pmpnn_data["aa"])
                    data_item["pos"] = np.array(pos)
                    data_item["num_contacts"] = num_contacts
                    data_item["local"] = local_features
                    data_item["dssp"] = dssp
                    data_item["sc_lddt"] = np.array(residue_sc_lddt)
                    if sc_lddt_mean > best_val or best_item is None:
                        best_val = sc_lddt_mean
                        best_item = data_item
                        best_result = result
                    is_success = sc_lddt_mean > 0.7
                    if opt.strict == "True":
                        is_success = (mean_plddt > 0.8) and (sc_rmsd < 2.0)
                    if is_success:
                        PDBFile(
                            result_data.update(plddt=residue_sc_lddt),
                            path=f"{opt.out_path}/predictions/result_{blk}_{idx}.pdb",
                        )
                        for k, v in data_item.items():
                            dataset[k].append(v)
                        attempt_done = True
                        break
            if not attempt_done:
                for k, v in best_item.items():
                    dataset[k].append(v)
                PDBFile(
                    best_result.to_data().unpad().update(plddt=best_item["sc_lddt"]),
                    path=f"{opt.out_path}/predictions/result_{blk}_{idx}.pdb",
                )
        dataset = {k: np.stack(v, axis=0) for k, v in dataset.items()}
        np.savez_compressed(f"{opt.out_path}/data/block_{blk}.npz", **dataset)


if __name__ == "__main__":
    main()
