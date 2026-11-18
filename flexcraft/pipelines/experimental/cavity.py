import os
import random
import time
from copy import deepcopy

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

import salad.inference as si

import flexcraft.sequence.aa_codes as aas
from flexcraft.files.csv import ScoreCSV
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
from flexcraft.structure.metrics import RMSD, sc_lddt
from flexcraft.utils import Keygen, data_from_protein, parse_options


def model_step(config, ligand_size: int):
    """Build a pure Haiku apply function for one `salad` cavity denoising step."""
    config = deepcopy(config)
    config.eval = True

    @hk.transform
    def step(data, prev):
        noise = si.StructureDiffusionNoise(config)
        predict = si.StructureDiffusionPredict(config)
        data["pos"] -= data["pos"][:, 1].mean(axis=0)
        ligand_pos = data["pos"][-ligand_size:]
        ligand_pos = ligand_pos - ligand_pos[:, 1].mean(axis=0)
        data["pos"] = data["pos"].at[-ligand_size:].set(ligand_pos)
        data.update(noise(data))
        out, prev = predict(data, prev)
        return out, prev

    return step.apply


def main():
    opt = parse_options(
        "Protein sequence design with custom protein MPNN",
        num_aa=72,
        out_path="outputs/",
        salad_params="params/salad/default_vp_scaled-200k.jax",
        pmpnn_params="params/pmpnn/v_48_030.pkl",
        af2_params="params/af/",
        af2_model="model_1_ptm",
        ligand_size=3,
        ligand_cyclic="False",
        num_blocks=10,
        num_designs=100,
        num_samples=10,
        std_buffer=2.0,
        strict="False",
        write_attempts="False",
        allow_redesign="False",
        random_dssp="True",
        seed=42,
    )

    def cloud_std_default(n_aa):
        minval = n_aa**0.4
        return minval + np.random.rand() * opt.std_buffer

    os.makedirs(f"{opt.out_path}/attempts/", exist_ok=True)
    os.makedirs(f"{opt.out_path}/success/", exist_ok=True)
    os.makedirs(f"{opt.out_path}/success_ligand/", exist_ok=True)

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
    salad_config.cyclic = True
    salad_step = model_step(salad_config, opt.ligand_size)

    af2_params = get_model_haiku_params(
        model_name=opt.af2_model, data_dir=opt.af2_params, fuse=True
    )
    af2_config = model_config(opt.af2_model)
    af2_config.model.global_config.use_dgram = False
    af2 = jax.jit(
        make_predict(
            make_af2(af2_config, use_multimer="multimer" in opt.af2_model),
            num_recycle=4,
        )
    )

    sampler = si.Sampler(salad_step, out_steps=400)

    score_keys = (
        "attempt",
        "seq_id",
        "T",
        "center",
        "sequence",
        "plddt",
        "monomer_plddt",
        "sc_rmsd",
        "monomer_rmsd",
        "ipae",
    )
    success = ScoreCSV(f"{opt.out_path}/success.csv", score_keys, default="none")

    print("Starting design...")
    success_count = 0
    idx = 0
    while success_count < opt.num_designs:
        ligand = str(opt.ligand_size)
        if opt.ligand_cyclic == "True":
            ligand = "c" + ligand
        num_aa_string = f"{opt.num_aa}:{ligand}"
        print(num_aa_string)
        num_aa, resi, chain, is_cyclic, cyclic_mask = si.parse_num_aa(num_aa_string)
        salad_data, init_prev = si.data.from_config(
            salad_config,
            num_aa=num_aa,
            residue_index=resi,
            chain_index=chain,
            cyclic_mask=cyclic_mask,
        )
        salad_data["aa_condition"] = jnp.concatenate(
            (
                jnp.full((opt.num_aa,), 20, dtype=jnp.int32),
                jax.random.randint(key(), (opt.ligand_size,), 0, 20, dtype=jnp.int32),
            ),
            axis=0,
        )
        if opt.allow_redesign == "True":
            salad_data["aa_condition"] = jnp.full_like(salad_data["aa_condition"], 20)
        salad_data["dssp_condition"], dssp_string = si.random_dssp(
            opt.num_aa, p=0.0, p_keep_loop=1.0, return_string=True
        )
        if opt.random_dssp == "True":
            print(dssp_string)
        else:
            salad_data["dssp_condition"] = jnp.full_like(
                salad_data["dssp_condition"], 3
            )
        salad_data["dssp_condition"] = jnp.concatenate(
            (salad_data["dssp_condition"], jnp.full((opt.ligand_size,), 3)),
            axis=0,
        )
        salad_data["cloud_std"] = cloud_std_default(num_aa)
        attempt_done = False
        for attempt in range(5):
            if attempt_done:
                break
            salad_data["cloud_std"] = cloud_std_default(num_aa)
            start = time.time()
            design = sampler(salad_params, key(), salad_data, init_prev)
            design = data_from_protein(si.data.to_protein(design))
            if opt.write_attempts == "True":
                design.save_pdb(f"{opt.out_path}/attempts/result_{idx}_{attempt}.pdb")
            print(f"Design {idx} in {time.time() - start:.2f} s")
            target_aa = aas.translate(
                salad_data["aa_condition"], aas.AF2_CODE, aas.PMPNN_CODE
            )
            pmpnn_data = design.update(aa=target_aa)
            step_0_logits = pmpnn(key(), pmpnn_data)["logits"]
            center = step_0_logits.mean(axis=0)
            for ids in range(opt.num_samples):
                data_item = dict()
                T = random.choice([0.01, 0.1])
                do_center = random.choice([True, False])
                data_item["attempt"] = idx
                data_item["seq_id"] = ids
                data_item["T"] = T
                data_item["center"] = do_center
                pmpnn_data = pmpnn_data.update(aa=target_aa)
                sampled, _ = pmpnn_sampler(center, T, do_center)(key(), pmpnn_data)
                pmpnn_data["aa"] = aas.translate(
                    sampled["aa"], aas.PMPNN_CODE, aas.AF2_CODE
                )
                sequence = pmpnn_data.to_sequence_string().split(":")[0]
                print(sequence)
                data_item["sequence"] = sequence
                af_data = AFInput.from_data(pmpnn_data)
                result: AFResult = af2(af2_params, key(), af_data).unpad()
                monomer_design = pmpnn_data[pmpnn_data["chain_index"] == 0]
                monomer_data = AFInput.from_data(monomer_design)
                monomer_result: AFResult = af2(af2_params, key(), monomer_data).unpad()
                data_item["monomer_plddt"] = monomer_result.mean_plddt
                data_item["monomer_rmsd"] = RMSD()(
                    monomer_result.to_data(), pmpnn_data[: opt.num_aa]
                )
                data_item["plddt"] = result.mean_plddt
                data_item["ipae"] = result.ipae
                residue_sc_lddt = sc_lddt(result, pmpnn_data)
                data_item["sc_rmsd"] = RMSD()(result.to_data(), pmpnn_data)
                data_item["sc_lddt"] = np.array(residue_sc_lddt)
                print(
                    f"attempt {attempt} {ids}:",
                    data_item["plddt"],
                    data_item["sc_rmsd"],
                    data_item["monomer_plddt"],
                    data_item["monomer_rmsd"],
                )
                if (data_item["monomer_plddt"] > 0.8) * (
                    data_item["monomer_rmsd"] < 2.0
                ):
                    if opt.strict == "True":
                        if not (
                            (data_item["sc_rmsd"] < 2.0) * (data_item["ipae"] < 0.3)
                        ):
                            continue
                    PDBFile(
                        monomer_result.to_data().update(plddt=monomer_result.plddt),
                        path=f"{opt.out_path}/success/result_{idx}_{ids}.pdb",
                    )
                    PDBFile(
                        result.to_data().update(plddt=result.plddt),
                        path=f"{opt.out_path}/success_ligand/result_{idx}_{ids}.pdb",
                    )
                    success.write_line(data_item)
                    attempt_done = True
                    success_count += 1
                    break
        idx += 1


if __name__ == "__main__":
    main()
