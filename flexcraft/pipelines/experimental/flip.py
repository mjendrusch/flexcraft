import os
from copy import deepcopy

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

import salad.inference as si
from salad.modules.utils.geometry import positions_to_ncacocb

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
    get_model_haiku_params,
    make_af2,
    make_predict,
    model_config,
)
from flexcraft.structure.metrics import RMSD, lddt, sc_lddt
from flexcraft.utils import Keygen, data_from_protein, parse_options


def model_step(config):
    """Build a pure Haiku apply function for one `salad` flip/reverse denoising step."""
    config = deepcopy(config)
    config.eval = True

    @hk.transform
    def step(data, prev):
        c = config
        t = data["t_pos"][0]
        noise = si.StructureDiffusionNoise(config)
        predict = si.StructureDiffusionPredict(config)
        target_ca = data["target_ca"]
        ca = data["pos"][:, 1]
        ca_update = (target_ca - ca)[:, None, :]
        fix_mask = (t > c.fix_threshold) * (data["dssp_condition"] < 3)
        if c.ca_mode == "gradient":
            data["pos"] = (
                data["pos"] + t * c.ca_lr * ca_update * fix_mask[:, None, None]
            )
        elif c.ca_mode == "fix":
            data["pos"] = jnp.where(
                fix_mask[:, None, None], data["pos"] + ca_update, data["pos"]
            )
        data.update(noise(data))
        out, prev = predict(data, prev)
        return out, prev

    return step.apply


def main():
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
        seed=42,
    )
    os.makedirs(f"{opt.out_path}/attempts", exist_ok=True)
    os.makedirs(f"{opt.out_path}/success", exist_ok=True)
    os.makedirs(f"{opt.out_path}/relaxed", exist_ok=True)

    score_keys = (
        "name",
        "mode",
        "sequence",
        "target_rmsd",
        "sc_rmsd",
        "lddt",
        "plddt",
        "sc_lddt",
    )
    success = ScoreCSV(f"{opt.out_path}/success.csv", score_keys, default="none")

    key = Keygen(opt.seed)
    pmpnn = jax.jit(make_pmpnn(f"{opt.fx_path}/{opt.pmpnn_params}", eps=0.05))
    transform = lambda center, T: transform_logits(
        [
            toggle_transform(center_logits(center=center), use=opt.center == "True"),
            scale_by_temperature(temperature=T),
            forbid("C", aas.PMPNN_CODE),
            norm_logits,
        ]
    )
    pmpnn_sampler = lambda center, T: sample(pmpnn, logit_transform=transform(center, T))

    salad_config, salad_params = si.make_salad_model(
        "default_vp", f"{opt.fx_path}/{opt.salad_params}"
    )
    salad_config.ca_mode = opt.ca_mode
    salad_config.ca_lr = opt.ca_lr
    salad_config.fix_threshold = opt.fix_threshold
    salad_step = model_step(salad_config)

    af2_params = get_model_haiku_params(
        model_name=opt.af2_model,
        data_dir=f"{opt.fx_path}/{opt.af2_params}",
        fuse=True,
    )
    af2_config = model_config(opt.af2_model)
    af2_config.model.global_config.use_dgram = False
    af2 = jax.jit(make_predict(make_af2(af2_config), num_recycle=2))

    sampler = si.Sampler(salad_step, start_steps=300, out_steps=400)
    print("Starting design...")
    file_names = os.listdir(opt.in_path)
    for name in file_names:
        if not name.endswith(".pdb"):
            continue
        base_name = ".".join(name.split(".")[:-1])
        full_path = f"{opt.in_path}/{name}"
        start_structure = PDBFile(path=full_path).to_data().unpad()
        raw_dssp = start_structure.dssp
        raw_dssp = jnp.where(raw_dssp == 0, 3, raw_dssp)
        ncacocb = np.array(positions_to_ncacocb(start_structure["atom_positions"]))
        center = ncacocb[:, 1].mean(axis=0)
        ncacocb -= center
        raw_init_pos = np.concatenate(
            (
                ncacocb,
                ncacocb[:, 1:2]
                + np.zeros(
                    (ncacocb.shape[0], salad_config.augment_size, 3), dtype=jnp.float32
                ),
            ),
            axis=1,
        )
        for mode in ("flip", "reverse", "revflip"):
            init_pos = raw_init_pos
            dssp = raw_dssp
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
                init_pos=init_pos,
            )
            salad_data["target_ca"] = init_pos[:, 1]
            salad_data["dssp_condition"] = dssp
            done = False
            for idx in range(50):
                if done:
                    break
                design = sampler(salad_params, key(), salad_data, init_prev)
                design = data_from_protein(si.data.to_protein(design))
                pmpnn_data = design.drop_aa()
                step_0_logits = pmpnn(key(), pmpnn_data)["logits"]
                center = step_0_logits.mean(axis=0)
                for ids in range(2):
                    if done:
                        break
                    T = opt.temperature
                    pmpnn_data = pmpnn_data.drop_aa()
                    sampled, _ = pmpnn_sampler(center, T)(key(), pmpnn_data)
                    pmpnn_data["aa"] = aas.translate(
                        sampled["aa"], aas.PMPNN_CODE, aas.AF2_CODE
                    )
                    sequence = pmpnn_data.to_sequence_string(code=aas.AF2_CODE)
                    af_data = (
                        AFInput.from_data(pmpnn_data)
                        .add_guess(pmpnn_data)
                        .add_pos(pmpnn_data)
                    )
                    result = af2(af2_params, key(), af_data).unpad()
                    plddt = result.mean_plddt
                    lddt_mean = lddt(result, pmpnn_data).mean()
                    residue_sc_lddt = sc_lddt(result, pmpnn_data)
                    sc_lddt_mean = residue_sc_lddt.mean()
                    pred_data = result.to_data()
                    sc_rmsd = RMSD(atoms=["CA"])(pred_data, pmpnn_data)
                    target_rmsd = RMSD(atoms=["CA"])(pred_data, init_pos)
                    print(idx, ids, sequence, plddt, lddt_mean, sc_lddt_mean, sc_rmsd)
                    if plddt >= 0.8 and sc_rmsd < 2.0:
                        PDBFile(
                            pred_data.update(plddt=residue_sc_lddt),
                            path=f"{opt.out_path}/success/{base_name}_{mode}_{idx}_{ids}.pdb",
                        )
                        success.write_line(
                            dict(
                                name=base_name,
                                mode=mode,
                                sequence=sequence,
                                target_rmsd=target_rmsd,
                                sc_rmsd=sc_rmsd,
                                lddt=lddt_mean,
                                plddt=plddt,
                                sc_lddt=sc_lddt_mean,
                            )
                        )
                        done = True


if __name__ == "__main__":
    main()
