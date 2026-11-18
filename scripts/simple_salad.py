import os
import time
from copy import deepcopy

import haiku as hk
import jax
import jax.numpy as jnp

import salad.inference as si

import flexcraft.sequence.aa_codes as aas
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


def _init_pyrosetta():
    """Lazily import and initialize PyRosetta."""
    try:
        import pyrosetta as pr
    except ImportError as exc:
        raise ImportError(
            "PyRosetta is required for Rosetta relaxation in `scripts/simple_salad.py`. "
            "Install PyRosetta into `.venv` and preserve it across syncs with "
            "`uv sync --inexact` (or run with `uv run --no-sync` / `--relax False`)."
        ) from exc
    if not pr.rosetta.basic.was_init_called():
        pr.init(
            "-ignore_unrecognized_res -ignore_zero_occupancy -mute all "
            "-corrections::beta_nov16 true -relax:default_repeats 1"
        )


def _relax_pdb(pdb: PDBFile, out_path: str) -> PDBFile:
    _init_pyrosetta()
    try:
        from flexcraft.rosetta.relax import fastrelax
    except ImportError as exc:
        raise ImportError(
            "PyRosetta is required for Rosetta relaxation in `scripts/simple_salad.py`. "
            "Install PyRosetta into `.venv` and preserve it across syncs with "
            "`uv sync --inexact` (or run with `uv run --no-sync` / `--relax False`)."
        ) from exc
    return fastrelax(pdb, out_path)


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
        salad_params="params/salad/default_vp-200k.jax",
        pmpnn_params="params/pmpnn/v_48_030.pkl",
        af2_params="params/af/",
        af2_model="model_1_ptm",
        temperature=0.1,
        center="True",
        relax="True",
        num_designs=100,
        num_samples=10,
        seed=42,
    )
    os.makedirs(opt.out_path, exist_ok=True)

    use_relax = opt.relax == "True"
    if use_relax:
        _init_pyrosetta()

    key = Keygen(opt.seed)
    pmpnn = jax.jit(make_pmpnn(opt.pmpnn_params, eps=0.05))
    transform = lambda center, T: transform_logits(
        [
            toggle_transform(center_logits(center=center), use=opt.center == "True"),
            scale_by_temperature(temperature=T),
            forbid("C", aas.PMPNN_CODE),
            norm_logits,
        ]
    )
    pmpnn_sampler = lambda center, T: sample(pmpnn, logit_transform=transform(center, T))

    salad_config, salad_params = si.make_salad_model("default_vp", opt.salad_params)
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
    print("Starting design...")
    for idx in range(opt.num_designs):
        start = time.time()
        salad_data["dssp_condition"], dssp_string = si.random_dssp(
            num_aa, p=0.0, return_string=True
        )
        print(dssp_string)
        design = sampler(salad_params, key(), salad_data, init_prev)
        print(f"Design {idx} in {time.time() - start:.2f} s")
        pmpnn_data = data_from_protein(si.data.to_protein(design)).drop_aa()
        step_0_logits = pmpnn(key(), pmpnn_data)["logits"]
        center = step_0_logits.mean(axis=0)
        p_W = jax.nn.softmax(step_0_logits - center)[:, aas.PMPNN_CODE.index("W")]
        best_W = jnp.argmax(p_W, axis=0)
        for ids in range(opt.num_samples):
            T = 1 - ids / opt.num_samples
            pmpnn_data = pmpnn_data.drop_aa()
            pmpnn_data["aa"] = pmpnn_data["aa"].at[best_W].set(aas.PMPNN_CODE.index("W"))
            sampled, _ = pmpnn_sampler(center, T)(key(), pmpnn_data)
            pmpnn_data["aa"] = aas.translate(
                sampled["aa"], aas.PMPNN_CODE, aas.AF2_CODE
            )
            sequence = aas.decode(pmpnn_data["aa"], aas.AF2_CODE)
            af_data = (
                AFInput.from_data(pmpnn_data).add_guess(pmpnn_data).add_pos(pmpnn_data)
            )
            result = af2(af2_params, key(), af_data).unpad()
            plddt = result.mean_plddt
            lddt_mean = lddt(result, pmpnn_data).mean()
            residue_sc_lddt = sc_lddt(result, pmpnn_data)
            sc_lddt_mean = residue_sc_lddt.mean()
            pred_data = result.to_data()
            sc_rmsd = RMSD(atoms=["CA"])(pred_data, pmpnn_data)
            print(idx, ids, sequence, plddt, lddt_mean, sc_lddt_mean, sc_rmsd)
            if sc_lddt_mean > 0.7:
                pdb = PDBFile(
                    pred_data.update(plddt=residue_sc_lddt),
                    path=f"{opt.out_path}/result_{idx}.pdb",
                )
                if use_relax:
                    relaxed = _relax_pdb(
                        pdb, f"{os.path.dirname(opt.out_path + '/')}_relaxed/result_{idx}.pdb"
                    )
                    relax_rmsd = RMSD(atoms=["CA"])(
                        pdb.to_data(), relaxed.to_data(), mask=None
                    )
                    print(f"relaxed RMSD {relax_rmsd:.2f} A")
                break


if __name__ == "__main__":
    main()
