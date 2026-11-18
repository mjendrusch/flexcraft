import os
from copy import deepcopy
import time

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

import salad.inference as si
from salad.modules.utils.geometry import index_count

import flexcraft.sequence.aa_codes as aas
from flexcraft.data.data import DesignData
from flexcraft.files.csv import ScoreCSV
from flexcraft.sequence.mpnn import make_pmpnn
from flexcraft.sequence.sample import (
    forbid,
    norm_logits,
    sample,
    scale_by_temperature,
    transform_logits,
)
from flexcraft.structure.af import (
    AFInput,
    get_model_haiku_params,
    make_af2,
    make_predict,
    model_config,
)
from flexcraft.structure.metrics import RMSD, lddt
from flexcraft.utils import Keygen, data_from_salad, parse_options


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


def init_tag_pair(init_data, tag_sequence, num_disambiguating=10, num_padding=5):
    num_aa = init_data["residue_index"].shape[0]
    tag_code = aas.encode(tag_sequence, aas.AF2_CODE)
    num_tag = tag_code.shape[0]
    tag_dssp = np.random.randint(1, 3)
    tag_start = np.random.randint(
        num_disambiguating + num_padding, num_aa - num_padding - num_tag
    )
    tag_end = tag_start + num_tag
    non_tag_mask = np.ones_like(init_data["residue_index"])
    non_tag_mask[tag_start:tag_end] = 0
    non_tag_mask = non_tag_mask > 0
    tag_data = {k: v.copy() for k, v in init_data.items()}
    tag_data["aa_condition"] = np.full_like(init_data["residue_index"], 20)
    tag_data["aa_condition"][tag_start:tag_end] = tag_code
    exposed_data = {k: v.copy() for k, v in tag_data.items()}
    exposed_data["dssp_condition"] = np.full_like(init_data["residue_index"], 3)
    exposed_data["dssp_condition"][tag_start:tag_end] = 0
    hidden_data = {k: v.copy() for k, v in tag_data.items()}
    hidden_data["dssp_condition"] = np.full_like(init_data["residue_index"], 3)
    hidden_data["dssp_condition"][tag_start:tag_end] = tag_dssp
    # tie all non-disambiguating amino acids
    tie_index_exposed = np.arange(num_aa, dtype=np.int32)
    tie_index_hidden = tie_index_exposed.copy()
    tie_index_hidden[:num_disambiguating] = np.arange(
        num_aa, num_aa + num_disambiguating, dtype=np.int32
    )
    tie_index = np.concatenate((tie_index_exposed, tie_index_hidden), axis=0)
    tie_weight = 1 / jnp.maximum(
        index_count(tie_index, jnp.ones_like(tie_index, dtype=jnp.bool_)), 1
    )
    paired_tie_info = dict(tie_index=tie_index, tie_weight=tie_weight)
    unpaired_tie_info = dict(
        tie_index=np.arange(2 * num_aa, dtype=np.int32),
        tie_weight=np.ones((2 * num_aa,), dtype=np.float32),
    )

    return dict(
        exposed=exposed_data,
        hidden=hidden_data,
        non_tag_mask=non_tag_mask,
        tie_info=paired_tie_info,
        unpaired_tie_info=unpaired_tie_info,
        pmpnn_aa=aas.translate(
            np.concatenate(2 * [tag_data["aa_condition"]], axis=-1),
            aas.AF2_CODE,
            aas.PMPNN_CODE,
        ),
    )


def mutate(sampler: si.Sampler, params, key):
    def _run(exposed, hidden, init_prev):
        start_steps = jax.random.randint(key(), (), 0, 4) * 100
        exposed = sampler(params, key(), exposed, init_prev, start_steps=start_steps)
        hidden = sampler(params, key(), hidden, init_prev, start_steps=start_steps)
        return exposed, hidden

    return _run


def evaluate(af2, af2_params, pmpnn_sampler, key=None, temperature: float = 0.1):
    if key is None:
        key = Keygen(0)

    def _run(exposed, hidden, tag_pair_info):
        exposed_data = data_from_salad(exposed)
        hidden_data = data_from_salad(hidden)
        non_tag_mask = tag_pair_info["non_tag_mask"]
        num_non_tag = jnp.maximum(1, jnp.asarray(non_tag_mask).sum())
        pmpnn_untied = DesignData.concatenate(
            [exposed_data, hidden_data], sep_batch=True
        )
        pmpnn_untied = pmpnn_untied.update(
            aa=tag_pair_info["pmpnn_aa"],
            tie_index=tag_pair_info["unpaired_tie_info"]["tie_index"],
            tie_weight=tag_pair_info["unpaired_tie_info"]["tie_weight"],
            tie_weights=tag_pair_info["unpaired_tie_info"]["tie_weight"],
        )
        pmpnn_tied = pmpnn_untied.update(
            aa=tag_pair_info["pmpnn_aa"],
            tie_index=tag_pair_info["tie_info"]["tie_index"],
            tie_weight=tag_pair_info["tie_info"]["tie_weight"],
            tie_weights=tag_pair_info["tie_info"]["tie_weight"],
        )
        result_tied, _ = pmpnn_sampler(None, temperature)(key(), pmpnn_tied)
        pmpnn_tied["aa"] = aas.translate(
            result_tied["aa"], aas.PMPNN_CODE, aas.AF2_CODE
        )
        pmpnn_result = pmpnn_tied.split(pmpnn_tied["batch_index"])

        modes = dict()
        for mode_name, pmpnn_out in zip(("exposed", "hidden"), pmpnn_result):
            af_input = AFInput.from_data(pmpnn_out).add_guess(pmpnn_out)
            mode_data = dict()
            af_res = af2(af2_params, key(), af_input).unpad()
            mode_data["result"] = af_res
            mode_data["sequence"] = pmpnn_out.to_sequence_string()
            lddt_val = lddt(af_res, pmpnn_out)
            plddt_val = af_res.plddt
            mode_data["lddt"] = (lddt_val * non_tag_mask).sum() / num_non_tag
            mode_data["plddt"] = (plddt_val * non_tag_mask).sum() / num_non_tag
            mode_data["sc_lddt"] = (
                (lddt_val * plddt_val * non_tag_mask).sum() / num_non_tag
            )
            mode_data["sc_rmsd"] = RMSD()(
                af_res.to_data(),
                pmpnn_out,
                weight=non_tag_mask,
                eval_mask=non_tag_mask,
            )
            mode_data["success"] = bool(
                (mode_data["plddt"] > 0.75) and (mode_data["sc_rmsd"] < 3.0)
            )
            modes[mode_name] = mode_data
        score = modes["exposed"]["sc_lddt"] * modes["hidden"]["sc_lddt"]
        is_success = bool(modes["exposed"]["success"] and modes["hidden"]["success"])
        return score, is_success, modes

    return _run


def search(
    salad_params,
    salad_sampler,
    af2,
    af2_params,
    pmpnn_sampler,
    key,
    population_size=4,
    multiplicity=4,
    num_steps=4,
    temperature: float = 0.1,
):
    _eval = evaluate(af2, af2_params, pmpnn_sampler, key=key, temperature=temperature)
    _mut = mutate(salad_sampler, salad_params, key)

    def _init(exposed, hidden, init_prev, tag_pair_info):
        population = [
            dict(
                score=0.0,
                success=False,
                exposed=salad_sampler(salad_params, key(), exposed, init_prev),
                hidden=salad_sampler(salad_params, key(), hidden, init_prev),
                aux=None,
            )
            for _ in range(population_size * multiplicity)
        ]
        for item in population:
            score, is_success, aux = _eval(
                item["exposed"], item["hidden"], tag_pair_info
            )
            item["score"] = score
            item["success"] = is_success
            item["aux"] = aux
        population = sorted(
            population, key=lambda x: float(x["score"]), reverse=True
        )[:population_size]
        return population

    def _run(exposed, hidden, init_prev, tag_pair_info):
        population = _init(exposed, hidden, init_prev, tag_pair_info)
        for _ in range(num_steps):
            candidates = []
            for item in population:
                for _ in range(multiplicity):
                    mut_exposed, mut_hidden = _mut(
                        item["exposed"], item["hidden"], init_prev
                    )
                    score, is_success, aux = _eval(
                        mut_exposed, mut_hidden, tag_pair_info
                    )
                    candidates.append(
                        dict(
                            score=score,
                            success=is_success,
                            exposed=mut_exposed,
                            hidden=mut_hidden,
                            aux=aux,
                        )
                    )
            population = sorted(
                candidates, key=lambda x: float(x["score"]), reverse=True
            )[:population_size]
        return population

    return _run


def main():
    opt = parse_options(
        "Protein sequence design with custom protein MPNN",
        num_aa="100",
        num_disambiguating=10,
        num_padding=5,
        tag="YPYDVPDYA",
        out_path="outputs/",
        salad_params="params/salad/default_vp-200k.jax",
        pmpnn_params="params/pmpnn/v_48_030.pkl",
        af2_params="params/af/",
        af2_model="model_1_ptm",
        temperature=0.1,
        center="True",
        num_designs=100,
        num_samples=10,
        seed=42,
    )
    os.makedirs(f"{opt.out_path}/attempts", exist_ok=True)
    os.makedirs(f"{opt.out_path}/success", exist_ok=True)

    score_keys = (
        "attempt",
        "seq_id",
        "T",
        "center",
        "sequence",
        "sc_rmsd",
        "relax_rmsd",
        "lddt",
        "plddt",
        "sc_lddt",
        "success",
    )

    _ = ScoreCSV(f"{opt.out_path}/success.csv", score_keys, default="none")
    _ = ScoreCSV(f"{opt.out_path}/all.csv", score_keys, default="none")

    key = Keygen(opt.seed)
    pmpnn = jax.jit(make_pmpnn(opt.pmpnn_params, eps=0.05))
    transform = lambda center, T: transform_logits(
        [
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
        design_info = dict(attempt=idx)
        start = time.time()
        tag_pair_info = init_tag_pair(
            salad_data, opt.tag, opt.num_disambiguating, opt.num_padding
        )
        non_tag_mask = tag_pair_info["non_tag_mask"]
        num_non_tag = num_aa - len(opt.tag)
        exposed = data_from_salad(
            sampler(salad_params, key(), tag_pair_info["exposed"], init_prev)
        )
        hidden = data_from_salad(
            sampler(salad_params, key(), tag_pair_info["hidden"], init_prev)
        )
        print(f"Design {idx} in {time.time() - start:.2f} s")
        exposed.save_pdb(f"{opt.out_path}/attempts/exposed_{idx}.pdb")
        hidden.save_pdb(f"{opt.out_path}/attempts/hidden_{idx}.pdb")
        for ids in range(opt.num_samples):
            T = opt.temperature
            design_info["seq_id"] = ids
            design_info["T"] = T
            design_info["center"] = True
            pmpnn_untied = DesignData.concatenate([exposed, hidden], sep_batch=True)
            pmpnn_untied = pmpnn_untied.update(
                aa=tag_pair_info["pmpnn_aa"],
                tie_index=tag_pair_info["unpaired_tie_info"]["tie_index"],
                tie_weight=tag_pair_info["unpaired_tie_info"]["tie_weight"],
                tie_weights=tag_pair_info["unpaired_tie_info"]["tie_weight"],
            )
            pmpnn_tied = pmpnn_untied.update(
                aa=tag_pair_info["pmpnn_aa"],
                tie_index=tag_pair_info["tie_info"]["tie_index"],
                tie_weight=tag_pair_info["tie_info"]["tie_weight"],
                tie_weights=tag_pair_info["tie_info"]["tie_weight"],
            )
            result_tied, _ = pmpnn_sampler(None, T)(key(), pmpnn_tied)
            pmpnn_tied["aa"] = aas.translate(
                result_tied["aa"], aas.PMPNN_CODE, aas.AF2_CODE
            )
            pmpnn_result = pmpnn_tied.split(pmpnn_tied["batch_index"])

            for mode_name, pmpnn_out in zip(("exposed", "hidden"), pmpnn_result):
                af_input = AFInput.from_data(pmpnn_out).add_guess(pmpnn_out)
                mode_data = dict()
                af_res = af2(af2_params, key(), af_input).unpad()
                mode_data["result"] = af_res
                mode_data["sequence"] = pmpnn_out.to_sequence_string()
                mode_data["plddt"] = (
                    (af_res.plddt * non_tag_mask).sum() / max(1, num_non_tag)
                )
                mode_data["sc_rmsd"] = RMSD()(
                    af_res.to_data(),
                    pmpnn_out,
                    weight=non_tag_mask,
                    eval_mask=non_tag_mask,
                )
                mode_data["success"] = (mode_data["plddt"] > 0.75) * (
                    mode_data["sc_rmsd"] < 3.0
                )
                print(
                    idx,
                    ids,
                    mode_name,
                    mode_data["sequence"],
                    mode_data["plddt"],
                    mode_data["sc_rmsd"],
                )
                if mode_data["success"]:
                    af_res.save_pdb(
                        f"{opt.out_path}/success/{mode_name}_{idx}_{ids}.pdb"
                    )


if __name__ == "__main__":
    main()
