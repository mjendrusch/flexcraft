import os
from copy import deepcopy

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax

import salad.inference as si

import flexcraft.sequence.aa_codes as aas
from flexcraft.data.data import DesignData
from flexcraft.files import ScoreCSV
from flexcraft.hallucination.opt import jit_loss_update, simplex_agpm
from flexcraft.pipelines.bind.hal import _report_result, _report_trajectory, masked_mean
from flexcraft.sequence.mpnn import batch_mpnn, make_pmpnn
from flexcraft.structure.boltz import Joltz2, JoltzInput, JoltzResult, JoltzSpec
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


def main():
    opt = parse_options(
        "Protein sequence design with custom protein MPNN",
        num_aa="100",
        out_path="outputs/",
        boltz_path="./params/boltz/",
        salad_params="params/salad/default_vp_scaled-200k.jax",
        pmpnn_params="params/pmpnn/v_48_030.pkl",
        af2_params="params/af/",
        af2_model="model_1_ptm",
        num_blocks=10,
        num_designs=100,
        num_samples=20,
        std_buffer=3.0,
        temperature=1.0,
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
    sampler = si.Sampler(salad_step, out_steps=400)

    _pmpnn, pmpnn_params = make_pmpnn(opt.pmpnn_params, eps=0.05, split_params=True)
    params = dict(mpnn=pmpnn_params)

    joltz_model = Joltz2(cache=opt.boltz_path)
    joltz, params["joltz"] = joltz_model.evaluator(num_recycle=4)
    joltz_spec = JoltzSpec().add_protein("X" * num_aa)
    params["joltz_input"], _ = joltz_spec.to_input(pad=False)
    joltz_pred = joltz_model.predictor(num_recycle=4, num_samples=4)

    def metrics(sequence, step_key, result: JoltzResult):
        pad_mask = jnp.asarray(result.token_pad_mask, dtype=jnp.bool_)
        seq_len = sequence.shape[0]
        seq_mask = pad_mask[:seq_len]
        data = result.to_data(return_samples=True)
        dssp = data[:seq_len].p_dssp
        pmpnn_input = data.update(
            aa=aas.translate(data["aa"], aas.AF2_CODE, aas.PMPNN_CODE)
        )
        logits_full = batch_mpnn(_pmpnn)(
            params["mpnn"], step_key, pmpnn_input.drop_aa()
        )["logits"]
        logits_full = aas.translate_onehot(
            logits_full, aas.PMPNN_CODE, aas.AF2_CODE
        )[..., :20]
        if not result.is_single_sample:
            logits_full = logits_full.mean(axis=0)
        logits = logits_full[:seq_len]
        prob = jax.lax.stop_gradient(
            jax.nn.softmax(
                logits.at[:, aas.AF2_CODE.index("C")].set(-1e6) / 0.1, axis=-1
            )
        )
        entropy = masked_mean(
            -(jax.nn.log_softmax(logits, axis=-1) * prob).sum(axis=-1), seq_mask
        )
        recovery = masked_mean((sequence * prob).sum(axis=-1), seq_mask)
        return dict(
            logits=logits,
            entropy=entropy,
            plddt=result.mean_plddt.mean(),
            recovery=recovery,
            contacts=result.contact_score(num_contacts=25, min_resi_distance=10),
            pae=result.max_pae * result.mean_pae.mean(),
            average_num_nonzero=masked_mean((sequence > 0.01).sum(axis=-1), seq_mask),
            L=dssp["L"],
            H=dssp["H"],
            E=dssp["E"],
        )

    def loss(sequence, key=None, context=None, params=None):
        keys = jax.random.split(key, 4)
        sequence = sequence.at[:, aas.AF2_CODE.index("C")].set(0.0)
        sequence = sequence / sequence.sum(axis=1, keepdims=True)
        joltz_input: JoltzInput = params["joltz_input"].set_aa(sequence)
        result: JoltzResult = joltz(params["joltz"]).predict(
            keys[0], joltz_input, num_samples=4
        )
        out = metrics(sequence, keys[1], result)
        out["result"] = result

        value = (
            out["contacts"]
            - 10.0 * out["recovery"]
            + 0.4 * out["pae"]
            - 0.1 * out["plddt"]
        )
        out["total"] = value
        return value, out

    loss_update = jit_loss_update(loss)

    os.makedirs(f"{opt.out_path}/blueprints/", exist_ok=True)
    os.makedirs(f"{opt.out_path}/attempts/", exist_ok=True)
    os.makedirs(f"{opt.out_path}/success/", exist_ok=True)
    common_keys = [
        "plddt",
        "entropy",
        "recovery",
        "contacts",
        "pae",
        "average_num_nonzero",
        "L",
        "H",
        "E",
    ]
    trajectory = ScoreCSV(
        f"{opt.out_path}/trajectory.csv", keys=["attempt", "step"] + common_keys
    )
    results_keys = ["attempt", "stage", "sequence"] + common_keys
    results = ScoreCSV(f"{opt.out_path}/results.csv", keys=results_keys)

    success_count = 0
    _attempt = 0
    while success_count < opt.num_designs:
        attempt = _attempt
        _attempt += 1
        context = dict(recovery=10.0, ipae=0.05)
        salad_data["dssp_condition"], dssp_string = si.random_dssp(
            num_aa, p=0.0, p_keep_loop=1.0, return_string=True
        )
        print("DSSP target:", dssp_string)
        salad_data["cloud_std"] = cloud_std_default(num_aa)
        salad_design = sampler(salad_params, key(), salad_data, init_prev)
        design = data_from_salad(salad_design)
        design.save_pdb(f"{opt.out_path}/blueprints/design_{attempt}.pdb")
        sequence = aas.translate_onehot(
            pmpnn(key(), design.drop_aa())["logits"],
            aas.PMPNN_CODE,
            aas.AF2_CODE,
        )[..., :20]
        sequence = jax.nn.softmax(sequence / opt.temperature, axis=-1)
        (_, aux), _ = loss_update(sequence, key=key(), params=params)
        aux_data: DesignData = aux["result"].unpad().to_data().unpad()
        aux_data.save_pdb(f"{opt.out_path}/attempts/design_{attempt}_init.pdb")
        sequence, _, aux = simplex_agpm(
            sequence,
            loss_update,
            key=key,
            params=params,
            context=context,
            num_steps=50,
            lr=0.1 * jnp.sqrt(sequence.shape[0]),
            momentum=0.3,
            scale=1.0,
            grad_transform=optax.clip_by_global_norm(1.0),
            param_transform=lambda x: x.at[:, aas.AF2_CODE.index("C")].set(0.0),
            verbose=True,
            return_last=False,
            early_stop=None,
            report=_report_trajectory(attempt, trajectory, start_step=0),
        )
        result = aux["result"].unpad()
        data = result.to_data().unpad()
        prediction_1 = joltz_pred(
            key(), JoltzSpec().add_protein(data.to_sequence_string())
        )
        _report_result(
            attempt,
            "stage_1",
            key(),
            results,
            sequence,
            prediction_1.result.unpad(),
            metrics,
        )
        prediction_1.save_pdb(f"{opt.out_path}/attempts/design_{attempt}.pdb")
        success_count += 1


if __name__ == "__main__":
    main()
