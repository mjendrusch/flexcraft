# Binder design protocol adapted from escalante-bio blogpost:
# https://blog.escalante.bio/teaching-generative-models-to-hallucinate/

import hashlib
import os
import uuid

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax

import flexcraft.sequence.aa_codes as aas
from flexcraft.data.data import DesignData
from flexcraft.files.csv import ScoreCSV
from flexcraft.hallucination.opt import jit_loss_update, simplex_agpm
from flexcraft.pipelines.bind.hal import (
    _parse_chunk,
    _parse_hotspots,
    _passing,
    _report_result,
    _report_trajectory,
    binder_hallucination_metrics,
    off_target_hallucination_metrics,
)
from flexcraft.sequence.mpnn import batch_mpnn, make_pmpnn
from flexcraft.structure.af import (
    AFInput,
    AFResult,
    get_model_haiku_params,
    make_af2,
    make_predict,
    model_config,
)
from flexcraft.structure.boltz import Joltz2, JoltzInput, JoltzResult, JoltzSpec
from flexcraft.utils import Keygen, data_from_salad, load_pdb, parse_options


def build_joltz_binder_spec(
    binder_length: int,
    target_sequence: str,
    target_template: str = "none",
    target_msa: str = "none",
) -> JoltzSpec:
    """Build a `JoltzSpec` for binder hallucination preserving both template and MSA inputs."""
    target_chains = target_sequence.split(":")
    use_target_msa = (target_template == "none") and (target_msa == "none")
    spec = (
        JoltzSpec()
        .add_protein("X" * binder_length)
        .add_protein(*target_chains, use_msa=use_target_msa)
    )
    if target_template != "none":
        spec = spec.add_template(
            target_template,
            to_chains=[
                "BCDEFGHIJKLMNOPQRSTUVWXYZ"[i] for i in range(len(target_chains))
            ],
        )
    if target_msa != "none":
        spec = spec.precomputed_msa(target_msa, start_chain=1)
    return spec


def main():
    opt = parse_options(
        "predict structures with AlphaFold",
        pmpnn_path="params/solmpnn/v_48_030.pkl",
        salad_path="params/salad/default_vp_scaled-200k.jax",
        boltz_path="params/boltz/",
        target_path="target.pdb",
        target_sequence="none",
        target_template="none",
        predict_template="False",
        target_msa="none",
        predict_msa="False",
        off_target_sequence="none",
        hotspots="none",
        target_chains="all",
        out_path="out",
        center="False",
        hallucination_model="joltz",
        model_name="model_1_ptm",
        param_path="params/af",
        sample_structure="False",
        inter_contact_distance=20.0,
        intra_contact_distance=14.0,
        hotspot_distance=0.0,
        num_designs=48,
        length=80,
        repeat=1,
        temperature=0.1,
        samples=10,
        seed=42,
        tmpdir="tmp/",
    )

    binder_length = opt.length
    target_template = opt.target_template
    target_msa = opt.target_msa
    if target_template == "none" and opt.predict_template == "True":
        os.makedirs(opt.tmpdir, exist_ok=True)
        target_template = f"{opt.tmpdir}/target_{str(uuid.uuid4())}.cif"
    if target_msa == "none" and opt.predict_msa == "True":
        os.makedirs(opt.tmpdir, exist_ok=True)
        target_msa = (
            f"{opt.tmpdir}/target_"
            f"{hashlib.sha256(opt.target_sequence.encode('utf-8')).hexdigest()}.npz"
        )
    key = Keygen(opt.seed)
    kval = key()

    pmpnn, pmpnn_params = make_pmpnn(opt.pmpnn_path, eps=0.05, split_params=True)
    jit_pmpnn = jax.jit(pmpnn)
    params = dict(mpnn=pmpnn_params)

    sample_structure = opt.sample_structure == "True"
    sampler = None
    if sample_structure:
        import salad.inference as si

        def cloud_std_default(n_aa):
            minval = n_aa**0.4
            return minval + np.random.rand() * 3.0

        def model_step(config):
            config.eval = True

            @hk.transform
            def step(data, prev):
                noise = si.StructureDiffusionNoise(config)
                predict = si.StructureDiffusionPredict(config)
                data.update(noise(data))
                out, prev = predict(data, prev)
                return out, prev

            return step.apply

        salad_config, salad_params = si.make_salad_model(
            "default_vp_scaled", opt.salad_path
        )
        num_aa, resi, chain, is_cyclic, cyclic_mask = si.parse_num_aa(f"{opt.length}")
        salad_data, init_prev = si.data.from_config(
            salad_config,
            num_aa=num_aa,
            residue_index=resi,
            chain_index=chain,
            cyclic_mask=cyclic_mask,
        )
        salad_step = model_step(salad_config)
        sampler = si.Sampler(salad_step, out_steps=400)

    hotspot_index = _parse_hotspots(opt.hotspots, binder_length=binder_length)

    target = None
    if opt.target_sequence != "none":
        target_sequence = opt.target_sequence
    else:
        target = load_pdb(opt.target_path).unpad()
        if opt.target_chains != "all":
            selected_chains = np.array(
                ["ABCDEFGHIJKLMNOPQRSTUVWXYZ".index(c) for c in opt.target_chains],
                dtype=np.int32,
            )
            target = target[(target.chain_index[:, None] == selected_chains).any(axis=-1)]
        target_sequence = target.to_sequence_string()

    off_target_sequence = None
    if opt.off_target_sequence != "none":
        off_target_sequence = opt.off_target_sequence.strip()

    model = Joltz2(cache=opt.boltz_path)
    joltz_pred = model.predictor(num_recycle=4)
    if opt.predict_msa == "True":
        if not os.path.isfile(target_msa):
            (
                JoltzSpec()
                .add_protein(*target_sequence.split(":"), use_msa=True)
                .save_msa(target_msa, cache=opt.boltz_path)
            )
    prediction = None
    if opt.predict_template == "True":
        prediction = joltz_pred(
            key(),
            JoltzSpec().add_protein(*target_sequence.split(":"), use_msa=True),
        )
        prediction.save_cif(target_template)

    joltz = None
    af2 = None
    af_template_data = None
    if opt.hallucination_model == "joltz":
        joltz, joltz_params = model.evaluator(num_recycle=4)
        joltz_spec = build_joltz_binder_spec(
            binder_length=binder_length,
            target_sequence=target_sequence,
            target_template=target_template,
            target_msa=target_msa,
        )
        joltz_input, _ = joltz_spec.to_input(pad=False)
        params["joltz"] = joltz_params
        params["joltz_input"] = joltz_input
        if off_target_sequence is not None:
            off_input, _ = (
                JoltzSpec()
                .add_protein("X" * binder_length)
                .add_protein(*off_target_sequence.split(":"), use_msa=True)
                .to_input(pad=False)
            )
            params["off_input"] = off_input
    elif opt.hallucination_model == "af2":
        af2_params = get_model_haiku_params(
            model_name=opt.model_name, data_dir=opt.param_path, fuse=True
        )
        af2_params = jax.tree.map(lambda x: jnp.array(x), af2_params)
        config = model_config(opt.model_name)
        config.model.global_config.use_dgram = False
        config.model.global_config.use_remat = True
        af2 = make_predict(make_af2(config), num_recycle=2)
        if target is None:
            if opt.predict_template != "True":
                prediction = joltz_pred(
                    key(),
                    JoltzSpec().add_protein(*target_sequence.split(":"), use_msa=True),
                )
            os.makedirs(opt.tmpdir, exist_ok=True)
            target_pdb = f"{opt.tmpdir}/target_{str(uuid.uuid4())}.pdb"
            prediction.save_pdb(target_pdb)
            target = load_pdb(target_pdb).unpad()
        design_data = DesignData.concatenate(
            (DesignData.from_length(binder_length), target)
        )
        is_target = jnp.zeros_like(design_data["mask"]).at[binder_length:].set(True)
        af_template_data = AFInput.from_data(design_data).add_template(
            design_data, where=is_target
        )
        params["af2"] = af2_params

    def metrics(sequence, step_key, result):
        return binder_hallucination_metrics(
            sequence,
            step_key,
            result,
            binder_length=binder_length,
            pmpnn=pmpnn,
            mpnn_params=params["mpnn"],
            inter_contact_distance=opt.inter_contact_distance,
            intra_contact_distance=opt.intra_contact_distance,
            hotspot_index=hotspot_index,
        )

    def off_target_metrics(sequence, step_key, target_logits, off_target_result):
        return off_target_hallucination_metrics(
            sequence,
            step_key,
            target_logits,
            off_target_result,
            binder_length=binder_length,
            pmpnn=pmpnn,
            mpnn_params=params["mpnn"],
        )

    def loss(sequence, key=kval, context=None, params=None):
        keys = jax.random.split(key, 4)
        sequence = sequence.at[:, aas.AF2_CODE.index("C")].set(0.0)
        sequence = sequence / sequence.sum(axis=1, keepdims=True)
        if "joltz" in params:
            joltz_input: JoltzInput = params["joltz_input"].set_aa(sequence)
            result: JoltzResult = joltz(params["joltz"]).predict(
                keys[0], joltz_input, num_samples=4
            )
        elif "af2" in params:
            target_mask = (af_template_data.data["seq_mask"][binder_length:] > 0)[:, None]
            target_seq_oh = (
                jax.nn.one_hot(
                    af_template_data.data["aatype"][binder_length:], 20, axis=-1
                )
                * target_mask
            )
            af_data = af_template_data.update_sequence(
                jnp.concatenate((sequence, target_seq_oh), axis=0)
            )
            af_data = af_data.block_diagonal(2)
            result: AFResult = af2(params["af2"], keys[0], af_data)
        out = metrics(sequence, keys[1], result)
        if "off_input" in params:
            off_input: JoltzInput = params["off_input"].set_aa(sequence)
            off_target_result: JoltzResult = joltz(params["joltz"]).predict(
                keys[2], off_input, num_samples=4
            )
            out.update(
                off_target_metrics(sequence, keys[3], out["logits"], off_target_result)
            )
        out["result"] = result

        value = (
            out["binder_target_contacts"]
            + out["within_binder_contacts"]
            - 10.0 * out["recovery"]
            + 0.05 * out["target_binder_pae"]
            + 0.05 * out["binder_target_pae"]
            + 0.4 * out["within_binder_pae"]
            - 0.1 * out["plddt"]
            - 0.025 * out["iptm"]
            - 0.025 * out["eptm"]
        )
        if opt.hotspot_distance > 0:
            value += opt.hotspot_distance * out["hotspot_distance"]
        if "off_input" in params:
            value += -out["off_target_contacts"].mean()
        out["total"] = value
        return value, out

    loss_update = jit_loss_update(loss)

    if sample_structure:
        os.makedirs(f"{opt.out_path}/blueprints/", exist_ok=True)
    os.makedirs(f"{opt.out_path}/attempts/", exist_ok=True)
    os.makedirs(f"{opt.out_path}/success/", exist_ok=True)
    common_keys = [
        "plddt",
        "recovery",
        "binder_target_contacts",
        "hotspot_contacts",
        "within_binder_contacts",
        "within_binder_pae",
        "binder_target_pae",
        "target_binder_pae",
        "iptm",
        "eptm",
        "ipsae",
        "average_num_nonzero",
        "entropy",
        "L",
        "H",
        "E",
    ]
    trajectory_keys = list(common_keys)
    if "off_input" in params:
        trajectory_keys += [
            "off_target_contacts",
            "off_target_binder_pae",
            "off_target_iptm",
            "off_target_recovery",
        ]
    trajectory = ScoreCSV(
        f"{opt.out_path}/trajectory.csv", keys=["attempt", "step"] + trajectory_keys
    )
    results_keys = ["attempt", "stage", "sequence"] + common_keys
    results = ScoreCSV(f"{opt.out_path}/results.csv", keys=results_keys)
    success = ScoreCSV(f"{opt.out_path}/success.csv", keys=results_keys)

    success_count = 0
    _attempt = 0
    while success_count < opt.num_designs:
        attempt = _attempt
        _attempt += 1
        context = dict(recovery=10.0, ipae=0.05)
        if sampler is not None:
            salad_data["dssp_condition"], dssp_string = si.random_dssp(
                num_aa, p=0.0, p_keep_loop=1.0, return_string=True
            )
            print("DSSP target:", dssp_string)
            salad_data["cloud_std"] = cloud_std_default(num_aa)
            salad_design = sampler(salad_params, key(), salad_data, init_prev)
            design = data_from_salad(salad_design)
            design.save_pdb(f"{opt.out_path}/blueprints/design_{attempt}.pdb")
            sequence = aas.translate_onehot(
                jit_pmpnn(pmpnn_params, key(), design.drop_aa())["logits"],
                aas.PMPNN_CODE,
                aas.AF2_CODE,
            )[..., :20]
            sequence = jax.nn.softmax(sequence, axis=-1)
        else:
            sequence = jax.random.gumbel(key(), (opt.length, 20), dtype=jnp.float32)
            sequence *= np.random.uniform(low=0.75, high=5.0)
            sequence = jnp.array(sequence, dtype=jnp.float32)
            sequence = jax.nn.softmax(sequence, axis=-1)
        sequence, _, aux = simplex_agpm(
            sequence,
            loss_update,
            key=key,
            params=params,
            context=context,
            num_steps=100,
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
        sequence_1 = sequence
        result = aux["result"].unpad()
        if opt.hallucination_model == "af2":
            result.save_pdb(f"{opt.out_path}/attempts/raw_{attempt}_0.pdb")
        data = result.to_data().unpad()
        spec = (
            JoltzSpec()
            .add_protein(data.to_sequence_string().split(":")[0][:binder_length])
            .add_protein(*target_sequence.split(":"), use_msa=True)
        )
        if target_msa != "none":
            spec = spec.precomputed_msa(target_msa, start_chain=1)
        prediction_1 = joltz_pred(key(), spec)
        stage_1 = _report_result(
            attempt,
            "stage_1",
            key(),
            results,
            sequence,
            prediction_1.result.unpad(),
            metrics,
        )
        prediction_1.save_pdb(f"{opt.out_path}/attempts/design_{attempt}_1.pdb")

        sequence = jnp.log(sequence + 1e-5)
        context["recovery"] = 10.0
        sequence, _, aux = simplex_agpm(
            sequence,
            loss_update,
            key=key,
            params=params,
            context=context,
            num_steps=50,
            lr=0.5 * jnp.sqrt(sequence.shape[0]),
            scale=1.25,
            grad_transform=optax.clip_by_global_norm(1.0),
            param_transform=lambda x: x.at[:, aas.AF2_CODE.index("C")].set(-1e6),
            verbose=True,
            return_last=True,
            opt_logits=True,
            report=_report_trajectory(attempt, trajectory, start_step=100),
        )
        sequence_2 = sequence
        result = aux["result"].unpad()
        if opt.hallucination_model == "af2":
            result.save_pdb(f"{opt.out_path}/attempts/raw_{attempt}_1.pdb")
        data = result.to_data().unpad()
        spec = (
            JoltzSpec()
            .add_protein(data.to_sequence_string().split(":")[0][:binder_length])
            .add_protein(*target_sequence.split(":"), use_msa=True)
        )
        if target_msa != "none":
            spec = spec.precomputed_msa(target_msa, start_chain=1)
        prediction_2 = joltz_pred(key(), spec)
        stage_2 = _report_result(
            attempt,
            "stage_2",
            key(),
            results,
            sequence,
            prediction_2.result.unpad(),
            metrics,
        )
        prediction_2.save_pdb(f"{opt.out_path}/attempts/design_{attempt}_2.pdb")

        stage = "stage_2"
        aux = stage_2
        prediction = prediction_2
        sequence = sequence_2
        if stage_2["ipsae"] < stage_1["ipsae"] and _passing(stage_1):
            stage = "stage_1"
            aux = stage_1
            prediction = prediction_1
            sequence = sequence_1
        if _passing(aux):
            _report_result(
                attempt,
                stage,
                key(),
                success,
                sequence,
                prediction.result.unpad(),
                metrics,
                aux=aux,
            )
            prediction.save_pdb(f"{opt.out_path}/success/design_{attempt}_{stage}.pdb")
            success_count += 1


if __name__ == "__main__":
    main()
