# Binder design protocol adapted from escalante-bio blogpost:
# https://blog.escalante.bio/teaching-generative-models-to-hallucinate/

import os
import time
import uuid
import numpy as np
import jax
import haiku as hk

from colabdesign.af.alphafold.model.config import model_config
from colabdesign.af.alphafold.model.data import get_model_haiku_params

import optax

from flexcraft.utils.options import parse_options
from flexcraft.utils.rng import Keygen
from flexcraft.utils import Keygen, parse_options, load_pdb, data_from_salad
from flexcraft.sequence.sample import *
from flexcraft.data.data import DesignData
from flexcraft.sequence.mpnn import make_pmpnn
import flexcraft.sequence.aa_codes as aas
from flexcraft.hallucination.opt import jit_loss_update, simplex_agpm
from flexcraft.structure.boltz._model import Joltz2, JoltzResult, Joltz2Writer
from flexcraft.structure.boltz._data import JoltzSpec, JoltzInput
from flexcraft.structure.af import AFInput, AFResult, make_af2, make_predict
from flexcraft.files.csv import ScoreCSV

def _parse_hotspots(spec: str, binder_length = 100) -> None | np.ndarray:
    if spec == "none":
        return None
    return binder_length + np.concatenate([
        _parse_chunk(c) for c in spec.split(",")], axis=0)

def _parse_chunk(spec: str) -> np.ndarray:
    if "-" not in spec:
        return np.array([int(spec) - 1], dtype=np.int32)
    start, end = [int(c) for c in spec.split("-")]
    return np.arange(start - 1, end, dtype=np.int32)

opt = parse_options(
    "predict structures with AlphaFold",
    pmpnn_path="params/solmpnn/v_48_030.pkl",
    salad_path="params/salad/default_vp_scaled-200k.jax",
    boltz_path="params/boltz/",
    target_path="target.pdb",
    target_sequence="none",
    target_template="none",
    predict_template="False",
    off_target_sequence="none",
    hotspots="none",
    target_chains="all",
    out_path="out",
    center="False",
    hallucination_model="joltz",
    model_name="model_1_ptm",
    param_path="params/af",
    sample_structure="False",
    num_designs=48,
    length=80,
    repeat=1,
    temperature=0.1,
    samples=10,
    seed=42,
    tmpdir="tmp/"
)

binder_length = opt.length
target_template = opt.target_template
if target_template == "none" and opt.predict_template == "True":
    os.makedirs(opt.tmpdir, exist_ok=True)
    target_template = f"{opt.tmpdir}/target_{str(uuid.uuid4())}.cif"
key = Keygen(opt.seed)
kval = key()
# set up protein mpnn
pmpnn, pmpnn_params = make_pmpnn(opt.pmpnn_path, eps=0.05, split_params=True)
jit_pmpnn = jax.jit(pmpnn)
params = dict(
    mpnn=pmpnn_params
)
# set up salad
sample_structure = opt.sample_structure == "True"
sampler = None
if sample_structure:
    import salad.inference as si
    def cloud_std_default(num_aa):
        minval = num_aa ** 0.4
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

    salad_config, salad_params = si.make_salad_model("default_vp_scaled", opt.salad_path)

    # initialize salad data and prev from the num_aa specification
    num_aa, resi, chain, is_cyclic, cyclic_mask = si.parse_num_aa(f"{opt.length}")
    salad_data, init_prev = si.data.from_config(
        salad_config,
        num_aa=num_aa,
        residue_index=resi,
        chain_index=chain,
        cyclic_mask=cyclic_mask)

    salad_step = model_step(salad_config)
    # build a sampler object for sampling
    sampler = si.Sampler(salad_step, out_steps=400)
# hotspot index
hotspot_index = _parse_hotspots(opt.hotspots, binder_length = binder_length)
# retrieve target
if opt.target_sequence != "none":
    target_sequence = opt.target_sequence
else:
    target = load_pdb(opt.target_path)
    if opt.target_chains != "all":
        selected_chains = np.array([
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ".index(c)
            for c in opt.target_chains], dtype=np.int32)
        target = target[(target.chain_index[:, None] == selected_chains).any(axis=-1)]
    target_sequence = target.to_sequence_string()
off_target_sequence = None
if opt.off_target_sequence != "none":
    off_target_sequence = opt.off_target_sequence.strip()
# set up Boltz models
model = Joltz2(cache=opt.boltz_path)
# full-atom predictor with context
joltz_pred = model.predictor(num_recycle=4)
# if applicable, predict & save a template structure
if opt.predict_template == "True":
    prediction = joltz_pred(key(), JoltzSpec().add_protein(
        *[c for c in target_sequence.split(":")], use_msa=True))
    prediction.save_cif(target_template)
# unknown-aa hallucination model
if opt.hallucination_model == "joltz":
    joltz, joltz_params = model.evaluator(num_recycle=4)
    if opt.target_template != "none":
        joltz_spec = (
            # start with empty spec
            JoltzSpec()
            # add a designable binder (all-X sequence), not generating MSA
            .add_protein("X" * binder_length)
            # add all target chains, generating MSA
            .add_protein(*[c for c in target_sequence.split(":")], use_msa=False)
            # add input template structure
            .add_template(target_template, to_chains=[
                "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[i]
                for i, _ in enumerate(target_sequence.split(":"))]))
    else:
        joltz_spec = (
            # start with empty spec
            JoltzSpec()
            # add a designable binder (all-X sequence), not generating MSA
            .add_protein("X" * binder_length)
            # add all target chains, generating MSA
            .add_protein(*[c for c in target_sequence.split(":")], use_msa=True))
    joltz_input, writer = joltz_spec.to_input(pad=False)
    params["joltz"] = joltz_params
    params["joltz_input"] = joltz_input
    joltz_off = None
    if off_target_sequence is not None:
        off_input, _ = (JoltzSpec()
            .add_protein("X" * binder_length)
            .add_protein(*[c for c in off_target_sequence.split(":")], use_msa=True)
            .to_input(pad=False))
        params["off_input"] = off_input
elif opt.hallucination_model == "af2":
    writer = ...
    af2_params = get_model_haiku_params(
        model_name=opt.model_name,
        data_dir=opt.param_path, fuse=True)
    af2_params = jax.tree.map(lambda x: jnp.array(x), af2_params)
    config = model_config(opt.model_name)
    config.model.global_config.use_dgram = False
    config.model.global_config.use_remat = True
    af2 = make_predict(make_af2(config), num_recycle=2)
    design_data = DesignData.concatenate((DesignData.from_length(binder_length), target))
    is_target = jnp.zeros_like(design_data["mask"]).at[binder_length:].set(True)
    af_template_data = AFInput.from_data(design_data).add_template(
        design_data, where=is_target)
    params["af2"] = af2_params

def batch_mpnn(mpnn):
    def _mpnn_map(params, key, data):
        if len(data["atom_positions"].shape) == 4:
            num_samples = data["atom_positions"].shape[0]
            # replicate all un-batched inputs
            xx = {
                key: jnp.repeat(data[key][None, :], num_samples, axis=0)
                for key in ["atom_mask", "mask", "aa", "residue_index", "chain_index", "batch_index"]
            }
            # add in the batched atom positions
            xx["atom_positions"] = data["atom_positions"]
            # split keys across batch
            keys = jax.random.split(key, num_samples)
            # vmap the ProteinMPNN
            return jax.vmap(mpnn, (None, 0, 0), 0)(params, keys, xx)
        else:
            return mpnn(params, key, data)
    return _mpnn_map

def metrics(sequence, key, result: JoltzResult):
    res_data = result.to_data(return_samples=True)
    binder_data = res_data[:binder_length]
    dssp = binder_data.p_dssp
    pmpnn_input = res_data.update(aa=aas.translate(res_data["aa"], aas.AF2_CODE, aas.PMPNN_CODE))
    is_binder = jnp.zeros((pmpnn_input.aa.shape[0],), dtype=jnp.bool_).at[:binder_length].set(True)
    logits = batch_mpnn(pmpnn)(params["mpnn"], key, pmpnn_input.drop_aa(where=is_binder))["logits"]
    logits = aas.translate_onehot(logits, aas.PMPNN_CODE, aas.AF2_CODE)[..., :20]
    if not result.is_single_sample:
        logits = logits.mean(axis=0)
    logits = logits[:binder_length]
    center = logits.mean(axis=0)
    logits = logits - center
    prob = jax.lax.stop_gradient(jax.nn.softmax(logits.at[:, aas.AF2_CODE.index("C")].set(-1e6) / 0.1, axis=-1))
    # losses
    plddt = result.plddt
    pae = 32 * result.pae # unnormalized pAE
    if not result.is_single_sample:
        plddt = plddt.mean(axis=0)
        pae = pae.mean(axis=0)
    binder_selector = result.chain_index == 0
    hotspot_selector = result.chain_index != 0
    if hotspot_index is not None:
        hotspot_selector = (np.arange(result.residue_index.shape[0])[:, None] == hotspot_index).any(axis=-1)
    hotspot_contacts = (
        result.index_contact_score(
            binder_selector, hotspot_selector, # NOTE: flip target and binder here
            num_contacts=3, contact_distance=20.0)
      + result.index_contact_score(
            hotspot_selector, binder_selector, # NOTE: flip target and binder here
            num_contacts=3, contact_distance=20.0)
    ) / 2
    binder_target_contacts = result.chain_contact_score(1, 0, num_contacts=3, contact_distance=20.0)
    if hotspot_index is not None:
        binder_target_contacts = hotspot_contacts
    return dict(
        logits = logits,
        plddt = plddt[:binder_length].mean(),
        recovery = (sequence * prob).sum(axis=-1).mean(),
        binder_target_contacts = binder_target_contacts,
        within_binder_contacts = result.chain_contact_score(0, 0, num_contacts=25, min_resi_distance=10),
        hotspot_contacts = hotspot_contacts,
        within_binder_pae = pae[:binder_length, :binder_length].mean(),
        binder_target_pae = pae[:binder_length, binder_length:].mean(),
        target_binder_pae = pae[binder_length:, :binder_length].mean(),
        iptm = result.index_iptm(chain_index=is_binder),
        eptm = result.index_ptm_score(chain_index=is_binder),
        ipsae = result.ipsae(chain_index=is_binder),
        average_num_nonzero = (sequence > 0.01).sum(axis=-1).mean(),
        L = dssp["L"], H = dssp["H"], E = dssp["E"]
    )

def off_target_metrics(sequence, key, target_logits: jax.Array, off_target_result: JoltzResult):
    res_data = off_target_result.to_data(return_samples=True)
    pmpnn_input = res_data.update(aa=aas.translate(res_data["aa"], aas.AF2_CODE, aas.PMPNN_CODE))
    is_binder = jnp.zeros((pmpnn_input.aa.shape[0],), dtype=jnp.bool_).at[:binder_length].set(True)
    logits = batch_mpnn(pmpnn)(params["mpnn"], key, pmpnn_input.drop_aa(where=is_binder))["logits"]
    logits = aas.translate_onehot(logits, aas.PMPNN_CODE, aas.AF2_CODE)[..., :20]
    if not off_target_result.is_single_sample:
        logits = logits.mean(axis=0)
    logits = logits[:binder_length]
    center = logits.mean(axis=0)
    logits = logits - center
    diff_mask = target_logits.argmax(axis=1) != logits.argmax(axis=1)
    # off_target_logits = logits
    # where the off-target logits differ from the target logits,
    # apply guidance to reinforce on-target interactions
    # delta_logits = target_logits - off_target_logits
    # change_positions = (abs(delta_logits) >= 0.1).any(axis=1)
    # jnp.where(change_positions[:, None], delta_logits, target_logits)
    # logits = target_logits
    prob = jax.lax.stop_gradient(jax.nn.softmax(logits.at[:, aas.AF2_CODE.index("C")].set(-1e6) / 0.1, axis=-1))
    pae = 32 * off_target_result.pae
    if not off_target_result.is_single_sample:
        pae = pae.mean(axis=0)
    return dict(
        # logits = logits,
        # recovery = (sequence * prob).sum(axis=-1).mean(),
        off_target_recovery = ((sequence * prob).sum(axis=-1) * diff_mask).sum() / jnp.maximum(1, diff_mask.sum()),
        off_target_contacts = off_target_result.chain_contact_score(1, 0, num_contacts=3, contact_distance=20.0),
        binder_off_target_pae = pae[:binder_length, binder_length:].mean(),
        off_target_binder_pae = pae[binder_length:, :binder_length].mean(),
        off_target_iptm = off_target_result.iptm,
        off_target_ipsae = off_target_result.ipsae(chain_index=is_binder),
    )

def _report_trajectory(index: int, writer: ScoreCSV, start_step=0):
    def _report(step: int, loss: float, aux: dict):
        print(loss.shape)
        line_info = dict(attempt=index, step=start_step + step, total=float(loss.mean()))
        for key in writer.keys:
            if key in aux:
                line_info[key] = float(aux[key].mean())
        writer.write_line(line_info)
    return _report

def _report_result(attempt: int, stage: str, key, writer: ScoreCSV, sequence, result: JoltzResult, aux=None):
    if aux is None:
        aux = metrics(sequence, key, result)
    line_info = dict(attempt=attempt, stage=stage, sequence=aas.decode(jnp.argmax(sequence, axis=-1), aas.AF2_CODE))
    for key in writer.keys:
        if key in aux:
            line_info[key] = float(aux[key].mean())
    writer.write_line(line_info)
    return aux

def _passing(aux):
    return (
        aux["ipsae"] > 0.75 and aux["plddt"] > 0.8
        and aux["binder_target_pae"] < 5.0 and aux["target_binder_pae"] < 5.0
        and aux["recovery"] > 0.6
    )

# fitness function for hallucination
def loss(sequence, key=kval, context=None, params=None):
    keys = jax.random.split(key, 4)
    # forbid cysteines
    sequence = sequence.at[:, aas.AF2_CODE.index("C")].set(0.0)
    sequence = sequence / sequence.sum(axis=1, keepdims=True)
    # set sequence
    joltz_input: JoltzInput = params["joltz_input"].set_aa(sequence)
    # predict structure
    if "joltz" in params:
        result: JoltzResult = joltz(params["joltz"]).predict(keys[0], joltz_input, num_samples=4)
        # res_data = result.to_data(return_samples=True)
    elif "af2" in params:
        target_sequence = jax.nn.one_hot(
                af_template_data.data["aatype"], 20, axis=-1)[binder_length:]
        af_data = af_template_data.update_sequence(
            jnp.concatenate((sequence, target_sequence), axis=0))
        af_data = af_data.block_diagonal(2)
        result: AFResult = af2(params["af2"], keys[0], af_data)
    # compute losses
    out = metrics(sequence, keys[1], result)
    # optionally include target binding loss
    if "off_input" in params:
        off_input: JoltzInput = params["off_input"].set_aa(sequence)
        off_target_result: JoltzResult = joltz(params["joltz"]).predict(keys[2], off_input, num_samples=4)
        out.update(off_target_metrics(sequence, keys[3], out["logits"], off_target_result))
    out["result"] = result

    # combination
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
    if "off_input" in params:
        value += (
            # also impacts on-target recovery
            #5.0 * out["off_target_recovery"].mean()
            - out["off_target_contacts"].mean()
            #- 0.2 * out["off_target_binder_pae"].mean()
            #- 0.2 * out["binder_off_target_pae"].mean()
            #+ 0.05 * out["off_target_iptm"].mean()
        )
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
    "L", "H", "E"
]
trajectory_keys = [k for k in common_keys]
if joltz_off is not None:
    trajectory_keys += [
        "off_target_contacts", "off_target_binder_pae",
        "off_target_iptm", "off_target_recovery"]
trajectory = ScoreCSV(f"{opt.out_path}/trajectory.csv", keys=["attempt", "step"] + trajectory_keys)
results_keys = [
    "attempt", "stage", "sequence"
] + common_keys
results = ScoreCSV(f"{opt.out_path}/results.csv", keys=results_keys)
success = ScoreCSV(f"{opt.out_path}/success.csv", keys=results_keys)

success_count = 0
attempt = 0
_attempt = 0
while success_count < opt.num_designs:
    attempt = _attempt
    _attempt += 1
    context = dict(
        recovery=10.0,
        ipae=0.05,
    )
    if sampler is not None:
        salad_data["dssp_condition"], dssp_string = si.random_dssp(
            num_aa, p=0.0, p_keep_loop=1.0, return_string=True)
        print("DSSP target:", dssp_string)
        salad_data["cloud_std"] = cloud_std_default(num_aa)
        salad_design = sampler(salad_params, key(), salad_data, init_prev)
        design = data_from_salad(salad_design)
        design.save_pdb(f"{opt.out_path}/blueprints/design_{attempt}.pdb")
        sequence = aas.translate_onehot(
            jit_pmpnn(pmpnn_params, key(), design.drop_aa())["logits"],
            aas.PMPNN_CODE, aas.AF2_CODE)[..., :20]
        sequence = jax.nn.softmax(sequence, axis=-1)
    else:
        sequence = jax.random.gumbel(key(), (opt.length, 20), dtype=jnp.float32)
        sequence *= np.random.uniform(low=0.75, high=5.0)
        sequence = jnp.array(sequence, dtype=jnp.float32)
        sequence = jax.nn.softmax(sequence, axis=-1)
    sequence, loss_val, aux = simplex_agpm(
        sequence, loss_update, key=key, params=params, context=context,
        num_steps=100, # 100
        lr=0.1 * jnp.sqrt(sequence.shape[0]), momentum=0.3,
        scale=1.0,
        grad_transform=optax.clip_by_global_norm(1.0),
        param_transform=lambda x: x.at[:, aas.AF2_CODE.index("C")].set(0.0),
        verbose=True, return_last=False,
        early_stop=None,
        report=_report_trajectory(attempt, trajectory, start_step=0))
    sequence_1 = sequence
    result = aux["result"]
    if opt.hallucination_model == "af2":
        result.save_pdb(f"{opt.out_path}/attempts/raw_{attempt}_0.pdb")
    data = result.to_data()
    prediction_1 = joltz_pred(
        key(),
        JoltzSpec()
            .add_protein(data.to_sequence_string()[:binder_length])
            .add_protein(*[c for c in target_sequence.split(":")], use_msa=True))
    stage_1 = _report_result(attempt, "stage_1", key(), results, sequence, prediction_1.result)
    prediction_1.save_pdb(f"{opt.out_path}/attempts/design_{attempt}_1.pdb")
    sequence = jnp.log(sequence + 1e-5)
    context["recovery"] = 10.0
    sequence, loss_val, aux = simplex_agpm(
        sequence, loss_update, key=key, params=params, context=context,
        num_steps=50, # 50
        lr=0.5 * jnp.sqrt(sequence.shape[0]),
        scale=1.25,
        grad_transform=optax.clip_by_global_norm(1.0),
        param_transform=lambda x: x.at[:, aas.AF2_CODE.index("C")].set(-1e6),
        verbose=True, return_last=True, opt_logits=True,
        report=_report_trajectory(attempt, trajectory, start_step=100))
    sequence_2 = sequence
    result = aux["result"]
    if opt.hallucination_model == "af2":
        result.save_pdb(f"{opt.out_path}/attempts/raw_{attempt}_1.pdb")
    data = result.to_data()
    prediction_2 = joltz_pred(
        key(),
        JoltzSpec()
            .add_protein(data.to_sequence_string()[:binder_length])
            .add_protein(*[c for c in target_sequence.split(":")], use_msa=True))
    stage_2 = _report_result(attempt, "stage_2", key(), results, sequence, prediction_2.result)
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
        _report_result(attempt, stage, key(), success, sequence, prediction.result, aux=aux)
        prediction.save_pdb(f"{opt.out_path}/success/design_{attempt}_{stage}.pdb")
        success_count += 1
