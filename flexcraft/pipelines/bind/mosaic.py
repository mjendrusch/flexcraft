# Binder design protocol adapted from escalante-bio blogpost:
# https://blog.escalante.bio/teaching-generative-models-to-hallucinate/

import os
import time
import jax
import haiku as hk

from colabdesign.af.alphafold.model.config import model_config
from colabdesign.af.alphafold.model.data import get_model_haiku_params

import optax

from flexcraft.utils.options import parse_options
from flexcraft.utils.rng import Keygen
from flexcraft.utils import Keygen, parse_options, load_pdb, strip_aa, tie_homomer
from flexcraft.sequence.sample import *
from flexcraft.data.data import DesignData
from flexcraft.sequence.mpnn import make_pmpnn
import flexcraft.sequence.aa_codes as aas
from flexcraft.hallucination.opt import jit_loss_update, simplex_agpm
from flexcraft.structure.boltz._model import Joltz2, JoltzResult
from flexcraft.structure.af import AFInput, AFResult, make_af2, make_predict
from flexcraft.files.csv import ScoreCSV

opt = parse_options(
    "predict structures with AlphaFold",
    pmpnn_path="params/solmpnn/v_48_030.pkl",
    target_path="target.pdb",
    target_chains="all",
    out_path="out",
    center="False",
    hallucination_model="joltz",
    model_name="model_1_ptm",
    param_path="params/af",
    num_designs=48,
    length=80,
    repeat=1,
    temperature=0.1,
    samples=10,
    seed=42
)
binder_length = opt.length
key = Keygen(opt.seed)
kval = key()
# set up protein mpnn
pmpnn, pmpnn_params = make_pmpnn(opt.pmpnn_path, eps=0.05, split_params=True)
params = dict(
    mpnn=pmpnn_params
)
# retrieve target
target = load_pdb(opt.target_path)
if opt.target_chains != "all":
    selected_chains = np.array([
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ".index(c) 
        for c in opt.target_chains], dtype=np.int32)
    target = target[(target.chain_index[:, None] == selected_chains).any(axis=-1)]
target_sequence = target.to_sequence_string()
# set up Boltz models
model = Joltz2()
# unknown-aa hallucination model
if opt.hallucination_model == "joltz":
    joltz, joltz_params, writer = model.evaluator(
        dict(kind="protein", sequence="X" * binder_length),
        dict(kind="protein", sequence=target_sequence, use_msa=True),
        num_recycle=4)
    params["joltz"] = joltz_params
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
# full-atom predictor with context
joltz_pred = model.predictor(
    dict(kind="protein", sequence=target_sequence, use_msa=True),
    num_recycle=4)

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
    pmpnn_input = res_data
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
    return dict(
        plddt = plddt[:binder_length].mean(),
        recovery = (sequence * prob).sum(axis=-1).mean(),
        binder_target_contacts = result.chain_contact_score(1, 0, num_contacts=3, contact_distance=20.0),
        within_binder_contacts = result.chain_contact_score(0, 0, num_contacts=25, min_resi_distance=10),
        within_binder_pae = pae[:binder_length, :binder_length].mean(),
        binder_target_pae = pae[:binder_length, binder_length:].mean(),
        target_binder_pae = pae[binder_length:, :binder_length].mean(),
        iptm = result.iptm,
        eptm = result.ptm_score,
        ipsae = result.ipsae(chain_index=is_binder),
        average_num_nonzero = (sequence > 0.01).sum(axis=-1).mean(),
    )

def _report_trajectory(index: int, writer: ScoreCSV, start_step=0):
    def _report(step: int, loss: float, aux: dict):
        line_info = dict(attempt=index, step=start_step + step, total=loss)
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
    # forbid cysteines
    sequence = sequence.at[:, aas.AF2_CODE.index("C")].set(0.0)
    sequence = sequence / sequence.sum(axis=1, keepdims=True)
    # predict structure
    if "joltz" in params:
        result: JoltzResult = joltz(params["joltz"]).predict(key, sequence, num_samples=4)
        # res_data = result.to_data(return_samples=True)
    elif "af2" in params:
        target_sequence = jax.nn.one_hot(
                af_template_data.data["aatype"], 20, axis=-1)[binder_length:]
        af_data = af_template_data.update_sequence(
            jnp.concatenate((sequence, target_sequence), axis=0))
        af_data = af_data.block_diagonal(2)
        result: AFResult = af2(params["af2"], key, af_data)
    # compute losses
    out = metrics(sequence, key, result)
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
    out["total"] = value
    return value, out
loss_update = jit_loss_update(loss)

os.makedirs(f"{opt.out_path}/attempts/", exist_ok=True)
os.makedirs(f"{opt.out_path}/success/", exist_ok=True)
trajectory_keys = [
    "plddt",
    "recovery",
    "binder_target_contacts",
    "within_binder_contacts",
    "within_binder_pae",
    "binder_target_pae",
    "target_binder_pae",
    "iptm",
    "eptm",
    "ipsae",
    "average_num_nonzero",
]
trajectory = ScoreCSV(f"{opt.out_path}/trajectory.csv", keys=["attempt", "step"] + trajectory_keys)
results_keys = [
    "attempt", "stage", "sequence"
] + trajectory_keys
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
    prediction_1 = joltz_pred(key(), dict(kind="protein", sequence=data.to_sequence_string()[:binder_length]))
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
    prediction_2 = joltz_pred(key(), dict(kind="protein", sequence=data.to_sequence_string()[:binder_length]))
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
