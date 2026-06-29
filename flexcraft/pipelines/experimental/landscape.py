
import os
import time
from copy import deepcopy, copy

import random
import optax
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
from flexcraft.utils import Keygen, parse_options, data_from_salad
from flexcraft.files import ScoreCSV, PDBFile
from flexcraft.structure.boltz import Joltz2, JoltzInput, JoltzResult, JoltzSpec
from flexcraft.hallucination.opt import jit_loss_update, simplex_agpm

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
        # apply noise
        data.update(noise(data))
        # run model
        out, prev = predict(data, prev)
        return out, prev
    # return the pure apply function generated
    # by haiku from our step
    return step.apply


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
    seed=42
)

def cloud_std_default(num_aa):
    minval = num_aa ** 0.4
    return minval + np.random.rand() * opt.std_buffer

os.makedirs(f"{opt.out_path}/predictions/", exist_ok=True)
os.makedirs(f"{opt.out_path}/relaxed/", exist_ok=True)
os.makedirs(f"{opt.out_path}/data/", exist_ok=True)

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
# build a sampler object for sampling
sampler = si.Sampler(salad_step, out_steps=400)

# set up params
_pmpnn, pmpnn_params = make_pmpnn(opt.pmpnn_params, eps=0.05, split_params=True)
params = dict(
    mpnn=pmpnn_params
)
# set up Joltz
joltz_model = Joltz2(cache=opt.boltz_path)
joltz, params["joltz"] = joltz_model.evaluator(num_recycle=4)
joltz_spec = JoltzSpec().add_protein("X" * num_aa)
params["joltz_input"], writer = joltz_spec.to_input(pad=False)
joltz_pred = joltz_model.predictor(num_recycle=4, num_samples=4)

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
    data = result.to_data(return_samples=True)
    dssp = data.p_dssp
    pmpnn_input = data.update(aa=aas.translate(data["aa"], aas.AF2_CODE, aas.PMPNN_CODE))
    logits = batch_mpnn(_pmpnn)(params["mpnn"], key, pmpnn_input.drop_aa())["logits"]
    logits = aas.translate_onehot(logits, aas.PMPNN_CODE, aas.AF2_CODE)[..., :20]
    if not result.is_single_sample:
        logits = logits.mean(axis=0)
    prob = jax.lax.stop_gradient(jax.nn.softmax(logits.at[:, aas.AF2_CODE.index("C")].set(-1e6) / 0.1, axis=-1))
    entropy = -(logits * prob).sum(axis=-1).mean()
    # losses
    plddt = result.plddt
    pae = 32 * result.pae # unnormalized pAE
    if not result.is_single_sample:
        plddt = plddt.mean(axis=0)
        pae = pae.mean(axis=0)
    return dict(
        logits = logits,
        entropy = entropy,
        plddt = plddt.mean(),
        recovery = (sequence * prob).sum(axis=-1).mean(),
        contacts = result.contact_score(num_contacts=25, min_resi_distance=10),
        pae = pae.mean(),
        average_num_nonzero = (sequence > 0.01).sum(axis=-1).mean(),
        L = dssp["L"], H = dssp["H"], E = dssp["E"]
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

# fitness function for hallucination
def loss(sequence, key=None, context=None, params=None):
    keys = jax.random.split(key, 4)
    # forbid cysteines
    sequence = sequence.at[:, aas.AF2_CODE.index("C")].set(0.0)
    sequence = sequence / sequence.sum(axis=1, keepdims=True)
    # set sequence
    joltz_input: JoltzInput = params["joltz_input"].set_aa(sequence)
    # predict structure
    result: JoltzResult = joltz(params["joltz"]).predict(keys[0], joltz_input, num_samples=4)
    # compute losses
    out = metrics(sequence, keys[1], result)
    out["result"] = result

    # combination
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
    "L", "H", "E"
]
trajectory_keys = [k for k in common_keys]
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
    salad_data["dssp_condition"], dssp_string = si.random_dssp(
        num_aa, p=0.0, p_keep_loop=1.0, return_string=True)
    print("DSSP target:", dssp_string)
    salad_data["cloud_std"] = cloud_std_default(num_aa)
    salad_design = sampler(salad_params, key(), salad_data, init_prev)
    design = data_from_salad(salad_design)
    design.save_pdb(f"{opt.out_path}/blueprints/design_{attempt}.pdb")
    sequence = aas.translate_onehot(
        pmpnn(key(), design.drop_aa())["logits"],
        aas.PMPNN_CODE, aas.AF2_CODE)[..., :20]
    sequence = jax.nn.softmax(sequence / opt.temperature, axis=-1)
    (_, aux), _ = loss_update(sequence, key=key(), params=params)
    aux_data: DesignData = aux["result"].to_data()
    aux_data.save_pdb(f"{opt.out_path}/attempts/design_{attempt}_init.pdb")
    sequence, loss_val, aux = simplex_agpm(
        sequence, loss_update, key=key, params=params, context=context,
        num_steps=50,
        lr=0.1 * jnp.sqrt(sequence.shape[0]), momentum=0.3,
        scale=1.0,
        grad_transform=optax.clip_by_global_norm(1.0),
        param_transform=lambda x: x.at[:, aas.AF2_CODE.index("C")].set(0.0),
        verbose=True, return_last=False,
        early_stop=None,
        report=_report_trajectory(attempt, trajectory, start_step=0))
    sequence_1 = sequence
    result = aux["result"]
    data = result.to_data()
    prediction_1 = joltz_pred(
        key(), JoltzSpec().add_protein(data.to_sequence_string()))
    stage_1 = _report_result(attempt, "stage_1", key(), results, sequence, prediction_1.result)
    prediction_1.save_pdb(f"{opt.out_path}/attempts/design_{attempt}.pdb")
    success_count += 1
