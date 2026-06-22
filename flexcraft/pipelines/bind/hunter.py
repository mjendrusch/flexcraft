# Protein Hunter-like binder design script

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
from flexcraft.utils import Keygen, parse_options, load_pdb, strip_aa, tie_homomer
from flexcraft.sequence.sample import *
from flexcraft.data.data import DesignData
from flexcraft.sequence.mpnn import make_pmpnn
import flexcraft.sequence.aa_codes as aas
from flexcraft.hallucination.opt import jit_loss_update, simplex_agpm
from flexcraft.structure.boltz._model import Joltz2, JoltzResult, Joltz2Writer
from flexcraft.structure.boltz._data import JoltzSpec, JoltzInput
from flexcraft.structure.boltz._result import JoltzPrediction
from flexcraft.structure.af import AFInput, AFResult, make_af2, make_predict
from flexcraft.files.csv import ScoreCSV

opt = parse_options(
    "predict structures with AlphaFold",
    pmpnn_path="params/solmpnn/v_48_030.pkl",
    target_sequence="none",
    use_msa="True",
    out_path="out",
    boltz_path="params/boltz/",
    num_designs=48,
    length=80,
    cycles=10,
    temperature=0.1,
    samples=10,
    seed=42,
    tmpdir="tmp/"
)
binder_length = opt.length
key = Keygen(opt.seed)
kval = key()
# set up protein mpnn
pmpnn = make_pmpnn(opt.pmpnn_path, eps=0.05, split_params=False)
params = dict()
# set up logit transform:
transform = lambda T: transform_logits((
    scale_by_temperature(T),
    forbid("C", aas.PMPNN_CODE),
    norm_logits
))
# retrieve target
target_sequence = opt.target_sequence
# set up optional template
target_template = None
if opt.use_msa == "False":
    os.makedirs(opt.tmpdir, exist_ok=True)
    target_template = f"{opt.tmpdir}/target_{str(uuid.uuid4())}.cif"
# set up Boltz models
model = Joltz2()
# unknown-aa hallucination model
joltz, joltz_params = model.evaluator(num_recycle=4)
init_input, writer = (JoltzSpec()
    .add_protein("X" * binder_length)
    .add_protein(*[c for c in target_sequence.split(":")], use_msa=True).to_input(pad=True, cache=opt.boltz_path))
params["joltz"] = joltz_params

def _hunter_step(params, key, data: JoltzInput) -> JoltzResult:
    result: JoltzResult = joltz(params).predict(key, data)
    return result
hunter_step = jax.jit(_hunter_step)

def protein_hunter(key, cycles=5):
    def _init_sequence():
        init_sequence = jax.random.gumbel(key(), (opt.length, 20), dtype=jnp.float32)
        init_sequence *= np.random.uniform(low=0.75, high=5.0)
        init_sequence = jnp.array(init_sequence, dtype=jnp.float32)
        init_sequence = jax.nn.softmax(0.01 * init_sequence, axis=-1) # FIXME
        return init_sequence
    def _pmpnn(data: DesignData):
        data = data.update(aa=data["aa"].at[:binder_length].set(20))
        data = data.update(aa=aas.translate(data["aa"], aas.AF2_CODE, aas.PMPNN_CODE))
        temperature = 0.1
        logit_transform = transform(temperature)
        pmpnn_sampler = sample(pmpnn, logit_transform=logit_transform)
        pmpnn_result, _ = pmpnn_sampler(key(), data)
        pmpnn_result = data.update(
            aa=aas.translate(pmpnn_result["aa"], aas.PMPNN_CODE, aas.AF2_CODE))
        return pmpnn_result.to_sequence_string().split(":")[0]
    def _inner(binder_length: int, template=None, use_msa=True, report=None) -> JoltzResult:
        spec = (
            JoltzSpec()
            .add_protein(binder_length * "X")
            .add_protein(*[c for c in target_sequence.split(":")])
        )
        if template is not None:
            spec = spec.add_template(template, to_chains=[
                "BCDEFGHIJKLMNOPQRSTUVWXYZ"[i] 
                for i, c in enumerate(target_sequence.split(":"))])
        joltz_input, joltz_writer = spec.to_input(pad=True, cache=opt.boltz_path)
        joltz_input: JoltzInput
        if use_msa:
            joltz_input = joltz_input.inherit_msa(init_input)
        init_sequence = _init_sequence()
        binder_sequence = aas.decode(np.argmax(init_sequence, axis=-1), aas.AF2_CODE)
        #joltz_input = joltz_input.set_aa(init_sequence)
        result: JoltzResult = hunter_step(joltz_params, key(), joltz_input)
        pred = JoltzPrediction(data=result.data, writer=joltz_writer)
        if report is not None:
            report(0, binder_sequence, pred)
        for c in range(cycles - 1):
            binder_sequence = _pmpnn(result.to_data())
            spec = (
                JoltzSpec()
                .add_protein(binder_sequence)
                .add_protein(*[c for c in target_sequence.split(":")])
            )
            if template is not None:
                spec = spec.add_template(template, to_chains=[
                    "BCDEFGHIJKLMNOPQRSTUVWXYZ"[i] 
                    for i, c in enumerate(target_sequence.split(":"))])
            joltz_input, joltz_writer = spec.to_input(pad=True, cache=opt.boltz_path)
            if use_msa:
                joltz_input = joltz_input.inherit_msa(init_input)
            result: JoltzResult = hunter_step(joltz_params, key(), joltz_input)
            pred = JoltzPrediction(data=result.data, writer=joltz_writer)
            if report is not None:
                report(c + 1, binder_sequence, pred)
        return JoltzPrediction(data=result.data, writer=joltz_writer)
    return _inner

def _report_trajectory(index: int, writer: ScoreCSV, start_step=0):
    def _report(step: int, sequence: str, pred: JoltzPrediction):
        result = pred.result
        line_info = dict(
            attempt=index, step=start_step + step,
            sequence=sequence,
            plddt=result.plddt[:binder_length].mean(),
            iptm=result.iptm)
        pred.save_pdb(f"{opt.out_path}/attempts/design_{index}_{step}.pdb")
        writer.write_line(line_info)
    return _report

os.makedirs(f"{opt.out_path}/attempts/", exist_ok=True)
os.makedirs(f"{opt.out_path}/final/", exist_ok=True)
common_keys = [
    "attempt",
    "step",
    "sequence",
    "plddt",
    "iptm",
]
trajectory_keys = [k for k in common_keys]
trajectory = ScoreCSV(f"{opt.out_path}/trajectory.csv", keys=["attempt", "step"] + trajectory_keys)
results_keys = [
    "attempt", "stage", "sequence"
] + common_keys

hunter = protein_hunter(key, cycles=opt.cycles)
success_count = 0
attempt = 0
_attempt = 0
while success_count < opt.num_designs:
    attempt = _attempt
    _attempt += 1

    prediction = hunter(
        binder_length,
        use_msa=opt.use_msa == "True",
        template=target_template,
        report=_report_trajectory(attempt, trajectory))
    success_count += 1
