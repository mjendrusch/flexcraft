import jax
import jax.numpy as jnp
import numpy as np
import random

import salad.inference as si
from flexcraft.data.data import DesignData
from flexcraft.structure.af import AFInput, AFResult
from flexcraft.utils.io import data_from_salad
import flexcraft.sequence.aa_codes as aas
from flexcraft.sequence.sample import *

def beam_search(sampler: si.Sampler, fitness,
                beam_width=4, expansion=4,
                beam_step=100, out_steps=400):
    def _rollout(params, key, data, current_step):
        sampler.return_prev = True
        sampler.start_steps = current_step
        sampler.out_steps = [current_step + 100, out_steps]
        init_data, init_prev = data
        outs, prevs = sampler(params, key(), init_data, init_prev)
        score, success, prediction = fitness(outs[-1])
        # FIXME: use prevs here?
        return score, success, dict(prediction=prediction, data=(outs[-1], init_prev)) # FIXME: intermediate step or full rollout?
    def _search_step(params, key, instances, successes, current_step, log_to=None):
        results = []
        for j, (score, _, data) in enumerate(instances):
            for i in range(expansion):
                score, success, rollout_data = _rollout(params, key, data["data"], current_step)
                if log_to is not None:
                    data_from_salad(rollout_data["data"][0]).save_pdb(f"{log_to}_search_{j}_{i}.pdb")
                print("step", current_step, "instance", f"{j}_{i}", score, success)
                results.append((score, success, rollout_data))
                if success:
                    successes.append((score, success, rollout_data))
        # sort results by score
        results = instances + results
        results = sorted(results, key=lambda x: x[0])[:beam_width]
        return results
    def _init_instances(data, prev):
        instances = []
        for i in range(beam_width):
            score = 10_000
            rollout = dict(prediction=data, data=(data, prev))
            instances.append((score, False, rollout))
        return instances
    def _run_search(params, key, data, prev, log_to=None):
        current_step = 0
        instances = _init_instances(data, prev)
        successes = []
        for i in range(out_steps // beam_step):
            instances = _search_step(params, key, instances, successes, current_step, log_to=log_to)
            current_step += beam_step
        return successes, instances
    return _run_search

# TODO
def genetic_search(proposal, fitness, init, population_size=4, expansion=4, steps=100):
    def _rollout(data, current_step):
        out: DesignData = proposal(data, step=current_step)
        score, success, prediction = fitness(out)
        return score, success, dict(prediction=prediction, data=out)
    def _search_step(instances, successes, current_step, report=None):
        results = []
        for j, (score, _, data) in enumerate(instances):
            for i in range(expansion):
                score, success, rollout_data = _rollout(data["data"], current_step)
                if report is not None:
                    report(rollout_data)
                print("step", current_step, "instance", f"{j}_{i}", score, success)
                results.append((score, success, rollout_data))
                if success:
                    successes.append((score, success, rollout_data))
        # sort results by score
        results = instances + results
        results = sorted(results, key=lambda x: x[0])[:population_size]
        return results
    def _init_instances(data):
        instances = []
        for i in range(population_size):
            score = 10_000
            rollout = dict(prediction=data, data=init(data))
            instances.append((score, False, rollout))
        return instances
    def _run_search(data, report=None):
        current_step = 0
        instances = _init_instances(data)
        successes = []
        for i in range(steps):
            instances = _search_step(
                instances, successes,
                current_step, report=report)
            current_step += 1
        return successes, instances
    return _run_search

def salad_proposal(salad_config, salad_params, pmpnn, keygen, step, **kwargs):
    sampler = si.Sampler(step, **kwargs)
    transform = lambda center, do_center, T: transform_logits((
        toggle_transform(
            center_logits(center), use=do_center),
        scale_by_temperature(T),
        forbid("C", aas.PMPNN_CODE),
        norm_logits
    ))
    def _proposal(data, step=0):
        init_prev = dict(
            pos=jnp.zeros_like(data["pos"]),
            local=jnp.zeros((data["pos"].shape[0], salad_config.local_size)))
        start_steps = random.choice([0, 100, 200, 300, 350])
        salad_design = sampler(salad_params, keygen(), data, init_prev, start_steps=start_steps)
        design = data_from_salad(salad_design)
        aa_condition = aas.translate(data["aa_condition"], aas.AF2_CODE, aas.PMPNN_CODE)
        pmpnn_input = design.update(aa=aa_condition)
        logit_center = pmpnn(keygen(), pmpnn_input)["logits"][~data["is_target"]].mean(axis=0)
        temperature = 0.1#random.choice((0.01, 0.1, 0.2))
        do_center_logits = random.choice((True, False))
        logit_transform = transform(logit_center, do_center_logits, temperature)
        pmpnn_sampler = sample(pmpnn, logit_transform=logit_transform)
        pmpnn_result, _ = pmpnn_sampler(keygen(), pmpnn_input)
        pmpnn_result = pmpnn_input.update(
            aa=aas.translate(pmpnn_result["aa"], aas.PMPNN_CODE, aas.AF2_CODE))
        salad_design["aa_gt"] = pmpnn_result["aa"]
        salad_design["aa"] = pmpnn_result["aa"]
        salad_design["aatype"] = pmpnn_result["aa"]
        return salad_design
    return _proposal

def genetic_binder_fitness(af2, af_params, keygen):
    def _inner(design):
        is_target = design["is_target"] > 0
        is_binder = ~is_target
        data = data_from_salad(design)
        af_input = (
            AFInput
            .from_data(data)
            .add_guess(data)
            .add_template(data, where=is_target)
        )
        num_chains = len(np.unique(design["chain_index"]))
        af_input = af_input.block_diagonal(num_sequences=num_chains)
        af_result: AFResult = af2(af_params, keygen(), af_input)
        af_result.inputs["chain_index"] = is_binder.astype(jnp.int32)
        fitness = af_result.ipae
        success = (af_result.ipae <= 0.2) * (af_result.plddt[is_binder].mean() > 0.8)
        print(fitness, result.to_sequence_string().split(":")[-1])
        return fitness, success, af_result.to_data()
    return _inner

def binder_fitness(pmpnn, transform, af2, af_params, key):
    def _inner(design):
        is_target = design["is_target"] > 0
        is_binder = ~is_target
        pmpnn_input = data_from_salad(design)
        pmpnn_input = pmpnn_input.update(aa=aas.translate(pmpnn_input.aa, aas.AF2_CODE, aas.PMPNN_CODE))
        pmpnn_input = pmpnn_input.drop_aa(where=is_binder)
        logit_center = pmpnn(key(), pmpnn_input)["logits"][is_binder].mean(axis=0)
        logit_transform = transform(logit_center, True, 0.01)
        pmpnn_sampler = sample(pmpnn, logit_transform=logit_transform)
        pmpnn_result, _ = pmpnn_sampler(key(), pmpnn_input)
        pmpnn_result = pmpnn_input.update(
            aa=aas.translate(pmpnn_result["aa"], aas.PMPNN_CODE, aas.AF2_CODE))
        af_input = (
            AFInput
            .from_data(pmpnn_result)
            .add_guess(pmpnn_result)
            .add_template(pmpnn_result, where=is_target)
        )
        num_chains = len(np.unique(design["chain_index"]))
        af_input = af_input.block_diagonal(num_sequences=num_chains)
        af_result: AFResult = af2(af_params, key(), af_input)
        af_result.inputs["chain_index"] = is_binder.astype(jnp.int32)
        fitness = af_result.ipae
        success = (af_result.ipae <= 0.25) * (af_result.plddt[is_binder].mean() > 0.8)
        result = af_result.to_data()
        print(fitness, result.to_sequence_string().split(":")[-1])
        return fitness, success, af_result.to_data()
    return _inner
