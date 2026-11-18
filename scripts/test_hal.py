import os
import time

import jax
import jax.numpy as jnp

from flexcraft.sequence.mpnn import make_pmpnn
from flexcraft.structure.af import (
    AFInput,
    AFResult,
    forbid_sequence,
    get_model_haiku_params,
    make_af2,
    make_predict,
    model_config,
    soft_sequence,
)
from flexcraft.utils import Keygen, parse_options


def main():
    opt = parse_options(
        "predict structures with AlphaFold",
        param_path="params/af/",
        pmpnn_path="params/pmpnn/v_48_030.pkl",
        model_name="model_1_ptm",
        out_path="out",
        center="False",
        length=50,
        repeat=1,
        temperature=0.1,
        samples=10,
        seed=42,
    )
    params = get_model_haiku_params(
        model_name=opt.model_name, data_dir=opt.param_path, fuse=True
    )
    config = model_config(opt.model_name)
    config.model.global_config.use_dgram = False
    config.model.global_config.use_remat = True
    af2 = make_predict(make_af2(config), num_recycle=2)
    key = Keygen(opt.seed)
    _ = jax.jit(make_pmpnn(opt.pmpnn_path, eps=0.05))

    def fitness(af_params, step_key, sequence, soft=0.0, hard=0.0, T=1.0):
        sequence = soft_sequence(sequence, soft=soft, hard=hard, temperature=T)
        sequence = forbid_sequence(sequence, value=0)
        repeat = jnp.concatenate(opt.repeat * [sequence], axis=0)
        data = AFInput.from_sequence(repeat)
        result: AFResult = af2(af_params, step_key, data)
        pad_mask = jnp.asarray(result.token_pad_mask, dtype=jnp.bool_)
        resi = result.inputs["residue_index"]
        valid_pairs = (abs(resi[:, None] - resi[None, :]) >= 10) & pad_mask[:, None] & pad_mask[None, :]
        contact = result.contact_entropy(contact_distance=14.0)
        contact = jnp.where(valid_pairs, contact, 1e6)
        per_res_contact = -contact.sort(axis=1)[:, :2].mean(axis=-1)
        contact_sum = (per_res_contact * pad_mask).sum() / jnp.maximum(1, pad_mask.sum())
        metrics = dict(
            result=result,
            pae=result.mean_pae,
            plddt=result.mean_plddt,
            pcontact=-contact_sum,
        )
        value = contact_sum + metrics["plddt"] + (1 - metrics["pae"])
        return value, metrics

    def gradient_step(af_params, step_key, sequence, soft=0.0, hard=0.0, T=1.0):
        (value, aux), grad = jax.value_and_grad(
            fitness, argnums=(2,), has_aux=True
        )(af_params, step_key, sequence, soft, hard, T)
        lr = (1 - soft) + (soft * T)
        grad = grad[0]
        grad = forbid_sequence(grad, value=0.0)
        grad /= jnp.sqrt(jnp.maximum((grad**2).sum(), 1e-3))
        new_sequence = sequence + lr * grad
        return forbid_sequence(new_sequence, value=0), value, aux

    gradient_step = jax.jit(gradient_step)

    os.makedirs(opt.out_path, exist_ok=True)
    sequence = jax.nn.softmax(
        forbid_sequence(jax.random.gumbel(key(), (opt.length, 20)), -1e9)
    )
    best = -1e6
    for idx in range(100):
        soft = 0.0
        start = time.time()
        sequence, value, aux = gradient_step(params, key(), sequence, soft=soft)
        print(
            f"Scores at step {idx} in {time.time() - start:.1f} s",
            f"fitness: {value:.2f}",
            f"pae: {aux['pae']:.2f}",
            f"plddt: {aux['plddt']:.2f}",
            f"contact: {aux['pcontact']:.2f}",
        )
        if value > best:
            result: AFResult = aux["result"].unpad()
            best = value
            result.save_pdb(f"{opt.out_path}/step_{idx}.pdb")


if __name__ == "__main__":
    main()
