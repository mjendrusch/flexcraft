from typing import List
import numpy as np

import jax
import jax.numpy as jnp


def scale_by_temperature(temperature: float | List[float] = 0.1):
    def inner(logits, data):
        return logits / temperature

    return inner


def center_logits(center=None):
    def inner(logits, data):
        cc = center
        if cc is None:
            cc = logits.mean(axis=0)
        return jax.nn.log_softmax(logits - cc, axis=-1)

    return inner


def forbid(amino_acids, aa_code):
    forbidden_index = np.array([aa_code.index(c) for c in amino_acids], dtype=np.int32)
    forbidden_mask = np.zeros((len(aa_code) + 1,), dtype=np.bool_)
    forbidden_mask[forbidden_index] = True

    def inner(logits, data):
        return jnp.where(forbidden_mask[None, :], -1e9, logits)

    return inner


def tie_logits(logits, data):
    tie_index = data["tie_index"]
    tie_weights = data["tie_weights"]
    logits = tie_weights[:, None] * logits
    # index-mean logits
    logit_sum = jnp.zeros_like(logits).at[tie_index].add(logits)
    # logit_denominator = jnp.maximum(jnp.zeros_like(logits).at[tie_index].add(1.0), 1e-6)
    logits = logit_sum[tie_index]#(logit_numerator / logit_denominator)[tie_index]
    return logits


def tie_update(aatype, position, update, data):
    tie_index = data["tie_index"]
    tie_mask = tie_index == tie_index[position]
    return jnp.where(tie_mask, update, aatype)


def transform_logits(transforms):
    def inner(logits, data):
        for transform in transforms:
            logits = transform(logits, data)
        return logits

    return inner


def toggle_transform(x, use=True):
    if use:
        return x
    return lambda logits, data: logits


def sample(model, select_next=None, logit_transform=None):
    if select_next is None:
        select_next = select_random_position

    def inner(key, data):
        log_p = 0.0
        # copy, don't write all over it
        data = {k: v for k, v in data.items()}
        data["aa"] = jnp.copy(data["aa"])
        while (data["aa"] == 20).any():
            key, subkey = jax.random.split(key, 2)
            # run MPNN on data
            res = model(subkey, data)
            logits = res["logits"]
            logits = tie_logits(logits, data)
            raw_logits = logits
            if logit_transform is not None:
                logits = logit_transform(logits, data)
            key, subkey = jax.random.split(key, 2)
            next_pos = select_next(subkey, logits, data)
            key, subkey = jax.random.split(key, 2)
            sampled = jax.random.categorical(subkey, logits=logits[next_pos])
            log_p += raw_logits[next_pos, sampled]
            data["aa"] = tie_update(data["aa"], next_pos, sampled, data)
        return data, log_p

    return inner


def select_random_position(key, logits, data):
    filled = data["aa"] != 20
    next_map = jnp.where(filled, jnp.inf, jax.random.uniform(key, (logits.shape[0],)))
    next_pos = jnp.argmin(next_map)
    return next_pos

def slice_dict(x, index):
    return {
        k: v[index]
        for k, v in x.items()
    }

def replace_value(x, **kwargs):
    result = {k: v for k, v in x.items()}
    for k, v in kwargs.items():
        result[k] = v
    return result

def norm_logits(logits, data):
    return jax.nn.log_softmax(logits, axis=-1)

if __name__ == "__main__":
    from flexcraft.sequence.mpnn import make_pmpnn
    import flexcraft.sequence.aa_codes as aas
    from flexcraft.utils import Keygen, parse_options, load_pdb, strip_aa, tie_homomer

    opt = parse_options(
        "Protein sequence design with custom protein MPNN",
        pdb_path="test.pdb",
        param_path="../prosesame/v_48_030.pkl",
        homomer=1,
        temperature=0.1,
        center="False",
        samples=10,
        seed=42
    )


    model = jax.jit(make_pmpnn(opt.param_path, eps=0.05))

    key = Keygen(opt.seed)
    data = load_pdb(opt.pdb_path)
    data = tie_homomer(data, opt.homomer)
    data = strip_aa(data)
    center = model(key(), data)["logits"].mean(axis=0)

    transform = transform_logits([
        toggle_transform(
            center_logits(center=center), use=opt.center == "True"),
        scale_by_temperature(temperature=opt.temperature),
        #forbid("C", aas.PMPNN_CODE),
        norm_logits
    ])
    sampler = sample(model, logit_transform=transform)
    for idx in range(opt.samples):
        result, log_p = sampler(key(), data)
        score_data = replace_value(data, aa=result["aa"])
        logits = model(key(), score_data)["logits"]
        if opt.center == "True":
            logits = jax.nn.log_softmax(logits - center, axis=-1)
        logprobs = logits[jnp.arange(result["aa"].shape[0]), result["aa"]]
        perplexity = np.exp(-logprobs.mean())
        print(f"> {float(perplexity):.2f} {float(np.exp(-log_p / result['aa'].shape[0])):.2f}")
        print(aas.decode(result["aa"], aas.PMPNN_CODE)[:data["aa"].shape[0] // opt.homomer])