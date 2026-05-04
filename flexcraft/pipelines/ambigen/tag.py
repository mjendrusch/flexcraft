import os
from copy import deepcopy
import shutil
import time

import jax
import jax.numpy as jnp
import numpy as np

import haiku as hk

import salad.inference as si
from salad.modules.utils.geometry import index_count

import flexcraft.sequence.aa_codes as aas
from flexcraft.sequence.mpnn import make_pmpnn
from flexcraft.sequence.sample import *
from flexcraft.structure.af import *
from flexcraft.utils import Keygen, parse_options, data_from_protein, data_from_salad
from flexcraft.files.pdb import PDBFile

from flexcraft.rosetta.relax import fastrelax
from flexcraft.files.csv import ScoreCSV
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
        # we can edit the structure before noise
        # is applied here:
        ...
        # apply noise
        data.update(noise(data))
        # and edit the noised structure here:
        ...
        # run model
        out, prev = predict(data, prev)
        # or the output of the model here:
        ...
        return out, prev
    # return the pure apply function generated
    # by haiku from our step
    return step.apply

def init_tag_pair(init_data, tag_sequence, num_disambiguating=10, num_padding=5):
    num_aa = init_data["residue_index"].shape[0]
    tag_code = aas.encode(tag_sequence, aas.AF2_CODE)
    num_tag = tag_code.shape[0]
    tag_dssp = np.random.randint(1, 3)
    tag_start = np.random.randint(num_disambiguating + num_padding, num_aa - num_padding - num_tag)
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
    tie_index_hidden[:num_disambiguating] = np.arange(num_aa, num_aa + num_disambiguating, dtype=np.int32)
    tie_index = np.concatenate((tie_index_exposed, tie_index_hidden), axis=0)
    tie_weight = 1 / jnp.maximum(index_count(tie_index, jnp.ones_like(tie_index, dtype=jnp.bool_)), 1)
    paired_tie_info = dict(tie_index=tie_index, tie_weight=tie_weight)
    unpaired_tie_info = dict(tie_index=np.arange(2 * num_aa, dtype=np.int32), tie_weight=np.ones((2 * num_aa,), dtype=np.float32))
    # set up PMPNN initialized sequences
    
    return dict(
        exposed=exposed_data,
        hidden=hidden_data,
        non_tag_mask=non_tag_mask,
        tie_info=paired_tie_info,
        unpaired_tie_info=unpaired_tie_info,
        pmpnn_aa=aas.translate(
            np.concatenate(2 * [tag_data["aa_condition"]], axis=-1),
            aas.AF2_CODE, aas.PMPNN_CODE)
    )

def mutate(sampler: si.Sampler, params, key):
    def _run(exposed, hidden, init_prev):
        start_steps = jax.random.randint(key(), (), 0, 4) * 100
        exposed = sampler(params, key(), exposed, init_prev, start_steps=start_steps)
        hidden = sampler(params, key(), hidden, init_prev, start_steps=start_steps)
        return exposed, hidden
    return _run

def evaluate(af2, af2_params, pmpnn_sampler):
    def _run(exposed, hidden, tag_pair_info):
        exposed = data_from_salad(exposed)
        hidden = data_from_salad(hidden)
        pmpnn_untied = DesignData.concatenate([exposed, hidden], sep_batch=True)
        pmpnn_untied = pmpnn_untied.update(
            aa=tag_pair_info["pmpnn_aa"],
            tie_index=tag_pair_info["unpaired_tie_info"]["tie_index"],
            tie_weight=tag_pair_info["unpaired_tie_info"]["tie_weight"])
        pmpnn_tied = pmpnn_untied.update(
           aa=tag_pair_info["pmpnn_aa"],
           tie_index=tag_pair_info["tie_info"]["tie_index"],
           tie_weight=tag_pair_info["tie_info"]["tie_weight"])
        # sample all non-fixed residues
        result_tied, _ = pmpnn_sampler(None, T)(key(), pmpnn_tied)
        # translate the amino acid order
        pmpnn_tied["aa"] = aas.translate(result_tied["aa"], aas.PMPNN_CODE, aas.AF2_CODE)
        pmpnn_result = pmpnn_tied.split(pmpnn_tied["batch_index"])

        modes = dict()
        for mode, pmpnn_out in zip(("exposed", "hidden"), pmpnn_result):
            af_input = AFInput.from_data(pmpnn_out).add_guess(pmpnn_out)
            mode_data = dict()
            mode_data["result"] = af2(af2_params, key(), af_input)
            mode_data["sequence"] = pmpnn_out.to_sequence_string()
            lddt_val = lddt(mode_data["result"], pmpnn_out)
            plddt_val = mode_data["result"].plddt
            mode_data["lddt"] = (lddt_val * non_tag_mask).sum() / num_non_tag
            mode_data["plddt"] = (plddt_val * non_tag_mask).sum() / num_non_tag
            mode_data["sc_lddt"] = (lddt_val * plddt_val * non_tag_mask).sum() / num_non_tag
            mode_data["sc_rmsd"] = RMSD()(mode_data["result"].to_data(), pmpnn_out,
                                          weight=non_tag_mask, eval_mask=non_tag_mask)
            mode_data["success"] = (mode_data["plddt"] > 0.75) and (mode_data["sc_rmsd"] < 3.0)
            print(idx, ids, mode, mode_data["sequence"], mode_data["plddt"], mode_data["sc_rmsd"])
            modes[mode] = mode_data
        score = modes["exposed"]["sc_lddt"] * modes["hidden"]["sc_lddt"]
        success = modes["exposed"]["success"] and mode["hidden"]["success"]
        return score, success, modes
    return _run

def search(salad_params, salad_sampler, af2, af2_params, pmpnn_sampler,
           key, population_size=4, multiplicity=4, num_steps=4):
    _eval = evaluate(af2, af2_params, pmpnn_sampler)
    _mut = mutate(salad_sampler, salad_params, key)
    def _init(exposed, hidden, init_prev, tag_pair_info):
        # initialize population
        population = [
            dict(
                score=0.0,
                success=False,
                exposed=salad_sampler(salad_params, key(), exposed, init_prev),
                hidden=salad_sampler(salad_params, key(), hidden, init_prev),
                aux=None
            )
            for _ in range(population_size * multiplicity)
        ]
        for item in population:
            score, success, aux = _eval(item["exposed"], item["hidden"], tag_pair_info)
            item["score"] = score
            item["success"] = success
            item["aux"] = aux
        population = sorted(population, key=lambda x: x["score"])[:population_size]
        return population

    def _run(exposed, hidden, init_prev, tag_pair_info):
        # initialize population
        population = _init(exposed, hidden, init_prev, tag_pair_info)
        for i in range(num_steps):
            tmp = []
            for item in population:
                for k in range(multiplicity):
                    tmp.append(_eval(*_mut(item), tag_pair_info))
            population = sorted(tmp, key=lambda x: x["score"])[:population_size]
        return population

    return _run


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
    seed=42
)
os.makedirs(f"{opt.out_path}/attempts", exist_ok=True)
os.makedirs(f"{opt.out_path}/success", exist_ok=True)

# set up output files
score_keys = (
    "attempt", "seq_id", "T", "center",
    "sequence", "sc_rmsd", "relax_rmsd", "lddt", "plddt", "sc_lddt", "success"
)

success = ScoreCSV(
    f"{opt.out_path}/success.csv", score_keys, default="none")
all_designs = ScoreCSV(
    f"{opt.out_path}/all.csv", score_keys, default="none")

key = Keygen(opt.seed)
# set up ProteinMPNN
pmpnn = jax.jit(make_pmpnn(opt.pmpnn_params, eps=0.05))
# set up logit transform
transform = lambda center, T: transform_logits([
    scale_by_temperature(temperature=T),
    forbid("C", aas.PMPNN_CODE),
    #tie_logits,
    norm_logits
])
# set up sampler
pmpnn_sampler = lambda center, T: sample(
    pmpnn, logit_transform=transform(center, T))

# make salad model
salad_config, salad_params = si.make_salad_model("default_vp", opt.salad_params)

# initialize salad data and prev from the num_aa specification
num_aa, resi, chain, is_cyclic, cyclic_mask = si.parse_num_aa(opt.num_aa)
salad_data, init_prev = si.data.from_config(
    salad_config,
    num_aa=num_aa,
    residue_index=resi,
    chain_index=chain,
    cyclic_mask=cyclic_mask)

salad_step = model_step(salad_config)

# make AF2 model
af2_params = get_model_haiku_params(
    model_name=opt.af2_model,
    data_dir=opt.af2_params, fuse=True)
af2_config = model_config(opt.af2_model)
af2_config.model.global_config.use_dgram = False
af2 = jax.jit(make_predict(make_af2(af2_config), num_recycle=2))

# 1 9 exposed GTKTRKVVAK VGDEEYEGTL TAESEVTAKLVTDKYGEVEVKLAGGLEEVTVAAGE YPYDVPDYA VEIGELKE 0.57304895 5.4396358
# 1 9 hidden  MESKTVTLAP NATITFKNGS TAESEVTAKLVTDKYGEVEVKLAGGLEEVTVAAGE YPYDVPDYA VEIGELKE 0.45158607 7.968858
# 0 9 exposed SLKEEIKKLVEEAAKKKKMEEEERKKVIEEVLKNVEKMTKELLKSLLENYPYDVPDYAKEKALYLREKLLLE 0.716066 10.444642
# 0 9 hidden  SLKEEIKKLVEEAAKKKKMEEEERKKVIEEVLKNVEKMTKELLKSLLENYPYDVPDYAKEKALYLREKLLLE 0.7401626 5.606064

# build a sampler object for sampling
sampler = si.Sampler(salad_step, out_steps=400)
# run a loop with num_design steps
print("Starting design...")
for idx in range(opt.num_designs):
    # generate a structure in each step
    design_info = dict(attempt=idx)
    start = time.time()
    tag_pair_info = init_tag_pair(
        salad_data, opt.tag,
        opt.num_disambiguating, opt.num_padding)
    non_tag_mask = tag_pair_info["non_tag_mask"]
    num_non_tag = num_aa - len(opt.tag)
    exposed = data_from_salad(
        sampler(salad_params, key(), tag_pair_info["exposed"], init_prev))
    hidden = data_from_salad(
        sampler(salad_params, key(), tag_pair_info["hidden"], init_prev))
    print(f"Design {idx} in {time.time() - start:.2f} s")
    exposed.save_pdb(f"{opt.out_path}/attempts/exposed_{idx}.pdb")
    hidden.save_pdb(f"{opt.out_path}/attempts/hidden_{idx}.pdb")
    # TODO: use genetic algorithm / search to reach better designs
    # # convert it to ProteinMPNN input
    # pmpnn_data = design.drop_aa()
    # step_0_logits = pmpnn(key(), pmpnn_data)["logits"]
    # center = step_0_logits.mean(axis=0)
    # # ensure one TRP/W residue
    # p_W = jax.nn.softmax(step_0_logits - center)[:, aas.PMPNN_CODE.index("W")]
    # best_W = jnp.argmax(p_W, axis=0)
    for ids in range(opt.num_samples):
        # anneal temperature
        T = 0.1
        # TODO search with sc_lddt
        design_info["seq_id"] = ids
        design_info["T"] = T
        design_info["center"] = True
        # prepare input
        pmpnn_untied = DesignData.concatenate([exposed, hidden], sep_batch=True)
        pmpnn_untied = pmpnn_untied.update(
            aa=tag_pair_info["pmpnn_aa"],
            tie_index=tag_pair_info["unpaired_tie_info"]["tie_index"],
            tie_weight=tag_pair_info["unpaired_tie_info"]["tie_weight"])
        # pmpnn_tied = pmpnn_untied # FIXME
        pmpnn_tied = pmpnn_untied.update(
           aa=tag_pair_info["pmpnn_aa"],
           tie_index=tag_pair_info["tie_info"]["tie_index"],
           tie_weight=tag_pair_info["tie_info"]["tie_weight"])
        # sample all non-fixed residues
        result_tied, _ = pmpnn_sampler(None, T)(key(), pmpnn_tied)
        result_untied, _ = pmpnn_sampler(None, T)(key(), pmpnn_tied)
        # ProteinMPNN has a different amino acid order than AF2
        # translate the amino acid order
        pmpnn_tied["aa"] = aas.translate(result_tied["aa"], aas.PMPNN_CODE, aas.AF2_CODE)
        pmpnn_result = pmpnn_tied.split(pmpnn_tied["batch_index"])

        results = dict()
        for mode, pmpnn_out in zip(("exposed", "hidden"), pmpnn_result):
            af_input = AFInput.from_data(pmpnn_out).add_guess(pmpnn_out)
            mode_data = dict()
            mode_data["result"] = af2(af2_params, key(), af_input)
            mode_data["sequence"] = pmpnn_out.to_sequence_string()
            mode_data["plddt"] = (mode_data["result"].plddt * non_tag_mask).sum() / num_non_tag
            mode_data["sc_rmsd"] = RMSD()(mode_data["result"].to_data(), pmpnn_out,
                                          weight=non_tag_mask, eval_mask=non_tag_mask)
            mode_data["success"] = (mode_data["plddt"] > 0.75) * (mode_data["sc_rmsd"] < 3.0)
            print(idx, ids, mode, mode_data["sequence"], mode_data["plddt"], mode_data["sc_rmsd"])
            if mode_data["success"]:
                mode_data["result"].save_pdb(f"{opt.out_path}/success/{mode}_{idx}_{ids}.pdb")

        # plddt = result.plddt.mean()
        # lddt_mean = lddt(result, pmpnn_data).mean()
        # residue_sc_lddt = sc_lddt(result, pmpnn_data)
        # sc_lddt_mean = residue_sc_lddt.mean()
        # sc_rmsd = RMSD(atoms=["CA"])(result.to_data(), pmpnn_data)
        # design_info.update(dict(
        #     sc_rmsd=sc_rmsd, sc_lddt=sc_lddt_mean, lddt=lddt_mean, plddt=plddt
        # ))
        # print(idx, ids, sequence, plddt, lddt_mean, sc_lddt_mean, sc_rmsd)
        # if sc_lddt_mean > 0.7:
        #     pdb = PDBFile(
        #         result.to_data().update(plddt=residue_sc_lddt),
        #         path=f"{opt.out_path}/success/design_{idx}_{ids}.pdb")
        #     relaxed = fastrelax(pdb, f"{opt.out_path}/relaxed/design_{idx}_{ids}.pdb")
        #     relax_rmsd = RMSD(atoms=["CA"])(pdb.to_data(), relaxed.to_data(), mask=None)
        #     design_info["relax_rmsd"] = relax_rmsd
        #     design_info["success"] = True
        #     print(f"relaxed RMSD {relax_rmsd:.2f} A")
        #     success.write_line(design_info)
        #     all_designs.write_line(design_info)
        #     break
        # else:
        #     all_designs.write_line(design_info)

