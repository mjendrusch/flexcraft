"""Graft one or more flexible peptides onto a design in a controlled manner."""

import os
import random
from copy import deepcopy
import jax
import jax.numpy as jnp
import haiku as hk
from collections import defaultdict

from flexcraft.pipelines.graft.common import (
    setup_directories, get_assembly_lengths, sample_data, pad_to_budget,
    make_peptide_assembly)
from flexcraft.pipelines.graft.options import MODEL_OPTIONS, SAMPLER_OPTIONS
from flexcraft.utils import *
from flexcraft.files.csv import ScoreCSV
from flexcraft.protocols.af2cycler import af_cycler
from flexcraft.sequence.mpnn import make_pmpnn
from flexcraft.sequence.sample import *
from flexcraft.structure.af import *
from flexcraft.structure.metrics import RMSD
from flexcraft.data.data import Block

import salad.inference as si

from salad.modules.utils.geometry import index_mean

def model_step(config, task):
    config = deepcopy(config)
    config.eval = True
    @hk.transform
    def step(data, prev):
        # set up salad models
        noise = si.StructureDiffusionNoise(config)
        predict = si.StructureDiffusionPredict(config)
        # get diffusion time
        t = data["t_pos"][0]
        # get motif information
        has_motif, group = data["has_motif"], data["motif_group"]
        # set up aa condition
        aatype = data["motif_aa"]
        aa_condition = jnp.where(has_motif, aas.AF2_CODE.index("S"), 20)
        if task["no_aa"]:
            aa_condition = jnp.full_like(aa_condition, 20)
        data["aa_condition"] = aa_condition
        # set up dssp condition
        data["dssp_condition"] = jnp.where(has_motif, 0, 3)
        pos = data["pos"]
        # get motif and protein centers of mass
        protein_center = pos[:, 1].mean(axis=0)
        motif_centers = index_mean(pos[:, 1], group, has_motif[:, None])
        # push non-motif inward
        push_non_motif = jnp.where(
            ~has_motif[:, None], protein_center - pos[:, 1], 0.0)
        # push motif centers apart
        other_group = group[:, None] != group[None, :]
        motif_pair = other_group * (has_motif[:, None] * has_motif[None, :]) > 0
        def motif_proximity_loss(ca):
            direction = (motif_pair[..., None] * (ca[:, None] - ca[None, :]))
            distance = jnp.sqrt(jnp.maximum((direction ** 2).sum(axis=-1), 1e-6))
            loss = (motif_pair * jax.nn.relu(task["inter_motif_distance"] - distance) ** 2).sum() / jnp.maximum(1, motif_pair.sum())
            loss *= group.max() ** 2
            return loss
        motif_proximity_step = -jax.grad(motif_proximity_loss, argnums=(0,))(motif_centers)[0]

        # apply updates in order:
        # first, compact the non-motif part of the protein
        pos = pos + t * task["push_non_motif"] * push_non_motif[:, None, :]
        # next, push motifs apart
        pos = pos + (t > task["motif_threshold"]) * task["push_apart"] * motif_proximity_step[:, None, :]
        
        # optionally align all-motif center of mass with the protein center
        # to ensure that motifs are distributed around the design and cannot
        # cluster to one side
        motif_centers = index_mean(pos[:, 1], group, has_motif[:, None])
        motifs_com = (motif_centers * has_motif[..., None]).sum(axis=0) / jnp.maximum(1, has_motif.sum())
        pos_com = pos.mean(axis=0)
        pos = pos - pos_com
        pos = jnp.where((t > task["centered_threshold"]) * has_motif[:, None, None], pos - motifs_com, pos)
        # finally, push motif centers outward to the specified radius
        motif_centers = index_mean(pos[:, 1], group, has_motif[:, None])
        motif_dir = jnp.where(
            has_motif[:, None], motif_centers - protein_center, 0.0)
        normed_motif_dir = motif_dir / jnp.maximum(1e-2, jnp.linalg.norm(motif_dir, axis=-1, keepdims=True))
        push_motif = task["motif_radius"] * normed_motif_dir - motif_dir
        pos = pos + (t > task["motif_threshold"]) * push_motif[:, None, :]
        
        # optionally symmetrize the design with cyclic symmetry
        if task["symmetrize"]:
            screw = si.symmetry.Screw(task["sym_count"], 360 / task["sym_count"], task["sym_radius"], task["sym_translation"])
            first = screw.select_pos(pos, do_radius=t > task["radius_threshold"])
            sym_pos = screw.replicate_pos(first, do_radius=t > task["radius_threshold"])
            threshold = t > task["sym_threshold"]
            pos = threshold * sym_pos + (1 - threshold) * pos
        data["pos"] = pos

        # run a single SALAD step
        # center positions
        data["pos"] = si.center_positions(
           data["pos"], data["batch_index"], data["mask"])
        # apply noise
        data.update(noise(data))
        # predict structure
        out, prev = predict(data, prev)
        return out, prev
    return step.apply


opt = parse_options(
    "Simple motif grafting with salad.",
    **MODEL_OPTIONS,
    **SAMPLER_OPTIONS,
    out_path="outputs/",
    motif_path="motifs/",
    peptide_length=12,
    peptide_count=1,
    peptides="MERRYCHRISTMAS",
    assembly="20-50,@0,20-50",
    assembly_budget="none",
    symmetrize="False",
    sym_count=1,
    sym_radius=12,
    sym_translation=0,
    h_bias=0.0,
    e_bias=0.0,
    l_bias=0.0,
    push_motif=1e-2,
    push_apart=0.0,
    motif_radius=10.0,
    motif_threshold=0.1,
    centered_threshold=2.0,
    inter_motif_distance=8.0,
    sym_threshold=0.5,
    radius_threshold=0.5,
    push_non_motif=1e-2,
    salad_only="False",
    redesign_motif_aa="False",
    write_failed="False",
    mask_motif_plddt="True",
    no_aa="True",
    timescale="ve(t, sigma_max=80.0)",
    f_motif_rmsd=10.0,
    f_plddt=0.8,
    f_sc_rmsd=2.0,
    f_pae=0.25,
    seed=42,
)

# set up output directories
setup_directories(opt.out_path)

# set up random key
key = Keygen(opt.seed)

# model setup
task = defaultdict(
    lambda: None,
    no_aa=opt.no_aa == "True",
    motif_radius=opt.motif_radius,
    motif_threshold=opt.motif_threshold,
    push_motif=opt.push_motif,
    push_apart=opt.push_apart,
    centered_threshold=opt.centered_threshold,
    inter_motif_distance=opt.inter_motif_distance,
    push_non_motif=opt.push_non_motif)
# options for symmetrizing designs
if opt.symmetrize == "True":
    task["symmetrize"] = True
    task["sym_count"] = opt.sym_count
    task["sym_radius"] = opt.sym_radius
    task["sym_translation"] = opt.sym_translation
    task["sym_threshold"] = opt.sym_threshold
    task["radius_threshold"] = opt.radius_threshold

# set up salad model
salad_config, salad_params = si.make_salad_model(
    opt.salad_config, opt.salad_params)
salad_sampler = si.Sampler(model_step(salad_config, task),
                           prev_threshold=opt.prev_threshold,
                           out_steps=[400],
                           timescale=opt.timescale)
# set up ProteinMPNN
pmpnn = jax.jit(make_pmpnn(opt.pmpnn_params, eps=0.05))
# set up logit transform:
transform = lambda center, do_center, T: transform_logits((
    toggle_transform(
        center_logits(center), use=do_center),
    scale_by_temperature(T),
    forbid("C", aas.PMPNN_CODE),
    norm_logits
))

# set up AlphaFold2
af2_params = get_model_haiku_params(
    model_name="model_1_ptm",
    data_dir=opt.af2_params, fuse=True)
af2_config = model_config("model_1_ptm")
af2_config.model.global_config.use_dgram = False
af2 = jax.jit(make_predict(make_af2(af2_config), num_recycle=4))
cycler = af_cycler(jax.jit(make_predict(make_af2(af2_config), num_recycle=0)),
                   pmpnn, confidence=None, fix_template=True)

# set up output files
score_keys = (
    "attempt", "seq_id", "T", "center",
    "sequence", "sc_rmsd", "motif_rmsd",
    "plddt", "motif_plddt", "pae",
    "success"
)

success = ScoreCSV(
    f"{opt.out_path}/success.csv", score_keys, default="none")
all_designs = ScoreCSV(
    f"{opt.out_path}/all.csv", score_keys, default="none")

for peptides, assembly in [(["S" * opt.peptide_length] * opt.peptide_count, opt.assembly)]:
    # base name of the motif path without the final file ending
    # other "." in the file name are preserved
    motif_name = "motif"
    # get settings dictionary for a motif and simple assembly specification
    settings = make_peptide_assembly(peptides, assembly)
    # get the maximum length of designs with that specification
    _, max_length = get_assembly_lengths(
        settings["segments"], settings["assembly"])
    success_count = 0
    attempt = 0
    while True:
        if success_count >= opt.num_designs:
            break
        # sample motif scaffolding task
        # if random lengths were provided, a specific length will be sampled
        data, init_prev = sample_data(
            salad_config, settings["segments"], settings["assembly"],
            pos_init=True)
        # pad to aa budget
        if opt.assembly_budget != "none":
            budget = [int(c) for c in opt.assembly_budget.strip().split(":")]
            data, init_prev = pad_to_budget(data, init_prev, budget)
        else:
            data = pad_dict(data, max_length)
            init_prev = pad_dict(init_prev, max_length)
        # bias secondary structure
        data["dssp_mean"] = jnp.array([
            opt.l_bias, opt.h_bias, opt.e_bias], dtype=jnp.float32)
        has_motif = data["has_motif"] > 0
        aa_condition = data["motif_aa"]
        steps = salad_sampler(salad_params, key(), data, init_prev)
        for ids, design in enumerate(steps):
            design = data_from_protein(si.data.to_protein(design))
            design.save_pdb(f"{opt.out_path}/attempts/{motif_name}_design_{attempt}_{ids}.pdb")
        # shortcut for only running salad
        if opt.salad_only == "True":
            attempt += 1
            success_count += 1
            continue
        # ProteinMPNN sequence design & AF2 filter
        ca = design["atom_positions"][:, 1]
        other_ca = ca[~has_motif]
        non_motif_contact = (jnp.linalg.norm(ca[:, None] - other_ca[None, :], axis=-1) < 8.0).any(axis=1)
        motif_aa_mask = (design.aa == aa_condition) * has_motif
        motif_aa = jnp.where(motif_aa_mask, aas.translate(aa_condition, aas.AF2_CODE, aas.PMPNN_CODE), 20)
        if opt.redesign_motif_aa == "True":
            motif_aa = jnp.full_like(motif_aa, 20)

        init_info = dict(motif=motif_name, attempt=attempt)
        logit_center = pmpnn(key(), design.update(aa=motif_aa))["logits"].mean(axis=0)
        num_sequences = opt.num_sequences
        attempt_success_count = 0
        for idx in range(num_sequences):
            if attempt_success_count >= opt.num_success:
                break
            # ensure that no overwriting occurs
            design_info = {k: v for k, v in init_info.items()}
            temperature = random.choice((0.01, 0.1, 0.2))#, 0.3, 0.5))
            do_center_logits = random.choice((True, False))
            logit_transform = transform(logit_center, do_center_logits, temperature)
            pmpnn_sampler = sample(pmpnn, logit_transform=logit_transform)
            design_info.update(
                seq_id=idx, T=temperature,
                center=do_center_logits)
            pmpnn_input = design.update(aa=motif_aa, mask=~has_motif)
            if opt.symmetrize == "True":
                pmpnn_input = pmpnn_input.tie([
                    Block(has_motif.shape[0] // opt.sym_count, "A", 1 / opt.sym_count)
                ] * opt.sym_count)
            pmpnn_result, _ = pmpnn_sampler(key(), pmpnn_input)
            sequence = aas.translate(pmpnn_result["aa"], aas.PMPNN_CODE, aas.AF2_CODE)
            #if opt.redesign_motif_aa == "True":
            sequence = jnp.where(has_motif, aas.AF2_CODE.index("S"), sequence)
            pmpnn_result = pmpnn_input.update(
                aa=sequence, mask=jnp.ones_like(sequence, dtype=jnp.bool_))
            # run AF2 filter
            af_input = AFInput.from_data(pmpnn_result).add_guess(pmpnn_result)
            af2_result: AFResult = af2(af2_params, key(), af_input)
            plddt = af2_result.plddt
            pae = af2_result.pae
            if opt.mask_motif_plddt == "True":
                print("masking")
                mean_plddt = plddt[~has_motif].mean()
                print(mean_plddt, plddt.mean())
                mean_pae = pae[~has_motif][:, ~has_motif].mean()
            else:
                mean_plddt = plddt.mean()
                mean_pae = pae.mean()
            # mask motif residues
            rmsd_ca = RMSD(["CA"])(af2_result.to_data(), design, mask=data["mask"] * (~has_motif))
            sequence = pmpnn_result.to_sequence_string(aas.AF2_CODE)
            split_sequence = "".join([c if not h else " " for c, h in zip(sequence, has_motif)])
            split_sequence = "_".join(split_sequence.split())
            design_info.update(
                sequence=split_sequence,
                plddt=mean_plddt,
                pae=mean_pae,
                sc_rmsd=rmsd_ca,
            )
            is_success = mean_plddt > opt.f_plddt and mean_pae < opt.f_pae and rmsd_ca < opt.f_sc_rmsd
            design_info["success"] = int(is_success)
            all_designs.write_line(design_info)
            af2_data: DesignData = af2_result.to_data()
            if not is_success:
                if opt.write_failed == "True":
                    af2_data.save_pdb(f"{opt.out_path}/fail/{motif_name}_{attempt}_{idx}.pdb")
                continue

            design_info.update(success=1, failure_reason="none")
            success.write_line(design_info)
            af2_data.save_pdb(f"{opt.out_path}/success/{motif_name}_{attempt}_{idx}.pdb")
            success_count += 1
            attempt_success_count += 1
        attempt += 1

