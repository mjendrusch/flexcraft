import os
import random
from copy import deepcopy
import jax
import jax.numpy as jnp
import haiku as hk
from salad.aflib.model.geometry import Vec3Array
from collections import defaultdict

from flexcraft.pipelines.graft.common import (
    setup_directories, get_assembly_lengths, sample_data, pad_to_budget,
    make_simple_assembly, make_ghosts, get_motif_paths, make_aa_condition)
from flexcraft.pipelines.graft.options import MODEL_OPTIONS, SAMPLER_OPTIONS
from flexcraft.utils import *
from flexcraft.files.csv import ScoreCSV
from flexcraft.protocols.af2cycler import af_cycler
from flexcraft.sequence.mpnn import make_pmpnn
from flexcraft.sequence.sample import *
from flexcraft.structure.af import *
from flexcraft.structure.metrics import RMSD

import salad.inference as si

from salad.modules.utils.geometry import positions_to_ncacocb, index_align, index_mean

def condition_step(config):
    config = deepcopy(config)
    config.eval = True
    @hk.transform
    def step(data, prev):
        # set up salad models
        noise = si.StructureDiffusionNoise(config)
        predict = si.StructureDiffusionPredict(config)
        # apply noise
        data.update(noise(data))
        # denoise
        out, prev = predict(data, prev)
        return out, prev
    return step.apply

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
        # get indices
        resi = data["residue_index"]
        chain = data["chain_index"]
        # get motif information
        motif, has_motif, group = data["motif"], data["has_motif"], data["motif_group"]
        use_motif = data["use_motif"]
        # compute distance map
        cb = Vec3Array.from_array(positions_to_ncacocb(motif)[:, 4])
        has_motif *= use_motif
        dmap = (cb[:, None] - cb[None, :]).norm()
        dmap_mask = has_motif[:, None] * has_motif[None, :]
        dmap_mask *= group[:, None] == group[None, :]
        dmap_mask = dmap_mask > 0
        resi_dist = jnp.where(chain[:, None] == chain[None, :],
                              abs(resi[:, None] - resi[None, :]),
                              jnp.inf)
        data["dmap"] = dmap
        data["dmap_mask"] = jnp.where(t < task["dmap_threshold"], dmap_mask, 0)
        aatype = data["motif_aa"]
        aa_condition = make_aa_condition(aatype, dmap, dmap_mask, resi_dist, task)
        data["aa_condition"] = aa_condition
        # optionally apply gradients
        pos = data["pos"]
        aln_motif = index_align(motif, pos, group, has_motif)
        mca = aln_motif[:, 1]
        def restraint_indexed(ca):
            val = ((ca - mca) ** 2).sum(axis=-1)
            return (val * has_motif).sum() / jnp.maximum(1, has_motif.sum())
        pos -= jnp.where(
            t >= task["grad_threshold"],
            t * jax.grad(restraint_indexed, argnums=(0,))(pos[:, 1])[0][:, None],
            0.0)
        # ghost chain gradients
        if "ghost" in data:
            ghost, ghost_group = data["ghost"], data["ghost_group"]
            aln_ghost = index_align(
                jnp.concatenate((motif, ghost), axis=0),
                jnp.concatenate((pos, jnp.zeros([ghost.shape[0]] + list(pos.shape[1:]))), axis=0),
                jnp.concatenate((group, ghost_group), axis=0),
                jnp.concatenate((has_motif, jnp.ones((ghost.shape[0],), dtype=jnp.bool_)), axis=0),
                jnp.concatenate((has_motif, jnp.zeros((ghost.shape[0],))), axis=0))
            ghost_ca = aln_ghost[has_motif.shape[0]:, 1]
            def ghost_repulsive(ca):
                distance = jnp.sqrt(jnp.maximum(3e-4, (ca[:, None] - ghost_ca[None, :]) ** 2).sum(axis=-1))
                return ((jax.nn.relu(12.0 - distance) ** 1.5).sum(axis=1) * (1 - has_motif) * data["mask"]).sum()# / jnp.maximum(1, data["mask"].sum())
            pos -= task["potentials"]["ghost_repulsive"] * t * jax.grad(ghost_repulsive, argnums=(0,))(pos[:, 1])[0][:, None]
        pos += task["potentials"]["compact"] * t * si.contacts.compact_step(pos, data["chain_index"], data["mask"])
        data["pos"] = pos
        # graft motif explicitly
        # aligning the denoised structure to the motif
        if task["reverse_align"]:
            pos_aligned = index_align(data["pos"], motif, group, has_motif)
            data["pos"] = pos_aligned.at[:, :4].set(
                jnp.where((t < task["align_threshold"]) * has_motif[:, None, None],
                        motif[:, :4],
                        pos_aligned[:, :4]))
            data["pos"] = data["pos"].at[:, 4:].set(
                jnp.where((t < task["align_threshold"]) * has_motif[:, None, None],
                        motif[:, 1, None],
                        data["pos"][:, 4:]))
        # or aligning the motif to the denoised structure (default)
        else:
            motif_aligned = index_align(motif, data["pos"], group, has_motif)
            data["pos"] = data["pos"].at[:, :4].set(
                jnp.where((t < task["align_threshold"]) * has_motif[:, None, None],
                        motif_aligned[:, :4],
                        data["pos"][:, :4]))
            data["pos"] = data["pos"].at[:, 4:].set(
                jnp.where((t < task["align_threshold"]) * has_motif[:, None, None],
                        motif_aligned[:, 1, None],
                        data["pos"][:, 4:]))
        # center positions
        data["pos"] = data["pos"] - index_mean(data["pos"][:, 1], data["batch_index"], data["mask"][:, None])[:, None]
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
    assembly="20-50,A1-20@0,20-50",
    assembly_budget="none",
    condition_dssp="False",
    h_bias=0.0,
    e_bias=0.0,
    l_bias=0.0,
    af_drop_chains="none",
    ghosts="none",
    ghost_repulsive_lr=0.0,
    ghost_attractive_lr=0.0,
    grad_threshold=2.0,
    align_threshold=2.0,
    dmap_threshold=0.5,
    align_final_to_motif="False",
    reverse_align="False",
    compact_lr=0.0,
    clash_lr=0.0,
    template_motif="False",
    use_motif_dssp="False",
    use_motif_aa="all",
    salad_only="False",
    redesign_motif_aa="False",
    write_failed="False",
    use_cycler="False",
    mask_motif_plddt="True",
    buried_contacts=6,
    center_to_chain="False",
    timescale="ve(t, sigma_max=80.0)",
    f_motif_rmsd=2.0,
    f_plddt=0.8,
    f_sc_rmsd=2.0,
    f_pae=0.25,
    seed=42,
)

# set up output directories
setup_directories(opt.out_path)
if opt.use_cycler == "True":
    os.makedirs(f"{opt.out_path}/cycles/", exist_ok=True)

# get motif PDB files
motif_paths = get_motif_paths(
    opt.motif_path, opt.assembly)

# set up random key
key = Keygen(opt.seed)

# model setup
task = defaultdict(
    lambda: None, mode=["grad"], 
    potentials=dict(compact=opt.compact_lr, clash=opt.clash_lr,
                    ghost_repulsive=opt.ghost_repulsive_lr),
    grad_threshold=opt.grad_threshold,
    align_threshold=opt.align_threshold,
    dmap_threshold=opt.dmap_threshold,
    use_motif_dssp=opt.use_motif_dssp == "True",
    use_motif_aa=opt.use_motif_aa,
    buried_contacts=opt.buried_contacts,
    center_to_chain=opt.center_to_chain == "True")
salad_config, salad_params = si.make_salad_model(
    opt.salad_config, opt.salad_params)
condition_sampler = si.Sampler(condition_step(salad_config),
                               prev_threshold=opt.prev_threshold,
                               out_steps=400,
                               timescale=opt.timescale)
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

for motif, assembly in motif_paths:
    # base name of the motif path without the final file ending
    # other "." in the file name are preserved
    motif_name = ".".join(os.path.basename(motif).split(".")[:-1])
    # get settings dictionary for a motif and simple assembly specification
    settings = make_simple_assembly(motif, assembly)
    # get ghost chains, if available
    ghost_data = dict()
    if opt.ghosts != "none":
        ghost_data = make_ghosts(motif, opt.ghosts)
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
        has_motif = data["has_motif"]
        aa_condition = data["motif_aa"]
        if opt.condition_dssp == "True":
            # NOTE: this is currently experimental
            cond_data = slice_dict(data, data["chain_index"] == 0, skip_keys=["dssp_mean"])
            cond_prev = slice_dict(init_prev, data["chain_index"] == 0, skip_keys=["dssp_mean"])
            design = condition_sampler(salad_params, key(), cond_data, cond_prev)
            design = data_from_protein(si.data.to_protein(design))
            dssp = design.dssp
            dssp = jnp.where(dssp == 0, 3, dssp)
            dssp = jnp.where(cond_data["has_motif"], 3, dssp)
            dssp = jnp.concatenate((dssp, jnp.zeros((has_motif.shape[0] - dssp.shape[0],), dtype=jnp.int32)))
            data["dssp_condition"] = dssp
        # add ghost chain info if available
        data.update(ghost_data)
        steps = salad_sampler(salad_params, key(), data, init_prev)
        for ids, design in enumerate(steps):
            design = data_from_protein(si.data.to_protein(design))
            if opt.align_final_to_motif == "True":
                design = design.align_to(
                    data["motif"], data["mask"], weight=data["has_motif"].astype(jnp.float32))
            design.save_pdb(f"{opt.out_path}/attempts/{motif_name}_design_{attempt}_{ids}.pdb")
        # shortcut for only running salad
        if opt.salad_only == "True":
            attempt += 1
            success_count += 1
            continue
        # optionally apply af2cycler
        if opt.use_cycler == "True":
            cycled = design
            for idc in range(10):
                cycled, predicted = cycler(af2_params, key, cycled, cycle_mask=~has_motif)
                predicted.save_pdb(f"{opt.out_path}/cycles/design_{attempt}_{idc}.pdb")
            design = cycled
        # ProteinMPNN sequence design & AF2 filter
        ca = design["atom_positions"][:, 1]
        other_ca = ca[~has_motif]
        non_motif_contact = (jnp.linalg.norm(ca[:, None] - other_ca[None, :], axis=-1) < 8.0).any(axis=1)
        motif_aa_mask = (design.aa == aa_condition) * has_motif
        motif_aa = jnp.where(motif_aa_mask, aas.translate(aa_condition, aas.AF2_CODE, aas.PMPNN_CODE), 20)
        if opt.redesign_motif_aa == "True":
            motif_aa = jnp.full_like(motif_aa, 20)

        # drop chains for design
        if opt.af_drop_chains != "none":
            drop_chains = np.array(["ABCDEFGHIJKLMNOPQRSTUVWXYZ".index(c.upper())
                                    for c in opt.af_drop_chains.strip().split(",")], dtype=np.int32)
            selector = ~(data["chain_index"][:, None] == drop_chains).any(axis=1)
            data = slice_dict(data, selector, skip_keys=["dssp_mean"])
            design = design[selector]
            motif_aa = motif_aa[selector]
            has_motif = has_motif[selector]

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
            pmpnn_input = design.update(aa=motif_aa)
            pmpnn_result, _ = pmpnn_sampler(key(), pmpnn_input)
            pmpnn_result = pmpnn_input.update(
                aa=aas.translate(pmpnn_result["aa"], aas.PMPNN_CODE, aas.AF2_CODE))
            # run AF2 filter
            af_input = AFInput.from_data(pmpnn_result).add_guess(pmpnn_result)
            if opt.template_motif == "True":
                af_input = af_input.add_template(
                    pmpnn_result.update(
                        atom_positions=data["motif"],
                        atom_mask=data["motif_mask"],
                        aa=data["motif_aa"]), where=has_motif,
                    template_sidechains=True)
            af2_result: AFResult = af2(af2_params, key(), af_input)
            plddt = af2_result.plddt
            pae = af2_result.pae
            if opt.mask_motif_plddt == "True":
                mean_plddt = plddt[~has_motif].mean()
                mean_pae = pae[~has_motif][:, ~has_motif].mean()
            else:
                mean_plddt = plddt.mean()
                mean_pae = pae.mean()
            motif_plddt = plddt[has_motif].mean()
            rmsd_ca = RMSD(["CA"])(af2_result.to_data(), design, mask=data["mask"])
            motif_rmsd_bb = RMSD(["N", "CA", "C", "O"])(
                af2_result.to_data(), data["motif"],
                index=data["motif_group"], mask=data["has_motif"])
            design_info.update(
                sequence=pmpnn_result.to_sequence_string(aas.AF2_CODE),#aas.decode(pmpnn_result["aa"], aas.AF2_CODE),
                plddt=mean_plddt,
                motif_plddt=motif_plddt,
                pae=mean_pae,
                sc_rmsd=rmsd_ca,
                motif_rmsd=motif_rmsd_bb
            )
            is_success = mean_plddt > opt.f_plddt and mean_pae < opt.f_pae and rmsd_ca < opt.f_sc_rmsd and motif_rmsd_bb < opt.f_motif_rmsd
            design_info["success"] = int(is_success)
            all_designs.write_line(design_info)
            af2_data: DesignData = af2_result.to_data()
            if opt.align_final_to_motif == "True":
                af2_data = af2_data.align_to(
                    data["motif"], mask=data["mask"],
                    weight=data["has_motif"].astype(jnp.float32))
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

