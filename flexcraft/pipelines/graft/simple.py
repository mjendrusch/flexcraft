import os
import random
from copy import deepcopy
import jax
import jax.numpy as jnp
import haiku as hk
from salad.aflib.model.geometry import Vec3Array
from collections import defaultdict

from flexcraft.pipelines.graft.common import (
    setup_directories, get_assembly_lengths, sample_data,
    make_simple_assembly, get_motif_paths, make_aa_condition)
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
        data["pos"] = pos
        # graft motif explicitly
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
    grad_threshold=1.0,
    align_threshold=0.0,
    dmap_threshold=1.0,
    compact_lr=0.0,
    clash_lr=0.0,
    template_motif="False",
    use_motif_dssp="False",
    use_motif_aa="none",
    salad_only="False",
    redesign_motif_aa="False",
    write_failed="False",
    use_cycler="False",
    mask_motif_plddt="False",
    buried_contacts=6,
    center_to_chain="False",
    timescale="ve(t, sigma_max=80.0)",
    seed=42
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
    potentials=dict(compact=opt.compact_lr, clash=opt.clash_lr),
    grad_threshold=opt.grad_threshold,
    align_threshold=opt.align_threshold,
    dmap_threshold=opt.dmap_threshold,
    use_motif_dssp=opt.use_motif_dssp == "True",
    use_motif_aa=opt.use_motif_aa,
    buried_contacts=opt.buried_contacts,
    center_to_chain=opt.center_to_chain == "True")
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
    "sequence", "sc_rmsd", "motif_rmsd", "plddt", "pae",
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
    # get the maximum length of designs with that specification
    _, max_length = get_assembly_lengths(
        settings["segments"], settings["assembly"])
    success_count = 0
    attempt = 0
    while True:
        if success_count >= opt.num_designs:
            break
        data, init_prev = sample_data(
            salad_config, settings["segments"], settings["assembly"],
            pos_init=True)
        has_motif = data["has_motif"]
        aa_condition = data["motif_aa"]
        data = pad_dict(data, max_length)
        init_prev = pad_dict(init_prev, max_length)
        steps = salad_sampler(salad_params, key(), data, init_prev)
        for ids, design in enumerate(steps):
            design = data_from_protein(si.data.to_protein(design))
            design.save_pdb(f"{opt.out_path}/attempts/{motif_name}_design_{attempt}_{ids}.pdb")
        # shortcut for only running salad
        if opt.salad_only == "True":
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
            rmsd_ca = RMSD(["CA"])(af2_result.to_data(), design, mask=data["mask"])
            motif_rmsd_bb = RMSD(["N", "CA", "C", "O"])(
                af2_result.to_data(), data["motif"],
                index=data["motif_group"], mask=data["has_motif"])
            design_info.update(
                sequence=aas.decode(pmpnn_result["aa"], aas.AF2_CODE),
                plddt=mean_plddt,
                pae=mean_pae,
                sc_rmsd=rmsd_ca,
                motif_rmsd=motif_rmsd_bb
            )
            is_success = mean_plddt > 0.8 and mean_pae < 0.25 and rmsd_ca < 2.0
            design_info["success"] = int(is_success)
            all_designs.write_line(design_info)
            if not is_success:
                if opt.write_failed == "True":
                    af2_result.save_pdb(f"{opt.out_path}/fail/{motif_name}_{attempt}_{idx}.pdb")
                continue

            # all_designs.write_line(design_info)
            design_info.update(success=1, failure_reason="none")
            success.write_line(design_info)
            af2_result.save_pdb(f"{opt.out_path}/success/{motif_name}_{attempt}_{idx}.pdb")
            success_count += 1
            attempt_success_count += 1
        attempt += 1

