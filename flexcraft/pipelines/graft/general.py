import os
import random
from collections import defaultdict
import numpy as np
import jax.numpy as jnp

from flexcraft.pipelines.graft.common import (
    setup_directories, model_step, get_assembly_lengths, sample_data,
    make_simple_assembly, get_motif_paths)
from flexcraft.pipelines.graft.options import MODEL_OPTIONS, SAMPLER_OPTIONS
from flexcraft.utils import *
from flexcraft.files.csv import ScoreCSV
from flexcraft.sequence.mpnn import make_pmpnn
from flexcraft.sequence.sample import *
from flexcraft.structure.af import *
from flexcraft.structure.metrics import RMSD
from flexcraft.utils import *

import salad.inference as si

def parse_center_file(path):
    with open(path, "rt") as f:
        center = []
        center_group = []
        center_time = []
        for line in f:
            length, group, x, y, z, ct = line.strip().split(",")
            length, group = int(length), int(group)
            x, y, z = float(x), float(y), float(z)
            ct = float(ct)
            center += [[x, y, z]] * length
            center_group += [group] * length
            center_time += [ct] * length
    center = np.array(center, dtype=np.float32)
    center -= center.mean(axis=0)
    center_group = np.array(center_group, dtype=np.int32)
    center_time = np.array(center_time, dtype=np.int32)
    return center, center_group, center_time

def parse_pseudo_center_file(path):
    with open(path, "rt") as f:
        center = []
        center_group = []
        center_time = []
        for line in f:
            group, x, y, z, ct = line.strip().split(",")
            group = int(group)
            x, y, z = float(x), float(y), float(z)
            ct = float(ct)
            center += [[x, y, z]]
            center_group += [group]
            center_time += [ct]
    center = np.array(center, dtype=np.float32)
    center_group = np.array(center_group, dtype=np.int32)
    center_time = np.array(center_time, dtype=np.int32)
    return dict(center=center, center_group=center_group, center_time=center_time)

opt = parse_options(
    "Simple motif grafting with salad.",
    **MODEL_OPTIONS,
    **SAMPLER_OPTIONS,
    out_path="outputs/",
    motif_path="motifs/",
    assembly="20-50,A1-20@0,20-50",
    symmetry="none",
    centers="none",
    center_radius=12.0,
    compact_lr=0.0,
    clash_lr=0.0,
    compact_by="chain",
    use_motif_dssp="False",
    use_motif_aa="none",
    center_to_chain="False",
    timescale="ve(t, sigma_max=80.0)",
    seed=42
)

# set up output directories
setup_directories(opt.out_path)

# get motif PDB files
motif_paths = get_motif_paths(
    opt.motif_path, opt.assembly)

# set up random key
key = Keygen(opt.seed)

# model setup
task = defaultdict(
    lambda: None, mode=["grad"], 
    potentials=dict(compact=opt.compact_lr, clash=opt.clash_lr),
    use_motif_dssp=opt.use_motif_dssp == "True",
    use_motif_aa=opt.use_motif_aa == "True",
    center_to_chain=opt.center_to_chain == "True",
    compact_by=opt.compact_by,
    buried_contacts=6)
pseudo_centers = None
if opt.centers != "none":
    task["center"] = True
    pseudo_centers = parse_pseudo_center_file(opt.centers)
if opt.symmetry != "none":
    task["sym"] = int(opt.symmetry)
salad_config, salad_params = si.make_salad_model(
    opt.salad_config, opt.salad_params)
salad_config.center_radius = opt.center_radius
salad_sampler = si.Sampler(model_step(salad_config, task),
                           prev_threshold=opt.prev_threshold,
                           out_steps=[400],
                           timescale=opt.timescale)
# # set up ProteinMPNN
# pmpnn = jax.jit(make_pmpnn(opt.pmpnn_params, eps=0.05))
# # set up logit transform:
# transform = lambda center, do_center, T: transform_logits((
#     toggle_transform(
#         center_logits(center), use=do_center),
#     scale_by_temperature(T),
#     forbid("C", aas.PMPNN_CODE),
#     norm_logits
# ))

# # set up AlphaFold2
# af2_params = get_model_haiku_params(
#     model_name="model_1_ptm",
#     data_dir=opt.af2_params, fuse=True)
# af2_config = model_config("model_1_ptm")
# af2_config.model.global_config.use_dgram = False
# af2 = jax.jit(make_predict(make_af2(af2_config), num_recycle=4))


# # set up output files
# score_keys = (
#     "attempt", "seq_id", "T", "center",
#     "sequence", "sc_rmsd", "motif_rmsd", "plddt", "pae",
#     "success"
# )

# success = ScoreCSV(
#     f"{opt.out_path}/success.csv", score_keys, default="none")
# all_designs = ScoreCSV(
#     f"{opt.out_path}/all.csv", score_keys, default="none")

for motif, assembly in motif_paths:
    # base name of the motif path without the final file ending
    # other "." in the file name are preserved
    motif_name = ".".join(os.path.basename(motif).split(".")[:-1])
    # get settings dictionary for a motif and simple assembly specification
    settings = make_simple_assembly(motif, assembly)
    # get the maximum length of designs with that specification
    _, max_length = get_assembly_lengths(
        settings["segments"], settings["assembly"])
    for idx in range(opt.num_designs):
        data, init_prev = sample_data(
            salad_config, settings["segments"], settings["assembly"],
            pos_init=True, pseudo_centers=pseudo_centers)
        has_motif = data["has_motif"]
        aa_condition = data["motif_aa"]
        data = pad_dict(data, max_length)
        init_prev = pad_dict(init_prev, max_length)
        data = pad_dict(data, max_length)
        init_prev = pad_dict(init_prev, max_length)
        steps = salad_sampler(salad_params, key(), data, init_prev)
        for ids, design in enumerate(steps):
            design = data_from_protein(si.data.to_protein(design))
            design.save_pdb(f"{opt.out_path}/attempts/{motif_name}_design_{idx}_{ids}.pdb")
        # # ProteinMPNN sequence design & AF2 filter
        # ca = design["atom_positions"][:, 1]
        # other_ca = ca[~has_motif]
        # non_motif_contact = (jnp.linalg.norm(ca[:, None] - other_ca[None, :], axis=-1) < 8.0).any(axis=1)
        # motif_aa_mask = (design.aa == aa_condition) * has_motif
        # motif_aa = jnp.where(motif_aa_mask, aas.translate(aa_condition, aas.AF2_CODE, aas.PMPNN_CODE), 20)

        # init_info = dict(motif=motif_name, attempt=attempt)
        # logit_center = pmpnn(key(), design.update(aa=motif_aa))["logits"].mean(axis=0)
        # num_sequences = opt.num_sequences
        # attempt_success_count = 0
        # for idx in range(num_sequences):
        #     if attempt_success_count >= opt.num_success:
        #         break
        #     # ensure that no overwriting occurs
        #     design_info = {k: v for k, v in init_info.items()}
        #     temperature = random.choice((0.01, 0.1, 0.2))#, 0.3, 0.5))
        #     do_center_logits = random.choice((True, False))
        #     logit_transform = transform(logit_center, do_center_logits, temperature)
        #     pmpnn_sampler = sample(pmpnn, logit_transform=logit_transform)
        #     design_info.update(
        #         seq_id=idx, T=temperature,
        #         center=do_center_logits)
        #     pmpnn_input = design.update(aa=motif_aa)
        #     pmpnn_result, _ = pmpnn_sampler(key(), pmpnn_input)
        #     pmpnn_result = pmpnn_input.update(
        #         aa=aas.translate(pmpnn_result["aa"], aas.PMPNN_CODE, aas.AF2_CODE))
        #     # run AF2 filter
        #     af_input = AFInput.from_data(pmpnn_result).add_guess(pmpnn_result).add_template(
        #         pmpnn_result.update(aa=data["motif_aa"]), where=has_motif)
        #     af2_result: AFResult = af2(af2_params, key(), af_input)
        #     mean_plddt = af2_result.plddt.mean()
        #     mean_pae = af2_result.pae.mean()
        #     rmsd_ca = RMSD(["CA"])(af2_result.to_data(), design, mask=data["mask"])
        #     motif_rmsd_bb = RMSD(["N", "CA", "C", "O"])(
        #         af2_result.to_data(), data["motif"],
        #         index=data["motif_group"], mask=data["has_motif"])
        #     design_info.update(
        #         sequence=aas.decode(pmpnn_result["aa"], aas.AF2_CODE),
        #         plddt=mean_plddt,
        #         pae=mean_pae,
        #         sc_rmsd=rmsd_ca,
        #         motif_rmsd=motif_rmsd_bb
        #     )
        #     is_success = mean_plddt > 0.8 and mean_pae < 0.25 and rmsd_ca < 2.0
        #     design_info["success"] = int(is_success)
        #     all_designs.write_line(design_info)
        #     if not is_success:
        #         continue

        #     # all_designs.write_line(design_info)
        #     design_info.update(success=1, failure_reason="none")
        #     success.write_line(design_info)
        #     af2_result.save_pdb(f"{opt.out_path}/success/{motif_name}_{attempt}_{idx}.pdb")
        #     success_count += 1
        #     attempt_success_count += 1
        # attempt += 1



