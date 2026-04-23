import os
from glob import glob
import numpy as np
import pandas as pd
import shutil
from flexcraft.utils import load_pdb
import flexcraft.sequence.aa_codes as aas
from flexcraft.utils.options import parse_options
import dnachisel as dc

def _codon_optimize(aa_sequence, prefix: str, suffix: str):
    prefix = prefix.upper()
    suffix = suffix.upper()
    sequence = prefix + dc.reverse_translate(aa_sequence) + suffix
    sequence = sequence.upper()
    #print(sequence)
    problem = dc.DnaOptimizationProblem(
        sequence=sequence,
        constraints=[
            dc.AvoidPattern("BsaI_site"),
            dc.AvoidPattern(dc.HomopolymerPattern("A", 4)),
            dc.AvoidPattern(dc.HomopolymerPattern("T", 4)),
            dc.AvoidPattern(dc.HomopolymerPattern("C", 4)),
            dc.AvoidPattern(dc.HomopolymerPattern("G", 4)),
            dc.EnforceGCContent(mini=0.3, maxi=0.7, window=50),
            dc.EnforceTranslation(translation=aa_sequence,
                                  location=(len(prefix), len(prefix) + 3 * len(aa_sequence))),
            dc.EnforceSequence(sequence=prefix, location=(0, len(prefix))),
            dc.EnforceSequence(sequence=suffix, location=(len(prefix) + 3 * len(aa_sequence), len(sequence)))
        ],
        objectives=[dc.CodonOptimize(species='e_coli', location=(len(prefix), len(prefix) + 3 * len(aa_sequence)))]
    )
    problem.resolve_constraints()
    problem.optimize()
    print(problem.sequence)
    #print(problem.constraints_text_summary())
    return problem.sequence

def _gs_string(count):
    result = ""
    for i in range(count):
        if i % 3 < 2:
            result += "G"
        else:
            result += "S"
    return result

def _gs_pad(sequence, num_aa=72):
    length = len(sequence)
    if length < num_aa:
        num_remaining = num_aa - length
        pad_left = num_remaining // 2
        pad_right = num_remaining - pad_left
        sequence = _gs_string(pad_left) + sequence + _gs_string(pad_right)
    return sequence

opt = parse_options(
    "Collect successful designs and metrics from a directory of flexcraft runs.",
    in_path="inputs/",
    out_path="outputs/",
    name="run_1",
    design_names="attempt:seq_id",
    collect_chains="all",
    dirs_match="none",
    success_name="success.csv",
    all_relaxed="False",
    prefix="",
    suffix="",
    sort_by="lowest-ipae",
    pad_to=72,
    sequence_field="binder_sequence",
    num_designs=96
)

os.makedirs(f"{opt.out_path}/designs/", exist_ok=True)
sort_mode, *sort_by = opt.sort_by.split("-")
sort_by = "-".join(sort_by)

with (open(f"{opt.out_path}/order.aa.fa", "at") as aa_f, 
      open(f"{opt.out_path}/order.dna.fa", "at") as dna_f):
    header_done = False
    subdirs = False
    if os.path.isfile(f"{opt.in_path}/{opt.success_name}"):
        path = f"{opt.in_path}/{opt.success_name}"
        shutil.copyfile(path, f"{opt.out_path}/all_{opt.name}.csv")
        data = pd.read_csv(f"{opt.out_path}/all_{opt.name}.csv", na_values=["NaN", "none", "None", "nan"])
        data.insert(0, "name", [f"{opt.name}_{a}_{s}" for a, s in zip(data["attempt"], data["seq_id"])])
    else:
        subdirs = True
        subitems = []
        for subdir in os.listdir(opt.in_path):
            if not os.path.isdir(f"{opt.in_path}/{subdir}"):
                continue
            if opt.dirs_match != "none" and not subdir.startswith(opt.dirs_match):
                continue
            path = f"{opt.in_path}/{subdir}/{opt.success_name}"
            data = pd.read_csv(f"{opt.in_path}/{subdir}/{opt.success_name}", na_values=["NaN", "none", "None", "nan"])
            data.insert(0, "name", [f"{opt.name}_{subdir}_{a}_{s}" for a, s in zip(data["attempt"], data["seq_id"])])
            subitems.append(data)
        data = pd.concat(subitems, axis=0)
        data.to_csv(f"{opt.out_path}/all_{opt.name}.csv")
    if sort_mode == "lowest":
        data = data.nsmallest(opt.num_designs, columns=[sort_by])
    else:
        data = data.nlargest(opt.num_designs, columns=[sort_by])
    data.to_csv(f"{opt.out_path}/filtered_{opt.name}.csv")
    names = np.array(data["name"])
    sequences = np.array(data[opt.sequence_field])
    for n, s in zip(names, sequences):
        s = _gs_pad(s, opt.pad_to)
        aa_f.write(f">{n}\n{s}\n")
        dna_s = _codon_optimize(s, opt.prefix, opt.suffix)
        dna_f.write(f">{n}\n{dna_s}\n")
        if subdirs:
            _, *subdir, attempt, seq_id = n.split("_")
            subdir = "_".join(subdir)
            for path in glob(f"{opt.in_path}/{subdir}/success/*_{attempt}_{seq_id}.pdb"):
                shutil.copyfile(path, f"{opt.out_path}/designs/{n}.pdb")
        else:
            _, *attempt, seq_id = n.split("_")
            attempt = "_".join(attempt)
            for path in glob(f"{opt.in_path}/success/*_{attempt}_{seq_id}.pdb"):
                shutil.copyfile(path, f"{opt.out_path}/designs/{n}.pdb")
