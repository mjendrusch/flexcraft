import os
import random
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
            dc.EnforceGCContent(mini=0.3, maxi=0.7, window=50),
            dc.EnforceTranslation(translation=aa_sequence,
                                  location=(len(prefix), len(prefix) + 3 * len(aa_sequence))),
            dc.EnforceSequence(sequence=prefix, location=(0, len(prefix))),
            dc.EnforceSequence(sequence=suffix, location=(len(prefix) + 3 * len(aa_sequence), len(sequence)))
        ],
        objectives=[
            dc.CodonOptimize(species='e_coli', location=(len(prefix), len(prefix) + 3 * len(aa_sequence))),
            dc.AvoidPattern(dc.HomopolymerPattern("A", 4)),
            dc.AvoidPattern(dc.HomopolymerPattern("T", 4)),
            dc.AvoidPattern(dc.HomopolymerPattern("C", 4)),
            dc.AvoidPattern(dc.HomopolymerPattern("G", 4)),
        ]
    )
    problem.resolve_constraints()
    problem.optimize()
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

def _scramble(sequence):
    result = np.array([c for c in sequence])
    np.random.shuffle(result)
    return "".join([c for c in result])

def _is_hydrophobic(c):
    return c in "ILVFYWMAC"

def _patterned_scramble(sequence):
    sequence = np.array([c for c in sequence])
    result = sequence.copy()
    hydrophobic_mask = np.array([_is_hydrophobic(c) for c in sequence], dtype=np.bool_)
    shuffled = sequence.copy()
    np.random.shuffle(shuffled)
    shuffled_hydrophobic_mask = np.array([_is_hydrophobic(c) for c in shuffled], dtype=np.bool_)
    result[hydrophobic_mask] = shuffled[shuffled_hydrophobic_mask]
    result[~hydrophobic_mask] = shuffled[~shuffled_hydrophobic_mask]
    return "".join([c for c in result])

opt = parse_options(
    "Collect successful designs and metrics from a directory of flexcraft runs.",
    in_path="inputs/",
    out_path="outputs/",
    name="run_1",
    design_names="attempt:seq_id",
    collect_chains="all",
    dirs_match="none",
    success_name="success.csv",
    prefix="",
    suffix="",
    pad_to=72,
    sequence_field="sequence",
    num_random=5,
)

os.makedirs(f"{opt.out_path}/designs/", exist_ok=True)

with (open(f"{opt.out_path}/order.aa.fa", "at") as aa_f, 
      open(f"{opt.out_path}/order.dna.fa", "at") as dna_f):
    header_done = False
    subdirs = False
    if os.path.isfile(f"{opt.in_path}/{opt.success_name}"):
        path = f"{opt.in_path}/{opt.success_name}"
        data = pd.read_csv(path, na_values=["NaN", "none", "None", "nan"])
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
    names = np.array(data["name"])
    sequences = np.array(data[opt.sequence_field])
    scramble_index = np.random.permutation(names.shape[0])[:opt.num_random]
    for n, s in zip(names[scramble_index], sequences[scramble_index]):
        random_s = _scramble(s)
        random_s = _gs_pad(random_s, opt.pad_to)
        patterned_s = _patterned_scramble(s)
        patterned_s = _gs_pad(patterned_s, opt.pad_to)
        aa_f.write(f">{n}_scramble\n{random_s}\n")
        dna_s = _codon_optimize(random_s, opt.prefix, opt.suffix)
        dna_f.write(f">{n}_scramble\n{dna_s}\n")
        aa_f.write(f">{n}_patterned_scramble\n{patterned_s}\n")
        dna_s = _codon_optimize(patterned_s, opt.prefix, opt.suffix)
        dna_f.write(f">{n}_patterned_scramble\n{dna_s}\n")
