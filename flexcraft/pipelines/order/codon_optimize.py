import os
from glob import glob
import numpy as np
import pandas as pd
import shutil
from flexcraft.utils import load_pdb
import flexcraft.sequence.aa_codes as aas
from flexcraft.utils.options import parse_options
import dnachisel as dc
import random

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

def _parse_library(s):
    library_mask = []
    result = ""
    for c in s:
        if c == "?":
            library_mask.append([True])
            result += "A"
        else:
            library_mask.append([False])
            result += c
    return np.array(library_mask), result

def _apply_library(dna_s, library_mask, codons):
    dna_s = np.array([c for c in dna_s])
    for i, val in enumerate(library_mask):
        if val:
            codon = np.array([c for c in random.choice(codons)])
            dna_s[3 * i:3 * (i + 1)] = codon
    return "".join(list(dna_s))

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
    "Codon optimize.",
    in_path="inputs/desings.aa.fa",
    out_path="outputs/",
    prefix="",
    suffix="",
    library_codons="NNT,NNC",
    pad_to=72
)

library_codons = opt.library_codons.strip().split(",")

with (open(opt.in_path, "rt") as in_f,
      open(f"{opt.out_path}/order.aa.fa", "at") as aa_f, 
      open(f"{opt.out_path}/order.dna.fa", "at") as dna_f):
    fasta_lines = in_f.read().split("\n")
    names = [c.strip()[1:] for c in fasta_lines if c.startswith(">")]
    sequences = [c.strip() for c in fasta_lines if not c.startswith(">")]
    for n, s in zip(names, sequences):
        s = _gs_pad(s, opt.pad_to)
        aa_f.write(f">{n}\n{s}\n")
        library_mask, sanitized_s = _parse_library(s)
        dna_s = _codon_optimize(sanitized_s, opt.prefix, opt.suffix)
        dna_s = _apply_library(dna_s, library_mask, library_codons)
        dna_f.write(f">{n}\n{dna_s}\n")
