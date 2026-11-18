import os
import random

import numpy as np

from flexcraft.pipelines.order.common import _codon_optimize, _gs_pad
from flexcraft.utils.options import parse_options


def _parse_library(sequence: str):
    library_mask = []
    result = ""
    for char in sequence:
        if char == "?":
            library_mask.append([True])
            result += "A"
        else:
            library_mask.append([False])
            result += char
    return np.array(library_mask), result


def _apply_library(dna_sequence: str, library_mask: np.ndarray, codons: list[str]) -> str:
    dna_arr = np.array(list(dna_sequence))
    for i, val in enumerate(library_mask):
        if val:
            codon = np.array(list(random.choice(codons)))
            dna_arr[3 * i : 3 * (i + 1)] = codon
    return "".join(dna_arr)


def main():
    opt = parse_options(
        "Codon optimize.",
        in_path="inputs/desings.aa.fa",
        out_path="outputs/",
        prefix="",
        suffix="",
        library_codons="NNT,NNC",
        pad_to=72,
    )

    os.makedirs(opt.out_path, exist_ok=True)
    library_codons = opt.library_codons.strip().split(",")

    with (
        open(opt.in_path, "rt") as in_f,
        open(f"{opt.out_path}/order.aa.fa", "at") as aa_f,
        open(f"{opt.out_path}/order.dna.fa", "at") as dna_f,
    ):
        fasta_lines = [line.strip() for line in in_f.read().splitlines() if line.strip()]
        names = [line[1:] for line in fasta_lines if line.startswith(">")]
        sequences = [line for line in fasta_lines if not line.startswith(">")]
        for n, s in zip(names, sequences):
            s = _gs_pad(s, opt.pad_to)
            aa_f.write(f">{n}\n{s}\n")
            library_mask, sanitized_s = _parse_library(s)
            dna_s = _codon_optimize(sanitized_s, opt.prefix, opt.suffix)
            print(dna_s)
            dna_s = _apply_library(dna_s, library_mask, library_codons)
            dna_f.write(f">{n}\n{dna_s}\n")


if __name__ == "__main__":
    main()
