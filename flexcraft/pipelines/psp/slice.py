import math
import numpy as np
from flexcraft.utils.options import parse_options
from flexcraft.files.fasta import FastaFile

def _slice_fasta(fasta: FastaFile, slice_spec: str, min_length=0) -> FastaFile:
    slice_spec = _parse_slice_spec(slice_spec)
    result = FastaFile()
    for begin, end in slice_spec:
        for name, sequence in fasta.items():
            if begin < len(sequence):
                subsequence = sequence[begin:end]
                if len(subsequence) >= min_length:
                    result.append(f"{name}_{begin}-{end}", subsequence)
    return result

def _parse_slice_spec(spec: str) -> list:
    subspecs = spec.strip().split(",")
    result = []
    for subspec in subspecs:
        result += _parse_subspec(subspec)
    return result

def _parse_subspec(spec: str) -> list:
    if "@" in spec:
        head, steps = spec.split("@")
        begin, end = _parse_range(head)
        length, overlap = map(int, steps.split(":"))
        step = length - overlap
        result = [
            (begin + step * i, begin + step * i + length)
            for i in range((end - begin) // step)
        ]
    else:
        begin, end = _parse_range(head)
        result = [(begin, end)]
    return result

def _parse_range(head):
    if head == "all":
        return (0, 1_000_000)
    return tuple([int(c) for c in head.split("-")])

if __name__ == "__main__":
    opt = parse_options(
        "slice an input sequence into subsequences.",
        fasta_path="input.fa",
        out_path="out.fa",
        slices="all@100:20",
        min_slice_length=0
    )

    inputs = FastaFile(path=opt.fasta_path)
    slices = _slice_fasta(inputs, opt.slices, min_length=opt.min_slice_length)
    slices.write(opt.out_path)
