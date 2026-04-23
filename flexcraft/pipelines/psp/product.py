from typing import List
from itertools import product
from flexcraft.utils.options import parse_options
from flexcraft.files.fasta import FastaFile

def _product(fasta_files: List[FastaFile]) -> FastaFile:
    result_items = []
    for itemlist in product(*[ff.items() for ff in fasta_files]):
        name = ":".join([k for k, _ in itemlist])
        sequence = ":".join([v for _, v in itemlist])
        result_items.append((name, sequence))
    return FastaFile(lines=result_items)

if __name__ == "__main__":
    opt = parse_options(
        "slice an input sequence into subsequences.",
        in_paths="a.fa,b.fa",
        out_path="out.fa"
    )

    fasta_lists = []
    for path in opt.in_paths.split(","):
        item = FastaFile(path=path)
        fasta_lists.append(item)

    output = _product(fasta_lists)
    output.write(opt.out_path)
