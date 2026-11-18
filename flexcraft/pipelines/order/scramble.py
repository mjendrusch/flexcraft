import os

import numpy as np
import pandas as pd

from flexcraft.pipelines.order.common import _codon_optimize, _gs_pad
from flexcraft.utils.options import parse_options


def _scramble(sequence: str) -> str:
    result = np.array(list(sequence))
    np.random.shuffle(result)
    return "".join(result)


def _is_hydrophobic(aa_char: str) -> bool:
    return aa_char in "ILVFYWMAC"


def _patterned_scramble(sequence: str) -> str:
    seq_arr = np.array(list(sequence))
    result = seq_arr.copy()
    hydrophobic_mask = np.array([_is_hydrophobic(c) for c in seq_arr], dtype=np.bool_)
    shuffled = seq_arr.copy()
    np.random.shuffle(shuffled)
    shuffled_hydrophobic_mask = np.array(
        [_is_hydrophobic(c) for c in shuffled], dtype=np.bool_
    )
    result[hydrophobic_mask] = shuffled[shuffled_hydrophobic_mask]
    result[~hydrophobic_mask] = shuffled[~shuffled_hydrophobic_mask]
    return "".join(result)


def main():
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

    with (
        open(f"{opt.out_path}/order.aa.fa", "at") as aa_f,
        open(f"{opt.out_path}/order.dna.fa", "at") as dna_f,
    ):
        if os.path.isfile(f"{opt.in_path}/{opt.success_name}"):
            path = f"{opt.in_path}/{opt.success_name}"
            data = pd.read_csv(path, na_values=["NaN", "none", "None", "nan"])
            data.insert(
                0,
                "name",
                [f"{opt.name}_{a}_{s}" for a, s in zip(data["attempt"], data["seq_id"])],
            )
        else:
            subitems = []
            for subdir in os.listdir(opt.in_path):
                if not os.path.isdir(f"{opt.in_path}/{subdir}"):
                    continue
                if opt.dirs_match != "none" and not subdir.startswith(opt.dirs_match):
                    continue
                data = pd.read_csv(
                    f"{opt.in_path}/{subdir}/{opt.success_name}",
                    na_values=["NaN", "none", "None", "nan"],
                )
                data.insert(
                    0,
                    "name",
                    [
                        f"{opt.name}_{subdir}_{a}_{s}"
                        for a, s in zip(data["attempt"], data["seq_id"])
                    ],
                )
                subitems.append(data)
            data = pd.concat(subitems, axis=0)
        names = np.array(data["name"])
        sequences = np.array(data[opt.sequence_field])
        scramble_index = np.random.permutation(names.shape[0])[: opt.num_random]
        for n, s in zip(names[scramble_index], sequences[scramble_index]):
            random_s = _gs_pad(_scramble(s), opt.pad_to)
            patterned_s = _gs_pad(_patterned_scramble(s), opt.pad_to)
            aa_f.write(f">{n}_scramble\n{random_s}\n")
            dna_s = _codon_optimize(
                random_s, opt.prefix, opt.suffix, homopolymer_as_objective=True
            )
            dna_f.write(f">{n}_scramble\n{dna_s}\n")
            aa_f.write(f">{n}_patterned_scramble\n{patterned_s}\n")
            dna_s = _codon_optimize(
                patterned_s, opt.prefix, opt.suffix, homopolymer_as_objective=True
            )
            dna_f.write(f">{n}_patterned_scramble\n{dna_s}\n")


if __name__ == "__main__":
    main()
