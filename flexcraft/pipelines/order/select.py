import os
from glob import glob
import shutil

import numpy as np
import pandas as pd

from flexcraft.pipelines.order.common import _codon_optimize, _gs_pad
from flexcraft.utils.options import parse_options


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
        all_relaxed="False",
        prefix="",
        suffix="",
        sort_by="lowest-ipae",
        pad_to=72,
        sequence_field="binder_sequence",
        num_designs=96,
    )

    os.makedirs(f"{opt.out_path}/designs/", exist_ok=True)
    sort_mode, *sort_by = opt.sort_by.split("-")
    sort_by = "-".join(sort_by)

    with (
        open(f"{opt.out_path}/order.aa.fa", "at") as aa_f,
        open(f"{opt.out_path}/order.dna.fa", "at") as dna_f,
    ):
        subdirs = False
        if os.path.isfile(f"{opt.in_path}/{opt.success_name}"):
            path = f"{opt.in_path}/{opt.success_name}"
            shutil.copyfile(path, f"{opt.out_path}/all_{opt.name}.csv")
            data = pd.read_csv(
                f"{opt.out_path}/all_{opt.name}.csv",
                na_values=["NaN", "none", "None", "nan"],
            )
            data.insert(
                0,
                "name",
                [f"{opt.name}_{a}_{s}" for a, s in zip(data["attempt"], data["seq_id"])],
            )
        else:
            subdirs = True
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
            print(dna_s)
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


if __name__ == "__main__":
    main()
