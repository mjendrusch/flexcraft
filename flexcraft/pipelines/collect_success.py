import os
import shutil
from flexcraft.utils import load_pdb
import flexcraft.sequence.aa_codes as aas
from flexcraft.utils.options import parse_options

opt = parse_options(
    "Collect successful designs and metrics from a directory of flexcraft runs.",
    in_path="inputs/",
    out_path="outputs/",
    collect_chains="all",
    dirs_match="none",
    all_relaxed="False"
)

os.makedirs(f"{opt.out_path}/designs/", exist_ok=True)

with open(f"{opt.out_path}/success.csv", "wt") as out_f, open(f"{opt.out_path}/success.fa", "wt") as fa_f:
    header_done = False
    for subdir in os.listdir(opt.in_path):
        if not os.path.isdir(f"{opt.in_path}/{subdir}"):
            continue
        if opt.dirs_match != "none" and not subdir.startswith(opt.dirs_match):
            continue
        if opt.all_relaxed == "True":
            path = f"{opt.in_path}/{subdir}/all.csv"
        else:
            path = f"{opt.in_path}/{subdir}/success.csv"
        if not os.path.isfile(path):
            continue
        with open(path, "rt") as in_f:
            header = next(in_f).strip().split(",")
            header = "run," + ",".join(header) + "\n"
            if not header_done:
                out_f.write(header)
                header_done = True
            if opt.all_relaxed == "True":
                reason_offset = header.split(",").index("reason") - 1
                for line in in_f:
                    reason = line.strip().split(",")[reason_offset]
                    if "designability" in reason:
                        continue
                    out_f.write(subdir + "," + line)
            else:
                for line in in_f:
                    out_f.write(subdir + "," + line)
        if opt.all_relaxed == "True":
            success_path = f"{opt.in_path}/{subdir}/relaxed/"
        else:
            success_path = f"{opt.in_path}/{subdir}/success/"
        for file_name in os.listdir(success_path):
            design_path = f"{success_path}/{file_name}"
            data = load_pdb(design_path)
            sequence = data.to_sequence_string()
            chains = {
                cid: seq
                for seq, cid in zip(
                    sequence.split(":"),
                    "ABCDEFGHIJKLMNOPQRSTUVWXYZ")
            }
            if opt.collect_chains != "all":
                sequence = ":".join([chains[c] for c in opt.collect_chains.split(",")])
            shutil.copyfile(f"{opt.in_path}/{subdir}/success/{file_name}",
                            f"{opt.out_path}/designs/{subdir}_{file_name}")
            fa_f.write(f">{subdir}_{file_name}\n{sequence}\n")
