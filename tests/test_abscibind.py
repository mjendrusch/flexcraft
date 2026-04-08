'''Test script for the AbsciBind protocol ipTM scoring.'''

import argparse
import json
from pathlib import Path

import jax
import numpy as np
import pandas as pd

from flexcraft.protocols.AbsciBind import AbsciBind
from flexcraft.files.pdb import PDBFile
from flexcraft.structure.af import AFInput, AFResult
from flexcraft.utils.rng import Keygen

from abscibind import *

def parse_args():
    p = argparse.ArgumentParser(description="Run AbsciBind ipTM scoring benchmark.")
    p.add_argument("--data_dir", type=Path, default=Path("data/o1_iptm_scoring"),
                   help="Directory containing annotation.json, PDBs and CSVs (default: data/o1_iptm_scoring).")
    p.add_argument("--af_params", type=Path, default=Path("params/af/params"),
                   help="Path to AlphaFold2 parameter directory (default: params/af/params).")
    p.add_argument("--seed", type=int, default=0,
                   help="JAX PRNG seed.")
    p.add_argument("--targets", nargs="*", default=None,
                   help="Subset of scaffold names to run (default: all).")
    p.add_argument("--max_designs", type=int, default=None,
                   help="Maximum number of designs per scaffold (default: all).")
    p.add_argument("--fetch_data", action="store_true",
                   help="Download missing PDB/CSV files before running.")
    p.add_argument("--verbose", action="store_true",
                   help="Run in verbose mode. With continuous output and predicted structures as .pdb.")
    p.add_argument("--af_model", type=str, default="model_2_multimer_v3",
                   help="Name of the af model to use for inference (default: model_2_multimer_v3).")
    p.add_argument("--clip_ab", action="store_true",
               help="Clip antibody variable regions to 120 aas.")
    return p.parse_args()


def main():
    args = parse_args()

    if args.fetch_data:
        load_data(args.data_dir)

    key = Keygen(args.seed)
    out = abscibind_pipe(
        data_dir=args.data_dir,
        af_parameter_path=args.af_params,
        af2_key=key,
        targets=args.targets,
        max_designs=args.max_designs,
        verbose=args.verbose,
        model=args.af_model,
        clip_ab=args.clip_ab,
    )
    print(out)


if __name__ == "__main__":
    main()
