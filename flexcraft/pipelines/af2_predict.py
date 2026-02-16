
import os
import time
from copy import deepcopy

import jax
import jax.numpy as jnp

import haiku as hk

import salad.inference as si

import flexcraft.sequence.aa_codes as aas
from flexcraft.structure.af import *
from flexcraft.utils import Keygen, parse_options, data_from_protein
from flexcraft.files.pdb import PDBFile
from flexcraft.structure.metrics import RMSD

opt = parse_options(
    "Protein structure prediction with AF2",
    fasta_path="none",
    pdb_path="none",
    out_path="outputs/",
    motif_sequence="none",
    template_motif="False",
    chains="0",
    af2_params="params/af/",
    af2_model="model_1_ptm",
    seed=42
)
os.makedirs(f"{opt.out_path}/predictions/", exist_ok=True)

key = Keygen(opt.seed)

# make AF2 model
af2_params = get_model_haiku_params(
    model_name=opt.af2_model,
    data_dir=opt.af2_params, fuse=True)
af2_config = model_config(opt.af2_model)
af2_config.model.global_config.use_dgram = False
af2 = jax.jit(make_predict(make_af2(af2_config), num_recycle=2))

if opt.fasta_path != "none":
    names = []
    sequences = []
    structures = []
    with open(opt.fasta_path, "rt") as f:
        while True:
            try:
                header = next(f)[1:].strip()
                sequence = next(f).strip()
                names.append(header)
                sequences.append(sequence)
                structures.append(None)
            except StopIteration:
                break

# FIXME
chains = np.array([int(c) for c in opt.chains.strip().split(",")])
if opt.pdb_path != "none":
    file_names = os.listdir(opt.pdb_path)
    names = [c.split(".")[0] for c in file_names]
    structures = []
    sequences = []
    for file_name in file_names:
        file_path = f"{opt.pdb_path}/{file_name}"
        structure = PDBFile(path=file_path).to_data()
        structure = structure[(structure["chain_index"][:, None] == chains).any(axis=1)]
        sequence = structure["aa"]
        structures.append(structure)
        sequences.append(sequence)

with open(f"{opt.out_path}/scores.csv", "wt") as out_f:
    out_f.write("name,sequence,plddt,pae,ptm,sc_rmsd,no_motif_rmsd\n")
    for name, sequence, structure in zip(names, sequences, structures):
        sequence = aas.decode(sequence, aas.AF2_CODE)
        motif_mask = jnp.zeros((len(sequence),), dtype=jnp.bool_)
        if opt.motif_sequence != "none":
            motif_start = sequence.find(opt.motif_sequence.strip())
            motif_end = motif_start + len(opt.motif_sequence.strip())
            motif_mask = motif_mask.at[motif_start:motif_end].set(True)
        if structure is not None:
            af_input = AFInput.from_data(structure).add_guess(structure)
        else:
            af_input = AFInput.from_sequence(sequence)
        result: AFResult = af2(af2_params, key(), af_input)
        plddt = (result.plddt * (~motif_mask)).sum() / jnp.maximum(1, (~motif_mask).sum())
        pae = result.pae.mean()
        ptm = result.bptm
        if structure is not None:
            sc_rmsd = RMSD()(result.to_data(), structure)
            no_motif_rmsd = RMSD()(result.to_data(), structure,
                                   weight=~motif_mask, mask=~motif_mask)
        else:
            sc_rmsd = -1.0

        out_f.write(f"{name},{sequence},{plddt:.3f},{pae:.3f},{ptm:.3f},{sc_rmsd:.2f},{no_motif_rmsd:.2f}\n")
        result.save_pdb(f"{opt.out_path}/predictions/{name}.pdb")
        out_f.flush()
