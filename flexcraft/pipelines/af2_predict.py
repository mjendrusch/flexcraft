import os

import jax
import jax.numpy as jnp
import numpy as np

import flexcraft.sequence.aa_codes as aas
from flexcraft.data.data import DesignData
from flexcraft.files.pdb import PDBFile
from flexcraft.structure.af import (
    AFInput,
    AFResult,
    get_model_haiku_params,
    make_af2,
    make_predict,
    model_config,
)
from flexcraft.structure.metrics import RMSD
from flexcraft.utils import Keygen, parse_options


def _decode_sequence(sequence) -> str:
    """Return a 1-letter amino acid string whether `sequence` is `str` or integer array."""
    if isinstance(sequence, str):
        return sequence
    return aas.decode(sequence, aas.AF2_CODE)


def main():
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
        seed=42,
    )
    os.makedirs(f"{opt.out_path}/predictions/", exist_ok=True)

    key = Keygen(opt.seed)

    af2_params = get_model_haiku_params(
        model_name=opt.af2_model, data_dir=opt.af2_params, fuse=True
    )
    af2_config = model_config(opt.af2_model)
    af2_config.model.global_config.use_dgram = False
    af2 = jax.jit(make_predict(make_af2(af2_config), num_recycle=2))

    names = []
    sequences = []
    structures = []

    if opt.fasta_path != "none":
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

    chains = np.array([int(c) for c in opt.chains.strip().split(",")])
    if opt.pdb_path != "none":
        file_names = [
            f for f in sorted(os.listdir(opt.pdb_path)) if f.endswith((".pdb", ".pdb1"))
        ]
        names = [c.split(".")[0] for c in file_names]
        structures = []
        sequences = []
        for file_name in file_names:
            file_path = f"{opt.pdb_path}/{file_name}"
            structure = PDBFile(path=file_path).to_data().unpad()
            structure = structure[
                (structure["chain_index"][:, None] == chains).any(axis=1)
            ]
            sequences.append(structure["aa"])
            structures.append(structure)

    with open(f"{opt.out_path}/scores.csv", "wt") as out_f:
        out_f.write("name,sequence,plddt,pae,ptm,sc_rmsd,no_motif_rmsd\n")
        for name, sequence, structure in zip(names, sequences, structures):
            sequence_str = _decode_sequence(sequence)
            flat_sequence = sequence_str.replace(":", "")
            motif_mask = jnp.zeros((len(flat_sequence),), dtype=jnp.bool_)
            if opt.motif_sequence != "none":
                motif_seq = opt.motif_sequence.strip()
                motif_start = flat_sequence.find(motif_seq)
                if motif_start >= 0:
                    motif_end = motif_start + len(motif_seq)
                    motif_mask = motif_mask.at[motif_start:motif_end].set(True)
            if structure is not None:
                af_input = AFInput.from_data(structure).add_guess(structure)
            elif ":" in sequence_str:
                af_input = AFInput.from_data(DesignData.from_sequence(sequence_str))
            else:
                af_input = AFInput.from_sequence(sequence_str)
            result: AFResult = af2(af2_params, key(), af_input).unpad()
            pad_mask = result.token_pad_mask
            eval_mask = (~motif_mask) & pad_mask
            plddt = float(
                (result.plddt * eval_mask).sum() / jnp.maximum(1, eval_mask.sum())
            )
            pae = float(result.mean_pae)
            ptm = float(result.ptm)
            if structure is not None:
                pred_data = result.to_data()
                sc_rmsd = float(RMSD()(pred_data, structure))
                no_motif_rmsd = float(
                    RMSD()(pred_data, structure, weight=~motif_mask, mask=~motif_mask)
                )
            else:
                sc_rmsd = -1.0
                no_motif_rmsd = -1.0

            out_f.write(
                f"{name},{sequence_str},{plddt:.3f},{pae:.3f},{ptm:.3f},"
                f"{sc_rmsd:.2f},{no_motif_rmsd:.2f}\n"
            )
            result.save_pdb(f"{opt.out_path}/predictions/{name}.pdb")
            out_f.flush()


if __name__ == "__main__":
    main()
