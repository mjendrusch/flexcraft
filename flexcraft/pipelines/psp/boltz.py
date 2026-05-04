
import os

import jax
import jax.numpy as jnp

from flexcraft.structure.af import *
from flexcraft.utils import Keygen, parse_options
from flexcraft.files import ScoreCSV, FastaFile

from flexcraft.structure.boltz._model import Joltz2, JoltzResult

def ptm_score(logits, pair_mask=None, num_res=None):
    if pair_mask is None:
        pair_mask = jnp.ones(shape=(logits.shape[-2], logits.shape[-2]), dtype=bool)
    if num_res is None:
        num_res = pair_mask.astype(jnp.int32).sum(axis=-1)
    # Clip num_res to avoid negative/undefined d0.
    clipped_num_res = jnp.maximum(num_res, 19)
    d0 = 1.24 * (clipped_num_res - 15) ** (1.0 / 3) - 1.8

    # Convert logits to probs.
    probs = jax.nn.softmax(logits, axis=-1)

    # TM-Score term for every bin.
    bin_centers = (jnp.arange(64) / 64 + 1 / 128) * 32
    tm_per_bin = 1.0 / (1 + jnp.square(bin_centers) / jnp.square(d0[:, None, None]))
    # E_distances tm(distance).
    predicted_tm_term = jnp.sum(probs * tm_per_bin, axis=-1)

    predicted_tm_term *= pair_mask

    pair_residue_weights = pair_mask
    normed_residue_mask = pair_residue_weights / (
        1e-8 + jnp.sum(pair_residue_weights, axis=-1, keepdims=True)
    )
    per_alignment = jnp.sum(predicted_tm_term * normed_residue_mask, axis=-1)
    return per_alignment.max(), per_alignment.reshape(per_alignment.shape[0], -1).max(axis=1)

def ipsae_score(
    logits: jnp.ndarray,
    asym_id: jnp.ndarray | None = None,
    pae_cutoff: float = 10.0,
) -> jnp.ndarray:
    bin_centers = (jnp.arange(64) / 64 + 1 / 128) * 32
    probs = jax.nn.softmax(logits, axis=-1)
    pae = jnp.sum(probs * bin_centers, axis=-1)

    pair_mask = jnp.ones_like(pae, dtype=bool)
    pair_mask *= asym_id[:, None] != asym_id[None, :]

    # only include residue pairs below the pae_cutoff
    pair_mask *= pae < pae_cutoff
    n_residues = jnp.sum(pair_mask, axis=-1, keepdims=True)

    # Compute adjusted d_0(num_res) per residue  as defined by eqn. (15) in
    # Dunbrack, R., "What's wrong with AlphaFold’s ipTM score and how to fix it."
    # 2025: https://pmc.ncbi.nlm.nih.gov/articles/PMC11844409/
    d0 = 1.24 * (jnp.clip(n_residues, min=27) - 15) ** (1.0 / 3) - 1.8

    tm_per_bin = 1.0 / (1 + jnp.square(bin_centers) / jnp.square(d0))
    predicted_tm_term = jnp.sum(probs * tm_per_bin, axis=-1)

    normed_residue_mask = pair_mask / (1e-8 + n_residues)
    per_alignment = jnp.sum(predicted_tm_term * normed_residue_mask, axis=-1)
    return per_alignment.max()

opt = parse_options(
    "Protein structure prediction with Boltz2",
    in_path="sequences.fa",
    out_path="outputs/",
    boltz_path="./params/boltz/",
    num_samples=1,
    num_recycle=4,
    seed=42
)
os.makedirs(f"{opt.out_path}/predictions/", exist_ok=True)

key = Keygen(opt.seed)
joltz = Joltz2(cache=opt.boltz_path).predictor_adhoc(
    num_recycle=opt.num_recycle, num_samples=opt.num_samples)
scores = ScoreCSV(
    path=f"{opt.out_path}/scores.csv",
    keys=["name", "sequence", "plddt", "ptm", "iptm", "ipsae"])
sequences = FastaFile(path=opt.in_path)

for name, sequence in sequences.items():
    name = name.replace(":", "_binds_")
    chain_sequences = sequence.split(":")
    input_chains = [
        dict(kind="protein", sequence=c, use_msa=True)
        for i, c in enumerate(chain_sequences)
    ]
    prediction = joltz(key(), *input_chains)
    plddt = prediction.data["confidence"].plddt.mean()
    chain_index = prediction.data["features"]["asym_id"][0]
    other_chain = chain_index[:, None] != chain_index[None, :]
    pae_logits = prediction.data["confidence"].pae_logits
    ptm, _ = ptm_score(pae_logits)
    iptm, index_iptm = ptm_score(pae_logits, pair_mask=other_chain)
    ipsae = ipsae_score(pae_logits, chain_index)
    for index in range(opt.num_samples):
        prediction.save_pdb(f"{opt.out_path}/predictions/{name}_model_{index}_{int(index_iptm[index] * 100)}.pdb", sample_index=index)
    scores.write_line(dict(name=name, sequence=sequence, plddt=plddt, ptm=ptm, iptm=iptm, ipsae=ipsae))
