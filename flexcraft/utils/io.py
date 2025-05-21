import numpy as np
from typing import Dict

from salad.aflib.common.protein import Protein, from_pdb_string
from salad.aflib.model.all_atom_multimer import atom37_to_atom14
from salad.aflib.model.geometry import Vec3Array

def data_from_protein(protein: Protein):
    positions, atom_mask = atom37_to_atom14(
        protein.aatype,
        Vec3Array.from_array(protein.atom_positions),
        protein.atom_mask)
    return dict(
        aa=protein.aatype,
        atom_positions=positions.to_array(),
        atom_mask=atom_mask,
        residue_index=protein.residue_index,
        chain_index=protein.chain_index,
        batch_index=np.zeros_like(protein.chain_index),
        tie_index=np.arange(protein.aatype.shape[0], dtype=np.int32),
        tie_weights=np.ones((protein.aatype.shape[0],), dtype=np.float32),
        mask=atom_mask.any(axis=1)
    )

def load_pdb(path: str) -> Dict[str, np.ndarray]:
    with open(path, "rt") as f:
        protein = from_pdb_string(f.read())
    return data_from_protein(protein)

def strip_aa(data):
    return {
        k: v if k != "aa" else np.full_like(v, 20)
        for k, v in data.items()
    }

def tie_homomer(data, num_monomers=1):
    monomer_L = data["aa"].shape[0] // num_monomers
    monomer_tie_index = np.arange(monomer_L, dtype=np.int32)
    homomer_tie_index = np.concatenate(num_monomers * [monomer_tie_index], axis=0)
    homomer_tie_weights = data["tie_weights"] / num_monomers
    out = {k: v for k, v in data.items()}
    out["tie_index"] = homomer_tie_index
    out["tie_weights"] = homomer_tie_weights
    return out
