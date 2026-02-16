import numpy as np
from typing import Dict

from salad.aflib.common.protein import Protein, from_pdb_string
from salad.aflib.model.all_atom_multimer import atom37_to_atom14, get_atom14_mask
from salad.aflib.model.geometry import Vec3Array

from flexcraft.data.data import DesignData

def data_from_protein(protein: Protein):
    """Convert an aflib or AlphaFold Protein to DesignData."""
    positions, atom_mask = atom37_to_atom14(
        protein.aatype,
        Vec3Array.from_array(protein.atom_positions),
        protein.atom_mask)
    return DesignData.from_dict(dict(
        aa=protein.aatype,
        atom_positions=positions.to_array(),
        atom_mask=atom_mask,
        residue_index=protein.residue_index,
        chain_index=protein.chain_index,
        batch_index=np.zeros_like(protein.chain_index),
        tie_index=np.arange(protein.aatype.shape[0], dtype=np.int32),
        tie_weights=np.ones((protein.aatype.shape[0],), dtype=np.float32),
        mask=atom_mask.any(axis=1),
        plddt=protein.b_factors[:, 1],
    ))

def data_from_salad(data: dict) -> DesignData:
    """Convert a salad data dictionary to DesignData."""
    atom_mask = get_atom14_mask(data["aatype"])
    return DesignData.from_dict(dict(
        aa=data["aatype"],
        atom_positions=data["atom_pos"],
        atom_mask=atom_mask,
        residue_index=data["residue_index"],
        chain_index=data["chain_index"],
        batch_index=data["batch_index"],
        tie_index=np.arange(data["aatype"].shape[0], dtype=np.int32),
        tie_weights=np.ones((data["aatype"].shape[0],), dtype=np.float32),
        mask=data["mask"] * get_atom14_mask(data["aatype"]).any(axis=1)
    ))

def load_pdb(path: str):
    """Load a PDB file as DesignData."""
    with open(path, "rt") as f:
        protein = from_pdb_string(f.read())
    return data_from_protein(protein)

def strip_aa(data):
    """Remove amino acid information form a data dictionary."""
    return {
        k: v if k != "aa" else np.full_like(v, 20)
        for k, v in data.items()
    }

def tie_homomer(data, num_monomers=1):
    """Tie a data dictionary as if it were a homooligomer."""
    monomer_L = data["aa"].shape[0] // num_monomers
    monomer_tie_index = np.arange(monomer_L, dtype=np.int32)
    homomer_tie_index = np.concatenate(num_monomers * [monomer_tie_index], axis=0)
    homomer_tie_weights = data["tie_weights"] / num_monomers
    out = {k: v for k, v in data.items()}
    out["tie_index"] = homomer_tie_index
    out["tie_weights"] = homomer_tie_weights
    return out
