import numpy as np

import jax
import jax.numpy as jnp

from salad.modules.utils.geometry import positions_to_ncacocb

def atom_array_to_atomX(atom_array, atom_to_token, num_atoms=14):
    residue_atom_count = atom_to_token.sum(axis=0)
    atomx_index = jnp.repeat(jnp.arange(num_atoms)[None], atom_to_token.shape[1], axis=0)
    atomx_mask = atomx_index < residue_atom_count[:, None]
    atomx_conindex = jnp.cumsum(atomx_mask.reshape(-1), axis=0).reshape(*atomx_mask.shape) - 1
    return atom_array[atomx_conindex], atomx_mask

def broadcast_array_to_atomX(residue_array, atom_to_token, num_atoms=14):
    residue_atom_count = atom_to_token.sum(axis=0)
    atomx_index = jnp.repeat(jnp.arange(num_atoms)[None], atom_to_token.shape[1], axis=0)
    atomx_mask = atomx_index < residue_atom_count[:, None]
    atomx = jnp.repeat(residue_array[:, None], num_atoms, axis=1)
    return atomx, atomx_mask

def atomX_to_atom_array(atomx, atom_to_token):
    consecutive_atoms = jnp.arange(atom_to_token.shape[0])
    consecutiveX, x_mask = atom_array_to_atomX(
        consecutive_atoms, atom_to_token, num_atoms=atomx.shape[1])
    atom_residue_index = np.argmax(atom_to_token, axis=1)
    first_atom_in_residue = consecutiveX[atom_residue_index, 0]
    start = atom_residue_index * atomx.shape[1]
    atomx_to_atom_array_index = consecutive_atoms + start - first_atom_in_residue
    return atomx.reshape(-1)[atomx_to_atom_array_index]

def _compute_padding_size(atom_to_token, mol_type, num_atoms=14):
    protein_residues = (mol_type == 0)
    num_protein_residues = protein_residues.astype(np.int32).sum()
    is_aa_atom = np.einsum("at,t->a", atom_to_token, protein_residues) > 0
    # get number of context atoms
    num_aa_atoms = is_aa_atom.astype(np.int32).sum()
    num_all_atoms = is_aa_atom.shape[0]
    num_non_aa_atoms = num_all_atoms - num_aa_atoms
    # get number of padded (to at most 14) amino acid atoms
    # add 1 to account for terminal OXT ?
    total_padded_size = num_protein_residues * num_atoms + num_non_aa_atoms
    total_padded_size = (total_padded_size // 32 + 1) * 32
    return total_padded_size

def _pad_to_size(atom_array, total_padded_size, axis=0):
    shape = list(atom_array.shape)
    axis_size = shape[axis]
    remaining = total_padded_size - axis_size
    result = atom_array
    if remaining > 0:
        padding_array_shape = shape[:axis] + [remaining] + shape[axis + 1:]
        padding_array = np.zeros(padding_array_shape, dtype=atom_array.dtype)
        result = np.concatenate((result, padding_array), axis=axis)
    return result

def pad_boltz_atom_features_for_compilation(features):
    atom_to_token = features["atom_to_token"][0]
    mol_type = features["mol_type"][0]
    total_padded_size = _compute_padding_size(atom_to_token, mol_type)
    num_atoms = atom_to_token.shape[0]
    num_tokens = atom_to_token.shape[1]
    result = dict()
    for key, value in features.items():
        vshape = list(value.shape)
        while num_atoms in vshape:
            axis = vshape.index(num_atoms)
            value = _pad_to_size(value, total_padded_size, axis=axis)
            vshape = list(value.shape)
        result[key] = value
    return result

def get_contact_atom(atom24, mol_type):
    protein = positions_to_ncacocb(atom24)[:, 4] # (pseudo CB)
    smolecule = atom24[:, 0] # first and only atom
    dna = atom24[:, 11] # base N atom
    rna = atom24[:, 12] # base N atom - one more because of 2' hydroxyl
    protein = jnp.where((mol_type == 0)[:, None], protein, 0.0)
    dna = jnp.where((mol_type == 1)[:, None], dna, 0.0)
    rna = jnp.where((mol_type == 2)[:, None], rna, 0.0)
    smolecule = jnp.where((mol_type == 3)[:, None], smolecule, 0.0)
    return protein + dna + rna + smolecule
