# Adapted from BindCraft, (c) 2025 Martin Pacesa
#
# MIT License

# Copyright (c) 2024 Martin Pacesa

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import jax
import jax.numpy as jnp
import numpy as np

import pyrosetta as pr
from pyrosetta.rosetta.core.select.residue_selector import ChainSelector, NotResidueSelector
from pyrosetta.rosetta.protocols.grafting.simple_movers import DeleteRegionMover

from pyrosetta.rosetta.protocols.analysis import InterfaceAnalyzerMover
from pyrosetta.rosetta.protocols.rosetta_scripts import XmlObjects

from flexcraft.files.pdb import PDBFile
from flexcraft.data.data import DesignData
import flexcraft.sequence.aa_codes as aas

def score_interface(pdb_file: PDBFile | str, is_target, set_int):
    # load pose
    if isinstance(pdb_file, str):
        pdb_file = PDBFile(path=str)
    pose = pr.pose_from_pdb(pdb_file.path)
    ###---------------------------
    # Rosetta cannot handle poses with more than 3 chains during interface calculation
    # the program crashes even if only two chains are specified for interface calculation if more then three total are present
    # therefore, delete everything except for the two selected chains from the Rosetta pose
    # select the two chains from set_int for Rosetta interface calculation
    if set_int:
        # Convert "A_B" format from set_int to "A,B" for ChainSelector
        # then select the chains we want to keep
        chains_to_keep = set_int.replace('_', ',')
        keep_selector = ChainSelector(chains_to_keep)
        keep_selection = keep_selector.apply(pose)
      
        if any(keep_selection):
            remove_selector = NotResidueSelector(keep_selector)
            remove_selection = remove_selector.apply(pose)
            # now, select the chains we do NOT want to keep and delete them from the Rosetta pose
            if any(remove_selection):
                deleter = DeleteRegionMover()
                deleter.set_residue_selector(remove_selector)
                deleter.set_rechain(False) # Keep original chain IDs (A, B, etc)
                deleter.apply(pose)
        else:
            print("Found no matching chains for the Rosetta interface calculations, please check set_int flag.")
    ###---------------------------
    # analyze interface statistics
    iam = InterfaceAnalyzerMover()
    iam.set_interface(set_int)
    scorefxn = pr.get_fa_scorefxn()
    iam.set_scorefunction(scorefxn)
    iam.set_compute_packstat(True)
    iam.set_compute_interface_energy(True)
    iam.set_calc_dSASA(True)
    iam.set_calc_hbond_sasaE(True)
    iam.set_compute_interface_sc(True)
    iam.set_pack_separated(True)
    iam.apply(pose)

    # Initialize dictionary with all amino acids
    interface_AA = {aa: 0 for aa in 'ACDEFGHIKLMNPQRSTVWY'}

    # Initialize list to store PDB residue IDs at the interface
    data: DesignData = pdb_file.data

    #-----------------------
    # for the interface residues, we need to ignore the same chains that we ignored in the rosetta pose
    if set_int:
        # convert set_int to list: "A_B" -> ['A', 'B'])
        allowed_chains = set_int.split('_')
        # Map integer chain indices (0, 1, 2) to characters ('A', 'B', 'C')
        chain_map = np.array(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"))
        # Get the character chain ID for every atom in the structure
        # data["chain_index"] contains integers, we map them to chars
        atom_chain_ids = chain_map[data["chain_index"]]
        # Create a boolean mask: True if the atom belongs to a chain in set_int
        allowed_mask = np.isin(atom_chain_ids, allowed_chains)
    else:
        # If no set_int provided, keep everything
        allowed_mask = np.ones(len(data["chain_index"]), dtype=bool)
    # ----------------------

    target_data = data[is_target & allowed_mask]
    binder_data = data[(~is_target) & allowed_mask]
    hotspot = jnp.linalg.norm(target_data["atom_positions"][:, None, 1] - binder_data["atom_positions"][None, :, 1], axis=-1)
    hotspot = (hotspot < 8.0).any(axis=1)
    aa = aas.decode(target_data["aa"][hotspot], aas.AF2_CODE)

    # Iterate over the interface residues
    for aa_type in aa:
        # Increase the count for this amino acid type
        interface_AA[aa_type] += 1

    # count interface residues
    interface_nres = len(aa)

    # Calculate the percentage of hydrophobic residues at the interface of the binder
    hydrophobic_aa = set('ACFILMPVWY')
    hydrophobic_count = sum(interface_AA[aa] for aa in hydrophobic_aa)
    if interface_nres != 0:
        interface_hydrophobicity = (hydrophobic_count / interface_nres) * 100
    else:
        interface_hydrophobicity = 0

    # retrieve statistics
    interfacescore = iam.get_all_data()
    interface_sc = interfacescore.sc_value # shape complementarity
    interface_interface_hbonds = interfacescore.interface_hbonds # number of interface H-bonds
    interface_dG = iam.get_interface_dG() # interface dG
    interface_dSASA = iam.get_interface_delta_sasa() # interface dSASA (interface surface area)
    interface_packstat = iam.get_interface_packstat() # interface pack stat score
    interface_dG_SASA_ratio = interfacescore.dG_dSASA_ratio * 100 # ratio of dG/dSASA (normalised energy for interface area size)
    buns_filter = XmlObjects.static_get_filter('<BuriedUnsatHbonds report_all_heavy_atom_unsats="true" scorefxn="scorefxn" ignore_surface_res="false" use_ddG_style="true" dalphaball_sasa="1" probe_radius="1.1" burial_cutoff_apo="0.2" confidence="0" />')
    interface_delta_unsat_hbonds = buns_filter.report_sm(pose)

    if interface_nres != 0:
        interface_hbond_percentage = (interface_interface_hbonds / interface_nres) * 100 # Hbonds per interface size percentage
        interface_bunsch_percentage = (interface_delta_unsat_hbonds / interface_nres) * 100 # Unsaturated H-bonds per percentage
    else:
        interface_hbond_percentage = 0
        interface_bunsch_percentage = 100
    
    ##########################################################
    # calculate binder energy score
    allowed_chains = set_int.split('_') # convert set_int to list: "A_B" -> ['A', 'B'])
    binder_chain=allowed_chains[-1]
    
    chain_design = ChainSelector(binder_chain)
    
    tem = pr.rosetta.core.simple_metrics.metrics.TotalEnergyMetric()
    tem.set_scorefunction(scorefxn)
    tem.set_residue_selector(chain_design)
    binder_score = tem.calculate(pose)

    # calculate binder SASA fraction
    bsasa = pr.rosetta.core.simple_metrics.metrics.SasaMetric()
    bsasa.set_residue_selector(chain_design)
    binder_sasa = bsasa.calculate(pose)

    if binder_sasa > 0:
        interface_binder_fraction = (interface_dSASA / binder_sasa) * 100
    else:
        interface_binder_fraction = 0

    # calculate surface hydrophobicity
    binder_pose = {pose.pdb_info().chain(pose.conformation().chain_begin(i)): p for i, p in zip(range(1, pose.num_chains()+1), pose.split_by_chain())}[binder_chain]

    layer_sel = pr.rosetta.core.select.residue_selector.LayerSelector()
    layer_sel.set_layers(pick_core = False, pick_boundary = False, pick_surface = True)
    surface_res = layer_sel.apply(binder_pose)

    exp_apol_count = 0
    total_count = 0 
    
    # count apolar and aromatic residues at the surface
    for i in range(1, len(surface_res) + 1):
        if surface_res[i] == True:
            res = binder_pose.residue(i)

            # count apolar and aromatic residues as hydrophobic
            if res.is_apolar() == True or res.name() == 'PHE' or res.name() == 'TRP' or res.name() == 'TYR':
                exp_apol_count += 1
            total_count += 1

    surface_hydrophobicity = exp_apol_count/total_count

    # output interface score array and amino acid counts at the interface
    interface_scores = {
        'binder_score': binder_score,
        'surface_hydrophobicity': surface_hydrophobicity,
        'interface_sc': interface_sc,
        'interface_packstat': interface_packstat,
        'interface_dG': interface_dG,
        'interface_dSASA': interface_dSASA,
        'interface_dG_SASA_ratio': interface_dG_SASA_ratio,
        'interface_fraction': interface_binder_fraction,
        'interface_hydrophobicity': interface_hydrophobicity,
        'interface_nres': interface_nres,
        'interface_interface_hbonds': interface_interface_hbonds,
        'interface_hbond_percentage': interface_hbond_percentage,
        'interface_delta_unsat_hbonds': interface_delta_unsat_hbonds,
        'interface_delta_unsat_hbonds_percentage': interface_bunsch_percentage
    }

    # round to two decimal places
    interface_scores = {k: round(v, 2) if isinstance(v, float) else v for k, v in interface_scores.items() if v is not None} # FIXME

    return interface_scores, interface_AA