from collections import defaultdict

import jax
import jax.numpy as jnp
import numpy as np
import sys

from flexcraft.files.pdb import PDBFile
from flexcraft.data.data import DesignData
from flexcraft.structure.af import (
    AFInput, AFResult, get_model_haiku_params, model_config, make_af2, make_predict)
from flexcraft.rosetta.relax import fastrelax
from flexcraft.rosetta.interface_analyzer import score_interface
from salad.modules.utils.geometry import index_align


class BindCraftProperties:
    def __init__(self, path, key, af_parameter_path, filter=None, 
                 use_guess=False, relaxed_name="relaxed", ipae_shortcut_threshold=0.35):
        
        AVAILABLE_FILTERS = {"default": default_filter}

        if filter is None:
            xfilter = default_filter
        elif isinstance(filter, str):
            if filter in AVAILABLE_FILTERS: 
                xfilter = AVAILABLE_FILTERS[filter]
            else: 
                raise ValueError(f"ERROR: The provided filter '{filter}' is not a valid filter option. Available filters are: {list(AVAILABLE_FILTERS.keys())}.")
        elif callable(filter):
            xfilter=filter
        else:
            raise ValueError("ERROR: The 'filter' argument must be a string, a callable function, or None.")

        self.ipae_shortcut_threshold = ipae_shortcut_threshold
        self.path = path
        self.relaxed_name = relaxed_name
        self.af2_params = [
            get_model_haiku_params(
                model_name=f"model_{model}_ptm",
                data_dir=af_parameter_path, fuse=True)
            for model in (1, 2)]
        af2_config = model_config("model_1_ptm")
        af2_config.model.global_config.use_dgram = False
        self.use_guess = use_guess
        self.af2 = jax.jit(make_predict(make_af2(af2_config), num_recycle=4))
        self.key = key
        self.filter=xfilter
    

    def __call__(self, name, design: DesignData, is_target: jnp.ndarray):
        # c = self.config
        af_input = (AFInput
                    .from_data(design)
                    .add_template(design, where=is_target))
        num_chains = len(np.unique(design["chain_index"]))
        af_input = af_input.block_diagonal(num_sequences=num_chains)
        if self.use_guess:
            af_input = af_input.add_guess(design)
        result = dict()
        af_results = []
        for model, params in enumerate(self.af2_params, 1):
            af_result: AFResult = self.af2(params, self.key(), af_input)
            af_results.append(af_result)
            metrics = self.metrics_from_result(name + f"_model_{model}", af_result, design, is_target)
            for key, value in metrics.items():
                if "AAs" in key:
                    continue
                result[f"{model}_{key}"] = value
                avg_key = f"Average_{key}"
                if avg_key not in result:
                    result[avg_key] = 0
                result[avg_key] += value / len(self.af2_params)
        result["success"] = self.filter(result)
        return af_results[0], result

    def metrics_from_result(self,
                           name: str,
                           af_result: AFResult,
                           design: DesignData,
                           is_target: jnp.ndarray,) -> dict:
        result = dict()
        is_binder = ~is_target
        pair_is_binder = is_binder[:, None] * is_binder[None, :]
        af_result.inputs["chain_index"] = is_target.astype(jnp.int32)
        # AlphaFold 2 metrics
        result["Binder_pLDDT"] = (is_binder * af_result.plddt).sum() / jnp.maximum(is_binder.sum(), 1)
        result["pLDDT"] = af_result.plddt.mean()
        result["Binder_pTM"] = ((pair_is_binder * af_result.ptm_matrix(is_binder.sum())).sum(axis=1) / jnp.maximum(pair_is_binder.sum(), 1)).max(axis=0)
        result["pTM"] = af_result.ptm
        result["i_pTM"] = af_result.iptm
        result["pAE"] = af_result.pae.mean()
        result["Binder_pAE"] = (af_result.pae * pair_is_binder).sum() / jnp.maximum(pair_is_binder.sum(), 1)
        result["i_pAE"] = af_result.ipae
        # FIXME if pass, relax the af2 structure. Else, continue
        if result["i_pAE"] >= self.ipae_shortcut_threshold:
            result.update({
                'Binder_Energy_Score': np.nan,
                'Surface_Hydrophobicity': np.nan,
                'ShapeComplementarity': np.nan,
                'PackStat': np.nan,
                'dG': np.nan,
                'dSASA': np.nan, 
                'dG/dSASA': np.nan,
                'Interface_SASA_%': np.nan,
                'Interface_Hydrophobicity': np.nan,
                'n_InterfaceResidues': np.nan,
                'n_InterfaceHbonds': np.nan,
                'InterfaceHbondsPercentage': np.nan,
                'n_InterfaceUnsatHbonds': np.nan,
                'InterfaceUnsatHbondsPercentage': np.nan,
                'InterfaceAAs': None,
            })
            return result
        # Rosetta relax the af2 structure
        relaxed: PDBFile = fastrelax(af_result.to_data(), f"{self.path}/{self.relaxed_name}/{name}.pdb")
        relaxed_data = relaxed.to_data()
        if_scores, if_aa = score_interface(relaxed, is_target)

        # alignment / RMSD
        aligned_positions = index_align(
            relaxed_data["atom_positions"], design["atom_positions"], design["batch_index"], design["mask"])
        result["RMSD"] = jnp.sqrt(((aligned_positions[:, 1] - design["atom_positions"][:, 1]) ** 2).mean())
        aligned_positions = index_align(
            relaxed_data["atom_positions"][is_target], design["atom_positions"][is_target],
            design["batch_index"][is_target], design["mask"][is_target])
        result["Target_RMSD"] = jnp.sqrt(((aligned_positions[:, 1] - design["atom_positions"][is_target, 1]) ** 2).mean())
        aligned_positions = index_align(
            relaxed_data["atom_positions"][is_binder], design["atom_positions"][is_binder],
            design["batch_index"][is_binder], design["mask"][is_binder])
        result["Binder_RMSD"] = jnp.sqrt(((aligned_positions[:, 1] - design["atom_positions"][is_binder, 1]) ** 2).mean())
        # adapted from BindCraft
        result.update({
            'Binder_Energy_Score': if_scores['binder_score'],
            'Surface_Hydrophobicity': if_scores['surface_hydrophobicity'],
            'ShapeComplementarity': if_scores['interface_sc'],
            'PackStat': if_scores['interface_packstat'],
            'dG': if_scores['interface_dG'],
            'dSASA': if_scores['interface_dSASA'], 
            'dG/dSASA': if_scores['interface_dG_SASA_ratio'],
            'Interface_SASA_%': if_scores['interface_fraction'],
            'Interface_Hydrophobicity': if_scores['interface_hydrophobicity'],
            'n_InterfaceResidues': if_scores['interface_nres'],
            'n_InterfaceHbonds': if_scores['interface_interface_hbonds'],
            'InterfaceHbondsPercentage': if_scores['interface_hbond_percentage'],
            'n_InterfaceUnsatHbonds': if_scores['interface_delta_unsat_hbonds'],
            'InterfaceUnsatHbondsPercentage': if_scores['interface_delta_unsat_hbonds_percentage'],
            'InterfaceAAs': if_aa,
        })
        print(result)
        return result

def default_filter(result):
    success = result["1_i_pAE"] < 0.35
    success = success and (result["2_i_pAE"] < 0.35)
    success = success and (result["1_pLDDT"] > 0.8)
    success = success and (result["2_pLDDT"] > 0.8)
    success = success and (result["1_pTM"] > 0.55)
    success = success and (result["2_pTM"] > 0.55)
    success = success and (result["1_i_pTM"] > 0.5)
    success = success and (result["2_i_pTM"] > 0.5)
    success = success and (result["1_Surface_Hydrophobicity"] < 0.35)
    success = success and (result["2_Surface_Hydrophobicity"] < 0.35)
    success = success and (result["1_ShapeComplementarity"] > 0.55)
    success = success and (result["2_ShapeComplementarity"] > 0.55)
    success = success and (result["Average_ShapeComplementarity"] > 0.6)
    success = success and (result["1_dSASA"] > 1.0) 
    success = success and (result["2_dSASA"] > 1.0) 
    success = success and (result["1_n_InterfaceResidues"] >= 7) 
    success = success and (result["2_n_InterfaceResidues"] >= 7) 
    success = success and (result["1_n_InterfaceHbonds"] >= 3) 
    success = success and (result["2_n_InterfaceHbonds"] >= 3) 
    success = success and (result["1_n_InterfaceUnsatHbonds"] < 10) 
    success = success and (result["2_n_InterfaceUnsatHbonds"] < 10) 
    success = success and (result["Average_RMSD"] < 2.0)
    return success
