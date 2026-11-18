import jax
import jax.numpy as jnp
import numpy as np

from flexcraft.files.pdb import PDBFile
from flexcraft.data.data import DesignData
from flexcraft.structure.af import (
    AFInput, AFResult, get_model_haiku_params, model_config, make_af2, make_predict)
from flexcraft.structure.metrics import RMSD


def _load_rosetta_helpers():
    """Lazily import PyRosetta relaxation and interface scoring helpers."""
    try:
        import pyrosetta  # noqa: F401
        from flexcraft.rosetta.relax import fastrelax
        from flexcraft.rosetta.interface_analyzer import score_interface
    except ImportError as exc:
        raise ImportError(
            "PyRosetta is required for BindCraft Rosetta relaxation and interface scoring. "
            "Install PyRosetta into `.venv` and preserve it across syncs with "
            "`uv sync --inexact` (or run commands with `uv run --no-sync`)."
        ) from exc
    return fastrelax, score_interface

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
        af_input = (AFInput
                    .from_data(design)
                    .add_template(design, where=is_target))
        valid_chains = np.asarray(design["chain_index"])
        valid_chains = valid_chains[valid_chains >= 0]
        num_chains = max(1, len(np.unique(valid_chains)))
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
        result["success"], result["reason"] = self.filter(result)
        return af_results[0], result

    def metrics_from_result(self,
                           name: str,
                           af_result: AFResult,
                           design: DesignData,
                           is_target: jnp.ndarray,) -> dict:
        result = dict()
        pad_mask = jnp.asarray(af_result.token_pad_mask, dtype=jnp.bool_)
        is_target_arr = jnp.asarray(is_target, dtype=jnp.bool_)
        if is_target_arr.shape[0] < pad_mask.shape[0]:
            is_target_arr = jnp.pad(
                is_target_arr,
                (0, pad_mask.shape[0] - is_target_arr.shape[0]),
                constant_values=False,
            )
        valid_binder = (~is_target_arr) & pad_mask
        binary_asym_id = jnp.where(pad_mask, valid_binder.astype(jnp.int32), -1)
        af_result.inputs = dict(af_result.inputs)
        af_result.inputs["asym_id"] = binary_asym_id
        if "chain_index" in af_result.inputs:
            af_result.inputs["chain_index"] = binary_asym_id
        # AlphaFold 2 metrics (complex pTM evaluated on af_result.ptm)
        result["Binder_pLDDT"] = af_result.index_plddt(valid_binder)
        result["pLDDT"] = af_result.mean_plddt
        result["Binder_pTM"] = af_result.index_ptm(valid_binder)
        result["pTM"] = af_result.ptm
        result["i_pTM"] = af_result.iptm
        result["pAE"] = af_result.mean_pae
        result["Binder_pAE"] = af_result.chain_pae(1, 1)
        result["i_pAE"] = af_result.ipae
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
        fastrelax, score_interface = _load_rosetta_helpers()
        unpadded_design = design.unpad()
        af_data = af_result.unpad().to_data().update(
            chain_index=unpadded_design["chain_index"]
        )
        is_target_host = np.asarray(is_target_arr[np.asarray(pad_mask, dtype=bool)], dtype=bool)
        is_binder_host = ~is_target_host
        relaxed: PDBFile = fastrelax(af_data, f"{self.path}/{self.relaxed_name}/{name}.pdb")
        relaxed_data = relaxed.to_data().unpad()
        if_scores, if_aa = score_interface(relaxed, is_target_host)
        # alignment / RMSD
        result["RMSD"] = RMSD()(
            relaxed_data, unpadded_design,
            unpadded_design["batch_index"], unpadded_design["mask"])
        relaxed_target = relaxed_data[is_target_host]
        design_target = unpadded_design[is_target_host]
        result["Target_RMSD"] = RMSD()(
            relaxed_target, design_target,
            design_target["batch_index"], design_target["mask"])
        relaxed_binder = relaxed_data[is_binder_host]
        design_binder = unpadded_design[is_binder_host]
        result["Binder_RMSD"] = RMSD()(
            relaxed_binder, design_binder,
            design_binder["batch_index"], design_binder["mask"])
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
    reason = []

    pae_success = result["1_i_pAE"] < 0.35
    pae_success = pae_success and (result["2_i_pAE"] < 0.35)
    if not pae_success:
        reason.append("ipAE")
    success = pae_success
    plddt_success = (result["1_pLDDT"] > 0.8)
    plddt_success = plddt_success and (result["2_pLDDT"] > 0.8)
    if not plddt_success:
        reason.append("pLDDT")
    success = success and plddt_success
    ptm_success = (result["1_pTM"] > 0.55)
    ptm_success = ptm_success and (result["2_pTM"] > 0.55)
    ptm_success = ptm_success and (result["1_i_pTM"] > 0.5)
    ptm_success = ptm_success and (result["2_i_pTM"] > 0.5)
    if not ptm_success:
        reason.append("pTM")
    success = success and ptm_success
    hydrophobicity_success =  (result["1_Surface_Hydrophobicity"] < 0.35)
    hydrophobicity_success = hydrophobicity_success and (result["2_Surface_Hydrophobicity"] < 0.35)
    if not hydrophobicity_success:
        reason.append("hydrophobicity")
    success = success and hydrophobicity_success
    shape_success = (result["1_ShapeComplementarity"] > 0.55)
    shape_success = shape_success and (result["2_ShapeComplementarity"] > 0.55)
    shape_success = shape_success and (result["Average_ShapeComplementarity"] > 0.6)
    if not shape_success:
        reason.append("shape_complementarity")
    success = success and shape_success
    sasa_success = (result["1_dSASA"] > 1.0)
    sasa_success = sasa_success and (result["2_dSASA"] > 1.0)
    if not sasa_success:
        reason.append("dSASA")
    success = success and sasa_success
    interface_success = (result["1_n_InterfaceResidues"] >= 7)
    interface_success = interface_success and (result["2_n_InterfaceResidues"] >= 7)
    interface_success = interface_success and (result["1_n_InterfaceHbonds"] >= 3)
    interface_success = interface_success and (result["2_n_InterfaceHbonds"] >= 3)
    interface_success = interface_success and (result["1_n_InterfaceUnsatHbonds"] < 10)
    interface_success = interface_success and (result["2_n_InterfaceUnsatHbonds"] < 10)
    if not interface_success:
        reason.append("interface")
    success = success and interface_success
    if success:
        rmsd_success = (result["Average_RMSD"] < 2.0)
        if not rmsd_success:
            reason.append("RMSD")
        success = success and rmsd_success
    if not reason:
        reason = "none"
    else:
        reason = ";".join(reason)
    return success, reason
