'''
AbsciBind protocol for scoring/filtering Binder Designs.
Adapted from: Levine, S. et al., 2026. Origin-1: a generative AI platform for de novo antibody design against novel epitopes. https://doi.org/10.64898/2026.01.14.699389
'''
from typing import Any

import jax
import jax.numpy as jnp
from flexcraft.structure.af import (
    AFInput, AFResult, get_model_haiku_params, model_config, make_af2, make_predict)
from flexcraft.structure.metrics import LRMSD, AbsciBindIPTM
from flexcraft.data.data import DesignData

class AbsciBind:
    '''
    Protocoll implementing AbsciBind from
    Levine, S. et al., 2026. Origin-1: a generative AI platform for de novo antibody design against novel epitopes. https://doi.org/10.64898/2026.01.14.699389
    for scoring multi-chain binder designs.
    '''
    def __init__(self, key, af_parameter_path:str, model:str|list = "model_2_multimer_v3", num_recycle:int=4):
        '''
        Initialize the AbsciBind class.
        Args:
            key: key for af predictions
            af_parameter_path: str, path to the af model parameters
            model: str|list, names of the models to use for the scoring, if multiple models are specified, ipTM for each model is calculated.
            num_recycle: numbers of cycles for refining the af predictions before ipTM scoring
        Returns:
            scores: dict, contains af model:ipTM pairs
        '''
        if not isinstance(model, list):
            model = [model]
        
        self.model = model
        self.key = key
        self.af_parameter_path = af_parameter_path
        self.num_recycle = num_recycle
        self.use_multimer = jnp.array(["multimer" in m for m in model]).all()
        if not self.use_multimer and jnp.array(["multimer" in m for m in model]).any():
            raise ValueError("Either use all multimer or all native models. Combination is not accepted!")
        self.af2_params = [
            get_model_haiku_params(
                model_name=model,
                data_dir=af_parameter_path, fuse=True)
            for model in self.model
        ]
        self.af2_config = model_config(self.model[0])
        self.af2m = jax.jit(make_predict(make_af2(self.af2_config, use_multimer=self.use_multimer), num_recycle=self.num_recycle))
        self.iptm = AbsciBindIPTM()
        
    def __call__(self, design:DesignData, is_target:jnp.ndarray, where:bool=True):
        '''
        Calculate ipTM scores for the design.
        Args:
            design: DesignData, the protein design to score
            is_target: bool array, True for the target protein/antigen
        '''
        af_input = (AFInput
                    .from_data(design)
                    .add_template(design,where=where))#TODO: do I need where=is_target))
        
        scores = {}
        for model_name, params in zip(self.model, self.af2_params):
            # mask diagonal to imply multimer prediction in base af
            if not self.use_multimer:
                num_chains = len(jnp.unique(design["chain_index"]))
                af_input_masked = af_input.block_diagonal(num_sequences=num_chains)
                af_result: AFResult = self.af2m(params, self.key(), af_input_masked)
            else:
                af_result: AFResult = self.af2m(params, self.key(), af_input)
            scores[model_name] =  self.iptm(
                result=af_result,
                is_target=is_target
            )
        return scores