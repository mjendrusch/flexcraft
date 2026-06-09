# adapted from mosaic ((c) 2025 escalante under MIT license)
import shutil
from typing import Any, Literal, List
import os
from pathlib import Path
from functools import cached_property
from tempfile import TemporaryDirectory
import joltz
from dataclasses import asdict, dataclass, replace
from boltz.model.models.boltz2 import Boltz2
from boltz.data.const import ref_atoms
from boltz.data.write.mmcif import to_mmcif
from boltz.data.write.pdb import to_pdb
import equinox as eqx
import boltz.data.const as const
import boltz.main as boltz_main
from boltz.data.types import StructureV2, Coords, Coords, Interface, Record
import equinox as eqx
import gemmi
import jax
import joltz
import numpy as np
import torch
from boltz.model.models.boltz2 import Boltz2
from boltz.data.const import ref_atoms
from jax import numpy as jnp
from jaxtyping import Array, Float, PyTree
from joltz import TrunkState

from salad.modules.utils.geometry import positions_to_ncacocb
from flexcraft.data.data import DesignData
from flexcraft.utils.rng import Keygen
import flexcraft.sequence.aa_codes as aas
from flexcraft.structure.boltz._utils import *
from flexcraft.structure.boltz._data import JoltzInput, JoltzSpec, Joltz2Writer
from flexcraft.structure.boltz._result import JoltzResult, JoltzPrediction


def load_boltz2(model="boltz2_conf.ckpt", cache=Path("./params/boltz/")):
    if isinstance(cache, str):
        cache = Path(cache)
    if not cache.exists():
        print(f"Downloading Boltz checkpoint to {cache}")
        cache.mkdir(parents=True, exist_ok=True)
        boltz_main.download_boltz2(cache)

    torch_model = Boltz2.load_from_checkpoint(
        cache / model,
        strict=True,
        map_location="cpu",
        # Note: these args ARE NOT USED during prediction, but are needed to load the model
        predict_args={
            "recycling_steps": 0,
            "sampling_steps": 25,
            "diffusion_samples": 1,
        },
        diffusion_process_args=asdict(boltz_main.Boltz2DiffusionParams()),
        # ema=False,
        msa_args=asdict(
            boltz_main.MSAModuleArgs(
                subsample_msa=True,
                num_subsampled_msa=1024,
                use_paired_feature=True,
            )
        ),
        pairformer_args=asdict(boltz_main.PairformerArgsV2()),
        weights_only=False
    ).eval()

    model = joltz.from_torch(torch_model)
    _model_params, _model_static = eqx.partition(model, eqx.is_inexact_array)
    return eqx.combine(jax.device_put(_model_params), _model_static)

def chain_yaml(chain_order: List[str], num_repeats: int = None, sequence: str = None,
               ccd: str = None, smiles: str = None, kind: str = "protein", use_msa: bool = False,
               **kwargs) -> str:
    if num_repeats is None:
        num_repeats = 1
    chain_names = [chain_order.pop(0) for i in range(num_repeats)]
    chain_name = ", ".join(chain_names)
    if ccd is not None:
        kind = "ligand"
        result = f"""  - ligand:
        id: [{chain_name}]
        ccd: {ccd}"""
        use_msa = False
    elif smiles is not None:
        kind = "ligand"
        result = f"""  - ligand:
        id: [{chain_name}]
        smiles: {smiles}"""
        use_msa = False
    else:
        result = f"""  - {kind.lower()}:
        id: [{chain_name}]
        sequence: {sequence}"""
    if not use_msa:
        result += """
        msa: empty"""

    return result

def _prefix():
    return """version: 1
sequences:"""

def make_features(raw_chains: list, cache="./params/boltz"):
    chain_order = [c for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"]
    chains = []
    for c in raw_chains:
        if isinstance(c, DesignData):
            subchains = c.split(c["chain_index"])
            for s in subchains:
                chains.append(dict(
                    sequence=s.to_sequence_string(),
                    kind="protein",
                    use_msa=False
                ))
        else:
            chains.append(c)
    yaml = "\n".join(
        [_prefix()]
        + [
            chain_yaml(chain_order, **c)
            for c in chains
        ]
    )

    # TODO: templates
    # tf, template_yaml = build_template_yaml("ABCDEFGHIJKLMNOPQRSTUVWXYZ", chains)
    # if tf is not None:
    #     yaml += template_yaml
    chain_order = [c for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"]
    has_template, template_yaml = build_chain_template_yaml(chain_order, chains)
    if has_template:
        yaml += template_yaml

    features, writer_spec = features_from_yaml(yaml, cache=cache)
    if has_template: # make sure we actually got a template
        assert np.sum(features["template_mask"]) > 0
    return features, writer_spec

def substitute_aa(features, aa_one_hot):
    features["res_type"] = jnp.array(features["res_type"]).astype(jnp.float32)
    features["msa"] = jnp.array(features["msa"]).astype(jnp.float32)
    features["profile"] = jnp.array(features["profile"]).astype(jnp.float32)
    assert len(aa_one_hot.shape) == 2
    assert aa_one_hot.shape[1] == 20
    binder_len = aa_one_hot.shape[0]

    # We only use the standard 20 amino acids, but boltz has 33 total tokens.
    # zero out non-standard AA types
    zero_padded_sequence = jnp.pad(aa_one_hot, ((0, 0), (2, 11)))
    n_msa = features["msa"].shape[1]

    # We assume there are no MSA hits for the binder sequence
    binder_profile = jnp.zeros_like(features["profile"][0, :binder_len])
    binder_profile = binder_profile.at[:binder_len].set(zero_padded_sequence) / n_msa
    binder_profile = binder_profile.at[:, 1].set((n_msa - 1) / n_msa)

    return features | {
        "res_type": features["res_type"]
        .at[0, :binder_len, :]
        .set(zero_padded_sequence),
        "msa": features["msa"].at[0, 0, :binder_len, :].set(zero_padded_sequence),
        "profile": features["profile"].at[0, :binder_len].set(binder_profile),
    }

def build_chain_template_yaml(chain_order: List[str], chains: list):
    # TODO: allow setting more fine-grained templates + PDB files
    num_templates = 0
    template_yaml = """
        
templates:"""
    for chain in chains:
        chain: dict
        num_repeats = chain.get("num_repeats", 1)
        if not "template_file" in chain:
            for i in range(num_repeats):
                chain_order.pop(0)
            continue
        if chain["template_file"] is None or not os.path.isfile(chain["template_file"]):
            for i in range(num_repeats):
                chain_order.pop(0)
            continue
        num_templates += 1
        template_yaml += f"""
  - cif: {chain['template_file']}
    chain_id: [{', '.join([chain_order.pop(0) for i in range(num_repeats)])}]
"""
    return num_templates, template_yaml


# def build_template_yaml(chain_names: str, chains: list):
#     # boltz wants perfect .cifs :( 
#     templates = {
#         chain_id: c.template_chain
#         for chain_id, c in zip(chain_names, chains)
#         if c.template_chain != None
#     }
#     if len(templates) > 0:
#         st = gemmi.Structure()
#         model = gemmi.Model("0")
#         entities = []

#         for chain_id, chain in templates.items():
#             chain.name = chain_id
#             ent = gemmi.Entity(chain_id)
#             ent.entity_type = gemmi.EntityType.Polymer
#             ent.polymer_type = gemmi.PolymerType.PeptideL
#             ent.subchains = [chain_id]
#             ent.full_sequence = [r.name for r in chain]
#             entities.append(ent)
#             for r in chain:
#                 r.subchain = chain_id
#             model.add_chain(chain)

#         st.add_model(model)
#         st.entities = gemmi.EntityList(entities)
#         st.assign_subchains()
#         st.setup_entities()
#         st.ensure_entities()
#         st.assign_label_seq_id()

#         tf = NamedTemporaryFile(suffix=".cif")

#         template_yaml = f"""
        
# templates:
#   - cif: {tf.name}
#     chain_id: [{', '.join(k for k in templates)}]
#     template_id: [{', '.join(k for k in templates)}]
# """
        
#         st.setup_entities()
#         doc = st.make_mmcif_document()
#         doc.write_file(tf.name)
#         return tf, template_yaml
#     else:
#         return None, None

# def binder_features(binder_length, chains: list):
#     return make_features([dict(sequence = "X" * binder_length, kind = "protein", use_msa=False)] + chains)

def features_from_yaml(
    input_yaml_str: str,
    cache=Path("./params/boltz/").expanduser(),
) -> PyTree:
    if not isinstance(cache, Path):
        cache = Path(cache).expanduser()
    out_dir_handle = (
        TemporaryDirectory()
    )  # this is sketchy -- we have to remember not to let this get garbage collected
    out_dir = Path(out_dir_handle.name)
    # dump the yaml to a file
    input_data_path = out_dir / "input.yaml"
    input_data_path.write_text(input_yaml_str)
    data = boltz_main.check_inputs(input_data_path)
    # TODO: loading of cached MSA should happen here
    # for each sequence, check if it is cached
    # if so, write csv to MSA directory and add that to dictionary
    # Process inputs
    ccd_path = cache / "ccd.pkl"
    mol_dir = cache / "mols"
    # TODO: make sure this caches MSAs
    boltz_main.process_inputs(
        data=data,
        out_dir=out_dir,
        ccd_path=ccd_path,
        mol_dir=mol_dir,
        use_msa_server=True,
        msa_server_url="https://api.colabfold.com",
        msa_pairing_strategy="greedy",
        boltz2=True,
    )

    # Load processed data
    processed_dir = out_dir / "processed"
    manifest = boltz_main.Manifest.load(processed_dir / "manifest.json")

    processed = boltz_main.BoltzProcessedInput(
        manifest=manifest,
        targets_dir=processed_dir / "structures",
        msa_dir=processed_dir / "msa",
        constraints_dir=(
            (processed_dir / "constraints")
            if (processed_dir / "constraints").exists()
            else None
        ),
        template_dir=(
            (processed_dir / "templates")
            if (processed_dir / "templates").exists()
            else None
        ),
        extra_mols_dir=(
            (processed_dir / "mols") if (processed_dir / "mols").exists() else None
        ),
    )
    # TODO: caching of MSA should happen here
    # check, if sequence is in cache. If not, try loading the MSA
    # Create data module
    data_module = boltz_main.Boltz2InferenceDataModule(
        manifest=manifest,
        target_dir=processed.targets_dir,
        msa_dir=processed.msa_dir,
        num_workers=0,
        mol_dir=mol_dir,
        constraints_dir=processed.constraints_dir,
        template_dir=processed.template_dir,
        extra_mols_dir=processed.extra_mols_dir,
        override_method=None,
    )

    # Load the features for the single example
    features_dict = list(data_module.predict_dataloader())[0]

    # convert features to numpy arrays
    features = {k: np.array(v) for k, v in features_dict.items() if k != "record"}

    ## one-hot the MSA
    features["msa"] = jax.nn.one_hot(features["msa"], const.num_tokens)
    # fix up some dtypes
    # features["method_feature"] = features["method_feature"].astype(np.int32)
    writer_spec = dict(
        features_dict=features_dict,
        target_dir=processed.targets_dir,
        output_dir=out_dir / "output",
        temp_dir_handle=out_dir_handle,
    )

    return features, writer_spec

class Joltz2:
    def __init__(self, model="boltz2_conf.ckpt", cache="./params/boltz/"):
        self.cache = cache
        self.model = load_boltz2(model=model, cache=cache)
        self.model_jit = jax.jit(self.model)

    def evaluator(self, num_recycle=4, num_sampling_steps=25, deterministic=False):
        evaluator = Joltz2Evaluator(
            joltz=self.model,
            num_recycle=num_recycle,
            num_sampling_steps=num_sampling_steps,
            deterministic=deterministic)
        # return evaluator
        evaluator_params, evaluator_static = eqx.partition(evaluator, eqx.is_array)
        def _evaluator(params):
            return eqx.combine(params, evaluator_static)
        return _evaluator, evaluator_params

    # def predictor_adhoc(self, num_recycle=2, num_samples=1,
    #                     num_sampling_steps=25, deterministic=False):
    #     jit_predict = eqx.filter_jit(Joltz2Evaluator(
    #         joltz=self.model,
    #         num_recycle=num_recycle,
    #         num_sampling_steps=num_sampling_steps,
    #         deterministic=deterministic)._predict)
    #     def _predict(key, *chains, aa=None):
    #         features, writer_spec = make_features(chains, cache=self.cache)
    #         features = pad_boltz_atom_features_for_compilation(features)
    #         writer_features = {
    #             k: torch.tensor(np.array(v))
    #             for k, v in features.items() if k != "record"
    #         }
    #         writer_features["record"] = writer_spec["features_dict"]["record"]
    #         writer_spec["features_dict"] = writer_features
    #         features = jax.tree.map(jnp.array, features)
    #         prediction = jit_predict(key, features, num_samples=num_samples)
    #         return JoltzPrediction(data=prediction.data,
    #                                writer=Joltz2Writer(**writer_spec))
    #     return _predict

    def predictor(self, num_recycle=2, num_samples=1,
                  num_sampling_steps=25, deterministic=False):
        jit_predict = eqx.filter_jit(Joltz2Evaluator(
            joltz=self.model,
            num_recycle=num_recycle,
            num_sampling_steps=num_sampling_steps,
            deterministic=deterministic)._predict)
        def _predict(key, joltz_spec: JoltzSpec):
            features, writer_spec = joltz_spec.to_features(pad=True, cache=self.cache)
            writer_features = {
                k: torch.tensor(np.array(v))
                for k, v in features.items() if k != "record"
            }
            writer_features["record"] = writer_spec["features_dict"]["record"]
            writer_spec["features_dict"] = writer_features
            features = jax.tree.map(jnp.array, features)
            prediction = jit_predict(key, features, num_samples=num_samples)
            return JoltzPrediction(data=prediction.data,
                                   writer=Joltz2Writer(**writer_spec))
        return _predict

class Joltz2Evaluator(eqx.Module):
    joltz: Any
    num_recycle: int = 3
    num_sampling_steps: int = 25
    deterministic: bool = False

    def initial_embedding(self, features):
        return self.joltz.embed_inputs(features)

    def trunk(self, base_key, features, embedding):
        def body_fn(carry, _):
            trunk_state, key = carry
            trunk_state = jax.tree.map(jax.lax.stop_gradient, trunk_state)
            trunk_state, key = self.joltz.trunk_iteration(
                trunk_state,
                embedding,
                features,
                key=key,
                deterministic=self.deterministic,
            )
            return (trunk_state, key), None

        state = TrunkState(
            s=jnp.zeros_like(embedding.s_init),
            z=jnp.zeros_like(embedding.z_init),
        )

        (final_state, _), _ = jax.lax.scan(
            body_fn,
            (state, base_key),
            None,
            length=self.num_recycle,
        )
        return final_state

    def structure_module(self, key, features, embedding, trunk_state,
                         positions=None, start_steps=10,
                         edit_fn=None, edit_settings=None):
        q, c, to_keys, atom_enc_bias, atom_dec_bias, token_trans_bias = (
            self.joltz.diffusion_conditioning(
                trunk_state.s,
                trunk_state.z,
                embedding.relative_position_encoding,
                features,
            )
        )
        with jax.default_matmul_precision("float32"):
            # TODO: implement partial diffusion
            # last_n=last_n,
            # atom_coords=positions,
            return jax.lax.stop_gradient(self._partial_sample(
                atom_coords=positions,
                s_trunk=trunk_state.s,
                s_inputs=embedding.s_inputs,
                feats=features,
                num_sampling_steps=self.num_sampling_steps,
                start_steps=start_steps,
                edit_fn=edit_fn,
                edit_settings=edit_settings,
                atom_mask=features["atom_pad_mask"],
                multiplicity=1,
                diffusion_conditioning={
                    "q": q,
                    "c": c,
                    "to_keys": to_keys,
                    "atom_enc_bias": atom_enc_bias,
                    "atom_dec_bias": atom_dec_bias,
                    "token_trans_bias": token_trans_bias,
                },
                key=jax.random.fold_in(key, 2),
            ))

    def _partial_sample(
        self,
        atom_coords,
        atom_mask,
        num_sampling_steps,
        *,
        key,
        start_steps = 0,
        edit_fn = None,
        edit_settings = None,
        feats = None,
        **network_condition_kwargs, 
    ):
        # set up structure module
        sm = self.joltz.structure_module
        shape = (*atom_mask.shape, 3)

        # get the schedule, which is returned as (sigma, gamma) tuple, and pair up with the next sigma and gamma
        sigmas = sm.sample_schedule(num_sampling_steps)
        gammas = jnp.where(sigmas > sm.gamma_min, sm.gamma_0, 0.0)

        step_scale = sm.step_scale

        @jax.checkpoint
        def sample_body_function(carry, input):
            (sigma_tm, sigma_t, gamma) = input
            atom_coords, key = carry
            random_R, random_tr = joltz.compute_random_augmentation(
                key = key
            )
            key = jax.random.fold_in(key, 1)
            atom_coords = atom_coords - atom_coords.mean(axis=-2, keepdims=True)
            atom_coords = (
                jnp.einsum("bmd,bds->bms", atom_coords, random_R) + random_tr
            )

            t_hat = sigma_tm * (1 + gamma)
            noise_var = sm.noise_scale**2 * (t_hat**2 - sigma_tm**2)
            eps = jnp.sqrt(noise_var) * jax.random.normal(shape = shape, key = key)
            key = jax.random.fold_in(key, 1)
            atom_coords_noisy = atom_coords + eps
            atom_coords_denoised = sm.preconditioned_network_forward(
                    atom_coords_noisy,
                    t_hat,
                    network_condition_kwargs=dict(
                        feats=feats,
                        **network_condition_kwargs,
                    ),
                    key = key
                )
            # enable custom structure editing / potentials
            if edit_fn is not None:
                atom_coords_denoised = edit_fn(
                    key, JoltzInput(features=feats),
                    atom_coords_denoised,
                    edit_settings=edit_settings)

            if sm.alignment_reverse_diff:
                atom_coords_noisy = joltz.weighted_rigid_align(
                    atom_coords_noisy,
                    atom_coords_denoised,
                    atom_mask,
                    atom_mask,
                )


            denoised_over_sigma = (atom_coords_noisy - atom_coords_denoised) / t_hat
            atom_coords_next = (
                atom_coords_noisy + step_scale * (sigma_t - t_hat) * denoised_over_sigma
            )

            return (atom_coords_next, jax.random.fold_in(key, 0)), None

        # initialize noise with normal distribution
        start_noise = jax.random.normal(shape = shape, key = key)
        start_coords = sigmas[0] * start_noise
        # initialize noise with input coordinates when using partial diffusion
        if atom_coords is not None:
            start_coords = atom_coords + sigmas[start_steps + 1] * start_noise
        (atom_coords, _), _ = jax.lax.scan(
            sample_body_function,
            (start_coords, jax.random.fold_in(key, 1)),
            (sigmas[:-1][start_steps:], sigmas[1:][start_steps:], gammas[1:][start_steps:])
        )

        return atom_coords

    def confidence_module(self, key, features, embedding,
                          trunk_state, distogram, positions):
        return self.joltz.confidence_module(
            s_inputs=embedding.s_inputs,
            s=trunk_state.s,
            z=trunk_state.z,
            x_pred=positions,
            feats=features,
            pred_distogram_logits=distogram[None],
            key=jax.random.fold_in(key, 5),
            deterministic=self.deterministic,
        )

    def _prepare(self, key, features):
        embedding = self.initial_embedding(features)
        trunk_state = self.trunk(key, features, embedding)
        distogram = self.joltz.distogram_module(trunk_state.z)[0, :, :, 0, :]
        return features, embedding, trunk_state, distogram

    def prepare(self, key, joltz_input: JoltzInput):
        return self._prepare(key, joltz_input.features)

    def _predict(self, key, features, positions=None, start_steps=0, num_samples=1):
        key, subkey = jax.random.split(key)
        features, embedding, trunk_state, distogram = self._prepare(subkey, features)
        def body(state, k):
            sample = self.structure_module(k, features, embedding, trunk_state,
                                           positions=positions, start_steps=start_steps)
            confidence = self.confidence_module(k, features, embedding, trunk_state, distogram, sample)
            return state, (sample, confidence)
        if num_samples > 1:
            _, (samples, confidence) = jax.lax.scan(body, None, jax.random.split(key, num_samples))
        else:
            _, (samples, confidence) = body(None, key)
        return JoltzResult(dict(
            features=features, embedding=embedding, state=trunk_state,
            distogram=distogram, samples=samples, confidence=confidence))

    def predict(self, key, joltz_input: JoltzInput, num_samples=1):
        features = joltz_input.features
        return self._predict(key, features, num_samples=num_samples)

    def score(self, key, joltz_input: JoltzInput, positions: jax.Array = None,
              data: DesignData = None):
        k1, k2 = jax.random.split(key, 2)
        features, embedding, trunk_state, distogram = self.prepare(k1, joltz_input)
        atom_to_token = features["atom_to_token"]
        if data is not None:
            sequence = data.aa
            positions = data["atom_positions"]
        if sequence.dtype == jnp.int32:
            sequence = jax.nn.one_hot(sequence, 20)
        if len(positions.shape) == 3:
            positions = atomX_to_atom_array(positions, atom_to_token[0])
            positions = positions[None]
        elif len(positions.shape) == 4:
            positions = atomX_to_atom_array(
                jnp.moveaxis(positions, 0, -1), atom_to_token[0])
            positions = jnp.moveaxis(positions, -1, 0)
            positions = positions[:, None]
        if positions.shape[-2] < atom_to_token.shape[0]:
            raise ValueError(f"Positions need to cover all atoms.")
        confidence = self.confidence_module(
            k2, features, embedding, trunk_state, distogram, positions)
        return JoltzResult(dict(
            features=features, embedding=embedding, state=trunk_state,
            distogram=distogram, samples=positions, confidence=confidence
        ))

if __name__ == "__main__":
    model = Joltz2()
    key = Keygen(42)
    joltz, params, writer = model.evaluator(dict(kind="protein", sequence="X" * 100), dict(ccd="TOP"), dict(kind="dna", sequence="GGCGCAATAAGCGCC"))
    joltz_pred = model.predictor(dict(ccd="TOP"), dict(kind="dna", sequence="GGCGCAATAAGCGCC"))
    x = jax.random.gumbel(key(), (100, 20), dtype=jnp.float32)
    x = jnp.array(x)
    x = jax.nn.log_softmax(x, axis=-1)

    pred_1 = joltz_pred(key(), dict(kind="protein", sequence="MIVKQRKINLKPATATITDPLEVNFAEALVESVKNNAPVKVNGMTVYGKGGNFEITRNGPNQLTVKAWGDIEIKIEAKLQPGYDAAQGFLDRIEAGSNRQ"))
    pred_1.save_pdb("test_pdb_1.pdb")
    pred_2 = joltz_pred(key(), dict(kind="protein", sequence="MIVKQRKIAAAPATATITDPLEVNFAEALVESVKNNAPVAAAAMTVYGKGGNFEITRNGPNQLTVKAWGDIEIKIEAKLQPGYDAAQGFLDRIEAGSNRQ"))
    pred_2.save_pdb("test_pdb_2.pdb")
    
    models = dict(joltz=joltz)
    def loss(sequence, scale, key=None, params=None):
        sequence = jax.nn.softmax(sequence * scale, axis=-1)
        result: JoltzResult = joltz(params).predict(key, sequence, num_samples=4)
        plddt_loss = (1 - result.plddt.mean())
        loss = result.contact_score().mean() + 0.25 * plddt_loss
        return loss, result
    loss_update = jax.jit(jax.value_and_grad(loss, argnums=0, has_aux=True))
    for i in range(100):
        print(f"starting {i}")
        scale = 1.0
        (loss, result), grad = loss_update(x, jnp.array(scale), key=key(), params=params)
        print(i, loss)
        grad /= jnp.linalg.norm(grad)
        x -= 0.5 * grad
        if i % 10 == 0:
            pred: JoltzPrediction = joltz_pred(
                key(), dict(kind="protein",
                            sequence=aas.decode(jnp.argmax(x, axis=1), aas.AF2_CODE)),
                aa=jax.nn.softmax(x, axis=-1))
            pred.save_cif(f"result_{i}.cif")
        print(f"done {i}")
    np.savez_compressed(f"joltz_prediction_{i}.npz", samples=result.data["samples"], dist=result.data["distogram"],
                        plddt_logits=result.data["confidence"].plddt_logits, pae_logits=result.data["confidence"].pae_logits,
                        pde_logits=result.data["confidence"].pde_logits,
                        **result.data["features"])
    print(result.plddt.mean())

