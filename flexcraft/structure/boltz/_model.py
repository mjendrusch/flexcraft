# adapted from mosaic (TODO: include license)
import shutil
from typing import Any, Literal
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
    ).eval()

    model = joltz.from_torch(torch_model)
    _model_params, _model_static = eqx.partition(model, eqx.is_inexact_array)
    return eqx.combine(jax.device_put(_model_params), _model_static)

def chain_yaml(chain_name: str, sequence: str = None, ccd: str = None,
               smiles: str = None, kind: str = "protein", use_msa: bool = False) -> str:
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
            chain_yaml(chain_id, **c)
            for chain_id, c in zip("ABCDEFGHIJKLMNOPQRSTUVWXYZ", chains)
        ]
    )

    # TODO: templates
    # tf, template_yaml = build_template_yaml("ABCDEFGHIJKLMNOPQRSTUVWXYZ", chains)
    # if tf is not None:
    #     yaml += template_yaml

    features, writer_spec = features_from_yaml(yaml, cache=cache)
    # if tf is not None: # make sure we actually got a template
    #     assert np.sum(features["template_mask"]) > 0
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


# TODO templates
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

    def evaluator(self, *chains, num_recycle=2, num_sampling_steps=25, deterministic=False):
        features, writer_spec = make_features(chains, cache=self.cache)
        # move to jax
        features = jax.tree.map(jnp.array, features)
        evaluator = Joltz2Evaluator(
            joltz=self.model, features=features,
            num_recycle=num_recycle,
            num_sampling_steps=num_sampling_steps,
            deterministic=deterministic)
        # return evaluator
        evaluator_params, evaluator_static = eqx.partition(evaluator, eqx.is_array)
        def _evaluator(params):
            return eqx.combine(params, evaluator_static)
        return _evaluator, evaluator_params, Joltz2Writer(**writer_spec)

    def predictor_adhoc(self, num_recycle=2, num_samples=1,
                        num_sampling_steps=25, deterministic=False):
        jit_predict = eqx.filter_jit(Joltz2Evaluator(
            joltz=self.model, features=None,
            num_recycle=num_recycle,
            num_sampling_steps=num_sampling_steps,
            deterministic=deterministic)._predict)
        def _predict(key, *chains, aa=None):
            features, writer_spec = make_features(chains, cache=self.cache)
            features = pad_boltz_atom_features_for_compilation(features)
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

    def predictor(self, *context_chains, num_recycle=2, num_samples=1,
                  num_sampling_steps=25, deterministic=False):
        jit_predict = eqx.filter_jit(Joltz2Evaluator(
            joltz=self.model, features=None,
            num_recycle=num_recycle,
            num_sampling_steps=num_sampling_steps,
            deterministic=deterministic)._predict)
        def _predict(key, *chains, aa=None):
            features, writer_spec = make_features(chains + context_chains, cache=self.cache)
            features = pad_boltz_atom_features_for_compilation(features)
            if aa is not None:
                if len(aa.shape) == 1:
                    aa = jax.nn.one_hot(aa, num_classes=20)
                features = substitute_aa(features, aa)
                features = jax.tree.map(np.array, features)
            # NOTE: turn features back into ``torch.tensor``s for the writer
            # the writer needs padded features to be able to properly write
            # TODO: should we modify the writer? Likely yes?
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

# Adapted from github.com/jwohlwend/boltz/src/boltz/data/write/writer.py
# at commit cb04aec, which is licensed under the Apache 2.0 License.
class _BoltzWriter:
    def __init__(
        self,
        data_dir: str,
        output_dir: str,
        output_format: Literal["pdb", "mmcif"] = "mmcif",
    ) -> None:
        """Initialize the writer.

        Parameters
        ----------
        output_dir : str
            The directory to save the predictions.

        """
        if output_format not in ["pdb", "mmcif"]:
            msg = f"Invalid output format: {output_format}"
            raise ValueError(msg)

        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_format = output_format
        self.failed = 0
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write(
        self,
        prediction: dict[str, torch.Tensor],
        batch
    ) -> None:
        """Write the predictions to disk."""
        if prediction["exception"]:
            self.failed += 1
            return

        # Get the records
        records: list[Record] = batch["record"]

        # Get the predictions
        coords = prediction["coords"]
        coords = coords.unsqueeze(0)

        pad_masks = prediction["masks"]
        idx_to_rank = {i: i for i in range(len(records))}

        # Iterate over the records
        for record, coord, pad_mask in zip(records, coords, pad_masks):
            # Load the structure
            path = self.data_dir / f"{record.id}.npz"
            structure: StructureV2 = StructureV2.load(path)

            # Compute chain map with masked removed, to be used later
            chain_map = {}
            for i, mask in enumerate(structure.mask):
                if mask:
                    chain_map[len(chain_map)] = i

            # Remove masked chains completely
            structure = structure.remove_invalid_chains()

            for model_idx in range(coord.shape[0]):
                # Get model coord
                model_coord = coord[model_idx]
                # Unpad
                coord_unpad = model_coord[pad_mask.bool()]
                coord_unpad = coord_unpad.cpu().numpy()

                # New atom table
                atoms = structure.atoms
                atoms["coords"] = coord_unpad
                atoms["is_present"] = True
                structure: StructureV2
                coord_unpad = [(x,) for x in coord_unpad]
                coord_unpad = np.array(coord_unpad, dtype=Coords)

                # Mew residue table
                residues = structure.residues
                residues["is_present"] = True

                # Update the structure
                interfaces = np.array([], dtype=Interface)
                new_structure: StructureV2 = replace(
                    structure,
                    atoms=atoms,
                    residues=residues,
                    interfaces=interfaces,
                    coords=coord_unpad,
                )

                # Update chain info
                chain_info = []
                for chain in new_structure.chains:
                    old_chain_idx = chain_map[chain["asym_id"]]
                    old_chain_info = record.chains[old_chain_idx]
                    new_chain_info = replace(
                        old_chain_info,
                        chain_id=int(chain["asym_id"]),
                        valid=True,
                    )
                    chain_info.append(new_chain_info)

                # Save the structure
                struct_dir = self.output_dir / record.id
                struct_dir.mkdir(exist_ok=True)

                # Get plddt's
                plddts = None
                if "plddt" in prediction:
                    plddts = prediction["plddt"][model_idx]

                # Create path name
                outname = f"{record.id}_model_{idx_to_rank[model_idx]}"

                # Save the structure
                if self.output_format == "pdb":
                    path = struct_dir / f"{outname}.pdb"
                    with path.open("w") as f:
                        f.write(
                            to_pdb(new_structure, plddts=plddts, boltz2=True)
                        )
                elif self.output_format == "mmcif":
                    path = struct_dir / f"{outname}.cif"
                    with path.open("w") as f:
                        f.write(
                            to_mmcif(new_structure, plddts=plddts, boltz2=True)
                        )
                else:
                    path = struct_dir / f"{outname}.npz"
                    np.savez_compressed(path, **asdict(new_structure))


class Joltz2Writer:
    writer: _BoltzWriter
    atom_pad_mask: torch.Tensor
    record: any
    out_dir: str
    temp_dir_handle: TemporaryDirectory

    def __init__(
        self,
        *,
        features_dict,
        target_dir: Path,
        output_dir: Path,
        temp_dir_handle: TemporaryDirectory,
    ):
        self.writer = _BoltzWriter(
            data_dir=target_dir,
            output_dir=output_dir,
            output_format="mmcif"
        )
        self.atom_pad_mask = features_dict["atom_pad_mask"].unsqueeze(0)
        self.record = features_dict["record"][0]
        self.out_dir = output_dir
        self.temp_dir_handle = temp_dir_handle

    def save_structure(self, path, sample_atom_coords, plddt=None, out_format="mmcif"):
        self.writer.output_format = out_format
        confidence = torch.ones(1)

        if len(sample_atom_coords.shape) == 3:
            coords = torch.tensor(np.array(sample_atom_coords)).unsqueeze(0)
        else:
            coords = torch.tensor(np.array(sample_atom_coords))
        pred_dict = {
            "exception": False,
            "coords": coords,
            "masks": self.atom_pad_mask,
            "confidence_score": confidence,
        }
        if plddt is not None:
            pred_dict["plddt"] = torch.tensor(np.array(plddt))
        self.writer.write(
            pred_dict,
            {"record": [self.record]}
        )

        if out_format == "mmcif":
            in_path = str((Path(self.out_dir) / self.record.id) / f"{self.record.id}_model_0.cif")
        else:
            in_path = str((Path(self.out_dir) / self.record.id) / f"{self.record.id}_model_0.pdb")
        shutil.copy(in_path, path)

    def save_pdb(self, path, sample_atom_coords, plddt=None):
        self.save_structure(path, sample_atom_coords, plddt=plddt, out_format="pdb")

    def save_cif(self, path, sample_atom_coords, plddt=None):
        self.save_structure(path, sample_atom_coords, plddt=plddt, out_format="mmcif")


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

class JoltzResult(eqx.Module):
    data: dict

    @property
    def log_distogram(self):
        logits = self.data["distogram"][0]
        return jax.nn.log_softmax(logits, axis=-1)

    @property
    def distogram(self):
        return jnp.exp(self.log_distogram)

    @property
    def distogram_bin_edges(self):
        return jnp.linspace(2.0, 22.0, 65)
    
    @property
    def residue_index(self):
        return self.data["features"]["residue_index"][0]
    
    @property
    def chain_index(self):
        return self.data["features"]["asym_id"][0]

    @chain_index.setter
    def set_chain_index(self, value):
        self.data["features"]["asym_id"] = value[None]
        return self.chain_index

    def contact_probability(self, contact_distance=10.0) -> jax.Array:
        """Compute the distogram predicted contact probability for each
        pair of amino acids.
        
        Args:
            contact_distance: Contact distance cutoff in Angstroms. Default: 10.0.
        """
        edge_mask = self.distogram_bin_edges[1:] < contact_distance
        return (edge_mask * self.distogram).sum(axis=-1)
    
    def contact_entropy(self, contact_distance=14.0) -> jax.Array:
        """Compute the distogram contact entropy for each pair of amino acids.
        This is the metric used for optimization in BoltzDesign-1, Cho et al. 2025 (10.1101/2025.04.06.647261).
        """
        edge_mask = self.distogram_bin_edges[1:] < contact_distance
        distogram_clipped = jax.nn.softmax(self.log_distogram - 1e9 * (1 - edge_mask), axis=-1)
        distogram_clipped = jnp.where(edge_mask, distogram_clipped, 0)
        return -(distogram_clipped * self.log_distogram).sum(axis=-1)

    def contact_score(self, contact_distance=14.0, min_resi_distance=10, num_contacts=25):
        entropy = self.contact_entropy(contact_distance=contact_distance)
        resi_dist = abs(self.residue_index[:, None] - self.residue_index[None, :])
        other_chain = self.chain_index[:, None] != self.chain_index[None, :]
        entropy = jnp.where((resi_dist >= min_resi_distance) + other_chain > 0, entropy, 1e6)
        contact_score = entropy.sort(axis=1)[:, :num_contacts].mean(axis=-1).mean()
        return contact_score

    def chain_contact_score(self, target_chain, source_chain=0, contact_distance=14.0,
                            min_resi_distance=10, num_contacts=25):
        entropy = self.contact_entropy(contact_distance=contact_distance)
        resi_dist = abs(self.residue_index[:, None] - self.residue_index[None, :])
        other_chain = self.chain_index[:, None] != self.chain_index[None, :]
        entropy = jnp.where((resi_dist >= min_resi_distance) + other_chain > 0, entropy, 1e6)
        selector = self.chain_index == target_chain
        entropy = jnp.where(selector[None, :], entropy, 1e6)
        binder_selector = ~selector
        if source_chain is not None:
            if target_chain == source_chain:
                binder_selector = selector
            else:
                binder_selector = binder_selector * (self.chain_index == source_chain)
        # NOTE: Ensure that only valid entries are averaged
        sorted_entropy = entropy.sort(axis=1)[:, :num_contacts]
        is_valid_entropy = sorted_entropy < 1e5
        mean_entropy = (sorted_entropy * is_valid_entropy).sum(axis=-1) / jnp.maximum(1, is_valid_entropy.sum(axis=-1))
        contact_score = (mean_entropy * binder_selector).sum()
        contact_score /= jnp.maximum(1, binder_selector.sum())
        return contact_score

    def intra_contact_score(self, chain, contact_distance=14.0,
                           min_resi_distance=10, num_contacts=25):
        return self.chain_contact_score(
            chain, chain, contact_distance=contact_distance,
            min_resi_distance=min_resi_distance, num_contacts=num_contacts)

    @property
    def atom_to_token(self):
        return self.data["features"]["atom_to_token"][0]

    @property
    def atom24_samples(self):
        sample = self.data["samples"]
        sample24, mask24 = self._transform_sampled(sample, num_atoms=24)
        return sample24, mask24

    @property
    def atom24(self):
        if self.is_single_sample:
            return self.atom24_samples
        sample24, mask24 = self.atom24_samples
        pae = self.pae.mean(axis=(-1, -2))
        best_pae = jnp.argmin(pae, axis=0)
        return sample24[0], mask24

    @property
    def atom14(self):
        atom24, mask24 = self.atom24
        return atom24[:, :14], mask24[:, :14]

    @property
    def atom4(self):
        atom24, mask24 = self.atom24
        return atom24[:, :4]

    @property
    def cb_samples(self):
        atom24, mask24 = self.atom24_samples
        return jax.vmap(get_contact_atom, (0, None), 0)(atom24, self.data["mol_type"])

    @property
    def cb(self):
        atom24, mask24 = self.atom24
        return get_contact_atom(atom24, self.data["mol_type"])

    def _transform_sampled(self, sampled_property, num_atoms=24):
        if self.is_single_sample:
            sampled_property = sampled_property[0]
        else:
            sampled_property = sampled_property[:, 0]
            sampled_property = jnp.moveaxis(sampled_property, 0, -1)
        sample24, mask24 = atom_array_to_atomX(
            sampled_property, self.atom_to_token, num_atoms=num_atoms)
        if not self.is_single_sample:
            sample24 = jnp.moveaxis(sample24, -1, 0)
        return sample24, mask24

    # TODO
    def _broadcast_sampled(self, sampled_property, num_atoms=24):
        if self.is_single_sample:
            sampled_property = sampled_property[0]
        else:
            sampled_property = sampled_property[:, 0]
            sampled_property = jnp.moveaxis(sampled_property, 0, -1)
        sample24, mask24 = broadcast_array_to_atomX(
            sampled_property, self.atom_to_token, num_atoms=num_atoms)
        if not self.is_single_sample:
            sample24 = jnp.moveaxis(sample24, -1, 0)
        return sample24, mask24

    @property
    def is_single_sample(self):
        return len(self.data["samples"].shape) == 3

    @property
    def plddt_logits(self):
        plddt_logits = self.data["confidence"].plddt_logits
        if self.is_single_sample:
            plddt_logits = plddt_logits[0]
        else:
            plddt_logits = plddt_logits[:, 0]
        return plddt_logits
        #plddt24, mask24 = self._transform_sampled(self.data["confidence"].plddt_logits, num_atoms=24)
        #return plddt24, mask24

    @property
    def plddt(self):
        plddt = self.data["confidence"].plddt
        if self.is_single_sample:
            plddt = plddt[0]
        else:
            plddt = plddt[:, 0]
        return plddt
        # plddt24, mask24 = self._transform_sampled(self.data["confidence"].plddt, num_atoms=24)
        # plddt = (plddt24 * mask24).sum(axis=-1) / jnp.maximum(1, mask24.sum(axis=-1))
        # return plddt
    
    @property
    def pae_logits(self):
        if self.is_single_sample:
            return self.data["confidence"].pae_logits[0]
        return self.data["confidence"].pae_logits[:, 0]
    
    @property
    def pae(self):
        if self.is_single_sample:
            return jnp.fill_diagonal(self.data["confidence"].pae[0], 0.0, inplace=False) / 32
        return jax.vmap(lambda x: jnp.fill_diagonal(x, 0.0, inplace=False))(self.data["confidence"].pae[:, 0]) / 32
    
    @property
    def ipae(self):
        pae = self.pae
        chain = self.chain_index
        other_chain = chain[:, None] != chain[None, :]
        return (pae * other_chain).sum() / jnp.maximum(1, other_chain.sum())

    def chain_selector(self, target_chain, source_chain=None):
        if source_chain is None:
            source_chain = target_chain
        chain = self.chain_index
        selector = (
            (chain == target_chain)[None, :]
            + (chain == source_chain)[:, None] > 0
        )
        return selector

    def chain_pae(self, target_chain, source_chain=None):
        selector = self.chain_selector(target_chain, source_chain)
        return (self.pae * selector).sum(axis=(-1, -2)) / jnp.maximum(1, selector.sum())

    @property
    def ptm(self):
        return self.ptm_matrix().mean(axis=-1).max()

    @property
    def iptm(self):
        ptm = self.ptm_matrix() # FIXME: adjust L?
        chain = self.chain_index
        other_chain = chain[:, None] != chain[None, :]
        return ((ptm * other_chain).sum(-1) / jnp.maximum(1, other_chain.sum(-1))).max()

    def ipsae(self, chain_index=None, raw_pae_threshold=15.0):
        raw_pae = self.pae * 32
        chain = chain_index
        if chain is None:
            chain = self.chain_index
        mask = chain[:, None] != chain[None, :]
        mask *= raw_pae < raw_pae_threshold
        mask_count = jnp.maximum(1, mask.astype(jnp.int32).sum(axis=-1))
        ptm = self.ptm_matrix(L = mask_count)
        ipsae = (ptm * mask / mask_count[..., None]).sum(axis=-1).max()
        return ipsae

    @property
    def ptm_score(self):
        pae_logits = self.pae_logits
        num_aa = pae_logits.shape[0]
        if not self.is_single_sample:
            num_aa = pae_logits.shape[1]
        num_aa = max(num_aa, 19)

        d0 = 1.24 * (num_aa - 15) ** (1.0 / 3) - 1.8
        bin_centers = (jnp.arange(64) / 64 + 1 / 128) * 32

        scale = 1.0 / (1 + bin_centers ** 2 / d0 ** 2)
        score = jax.nn.logsumexp(a=pae_logits, b=scale, axis=-1)
        if not self.is_single_sample:
            score = score.mean(axis=0)
        chain = self.chain_index
        other_chain = chain[:, None] != chain[None, :]
        score = (score * other_chain).sum() / jnp.maximum(1, other_chain.sum())
        return score

    def chain_ptm(self, target_chain, source_chain=None):
        ptm = self.ptm_matrix()
        selector = self.chain_selector(target_chain, source_chain)
        mean_ptm = (ptm * selector).sum(axis=-1) / jnp.maximum(1, selector.sum(axis=-1))
        max_ptm = mean_ptm.max()
        return max_ptm

    def ptm_matrix(self, L=None):
        logits = self.pae_logits
        probabilities = jax.nn.softmax(logits, axis=-1)
        # if not self.is_single_sample:
        #     probabilities = probabilities.mean(axis=0)
        bin_centers = jnp.arange(probabilities.shape[-1]) / probabilities.shape[-1]
        bin_centers += 1 / probabilities.shape[-1] / 2
        bin_centers *= 32
        if L is None:
            L = probabilities.shape[0]
        d0 = 1.24 * (jnp.maximum(27, L) - 15) ** (1 / 3) - 1.8
        d0 = jnp.where(L < 27, 1, d0)
        # if L was an array, broadcast it
        if isinstance(L, jax.Array) and len(L.shape) >= 1:
            d0 = d0[..., None, None]
        return (probabilities * (1 / (1 + (bin_centers / d0) ** 2))).sum(axis=-1)

    @property
    def restype(self):
        # get one-hot residue type
        res_type_one_hot = self.data["features"]["res_type"][0]
        res_type = jnp.argmax(res_type_one_hot, axis=-1)
        # flip order, such that 20 amino acids are first
        # and padding / gap tokens are the last two tokens
        num_types = res_type_one_hot.shape[-1]
        res_type = res_type - 2
        res_type = jnp.where(res_type < 0, res_type + num_types, res_type)
        return res_type

    def to_data(self, return_samples=False) -> DesignData:
        """Convert an AFResult to DesignData."""
        if return_samples:
            atom24, mask24 = self.atom24_samples
        else:
            atom24, mask24 = self.atom24
        return DesignData(data=dict(
            atom_positions=atom24,
            atom_mask=mask24,
            aa=self.restype,
            mask=mask24.any(axis=1),
            residue_index=self.residue_index,
            chain_index=self.chain_index,
            batch_index=jnp.zeros_like(self.residue_index),
            plddt=self.plddt.mean(axis=0) if len(self.plddt.shape) == 2 else self.plddt,
        ))

@dataclass
class JoltzPrediction:
    data: Any
    writer: Joltz2Writer
    @property
    def result(self):
        return JoltzResult(data=self.data)

    def save_pdb(self, path, sample_index=0):
        is_multisample = len(self.data["samples"].shape) == 4
        if is_multisample:
            self.writer.save_pdb(path, self.data["samples"][sample_index],
                                 plddt=self.data["confidence"].plddt[sample_index])
        else:
            self.writer.save_pdb(path, self.data["samples"],
                                 plddt=self.data["confidence"].plddt)

    def save_cif(self, path, sample_index=0):
        self.writer.save_cif(path, self.data["samples"][sample_index][None],
                             plddt=self.data["confidence"].plddt[sample_index][None])

class Joltz2Evaluator(eqx.Module):
    joltz: Any
    features: Any
    num_recycle: int = 3
    num_sampling_steps: int = 25
    deterministic: bool = False

    def substitute_aa(self, sequence):
        self.features = substitute_aa(
            self.features, aa_one_hot=sequence)
        return self

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

    def structure_module(self, key, features, embedding, trunk_state):
        q, c, to_keys, atom_enc_bias, atom_dec_bias, token_trans_bias = (
            self.joltz.diffusion_conditioning(
                trunk_state.s,
                trunk_state.z,
                embedding.relative_position_encoding,
                features,
            )
        )
        with jax.default_matmul_precision("float32"):
            return jax.lax.stop_gradient(self.joltz.structure_module.sample(
                s_trunk=trunk_state.s,
                s_inputs=embedding.s_inputs,
                feats=features,
                num_sampling_steps=self.num_sampling_steps,
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

    def prepare(self, key, sequence):
        if sequence is not None:
            features = substitute_aa(self.features, sequence)
        return self._prepare(key, features)

    def _predict(self, key, features, num_samples=1):
        key, subkey = jax.random.split(key)
        features, embedding, trunk_state, distogram = self._prepare(subkey, features)
        def body(state, k):
            sample = self.structure_module(k, features, embedding, trunk_state)
            confidence = self.confidence_module(k, features, embedding, trunk_state, distogram, sample)
            return state, (sample, confidence)
        if num_samples > 1:
            _, (samples, confidence) = jax.lax.scan(body, None, jax.random.split(key, num_samples))
        else:
            _, (samples, confidence) = body(None, key)
        return JoltzResult(dict(
            features=features, embedding=embedding, state=trunk_state,
            distogram=distogram, samples=samples, confidence=confidence))

    def predict(self, key, sequence, num_samples=1):
        if sequence is not None:
            features = substitute_aa(self.features, sequence)
        return self._predict(key, features, num_samples=num_samples)

    def score(self, key, sequence: jax.Array = None, positions: jax.Array = None,
              data: DesignData = None):
        k1, k2 = jax.random.split(key, 2)
        features, embedding, trunk_state, distogram = self.prepare(k1, sequence)
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

