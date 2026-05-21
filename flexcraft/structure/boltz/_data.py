# adapted from mosaic ((c) 2025 escalante under MIT license)
import shutil
from typing import Any, Literal, List
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from dataclasses import asdict, replace
from boltz.data.write.mmcif import to_mmcif
from boltz.data.write.pdb import to_pdb
import equinox as eqx
import boltz.data.const as const
import boltz.main as boltz_main
from boltz.data.types import StructureV2, Coords, Coords, Interface, Record
import equinox as eqx
import jax
import joltz
import numpy as np
import torch
from jax import numpy as jnp
from jaxtyping import Array, Float, PyTree
from flexcraft.data.data import DesignData
from flexcraft.files.pdb import PDBFile
from flexcraft.structure.boltz._utils import *


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

def _make_features(chains: list, templates: list, constraints: list,
                   cache="./params/boltz/"):
    chain_order = [c for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"]
    yaml = "\n".join(
        [_prefix()]
        + [
            chain_yaml(chain_order, **c)
            for c in chains
        ]
    )
    chain_order = [c for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"]
    has_template, template_yaml = build_template_yaml(templates)
    if has_template:
        yaml += template_yaml
    has_constraint, constraint_yaml = build_constraint_yaml(constraints)
    if has_constraint:
        yaml += constraint_yaml
    features, writer_spec = _features_from_yaml(yaml, cache=cache)
    if has_template:
        assert np.sum(features["template_mask"]) > 0
    return features, writer_spec

class JoltzSpec:
    def __init__(self, *chains, templates=None, constraints=None):
        self.chains = list(chains)
        self.templates = templates or list()
        self.constraints = constraints or list()
        self.temporaries = list()

    def add_chain(self, *chains):
        _chains = []
        for c in chains:
            if isinstance(c, DesignData):
                subchains = c.split(c["chain_index"])
                for s in subchains:
                    _chains.append(dict(
                        sequence=s.to_sequence_string(),
                        kind="protein",
                        use_msa=False
                    ))
            else:
                _chains.append(c)
        self.chains += _chains
        return self

    def add_polymer(self, *sequences, kind="protein", use_msa=False, num_repeats=1):
        _chains = []
        for c in sequences:
            subchains = c.split(":")
            for s in subchains:
                _chains.append(dict(
                    sequence=s,
                    kind=kind,
                    use_msa=use_msa,
                    num_repeats=num_repeats
                ))
        return self.add_chain(*_chains)

    def add_protein(self, *sequences, use_msa=False, num_repeats=1):
        return self.add_polymer(*sequences, kind="protein", use_msa=use_msa, num_repeats=num_repeats)

    def add_dna(self, *sequences, use_msa=False, num_repeats=1):
        return self.add_polymer(*sequences, kind="dna", use_msa=use_msa, num_repeats=num_repeats)

    def add_rna(self, *sequences, use_msa=False, num_repeats=1):
        return self.add_polymer(*sequences, kind="rna", use_msa=use_msa, num_repeats=num_repeats)

    def add_smiles(self, smiles, num_repeats=1):
        _chains = [
            dict(kind="ligand", smiles=smiles, num_repeats=num_repeats)
        ]
        return self.add_chain(*_chains)

    def add_ccd(self, ccd, num_repeats=1):
        _chains = [
            dict(kind="ligand", ccd=ccd, num_repeats=num_repeats)
        ]
        return self.add_chain(*_chains)

    def add_template(self, path_or_object, to_chains=None):
        template_info = dict(template=path_or_object, template_chains=to_chains)
        if isinstance(path_or_object, str):
            if path_or_object.endswith((".pdb", ".pdb1")):
                template_info["pdb"] = path_or_object
            elif path_or_object.endswith(".cif"):
                template_info["cif"] = path_or_object
            else:
                raise NotImplementedError(f"Invalid template file '{path_or_object}'. "
                                          f"Has to be either '.pdb' or '.cif'.")
        elif isinstance(path_or_object, DesignData):
            tmpfile = PDBFile(data = path_or_object)
            self.temporaries.append(tmpfile)
            template_info["pdb"] = tmpfile.path
        else:
            raise NotImplementedError(
                "Input template has to be a path to a '.pdb' or '.cif' file.")
        self.templates.append(template_info)
        return self

    def add_msa(self, to_chains=None):
        if to_chains is None:
            to_chains = list(range(len(self.chains)))
        for c in to_chains:
            chain = self.chains[c]
            if not all(x == "X" for x in chain["sequence"]):
                self.chains[c]["use_msa"] = True
        return self

    def add_bond(self, chain_1, residue_1, atom_1,
                 chain_2, residue_2, atom_2):
        self.constraints.append(dict(
            kind="bond",
            atom_1=dict(chain=chain_1, residue=residue_1, atom=atom_1),
            atom_2=dict(chain=chain_2, residue=residue_2, atom=atom_2)))
        return self

    def add_contact(self, chain_1, residue_or_atom_1,
                    chain_2, residue_or_atom_2, max_distance=6.0):
        self.constraints.append(dict(
            kind="contact",
            token_1=dict(chain=chain_1, target=residue_or_atom_1),
            token_2=dict(chain=chain_2, target=residue_or_atom_2),
            max_distance=max_distance
        ))
        return self

    def add_constraint(self, constraint):
        self.constraints.append(constraint)
        return self

    def to_features(self, pad=True, cache="./params/boltz/") -> dict:
        features, writer_spec = _make_features(
            self.chains, self.templates, self.constraints, cache=cache)
        if pad:
            features = pad_boltz_atom_features_for_compilation(features)
        writer_features = {
            k: torch.tensor(np.array(v))
            for k, v in features.items() if k != "record"
        }
        writer_features["record"] = writer_spec["features_dict"]["record"]
        writer_spec["features_dict"] = writer_features
        features = jax.tree.map(jnp.array, features)
        return features, writer_spec

    def to_input(self, pad=True, cache="./params/boltz/") -> "JoltzInput":
        features, writer_spec = self.to_features(pad=pad, cache=cache)
        return JoltzInput(features=features), Joltz2Writer(**writer_spec)

_AA_SLICE = slice(2, 22)
_AA_UNK = 22
_RNA_SLICE = slice(23, 27)
_RNA_UNK = 27
_DNA_SLICE = slice(28, 32)
_DNA_UNK = 32

class JoltzInput(eqx.Module):
    features: dict
    @property
    def residue_index(self):
        return self.features["residue_index"][0]
    @property
    def chain_index(self):
        return self.features["asym_id"][0]
    @property
    def residue_type(self):
        return jnp.argmax(self.features["res_type"][0], axis=-1)

    def _set_res_type(self, sequence, start=0, seq_slice=_AA_SLICE, seq_count=20):
        self.features["res_type"] = jnp.array(self.features["res_type"]).astype(jnp.float32)
        self.features["res_type"] = self.features["res_type"].at[0, start:start + sequence.shape[0]].set(0.0)
        self.features["res_type"] = self.features["res_type"].at[0, start:start + sequence.shape[0], seq_slice].set(sequence[:, :seq_count])
        return self

    def _set_profile(self, sequence, start=0, seq_slice=_AA_SLICE, seq_count=20):
        num_msa = self.features["msa"].shape[1]
        self.features["profile"] = jnp.array(self.features["profile"]).astype(jnp.float32)
        self.features["profile"] = self.features["profile"].at[0, start:start + sequence.shape[0]].set(0.0)
        self.features["profile"] = self.features["profile"].at[0, start:start + sequence.shape[0], seq_slice].set(sequence[:, :seq_count] / num_msa)
        self.features["profile"] = self.features["profile"].at[0, start:start + sequence.shape[0], 1].set((num_msa - 1) / num_msa)
        return self

    def _set_msa(self, sequence, start=0, seq_slice=_AA_SLICE, seq_count=20):
        self.features["msa"] = jnp.array(self.features["msa"]).astype(jnp.float32)
        # reset MSA for all positions we're setting:
        # setting all msa positions to zero
        self.features["msa"] = self.features["msa"].at[0, :, start:start + sequence.shape[0]].set(0.0)
        # setting all msa positions from the 2nd sequence onwards to "-"
        self.features["msa"] = self.features["msa"].at[0, 1:, start:start + sequence.shape[0], 1].set(1.0)
        # finally, setting the first sequence to the input sequence
        self.features["msa"] = self.features["msa"].at[0, 0, start:start + sequence.shape[0], seq_slice].set(sequence[:, :seq_count])
        return self

    def _set_sequence(self, sequence, start=0, seq_slice=_AA_SLICE, seq_count=20):
        self = self._set_res_type(sequence, start=start, seq_slice=seq_slice, seq_count=seq_count)
        self = self._set_profile(sequence, start=start, seq_slice=seq_slice, seq_count=seq_count)
        self = self._set_msa(sequence, start=start, seq_slice=seq_slice, seq_count=seq_count)
        return self

    def set_aa(self, sequence, start=0):
        return self._set_sequence(sequence, start=start)

    def inherit_msa(self, data: "JoltzInput"):
        self.features["msa"] = data.features["msa"]
        self.features["msa_mask"] = data.features["msa_mask"]
        self.features["profile"] = data.features["profile"]
        self.features["has_deletion"] = data.features["has_deletion"]
        self.features["deletion_value"] = data.features["deletion_value"]
        self.features["msa_paired"] = data.features["msa_paired"]
        return self

    def set_msa(self, msa, start=0):
        self.features["msa"] = jnp.array(self.features["msa"]).astype(jnp.float32)
        # reset MSA for all positions we're setting:
        # setting all msa positions to zero
        self.features["msa"] = self.features["msa"].at[0, :, start:start + msa.shape[1]].set(0.0)
        # setting all msa positions from the 2nd sequence onwards to "-"
        self.features["msa"] = self.features["msa"].at[0, 1:, start:start + msa.shape[1], 1].set(1.0)
        # finally, setting the first sequence to the input sequence
        self.features["msa"] = self.features["msa"].at[0, 0:msa.shape[0], start:start + msa.shape[1], :].set(msa)
        # compute & set the profile
        self.features["profile"] = self.features["msa"].at[0, start:start + msa.shape[1]].set(msa.mean(axis=0))
        return self

    def set_rna(self, sequence, start=0):
        return self._set_sequence(sequence, start=start, seq_slice=_RNA_SLICE, seq_count=4)
    
    def set_dna(self, sequence, start=0):
        return self._set_sequence(sequence, start=start, seq_slice=_DNA_SLICE, seq_count=4)

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

def build_template_yaml(templates: list):
    num_templates = len(templates)
    template_yaml = """
        
templates:"""
    for template in templates:
        kind = "cif" if "cif" in template else "pdb"
        template_yaml += f"""
  - {kind}: {template[kind]}"""
        if "chains" in template:
            template_yaml += f"""
    chain_id: [{', '.join(template['chains'])}]"""
        if "template_chains" in template:
            template_yaml += f"""
    template_chain_id: [{', '.join(template['template_chains'])}]"""
        template_yaml += "\n"
    return num_templates, template_yaml

def build_constraint_yaml(constraints: list):
    num_constraints = len(constraints)
    constraint_yaml = """

constraints:"""
    for constraint in constraints:
        if constraint["kind"] == "bond":
            constraint_yaml += f"""
    - bond:
        atom1: [{constraint['atom_1']['chain']}, {constraint['atom_1']['residue']}, {constraint['atom_1']['atom']}]
        atom2: [{constraint['atom_2']['chain']}, {constraint['atom_2']['residue']}, {constraint['atom_2']['atom']}]"""
        elif constraint["kind"] == "contact":
            constraint_yaml += f"""
    - contact:
        token1: [{constraint['token_1']['chain']}, {constraint['token_1']['target']}]
        token2: [{constraint['token_2']['chain']}, {constraint['token_2']['target']}]
        max_distance: {constraint['max_distance']}"""
        else:
            raise NotImplementedError(f"Unknown constraint type {constraint['kind']}.")
    constraint_yaml += "\n"
    return num_constraints, constraint_yaml

def _features_from_yaml(
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
