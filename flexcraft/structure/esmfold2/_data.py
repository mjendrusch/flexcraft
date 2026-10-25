import warnings
from dataclasses import fields
from pathlib import Path

import gemmi
import numpy as np

import jax
import jax.numpy as jnp

from esm.models.esmfold2 import (
    DNAInput,
    ESMFold2InputBuilder,
    LigandInput,
    ProteinInput,
    RNAInput,
    StructurePredictionInput,
)
from esmjfold2 import Features, output_to_molecular_complex

from flexcraft.structure.common._data import InputSpec, ModelInput, _AA_SLICE

NUM_RES_TYPES = 33
_CHAIN_NAMES = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# the feature keys ESMFold2 itself consumes; we carry a few more for the language model and the writer
FEATURE_FIELDS = tuple(field.name for field in fields(Features))

class ESMFold2Spec(InputSpec):
    def __init__(self, *chains, templates=None, constraints=None):
        super().__init__(*chains, templates=templates, constraints=constraints)

    def to_features(self) -> dict:
        return to_features(self.chains, self.templates, self.constraints)

    def to_input(self, **kwargs) -> "ESMFold2Input":
        features, chain_infos = self.to_features()
        return ESMFold2Input(features=features), ESMFold2Writer(features, chain_infos)

class ESMFold2Writer:
    def __init__(self, features, chain_infos):
        self.atom_mask = np.asarray(features["atom_attention_mask"][0])
        self.ref_element = np.asarray(features["ref_element"][0])
        self.ref_atom_name_chars = np.asarray(features["ref_atom_name_chars"][0])
        self.num_tokens = features["residue_index"].shape[1]
        self.chain_infos = chain_infos

    def to_complex(self, coords, plddt=None, complex_id="model"):
        if plddt is None:
            plddt = np.zeros((self.num_tokens,), dtype=np.float32)
        return output_to_molecular_complex(
            coords=coords,
            plddt=plddt,
            atom_mask=self.atom_mask,
            ref_element=self.ref_element,
            ref_atom_name_chars=self.ref_atom_name_chars,
            chain_infos=self.chain_infos,
            complex_id=complex_id)

    def to_mmcif(self, sample_atom_coords, plddt=None, complex_id="model"):
        return self.to_complex(sample_atom_coords, plddt=plddt,
                               complex_id=complex_id).to_mmcif()

    def save_pdb(self, path, sample_atom_coords, plddt=None, keep_chains=None):
        # MolecularComplex only serializes to mmCIF, so round-trip through gemmi
        cif = self.to_mmcif(sample_atom_coords, plddt=plddt,
                            complex_id=Path(path).stem)
        block = gemmi.cif.read_string(cif).sole_block()
        structure = gemmi.make_structure_from_block(block)
        structure.setup_entities()
        structure.write_minimal_pdb(path)
        if keep_chains is not None:
            keep_pdb_chains(path, keep_chains)

    def save_cif(self, path, sample_atom_coords, plddt=None):
        Path(path).write_text(self.to_mmcif(
            sample_atom_coords, plddt=plddt, complex_id=Path(path).stem))

def keep_pdb_chains(path, keep_chains):
    """Drop every chain but keep_chains from a PDB file, in place.

    Filters the written text rather than the gemmi model: the chain id lives in a
    fixed-width column, so this needs no api we have not exercised.
    """
    keep = set(keep_chains)
    lines = Path(path).read_text().splitlines(keepends=True)
    Path(path).write_text("".join(
        line for line in lines
        if not line.startswith(("ATOM", "HETATM", "ANISOU", "TER"))
        or line[21:22] in keep))

def to_features(chains, templates, constraints):
    if templates:
        warnings.warn("Skipping provided templates: ESMFold2 has no template pathway.")
    if constraints:
        warnings.warn("Skipping provided constraints for ESMFold2: covalent bonds "
                      "and pockets are not wired up yet.")
    sequences = _to_esm_chains(chains)
    if not sequences:
        raise ValueError("ESMFold2 requires at least one chain.")
    features, chain_infos = ESMFold2InputBuilder().prepare_input(
        StructurePredictionInput(sequences=sequences))
    features = {name: _to_jax(value) for name, value in features.items()}
    # one-hot up front, so soft sequences can be written in place by set_aa
    features["res_type"] = jax.nn.one_hot(features["res_type"], NUM_RES_TYPES)
    if features.get("msa") is not None:
        features["msa"] = jax.nn.one_hot(features["msa"], NUM_RES_TYPES)
    return features, chain_infos

def to_model_features(features: dict) -> Features:
    """Select the keys ESMFold2 consumes and wrap them in its Features module."""
    return Features(**{name: features[name]
                       for name in FEATURE_FIELDS if name in features})

def _to_jax(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return jnp.asarray(value)

def _to_esm_chains(chains):
    """Map flexcraft chain dicts onto the esm input classes.

    Chain ids are handed out in order and only to chains we accept, so a skipped
    chain does not leave a gap that a covalent bond would later resolve against.
    """
    sequences = []
    next_id = 0
    for chain in chains:
        kind = chain.get("kind", "protein")
        if kind not in _CHAIN_KINDS:
            warnings.warn(f"Skipping invalid chain type '{kind}' for ESMFold2.")
            continue
        num_repeats = chain.get("num_repeats", 1)
        if next_id + num_repeats > len(_CHAIN_NAMES):
            raise ValueError(
                f"More than {len(_CHAIN_NAMES)} chains requested; ESMFold2 writes "
                f"single-character chain ids.")
        ids = [_CHAIN_NAMES[next_id + i] for i in range(num_repeats)]
        next_id += num_repeats
        # a list of ids is how esm expresses several copies of one entity
        sequences.append(_CHAIN_KINDS[kind](
            ids if num_repeats > 1 else ids[0], chain))
    return sequences

def _protein_input(ids, chain):
    return ProteinInput(id=ids, sequence=chain["sequence"], msa=_load_msa(chain))

def _rna_input(ids, chain):
    return RNAInput(id=ids, sequence=chain["sequence"], msa=_load_msa(chain))

def _dna_input(ids, chain):
    # DNAInput has no msa field
    return DNAInput(id=ids, sequence=chain["sequence"])

def _ligand_input(ids, chain):
    ccd = chain.get("ccd")
    if ccd is not None:
        # LigandInput.__post_init__ rejects a bare string
        return LigandInput(id=ids, ccd=[ccd] if isinstance(ccd, str) else list(ccd))
    if chain.get("smiles") is None:
        raise ValueError("A ligand chain needs either 'smiles' or 'ccd'.")
    return LigandInput(id=ids, smiles=chain["smiles"])

_CHAIN_KINDS = dict(
    protein=_protein_input,
    rna=_rna_input,
    dna=_dna_input,
    ligand=_ligand_input,
)

def _load_msa(chain):
    if not chain.get("use_msa", False):
        return None
    path = chain.get("msa_path")
    if path is None:
        warnings.warn("ESMFold2 does not run MSA search. Set chain['msa_path'] "
                      "to an a3m file to condition on an MSA.")
        return None
    from esm.utils.msa import MSA
    return MSA.from_a3m(path=path, remove_insertions=True,
                        max_sequences=chain.get("msa_depth", 1024))

class ESMFold2Input(ModelInput):
    features: dict

    # set_aa is inherited: ModelInput.set_aa dispatches to the _set_sequence
    # override below, which writes the Boltz-layout features["res_type"] and
    # skips the profile, which ESMFold2 averages out of the MSA internally.
    def _set_sequence(self, sequence, start=0, seq_slice=_AA_SLICE, seq_count=20,
                      reset_msa=True, reset_profile=True):
        result = self._set_res_type(sequence, start=start,
                                    seq_slice=seq_slice, seq_count=seq_count)
        if reset_msa and result.features.get("msa") is not None:
            result = result._set_msa(sequence, start=start,
                                     seq_slice=seq_slice, seq_count=seq_count)
        return result

    def set_template(self, *args, **kwargs):
        raise NotImplementedError(
            "ESMFold2 has no template pathway. Use Boltz or Protenix for "
            "template conditioned prediction.")
