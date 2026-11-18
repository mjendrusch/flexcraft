import os
import warnings
from copy import copy, deepcopy
from typing import Dict, Sequence

import gemmi
import numpy as np
import jax
import jax.numpy as jnp

import tempfile
import shutil
from collections import defaultdict
from protenix.data.json_to_feature import SampleDictToFeatures
from protenix.data.msa_featurizer import InferenceMSAFeaturizer
from protenix.data.utils import data_type_transform

from flexcraft.structure.protenix.msa.colab_request_parser import RequestParser
from protenix.openfold_local.np.residue_constants import (
    unk_restype_index, atom_order, restype_order,
)

from flexcraft.structure.common._data import InputSpec, ModelInput
import flexcraft.sequence.aa_codes as aas
from salad.modules.utils.geometry import positions_to_ncacocb


def _apply_pad_spec(inp, pad_spec):
    """Apply a _pad_spec (dict, tuple, or list) to a model input."""
    if pad_spec is None:
        return inp
    if isinstance(pad_spec, dict):
        method = pad_spec.get("method") or pad_spec.get("mode") or pad_spec.get("kind")
        if method == "pad_atoms":
            return inp.pad_atoms(pad_spec.get("num_residue_atoms", 24))
        if pad_spec.get("_multiple") is not None or method == "pad_to_nearest_multiple":
            mult = pad_spec.get("_multiple", pad_spec.get("multiple"))
            return inp.pad_to_nearest_multiple(
                mult,
                pad_spec.get("num_residue_atoms", 24),
                num_msa=pad_spec.get("num_msa"),
                num_templates=pad_spec.get("num_templates"))
        return inp.pad_to(
            pad_spec.get("size"),
            pad_spec.get("num_residue_atoms", 24),
            num_msa=pad_spec.get("num_msa"),
            num_templates=pad_spec.get("num_templates"))
    if isinstance(pad_spec, tuple) and len(pad_spec) == 3:
        method_name, args, kwargs = pad_spec
        return getattr(inp, method_name)(*args, **kwargs)
    if isinstance(pad_spec, list):
        for op in pad_spec:
            inp = _apply_pad_spec(inp, op)
        return inp
    raise TypeError(f"Unsupported _pad_spec format: {pad_spec!r}")


class ProtenixSpec(InputSpec):
    def __init__(self, *chains, templates=None, constraints=None):
        super().__init__(*chains, templates=templates, constraints=constraints)
        self._pad_spec = None

    def to_features(self) -> dict:
        features, atom_array, token_array = to_features(
            self.chains, self.templates, self.constraints)
        return features, atom_array, token_array

    def to_input(self, **kwargs) -> "ProtenixInput":
        features, atom_array, token_array = self.to_features()
        inp = ProtenixInput(features=features)
        # Apply padding spec if set
        pad_spec = getattr(self, "_pad_spec", None)
        if pad_spec is not None:
            inp = _apply_pad_spec(inp, pad_spec)
        return inp, ProtenixWriter(atom_array, token_array)

class ProtenixWriter:
    def __init__(self, atom_array, token_array):
        self.atom_array = atom_array
        self.token_array = token_array

    # adapted from escalante-bio/mosaic under Apache 2.0 license, see NOTICE
    def to_gemmi(self, coords, plddt=None):
        if coords is not None:
            atom_array = deepcopy(self.atom_array)
            atom_array.coord = coords
        structure = gemmi.Structure()
        model = gemmi.Model("0")
        chains = {}
        for atom_idx, atom in enumerate(atom_array):
            chain = chains.setdefault(atom.chain_id, {})
            residue = chain.setdefault(int(atom.res_id), _new_gemmi_residue(atom))
            gemmi_atom = _biotite_atom_to_gemmi_atom(atom)
            if plddt is not None:
                gemmi_atom.b_iso = plddt[atom_idx]
            residue.add_atom(gemmi_atom)
        for k in chains:
            chain = gemmi.Chain(k)
            chain.append_residues(list(chains[k].values()))
            model.add_chain(chain)
        structure.add_model(model)
        return structure

    def result_to_data(self, result, sample_index=0):
        is_multisample = len(result.data["samples"].shape) == 4
        plddt = (jax.nn.softmax(result.data["confidence"].plddt_logits, axis=-1)
                 * (jnp.arange(50) + 0.5)).sum(axis=-1) / 50
        coords = result.data["samples"]
        if is_multisample:
            plddt = plddt[sample_index, 0]
            coords = coords[sample_index, 0]
        else:
            plddt = plddt[0]
            coords = coords[0]
        return np.array(coords), np.array(plddt)

    def save_pdb(self, path, sample_atom_coords=None, plddt=None, result=None, ):
        if result is not None:
            sample_atom_coords, plddt = self.result_to_data(result)
        self.to_gemmi(sample_atom_coords, plddt=plddt).write_minimal_pdb(str(path))

    def save_cif(self, path, sample_atom_coords=None, plddt=None, result=None):
        if result is not None:
            sample_atom_coords, plddt = self.result_to_data(result)
        structure = self.to_gemmi(sample_atom_coords, plddt=plddt)
        document = structure.make_mmcif_document()
        document.write_file(str(path))

# adapted from escalante-bio/mosaic under Apache 2.0 license, see NOTICE
def _biotite_atom_to_gemmi_atom(atom):
    ga = gemmi.Atom()
    ga.pos = gemmi.Position(*atom.coord)
    ga.element = gemmi.Element(atom.element)
    ga.name = atom.atom_name
    return ga

# adapted from escalante-bio/mosaic under Apache 2.0 license, see NOTICE
def _new_gemmi_residue(atom):
    r = gemmi.Residue()
    r.name = atom.res_name
    r.seqid = gemmi.SeqId(atom.res_id, " ")
    r.entity_type = gemmi.EntityType.Polymer
    return r

# adapted from escalante-bio/mosaic under Apache 2.0 license, see NOTICE
def to_features(chains, templates, constraints):
    temp_dirs = []

    # Collect sequences needing MSA search
    seqs_needing_search = []
    for chain in chains:
        if "msa_dir" not in chain:
            chain["msa_dir"] = None
        if "num_repeats" not in chain:
            chain["num_repeats"] = 1
        if chain.get("use_msa", False) and chain["msa_dir"] is None:
            seqs_needing_search.append(chain["sequence"])

    # Run MSA search if needed
    seq_to_msa_dir: Dict[str, str] = {}
    if seqs_needing_search:
        msa_tmpdir = tempfile.mkdtemp(prefix="protenix_msa_")
        temp_dirs.append(msa_tmpdir)
        unique_seqs = sorted(set(seqs_needing_search))
        msa_subdirs = _run_msa_search(unique_seqs, msa_tmpdir, None)
        seq_to_msa_dir = dict(zip(unique_seqs, msa_subdirs))

    # Build sequences list for sample dict
    sequences = []
    chain_names = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    chain_to_entity = dict()
    chain_to_copy = dict()
    chain_id = 0
    entity_id = 0
    for i, chain in enumerate(chains):
        chain: Dict[str, str | int]
        if "num_repeats" not in chain:
            chain["num_repeats"] = 1
        for copy_id in range(chain["num_repeats"]):
            chain_to_entity[chain_names[chain_id]] = entity_id + 1
            chain_to_copy[chain_names[chain_id]] = copy_id + 1
            chain_id += 1
        entity_id += 1
        if chain["kind"] == "protein":
            protein_chain = {"sequence": chain["sequence"], "count": chain["num_repeats"]}

            if "msa_dir" in chain and chain["msa_dir"] is not None:
                msa_dir = chain["msa_dir"]
            elif chain.get("use_msa", False):
                msa_dir = seq_to_msa_dir[chain["sequence"]]
            else:
                # Create dummy MSA
                msa_dir = tempfile.mkdtemp(prefix=f"protenix_msa_dummy_{i}_")
                temp_dirs.append(msa_dir)
                _make_dummy_msa(chain["sequence"], msa_dir)

            protein_chain["msa"] = {
                "precomputed_msa_dir": msa_dir,
                "pairing_db": "uniref100",
            }
            if "modifications" in chain:
                protein_chain["modifications"] = [
                    {
                        "ptmType": "CCD_" + mod["ccd"],
                        "ptmLocation": mod["residue"]
                    }
                    for mod in chain["modifications"]
                ]
            sequences.append({"proteinChain": protein_chain})
        elif chain["kind"] == "dna":
            dna_chain = {
                "sequence": chain["sequence"],
                "count": chain["num_repeats"],
            }
            if "modifications" in chain:
                dna_chain["modifications"] = [
                    {
                        "modificationType": "CCD_" + mod["ccd"],
                        "basePosition": mod["residue"]
                    }
                    for mod in chain["modifications"]
                ]
            sequences.append({"dnaChain": dna_chain})
        elif chain["kind"] == "rna":
            rna_chain = {
                "sequence": chain["sequence"],
                "count": chain["num_repeats"],
            }
            if "modifications" in chain:
                rna_chain["modifications"] = [
                    {
                        "modificationType": "CCD_" + mod["ccd"],
                        "basePosition": mod["residue"]
                    }
                    for mod in chain["modifications"]
                ]
            if "use_msa" in chain and chain["use_msa"] == True:
                raise NotImplementedError("RNA MSA not supported yet.")
            sequences.append({"rnaChain": rna_chain})
        elif chain["kind"] in ("smol", "ligand"):
            # Accept both "smol" and "ligand" kinds
            if "ccd" in chain:
                ligand = "CCD_" + chain["ccd"].upper()
            elif "smiles" in chain:
                ligand = chain["smiles"]
            else:
                raise ValueError("Small molecule chains should have either 'smiles' or 'ccd' set.")
            sequences.append({"ligand": {"ligand": ligand, "count": chain["num_repeats"]}})
        elif chain["kind"] == "ion":
            sequences.append({"ion": {"ion": chain["ccd"], "count": chain["num_repeats"]}})
        else:
            raise NotImplementedError(f"Unknown chain type {chain['kind']}.")

    sample = {"name": "pred", "sequences": sequences}
    if constraints:
        sample["constraint"] = dict()

    # add constraints
    contact_list = []
    bond_list = []
    for constraint in constraints:
        if constraint["kind"] == "contact":
            token_1 = constraint["token_1"]
            chain_1 = token_1["chain"]
            atom_1 = None
            target_1 = token_1["target"]

            token_2 = constraint["token_2"]
            chain_2 = token_2["chain"]
            atom_2 = None
            target_2 = token_2["target"]
            contact = dict(
                entity_1=chain_to_entity[chain_1],
                copy_1=chain_to_copy[chain_1],
                entity_2=chain_to_entity[chain_2],
                copy_2=chain_to_copy[chain_2],
                min_distance=0,
                max_distance=constraint["max_distance"]
            )
            if isinstance(target_1, tuple):
                target_1, atom_1 = target_1
                contact["position_1"] = target_1
                contact["atom_1"] = atom_1
            if isinstance(target_1, str):
                contact["atom_1"] = target_1
            if isinstance(target_2, tuple):
                target_2, atom_2 = target_2
                contact["position_2"] = target_2
                contact["atom_2"] = atom_2
            if isinstance(target_2, str):
                contact["atom_2"] = target_2

            contact_list.append(contact)
        if constraint["kind"] == "pocket":
            if "pocket" in sample:
                raise ValueError("ProtenixSpec may contain only a single pocket constraint.")
            pocket = dict()
            binder = constraint["binder"]
            binder_entity = chain_to_entity[binder]
            binder_copy = chain_to_copy[binder]
            pocket["binder_chain"] = [binder_entity, binder_copy]
            pocket["contact_residues"] = [
                (chain_to_entity[chain], chain_to_copy[chain], residue)
                for chain, residue in constraint["contacts"]
            ]
            pocket["max_distance"] = constraint["max_distance"]
            sample["constraint"]["pocket"] = pocket
        if constraint["kind"] == "bond":
            bond_list.append(dict(
                entity1=chain_to_entity[constraint["atom_1"]["chain"]],
                copy1=chain_to_copy[constraint["atom_1"]["chain"]],
                position1=constraint["atom_1"]["residue"],
                atom1=constraint["atom_1"]["atom"],
                entity2=chain_to_entity[constraint["atom_2"]["chain"]],
                copy2=chain_to_copy[constraint["atom_2"]["chain"]],
                position2=constraint["atom_2"]["residue"],
                atom2=constraint["atom_2"]["atom"],
            ))
    if contact_list:
        sample["constraint"]["contacts"] = contact_list
    if bond_list:
        sample["constraint"]["covalent_bonds"] = bond_list

    # featurize
    sample2feat = SampleDictToFeatures(sample)
    features_dict, atom_array, token_array = sample2feat.get_feature_dict()
    features_dict["distogram_rep_atom_mask"] = np.asarray(
        atom_array.distogram_rep_atom_mask, dtype=np.int64
    )

    # featurize MSA
    entity_to_asym_id: Dict[str, set] = defaultdict(set)
    for entity_id, asym_id_int in zip(
        atom_array.label_entity_id, atom_array.asym_id_int
    ):
        entity_to_asym_id[entity_id].add(asym_id_int)
    entity_to_asym_id = dict(entity_to_asym_id)

    msa_feats = InferenceMSAFeaturizer.make_msa_feature(
        bioassembly=sequences,
        entity_to_asym_id=entity_to_asym_id,
        token_array=token_array,
        atom_array=atom_array,
    )
    if msa_feats:
        for k, v in msa_feats.items():
            features_dict[k] = np.asarray(v) if isinstance(v, np.ndarray) else v

    # Apply data type transforms (still uses torch internally)
    features_dict = data_type_transform(features_dict)

    result = {}
    for k, v in features_dict.items():
        if isinstance(v, np.ndarray):
            result[k] = v
        elif isinstance(v, dict):
            result[k] = {k2: np.asarray(v2) if not isinstance(v2, np.ndarray) else v2 for k2, v2 in v.items()}
        else:
            result[k] = v

    # Add atom_rep_atom_idx (needed by JAX model)
    result["atom_rep_atom_idx"] = result["distogram_rep_atom_mask"].nonzero()[0]

    num_res = result["restype"].shape[0]

    # Add templates
    if templates:
        aatype = []
        atom_positions = []
        atom_mask = []
        for template in templates:
            if "template_chains" not in template:
                raise NotImplementedError("Templates require a selector.")
            _aatype, _atom_positions, _atom_mask = _load_template(chains, template)
            aatype.append(_aatype)
            atom_positions.append(_atom_positions)
            atom_mask.append(_atom_mask)
        aatype = np.stack(aatype, axis=0)
        atom_positions = np.stack(atom_positions, axis=0)
        atom_mask = np.stack(atom_mask, axis=0)
        from protenix.data.template.template_featurizer import Templates
        result.update(
            Templates(aatype, atom_positions, atom_mask).as_protenix_dict())
    else:
        aatype = np.full(num_res, unk_restype_index, dtype=np.int64)
        atom_positions = np.zeros((num_res, 37, 3), dtype=np.float32)
        atom_mask = np.zeros((num_res, 37), dtype=np.float32)
        from protenix.data.template.template_featurizer import Templates
        result.update(
            Templates(aatype[None], atom_positions[None], atom_mask[None]).as_protenix_dict())

    # Clean up temporary directories
    for d in temp_dirs:
        try:
            shutil.rmtree(d, ignore_errors=True)
        except Exception:
            pass

    return result, atom_array, token_array

def _load_template(chains, template):
    path = template["template"]
    target_chains = template["template_chains"]
    _aatype, _atom_positions, _atom_mask = load_template_from_structure(path)
    target_chain_mask = []
    idx = 0
    full_length = 0
    names = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    for chain in chains:
        for i in range(chain["num_repeats"]):
            name = names[idx]
            is_target = name in target_chains
            length = len(chain["sequence"])
            full_length += length
            target_chain_mask += [is_target] * length
            idx += 1
    target_chain_mask = np.array(target_chain_mask, dtype=np.bool_)
    aatype = np.zeros((full_length,), dtype=np.int32)
    atom_positions = np.zeros((full_length, 37, 3), dtype=np.float32)
    atom_mask = np.zeros((full_length, 37,), dtype=np.bool_)
    aatype[target_chain_mask] = _aatype
    atom_positions[target_chain_mask] = _atom_positions
    atom_mask[target_chain_mask] = _atom_mask
    return aatype, atom_positions, atom_mask

def load_template_from_structure(
    path: str,
):
    structure = gemmi.read_structure(path)
    model = structure[0]
    _aatype = []
    _atom_positions = []
    _atom_mask = []
    for chain in model:
        residues = [
            res for res in chain
            if gemmi.find_tabulated_residue(res.name).is_amino_acid()
        ]
        num_residues = len(residues)

        atom_positions = np.zeros((num_residues, 37, 3), dtype=np.float32)
        atom_mask = np.zeros((num_residues, 37), dtype=np.float32)
        res_names = []

        for i, res in enumerate(residues):
            res_names.append(res.name)
            for atom in res:
                aname = atom.name
                if aname in atom_order:
                    aidx = atom_order[aname]
                    atom_positions[i, aidx] = [atom.pos.x, atom.pos.y, atom.pos.z]
                    atom_mask[i, aidx] = 1.0

        aatype = np.array(
            [
                restype_order.get(
                    gemmi.find_tabulated_residue(name).one_letter_code,
                    unk_restype_index,
                )
                for name in res_names
            ],
            dtype=np.int64,
        )
        _aatype.append(aatype)
        _atom_positions.append(atom_positions)
        _atom_mask.append(atom_mask)
    _aatype = np.concatenate(_aatype, axis=0)
    _atom_positions = np.concatenate(_atom_positions, axis=0)
    _atom_mask = np.concatenate(_atom_mask, axis=0)

    return _aatype, _atom_positions, _atom_mask

# adapted from escalante.bio/protenij, under Apache 2.0 license
def _make_dummy_msa(sequence: str, msa_dir: str) -> None:
    """Create dummy MSA files (just the query sequence)."""
    os.makedirs(msa_dir, exist_ok=True)
    for fname in ["pairing.a3m", "non_pairing.a3m"]:
        with open(os.path.join(msa_dir, fname), "w") as f:
            f.write(f">query\n{sequence}\n")

# adapted from escalante.bio/protenij, under Apache 2.0 license
def _run_msa_search(sequences: Sequence[str], msa_dir: str, email: str = "") -> list:
    """Run MMSeqs2 search for sequences, return list of result directories."""
    os.makedirs(msa_dir, exist_ok=True)
    seqs_sorted = sorted(set(sequences))
    tmp_fasta = os.path.join(msa_dir, "msa_input.fasta")

    return RequestParser.msa_search(
        seqs_pending_msa=seqs_sorted,
        tmp_fasta_fpath=tmp_fasta,
        msa_res_dir=msa_dir,
        mode="colabfold",
        email=email,
    )

class ProtenixInput(ModelInput):
    features: dict

    @property
    def residue_index(self):
        return self.features["residue_index"]

    @property
    def chain_index(self):
        return self.features["asym_id"]

    @property
    def residue_type(self):
        return jnp.argmax(self.features["restype"], axis=-1)

    def copy(self):
        return type(self)(features={k: v for k, v in self.features.items()})

    def set_aa(self, sequence, start=0):
        return self._set_sequence(sequence, start=start, seq_slice=slice(0, 20), seq_count=20)

    def set_rna(self, sequence, start=0):
        return self._set_sequence(sequence, start=start, seq_slice=slice(21, 25), seq_count=4)

    def set_dna(self, sequence, start=0):
        return self._set_sequence(sequence, start=start, seq_slice=slice(26, 30), seq_count=4)

    def _token_valid_slice(self, start: int, end: int) -> jax.Array:
        if "token_pad_mask" in self.features and self.features["token_pad_mask"] is not None:
            return (jnp.asarray(self.features["token_pad_mask"]) > 0)[start:end]
        return jnp.ones((end - start,), dtype=jnp.bool_)

    def _compute_masked_profile(self, msa: jax.Array) -> jax.Array:
        one_hot = jax.nn.one_hot(jnp.asarray(msa), 32, dtype=jnp.float32)
        msa_mask = self.features.get("msa_mask", None)
        if msa_mask is not None and jnp.asarray(msa_mask).ndim >= 2:
            row_valid = (jnp.asarray(msa_mask) > 0).any(axis=-1).astype(jnp.float32)
            row_valid = row_valid[: one_hot.shape[0]]
            profile = (one_hot * row_valid[:, None, None]).sum(axis=0) / jnp.maximum(
                row_valid.sum(), 1.0
            )
        else:
            profile = one_hot.mean(axis=0)
        if "token_pad_mask" in self.features and self.features["token_pad_mask"] is not None:
            tok_mask = (jnp.asarray(self.features["token_pad_mask"]) > 0).astype(jnp.float32)
            profile = profile * tok_mask[:, None]
        return profile

    def _set_res_type(self, sequence, start=0, seq_slice=slice(0, 20), seq_count=20):
        result = self.copy()
        end = start + sequence.shape[0]
        tok_mask = result._token_valid_slice(start, end).astype(jnp.float32)[:, None]
        result.features["restype"] = jnp.array(result.features["restype"]).astype(jnp.float32)
        result.features["restype"] = result.features["restype"].at[start:end].set(0.0)
        result.features["restype"] = result.features["restype"].at[start:end, seq_slice].set(
            sequence[:, :seq_count] * tok_mask
        )
        return result

    def _set_profile(self, sequence, start=0, seq_slice=slice(0, 20), seq_count=20):
        """Update profile from MSA. MSA is 2D int (N_msa, L)."""
        result = self.copy()
        end = start + sequence.shape[0]
        profile = result._compute_masked_profile(result.features["msa"])
        result.features["profile"] = jnp.array(result.features["profile"]).astype(jnp.float32)
        result.features["profile"] = result.features["profile"].at[start:end].set(
            profile[start:end]
        )
        return result

    def _set_msa(self, sequence, start=0, seq_slice=slice(0, 20), seq_count=20):
        """Update MSA. MSA is 2D integer (N_msa, L) with values 0..31."""
        result = self.copy()
        msa = jnp.asarray(result.features["msa"])
        length = sequence.shape[0]
        end = start + length
        tok_mask = result._token_valid_slice(start, end)
        # Row 0: set to argmax of one-hot sequence on valid tokens, keep gap (31) on padded
        query_idx = jnp.argmax(sequence[:, :seq_count], axis=-1)
        if seq_slice.start > 0:
            query_idx = query_idx + seq_slice.start
        query_idx = jnp.where(tok_mask, query_idx, jnp.asarray(31, dtype=query_idx.dtype))
        msa = msa.at[0, start:end].set(query_idx.astype(msa.dtype))
        # Rows 1+: set to gap (31)
        msa = msa.at[1:, start:end].set(31)
        result.features["msa"] = msa
        return result

    def _set_sequence(self, sequence, start=0, seq_slice=slice(0, 20), seq_count=20,
                      reset_msa=True, reset_profile=True):
        result = self.copy()
        length = sequence.shape[0]
        previous_res_type = result.features["restype"][start:start+length]
        # Build full-width one-hot at the correct slice position
        full_seq = jnp.zeros_like(previous_res_type).at[:, seq_slice].set(sequence[:, :seq_count])
        result = result._set_res_type(full_seq, start=start,
                                       seq_slice=slice(0, full_seq.shape[-1]),
                                       seq_count=full_seq.shape[-1])
        if reset_msa:
            result = result._set_msa(sequence, start=start, seq_slice=seq_slice, seq_count=seq_count)
        if reset_profile:
            result = result._set_profile(sequence, start=start, seq_slice=seq_slice, seq_count=seq_count)
        return result

    def set_template(self, coords, start=None,
                     mask=None, restype=None, template_id=0):
        """Set template features.

        Args:
            coords: Template coordinates (L_template, 14, 3) or (L_template, 24, 3).
            start: Start position in residue index. Defaults to 0.
            mask: Residue mask (L_template,). Defaults to all ones.
            restype: Residue type one-hot (L, 32). Defaults to current restype.
            template_id: Template index (unused, templates are set as single).
        """
        if start is None:
            start = 0
        if restype is None:
            restype = self.features["restype"]
        # Slice to atom14 if atom24
        if coords.shape[1] > 14:
            coords = coords[:, :14]
        num_template = coords.shape[0]
        num_protein = self.features["restype"].shape[0]
        if mask is None:
            mask = jnp.ones((num_template,), dtype=jnp.bool_)
        if num_protein > num_template:
            _mask = jnp.zeros((num_protein,), dtype=jnp.bool_)
            _mask = _mask.at[start:start+num_template].set(mask)
            _coords = jnp.zeros((num_protein, 14, 3), dtype=jnp.float32)
            _coords = _coords.at[start:start+num_template].set(coords)
            mask = _mask
            coords = _coords
        ncacocb = positions_to_ncacocb(coords)
        pb_mask = mask
        bb_mask = mask
        aatype_idx = jnp.argmax(restype, axis=-1) if len(restype.shape) > 1 else restype
        pseudo_beta = jnp.where(
            (aatype_idx == aas.AF2_CODE.index("G"))[:, None],
            ncacocb[:, 1], ncacocb[:, 4])
        dist = jnp.linalg.norm(pseudo_beta[:, None] - pseudo_beta[None, :], axis=-1)
        min_bin: float = 3.25
        max_bin: float = 50.75
        bin_edges = jnp.linspace(min_bin, max_bin, 39)
        bin_edges = jnp.concatenate((bin_edges, jnp.array([1e8])), axis=-1)
        dgram = (dist[:, :, None] > bin_edges[:-1]) * (dist[:, :, None] < bin_edges[1:])
        unit_vector, uv_mask = compute_unit_vector(coords, mask)
        pb_mask_2d = pb_mask[:, None] * pb_mask[None, :]
        # All template features have leading template axis (T, ...)
        # template_aatype is integer (T, L) matching Templates.as_protenix_dict()
        template_features = {
            "template_aatype": aatype_idx[None],
            "template_distogram": dgram[None].astype(jnp.float32),
            "template_pseudo_beta_mask": pb_mask_2d[None].astype(jnp.float32),
            "template_unit_vector": unit_vector[None],
            "template_backbone_frame_mask": uv_mask[None].astype(jnp.float32),
        }
        result = self.copy()
        result.features.update(template_features)
        return result

    def set_msa(self, msa, start=0):
        """Set MSA features directly. Supports 2D int (N, L) or 3D one-hot (N, L, C)."""
        result = self.copy()
        if len(msa.shape) == 3:
            msa = jnp.argmax(msa, axis=-1)
        current_msa = jnp.asarray(result.features["msa"])
        end = start + msa.shape[1]
        tok_mask = result._token_valid_slice(start, end)
        msa_Slice = jnp.where(
            tok_mask[None, :],
            jnp.asarray(msa, dtype=current_msa.dtype),
            jnp.asarray(31, dtype=current_msa.dtype),
        )
        current_msa = current_msa.at[:msa.shape[0], start:end].set(msa_Slice)
        result.features["msa"] = current_msa
        result.features["profile"] = result._compute_masked_profile(current_msa)
        return result

    def load_msa(self, path: str, start=0, msa_start=0, msa_end=None) -> "ProtenixInput":
        """Load MSA from npz file."""
        msa_features = np.load(path)
        result = self.copy()
        msa = msa_features["msa"]
        if len(msa.shape) == 4:
            msa = msa[0]  # Drop batch dim
        if len(msa.shape) == 3:
            msa = np.argmax(msa, axis=-1)
        msa_slice = msa[:, msa_start:msa_end]
        msa_length = msa_slice.shape[1]
        sequence_count = msa_slice.shape[0]
        current_msa = jnp.asarray(result.features["msa"])
        current_sequence_count = current_msa.shape[0]
        if sequence_count > current_sequence_count:
            difference = sequence_count - current_sequence_count
            padding = jnp.full(
                (difference, current_msa.shape[1]),
                31, dtype=current_msa.dtype)  # 31 = gap
            current_msa = jnp.concatenate((current_msa, padding), axis=0)
        end = start + msa_length
        tok_mask = result._token_valid_slice(start, end)
        masked_slice = jnp.where(
            tok_mask[None, :],
            jnp.asarray(msa_slice, dtype=current_msa.dtype),
            jnp.asarray(31, dtype=current_msa.dtype),
        )
        current_msa = current_msa.at[:sequence_count, start:end].set(masked_slice)
        result.features["msa"] = current_msa
        result.features["profile"] = result._compute_masked_profile(current_msa)
        return result

    def pad_to(self, size: int | None,
               num_residue_atoms: int = 24, *,
               num_msa: int | None = None,
               num_templates: int | None = None) -> "ProtenixInput":
        """Pad ProtenixInput to target token/atom counts.

        Delegates to protenix.token_padding.pad_token_features and
        protenix.atom_padding.pad_atom_features.

        Args:
            size: Target number of tokens. If None, keeps current.
            num_residue_atoms: Atoms per residue for atom padding target.
            num_msa: Target MSA depth. Monotonic (warn+no-op if smaller).
            num_templates: Target template count. Monotonic.
        """
        current_tokens = int(self.features["restype"].shape[0])
        current_atoms = int(self.features["atom_to_token_idx"].shape[-1])

        if size is None:
            target_tokens = current_tokens
            target_atoms = current_atoms
        elif size < current_tokens:
            warnings.warn(
                f"pad_to({size}) < current token count ({current_tokens}), "
                f"keeping current size.",
                UserWarning,
                stacklevel=2,
            )
            target_tokens = current_tokens
            target_atoms = current_atoms
        else:
            target_tokens = int(size)
            requested_atoms = int(num_residue_atoms) * target_tokens
            if current_atoms > requested_atoms:
                warnings.warn(
                    f"Current atoms ({current_atoms}) > num_residue_atoms * "
                    f"target_tokens ({requested_atoms}), "
                    f"keeping current atom count.",
                    UserWarning,
                    stacklevel=2,
                )
            target_atoms = max(current_atoms, requested_atoms)

        # Check MSA monotonicity
        if num_msa is not None:
            if "msa" in self.features:
                current_msa = self.features["msa"].shape[0]
                if num_msa < current_msa:
                    warnings.warn(
                        f"num_msa={num_msa} < current MSA depth ({current_msa}), "
                        f"keeping current depth.",
                        UserWarning,
                        stacklevel=2,
                    )
                    num_msa = None

        # Check template monotonicity
        if num_templates is not None:
            if "template_aatype" in self.features:
                current_templates = self.features["template_aatype"].shape[0]
                if num_templates < current_templates:
                    warnings.warn(
                        f"num_templates={num_templates} < current count ({current_templates}), "
                        f"keeping current count.",
                        UserWarning,
                        stacklevel=2,
                    )
                    num_templates = None

        no_token_pad = (target_tokens == current_tokens)
        no_atom_pad = (target_atoms == current_atoms)
        if no_token_pad and no_atom_pad and num_msa is None and num_templates is None:
            return self

        features = dict(self.features)

        # Initialize pad masks BEFORE padding (so pad_token/atom_features see them)
        features.setdefault("token_pad_mask", jnp.ones(current_tokens, dtype=jnp.bool_))
        features.setdefault("atom_pad_mask", jnp.ones(current_atoms, dtype=jnp.bool_))

        # Pad token-axis features
        if not no_token_pad:
            try:
                from protenix.token_padding import pad_token_features
                features = pad_token_features(features, target_tokens)
            except ImportError:
                features = _pad_token_features_fallback(features, current_tokens, target_tokens)

        # Pad atom-axis features
        if not no_atom_pad:
            try:
                from protenix.atom_padding import pad_atom_features
                features = pad_atom_features(features, target_atoms)
            except ImportError:
                features = _pad_atom_features_fallback(features, current_atoms, target_atoms)

        # Extend pad masks for newly padded positions
        if not no_token_pad:
            existing = jnp.asarray(features["token_pad_mask"])
            if existing.shape[0] < target_tokens:
                features["token_pad_mask"] = jnp.pad(
                    existing, [(0, target_tokens - existing.shape[0])],
                    constant_values=False)

        if not no_atom_pad:
            existing = jnp.asarray(features["atom_pad_mask"])
            if existing.shape[0] < target_atoms:
                features["atom_pad_mask"] = jnp.pad(
                    existing, [(0, target_atoms - existing.shape[0])],
                    constant_values=False)

        # If msa_mask already exists and token count grew, right-pad axis 1 with 0.0
        if "msa_mask" in features:
            msa_mask_existing = jnp.asarray(features["msa_mask"])
            if msa_mask_existing.ndim == 2 and msa_mask_existing.shape[1] < target_tokens:
                features["msa_mask"] = jnp.pad(
                    msa_mask_existing,
                    [(0, 0), (0, target_tokens - msa_mask_existing.shape[1])],
                )

        # Pad MSA rows
        if num_msa is not None and "msa" in features:
            current_msa = features["msa"].shape[0]
            if num_msa > current_msa:
                diff = num_msa - current_msa
                msa = jnp.asarray(features["msa"])
                # MSA is 2D int: pad with gap (31)
                if msa.ndim == 2:
                    features["msa"] = jnp.pad(
                        msa, [(0, diff), (0, 0)],
                        constant_values=31)
                else:
                    features["msa"] = jnp.pad(msa,
                        [(0, diff)] + [(0, 0)] * (msa.ndim - 1))
                # Create/pad msa_mask: 1 on valid rows & valid tokens, 0 on padded
                if "msa_mask" not in features:
                    tok_mask_f = (jnp.asarray(features["token_pad_mask"]) > 0).astype(jnp.float32)
                    features["msa_mask"] = jnp.broadcast_to(
                        tok_mask_f[None, :], (current_msa, features["msa"].shape[1])
                    )
                msa_mask = jnp.asarray(features["msa_mask"])
                features["msa_mask"] = jnp.pad(msa_mask,
                    [(0, diff)] + [(0, 0)] * (msa_mask.ndim - 1))
                # Pad has_deletion and deletion_value on axis 0
                for msa_key in ("has_deletion", "deletion_value"):
                    if msa_key in features:
                        val = jnp.asarray(features[msa_key])
                        features[msa_key] = jnp.pad(val,
                            [(0, diff)] + [(0, 0)] * (val.ndim - 1))

        # Pad template axis
        if num_templates is not None and "template_aatype" in features:
            current_templates = features["template_aatype"].shape[0]
            if num_templates > current_templates:
                diff = num_templates - current_templates
                for tkey in ["template_aatype", "template_distogram",
                             "template_pseudo_beta_mask", "template_unit_vector",
                             "template_backbone_frame_mask"]:
                    if tkey in features:
                        features[tkey] = jnp.pad(jnp.asarray(features[tkey]),
                            [(0, diff)] + [(0, 0)] * (len(features[tkey].shape) - 1))
                if "template_mask" in features:
                    features["template_mask"] = jnp.pad(
                        jnp.asarray(features["template_mask"]), [(0, diff)])

        # Set asym_id = -1 on padded tokens
        if not no_token_pad and "asym_id" in features:
            features["asym_id"] = jnp.asarray(features["asym_id"]).at[current_tokens:].set(-1)

        return ProtenixInput(features=features)

    def pad_to_nearest_multiple(self, multiple: int,
                                num_residue_atoms: int = 24, *,
                                num_msa: int | None = None,
                                num_templates: int | None = None) -> "ProtenixInput":
        """Pad to the nearest multiple of `multiple`."""
        current_tokens = self.features["restype"].shape[0]
        target = ((current_tokens + multiple - 1) // multiple) * multiple
        return self.pad_to(target, num_residue_atoms=num_residue_atoms,
                           num_msa=num_msa, num_templates=num_templates)

    def pad_atoms(self, num_residue_atoms: int = 24) -> "ProtenixInput":
        """Right-pad only the atom axis to `max(current_atoms, num_residue_atoms * L)`."""
        if num_residue_atoms <= 0:
            raise ValueError(
                f"num_residue_atoms must be positive, got {num_residue_atoms}."
            )
        return self.pad_to(
            int(self.residue_index.shape[0]),
            num_residue_atoms=num_residue_atoms,
        )


def _pad_token_features_fallback(features, current_tokens, target_tokens):
    """Fallback token padding when protenix.token_padding is not available."""
    diff = target_tokens - current_tokens
    result = dict(features)
    for key, value in features.items():
        if not isinstance(value, (jnp.ndarray, np.ndarray)):
            continue
        shape = value.shape
        if len(shape) == 0:
            continue
        value = jnp.asarray(value)
        # (L, L, ...) pair features — check first to avoid (L, ...) catch
        if len(shape) >= 2 and shape[0] == current_tokens and shape[1] == current_tokens:
            pad_widths = [(0, diff), (0, diff)] + [(0, 0)] * (len(shape) - 2)
            result[key] = jnp.pad(value, pad_widths)
        # (L, ...) token features
        elif shape[0] == current_tokens:
            pad_widths = [(0, diff)] + [(0, 0)] * (len(shape) - 1)
            padded = jnp.pad(value, pad_widths)
            if key == "asym_id":
                padded = padded.at[current_tokens:].set(-1)
            result[key] = padded
        # (*, L, ...) MSA features
        elif len(shape) >= 2 and shape[1] == current_tokens:
            pad_widths = [(0, 0), (0, diff)] + [(0, 0)] * (len(shape) - 2)
            result[key] = jnp.pad(value, pad_widths)
    return result


def _pad_atom_features_fallback(features, current_atoms, target_atoms):
    """Fallback atom padding when protenix.atom_padding is not available."""
    diff = target_atoms - current_atoms
    result = dict(features)
    for key, value in features.items():
        if not isinstance(value, (jnp.ndarray, np.ndarray)):
            continue
        shape = value.shape
        if len(shape) == 0:
            continue
        value = jnp.asarray(value)
        if shape[0] == current_atoms:
            pad_widths = [(0, diff)] + [(0, 0)] * (len(shape) - 1)
            result[key] = jnp.pad(value, pad_widths)
        elif len(shape) >= 2 and shape[-1] == current_atoms:
            pad_widths = [(0, 0)] * (len(shape) - 1) + [(0, diff)]
            result[key] = jnp.pad(value, pad_widths)
    return result


# adapted from escalante-bio/protenij under Apache 2.0 license
def compute_unit_vector(
    atom_positions: jnp.ndarray,
    mask: jnp.ndarray,
    epsilon: float = 1e-6,
) -> tuple[jnp.ndarray, jnp.ndarray]:

    n_pos = atom_positions[:, 0]
    ca_pos = atom_positions[:, 1]
    c_pos = atom_positions[:, 2]

    # Build local frame: origin at CA, x-axis along C-CA
    v1 = c_pos - ca_pos  # C-CA direction
    v2 = n_pos - ca_pos  # N-CA direction

    # Gram-Schmidt orthogonalization
    e1 = v1 / (jnp.linalg.norm(v1, axis=-1, keepdims=True) + epsilon)
    e2 = v2 - jnp.sum(v2 * e1, axis=-1, keepdims=True) * e1
    e2 = e2 / (jnp.linalg.norm(e2, axis=-1, keepdims=True) + epsilon)
    e3 = jnp.cross(e1, e2)

    # Relative positions: diff[i, j] = CA[j] - CA[i]
    diff = ca_pos[None, :, :] - ca_pos[:, None, :]

    # Transform to local frame
    ux = jnp.sum(e1[:, None, :] * diff, axis=-1)
    uy = jnp.sum(e2[:, None, :] * diff, axis=-1)
    uz = jnp.sum(e3[:, None, :] * diff, axis=-1)

    unit_vector = jnp.stack([ux, uy, uz], axis=-1)

    # Normalize to unit vector
    uv_norm = jnp.linalg.norm(unit_vector, axis=-1, keepdims=True)
    unit_vector = unit_vector / (uv_norm + epsilon)

    # 2D mask: valid only if both residues have backbone
    mask_2d = mask[:, None] * mask[None, :]

    return unit_vector, mask_2d
