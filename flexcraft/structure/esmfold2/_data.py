import warnings
from dataclasses import fields
from pathlib import Path
from typing import Any

import gemmi
import jax
import jax.numpy as jnp
import numpy as np
from esm.models.esmfold2 import (
    DNAInput,
    ESMFold2InputBuilder,
    LigandInput,
    ProteinInput,
    RNAInput,
    StructurePredictionInput,
)
from esmjfold2 import Features, output_to_molecular_complex

from flexcraft.sequence.aa_codes import AF2_CODE
from flexcraft.structure.common._data import (
    InputSpec,
    ModelInput,
    _AA_SLICE,
    _AA_UNK,
)
from flexcraft.structure.common._utils import (
    align_up,
    pad_features_by_schema,
)

NUM_RES_TYPES = 33
ESM_PAD_ID = 1
ESM_NON_PROTEIN_ID = 24
_CHAIN_NAMES = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# Boltz 20-AA order (ARNDCQEGHILKMFPSTWYV) -> ESM_PROTEIN_VOCAB token IDs
_BOLTZ_AA_TO_ESM_ID = jnp.array(
    [5, 10, 17, 13, 23, 16, 9, 6, 21, 12, 4, 15, 20, 18, 14, 8, 11, 22, 19, 7],
    dtype=jnp.int32,
)

# Feature keys ESMFold2 consumes directly
FEATURE_FIELDS = tuple(field.name for field in fields(Features))

ESMFOLD2_FEATURE_SCHEMA: dict[str, Any] = {
    "res_type": (None, "token", None),
    "profile": (None, "token", None),
    "deletion_mean": (None, "token"),
    "residue_index": (None, "token"),
    "token_index": (None, "token"),
    "asym_id": (None, "token"),
    "entity_id": (None, "token"),
    "sym_id": (None, "token"),
    "mol_type": (None, "token"),
    "input_ids": (None, "token"),
    "token_attention_mask": (None, "token"),
    "token_pad_mask": (None, "token"),
    "distogram_atom_idx": (None, "token"),
    "center_atom_idx": (None, "token"),
    "cyclic_period": (None, "token"),
    "pocket_feature": lambda arr: (
        (None, "token", None) if arr.ndim == 3 else (None, "token")
    ),
    "frames_idx": lambda arr: (
        (None, None, "token", None)
        if arr.ndim == 4
        else (None, "token", None)
    ),
    "disto_cond": (None, "token", "token"),
    "disto_cond_mask": (None, "token", "token"),
    "token_bonds": lambda arr: (
        (None, "token", "token", None)
        if arr.ndim == 4
        else (None, "token", "token")
    ),
    "type_bonds": (None, "token", "token"),
    "pair_mask": (None, "token", "token"),
    "atom_attention_mask": (None, "atom"),
    "atom_pad_mask": (None, "atom"),
    "atom_to_token": lambda arr: (
        (None, "atom", "token") if arr.ndim == 3 else (None, "atom")
    ),
    "ref_pos": (None, "atom", None),
    "ref_charge": (None, "atom"),
    "ref_element": lambda arr: (
        (None, "atom", None) if arr.ndim == 3 else (None, "atom")
    ),
    "ref_atom_name_chars": lambda arr: (
        (None, "atom", None, None)
        if arr.ndim == 4
        else (None, "atom", None)
    ),
    "ref_space_uid": (None, "atom"),
    "ref_chirality": (None, "atom"),
    "is_resolved": (None, "atom"),
    "gt_coords": lambda arr: (
        (None, None, "atom", None)
        if arr.ndim == 4
        else (None, "atom", None)
    ),
    "msa": (None, "msa", "token", None),
    "msa_attention_mask": (None, "msa", "token"),
    "has_deletion": (None, "msa", "token"),
    "deletion_value": (None, "msa", "token"),
    "msa_paired": (None, "msa", "token"),
}

ESMFOLD2_PAD_VALUES: dict[str, Any] = {
    "res_type": 0.0,
    "asym_id": -1,
    "entity_id": -1,
    "input_ids": ESM_PAD_ID,
    "token_attention_mask": False,
    "deletion_mean": 0.0,
    "atom_attention_mask": False,
    "msa_attention_mask": False,
}


def align_esmfold2_atoms_w32(features: dict) -> dict:
    """Ensure ESMFold2 atom-axis features have `N_atoms % 32 == 0`."""
    if (
        "atom_attention_mask" not in features
        or features["atom_attention_mask"] is None
    ):
        return features
    current_atoms = int(features["atom_attention_mask"].shape[1])
    target_atoms = align_up(max(current_atoms, 32), 32)
    if target_atoms == current_atoms:
        return features
    return pad_features_by_schema(
        features,
        axis_targets={"atom": target_atoms},
        schema=ESMFOLD2_FEATURE_SCHEMA,
        pad_values=ESMFOLD2_PAD_VALUES,
    )


def _parse_a3m_to_boltz_onehot(path: str, max_sequences: int = 1024) -> jax.Array:
    """Parse an A3M file into `(1, M, L, 33)` Boltz-layout one-hot MSA."""
    char_to_boltz = {aa: idx + 2 for idx, aa in enumerate(AF2_CODE)}
    char_to_boltz["-"] = 1
    sequences = []
    current = []
    for raw_line in Path(path).read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith(">"):
            if current:
                sequences.append("".join(current))
                if len(sequences) >= max_sequences:
                    current = []
                    break
                current = []
            continue
        current.append(
            "".join(ch for ch in line if not ch.islower() and ch != ".")
        )
    if current and len(sequences) < max_sequences:
        sequences.append("".join(current))
    if not sequences:
        raise ValueError(f"No sequences found in A3M file '{path}'.")
    encoded = np.array(
        [
            [char_to_boltz.get(ch, _AA_UNK) for ch in seq]
            for seq in sequences
        ],
        dtype=np.int32,
    )
    return jax.nn.one_hot(jnp.asarray(encoded)[None], NUM_RES_TYPES)


class ESMFold2Spec(InputSpec):
    def __init__(self, *chains, templates=None, constraints=None):
        super().__init__(*chains, templates=templates, constraints=constraints)

    def to_features(
        self, pad: bool = True, cache: str = "./params/esmfold2/", **kwargs
    ):
        return to_features(self.chains, self.templates, self.constraints)

    def to_input(
        self, pad: bool = True, cache: str = "./params/esmfold2/", **kwargs
    ) -> tuple["ESMFold2Input", "ESMFold2Writer"]:
        features, chain_infos = self.to_features(
            pad=pad, cache=cache, **kwargs
        )
        writer = ESMFold2Writer(features, chain_infos)
        esm_input = ESMFold2Input(features=features)
        if self.precomputed_msas:
            for msa_info in self.precomputed_msas:
                esm_input = esm_input.load_msa(
                    msa_info["path"], start=msa_info["offset"]
                )
        esm_input = self._apply_pad_spec(esm_input)
        return esm_input, writer

    def save_msa(self, path: str, cache: str = "./params/esmfold2/"):
        esm_input, _ = self.to_input(pad=False, cache=cache)
        esm_input.save_msa(path)


class ESMFold2Writer:
    def __init__(self, features, chain_infos):
        valid_atoms = np.asarray(
            features["atom_attention_mask"][0], dtype=bool
        )
        self.n_valid_atoms = int(valid_atoms.sum())
        self.atom_mask = np.ones((self.n_valid_atoms,), dtype=bool)
        self.ref_element = np.asarray(features["ref_element"][0])[valid_atoms]
        self.ref_atom_name_chars = np.asarray(
            features["ref_atom_name_chars"][0]
        )[valid_atoms]
        if (
            "token_attention_mask" in features
            and features["token_attention_mask"] is not None
        ):
            self.num_tokens = int(
                np.asarray(
                    features["token_attention_mask"][0], dtype=bool
                ).sum()
            )
        else:
            self.num_tokens = int(features["residue_index"].shape[1])
        self.chain_infos = chain_infos

    def to_complex(self, coords, plddt=None, complex_id="model"):
        coords_np = np.asarray(coords)[: self.n_valid_atoms]
        if plddt is None:
            plddt_np = np.zeros((self.num_tokens,), dtype=np.float32)
        else:
            plddt_np = np.asarray(plddt)[: self.num_tokens]
        return output_to_molecular_complex(
            coords=coords_np,
            plddt=plddt_np,
            atom_mask=self.atom_mask,
            ref_element=self.ref_element,
            ref_atom_name_chars=self.ref_atom_name_chars,
            chain_infos=self.chain_infos,
            complex_id=complex_id,
        )

    def to_mmcif(self, sample_atom_coords, plddt=None, complex_id="model"):
        return self.to_complex(
            sample_atom_coords, plddt=plddt, complex_id=complex_id
        ).to_mmcif()

    def save_pdb(
        self, path, sample_atom_coords, plddt=None, keep_chains=None
    ):
        cif = self.to_mmcif(
            sample_atom_coords, plddt=plddt, complex_id=Path(path).stem
        )
        block = gemmi.cif.read_string(cif).sole_block()
        structure = gemmi.make_structure_from_block(block)
        structure.setup_entities()
        structure.write_minimal_pdb(str(path))
        if keep_chains is not None:
            keep_pdb_chains(path, keep_chains)

    def save_cif(self, path, sample_atom_coords, plddt=None):
        Path(path).write_text(
            self.to_mmcif(
                sample_atom_coords, plddt=plddt, complex_id=Path(path).stem
            )
        )


def keep_pdb_chains(path, keep_chains):
    """Drop every chain except `keep_chains` from a PDB file in place."""
    keep = set(keep_chains)
    lines = Path(path).read_text().splitlines(keepends=True)
    Path(path).write_text(
        "".join(
            line
            for line in lines
            if not line.startswith(("ATOM", "HETATM", "ANISOU", "TER"))
            or line[21:22] in keep
        )
    )


def to_features(chains, templates, constraints):
    if templates:
        warnings.warn(
            "Skipping provided templates: ESMFold2 has no template pathway."
        )
    if constraints:
        warnings.warn(
            "Skipping provided constraints for ESMFold2: covalent bonds "
            "and pockets are not wired up yet."
        )
    sequences = _to_esm_chains(chains)
    if not sequences:
        raise ValueError("ESMFold2 requires at least one chain.")
    features, chain_infos = ESMFold2InputBuilder().prepare_input(
        StructurePredictionInput(sequences=sequences)
    )
    features = {name: _to_jax(value) for name, value in features.items()}
    features["res_type"] = jax.nn.one_hot(features["res_type"], NUM_RES_TYPES)
    if features.get("msa") is not None:
        features["msa"] = jax.nn.one_hot(features["msa"], NUM_RES_TYPES)
    if features.get("deletion_mean") is None:
        features["deletion_mean"] = jnp.zeros(
            features["residue_index"].shape, dtype=jnp.float32
        )
    features = align_esmfold2_atoms_w32(features)
    return features, chain_infos


def to_model_features(features: dict) -> Features:
    """Select the keys ESMFold2 consumes and wrap them in its Features module."""
    return Features(
        **{name: features[name] for name in FEATURE_FIELDS if name in features}
    )


def _to_jax(value):
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return jnp.asarray(value)


def _to_esm_chains(chains):
    """Map flexcraft chain dicts onto the esm input classes."""
    sequences = []
    next_id = 0
    for chain in chains:
        kind = chain.get("kind", "protein")
        if kind not in _CHAIN_KINDS:
            warnings.warn(
                f"Skipping invalid chain type '{kind}' for ESMFold2."
            )
            continue
        num_repeats = chain.get("num_repeats", 1)
        if next_id + num_repeats > len(_CHAIN_NAMES):
            raise ValueError(
                f"More than {len(_CHAIN_NAMES)} chains requested; ESMFold2 writes "
                f"single-character chain ids."
            )
        ids = [_CHAIN_NAMES[next_id + i] for i in range(num_repeats)]
        next_id += num_repeats
        sequences.append(
            _CHAIN_KINDS[kind](ids if num_repeats > 1 else ids[0], chain)
        )
    return sequences


def _protein_input(ids, chain):
    return ProteinInput(
        id=ids, sequence=chain["sequence"], msa=_load_msa(chain)
    )


def _rna_input(ids, chain):
    return RNAInput(id=ids, sequence=chain["sequence"], msa=_load_msa(chain))


def _dna_input(ids, chain):
    return DNAInput(id=ids, sequence=chain["sequence"])


def _ligand_input(ids, chain):
    ccd = chain.get("ccd")
    if ccd is not None:
        return LigandInput(
            id=ids, ccd=[ccd] if isinstance(ccd, str) else list(ccd)
        )
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
        warnings.warn(
            "ESMFold2 does not run MSA search. Set chain['msa_path'] "
            "to an a3m file to condition on an MSA."
        )
        return None
    from esm.utils.msa import MSA

    return MSA.from_a3m(
        path=path,
        remove_insertions=True,
        max_sequences=chain.get("msa_depth", 1024),
    )


def _ensure_msa_arrays(features: dict, min_rows: int = 1) -> dict:
    """Initialize 1-row query MSA tensors if `features['msa']` is currently None."""
    result = dict(features)
    tok_mask = result["token_attention_mask"].astype(jnp.bool_)
    num_tokens = int(result["res_type"].shape[1])
    if result.get("msa") is None:
        query_row = (
            jnp.asarray(result["res_type"], dtype=jnp.float32)[:, None, :, :]
            * tok_mask[:, None, :, None]
        )
        result["msa"] = query_row
        result["msa_attention_mask"] = tok_mask[:, None, :]
        for opt_key in ("has_deletion", "deletion_value"):
            if opt_key in FEATURE_FIELDS and result.get(opt_key) is None:
                result[opt_key] = jnp.zeros(
                    (1, 1, num_tokens), dtype=jnp.float32
                )
    current_rows = int(result["msa"].shape[1])
    if min_rows > current_rows:
        diff = min_rows - current_rows
        gap_pad = jnp.zeros(
            (1, diff, num_tokens, NUM_RES_TYPES), dtype=jnp.float32
        )
        gap_pad = gap_pad.at[..., 1].set(
            tok_mask[:, None, :].astype(jnp.float32)
        )
        result["msa"] = jnp.concatenate([result["msa"], gap_pad], axis=1)
        mask_pad = jnp.broadcast_to(tok_mask[:, None, :], (1, diff, num_tokens))
        result["msa_attention_mask"] = jnp.concatenate(
            [result["msa_attention_mask"].astype(jnp.bool_), mask_pad], axis=1
        )
        for opt_key in ("has_deletion", "deletion_value"):
            if result.get(opt_key) is not None:
                z_pad = jnp.zeros((1, diff, num_tokens), dtype=jnp.float32)
                result[opt_key] = jnp.concatenate(
                    [result[opt_key], z_pad], axis=1
                )
    return result


class ESMFold2Input(ModelInput):
    features: dict

    def __init__(self, features: dict):
        if isinstance(features, dict):
            updated = dict(features)
            if (
                "residue_index" in updated
                and updated.get("deletion_mean") is None
            ):
                updated["deletion_mean"] = jnp.zeros(
                    updated["residue_index"].shape, dtype=jnp.float32
                )
            if (
                "atom_attention_mask" in updated
                and updated["atom_attention_mask"] is not None
                and updated["atom_attention_mask"].shape[1] % 32 != 0
            ):
                updated = align_esmfold2_atoms_w32(updated)
            features = updated
        self.features = features

    def _set_sequence(
        self,
        sequence,
        start=0,
        seq_slice=_AA_SLICE,
        seq_count=20,
        reset_msa=True,
        reset_profile=True,
    ):
        result = self._set_res_type(
            sequence, start=start, seq_slice=seq_slice, seq_count=seq_count
        )
        tok_mask = result.features["token_attention_mask"].astype(jnp.bool_)
        result.features["res_type"] = (
            result.features["res_type"] * tok_mask[..., None]
        )

        end = start + sequence.shape[0]
        if seq_slice == _AA_SLICE and "input_ids" in result.features:
            aa_idx = jnp.argmax(sequence[:, :20], axis=-1)
            esm_ids = _BOLTZ_AA_TO_ESM_ID[aa_idx]
            mol_slice = result.features["mol_type"][0, start:end]
            tok_valid = tok_mask[0, start:end]
            id_dtype = result.features["input_ids"].dtype
            updated_ids = jnp.where(
                ~tok_valid,
                jnp.asarray(ESM_PAD_ID, dtype=id_dtype),
                jnp.where(
                    mol_slice == 0,
                    esm_ids.astype(id_dtype),
                    jnp.asarray(ESM_NON_PROTEIN_ID, dtype=id_dtype),
                ),
            )
            result.features["input_ids"] = (
                result.features["input_ids"].at[0, start:end].set(updated_ids)
            )

        if reset_msa and result.features.get("msa") is not None:
            result = result._set_msa(
                sequence,
                start=start,
                seq_slice=seq_slice,
                seq_count=seq_count,
            )
            msa_mask = result.features["msa_attention_mask"].astype(jnp.bool_)
            result.features["msa"] = (
                result.features["msa"] * msa_mask[..., None]
            )
        return result

    def set_msa(self, msa, start=0) -> "ESMFold2Input":
        msa = jnp.asarray(msa, dtype=jnp.float32)
        if msa.shape[-1] == 20:
            msa = jnp.pad(
                msa, ((0, 0), (0, 0), (2, NUM_RES_TYPES - 20 - 2))
            )
        end = start + msa.shape[1]
        features = _ensure_msa_arrays(self.features, min_rows=msa.shape[0])
        tok_valid = features["token_attention_mask"][0, start:end].astype(
            jnp.bool_
        )
        msa_arr = jnp.asarray(features["msa"], dtype=jnp.float32)
        msa_mask = features["msa_attention_mask"].astype(jnp.bool_)
        row_was_active = msa_mask[0].any(axis=-1)

        msa_arr = msa_arr.at[0, :, start:end].set(0.0)
        msa_arr = msa_arr.at[0, 1:, start:end, 1].set(
            (row_was_active[1:, None] & tok_valid[None, :]).astype(jnp.float32)
        )
        masked_in = msa * tok_valid[None, :, None]
        msa_arr = msa_arr.at[0, : msa.shape[0], start:end, :].set(masked_in)
        msa_mask = msa_mask.at[0, : msa.shape[0], start:end].set(
            jnp.broadcast_to(tok_valid[None, :], (msa.shape[0], msa.shape[1]))
        )
        features["msa"] = msa_arr * msa_mask[..., None]
        features["msa_attention_mask"] = msa_mask
        return ESMFold2Input(features=features)

    def save_msa(self, path: str) -> "ESMFold2Input":
        features = _ensure_msa_arrays(self.features, min_rows=1)
        save_dict = {
            "msa": np.asarray(features["msa"]),
            "msa_attention_mask": np.asarray(features["msa_attention_mask"]),
            "deletion_mean": np.asarray(features["deletion_mean"]),
        }
        for opt_key in ("has_deletion", "deletion_value"):
            if features.get(opt_key) is not None:
                save_dict[opt_key] = np.asarray(features[opt_key])
        np.savez_compressed(path, **save_dict)
        return self

    def load_msa(
        self, path: str, start=0, msa_start=0, msa_end=None
    ) -> "ESMFold2Input":
        if str(path).endswith(".a3m"):
            msa_onehot = _parse_a3m_to_boltz_onehot(str(path))
            msa_slice = msa_onehot[0, :, msa_start:msa_end, :]
            return self.set_msa(msa_slice, start=start)

        loaded = np.load(path)
        raw_msa = jnp.asarray(loaded["msa"])
        if raw_msa.ndim == 3:
            raw_msa = jax.nn.one_hot(raw_msa, NUM_RES_TYPES)
        raw_msa = raw_msa[:, :, msa_start:msa_end, :].astype(jnp.float32)
        seq_count = int(raw_msa.shape[1])
        msa_len = int(raw_msa.shape[2])
        end = start + msa_len

        if "msa_attention_mask" in loaded:
            in_mask = jnp.asarray(
                loaded["msa_attention_mask"][:, :, msa_start:msa_end],
                dtype=jnp.bool_,
            )
        elif "msa_mask" in loaded:
            in_mask = (
                jnp.asarray(loaded["msa_mask"][:, :, msa_start:msa_end]) > 0
            )
        else:
            in_mask = jnp.ones((1, seq_count, msa_len), dtype=jnp.bool_)

        features = _ensure_msa_arrays(self.features, min_rows=seq_count)
        tok_valid = features["token_attention_mask"][0, start:end].astype(
            jnp.bool_
        )
        in_mask = in_mask & tok_valid[None, None, :]

        features["msa"] = (
            features["msa"]
            .at[0, :seq_count, start:end, :]
            .set(raw_msa[0] * in_mask[0, :, :, None])
        )
        features["msa_attention_mask"] = (
            features["msa_attention_mask"]
            .astype(jnp.bool_)
            .at[0, :seq_count, start:end]
            .set(in_mask[0])
        )
        for opt_key in ("has_deletion", "deletion_value"):
            if opt_key in loaded and features.get(opt_key) is not None:
                del_slice = jnp.asarray(
                    loaded[opt_key][0, :, msa_start:msa_end], dtype=jnp.float32
                )
                features[opt_key] = (
                    features[opt_key]
                    .at[0, :seq_count, start:end]
                    .set(del_slice * in_mask[0])
                )
        if "deletion_value" in loaded:
            del_val = jnp.asarray(
                loaded["deletion_value"][0, :, msa_start:msa_end],
                dtype=jnp.float32,
            )
            weights = in_mask[0].astype(jnp.float32)
            del_mean = (del_val * weights).sum(axis=0) / jnp.maximum(
                1.0, weights.sum(axis=0)
            )
            features["deletion_mean"] = (
                jnp.asarray(features["deletion_mean"], dtype=jnp.float32)
                .at[0, start:end]
                .set(del_mean * tok_valid)
            )
        elif "deletion_mean" in loaded:
            del_mean = jnp.asarray(
                loaded["deletion_mean"][0, msa_start:msa_end], dtype=jnp.float32
            )
            features["deletion_mean"] = (
                jnp.asarray(features["deletion_mean"], dtype=jnp.float32)
                .at[0, start:end]
                .set(del_mean * tok_valid)
            )
        return ESMFold2Input(features=features)

    def pad_to(
        self,
        size: int | None,
        num_residue_atoms: int = 24,
        *,
        num_msa: int | None = None,
        num_templates: int | None = None,
    ) -> "ESMFold2Input":
        """Right-pad token, atom (`W=32`), and MSA axes (`num_templates` is a no-op)."""
        del num_templates  # ESMFold2 has no template pathway; silent no-op.

        features = dict(self.features)
        if features.get("deletion_mean") is None:
            features["deletion_mean"] = jnp.zeros(
                features["residue_index"].shape, dtype=jnp.float32
            )

        current_tokens = int(features["res_type"].shape[1])
        current_atoms = int(features["atom_attention_mask"].shape[1])

        if size is None:
            target_tokens = current_tokens
            target_atoms = align_up(max(current_atoms, 32), 32)
        elif size < current_tokens:
            warnings.warn(
                f"Requested token pad size {size} < current token count "
                f"{current_tokens}; keeping {current_tokens} tokens.",
                UserWarning,
                stacklevel=2,
            )
            target_tokens = current_tokens
            target_atoms = align_up(max(current_atoms, 32), 32)
        else:
            target_tokens = int(size)
            requested_atoms = int(num_residue_atoms) * target_tokens
            if current_atoms > requested_atoms:
                warnings.warn(
                    f"Current atom count {current_atoms} exceeds "
                    f"num_residue_atoms * target_tokens ({requested_atoms}); "
                    f"keeping aligned atom count.",
                    UserWarning,
                    stacklevel=2,
                )
            target_atoms = align_up(max(current_atoms, requested_atoms, 32), 32)

        axis_targets: dict[str, int] = {
            "token": target_tokens,
            "atom": target_atoms,
        }

        if num_msa is not None:
            if features.get("msa") is None:
                features = _ensure_msa_arrays(features, min_rows=1)
            current_msa = int(features["msa"].shape[1])
            if num_msa < current_msa:
                warnings.warn(
                    f"Requested num_msa {num_msa} < current MSA depth "
                    f"{current_msa}; keeping {current_msa}.",
                    UserWarning,
                    stacklevel=2,
                )
            else:
                axis_targets["msa"] = int(num_msa)

        padded = pad_features_by_schema(
            features,
            axis_targets=axis_targets,
            schema=ESMFOLD2_FEATURE_SCHEMA,
            pad_values=ESMFOLD2_PAD_VALUES,
        )
        return ESMFold2Input(features=padded)

    def set_template(self, *args, **kwargs):
        raise NotImplementedError(
            "ESMFold2 has no template pathway. Use Boltz or Protenix for "
            "template conditioned prediction."
        )
