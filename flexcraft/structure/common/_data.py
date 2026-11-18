import copy
from typing import Iterable

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from flexcraft.data.data import DesignData
from flexcraft.files import PDBFile
from flexcraft.structure.common._utils import align_up


def _apply_single_pad_op(model_input, op):
    if op is None:
        return model_input
    if isinstance(op, tuple) and len(op) == 3 and isinstance(op[0], str):
        method_name, args, kwargs = op
        return getattr(model_input, method_name)(*args, **kwargs)
    if isinstance(op, dict):
        if op.get("_multiple") is not None:
            return model_input.pad_to_nearest_multiple(
                op["_multiple"],
                op.get("num_residue_atoms", 24),
                num_msa=op.get("num_msa"),
                num_templates=op.get("num_templates"),
            )
        method_name = op.get("method") or op.get("mode") or op.get("kind")
        if method_name == "pad_atoms":
            return model_input.pad_atoms(op.get("num_residue_atoms", 24))
        if method_name is None:
            method_name = (
                "pad_to_nearest_multiple" if "multiple" in op else "pad_to"
            )
        num_residue_atoms = op.get("num_residue_atoms", 24)
        kwargs = {
            "num_msa": op.get("num_msa"),
            "num_templates": op.get("num_templates"),
        }
        first_arg = (
            op["multiple"]
            if method_name == "pad_to_nearest_multiple"
            else op.get("size")
        )
        return getattr(model_input, method_name)(
            first_arg, num_residue_atoms, **kwargs
        )
    raise TypeError(f"Unsupported _pad_spec format: {op!r}")


class InputSpec:
    def __init__(self, *chains, templates=None, constraints=None):
        self.chains = list(chains)
        self.templates = templates or list()
        self.constraints = constraints or list()
        self.temporaries = list()
        self.precomputed_msas = list()
        self._pad_spec = None

    def _clone(self) -> "InputSpec":
        cloned = copy.copy(self)
        cloned.chains = [
            dict(c) if isinstance(c, dict) else c for c in self.chains
        ]
        cloned.templates = list(self.templates)
        cloned.constraints = list(self.constraints)
        cloned.temporaries = list(self.temporaries)
        cloned.precomputed_msas = list(self.precomputed_msas)
        return cloned

    def _with_pad_op(self, op: tuple) -> "InputSpec":
        cloned = self._clone()
        existing = getattr(self, "_pad_spec", None)
        if existing is None:
            cloned._pad_spec = op
        elif isinstance(existing, list):
            cloned._pad_spec = [*existing, op]
        else:
            cloned._pad_spec = [existing, op]
        return cloned

    def _apply_pad_spec(self, model_input):
        pad_spec = getattr(self, "_pad_spec", None)
        if pad_spec is None:
            return model_input
        if isinstance(pad_spec, list):
            for op in pad_spec:
                model_input = _apply_single_pad_op(model_input, op)
            return model_input
        return _apply_single_pad_op(model_input, pad_spec)

    def pad_to(
        self,
        size: int | None,
        num_residue_atoms: int = 24,
        *,
        num_msa: int | None = None,
        num_templates: int | None = None,
    ) -> "InputSpec":
        """Configure static bucket padding to apply when `to_input()` runs."""
        cloned = self._clone()
        cloned._pad_spec = dict(
            size=size,
            num_residue_atoms=num_residue_atoms,
            num_msa=num_msa,
            num_templates=num_templates,
        )
        return cloned

    def pad_to_nearest_multiple(
        self,
        multiple: int,
        num_residue_atoms: int = 24,
        *,
        num_msa: int | None = None,
        num_templates: int | None = None,
    ) -> "InputSpec":
        """Configure multiple-aligned bucket padding to apply when `to_input()` runs."""
        if multiple is not None and multiple <= 0:
            raise ValueError(f"multiple must be positive, got {multiple}.")
        cloned = self._clone()
        cloned._pad_spec = dict(
            size=None,
            num_residue_atoms=num_residue_atoms,
            num_msa=num_msa,
            num_templates=num_templates,
            _multiple=multiple,
        )
        return cloned

    def pad_atoms(self, num_residue_atoms: int = 24) -> "InputSpec":
        """Configure atom-only bucket padding (`num_residue_atoms * L`) on `to_input()`."""
        if num_residue_atoms <= 0:
            raise ValueError(
                f"num_residue_atoms must be positive, got {num_residue_atoms}."
            )
        return self._with_pad_op(("pad_atoms", (int(num_residue_atoms),), {}))

    def add_chain(self, *chains):
        _chains = []
        for c in chains:
            if isinstance(c, DesignData):
                subchains = c.split(c["chain_index"])
                for s in subchains:
                    _chains.append(
                        dict(
                            sequence=s.to_sequence_string(),
                            kind="protein",
                            use_msa=False,
                        )
                    )
            else:
                _chains.append(c)
        self.chains += _chains
        return self

    def add_polymer(
        self, *sequences, kind="protein", use_msa=False, num_repeats=1
    ):
        _chains = []
        for c in sequences:
            subchains = c.split(":")
            for s in subchains:
                _chains.append(
                    dict(
                        sequence=s,
                        kind=kind,
                        use_msa=use_msa,
                        num_repeats=num_repeats,
                    )
                )
        return self.add_chain(*_chains)

    def add_protein(self, *sequences, use_msa=False, num_repeats=1):
        return self.add_polymer(
            *sequences, kind="protein", use_msa=use_msa, num_repeats=num_repeats
        )

    def add_dna(self, *sequences, use_msa=False, num_repeats=1):
        return self.add_polymer(
            *sequences, kind="dna", use_msa=use_msa, num_repeats=num_repeats
        )

    def add_rna(self, *sequences, use_msa=False, num_repeats=1):
        return self.add_polymer(
            *sequences, kind="rna", use_msa=use_msa, num_repeats=num_repeats
        )

    def add_smiles(self, smiles, num_repeats=1):
        _chains = [
            dict(kind="ligand", smiles=smiles, num_repeats=num_repeats)
        ]
        return self.add_chain(*_chains)

    def add_ccd(self, ccd, num_repeats=1):
        _chains = [dict(kind="ligand", ccd=ccd, num_repeats=num_repeats)]
        return self.add_chain(*_chains)

    def add_template(self, path_or_object, to_chains=None):
        template_info = dict(
            template=path_or_object, template_chains=to_chains
        )
        if isinstance(path_or_object, str):
            if path_or_object.endswith((".pdb", ".pdb1")):
                template_info["pdb"] = path_or_object
            elif path_or_object.endswith(".cif"):
                template_info["cif"] = path_or_object
            else:
                raise NotImplementedError(
                    f"Invalid template file '{path_or_object}'. "
                    f"Has to be either '.pdb' or '.cif'."
                )
        elif isinstance(path_or_object, DesignData):
            tmpfile = PDBFile(data=path_or_object, temporary=True)
            self.temporaries.append(tmpfile)
            template_info["pdb"] = tmpfile.path
        else:
            raise NotImplementedError(
                "Input template has to be a path to a '.pdb' or '.cif' file."
            )
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

    def precomputed_msa(self, path, start_chain=0):
        offset = 0
        for chain in self.chains[:start_chain]:
            offset += len(chain["sequence"]) * chain.get("num_repeats", 1)
        self.precomputed_msas.append(dict(path=path, offset=offset))
        for chain in self.chains:
            chain["use_msa"] = False
        return self

    def add_bond(
        self, chain_1, residue_1, atom_1, chain_2, residue_2, atom_2
    ):
        self.constraints.append(
            dict(
                kind="bond",
                atom_1=dict(chain=chain_1, residue=residue_1, atom=atom_1),
                atom_2=dict(chain=chain_2, residue=residue_2, atom=atom_2),
            )
        )
        return self

    def add_contact(
        self,
        chain_1,
        residue_or_atom_1,
        chain_2,
        residue_or_atom_2,
        max_distance=6.0,
    ):
        self.constraints.append(
            dict(
                kind="contact",
                token_1=dict(chain=chain_1, target=residue_or_atom_1),
                token_2=dict(chain=chain_2, target=residue_or_atom_2),
                max_distance=max_distance,
            )
        )
        return self

    def add_pocket(self, binder, *contacts, max_distance=6.0):
        self.constraints.append(
            dict(
                kind="pocket",
                binder=binder,
                contacts=contacts,
                max_distance=max_distance,
            )
        )
        return self

    def add_constraint(self, constraint):
        self.constraints.append(constraint)
        return self

    def to_features(self, pad=True, cache="./params/model/") -> dict:
        raise NotImplementedError("Implement to_features per model.")

    def to_input(self, pad=True, cache="./params/model/"):
        raise NotImplementedError("Implement to_input per model.")

    def save_msa(self, path: str, cache="./params/model/"):
        raise NotImplementedError("Implement save_msa per model.")


_AA_SLICE = slice(2, 22)
_AA_UNK = 22
_RNA_SLICE = slice(23, 27)
_RNA_UNK = 27
_DNA_SLICE = slice(28, 32)
_DNA_UNK = 32


def _token_valid_slice(features: dict, start: int, end: int):
    if (
        "token_pad_mask" in features
        and features["token_pad_mask"] is not None
    ):
        return jnp.asarray(
            features["token_pad_mask"][0, start:end] > 0, dtype=jnp.float32
        )
    if (
        "token_attention_mask" in features
        and features["token_attention_mask"] is not None
    ):
        return jnp.asarray(
            features["token_attention_mask"][0, start:end], dtype=jnp.bool_
        ).astype(jnp.float32)
    return None


class ModelInput(eqx.Module):
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

    def copy(self):
        return type(self)(features={k: v for k, v in self.features.items()})

    def pad_to(
        self,
        size: int | None,
        num_residue_atoms: int = 24,
        *,
        num_msa: int | None = None,
        num_templates: int | None = None,
    ) -> "ModelInput":
        raise NotImplementedError("Implement pad_to per model.")

    def pad_to_nearest_multiple(
        self,
        multiple: int,
        num_residue_atoms: int = 24,
        *,
        num_msa: int | None = None,
        num_templates: int | None = None,
    ) -> "ModelInput":
        """Pad token count up to the nearest multiple of `multiple`."""
        if multiple is not None and multiple <= 0:
            raise ValueError(f"multiple must be positive, got {multiple}.")
        current_tokens = int(self.residue_index.shape[0])
        target_tokens = (
            None if multiple is None else align_up(current_tokens, multiple)
        )
        return self.pad_to(
            target_tokens,
            num_residue_atoms=num_residue_atoms,
            num_msa=num_msa,
            num_templates=num_templates,
        )

    def pad_atoms(self, num_residue_atoms: int = 24) -> "ModelInput":
        """Right-pad atom features to `num_residue_atoms * L` without padding tokens."""
        if num_residue_atoms <= 0:
            raise ValueError(
                f"num_residue_atoms must be positive, got {num_residue_atoms}."
            )
        return self.pad_to(
            int(self.residue_index.shape[0]),
            num_residue_atoms=int(num_residue_atoms),
        )

    def _set_res_type(
        self, sequence, start=0, seq_slice=_AA_SLICE, seq_count=20
    ):
        result = self.copy()
        end = start + sequence.shape[0]
        seq_vals = jnp.asarray(sequence[:, :seq_count], dtype=jnp.float32)
        tok_valid = _token_valid_slice(result.features, start, end)
        if tok_valid is not None:
            seq_vals = seq_vals * tok_valid[:, None]
        res_type = jnp.asarray(result.features["res_type"], dtype=jnp.float32)
        res_type = res_type.at[0, start:end].set(0.0)
        res_type = res_type.at[0, start:end, seq_slice].set(seq_vals)
        result.features["res_type"] = res_type
        return result

    def _set_profile(
        self, sequence, start=0, seq_slice=_AA_SLICE, seq_count=20
    ):
        result = self.copy()
        end = start + sequence.shape[0]
        if (
            "msa_mask" in result.features
            and result.features["msa_mask"] is not None
        ):
            msa_mask_slice = jnp.asarray(
                result.features["msa_mask"][0, :, start:end], dtype=jnp.float32
            )
            num_msa_1d = jnp.maximum(1.0, msa_mask_slice.sum(axis=0))
            num_msa = num_msa_1d[:, None]
            gap_frac = (num_msa_1d - 1.0) / num_msa_1d
        else:
            m_depth = float(result.features["msa"].shape[1])
            num_msa = m_depth
            gap_frac = jnp.full(
                (end - start,), (m_depth - 1.0) / m_depth, dtype=jnp.float32
            )
        seq_vals = (
            jnp.asarray(sequence[:, :seq_count], dtype=jnp.float32) / num_msa
        )
        tok_valid = _token_valid_slice(result.features, start, end)
        if tok_valid is not None:
            seq_vals = seq_vals * tok_valid[:, None]
            gap_frac = gap_frac * tok_valid
        profile = jnp.asarray(result.features["profile"], dtype=jnp.float32)
        profile = profile.at[0, start:end].set(0.0)
        profile = profile.at[0, start:end, seq_slice].set(seq_vals)
        profile = profile.at[0, start:end, 1].set(gap_frac)
        result.features["profile"] = profile
        return result

    def _set_msa(self, sequence, start=0, seq_slice=_AA_SLICE, seq_count=20):
        result = self.copy()
        end = start + sequence.shape[0]
        seq_vals = jnp.asarray(sequence[:, :seq_count], dtype=jnp.float32)
        tok_valid = _token_valid_slice(result.features, start, end)
        msa = jnp.asarray(result.features["msa"], dtype=jnp.float32)
        msa = msa.at[0, :, start:end].set(0.0)
        if tok_valid is not None:
            if (
                "msa_mask" in result.features
                and result.features["msa_mask"] is not None
            ):
                gap_mask = (
                    jnp.asarray(
                        result.features["msa_mask"][0, 1:, start:end] > 0,
                        dtype=jnp.float32,
                    )
                    * tok_valid[None, :]
                )
            else:
                gap_mask = jnp.broadcast_to(
                    tok_valid[None, :], (msa.shape[1] - 1, end - start)
                )
            msa = msa.at[0, 1:, start:end, 1].set(gap_mask)
            msa = msa.at[0, 0, start:end, seq_slice].set(
                seq_vals * tok_valid[:, None]
            )
        else:
            msa = msa.at[0, 1:, start:end, 1].set(1.0)
            msa = msa.at[0, 0, start:end, seq_slice].set(seq_vals)
        result.features["msa"] = msa
        return result

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
        if reset_profile:
            result = result._set_profile(
                sequence, start=start, seq_slice=seq_slice, seq_count=seq_count
            )
        if reset_msa:
            result = result._set_msa(
                sequence, start=start, seq_slice=seq_slice, seq_count=seq_count
            )
        return result

    def set_aa(self, sequence, start=0, reset_msa=True, reset_profile=True):
        return self._set_sequence(
            sequence,
            start=start,
            reset_msa=reset_msa,
            reset_profile=reset_profile,
        )

    def inherit_features(
        self, data: "ModelInput", features: Iterable[str] = None
    ) -> "ModelInput":
        result = self.copy()
        for name in features:
            result.features[name] = data.features[name]
        return result

    def save_features(
        self, path: str, features: Iterable[str]
    ) -> "ModelInput":
        features_dict = {name: self.features[name] for name in features}
        np.savez_compressed(path, **features_dict)
        return self

    def load_msa(
        self, path: str, start=0, msa_start=0, msa_end=None
    ) -> "ModelInput":
        msa_features = np.load(path)
        result = self.copy()
        msa_names = [
            "msa",
            "msa_mask",
            "msa_paired",
            "has_deletion",
            "deletion_value",
        ]
        msa = msa_features["msa"]
        msa_length = msa[:, :, msa_start:msa_end].shape[2]
        sequence_count = msa.shape[1]
        current_sequence_count = self.features["msa"].shape[1]
        for name in msa_names:
            if sequence_count > current_sequence_count:
                difference = sequence_count - current_sequence_count
                padding = jnp.zeros(
                    [1, difference] + list(result.features[name].shape[2:]),
                    dtype=result.features[name].dtype,
                )
                if name == "msa":
                    padding = padding.at[..., 1].set(1)
                if name == "msa_mask":
                    padding = padding.at[0].set(
                        msa_features["msa_mask"][0, -difference:, 0:1]
                    )
                result.features[name] = jnp.concatenate(
                    (result.features[name], padding), axis=1
                )
            result.features[name] = (
                result.features[name]
                .at[0, :sequence_count, start : start + msa_length]
                .set(msa_features[name][0, :, msa_start:msa_end])
            )
        if "msa_mask" in result.features:
            weights = jnp.asarray(
                result.features["msa_mask"], dtype=jnp.float32
            )
            denom = jnp.maximum(1.0, weights.sum(axis=1))
            result.features["profile"] = (
                result.features["msa"] * weights[..., None]
            ).sum(axis=1) / denom[..., None]
            result.features["deletion_mean"] = (
                result.features["deletion_value"] * weights
            ).sum(axis=1) / denom
        else:
            result.features["profile"] = result.features["msa"].mean(axis=1)
            result.features["deletion_mean"] = result.features[
                "deletion_value"
            ].mean(axis=1)
        return result

    def set_msa(self, msa, start=0):
        result = self.copy()
        end = start + msa.shape[1]
        msa = jnp.asarray(msa, dtype=jnp.float32)
        current_count = result.features["msa"].shape[1]
        if msa.shape[0] > current_count:
            difference = msa.shape[0] - current_count
            for name in (
                "msa",
                "msa_mask",
                "msa_paired",
                "has_deletion",
                "deletion_value",
            ):
                if name not in result.features or result.features[name] is None:
                    continue
                padding = jnp.zeros(
                    [1, difference] + list(result.features[name].shape[2:]),
                    dtype=result.features[name].dtype,
                )
                if name == "msa":
                    padding = padding.at[..., 1].set(1)
                elif name == "msa_mask":
                    tok_mask = result.features.get("token_pad_mask")
                    if tok_mask is not None:
                        padding = jnp.broadcast_to(
                            tok_mask[:, None, :].astype(padding.dtype),
                            padding.shape,
                        )
                    else:
                        padding = jnp.ones_like(padding)
                result.features[name] = jnp.concatenate(
                    [result.features[name], padding], axis=1
                )
        tok_valid = _token_valid_slice(result.features, start, end)
        msa_arr = jnp.asarray(result.features["msa"], dtype=jnp.float32)
        msa_arr = msa_arr.at[0, :, start:end].set(0.0)
        if tok_valid is not None:
            if (
                "msa_mask" in result.features
                and result.features["msa_mask"] is not None
            ):
                gap_mask = (
                    jnp.asarray(
                        result.features["msa_mask"][0, 1:, start:end] > 0,
                        dtype=jnp.float32,
                    )
                    * tok_valid[None, :]
                )
                result.features["msa_mask"] = (
                    jnp.asarray(result.features["msa_mask"])
                    .at[0, 0 : msa.shape[0], start:end]
                    .set(
                        jnp.broadcast_to(
                            tok_valid[None, :].astype(
                                result.features["msa_mask"].dtype
                            ),
                            (msa.shape[0], end - start),
                        )
                    )
                )
            else:
                gap_mask = jnp.broadcast_to(
                    tok_valid[None, :], (msa_arr.shape[1] - 1, end - start)
                )
            msa_in = msa * tok_valid[None, :, None]
            msa_arr = msa_arr.at[0, 1:, start:end, 1].set(gap_mask)
            msa_arr = msa_arr.at[0, 0 : msa.shape[0], start:end, :].set(msa_in)
            profile_update = msa_in.mean(axis=0)
        else:
            msa_arr = msa_arr.at[0, 1:, start:end, 1].set(1.0)
            msa_arr = msa_arr.at[0, 0 : msa.shape[0], start:end, :].set(msa)
            profile_update = msa.mean(axis=0)
        result.features["msa"] = msa_arr
        profile = jnp.asarray(result.features["profile"], dtype=jnp.float32)
        result.features["profile"] = profile.at[0, start:end].set(
            profile_update
        )
        return result

    def set_aa_msa(self, msa, start=0):
        if msa.shape[-1] == 20:
            msa_feat = self.features.get("msa")
            msa_size = (
                msa_feat.shape[-1]
                if msa_feat is not None
                else self.features["res_type"].shape[-1]
            )
            seq_start = 2
            msa = jnp.pad(
                msa, ((0, 0), (0, 0), (seq_start, msa_size - 20 - seq_start))
            )
        aa = msa[0, ..., 2:22]
        return self.set_aa(aa, start=start).set_msa(msa, start=start)

    def set_aa_msa_random(self, key, msa, start=0):
        index = jax.random.permutation(key, msa.shape[0])
        msa = msa[index]
        return self.set_aa_msa(msa, start=start)

    def set_rna(self, sequence, start=0):
        return self._set_sequence(
            sequence, start=start, seq_slice=_RNA_SLICE, seq_count=4
        )

    def set_dna(self, sequence, start=0):
        return self._set_sequence(
            sequence, start=start, seq_slice=_DNA_SLICE, seq_count=4
        )
