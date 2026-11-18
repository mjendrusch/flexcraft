# adapted from mosaic ((c) 2025 escalante under MIT license)
import os
import shutil
import warnings
from dataclasses import asdict, replace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, List, Literal, Tuple

import boltz.data.const as const
import boltz.main as boltz_main
import equinox as eqx
import jax
import numpy as np
import torch
from boltz.data.types import Coords, Interface, Record, StructureV2
from boltz.data.write.mmcif import to_mmcif
from boltz.data.write.pdb import to_pdb
from jax import numpy as jnp
from jaxtyping import PyTree

from flexcraft.structure.common._data import (
    InputSpec,
    ModelInput,
    _AA_SLICE,
    _AA_UNK,
    _DNA_SLICE,
    _DNA_UNK,
    _RNA_SLICE,
    _RNA_UNK,
)
from flexcraft.structure.common._utils import (
    align_up,
    backbone_to_template_geometry,
    compute_pseudo_cb,
    pad_features_by_schema,
)


def _ensemble_ref_axes(current_tokens: int):
    def _axes(arr):
        if arr.ndim >= 2 and arr.shape[1] == current_tokens:
            return (None, "token") + (None,) * (arr.ndim - 2)
        return (None,) * arr.ndim

    return _axes


def get_boltz_feature_schema(current_tokens: int | None = None) -> dict[str, Any]:
    """Return explicit per-feature named-axis map for Boltz2 feature dicts."""
    return {
        "token_index": (None, "token"),
        "residue_index": (None, "token"),
        "asym_id": (None, "token"),
        "entity_id": (None, "token"),
        "sym_id": (None, "token"),
        "mol_type": (None, "token"),
        "res_type": (None, "token", None),
        "disto_center": (None, "token", None),
        "disto_coords_ensemble": lambda arr: (
            (None, None, "token", None)
            if arr.ndim == 4
            else (None, "token", None)
        ),
        "disto_target": lambda arr: (
            (None, "token", "token", None, None)
            if arr.ndim == 5
            else (None, "token", "token", None)
        ),
        "token_pad_mask": (None, "token"),
        "token_resolved_mask": (None, "token"),
        "token_disto_mask": (None, "token"),
        "pocket_feature": lambda arr: (
            (None, "token", None) if arr.ndim == 3 else (None, "token")
        ),
        "method_feature": (None, "token"),
        "modified": (None, "token"),
        "cyclic_period": (None, "token"),
        "affinity_token_mask": (None, "token"),
        "profile": (None, "token", None),
        "deletion_mean": (None, "token"),
        "ensemble_ref_idxs": (
            _ensemble_ref_axes(current_tokens)
            if current_tokens is not None
            else (None, "token")
        ),
        "frames_idx": lambda arr: (
            (None, None, "token", None)
            if arr.ndim == 4
            else (None, "token", None)
        ),
        "frame_resolved_mask": lambda arr: (
            (None, None, "token") if arr.ndim == 3 else (None, "token")
        ),
        "token_bonds": lambda arr: (
            (None, "token", "token", None)
            if arr.ndim == 4
            else (None, "token", "token")
        ),
        "type_bonds": (None, "token", "token"),
        "contact_conditioning": (None, "token", "token", None),
        "contact_threshold": (None, "token", "token"),
        "pair_mask": (None, "token", "token"),
        "atom_pad_mask": (None, "atom"),
        "atom_resolved_mask": lambda arr: (
            (None, None, "atom") if arr.ndim == 3 else (None, "atom")
        ),
        "atom_to_token": (None, "atom", "token"),
        "token_to_rep_atom": (None, "token", "atom"),
        "token_to_center_atom": (None, "token", "atom"),
        "r_set_to_rep_atom": (None, "token", "atom"),
        "ref_pos": (None, "atom", None),
        "ref_space_uid": (None, "atom"),
        "ref_charge": (None, "atom"),
        "ref_element": (None, "atom", None),
        "ref_atom_name_chars": (None, "atom", None, None),
        "ref_chirality": (None, "atom"),
        "atom_backbone_feat": (None, "atom", None),
        "coords": lambda arr: (
            (None, None, "atom", None)
            if arr.ndim == 4
            else (None, "atom", None)
        ),
        "bfactor": (None, "atom"),
        "plddt": (None, "atom"),
        "msa": (None, "msa", "token", None),
        "msa_mask": (None, "msa", "token"),
        "msa_paired": (None, "msa", "token"),
        "has_deletion": (None, "msa", "token"),
        "deletion_value": (None, "msa", "token"),
        "template_frame_rot": (None, "template", "token", None, None),
        "template_frame_t": (None, "template", "token", None),
        "template_ca": (None, "template", "token", None),
        "template_cb": (None, "template", "token", None),
        "template_mask": (None, "template", "token"),
        "template_mask_cb": (None, "template", "token"),
        "template_mask_frame": (None, "template", "token"),
        "template_restype": (None, "template", "token", None),
        "visibility_ids": (None, "template", "token"),
        "query_to_template": (None, "template", "token"),
        "template_force": (None, "template"),
        "template_force_threshold": (None, "template"),
    }


BOLTZ_PAD_VALUES = {
    "contact_conditioning": lambda arr: (
        jnp.zeros((arr.shape[-1],), dtype=arr.dtype).at[0].set(1.0)
        if isinstance(arr, jax.Array)
        else np.array(
            [1.0] + [0.0] * (arr.shape[-1] - 1), dtype=arr.dtype
        )
    ),
    "asym_id": -1,
    "entity_id": -1,
    "ref_space_uid": -1,
    "visibility_ids": -1.0,
    "ensemble_ref_idxs": -1.0,
    "token_pad_mask": 0.0,
    "atom_pad_mask": 0.0,
    "msa_mask": 0.0,
    "template_mask": 0.0,
    "template_mask_cb": 0.0,
    "template_mask_frame": 0.0,
}


def align_boltz_atoms_w32(features: dict) -> dict:
    """Ensure Boltz atom-axis features have `N_atoms % 32 == 0`."""
    if "atom_pad_mask" not in features:
        return features
    current_atoms = int(features["atom_pad_mask"].shape[1])
    target_atoms = align_up(max(current_atoms, 32), 32)
    if target_atoms == current_atoms:
        return features
    current_tokens = (
        int(features["res_type"].shape[1]) if "res_type" in features else None
    )
    return pad_features_by_schema(
        features,
        axis_targets={"atom": target_atoms},
        schema=get_boltz_feature_schema(current_tokens),
        pad_values=BOLTZ_PAD_VALUES,
    )


def chain_yaml(
    chain_order: List[str],
    num_repeats: int = None,
    sequence: str = None,
    ccd: str = None,
    smiles: str = None,
    kind: str = "protein",
    use_msa: bool = False,
    **kwargs,
) -> str:
    if num_repeats is None:
        num_repeats = 1
    chain_names = [chain_order.pop(0) for _ in range(num_repeats)]
    chain_name = ", ".join(chain_names)
    if ccd is not None:
        result = f"""  - ligand:
        id: [{chain_name}]
        ccd: {ccd}"""
        use_msa = False
    elif smiles is not None:
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


def _make_features(
    chains: list, templates: list, constraints: list, cache="./params/boltz/"
):
    chain_order = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    yaml = "\n".join(
        [_prefix()] + [chain_yaml(chain_order, **c) for c in chains]
    )
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


class JoltzSpec(InputSpec):
    def to_features(self, pad=True, cache="./params/boltz/"):
        chains = [
            {**c, "use_msa": False} if self.precomputed_msas else dict(c)
            for c in self.chains
        ]
        features, writer_spec = _make_features(
            chains, self.templates, self.constraints, cache=cache
        )
        writer_features = {
            k: torch.tensor(np.array(v))
            for k, v in features.items()
            if k != "record"
        }
        writer_features["record"] = writer_spec["features_dict"]["record"]
        writer_spec["features_dict"] = writer_features
        features = jax.tree.map(jnp.array, features)
        features = align_boltz_atoms_w32(features)
        return features, writer_spec

    def to_input(
        self, pad=True, cache="./params/boltz/"
    ) -> Tuple["JoltzInput", "Joltz2Writer"]:
        features, writer_spec = self.to_features(pad=pad, cache=cache)
        joltz_input = JoltzInput(features=features)
        if self.precomputed_msas:
            for msa in self.precomputed_msas:
                joltz_input = joltz_input.load_msa(msa["path"], msa["offset"])
        joltz_input = self._apply_pad_spec(joltz_input)
        joltz_writer = Joltz2Writer(**writer_spec)
        return joltz_input, joltz_writer

    def save_msa(self, path: str, cache="./params/boltz/"):
        joltz_input, _ = self.to_input(pad=False, cache=cache)
        joltz_input.save_msa(path)


class JoltzInput(ModelInput):
    features: dict

    def __init__(self, features: dict):
        if (
            isinstance(features, dict)
            and "atom_pad_mask" in features
            and features["atom_pad_mask"].shape[1] % 32 != 0
        ):
            features = align_boltz_atoms_w32(features)
        self.features = features

    def pad_to(
        self,
        size: int | None,
        num_residue_atoms: int = 24,
        *,
        num_msa: int | None = None,
        num_templates: int | None = None,
    ) -> "JoltzInput":
        """Right-pad token, atom (`W=32`), MSA, and template axes monotonically."""
        current_tokens = int(self.features["res_type"].shape[1])
        current_atoms = int(self.features["atom_pad_mask"].shape[1])

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

        axis_targets = {"token": target_tokens, "atom": target_atoms}

        if num_msa is not None and "msa" in self.features:
            current_msa = int(self.features["msa"].shape[1])
            if num_msa < current_msa:
                warnings.warn(
                    f"Requested num_msa {num_msa} < current MSA depth "
                    f"{current_msa}; keeping {current_msa}.",
                    UserWarning,
                    stacklevel=2,
                )
            else:
                axis_targets["msa"] = int(num_msa)

        if num_templates is not None and "template_mask" in self.features:
            current_templates = int(self.features["template_mask"].shape[1])
            if num_templates < current_templates:
                warnings.warn(
                    f"Requested num_templates {num_templates} < current "
                    f"template count {current_templates}; keeping {current_templates}.",
                    UserWarning,
                    stacklevel=2,
                )
            else:
                axis_targets["template"] = int(num_templates)

        padded_features = pad_features_by_schema(
            self.features,
            axis_targets=axis_targets,
            schema=get_boltz_feature_schema(current_tokens),
            pad_values=BOLTZ_PAD_VALUES,
        )
        return JoltzInput(features=padded_features)

    def inherit_msa(
        self,
        data: "JoltzInput",
        start: int = 0,
        msa_start: int = 0,
        msa_end: int | None = None,
    ) -> "JoltzInput":
        """Return a copy inheriting MSA features from `data`."""
        msa_keys = (
            "msa",
            "msa_mask",
            "profile",
            "deletion_mean",
            "has_deletion",
            "deletion_value",
            "msa_paired",
        )
        src_msa = data.features["msa"]
        dst_msa = self.features["msa"]
        if (
            start == 0
            and msa_start == 0
            and msa_end is None
            and src_msa.shape[2] == dst_msa.shape[2]
            and src_msa.shape[1] >= dst_msa.shape[1]
        ):
            return self.inherit_features(data, features=msa_keys)

        result = self.copy()
        msa_slice = src_msa[:, :, msa_start:msa_end]
        msa_length = int(msa_slice.shape[2])
        sequence_count = int(msa_slice.shape[1])
        current_sequence_count = int(dst_msa.shape[1])
        tok_mask = jnp.asarray(
            result.features["token_pad_mask"], dtype=jnp.float32
        )
        row_names = (
            "msa",
            "msa_mask",
            "msa_paired",
            "has_deletion",
            "deletion_value",
        )
        for name in row_names:
            arr = jnp.asarray(result.features[name])
            if sequence_count > current_sequence_count:
                difference = sequence_count - current_sequence_count
                padding = jnp.zeros(
                    [1, difference] + list(arr.shape[2:]),
                    dtype=arr.dtype,
                )
                if name == "msa":
                    padding = padding.at[..., 1].set(
                        tok_mask[:, None, :].astype(arr.dtype)
                    )
                elif name == "msa_mask":
                    row_active = (
                        jnp.asarray(
                            data.features["msa_mask"][
                                0, -difference:, msa_start:msa_end
                            ]
                        )
                        > 0
                    ).any(axis=-1)
                    padding = (
                        row_active[None, :, None] & (tok_mask[:, None, :] > 0)
                    ).astype(arr.dtype)
                arr = jnp.concatenate([arr, padding], axis=1)
            src_slice = jnp.asarray(
                data.features[name][0, :, msa_start:msa_end], dtype=arr.dtype
            )
            arr = arr.at[0, :sequence_count, start : start + msa_length].set(
                src_slice
            )
            result.features[name] = arr

        weights = jnp.asarray(result.features["msa_mask"], dtype=jnp.float32)
        denom = jnp.maximum(1.0, weights.sum(axis=1))
        profile = (
            jnp.asarray(result.features["msa"], dtype=jnp.float32)
            * weights[..., None]
        ).sum(axis=1) / denom[..., None]
        deletion_mean = (
            jnp.asarray(result.features["deletion_value"], dtype=jnp.float32)
            * weights
        ).sum(axis=1) / denom
        result.features["profile"] = profile * tok_mask[..., None]
        result.features["deletion_mean"] = deletion_mean * tok_mask
        return result

    def save_msa(self, path: str) -> "JoltzInput":
        return self.save_features(
            path,
            features=(
                "msa",
                "msa_mask",
                "profile",
                "has_deletion",
                "deletion_value",
                "deletion_mean",
                "msa_paired",
            ),
        )

    def set_template(
        self,
        coords=None,
        start=0,
        template=None,
        mask=None,
        restype=None,
        template_id=0,
    ) -> "JoltzInput":
        """Set template geometry directly from `(L, >=4, 3)` coordinates."""
        result = self.copy()
        if template is not None:
            result.features.update(template)
            result.features["template_restype"] = (
                result.features["template_restype"]
                .at[0]
                .set(result.features["res_type"][0])
            )
            return result
        length = coords.shape[0]
        sl = slice(start, start + length)
        tid = template_id

        rot, trans = backbone_to_template_geometry(coords)
        cb = compute_pseudo_cb(coords)

        finite = jnp.isfinite(coords[:, :3]).all(axis=(1, 2))
        valid = finite if mask is None else (mask.astype(bool) & finite)
        valid = valid.astype(bool)
        keep = valid[:, None]
        rot = jnp.where(keep[..., None], rot, 0.0)
        trans = jnp.where(keep, trans, 0.0)
        cb = jnp.where(keep, cb, 0.0)

        if restype is None:
            restype = result.features["res_type"][0, sl]
        elif restype.ndim == 1:
            restype = jnp.asarray(restype)
            restype = jnp.where(restype == 20, _AA_UNK, restype + 2)
            restype = jax.nn.one_hot(restype, 33)
        elif restype.shape[-1] == 20:
            restype = jnp.pad(restype, ((0, 0), (2, 11)))
        elif restype.shape[-1] != 33:
            raise ValueError(
                f"restype must be shape (L,), (L, 20), or (L, 33), got {restype.shape}"
            )
        restype = restype.astype(jnp.float32)

        asym = result.features["asym_id"][0]
        valid_tok = result.features["token_pad_mask"][0] > 0
        in_region_chain = valid_tok & (
            asym[:, None] == asym[sl][None, :]
        ).any(axis=1)
        vis = jnp.where(
            ~valid_tok,
            -1.0,
            jnp.where(in_region_chain, float(tid), (-1.0 - asym)),
        ).astype(jnp.float32)

        num_tokens = result.features["template_mask"].shape[2]
        assert (
            start + length <= num_tokens
        ), f"template region {start}:{start + length} exceeds {num_tokens} tokens"

        writes = [
            ("template_frame_rot", rot, (0, tid, sl)),
            ("template_frame_t", trans, (0, tid, sl)),
            ("template_ca", trans, (0, tid, sl)),
            ("template_cb", cb, (0, tid, sl)),
            ("template_mask", valid, (0, tid, sl)),
            ("template_mask_cb", valid, (0, tid, sl)),
            ("template_mask_frame", valid, (0, tid, sl)),
            ("template_restype", restype, (0, tid, sl)),
            ("visibility_ids", vis, (0, tid)),
        ]
        for k, value, idx in writes:
            result.features[k] = (
                jnp.asarray(result.features[k], jnp.float32).at[idx].set(value)
            )

        return result

    @property
    def template(self):
        return {
            k: self.features[k]
            for k in (
                "template_frame_rot",
                "template_frame_t",
                "template_ca",
                "template_cb",
                "template_mask",
                "template_mask_cb",
                "template_mask_frame",
                "visibility_ids",
            )
        }


def substitute_aa(features, aa_one_hot):
    features["res_type"] = jnp.array(features["res_type"]).astype(jnp.float32)
    features["msa"] = jnp.array(features["msa"]).astype(jnp.float32)
    features["profile"] = jnp.array(features["profile"]).astype(jnp.float32)
    assert len(aa_one_hot.shape) == 2
    assert aa_one_hot.shape[1] == 20
    binder_len = aa_one_hot.shape[0]

    zero_padded_sequence = jnp.pad(aa_one_hot, ((0, 0), (2, 11)))
    if "msa_mask" in features:
        n_msa = jnp.maximum(
            1.0,
            jnp.asarray(features["msa_mask"][0, :, :binder_len], jnp.float32).sum(
                axis=0
            )[:, None],
        )
    else:
        n_msa = float(features["msa"].shape[1])

    binder_profile = jnp.zeros_like(features["profile"][0, :binder_len])
    binder_profile = (
        binder_profile.at[:binder_len].set(zero_padded_sequence) / n_msa
    )
    binder_profile = binder_profile.at[:, 1].set(
        jnp.squeeze((n_msa - 1.0) / n_msa)
    )

    return features | {
        "res_type": features["res_type"]
        .at[0, :binder_len, :]
        .set(zero_padded_sequence),
        "msa": features["msa"]
        .at[0, 0, :binder_len, :]
        .set(zero_padded_sequence),
        "profile": features["profile"].at[0, :binder_len].set(binder_profile),
    }


def build_chain_template_yaml(chain_order: List[str], chains: list):
    num_templates = 0
    template_yaml = """
        
templates:"""
    for chain in chains:
        num_repeats = chain.get("num_repeats", 1)
        if "template_file" not in chain:
            for _ in range(num_repeats):
                chain_order.pop(0)
            continue
        if chain["template_file"] is None or not os.path.isfile(
            chain["template_file"]
        ):
            for _ in range(num_repeats):
                chain_order.pop(0)
            continue
        num_templates += 1
        template_yaml += f"""
  - cif: {chain['template_file']}
    chain_id: [{', '.join([chain_order.pop(0) for _ in range(num_repeats)])}]
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
        if (
            "template_chains" in template
            and template["template_chains"] is not None
        ):
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
        elif constraint["kind"] == "pocket":
            contacts = ", ".join(
                [
                    f"[{chain}, {res_atm}]"
                    for chain, res_atm in constraint["contacts"]
                ]
            )
            constraint_yaml += f"""
    - pocket:
        binder: {constraint['binder']}
        contacts: [{contacts}]
        max_distance: {constraint['max_distance']}"""
        else:
            raise NotImplementedError(
                f"Unknown constraint type {constraint['kind']}."
            )
    constraint_yaml += "\n"
    return num_constraints, constraint_yaml


def _features_from_yaml(
    input_yaml_str: str,
    cache=Path("./params/boltz/").expanduser(),
) -> PyTree:
    if not isinstance(cache, Path):
        cache = Path(cache).expanduser()
    out_dir_handle = TemporaryDirectory()
    out_dir = Path(out_dir_handle.name)
    input_data_path = out_dir / "input.yaml"
    input_data_path.write_text(input_yaml_str)
    data = boltz_main.check_inputs(input_data_path)
    ccd_path = cache / "ccd.pkl"
    mol_dir = cache / "mols"
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
            (processed_dir / "mols")
            if (processed_dir / "mols").exists()
            else None
        ),
    )
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

    features_dict = list(data_module.predict_dataloader())[0]
    features = {
        k: np.array(v) for k, v in features_dict.items() if k != "record"
    }
    features["msa"] = jax.nn.one_hot(features["msa"], const.num_tokens)
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
        if output_format not in ["pdb", "mmcif"]:
            raise ValueError(f"Invalid output format: {output_format}")

        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_format = output_format
        self.failed = 0
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write(self, prediction: dict[str, torch.Tensor], batch) -> None:
        """Write the predictions to disk."""
        if prediction["exception"]:
            self.failed += 1
            return

        records: list[Record] = batch["record"]
        coords = prediction["coords"].unsqueeze(0)
        pad_masks = prediction["masks"]
        idx_to_rank = {i: i for i in range(len(records))}

        for record, coord, pad_mask in zip(records, coords, pad_masks):
            path = self.data_dir / f"{record.id}.npz"
            structure: StructureV2 = StructureV2.load(path)

            chain_map = {}
            for i, mask in enumerate(structure.mask):
                if mask:
                    chain_map[len(chain_map)] = i

            structure = structure.remove_invalid_chains()

            for model_idx in range(coord.shape[0]):
                model_coord = coord[model_idx]
                coord_unpad = model_coord[pad_mask.bool()].cpu().numpy()

                atoms = structure.atoms
                atoms["coords"] = coord_unpad
                atoms["is_present"] = True
                coord_unpad = np.array(
                    [(x,) for x in coord_unpad], dtype=Coords
                )

                residues = structure.residues
                residues["is_present"] = True

                interfaces = np.array([], dtype=Interface)
                new_structure: StructureV2 = replace(
                    structure,
                    atoms=atoms,
                    residues=residues,
                    interfaces=interfaces,
                    coords=coord_unpad,
                )

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

                struct_dir = self.output_dir / record.id
                struct_dir.mkdir(exist_ok=True)

                plddts = None
                if "plddt" in prediction:
                    plddts = prediction["plddt"][model_idx]

                outname = f"{record.id}_model_{idx_to_rank[model_idx]}"
                if self.output_format == "pdb":
                    out_path = struct_dir / f"{outname}.pdb"
                    with out_path.open("w") as f:
                        f.write(
                            to_pdb(new_structure, plddts=plddts, boltz2=True)
                        )
                elif self.output_format == "mmcif":
                    out_path = struct_dir / f"{outname}.cif"
                    with out_path.open("w") as f:
                        f.write(
                            to_mmcif(new_structure, plddts=plddts, boltz2=True)
                        )
                else:
                    out_path = struct_dir / f"{outname}.npz"
                    np.savez_compressed(out_path, **asdict(new_structure))


class Joltz2Writer:
    writer: _BoltzWriter
    atom_pad_mask: torch.Tensor
    n_valid_atoms: int
    n_valid_tokens: int
    record: Any
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
            output_format="mmcif",
        )
        raw_atom_mask = features_dict["atom_pad_mask"]
        self.atom_pad_mask = raw_atom_mask.unsqueeze(0)
        self.n_valid_atoms = int((raw_atom_mask > 0).sum().item())
        raw_tok_mask = features_dict.get("token_pad_mask")
        self.n_valid_tokens = (
            int((raw_tok_mask > 0).sum().item())
            if raw_tok_mask is not None
            else int(features_dict["residue_index"].shape[-1])
        )
        self.record = features_dict["record"][0]
        self.out_dir = output_dir
        self.temp_dir_handle = temp_dir_handle

    def save_structure(
        self, path, sample_atom_coords, plddt=None, out_format="mmcif"
    ):
        self.writer.output_format = out_format
        confidence = torch.ones(1)

        coords_np = np.asarray(sample_atom_coords)
        if coords_np.ndim == 2:
            coords = torch.tensor(coords_np).unsqueeze(0).unsqueeze(0)
        elif coords_np.ndim == 3:
            coords = torch.tensor(coords_np).unsqueeze(0)
        else:
            coords = torch.tensor(coords_np)

        num_in_atoms = coords.shape[-2]
        if num_in_atoms == self.atom_pad_mask.shape[-1]:
            masks = self.atom_pad_mask
        else:
            masks = torch.zeros((1, 1, num_in_atoms), dtype=torch.bool)
            masks[:, :, : self.n_valid_atoms] = True

        pred_dict = {
            "exception": False,
            "coords": coords,
            "masks": masks,
            "confidence_score": confidence,
        }
        if plddt is not None:
            plddt_np = np.asarray(plddt)
            if plddt_np.ndim == 3:
                plddt_np = plddt_np.squeeze(0)
            elif plddt_np.ndim == 1:
                plddt_np = plddt_np[None, :]
            plddt_np = plddt_np[:, : self.n_valid_tokens]
            pred_dict["plddt"] = torch.tensor(plddt_np)

        self.writer.write(pred_dict, {"record": [self.record]})

        suffix = "cif" if out_format == "mmcif" else "pdb"
        in_path = str(
            (Path(self.out_dir) / self.record.id)
            / f"{self.record.id}_model_0.{suffix}"
        )
        shutil.copy(in_path, path)

    def save_pdb(self, path, sample_atom_coords, plddt=None):
        self.save_structure(
            path, sample_atom_coords, plddt=plddt, out_format="pdb"
        )

    def save_cif(self, path, sample_atom_coords, plddt=None):
        self.save_structure(
            path, sample_atom_coords, plddt=plddt, out_format="mmcif"
        )
