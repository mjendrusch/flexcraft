"""Unit tests for padding contracts and unified Result metrics across AF2, Protenix, Joltz2, ESMFold2, and AF3Like."""

import inspect
from unittest.mock import patch

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from flexcraft.structure.af._af3_like_wrapper import AF2Spec
from flexcraft.structure.af._data import AFInput, AFResult
from flexcraft.structure.af3_like.model import AF3LikeSpec, _any_spec_to_model_spec
from flexcraft.structure.boltz._data import JoltzInput, JoltzSpec
from flexcraft.structure.boltz._result import JoltzResult
from flexcraft.structure.common._data import InputSpec, ModelInput
from flexcraft.structure.esmfold2._data import (
    ESM_NON_PROTEIN_ID,
    ESM_PAD_ID,
    ESMFold2Input,
    ESMFold2Spec,
    _BOLTZ_AA_TO_ESM_ID,
)
from flexcraft.structure.esmfold2._result import ESMFold2Confidence, ESMFold2Result
from flexcraft.structure.metrics import pae_bin_centers
from flexcraft.structure.protenix._data import ProtenixInput, ProtenixSpec
from flexcraft.structure.protenix._result import ProtenixResult


class _SyntheticConfidence(eqx.Module):
    plddt: jax.Array
    plddt_logits: jax.Array
    pae: jax.Array
    pae_logits: jax.Array


def _make_synthetic_protenix_features(
    num_tokens: int = 10,
    atoms_per_token: int = 8,
    num_msa: int = 2,
    num_templates: int = 1,
) -> dict:
    """Create a realistic unbatched Protenix feature dictionary."""
    num_atoms = num_tokens * atoms_per_token
    atom_to_token_idx = jnp.repeat(jnp.arange(num_tokens, dtype=jnp.int32), atoms_per_token)
    rep_idx = jnp.arange(num_tokens, dtype=jnp.int32) * atoms_per_token + 1
    disto_mask = jnp.zeros((num_atoms,), dtype=jnp.int32).at[rep_idx].set(1)
    restype = jax.nn.one_hot(jnp.arange(num_tokens) % 20, 32, dtype=jnp.float32)
    msa = jnp.zeros((num_msa, num_tokens), dtype=jnp.int32)
    return {
        "token_index": jnp.arange(num_tokens, dtype=jnp.int32),
        "residue_index": jnp.arange(1, num_tokens + 1, dtype=jnp.int32),
        "asym_id": jnp.array(
            [0] * (num_tokens // 2) + [1] * (num_tokens - num_tokens // 2),
            dtype=jnp.int32,
        ),
        "entity_id": jnp.zeros((num_tokens,), dtype=jnp.int32),
        "sym_id": jnp.zeros((num_tokens,), dtype=jnp.int32),
        "restype": restype,
        "profile": restype,
        "deletion_mean": jnp.zeros((num_tokens,), dtype=jnp.float32),
        "atom_rep_atom_idx": rep_idx,
        "distogram_rep_atom_mask": disto_mask,
        "token_bonds": jnp.zeros((num_tokens, num_tokens), dtype=jnp.float32),
        "atom_to_token_idx": atom_to_token_idx,
        "atom_to_tokatom_idx": jnp.tile(jnp.arange(atoms_per_token, dtype=jnp.int32), num_tokens),
        "is_protein": jnp.ones((num_atoms,), dtype=jnp.int32),
        "is_dna": jnp.zeros((num_atoms,), dtype=jnp.int32),
        "is_rna": jnp.zeros((num_atoms,), dtype=jnp.int32),
        "is_ligand": jnp.zeros((num_atoms,), dtype=jnp.int32),
        "ref_pos": jnp.zeros((num_atoms, 3), dtype=jnp.float32),
        "ref_mask": jnp.ones((num_atoms,), dtype=jnp.int32),
        "ref_space_uid": atom_to_token_idx,
        "ref_charge": jnp.zeros((num_atoms,), dtype=jnp.int32),
        "ref_element": jnp.zeros((num_atoms, 128), dtype=jnp.int32),
        "ref_atom_name_chars": jnp.zeros((num_atoms, 4, 64), dtype=jnp.int32),
        "msa": msa,
        "has_deletion": jnp.zeros((num_msa, num_tokens), dtype=jnp.float32),
        "deletion_value": jnp.zeros((num_msa, num_tokens), dtype=jnp.float32),
        "template_aatype": jnp.full((num_templates, num_tokens), 20, dtype=jnp.int32),
        "template_distogram": jnp.zeros((num_templates, num_tokens, num_tokens, 39), dtype=jnp.float32),
        "template_pseudo_beta_mask": jnp.zeros((num_templates, num_tokens, num_tokens), dtype=jnp.float32),
        "template_unit_vector": jnp.zeros((num_templates, num_tokens, num_tokens, 3), dtype=jnp.float32),
        "template_backbone_frame_mask": jnp.zeros((num_templates, num_tokens, num_tokens), dtype=jnp.float32),
    }


def _make_synthetic_joltz_features(
    num_tokens: int = 10,
    atoms_per_token: int = 5,
    num_msa: int = 2,
    num_templates: int = 1,
) -> dict:
    """Create a realistic batched (1, ...) Boltz2/Joltz feature dictionary."""
    num_atoms = num_tokens * atoms_per_token
    atom_to_token_idx = jnp.repeat(jnp.arange(num_tokens, dtype=jnp.int32), atoms_per_token)
    atom_to_token = jax.nn.one_hot(atom_to_token_idx, num_tokens, dtype=jnp.float32)[None]
    res_type = jax.nn.one_hot(2 + (jnp.arange(num_tokens) % 20), 33, dtype=jnp.float32)[None]
    msa = jnp.broadcast_to(res_type[:, None, :, :], (1, num_msa, num_tokens, 33))
    return {
        "token_index": jnp.arange(num_tokens, dtype=jnp.int32)[None],
        "residue_index": jnp.arange(num_tokens, dtype=jnp.int32)[None],
        "asym_id": jnp.array(
            [[0] * (num_tokens // 2) + [1] * (num_tokens - num_tokens // 2)],
            dtype=jnp.int32,
        ),
        "entity_id": jnp.zeros((1, num_tokens), dtype=jnp.int32),
        "sym_id": jnp.zeros((1, num_tokens), dtype=jnp.int32),
        "mol_type": jnp.zeros((1, num_tokens), dtype=jnp.int32),
        "res_type": res_type,
        "profile": res_type,
        "deletion_mean": jnp.zeros((1, num_tokens), dtype=jnp.float32),
        "token_pad_mask": jnp.ones((1, num_tokens), dtype=jnp.float32),
        "token_resolved_mask": jnp.ones((1, num_tokens), dtype=jnp.float32),
        "token_disto_mask": jnp.ones((1, num_tokens), dtype=jnp.float32),
        "token_bonds": jnp.zeros((1, num_tokens, num_tokens, 1), dtype=jnp.float32),
        "type_bonds": jnp.zeros((1, num_tokens, num_tokens), dtype=jnp.int32),
        "contact_conditioning": jnp.zeros((1, num_tokens, num_tokens, 4), dtype=jnp.float32)
        .at[:, :, :, 0]
        .set(1.0),
        "contact_threshold": jnp.zeros((1, num_tokens, num_tokens), dtype=jnp.float32),
        "atom_pad_mask": jnp.ones((1, num_atoms), dtype=jnp.float32),
        "atom_resolved_mask": jnp.ones((1, num_atoms), dtype=jnp.float32),
        "atom_to_token": atom_to_token,
        "ref_pos": jnp.zeros((1, num_atoms, 3), dtype=jnp.float32),
        "ref_space_uid": atom_to_token_idx[None],
        "ref_charge": jnp.zeros((1, num_atoms), dtype=jnp.float32),
        "ref_element": jnp.zeros((1, num_atoms, 128), dtype=jnp.float32),
        "ref_atom_name_chars": jnp.zeros((1, num_atoms, 4, 64), dtype=jnp.float32),
        "ref_chirality": jnp.zeros((1, num_atoms), dtype=jnp.int32),
        "atom_backbone_feat": jnp.zeros((1, num_atoms, 4), dtype=jnp.float32),
        "coords": jnp.zeros((1, 1, num_atoms, 3), dtype=jnp.float32),
        "msa": msa,
        "msa_mask": jnp.ones((1, num_msa, num_tokens), dtype=jnp.float32),
        "msa_paired": jnp.zeros((1, num_msa, num_tokens), dtype=jnp.float32),
        "has_deletion": jnp.zeros((1, num_msa, num_tokens), dtype=jnp.float32),
        "deletion_value": jnp.zeros((1, num_msa, num_tokens), dtype=jnp.float32),
        "template_frame_rot": jnp.zeros((1, num_templates, num_tokens, 3, 3), dtype=jnp.float32),
        "template_frame_t": jnp.zeros((1, num_templates, num_tokens, 3), dtype=jnp.float32),
        "template_ca": jnp.zeros((1, num_templates, num_tokens, 3), dtype=jnp.float32),
        "template_cb": jnp.zeros((1, num_templates, num_tokens, 3), dtype=jnp.float32),
        "template_mask": jnp.ones((1, num_templates, num_tokens), dtype=jnp.float32),
        "template_mask_cb": jnp.ones((1, num_templates, num_tokens), dtype=jnp.float32),
        "template_mask_frame": jnp.ones((1, num_templates, num_tokens), dtype=jnp.float32),
        "template_restype": jnp.zeros((1, num_templates, num_tokens, 33), dtype=jnp.float32),
        "visibility_ids": jnp.zeros((1, num_templates, num_tokens), dtype=jnp.float32),
    }


def _make_synthetic_esmfold2_features(num_tokens: int = 10, atoms_per_token: int = 5) -> dict:
    """Create a realistic batched (1, ...) ESMFold2 feature dictionary."""
    num_atoms = num_tokens * atoms_per_token
    atom_to_token = jnp.repeat(jnp.arange(num_tokens, dtype=jnp.int32), atoms_per_token)[None]
    rep_idx = (jnp.arange(num_tokens, dtype=jnp.int32) * atoms_per_token + 1)[None]
    aa_idx = jnp.arange(num_tokens, dtype=jnp.int32) % 20
    res_type = jax.nn.one_hot(2 + aa_idx, 33, dtype=jnp.float32)[None]
    input_ids = _BOLTZ_AA_TO_ESM_ID[aa_idx][None]
    return {
        "res_type": res_type,
        "residue_index": jnp.arange(num_tokens, dtype=jnp.int32)[None],
        "token_index": jnp.arange(num_tokens, dtype=jnp.int32)[None],
        "asym_id": jnp.array(
            [[0] * (num_tokens // 2) + [1] * (num_tokens - num_tokens // 2)],
            dtype=jnp.int32,
        ),
        "entity_id": jnp.zeros((1, num_tokens), dtype=jnp.int32),
        "sym_id": jnp.zeros((1, num_tokens), dtype=jnp.int32),
        "mol_type": jnp.zeros((1, num_tokens), dtype=jnp.int32),
        "input_ids": input_ids,
        "token_attention_mask": jnp.ones((1, num_tokens), dtype=jnp.bool_),
        "distogram_atom_idx": rep_idx,
        "center_atom_idx": rep_idx,
        "cyclic_period": jnp.zeros((1, num_tokens), dtype=jnp.int32),
        "token_bonds": jnp.zeros((1, num_tokens, num_tokens), dtype=jnp.float32),
        "type_bonds": jnp.zeros((1, num_tokens, num_tokens), dtype=jnp.int32),
        "atom_attention_mask": jnp.ones((1, num_atoms), dtype=jnp.bool_),
        "atom_to_token": atom_to_token,
        "ref_pos": jnp.zeros((1, num_atoms, 3), dtype=jnp.float32),
        "ref_charge": jnp.zeros((1, num_atoms), dtype=jnp.float32),
        "ref_element": jnp.zeros((1, num_atoms), dtype=jnp.int32),
        "ref_atom_name_chars": jnp.zeros((1, num_atoms, 4), dtype=jnp.int32),
        "ref_space_uid": atom_to_token,
        "ref_chirality": jnp.zeros((1, num_atoms), dtype=jnp.int32),
        "msa": None,
        "msa_attention_mask": None,
        "has_deletion": None,
        "deletion_value": None,
        "deletion_mean": jnp.zeros((1, num_tokens), dtype=jnp.float32),
    }


def test_unified_pad_signatures_across_all_classes():
    """Verify identical pad_to, pad_to_nearest_multiple, and pad_atoms signatures across all Spec and Input classes."""
    classes = (
        InputSpec,
        ModelInput,
        AF2Spec,
        AFInput,
        ProtenixSpec,
        ProtenixInput,
        JoltzSpec,
        JoltzInput,
        ESMFold2Spec,
        ESMFold2Input,
        AF3LikeSpec,
    )
    for cls in classes:
        sig_pad = inspect.signature(cls.pad_to)
        params_pad = list(sig_pad.parameters.values())
        assert [p.name for p in params_pad] == [
            "self",
            "size",
            "num_residue_atoms",
            "num_msa",
            "num_templates",
        ], f"Mismatch in {cls.__name__}.pad_to: {sig_pad}"
        assert params_pad[2].default == 24
        assert params_pad[3].kind == inspect.Parameter.KEYWORD_ONLY and params_pad[3].default is None
        assert params_pad[4].kind == inspect.Parameter.KEYWORD_ONLY and params_pad[4].default is None

        sig_mult = inspect.signature(cls.pad_to_nearest_multiple)
        params_mult = list(sig_mult.parameters.values())
        assert [p.name for p in params_mult] == [
            "self",
            "multiple",
            "num_residue_atoms",
            "num_msa",
            "num_templates",
        ], f"Mismatch in {cls.__name__}.pad_to_nearest_multiple: {sig_mult}"
        assert params_mult[2].default == 24
        assert params_mult[3].kind == inspect.Parameter.KEYWORD_ONLY and params_mult[3].default is None
        assert params_mult[4].kind == inspect.Parameter.KEYWORD_ONLY and params_mult[4].default is None

        sig_atoms = inspect.signature(cls.pad_atoms)
        params_atoms = list(sig_atoms.parameters.values())
        assert [p.name for p in params_atoms] == [
            "self",
            "num_residue_atoms",
        ], f"Mismatch in {cls.__name__}.pad_atoms: {sig_atoms}"
        assert params_atoms[1].default == 24


def test_pad_atoms_behavior_across_all_9_classes():
    """Verify pad_atoms across all 9 predictor Spec and Input classes, preserving tokens/MSA/templates."""
    # Reject non-positive num_residue_atoms across all Spec classes
    for spec_cls in (AF2Spec, ProtenixSpec, JoltzSpec, ESMFold2Spec, AF3LikeSpec):
        spec = spec_cls().add_protein("ACDEFGHIKL")
        with pytest.raises(ValueError, match="positive"):
            spec.pad_atoms(0)
        with pytest.raises(ValueError, match="positive"):
            spec.pad_atoms(-4)

    # 1. AF2Spec & AFInput: fixed 14/37 atom slots -> no-op for >= 14, UserWarning for < 14, ValueError for <= 0
    af_spec = AF2Spec().add_protein("ACDEFGHIKL:MNPQRSTVWY").pad_atoms(24)
    af_in = af_spec.to_input()
    assert af_in["aatype"].shape == (20,)
    assert int(af_in["seq_mask"].sum()) == 20
    assert af_in["msa_feat"].shape[0] == 1
    assert af_in.pad_atoms(14) is af_in
    with pytest.warns(UserWarning, match="14/37"):
        af_warn = af_in.pad_atoms(13)
    assert af_warn is af_in
    with pytest.raises(ValueError, match="positive"):
        af_in.pad_atoms(0)

    # Dict-formatted _pad_spec={"method": "pad_atoms", "num_residue_atoms": 24} also works on AF2Spec
    af_dict_spec = AF2Spec().add_protein("ACDEFGHIKL")
    af_dict_spec._pad_spec = {"method": "pad_atoms", "num_residue_atoms": 24}
    assert af_dict_spec.to_input()["aatype"].shape == (10,)

    # 2. ProtenixSpec & ProtenixInput: exact L * num_residue_atoms atom padding; tokens/MSA/templates unchanged
    p_base = ProtenixInput(
        features=_make_synthetic_protenix_features(num_tokens=10, atoms_per_token=8, num_msa=3, num_templates=2)
    )
    with pytest.raises(ValueError, match="positive"):
        p_base.pad_atoms(0)
    p_atoms = p_base.pad_atoms(24)
    assert p_atoms.features["restype"].shape == (10, 32)
    assert int(p_atoms.features["token_pad_mask"].sum()) == 10
    assert p_atoms.features["msa"].shape == (3, 10)
    assert p_atoms.features["template_aatype"].shape == (2, 10)
    assert p_atoms.features["atom_to_token_idx"].shape == (240,)
    assert int(p_atoms.features["atom_pad_mask"].sum()) == 80
    # Monotonic warning & no-op when requested atom budget (5 * 10 = 50) < current atoms (80)
    with pytest.warns(UserWarning):
        p_smaller = p_base.pad_atoms(5)
    assert p_smaller.features["atom_to_token_idx"].shape == (80,)

    with patch(
        "flexcraft.structure.protenix._data.to_features",
        return_value=(_make_synthetic_protenix_features(10, 8, num_msa=3, num_templates=2), [], []),
    ):
        p_from_spec, _ = ProtenixSpec().add_protein("ACDEFGHIKL").pad_atoms(24).to_input()
        assert p_from_spec.features["restype"].shape == (10, 32)
        assert p_from_spec.features["msa"].shape == (3, 10)
        assert p_from_spec.features["template_aatype"].shape == (2, 10)
        assert p_from_spec.features["atom_to_token_idx"].shape == (240,)

        p_dict_spec = ProtenixSpec().add_protein("ACDEFGHIKL")
        p_dict_spec._pad_spec = {"method": "pad_atoms", "num_residue_atoms": 20}
        p_from_dict, _ = p_dict_spec.to_input()
        assert p_from_dict.features["atom_to_token_idx"].shape == (200,)

    # 3. JoltzSpec & JoltzInput: W=32-aligned atom padding; tokens/MSA/templates unchanged
    j_base = JoltzInput(
        features=_make_synthetic_joltz_features(num_tokens=10, atoms_per_token=5, num_msa=3, num_templates=2)
    )
    assert j_base.features["atom_pad_mask"].shape == (1, 64)
    with pytest.raises(ValueError, match="positive"):
        j_base.pad_atoms(0)
    j_atoms = j_base.pad_atoms(24)
    assert j_atoms.features["res_type"].shape == (1, 10, 33)
    assert int(j_atoms.features["token_pad_mask"].sum()) == 10
    assert j_atoms.features["msa"].shape == (1, 3, 10, 33)
    assert j_atoms.features["template_mask"].shape == (1, 2, 10)
    assert j_atoms.features["atom_pad_mask"].shape == (1, 256)  # align_up(10 * 24, 32) == 256
    assert int(j_atoms.features["atom_pad_mask"].sum()) == 50
    with pytest.warns(UserWarning):
        j_smaller = j_atoms.pad_atoms(10)  # 100 < 256 -> warns and keeps 256
    assert j_smaller.features["atom_pad_mask"].shape == (1, 256)

    with patch(
        "flexcraft.structure.boltz._data._make_features",
        return_value=(
            _make_synthetic_joltz_features(10, 5, num_msa=3, num_templates=2),
            {"features_dict": {"record": []}, "target_dir": ".", "output_dir": ".", "temp_dir_handle": None},
        ),
    ), patch("flexcraft.structure.boltz._data.Joltz2Writer", return_value=None):
        j_from_spec, _ = JoltzSpec().add_protein("ACDEFGHIKL").pad_atoms(24).to_input()
        assert j_from_spec.features["res_type"].shape == (1, 10, 33)
        assert j_from_spec.features["msa"].shape == (1, 3, 10, 33)
        assert j_from_spec.features["template_mask"].shape == (1, 2, 10)
        assert j_from_spec.features["atom_pad_mask"].shape == (1, 256)

    # 4. ESMFold2Spec & ESMFold2Input: W=32-aligned atom padding; tokens/MSA unchanged
    e_base = ESMFold2Input(features=_make_synthetic_esmfold2_features(num_tokens=10, atoms_per_token=5))
    with pytest.raises(ValueError, match="positive"):
        e_base.pad_atoms(0)
    e_atoms = e_base.pad_atoms(24)
    assert e_atoms.features["res_type"].shape == (1, 10, 33)
    assert int(e_atoms.features["token_attention_mask"].sum()) == 10
    assert e_atoms.features["msa"] is None
    assert e_atoms.features["atom_attention_mask"].shape == (1, 256)
    assert int(e_atoms.features["atom_attention_mask"].sum()) == 50
    with pytest.warns(UserWarning):
        e_smaller = e_atoms.pad_atoms(10)
    assert e_smaller.features["atom_attention_mask"].shape == (1, 256)

    with patch(
        "flexcraft.structure.esmfold2._data.to_features",
        return_value=(_make_synthetic_esmfold2_features(10, 5), []),
    ):
        e_from_spec, _ = ESMFold2Spec().add_protein("ACDEFGHIKL").pad_atoms(24).to_input()
        assert e_from_spec.features["res_type"].shape == (1, 10, 33)
        assert e_from_spec.features["atom_attention_mask"].shape == (1, 256)

    # 5. AF3LikeSpec.pad_atoms(24) dispatch across all 4 backends (plus dict _pad_spec fallback)
    af3_atoms_spec = AF3LikeSpec().add_protein("ACDEFGHIKL").pad_atoms(24)
    af_via_af3 = af3_atoms_spec.to_input(model="af2_model_1_ptm")
    assert isinstance(af_via_af3, AFInput) and af_via_af3["aatype"].shape == (10,)

    with patch(
        "flexcraft.structure.protenix._data.to_features",
        return_value=(_make_synthetic_protenix_features(10, 8), [], []),
    ):
        p_via_af3, _ = af3_atoms_spec.to_input(model="protenix_base_default_v1.0.0")
        assert p_via_af3.features["restype"].shape == (10, 32)
        assert p_via_af3.features["atom_to_token_idx"].shape == (240,)

    with patch(
        "flexcraft.structure.boltz._data._make_features",
        return_value=(
            _make_synthetic_joltz_features(10, 5),
            {"features_dict": {"record": []}, "target_dir": ".", "output_dir": ".", "temp_dir_handle": None},
        ),
    ), patch("flexcraft.structure.boltz._data.Joltz2Writer", return_value=None):
        j_via_af3, _ = af3_atoms_spec.to_input(model="joltz")
        assert j_via_af3.features["res_type"].shape == (1, 10, 33)
        assert j_via_af3.features["atom_pad_mask"].shape == (1, 256)

        j_dict = JoltzSpec().add_protein("ACDEFGHIKL")
        j_dict._pad_spec = {"method": "pad_atoms", "num_residue_atoms": 24}
        j_from_dict, _ = j_dict.to_input()
        assert j_from_dict.features["atom_pad_mask"].shape == (1, 256)

    with patch(
        "flexcraft.structure.esmfold2._data.to_features",
        return_value=(_make_synthetic_esmfold2_features(10, 5), []),
    ):
        e_via_af3, _ = af3_atoms_spec.to_input(model="esmfold2_fast")
        assert e_via_af3.features["res_type"].shape == (1, 10, 33)
        assert e_via_af3.features["atom_attention_mask"].shape == (1, 256)


def test_af2_spec_and_input_padding_contracts():
    """Verify AF2Spec and AFInput pad_to, pad_to_nearest_multiple, monotonicity, and update_sequence masking."""
    spec = AF2Spec().add_protein("ACDEFGHIKL:MNPQRSTVWY")
    spec_padded = spec.pad_to(32, num_msa=6, num_templates=3)
    assert spec._pad_spec is None
    assert spec_padded._pad_spec is not None

    af_in = spec_padded.to_input()
    assert af_in["aatype"].shape == (32,)
    assert af_in["seq_mask"].shape == (32,)
    assert int(af_in["seq_mask"].sum()) == 20
    assert jnp.all(af_in["asym_id"][20:] == -1)
    assert af_in["msa_feat"].shape == (6, 32, 49)
    assert af_in["msa_row_mask"].shape == (6,)
    assert int(af_in["msa_row_mask"].sum()) == 1
    assert af_in["template_aatype"].shape == (3, 32)
    assert af_in["template_mask"].shape == (3,)
    assert af_in["prev"]["prev_pair"].shape == (32, 32, 128)

    with pytest.warns(UserWarning):
        af_smaller = af_in.pad_to(16, num_msa=2, num_templates=1)
    assert af_smaller["aatype"].shape == (32,)
    assert af_smaller["msa_feat"].shape == (6, 32, 49)
    assert af_smaller["template_aatype"].shape == (3, 32)

    af_mult = AF2Spec().add_protein("ACDEFGHIKLMNPQRSTVWY").pad_to_nearest_multiple(16).to_input()
    assert af_mult["aatype"].shape == (32,)

    full_seq = jnp.ones((32, 20), dtype=jnp.float32) / 20.0
    updated = af_in.update_sequence(full_seq)
    assert jnp.all(updated["target_feat"][20:] == 0.0)


def test_protenix_spec_and_input_padding_contracts():
    """Verify ProtenixSpec and ProtenixInput padding contracts including size==current atom padding and double pad."""
    p_in = ProtenixInput(
        features=_make_synthetic_protenix_features(num_tokens=10, atoms_per_token=8, num_msa=2, num_templates=1)
    )

    p_same_tok = p_in.pad_to(10, num_residue_atoms=24)
    assert p_same_tok.features["restype"].shape == (10, 32)
    assert p_same_tok.features["atom_to_token_idx"].shape == (240,)
    assert int(p_same_tok.features["atom_pad_mask"].sum()) == 80

    p_pad = p_in.pad_to(16, num_residue_atoms=24, num_msa=5, num_templates=3)
    assert p_pad.features["restype"].shape == (16, 32)
    assert int(p_pad.features["token_pad_mask"].sum()) == 10
    assert jnp.all(p_pad.features["asym_id"][10:] == -1)
    assert p_pad.features["atom_to_token_idx"].shape == (16 * 24,)
    assert int(p_pad.features["atom_pad_mask"].sum()) == 80
    assert p_pad.features["msa"].shape == (5, 16)
    assert jnp.all(p_pad.features["msa"][2:] == 31)
    assert p_pad.features["msa_mask"].shape == (5, 16)
    assert jnp.all(p_pad.features["msa_mask"][2:] == 0)
    assert jnp.all(p_pad.features["msa_mask"][:, 10:] == 0)
    assert p_pad.features["template_aatype"].shape == (3, 16)
    assert p_pad.features["template_distogram"].shape == (3, 16, 16, 39)

    p_pad2 = p_pad.pad_to_nearest_multiple(32, num_residue_atoms=24)
    assert p_pad2.features["restype"].shape == (32, 32)
    assert int(p_pad2.features["token_pad_mask"].sum()) == 10
    assert int(p_pad2.features["atom_pad_mask"].sum()) == 80

    with pytest.warns(UserWarning):
        p_noop = p_in.pad_to(6, num_residue_atoms=24)
    assert p_noop.features["restype"].shape == (10, 32)
    assert p_noop.features["atom_to_token_idx"].shape == (80,)

    new_aa = jax.nn.one_hot(jnp.full((16,), 3), 20, dtype=jnp.float32)
    p_seq = p_pad.set_aa(new_aa)
    assert jnp.all(p_seq.features["restype"][10:] == 0.0)
    assert jnp.all(p_seq.features["profile"][10:] == 0.0)
    assert jnp.all(p_seq.features["msa"][:, 10:] == 31)


def test_joltz_and_esmfold2_padding_contracts_and_w32_invariant():
    """Verify JoltzInput and ESMFold2Input W=32 atom alignment, size==current atom padding, and ESMFold2 specifics."""
    j_in = JoltzInput(
        features=_make_synthetic_joltz_features(num_tokens=10, atoms_per_token=5, num_msa=2, num_templates=1)
    )
    assert j_in.features["atom_pad_mask"].shape == (1, 64)
    assert int(j_in.features["atom_pad_mask"].sum()) == 50

    j_same = j_in.pad_to(10, num_residue_atoms=24)
    assert j_same.features["res_type"].shape == (1, 10, 33)
    assert j_same.features["atom_pad_mask"].shape == (1, 256)
    assert int(j_same.features["atom_pad_mask"].sum()) == 50

    j_pad = j_in.pad_to_nearest_multiple(16, num_residue_atoms=24, num_msa=6, num_templates=3)
    assert j_pad.features["res_type"].shape == (1, 16, 33)
    assert j_pad.features["atom_pad_mask"].shape == (1, 384)
    assert j_pad.features["atom_to_token"].shape == (1, 384, 16)
    assert jnp.all(j_pad.features["atom_to_token"][0, 50:] == 0.0)
    assert jnp.all(j_pad.features["asym_id"][0, 10:] == -1)
    assert j_pad.features["msa"].shape == (1, 6, 16, 33)
    assert j_pad.features["template_mask"].shape == (1, 3, 16)

    with pytest.warns(UserWarning):
        j_small = j_pad.pad_to(8, num_msa=2, num_templates=1)
    assert j_small.features["res_type"].shape == (1, 16, 33)
    assert j_small.features["msa"].shape == (1, 6, 16, 33)

    e_in = ESMFold2Input(features=_make_synthetic_esmfold2_features(num_tokens=10, atoms_per_token=5))
    assert e_in.features["atom_attention_mask"].shape == (1, 64)
    assert int(e_in.features["atom_attention_mask"].sum()) == 50

    e_same = e_in.pad_to(10, num_residue_atoms=24)
    assert e_same.features["res_type"].shape == (1, 10, 33)
    assert e_same.features["atom_attention_mask"].shape == (1, 256)

    e_pad = e_in.pad_to(16, num_residue_atoms=24, num_msa=4, num_templates=5)
    assert e_pad.features["res_type"].shape == (1, 16, 33)
    assert jnp.all(e_pad.features["res_type"][0, 10:] == 0.0)
    assert jnp.all(e_pad.features["input_ids"][0, 10:] == ESM_PAD_ID)
    assert jnp.all(e_pad.features["asym_id"][0, 10:] == -1)
    assert e_pad.features["atom_attention_mask"].shape == (1, 384)
    assert e_pad.features["msa"].shape == (1, 4, 16, 33)
    assert jnp.all(e_pad.features["msa"][0, :, 10:] == 0.0)

    e_mixed = e_pad.copy()
    e_mixed.features["mol_type"] = e_mixed.features["mol_type"].at[0, 8:10].set(1)
    seq_onehot = jax.nn.one_hot(jnp.full((16,), 4), 20, dtype=jnp.float32)
    e_updated = e_mixed.set_aa(seq_onehot)
    expected_esm_id = int(_BOLTZ_AA_TO_ESM_ID[4])
    assert jnp.all(e_updated.features["input_ids"][0, :8] == expected_esm_id)
    assert jnp.all(e_updated.features["input_ids"][0, 8:10] == ESM_NON_PROTEIN_ID)
    assert jnp.all(e_updated.features["input_ids"][0, 10:] == ESM_PAD_ID)
    assert jnp.all(e_updated.features["res_type"][0, 10:] == 0.0)


def test_af3_like_spec_dispatch_and_pad_spec_preservation():
    """Verify AF3LikeSpec preserves _pad_spec when converting to all 4 backend specs."""
    af3_spec = AF3LikeSpec().add_protein("ACDEFGHIKL").pad_to(32, num_residue_atoms=24, num_msa=8, num_templates=2)
    for kind, expected_cls in (
        ("af2", AF2Spec),
        ("protenix", ProtenixSpec),
        ("joltz", JoltzSpec),
        ("esmfold2", ESMFold2Spec),
    ):
        backend_spec = _any_spec_to_model_spec(af3_spec, kind)
        assert isinstance(backend_spec, expected_cls)
        assert backend_spec._pad_spec == af3_spec._pad_spec

    af_in = af3_spec.to_input(model="af2_model_1_ptm")
    assert isinstance(af_in, AFInput)
    assert af_in["aatype"].shape == (32,)
    assert af_in["msa_feat"].shape == (8, 32, 49)
    assert af_in["template_aatype"].shape == (2, 32)

    with patch(
        "flexcraft.structure.protenix._data.to_features",
        return_value=(_make_synthetic_protenix_features(10, 8), [], []),
    ):
        p_in, _ = af3_spec.to_input(model="protenix_base_default_v1.0.0")
        assert isinstance(p_in, ProtenixInput)
        assert p_in.features["restype"].shape == (32, 32)
        assert p_in.features["atom_to_token_idx"].shape == (32 * 24,)

    with patch(
        "flexcraft.structure.boltz._data._make_features",
        return_value=(
            _make_synthetic_joltz_features(10, 5),
            {"features_dict": {"record": []}, "target_dir": ".", "output_dir": ".", "temp_dir_handle": None},
        ),
    ), patch("flexcraft.structure.boltz._data.Joltz2Writer", return_value=None):
        j_in, _ = af3_spec.to_input(model="joltz")
        assert isinstance(j_in, JoltzInput)
        assert j_in.features["res_type"].shape == (1, 32, 33)
        assert j_in.features["atom_pad_mask"].shape == (1, 32 * 24)

    with patch(
        "flexcraft.structure.esmfold2._data.to_features",
        return_value=(_make_synthetic_esmfold2_features(10, 5), []),
    ):
        e_in, _ = af3_spec.to_input(model="esmfold2_fast")
        assert isinstance(e_in, ESMFold2Input)
        assert e_in.features["res_type"].shape == (1, 32, 33)
        assert e_in.features["atom_attention_mask"].shape == (1, 32 * 24)


def _evaluate_full_result_surface(res, sub_mask, alt_chain):
    """Evaluate the full unified <model>Result metric surface inside @jax.jit."""
    return {
        "max_pae": jnp.asarray(res.max_pae, dtype=jnp.float32),
        "pae_bin_edges": res.pae_bin_edges,
        "pae_bin_centers": res.pae_bin_centers,
        "distogram_mean": res.distogram_mean,
        "distance": res.distance,
        "ptm": res.ptm,
        "index_ptm": res.index_ptm(sub_mask),
        "iptm": res.iptm,
        "index_iptm": res.index_iptm(alt_chain),
        "chain_ptm": res.chain_ptm(1, 0),
        "ipae": res.ipae,
        "index_ipae": res.index_ipae(alt_chain),
        "min_ipae": res.min_ipae,
        "index_min_ipae": res.index_min_ipae(alt_chain),
        "chain_pae": res.chain_pae(1, 0),
        "ipsae": res.ipsae(),
        "index_ipsae": res.index_ipsae(alt_chain, pae_cutoff=0.4),
        "ptm_score": res.ptm_score,
        "index_ptm_score": res.index_ptm_score(alt_chain),
        "mean_plddt": res.mean_plddt,
        "index_plddt": res.index_plddt(sub_mask),
        "mean_pae": res.mean_pae,
    }


_PER_SAMPLE_SCALAR_KEYS = (
    "ptm",
    "index_ptm",
    "iptm",
    "index_iptm",
    "chain_ptm",
    "ipae",
    "index_ipae",
    "min_ipae",
    "index_min_ipae",
    "chain_pae",
    "ipsae",
    "index_ipsae",
    "ptm_score",
    "index_ptm_score",
    "mean_plddt",
    "index_plddt",
    "mean_pae",
)


def test_unified_result_surface_single_and_multisample_under_jit():
    """Verify full <model>Result surface under @jax.jit for single-sample () and multi-sample (S,) across all 4 predictors."""
    centers = pae_bin_centers(64, 32.0)
    key = jax.random.key(123)
    k1, k2 = jax.random.split(key)

    # Distinct 3-sample logits/PAE/pLDDT on 10 valid tokens padded to 16 with garbage tail [10:16]
    valid_logits_3 = jax.random.normal(k1, (3, 10, 10, 64), dtype=jnp.float32)
    valid_logits_3 = valid_logits_3.at[0, :, :, 2].add(6.0)
    valid_logits_3 = valid_logits_3.at[1, :, :, 10].add(6.0)
    valid_logits_3 = valid_logits_3.at[2, :, :, 25].add(6.0)
    noise_logits_3 = jax.random.normal(k2, (3, 16, 16, 64), dtype=jnp.float32) * 4.0
    pae_logits_3 = noise_logits_3.at[:, :10, :10, :].set(valid_logits_3)
    pae_raw_3 = (jax.nn.softmax(pae_logits_3, axis=-1) * centers).sum(axis=-1)
    pae_raw_3 = pae_raw_3.at[:, 10:, :].set(30.0).at[:, :, 10:].set(30.0)

    plddt_tok_3 = jnp.stack(
        [
            jnp.linspace(0.70, 0.90, 16, dtype=jnp.float32),
            jnp.linspace(0.50, 0.80, 16, dtype=jnp.float32),
            jnp.linspace(0.85, 0.95, 16, dtype=jnp.float32),
        ],
        axis=0,
    ).at[:, 10:].set(0.05)

    atom_plddt_logits_3 = jnp.zeros((3, 384, 50), dtype=jnp.float32)
    atom_plddt_logits_3 = atom_plddt_logits_3.at[0, :, 40].set(8.0)
    atom_plddt_logits_3 = atom_plddt_logits_3.at[1, :, 25].set(8.0)
    atom_plddt_logits_3 = atom_plddt_logits_3.at[2, :, 46].set(8.0)
    atom_plddt_logits_3 = atom_plddt_logits_3.at[:, 80:, :].set(0.0)

    sub_mask_pad = jnp.arange(16) < 5
    sub_mask_unpad = jnp.arange(10) < 5
    alt_chain_pad = jnp.array([0, 0, 0, 1, 1, 1, 1, 0, 0, 1, -1, -1, -1, -1, -1, -1], dtype=jnp.int32)
    alt_chain_unpad = alt_chain_pad[:10]

    # 1. AFResult (single-sample (), plus test breaks / bin_edges / fallback PAE edge conventions)
    af_in = AF2Spec().add_protein("ACDEF:GHIKL").pad_to(16).to_input()
    for pae_edge_dict in (
        {"breaks": jnp.linspace(0.0, 31.0, 63)},
        {"bin_edges": jnp.linspace(0.5, 31.5, 63)},
        {},
    ):
        af_res = AFResult(
            inputs=af_in.data,
            result={
                "structure_module": {
                    "final_atom_positions": jnp.ones((16, 37, 3), dtype=jnp.float32),
                    "final_atom_mask": jnp.ones((16, 37), dtype=jnp.float32),
                },
                "predicted_lddt": {"logits": jnp.zeros((16, 50), dtype=jnp.float32).at[:10, 42].set(6.0)},
                "predicted_aligned_error": {"logits": pae_logits_3[0], **pae_edge_dict},
                "distogram": {
                    "logits": jnp.zeros((16, 16, 64), dtype=jnp.float32),
                    "bin_edges": jnp.linspace(2.0, 22.0, 63),
                },
                "representations": {
                    "msa_first_row": jnp.zeros((16, 256), dtype=jnp.float32),
                    "single": jnp.zeros((16, 384), dtype=jnp.float32),
                    "pair": jnp.zeros((16, 16, 128), dtype=jnp.float32),
                },
            },
        )
        af_out = jax.jit(_evaluate_full_result_surface)(af_res, sub_mask_pad, alt_chain_pad)
        assert jnp.allclose(af_out["max_pae"], 32.0, atol=1e-5)
        assert af_out["pae_bin_edges"].shape == (65,)
        assert jnp.allclose(af_out["pae_bin_edges"][0], 0.0, atol=1e-5)
        assert jnp.allclose(af_out["pae_bin_edges"][-1], 32.0, atol=1e-5)
        assert jnp.allclose(af_out["pae_bin_centers"], centers, atol=1e-5)
        assert jnp.allclose(af_out["distogram_mean"], af_out["distance"], atol=1e-6)
        for k in _PER_SAMPLE_SCALAR_KEYS:
            assert af_out[k].shape == (), f"Expected () for AFResult.{k}, got {af_out[k].shape}"
            assert jnp.isfinite(af_out[k]), f"Non-finite AFResult.{k}"
        assert 0.0 <= float(af_out["ipae"]) <= 1.0
        assert 0.0 <= float(af_out["min_ipae"]) <= 1.0
        assert jnp.allclose(
            af_res.ipsae(pae_cutoff=0.4),
            af_res.index_ipsae(raw_pae_threshold=12.8),
            atol=1e-6,
        )

        af_unpad = af_res.unpad()
        af_unpad_out = jax.jit(_evaluate_full_result_surface)(af_unpad, sub_mask_unpad, alt_chain_unpad)
        for k in _PER_SAMPLE_SCALAR_KEYS:
            assert jnp.allclose(af_out[k], af_unpad_out[k], atol=1e-6), f"AFResult pad mismatch in {k}"

    # 2. ProtenixResult, JoltzResult, ESMFold2Result: both single-sample () and 3-sample (3,)
    p_in = ProtenixInput(features=_make_synthetic_protenix_features(10, 8)).pad_to(16, num_residue_atoms=24)
    j_in = JoltzInput(features=_make_synthetic_joltz_features(10, 5)).pad_to(16, num_residue_atoms=24)
    e_in = ESMFold2Input(features=_make_synthetic_esmfold2_features(10, 5)).pad_to(16, num_residue_atoms=24)

    def _build_single_and_multi():
        p_single = ProtenixResult(
            data={
                "features": p_in.features,
                "samples": jnp.ones((1, 384, 3), dtype=jnp.float32),
                "distogram": jnp.zeros((16, 16, 64), dtype=jnp.float32),
                "confidence": _SyntheticConfidence(
                    plddt=jnp.full((1, 384), 0.8, dtype=jnp.float32),
                    plddt_logits=atom_plddt_logits_3[0:1],
                    pae=pae_raw_3[0:1],
                    pae_logits=pae_logits_3[0:1],
                ),
            }
        )
        p_multi = ProtenixResult(
            data={
                "features": p_in.features,
                "samples": jnp.ones((3, 1, 384, 3), dtype=jnp.float32),
                "distogram": jnp.zeros((16, 16, 64), dtype=jnp.float32),
                "confidence": _SyntheticConfidence(
                    plddt=jnp.full((3, 1, 384), 0.8, dtype=jnp.float32),
                    plddt_logits=atom_plddt_logits_3[:, None, :, :],
                    pae=pae_raw_3[:, None, :, :],
                    pae_logits=pae_logits_3[:, None, :, :, :],
                ),
            }
        )

        j_single = JoltzResult(
            data={
                "features": j_in.features,
                "samples": jnp.ones((1, 384, 3), dtype=jnp.float32),
                "distogram": jnp.zeros((16, 16, 64), dtype=jnp.float32),
                "confidence": _SyntheticConfidence(
                    plddt=plddt_tok_3[0:1],
                    plddt_logits=jnp.zeros((1, 16, 50), dtype=jnp.float32),
                    pae=pae_raw_3[0:1],
                    pae_logits=pae_logits_3[0:1],
                ),
            }
        )
        j_multi = JoltzResult(
            data={
                "features": j_in.features,
                "samples": jnp.ones((3, 1, 384, 3), dtype=jnp.float32),
                "distogram": jnp.zeros((16, 16, 64), dtype=jnp.float32),
                "confidence": _SyntheticConfidence(
                    plddt=plddt_tok_3[:, None, :],
                    plddt_logits=jnp.zeros((3, 1, 16, 50), dtype=jnp.float32),
                    pae=pae_raw_3[:, None, :, :],
                    pae_logits=pae_logits_3[:, None, :, :, :],
                ),
            }
        )

        e_single = ESMFold2Result(
            data={
                "features": e_in.features,
                "samples": jnp.ones((1, 384, 3), dtype=jnp.float32),
                "distogram": jnp.zeros((16, 16, 64), dtype=jnp.float32),
                "confidence": ESMFold2Confidence(
                    plddt=plddt_tok_3[0:1],
                    plddt_logits=atom_plddt_logits_3[0:1],
                    pae=pae_raw_3[0:1],
                    pae_logits=pae_logits_3[0:1],
                    ptm=jnp.array([0.8]),
                    iptm=jnp.array([0.75]),
                ),
            }
        )
        e_multi = ESMFold2Result(
            data={
                "features": e_in.features,
                "samples": jnp.ones((3, 1, 384, 3), dtype=jnp.float32),
                "distogram": jnp.zeros((16, 16, 64), dtype=jnp.float32),
                "confidence": ESMFold2Confidence(
                    plddt=plddt_tok_3[:, None, :],
                    plddt_logits=atom_plddt_logits_3[:, None, :, :],
                    pae=pae_raw_3[:, None, :, :],
                    pae_logits=pae_logits_3[:, None, :, :, :],
                    ptm=jnp.array([0.8, 0.7, 0.6]),
                    iptm=jnp.array([0.75, 0.65, 0.55]),
                ),
            }
        )
        return (p_single, p_multi), (j_single, j_multi), (e_single, e_multi)

    for single_res, multi_res in _build_single_and_multi():
        cls_name = type(single_res).__name__
        out_s = jax.jit(_evaluate_full_result_surface)(single_res, sub_mask_pad, alt_chain_pad)
        out_m = jax.jit(_evaluate_full_result_surface)(multi_res, sub_mask_pad, alt_chain_pad)

        assert jnp.allclose(out_s["max_pae"], 32.0)
        assert out_s["pae_bin_edges"].shape == (65,)
        assert jnp.allclose(out_s["pae_bin_centers"], centers, atol=1e-5)
        assert jnp.allclose(out_s["distogram_mean"], out_s["distance"], atol=1e-6)
        assert jnp.allclose(out_m["distogram_mean"], out_m["distance"], atol=1e-6)

        for k in _PER_SAMPLE_SCALAR_KEYS:
            assert out_s[k].shape == (), f"{cls_name} single-sample {k} expected (), got {out_s[k].shape}"
            assert out_m[k].shape == (3,), f"{cls_name} multi-sample {k} expected (3,), got {out_m[k].shape}"
            # Sample 0 of multi_res matches single_res (constructed from slice [0:1])
            assert jnp.allclose(out_m[k][0], out_s[k], atol=1e-6), f"{cls_name} sample-0 mismatch in {k}"
        # Distinct samples produce distinct pTM / ipAE values (verifying no premature cross-sample reduction)
        assert not jnp.allclose(out_m["ptm"][0], out_m["ptm"][2], atol=1e-3)
        assert not jnp.allclose(out_m["ipae"][0], out_m["ipae"][2], atol=1e-3)
        assert not jnp.allclose(out_m["mean_plddt"][0], out_m["mean_plddt"][1], atol=1e-3)

        # Monomer chain_index produces +inf on index_min_ipae in both () and (3,)
        mono_chain = jnp.zeros((16,), dtype=jnp.int32)
        assert jnp.isinf(single_res.index_min_ipae(mono_chain))
        assert jnp.all(jnp.isinf(multi_res.index_min_ipae(mono_chain)))

        # .to_data() works inside @jax.jit, .unpad() raises inside @jax.jit, and host .unpad() preserves metrics
        d_s, p_dssp = jax.jit(lambda r: (r.to_data(), r.to_data().p_dssp))(single_res)
        assert d_s["atom_positions"].shape == (16, 24, 3)
        assert int(d_s["mask"].sum()) == 10
        assert jnp.all(d_s["chain_index"][10:] == -1)
        assert jnp.isfinite(p_dssp["L"])

        with pytest.raises(RuntimeError, match="cannot be called inside jax.jit"):
            jax.jit(lambda r: r.unpad())(multi_res)

        unpad_m = multi_res.unpad()
        out_unpad_m = jax.jit(_evaluate_full_result_surface)(unpad_m, sub_mask_unpad, alt_chain_unpad)
        for k in _PER_SAMPLE_SCALAR_KEYS:
            assert out_unpad_m[k].shape == (3,)
            assert jnp.allclose(out_m[k], out_unpad_m[k], atol=1e-6), f"{cls_name} unpad mismatch in {k}"
