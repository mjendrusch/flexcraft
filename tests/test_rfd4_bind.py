"""Unit and @jax.jit tests for flexcraft.pipelines.rfd4.bind."""

import inspect
import pathlib
import types
from unittest.mock import patch

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import flexcraft.sequence.aa_codes as aas
from flexcraft.files.csv import ScoreCSV
from flexcraft.pipelines.rfd4 import bind_disordered
from flexcraft.pipelines.rfd4.bind import (
    SCORE_KEYS,
    build_parser,
    evaluate_binder_prediction,
    extract_binder_sequence,
    extract_target_sequences,
    prepare_joltz_input,
    score_prediction_samples,
)
from flexcraft.structure.af._af3_like_wrapper import AF2Spec
from flexcraft.structure.af._data import AFResult
from flexcraft.structure.boltz._data import JoltzInput
from flexcraft.structure.boltz._result import JoltzResult
from flexcraft.structure.metrics import pae_bin_centers


class _SyntheticConfidence(eqx.Module):
    plddt: jax.Array
    plddt_logits: jax.Array
    pae: jax.Array
    pae_logits: jax.Array


def _encode_rfd4_seq(seq: str) -> np.ndarray:
    """Encode a 1-letter amino acid string into 1-based RFD4 token indices (1..20)."""
    return np.array([aas.AF2_CODE.index(ch) + 1 for ch in seq], dtype=np.int32)


def _make_joltz_features(
    chain_lengths: tuple[int, ...],
    *,
    atoms_per_token: int = 5,
    num_msa: int = 1,
    num_templates: int = 0,
) -> dict:
    """Construct a synthetic batched (1, ...) Joltz feature dict for multi-chain proteins."""
    num_tokens = sum(chain_lengths)
    num_atoms = num_tokens * atoms_per_token
    asym_list = []
    for cid, length in enumerate(chain_lengths):
        asym_list.extend([cid] * length)
    asym_id = jnp.array([asym_list], dtype=jnp.int32)
    atom_to_token_idx = jnp.repeat(jnp.arange(num_tokens, dtype=jnp.int32), atoms_per_token)
    atom_to_token = jax.nn.one_hot(atom_to_token_idx, num_tokens, dtype=jnp.float32)[None]
    res_type = jax.nn.one_hot(2 + (jnp.arange(num_tokens) % 20), 33, dtype=jnp.float32)[None]
    msa = jnp.broadcast_to(res_type[:, None, :, :], (1, num_msa, num_tokens, 33))
    return {
        "token_index": jnp.arange(num_tokens, dtype=jnp.int32)[None],
        "residue_index": jnp.arange(num_tokens, dtype=jnp.int32)[None],
        "asym_id": asym_id,
        "entity_id": asym_id,
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
        "template_mask": jnp.zeros((1, num_templates, num_tokens), dtype=jnp.float32),
        "template_mask_cb": jnp.zeros((1, num_templates, num_tokens), dtype=jnp.float32),
        "template_mask_frame": jnp.zeros((1, num_templates, num_tokens), dtype=jnp.float32),
        "template_restype": jnp.zeros((1, num_templates, num_tokens, 33), dtype=jnp.float32),
        "visibility_ids": jnp.zeros((1, num_templates, num_tokens), dtype=jnp.float32),
    }


def _make_ideal_backbone_coords24(num_residues: int) -> jnp.ndarray:
    """Create valid non-collinear (L, 24, 3) backbone coordinates for template frame tests."""
    idx = jnp.arange(num_residues, dtype=jnp.float32) * 3.8
    coords = jnp.zeros((num_residues, 24, 3), dtype=jnp.float32)
    coords = coords.at[:, 0].set(jnp.stack([idx - 1.2, jnp.full_like(idx, 0.8), jnp.zeros_like(idx)], axis=-1))
    coords = coords.at[:, 1].set(jnp.stack([idx, jnp.zeros_like(idx), jnp.zeros_like(idx)], axis=-1))
    coords = coords.at[:, 2].set(jnp.stack([idx + 1.2, jnp.full_like(idx, 0.9), jnp.full_like(idx, 0.3)], axis=-1))
    coords = coords.at[:, 3].set(jnp.stack([idx + 1.5, jnp.full_like(idx, 2.0), jnp.full_like(idx, 0.3)], axis=-1))
    return coords


def test_build_parser_defaults_and_boolean_flags():
    """Verify CLI parser defaults and --use_template / --auto_condition flag handling."""
    parser = build_parser()
    args = parser.parse_args(["--in_path", "target.pdb"])
    assert args.in_path == pathlib.Path("target.pdb")
    assert args.target_selection == ""
    assert args.hotspots == ""
    assert args.auto_condition is True
    assert args.num_designs == 5
    assert args.max_attempts == 200
    assert args.num_aa == 80
    assert args.use_template is False
    assert args.f_min_ipae == pytest.approx(2.0)
    assert args.f_plddt == pytest.approx(0.8)
    assert args.num_samples == 4
    assert args.num_recycle == 3
    assert args.boltz_path == pathlib.Path("./params/boltz/")
    assert args.forbidden == "C"
    assert args.seed == 0

    assert parser.parse_args(["--in_path", "t.pdb", "--forbidden", "CK"]).forbidden == "CK"
    with pytest.raises(SystemExit):
        from flexcraft.pipelines.rfd4.bind import main as bind_main

        bind_main(["--in_path", "t.pdb", "--forbidden", "XZ"])

    assert parser.parse_args(["--in_path", "t.pdb", "--use_template"]).use_template is True
    assert parser.parse_args(["--in_path", "t.pdb", "--use_template", "True"]).use_template is True
    assert parser.parse_args(["--in_path", "t.pdb", "--use_template", "False"]).use_template is False
    assert parser.parse_args(["--in_path", "t.pdb", "--use_template", "--no-use_template"]).use_template is False
    assert parser.parse_args(["--in_path", "t.pdb", "--no-auto_condition"]).auto_condition is False

    with pytest.raises(SystemExit):
        parser.parse_args(["--in_path", "t.pdb", "--use_template", "not_a_bool"])


def test_extract_target_and_binder_sequences():
    """Verify target chain extraction preserves first-appearance chain order and excludes binder & padding."""
    binder_len = 6
    target_seq_1 = "ACDEFGH"
    target_seq_2 = "KLMNPQ"
    cond_seq = np.concatenate(
        [
            np.zeros((binder_len,), dtype=np.int32),
            _encode_rfd4_seq(target_seq_1),
            _encode_rfd4_seq(target_seq_2),
            np.full((4,), 5, dtype=np.int32),
        ]
    )[None, :]
    chain_iid = np.concatenate(
        [
            np.zeros((binder_len,), dtype=np.float32),
            np.full((len(target_seq_1),), 5.0, dtype=np.float32),
            np.full((len(target_seq_2),), 2.0, dtype=np.float32),
            np.full((4,), 2.0, dtype=np.float32),
        ]
    )[None, :]
    padding_mask = np.array(
        [[True] * (binder_len + len(target_seq_1) + len(target_seq_2)) + [False] * 4],
        dtype=bool,
    )

    fake_rfd4_input = types.SimpleNamespace(
        padding_mask=padding_mask,
        cond_seq=cond_seq,
        chain_iid=chain_iid,
    )
    extracted = extract_target_sequences(fake_rfd4_input, binder_len)
    assert extracted == [target_seq_1, target_seq_2]

    empty_target_input = types.SimpleNamespace(
        padding_mask=np.array([[True] * binder_len + [False] * 4]),
        cond_seq=cond_seq[:, : binder_len + 4],
        chain_iid=None,
    )
    with pytest.raises(ValueError, match="no target tokens"):
        extract_target_sequences(empty_target_input, binder_len)

    no_cond_input = types.SimpleNamespace(
        padding_mask=padding_mask,
        cond_seq=None,
        chain_iid=chain_iid,
    )
    with pytest.raises(ValueError, match="cond_seq is unset"):
        extract_target_sequences(no_cond_input, binder_len)

    with_method = types.SimpleNamespace(sequence=lambda num_aa: "WYRSTV"[:num_aa])
    assert extract_binder_sequence(with_method, 6) == "WYRSTV"
    without_method = types.SimpleNamespace()
    assert extract_binder_sequence(without_method, 5, rfd4_restype=_encode_rfd4_seq("GHIKLMP")) == "GHIKL"


def test_prepare_joltz_input_pad_atoms_msa_inheritance_and_template_modes():
    """Verify prepare_joltz_input uses pad_atoms(24) (keeping L=20 tokens and padding atoms to 480) in both modes."""
    sig_params = inspect.signature(prepare_joltz_input).parameters
    assert "pad_multiple" not in sig_params
    assert sig_params["num_residue_atoms"].default == 24

    binder_seq = "ACDEFG"  # L_binder = 6
    target_seqs = ["HIKLMNPQ", "RSTVWY"]  # L_target = 8 + 6 = 14 -> L_total = 20
    chain_lengths = (6, 8, 6)
    sentinel_writer = object()

    base_complex_input = JoltzInput(
        features=_make_joltz_features(chain_lengths, atoms_per_token=5, num_msa=1, num_templates=0)
    )
    target_msa_feats = _make_joltz_features((8, 6), atoms_per_token=5, num_msa=4, num_templates=0)
    distinct_msa = jnp.zeros((1, 4, 14, 33), dtype=jnp.float32).at[:, :, :, 7].set(1.0)
    target_msa_feats["msa"] = distinct_msa
    target_msa_input = JoltzInput(features=target_msa_feats)

    # 1. use_template=False: inherits target MSA at start=6, keeps L=20 valid tokens, pads atoms to 20*24 = 480
    with patch(
        "flexcraft.pipelines.rfd4.bind.JoltzSpec.to_input",
        return_value=(base_complex_input, sentinel_writer),
    ):
        j_msa, writer_out = prepare_joltz_input(
            binder_seq,
            target_seqs,
            use_template=False,
            target_msa_input=target_msa_input,
            rfd4_coords24=None,
            rfd4_mask24=None,
            boltz_path=pathlib.Path("./params/boltz"),
            num_residue_atoms=24,
        )
    assert writer_out is sentinel_writer
    assert j_msa.features["res_type"].shape == (1, 20, 33)
    assert int(j_msa.features["token_pad_mask"].sum()) == 20
    assert j_msa.features["atom_pad_mask"].shape == (1, 480)
    assert int(j_msa.features["atom_pad_mask"].sum()) == 100
    assert j_msa.features["msa"].shape == (1, 4, 20, 33)
    assert jnp.allclose(j_msa.features["msa"][0, :, 6:20, :], distinct_msa[0])

    with patch(
        "flexcraft.pipelines.rfd4.bind.JoltzSpec.to_input",
        return_value=(base_complex_input, sentinel_writer),
    ):
        with pytest.raises(ValueError, match="target_msa_input is required"):
            prepare_joltz_input(
                binder_seq,
                target_seqs,
                use_template=False,
                target_msa_input=None,
                rfd4_coords24=None,
                rfd4_mask24=None,
                boltz_path=pathlib.Path("./params/boltz"),
            )

    # 2. use_template=True: allocates template slot, populates target [6:20], keeps L=20 tokens & pads atoms to 480
    coords24 = _make_ideal_backbone_coords24(20)
    mask24 = jnp.ones((20, 24), dtype=jnp.bool_)
    with patch(
        "flexcraft.pipelines.rfd4.bind.JoltzSpec.to_input",
        return_value=(base_complex_input, sentinel_writer),
    ):
        j_tmpl, _ = prepare_joltz_input(
            binder_seq,
            target_seqs,
            use_template=True,
            target_msa_input=None,
            rfd4_coords24=coords24,
            rfd4_mask24=mask24,
            boltz_path=pathlib.Path("./params/boltz"),
            num_residue_atoms=24,
        )
    assert j_tmpl.features["res_type"].shape == (1, 20, 33)
    assert int(j_tmpl.features["token_pad_mask"].sum()) == 20
    assert j_tmpl.features["atom_pad_mask"].shape == (1, 480)
    assert j_tmpl.features["template_mask"].shape == (1, 1, 20)
    assert jnp.all(j_tmpl.features["template_mask"][0, 0, :6] == 0.0)
    assert jnp.all(j_tmpl.features["template_mask"][0, 0, 6:20] == 1.0)
    assert jnp.allclose(j_tmpl.features["template_ca"][0, 0, 6:20], coords24[6:20, 1], atol=1e-5)

    with patch(
        "flexcraft.pipelines.rfd4.bind.JoltzSpec.to_input",
        return_value=(base_complex_input, sentinel_writer),
    ):
        with pytest.raises(ValueError, match="rfd4_coords24 and rfd4_mask24 are required"):
            prepare_joltz_input(
                binder_seq,
                target_seqs,
                use_template=True,
                target_msa_input=None,
                rfd4_coords24=None,
                rfd4_mask24=None,
                boltz_path=pathlib.Path("./params/boltz"),
            )


def _build_synthetic_joltz_result(
    chain_lengths: tuple[int, ...],
    *,
    atoms_per_token: int,
    pae_raw: jax.Array,
    pae_logits: jax.Array,
    plddt: jax.Array,
    coords24_samples: jax.Array,
    single_sample: bool,
) -> JoltzResult:
    """Build a pad_atoms(24) JoltzResult for either S=1 (3D samples) or S>1 (4D samples)."""
    num_tokens = sum(chain_lengths)
    j_in = JoltzInput(
        features=_make_joltz_features(chain_lengths, atoms_per_token=atoms_per_token)
    ).pad_atoms(24)
    n_atoms_padded = j_in.features["atom_pad_mask"].shape[1]
    n_valid_atoms = num_tokens * atoms_per_token

    if single_sample:
        flat_valid = coords24_samples[0, :, :atoms_per_token, :].reshape(n_valid_atoms, 3)
        samples_atoms = jnp.zeros((1, n_atoms_padded, 3), dtype=jnp.float32).at[0, :n_valid_atoms, :].set(flat_valid)
        conf = _SyntheticConfidence(
            plddt=plddt[0:1],
            plddt_logits=jnp.zeros((1, num_tokens, 50), dtype=jnp.float32),
            pae=pae_raw[0:1],
            pae_logits=pae_logits[0:1],
        )
    else:
        num_samples = coords24_samples.shape[0]
        flat_valid = coords24_samples[:, :, :atoms_per_token, :].reshape(num_samples, n_valid_atoms, 3)
        samples_atoms = (
            jnp.zeros((num_samples, 1, n_atoms_padded, 3), dtype=jnp.float32)
            .at[:, 0, :n_valid_atoms, :]
            .set(flat_valid)
        )
        conf = _SyntheticConfidence(
            plddt=plddt[:, None, :],
            plddt_logits=jnp.zeros((num_samples, 1, num_tokens, 50), dtype=jnp.float32),
            pae=pae_raw[:, None, :, :],
            pae_logits=pae_logits[:, None, :, :, :],
        )

    return JoltzResult(
        data={
            "features": j_in.features,
            "samples": samples_atoms,
            "distogram": jnp.zeros((num_tokens, num_tokens, 64), dtype=jnp.float32),
            "confidence": conf,
        }
    )


def test_score_prediction_samples_single_multi_compile_reuse_and_rmsd_jit(tmp_path):
    """Verify simplified @jax.jit score_prediction_samples (S=1 & S=4, compile_count==1 across pad_atoms(24), target collapsing, RMSD)."""
    chain_lengths = (4, 4, 4)  # 4 binder (chain 0) + 4 target A (chain 1) + 4 target B (chain 2) = 12 tokens
    num_tokens = 12
    num_samples = 4
    centers = pae_bin_centers(64, 32.0)

    pae_logits_list = []
    pae_raw_list = []
    for bin_choice in (2, 2, 1, 2):
        logits = jnp.full((num_tokens, num_tokens, 64), -20.0, dtype=jnp.float32)
        logits = logits.at[:, :, bin_choice].set(10.0)
        # Huge intra-target PAE (bin 56 ~ 28.25 A) between target chains 1 and 2 — must be excluded!
        logits = logits.at[4:8, 8:12, :].set(-20.0).at[4:8, 8:12, 56].set(10.0)
        logits = logits.at[8:12, 4:8, :].set(-20.0).at[8:12, 4:8, 56].set(10.0)
        expected_raw = (jax.nn.softmax(logits, axis=-1) * centers).sum(axis=-1)
        expected_raw = jnp.fill_diagonal(expected_raw, 0.0, inplace=False)
        pae_logits_list.append(logits)
        pae_raw_list.append(expected_raw)

    pae_logits_4 = jnp.stack(pae_logits_list, axis=0)
    pae_raw_4 = jnp.stack(pae_raw_list, axis=0)
    plddt_4 = jnp.full((num_samples, num_tokens), 0.5, dtype=jnp.float32).at[:, :4].set(0.9)

    rfd4_coords = _make_ideal_backbone_coords24(num_tokens)
    rfd4_mask = jnp.ones((num_tokens, 24), dtype=jnp.bool_)

    # Sample 0..2 match backbone (sidechain atom 5 perturbed on sample 1 -> RMSD stays 0!);
    # Sample 3 has backbone CA perturbed -> RMSD > 0!
    pred_coords_4 = jnp.broadcast_to(rfd4_coords[None], (num_samples, num_tokens, 24, 3))
    pred_coords_4 = pred_coords_4.at[1, :, 5, :].add(25.0)
    pred_coords_4 = pred_coords_4.at[3, 0, 1, 0].add(3.0)

    # 1. Compile reuse check: two complexes at L=12 with different valid atom counts (4 vs 9 atoms/token -> 48 vs 108)
    #    padded with pad_atoms(24) (both align to 288 flat atoms) trigger only 1 XLA compilation!
    j_res_4atoms = _build_synthetic_joltz_result(
        chain_lengths,
        atoms_per_token=4,
        pae_raw=pae_raw_4,
        pae_logits=pae_logits_4,
        plddt=plddt_4,
        coords24_samples=pred_coords_4,
        single_sample=False,
    )
    j_res_9atoms = _build_synthetic_joltz_result(
        chain_lengths,
        atoms_per_token=9,
        pae_raw=pae_raw_4,
        pae_logits=pae_logits_4,
        plddt=plddt_4,
        coords24_samples=pred_coords_4,
        single_sample=False,
    )
    assert j_res_4atoms.data["features"]["atom_pad_mask"].shape == (1, 288)
    assert j_res_9atoms.data["features"]["atom_pad_mask"].shape == (1, 288)
    assert int(j_res_4atoms.atom_pad_mask.sum()) == 48
    assert int(j_res_9atoms.atom_pad_mask.sum()) == 108

    score_prediction_samples.clear_cache()
    metrics_4a = score_prediction_samples(j_res_4atoms, rfd4_coords, rfd4_mask)
    assert score_prediction_samples._cache_size() == 1
    metrics_9a = score_prediction_samples(j_res_9atoms, rfd4_coords, rfd4_mask)
    assert score_prediction_samples._cache_size() == 1

    for key_name in ("min_ipae", "mean_ipae", "ipsae", "iptm", "binder_plddt", "rmsd"):
        assert metrics_4a[key_name].shape == (4,)
        assert jnp.allclose(metrics_4a[key_name], metrics_9a[key_name], atol=1e-5)

    # Intra-target 28.25 A excluded; sidechain perturbation on sample 1 ignored by 4-atom backbone RMSD;
    # CA perturbation on sample 3 detected by RMSD > 0.
    assert jnp.allclose(metrics_9a["min_ipae"], jnp.array([1.25, 1.25, 0.75, 1.25]), atol=1e-3)
    assert jnp.allclose(metrics_9a["mean_ipae"], jnp.array([1.25, 1.25, 0.75, 1.25]), atol=1e-3)
    assert jnp.allclose(metrics_9a["binder_plddt"], 0.9, atol=1e-5)
    assert jnp.allclose(metrics_9a["rmsd"][:3], 0.0, atol=1e-4)
    assert float(metrics_9a["rmsd"][3]) > 0.2

    # Masking residue 0's backbone in rfd4_mask24 excludes sample 3's residue-0 CA perturbation -> RMSD == 0!
    masked_res0 = rfd4_mask.at[0, 1].set(False)
    metrics_masked_bb = score_prediction_samples(j_res_9atoms, rfd4_coords, masked_res0)
    assert jnp.allclose(metrics_masked_bb["rmsd"], 0.0, atol=1e-4)

    # 2. Single-sample S=1 (3D samples) returns 1D (1,) arrays matching sample 0 of S=4
    j_res_single = _build_synthetic_joltz_result(
        chain_lengths,
        atoms_per_token=9,
        pae_raw=pae_raw_4,
        pae_logits=pae_logits_4,
        plddt=plddt_4,
        coords24_samples=pred_coords_4,
        single_sample=True,
    )
    assert j_res_single.is_single_sample
    metrics_single = score_prediction_samples(j_res_single, rfd4_coords, rfd4_mask)
    for key_name in ("min_ipae", "mean_ipae", "ipsae", "iptm", "binder_plddt", "rmsd"):
        assert metrics_single[key_name].shape == (1,)
        assert jnp.allclose(metrics_single[key_name][0], metrics_9a[key_name][0], atol=1e-5)

    # 3. evaluate_binder_prediction aggregation & ScoreCSV logging
    summary, best_idx = evaluate_binder_prediction(
        j_res_9atoms,
        rfd4_coords,
        rfd4_mask,
        f_min_ipae=2.0,
        f_plddt=0.8,
    )
    assert best_idx == 2
    assert summary["success"] is True
    assert summary["min_ipae"] == pytest.approx(1.125, abs=1e-3)
    assert summary["min_ipae_min"] == pytest.approx(0.75, abs=1e-3)
    assert summary["binder_plddt"] == pytest.approx(0.9, abs=1e-4)

    summary_fail, _ = evaluate_binder_prediction(
        j_res_9atoms,
        rfd4_coords,
        rfd4_mask,
        f_min_ipae=1.0,
        f_plddt=0.8,
    )
    assert summary_fail["success"] is False

    csv_path = tmp_path / "scores.csv"
    scores_csv = ScoreCSV(csv_path, keys=list(SCORE_KEYS))
    row = {"attempt": 0, "sequence": "ACDE", **summary}
    scores_csv.write_line(row)
    lines = csv_path.read_text().strip().splitlines()
    assert len(lines) == 2
    assert lines[0].split(",") == list(SCORE_KEYS)


def test_bind_disordered_parser_and_af3_spec():
    """Verify bind_disordered CLI defaults, no template option, interaction flags, and MSA-free AF3LikeSpec."""
    parser = bind_disordered.build_parser()
    option_strings = set(parser._option_string_actions.keys())
    assert "--use_template" not in option_strings
    assert "--no-use_template" not in option_strings

    args = parser.parse_args(["--target_sequence", "MERRYCHRISTMAS", "--num_aa", "72"])
    assert args.target_sequence == "MERRYCHRISTMAS"
    assert args.kind == "aa"
    assert args.hotspots == ""
    assert args.auto_condition is False
    assert args.model == "joltz"
    assert args.cache is None
    assert args.num_designs == 5
    assert args.max_attempts == 200
    assert args.num_aa == 72
    assert args.f_min_ipae == pytest.approx(2.0)
    assert args.f_plddt == pytest.approx(0.8)
    assert args.num_samples == 4
    assert args.num_recycle == 3
    assert args.forbidden == "C"
    assert args.seed == 0

    assert (
        parser.parse_args(["--target_sequence", "ACDE", "--forbidden", "CK"]).forbidden == "CK"
    )
    with pytest.raises(SystemExit):
        bind_disordered.main(["--target_sequence", "MERRYCHRISTMAS", "--forbidden", "B"])

    assert parser.parse_args(["--target_sequence", "ACDE", "--auto_condition"]).auto_condition is True
    assert parser.parse_args(["--target_sequence", "ACDE", "--auto_condition", "True"]).auto_condition is True
    assert parser.parse_args(["--target_sequence", "ACDE", "--auto_condition", "False"]).auto_condition is False

    # Mutual exclusion between --auto_condition and explicit interaction flags in main()
    with pytest.raises(SystemExit):
        bind_disordered.main(["--target_sequence", "MERRYCHRISTMAS", "--auto_condition", "--h_bond", "1"])

    # AF2 with non-protein target raises ValueError
    with pytest.raises(ValueError, match="only supports protein targets"):
        bind_disordered.main(["--target_sequence", "ACGT", "--kind", "dna", "--model", "af2_model_1_ptm"])

    # Verify prepare_af3_spec sets use_msa=False on all chains, no templates, and pad_atoms(24)
    for kind, expected_polymer in (("aa", "protein"), ("dna", "dna"), ("rna", "rna")):
        spec = bind_disordered.prepare_af3_spec("ACDEFG", "ACGU" if kind == "rna" else "ACGT", kind=kind)
        assert len(spec.chains) == 2
        assert spec.chains[0] == {"sequence": "ACDEFG", "kind": "protein", "use_msa": False, "num_repeats": 1}
        assert spec.chains[1]["kind"] == expected_polymer
        assert spec.chains[1]["use_msa"] is False
        assert spec.templates == []
        assert spec._pad_spec == ("pad_atoms", (24,), {})


def test_bind_disordered_end_to_end_mocked_loop_and_af_result_scoring(tmp_path):
    """Verify AFResult.atom24_samples scoring compatibility and mocked bind_disordered.main loop."""
    # 1. AFResult compatibility in score_prediction_samples / evaluate_binder_prediction
    af_in = AF2Spec().add_protein("ACDE:MERR").pad_atoms(24).to_input()
    af_res = AFResult(
        inputs=af_in.data,
        result={
            "structure_module": {
                "final_atom_positions": jnp.ones((8, 37, 3), dtype=jnp.float32),
                "final_atom_mask": jnp.ones((8, 37), dtype=jnp.float32),
            },
            "predicted_lddt": {"logits": jnp.zeros((8, 50), dtype=jnp.float32).at[:, -1].set(10.0)},
            "predicted_aligned_error": {
                "logits": jnp.full((8, 8, 64), -20.0, dtype=jnp.float32).at[:, :, 1].set(10.0),
                "breaks": jnp.linspace(0.0, 31.0, 63),
            },
            "distogram": {
                "logits": jnp.zeros((8, 8, 64), dtype=jnp.float32),
                "bin_edges": jnp.linspace(2.0, 22.0, 63),
            },
            "representations": {
                "msa_first_row": jnp.zeros((8, 256), dtype=jnp.float32),
                "single": jnp.zeros((8, 384), dtype=jnp.float32),
                "pair": jnp.zeros((8, 8, 128), dtype=jnp.float32),
            },
        },
    )
    af_coords24, af_mask24 = af_res.atom24
    af_summary, af_best = evaluate_binder_prediction(af_res, af_coords24, af_mask24)
    assert af_best == 0
    assert af_summary["success"] is True
    assert af_summary["rmsd"] == pytest.approx(0.0, abs=1e-4)

    # 2. Mocked bind_disordered.main loop verifying MSA-free, template-free specs and file outputs
    rfd4_coords = _make_ideal_backbone_coords24(8)
    rfd4_mask = jnp.ones((8, 24), dtype=jnp.bool_)
    centers = pae_bin_centers(64, 32.0)
    logits = jnp.full((2, 8, 8, 64), -20.0, dtype=jnp.float32).at[:, :, :, 1].set(10.0)
    pae_raw = jax.vmap(lambda m: jnp.fill_diagonal(m, 0.0, inplace=False))(
        (jax.nn.softmax(logits, axis=-1) * centers).sum(axis=-1)
    )
    plddt = jnp.full((2, 8), 0.92, dtype=jnp.float32)
    j_res_pass = _build_synthetic_joltz_result(
        (4, 4),
        atoms_per_token=5,
        pae_raw=pae_raw,
        pae_logits=logits,
        plddt=plddt,
        coords24_samples=jnp.broadcast_to(rfd4_coords[None], (2, 8, 24, 3)),
        single_sample=False,
    )

    seen_specs = []

    class _FakeRFD4Result:
        def save_pdb(self, path):
            pathlib.Path(path).write_text("MODEL 1\nENDMDL\n")

        def to_atom24(self):
            return types.SimpleNamespace(
                coords=np.asarray(rfd4_coords),
                mask=np.asarray(rfd4_mask),
                restype=_encode_rfd4_seq("ACDEMERR"),
            )

        def sequence(self, num_aa=None):
            return "ACDE"[:num_aa]

    class _FakePrediction:
        result = j_res_pass

        def save_pdb(self, path, sample_index=0):
            pathlib.Path(path).write_text(f"REMARK sample={sample_index}\nEND\n")

    class _FakeAF3Like:
        def __init__(self, model="joltz", **kwargs):
            self.model = model
            self.kwargs = kwargs

        def predictor(self, **kwargs):
            def _run(key, spec):
                seen_specs.append(spec)
                return _FakePrediction()

            return _run

    out_dir = tmp_path / "disordered_out"
    with (
        patch.object(bind_disordered.RFD4, "from_pretrained", return_value=types.SimpleNamespace(step=lambda: None)),
        patch.object(
            bind_disordered,
            "RFD4Sampler",
            return_value=types.SimpleNamespace(sample=lambda batch, key, recycle_schedule: _FakeRFD4Result()),
        ),
        patch.object(bind_disordered, "AF3Like", _FakeAF3Like),
    ):
        bind_disordered.main(
            [
                "--target_sequence",
                "MERR",
                "--num_aa",
                "4",
                "--num_designs",
                "2",
                "--max_attempts",
                "5",
                "--num_samples",
                "2",
                "--out_path",
                str(out_dir),
            ]
        )

    assert len(seen_specs) == 2
    for spec in seen_specs:
        assert all(c["use_msa"] is False for c in spec.chains)
        assert spec.templates == []
        assert spec._pad_spec == ("pad_atoms", (24,), {})

    assert (out_dir / "attempts" / "design_0.pdb").exists()
    assert (out_dir / "attempts" / "design_1.pdb").exists()
    assert (out_dir / "predictions" / "design_0.pdb").exists()
    assert (out_dir / "predictions" / "design_1.pdb").exists()
    csv_lines = (out_dir / "scores.csv").read_text().strip().splitlines()
    assert len(csv_lines) == 3
    assert csv_lines[0].split(",") == list(SCORE_KEYS)
