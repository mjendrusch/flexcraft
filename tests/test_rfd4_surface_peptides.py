"""Unit and @jax.jit / @eqx.filter_jit tests for flexcraft.pipelines.rfd4.surface_peptides."""

from __future__ import annotations

import csv
import pathlib
import types
from unittest.mock import patch

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from rfd4fx.encoding import UNIFIED_ATOM37_ENCODING
from rfd4fx.flow_matching import RFD4Sampler
from rfd4fx.types import DecoderOutput, TrunkOutput

from flexcraft.pipelines.rfd4 import bind, bind_disordered, common as rfd4_common, surface_peptides as sp
from test_rfd4_bind import (
    _build_synthetic_joltz_result,
    _encode_rfd4_seq,
    _make_ideal_backbone_coords24,
)


class _DummyDecoder:
    def __init__(self, vocab_size: int = 28):
        allowed = np.zeros((4, vocab_size), dtype=bool)
        allowed[0, 1:21] = True
        self.allowed_residue_types = jnp.asarray(allowed)

    def __call__(self, batch):
        bsz, n_tok = batch.padding_mask.shape
        ca_nm = batch.ca_coors_nm
        offsets = jnp.zeros((1, 1, 37, 3), dtype=jnp.float32)
        offsets = offsets.at[:, :, 0, :].set(jnp.array([-0.12, 0.08, 0.0]))
        offsets = offsets.at[:, :, 2, :].set(jnp.array([0.12, 0.09, 0.03]))
        offsets = offsets.at[:, :, 3, :].set(jnp.array([0.15, 0.20, 0.03]))
        coors_nm = ca_nm[:, :, None, :] + offsets
        atom_mask = jnp.zeros((bsz, n_tok, 37), dtype=bool).at[:, :, :5].set(True)
        seq_logits = jnp.zeros((bsz, n_tok, 28), dtype=jnp.float32).at[:, :, 1].set(5.0)
        aatype_max = jnp.where(batch.cond_seq != 0, batch.cond_seq, 1)
        return DecoderOutput(
            coors_nm=coors_nm,
            seq_logits=seq_logits,
            emit_mask=batch.padding_mask,
            aatype_max=aatype_max,
            atom_mask=atom_mask,
            atom_presence_logits=jnp.ones((bsz, n_tok, 37), dtype=jnp.float32),
        )


class _DummyTrunk:
    """Asymmetric trunk model to stress-test Screw symmetrization and latent non-symmetrization."""

    def __call__(self, data, *, n_recycle=None, key=None):
        del n_recycle, key
        bsz, n_tok = data.padding_mask.shape
        idx = jnp.arange(n_tok, dtype=jnp.float32)[None, :, None]
        asymmetric_bias = jnp.concatenate(
            [0.05 * jnp.sin(idx), 0.05 * jnp.cos(idx), 0.02 * idx / jnp.maximum(n_tok, 1)],
            axis=-1,
        )
        v_ca = -0.3 * data.x_t_bb_ca + asymmetric_bias
        v_lat = -0.2 * data.x_t_local_latents + 0.01 * idx
        # Token-dependent logits so different units would pick different AAs unless tied:
        aa_choice = 1 + (jnp.arange(n_tok, dtype=jnp.int32) % 20)
        seq_logits = jax.nn.one_hot(aa_choice, 28, dtype=jnp.float32)[None] * 10.0
        return TrunkOutput(
            bb_ca=jnp.broadcast_to(v_ca, (bsz, n_tok, 3)),
            local_latents=v_lat,
            seq_logits=seq_logits,
        )


class _DummyRFD4Model:
    def __init__(self):
        self.nn = _DummyTrunk()
        self.decoder = _DummyDecoder()

    def step(self):
        from rfd4fx.model import RFD4

        return RFD4.step(self)  # type: ignore[arg-type]

    def decode(self, data, x1_bb, z1, aa_pred=None):
        from rfd4fx.model import RFD4

        return RFD4.decode(self, data, x1_bb, z1, aa_pred=aa_pred)  # type: ignore[arg-type]


def test_cli_parser_defaults_and_validation():
    """Verify CLI flags, defaults, absence of forbidden flags, and upfront validation errors."""
    parser = sp.build_parser()
    args = parser.parse_args([])
    assert args.assembly == "20-50,@0,20-50"
    assert args.assembly_budget == "none"
    assert args.peptide_length == 0
    assert args.peptides == "MERRYCHRISTMAS"
    assert args.p_exposed == pytest.approx(0.9)
    assert args.symmetrize is False
    assert args.sym_count == 1
    assert args.force_loop is False
    assert args.forbidden == "C"
    assert args.gen_only is False
    assert args.model == "protenix-v2"
    assert args.f_sc_rmsd == pytest.approx(2.0)
    assert args.f_plddt == pytest.approx(0.8)
    assert not hasattr(sp, "SurfaceRFD4Sampler")

    # Explicitly forbidden options must not exist on parser:
    for forbidden in ("--sym_radius", "--sym_translation", "--push_motif", "--motif_radius", "--num_aa"):
        with pytest.raises(SystemExit):
            parser.parse_args([forbidden, "1.0"])

    # Boolean variants on --symmetrize, --force_loop, and --gen_only:
    assert parser.parse_args(["--symmetrize", "--sym_count", "3"]).symmetrize is True
    assert parser.parse_args(["--symmetrize", "True", "--sym_count", "3"]).symmetrize is True
    assert parser.parse_args(["--symmetrize", "False"]).symmetrize is False
    assert parser.parse_args(["--symmetrize", "--no-symmetrize"]).symmetrize is False
    assert parser.parse_args(["--force_loop"]).force_loop is True
    assert parser.parse_args(["--force_loop", "True"]).force_loop is True
    assert parser.parse_args(["--force_loop", "False"]).force_loop is False
    assert parser.parse_args(["--force_loop", "--no-force_loop"]).force_loop is False
    assert parser.parse_args(["--forbidden", "CK"]).forbidden == "CK"
    assert parser.parse_args(["--forbidden", ""]).forbidden == ""
    assert parser.parse_args(["--gen_only"]).gen_only is True
    assert parser.parse_args(["--gen_only", "True"]).gen_only is True
    assert parser.parse_args(["--gen_only", "False"]).gen_only is False
    assert parser.parse_args(["--gen_only", "--no-gen_only"]).gen_only is False

    # Upfront parser.error checks in main() before model loading:
    for bad_argv in (
        ["--symmetrize", "--sym_count", "1"],
        ["--p_exposed", "1.2"],
        ["--peptide_length", "-1"],
        ["--forbidden", "X"],
        ["--forbidden", "B"],
        ["--forbidden", "ARNDCQEGHILKMFPSTWYV"],
        ["--assembly", "10,@-1,10"],
        ["--assembly", "10,@2,10"],
        ["--assembly", "20-50,@0,20-50", "--assembly_budget", "50"],
        ["--assembly", "0-5:10,@0,10"],
        ["--assembly", "0,@0,0"],
        ["--assembly", ":".join(["10"] * 27)],
    ):
        with pytest.raises(SystemExit):
            sp.main(bad_argv)


def test_parse_peptides_zero_disables_padding_and_ggs_pads():
    """Verify --peptide_length=0 disables padding and >0 pads symmetrically with GGS linkers."""
    # peptide_length == 0 -> exact input peptides preserved
    assert sp.parse_peptides("ACDE,MERRYCHRISTMAS", peptide_length=0) == [
        "ACDE",
        "MERRYCHRISTMAS",
    ]
    # peptide_length > len(pep) -> symmetric N/C GGS padding; >= peptide_length -> unchanged
    padded = sp.parse_peptides("ACDE,MERRYCHRISTMAS", peptide_length=10)
    assert padded[0] == "GGSACDEGGS"
    assert len(padded[0]) == 10
    assert padded[1] == "MERRYCHRISTMAS"

    with pytest.raises(ValueError, match="non-canonical"):
        sp.parse_peptides("ACDXE", peptide_length=0)


def test_assembly_sampling_budget_and_single_vs_multichain_symmetry():
    """Verify budget N/C padding (including remaining==0) and single- vs multi-chain symmetry."""
    peptides = sp.parse_peptides("ACDEF", peptide_length=7)  # GACAGDEF? -> 'G' + 'ACDEF' + 'G'
    assert len(peptides[0]) == 7

    # 1. Exact budget == chain length (remaining == 0 edge case) + larger budget
    spec_budget = sp.parse_assembly_spec(
        "5,@0,5:4,@0,4",
        peptides,
        assembly_budget="17:20",
    )
    rng = np.random.default_rng(123)
    chains, is_pep, exposed = sp.sample_attempt_chains(spec_budget, 0.5, rng)
    assert [len(c) for c in chains] == [17, 20]
    assert int(is_pep.sum()) == 14
    assert not bool((exposed & ~is_pep).any())

    # Default "20-50,@0,20-50" (min=54, max=114) with assembly_budget="80" caps segments cleanly to 80:
    spec_capped = sp.parse_assembly_spec(
        "20-50,@0,20-50",
        ["MERRYCHRISTMAS"],
        assembly_budget="80",
    )
    for trial in range(20):
        c_seqs, c_pep, _ = sp.sample_attempt_chains(
            spec_capped, 0.9, np.random.default_rng(trial)
        )
        assert len(c_seqs) == 1 and len(c_seqs[0]) == 80
        assert int(c_pep.sum()) == 14

    # 2. Multi-chain C3 symmetry: 3 chains tied identically
    spec_mc = sp.parse_assembly_spec(
        "4-8,@0,4-8:4-8,@0,4-8:4-8,@0,4-8",
        peptides,
        assembly_budget="24:24:24",
        symmetrize=True,
        sym_count=3,
    )
    assert spec_mc.rfd4_pad_to == 3 * 64
    mc_chains, mc_pep, mc_exp = sp.sample_attempt_chains(
        spec_mc, 0.5, np.random.default_rng(7)
    )
    assert len(mc_chains) == 3
    assert mc_chains[0] == mc_chains[1] == mc_chains[2]
    assert np.array_equal(mc_pep[:24], mc_pep[24:48])
    assert np.array_equal(mc_exp[:24], mc_exp[24:48])

    # 3. Single-chain C3 symmetry: 1 chain containing 3 tandem repeats
    spec_sc = sp.parse_assembly_spec(
        "3-6,@0,3-6,3-6,@0,3-6,3-6,@0,3-6",
        peptides,
        symmetrize=True,
        sym_count=3,
    )
    sc_chains, sc_pep, sc_exp = sp.sample_attempt_chains(
        spec_sc, 0.5, np.random.default_rng(11)
    )
    assert len(sc_chains) == 1
    unit_len = len(sc_chains[0]) // 3
    assert sc_chains[0][:unit_len] == sc_chains[0][unit_len : 2 * unit_len]
    assert np.array_equal(sc_pep[:unit_len], sc_pep[unit_len : 2 * unit_len])
    assert np.array_equal(sc_exp[:unit_len], sc_exp[unit_len : 2 * unit_len])


def test_rfd4_input_rasa_slots_and_pytree_invariance():
    """Verify RFD4Input.add_rasa sets C_RAS=3 strictly on peptide sidechains and --force_loop gates C_SEC."""
    chain_seqs = ["XXGKRXX"]
    exposed_all = np.array([False, False, True, True, True, False, False], dtype=bool)
    inp_exp = sp.build_surface_rfd4_input(chain_seqs, exposed_all, pad_to=64, np_seed=42)

    rasa = np.asarray(inp_exp.rasa)[0]
    atom_mask = np.asarray(inp_exp.atom37_mask)[0]
    cond_seq = np.asarray(inp_exp.cond_seq)[0]
    sec_struct = np.asarray(inp_exp.secondary_structure)[0]

    # Default force_loop=False leaves secondary_structure == 0 everywhere
    assert (cond_seq[:2] == 0).all() and (cond_seq[5:] == 0).all()
    assert (cond_seq[2:5] != 0).all()
    assert (sec_struct == 0).all()
    assert (rasa[:2] == 0).all() and (rasa[5:] == 0).all()

    # force_loop=True sets SS_LOOP = 3 strictly on is_peptide residues with identical PyTree
    inp_loop = sp.build_surface_rfd4_input(
        chain_seqs, exposed_all, force_loop=True, pad_to=64, np_seed=42
    )
    sec_loop = np.asarray(inp_loop.secondary_structure)[0]
    assert (sec_loop[:2] == 0).all() and (sec_loop[5:] == 0).all()
    assert (sec_loop[2:5] == 3).all()
    assert jax.tree_util.tree_structure(inp_exp.to_batch()) == jax.tree_util.tree_structure(
        inp_loop.to_batch()
    )

    # GLY at index 2 has NO sidechain -> zero conditioned slots (backbone CA is NOT conditioned)
    assert int((rasa[2] != 0).sum()) == 0
    assert not bool(atom_mask[2].any())

    # LYS (index 3) and ARG (index 4) -> sidechain heavy atoms == 3, backbone N/CA/C/O == 0
    for res_idx, res_name in ((3, "LYS"), (4, "ARG")):
        layout = [str(a).strip() for a in UNIFIED_ATOM37_ENCODING.token_atoms[res_name]]
        for slot_idx, atom_name in enumerate(layout):
            if atom_name and atom_name not in {"N", "CA", "C", "O", "OXT"}:
                assert rasa[res_idx, slot_idx] == 3
                assert bool(atom_mask[res_idx, slot_idx]) is True
            else:
                assert rasa[res_idx, slot_idx] == 0

    # All-False exposed_mask (p_exposed == 0.0) keeps identical FlowBatch PyTree structure
    inp_zero = sp.build_surface_rfd4_input(
        chain_seqs, np.zeros(7, dtype=bool), pad_to=64, np_seed=42
    )
    assert jax.tree_util.tree_structure(inp_exp.to_batch()) == jax.tree_util.tree_structure(
        inp_zero.to_batch()
    )
    assert int((np.asarray(inp_zero.rasa) != 0).sum()) == 0


def test_symmetric_step_60_step_vf_tsr_rollout_and_single_compile():
    """Verify regular RFD4Sampler + symmetric step stays finite, C3-symmetric, preserves user keys, and compiles once."""
    dummy_model = _DummyRFD4Model()
    raw_step = sp.make_surface_peptide_step(dummy_model, symmetrize=True, sym_count=3)  # type: ignore[arg-type]

    compile_counter = {"count": 0}

    def traced_step(model, key, data):
        compile_counter["count"] += 1
        assert data.extra is not None and "some_user_key" in data.extra
        out = raw_step(model, key, data)
        updated_extra = {**out.extra, "some_user_key": out.extra["some_user_key"] + 1}
        return eqx.tree_at(lambda b: b.extra, out, updated_extra)

    sampler = RFD4Sampler(traced_step, dummy_model)  # type: ignore[arg-type]

    # Rollout 1: 3 chains of length 10 (total 30, padded to 192 = 3 * 64)
    seqs_1 = ["XXXGKRXXXX"] * 3
    exp_1 = np.tile(np.array([False, False, False, True, True, True, False, False, False, False]), 3)
    batch_1 = sp.build_surface_rfd4_input(seqs_1, exp_1, pad_to=192, np_seed=1).to_batch()
    batch_1 = eqx.tree_at(
        lambda b: b.extra,
        batch_1,
        {"some_user_key": jnp.asarray(10, dtype=jnp.int32)},
        is_leaf=lambda l: l is None,
    )
    res_1 = sampler.sample(batch_1, key=jax.random.key(0), nsteps=60, jit_decode=False)

    assert compile_counter["count"] == 1
    assert int(res_1.data.extra["some_user_key"]) == 70
    ca_final = np.asarray(res_1.x["bb_ca"])[0, :30]
    ca_x1 = np.asarray(res_1.x_1["bb_ca"])[0, :30]
    lat_final = np.asarray(res_1.x["local_latents"])[0, :30]
    assert np.isfinite(ca_final).all()
    assert np.isfinite(ca_x1).all()

    # Exact 120-degree Z-axis C3 rotational symmetry on both x["bb_ca"] and x_1["bb_ca"]:
    angle = 2.0 * np.pi / 3.0
    rot_z = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    for arr in (ca_final, ca_x1):
        u0, u1, u2 = arr[:10], arr[10:20], arr[20:30]
        np.testing.assert_allclose(u1, u0 @ rot_z, atol=1e-5)
        np.testing.assert_allclose(u2, u0 @ (rot_z @ rot_z), atol=1e-5)

    # Local latents MUST NOT be symmetrized or replicated across units:
    assert not np.allclose(lat_final[:10], lat_final[10:20], atol=1e-3)

    # Sequences across all 3 symmetric units are tied and preserve grafted "GKR":
    designed_seqs = rfd4_common.extract_chain_sequences(res_1, [10, 10, 10])
    assert designed_seqs[0] == designed_seqs[1] == designed_seqs[2]
    assert designed_seqs[0][3:6] == "GKR"

    # Rollout 2: different unit length (3 chains of length 14 -> total 42, same pad_to=192) -> NO recompile!
    seqs_2 = ["XXXXXGKRXXXXXX"] * 3
    exp_2 = np.zeros(42, dtype=bool)
    batch_2 = sp.build_surface_rfd4_input(seqs_2, exp_2, pad_to=192, np_seed=2).to_batch()
    batch_2 = eqx.tree_at(
        lambda b: b.extra,
        batch_2,
        {"some_user_key": jnp.asarray(0, dtype=jnp.int32)},
        is_leaf=lambda l: l is None,
    )
    res_2 = sampler.sample(batch_2, key=jax.random.key(1), nsteps=60, jit_decode=False)
    assert compile_counter["count"] == 1
    assert int(res_2.data.extra["some_user_key"]) == 60
    ca2 = np.asarray(res_2.x_1["bb_ca"])[0, :42]
    np.testing.assert_allclose(ca2[14:28], ca2[:14] @ rot_z, atol=1e-5)


def test_prepare_af3_spec_multichain_vs_singlechain_symmetry():
    """Verify num_repeats=sym_count is applied ONLY to multi-chain symmetric designs, NEVER single-chain."""
    # Multi-chain symmetric homooligomer (3 chains, sym_count=3) -> 1 chain spec entry with count=3
    spec_mc = sp.prepare_surface_af3_spec(
        ["ACDEFGHIK", "ACDEFGHIK", "ACDEFGHIK"],
        model="protenix-v2",
        max_total_length=27,
        variable_length=False,
        symmetrize=True,
        sym_count=3,
    )
    assert len(spec_mc.chains) == 1
    assert spec_mc.chains[0]["num_repeats"] == 3
    assert spec_mc.chains[0]["use_msa"] is False

    # Single-chain symmetric design (1 chain, sym_count=3) -> MUST remain 1 chain with num_repeats=1!
    spec_sc = sp.prepare_surface_af3_spec(
        ["ACDEFGHIKACDEFGHIKACDEFGHIK"],
        model="protenix-v2",
        max_total_length=27,
        variable_length=False,
        symmetrize=True,
        sym_count=3,
    )
    assert len(spec_sc.chains) == 1
    assert spec_sc.chains[0]["num_repeats"] == 1
    assert spec_sc.chains[0]["use_msa"] is False


def test_evaluate_excludes_peptide_handles_single_sample_and_c4_permutations():
    """Verify peptide exclusion, S=1 3D handling, incomplete backbone guard, and C4 (0,2,1,3) assignment."""
    # 1. Peptide perturbation (+50 A, pLDDT=0.1, PAE=30 A) is completely excluded from scaffold metrics:
    num_tokens = 16
    is_peptide = np.array([False] * 5 + [True] * 6 + [False] * 5, dtype=bool)
    rfd4_coords = _make_ideal_backbone_coords24(num_tokens)
    rfd4_mask = jnp.zeros((num_tokens, 24), dtype=jnp.bool_).at[:, :5].set(True)

    pred_coords = jnp.broadcast_to(rfd4_coords[None], (2, num_tokens, 24, 3)).copy()
    pred_coords = pred_coords.at[:, 5:11, :, :].add(50.0)

    plddt = jnp.where(jnp.asarray(is_peptide)[None, :], 0.10, 0.92)
    plddt = jnp.broadcast_to(plddt, (2, num_tokens))
    scaf_pair = (~jnp.asarray(is_peptide))[:, None] & (~jnp.asarray(is_peptide))[None, :]
    pae_raw = jnp.where(scaf_pair[None, :, :], 1.5, 30.0)
    pae_raw = jnp.broadcast_to(pae_raw, (2, num_tokens, num_tokens))
    pae_logits = jax.nn.one_hot(
        jnp.where(scaf_pair[None], 2, 60), 64, dtype=jnp.float32
    )
    pae_logits = jnp.broadcast_to(pae_logits, (2, num_tokens, num_tokens, 64))

    j_res_multi = _build_synthetic_joltz_result(
        (num_tokens,),
        atoms_per_token=5,
        pae_raw=pae_raw,
        pae_logits=pae_logits,
        plddt=plddt,
        coords24_samples=pred_coords,
        single_sample=False,
    )
    summary, best_idx = sp.evaluate_surface_peptide_prediction(
        j_res_multi,
        rfd4_coords,
        rfd4_mask,
        is_peptide,
        num_chains=1,
    )
    assert best_idx == 0
    assert summary["sc_rmsd"] == pytest.approx(0.0, abs=1e-4)
    assert summary["plddt"] == pytest.approx(0.92, abs=1e-4)
    assert summary["motif_plddt"] == pytest.approx(0.10, abs=1e-4)
    assert summary["mean_pae"] < 2.0
    assert summary["success"] is True

    # 2. Single-sample (3D atom24_samples) support:
    j_res_single = _build_synthetic_joltz_result(
        (num_tokens,),
        atoms_per_token=5,
        pae_raw=pae_raw[:1],
        pae_logits=pae_logits[:1],
        plddt=plddt[:1],
        coords24_samples=pred_coords[:1],
        single_sample=True,
    )
    summary_s1, _ = sp.evaluate_surface_peptide_prediction(
        j_res_single, rfd4_coords, rfd4_mask, is_peptide, num_chains=1
    )
    assert summary_s1["sc_rmsd"] == pytest.approx(0.0, abs=1e-4)
    assert summary_s1["success"] is True

    # 3. Incomplete (CA-only) scaffold backbone must yield sc_rmsd = inf and success = False:
    broken_mask = jnp.zeros((num_tokens, 24), dtype=jnp.bool_).at[:, 1].set(True)
    summary_broken, _ = sp.evaluate_surface_peptide_prediction(
        j_res_single, rfd4_coords, broken_mask, is_peptide, num_chains=1
    )
    assert np.isinf(summary_broken["sc_rmsd"])
    assert summary_broken["success"] is False

    # 4. C4 non-dihedral permutation (0, 2, 1, 3): 0.0 A for multi-chain C4, >0 A for single-chain C4!
    unit_coords = _make_ideal_backbone_coords24(6).at[:, :4, 0].add(8.0)
    c4_units = []
    for k_idx in range(4):
        ang = k_idx * (np.pi / 2.0)
        rot = jnp.array(
            [[np.cos(ang), -np.sin(ang), 0.0], [np.sin(ang), np.cos(ang), 0.0], [0.0, 0.0, 1.0]],
            dtype=jnp.float32,
        )
        c4_units.append(jnp.einsum("lad,de->lae", unit_coords, rot))
    rfd4_c4 = jnp.concatenate(c4_units, axis=0)  # (24, 24, 3)
    pred_c4 = jnp.concatenate([c4_units[0], c4_units[2], c4_units[1], c4_units[3]], axis=0)[None]
    is_pep_c4 = np.tile(np.array([False, False, True, True, False, False]), 4)
    mask_c4 = jnp.zeros((24, 24), dtype=jnp.bool_).at[:, :5].set(True)

    j_c4 = _build_synthetic_joltz_result(
        (6, 6, 6, 6),
        atoms_per_token=5,
        pae_raw=jnp.ones((1, 24, 24), dtype=jnp.float32),
        pae_logits=jax.nn.one_hot(jnp.full((1, 24, 24), 2), 64, dtype=jnp.float32),
        plddt=jnp.full((1, 24), 0.95, dtype=jnp.float32),
        coords24_samples=pred_c4,
        single_sample=True,
    )
    sum_mc_c4, _ = sp.evaluate_surface_peptide_prediction(
        j_c4, rfd4_c4, mask_c4, is_pep_c4, num_chains=4, symmetrize=True, sym_count=4
    )
    assert sum_mc_c4["sc_rmsd"] == pytest.approx(0.0, abs=1e-4)
    assert sum_mc_c4["success"] is True

    # For single-chain C4 (num_chains=1), subunits are covalently ordered so (0,2,1,3) is NOT equivalent:
    sum_sc_c4, _ = sp.evaluate_surface_peptide_prediction(
        j_c4, rfd4_c4, mask_c4, is_pep_c4, num_chains=1, symmetrize=True, sym_count=4
    )
    assert sum_sc_c4["sc_rmsd"] > 5.0


def test_main_end_to_end_mocked(tmp_path: pathlib.Path):
    """Verify main() writes attempts/, predictions/, and scores.csv with negative --seed support."""
    total_len = 14
    seq_str = "AAAAGMERRYAAAA"
    coords24 = np.asarray(_make_ideal_backbone_coords24(total_len), dtype=np.float32)
    mask24 = np.zeros((total_len, 24), dtype=bool)
    mask24[:, :5] = True

    class _FakeRFD4Result:
        def save_pdb(self, path: pathlib.Path) -> None:
            path.write_text("MODEL 1\nENDMDL\n")

        def sequence(self) -> str:
            return seq_str

        def to_atom24(self):
            return types.SimpleNamespace(
                coords=coords24,
                mask=mask24,
                restype=_encode_rfd4_seq(seq_str),
            )

    class _FakeSampler:
        def __init__(self, step_fn, model):
            self.step_fn = step_fn
            self.model = model

        def sample(self, batch, *, key, recycle_schedule):
            del batch, key, recycle_schedule
            return _FakeRFD4Result()

    j_res = _build_synthetic_joltz_result(
        (total_len,),
        atoms_per_token=5,
        pae_raw=jnp.ones((2, total_len, total_len), dtype=jnp.float32),
        pae_logits=jax.nn.one_hot(jnp.full((2, total_len, total_len), 2), 64, dtype=jnp.float32),
        plddt=jnp.full((2, total_len), 0.94, dtype=jnp.float32),
        coords24_samples=jnp.broadcast_to(jnp.asarray(coords24)[None], (2, total_len, 24, 3)),
        single_sample=False,
    )

    class _FakePrediction:
        result = j_res

        def save_pdb(self, path: pathlib.Path, sample_index: int = 0) -> None:
            path.write_text(f"SAMPLE {sample_index}\n")

    class _FakeAF3Like:
        def __init__(self, model: str, **kwargs):
            self.model = model
            self.kwargs = kwargs

        def predictor(self, **kwargs):
            del kwargs
            return lambda key, spec: _FakePrediction()

    out_dir = tmp_path / "run_sp"
    with (
        patch.object(sp.RFD4, "from_pretrained", return_value=_DummyRFD4Model()),
        patch.object(sp, "RFD4Sampler", _FakeSampler),
        patch.object(sp, "AF3Like", _FakeAF3Like),
    ):
        sp.main(
            [
                "--peptides",
                "MERRY",
                "--peptide_length",
                "6",
                "--assembly",
                "4,@0,4",
                "--p_exposed",
                "0.9",
                "--num_designs",
                "2",
                "--max_attempts",
                "5",
                "--seed",
                "-7",
                "--out_path",
                str(out_dir),
            ]
        )

    assert (out_dir / "attempts" / "design_0.pdb").is_file()
    assert (out_dir / "predictions" / "design_0.pdb").is_file()
    with open(out_dir / "scores.csv", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert all(r["success"] == "True" for r in rows)
    assert rows[0]["sequence"] == seq_str


def test_gen_only_skips_predictor_and_scores_across_rfd4_pipelines(tmp_path: pathlib.Path):
    """Verify --gen_only True skips structure predictors, predictions/, and scores.csv in all 3 pipelines."""

    class _FakeGenResult:
        def save_pdb(self, path: pathlib.Path) -> None:
            pathlib.Path(path).write_text("MODEL 1\nENDMDL\n")

    class _FakeSampler:
        def __init__(self, step_fn, model):
            self.step_fn = step_fn
            self.model = model

        def sample(self, batch, *, key, recycle_schedule):
            del batch, key, recycle_schedule
            return _FakeGenResult()

    class _ForbiddenPredictor:
        def __init__(self, *args, **kwargs):
            raise AssertionError("Structure predictor must NOT be initialized when --gen_only True")

    # 1. surface_peptides --gen_only True
    sp_out = tmp_path / "sp_gen"
    with (
        patch.object(sp.RFD4, "from_pretrained", return_value=_DummyRFD4Model()),
        patch.object(sp, "RFD4Sampler", _FakeSampler),
        patch.object(sp, "AF3Like", _ForbiddenPredictor),
    ):
        sp.main(
            [
                "--peptides",
                "MERRY",
                "--assembly_budget",
                "60",
                "--p_exposed",
                "0.5",
                "--num_designs",
                "3",
                "--gen_only",
                "True",
                "--out_path",
                str(sp_out),
            ]
        )
    assert [p.name for p in sorted((sp_out / "attempts").glob("*.pdb"))] == [
        "design_0.pdb",
        "design_1.pdb",
        "design_2.pdb",
    ]
    assert not (sp_out / "predictions").exists()
    assert not (sp_out / "scores.csv").exists()

    # 2. bind_disordered --gen_only True
    bd_out = tmp_path / "bd_gen"
    with (
        patch.object(bind_disordered.RFD4, "from_pretrained", return_value=_DummyRFD4Model()),
        patch.object(bind_disordered, "RFD4Sampler", _FakeSampler),
        patch.object(bind_disordered, "AF3Like", _ForbiddenPredictor),
    ):
        bind_disordered.main(
            [
                "--target_sequence",
                "MERRY",
                "--num_aa",
                "16",
                "--num_designs",
                "2",
                "--gen_only",
                "True",
                "--out_path",
                str(bd_out),
            ]
        )
    assert [p.name for p in sorted((bd_out / "attempts").glob("*.pdb"))] == [
        "design_0.pdb",
        "design_1.pdb",
    ]
    assert not (bd_out / "predictions").exists()
    assert not (bd_out / "scores.csv").exists()

    # 3. bind --gen_only True
    fake_inp = types.SimpleNamespace(
        padding_mask=np.ones((1, 20), dtype=bool),
        cond_seq=np.concatenate(
            [np.zeros((1, 10), dtype=np.int32), np.ones((1, 10), dtype=np.int32)],
            axis=1,
        ),
        chain_iid=np.concatenate(
            [np.zeros((1, 10), dtype=np.float32), np.ones((1, 10), dtype=np.float32)],
            axis=1,
        ),
        pad_to_nearest=lambda mult: types.SimpleNamespace(to_batch=lambda: object()),
    )
    b_out = tmp_path / "b_gen"
    with (
        patch.object(bind.RFD4Input, "from_chains", return_value=fake_inp),
        patch.object(bind.RFD4, "from_pretrained", return_value=_DummyRFD4Model()),
        patch.object(bind, "RFD4Sampler", _FakeSampler),
        patch.object(bind, "Joltz2", _ForbiddenPredictor),
    ):
        bind.main(
            [
                "--in_path",
                "dummy.pdb",
                "--num_aa",
                "10",
                "--num_designs",
                "2",
                "--gen_only",
                "True",
                "--out_path",
                str(b_out),
            ]
        )
    assert [p.name for p in sorted((b_out / "attempts").glob("*.pdb"))] == [
        "design_0.pdb",
        "design_1.pdb",
    ]
    assert not (b_out / "predictions").exists()
    assert not (b_out / "scores.csv").exists()


def test_forbid_amino_acids_sets_c_seq_prob_zero_only_on_designable_residues():
    """Verify --forbidden sets C_SEQ_PROB_<AA>=0.0 on designed residues while preserving motifs/peptides."""
    from rfd4fx.featurize import SEQ_PROB_ORDER
    from rfd4fx.nn.features import SequenceProbabilitiesSeqFeat
    from rfd4fx.rfd4_input import RFD4Input

    cys_idx = SEQ_PROB_ORDER.index("condition_sequence-probability-CYS_1_residue")
    lys_idx = SEQ_PROB_ORDER.index("condition_sequence-probability-LYS_1_residue")

    # 1. surface_peptides with KTCKQSGTCPPDVVPKVEG (contains both C and K) and --forbidden "CK"
    peptide = "KTCKQSGTCPPDVVPKVEG"
    cond_seq_str = ("X" * 5) + peptide + ("X" * 6)
    is_pep = np.array([False] * 5 + [True] * len(peptide) + [False] * 6, dtype=bool)
    inp_ck = sp.build_surface_rfd4_input(
        [cond_seq_str],
        np.zeros(len(cond_seq_str), dtype=bool),
        is_peptide=is_pep,
        forbidden="ck",
        pad_to=64,
        np_seed=0,
    )
    probs = np.asarray(inp_ck.seq_probabilities)[0]
    designable_idx = np.flatnonzero(~is_pep)  # [0..4, 24..29]

    # Designable scaffold tokens: 0.0 on CYS and LYS, NaN on all other 26 channels
    assert (probs[designable_idx, cys_idx] == 0.0).all()
    assert (probs[designable_idx, lys_idx] == 0.0).all()
    other_channels = [i for i in range(28) if i not in (cys_idx, lys_idx)]
    assert np.isnan(probs[np.ix_(designable_idx, other_channels)]).all()

    # Peptide tokens: preserve exact one-hot CYS (indices 7, 13) and LYS (indices 5, 8, 20)
    for pos, ch in enumerate(cond_seq_str):
        if is_pep[pos]:
            assert np.isfinite(probs[pos]).all()
            if ch == "C":
                assert probs[pos, cys_idx] == pytest.approx(1.0)
            elif ch == "K":
                assert probs[pos, lys_idx] == pytest.approx(1.0)

    # Padded tokens (30..63): remain all NaN
    assert np.isnan(probs[len(cond_seq_str) :]).all()

    # SequenceProbabilitiesSeqFeat emits (prob=0.0, mask=1.0) strictly on CYS/LYS for designable tokens
    feat = np.asarray(SequenceProbabilitiesSeqFeat()(inp_ck.to_batch()))[0]
    feat_probs, feat_masks = feat[:, :28], feat[:, 28:]
    assert (feat_probs[designable_idx] == 0.0).all()
    assert (feat_masks[designable_idx, cys_idx] == 1.0).all()
    assert (feat_masks[designable_idx, lys_idx] == 1.0).all()
    assert (feat_masks[np.ix_(designable_idx, other_channels)] == 0.0).all()

    # 2. Disabling via "" or "none" leaves designable tokens all-NaN while keeping PyTree identical
    for empty_val in ("", "none"):
        inp_none = sp.build_surface_rfd4_input(
            [cond_seq_str],
            np.zeros(len(cond_seq_str), dtype=bool),
            is_peptide=is_pep,
            forbidden=empty_val,
            pad_to=64,
            np_seed=0,
        )
        probs_none = np.asarray(inp_none.seq_probabilities)[0]
        assert np.isnan(probs_none[designable_idx]).all()
        assert jax.tree_util.tree_structure(inp_ck.to_batch()) == jax.tree_util.tree_structure(
            inp_none.to_batch()
        )

    # 3. bind_disordered style: binder (0..7) + disordered target with 'C' ("MERRYCHRISTMAS") + pad_to_nearest(64)
    inp_bd = RFD4Input.from_chains(8, {"kind": "aa", "sequence": "MERRYCHRISTMAS"})
    binder_mask = np.zeros(int(inp_bd.padding_mask.shape[1]), dtype=bool)
    binder_mask[:8] = True
    inp_bd = rfd4_common.forbid_amino_acids(inp_bd, "C", designable_mask=binder_mask).pad_to_nearest(64)
    probs_bd = np.asarray(inp_bd.seq_probabilities)[0]
    assert (probs_bd[:8, cys_idx] == 0.0).all()
    assert np.isnan(probs_bd[:8, [i for i in range(28) if i != cys_idx]]).all()
    assert probs_bd[8 + 5, cys_idx] == pytest.approx(1.0)  # 'C' in MERRYCHRISTMAS
    assert np.isnan(probs_bd[22:]).all()

    # 4. Direct RFD4Input.add_probability + RFD4Input.forbid (AF2_CODE order, <0 -> NaN, silent skip on cond_seq!=0, jit)
    raw = RFD4Input.from_chains(6, {"kind": "aa", "sequence": "CKW"}, pad_to=16)
    p_mat = np.full((9, 20), -1.0, dtype=np.float32)
    p_mat[:, 0] = 0.35  # ALA = index 0 in AF2_CODE
    edited = raw.add_probability(p_mat, where=True, start=0).forbid("CK")
    sp_ed = np.asarray(edited.seq_probabilities)[0]
    np.testing.assert_allclose(sp_ed[:6, 0], 0.35)
    assert (sp_ed[:6, cys_idx] == 0.0).all()
    assert (sp_ed[:6, lys_idx] == 0.0).all()
    # Conditioned "CKW" (tokens 6..8) silently skipped -> retains 1.0 on CYS, LYS, TRP:
    assert sp_ed[6, cys_idx] == pytest.approx(1.0)
    assert sp_ed[7, lys_idx] == pytest.approx(1.0)
    assert np.isnan(sp_ed[9:]).all()

    @eqx.filter_jit
    def _jit_prob_and_forbid(inp_obj, probs_20, mask_1d):
        return inp_obj.add_probability(probs_20, where=mask_1d).forbid("W", where=mask_1d)

    jit_res = _jit_prob_and_forbid(
        raw,
        jnp.full((6, 20), -1.0, dtype=jnp.float32).at[:, 0].set(0.5),
        jnp.array([True, False, True, False, True, False]),
    )
    sp_jit = np.asarray(jit_res.seq_probabilities)[0]
    assert sp_jit[0, 0] == pytest.approx(0.5)
    assert sp_jit[0, SEQ_PROB_ORDER.index("condition_sequence-probability-TRP_1_residue")] == 0.0
    assert np.isnan(sp_jit[1]).all()
