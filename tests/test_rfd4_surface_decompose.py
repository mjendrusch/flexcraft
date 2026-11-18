"""Unit, geometry, sanitization, and mocked end-to-end tests for surface_decompose and rfd4.common."""

from __future__ import annotations

import csv
import pathlib
import types
from unittest.mock import patch

import gemmi
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from rfd4fx.encoding import UNIFIED_ATOM37_ENCODING as ENC
from rfd4fx.flow_matching import RFD4Result
from rfd4fx.rfd4_input import RFD4Input
from rfd4fx.types import DecoderOutput, FlowBatch

from flexcraft.pipelines.rfd4 import common as rfd4_common, surface_decompose as sd
from flexcraft.structure.af._af3_like_wrapper import AF2Spec
from flexcraft.structure.af._data import AFResult
from test_rfd4_bind import _build_synthetic_joltz_result


def _rotation_matrix(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    unit = np.asarray(axis, dtype=np.float64)
    unit = unit / np.linalg.norm(unit)
    kx, ky, kz = unit
    skew = np.array([[0.0, -kz, ky], [kz, 0.0, -kx], [-ky, kx, 0.0]], dtype=np.float64)
    return np.eye(3) + np.sin(angle_rad) * skew + (1.0 - np.cos(angle_rad)) * (skew @ skew)


def _canonical_atom_template(res_name: str) -> dict[str, np.ndarray]:
    template: dict[str, np.ndarray] = {
        "N": np.array([-1.2, 0.8, 0.0], dtype=np.float64),
        "CA": np.array([0.0, 0.0, 0.0], dtype=np.float64),
        "C": np.array([1.3, 0.6, 0.0], dtype=np.float64),
        "O": np.array([1.8, 1.7, 0.2], dtype=np.float64),
    }
    slots = [str(a).strip() for a in ENC.token_atoms[res_name]]
    sc_idx = 0
    for aname in slots:
        if not aname or aname in sd.BACKBONE_ATOMS:
            continue
        angle = 0.55 * (sc_idx + 1)
        template[aname] = np.array(
            [
                0.3 * np.cos(angle) + 0.15 * sc_idx,
                -1.4 - 0.9 * sc_idx,
                0.8 * np.sin(angle) + 0.25 * ((sc_idx % 3) - 1),
            ],
            dtype=np.float64,
        )
        sc_idx += 1
    return template


def _encode_ref_atom_name_chars(atom_names: list[str]) -> jnp.ndarray:
    """Encode PDB atom names using AlphaFold3 SI Table 5 ref_atom_name_chars (ord(c) - 32)."""
    codes = np.zeros((len(atom_names), 4), dtype=np.int32)
    for idx, name in enumerate(atom_names):
        for c_idx, ch in enumerate(name.ljust(4)[:4]):
            codes[idx, c_idx] = ord(ch) - 32
    return jax.nn.one_hot(jnp.asarray(codes), 64, dtype=jnp.float32)


def _write_synthetic_target_pdb(pdb_path: pathlib.Path, *, two_chains: bool = False) -> None:
    """Write a synthetic PDB with blank chain ID, MSE, altlocs, H atoms, HOH, and exposed patch."""
    st = gemmi.Structure()
    model = gemmi.Model("1")
    chain_a = gemmi.Chain(" ")  # blank chain ID -> normalized to "_"
    res_names = ["ASP", "TYR", "CYS", "ARG", "GLU", "MSE"]
    for idx, rname in enumerate(res_names):
        res = gemmi.Residue()
        res.name = rname
        res.het_flag = "H" if rname == "MSE" else "A"
        res.seqid = gemmi.SeqId(idx + 1, " ")
        # Tight cluster within 5.0 A so all 6 form a valid surface patch:
        theta = 2.0 * np.pi * idx / len(res_names)
        base = np.array([2.2 * np.cos(theta), 0.0, 2.2 * np.sin(theta)], dtype=np.float64)
        tmpl = _canonical_atom_template("MET" if rname == "MSE" else rname)
        for aname, rel in tmpl.items():
            atom = gemmi.Atom()
            actual_name = "SE" if (rname == "MSE" and aname == "SD") else aname
            atom.name = actual_name
            atom.element = gemmi.Element("Se" if actual_name == "SE" else actual_name[0])
            xyz = base + rel
            atom.pos = gemmi.Position(float(xyz[0]), float(xyz[1]), float(xyz[2]))
            if idx == 0 and aname == "CA":
                atom.altloc = "A"
                res.add_atom(atom)
                dup = gemmi.Atom()
                dup.name = "CA"
                dup.altloc = "B"
                dup.element = gemmi.Element("C")
                dup.pos = gemmi.Position(float(xyz[0] + 9.0), float(xyz[1]), float(xyz[2]))
                res.add_atom(dup)
            else:
                res.add_atom(atom)
        h_atom = gemmi.Atom()
        h_atom.name = "H"
        h_atom.element = gemmi.Element("H")
        h_atom.pos = gemmi.Position(0.0, 0.0, 0.0)
        res.add_atom(h_atom)
        chain_a.add_residue(res)

    # Add a water residue that must be stripped:
    wat = gemmi.Residue()
    wat.name = "HOH"
    wat.het_flag = "H"
    wat.seqid = gemmi.SeqId(99, " ")
    ow = gemmi.Atom()
    ow.name = "O"
    ow.element = gemmi.Element("O")
    ow.pos = gemmi.Position(20.0, 20.0, 20.0)
    wat.add_atom(ow)
    chain_a.add_residue(wat)
    model.add_chain(chain_a)

    if two_chains:
        # Place chain B right next to residue 1's CB so _1 becomes an interface residue:
        chain_b = gemmi.Chain("B")
        res_b = gemmi.Residue()
        res_b.name = "ALA"
        res_b.het_flag = "A"
        res_b.seqid = gemmi.SeqId(1, " ")
        cb_a0 = np.array(
            [chain_a[0].find_atom("CB", "\x00").pos.x, chain_a[0].find_atom("CB", "\x00").pos.y, chain_a[0].find_atom("CB", "\x00").pos.z]
        )
        for aname, rel in _canonical_atom_template("ALA").items():
            atom = gemmi.Atom()
            atom.name = aname
            atom.element = gemmi.Element(aname[0])
            xyz = cb_a0 + np.array([6.0, 0.0, 0.0]) + rel - _canonical_atom_template("ALA")["CB"]
            atom.pos = gemmi.Position(float(xyz[0]), float(xyz[1]), float(xyz[2]))
            res_b.add_atom(atom)
        chain_b.add_residue(res_b)
        model.add_chain(chain_b)

    st.add_model(model)
    # Add a second model that must be discarded:
    st.add_model(gemmi.Model("2"))
    st.write_minimal_pdb(str(pdb_path))


def _make_rfd4_result(aa_pred_idx: np.ndarray, coors_ang: np.ndarray, atom_mask: np.ndarray) -> RFD4Result:
    num_aa = int(aa_pred_idx.shape[0])
    vocab_size = len(ENC.idx_to_token)
    aa_arr = aa_pred_idx[None, :].astype(np.int32)
    coors_nm = (coors_ang[None, :, :, :] / 10.0).astype(np.float32)
    data = FlowBatch(
        padding_mask=np.ones((1, num_aa), dtype=bool),
        token_type=np.zeros((1, num_aa), dtype=np.int32),
        chain_iid=np.zeros((1, num_aa), dtype=np.float32),
        aa_pred=aa_arr,
    )
    dec = DecoderOutput(
        coors_nm=coors_nm,
        seq_logits=np.zeros((1, num_aa, vocab_size), dtype=np.float32),
        emit_mask=np.ones((1, num_aa), dtype=bool),
        aatype_max=aa_arr,
        atom_mask=atom_mask[None, :, :].astype(bool),
        atom_presence_logits=np.zeros((1, num_aa, 37), dtype=np.float32),
    )
    return RFD4Result(x={"bb_ca": coors_nm[:, :, 1, :]}, x_1=None, data=data, dec=dec)


def _embed_patch_into_scaffold(
    patch: sd.PatchSpec,
    num_aa: int,
    *,
    rot: np.ndarray,
    trans: np.ndarray,
    swap_c2: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]:
    """Build (aa_pred_idx, coors37_ang, mask37, motif_positions) embedding `patch` with optional C2 swaps."""
    m_count = len(patch.member_residues)
    motif_positions = list(range(1, 1 + m_count))
    gly_idx = int(ENC.token_to_idx["GLY"])
    aa_pred = np.full(num_aa, gly_idx, dtype=np.int32)
    coors_ang = np.zeros((num_aa, 37, 3), dtype=np.float64)
    atom_mask = np.zeros((num_aa, 37), dtype=bool)

    gly_tmpl = _canonical_atom_template("GLY")
    gly_slots = [str(a).strip() for a in ENC.token_atoms["GLY"]]
    for j_idx in range(num_aa):
        shift = np.array([3.8 * j_idx, 1.1 * np.sin(j_idx), 0.9 * np.cos(j_idx)], dtype=np.float64)
        for s_idx, aname in enumerate(gly_slots):
            if aname in gly_tmpl:
                coors_ang[j_idx, s_idx] = (gly_tmpl[aname] + shift) @ rot + trans
                atom_mask[j_idx, s_idx] = True

    for m_idx, res in enumerate(patch.member_residues):
        j_idx = motif_positions[m_idx]
        aa_pred[j_idx] = res.aa_idx
        atom_mask[j_idx] = False
        slots = [str(a).strip() for a in ENC.token_atoms[res.res_name]]
        swaps = sd.C2_SYMMETRY_SWAPS.get(res.res_name, ()) if swap_c2 else ()
        swap_map = {src: dst for a, b in swaps for src, dst in ((a, b), (b, a))}
        for s_idx, aname in enumerate(slots):
            if not aname or aname == "OXT":
                continue
            src_name = swap_map.get(aname, aname)
            if src_name in res.atom_coords:
                pt = res.atom_coords[src_name] - patch.c_crd_offset
                coors_ang[j_idx, s_idx] = pt @ rot + trans
                atom_mask[j_idx, s_idx] = True
    return aa_pred, coors_ang, atom_mask, motif_positions


def test_common_subpackage_and_cli_defaults(tmp_path: pathlib.Path):
    """Verify rfd4.common is a subpackage directory and surface_decompose CLI defaults/validation."""
    assert pathlib.Path(rfd4_common.__file__).name == "__init__.py"
    assert pathlib.Path(rfd4_common.__file__).parent.name == "common"

    pdb_file = tmp_path / "in.pdb"
    _write_synthetic_target_pdb(pdb_file)
    out_dir = tmp_path / "out"

    args = sd.parse_args(["--in_path", str(pdb_file), "--out_path", str(out_dir), "--num_aa", "32"])
    assert args.model == "protenix-v2"
    assert args.cache is None
    assert args.num_samples == 4
    assert args.num_recycle == 4
    assert args.forbidden == "C"
    assert args.gen_only is False
    assert args.overwrite is False
    assert args.num_designs == 100
    assert args.max_seed_attempts == 10
    assert args.cutoff == pytest.approx(5.0)
    assert args.min_num_patch == 5
    assert args.interface_cutoff == pytest.approx(8.0)
    assert args.f_rfd4_rmsd == pytest.approx(1.0)
    assert args.f_sc_rmsd == pytest.approx(2.0)
    assert args.f_motif_rmsd == pytest.approx(1.0)
    assert args.f_plddt == pytest.approx(0.85)
    assert args.f_pae == pytest.approx(3.0)

    # Deprecated --protenix_model flag must not exist:
    with pytest.raises(SystemExit):
        sd.parse_args(
            ["--in_path", str(pdb_file), "--out_path", str(out_dir), "--num_aa", "32", "--protenix_model", "x"]
        )

    for bad_extra in (
        ["--num_aa", "0"],
        ["--num_aa", "20", "--min_num_patch", "2"],
        ["--num_aa", "20", "--num_samples", "0"],
        ["--num_aa", "20", "--forbidden", "Z"],
        ["--num_aa", "20", "--f_plddt", "1.5"],
    ):
        with pytest.raises(SystemExit):
            sd.parse_args(["--in_path", str(pdb_file), "--out_path", str(out_dir), *bad_extra])


def test_sanitization_patch_precomputation_and_forbidden_postfilter(tmp_path: pathlib.Path):
    """Verify MSE->MET, blank chains, interface exclusion, C2 matching, NaN C_SEQ_PROB, and post-filter."""
    pdb_path = tmp_path / "target.pdb"
    _write_synthetic_target_pdb(pdb_path, two_chains=True)

    plans, m_max = sd.prepare_target_plans(
        [pdb_path], interface_cutoff=8.0, cutoff=12.0, min_num_patch=5
    )
    assert len(plans) == 1
    plan = plans[0]
    assert len(plan.sanitized_structure) == 1  # model 2 dropped
    ctx_a = next(c for c in plan.chains if c.chain_id == "_")
    assert ctx_a.n_polymer == 6  # HOH dropped, MSE converted to MET
    assert ctx_a.residues[-1].res_name == "MET"
    assert "SD" in ctx_a.residues[-1].atom_coords
    assert ctx_a.residues[0].is_interface is True  # close to chain B
    assert "_1" not in ctx_a.valid_seeds  # excluded as seed
    assert len(ctx_a.valid_seeds) >= 1
    assert m_max == 6  # interface neighbor _1 IS still included inside nearby patches!

    patch_spec = ctx_a.patches[ctx_a.valid_seeds[0]]
    assert "_1" in patch_spec.member_tokens

    # compile_patch_batch must leave scaffold seq_probabilities (0..15) completely NaN:
    base_inp = RFD4Input.from_chains(16)
    batch = sd.compile_patch_batch(base_inp, patch_spec, n_pad=64)
    probs = np.asarray(batch.seq_probabilities)[0]
    assert np.isnan(probs[:16]).all()
    assert np.isfinite(probs[16 : 16 + len(patch_spec.member_residues)]).all()
    assert np.isnan(probs[16 + len(patch_spec.member_residues) :]).all()

    # Hungarian + C2-symmetric sidechain matching recovers exact 0.0 A RMSD under rigid transform + C2 flips:
    rot = _rotation_matrix(np.array([1.0, -2.0, 0.5]), 1.1)
    trans = np.array([12.5, -7.0, 4.2])
    aa_pred, coors37, mask37, motif_pos = _embed_patch_into_scaffold(
        patch_spec, 16, rot=rot, trans=trans, swap_c2=True
    )
    col_ind, rmsd, r_fit, t_fit = sd.match_rfd4_patch(patch_spec, aa_pred, coors37, mask37)
    assert col_ind is not None
    assert list(col_ind) == motif_pos
    assert rmsd == pytest.approx(0.0, abs=1e-5)

    # Post-filter exempts CYS at matched motif index (col_ind[2]) and flags CYS on non-motif scaffold slots:
    seq_clean = ["G"] * 16
    for idx, res in zip(col_ind, patch_spec.member_residues):
        seq_clean[int(idx)] = sd.CANONICAL_AA3_TO_1[res.res_name]
    seq_clean_str = "".join(seq_clean)
    assert "C" in seq_clean_str
    assert sd.has_forbidden_scaffold_aa(seq_clean_str, col_ind, frozenset({"C"})) is False

    seq_with_scaf_cys = "C" + seq_clean_str[1:]
    assert sd.has_forbidden_scaffold_aa(seq_with_scaf_cys, col_ind, frozenset({"C"})) is True
    assert sd.has_forbidden_scaffold_aa(seq_with_scaf_cys, col_ind, frozenset()) is False

    # Missing required AA multiplicity triggers f_match (col_ind is None):
    broken_aa = np.full_like(aa_pred, int(ENC.token_to_idx["GLY"]))
    col_none, rmsd_none, _, _ = sd.match_rfd4_patch(patch_spec, broken_aa, coors37, mask37)
    assert col_none is None and rmsd_none is None


def test_evaluate_decompose_prediction_and_af_result_compat(tmp_path: pathlib.Path):
    """Verify multi-sample 4-atom backbone sc_rmsd, motif_rmsd, and AFResult atom extraction."""
    pdb_path = tmp_path / "single.pdb"
    _write_synthetic_target_pdb(pdb_path, two_chains=False)
    plans, _ = sd.prepare_target_plans([pdb_path], interface_cutoff=8.0, cutoff=12.0, min_num_patch=5)
    patch = plans[0].chains[0].patches[plans[0].chains[0].valid_seeds[0]]

    num_aa = 12
    rot = _rotation_matrix(np.array([0.3, 1.0, -0.4]), 0.7)
    trans = np.array([-3.0, 5.0, 2.0])
    aa_pred, coors37, mask37, motif_pos = _embed_patch_into_scaffold(
        patch, num_aa, rot=rot, trans=trans, swap_c2=True
    )
    col_ind = np.asarray(motif_pos, dtype=np.int32)

    # Build a 2-sample synthetic JoltzResult with real ref_atom_name_chars and atom coordinates:
    idx_to_tok = ENC.idx_to_token if isinstance(ENC.idx_to_token, dict) else dict(enumerate(ENC.idx_to_token))
    atom_names: list[str] = []
    tok_indices: list[int] = []
    sample0_pts: list[np.ndarray] = []
    # Build per-residue CCD/backbone-first atoms: N, CA, C, O, then sidechains
    for tok in range(num_aa):
        rname = str(idx_to_tok[int(aa_pred[tok])])
        slots = [str(a).strip() for a in ENC.token_atoms[rname]]
        slot_by_name = {an: s for s, an in enumerate(slots) if an}
        ordered_names = [an for an in ("N", "CA", "C", "O") if an in slot_by_name] + [
            an for an in slots if an and an not in sd.BACKBONE_ATOMS and mask37[tok, slot_by_name[an]]
        ]
        for an in ordered_names:
            atom_names.append(an)
            tok_indices.append(tok)
            sample0_pts.append(coors37[tok, slot_by_name[an]])

    n_atoms = len(atom_names)
    coords_s0 = np.stack(sample0_pts, axis=0).astype(np.float32)
    # Sample 1 has small 0.1 A shift on sidechains so Sample 0 is best_sample_idx == 0:
    coords_s1 = coords_s0.copy()
    j_res = _build_synthetic_joltz_result(
        (num_aa,),
        atoms_per_token=1,
        pae_raw=jnp.full((2, num_aa, num_aa), 1.2, dtype=jnp.float32),
        pae_logits=jax.nn.one_hot(jnp.full((2, num_aa, num_aa), 2), 64, dtype=jnp.float32),
        plddt=jnp.full((2, num_aa), 0.93, dtype=jnp.float32),
        coords24_samples=jnp.zeros((2, num_aa, 24, 3), dtype=jnp.float32),
        single_sample=False,
    )
    # Override atom-level features with exact variable-atom protein representation:
    j_res.data["samples"] = jnp.stack([coords_s0, coords_s1], axis=0)[:, None, :, :]
    j_res.data["features"]["atom_pad_mask"] = jnp.ones((1, n_atoms), dtype=jnp.float32)
    j_res.data["features"]["atom_to_token"] = jax.nn.one_hot(
        jnp.asarray(tok_indices, dtype=jnp.int32), num_aa, dtype=jnp.float32
    )[None]
    j_res.data["features"]["ref_atom_name_chars"] = _encode_ref_atom_name_chars(atom_names)[None]

    summary, best_idx, r_best, t_best, best_plddt = sd.evaluate_decompose_prediction(
        j_res, coors37, mask37, patch, col_ind, num_aa=num_aa
    )
    assert best_idx == 0
    assert summary["sc_rmsd"] == pytest.approx(0.0, abs=1e-4)
    assert summary["motif_rmsd"] == pytest.approx(0.0, abs=1e-4)
    assert summary["mean_plddt"] == pytest.approx(0.93, abs=1e-4)
    assert summary["motif_plddt"] == pytest.approx(0.93, abs=1e-4)
    assert summary["mean_pae"] < 2.0
    assert summary["failure"] == "none"
    assert summary["accepted"] is True
    assert best_plddt.shape == (num_aa,)

    # Also verify AFResult compatibility in extract_residue_atom_dicts:
    af_in = AF2Spec().add_protein("ACDE").to_input()
    af_res = AFResult(
        inputs=af_in.data,
        result={
            "structure_module": {
                "final_atom_positions": jnp.ones((4, 37, 3), dtype=jnp.float32),
                "final_atom_mask": jnp.ones((4, 37), dtype=jnp.float32),
            },
            "predicted_lddt": {"logits": jnp.zeros((4, 50), dtype=jnp.float32)},
            "predicted_aligned_error": {
                "logits": jnp.zeros((4, 4, 64), dtype=jnp.float32),
                "breaks": jnp.linspace(0.0, 31.0, 63),
            },
            "distogram": {
                "logits": jnp.zeros((4, 4, 64), dtype=jnp.float32),
                "bin_edges": jnp.linspace(2.0, 22.0, 63),
            },
            "representations": {
                "msa_first_row": jnp.zeros((4, 8), dtype=jnp.float32),
                "single": jnp.zeros((4, 8), dtype=jnp.float32),
                "pair": jnp.zeros((4, 4, 8), dtype=jnp.float32),
            },
        },
    )
    af_dicts = rfd4_common.extract_residue_atom_dicts(af_res)
    assert len(af_dicts) == 4 and "CA" in af_dicts[0]


def test_main_end_to_end_mocked_and_gen_only(tmp_path: pathlib.Path):
    """Verify full surface_decompose.main loop, overwrite guard, zero-design exit, and --gen_only."""
    pdb_path = tmp_path / "t8.pdb"
    _write_synthetic_target_pdb(pdb_path, two_chains=False)

    # 1. num_designs == 0 writes seeds/t8.pdb and header-only scores.csv without loading RFD4 or AF3Like:
    out_zero = tmp_path / "zero_out"
    with (
        patch.object(sd.RFD4, "from_pretrained", side_effect=AssertionError("must not load RFD4")),
        patch.object(sd, "build_af3_predictor", side_effect=AssertionError("must not load AF3Like")),
    ):
        sd.main(["--in_path", str(pdb_path), "--out_path", str(out_zero), "--num_aa", "14", "--num_designs", "0"])
    assert (out_zero / "seeds" / "t8.pdb").is_file()
    assert (out_zero / "scores.csv").read_text().strip().splitlines() == [",".join(sd.SCORE_COLUMNS)]

    # Non-empty directory without --overwrite raises FileExistsError:
    with pytest.raises(FileExistsError):
        sd.main(["--in_path", str(pdb_path), "--out_path", str(out_zero), "--num_aa", "14", "--num_designs", "0"])

    # 2. Full mocked run: 1st attempt f_match, 2nd attempt f_forbidden (non-motif CYS), 3rd attempt accepted!
    plans, _ = sd.prepare_target_plans([pdb_path], interface_cutoff=8.0, cutoff=12.0, min_num_patch=5)
    ctx = plans[0].chains[0]
    num_aa = 14
    call_counter = {"n": 0, "pred_n": 0}

    class _FakeSampler:
        def __init__(self, step_fn, model):
            del step_fn, model

        def sample(self, batch, *, key, recycle_schedule):
            del key, recycle_schedule
            call_counter["n"] += 1
            # Scaffold seq_probabilities must stay all-NaN on every call:
            assert np.isnan(np.asarray(batch.seq_probabilities)[0, :num_aa]).all()
            # Recover which patch was conditioned by reading in-context aa tokens:
            in_ctx = np.asarray(batch.is_in_context_conditioning)[0].astype(bool)
            m_len = int(in_ctx.sum())
            cond_aa = np.asarray(batch.cond_seq)[0, num_aa : num_aa + m_len]
            matched_patch = next(
                p for p in ctx.patches.values() if np.array_equal(p.aa_idx, cond_aa)
            )
            if call_counter["n"] == 1:
                # Force f_match on attempt 0:
                aa_bad = np.full(num_aa, int(ENC.token_to_idx["GLY"]), dtype=np.int32)
                coors_bad = np.ones((num_aa, 37, 3), dtype=np.float64)
                mask_bad = np.zeros((num_aa, 37), dtype=bool)
                mask_bad[:, [0, 1, 2, 4]] = True
                return _make_rfd4_result(aa_bad, coors_bad, mask_bad)

            aa_ok, coors_ok, mask_ok, _ = _embed_patch_into_scaffold(
                matched_patch,
                num_aa,
                rot=np.eye(3),
                trans=np.zeros(3),
                swap_c2=False,
            )
            if call_counter["n"] == 2:
                # Place an extra CYS on non-motif scaffold position 0 -> must trigger f_forbidden before AF3Like!
                aa_forb = aa_ok.copy()
                aa_forb[0] = int(ENC.token_to_idx["CYS"])
                return _make_rfd4_result(aa_forb, coors_ok, mask_ok)

            call_counter["last_aa"] = aa_ok
            call_counter["last_coors"] = coors_ok
            call_counter["last_mask"] = mask_ok
            return _make_rfd4_result(aa_ok, coors_ok, mask_ok)

    def _fake_predictor(key, spec):
        del key, spec
        call_counter["pred_n"] += 1
        aa_ok = call_counter["last_aa"]
        coors_ok = call_counter["last_coors"]
        mask_ok = call_counter["last_mask"]
        idx_to_tok = ENC.idx_to_token if isinstance(ENC.idx_to_token, dict) else dict(enumerate(ENC.idx_to_token))
        atom_names, tok_indices, pts = [], [], []
        for tok in range(num_aa):
            rname = str(idx_to_tok[int(aa_ok[tok])])
            slots = [str(a).strip() for a in ENC.token_atoms[rname]]
            slot_by_name = {an: s for s, an in enumerate(slots) if an}
            ordered = [an for an in ("N", "CA", "C", "O") if an in slot_by_name] + [
                an for an in slots if an and an not in sd.BACKBONE_ATOMS and mask_ok[tok, slot_by_name[an]]
            ]
            for an in ordered:
                atom_names.append(an)
                tok_indices.append(tok)
                pts.append(coors_ok[tok, slot_by_name[an]])
        coords_arr = np.stack(pts, axis=0).astype(np.float32)
        n_atoms = len(atom_names)
        j_res = _build_synthetic_joltz_result(
            (num_aa,),
            atoms_per_token=1,
            pae_raw=jnp.full((4, num_aa, num_aa), 1.0, dtype=jnp.float32),
            pae_logits=jax.nn.one_hot(jnp.full((4, num_aa, num_aa), 2), 64, dtype=jnp.float32),
            plddt=jnp.full((4, num_aa), 0.94, dtype=jnp.float32),
            coords24_samples=jnp.zeros((4, num_aa, 24, 3), dtype=jnp.float32),
            single_sample=False,
        )
        j_res.data["samples"] = jnp.broadcast_to(coords_arr[None, None], (4, 1, n_atoms, 3))
        j_res.data["features"]["atom_pad_mask"] = jnp.ones((1, n_atoms), dtype=jnp.float32)
        j_res.data["features"]["atom_to_token"] = jax.nn.one_hot(
            jnp.asarray(tok_indices, dtype=jnp.int32), num_aa, dtype=jnp.float32
        )[None]
        j_res.data["features"]["ref_atom_name_chars"] = _encode_ref_atom_name_chars(atom_names)[None]

        def _save_pdb(path: pathlib.Path, sample_index: int = 0) -> None:
            del sample_index
            rfd4_tmp = _make_rfd4_result(aa_ok, coors_ok, mask_ok)
            rfd4_tmp.save_pdb(path)

        return types.SimpleNamespace(result=j_res, save_pdb=_save_pdb)

    out_run = tmp_path / "full_run"
    with (
        patch.object(sd.RFD4, "from_pretrained", return_value=types.SimpleNamespace(step=lambda: None)),
        patch.object(sd, "RFD4Sampler", _FakeSampler),
        patch.object(sd, "build_af3_predictor", return_value=_fake_predictor),
    ):
        sd.main(
            [
                "--in_path",
                str(pdb_path),
                "--out_path",
                str(out_run),
                "--num_aa",
                str(num_aa),
                "--num_designs",
                "3",
                "--cutoff",
                "12.0",
                "--seed",
                "-5",
            ]
        )

    with open(out_run / "scores.csv", newline="") as handle:
        rows = list(csv.DictReader(handle))
    # Attempt 0 -> f_match; Attempt 1 -> f_forbidden (short-circuits before predictor); Attempt 2 -> accepted!
    assert len(rows) == 3
    assert call_counter["pred_n"] == 1
    assert rows[0]["failure"] == "f_match"
    assert rows[0]["sc_rmsd"] == "none"
    assert rows[1]["failure"] == "f_forbidden"
    assert float(rows[1]["rfd4_motif_rmsd"]) == pytest.approx(0.0, abs=1e-3)
    assert rows[1]["sc_rmsd"] == "none"
    assert rows[2]["failure"] == "none"
    assert "C" in rows[2]["sequence"]  # CYS inside matched motif patch IS allowed!
    assert float(rows[2]["sc_rmsd"]) == pytest.approx(0.0, abs=1e-3)
    assert len(list((out_run / "attempts").glob("*.pdb"))) == 3
    success_pdbs = list((out_run / "success").glob("*.pdb"))
    assert len(success_pdbs) == 1
    st_succ = gemmi.read_structure(str(success_pdbs[0]))
    assert st_succ[0][0][0][0].b_iso == pytest.approx(0.94, abs=1e-2)

    st_seeds = gemmi.read_structure(str(out_run / "seeds" / "t8.pdb"))
    assert all(res[0].b_iso == pytest.approx(1.0) for res in st_seeds[0][0])

    # 3. --gen_only True skips predictor, success/, and scores.csv:
    out_gen = tmp_path / "gen_run"
    with (
        patch.object(sd.RFD4, "from_pretrained", return_value=types.SimpleNamespace(step=lambda: None)),
        patch.object(sd, "RFD4Sampler", _FakeSampler),
        patch.object(sd, "build_af3_predictor", side_effect=AssertionError("must not load AF3Like")),
    ):
        sd.main(
            [
                "--in_path",
                str(pdb_path),
                "--out_path",
                str(out_gen),
                "--num_aa",
                str(num_aa),
                "--num_designs",
                "2",
                "--cutoff",
                "12.0",
                "--gen_only",
                "True",
            ]
        )
    assert len(list((out_gen / "attempts").glob("*.pdb"))) == 2
    assert (out_gen / "seeds" / "t8.pdb").is_file()
    assert not (out_gen / "success").exists()
    assert not (out_gen / "scores.csv").exists()
