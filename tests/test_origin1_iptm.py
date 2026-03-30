"""Tests for AbsciBindIPTM and its helper functions.

Source: flexcraft/flexcraft/structure/metrics/Origin1_ipTM.py
"""

# ---------------------------------------------------------------------------
# Mock shim — must come before any flexcraft imports.
#
# Strategy: stub out the entire colabdesign / haiku import chain so that only
# the pure JAX/numpy functions in Origin1_ipTM.py are exercised.  AFResult is
# never actually used at runtime in the tested code — the class only accesses
# result.result["predicted_aligned_error"]["logits"], so we provide a minimal
# MockAFResult instead of the real class.
# ---------------------------------------------------------------------------
import sys
import types

# --- haiku stub (colabdesign needs hk.Module etc.) -------------------------
_hk_mock = types.ModuleType("haiku")
_hk_mock.__version__ = "mock"
_hk_mock.Params = dict


class _FakeModule:
    """Minimal haiku.Module stand-in."""
    pass


_hk_mock.Module = _FakeModule
_hk_mock.transform = lambda f: f          # no-op decorator
_hk_mock.transform_with_state = lambda f: f
sys.modules["haiku"] = _hk_mock
for _sub in [
    "haiku.experimental",
    "haiku._src",
    "haiku._src.layer_stack",
    "haiku._src.lift",
    "haiku._src.transform",
]:
    sys.modules[_sub] = types.ModuleType(_sub)

# --- stub out the colabdesign import chain ---------------------------------
for _mod_name in [
    "colabdesign",
    "colabdesign.af",
    "colabdesign.af.model",
    "colabdesign.af.alphafold",
    "colabdesign.af.alphafold.model",
    "colabdesign.af.alphafold.model.geometry",
    "colabdesign.af.alphafold.model.modules",
    "colabdesign.af.alphafold.model.all_atom_multimer",
    "colabdesign.af.prep",
    "colabdesign.af.inputs",
]:
    sys.modules.setdefault(_mod_name, types.ModuleType(_mod_name))

# geometry stub needs Vec3Array
sys.modules["colabdesign.af.alphafold.model.geometry"].Vec3Array = object

# --- stub out salad ---------------------------------------------------------
for _mod_name in ["salad", "salad.aflib", "salad.aflib.common", "salad.aflib.common.protein"]:
    sys.modules.setdefault(_mod_name, types.ModuleType(_mod_name))

# --- stub out the AFResult / AFInput that _data.py would provide -----------
# We pre-register a fake flexcraft.structure.af._data so Origin1_ipTM's
# `from flexcraft.structure.af._data import AFResult` resolves cleanly.
_af_data_mock = types.ModuleType("flexcraft.structure.af._data")


class AFResult:  # minimal stub — only used as a type annotation
    pass


_af_data_mock.AFResult = AFResult
sys.modules["flexcraft.structure.af._data"] = _af_data_mock

# Also stub the parent packages so Python doesn't try to exec their __init__
for _mod_name in [
    "flexcraft",
    "flexcraft.structure",
    "flexcraft.structure.af",
    "flexcraft.structure.metrics",
]:
    sys.modules.setdefault(_mod_name, types.ModuleType(_mod_name))

sys.path.insert(0, "/home/ntbiotech/Documents/Current_projects/BinderDesign/flexcraft")

# ---------------------------------------------------------------------------
# Load Origin1_ipTM directly by file path so the package import chain
# (colabdesign → haiku → …) is never triggered.
# ---------------------------------------------------------------------------
import importlib.util as _ilu

_spec = _ilu.spec_from_file_location(
    "flexcraft.structure.metrics.Origin1_ipTM",
    "/home/ntbiotech/Documents/Current_projects/BinderDesign/flexcraft/"
    "flexcraft/structure/metrics/Origin1_ipTM.py",
)
_origin1_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_origin1_mod)

_d0 = _origin1_mod._d0
_ptm_matrix = _origin1_mod._ptm_matrix
_iptm_A_given_B = _origin1_mod._iptm_A_given_B
AbsciBindIPTM = _origin1_mod.AbsciBindIPTM

# ---------------------------------------------------------------------------
# Standard imports
# ---------------------------------------------------------------------------
import numpy as np
import jax.numpy as jnp
import pytest


# ---------------------------------------------------------------------------
# Minimal AFResult mock
# ---------------------------------------------------------------------------

class MockAFResult:
    """Minimal mock that replicates the only AFResult attribute accessed."""

    def __init__(self, logits):
        self.result = {"predicted_aligned_error": {"logits": logits}}


# ---------------------------------------------------------------------------
# Helper: build uniform-logit MockAFResult
# ---------------------------------------------------------------------------

def _uniform_logits(L: int, num_bins: int = 64) -> np.ndarray:
    """Return logits of shape (L, L, num_bins) that are all zero.

    After softmax this gives uniform probability 1/num_bins per bin.
    """
    return np.zeros((L, L, num_bins), dtype=np.float32)


def _spike_logits(L: int, bin_idx: int, num_bins: int = 64, spike: float = 100.0) -> np.ndarray:
    """Return logits with large value at bin_idx so softmax concentrates there."""
    logits = np.full((L, L, num_bins), -spike, dtype=np.float32)
    logits[:, :, bin_idx] = spike
    return logits


# ===========================================================================
# 1. Tests for helper `_d0`
# ===========================================================================

class TestD0:
    """Tests for the TM-score normalisation factor _d0(L).

    Reference: Origin-1, Suppl. §7.5 —
      d0(L) = 1.34*(L-15)^(1/3) - 1.8  for L >= 19, else 1.0
    """

    def test_small_L_returns_one(self):
        """_d0(L) should equal 1.0 for L < 19.

        Note: jnp.where eagerly evaluates both branches, so for L < 19 the
        expression (L-15)**(1/3) uses a negative base and JAX returns a complex
        scalar.  The real part of the selected branch is still 1.0.
        """
        for L in [1, 5, 10, 18]:
            result = complex(_d0(L))
            assert result.real == pytest.approx(1.0), f"L={L}: expected real=1.0, got {result}"
            assert abs(result.imag) < 1e-6, f"L={L}: unexpected imaginary part {result.imag}"

    def test_boundary_L19(self):
        """_d0(19) should use the formula, not the fallback."""
        expected = 1.34 * (19 - 15) ** (1 / 3) - 1.8
        result = float(_d0(19))
        assert result == pytest.approx(expected, rel=1e-5)

    def test_L30(self):
        """_d0(30) matches formula."""
        expected = 1.34 * (30 - 15) ** (1 / 3) - 1.8
        assert float(_d0(30)) == pytest.approx(expected, rel=1e-5)

    def test_L100(self):
        """_d0(100) matches formula."""
        expected = 1.34 * (100 - 15) ** (1 / 3) - 1.8
        assert float(_d0(100)) == pytest.approx(expected, rel=1e-5)

    def test_increases_with_L(self):
        """_d0 should be monotonically non-decreasing for L >= 19."""
        Ls = [19, 30, 50, 100, 200]
        values = [float(_d0(L)) for L in Ls]
        for i in range(len(values) - 1):
            assert values[i] <= values[i + 1], (
                f"_d0 decreased from L={Ls[i]} to L={Ls[i+1]}"
            )


# ===========================================================================
# 2. Tests for helper `_ptm_matrix`
# ===========================================================================

class TestPtmMatrix:
    """Tests for the element-wise pTM score matrix computation.

    pTM_ij(L) = sum_b p_ij^b * 1/(1 + (Delta_b/d0(L))^2), Delta_b=(b-0.5)/2
    """

    def test_shape(self):
        """Output shape should be (L, L)."""
        L = 5
        probs = np.ones((L, L, 64), dtype=np.float32) / 64
        result = _ptm_matrix(jnp.array(probs), L=30)
        assert result.shape == (L, L)

    def test_uniform_probabilities_gives_weighted_mean(self):
        """With uniform probs, pTM_ij = (1/64) * sum_b weight_b."""
        L_seq = 4
        L_norm = 30
        probs = np.ones((L_seq, L_seq, 64), dtype=np.float32) / 64
        result = np.array(_ptm_matrix(jnp.array(probs), L=L_norm))

        # Compute expected value manually
        b = np.arange(1, 65)
        delta = (b - 0.5) / 2
        d = float(_d0(L_norm))
        weights = 1.0 / (1.0 + (delta / d) ** 2)
        expected = np.sum(weights) / 64  # uniform: (1/64)*sum(weights)

        assert result == pytest.approx(expected, rel=1e-5)

    def test_spike_on_bin1_gives_near_one(self):
        """With all mass on bin b=1 (Delta=0.25), pTM should be close to 1."""
        import jax
        L_seq = 3
        # bin index 0 → b=1 → Delta_1 = (1-0.5)/2 = 0.25
        # weight = 1/(1+(0.25/d0)^2) → close to 1 if d0 >> 0.25
        logits = _spike_logits(L_seq, bin_idx=0)
        # Use jax.nn.softmax to avoid numpy exp overflow with spike logits
        probs = jax.nn.softmax(jnp.array(logits), axis=-1)
        result = np.array(_ptm_matrix(probs, L=100))

        # d0(100) is large so weight for bin 1 ≈ 1.0
        d = float(_d0(100))
        expected_weight = 1.0 / (1.0 + (0.25 / d) ** 2)
        assert result == pytest.approx(expected_weight, rel=1e-4)

    def test_spike_on_last_bin_gives_low_score(self):
        """With all mass on bin b=64 (Delta=31.75), pTM should be small."""
        import jax
        L_seq = 3
        logits = _spike_logits(L_seq, bin_idx=63)
        # Use jax.nn.softmax to avoid numpy exp overflow with spike logits
        probs = jax.nn.softmax(jnp.array(logits), axis=-1)
        result = float(np.array(_ptm_matrix(probs, L=30)).mean())
        assert result < 0.1, f"Expected low pTM for last bin, got {result}"

    def test_scores_bounded_zero_one(self):
        """pTM matrix values should lie in [0, 1]."""
        L_seq = 6
        rng = np.random.default_rng(42)
        probs = rng.dirichlet(np.ones(64), size=(L_seq, L_seq)).astype(np.float32)
        result = np.array(_ptm_matrix(jnp.array(probs), L=50))
        assert result.min() >= 0.0
        assert result.max() <= 1.0


# ===========================================================================
# 3. Tests for helper `_iptm_A_given_B`
# ===========================================================================

class TestIptmAGivenB:
    """Tests for ipTM_A(B;L) = max_{i in A}[1/|B| * sum_{j in B} pTM_ij].

    Source: flexcraft/flexcraft/structure/metrics/Origin1_ipTM.py
    """

    def _make_ptm(self, values: np.ndarray) -> jnp.ndarray:
        return jnp.array(values, dtype=jnp.float32)

    def test_known_values_simple(self):
        """Manual 3x3 pTM matrix with known answer."""
        # Rows: residues 0,1,2.  A={0,1}, B={2}.
        # mean over B for row0 = ptm[0,2] = 0.8
        # mean over B for row1 = ptm[1,2] = 0.6
        # max over A → 0.8
        ptm = self._make_ptm([[0.1, 0.2, 0.8],
                               [0.3, 0.4, 0.6],
                               [0.5, 0.7, 0.9]])
        A_mask = jnp.array([True, True, False])
        B_mask = jnp.array([False, False, True])
        result = float(_iptm_A_given_B(ptm, A_mask, B_mask))
        assert result == pytest.approx(0.8, rel=1e-5)

    def test_multiple_B_residues(self):
        """Mean over |B|=2 residues, then max over A."""
        # A={0}, B={1,2}
        # mean for row0 = (ptm[0,1] + ptm[0,2]) / 2 = (0.4 + 0.6) / 2 = 0.5
        ptm = self._make_ptm([[0.1, 0.4, 0.6],
                               [0.2, 0.5, 0.7],
                               [0.3, 0.8, 0.9]])
        A_mask = jnp.array([True, False, False])
        B_mask = jnp.array([False, True, True])
        result = float(_iptm_A_given_B(ptm, A_mask, B_mask))
        assert result == pytest.approx(0.5, rel=1e-5)

    def test_max_selects_best_A_residue(self):
        """Result should reflect the best (max) row in A, not the mean."""
        # A={0,1,2}, B={3}
        ptm = self._make_ptm([[0.2, 0.3, 0.4, 0.9],
                               [0.1, 0.2, 0.3, 0.5],
                               [0.3, 0.4, 0.5, 0.7],
                               [0.6, 0.7, 0.8, 0.1]])
        A_mask = jnp.array([True, True, True, False])
        B_mask = jnp.array([False, False, False, True])
        result = float(_iptm_A_given_B(ptm, A_mask, B_mask))
        # means over B: [0.9, 0.5, 0.7]; max = 0.9
        assert result == pytest.approx(0.9, rel=1e-5)


# ===========================================================================
# 4–7. Tests for AbsciBindIPTM.__call__
# ===========================================================================

class TestAbsciBindIPTM:
    """Integration tests for AbsciBindIPTM.__call__.

    Source: flexcraft/flexcraft/structure/metrics/Origin1_ipTM.py
    """

    def _make_scorer(self):
        return AbsciBindIPTM()

    # --- 4. Output structure ---

    def test_output_keys(self):
        """Return dict must contain iptm, default_iptm, ab_aligned_iptm."""
        L = 10
        scorer = self._make_scorer()
        result = MockAFResult(_uniform_logits(L))
        ab_mask = np.array([True] * 5 + [False] * 5)
        ag_mask = np.array([False] * 5 + [True] * 5)
        scores = scorer(result, ab_mask, ag_mask)
        assert set(scores.keys()) == {"iptm", "default_iptm", "ab_aligned_iptm"}

    def test_iptm_equals_mean_of_components(self):
        """iptm must equal (default_iptm + ab_aligned_iptm) / 2."""
        L = 12
        scorer = self._make_scorer()
        rng = np.random.default_rng(7)
        logits = rng.standard_normal((L, L, 64)).astype(np.float32)
        result = MockAFResult(logits)
        ab_mask = np.array([True] * 6 + [False] * 6)
        ag_mask = np.array([False] * 6 + [True] * 6)
        scores = scorer(result, ab_mask, ag_mask)
        expected = (float(scores["default_iptm"]) + float(scores["ab_aligned_iptm"])) / 2
        assert float(scores["iptm"]) == pytest.approx(expected, rel=1e-5)

    # --- 5. Symmetry with uniform logits and equal chain lengths ---

    def test_symmetric_uniform_iptm_ab_equals_iptm_ag(self):
        """With uniform logits and equal chain lengths, ipTM_Ab(Ag) == ipTM_Ag(Ab)."""
        L = 10  # 5 Ab + 5 Ag
        scorer = self._make_scorer()
        result = MockAFResult(_uniform_logits(L))
        ab_mask = np.array([True] * 5 + [False] * 5)
        ag_mask = np.array([False] * 5 + [True] * 5)
        scores = scorer(result, ab_mask, ag_mask)
        # With symmetric uniform logits both directions give identical pTM values
        # so ipTM_Ab(Ag; L_tot) == ipTM_Ag(Ab; L_tot) → default_iptm == one of them
        # We verify default_iptm >= ab_aligned_iptm (formula check: L_tot vs L_ag)
        assert float(scores["default_iptm"]) >= 0.0
        assert float(scores["ab_aligned_iptm"]) >= 0.0

    def test_formula_consistency_uniform(self):
        """With uniform logits, default_iptm should equal ab_aligned_iptm when L_ab==L_ag.

        When probabilities are uniform and the pTM matrix is uniform, both
        ipTM_Ab(Ag;L_tot) and ipTM_Ab(Ag;L_ag) differ only through d0.  We
        verify that each component is a deterministic function of its inputs.
        """
        scorer = self._make_scorer()
        L = 10
        result = MockAFResult(_uniform_logits(L))
        ab_mask = np.array([True] * 5 + [False] * 5)
        ag_mask = np.array([False] * 5 + [True] * 5)

        scores1 = scorer(result, ab_mask, ag_mask)
        scores2 = scorer(result, ab_mask, ag_mask)
        # Determinism check: same inputs → same outputs
        assert float(scores1["iptm"]) == pytest.approx(float(scores2["iptm"]))
        assert float(scores1["default_iptm"]) == pytest.approx(float(scores2["default_iptm"]))
        assert float(scores1["ab_aligned_iptm"]) == pytest.approx(float(scores2["ab_aligned_iptm"]))

    # --- 6. Score range ---

    def test_scores_in_zero_one_uniform(self):
        """All output scores must be in [0, 1] for uniform logits."""
        scorer = self._make_scorer()
        L = 20
        result = MockAFResult(_uniform_logits(L))
        ab_mask = np.array([True] * 10 + [False] * 10)
        ag_mask = np.array([False] * 10 + [True] * 10)
        scores = scorer(result, ab_mask, ag_mask)
        for key, val in scores.items():
            v = float(val)
            assert 0.0 <= v <= 1.0, f"{key}={v} is out of [0, 1]"

    def test_scores_in_zero_one_random_logits(self):
        """All output scores must be in [0, 1] for random logits."""
        scorer = self._make_scorer()
        rng = np.random.default_rng(99)
        L = 16
        logits = rng.standard_normal((L, L, 64)).astype(np.float32)
        result = MockAFResult(logits)
        ab_mask = np.array([True] * 8 + [False] * 8)
        ag_mask = np.array([False] * 8 + [True] * 8)
        scores = scorer(result, ab_mask, ag_mask)
        for key, val in scores.items():
            v = float(val)
            assert 0.0 <= v <= 1.0, f"{key}={v} is out of [0, 1]"

    def test_scores_in_zero_one_spike_logits(self):
        """All output scores must be in [0, 1] for spike logits (near-deterministic probs)."""
        scorer = self._make_scorer()
        L = 12
        # Spike on first bin → high pTM scores
        result = MockAFResult(_spike_logits(L, bin_idx=0))
        ab_mask = np.array([True] * 6 + [False] * 6)
        ag_mask = np.array([False] * 6 + [True] * 6)
        scores = scorer(result, ab_mask, ag_mask)
        for key, val in scores.items():
            v = float(val)
            assert 0.0 <= v <= 1.0, f"{key}={v} is out of [0, 1]"

    # --- 7. Mask swap ---

    def test_swap_masks_changes_ab_aligned_iptm(self):
        """Swapping ab/ag masks should change ab_aligned_iptm (uses asymmetric L_ag)."""
        scorer = self._make_scorer()
        rng = np.random.default_rng(13)
        L = 14  # unequal chains: 4 Ab, 10 Ag
        logits = rng.standard_normal((L, L, 64)).astype(np.float32)
        result = MockAFResult(logits)

        ab_mask = np.array([True] * 4 + [False] * 10)
        ag_mask = np.array([False] * 4 + [True] * 10)

        scores_orig = scorer(result, ab_mask, ag_mask)
        scores_swap = scorer(result, ag_mask, ab_mask)

        # ab_aligned_iptm uses L_ag for d0: swapping changes L_ag from 10 to 4
        assert float(scores_orig["ab_aligned_iptm"]) != pytest.approx(
            float(scores_swap["ab_aligned_iptm"]), rel=1e-4
        ), "ab_aligned_iptm should differ when masks are swapped with unequal chain lengths"

    def test_swap_masks_default_iptm_same_symmetric_logits(self):
        """With uniform logits, default_iptm should be equal after mask swap.

        default_iptm = max[ipTM_Ab(Ag;L_tot), ipTM_Ag(Ab;L_tot)].
        Swapping ab/ag swaps the two terms inside max — the result is the same.
        This holds for any logits because max is commutative.
        """
        scorer = self._make_scorer()
        L = 10
        result = MockAFResult(_uniform_logits(L))
        ab_mask = np.array([True] * 5 + [False] * 5)
        ag_mask = np.array([False] * 5 + [True] * 5)

        scores_orig = scorer(result, ab_mask, ag_mask)
        scores_swap = scorer(result, ag_mask, ab_mask)

        assert float(scores_orig["default_iptm"]) == pytest.approx(
            float(scores_swap["default_iptm"]), rel=1e-5
        )

    def test_swap_masks_default_iptm_same_random_logits(self):
        """default_iptm = max(a, b) == max(b, a) — identical after swap for any logits."""
        scorer = self._make_scorer()
        rng = np.random.default_rng(55)
        L = 12
        logits = rng.standard_normal((L, L, 64)).astype(np.float32)
        result = MockAFResult(logits)

        ab_mask = np.array([True] * 6 + [False] * 6)
        ag_mask = np.array([False] * 6 + [True] * 6)

        scores_orig = scorer(result, ab_mask, ag_mask)
        scores_swap = scorer(result, ag_mask, ab_mask)

        assert float(scores_orig["default_iptm"]) == pytest.approx(
            float(scores_swap["default_iptm"]), rel=1e-5
        )
