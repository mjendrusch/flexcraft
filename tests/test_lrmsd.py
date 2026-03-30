"""
Tests for LRMSD (Ligand RMSD) metric.
Source: flexcraft/structure/metrics/rmsd.py :: LRMSD
"""

import sys
import types

# ---------------------------------------------------------------------------
# Haiku compatibility shim
# jax 0.9.0 breaks the dm-haiku import chain via xla_extension; mock haiku
# before any salad/flexcraft imports so the geometry utilities can load.
# ---------------------------------------------------------------------------
_hk_mock = types.ModuleType("haiku")
_hk_mock.__version__ = "mock"
_hk_mock.Params = dict
sys.modules["haiku"] = _hk_mock
for _sub in [
    "haiku.experimental",
    "haiku._src",
    "haiku._src.layer_stack",
    "haiku._src.lift",
    "haiku._src.transform",
]:
    sys.modules[_sub] = types.ModuleType(_sub)

sys.path.insert(0, "/home/ntbiotech/Documents/Current_projects/BinderDesign/flexcraft")

import numpy as np
import pytest

from flexcraft.structure.metrics.rmsd import LRMSD


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _zeros(L: int) -> np.ndarray:
    """Return an all-zero atom14 array of length L."""
    return np.zeros((L, 14, 3), dtype=np.float32)


def _linspace_coords(L: int, scale: float = 1.0) -> np.ndarray:
    """Return atom14 positions with CA (index 1) spread along the x-axis."""
    pos = np.zeros((L, 14, 3), dtype=np.float32)
    pos[:, 1, 0] = np.linspace(0, scale * (L - 1), L)
    return pos


def rotation_matrix_x(theta: float) -> np.ndarray:
    """3x3 rotation matrix around the x-axis by angle theta (radians)."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float32)


def apply_rotation(pos: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Apply rotation R (3x3) to all atom coordinates in pos (L, 14, 3)."""
    return (pos.reshape(-1, 3) @ R.T).reshape(pos.shape)


def apply_translation(pos: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Apply translation vector t (3,) to all atom coordinates in pos."""
    return pos + t[np.newaxis, np.newaxis, :]


# ---------------------------------------------------------------------------
# Test: identity — LRMSD of a structure with itself is 0
# ---------------------------------------------------------------------------

class TestLRMSDIdentity:
    """
    Source: LRMSD.__call__
    Description: LRMSD of a structure with itself must be exactly zero.
    """

    def test_identity_a_is_receptor(self):
        """A (receptor, len>=B) compared to itself gives LRMSD=0."""
        lrmsd = LRMSD()
        A = _linspace_coords(6)
        B = _linspace_coords(3)
        result = float(lrmsd(A, B, A, B))
        assert result == pytest.approx(0.0, abs=1e-5), (
            f"Expected LRMSD=0 for identity, got {result}"
        )

    def test_identity_b_is_receptor(self):
        """B (receptor, len>A) compared to itself gives LRMSD=0."""
        lrmsd = LRMSD()
        A = _linspace_coords(3)
        B = _linspace_coords(6)
        result = float(lrmsd(A, B, A, B))
        assert result == pytest.approx(0.0, abs=1e-5), (
            f"Expected LRMSD=0 for identity (B receptor), got {result}"
        )

    def test_identity_equal_lengths(self):
        """Equal-length A and B (A treated as receptor) compared to itself gives 0."""
        lrmsd = LRMSD()
        A = _linspace_coords(4)
        B = _linspace_coords(4)
        result = float(lrmsd(A, B, A, B))
        assert result == pytest.approx(0.0, abs=1e-5), (
            f"Expected LRMSD=0 for equal-length identity, got {result}"
        )


# ---------------------------------------------------------------------------
# Test: known translation of ligand only
# ---------------------------------------------------------------------------

class TestLRMSDKnownTranslation:
    """
    Source: LRMSD.__call__
    Description: Translating only the ligand by a known vector d should
    produce LRMSD equal to |d| (the Euclidean distance of the translation).
    The receptor is the same in both structures so alignment is perfect.
    """

    def test_ligand_translated_along_x(self):
        """Ligand shifted by (3, 0, 0): LRMSD should equal 3.0."""
        lrmsd = LRMSD()
        # A is receptor (longer)
        A = _linspace_coords(6)
        B_x = _linspace_coords(3)                      # structure x ligand
        B_y = apply_translation(B_x, np.array([3.0, 0.0, 0.0]))  # translated ligand in y

        result = float(lrmsd(A, B_x, A, B_y))
        assert result == pytest.approx(3.0, abs=1e-4), (
            f"Expected LRMSD=3.0 for x-translation, got {result}"
        )

    def test_ligand_translated_along_xyz(self):
        """Ligand shifted by (1, 2, 2): LRMSD should equal 3.0 (||(1,2,2)||=3)."""
        lrmsd = LRMSD()
        A = _linspace_coords(8)
        B_x = _linspace_coords(4)
        delta = np.array([1.0, 2.0, 2.0])
        B_y = apply_translation(B_x, delta)

        expected = float(np.linalg.norm(delta))
        result = float(lrmsd(A, B_x, A, B_y))
        assert result == pytest.approx(expected, abs=1e-4), (
            f"Expected LRMSD={expected:.4f}, got {result}"
        )

    def test_zero_translation_gives_zero(self):
        """A zero translation should give LRMSD=0."""
        lrmsd = LRMSD()
        A = _linspace_coords(5)
        B = _linspace_coords(3)
        result = float(lrmsd(A, B, A, B))
        assert result == pytest.approx(0.0, abs=1e-5), (
            f"Expected LRMSD=0 for zero translation, got {result}"
        )


# ---------------------------------------------------------------------------
# Test: receptor vs ligand assignment
# ---------------------------------------------------------------------------

class TestReceptorLigandAssignment:
    """
    Source: LRMSD.__call__ — len(A_x) >= len(B_x) => A is receptor, B is ligand;
                              len(B_x) > len(A_x) => B is receptor, A is ligand.
    Description: Verify that the correct chain is treated as the ligand (scored).
    When only the ligand is displaced, LRMSD > 0; when only the receptor is
    displaced, LRMSD should be 0 because the ligand did not move relative to
    the receptor after alignment.
    """

    def test_a_is_receptor_ligand_displaced(self):
        """A is receptor (len>=B): displacing B gives LRMSD > 0."""
        lrmsd = LRMSD()
        A = _linspace_coords(6)
        B_x = _linspace_coords(3)
        B_y = apply_translation(B_x, np.array([5.0, 0.0, 0.0]))
        result = float(lrmsd(A, B_x, A, B_y))
        assert result > 0.1, (
            f"Expected LRMSD>0 when ligand B is displaced, got {result}"
        )

    def test_a_is_receptor_only_receptor_displaced(self):
        """A is receptor (len>=B): rigid translation of entire complex gives LRMSD~0."""
        lrmsd = LRMSD()
        A_x = _linspace_coords(6)
        B_x = _linspace_coords(3)
        # Translate the ENTIRE complex rigidly — after receptor alignment LRMSD must be 0
        t = np.array([10.0, 5.0, -3.0])
        A_y = apply_translation(A_x, t)
        B_y = apply_translation(B_x, t)
        result = float(lrmsd(A_x, B_x, A_y, B_y))
        assert result == pytest.approx(0.0, abs=1e-4), (
            f"Expected LRMSD~0 when whole complex translated rigidly, got {result}"
        )

    def test_b_is_receptor_when_longer(self):
        """B is receptor (len(B)>len(A)): displacing A gives LRMSD > 0."""
        lrmsd = LRMSD()
        B = _linspace_coords(6)          # B longer -> B is receptor
        A_x = _linspace_coords(3)
        A_y = apply_translation(A_x, np.array([5.0, 0.0, 0.0]))
        result = float(lrmsd(A_x, B, A_y, B))
        assert result > 0.1, (
            f"Expected LRMSD>0 when ligand A is displaced (B is receptor), got {result}"
        )

    def test_b_is_receptor_receptor_displaced_gives_zero(self):
        """B is receptor (len>A): rigid translation of whole complex gives LRMSD~0."""
        lrmsd = LRMSD()
        A_x = _linspace_coords(3)
        B_x = _linspace_coords(7)        # B longer -> B is receptor
        t = np.array([4.0, -2.0, 6.0])
        A_y = apply_translation(A_x, t)
        B_y = apply_translation(B_x, t)
        result = float(lrmsd(A_x, B_x, A_y, B_y))
        assert result == pytest.approx(0.0, abs=1e-4), (
            f"Expected LRMSD~0 when whole complex translated rigidly (B receptor), got {result}"
        )


# ---------------------------------------------------------------------------
# Test: alignment invariance — rigid transformation of the whole complex
# ---------------------------------------------------------------------------

class TestAlignmentInvariance:
    """
    Source: LRMSD.__call__
    Description: Rigidly rotating and/or translating the ENTIRE complex (both
    chains together) must not change the LRMSD value, because the Kabsch
    alignment on the receptor absorbs the global transformation.
    """

    def test_global_translation_invariance(self):
        """Translating both complexes by the same offset does not change LRMSD."""
        lrmsd = LRMSD()
        A_x = _linspace_coords(6)
        B_x = _linspace_coords(3)
        # Original LRMSD with a displaced ligand
        B_y = apply_translation(B_x, np.array([2.0, 1.0, 0.0]))
        A_y = A_x.copy()
        ref_result = float(lrmsd(A_x, B_x, A_y, B_y))

        # Now shift both input structures by a large global offset
        global_t = np.array([100.0, -50.0, 30.0])
        result = float(lrmsd(
            apply_translation(A_x, global_t),
            apply_translation(B_x, global_t),
            apply_translation(A_y, global_t),
            apply_translation(B_y, global_t),
        ))
        assert result == pytest.approx(ref_result, abs=1e-4), (
            f"LRMSD changed under global translation: {ref_result} -> {result}"
        )

    def test_global_rotation_invariance(self):
        """Rotating both complexes by the same rotation does not change LRMSD."""
        lrmsd = LRMSD()
        A_x = _linspace_coords(6)
        B_x = _linspace_coords(3)
        B_y = apply_translation(B_x, np.array([2.0, 0.0, 0.0]))
        A_y = A_x.copy()
        ref_result = float(lrmsd(A_x, B_x, A_y, B_y))

        R = rotation_matrix_x(np.pi / 4)  # 45-degree rotation
        result = float(lrmsd(
            apply_rotation(A_x, R),
            apply_rotation(B_x, R),
            apply_rotation(A_y, R),
            apply_rotation(B_y, R),
        ))
        assert result == pytest.approx(ref_result, abs=1e-4), (
            f"LRMSD changed under global rotation: {ref_result} -> {result}"
        )

    def test_global_rototranslation_invariance(self):
        """Combined rotation + translation of both complexes does not change LRMSD."""
        lrmsd = LRMSD()
        A_x = _linspace_coords(5)
        B_x = _linspace_coords(2)
        B_y = apply_translation(B_x, np.array([0.0, 3.0, 4.0]))
        A_y = A_x.copy()
        ref_result = float(lrmsd(A_x, B_x, A_y, B_y))

        R = rotation_matrix_x(np.pi / 3)
        t = np.array([7.0, -3.0, 11.0])
        result = float(lrmsd(
            apply_translation(apply_rotation(A_x, R), t),
            apply_translation(apply_rotation(B_x, R), t),
            apply_translation(apply_rotation(A_y, R), t),
            apply_translation(apply_rotation(B_y, R), t),
        ))
        assert result == pytest.approx(ref_result, abs=1e-4), (
            f"LRMSD changed under rototranslation: {ref_result} -> {result}"
        )
