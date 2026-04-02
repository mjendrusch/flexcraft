"""
Tests for LRMSD (Ligand RMSD) metric.
Source: flexcraft/structure/metrics/rmsd.py :: LRMSD
"""

import numpy as np
import pytest
from flexcraft.structure.metrics import LRMSD

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


def _concat(receptor: np.ndarray, ligand: np.ndarray):
    """Concatenate receptor and ligand into one array and build is_target mask.

    is_target[i] = True  -> residue i is the ligand (scored in LRMSD)
    is_target[i] = False -> residue i is the receptor (used for alignment)
    """
    L_rec = receptor.shape[0]
    L_lig = ligand.shape[0]
    combined = np.concatenate([receptor, ligand], axis=0)
    is_target = np.array(
        [False] * L_rec + [True] * L_lig, dtype=np.bool_
    )
    return combined, is_target


# ---------------------------------------------------------------------------
# Test: identity — LRMSD of a structure with itself is 0
# ---------------------------------------------------------------------------

class TestLRMSDIdentity:
    """
    Source: LRMSD.__call__
    Description: LRMSD of a structure with itself must be exactly zero.
    """

    def test_identity_a_is_receptor(self):
        """Receptor (len 6) + ligand (len 3): LRMSD of structure with itself is 0."""
        lrmsd = LRMSD()
        receptor = _linspace_coords(6)
        ligand = _linspace_coords(3)
        x, is_target = _concat(receptor, ligand)
        result = float(lrmsd(x, x, is_target))
        assert result == pytest.approx(0.0, abs=1e-5), (
            f"Expected LRMSD=0 for identity, got {result}"
        )

    def test_identity_b_is_receptor(self):
        """Receptor (len 6) + ligand (len 3): reversed role assignment gives LRMSD=0."""
        lrmsd = LRMSD()
        receptor = _linspace_coords(6)
        ligand = _linspace_coords(3)
        # Here ligand comes first in the array; is_target marks the ligand section
        L_lig = ligand.shape[0]
        L_rec = receptor.shape[0]
        x = np.concatenate([ligand, receptor], axis=0)
        is_target = np.array(
            [True] * L_lig + [False] * L_rec, dtype=np.bool_
        )
        result = float(lrmsd(x, x, is_target))
        assert result == pytest.approx(0.0, abs=1e-5), (
            f"Expected LRMSD=0 for identity (ligand first), got {result}"
        )

    def test_identity_equal_lengths(self):
        """Equal-length receptor and ligand: LRMSD of structure with itself is 0."""
        lrmsd = LRMSD()
        receptor = _linspace_coords(4)
        ligand = _linspace_coords(4)
        x, is_target = _concat(receptor, ligand)
        result = float(lrmsd(x, x, is_target))
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
        receptor = _linspace_coords(6)
        ligand_x = _linspace_coords(3)
        ligand_y = apply_translation(ligand_x, np.array([3.0, 0.0, 0.0]))

        x, is_target = _concat(receptor, ligand_x)
        y, _ = _concat(receptor, ligand_y)

        result = float(lrmsd(x, y, is_target))
        assert result == pytest.approx(3.0, abs=1e-4), (
            f"Expected LRMSD=3.0 for x-translation, got {result}"
        )

    def test_ligand_translated_along_xyz(self):
        """Ligand shifted by (1, 2, 2): LRMSD should equal 3.0 (||(1,2,2)||=3)."""
        lrmsd = LRMSD()
        receptor = _linspace_coords(8)
        ligand_x = _linspace_coords(4)
        delta = np.array([1.0, 2.0, 2.0])
        ligand_y = apply_translation(ligand_x, delta)

        x, is_target = _concat(receptor, ligand_x)
        y, _ = _concat(receptor, ligand_y)

        expected = float(np.linalg.norm(delta))
        result = float(lrmsd(x, y, is_target))
        assert result == pytest.approx(expected, abs=1e-4), (
            f"Expected LRMSD={expected:.4f}, got {result}"
        )

    def test_zero_translation_gives_zero(self):
        """A zero translation should give LRMSD=0."""
        lrmsd = LRMSD()
        receptor = _linspace_coords(5)
        ligand = _linspace_coords(3)
        x, is_target = _concat(receptor, ligand)
        result = float(lrmsd(x, x, is_target))
        assert result == pytest.approx(0.0, abs=1e-5), (
            f"Expected LRMSD=0 for zero translation, got {result}"
        )


# ---------------------------------------------------------------------------
# Test: receptor vs ligand assignment
# ---------------------------------------------------------------------------

class TestReceptorLigandAssignment:
    """
    Source: LRMSD.__call__ — is_target marks the ligand (scored); ~is_target marks
                              the receptor (used for Kabsch alignment).
    Description: Verify that the correct chain is treated as the ligand (scored).
    When only the ligand is displaced, LRMSD > 0; when the entire complex is
    rigidly transformed, LRMSD should be 0 because alignment absorbs the motion.
    """

    def test_ligand_displaced_gives_nonzero(self):
        """Displacing the ligand (is_target=True) gives LRMSD > 0."""
        lrmsd = LRMSD()
        receptor = _linspace_coords(6)
        ligand_x = _linspace_coords(3)
        ligand_y = apply_translation(ligand_x, np.array([5.0, 0.0, 0.0]))

        x, is_target = _concat(receptor, ligand_x)
        y, _ = _concat(receptor, ligand_y)

        result = float(lrmsd(x, y, is_target))
        assert result > 0.1, (
            f"Expected LRMSD>0 when ligand is displaced, got {result}"
        )

    def test_rigid_whole_complex_translation_gives_zero(self):
        """Rigid translation of the entire complex gives LRMSD~0."""
        lrmsd = LRMSD()
        receptor_x = _linspace_coords(6)
        ligand_x = _linspace_coords(3)
        t = np.array([10.0, 5.0, -3.0])
        receptor_y = apply_translation(receptor_x, t)
        ligand_y = apply_translation(ligand_x, t)

        x, is_target = _concat(receptor_x, ligand_x)
        y, _ = _concat(receptor_y, ligand_y)

        result = float(lrmsd(x, y, is_target))
        assert result == pytest.approx(0.0, abs=1e-4), (
            f"Expected LRMSD~0 when whole complex translated rigidly, got {result}"
        )

    def test_receptor_only_displacement_gives_zero(self):
        """Displacing only the receptor (~is_target) gives LRMSD~0 (alignment absorbs it)."""
        lrmsd = LRMSD()
        receptor_x = _linspace_coords(6)
        ligand = _linspace_coords(3)
        receptor_y = apply_translation(receptor_x, np.array([5.0, 0.0, 0.0]))

        x, is_target = _concat(receptor_x, ligand)
        y, _ = _concat(receptor_y, ligand)

        result = float(lrmsd(x, y, is_target))
        assert result == pytest.approx(0.0, abs=1e-4), (
            f"Expected LRMSD~0 when only receptor is displaced (alignment absorbs it), got {result}"
        )

    def test_ligand_first_in_array_displaced_gives_nonzero(self):
        """Ligand placed first in the concatenated array: displacement gives LRMSD > 0."""
        lrmsd = LRMSD()
        ligand_x = _linspace_coords(3)
        receptor = _linspace_coords(6)
        ligand_y = apply_translation(ligand_x, np.array([5.0, 0.0, 0.0]))

        L_lig = ligand_x.shape[0]
        L_rec = receptor.shape[0]
        x = np.concatenate([ligand_x, receptor], axis=0)
        y = np.concatenate([ligand_y, receptor], axis=0)
        is_target = np.array([True] * L_lig + [False] * L_rec, dtype=np.bool_)

        result = float(lrmsd(x, y, is_target))
        assert result > 0.1, (
            f"Expected LRMSD>0 when ligand (first in array) is displaced, got {result}"
        )

    def test_rigid_complex_ligand_first_gives_zero(self):
        """Ligand first in array: rigid translation of whole complex gives LRMSD~0."""
        lrmsd = LRMSD()
        ligand_x = _linspace_coords(3)
        receptor_x = _linspace_coords(7)
        t = np.array([4.0, -2.0, 6.0])
        ligand_y = apply_translation(ligand_x, t)
        receptor_y = apply_translation(receptor_x, t)

        L_lig = ligand_x.shape[0]
        L_rec = receptor_x.shape[0]
        x = np.concatenate([ligand_x, receptor_x], axis=0)
        y = np.concatenate([ligand_y, receptor_y], axis=0)
        is_target = np.array([True] * L_lig + [False] * L_rec, dtype=np.bool_)

        result = float(lrmsd(x, y, is_target))
        assert result == pytest.approx(0.0, abs=1e-4), (
            f"Expected LRMSD~0 when whole complex translated rigidly (ligand first), got {result}"
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
        receptor_x = _linspace_coords(6)
        ligand_x = _linspace_coords(3)
        ligand_y = apply_translation(ligand_x, np.array([2.0, 1.0, 0.0]))
        receptor_y = receptor_x.copy()

        x, is_target = _concat(receptor_x, ligand_x)
        y, _ = _concat(receptor_y, ligand_y)
        ref_result = float(lrmsd(x, y, is_target))

        global_t = np.array([100.0, -50.0, 30.0])
        x_shifted, _ = _concat(
            apply_translation(receptor_x, global_t),
            apply_translation(ligand_x, global_t),
        )
        y_shifted, _ = _concat(
            apply_translation(receptor_y, global_t),
            apply_translation(ligand_y, global_t),
        )
        result = float(lrmsd(x_shifted, y_shifted, is_target))
        assert result == pytest.approx(ref_result, abs=1e-4), (
            f"LRMSD changed under global translation: {ref_result} -> {result}"
        )

    def test_global_rotation_invariance(self):
        """Rotating both complexes by the same rotation does not change LRMSD."""
        lrmsd = LRMSD()
        receptor_x = _linspace_coords(6)
        ligand_x = _linspace_coords(3)
        ligand_y = apply_translation(ligand_x, np.array([2.0, 0.0, 0.0]))
        receptor_y = receptor_x.copy()

        x, is_target = _concat(receptor_x, ligand_x)
        y, _ = _concat(receptor_y, ligand_y)
        ref_result = float(lrmsd(x, y, is_target))

        R = rotation_matrix_x(np.pi / 4)
        x_rot, _ = _concat(
            apply_rotation(receptor_x, R),
            apply_rotation(ligand_x, R),
        )
        y_rot, _ = _concat(
            apply_rotation(receptor_y, R),
            apply_rotation(ligand_y, R),
        )
        result = float(lrmsd(x_rot, y_rot, is_target))
        assert result == pytest.approx(ref_result, abs=1e-4), (
            f"LRMSD changed under global rotation: {ref_result} -> {result}"
        )

    def test_global_rototranslation_invariance(self):
        """Combined rotation + translation of both complexes does not change LRMSD."""
        lrmsd = LRMSD()
        receptor_x = _linspace_coords(5)
        ligand_x = _linspace_coords(2)
        ligand_y = apply_translation(ligand_x, np.array([0.0, 3.0, 4.0]))
        receptor_y = receptor_x.copy()

        x, is_target = _concat(receptor_x, ligand_x)
        y, _ = _concat(receptor_y, ligand_y)
        ref_result = float(lrmsd(x, y, is_target))

        R = rotation_matrix_x(np.pi / 3)
        t = np.array([7.0, -3.0, 11.0])
        x_rt, _ = _concat(
            apply_translation(apply_rotation(receptor_x, R), t),
            apply_translation(apply_rotation(ligand_x, R), t),
        )
        y_rt, _ = _concat(
            apply_translation(apply_rotation(receptor_y, R), t),
            apply_translation(apply_rotation(ligand_y, R), t),
        )
        result = float(lrmsd(x_rt, y_rt, is_target))
        assert result == pytest.approx(ref_result, abs=1e-4), (
            f"LRMSD changed under rototranslation: {ref_result} -> {result}"
        )
