"""FastRelax via tmol (open, license-free, GPU) -- a drop-in alternative to the
PyRosetta backend in ``flexcraft.rosetta.relax``.

tmol (https://github.com/uw-ipd/tmol, BSD-3) is a PyTorch reimplementation of the
Rosetta full-atom energy function and FastRelax protocol. It needs no PyRosetta
licence and runs on GPU. The API used here (``pose_stack_from_pdb``,
``beta2016_score_function``, ``MoveMap``, ``PackerPalette``, ``FoldForest``,
``fast_relax``, ``write_pose_stack_pdb``) is tmol's public interface; the call
pattern follows tmol's own ``tests/relax/test_fast_relax.py``. tmol is imported
lazily so flexcraft installs without it.

The FastRelax behaviour mirrors ``flexcraft.rosetta.relax.fastrelax``: relax
backbone + side chains under mainchain coordinate constraints to the input,
realign onto the starting structure, and restore the input B-factors.

(An open tmol scoring engine for a BindCraft-lineage pipeline first appeared in
FreeBindCraft; this is an independent reimplementation against upstream tmol, not
a port of that Elastic-License code.)
"""

import numpy as np

from flexcraft.data.data import DesignData
from flexcraft.files.pdb import PDBFile


def _require_tmol():
    try:
        import tmol  # noqa: F401
        import torch  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "tmol is required for flexcraft.tmol. Install upstream tmol "
            "(https://github.com/uw-ipd/tmol); its compiled extensions must match "
            "the installed torch build. No PyRosetta licence is needed."
        ) from exc
    import tmol
    import torch
    return tmol, torch


def _select_device(device, torch):
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _ca_coords_and_bfactors(pdb_path):
    """Per-residue CA coordinates and per-atom B-factors from a PDB (biotite)."""
    from biotite.structure.io.pdb import PDBFile as BiotitePDB

    arr = BiotitePDB.read(pdb_path).get_structure(model=1, extra_fields=["b_factor"])
    ca = arr[(arr.atom_name == "CA") & (arr.element != "H")]
    return arr, ca.coord


def _kabsch(mobile, target):
    """Rotation+translation aligning `mobile` onto `target` (both (N,3))."""
    mc, tc = mobile.mean(0), target.mean(0)
    h = (mobile - mc).T @ (target - tc)
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    r = vt.T @ np.diag([1, 1, d]) @ u.T
    return r, tc - r @ mc


def fastrelax(
    pdb_file: "str | PDBFile | DesignData",
    relaxed_pdb_path: str | None = None,
    tmpdir: str | None = None,
    return_pose: bool = False,
    num_repeats: int = 1,
    constrain_to_start: bool = True,
    device=None,
):
    """Run tmol FastRelax on `pdb_file`, mirroring ``flexcraft.rosetta.relax.fastrelax``.

    Args:
        pdb_file: path, ``PDBFile``, or ``DesignData`` to relax.
        relaxed_pdb_path: output path for the relaxed PDB (temp file if ``None``).
        tmpdir: directory for temporary files.
        return_pose: also return the relaxed tmol ``PoseStack`` (not a Rosetta Pose).
        num_repeats: FastRelax pack/min ramps.
        constrain_to_start: apply ramped mainchain coordinate constraints to the input.
        device: torch device (auto-selects CUDA if available, else CPU).

    Returns:
        ``PDBFile`` of the relaxed structure, or ``(PDBFile, PoseStack)`` if ``return_pose``.
    """
    tmol, torch = _require_tmol()
    from tmol.pack import PackerPalette

    dev = _select_device(device, torch)

    uses_tmpfile = False
    if isinstance(pdb_file, PDBFile):
        pdb_file = pdb_file.path
    if isinstance(pdb_file, DesignData):
        uses_tmpfile = True
        tmp_file = PDBFile(pdb_file, prefix="tmp", tmpdir=tmpdir)
        pdb_file = tmp_file.path
    if relaxed_pdb_path is None:
        relaxed_pdb_path = PDBFile.get_tmp_path(prefix="relaxed", tmpdir=tmpdir)

    start_arr, start_ca = _ca_coords_and_bfactors(pdb_file)

    pose_stack = tmol.pose_stack_from_pdb(pdb_file, dev)
    sfxn = tmol.beta2016_score_function(dev)
    if constrain_to_start:
        pose_stack = tmol.create_mainchain_coordinate_constraints(pose_stack)
        sfxn.set_weight(tmol.ScoreType.constraint, 1.0)

    move_map = tmol.MoveMap.from_pose_stack(pose_stack)
    move_map.move_all_jumps = True
    move_map.move_all_named_torsions = True
    palette = PackerPalette()
    fold_forest = tmol.FoldForest.reasonable_fold_forest(pose_stack)

    # With task_operations left unset, fast_relax builds its default packer task:
    # restrict-to-repacking with Dunbrack + fixed-AA + include-current rotamer samplers.
    try:
        relaxed = tmol.fast_relax(
            pose_stack,
            sfxn,
            palette,
            move_map,
            fold_forest,
            num_repeats=int(num_repeats),
            ramp_constraints=bool(constrain_to_start),
        )
    finally:
        if constrain_to_start:
            sfxn.set_weight(tmol.ScoreType.constraint, 0.0)

    tmol.write_pose_stack_pdb(relaxed, relaxed_pdb_path)

    # Realign onto the starting structure and restore input B-factors (flexcraft semantics).
    rel_arr, rel_ca = _ca_coords_and_bfactors(relaxed_pdb_path)
    if len(rel_ca) == len(start_ca) and len(start_ca) >= 3:
        r, t = _kabsch(rel_ca, start_ca)
        rel_arr.coord = (rel_arr.coord @ r.T) + t
    if start_arr.array_length() == rel_arr.array_length():
        rel_arr.set_annotation("b_factor", start_arr.b_factor)
    from biotite.structure.io.pdb import PDBFile as BiotitePDB

    out = BiotitePDB()
    out.set_structure(rel_arr)
    out.write(relaxed_pdb_path)

    result = PDBFile(path=relaxed_pdb_path)
    if uses_tmpfile:
        tmp_file.remove()
    if return_pose:
        return result, relaxed
    return result
