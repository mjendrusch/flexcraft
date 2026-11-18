"""Interface to Rosetta FastRelax and residue pair energies via PyRosetta.

PyRosetta is imported lazily when `fastrelax` or `pair_energies` is called.
When installing PyRosetta into a `uv`-managed environment, run subsequent syncs
with `uv sync --inexact` (or commands with `uv run --no-sync`) so `uv` preserves
the out-of-tree PyRosetta installation.
"""

# Adapted from BindCraft

import os

from flexcraft.data.data import DesignData
from flexcraft.files.pdb import PDBFile


def _require_pyrosetta():
    """Import PyRosetta lazily, raising a clear ImportError if unavailable."""
    try:
        import pyrosetta as pr
    except ImportError as exc:
        raise ImportError(
            "PyRosetta is not installed, and is required for flexcraft.rosetta.relax. "
            "Install PyRosetta into your environment (e.g. `pip install pyrosetta-installer && "
            "python -c 'import pyrosetta_installer; pyrosetta_installer.install_pyrosetta()'`; "
            "commercial use requires a license: https://els2.comotion.uw.edu/product/pyrosetta), "
            "and use `uv sync --inexact` (or `uv run --no-sync`) so `uv` does not remove it."
        ) from exc
    return pr


def fastrelax(
    pdb_file: str | PDBFile | DesignData,
    relaxed_pdb_path: str | None = None,
    tmpdir: str | None = None,
    return_pose: bool = False,
):
    """Run FastRelax with default settings on `pdb_file`.

    Args:
        pdb_file: Path to a PDB file, `PDBFile`, or `DesignData` object to relax.
        relaxed_pdb_path: Optional output path for the relaxed PDB file.
        tmpdir: Optional temporary directory path.
        return_pose: Whether to also return the relaxed PyRosetta Pose.

    Returns:
        `PDBFile` of the relaxed structure, or `(PDBFile, Pose)` if `return_pose=True`.
    """
    pr = _require_pyrosetta()
    from pyrosetta.rosetta.core.kinematics import MoveMap
    from pyrosetta.rosetta.protocols.relax import FastRelax
    from pyrosetta.rosetta.protocols.simple_moves import AlignChainMover

    uses_tmpfile = False
    if isinstance(pdb_file, PDBFile):
        pdb_file = pdb_file.path
    if isinstance(pdb_file, DesignData):
        uses_tmpfile = True
        tmp_file = PDBFile(pdb_file, prefix="tmp", tmpdir=tmpdir)
        pdb_file = tmp_file.path
    if relaxed_pdb_path is None:
        relaxed_pdb_path = PDBFile.get_tmp_path(prefix="relaxed", tmpdir=tmpdir)
    relaxed_dir = os.path.dirname(relaxed_pdb_path) if relaxed_pdb_path else ""
    if relaxed_dir and not os.path.isdir(relaxed_dir):
        os.makedirs(relaxed_dir, exist_ok=True)

    pose = pr.pose_from_pdb(pdb_file)
    start_pose = pose.clone()

    movemap = MoveMap()
    movemap.set_chi(True)
    movemap.set_bb(True)
    movemap.set_jump(False)

    relax_mover = FastRelax()
    scorefxn = pr.get_fa_scorefxn()
    relax_mover.set_scorefxn(scorefxn)
    relax_mover.set_movemap(movemap)
    relax_mover.max_iter(200)
    relax_mover.min_type("lbfgs_armijo_nonmonotone")
    relax_mover.constrain_relax_to_start_coords(True)
    relax_mover.apply(pose)

    align = AlignChainMover()
    align.source_chain(0)
    align.target_chain(0)
    align.pose(start_pose)
    align.apply(pose)

    for resid in range(1, pose.total_residue() + 1):
        if pose.residue(resid).is_protein():
            bfactor = start_pose.pdb_info().bfactor(resid, 1)
            for atom_id in range(1, pose.residue(resid).natoms() + 1):
                pose.pdb_info().bfactor(resid, atom_id, bfactor)

    pose.dump_pdb(relaxed_pdb_path)
    result = PDBFile(path=relaxed_pdb_path)
    result.clean()
    if uses_tmpfile:
        tmp_file.remove()
    if return_pose:
        return result, pose
    return result


def pair_energies(pdb_file: str | PDBFile | DesignData, tmpdir: str | None = None):
    """Compute residue pair energies for `pdb_file` using the PyRosetta full-atom scorefxn."""
    pr = _require_pyrosetta()
    uses_tmpfile = False
    if isinstance(pdb_file, pr.rosetta.core.pose.Pose):
        pose = pdb_file
    elif isinstance(pdb_file, PDBFile):
        pose = pr.pose_from_pdb(pdb_file.path)
    elif isinstance(pdb_file, DesignData):
        uses_tmpfile = True
        tmp_file = PDBFile(pdb_file, prefix="tmp", tmpdir=tmpdir)
        pose = pr.pose_from_pdb(tmp_file.path)
    else:
        pose = pr.pose_from_pdb(pdb_file)
    scorefxn = pr.get_fa_scorefxn()
    scorefxn.score(pose)
    energies = pose.energies().residue_pair_energies_array()
    if uses_tmpfile:
        tmp_file.remove()
    return energies
