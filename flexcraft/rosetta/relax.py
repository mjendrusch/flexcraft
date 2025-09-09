"""This model provides an interface to Rosetta FastRelax, via PyRosetta.

Importing this module requires a PyRosetta installation. To install PyRosetta run:
pip install pyrosetta-installer
python -c 'import pyrosetta_installer; pyrosetta_installer.install_pyrosetta()'
"""

# Adapted from BindCraft

import os
try:
    import pyrosetta as pr
except ImportError:
    raise ImportError(
        "PyRosetta is not installed. " \
        "Please install it using: pip install pyrosetta-installer; " \
        "python -c 'import pyrosetta_installer; pyrosetta_installer.install_pyrosetta()'.\n"
        "Please note that using PyRosetta for commercial purposes requires " \
        "acquiring a license (https://els2.comotion.uw.edu/product/pyrosetta)."
    )

from pyrosetta.rosetta.core.kinematics import MoveMap
from pyrosetta.rosetta.protocols.simple_moves import AlignChainMover
from pyrosetta.rosetta.protocols.relax import FastRelax
from flexcraft.files.pdb import PDBFile
from flexcraft.data.data import DesignData

def fastrelax(pdb_file: str | PDBFile | DesignData,
              relaxed_pdb_path: str | None=None, tmpdir=None):
    """Run FastRelax with default settings on a `pdb_file` and write the output to `relaxed_pdb_path`.
    
    Args:
        pdb_file: path to a PDB file, `PDBFile` or `DesignData` object to be relaxed.
        relaxed_pdb_path: optional path to the output relaxed PDB.
            If not provided, this will produce and remove temporary PDB files.
        tmpdir: optional path to a temporary file directory. If not provided, will write to "tmp/".
    
    Returns:
        `PDBFile` object of the relaxed structure.
    """
    if isinstance(pdb_file, PDBFile):
        pdb_file = pdb_file.path
    if isinstance(pdb_file, DesignData):
        pdb_file = PDBFile(pdb_file, tmpdir=tmpdir).path
    if relaxed_pdb_path is None:
        relaxed_pdb_path = PDBFile.get_tmp_path(prefix="relaxed", tmpdir=tmpdir)
    if relaxed_pdb_path and not os.path.isdir(os.path.dirname(relaxed_pdb_path)):
        os.makedirs(os.path.dirname(relaxed_pdb_path))

    # Generate pose
    pose = pr.pose_from_pdb(pdb_file)
    start_pose = pose.clone()

    ### Generate movemaps
    mmf = MoveMap()
    mmf.set_chi(True) # enable sidechain movement
    mmf.set_bb(True) # enable backbone movement, can be disabled to increase speed by 30% but makes metrics look worse on average
    mmf.set_jump(False) # disable whole chain movement

    # Run FastRelax
    fastrelax = FastRelax()
    scorefxn = pr.get_fa_scorefxn()
    fastrelax.set_scorefxn(scorefxn)
    fastrelax.set_movemap(mmf) # set MoveMap
    fastrelax.max_iter(200) # default iterations is 2500
    fastrelax.min_type("lbfgs_armijo_nonmonotone")
    fastrelax.constrain_relax_to_start_coords(True)
    fastrelax.apply(pose)

    # Align relaxed structure to original trajectory
    align = AlignChainMover()
    align.source_chain(0)
    align.target_chain(0)
    align.pose(start_pose)
    align.apply(pose)

    # Copy B factors from start_pose to pose
    for resid in range(1, pose.total_residue() + 1):
        if pose.residue(resid).is_protein():
            # Get the B factor of the first heavy atom in the residue
            bfactor = start_pose.pdb_info().bfactor(resid, 1)
            for atom_id in range(1, pose.residue(resid).natoms() + 1):
                pose.pdb_info().bfactor(resid, atom_id, bfactor)

    # output relaxed and aligned PDB
    pose.dump_pdb(relaxed_pdb_path)
    result = PDBFile(path=relaxed_pdb_path)
    result.clean()
    return result
