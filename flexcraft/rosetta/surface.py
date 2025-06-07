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

from flexcraft.files.pdb import PDBFile
from flexcraft.data.data import DesignData

def get_sasa(pose):
    """Calculate the total and hydrophobic sasa"""
    rsd_sasa = pr.rosetta.utility.vector1_double()
    rsd_hydrophobic_sasa = pr.rosetta.utility.vector1_double()
    pr.rosetta.core.scoring.calc_per_res_hydrophobic_sasa(
        pose, rsd_sasa, rsd_hydrophobic_sasa, 1.4)
    return sum(rsd_sasa), sum(rsd_hydrophobic_sasa)

def sap_per_residue(pdb_file, tmpdir=None):
    if isinstance(pdb_file, PDBFile):
        pdb_file = pdb_file.path
    if isinstance(pdb_file, DesignData):
        pdb_file = PDBFile(pdb_file, tmpdir=tmpdir).path
    pose = pr.pose_from_file(pdb_file)
    true_sel = (
        pr.rosetta.core.select.residue_selector.TrueResidueSelector()
    )

    total_sap_score = pr.rosetta.core.pack.guidance_scoreterms.sap.calculate_sap(
        pose,
        true_sel,
        true_sel,
        true_sel,
    )

    no_res = len(pose.sequence())
    sap_per_res = total_sap_score / float(no_res)

    # rsd_sasa, rsd_hydrophobic_sasa = get_sasa(pose)
    # rsd_sasa_per_res = rsd_sasa / float(no_res)
    # rsd_hydrophobic_sasa_per_res = rsd_hydrophobic_sasa / float(no_res)
    return sap_per_res

