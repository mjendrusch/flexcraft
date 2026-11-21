"""Surface hydrophobicity and SAP, license-free (no PyRosetta).

A drop-in alternative to :mod:`flexcraft.rosetta.surface`. The quantities here are
geometric (solvent-accessible surface area) plus a residue hydrophobicity scale,
not terms of a force field, so they do not come from tmol's energy function (tmol
exposes no SASA). SASA is computed with biotite's Shrake-Rupley implementation,
which flexcraft already depends on, so this module needs no PyRosetta licence.

``get_sasa`` mirrors ``flexcraft.rosetta.surface.get_sasa`` (total and hydrophobic
SASA). ``sap_per_residue`` mirrors ``flexcraft.rosetta.surface.sap_per_residue`` and
returns a per-residue-normalised spatial aggregation propensity (SAP) in the sense of
Lauer et al., *J. Pharm. Sci.* 101:102-115 (2012). It is an open reimplementation
built on biotite SASA; it is NOT guaranteed bit-identical to Rosetta's
``calculate_sap`` (which uses its own SASA engine and a per-atom reference). Use the
``flexcraft.rosetta`` backend when exact Rosetta SAP is required.
"""

import numpy as np

from flexcraft.data.data import DesignData
from flexcraft.files.pdb import PDBFile

# Hydrophobic atoms for the "hydrophobic SASA" split: carbon and sulfur (nonpolar).
_HYDROPHOBIC_ELEMENTS = ("C", "S")

# Tien et al. 2013 THEORETICAL maximum solvent-accessible surface area (A^2) per
# residue, used to turn per-residue SASA into a relative exposure.
# Source: Tien et al., PLoS ONE 8:e80635 (2013), "theoretical" column
# (en.wikipedia.org/wiki/Relative_accessible_surface_area).
MAX_ASA: dict[str, float] = {
    "ALA": 129.0, "ARG": 274.0, "ASN": 195.0, "ASP": 193.0, "CYS": 167.0,
    "GLN": 225.0, "GLU": 223.0, "GLY": 104.0, "HIS": 224.0, "ILE": 197.0,
    "LEU": 201.0, "LYS": 236.0, "MET": 224.0, "PHE": 240.0, "PRO": 159.0,
    "SER": 155.0, "THR": 172.0, "TRP": 285.0, "TYR": 263.0, "VAL": 174.0,
}

# Black & Mould 1991 hydrophobicity (Anal. Biochem. 193:72-82), normalised so
# Gly = 0.0 (more hydrophobic = more positive); the scale SAP / BIOVIA Protein
# Aggregation Scores use. Values are the raw ProtScale scale
# (web.expasy.org/protscale/pscale/Hphob.Black.html) minus the Gly value 0.501.
HYDROPHOBICITY: dict[str, float] = {
    "ALA": 0.115, "ARG": -0.501, "ASN": -0.265, "ASP": -0.473, "CYS": 0.179,
    "GLN": -0.250, "GLU": -0.458, "GLY": 0.000, "HIS": -0.336, "ILE": 0.442,
    "LEU": 0.442, "LYS": -0.218, "MET": 0.237, "PHE": 0.499, "PRO": 0.210,
    "SER": -0.142, "THR": -0.051, "TRP": 0.377, "TYR": 0.379, "VAL": 0.324,
}


def _load_pdb_path(pdb_file, tmpdir=None):
    """Resolve a path from str / PDBFile / DesignData; return (path, cleanup_fn)."""
    if isinstance(pdb_file, PDBFile):
        return pdb_file.path, None
    if isinstance(pdb_file, DesignData):
        tmp = PDBFile(pdb_file, prefix="tmp", tmpdir=tmpdir)
        return tmp.path, tmp.remove
    return pdb_file, None


def _structure(pdb_path):
    from biotite.structure.io.pdb import PDBFile as BiotitePDB

    return BiotitePDB.read(pdb_path).get_structure(model=1)


def _per_atom_sasa(atoms):
    """Per-atom SASA (A^2) via biotite Shrake-Rupley; NaN entries treated as 0."""
    import biotite.structure as struc

    sasa = struc.sasa(atoms, probe_radius=1.4, vdw_radii="ProtOr")
    return np.nan_to_num(sasa, nan=0.0)


def get_sasa(pdb_file: "str | PDBFile | DesignData", tmpdir=None):
    """Total and hydrophobic SASA (A^2), mirroring the Rosetta backend's ``get_sasa``.

    Hydrophobic SASA is the area contributed by carbon and sulfur atoms.

    Returns:
        ``(total_sasa, hydrophobic_sasa)``.
    """
    path, cleanup = _load_pdb_path(pdb_file, tmpdir)
    try:
        atoms = _structure(path)
        sasa = _per_atom_sasa(atoms)
        total = float(sasa.sum())
        hydrophobic = float(sasa[np.isin(atoms.element, _HYDROPHOBIC_ELEMENTS)].sum())
    finally:
        if cleanup is not None:
            cleanup()
    return total, hydrophobic


def surface_hydrophobicity(pdb_file: "str | PDBFile | DesignData", tmpdir=None):
    """Fraction of solvent-accessible surface that is hydrophobic (0-1)."""
    total, hydrophobic = get_sasa(pdb_file, tmpdir=tmpdir)
    return hydrophobic / total if total > 0 else 0.0


def sap_per_residue(pdb_file: "str | PDBFile | DesignData", tmpdir=None, radius: float = 5.0):
    """Per-residue-normalised SAP (Lauer et al. 2012), open reimplementation.

    For every heavy atom i, SAP(i) is the sum over heavy atoms j within ``radius``
    (A) of ``RSA(res_j) * hydrophobicity(res_j)``, where ``RSA`` is the residue's
    relative solvent accessibility (its SASA over the Tien 2013 theoretical maximum)
    and hydrophobicity is the Black & Mould scale normalised to Gly = 0. The molecule
    SAP is the sum over atoms of SAP(i); this returns it divided by the residue count,
    matching the Rosetta backend's return. Not bit-identical to Rosetta ``calculate_sap``.
    """
    if not MAX_ASA or not HYDROPHOBICITY:
        raise RuntimeError(
            "surface.py constant tables (MAX_ASA, HYDROPHOBICITY) are unset; "
            "they must be populated from their primary sources before use."
        )
    import biotite.structure as struc

    path, cleanup = _load_pdb_path(pdb_file, tmpdir)
    try:
        atoms = _structure(path)
        atoms = atoms[atoms.element != "H"]
        sasa = _per_atom_sasa(atoms)

        # residue-level SASA -> relative accessibility -> per-atom weight RSA*hydrophobicity
        res_ids = atoms.res_id
        chain_ids = atoms.chain_id
        res_names = atoms.res_name
        # group atoms by (chain, res_id)
        keys = np.char.add(np.char.add(chain_ids.astype(str), "_"), res_ids.astype(str))
        weight = np.zeros(len(atoms), dtype=float)
        n_res = 0
        for k in np.unique(keys):
            mask = keys == k
            resname = res_names[mask][0]
            max_asa = MAX_ASA.get(resname)
            h = HYDROPHOBICITY.get(resname)
            n_res += 1
            if max_asa is None or h is None or max_asa <= 0:
                continue  # unknown residue (ligand/noncanonical): contributes 0, still counted
            rsa = min(sasa[mask].sum() / max_asa, 1.0)
            weight[mask] = rsa * h

        # atom neighbourhood sum within `radius`
        cell_list = struc.CellList(atoms, cell_size=radius)
        total_sap = 0.0
        neighbours = cell_list.get_atoms(atoms.coord, radius=radius)  # (n_atoms, max_nbr), -1 padded
        for i in range(len(atoms)):
            nbr = neighbours[i]
            nbr = nbr[nbr >= 0]
            total_sap += weight[nbr].sum()
    finally:
        if cleanup is not None:
            cleanup()
    return total_sap / float(n_res) if n_res else 0.0
