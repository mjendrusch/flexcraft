"""Interface scoring via tmol -- the energetic metrics from
``flexcraft.rosetta.interface_analyzer.score_interface`` without PyRosetta.

tmol's beta2016 full-atom energy gives a separated interface ``dG`` and the binder's
own energy. The dict returned here matches the keys of the PyRosetta ``score_interface``
so it is drop-in; metrics that are PyRosetta-specific scoreterms with no tmol equivalent
(shape complementarity / ``interface_sc``, SAP, packstat, buried-unsat hbonds, SASA-derived
terms) are returned as ``None`` and dropped by the same ``is not None`` filter the
PyRosetta path uses -- they are not faked.
"""

import tempfile

import numpy as np

from flexcraft.files.pdb import PDBFile
from .relax import _require_tmol, _select_device

_PYROSETTA_ONLY = (
    "surface_hydrophobicity", "binder_sap_score", "binder_sap_per_res",
    "contact_molecular_surface", "interface_sc", "interface_packstat",
    "interface_dSASA", "interface_dG_SASA_ratio", "interface_fraction",
    "interface_hydrophobicity", "interface_nres", "interface_hbond_percentage",
    "interface_delta_unsat_hbonds", "interface_delta_unsat_hbonds_percentage",
)


def _write_chains(in_pdb, chains, out_pdb):
    from biotite.structure.io.pdb import PDBFile as BiotitePDB

    arr = BiotitePDB.read(in_pdb).get_structure(model=1)
    sub = arr[np.isin(arr.chain_id, list(chains))]
    out = BiotitePDB()
    out.set_structure(sub)
    out.write(out_pdb)


def _score_pdb(pdb_path, tmol, dev):
    ps = tmol.pose_stack_from_pdb(pdb_path, dev)
    sfxn = tmol.beta2016_score_function(dev)
    scorer = sfxn.render_whole_pose_scoring_module(ps)
    return float(scorer(ps.coords).sum())


def score_interface(
    pdb_file: "str | PDBFile",
    binder_chains=("B",),
    target_chains=("A",),
    device=None,
):
    """Interface energetics via tmol: ``dG = E(complex) - E(binder) - E(target)``.

    Returns ``(interface_scores, interface_AA)`` with the same keys as the PyRosetta
    ``score_interface``; tmol fills the energetic terms, the rest are ``None``.
    """
    tmol, torch = _require_tmol()
    dev = _select_device(device, torch)
    if isinstance(pdb_file, PDBFile):
        pdb_file = pdb_file.path
    binder_chains, target_chains = list(binder_chains), list(target_chains)

    with tempfile.TemporaryDirectory() as tmp:
        complex_pdb = f"{tmp}/complex.pdb"
        binder_pdb = f"{tmp}/binder.pdb"
        target_pdb = f"{tmp}/target.pdb"
        _write_chains(pdb_file, binder_chains + target_chains, complex_pdb)
        _write_chains(pdb_file, binder_chains, binder_pdb)
        _write_chains(pdb_file, target_chains, target_pdb)
        e_complex = _score_pdb(complex_pdb, tmol, dev)
        e_binder = _score_pdb(binder_pdb, tmol, dev)
        e_target = _score_pdb(target_pdb, tmol, dev)

    interface_scores = {
        "binder_score": round(e_binder, 2),
        "interface_dG": round(e_complex - e_binder - e_target, 2),
        "interface_interface_hbonds": None,  # TODO: count via tmol ScoreType.hbond block-pairs
    }
    for k in _PYROSETTA_ONLY:
        interface_scores[k] = None
    interface_scores = {k: v for k, v in interface_scores.items() if v is not None}

    # per-chain amino-acid composition of the binder (sequence-only; no PyRosetta)
    from biotite.structure.io.pdb import PDBFile as BiotitePDB
    from biotite.structure import get_residues
    from biotite.sequence import ProteinSequence

    arr = BiotitePDB.read(pdb_file).get_structure(model=1)
    binder = arr[np.isin(arr.chain_id, binder_chains)]
    interface_AA = {aa: 0 for aa in "ACDEFGHIKLMNPQRSTVWY"}
    for resn in get_residues(binder)[1]:
        try:
            interface_AA[ProteinSequence.convert_letter_3to1(resn)] += 1
        except KeyError:
            pass
    return interface_scores, interface_AA
