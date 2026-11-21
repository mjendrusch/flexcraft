"""Open, GPU, licence-free physics backend for flexcraft via tmol.

A drop-in alternative to ``flexcraft.rosetta`` (PyRosetta): ``fastrelax``,
``score_interface`` and the surface/SAP metrics with the same signatures, backed by
upstream tmol (https://github.com/uw-ipd/tmol, BSD-3) for relax/interface energetics
and biotite SASA for the surface terms, instead of PyRosetta. Select it with the
pipeline ``relax_backend="tmol"`` flag.
"""

from .relax import fastrelax  # noqa: F401
from .interface import score_interface  # noqa: F401
from .surface import get_sasa, sap_per_residue, surface_hydrophobicity  # noqa: F401
