"""Unit tests for runtime environment invariants and lazy submodule imports."""

import importlib
import subprocess
import sys
import types
import warnings

from packaging.version import Version
import pytest


def test_torch_is_cpu_only_and_jax_version():
    """Verify CPU-only PyTorch and JAX >= 0.12 in the unified environment."""
    import jax
    import torch

    assert torch.version.cuda is None, (
        f"Expected CPU-only torch wheel (torch.version.cuda is None), got {torch.version.cuda!r}"
    )
    assert Version(jax.__version__) >= Version("0.12.0"), (
        f"Expected jax >= 0.12.0, got {jax.__version__}"
    )


def test_core_and_predictor_packages_import_without_pyrosetta_or_dnachisel():
    """Verify all core packages and pipelines import cleanly without optional extras."""
    modules_to_import = (
        "flexcraft",
        "flexcraft.structure",
        "salad",
        "rfd4fx",
        "joltz",
        "boltz",
        "protenix",
        "esmjfold2",
        "flexcraft.rosetta.relax",
        "flexcraft.protocols.bindcraft_filter",
        "flexcraft.pipelines.denovo",
        "flexcraft.pipelines.order.select",
        "flexcraft.pipelines.order.codon_optimize",
    )
    for mod_name in modules_to_import:
        mod = importlib.import_module(mod_name)
        assert mod is not None

    assert "pyrosetta" not in sys.modules
    assert "dnachisel" not in sys.modules


def test_structure_lazy_getattr_in_fresh_subprocess():
    """Verify flexcraft.structure keeps heavy predictor backends unloaded until accessed."""
    code = """
import sys
import flexcraft.structure as st

heavy_backends = (
    "flexcraft.structure.af",
    "flexcraft.structure.boltz",
    "flexcraft.structure.protenix",
    "flexcraft.structure.esmfold2",
    "joltz",
    "boltz",
    "protenix",
    "esmjfold2",
)
loaded_early = [m for m in heavy_backends if m in sys.modules]
assert not loaded_early, f"Backends loaded eagerly on import flexcraft.structure: {loaded_early}"

metrics_mod = st.metrics
assert "flexcraft.structure.metrics" in sys.modules
still_unloaded = [m for m in ("joltz", "boltz", "protenix", "esmjfold2") if m in sys.modules]
assert not still_unloaded, f"Heavy predictors loaded when accessing st.metrics: {still_unloaded}"
"""
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"Subprocess failed:\nSTDOUT: {proc.stdout}\nSTDERR: {proc.stderr}"


def test_structure_dir_and_attribute_error():
    """Verify flexcraft.structure.__dir__ and AttributeError on unknown attributes."""
    import flexcraft.structure as st

    names = dir(st)
    for expected in ("af", "boltz", "protenix", "esmfold2", "af3_like", "common", "metrics"):
        assert expected in names

    with pytest.raises(AttributeError, match="has no attribute 'nonexistent_backend'"):
        _ = st.nonexistent_backend


def test_check_runtime_environment_warns_on_simulated_cuda_torch(monkeypatch):
    """Verify check_runtime_environment emits a RuntimeWarning if CUDA torch is detected."""
    import flexcraft.structure as st

    fake_torch = types.SimpleNamespace(version=types.SimpleNamespace(cuda="12.4"))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    with pytest.warns(RuntimeWarning, match="CUDA-enabled PyTorch"):
        st.check_runtime_environment(reset=True)

    st.check_runtime_environment(reset=True)


def test_lazy_pyrosetta_import_error_message(monkeypatch):
    """Verify flexcraft.rosetta.relax raises an actionable ImportError when PyRosetta is absent."""
    from flexcraft.rosetta import relax

    monkeypatch.setitem(sys.modules, "pyrosetta", None)
    with pytest.raises(ImportError, match="PyRosetta is not installed") as exc_info:
        relax._require_pyrosetta()
    assert "uv sync --inexact" in str(exc_info.value)

    with pytest.raises(ImportError, match="PyRosetta is not installed"):
        relax.fastrelax("dummy.pdb")

    with pytest.raises(ImportError, match="PyRosetta is not installed"):
        relax.pair_energies("dummy.pdb")


def test_lazy_dnachisel_import_error_message(monkeypatch):
    """Verify flexcraft.pipelines.order raises an actionable ImportError when dnachisel is absent."""
    from flexcraft.pipelines.order import common as order_common

    monkeypatch.setitem(sys.modules, "dnachisel", None)
    with pytest.raises(ImportError, match="dnachisel is required") as exc_info:
        order_common._load_dnachisel()
    assert "uv sync --extra order" in str(exc_info.value)

    with pytest.raises(ImportError, match="uv sync --extra order"):
        order_common.codon_optimize("ACDEFG")


def test_flexcraft_tools_deprecation_warning():
    """Verify importing flexcraft.tools emits a DeprecationWarning."""
    sys.modules.pop("flexcraft.tools", None)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        importlib.import_module("flexcraft.tools")

    dep_warnings = [
        w for w in caught
        if issubclass(w.category, DeprecationWarning)
        and "flexcraft.tools is deprecated" in str(w.message)
    ]
    assert len(dep_warnings) == 1
