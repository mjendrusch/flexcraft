"""Lazy submodule access and startup runtime checks for flexcraft.structure."""

import importlib
import os
import sys
import warnings

_LAZY_SUBMODULES = (
    "af",
    "boltz",
    "protenix",
    "esmfold2",
    "af3_like",
    "common",
    "metrics",
)

__all__ = [*_LAZY_SUBMODULES, "check_runtime_environment"]

_WARNED_TORCH_CUDA = False
_WARNED_LINUX_CPU_JAX = False


def check_runtime_environment(*, reset: bool = False) -> None:
    """Emit one-time warnings for CUDA PyTorch or CPU-only JAX on Linux GPUs.

    Args:
        reset: Reset warning guards before checking.
    """
    global _WARNED_TORCH_CUDA, _WARNED_LINUX_CPU_JAX
    if reset:
        _WARNED_TORCH_CUDA = False
        _WARNED_LINUX_CPU_JAX = False

    torch_module = sys.modules.get("torch")
    if not _WARNED_TORCH_CUDA and torch_module is not None:
        torch_cuda_version = getattr(getattr(torch_module, "version", None), "cuda", None)
        if torch_cuda_version is not None:
            _WARNED_TORCH_CUDA = True
            warnings.warn(
                f"CUDA-enabled PyTorch (torch.version.cuda={torch_cuda_version}) is loaded. "
                "flexcraft executes models in JAX and expects CPU-only PyTorch wheels; "
                "run `uv sync` in the flexcraft workspace root to avoid CUDA runtime conflicts.",
                RuntimeWarning,
                stacklevel=2,
            )

    if not _WARNED_LINUX_CPU_JAX and sys.platform == "linux" and os.path.exists("/dev/nvidia0"):
        try:
            import jax

            default_backend = jax.default_backend()
        except Exception:
            default_backend = None
        if default_backend == "cpu":
            _WARNED_LINUX_CPU_JAX = True
            warnings.warn(
                "NVIDIA GPU (/dev/nvidia0) detected on Linux while JAX default backend is 'cpu'. "
                "Install GPU JAX support with `uv sync --extra cuda12` or `uv sync --extra cuda13`.",
                RuntimeWarning,
                stacklevel=2,
            )


def __getattr__(name: str):
    if name in _LAZY_SUBMODULES:
        module = importlib.import_module(f"{__name__}.{name}")
        globals()[name] = module
        check_runtime_environment()
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals().keys()) | set(__all__))


check_runtime_environment()
