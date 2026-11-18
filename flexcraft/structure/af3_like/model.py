"""Model-agnostic wrapper and input specification for structure predictors."""

from __future__ import annotations

from copy import copy
from typing import TYPE_CHECKING, Any

from flexcraft.structure.common._data import InputSpec, ModelInput

if TYPE_CHECKING:
    from flexcraft.structure.af import AFInput, AFResult
    from flexcraft.structure.boltz._data import JoltzInput
    from flexcraft.structure.boltz._result import JoltzResult
    from flexcraft.structure.esmfold2._data import ESMFold2Input
    from flexcraft.structure.esmfold2._result import ESMFold2Result
    from flexcraft.structure.protenix._data import ProtenixInput
    from flexcraft.structure.protenix._result import ProtenixResult

    AF3LikeResult = ProtenixResult | JoltzResult | ESMFold2Result | AFResult
    AF3LikeInput = ProtenixInput | JoltzInput | ESMFold2Input | AFInput


def _get_model_kind(model: str) -> str:
    """Resolve predictor backend kind from a model name string."""
    if model == "joltz" or model.startswith("joltz"):
        return "joltz"
    if model.startswith("protenix"):
        return "protenix"
    if model.startswith("esmfold2"):
        return "esmfold2"
    if model.startswith("af2"):
        return "af2"
    raise ValueError(
        f"Unsupported AF3Like model {model!r}. "
        "Expected 'joltz', 'protenix_*', 'esmfold2_*', or 'af2_*'."
    )


def _get_model_spec_cls(kind: str) -> type[InputSpec]:
    """Lazily import and return the InputSpec subclass for `kind`."""
    if kind == "joltz":
        from flexcraft.structure.boltz._data import JoltzSpec

        return JoltzSpec
    if kind == "protenix":
        from flexcraft.structure.protenix._data import ProtenixSpec

        return ProtenixSpec
    if kind == "esmfold2":
        from flexcraft.structure.esmfold2._data import ESMFold2Spec

        return ESMFold2Spec
    if kind == "af2":
        from flexcraft.structure.af import AF2Spec

        return AF2Spec
    raise ValueError(f"Unknown model kind {kind!r}.")


def _any_spec_to_model_spec(spec: InputSpec, kind: str) -> InputSpec:
    """Convert any InputSpec into the backend-specific InputSpec for `kind`."""
    spec_cls = _get_model_spec_cls(kind)
    model_spec = spec_cls(
        *copy(spec.chains),
        templates=copy(spec.templates),
        constraints=copy(spec.constraints),
    )
    for attr_name in ("precomputed_msas", "temporaries", "_pad_spec"):
        attr_value = getattr(spec, attr_name, None)
        if attr_value is not None:
            setattr(model_spec, attr_name, copy(attr_value))
    return model_spec


class AF3LikeSpec(InputSpec):
    """Backend-agnostic InputSpec dispatching to the requested predictor."""

    def to_features(self, model: str = "joltz", **kwargs) -> dict:
        model_kind = _get_model_kind(model)
        return _any_spec_to_model_spec(self, model_kind).to_features(**kwargs)

    def to_input(self, model: str = "joltz", **kwargs):
        model_kind = _get_model_kind(model)
        return _any_spec_to_model_spec(self, model_kind).to_input(**kwargs)


class AF3Like:
    """Model-agnostic wrapper across Joltz2, Protenix, ESMFold2, and AF2."""

    def __init__(self, model: str = "protenix_base_default_v1.0.0", **kwargs):
        """Initialize the requested predictor backend lazily.

        Args:
            model: Predictor identifier (`joltz`, `protenix_*`, `esmfold2_*`, `af2_*`).
            **kwargs: Backend keyword arguments (e.g. `cache`, `load_esmc`).
        """
        self.model_kind = _get_model_kind(model)
        if self.model_kind == "joltz":
            from flexcraft.structure.boltz._model import Joltz2

            self.base = Joltz2(**kwargs)
        elif self.model_kind == "protenix":
            from flexcraft.structure.protenix._model import Protenix

            self.base = Protenix(model=model, **kwargs)
        elif self.model_kind == "esmfold2":
            from flexcraft.structure.esmfold2._model import ESMFold2

            self.base = ESMFold2(model=model, **kwargs)
        elif self.model_kind == "af2":
            from flexcraft.structure.af import AFWrapper

            self.base = AFWrapper(model=model, **kwargs)

    def predictor(self, **kwargs):
        """Return an eager predictor accepting an InputSpec, ModelInput, or AFInput."""
        _predictor = self.base.predictor(**kwargs)

        def _wrapper(key, any_spec: InputSpec | ModelInput | Any, **call_kwargs):
            if isinstance(any_spec, InputSpec):
                spec = _any_spec_to_model_spec(any_spec, self.model_kind)
                return _predictor(key, spec, **call_kwargs)
            return _predictor(key, any_spec, **call_kwargs)

        return _wrapper

    def evaluator(self, **kwargs):
        """Return the underlying JIT-compatible evaluator for this backend."""
        return self.base.evaluator(**kwargs)
