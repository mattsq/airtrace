"""Registry signature guarantees for models.

These tests help catch constructor contract regressions early, ensuring that
models can be instantiated by generic tooling (e.g., model validation) without
needing bespoke argument handling.
"""

from __future__ import annotations

import inspect
from typing import Iterable

import pytest

from airtrace.models.registry import MODEL_REGISTRY


def _required_positional_parameters(parameters: Iterable[inspect.Parameter]) -> list[str]:
    """Return required positional-or-keyword parameters excluding ``self``.

    Args:
        parameters: Iterable of parameters from an ``inspect.Signature``.

    Returns:
        List of required positional-or-keyword parameter names.
    """

    required: list[str] = []
    for param in parameters:
        if param.name == "self":
            continue
        if param.kind not in {
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        }:
            continue
        if param.default is inspect._empty:
            required.append(param.name)
    return required


def test_registered_models_have_no_extra_required_positional_args() -> None:
    """Ensure registry models can be constructed with only ``input_dim``/``output_dim``.

    This guards against introducing additional mandatory positional parameters
    (like ``pred_len``) that generic tooling would not supply when instantiating
    models for validation.
    """

    allowed_required = {"input_dim", "output_dim"}
    model_specific_allowlist = {
        # SOFTS explicitly requires sequence and prediction lengths.
        "softs": allowed_required | {"seq_len", "pred_len"},
    }

    for name, cls in MODEL_REGISTRY.items():
        signature = inspect.signature(cls.__init__)
        required = set(_required_positional_parameters(signature.parameters.values()))
        allowed = model_specific_allowlist.get(name, allowed_required)
        assert required.issubset(
            allowed
        ), f"Model '{name}' requires unsupported positional args: {sorted(required - allowed)}"
