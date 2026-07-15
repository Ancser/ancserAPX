"""Production model registry.

Factors are reusable input signals.  A model is the scoring system that turns
those inputs (or any future non-factor signal source) into a cross-sectional
score.  Keeping the registry explicit prevents the UI or a saved live config
from silently selecting an idea that has no backtest/live implementation.
"""

from __future__ import annotations

from typing import Dict, List


DEFAULT_MODEL_ID = "factor_composite"


MODEL_REGISTRY: Dict[str, Dict[str, object]] = {
    DEFAULT_MODEL_ID: {
        "id": DEFAULT_MODEL_ID,
        "label": "Factor Composite",
        "description": "Cross-sectional ranked factor blend used by the current production strategies.",
        "uses_factors": True,
        "production_ready": True,
    },
}


def list_models(*, production_only: bool = True) -> List[Dict[str, object]]:
    """Return serialisable model metadata for the API/UI."""
    models = []
    for definition in MODEL_REGISTRY.values():
        if production_only and not bool(definition.get("production_ready", False)):
            continue
        models.append(dict(definition))
    return models


def require_model(model_id: str | None) -> Dict[str, object]:
    """Return a production model definition or raise a clear configuration error."""
    resolved = str(model_id or DEFAULT_MODEL_ID).strip() or DEFAULT_MODEL_ID
    definition = MODEL_REGISTRY.get(resolved)
    if definition is None:
        raise ValueError(f"Unknown model_id '{resolved}'.")
    if not bool(definition.get("production_ready", False)):
        raise ValueError(f"Model '{resolved}' is not approved for production/backtest use.")
    return dict(definition)

