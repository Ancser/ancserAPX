from backend.alpha.models import DEFAULT_MODEL_ID, list_models, require_model
from backend.execution.strategy import resolve_factor_weights
from backend.utils.performance import fifo_realized_pnl


def test_only_registered_production_models_are_exposed():
    models = list_models()
    assert models
    assert models[0]["id"] == DEFAULT_MODEL_ID
    assert models[0]["uses_factors"] is True
    assert require_model(None)["id"] == DEFAULT_MODEL_ID


def test_unknown_model_never_silently_falls_back():
    try:
        require_model("not-a-real-model")
    except ValueError as exc:
        assert "Unknown model_id" in str(exc)
    else:
        raise AssertionError("unknown model should have been rejected")


def test_fifo_realized_pnl_is_realized_only_and_auditable():
    fills = [
        {"time": "2026-01-01T14:30:00Z", "symbol": "ABC", "side": "buy", "qty": 2, "price": 10},
        {"time": "2026-01-02T14:30:00Z", "symbol": "ABC", "side": "buy", "qty": 1, "price": 20},
        {"time": "2026-01-03T14:30:00Z", "symbol": "ABC", "side": "sell", "qty": 2.5, "price": 30},
    ]
    result = fifo_realized_pnl(fills)
    # 2 * (30 - 10) + 0.5 * (30 - 20)
    assert result["realized_pnl"] == 45.0
    assert result["unmatched_sell_qty"] == {}


def test_fifo_flags_sells_without_available_cost_basis():
    result = fifo_realized_pnl([
        {"time": "2026-01-03", "symbol": "OLD", "side": "sell", "qty": 3, "price": 30},
    ])
    assert result["realized_pnl"] == 0.0
    assert result["unmatched_sell_qty"] == {"OLD": 3.0}


def test_custom_live_factor_weights_use_the_ui_values():
    weights = resolve_factor_weights(["Momentum", "Reversion"], {
        "Momentum": 0.7,
        "Reversion": 0.3,
    })
    assert weights == {"Momentum": 0.7, "Reversion": 0.3}


def test_custom_live_factor_weights_fall_back_to_equal_when_auto():
    weights = resolve_factor_weights(["Momentum", "Reversion"], {
        "Momentum": 0.0,
        "Reversion": 0.0,
    })
    assert weights == {"Momentum": 0.5, "Reversion": 0.5}
