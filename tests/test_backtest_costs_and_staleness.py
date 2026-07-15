from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backend.backtest.engine import (
    BacktestCostConfig,
    BacktestEngine,
    _estimate_trade_costs,
    compute_metrics,
)


def _synthetic_factor_data(periods: int = 16) -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-02", periods=periods)
    rows = []
    for i, date in enumerate(dates):
        # A leads for the first week; B leads from the second week onward.
        a_score = 2.0 if i < 5 else 0.0
        b_score = 0.0 if i < 5 else 2.0
        for symbol, score in (("A", a_score), ("B", b_score)):
            rows.append(
                {
                    "timestamp": date,
                    "symbol": symbol,
                    "close": 100.0,
                    "high": 101.0,
                    "low": 99.0,
                    "volume": 1_000_000.0,
                    "fwd_ret": 0.001 if symbol == "A" else -0.001,
                    "factor_ts_mom": score,
                }
            )
    return pd.DataFrame(rows)


def _run_synthetic(*, delay: int, slippage_bps: float = 0.0):
    engine = BacktestEngine(initial_capital=10_000.0)
    data = _synthetic_factor_data()
    engine.fetch_and_prepare_data = lambda *_args, **_kwargs: data.copy()
    return engine.run_strategy(
        symbols=["A", "B"],
        start_date="2024-01-02",
        end_date="2024-01-31",
        sleeves=[
            {
                "name": "Core",
                "alloc": 1.0,
                "factors": ["Momentum"],
                "weights": {"Momentum": 1.0},
                "winner_lock": False,
            }
        ],
        leverage=1.0,
        top_n=1,
        rebalance_days=5,
        commission_bps=0.0,
        slippage_bps=slippage_bps,
        regulatory_sell_bps=0.0,
        signal_delay_days=delay,
    )


def test_trade_cost_components_charge_each_traded_dollar_once() -> None:
    costs = _estimate_trade_costs(
        {"A": 0.50, "B": 0.50},
        {"B": 0.25, "C": 0.75},
        100_000.0,
        BacktestCostConfig(
            commission_bps=1.0,
            slippage_bps=2.0,
            regulatory_sell_bps=3.0,
        ),
    )

    assert costs["buy_turnover"] == pytest.approx(0.75)
    assert costs["sell_turnover"] == pytest.approx(0.75)
    assert costs["gross_turnover"] == pytest.approx(1.50)
    assert costs["one_way_turnover"] == pytest.approx(0.75)
    assert costs["traded_notional"] == pytest.approx(150_000.0)
    assert costs["commission_cost"] == pytest.approx(15.0)
    assert costs["slippage_cost"] == pytest.approx(30.0)
    assert costs["regulatory_cost"] == pytest.approx(22.5)
    assert costs["transaction_cost"] == pytest.approx(67.5)


def test_alpaca_appropriate_defaults_separate_commission_and_slippage() -> None:
    config = BacktestCostConfig()
    assert config.commission_bps == 0.0
    assert config.slippage_bps == 5.0
    assert config.regulatory_sell_bps == 0.0
    with pytest.raises(ValueError):
        BacktestCostConfig(slippage_bps=-1.0)


def test_signal_delay_is_exactly_five_sessions_and_never_looks_forward() -> None:
    _, _, normal_holdings = _run_synthetic(delay=0)
    delayed_result, _, delayed_holdings = _run_synthetic(delay=5)

    normal = normal_holdings.reset_index()
    delayed = delayed_holdings.reset_index()
    second_week = pd.Timestamp("2024-01-09")
    normal_row = normal.loc[normal["date"] == second_week].iloc[0]
    delayed_row = delayed.loc[delayed["date"] == second_week].iloc[0]

    assert normal_row["signal_date"] == second_week
    assert normal_row["long"] == "B"
    assert delayed_row["signal_date"] == pd.Timestamp("2024-01-02")
    assert delayed_row["long"] == "A"
    assert delayed_result.loc[:"2024-01-08", "gross_turnover"].sum() == 0.0


def test_cost_columns_reconcile_with_metrics_and_reduce_equity() -> None:
    free_result, _, _ = _run_synthetic(delay=0, slippage_bps=0.0)
    costed_result, _, _ = _run_synthetic(delay=0, slippage_bps=5.0)
    metrics = compute_metrics(costed_result, 10_000.0, holding_period_days=5)

    assert costed_result["commission_cost"].sum() == 0.0
    assert costed_result["regulatory_cost"].sum() == 0.0
    assert costed_result["slippage_cost"].sum() > 0.0
    assert costed_result["transaction_cost"].sum() == pytest.approx(
        costed_result["slippage_cost"].sum()
    )
    assert metrics["total_transaction_cost"] == pytest.approx(
        round(float(costed_result["transaction_cost"].sum()), 2)
    )
    assert metrics["total_gross_turnover"] == pytest.approx(
        round(float(costed_result["gross_turnover"].sum()), 4)
    )
    assert np.isfinite(costed_result["equity"]).all()
    assert costed_result["equity"].iloc[-1] < free_result["equity"].iloc[-1]


def test_legacy_mwu_neutralization_path_also_accounts_for_costs() -> None:
    engine = BacktestEngine(initial_capital=10_000.0)
    result, _, _ = engine.run_simulation(
        _synthetic_factor_data(),
        active_factors=["Momentum"],
        leverage=1.0,
        use_mwu=False,
        use_vol_target=False,
        top_n=1,
        rebalance_days=5,
        commission_bps=0.0,
        slippage_bps=5.0,
        regulatory_sell_bps=0.0,
    )

    assert result["gross_turnover"].sum() > 0.0
    assert result["slippage_cost"].sum() > 0.0
    assert result["transaction_cost"].sum() == pytest.approx(
        result["slippage_cost"].sum()
    )
