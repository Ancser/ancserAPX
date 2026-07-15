from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research_v2.fundamental_value import (
    PESectorStrategyConfig,
    build_pe_sector_balanced_portfolio,
    joint_sector_size_residual,
    sector_relative_market_cap_filter,
    size_filter_sector_diagnostics,
    validate_fundamental_snapshot,
)


def _snapshot() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": ["A1", "A2", "A3", "A4", "B1", "B2", "B3", "B4"],
            "sector": ["A"] * 4 + ["B"] * 4,
            "pe_ttm": [50.0, -4.0, 10.0, 5.0, np.nan, 0.0, 9.0, 7.0],
            "market_cap": [1.0, 2.0, 3.0, 4.0, 100.0, 200.0, 300.0, 400.0],
            "available_at": ["2026-07-14T20:00:00Z"] * 8,
        }
    )


def test_sector_median_filter_avoids_global_sector_retention_bias():
    screened = sector_relative_market_cap_filter(_snapshot(), quantile=0.5, min_sector_names=4)
    assert set(screened.loc[screened["sector_size_eligible"], "symbol"]) == {"A3", "A4", "B3", "B4"}
    diagnostics = size_filter_sector_diagnostics(_snapshot(), quantile=0.5).set_index("sector")
    assert diagnostics.loc["A", "sector_relative_retention"] == pytest.approx(0.5)
    assert diagnostics.loc["B", "sector_relative_retention"] == pytest.approx(0.5)
    assert diagnostics.loc["A", "global_retention"] == pytest.approx(0.0)
    assert diagnostics.loc["B", "global_retention"] == pytest.approx(1.0)


def test_pe_only_strategy_excludes_nonpositive_and_equalizes_sector_gross_budget():
    screened, portfolio = build_pe_sector_balanced_portfolio(
        _snapshot(),
        PESectorStrategyConfig(
            top_n=4,
            gross_target=1.0,
            sector_market_cap_quantile=0.5,
            min_sector_names=4,
        ),
        decision_at="2026-07-14T20:00:00Z",
    )
    assert set(portfolio["symbol"]) == {"A3", "A4", "B3", "B4"}
    assert portfolio.set_index("symbol").loc["A4", "pe_score"] > portfolio.set_index("symbol").loc["A3", "pe_score"]
    assert portfolio.set_index("symbol").loc["B4", "pe_score"] > portfolio.set_index("symbol").loc["B3", "pe_score"]
    assert portfolio.groupby("sector")["weight"].sum().to_dict() == pytest.approx({"A": 0.5, "B": 0.5})
    assert portfolio["weight"].sum() == pytest.approx(1.0)
    assert not screened.loc[screened["pe_ttm"].fillna(0) <= 0, "pe_eligible"].any()


def test_pe_top_n_is_global_and_sector_budget_does_not_change_selection():
    snapshot = pd.DataFrame(
        {
            "symbol": ["A1", "A2", "A3", "A4", "B1", "B2", "B3", "B4"],
            "sector": ["A"] * 4 + ["B"] * 4,
            "pe_ttm": [1.0, 2.0, 3.0, 4.0, 100.0, 101.0, 102.0, 103.0],
            "market_cap": np.nan,
            "available_at": "2026-07-14T20:00:00Z",
        }
    )
    _, portfolio = build_pe_sector_balanced_portfolio(
        snapshot,
        PESectorStrategyConfig(
            top_n=4,
            gross_target=1.0,
            apply_sector_size_filter=False,
        ),
        decision_at="2026-07-14T20:00:00Z",
    )
    assert set(portfolio["symbol"]) == {"A1", "A2", "A3", "A4"}
    assert portfolio.groupby("sector")["weight"].sum().to_dict() == pytest.approx({"A": 1.0})


def test_point_in_time_validation_fails_closed_on_future_snapshot():
    with pytest.raises(ValueError, match="look-ahead"):
        validate_fundamental_snapshot(_snapshot(), decision_at="2026-07-14T19:59:59Z")
    validated = validate_fundamental_snapshot(
        _snapshot(), decision_at="2026-07-14T20:00:00Z"
    )
    assert len(validated) == 8


def test_validation_rejects_null_identifiers():
    snapshot = _snapshot()
    snapshot.loc[0, "sector"] = None
    with pytest.raises(ValueError, match="non-null"):
        validate_fundamental_snapshot(snapshot, decision_at="2026-07-14T20:00:00Z")


def test_joint_residual_is_orthogonal_to_size_and_sector_dummies():
    rows = []
    for sector_index, sector in enumerate(["A", "B", "C"]):
        for index in range(8):
            cap = float(np.exp(4.0 + sector_index + index / 5.0))
            earnings_yield = 0.03 + 0.015 * np.log(cap) + 0.02 * sector_index + ((index % 3) - 1) * 0.001
            rows.append(
                {
                    "symbol": f"{sector}{index}",
                    "sector": sector,
                    "pe_ttm": 1.0 / earnings_yield,
                    "market_cap": cap,
                    "earnings_yield": earnings_yield,
                    "available_at": "2026-07-14T20:00:00Z",
                }
            )
    snapshot = pd.DataFrame(rows).sample(frac=1.0, random_state=17)
    snapshot.index = np.arange(100, 100 + len(snapshot))
    residual = joint_sector_size_residual(snapshot)
    assert residual.notna().all()
    log_cap = np.log(snapshot["market_cap"].to_numpy(float))
    assert abs(float(np.corrcoef(residual.to_numpy(float), log_cap)[0, 1])) < 1e-10
    means = pd.DataFrame({"sector": snapshot["sector"], "residual": residual}).groupby("sector")["residual"].mean()
    assert float(means.abs().max()) < 1e-12
