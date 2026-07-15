from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research_v2.delta_pe import (
    DeltaPEConfig,
    build_delta_pe_portfolio,
    compute_delta_pe_features,
)


def _snapshots() -> tuple[pd.DataFrame, pd.DataFrame]:
    symbols = ["A1", "A2", "A3", "B1", "B2", "B3"]
    sectors = ["A", "A", "A", "B", "B", "B"]
    previous = pd.DataFrame(
        {
            "symbol": symbols,
            "sector": sectors,
            "pe_ttm": [10.0] * 6,
            "trailing_eps": [10.0] * 6,
            "market_cap": [100.0] * 6,
            "available_at": ["2026-01-01T00:00:00Z"] * 6,
        }
    )
    current = previous.copy()
    current["available_at"] = "2026-02-01T00:00:00Z"
    return previous, current


def test_sector_common_shock_is_removed_from_stock_relative_component():
    previous, current = _snapshots()
    current.loc[current["sector"] == "A", "pe_ttm"] = 20.0
    features = compute_delta_pe_features(
        previous,
        current,
        decision_at="2026-02-02T00:00:00Z",
        config=DeltaPEConfig(min_sector_names=2),
    ).set_index("symbol")
    assert features.loc["A1", "sector_delta_log_pe"] == pytest.approx(np.log(2.0))
    assert features.loc[["A1", "A2", "A3"], "relative_delta_log_pe"].abs().max() < 1e-12
    assert not features.loc[["A1", "A2", "A3"], "surge_and_eligible"].any()
    assert features.loc[["A1", "A2", "A3"], "literal_surge_and_eligible"].all()


def test_idiosyncratic_expansion_passes_raw_and_but_eps_collapse_fails_quality_guard():
    previous, current = _snapshots()
    current.loc[current["sector"] == "A", "pe_ttm"] = [40.0, 20.0, 20.0]
    current.loc[current["sector"] == "A", "trailing_eps"] = [2.5, 10.0, 10.0]
    features = compute_delta_pe_features(
        previous,
        current,
        decision_at="2026-02-02T00:00:00Z",
        config=DeltaPEConfig(min_sector_names=2),
    ).set_index("symbol")
    assert bool(features.loc["A1", "surge_and_eligible"])
    assert bool(features.loc["A1", "literal_surge_and_eligible"])
    assert not bool(features.loc["A1", "reported_eps_guarded_surge_eligible"])
    assert not bool(features.loc["A1", "reported_eps_guarded_literal_surge_eligible"])
    assert np.isnan(features.loc["A1", "reported_eps_guarded_surge_score"])
    assert not bool(features.loc["A1", "reported_eps_guard_is_corporate_action_certified"])


def test_missing_pe_transitions_never_receive_delta_scores():
    previous, current = _snapshots()
    previous.loc[previous["symbol"] == "B1", "pe_ttm"] = np.nan
    current.loc[current["symbol"] == "B2", "pe_ttm"] = np.nan
    features = compute_delta_pe_features(
        previous,
        current,
        decision_at="2026-02-02T00:00:00Z",
        config=DeltaPEConfig(min_sector_names=2),
    ).set_index("symbol")
    assert features.loc["B1", "pe_state_transition"] == "missing_or_nonpositive_to_positive"
    assert features.loc["B2", "pe_state_transition"] == "positive_to_missing_or_nonpositive"
    assert np.isnan(features.loc["B1", "delta_log_pe"])
    assert np.isnan(features.loc["B2", "delta_log_pe"])


def test_point_in_time_shuffle_invariance_and_portfolio_does_not_force_top_n():
    previous, current = _snapshots()
    current.loc[current["sector"] == "A", "pe_ttm"] = [40.0, 20.0, 20.0]
    config = DeltaPEConfig(top_n=5, min_sector_names=2, eps_log_floor=-1.0)
    with pytest.raises(ValueError, match="look-ahead"):
        compute_delta_pe_features(
            previous,
            current,
            decision_at="2026-01-31T23:59:59Z",
            config=config,
        )
    ordered = compute_delta_pe_features(
        previous,
        current,
        decision_at="2026-02-02T00:00:00Z",
        config=config,
    ).set_index("symbol")
    shuffled = compute_delta_pe_features(
        previous.sample(frac=1.0, random_state=1),
        current.sample(frac=1.0, random_state=2),
        decision_at="2026-02-02T00:00:00Z",
        config=config,
    ).set_index("symbol")
    pd.testing.assert_series_equal(
        ordered["surge_and_score"].sort_index(),
        shuffled["surge_and_score"].sort_index(),
    )
    portfolio = build_delta_pe_portfolio(
        ordered.reset_index(), score_column="surge_and_score", config=config
    )
    assert list(portfolio["symbol"]) == ["A1"]
    assert portfolio["weight"].sum() == pytest.approx(1.0)


def test_equal_sector_gross_changes_weights_not_literal_selection():
    previous, current = _snapshots()
    current.loc[current["sector"] == "A", "pe_ttm"] = [40.0, 20.0, 20.0]
    current.loc[current["sector"] == "B", "pe_ttm"] = [18.0, 12.0, 12.0]
    config = DeltaPEConfig(top_n=4, min_sector_names=2, eps_log_floor=-1.0)
    features = compute_delta_pe_features(
        previous,
        current,
        decision_at="2026-02-02T00:00:00Z",
        config=config,
    )
    equal_name = build_delta_pe_portfolio(
        features, score_column="literal_surge_and_score", config=config
    )
    equal_sector = build_delta_pe_portfolio(
        features,
        score_column="literal_surge_and_score",
        config=config,
        equal_sector_gross=True,
    )
    assert set(equal_name["symbol"]) == set(equal_sector["symbol"])
    assert equal_sector.groupby("sector")["weight"].sum().to_dict() == pytest.approx(
        {"A": 0.5, "B": 0.5}
    )
