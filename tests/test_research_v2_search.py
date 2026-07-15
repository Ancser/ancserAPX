from __future__ import annotations

import json

import pandas as pd
import pytest

from research_v2.backtest import MarketBar, RiskConfig, RiskObservation
from research_v2.costs import CostConfig
from research_v2.experiment import MarketContext
from research_v2.portfolio import PortfolioConfig
from research_v2.search import (
    Cadence,
    SearchPolicy,
    run_staged_search,
    write_search_artifacts,
)


def _synthetic_inputs():
    sessions = tuple(pd.bdate_range("2024-01-02", periods=36))
    symbols = ("A", "B", "C", "D")
    daily_returns = {
        "A": (0.010, 0.003),
        "B": (0.003, -0.001),
        "C": (-0.002, 0.001),
        "D": (-0.008, -0.002),
    }
    prices = {symbol: 100.0 for symbol in symbols}
    market = {}
    observations = {}
    for index, session in enumerate(sessions):
        bars = {}
        for symbol_number, symbol in enumerate(symbols):
            open_price = prices[symbol]
            close_price = open_price * (1.0 + daily_returns[symbol][index % 2])
            prices[symbol] = close_price
            bars[symbol] = MarketBar(
                open=open_price,
                close=close_price,
                adv_dollars=1_000_000_000.0,
                daily_volatility=0.015 + 0.002 * symbol_number,
                spread_proxy_bps=0.0,
                beta=1.0,
            )
        market[session] = bars
        observations[session] = RiskObservation(betas={symbol: 1.0 for symbol in symbols})

    context = MarketContext(
        market=market,
        full_risk_observations=observations,
        beta_only_observations=observations,
        sectors={symbol: symbol for symbol in symbols},
        sessions=sessions,
        symbols=symbols,
        metadata={"fixture": True},
    )

    rows = []
    good = {"A": 4.0, "B": 3.0, "C": 2.0, "D": 1.0}
    bad = {"A": 1.0, "B": 2.0, "C": 3.0, "D": 4.0}
    for index, session in enumerate(sessions):
        fold = "F1" if index < 12 else "F2" if index < 24 else "LOCKBOX"
        for symbol in symbols:
            rows.append({
                "timestamp": session,
                "symbol": symbol,
                "fold_id": fold,
                "score_good": good[symbol],
                "score_bad": bad[symbol],
            })
    frame = pd.DataFrame(rows)
    selection = frame.loc[frame["fold_id"] != "LOCKBOX"].copy()
    lockbox = frame.loc[frame["fold_id"] == "LOCKBOX"].copy()
    return context, selection, lockbox


def _run(*, progress=None):
    context, selection, lockbox = _synthetic_inputs()
    return run_staged_search(
        context,
        selection,
        lockbox,
        score_columns=("score_good", "score_bad"),
        base_portfolio=PortfolioConfig(
            top_n=1,
            weighting="equal",
            gross_target=1.0,
            single_name_cap=2.0,
            sector_cap=2.0,
            rank_buffer=0,
            no_trade_band=0.0,
            staggered_tranches=1,
            max_adv_participation=None,
        ),
        base_cost=CostConfig(
            commission_bps=0.0,
            spread_multiplier=0.0,
            impact_coefficient=0.0,
            max_adv_participation=None,
        ),
        base_risk=RiskConfig(),
        top_n_grid=(1,),
        cadence_grid=(
            Cadence("every_two_days", 2, 1),
            Cadence("daily_5_tranches", 1, 5),
        ),
        weighting_grid=("equal", "inverse_vol"),
        risk_variants={
            "none": RiskConfig(),
            "inactive_drawdown": RiskConfig(drawdown_steps=((0.50, 0.50),)),
        },
        leverage_grid=(0.75, 1.0),
        cost_sensitivity_bps=(0.0, 5.0, 10.0, 20.0),
        selection_cost_bps=10.0,
        base_rebalance_days=2,
        policy=SearchPolicy(max_drawdown_limit=-0.50),
        initial_capital=100_000.0,
        progress=progress,
    )


def test_stages_cost_curve_neighborhood_and_lockbox_boundary():
    events = []
    result = _run(progress=events.append)

    assert len(result.stage_a) == 2
    stage_a_best = max(result.stage_a, key=lambda item: item.objective_value)
    assert stage_a_best.candidate.score_column == "score_good"
    assert result.champion.candidate.score_column == "score_good"

    assert any(
        item.candidate.cadence.rebalance_days == 1
        and item.candidate.cadence.staggered_tranches == 5
        for item in result.stage_b
    )
    assert {item.candidate.portfolio_config.weighting for item in result.stage_b} == {
        "equal", "inverse_vol"
    }
    assert {item.candidate.risk_variant for item in result.stage_b} == {
        "none", "inactive_drawdown"
    }

    for leverage in (0.75, 1.0):
        curve = {
            item.candidate.extra_cost_bps
            for item in result.stage_c
            if item.candidate.portfolio_config.gross_target == leverage
        }
        assert curve == {0.0, 5.0, 10.0, 20.0}
    assert result.champion.candidate.extra_cost_bps == 10.0
    assert result.champion.cost_breakdown["spread"] == 0.0
    assert result.champion.cost_breakdown["market_impact"] == 0.0
    assert result.champion.cost_breakdown["commission"] > 0.0

    assert result.neighborhood
    assert any(item.label == "score_column=score_bad" for item in result.neighborhood)
    assert len(result.offset_sensitivity) == result.champion.candidate.cadence.rebalance_days
    assert "objective_fold_metrics" in result.champion.worst_metrics

    names = [event["event"] for event in events]
    assert names.index("champion_locked") < names.index("lockbox_completed")
    assert result.lockbox is not None
    assert result.lockbox.candidate.candidate_id == result.champion.candidate.candidate_id
    assert result.audit["selection_only_champion"] is True
    assert result.audit["fold_state_reset"] is False
    assert result.audit["daily_five_tranche_tested"] is True


def test_fold_metrics_are_slices_of_one_continuous_ledger():
    result = _run()
    good = next(item for item in result.stage_a if item.candidate.score_column == "score_good")
    first = good.fold_metrics["F1"]
    second = good.fold_metrics["F2"]

    # With a reset, F2 would restart from the configured 100,000.  Instead its
    # opening equity is exactly the previous fold's closing equity.
    assert second["initial_equity"] == pytest.approx(first["final_equity"])
    assert second["initial_equity"] != pytest.approx(100_000.0)
    assert good.continuous_oos_state is True


def test_artifact_writer_emits_json_curves_folds_and_sensitivities(tmp_path):
    result = _run()
    root = tmp_path / "research_v2"
    output = root / "runs" / "synthetic"
    paths = write_search_artifacts(result, output, research_root=root)

    assert set(paths) == {
        "summary", "champion", "stage_a", "stage_b", "stage_c",
        "neighborhood", "offset_sensitivity", "fold_metrics",
    }
    assert all(path.exists() for path in paths.values())
    payload = json.loads(paths["summary"].read_text(encoding="utf-8"))
    assert payload["champion"]["candidate"]["candidate_id"] == result.champion.candidate.candidate_id
    assert payload["lockbox"]["stage"] == "LOCKBOX"
    assert "F1" in payload["champion"]["fold_metrics"]
    assert "F2" in payload["champion"]["fold_metrics"]


def test_interleaved_fold_dates_are_rejected_before_search():
    context, selection, lockbox = _synthetic_inputs()
    dates = sorted(selection["timestamp"].unique())
    selection["fold_id"] = selection["timestamp"].map(
        {date: ("F1" if index % 2 == 0 else "F2") for index, date in enumerate(dates)}
    )
    with pytest.raises(ValueError, match="overlap or interleave"):
        run_staged_search(
            context,
            selection,
            lockbox,
            score_columns=("score_good",),
            base_portfolio=PortfolioConfig(
                top_n=1, single_name_cap=2.0, sector_cap=2.0,
                max_adv_participation=None,
            ),
            base_cost=CostConfig(max_adv_participation=None),
            base_risk=RiskConfig(),
            top_n_grid=(1,),
            cadence_grid=(1,),
            weighting_grid=("equal",),
            leverage_grid=(1.0,),
        )
