"""Corrective, immutable sector-neutralization study (v3).

This runner supersedes the first exploratory output after an independent audit
found two comparison-label problems: every-five-prediction-session schedules
are not the Main account's Friday schedule, and the historical search artifact
labelled ``risk_variant=none`` actually received an implicit slow-trend overlay.

V3 keeps both causal views and corrects one further clock-label issue found
during the v2 post-publication audit: Main trades around 09:35 ET on the last
NYSE session on/before Friday, so it can normally use only the prior session's
completed daily bar.  The research proxy is therefore prior close -> scheduled
rebalance-session open, not Friday close -> Monday open.

V3 keeps both causal views:

* current static sector map + explicit true-no-risk signal isolation; and
* exact legacy champion behaviour, including its frozen old sector map and
  explicit trend overlay, guarded by a hard parity assertion.

The completed directory is immutable.  There is no refresh or overwrite mode.
"""

from __future__ import annotations

import argparse
import ast
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
from importlib import metadata as importlib_metadata
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import uuid
from typing import Dict, Mapping, Sequence

import numpy as np
import pandas as pd

from research_v2.run_neutralization_study import (
    _build_full_universe_factor_variants,
    _build_signal_variants,
    _canonical_sha256,
    _context_slice,
    _file_sha256,
    _json_safe,
    _load_predictions,
    _merge_factors,
    _moving_block_bootstrap,
    _offset_summary,
    _performance_rows,
    _portfolio_allocation_diagnostics,
    _signal_diagnostics,
    _signal_specs,
    _snapshot_sectors,
    _write_json,
)


ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent
DEFAULT_RUN = ROOT / "runs" / "20260710_full_v1"
DEFAULT_CACHE = ROOT / "cache" / "canonical_features_h5.parquet"
OUTPUT_NAME = "neutralization_study_v3"

GATE_THRESHOLDS = {
    "minimum_sector_r2_reduction": 0.80,
    "maximum_rank_ic_degradation": 0.003,
    "maximum_sharpe_degradation": 0.05,
    "maximum_drawdown_degradation_percentage_points": 0.02,
    "maximum_turnover_increase": 0.15,
}

LEGACY_PARITY_FIELDS = (
    "final_equity",
    "cagr",
    "sharpe",
    "max_drawdown",
    "total_gross_turnover",
    "commission",
    "spread",
    "impact",
    "total_cost",
    "periods",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--bootstrap-repetitions", type=int, default=5_000)
    return parser.parse_args(argv)


def _write_json_fsync(path: Path, payload: object) -> None:
    encoded = json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n"
    with path.open("w", encoding="utf-8") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())


def _legacy_sector_snapshot(symbols: Sequence[str]) -> tuple[dict[str, str], dict[str, object]]:
    """Reconstruct the pre-supplement map used by the frozen 20260710 search."""

    source_path = WORKSPACE / "backend" / "alpha" / "neutralization.py"
    source = source_path.read_text(encoding="utf-8")
    marker = "# Current SPY + QQQ constituent coverage supplement"
    marker_lines = [
        index
        for index, line in enumerate(source.splitlines(), start=1)
        if marker in line
    ]
    if len(marker_lines) != 1:
        raise ValueError("cannot uniquely locate the current-map supplement boundary")
    marker_line = marker_lines[0]
    tree = ast.parse(source)
    dictionary: ast.Dict | None = None
    for node in tree.body:
        if (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "SECTOR_MAP"
            and isinstance(node.value, ast.Dict)
        ):
            dictionary = node.value
            break
    if dictionary is None:
        raise ValueError("SECTOR_MAP dictionary literal was not found")
    legacy_runtime: dict[str, str] = {}
    for key_node, value_node in zip(dictionary.keys, dictionary.values):
        if key_node is None or getattr(key_node, "lineno", marker_line) >= marker_line:
            continue
        key = str(ast.literal_eval(key_node))
        value = str(ast.literal_eval(value_node))
        legacy_runtime[key] = value
    snapshot = {
        str(symbol): str(legacy_runtime.get(str(symbol), "Unknown"))
        for symbol in sorted(symbols)
    }
    known = sum(value != "Unknown" for value in snapshot.values())
    # These counts are part of legacy parity.  A source edit must create a new
    # study instead of silently changing the frozen comparator.
    if len(legacy_runtime) != 382 or (len(snapshot) == 480 and known != 356):
        raise AssertionError(
            f"legacy sector reconstruction drifted: entries={len(legacy_runtime)}, known={known}"
        )
    return snapshot, {
        "source": str(source_path.resolve()),
        "source_sha256": _file_sha256(source_path),
        "supplement_boundary_line": marker_line,
        "runtime_entries": len(legacy_runtime),
        "oos_known": known,
        "oos_unknown": len(snapshot) - known,
        "canonical_sha256": _canonical_sha256(snapshot),
        "unknown_policy_for_portfolio_caps": "unique Unknown:<symbol> pseudo-sector, matching frozen search",
    }


def _legacy_context(current, legacy_snapshot: Mapping[str, str]):
    from research_v2.experiment import MarketContext

    sectors = {
        symbol: (
            str(legacy_snapshot[symbol])
            if legacy_snapshot[symbol] != "Unknown"
            else f"Unknown:{symbol}"
        )
        for symbol in current.symbols
    }
    return MarketContext(
        market=current.market,
        full_risk_observations=current.full_risk_observations,
        beta_only_observations=current.beta_only_observations,
        sectors=sectors,
        sessions=current.sessions,
        symbols=current.symbols,
        metadata={
            **current.metadata,
            "sector_map": "legacy pre-2026-07-supplement reconstruction",
            "sector_known": sum(not value.startswith("Unknown:") for value in sectors.values()),
            "sector_unknown": sum(value.startswith("Unknown:") for value in sectors.values()),
        },
    )


def _weekly_last_session_signal_map(
    predictions: pd.DataFrame,
    *,
    score_column: str,
    eligible_symbols: Sequence[str],
) -> dict[pd.Timestamp, dict[str, float]]:
    """Signal on the last available market session of each ISO week."""

    required = {"timestamp", "symbol", score_column}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"weekly signal frame missing: {sorted(missing)}")
    frame = predictions.loc[
        predictions["symbol"].astype(str).isin(set(map(str, eligible_symbols))),
        ["timestamp", "symbol", score_column],
    ].copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    dates = pd.DatetimeIndex(frame["timestamp"].drop_duplicates().sort_values())
    iso = dates.isocalendar()
    schedule = pd.DataFrame(
        {"timestamp": dates, "iso_year": iso.year.to_numpy(), "iso_week": iso.week.to_numpy()}
    )
    chosen = set(
        pd.to_datetime(
            schedule.groupby(["iso_year", "iso_week"], sort=True)["timestamp"].max()
        )
    )
    signals: dict[pd.Timestamp, dict[str, float]] = {}
    for date in sorted(chosen):
        day = frame.loc[frame["timestamp"] == date, ["symbol", score_column]].dropna()
        scores = {
            str(row[0]): float(row[1])
            for row in day.itertuples(index=False, name=None)
        }
        if scores:
            signals[pd.Timestamp(date)] = scores
    if not signals:
        raise ValueError("weekly-last-session schedule generated no signals")
    return signals


def _weekly_pre_rebalance_signal_map(
    predictions: pd.DataFrame,
    *,
    score_column: str,
    eligible_symbols: Sequence[str],
    market_sessions: Sequence[pd.Timestamp],
) -> dict[pd.Timestamp, dict[str, float]]:
    """Proxy Main's Friday 09:35 trade with the prior completed daily bar.

    The backtest engine executes a decision made at close(t) at open(t+1).
    Main's weekly scheduler instead triggers on the final NYSE session on or
    before Friday.  Selecting the market session immediately before that
    execution session therefore represents Thursday close -> Friday open in a
    normal week, or Wednesday close -> Thursday open when Friday is a holiday.
    The official open remains an approximation to the production 09:35 fill.
    """

    required = {"timestamp", "symbol", score_column}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"weekly signal frame missing: {sorted(missing)}")
    frame = predictions.loc[
        predictions["symbol"].astype(str).isin(set(map(str, eligible_symbols))),
        ["timestamp", "symbol", score_column],
    ].copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    prediction_dates = pd.DatetimeIndex(frame["timestamp"].drop_duplicates().sort_values())
    if len(prediction_dates) < 2:
        raise ValueError("weekly pre-rebalance schedule needs at least two sessions")
    sessions = pd.DatetimeIndex(pd.to_datetime(tuple(market_sessions))).drop_duplicates().sort_values()
    sessions = sessions[(sessions >= prediction_dates.min()) & (sessions <= prediction_dates.max())]
    if len(sessions) < 2:
        raise ValueError("weekly pre-rebalance schedule has insufficient market sessions")
    iso = sessions.isocalendar()
    schedule = pd.DataFrame(
        {"timestamp": sessions, "iso_year": iso.year.to_numpy(), "iso_week": iso.week.to_numpy()}
    )
    execution_dates = pd.DatetimeIndex(
        pd.to_datetime(
            schedule.groupby(["iso_year", "iso_week"], sort=True)["timestamp"].max()
        )
    )
    # Every non-terminal ISO week is known complete because a later week's
    # market session exists.  The terminal week is unknowable without looking
    # beyond the frozen sample: accept it only when it reaches Friday, otherwise
    # conservatively omit it (even if a Thursday might truly be a holiday week).
    if len(execution_dates):
        completed = list(execution_dates[:-1])
        terminal = pd.Timestamp(execution_dates[-1])
        if terminal.dayofweek == 4:
            completed.append(terminal)
        execution_dates = pd.DatetimeIndex(completed)
    previous = {
        pd.Timestamp(sessions[index]): pd.Timestamp(sessions[index - 1])
        for index in range(1, len(sessions))
    }
    available_predictions = set(map(pd.Timestamp, prediction_dates))
    chosen = sorted(
        {
            previous[pd.Timestamp(date)]
            for date in execution_dates
            if pd.Timestamp(date) in previous
            and previous[pd.Timestamp(date)] in available_predictions
        }
    )
    signals: dict[pd.Timestamp, dict[str, float]] = {}
    for date in chosen:
        day = frame.loc[frame["timestamp"] == date, ["symbol", score_column]].dropna()
        scores = {
            str(row[0]): float(row[1])
            for row in day.itertuples(index=False, name=None)
        }
        if scores:
            signals[pd.Timestamp(date)] = scores
    if not signals:
        raise ValueError("weekly pre-rebalance schedule generated no signals")
    return signals


def _portfolio_plans():
    from research_v2.backtest import RiskConfig
    from research_v2.portfolio import PortfolioConfig

    claude = PortfolioConfig(
        top_n=20,
        weighting="equal",
        gross_target=1.5,
        single_name_cap=1.5,
        sector_cap=1.5,
        rank_buffer=0,
        no_trade_band=0.0,
        staggered_tranches=1,
        max_adv_participation=0.02,
    )
    hybrid = PortfolioConfig(
        top_n=30,
        weighting="inverse_vol",
        gross_target=0.75,
        single_name_cap=0.10,
        sector_cap=0.30,
        inverse_vol_floor=0.005,
        rank_buffer=5,
        no_trade_band=0.0025,
        staggered_tranches=1,
        max_adv_participation=0.02,
    )
    claude_scores = {
        "raw": "score_production_claude1",
        "factor_sector_residual": "score_claude1_factorwise__sector_residual",
        "factor_sector_zscore": "score_claude1_factorwise__sector_zscore",
        "factor_within_sector_rank": "score_claude1_factorwise__within_sector_rank",
        "score_sector_residual": "score_production_claude1__sector_residual",
    }
    hybrid_scores = {
        "raw": "score_hybrid_tab50_seq50",
        "sector_residual": "score_hybrid_tab50_seq50__sector_residual",
        "sector_zscore": "score_hybrid_tab50_seq50__sector_zscore",
        "within_sector_rank": "score_hybrid_tab50_seq50__within_sector_rank",
    }
    return (
        {
            "scenario": "claude1_weekly_prior_close_proxy_480",
            "family": "claude1",
            "context": "current",
            "config": claude,
            "scores": claude_scores,
            "schedule": "prior_session_close_to_last_trading_day_open",
            "rebalance_days": 5,
            "offsets": (0,),
            "costs": (0.0, 5.0, 10.0, 20.0),
            "risk": RiskConfig(),
            "use_full_risk": False,
        },
        {
            "scenario": "claude1_every_5_prediction_sessions_diagnostic",
            "family": "claude1",
            "context": "current",
            "config": claude,
            "scores": claude_scores,
            "schedule": "periodic_prediction_sessions",
            "rebalance_days": 5,
            "offsets": tuple(range(5)),
            "costs": (0.0, 5.0, 10.0, 20.0),
            "risk": RiskConfig(),
            "use_full_risk": False,
        },
        {
            "scenario": "claude1_sector_cap_30pct_prior_close_weekly",
            "family": "claude1",
            "context": "current",
            "config": replace(claude, sector_cap=0.45),
            "scores": {
                key: claude_scores[key]
                for key in ("raw", "factor_sector_residual")
            },
            "schedule": "prior_session_close_to_last_trading_day_open",
            "rebalance_days": 5,
            "offsets": (0,),
            "costs": (0.0, 5.0, 10.0, 20.0),
            "risk": RiskConfig(),
            "use_full_risk": False,
        },
        {
            "scenario": "hybrid50_signal_only_current_map",
            "family": "hybrid50",
            "context": "current",
            "config": hybrid,
            "scores": hybrid_scores,
            "schedule": "periodic_prediction_sessions",
            "rebalance_days": 21,
            "offsets": tuple(range(21)),
            "costs": (0.0, 5.0, 10.0, 20.0),
            "risk": RiskConfig(target_change_buffer=0.0025),
            "use_full_risk": False,
        },
        {
            "scenario": "hybrid50_legacy_champion_parity",
            "family": "hybrid50",
            "context": "legacy",
            "config": hybrid,
            "scores": hybrid_scores,
            "schedule": "periodic_prediction_sessions",
            "rebalance_days": 21,
            "offsets": tuple(range(21)),
            "costs": (0.0, 5.0, 10.0, 20.0),
            "risk": RiskConfig(trend_filter=True, target_change_buffer=0.0025),
            "use_full_risk": True,
        },
        {
            "scenario": "hybrid50_uncapped_signal_only_diagnostic",
            "family": "hybrid50",
            "context": "current",
            "config": replace(hybrid, single_name_cap=0.75, sector_cap=0.75),
            "scores": hybrid_scores,
            "schedule": "periodic_prediction_sessions",
            "rebalance_days": 21,
            "offsets": (0,),
            "costs": (10.0,),
            "risk": RiskConfig(target_change_buffer=0.0025),
            "use_full_risk": False,
        },
    )


def _plan_payload(
    code_sha256: Mapping[str, str],
    input_sha256: Mapping[str, str],
    live_proxy_audit: Mapping[str, object],
    bootstrap_repetitions: int,
) -> dict[str, object]:
    plans = []
    for plan in _portfolio_plans():
        plans.append(
            {
                "scenario": plan["scenario"],
                "family": plan["family"],
                "context": plan["context"],
                "schedule": plan["schedule"],
                "rebalance_days": plan["rebalance_days"],
                "offsets": list(plan["offsets"]),
                "all_offsets_friction_bps": 10.0,
                "reference_offset_cost_sensitivity_bps": list(plan["costs"]),
                "scores": dict(plan["scores"]),
                "portfolio": asdict(plan["config"]),
                "risk": asdict(plan["risk"]),
                "use_full_risk_observations": plan["use_full_risk"],
            }
        )
    return {
        "study": "sector_neutralization_oos_corrective_v3",
        "created_at_utc_before_result_computation": _utc_now(),
        "registration_status": (
            "corrective specification frozen after v1/v2 audits; gates inherited unchanged, "
            "not a pristine pre-registration"
        ),
        "primary_methods": ["raw", "sector_residual", "sector_zscore", "within_sector_rank"],
        "primary_gate_tests": [
            "claude1_factor_sector_residual_weekly_prior_close_proxy",
            "hybrid50_sector_residual_true_no_risk",
            "hybrid50_sector_residual_legacy_champion_behavior",
        ],
        "gate_thresholds": dict(GATE_THRESHOLDS),
        "portfolio_plans": plans,
        "cost_scope": "10 bps for every offset; 0/5/20 bps sensitivity on reference offset/schedule only",
        "bootstrap": {
            "method": "circular moving-block paired daily net-return bootstrap",
            "repetitions": bootstrap_repetitions,
            "scope": "reference offset/schedule at 10 bps only",
            "block_length": "5 Claude / 21 Hybrid",
        },
        "confirmation_policy": "previously opened 2026 period is descriptive only",
        "market_cap_policy": "not tested without historical PIT market cap/shares; no proxy substitution",
        "weekly_live_proxy": dict(live_proxy_audit),
        "input_sha256_before_result_computation": dict(input_sha256),
        "code_sha256": dict(code_sha256),
    }


def _make_signal_map(plan, combined: pd.DataFrame, score_col: str, context):
    from research_v2.experiment import make_signal_map

    if plan["schedule"] == "prior_session_close_to_last_trading_day_open":
        return _weekly_pre_rebalance_signal_map(
            combined,
            score_column=score_col,
            eligible_symbols=context.symbols,
            market_sessions=context.sessions,
        )
    return make_signal_map(
        combined[["timestamp", "symbol", score_col]],
        score_column=score_col,
        eligible_symbols=context.symbols,
        rebalance_days=int(plan["rebalance_days"]),
        offset=int(plan["active_offset"]),
        liquidate_at_end=False,
    )


def _run_portfolios_v3(
    contexts: Mapping[str, object],
    combined: pd.DataFrame,
    selection: pd.DataFrame,
    confirmation: pd.DataFrame,
):
    from research_v2.costs import CostConfig
    from research_v2.experiment import evaluate_strategy
    from research_v2.metrics import compute_performance_metrics

    boundaries = {
        "selection": (
            pd.Timestamp(selection["timestamp"].min()),
            pd.Timestamp(selection["timestamp"].max()),
        ),
        "opened_2026": (
            pd.Timestamp(confirmation["timestamp"].min()),
            pd.Timestamp(confirmation["timestamp"].max()),
        ),
    }
    sliced_contexts = {
        key: _context_slice(
            value,
            pd.Timestamp(combined["timestamp"].min()),
            pd.Timestamp(combined["timestamp"].max()),
        )
        for key, value in contexts.items()
    }
    metric_rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []
    daily_rows: list[dict[str, object]] = []
    allocation_daily: list[pd.DataFrame] = []
    allocation_symbol: list[pd.DataFrame] = []
    plans = _portfolio_plans()
    total = sum(
        len(plan["scores"])
        * (len(plan["offsets"]) + len(set(plan["costs"]) - {10.0}))
        for plan in plans
    )
    completed = 0

    for frozen_plan in plans:
        context = sliced_contexts[str(frozen_plan["context"])]
        for variant, raw_score_col in dict(frozen_plan["scores"]).items():
            score_col = str(raw_score_col)
            jobs = {(int(offset), 10.0) for offset in frozen_plan["offsets"]}
            jobs.update((0, float(cost)) for cost in frozen_plan["costs"])
            for offset, friction_bps in sorted(jobs):
                plan = dict(frozen_plan)
                plan["active_offset"] = offset
                signals = _make_signal_map(plan, combined, score_col, context)
                cost = CostConfig(
                    commission_bps=friction_bps,
                    spread_multiplier=1.0,
                    min_spread_bps=0.0,
                    max_spread_bps=30.0,
                    impact_coefficient=0.10,
                    max_impact_bps=50.0,
                    max_adv_participation=0.02,
                    annual_funding_rate=0.055,
                    periods_per_year=252,
                )
                evaluation = evaluate_strategy(
                    context,
                    signals,
                    portfolio_config=plan["config"],
                    cost_config=cost,
                    risk_config=plan["risk"],
                    use_full_risk_observations=bool(plan["use_full_risk"]),
                    initial_capital=100_000.0,
                )
                metrics, ledgers = _performance_rows(evaluation.result, boundaries)
                for period, values in metrics.items():
                    metric_rows.append(
                        {
                            "period": period,
                            "scenario": plan["scenario"],
                            "family": plan["family"],
                            "variant": variant,
                            "score_col": score_col,
                            "context": plan["context"],
                            "schedule": plan["schedule"],
                            "rebalance_days": plan["rebalance_days"],
                            "offset": offset,
                            "extra_friction_bps": friction_bps,
                            "trend_filter": bool(plan["risk"].trend_filter),
                            **values,
                        }
                    )

                if offset == 0 and friction_bps == 10.0:
                    for period, ledger in ledgers.items():
                        for row in ledger:
                            daily_rows.append(
                                {
                                    "period": period,
                                    "scenario": plan["scenario"],
                                    "family": plan["family"],
                                    "variant": variant,
                                    "schedule": plan["schedule"],
                                    "timestamp": pd.Timestamp(row.session),
                                    "net_return": float(row.ending_equity / row.starting_equity - 1.0),
                                }
                            )
                        daily, symbols = _portfolio_allocation_diagnostics(
                            ledger,
                            context,
                            period,
                            str(plan["scenario"]),
                            str(variant),
                        )
                        daily.insert(3, "schedule", str(plan["schedule"]))
                        symbols.insert(3, "schedule", str(plan["schedule"]))
                        allocation_daily.append(daily)
                        allocation_symbol.append(symbols)
                    for fold_id, fold in selection.groupby("fold_id", sort=False, observed=True):
                        start = pd.Timestamp(fold["timestamp"].min())
                        end = pd.Timestamp(fold["timestamp"].max())
                        fold_ledger = [
                            row
                            for row in ledgers["selection"]
                            if start <= pd.Timestamp(row.session) <= end
                        ]
                        fold_rows.append(
                            {
                                "scenario": plan["scenario"],
                                "family": plan["family"],
                                "variant": variant,
                                "schedule": plan["schedule"],
                                "fold_id": str(fold_id),
                                **compute_performance_metrics(fold_ledger).to_dict(),
                            }
                        )
                completed += 1
                if completed % 10 == 0 or completed == total:
                    print(
                        json.dumps(
                            {
                                "event": "neutralization_v3_portfolio_progress",
                                "completed": completed,
                                "total": total,
                                "scenario": plan["scenario"],
                                "variant": variant,
                                "offset": offset,
                                "bps": friction_bps,
                            }
                        ),
                        flush=True,
                    )
    return (
        pd.DataFrame(metric_rows),
        pd.DataFrame(fold_rows),
        pd.DataFrame(daily_rows),
        pd.concat(allocation_daily, ignore_index=True),
        pd.concat(allocation_symbol, ignore_index=True),
    )


def _portfolio_comparisons_v2(daily: pd.DataFrame, repetitions: int) -> pd.DataFrame:
    rows = []
    for (scenario, period), group in daily.groupby(["scenario", "period"], sort=True):
        raw = group.loc[group["variant"] == "raw", ["timestamp", "net_return"]].rename(
            columns={"net_return": "raw"}
        )
        if raw.empty:
            continue
        block = 5 if str(scenario).startswith("claude1") else 21
        for variant, part in group.groupby("variant", sort=True):
            if variant == "raw":
                continue
            aligned = raw.merge(
                part[["timestamp", "net_return"]].rename(columns={"net_return": "variant"}),
                on="timestamp",
                how="inner",
                validate="one_to_one",
            )
            differences = aligned["variant"].to_numpy(float) - aligned["raw"].to_numpy(float)
            rows.append(
                {
                    "scenario": scenario,
                    "period": period,
                    "variant": variant,
                    **_moving_block_bootstrap(
                        differences,
                        block_length=block,
                        repetitions=repetitions,
                        seed=20260715 + sum(map(ord, f"{scenario}:{period}:{variant}")),
                    ),
                }
            )
    return pd.DataFrame(rows)


def _assert_legacy_parity(
    portfolio: pd.DataFrame,
    champion_payload: Mapping[str, object],
) -> dict[str, object]:
    rows = portfolio.loc[
        (portfolio["period"] == "selection")
        & (portfolio["scenario"] == "hybrid50_legacy_champion_parity")
        & (portfolio["variant"] == "raw")
        & (portfolio["offset"] == 0)
        & (portfolio["extra_friction_bps"] == 10.0)
    ]
    if len(rows) != 1:
        raise AssertionError(f"legacy parity row count is {len(rows)}, expected 1")
    actual = rows.iloc[0]
    expected = dict(champion_payload["metrics"])
    differences = {}
    for field in LEGACY_PARITY_FIELDS:
        a = float(actual[field])
        e = float(expected[field])
        tolerance = max(1e-10, abs(e) * 1e-10)
        differences[field] = {"actual": a, "expected": e, "absolute_error": abs(a - e)}
        if abs(a - e) > tolerance:
            raise AssertionError(
                f"legacy champion parity failed for {field}: actual={a}, expected={e}"
            )
    return {
        "passed": True,
        "fields": differences,
        "champion_candidate_id": champion_payload["candidate"]["candidate_id"],
        "semantic_note": (
            "the frozen artifact called this risk_variant=none, but exact parity requires "
            "trend_filter=True because the old engine enabled trend from rich observations"
        ),
    }


def _acceptance_gates_v3(signal: pd.DataFrame, offsets: pd.DataFrame) -> list[dict[str, object]]:
    definitions = (
        (
            "claude1_factor_sector_residual_weekly_prior_close_proxy",
            "claude1",
            "factor_sector_residual",
            "claude1_weekly_prior_close_proxy_480",
        ),
        (
            "hybrid50_sector_residual_true_no_risk",
            "hybrid50",
            "sector_residual",
            "hybrid50_signal_only_current_map",
        ),
        (
            "hybrid50_sector_residual_legacy_champion_behavior",
            "hybrid50",
            "sector_residual",
            "hybrid50_legacy_champion_parity",
        ),
    )
    rows = []
    for name, family, variant, scenario in definitions:
        signal_rows = signal.loc[
            (signal["period"] == "selection") & (signal["family"] == family)
        ].set_index("variant")
        portfolio_rows = offsets.loc[
            (offsets["period"] == "selection") & (offsets["scenario"] == scenario)
        ].set_index("variant")
        raw_signal = signal_rows.loc["raw"]
        new_signal = signal_rows.loc[variant]
        raw_portfolio = portfolio_rows.loc["raw"]
        new_portfolio = portfolio_rows.loc[variant]
        r2_reduction = 1.0 - float(new_signal["mean_sector_r2"]) / max(
            float(raw_signal["mean_sector_r2"]), 1e-15
        )
        ic_delta = float(new_signal["mean_rank_ic"] - raw_signal["mean_rank_ic"])
        sharpe_delta = float(new_portfolio["median_sharpe"] - raw_portfolio["median_sharpe"])
        drawdown_delta = float(
            new_portfolio["median_max_drawdown"] - raw_portfolio["median_max_drawdown"]
        )
        turnover_ratio = float(
            new_portfolio["median_gross_turnover"] / raw_portfolio["median_gross_turnover"]
        )
        checks = {
            "sector_r2_reduction_pass": r2_reduction >= GATE_THRESHOLDS["minimum_sector_r2_reduction"],
            "rank_ic_pass": ic_delta >= -GATE_THRESHOLDS["maximum_rank_ic_degradation"],
            "sharpe_pass": sharpe_delta >= -GATE_THRESHOLDS["maximum_sharpe_degradation"],
            "drawdown_pass": drawdown_delta
            >= -GATE_THRESHOLDS["maximum_drawdown_degradation_percentage_points"],
            "turnover_pass": turnover_ratio <= 1.0 + GATE_THRESHOLDS["maximum_turnover_increase"],
        }
        rows.append(
            {
                "test": name,
                "family": family,
                "variant": variant,
                "scenario": scenario,
                "r2_reduction": r2_reduction,
                "rank_ic_delta": ic_delta,
                "sharpe_delta": sharpe_delta,
                "max_drawdown_delta": drawdown_delta,
                "turnover_ratio": turnover_ratio,
                **checks,
                "all_gates_pass": all(checks.values()),
            }
        )
    return rows


def _universe_sensitivity(
    selection: pd.DataFrame,
    confirmation: pd.DataFrame,
    sector_snapshot: Mapping[str, str],
) -> dict[str, object]:
    from research_v2.neutralization import build_claude1_factor_variants

    result = {}
    for name, frame in (("selection", selection), ("opened_2026", confirmation)):
        rebuilt, _ = build_claude1_factor_variants(
            frame,
            sector_snapshot,
            methods=("none",),
            min_sector_names=10,
        )
        candidate = rebuilt["score_claude1_factorwise__none"]
        overlaps = []
        max_error = float(np.max(np.abs(candidate - frame["score_production_claude1"])))
        local = frame[["timestamp", "symbol", "score_production_claude1"]].copy()
        local["complete_case_recomputed"] = candidate.to_numpy(float)
        for _date, day in local.groupby("timestamp", sort=True):
            raw = set(day.nlargest(20, "score_production_claude1")["symbol"])
            new = set(day.nlargest(20, "complete_case_recomputed")["symbol"])
            overlaps.append(len(raw & new) / 20.0)
        result[name] = {
            "max_abs_score_error": max_error,
            "mean_top20_overlap": float(np.mean(overlaps)),
            "minimum_top20_overlap": float(np.min(overlaps)),
            "interpretation": (
                "480-complete-case ranking versus frozen artifact ranking on the larger daily cross-section; "
                "this is not exact current-live-universe parity"
            ),
        }
    return result


def _code_state() -> tuple[dict[str, str], dict[str, object]]:
    paths = tuple(sorted(ROOT.glob("*.py"))) + (
        WORKSPACE / "backend" / "alpha" / "neutralization.py",
        WORKSPACE / "backend" / "execution" / "scheduler.py",
        WORKSPACE / "backend" / "execution" / "strategy.py",
        WORKSPACE / "backend" / "data" / "fetcher.py",
        WORKSPACE / "backend" / "data" / "store.py",
        WORKSPACE / "config" / "live_strategy.json",
    )
    hashes = {str(path.relative_to(WORKSPACE)): _file_sha256(path) for path in paths}
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=WORKSPACE, text=True, encoding="utf-8"
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--short"], cwd=WORKSPACE, text=True, encoding="utf-8"
        ).splitlines()
    except (OSError, subprocess.CalledProcessError):
        commit = None
        status = []
    packages = {}
    for package in ("numpy", "pandas", "polars", "pyarrow", "scikit-learn", "torch"):
        try:
            packages[package] = importlib_metadata.version(package)
        except importlib_metadata.PackageNotFoundError:
            packages[package] = None
    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
        "git_commit": commit,
        "git_status_lines": status,
        "git_status_sha256": _canonical_sha256(status),
    }
    return hashes, environment


def _fmt(value: object, digits: int = 3) -> str:
    return "—" if pd.isna(value) else f"{float(value):.{digits}f}"


def _pct(value: object) -> str:
    return "—" if pd.isna(value) else f"{float(value):.2%}"


def _report_v3(
    manifest: Mapping[str, object],
    signal: pd.DataFrame,
    offsets: pd.DataFrame,
    gates: Sequence[Mapping[str, object]],
) -> str:
    lines = [
        "# 行業中性化可驗證研究報告（Corrective v3）",
        "",
        f"完成時間：{manifest['completed_at_utc']}",
        "",
        "## 結論",
        "",
    ]
    for gate in gates:
        verdict = "通過" if gate["all_gates_pass"] else "未通過"
        lines.append(
            f"- **{gate['test']}：{verdict}**；Sector R² 降低 {_pct(gate['r2_reduction'])}，"
            f"IC 差 {_fmt(gate['rank_ic_delta'], 4)}，Sharpe 差 {_fmt(gate['sharpe_delta'])}，"
            f"MaxDD 差 {_pct(gate['max_drawdown_delta'])}，換手比 {_fmt(gate['turnover_ratio'])}。"
        )
    lines.extend(
        [
            "",
            "沒有任何 residual 主假設可以因本研究直接進入 Main daily。中性化被證實能降低行業暴露與集中度；是否提高 alpha 則沒有通過非劣 gate。次要版本最多只能進新的 shadow-forward。",
            "",
            "## v3 修正了什麼",
            "",
            "- Claude #1 的 weekly proxy 改用 09:35 ET 調倉前一市場交易日的完成日線訊號；正常週代理為週四收盤→週五開盤，週五休市則為週三收盤→週四開盤。官方開盤價仍只是 09:35 成交的近似。",
            "- Hybrid 同時保留 current-map true-no-risk 因果測試，以及 frozen legacy champion 行為。Legacy raw 由 final equity、Sharpe、MaxDD、換手與所有成本做 hard parity。",
            "- 舊 search 的 `risk_variant=none` 實際含隱式 slow-trend cash overlay；v3 已把 trend filter 改成顯式設定，未來的 `none` 才是真正無 regime overlay。",
            "- 518 檔 factorwise 截面中 9 檔 Unknown 不再把原始 RSI 0–100 混進 residual／z-score 尺度；它們取得自己的 raw global rank，已映射股票另行校準。",
            "- 完成目錄不可 refresh／overwrite；plan 在結果前寫入並 hash，輸入、程式與輸出均有 SHA-256，第三方依賴則記錄精確版本。",
            "",
            "## 資料與限制",
            "",
            "- 評估股票 480 檔、11 個 static broad sectors、100% 映射；分類是 2026-07 靜態快照，不是 PIT GICS。",
            "- Claude factorwise 為重建 frozen artifact，先在 518 檔原排名截面處理，其中 9 檔 Unknown 採 rank-scale passthrough，再切回 480 檔。",
            f"- Main config 目前有 {manifest['weekly_live_proxy']['configured_unique_universe_count']} 檔股票，研究 complete-case universe 為 {manifest['weekly_live_proxy']['research_complete_case_universe_count']} 檔；因此 Claude weekly 場景只是 clock／portfolio-rule proxy，不是 exact production parity。",
            "- Weekly proxy 只模擬 calendar API 成功且已有持倉歷史的正常週期；首次啟動立即建倉、長時間中斷後的 stale catch-up、以及 calendar fallback 路徑不在本次 A/B。",
            "- 沒有歷史 PIT market cap／shares，因此市值中性化未測；沒有用價格、ADV 或成交額冒充市值。",
            f"- Selection 與 2026 之間有 {manifest['chronology']['no_prediction_market_sessions']} 個無 prediction 市場交易日；持倉與 engine state 連續穿越此 gap。Periodic offset 以可用 prediction dates 計數，與 frozen search 一致。",
            "- 2026 已開封且只有 123 日，只作描述，不用來挑方法。",
            "",
            "## Selection 訊號",
            "",
            "| 家族 | 版本 | Rank IC | NW t | Within-sector IC | Sector R² | Top-N 最大行業占比 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    primary = signal.loc[
        (signal["period"] == "selection")
        & signal["family"].isin(["claude1", "tabular_ensemble", "ridge", "gbdt", "sequence_ensemble", "hybrid50"])
    ]
    for row in primary.itertuples(index=False):
        lines.append(
            f"| {row.family} | {row.variant} | {_fmt(row.mean_rank_ic, 4)} | {_fmt(row.rank_ic_nw_t, 2)} | "
            f"{_fmt(row.mean_within_sector_rank_ic, 4)} | {_pct(row.mean_sector_r2)} | "
            f"{_pct(row.mean_max_sector_share)} |"
        )
    lines.extend(
        [
            "",
            "## 扣成本組合（10 bps 額外摩擦）",
            "",
            "0/5/20 bps 只對 reference schedule／offset 0 做敏感度；10 bps 才涵蓋全部 5／21 offsets。成本另含 spread proxy、impact 與 5.5% 槓桿融資。",
            "",
            "| 期間 | 場景 | 版本 | schedules/offsets | 中位 Sharpe | 最差 Sharpe | 中位 CAGR | 中位 MaxDD | 最差 MaxDD |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    show = offsets.loc[
        offsets["scenario"].isin(
            [
                "claude1_weekly_prior_close_proxy_480",
                "hybrid50_signal_only_current_map",
                "hybrid50_legacy_champion_parity",
            ]
        )
    ]
    for row in show.itertuples(index=False):
        lines.append(
            f"| {row.period} | {row.scenario} | {row.variant} | {row.offsets} | "
            f"{_fmt(row.median_sharpe)} | {_fmt(row.worst_sharpe)} | {_pct(row.median_cagr)} | "
            f"{_pct(row.median_max_drawdown)} | {_pct(row.worst_max_drawdown)} |"
        )
    lines.extend(
        [
            "",
            "## 如何移植",
            "",
            "可移植的是 pure score transform、sector exposure／持倉 concentration 診斷、顯式 trend-filter、prior-close Friday schedule builder、legacy parity guard 與 immutable research manifest。不可直接移植的是任何『最佳中性化版本』，因為主 gate 沒有通過。",
            "",
            "若要繼續，應把 Claude score-level residual 定義為全新 shadow challenger，或預先固定 partial shrinkage `score=(1-λ)·raw+λ·residual`；λ 必須在新的 forward 資料前鎖定。取得 PIT market cap 後，才測 joint OLS `score ~ z(log_market_cap) + sector dummies`。",
            "",
            "完整數據見同目錄 CSV／Parquet；`study_plan.json`、`manifest.json` 與 `_SUCCESS.json` 提供完整 hashes。",
            "",
        ]
    )
    return "\n".join(lines)


def _verify_published(output: Path) -> None:
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    success = json.loads((output / "_SUCCESS.json").read_text(encoding="utf-8"))
    if _file_sha256(output / "manifest.json") != success["manifest_sha256"]:
        raise AssertionError("published manifest hash mismatch")
    declared = set(manifest["output_sha256"])
    actual = {
        str(path.relative_to(output)).replace("\\", "/")
        for path in output.rglob("*")
        if path.is_file()
    }
    expected_files = declared | {"manifest.json", "_SUCCESS.json"}
    if actual != expected_files:
        raise AssertionError(
            f"published file set mismatch: missing={sorted(expected_files - actual)}, "
            f"extra={sorted(actual - expected_files)}"
        )
    for relative, expected in manifest["output_sha256"].items():
        path = output / relative
        if not path.is_file() or _file_sha256(path) != expected:
            raise AssertionError(f"published output hash mismatch: {relative}")
    if _canonical_sha256(manifest["output_sha256"]) != success["output_hashes_sha256"]:
        raise AssertionError("published aggregate output hash mismatch")
    plan_hash = _file_sha256(output / "study_plan.json")
    if plan_hash != manifest["study_plan_sha256"] or plan_hash != success["study_plan_sha256"]:
        raise AssertionError("published study plan hash mismatch")
    code_hash = _canonical_sha256(manifest["code_sha256"])
    if code_hash != manifest["code_sha256_aggregate"] or code_hash != success["code_sha256_aggregate"]:
        raise AssertionError("published code aggregate hash mismatch")
    if success.get("legacy_champion_parity_passed") is not True or manifest["legacy_champion_parity"].get("passed") is not True:
        raise AssertionError("published legacy champion parity flag mismatch")
    if success.get("market_cap_neutralization_tested") is not False or manifest["market_cap_neutralization"].get("status") != "not_tested":
        raise AssertionError("published market-cap status mismatch")
    if success.get("immutable") is not True or success.get("completed_at_utc") != manifest.get("completed_at_utc"):
        raise AssertionError("published completion metadata mismatch")


def main(argv: Sequence[str] | None = None) -> int:
    args = _arguments(argv)
    if args.bootstrap_repetitions < 100:
        raise ValueError("bootstrap repetitions must be at least 100")
    from research_v2.safety import ensure_research_output_path, offline_context

    run_dir = ensure_research_output_path(args.run_dir, research_root=ROOT)
    cache = args.cache.expanduser().resolve(strict=True)
    input_paths = [
        cache,
        run_dir / "tabular" / "selection_oos_predictions.parquet",
        run_dir / "tabular" / "lockbox_predictions.parquet",
        run_dir / "sequence_full60_all" / "selection" / "predictions.parquet",
        run_dir / "sequence_full60_all" / "lockbox" / "predictions.parquet",
        run_dir / "search" / "champion.json",
    ]
    for path in input_paths:
        path.resolve(strict=True)
    input_sha256 = {str(path.resolve()): _file_sha256(path) for path in input_paths}
    live_config_path = WORKSPACE / "config" / "live_strategy.json"
    live_config = json.loads(live_config_path.read_text(encoding="utf-8"))
    live_universe = tuple(map(str, live_config.get("universe", ())))
    live_proxy_audit = {
        "configured_universe_count": len(live_universe),
        "configured_unique_universe_count": len(set(live_universe)),
        "configured_universe_sha256": _canonical_sha256(sorted(live_universe)),
        "rebalance_frequency": live_config.get("rebalance_frequency"),
        "rebalance_weekday": live_config.get("rebalance_weekday", 4),
        "scheduler_trigger": "09:35 ET on the last NYSE session on/before Friday",
        "execution_proxy": "prior completed daily close -> official open; not exact 09:35 fill",
        "scope": "normal mature weekly cycle with successful exchange-calendar lookup",
        "excluded_scheduler_paths": [
            "first run with no rebalance history may establish positions immediately",
            "max_stale_days catch-up may trade off-cycle after an extended outage",
            "calendar API failure falls back to the configured weekday",
        ],
    }
    output = ensure_research_output_path(run_dir / OUTPUT_NAME, research_root=ROOT)
    if output.exists():
        raise FileExistsError(f"immutable study already exists: {output}")
    staging = ensure_research_output_path(
        run_dir / f".{OUTPUT_NAME}.partial-{uuid.uuid4().hex}", research_root=ROOT
    )
    staging.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    started_at = _utc_now()
    code_sha256, environment = _code_state()
    plan = _plan_payload(
        code_sha256, input_sha256, live_proxy_audit, args.bootstrap_repetitions
    )
    _write_json_fsync(staging / "study_plan.json", plan)
    plan_sha256 = _file_sha256(staging / "study_plan.json")

    try:
        with offline_context():
            import polars as pl

            from research_v2.experiment import MarketContext, build_market_context, complete_case_symbols

            selection, confirmation = _load_predictions(run_dir)
            market_columns = [
                "timestamp",
                "symbol",
                "open",
                "close",
                "adv20_v2",
                "realized_vol_20d_v2",
                "range_pct_v2",
                "breadth_200d_v2",
            ]
            panel = pl.read_parquet(
                cache,
                columns=market_columns + ["factor_ts_mom", "factor_rsi"],
            )
            symbols = complete_case_symbols(panel.select("timestamp", "symbol"))
            prediction_symbols = tuple(sorted(selection["symbol"].astype(str).unique()))
            if prediction_symbols != tuple(sorted(symbols)):
                raise ValueError("OOS predictions are not the cache complete-case universe")
            current_sector_snapshot = _snapshot_sectors(symbols)
            legacy_sector_snapshot, legacy_sector_audit = _legacy_sector_snapshot(symbols)
            all_factor_symbols = tuple(sorted(panel["symbol"].unique().to_list()))
            factor_sector_snapshot = _snapshot_sectors(
                all_factor_symbols, require_complete=False
            )
            factor_panel = panel.select(
                "timestamp", "symbol", "factor_ts_mom", "factor_rsi"
            ).to_pandas()
            factor_panel["timestamp"] = pd.to_datetime(factor_panel["timestamp"])
            factor_panel["symbol"] = factor_panel["symbol"].astype(str)
            selection = _merge_factors(selection, factor_panel)
            confirmation = _merge_factors(confirmation, factor_panel)
            selection_dates = set(pd.to_datetime(selection["timestamp"].unique()))
            confirmation_dates = set(pd.to_datetime(confirmation["timestamp"].unique()))
            selection_factor, selection_factor_audit = _build_full_universe_factor_variants(
                factor_panel.loc[factor_panel["timestamp"].isin(selection_dates)],
                factor_sector_snapshot,
            )
            confirmation_factor, confirmation_factor_audit = _build_full_universe_factor_variants(
                factor_panel.loc[factor_panel["timestamp"].isin(confirmation_dates)],
                factor_sector_snapshot,
            )
            selection, selection_audit = _build_signal_variants(
                selection, current_sector_snapshot, selection_factor
            )
            confirmation, confirmation_audit = _build_signal_variants(
                confirmation, current_sector_snapshot, confirmation_factor
            )
            selection_audit["factorwise_full_universe"] = selection_factor_audit
            confirmation_audit["factorwise_full_universe"] = confirmation_factor_audit
            signal, signal_folds, exposure_daily, signal_comparisons = _signal_diagnostics(
                selection, confirmation, _signal_specs()
            )
            universe_sensitivity = _universe_sensitivity(
                selection, confirmation, current_sector_snapshot
            )
            combined = pd.concat([selection, confirmation], ignore_index=True).sort_values(
                ["timestamp", "symbol"], kind="mergesort"
            )
            current_context = build_market_context(
                panel.select(market_columns),
                symbols=symbols,
                start=combined["timestamp"].min(),
                end=combined["timestamp"].max(),
                beta_lookback=126,
                spread_range_fraction=0.02,
                min_spread_bps=1.0,
                max_spread_bps=30.0,
            )
            legacy_context = _legacy_context(current_context, legacy_sector_snapshot)
            portfolio, portfolio_folds, daily_returns, allocation_daily, allocation_symbol = _run_portfolios_v3(
                {"current": current_context, "legacy": legacy_context},
                combined,
                selection,
                confirmation,
            )
            offset_summary = _offset_summary(portfolio)
            portfolio_comparisons = _portfolio_comparisons_v2(
                daily_returns, args.bootstrap_repetitions
            )
            champion_path = run_dir / "search" / "champion.json"
            champion = json.loads(champion_path.read_text(encoding="utf-8"))
            legacy_parity = _assert_legacy_parity(portfolio, champion)
            gates = _acceptance_gates_v3(signal, offset_summary)

            selection_start = pd.Timestamp(selection["timestamp"].min())
            selection_end = pd.Timestamp(selection["timestamp"].max())
            confirmation_start = pd.Timestamp(confirmation["timestamp"].min())
            confirmation_end = pd.Timestamp(confirmation["timestamp"].max())
            gap_sessions = [
                session
                for session in current_context.sessions
                if selection_end < pd.Timestamp(session) < confirmation_start
            ]
            final_input_sha256 = {
                str(path.resolve()): _file_sha256(path) for path in input_paths
            }
            if final_input_sha256 != input_sha256:
                raise AssertionError("research input changed during result computation")
            final_code_sha256, _final_environment = _code_state()
            if final_code_sha256 != code_sha256:
                raise AssertionError("research code or live-clock semantic input changed during result computation")
            completed_at = _utc_now()
            manifest: dict[str, object] = {
                "study": "sector_neutralization_oos_corrective_v3",
                "supersedes": (
                    "neutralization_study v1 and corrective v2; v3 additionally fixes the "
                    "Main weekly decision/execution clock label"
                ),
                "started_at_utc": started_at,
                "completed_at_utc": completed_at,
                "elapsed_seconds": time.perf_counter() - started,
                "study_plan_sha256": plan_sha256,
                "selection_only_method_gates": True,
                "confirmation_status": "previously_opened_2026_descriptive_only",
                "decision_clock": "prior completed session close(t) signal -> scheduled weekly session open(t+1)",
                "production_clock_proxy": {
                    "scheduler": "last NYSE session on/before Friday at approximately 09:35 ET",
                    "normal_week": "Thursday close signal -> Friday official-open execution proxy",
                    "friday_holiday_week": "Wednesday close signal -> Thursday official-open execution proxy",
                    "limitation": "daily official open does not reproduce the exact 09:35 fill",
                },
                "weekly_live_proxy": {
                    **live_proxy_audit,
                    "research_complete_case_universe_count": len(symbols),
                    "exact_production_parity": False,
                    "non_parity_reasons": [
                        "research uses the 480-name complete-case OOS universe, not the configured live universe",
                        "daily official open is only a proxy for the 09:35 ET fill",
                    ],
                },
                "chronology": {
                    "selection_start": selection_start,
                    "selection_end": selection_end,
                    "confirmation_start": confirmation_start,
                    "confirmation_end": confirmation_end,
                    "no_prediction_market_sessions": len(gap_sessions),
                    "gap_session_start": gap_sessions[0] if gap_sessions else None,
                    "gap_session_end": gap_sessions[-1] if gap_sessions else None,
                    "state_continuity": "positions/equity/risk state cross the gap",
                    "periodic_cadence": "index over available prediction dates, matching frozen search",
                },
                "universe": {
                    "symbols": len(symbols),
                    "selection_rows": len(selection),
                    "selection_days": int(selection["timestamp"].nunique()),
                    "confirmation_rows": len(confirmation),
                    "confirmation_days": int(confirmation["timestamp"].nunique()),
                },
                "current_sector_snapshot": {
                    "review_date": "2026-07",
                    "point_in_time": False,
                    "coverage": 1.0,
                    "canonical_sha256": _canonical_sha256(current_sector_snapshot),
                    "sectors": int(len(set(current_sector_snapshot.values()))),
                    "GPN_runtime_value": current_sector_snapshot.get("GPN"),
                },
                "legacy_sector_snapshot": legacy_sector_audit,
                "factor_neutralization_universe": {
                    "symbols": len(all_factor_symbols),
                    "unknown_symbols_rank_scale_passthrough": sorted(
                        symbol
                        for symbol, sector in factor_sector_snapshot.items()
                        if sector == "Unknown"
                    ),
                    "canonical_sha256": _canonical_sha256(factor_sector_snapshot),
                    "unknown_semantics": (
                        "unknown names receive raw global centered rank after known-sector calibration; "
                        "raw units are never mixed with residual/z/rank units"
                    ),
                },
                "legacy_champion_parity": legacy_parity,
                "risk_semantics_correction": {
                    "old_issue": (
                        "rich benchmark observations implicitly enabled slow-trend regime even when "
                        "RiskConfig was labelled none"
                    ),
                    "new_semantics": "trend_filter must be explicitly true",
                    "daily_production_impact": "none; change is confined to research_v2",
                },
                "complete_case_universe_sensitivity": universe_sensitivity,
                "market_cap_neutralization": {
                    "status": "not_tested",
                    "reason": "no historical point-in-time market cap or shares outstanding",
                    "prohibited_substitutes": ["price", "dollar volume", "ADV", "current market cap backfill"],
                },
                "gate_thresholds": dict(GATE_THRESHOLDS),
                "cost_scope": {
                    "all_offsets": [10.0],
                    "reference_offset_or_schedule": [0.0, 5.0, 10.0, 20.0],
                    "base_layers": "lagged spread proxy + square-root impact + 5.5% leverage funding + 2% ADV cap",
                },
                "bootstrap": {
                    "method": "circular moving-block paired daily net-return bootstrap",
                    "repetitions": args.bootstrap_repetitions,
                    "scope": "reference schedule/offset0 at 10bps, not the offset median",
                    "block_lengths": {"claude1": 5, "hybrid50": 21},
                },
                "input_sha256": input_sha256,
                "code_sha256": code_sha256,
                "code_sha256_aggregate": _canonical_sha256(code_sha256),
                "environment": environment,
                "audits": {
                    "selection": selection_audit,
                    "confirmation": confirmation_audit,
                    "current_market_context": current_context.metadata,
                    "legacy_market_context": legacy_context.metadata,
                },
                "production_mutations": [],
                "output_sha256": {},
            }

            _write_json(staging / "current_sector_map_snapshot.json", current_sector_snapshot)
            _write_json(staging / "legacy_sector_map_snapshot.json", legacy_sector_snapshot)
            _write_json(staging / "factor_universe_sector_map_snapshot.json", factor_sector_snapshot)
            _write_json(staging / "acceptance_gates.json", gates)
            _write_json(staging / "legacy_champion_parity.json", legacy_parity)
            signal.to_csv(staging / "signal_diagnostics.csv", index=False)
            signal_folds.to_csv(staging / "signal_fold_diagnostics.csv", index=False)
            exposure_daily.to_parquet(staging / "signal_daily_sector_exposure.parquet", index=False)
            signal_comparisons.to_csv(staging / "signal_comparisons_vs_raw.csv", index=False)
            portfolio.to_csv(
                staging / "portfolio_metrics_10bps_all_offsets_plus_reference_cost_sensitivity.csv",
                index=False,
            )
            portfolio_folds.to_csv(staging / "portfolio_fold_metrics_reference_10bps.csv", index=False)
            offset_summary.to_csv(staging / "portfolio_offset_summary_10bps.csv", index=False)
            daily_returns.to_parquet(
                staging / "portfolio_daily_returns_reference_10bps.parquet", index=False
            )
            portfolio_comparisons.to_csv(
                staging / "portfolio_paired_bootstrap_reference_10bps.csv", index=False
            )
            allocation_daily.to_parquet(
                staging / "portfolio_daily_allocation_reference_10bps.parquet", index=False
            )
            allocation_symbol.to_csv(
                staging / "portfolio_average_symbol_allocation_reference_10bps.csv", index=False
            )
            summary = {
                "acceptance_gates": gates,
                "legacy_champion_parity": legacy_parity,
                "selection_signal": signal.loc[signal["period"] == "selection"].to_dict("records"),
                "confirmation_signal": signal.loc[signal["period"] == "opened_2026"].to_dict("records"),
                "offset_summary": offset_summary.to_dict("records"),
            }
            _write_json(staging / "summary.json", summary)
            (staging / "report.md").write_text(
                _report_v3(manifest, signal, offset_summary, gates), encoding="utf-8"
            )

            artifact_hashes = {
                str(path.relative_to(staging)).replace("\\", "/"): _file_sha256(path)
                for path in sorted(staging.rglob("*"))
                if path.is_file() and path.name not in {"manifest.json", "_SUCCESS.json"}
            }
            manifest["output_sha256"] = artifact_hashes
            _write_json_fsync(staging / "manifest.json", manifest)
            success = {
                "completed_at_utc": completed_at,
                "immutable": True,
                "study_plan_sha256": plan_sha256,
                "manifest_sha256": _file_sha256(staging / "manifest.json"),
                "output_hashes_sha256": _canonical_sha256(artifact_hashes),
                "code_sha256_aggregate": manifest["code_sha256_aggregate"],
                "legacy_champion_parity_passed": True,
                "market_cap_neutralization_tested": False,
            }
            _write_json_fsync(staging / "_SUCCESS.json", success)
        os.replace(staging, output)
        _verify_published(output)
        print(
            json.dumps(
                {
                    "event": "neutralization_study_v3_completed",
                    "output": str(output),
                    "elapsed_seconds": time.perf_counter() - started,
                }
            ),
            flush=True,
        )
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise
