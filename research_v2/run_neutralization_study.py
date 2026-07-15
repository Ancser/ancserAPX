"""Run an isolated, reproducible sector-neutralization OOS study.

The study never changes production configuration or daily-run code.  It uses
already-frozen OOS predictions, applies date-local transforms, and compares
signals and portfolios under identical execution assumptions.  The 2026
period is reported only after selection-period rules and gates are fixed.

There is intentionally no market-cap result: this repository has no historical
point-in-time market cap or shares-outstanding data.  Dollar volume is not used
as a dishonest substitute for company size.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
import uuid
from typing import Dict, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DEFAULT_RUN = ROOT / "runs" / "20260710_full_v1"
DEFAULT_CACHE = ROOT / "cache" / "canonical_features_h5.parquet"
OUTPUT_NAME = "neutralization_study"
METHODS = ("sector_residual", "sector_zscore", "within_sector_rank")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--bootstrap-repetitions", type=int, default=5_000)
    return parser.parse_args(argv)


def _file_sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _json_safe(value: object) -> object:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    return value


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _load_predictions(run_dir: Path):
    from research_v2.run_full_search import _add_fixed_hybrids, _merge

    sequence = run_dir / "sequence_full60_all"
    selection = _add_fixed_hybrids(
        _merge(
            run_dir / "tabular" / "selection_oos_predictions.parquet",
            sequence / "selection" / "predictions.parquet",
        )
    )
    confirmation = _add_fixed_hybrids(
        _merge(
            run_dir / "tabular" / "lockbox_predictions.parquet",
            sequence / "lockbox" / "predictions.parquet",
        )
    )
    for frame in (selection, confirmation):
        frame["timestamp"] = pd.to_datetime(frame["timestamp"])
        frame["symbol"] = frame["symbol"].astype(str)
        if frame.duplicated(["timestamp", "symbol"]).any():
            raise ValueError("prediction artifact contains duplicate keys")
    if confirmation["timestamp"].min() <= selection["timestamp"].max():
        raise ValueError("confirmation must begin after selection")
    return selection, confirmation


def _snapshot_sectors(symbols: Sequence[str], *, require_complete: bool = True):
    from backend.alpha.neutralization import SECTOR_MAP

    snapshot = {symbol: str(SECTOR_MAP.get(symbol, "Unknown")) for symbol in sorted(symbols)}
    unknown = sorted(symbol for symbol, sector in snapshot.items() if sector in {"", "Unknown"})
    if unknown and require_complete:
        raise ValueError("primary neutralization requires complete sector coverage: " + ", ".join(unknown))
    return snapshot


def _merge_factors(predictions: pd.DataFrame, factor_panel: pd.DataFrame) -> pd.DataFrame:
    keys = ["timestamp", "symbol"]
    merged = predictions.merge(
        factor_panel,
        on=keys,
        how="left",
        validate="one_to_one",
    )
    if merged[["factor_ts_mom", "factor_rsi"]].isna().any().any():
        raise ValueError("factor cache failed to cover all OOS prediction keys")
    return merged.sort_values(keys, kind="mergesort").reset_index(drop=True)


def _build_full_universe_factor_variants(
    factor_frame: pd.DataFrame,
    sector_snapshot: Mapping[str, str],
):
    from research_v2.neutralization import build_claude1_factor_variants

    variants, audit = build_claude1_factor_variants(
        factor_frame,
        sector_snapshot,
        min_sector_names=10,
        unknown_policy="passthrough_global",
    )
    keep = [
        "timestamp",
        "symbol",
        *[
            column
            for column in variants
            if column.startswith("score_claude1_factorwise__")
        ],
    ]
    return variants.loc[:, keep], audit


def _build_signal_variants(
    frame: pd.DataFrame,
    sector_snapshot: Mapping[str, str],
    factor_variants: pd.DataFrame,
):
    from research_v2.neutralization import (
        NeutralizationSpec,
        attach_sector_snapshot,
        neutralize_cross_sections,
    )

    result = attach_sector_snapshot(frame, sector_snapshot)
    result = result.merge(
        factor_variants,
        on=["timestamp", "symbol"],
        how="left",
        validate="one_to_one",
    )
    factor_score_cols = [
        column
        for column in result
        if column.startswith("score_claude1_factorwise__")
    ]
    if result[factor_score_cols].isna().any().any():
        raise ValueError("full-universe factor variants failed to cover prediction keys")

    parity_error = float(
        np.nanmax(
            np.abs(
                result["score_claude1_factorwise__none"].to_numpy(float)
                - result["score_production_claude1"].to_numpy(float)
            )
        )
    )
    if parity_error > 1e-12:
        raise AssertionError(f"Claude #1 raw factor reconstruction parity failed: {parity_error}")

    base_scores = (
        "score_production_claude1",
        "score_ensemble",
        "score_ridge",
        "score_gbdt",
        "score_sequence_locked",
        "score_hybrid_tab50_seq50",
    )
    score_audit: Dict[str, object] = {}
    for method in METHODS:
        transformed, audit = neutralize_cross_sections(
            result,
            base_scores,
            sector_snapshot,
            spec=NeutralizationSpec(
                method=method,
                min_sector_names=10,
                unknown_policy="error",
                final_cross_section_rank=True,
            ),
            output_prefix=f"{method}__",
        )
        for base in base_scores:
            result[f"{base}__{method}"] = transformed[f"{method}__{base}"].to_numpy(float)
        score_audit[method] = audit
    return result, {
        "scorewise": score_audit,
        "raw_claude1_reconstruction_max_abs_error": parity_error,
    }


def _signal_specs() -> list[dict[str, object]]:
    specs: list[dict[str, object]] = [
        {
            "family": "claude1",
            "variant": "raw",
            "level": "none",
            "method": "none",
            "score_col": "score_production_claude1",
            "top_n": 20,
        },
        {
            "family": "claude1",
            "variant": "factor_sector_residual",
            "level": "factorwise",
            "method": "sector_residual",
            "score_col": "score_claude1_factorwise__sector_residual",
            "top_n": 20,
        },
        {
            "family": "claude1",
            "variant": "factor_sector_zscore",
            "level": "factorwise",
            "method": "sector_zscore",
            "score_col": "score_claude1_factorwise__sector_zscore",
            "top_n": 20,
        },
        {
            "family": "claude1",
            "variant": "factor_within_sector_rank",
            "level": "factorwise",
            "method": "within_sector_rank",
            "score_col": "score_claude1_factorwise__within_sector_rank",
            "top_n": 20,
        },
        {
            "family": "claude1",
            "variant": "score_sector_residual",
            "level": "scorewise",
            "method": "sector_residual",
            "score_col": "score_production_claude1__sector_residual",
            "top_n": 20,
        },
    ]
    for base, family in (
        ("score_ensemble", "tabular_ensemble"),
        ("score_ridge", "ridge"),
        ("score_gbdt", "gbdt"),
        ("score_sequence_locked", "sequence_ensemble"),
        ("score_hybrid_tab50_seq50", "hybrid50"),
    ):
        specs.append(
            {
                "family": family,
                "variant": "raw",
                "level": "none",
                "method": "none",
                "score_col": base,
                "top_n": 30 if family == "hybrid50" else 20,
            }
        )
        for method in METHODS:
            specs.append(
                {
                    "family": family,
                    "variant": method,
                    "level": "scorewise",
                    "method": method,
                    "score_col": f"{base}__{method}",
                    "top_n": 30 if family == "hybrid50" else 20,
                }
            )
    return specs


def _daily_ic(frame: pd.DataFrame, score_col: str) -> pd.Series:
    from research_v2.validation import daily_rank_ic

    return daily_rank_ic(frame, prediction_col=score_col, label_col="label_rank")


def _selection_overlap(
    frame: pd.DataFrame, raw_col: str, variant_col: str, top_n: int
) -> Dict[str, float]:
    rows = []
    for _date, day in frame.groupby("timestamp", sort=True, observed=True):
        raw = set(day.nlargest(top_n, raw_col)["symbol"].astype(str))
        variant = set(day.nlargest(top_n, variant_col)["symbol"].astype(str))
        union = raw | variant
        rows.append(
            (
                len(raw & variant) / top_n,
                len(raw & variant) / len(union) if union else 1.0,
                day[[raw_col, variant_col]].corr(method="spearman").iloc[0, 1],
            )
        )
    values = np.asarray(rows, dtype=float)
    return {
        "mean_top_n_overlap": float(values[:, 0].mean()),
        "mean_top_n_jaccard": float(values[:, 1].mean()),
        "mean_score_rank_correlation": float(values[:, 2].mean()),
    }


def _signal_diagnostics(
    selection: pd.DataFrame,
    confirmation: pd.DataFrame,
    specs: Sequence[Mapping[str, object]],
):
    from research_v2.neutralization import (
        sector_conditioned_prediction_diagnostics,
        sector_exposure_diagnostics,
        top_n_sector_concentration,
    )
    from research_v2.validation import newey_west_mean_stats, prediction_diagnostics

    rows: list[dict[str, object]] = []
    folds: list[dict[str, object]] = []
    daily_exposure: list[pd.DataFrame] = []
    comparisons: list[dict[str, object]] = []
    frames = {"selection": selection, "opened_2026": confirmation}
    raw_by_family = {
        str(spec["family"]): str(spec["score_col"])
        for spec in specs
        if spec["variant"] == "raw"
    }
    top_by_family = {
        str(spec["family"]): int(spec["top_n"])
        for spec in specs
        if spec["variant"] == "raw"
    }

    for period, frame in frames.items():
        for spec in specs:
            score_col = str(spec["score_col"])
            base = {key: value for key, value in spec.items() if key != "score_col"}
            standard = prediction_diagnostics(frame, prediction_col=score_col, horizon=5)
            conditioned = sector_conditioned_prediction_diagnostics(
                frame, score_col=score_col, horizon=5
            )
            exposure, exposure_daily = sector_exposure_diagnostics(
                frame, score_col=score_col
            )
            concentration, _ = top_n_sector_concentration(
                frame, score_col=score_col, top_n=int(spec["top_n"])
            )
            rows.append(
                {
                    "period": period,
                    **base,
                    "score_col": score_col,
                    **standard,
                    **conditioned,
                    **exposure,
                    **concentration,
                }
            )
            tagged = exposure_daily.copy()
            tagged.insert(0, "period", period)
            tagged.insert(1, "family", str(spec["family"]))
            tagged.insert(2, "variant", str(spec["variant"]))
            daily_exposure.append(tagged)

            for fold_id, part in frame.groupby("fold_id", sort=False, observed=True):
                fold_standard = prediction_diagnostics(part, prediction_col=score_col, horizon=5)
                fold_exposure, _ = sector_exposure_diagnostics(part, score_col=score_col)
                folds.append(
                    {
                        "period": period,
                        "fold_id": str(fold_id),
                        **base,
                        **fold_standard,
                        **fold_exposure,
                    }
                )

            if spec["variant"] != "raw":
                family = str(spec["family"])
                raw_col = raw_by_family[family]
                raw_ic = _daily_ic(frame, raw_col)
                variant_ic = _daily_ic(frame, score_col)
                aligned = pd.concat([raw_ic.rename("raw"), variant_ic.rename("variant")], axis=1).dropna()
                delta = aligned["variant"] - aligned["raw"]
                nw = newey_west_mean_stats(delta, max_lag=4)
                comparisons.append(
                    {
                        "period": period,
                        "family": family,
                        "variant": str(spec["variant"]),
                        "raw_score_col": raw_col,
                        "variant_score_col": score_col,
                        "mean_daily_ic_delta": nw["mean"],
                        "ic_delta_nw_t": nw["nw_t"],
                        "ic_delta_days": nw["n"],
                        **_selection_overlap(
                            frame,
                            raw_col,
                            score_col,
                            top_by_family[family],
                        ),
                    }
                )
    return (
        pd.DataFrame(rows),
        pd.DataFrame(folds),
        pd.concat(daily_exposure, ignore_index=True),
        pd.DataFrame(comparisons),
    )


def _context_slice(context, start: pd.Timestamp, end: pd.Timestamp):
    from research_v2.experiment import MarketContext

    sessions = tuple(session for session in context.sessions if start <= session <= end)
    return MarketContext(
        market={session: context.market[session] for session in sessions},
        full_risk_observations={session: context.full_risk_observations[session] for session in sessions},
        beta_only_observations={session: context.beta_only_observations[session] for session in sessions},
        sectors=context.sectors,
        sessions=sessions,
        symbols=context.symbols,
        metadata={**context.metadata, "study_slice_start": str(start), "study_slice_end": str(end)},
    )


def _performance_rows(result, boundaries: Mapping[str, tuple[pd.Timestamp, pd.Timestamp]]):
    from research_v2.metrics import compute_performance_metrics

    metrics: Dict[str, Mapping[str, float | int]] = {}
    ledgers: Dict[str, list[object]] = {}
    for period, (start, end) in boundaries.items():
        rows = [row for row in result.ledger if start <= pd.Timestamp(row.session) <= end]
        if not rows:
            raise ValueError(f"no ledger rows for {period}")
        ledgers[period] = rows
        metrics[period] = compute_performance_metrics(rows).to_dict()
    return metrics, ledgers


def _portfolio_specs():
    from research_v2.portfolio import PortfolioConfig

    claude_uncapped = PortfolioConfig(
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
    claude_cap = replace(claude_uncapped, sector_cap=0.45)
    hybrid_capped = PortfolioConfig(
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
    hybrid_uncapped = replace(hybrid_capped, single_name_cap=0.75, sector_cap=0.75)
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
    return [
        {
            "scenario": "claude1_live_like_uncapped",
            "family": "claude1",
            "config": claude_uncapped,
            "scores": claude_scores,
            "rebalance_days": 5,
            "offsets": tuple(range(5)),
            "costs": (0.0, 5.0, 10.0, 20.0),
        },
        {
            "scenario": "claude1_relative_sector_cap_30pct",
            "family": "claude1",
            "config": claude_cap,
            "scores": {key: claude_scores[key] for key in ("raw", "factor_sector_residual")},
            "rebalance_days": 5,
            "offsets": tuple(range(5)),
            "costs": (0.0, 5.0, 10.0, 20.0),
        },
        {
            "scenario": "hybrid50_champion_cap_absolute_0.30",
            "family": "hybrid50",
            "config": hybrid_capped,
            "scores": hybrid_scores,
            "rebalance_days": 21,
            "offsets": tuple(range(21)),
            "costs": (0.0, 5.0, 10.0, 20.0),
        },
        {
            "scenario": "hybrid50_uncapped_diagnostic",
            "family": "hybrid50",
            "config": hybrid_uncapped,
            "scores": hybrid_scores,
            "rebalance_days": 21,
            "offsets": (0,),
            "costs": (10.0,),
        },
    ]


def _portfolio_allocation_diagnostics(
    ledger: Sequence[object], context, period: str, scenario: str, variant: str
):
    universe_counts = pd.Series(context.sectors).value_counts(normalize=True)
    daily: list[dict[str, object]] = []
    accum: Dict[str, dict[str, float]] = {
        symbol: {"abs_weight_sum": 0.0, "held_weight_sum": 0.0, "held_days": 0.0}
        for symbol in context.symbols
    }
    for row in ledger:
        session = pd.Timestamp(row.session)
        bars = context.market[session]
        weights = {
            symbol: float(quantity) * float(bars[symbol].close) / float(row.ending_equity)
            for symbol, quantity in row.positions.items()
            if abs(float(quantity)) > 1e-12
        }
        gross = sum(abs(weight) for weight in weights.values())
        sector_weights: Dict[str, float] = {}
        for symbol, weight in weights.items():
            sector = context.sectors[symbol]
            sector_weights[sector] = sector_weights.get(sector, 0.0) + abs(weight)
            state = accum[symbol]
            state["abs_weight_sum"] += abs(weight)
            state["held_weight_sum"] += abs(weight)
            state["held_days"] += 1.0
        if gross > 1e-12:
            shares = pd.Series(sector_weights, dtype=float) / gross
            hhi = float(np.square(shares).sum())
            active = shares.reindex(universe_counts.index, fill_value=0.0) - universe_counts
            daily.append(
                {
                    "period": period,
                    "scenario": scenario,
                    "variant": variant,
                    "timestamp": session,
                    "held_names": len(weights),
                    "gross_exposure": gross,
                    "max_name_weight": max(map(abs, weights.values()), default=0.0),
                    "max_sector_share_of_gross": float(shares.max()),
                    "sector_hhi": hhi,
                    "effective_sectors": 1.0 / hhi if hhi > 0 else np.nan,
                    "max_abs_active_sector_share": float(active.abs().max()),
                }
            )
    count = max(len(ledger), 1)
    by_symbol = []
    for symbol, state in accum.items():
        held_days = int(state["held_days"])
        by_symbol.append(
            {
                "period": period,
                "scenario": scenario,
                "variant": variant,
                "symbol": symbol,
                "sector": context.sectors[symbol],
                "mean_abs_weight_including_zero": state["abs_weight_sum"] / count,
                "mean_abs_weight_when_held": (
                    state["held_weight_sum"] / held_days if held_days else 0.0
                ),
                "holding_frequency": held_days / count,
                "held_days": held_days,
            }
        )
    return pd.DataFrame(daily), pd.DataFrame(by_symbol)


def _run_portfolios(
    context,
    combined: pd.DataFrame,
    selection: pd.DataFrame,
    confirmation: pd.DataFrame,
):
    from research_v2.backtest import RiskConfig
    from research_v2.costs import CostConfig
    from research_v2.experiment import evaluate_strategy, make_signal_map

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
    sliced = _context_slice(
        context,
        pd.Timestamp(combined["timestamp"].min()),
        pd.Timestamp(combined["timestamp"].max()),
    )
    metrics_rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []
    daily_returns: list[dict[str, object]] = []
    allocation_daily: list[pd.DataFrame] = []
    allocation_symbol: list[pd.DataFrame] = []
    progress_total = 0
    for plan in _portfolio_specs():
        progress_total += len(plan["scores"]) * (
            len(plan["offsets"]) + len(set(plan["costs"]) - {10.0})
        )
    completed = 0

    for plan in _portfolio_specs():
        scenario = str(plan["scenario"])
        rebalance_days = int(plan["rebalance_days"])
        for variant, score_col_raw in dict(plan["scores"]).items():
            score_col = str(score_col_raw)
            jobs = {(int(offset), 10.0) for offset in plan["offsets"]}
            jobs.update((0, float(cost)) for cost in plan["costs"])
            for offset, friction_bps in sorted(jobs):
                signals = make_signal_map(
                    combined[["timestamp", "symbol", score_col]],
                    score_column=score_col,
                    eligible_symbols=sliced.symbols,
                    rebalance_days=rebalance_days,
                    offset=offset,
                    liquidate_at_end=False,
                )
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
                    sliced,
                    signals,
                    portfolio_config=plan["config"],
                    cost_config=cost,
                    risk_config=RiskConfig(),
                    use_full_risk_observations=False,
                    initial_capital=100_000.0,
                )
                metrics, ledgers = _performance_rows(evaluation.result, boundaries)
                for period, values in metrics.items():
                    metrics_rows.append(
                        {
                            "period": period,
                            "scenario": scenario,
                            "family": plan["family"],
                            "variant": variant,
                            "score_col": score_col,
                            "rebalance_days": rebalance_days,
                            "offset": offset,
                            "extra_friction_bps": friction_bps,
                            **values,
                        }
                    )

                if offset == 0 and friction_bps == 10.0:
                    for period, ledger in ledgers.items():
                        for row in ledger:
                            daily_returns.append(
                                {
                                    "period": period,
                                    "scenario": scenario,
                                    "family": plan["family"],
                                    "variant": variant,
                                    "timestamp": pd.Timestamp(row.session),
                                    "net_return": float(row.ending_equity / row.starting_equity - 1.0),
                                }
                            )
                        daily, symbols = _portfolio_allocation_diagnostics(
                            ledger, sliced, period, scenario, str(variant)
                        )
                        allocation_daily.append(daily)
                        allocation_symbol.append(symbols)

                    for fold_id, fold in selection.groupby("fold_id", sort=False, observed=True):
                        start = pd.Timestamp(fold["timestamp"].min())
                        end = pd.Timestamp(fold["timestamp"].max())
                        fold_ledger = [
                            row for row in ledgers["selection"]
                            if start <= pd.Timestamp(row.session) <= end
                        ]
                        from research_v2.metrics import compute_performance_metrics

                        fold_rows.append(
                            {
                                "scenario": scenario,
                                "family": plan["family"],
                                "variant": variant,
                                "fold_id": str(fold_id),
                                **compute_performance_metrics(fold_ledger).to_dict(),
                            }
                        )
                completed += 1
                if completed % 10 == 0 or completed == progress_total:
                    print(
                        json.dumps(
                            {
                                "event": "neutralization_portfolio_progress",
                                "completed": completed,
                                "total": progress_total,
                                "scenario": scenario,
                                "variant": variant,
                                "offset": offset,
                                "bps": friction_bps,
                            }
                        ),
                        flush=True,
                    )
    return (
        pd.DataFrame(metrics_rows),
        pd.DataFrame(fold_rows),
        pd.DataFrame(daily_returns),
        pd.concat(allocation_daily, ignore_index=True),
        pd.concat(allocation_symbol, ignore_index=True),
    )


def _moving_block_bootstrap(
    differences: np.ndarray,
    *,
    block_length: int,
    repetitions: int,
    seed: int,
) -> Dict[str, float]:
    values = np.asarray(differences, dtype=float)
    values = values[np.isfinite(values)]
    n = len(values)
    if n < block_length or repetitions < 100:
        raise ValueError("insufficient observations or bootstrap repetitions")
    rng = np.random.default_rng(seed)
    blocks = math.ceil(n / block_length)
    samples = np.empty(repetitions, dtype=float)
    offsets = np.arange(block_length)
    for index in range(repetitions):
        # Circular blocks give every observation equal inclusion probability.
        # Non-circular blocks underweight the two sample edges and can heavily
        # bias short confirmation windows when block_length is 21 sessions.
        starts = rng.integers(0, n, size=blocks)
        positions = ((starts[:, None] + offsets[None, :]) % n).reshape(-1)[:n]
        samples[index] = float(values[positions].mean())
    return {
        "mean_daily_return_delta": float(values.mean()),
        "annualized_mean_return_delta": float(values.mean() * 252.0),
        "bootstrap_ci_2_5": float(np.quantile(samples, 0.025) * 252.0),
        "bootstrap_ci_97_5": float(np.quantile(samples, 0.975) * 252.0),
        "bootstrap_probability_nonpositive": float(np.mean(samples <= 0.0)),
        "bootstrap_repetitions": int(repetitions),
        "bootstrap_block_length": int(block_length),
    }


def _portfolio_comparisons(daily: pd.DataFrame, repetitions: int) -> pd.DataFrame:
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
            merged = raw.merge(
                part[["timestamp", "net_return"]].rename(columns={"net_return": "variant"}),
                on="timestamp",
                how="inner",
                validate="one_to_one",
            )
            diff = merged["variant"].to_numpy(float) - merged["raw"].to_numpy(float)
            rows.append(
                {
                    "scenario": scenario,
                    "period": period,
                    "variant": variant,
                    **_moving_block_bootstrap(
                        diff,
                        block_length=block,
                        repetitions=repetitions,
                        seed=20260714 + sum(map(ord, f"{scenario}:{period}:{variant}")),
                    ),
                }
            )
    return pd.DataFrame(rows)


def _offset_summary(portfolio: pd.DataFrame) -> pd.DataFrame:
    fixed = portfolio.loc[portfolio["extra_friction_bps"] == 10.0]
    rows = []
    for keys, group in fixed.groupby(
        ["period", "scenario", "family", "variant"], sort=True, observed=True
    ):
        rows.append(
            {
                "period": keys[0],
                "scenario": keys[1],
                "family": keys[2],
                "variant": keys[3],
                "offsets": int(group["offset"].nunique()),
                "median_sharpe": float(group["sharpe"].median()),
                "worst_sharpe": float(group["sharpe"].min()),
                "best_sharpe": float(group["sharpe"].max()),
                "median_cagr": float(group["cagr"].median()),
                "worst_cagr": float(group["cagr"].min()),
                "median_max_drawdown": float(group["max_drawdown"].median()),
                "worst_max_drawdown": float(group["max_drawdown"].min()),
                "median_gross_turnover": float(group["total_gross_turnover"].median()),
                "median_total_cost": float(group["total_cost"].median()),
            }
        )
    return pd.DataFrame(rows)


def _acceptance_gates(signal: pd.DataFrame, offsets: pd.DataFrame) -> list[dict[str, object]]:
    definitions = (
        (
            "claude1_factor_sector_residual",
            "claude1",
            "factor_sector_residual",
            "claude1_live_like_uncapped",
        ),
        (
            "hybrid50_score_sector_residual",
            "hybrid50",
            "sector_residual",
            "hybrid50_champion_cap_absolute_0.30",
        ),
    )
    rows = []
    for name, family, variant, scenario in definitions:
        period_signal = signal.loc[
            (signal["period"] == "selection") & (signal["family"] == family)
        ].set_index("variant")
        period_portfolio = offsets.loc[
            (offsets["period"] == "selection") & (offsets["scenario"] == scenario)
        ].set_index("variant")
        raw_s = period_signal.loc["raw"]
        new_s = period_signal.loc[variant]
        raw_p = period_portfolio.loc["raw"]
        new_p = period_portfolio.loc[variant]
        r2_reduction = 1.0 - float(new_s["mean_sector_r2"]) / max(
            float(raw_s["mean_sector_r2"]), 1e-15
        )
        ic_delta = float(new_s["mean_rank_ic"] - raw_s["mean_rank_ic"])
        sharpe_delta = float(new_p["median_sharpe"] - raw_p["median_sharpe"])
        drawdown_delta = float(new_p["median_max_drawdown"] - raw_p["median_max_drawdown"])
        turnover_ratio = float(new_p["median_gross_turnover"] / raw_p["median_gross_turnover"])
        gates = {
            "sector_r2_reduction_at_least_80pct": r2_reduction >= 0.80,
            "rank_ic_degradation_no_more_than_0_003": ic_delta >= -0.003,
            "median_offset_sharpe_degradation_no_more_than_0_05": sharpe_delta >= -0.05,
            "median_max_drawdown_not_worse_by_more_than_2pp": drawdown_delta >= -0.02,
            "turnover_increase_no_more_than_15pct": turnover_ratio <= 1.15,
        }
        rows.append(
            {
                "test": name,
                "family": family,
                "variant": variant,
                "scenario": scenario,
                "r2_reduction": r2_reduction,
                "rank_ic_delta": ic_delta,
                "median_offset_sharpe_delta": sharpe_delta,
                "median_max_drawdown_delta": drawdown_delta,
                "turnover_ratio": turnover_ratio,
                **gates,
                "all_gates_pass": all(gates.values()),
            }
        )
    return rows


def _format_pct(value: object) -> str:
    return "—" if pd.isna(value) else f"{float(value):.2%}"


def _format_num(value: object, digits: int = 3) -> str:
    return "—" if pd.isna(value) else f"{float(value):.{digits}f}"


def _report(
    manifest: Mapping[str, object],
    signal: pd.DataFrame,
    offsets: pd.DataFrame,
    gates: Sequence[Mapping[str, object]],
) -> str:
    lines = [
        "# 行業中性化可驗證研究報告",
        "",
        f"完成時間：{manifest['completed_at_utc']}",
        "",
        "## 先講結論",
        "",
    ]
    for gate in gates:
        verdict = "通過預先門檻" if gate["all_gates_pass"] else "未通過預先門檻"
        lines.append(
            f"- **{gate['test']}：{verdict}**。行業 R² 降低 {_format_pct(gate['r2_reduction'])}，"
            f"Rank IC 差 {_format_num(gate['rank_ic_delta'], 4)}，offset 中位 Sharpe 差 "
            f"{_format_num(gate['median_offset_sharpe_delta'])}，換手比 "
            f"{_format_num(gate['turnover_ratio'])}。"
        )
    lines.extend(
        [
            "",
            "這是 discovery／探索性證據，不是全新未開封 lockbox。Selection OOS 與 2026 確認期先前均已被研究流程查看；2026 在本次只用來描述方向是否延續，沒有用來挑方法。",
            "",
            "## 資料邊界",
            "",
            f"- 股票：{manifest['universe']['symbols']}；行業：{manifest['universe']['sectors']}；映射覆蓋率：100%。",
            f"- 行業映射 canonical-content SHA-256：`{manifest['sector_snapshot']['sha256']}`。它是 2026-07 靜態 broad-sector 標籤，不是歷史 point-in-time GICS。",
            "- 為精確重建原版 Claude #1，factorwise 先在原排名用的 518 檔完整截面處理，再切回 480 檔 OOS 股票；其中 9 檔無行業標籤，只保留全市場排名、不參與行業回歸。scorewise／ML 比較則固定在 480 檔完整樣本。",
            "- 資料沒有逐日 market cap、shares outstanding 或歷史可用時間戳；所以**市值中性化未測**。ADV／成交額沒有被冒充為市值。",
            "- 所有轉換只用當日橫截面；訊號在 close(t) 形成、open(t+1) 執行。原 daily run、live config、production scheduler 均未修改。",
            "",
            "## 方法差異",
            "",
            "- `sector_residual`：每日 score 對 intercept + sector dummy 的 OLS 殘差；數學上等於減去當日行業均值。",
            "- `sector_zscore`：先減行業均值，再除以當日行業內標準差；同時消除平均與尺度。",
            "- `within_sector_rank`：各行業內百分位排序；最抗離群值，但丟掉距離資訊。",
            "- Claude #1 的 factorwise 測試先分別處理 Momentum 與 RSI，再按固定 70/30 合成；ML/hybrid 則是固定模型後處理 score，沒有重新訓練。",
            "- 因子中性化不等於持倉中性化；long-only Top-N 仍可能集中，所以另報實際持倉並分開測 sector cap。",
            "",
            "## 訊號結果（Selection OOS）",
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
            f"| {row.family} | {row.variant} | {_format_num(row.mean_rank_ic, 4)} | "
            f"{_format_num(row.rank_ic_nw_t, 2)} | {_format_num(row.mean_within_sector_rank_ic, 4)} | "
            f"{_format_pct(row.mean_sector_r2)} | {_format_pct(row.mean_max_sector_share)} |"
        )
    lines.extend(
        [
            "",
            "## 組合結果（10 bps 額外摩擦；所有調倉 offsets）",
            "",
            "下表用中位與最差 offset，避免只展示最好的一個調倉日。成本仍另含 spread proxy、impact 與槓桿融資。",
            "",
            "| 期間 | 場景 | 版本 | offsets | 中位 Sharpe | 最差 Sharpe | 中位 CAGR | 最差 MaxDD | 中位換手 |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    main_offsets = offsets.loc[
        offsets["scenario"].isin(
            ["claude1_live_like_uncapped", "hybrid50_champion_cap_absolute_0.30"]
        )
    ]
    for row in main_offsets.itertuples(index=False):
        lines.append(
            f"| {row.period} | {row.scenario} | {row.variant} | {row.offsets} | "
            f"{_format_num(row.median_sharpe)} | {_format_num(row.worst_sharpe)} | "
            f"{_format_pct(row.median_cagr)} | {_format_pct(row.worst_max_drawdown)} | "
            f"{_format_num(row.median_gross_turnover, 2)} |"
        )
    lines.extend(
        [
            "",
            "## 如何解讀",
            "",
            "中性化若降低 Sector R² 卻同時顯著降低 IC／Sharpe，代表原訊號的一部分其實是有效的 sector timing，而非純污染。反之，若暴露明顯下降、alpha 與成本後結果不劣，才值得進 shadow-forward。報告的 gate 採這個『去風險且 alpha 非劣』原則，不用最高 Sharpe 追逐版本。",
            "",
            "下一個真正可部署的驗證步驟，是固定通過版本做新的 shadow-forward；取得 PIT market cap、shares 與歷史 sector effective dates 後，再跑 joint OLS：`score ~ z(log_market_cap) + sector dummies`。",
            "",
            "## 可重現輸出",
            "",
            "`manifest.json` 固定輸入 hash、行業快照與限制；CSV／Parquet 包含 signal、fold、offset、成本敏感度、每日報酬與平均持倉分配。`_SUCCESS.json` 只會在全部輸出完成後寫入。",
            "",
        ]
    )
    return "\n".join(lines)


def _refresh_derived_outputs(output: Path, repetitions: int) -> None:
    """Repair/rebuild statistics that depend only on completed study outputs."""

    required = (
        output / "manifest.json",
        output / "signal_diagnostics.csv",
        output / "portfolio_offset_summary_10bps.csv",
        output / "portfolio_daily_returns_offset0_10bps.parquet",
        output / "_SUCCESS.json",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("cannot refresh incomplete study: " + ", ".join(missing))
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    signal = pd.read_csv(output / "signal_diagnostics.csv")
    offsets = pd.read_csv(output / "portfolio_offset_summary_10bps.csv")
    daily = pd.read_parquet(output / "portfolio_daily_returns_offset0_10bps.parquet")
    comparisons = _portfolio_comparisons(daily, repetitions=repetitions)
    gates = _acceptance_gates(signal, offsets)
    refreshed = _utc_now()
    manifest["bootstrap"] = {
        "method": "deterministic circular moving-block bootstrap of paired daily net returns",
        "repetitions": repetitions,
        "block_length": "rebalance cadence: 5 Claude1 / 21 hybrid50",
        "equal_observation_inclusion_probability": True,
    }
    manifest["derived_outputs_refreshed_at_utc"] = refreshed
    comparisons.to_csv(output / "portfolio_paired_bootstrap.csv", index=False)
    _write_json(output / "acceptance_gates.json", gates)
    _write_json(output / "manifest.json", manifest)
    summary_path = output / "summary.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["acceptance_gates"] = gates
        summary["derived_outputs_refreshed_at_utc"] = refreshed
        _write_json(summary_path, summary)
    (output / "report.md").write_text(
        _report(manifest, signal, offsets, gates), encoding="utf-8"
    )
    success = json.loads((output / "_SUCCESS.json").read_text(encoding="utf-8"))
    success["derived_outputs_refreshed_at_utc"] = refreshed
    success["paired_bootstrap"] = "circular_moving_block"
    _write_json(output / "_SUCCESS.json", success)


def main(argv: Sequence[str] | None = None) -> int:
    args = _arguments(argv)
    if args.bootstrap_repetitions < 100:
        raise ValueError("bootstrap repetitions must be at least 100")
    from research_v2.safety import ensure_research_output_path, offline_context

    run_dir = ensure_research_output_path(args.run_dir, research_root=ROOT)
    cache = args.cache.expanduser().resolve(strict=True)
    output = ensure_research_output_path(run_dir / OUTPUT_NAME, research_root=ROOT)
    if output.exists():
        raise FileExistsError(f"study output already exists; refusing overwrite: {output}")
    staging = ensure_research_output_path(
        run_dir / f".{OUTPUT_NAME}.partial-{uuid.uuid4().hex}", research_root=ROOT
    )
    staging.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()

    try:
        with offline_context():
            import polars as pl

            from research_v2.experiment import build_market_context, complete_case_symbols

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
            prediction_symbols = tuple(sorted(selection["symbol"].unique()))
            if prediction_symbols != tuple(sorted(confirmation["symbol"].unique())):
                raise ValueError("selection and confirmation universes differ")
            if set(prediction_symbols) != set(symbols):
                raise ValueError("OOS predictions are not the cache complete-case universe")
            sector_snapshot = _snapshot_sectors(symbols)
            all_factor_symbols = tuple(sorted(panel["symbol"].unique().to_list()))
            full_factor_sector_snapshot = _snapshot_sectors(
                all_factor_symbols, require_complete=False
            )
            factor_panel = (
                panel.select("timestamp", "symbol", "factor_ts_mom", "factor_rsi")
                .to_pandas()
            )
            factor_panel["timestamp"] = pd.to_datetime(factor_panel["timestamp"])
            factor_panel["symbol"] = factor_panel["symbol"].astype(str)
            selection = _merge_factors(selection, factor_panel)
            confirmation = _merge_factors(confirmation, factor_panel)

            selection_dates = set(pd.to_datetime(selection["timestamp"].unique()))
            confirmation_dates = set(pd.to_datetime(confirmation["timestamp"].unique()))
            selection_factor_variants, selection_factor_audit = _build_full_universe_factor_variants(
                factor_panel.loc[factor_panel["timestamp"].isin(selection_dates)],
                full_factor_sector_snapshot,
            )
            confirmation_factor_variants, confirmation_factor_audit = _build_full_universe_factor_variants(
                factor_panel.loc[factor_panel["timestamp"].isin(confirmation_dates)],
                full_factor_sector_snapshot,
            )
            selection, selection_audit = _build_signal_variants(
                selection, sector_snapshot, selection_factor_variants
            )
            confirmation, confirmation_audit = _build_signal_variants(
                confirmation, sector_snapshot, confirmation_factor_variants
            )
            selection_audit["factorwise_full_universe"] = selection_factor_audit
            confirmation_audit["factorwise_full_universe"] = confirmation_factor_audit
            specs = _signal_specs()
            signal, signal_folds, exposure_daily, signal_comparisons = _signal_diagnostics(
                selection, confirmation, specs
            )

            combined = pd.concat([selection, confirmation], ignore_index=True).sort_values(
                ["timestamp", "symbol"], kind="mergesort"
            )
            context = build_market_context(
                panel.select(market_columns),
                symbols=symbols,
                start=combined["timestamp"].min(),
                end=combined["timestamp"].max(),
                beta_lookback=126,
                spread_range_fraction=0.02,
                min_spread_bps=1.0,
                max_spread_bps=30.0,
            )
            portfolio, portfolio_folds, daily_returns, allocation_daily, allocation_symbol = _run_portfolios(
                context, combined, selection, confirmation
            )
            offset_summary = _offset_summary(portfolio)
            portfolio_comparisons = _portfolio_comparisons(
                daily_returns, repetitions=args.bootstrap_repetitions
            )
            gates = _acceptance_gates(signal, offset_summary)

            selection_start = pd.Timestamp(selection["timestamp"].min())
            selection_end = pd.Timestamp(selection["timestamp"].max())
            confirmation_start = pd.Timestamp(confirmation["timestamp"].min())
            confirmation_end = pd.Timestamp(confirmation["timestamp"].max())
            input_paths = [
                cache,
                run_dir / "tabular" / "selection_oos_predictions.parquet",
                run_dir / "tabular" / "lockbox_predictions.parquet",
                run_dir / "sequence_full60_all" / "selection" / "predictions.parquet",
                run_dir / "sequence_full60_all" / "lockbox" / "predictions.parquet",
            ]
            sector_counts = pd.Series(sector_snapshot).value_counts().sort_index().to_dict()
            completed_at = _utc_now()
            manifest = {
                "study": "sector_neutralization_oos_v1",
                "started_at_utc": datetime.fromtimestamp(
                    datetime.now().timestamp() - (time.perf_counter() - started), timezone.utc
                ).isoformat(),
                "completed_at_utc": completed_at,
                "elapsed_seconds": time.perf_counter() - started,
                "selection_only_method_gates": True,
                "confirmation_status": "previously_opened_2026_confirmation_not_used_for_selection",
                "decision_clock": "close(t) signal -> open(t+1) execution",
                "universe": {
                    "symbols": len(symbols),
                    "sectors": len(sector_counts),
                    "sector_counts": sector_counts,
                    "selection_rows": len(selection),
                    "selection_days": int(selection["timestamp"].nunique()),
                    "selection_start": selection_start,
                    "selection_end": selection_end,
                    "confirmation_rows": len(confirmation),
                    "confirmation_days": int(confirmation["timestamp"].nunique()),
                    "confirmation_start": confirmation_start,
                    "confirmation_end": confirmation_end,
                },
                "sector_snapshot": {
                    "source": "backend.alpha.neutralization.SECTOR_MAP working-tree snapshot",
                    "review_date": "2026-07",
                    "point_in_time": False,
                    "coverage": 1.0,
                    "sha256": _canonical_sha256(sector_snapshot),
                    "known_data_quality_notes": [
                        "broad 11-sector static classification projected backward",
                        "source file defines GPN twice; Python runtime value Industrials is snapshotted",
                    ],
                },
                "factor_neutralization_universe": {
                    "symbols": len(all_factor_symbols),
                    "unknown_symbols_passthrough_global_rank": sorted(
                        symbol
                        for symbol, sector in full_factor_sector_snapshot.items()
                        if sector in {"", "Unknown"}
                    ),
                    "sector_snapshot_sha256": _canonical_sha256(
                        full_factor_sector_snapshot
                    ),
                    "reason": "match original Claude1 ranking cross-section before slicing to 480 OOS names",
                },
                "market_cap_neutralization": {
                    "status": "not_tested",
                    "reason": "no historical point-in-time market cap or shares outstanding",
                    "prohibited_substitutes": ["price", "dollar volume", "ADV", "current market cap backfill"],
                },
                "predeclared_gates": {
                    "sector_r2_reduction": 0.80,
                    "maximum_rank_ic_degradation": 0.003,
                    "maximum_median_offset_sharpe_degradation": 0.05,
                    "maximum_drawdown_degradation_percentage_points": 0.02,
                    "maximum_turnover_increase": 0.15,
                },
                "bootstrap": {
                    "method": "deterministic circular moving-block bootstrap of paired daily net returns",
                    "repetitions": args.bootstrap_repetitions,
                    "block_length": "rebalance cadence: 5 Claude1 / 21 hybrid50",
                    "equal_observation_inclusion_probability": True,
                },
                "input_sha256": {str(path.resolve()): _file_sha256(path) for path in input_paths},
                "audits": {
                    "selection": selection_audit,
                    "confirmation": confirmation_audit,
                    "market_context": context.metadata,
                },
                "production_mutations": [],
            }

            _write_json(staging / "sector_map_snapshot.json", sector_snapshot)
            _write_json(
                staging / "factor_universe_sector_map_snapshot.json",
                full_factor_sector_snapshot,
            )
            _write_json(staging / "manifest.json", manifest)
            _write_json(staging / "acceptance_gates.json", gates)
            signal.to_csv(staging / "signal_diagnostics.csv", index=False)
            signal_folds.to_csv(staging / "signal_fold_diagnostics.csv", index=False)
            exposure_daily.to_parquet(staging / "signal_daily_sector_exposure.parquet", index=False)
            signal_comparisons.to_csv(staging / "signal_comparisons_vs_raw.csv", index=False)
            portfolio.to_csv(staging / "portfolio_metrics_all_offsets_costs.csv", index=False)
            portfolio_folds.to_csv(staging / "portfolio_fold_metrics.csv", index=False)
            offset_summary.to_csv(staging / "portfolio_offset_summary_10bps.csv", index=False)
            daily_returns.to_parquet(staging / "portfolio_daily_returns_offset0_10bps.parquet", index=False)
            portfolio_comparisons.to_csv(staging / "portfolio_paired_bootstrap.csv", index=False)
            allocation_daily.to_parquet(staging / "portfolio_daily_allocation_offset0_10bps.parquet", index=False)
            allocation_symbol.to_csv(staging / "portfolio_average_symbol_allocation.csv", index=False)
            report = _report(manifest, signal, offset_summary, gates)
            (staging / "report.md").write_text(report, encoding="utf-8")
            _write_json(
                staging / "summary.json",
                {
                    "acceptance_gates": gates,
                    "selection_signal": signal.loc[signal["period"] == "selection"].to_dict("records"),
                    "confirmation_signal": signal.loc[signal["period"] == "opened_2026"].to_dict("records"),
                    "offset_summary": offset_summary.to_dict("records"),
                },
            )
            _write_json(
                staging / "_SUCCESS.json",
                {
                    "completed_at_utc": completed_at,
                    "selection_only_method_gates": True,
                    "market_cap_neutralization_tested": False,
                    "sector_snapshot_sha256": manifest["sector_snapshot"]["sha256"],
                },
            )
        os.replace(staging, output)
        print(
            json.dumps(
                {
                    "event": "neutralization_study_completed",
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
