"""Adapters from frozen feature artifacts to the pure Research v2 engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import polars as pl

from .backtest import MarketBar, RiskConfig, RiskObservation, BacktestResult, run_backtest
from .costs import CostConfig
from .metrics import PerformanceMetrics, compute_performance_metrics
from .portfolio import PortfolioConfig


@dataclass(frozen=True)
class MarketContext:
    market: Mapping[pd.Timestamp, Mapping[str, MarketBar]]
    full_risk_observations: Mapping[pd.Timestamp, RiskObservation]
    beta_only_observations: Mapping[pd.Timestamp, RiskObservation]
    sectors: Mapping[str, str]
    sessions: Tuple[pd.Timestamp, ...]
    symbols: Tuple[str, ...]
    metadata: Mapping[str, object]


@dataclass(frozen=True)
class EvaluationResult:
    result: BacktestResult
    metrics: PerformanceMetrics
    evaluation_ledger_start: int


def complete_case_symbols(panel: pl.DataFrame) -> Tuple[str, ...]:
    if not {"timestamp", "symbol"}.issubset(panel.columns):
        raise ValueError("panel requires timestamp and symbol")
    total_sessions = panel["timestamp"].n_unique()
    complete = (
        panel.group_by("symbol")
        .agg(pl.col("timestamp").n_unique().alias("sessions"))
        .filter(pl.col("sessions") == total_sessions)
        .select(pl.col("symbol").cast(pl.Utf8))
        .sort("symbol")["symbol"]
        .to_list()
    )
    if not complete:
        raise ValueError("no complete-case symbols")
    return tuple(complete)


def build_sector_map(symbols: Sequence[str]) -> Tuple[Dict[str, str], Dict[str, object]]:
    from backend.alpha.neutralization import SECTOR_MAP

    sectors: Dict[str, str] = {}
    known = 0
    for symbol in symbols:
        if symbol in SECTOR_MAP and SECTOR_MAP[symbol] != "Unknown":
            sectors[symbol] = str(SECTOR_MAP[symbol])
            known += 1
        else:
            # A single giant Unknown bucket would impose a fictional industry
            # constraint.  Keep each unmapped name separate and report that the
            # sector cap cannot protect those names from true industry crowding.
            sectors[symbol] = f"Unknown:{symbol}"
    return sectors, {
        "sector_known": known,
        "sector_unknown": len(symbols) - known,
        "sector_coverage": known / len(symbols),
        "unknown_policy": "unique pseudo-sector; no fabricated shared Unknown bucket",
    }


def _rolling_betas(close: pd.DataFrame, lookback: int = 126, minimum: int = 60) -> Tuple[pd.DataFrame, pd.Series]:
    returns = close.pct_change(fill_method=None)
    market = returns.mean(axis=1, skipna=True)
    mean_r = returns.rolling(lookback, min_periods=minimum).mean()
    mean_m = market.rolling(lookback, min_periods=minimum).mean()
    mean_rm = returns.mul(market, axis=0).rolling(lookback, min_periods=minimum).mean()
    mean_m2 = market.pow(2).rolling(lookback, min_periods=minimum).mean()
    covariance = mean_rm.sub(mean_r.mul(mean_m, axis=0), axis=0)
    variance = (mean_m2 - mean_m.pow(2)).replace(0.0, np.nan)
    beta = covariance.div(variance, axis=0).clip(-3.0, 3.0)
    return beta, market.fillna(0.0)


def build_market_context(
    feature_panel: pl.DataFrame,
    *,
    symbols: Sequence[str],
    start: object,
    end: object,
    beta_lookback: int = 126,
    spread_range_fraction: float = 0.02,
    min_spread_bps: float = 1.0,
    max_spread_bps: float = 30.0,
) -> MarketContext:
    required = {
        "timestamp", "symbol", "open", "close", "adv20_v2",
        "realized_vol_20d_v2", "range_pct_v2", "breadth_200d_v2",
    }
    missing = required - set(feature_panel.columns)
    if missing:
        raise ValueError(f"feature panel missing market columns: {sorted(missing)}")
    symbol_set = set(map(str, symbols))
    df = feature_panel.filter(pl.col("symbol").cast(pl.Utf8).is_in(symbol_set)).select(sorted(required))
    pdf = df.to_pandas()
    pdf["timestamp"] = pd.to_datetime(pdf["timestamp"])
    pdf["symbol"] = pdf["symbol"].astype(str)
    pdf = pdf.sort_values(["timestamp", "symbol"])

    close = pdf.pivot(index="timestamp", columns="symbol", values="close").sort_index()
    missing_symbols = sorted(symbol_set - set(close.columns))
    if missing_symbols:
        raise ValueError(f"market context missing symbols: {missing_symbols[:10]}")
    close = close.loc[:, sorted(symbol_set)]
    beta, market_return = _rolling_betas(close, lookback=beta_lookback)
    benchmark = (1.0 + market_return).cumprod()
    slow = benchmark.rolling(200, min_periods=100).mean()
    fast = benchmark.rolling(20, min_periods=10).mean()
    breadth = pdf.groupby("timestamp")["breadth_200d_v2"].first().reindex(close.index)

    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    sessions = tuple(pd.Timestamp(d) for d in close.index if start_ts <= d <= end_ts)
    if not sessions:
        raise ValueError("market context date range is empty")

    indexed = pdf.set_index(["timestamp", "symbol"]).sort_index()
    market: Dict[pd.Timestamp, Dict[str, MarketBar]] = {}
    full_risk: Dict[pd.Timestamp, RiskObservation] = {}
    beta_only: Dict[pd.Timestamp, RiskObservation] = {}
    for session in sessions:
        try:
            day = indexed.loc[session]
        except KeyError as exc:
            raise ValueError(f"feature panel has no rows on {session}") from exc
        bars: Dict[str, MarketBar] = {}
        betas: Dict[str, float] = {}
        for symbol in sorted(symbol_set):
            if symbol not in day.index:
                raise ValueError(f"complete-case symbol {symbol} missing on {session}")
            row = day.loc[symbol]
            adv = float(row["adv20_v2"])
            vol = float(row["realized_vol_20d_v2"])
            range_pct = float(row["range_pct_v2"])
            if not np.isfinite(adv) or adv <= 0:
                raise ValueError(f"invalid point-in-time ADV for {symbol} on {session}")
            if not np.isfinite(vol) or vol < 0:
                raise ValueError(f"invalid point-in-time volatility for {symbol} on {session}")
            spread = float(np.clip(range_pct * 10_000.0 * spread_range_fraction, min_spread_bps, max_spread_bps))
            beta_value = float(beta.at[session, symbol]) if session in beta.index else np.nan
            if not np.isfinite(beta_value):
                beta_value = 1.0
            betas[symbol] = beta_value
            bars[symbol] = MarketBar(
                open=float(row["open"]),
                close=float(row["close"]),
                adv_dollars=adv,
                daily_volatility=vol,
                spread_proxy_bps=spread,
                beta=beta_value,
            )
        market[session] = bars
        beta_obs = RiskObservation(betas=betas)
        beta_only[session] = beta_obs
        b = float(benchmark.get(session, np.nan))
        s = float(slow.get(session, np.nan))
        f = float(fast.get(session, np.nan))
        br = float(breadth.get(session, np.nan))
        full_risk[session] = RiskObservation(
            benchmark_close=b if np.isfinite(b) else None,
            benchmark_slow=s if np.isfinite(s) else None,
            benchmark_fast=f if np.isfinite(f) else None,
            breadth=br if np.isfinite(br) else None,
            crowding_score=None,
            betas=betas,
        )

    sectors, sector_meta = build_sector_map(sorted(symbol_set))
    metadata = {
        **sector_meta,
        "symbols": len(symbol_set),
        "sessions": len(sessions),
        "start": str(sessions[0]),
        "end": str(sessions[-1]),
        "benchmark": "cumulative equal-weight complete-case universe proxy",
        "beta": f"rolling {beta_lookback}-session beta to local universe proxy",
        "spread_proxy": f"clip(daily high-low range * {spread_range_fraction}, {min_spread_bps}-{max_spread_bps} bps)",
    }
    return MarketContext(market, full_risk, beta_only, sectors, sessions, tuple(sorted(symbol_set)), metadata)


def make_signal_map(
    predictions: pd.DataFrame,
    *,
    score_column: str,
    eligible_symbols: Sequence[str],
    rebalance_days: int,
    offset: int = 0,
    liquidate_at_end: bool = True,
) -> Dict[pd.Timestamp, Dict[str, float]]:
    if rebalance_days < 1 or not 0 <= offset < rebalance_days:
        raise ValueError("invalid rebalance_days/offset")
    required = {"timestamp", "symbol", score_column}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"predictions missing signal columns: {sorted(missing)}")
    frame = predictions.loc[predictions["symbol"].astype(str).isin(set(map(str, eligible_symbols)))].copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    dates = pd.DatetimeIndex(frame["timestamp"].drop_duplicates().sort_values())
    signals: Dict[pd.Timestamp, Dict[str, float]] = {}
    for index, date in enumerate(dates):
        if index % rebalance_days != offset:
            continue
        day = frame.loc[frame["timestamp"] == date, ["symbol", score_column]].dropna()
        scores = {str(row.symbol): float(getattr(row, score_column)) for row in day.itertuples(index=False)}
        if scores:
            signals[pd.Timestamp(date)] = scores
    if liquidate_at_end and len(dates):
        signals[pd.Timestamp(dates[-1])] = {}
    return signals


def evaluate_strategy(
    context: MarketContext,
    signals: Mapping[pd.Timestamp, Mapping[str, float]],
    *,
    portfolio_config: PortfolioConfig,
    cost_config: CostConfig,
    risk_config: RiskConfig,
    use_full_risk_observations: bool,
    initial_capital: float = 100_000.0,
) -> EvaluationResult:
    observations = context.full_risk_observations if use_full_risk_observations else context.beta_only_observations
    result = run_backtest(
        context.market,
        signals,
        context.sectors,
        portfolio_config,
        cost_config,
        risk_config,
        observations,
        initial_capital=initial_capital,
    )
    first_execution = next((i for i, row in enumerate(result.ledger) if row.executed_signal_session is not None), None)
    if first_execution is None:
        raise ValueError("strategy never executed a signal")
    evaluation_ledger = result.ledger[first_execution:]
    metrics = compute_performance_metrics(evaluation_ledger)
    return EvaluationResult(result, metrics, first_execution)
