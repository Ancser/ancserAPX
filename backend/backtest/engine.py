"""
BacktestEngine — reads from local Parquet store first, falls back to Alpaca live fetch.
Ported from ancserAPX with data-source abstraction replaced by local store.
"""

import polars as pl
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from datetime import datetime

from backend.data import store
from backend.alpha.factors import compute_all_factors, RUNTIME_FACTOR_META
from backend.alpha.mwu import MWUEngine


_BPS = 10_000.0


@dataclass(frozen=True)
class BacktestCostConfig:
    """Simple, auditable production-backtest trading-cost assumptions.

    All rates are *one-way basis points of traded notional*:

    - ``commission_bps`` applies to buys and sells.  Its default is zero,
      matching the broker commission normally charged by Alpaca for US stocks.
    - ``slippage_bps`` applies to buys and sells and represents spread plus
      execution slippage.  It is deliberately not labelled commission.
    - ``regulatory_sell_bps`` applies only to sales.  It defaults to zero
      because SEC/FINRA rates change over time and TAF is share/cap based; a
      caller can provide a documented blended bps assumption for its period.

    The 5 bps default slippage makes a normal production backtest cost-aware
    without pretending that zero broker commission means zero trading cost.
    Research v2 retains the richer spread/impact/ADV model.
    """

    commission_bps: float = 0.0
    slippage_bps: float = 5.0
    regulatory_sell_bps: float = 0.0

    def __post_init__(self) -> None:
        for name in ("commission_bps", "slippage_bps", "regulatory_sell_bps"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")


def _estimate_trade_costs(
    pretrade_weights: Dict[str, float],
    target_weights: Dict[str, float],
    equity: float,
    config: BacktestCostConfig,
) -> Dict[str, float]:
    """Return turnover and dollar costs for one target-weight transition.

    ``gross_turnover`` is ``sum(abs(target - pretrade))``.  Each traded dollar
    is therefore charged exactly once.  ``one_way_turnover`` is the conventional
    half-gross statistic; it is reported, never used as another cost base.
    """

    symbols = set(pretrade_weights) | set(target_weights)
    deltas = {
        symbol: float(target_weights.get(symbol, 0.0))
        - float(pretrade_weights.get(symbol, 0.0))
        for symbol in symbols
    }
    buy_turnover = sum(max(delta, 0.0) for delta in deltas.values())
    sell_turnover = sum(max(-delta, 0.0) for delta in deltas.values())
    gross_turnover = buy_turnover + sell_turnover
    traded_notional = max(float(equity), 0.0) * gross_turnover
    sell_notional = max(float(equity), 0.0) * sell_turnover
    commission = traded_notional * float(config.commission_bps) / _BPS
    slippage = traded_notional * float(config.slippage_bps) / _BPS
    regulatory = sell_notional * float(config.regulatory_sell_bps) / _BPS
    return {
        "buy_turnover": float(buy_turnover),
        "sell_turnover": float(sell_turnover),
        "gross_turnover": float(gross_turnover),
        "one_way_turnover": float(0.5 * gross_turnover),
        "traded_notional": float(traded_notional),
        "commission_cost": float(commission),
        "slippage_cost": float(slippage),
        "regulatory_cost": float(regulatory),
        "transaction_cost": float(commission + slippage + regulatory),
    }


def _zero_trade_costs() -> Dict[str, float]:
    return {
        "buy_turnover": 0.0,
        "sell_turnover": 0.0,
        "gross_turnover": 0.0,
        "one_way_turnover": 0.0,
        "traded_notional": 0.0,
        "commission_cost": 0.0,
        "slippage_cost": 0.0,
        "regulatory_cost": 0.0,
        "transaction_cost": 0.0,
    }


# ── Metrics helper ────────────────────────────────────────────────────────────

def compute_metrics(
    res_df: pd.DataFrame,
    initial_capital: float,
    holding_period_days: int = 1,
) -> Dict:
    equity = res_df["equity"].dropna()
    if len(equity) < 2:
        return {}

    returns = equity.pct_change().dropna()
    final = equity.iloc[-1]
    n_years = max(len(equity) / 252, 0.01)

    cagr = ((final / initial_capital) ** (1 / n_years) - 1) * 100
    sharpe = float(returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0.0
    rolling_max = equity.cummax()
    dd = (equity - rolling_max) / rolling_max
    max_dd = float(dd.min() * 100)
    trough_date = dd.idxmin()
    peak_date = equity.loc[:trough_date].idxmax()
    peak_value = float(equity.loc[peak_date])
    labels = list(equity.index)
    peak_pos = labels.index(peak_date)
    trough_pos = labels.index(trough_date)
    dd_fall_days = max(0, trough_pos - peak_pos)
    recovery_slice = equity.iloc[trough_pos:]
    recovered = recovery_slice[recovery_slice >= peak_value]
    if not recovered.empty:
        recovery_date = recovered.index[0]
        recovery_pos = labels.index(recovery_date)
        dd_recovery_days = max(0, recovery_pos - trough_pos)
    else:
        recovery_date = None
        dd_recovery_days = None
    calmar = cagr / abs(max_dd) if abs(max_dd) > 0.01 else 0.0
    win_rate = float((returns > 0).mean() * 100)
    hold_days = max(1, int(holding_period_days or 1))
    holding_rets = equity.pct_change(hold_days).iloc[hold_days::hold_days].dropna()
    holding_win_rate = (
        float((holding_rets > 0).mean() * 100) if len(holding_rets) else None
    )
    total_return = (final / initial_capital - 1) * 100
    pnl = equity.diff().dropna()
    gross_profit = float(pnl[pnl > 0].sum())
    gross_loss = float(-pnl[pnl < 0].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else None

    def _sum_column(name: str) -> float:
        if name not in res_df.columns:
            return 0.0
        return float(pd.to_numeric(res_df[name], errors="coerce").fillna(0.0).sum())

    total_gross_turnover = _sum_column("gross_turnover")
    total_one_way_turnover = _sum_column("one_way_turnover")
    total_traded_notional = _sum_column("traded_notional")
    total_commission = _sum_column("commission_cost")
    total_slippage = _sum_column("slippage_cost")
    total_regulatory = _sum_column("regulatory_cost")
    total_transaction_cost = _sum_column("transaction_cost")
    total_borrow_cost = _sum_column("borrow_cost")
    total_cost = total_transaction_cost + total_borrow_cost

    return {
        "final_equity": round(final, 2),
        "initial_capital": round(initial_capital, 2),
        "total_return_pct": round(total_return, 2),
        "cagr_pct": round(cagr, 2),
        "sharpe": round(sharpe, 2),
        "calmar": round(calmar, 2),
        "profit_factor": round(profit_factor, 2) if profit_factor is not None else None,
        "max_dd_pct": round(max_dd, 2),
        "max_dd_start": str(peak_date)[:10],
        "max_dd_trough": str(trough_date)[:10],
        "max_dd_recovery": str(recovery_date)[:10] if recovery_date is not None else None,
        "max_dd_fall_days": int(dd_fall_days),
        "max_dd_recovery_days": int(dd_recovery_days) if dd_recovery_days is not None else None,
        "win_rate_pct": round(win_rate, 1),
        "holding_win_rate_pct": round(holding_win_rate, 1) if holding_win_rate is not None else None,
        "holding_period_days": hold_days,
        "total_days": len(equity),
        "total_gross_turnover": round(total_gross_turnover, 4),
        "total_one_way_turnover": round(total_one_way_turnover, 4),
        "annualized_gross_turnover": round(total_gross_turnover / n_years, 4),
        "annualized_one_way_turnover": round(total_one_way_turnover / n_years, 4),
        "total_traded_notional": round(total_traded_notional, 2),
        "total_commission": round(total_commission, 2),
        "total_slippage": round(total_slippage, 2),
        "total_regulatory_fees": round(total_regulatory, 2),
        "total_transaction_cost": round(total_transaction_cost, 2),
        "total_borrow_cost": round(total_borrow_cost, 2),
        "total_cost": round(total_cost, 2),
        "total_cost_pct_initial": round(total_cost / initial_capital * 100, 4),
    }


# ── Benchmark (QQQ) ───────────────────────────────────────────────────────────

def compute_benchmark_relative_metrics(
    res_df: pd.DataFrame,
    benchmark_curve: List[Dict],
    periods_per_year: int = 252,
) -> Dict:
    """Compare strategy equity with a benchmark on their shared sessions.

    Alpha uses a zero risk-free rate. Capture ratios use arithmetic mean daily
    returns on benchmark up/down sessions. Rolling beat rates use every
    available overlapping 1Y/3Y window.
    """
    if res_df.empty or "equity" not in res_df.columns or not benchmark_curve:
        return {}

    strategy = pd.Series(
        pd.to_numeric(res_df["equity"], errors="coerce").to_numpy(),
        index=pd.to_datetime(res_df.index, errors="coerce"),
        name="strategy",
    )
    benchmark_frame = pd.DataFrame(benchmark_curve)
    if not {"date", "value"}.issubset(benchmark_frame.columns):
        return {}
    benchmark = pd.Series(
        pd.to_numeric(benchmark_frame["value"], errors="coerce").to_numpy(),
        index=pd.to_datetime(benchmark_frame["date"], errors="coerce"),
        name="benchmark",
    )

    # Vendor timestamps can be tz-aware. Daily comparisons only need dates.
    strategy.index = pd.DatetimeIndex(strategy.index).tz_localize(None).normalize()
    benchmark.index = pd.DatetimeIndex(benchmark.index).tz_localize(None).normalize()
    aligned = pd.concat([strategy, benchmark], axis=1, join="inner")
    aligned = aligned[~aligned.index.duplicated(keep="last")].sort_index().dropna()
    aligned = aligned[(aligned["strategy"] > 0) & (aligned["benchmark"] > 0)]
    if len(aligned) < 3:
        return {}

    returns = aligned.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan).dropna()
    if len(returns) < 2:
        return {}

    strategy_ret = returns["strategy"]
    benchmark_ret = returns["benchmark"]
    benchmark_var = float(benchmark_ret.var(ddof=1))
    beta = (
        float(strategy_ret.cov(benchmark_ret) / benchmark_var)
        if benchmark_var > 1e-16 else None
    )
    alpha = (
        float((strategy_ret.mean() - beta * benchmark_ret.mean()) * periods_per_year * 100)
        if beta is not None else None
    )
    active_ret = strategy_ret - benchmark_ret
    active_std = float(active_ret.std(ddof=1))
    tracking_error = active_std * np.sqrt(periods_per_year) * 100
    information_ratio = (
        float(active_ret.mean() / active_std * np.sqrt(periods_per_year))
        if active_std > 1e-16 else None
    )
    correlation = float(strategy_ret.corr(benchmark_ret))

    up_mask = benchmark_ret > 0
    down_mask = benchmark_ret < 0
    up_benchmark_mean = float(benchmark_ret[up_mask].mean()) if up_mask.any() else 0.0
    down_benchmark_mean = float(benchmark_ret[down_mask].mean()) if down_mask.any() else 0.0
    upside_capture = (
        float(strategy_ret[up_mask].mean() / up_benchmark_mean * 100)
        if up_mask.any() and abs(up_benchmark_mean) > 1e-16 else None
    )
    downside_capture = (
        float(strategy_ret[down_mask].mean() / down_benchmark_mean * 100)
        if down_mask.any() and abs(down_benchmark_mean) > 1e-16 else None
    )

    n_years = max((len(aligned) - 1) / periods_per_year, 1.0 / periods_per_year)
    strategy_cagr = ((aligned["strategy"].iloc[-1] / aligned["strategy"].iloc[0]) ** (1 / n_years) - 1) * 100
    benchmark_cagr = ((aligned["benchmark"].iloc[-1] / aligned["benchmark"].iloc[0]) ** (1 / n_years) - 1) * 100

    def _rolling_stats(window: int) -> Tuple[Optional[float], int, Optional[float]]:
        rolling = aligned.pct_change(window, fill_method=None).dropna()
        if rolling.empty:
            return None, 0, None
        excess = rolling["strategy"] - rolling["benchmark"]
        return (
            float((excess > 0).mean() * 100),
            int(len(excess)),
            float(excess.iloc[-1] * 100),
        )

    rolling_1y, rolling_1y_windows, latest_1y_excess = _rolling_stats(periods_per_year)
    rolling_3y, rolling_3y_windows, latest_3y_excess = _rolling_stats(periods_per_year * 3)

    def _round(value, digits=2):
        return round(float(value), digits) if value is not None and np.isfinite(value) else None

    return {
        "matched_days": int(len(aligned)),
        "benchmark_cagr_pct": _round(benchmark_cagr),
        "strategy_aligned_cagr_pct": _round(strategy_cagr),
        "excess_cagr_pct": _round(strategy_cagr - benchmark_cagr),
        "alpha_pct_annual": _round(alpha),
        "beta": _round(beta, 3),
        "tracking_error_pct": _round(tracking_error),
        "information_ratio": _round(information_ratio, 3),
        "correlation": _round(correlation, 3),
        "upside_capture_pct": _round(upside_capture),
        "downside_capture_pct": _round(downside_capture),
        "rolling_1y_win_rate_pct": _round(rolling_1y),
        "rolling_1y_windows": rolling_1y_windows,
        "latest_1y_excess_pct": _round(latest_1y_excess),
        "rolling_3y_win_rate_pct": _round(rolling_3y),
        "rolling_3y_windows": rolling_3y_windows,
        "latest_3y_excess_pct": _round(latest_3y_excess),
    }


def _compute_benchmark_curve(
    start_date: str,
    end_date: str,
    initial_capital: float,
    symbol: str = "QQQ",
) -> List[Dict]:
    """Load the benchmark ETF (store first, Alpaca fallback) and return a
    buy-and-hold equity curve normalised to `initial_capital` — i.e. it starts
    at exactly the same funding as the strategy curve, so the two lines are
    directly comparable on the chart."""
    try:
        # NOTE: check emptiness BEFORE .sort() — an empty store result has no
        # columns at all, so sorting on "timestamp" would raise.
        bench_df = store.load([symbol], start_date, end_date).collect()
        if bench_df.is_empty():
            # QQQ is an ETF, not an index constituent, so it is rarely in the
            # local store. Fall back to a live Alpaca fetch like the main
            # data path does, otherwise the benchmark line never renders.
            try:
                from backend.data.alpaca_adapter import AlpacaAdapter
                bench_df = AlpacaAdapter().fetch_history([symbol], start_date, end_date).collect()
            except Exception:
                return []
        if bench_df.is_empty():
            return []
        bench_df = bench_df.sort("timestamp")
        # cast to native float — Alpaca returns float32 which FastAPI cannot serialize
        closes = bench_df["close"].to_numpy().astype(float)
        if len(closes) < 2:
            return []
        equity = [float(initial_capital)]
        for i in range(1, len(closes)):
            r = (closes[i] / closes[i - 1]) - 1
            equity.append(equity[-1] * (1 + r))
        dates = bench_df["timestamp"].to_list()
        return [
            {"date": str(d)[:10], "value": round(float(v), 2)}
            for d, v in zip(dates, equity)
        ]
    except Exception:
        return []


# Backwards-compatible alias (older callers may import the SPY name).
_compute_spy_curve = _compute_benchmark_curve


def _load_regime_ema(
    start_date: str,
    end_date: str,
    warmup_days: int = 450,
    span_slow: int = 200,
    span_fast: int = 20,
) -> Optional[pd.DataFrame]:
    """Load the market-regime gauge (QQQ, SPY fallback) with a warm-up buffer and
    return a DataFrame indexed by timestamp with columns [close, ema_slow,
    ema_fast]. Used by the risk-management kill-switch: liquidate when close drops
    below the slow EMA (200), re-enter when it reclaims the fast EMA (20).

    The warm-up buffer is essential — a 200-period EMA needs ~200 prior bars to
    be meaningful, so we load `warmup_days` before start_date, compute the EMAs on
    the full series, and the caller slices to its simulation dates."""
    from datetime import timedelta
    try:
        sim_start = datetime.strptime(start_date[:10], "%Y-%m-%d")
        load_start = (sim_start - timedelta(days=warmup_days)).strftime("%Y-%m-%d")
    except Exception:
        load_start = start_date

    df = None
    for sym in ("QQQ", "SPY"):
        try:
            d = store.load([sym], load_start, end_date).collect()
            if d.is_empty():
                try:
                    from backend.data.alpaca_adapter import AlpacaAdapter
                    d = AlpacaAdapter().fetch_history([sym], load_start, end_date).collect()
                except Exception:
                    d = None
            if d is not None and not d.is_empty():
                df = d
                break
        except Exception:
            continue
    if df is None or df.is_empty():
        return None

    df = df.sort("timestamp")
    pdf = df.select(["timestamp", "close"]).to_pandas()
    ts = pd.to_datetime(pdf["timestamp"])
    if getattr(ts.dt, "tz", None) is not None:
        ts = ts.dt.tz_localize(None)   # match the tz-naive simulation dates
    pdf["timestamp"] = ts
    pdf = pdf.set_index("timestamp").sort_index()
    s = pdf["close"].astype(float)
    return pd.DataFrame({
        "close": s,
        "ema_slow": s.ewm(span=span_slow, adjust=False).mean(),
        "ema_fast": s.ewm(span=span_fast, adjust=False).mean(),
    })


# ── Engine ────────────────────────────────────────────────────────────────────

class BacktestEngine:
    def __init__(self, initial_capital: float = 100_000.0):
        self.initial_capital = initial_capital

    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------

    # Calendar-day warm-up buffer loaded BEFORE start_date so long-lookback
    # factors (252-trading-day momentum ≈ 365 calendar days) are already valid
    # on the very first simulation day. Without this the equity curve is a flat
    # line for the first ~year and no holdings are logged (the warm-up bug).
    WARMUP_DAYS = 450  # matches LiveStrategy's 450-day load → identical factor values

    def fetch_and_prepare_data(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """
        Load from local store. Falls back to Alpaca live-fetch if data is missing.

        Loads an extra WARMUP_DAYS of history before `start_date` so momentum /
        long-lookback factors are warmed up, computes factors over the full
        window, then trims the returned frame to timestamps >= start_date.
        """
        from datetime import timedelta
        try:
            sim_start = datetime.strptime(start_date[:10], "%Y-%m-%d")
            load_start = (sim_start - timedelta(days=self.WARMUP_DAYS)).strftime("%Y-%m-%d")
        except Exception:
            load_start = start_date

        # Try local store first (with warm-up buffer)
        lf = store.load(symbols, load_start, end_date)
        try:
            schema_df = lf.collect()
        except Exception:
            schema_df = pl.DataFrame()

        covered = set(schema_df["symbol"].cast(pl.Utf8).unique().to_list()) if not schema_df.is_empty() else set()
        missing = [s for s in symbols if s not in covered]

        # Fallback: fetch missing symbols from Alpaca
        if missing:
            print(f"[BacktestEngine] {len(missing)} symbols missing from store — fetching from Alpaca...")
            try:
                from backend.data.alpaca_adapter import AlpacaAdapter
                adapter = AlpacaAdapter()
                CHUNK = 50
                chunks = [missing[i : i + CHUNK] for i in range(0, len(missing), CHUNK)]
                frames = []
                for chunk in chunks:
                    try:
                        df = adapter.fetch_history(chunk, load_start, end_date).collect()
                        if not df.is_empty():
                            frames.append(df)
                    except Exception as e:
                        print(f"[BacktestEngine] Alpaca chunk failed: {e}")
                if frames:
                    fetched = pl.concat(frames)
                    schema_df = pl.concat([schema_df, fetched]) if not schema_df.is_empty() else fetched
            except Exception as e:
                print(f"[BacktestEngine] Alpaca fallback failed entirely: {e}")

        if schema_df.is_empty():
            print("[BacktestEngine] No data available.")
            return pd.DataFrame()

        print("[BacktestEngine] Computing factors...")
        factor_df = compute_all_factors(schema_df.lazy()).collect()
        factor_df = factor_df.sort(["symbol", "timestamp"])
        factor_df = factor_df.with_columns([
            (pl.col("close").shift(-1).over("symbol") / pl.col("close") - 1).alias("fwd_ret")
        ])

        pdf = factor_df.to_pandas()
        pdf["timestamp"] = pd.to_datetime(pdf["timestamp"])

        # Trim the warm-up buffer: factors are now computed, so the simulation
        # only sees dates from the requested start_date onward.
        try:
            sim_start_ts = pd.to_datetime(start_date[:10])
            pdf = pdf[pdf["timestamp"] >= sim_start_ts].reset_index(drop=True)
        except Exception:
            pass
        return pdf

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def run_simulation(
        self,
        data: pd.DataFrame,
        active_factors: List[str],
        leverage: float = 1.0,
        use_mwu: bool = False,
        use_vol_target: bool = True,
        vol_target_pct: float = 0.20,
        vol_window: int = 20,
        strategy_mode: str = "long_only",
        top_n: int = 30,
        neutralize_sector: bool = False,
        factor_weights: Optional[Dict[str, float]] = None,
        rebalance_days: int = 1,
        commission_bps: float = 0.0,
        slippage_bps: float = 5.0,
        regulatory_sell_bps: float = 0.0,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

        if data.empty:
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        # Map friendly name → internal column
        col_map = {name: meta["col"] for name, meta in RUNTIME_FACTOR_META.items()}
        descending_factors = {
            name for name, meta in RUNTIME_FACTOR_META.items() if meta["descending"]
        }

        valid_factors = [f for f in active_factors if col_map.get(f) in data.columns]
        factor_cols = [col_map[f] for f in valid_factors]

        if not valid_factors:
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        dates = sorted(data["timestamp"].unique())

        # Sector neutralization
        if neutralize_sector and "sector" not in data.columns:
            from backend.alpha.neutralization import add_sector_column
            add_sector_column(data, symbol_col="symbol")

        if neutralize_sector:
            for col in factor_cols:
                if col in data.columns:
                    data[col] = data.groupby(["timestamp", "sector"])[col].transform(lambda x: x - x.mean())

        # Pre-compute rank pivots
        _temp_cols = []
        rank_pivots = {}
        for f, col in zip(valid_factors, factor_cols):
            if col not in data.columns:
                continue
            ascending = f not in descending_factors
            rank_col = f"_rank_{col}"
            data[rank_col] = data.groupby("timestamp")[col].rank(ascending=ascending, pct=True)
            rank_pivots[f] = data.pivot_table(index="timestamp", columns="symbol", values=rank_col)
            _temp_cols.append(rank_col)
        data.drop(columns=_temp_cols, inplace=True, errors="ignore")

        fwd_ret_pivot = data.pivot_table(index="timestamp", columns="symbol", values="fwd_ret")

        factor_pivots = {}
        if use_mwu:
            for f, col in zip(valid_factors, factor_cols):
                if col in data.columns:
                    factor_pivots[f] = data.pivot_table(index="timestamp", columns="symbol", values=col)

        mwu = MWUEngine(valid_factors)
        current_weights = mwu.weights.copy()

        # Static factor weighting (e.g. v1.5S 70/30). Overrides the equal-weight
        # default when MWU is off. Only weights for active/valid factors are
        # kept, then re-normalised so they sum to 1.0.
        if factor_weights and not use_mwu:
            picked = {f: float(factor_weights[f]) for f in valid_factors if f in factor_weights}
            total_w = sum(picked.values())
            if total_w > 0:
                current_weights = {
                    f: (picked.get(f, 0.0) / total_w) for f in valid_factors
                }

        equity = [self.initial_capital]
        weights_history = []
        holdings_history = []
        daily_returns_buffer = []
        current_scalar = leverage
        rebalance_days = max(1, int(rebalance_days or 1))
        current_longs: List[str] = []
        current_shorts: List[str] = []
        position_weights: Dict[str, float] = {}
        cost_config = BacktestCostConfig(
            commission_bps=commission_bps,
            slippage_bps=slippage_bps,
            regulatory_sell_bps=regulatory_sell_bps,
        )
        cost_history = [_zero_trade_costs()]

        for i, date in enumerate(dates[:-1]):
            # P0-2 FIX: MWU uses dates[i-1] to avoid look-ahead bias
            if use_mwu and i >= 2:
                prev_date = dates[i - 1]
                day_ics = {}
                if prev_date in fwd_ret_pivot.index:
                    fwd_row = fwd_ret_pivot.loc[prev_date].dropna()
                    for f in valid_factors:
                        if f in factor_pivots and prev_date in factor_pivots[f].index:
                            fac_row = factor_pivots[f].loc[prev_date].dropna()
                            common = fac_row.index.intersection(fwd_row.index)
                            if len(common) > 5:
                                corr = fac_row[common].corr(fwd_row[common], method="spearman")
                                day_ics[f] = 0.0 if np.isnan(corr) else corr
                current_weights = mwu.update(date, day_ics)

            if date not in fwd_ret_pivot.index:
                daily_returns_buffer.append(0.0)
                equity.append(equity[-1])
                cost_history.append(_zero_trade_costs())
                continue

            do_rebalance = (i % rebalance_days == 0) or not current_longs
            if do_rebalance:
                score_series = None
                for f in valid_factors:
                    if f not in rank_pivots or date not in rank_pivots[f].index:
                        continue
                    rank_row = rank_pivots[f].loc[date].dropna()
                    weighted = rank_row * current_weights[f]
                    score_series = weighted if score_series is None else score_series.add(weighted, fill_value=0.0)

                if score_series is not None and not score_series.empty:
                    if strategy_mode == "long_short":
                        n_side = min(top_n, max(1, len(score_series) // 2))
                        current_longs = score_series.nlargest(n_side).index.tolist()
                        current_shorts = score_series.nsmallest(n_side).index.tolist()
                    else:
                        current_longs = score_series.nlargest(min(top_n, len(score_series))).index.tolist()
                        current_shorts = []

                    holdings_history.append({
                        "date": date,
                        "long": ", ".join(current_longs),
                        "short": ", ".join(current_shorts) if current_shorts else "",
                    })

            if not current_longs:
                daily_returns_buffer.append(0.0)
                equity.append(equity[-1])
                cost_history.append(_zero_trade_costs())
                continue

            # Volatility targeting
            if use_vol_target and len(daily_returns_buffer) >= vol_window:
                recent = np.array(daily_returns_buffer[-vol_window:])
                realized_vol = np.std(recent, ddof=1) * np.sqrt(252)
                current_scalar = min(leverage, vol_target_pct / realized_vol) if realized_vol > 0.001 else leverage
            else:
                current_scalar = leverage

            # The legacy path implicitly restores equal name weights every
            # session (its return is the arithmetic mean of held names).  Make
            # that hidden trading explicit so turnover and costs are honest.
            if strategy_mode == "long_short":
                long_weight = current_scalar / (2.0 * len(current_longs)) if current_longs else 0.0
                short_weight = -current_scalar / (2.0 * len(current_shorts)) if current_shorts else 0.0
                desired_weights = {
                    **{symbol: long_weight for symbol in current_longs},
                    **{symbol: short_weight for symbol in current_shorts},
                }
            else:
                name_weight = current_scalar / len(current_longs)
                desired_weights = {symbol: name_weight for symbol in current_longs}

            day_costs = _estimate_trade_costs(
                position_weights, desired_weights, equity[-1], cost_config
            )
            position_weights = desired_weights

            # P&L
            fwd_row = fwd_ret_pivot.loc[date]
            actual = 0.0
            for symbol, weight in position_weights.items():
                value = fwd_row.get(symbol, 0.0)
                value = 0.0 if value is None or pd.isna(value) else float(value)
                actual += weight * value
            raw = actual / current_scalar if current_scalar > 0 else 0.0
            daily_returns_buffer.append(raw)
            cost_fraction = day_costs["transaction_cost"] / equity[-1] if equity[-1] > 0 else 0.0
            net_return = actual - cost_fraction
            equity.append(equity[-1] * (1 + net_return))

            denominator = 1.0 + net_return
            if denominator > 0:
                position_weights = {
                    symbol: weight
                    * (1.0 + (0.0 if pd.isna(fwd_row.get(symbol, 0.0)) else float(fwd_row.get(symbol, 0.0))))
                    / denominator
                    for symbol, weight in position_weights.items()
                }
            cost_history.append(day_costs)

            weights_history.append({"date": date, **current_weights, "vol_scalar": current_scalar})

        res_df = pd.DataFrame({"date": dates, "equity": equity, **{
            key: [row[key] for row in cost_history]
            for key in _zero_trade_costs()
        }}).set_index("date")
        self.last_target_weights = dict(position_weights)
        self.last_cost_config = cost_config
        w_df = pd.DataFrame(weights_history).set_index("date") if weights_history else pd.DataFrame()
        h_df = pd.DataFrame(holdings_history).set_index("date") if holdings_history else pd.DataFrame()

        return res_df, w_df, h_df

    # ------------------------------------------------------------------
    # Full run
    # ------------------------------------------------------------------

    def run(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        active_factors: List[str],
        leverage: float = 1.0,
        use_mwu: bool = False,
        use_vol_target: bool = True,
        vol_target_pct: float = 0.20,
        vol_window: int = 20,
        strategy_mode: str = "long_only",
        top_n: int = 30,
        neutralize_sector: bool = False,
        factor_weights: Optional[Dict[str, float]] = None,
        rebalance_days: int = 1,
        commission_bps: float = 0.0,
        slippage_bps: float = 5.0,
        regulatory_sell_bps: float = 0.0,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        data = self.fetch_and_prepare_data(symbols, start_date, end_date)
        self.last_prepared_data = data
        return self.run_simulation(
            data, active_factors, leverage, use_mwu, use_vol_target,
            vol_target_pct, vol_window, strategy_mode, top_n, neutralize_sector,
            factor_weights=factor_weights,
            rebalance_days=rebalance_days,
            commission_bps=commission_bps,
            slippage_bps=slippage_bps,
            regulatory_sell_bps=regulatory_sell_bps,
        )

    # ------------------------------------------------------------------
    # Multi-sleeve strategy run (sleeves + leverage + winner-lock)
    #
    # PARITY-CRITICAL: this builds each rebalance day's target weights with the
    # SHARED backend.alpha.portfolio.combined_target_weights() — the very same
    # function the live executor calls. The backtest then just holds those
    # weights (letting them drift on real returns) until the next weekly
    # rebalance. So a Claude #1 backtest and a Claude #1 live account make
    # identical decisions on identical data. See SKILL.md. DO NOT fork this.
    # ------------------------------------------------------------------

    def run_strategy(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        sleeves: List[Dict],
        leverage: float = 1.0,
        top_n: int = 20,
        lock_rules: Optional[Dict[str, float]] = None,
        rebalance_days: int = 5,
        borrow_rate: float = 0.0,
        ema_kill_switch: bool = False,
        risk_management: Optional[Dict] = None,
        commission_bps: float = 0.0,
        slippage_bps: float = 5.0,
        regulatory_sell_bps: float = 0.0,
        signal_delay_days: int = 0,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Weekly-rebalanced, weight-based multi-sleeve simulation that mirrors
        live execution exactly. Returns (equity_df, weights_df, holdings_df)
        shaped like run() so the server consumes it unchanged.

        Risk-management overlay (checked DAILY, off the weekly cadence):
          • ema_kill_switch — when the market gauge (QQQ) closes below its 200-EMA,
            liquidate the whole book to cash; stay in cash until it reclaims its
            20-EMA, then re-enter IMMEDIATELY (not bound by the weekly schedule).
        """
        from backend.alpha.portfolio import combined_target_weights

        lock_rules = lock_rules or {}
        risk_management = risk_management or {}
        regime_mode = str(
            risk_management.get("regime_mode", "cash" if ema_kill_switch else "off")
        ).lower()
        if ema_kill_switch and regime_mode == "off":
            regime_mode = "cash"
        risk_off_leverage = float(risk_management.get("risk_off_leverage", min(leverage, 1.0)))
        volatility_throttle = bool(risk_management.get("volatility_throttle", False))
        vol_target_pct = float(risk_management.get("vol_target_pct", 0.25))
        vol_lookback = int(risk_management.get("vol_lookback", 20))
        liquidity_filter = bool(risk_management.get("liquidity_filter", False))
        min_price = float(risk_management.get("min_price", 5.0))
        min_avg_dollar_vol = float(risk_management.get("min_avg_dollar_vol", 20_000_000.0))
        crowding_shock_guard = bool(risk_management.get("crowding_shock_guard", False))
        max_avg_range_pct = float(risk_management.get("max_avg_range_pct", 0.12))
        sector_balance = bool(risk_management.get("sector_balance", False))
        data = self.fetch_and_prepare_data(symbols, start_date, end_date)
        self.last_prepared_data = data
        if data.empty:
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        rebalance_days = max(1, int(rebalance_days or 1))
        signal_delay_days = int(signal_delay_days or 0)
        if signal_delay_days < 0:
            raise ValueError("signal_delay_days must be non-negative")
        cost_config = BacktestCostConfig(
            commission_bps=commission_bps,
            slippage_bps=slippage_bps,
            regulatory_sell_bps=regulatory_sell_bps,
        )

        col_map = {n: m["col"] for n, m in RUNTIME_FACTOR_META.items()}
        all_factors: List[str] = []
        for sl in sleeves:
            for f in sl.get("factors", []):
                if f not in all_factors:
                    all_factors.append(f)

        dates = sorted(data["timestamp"].unique())
        close_pivot = data.pivot_table(index="timestamp", columns="symbol", values="close")
        fwd_pivot = data.pivot_table(index="timestamp", columns="symbol", values="fwd_ret")
        dollar_vol_pivot = data.assign(_dollar_vol=data["close"] * data["volume"]).pivot_table(
            index="timestamp", columns="symbol", values="_dollar_vol"
        )
        range_pivot = data.assign(_range_pct=(data["high"] - data["low"]) / data["close"]).pivot_table(
            index="timestamp", columns="symbol", values="_range_pct"
        )
        avg_dollar_vol = dollar_vol_pivot.rolling(20, min_periods=5).mean()
        avg_range_pct = range_pivot.rolling(20, min_periods=5).mean()
        market_ret = close_pivot.pct_change().mean(axis=1)
        market_vol = market_ret.rolling(
            vol_lookback, min_periods=max(5, min(vol_lookback, 10))
        ).std() * np.sqrt(252)
        fac_pivot = {
            f: data.pivot_table(index="timestamp", columns="symbol", values=col_map[f])
            for f in all_factors if col_map.get(f) in data.columns
        }

        # Market-regime gauge for the 200-EMA kill-switch (aligned to sim dates).
        regime = None
        use_regime_guard = regime_mode in {"cash", "throttle"}
        if use_regime_guard:
            r = _load_regime_ema(start_date, end_date)
            if r is not None:
                regime = r.reindex(pd.DatetimeIndex(dates), method="ffill")

        state: Dict[str, Dict] = {}     # winner-lock state, carried across rebalances
        target_w: Dict[str, float] = {}  # combined target weights (already ×leverage)
        in_market = True                # kill-switch regime state
        equity = [self.initial_capital]
        holdings_history = []
        cost_history = [{**_zero_trade_costs(), "borrow_cost": 0.0}]

        def _eligible_symbols(date):
            eligible = set(close_pivot.columns)
            if liquidity_filter and date in close_pivot.index:
                price_row = close_pivot.loc[date]
                dvol_row = avg_dollar_vol.loc[date] if date in avg_dollar_vol.index else pd.Series(dtype=float)
                eligible &= set(price_row[price_row >= min_price].index)
                eligible &= set(dvol_row[dvol_row >= min_avg_dollar_vol].index)
            if crowding_shock_guard and date in avg_range_pct.index:
                range_row = avg_range_pct.loc[date]
                eligible &= set(range_row[range_row <= max_avg_range_pct].index)
            return eligible

        def _scale_weights(weights: Dict[str, float], target_gross: float) -> Dict[str, float]:
            gross = sum(abs(float(w)) for w in weights.values())
            if gross <= 0 or target_gross <= 0:
                return {}
            scalar = target_gross / gross
            return {s: float(w) * scalar for s, w in weights.items()}

        def _desired_leverage(date) -> float:
            lev = float(leverage)
            if regime_mode == "throttle" and not in_market:
                lev = min(lev, risk_off_leverage)
            if volatility_throttle and date in market_vol.index:
                rv = market_vol.loc[date]
                if pd.notna(rv) and rv > vol_target_pct and rv > 0:
                    lev = min(lev, float(leverage) * vol_target_pct / float(rv))
            return max(0.0, lev)

        def _rebalance(signal_date, lev_override: Optional[float] = None):
            eligible = _eligible_symbols(signal_date)
            factor_values = {
                f: fac_pivot[f].loc[signal_date].dropna().loc[
                    lambda s: s.index.isin(eligible)
                ]
                for f in fac_pivot if signal_date in fac_pivot[f].index
            }
            price = (
                close_pivot.loc[signal_date].dropna()
                if signal_date in close_pivot.index else pd.Series(dtype=float)
            )
            price = price.loc[price.index.isin(eligible)]
            return combined_target_weights(
                sleeves, factor_values, price, state, top_n, lock_rules,
                float(leverage if lev_override is None else lev_override),
                sector_balance=sector_balance,
            )

        for i, date in enumerate(dates[:-1]):
            # ── Risk: 200-EMA kill-switch regime transitions (daily) ─────────
            prev_in_market = in_market
            if use_regime_guard and regime is not None:
                row = regime.iloc[i]
                mc, es, ef = row["close"], row["ema_slow"], row["ema_fast"]
                if in_market and pd.notna(es) and mc < es:
                    in_market = False
                elif (not in_market) and pd.notna(ef) and mc > ef:
                    in_market = True

            # ── Decide whether to (re)build target weights today ─────────────
            desired_lev = _desired_leverage(date)
            trade_requested = False
            next_target = dict(target_w)
            if regime_mode == "cash":
                if not in_market:
                    do_rebalance = False
                    if prev_in_market:  # just dropped below 200-EMA → go to cash
                        next_target = {}
                        trade_requested = True
                        holdings_history.append({
                            "date": date,
                            "signal_date": date,
                            "long": "(cash · 200EMA kill-switch)",
                            "short": "",
                        })
                elif not prev_in_market:
                    do_rebalance = True   # reclaimed 20-EMA → re-enter off-schedule
                else:
                    do_rebalance = (i % rebalance_days == 0)
            else:
                do_rebalance = (i % rebalance_days == 0) or (
                    regime_mode == "throttle" and prev_in_market != in_market
                )

            if do_rebalance:
                signal_index = i - signal_delay_days
                # Do not substitute a newer observation: remain in the prior
                # book (cash at inception) until the exact lagged session exists.
                if signal_index >= 0:
                    signal_date = dates[signal_index]
                    next_target, state = _rebalance(signal_date, desired_lev)
                    trade_requested = True
                    holdings_history.append({
                        "date": date,
                        "signal_date": signal_date,
                        "long": ", ".join(sorted(next_target.keys())),
                        "short": "",
                    })
            elif target_w and (volatility_throttle or regime_mode == "throttle"):
                next_target = _scale_weights(target_w, desired_lev)
                trade_requested = any(
                    abs(float(next_target.get(symbol, 0.0)) - float(target_w.get(symbol, 0.0))) > 1e-12
                    for symbol in set(next_target) | set(target_w)
                )

            day_costs = _zero_trade_costs()
            if trade_requested:
                day_costs = _estimate_trade_costs(
                    target_w, next_target, equity[-1], cost_config
                )
                target_w = next_target

            # ── Daily P&L on held weights, then drift the weights ────────────
            if date not in fwd_pivot.index or not target_w:
                equity.append(equity[-1] - day_costs["transaction_cost"])
                cost_history.append({**day_costs, "borrow_cost": 0.0})
                continue

            fwd_row = fwd_pivot.loc[date]
            port_ret = 0.0
            for s, w in target_w.items():
                r = fwd_row.get(s, 0.0)
                r = 0.0 if (r is None or pd.isna(r)) else float(r)
                port_ret += w * r
            gross = sum(target_w.values())
            daily_borrow = max(gross - 1.0, 0.0) * borrow_rate / 252.0
            borrow_cost = equity[-1] * daily_borrow
            transaction_fraction = (
                day_costs["transaction_cost"] / equity[-1] if equity[-1] > 0 else 0.0
            )
            net_return = port_ret - daily_borrow - transaction_fraction
            denom = 1.0 + net_return
            target_w = {
                s: (w * (1.0 + (0.0 if pd.isna(fwd_row.get(s, 0.0)) else float(fwd_row.get(s, 0.0)))) / denom)
                for s, w in target_w.items()
            } if denom > 0 else target_w
            equity.append(equity[-1] * (1.0 + net_return))
            cost_history.append({**day_costs, "borrow_cost": float(borrow_cost)})

        self.last_target_weights = dict(target_w)
        self.last_cost_config = cost_config
        self.last_signal_delay_days = signal_delay_days
        audit_keys = list(_zero_trade_costs()) + ["borrow_cost"]
        res_df = pd.DataFrame({
            "equity": equity,
            **{key: [row[key] for row in cost_history] for key in audit_keys},
        }, index=pd.Index(dates, name="date"))
        h_df = pd.DataFrame(holdings_history).set_index("date") if holdings_history else pd.DataFrame()
        return res_df, pd.DataFrame(), h_df
