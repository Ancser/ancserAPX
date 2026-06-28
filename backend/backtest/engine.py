"""
BacktestEngine — reads from local Parquet store first, falls back to Alpaca live fetch.
Ported from ancserFX with data-source abstraction replaced by local store.
"""

import polars as pl
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from datetime import datetime

from backend.data import store
from backend.alpha.factors import compute_all_factors, FACTOR_META
from backend.alpha.mwu import MWUEngine


# ── Metrics helper ────────────────────────────────────────────────────────────

def compute_metrics(res_df: pd.DataFrame, initial_capital: float) -> Dict:
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
    calmar = cagr / abs(max_dd) if abs(max_dd) > 0.01 else 0.0
    win_rate = float((returns > 0).mean() * 100)
    total_return = (final / initial_capital - 1) * 100

    return {
        "final_equity": round(final, 2),
        "initial_capital": round(initial_capital, 2),
        "total_return_pct": round(total_return, 2),
        "cagr_pct": round(cagr, 2),
        "sharpe": round(sharpe, 2),
        "calmar": round(calmar, 2),
        "max_dd_pct": round(max_dd, 2),
        "win_rate_pct": round(win_rate, 1),
        "total_days": len(equity),
    }


# ── Benchmark (QQQ) ───────────────────────────────────────────────────────────

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


# ── Engine ────────────────────────────────────────────────────────────────────

class BacktestEngine:
    def __init__(self, initial_capital: float = 100_000.0):
        self.initial_capital = initial_capital

    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------

    def fetch_and_prepare_data(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """
        Load from local store. Falls back to Alpaca live-fetch if data is missing.
        """
        # Try local store first
        lf = store.load(symbols, start_date, end_date)
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
                        df = adapter.fetch_history(chunk, start_date, end_date).collect()
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
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

        if data.empty:
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        # Map friendly name → internal column
        col_map = {name: meta["col"] for name, meta in FACTOR_META.items()}
        descending_factors = {name for name, meta in FACTOR_META.items() if meta["descending"]}

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
                continue

            score_series = None
            for f in valid_factors:
                if f not in rank_pivots or date not in rank_pivots[f].index:
                    continue
                rank_row = rank_pivots[f].loc[date].dropna()
                weighted = rank_row * current_weights[f]
                score_series = weighted if score_series is None else score_series.add(weighted, fill_value=0.0)

            if score_series is None or score_series.empty:
                daily_returns_buffer.append(0.0)
                equity.append(equity[-1])
                continue

            if strategy_mode == "long_short":
                n_side = min(top_n, max(1, len(score_series) // 2))
                top_stocks = score_series.nlargest(n_side).index.tolist()
                bottom_n = score_series.nsmallest(n_side).index.tolist()
            else:
                top_stocks = score_series.nlargest(min(top_n, len(score_series))).index.tolist()
                bottom_n = []

            holdings_history.append({
                "date": date,
                "long": ", ".join(top_stocks),
                "short": ", ".join(bottom_n) if bottom_n else "",
            })

            # Volatility targeting
            if use_vol_target and len(daily_returns_buffer) >= vol_window:
                recent = np.array(daily_returns_buffer[-vol_window:])
                realized_vol = np.std(recent, ddof=1) * np.sqrt(252)
                current_scalar = min(leverage, vol_target_pct / realized_vol) if realized_vol > 0.001 else leverage
            else:
                current_scalar = leverage

            # P&L
            fwd_row = fwd_ret_pivot.loc[date]
            if strategy_mode == "long_short":
                long_ret = fwd_row.reindex(top_stocks).dropna().mean() if top_stocks else 0.0
                short_ret = fwd_row.reindex(bottom_n).dropna().mean() if bottom_n else 0.0
                raw = (long_ret - short_ret) / 2
            else:
                raw = fwd_row.reindex(top_stocks).dropna().mean() if top_stocks else 0.0

            raw = 0.0 if np.isnan(raw) else raw
            actual = raw * current_scalar
            daily_returns_buffer.append(raw)
            equity.append(equity[-1] * (1 + actual))

            weights_history.append({"date": date, **current_weights, "vol_scalar": current_scalar})

        res_df = pd.DataFrame({"date": dates, "equity": equity}).set_index("date")
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
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        data = self.fetch_and_prepare_data(symbols, start_date, end_date)
        self.last_prepared_data = data
        return self.run_simulation(
            data, active_factors, leverage, use_mwu, use_vol_target,
            vol_target_pct, vol_window, strategy_mode, top_n, neutralize_sector,
            factor_weights=factor_weights,
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
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Weekly-rebalanced, weight-based multi-sleeve simulation that mirrors
        live execution exactly. Returns (equity_df, weights_df, holdings_df)
        shaped like run() so the server consumes it unchanged."""
        from backend.alpha.portfolio import combined_target_weights

        lock_rules = lock_rules or {}
        data = self.fetch_and_prepare_data(symbols, start_date, end_date)
        self.last_prepared_data = data
        if data.empty:
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        col_map = {n: m["col"] for n, m in FACTOR_META.items()}
        all_factors: List[str] = []
        for sl in sleeves:
            for f in sl.get("factors", []):
                if f not in all_factors:
                    all_factors.append(f)

        dates = sorted(data["timestamp"].unique())
        close_pivot = data.pivot_table(index="timestamp", columns="symbol", values="close")
        fwd_pivot = data.pivot_table(index="timestamp", columns="symbol", values="fwd_ret")
        fac_pivot = {
            f: data.pivot_table(index="timestamp", columns="symbol", values=col_map[f])
            for f in all_factors if col_map.get(f) in data.columns
        }

        state: Dict[str, Dict] = {}     # winner-lock state, carried across rebalances
        target_w: Dict[str, float] = {}  # combined target weights (already ×leverage)
        equity = [self.initial_capital]
        holdings_history = []

        for i, date in enumerate(dates[:-1]):
            # ── Weekly rebalance: rebuild target weights via SHARED logic ────
            if i % rebalance_days == 0:
                factor_values = {
                    f: fac_pivot[f].loc[date].dropna()
                    for f in fac_pivot if date in fac_pivot[f].index
                }
                price = close_pivot.loc[date].dropna() if date in close_pivot.index else pd.Series(dtype=float)
                target_w, state = combined_target_weights(
                    sleeves, factor_values, price, state, top_n, lock_rules, leverage,
                )
                holdings_history.append({
                    "date": date,
                    "long": ", ".join(sorted(target_w.keys())),
                    "short": "",
                })

            # ── Daily P&L on held weights, then drift the weights ────────────
            if date not in fwd_pivot.index or not target_w:
                equity.append(equity[-1])
                continue
            fwd_row = fwd_pivot.loc[date]
            port_ret = 0.0
            for s, w in target_w.items():
                r = fwd_row.get(s, 0.0)
                r = 0.0 if (r is None or pd.isna(r)) else float(r)
                port_ret += w * r
            gross = sum(target_w.values())
            daily_borrow = max(gross - 1.0, 0.0) * borrow_rate / 252.0
            denom = 1.0 + port_ret
            target_w = {
                s: (w * (1.0 + (0.0 if pd.isna(fwd_row.get(s, 0.0)) else float(fwd_row.get(s, 0.0)))) / denom)
                for s, w in target_w.items()
            } if denom > 0 else target_w
            equity.append(equity[-1] * (1.0 + port_ret - daily_borrow))

        res_df = pd.DataFrame({"equity": equity}, index=pd.Index(dates, name="date"))
        h_df = pd.DataFrame(holdings_history).set_index("date") if holdings_history else pd.DataFrame()
        return res_df, pd.DataFrame(), h_df
