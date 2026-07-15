"""
Parity verification: backtest engine and live strategy must produce IDENTICAL
target weights for the same preset, same data, same winner-lock state.

Both call backend.alpha.portfolio.combined_target_weights, so the only thing
this test can catch is a divergence in how each side *builds the inputs*
(factor_values dict + price Series + state) before that shared call.

Run:  python scripts/verify_parity.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from backend.alpha.factors import (
    STRATEGY_PRESETS, RUNTIME_FACTOR_META, compute_all_factors,
)
from backend.alpha.portfolio import combined_target_weights
from backend.backtest.engine import BacktestEngine
from backend.data import store
from backend.data.constituents import SPY_QQQ_TICKERS

PRESET = "Claude #1"
START = "2024-01-01"
END = "2026-06-28"
# keep it quick: a subset of the universe is enough to prove parity of logic
SYMBOLS = SPY_QQQ_TICKERS

sp = STRATEGY_PRESETS[PRESET]
sleeves = sp["sleeves"]
leverage = float(sp["leverage"])
top_n = int(sp["top_n"])
lock_rules = sp.get("winner_lock", {})
col_map = {n: m["col"] for n, m in RUNTIME_FACTOR_META.items()}

all_factors = []
for sl in sleeves:
    for f in sl.get("factors", []):
        if f not in all_factors:
            all_factors.append(f)


def engine_weights_at_latest():
    """Replicate the engine's input construction at its LAST available date."""
    eng = BacktestEngine(initial_capital=10000)
    data = eng.fetch_and_prepare_data(SYMBOLS, START, END)
    if data.empty:
        raise SystemExit("engine: no data")
    dates = sorted(data["timestamp"].unique())
    last = dates[-1]
    close_pivot = data.pivot_table(index="timestamp", columns="symbol", values="close")
    fac_pivot = {
        f: data.pivot_table(index="timestamp", columns="symbol", values=col_map[f])
        for f in all_factors if col_map.get(f) in data.columns
    }
    factor_values = {f: fac_pivot[f].loc[last].dropna() for f in fac_pivot if last in fac_pivot[f].index}
    price = close_pivot.loc[last].dropna() if last in close_pivot.index else pd.Series(dtype=float)
    w, _ = combined_target_weights(sleeves, factor_values, price, {}, top_n, lock_rules, leverage)
    return last, w


def live_weights_at_latest():
    """Replicate LiveStrategy.calculate_targets input construction (store path),
    state empty so it matches a fresh backtest rebalance."""
    hist_pl = store.load(SYMBOLS, START, END).collect()
    factor_df = compute_all_factors(hist_pl.lazy()).collect().to_pandas()
    factor_df["timestamp"] = pd.to_datetime(factor_df["timestamp"])
    hist = hist_pl.to_pandas()
    hist["timestamp"] = pd.to_datetime(hist["timestamp"])
    closes = hist.pivot(index="timestamp", columns="symbol", values="close")
    latest_date = factor_df["timestamp"].max()
    latest_data = factor_df[factor_df["timestamp"] == latest_date].set_index("symbol")
    factor_values = {}
    for f in all_factors:
        col = col_map.get(f)
        if col and col in latest_data.columns:
            factor_values[f] = latest_data[col].reindex(closes.columns).dropna()
    price = closes.iloc[-1].dropna()
    w, _ = combined_target_weights(sleeves, factor_values, price, {}, top_n, lock_rules, leverage)
    return latest_date, w


def main():
    d_bt, w_bt = engine_weights_at_latest()
    d_lv, w_lv = live_weights_at_latest()
    print(f"engine as-of {str(d_bt)[:10]} | {len(w_bt)} names | gross {sum(w_bt.values()):.4f}")
    print(f"live   as-of {str(d_lv)[:10]} | {len(w_lv)} names | gross {sum(w_lv.values()):.4f}")

    keys = sorted(set(w_bt) | set(w_lv))
    max_diff = 0.0
    mismatches = []
    for k in keys:
        a, b = w_bt.get(k, 0.0), w_lv.get(k, 0.0)
        d = abs(a - b)
        max_diff = max(max_diff, d)
        if d > 1e-9:
            mismatches.append((k, a, b, d))

    print(f"\nmax weight diff: {max_diff:.2e}  | mismatched names: {len(mismatches)}")
    for k, a, b, d in mismatches[:20]:
        print(f"  {k:6s} engine={a:.6f} live={b:.6f} diff={d:.2e}")

    if max_diff < 1e-9 and set(w_bt) == set(w_lv):
        print("\nPARITY OK: backtest and live produce identical target weights.")
        return 0
    print("\nPARITY FAIL: weights differ.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
