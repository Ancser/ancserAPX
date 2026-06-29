"""
Verify the chosen preset with the REAL engine.run_strategy (parity check) and
produce attribution: per-year returns, top contributing stocks, and the
windows where the strategy was unstable.
"""
import sys, io
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)

import numpy as np
import pandas as pd
import importlib.util

spec = importlib.util.spec_from_file_location("ps", str(Path(__file__).parent / "preset_search.py"))
ps = importlib.util.module_from_spec(spec); spec.loader.exec_module(ps)

from backend.backtest.engine import BacktestEngine
from backend.data.constituents import SPY_QQQ_TICKERS

# ── chosen preset ─────────────────────────────────────────────────────────────
SLEEVES = [
    {"name": "Core", "alloc": 0.70, "factors": ["Momentum"], "weights": {"Momentum": 1.0}, "winner_lock": False},
    {"name": "Defensive", "alloc": 0.30, "factors": ["Volatility", "Reversion"],
     "weights": {"Volatility": 0.6, "Reversion": 0.4}, "winner_lock": False},
]
LEV, TOPN, REB = 1.5, 15, 5

# ── 1. REAL engine parity check ──────────────────────────────────────────────
print("=== REAL engine.run_strategy (full period) ===")
eng = BacktestEngine(initial_capital=10000)
res, _, _ = eng.run_strategy(SPY_QQQ_TICKERS, "2021-07-01", "2026-06-30",
                             sleeves=SLEEVES, leverage=LEV, top_n=TOPN,
                             lock_rules={}, rebalance_days=REB)
eq = res["equity"]
rets = eq.pct_change().dropna(); yrs = len(eq) / 252
cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1 / yrs) - 1
sharpe = rets.mean() / rets.std() * np.sqrt(252)
dd = (eq / eq.cummax() - 1).min()
print(f"engine: CAGR {cagr*100:.1f}%  Sharpe {sharpe:.2f}  MaxDD {dd*100:.1f}%  final ${eq.iloc[-1]:,.0f}")

# search-simulator equivalent
sim = ps.simulate(SLEEVES, LEV, TOPN, {}, REB, ps.ALL_DATES, track_contrib=True)
m = ps.metrics(sim["equity"])
print(f"search: CAGR {m['cagr']*100:.1f}%  Sharpe {m['sharpe']:.2f}  MaxDD {m['maxdd']*100:.1f}%")

# ── 2. Per-regime / rolling stability ────────────────────────────────────────
print("\n=== Per-window stability (instability = low/negative Sharpe) ===")
allw = {**ps.REGIME_WINDOWS, **ps.ROLL_WINDOWS}
rows = []
for wname, (a, b) in allw.items():
    ds = ps.dates_between(a, b)
    if len(ds) < 20:
        continue
    wm = ps.metrics(ps.simulate(SLEEVES, LEV, TOPN, {}, REB, ds)["equity"])
    rows.append((wname, wm["cagr"], wm["sharpe"], wm["maxdd"]))
for n, c, s, d in rows:
    flag = "  <-- UNSTABLE" if s < 0.5 else ""
    print(f"  {n:18s} CAGR {c*100:7.1f}%  Sharpe {s:5.2f}  MaxDD {d*100:7.1f}%{flag}")

# ── 3. Per-year contribution ─────────────────────────────────────────────────
print("\n=== Per-year additive return contribution (sum of w*r) ===")
for y in sorted(sim["year_ret"]):
    print(f"  {y}: {sim['year_ret'][y]*100:+7.1f}%")

# ── 4. Top contributing stocks (full period) ─────────────────────────────────
print("\n=== Top 25 contributing stocks (additive sum of weight*fwd_ret) ===")
contrib = pd.Series(sim["contrib"]).sort_values(ascending=False)
for s, v in contrib.head(25).items():
    print(f"  {s:6s} {v*100:+7.2f}%")
print("\n=== Worst 10 detractors ===")
for s, v in contrib.tail(10).items():
    print(f"  {s:6s} {v*100:+7.2f}%")

# ── 5. Contribution by year x top names ──────────────────────────────────────
print("\n=== Per-year top-5 contributors ===")
hold = sim["holdings"]   # list of (date, {sym:w})
# rebuild per-year per-symbol contribution
fwd = ps.fwd_pivot
peryear = {}
# reconstruct daily holdings by forward-filling rebalance weights
cur = {}
ridx = 0
rdates = [d for d, _ in hold]
for date in ps.ALL_DATES[:-1]:
    while ridx < len(rdates) and rdates[ridx] == date:
        cur = hold[ridx][1]; ridx += 1
    if date not in fwd.index or not cur:
        continue
    row = fwd.loc[date]; y = date.year
    d = peryear.setdefault(y, {})
    for s, w in cur.items():
        r = row.get(s, 0.0); r = 0.0 if pd.isna(r) else float(r)
        d[s] = d.get(s, 0.0) + w * r
for y in sorted(peryear):
    top = sorted(peryear[y].items(), key=lambda x: x[1], reverse=True)[:5]
    s = ", ".join(f"{k}({v*100:+.0f}%)" for k, v in top)
    print(f"  {y}: {s}")
