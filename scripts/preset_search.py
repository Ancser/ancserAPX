"""
Overnight preset search.

Goal: find a ROBUST strategy preset that
  1. stays stable across different market regimes (rolling-window backtests),
  2. keeps worst-window MaxDD below the leverage liquidation (爆倉) threshold,
  3. is built only from the factors/structures the app already supports.

Method (parity-honest): factors are computed ONCE with the production
compute_all_factors(); every candidate is simulated with the SAME
combined_target_weights() the engine and live executor use, just over
precomputed pivots and arbitrary date windows. So a winner here is directly
runnable as a STRATEGY_PRESET.

Staged search keeps it tractable & interpretable:
  Stage A  best CORE factor combo            (single sleeve, 1x, top20, weekly)
  Stage B  add structure (sleeve split + winner-lock) on the best cores, 1x
  Stage C  sweep leverage / top_n / rebalance on best structures, apply margin

Results are streamed to scripts/out/*.csv and a final scripts/out/summary.json.
Run:  python scripts/preset_search.py        (no timeout; takes a while)
"""
import sys, os, io, json, itertools, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if (getattr(sys.stdout, "encoding", "") or "").lower() not in ("utf-8", "utf8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)

import numpy as np
import pandas as pd

from backend.alpha.factors import FACTOR_META, compute_all_factors
from backend.alpha.portfolio import combined_target_weights
from backend.data import store
from backend.data.constituents import SPY_QQQ_TICKERS

OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(exist_ok=True)

COL = {n: m["col"] for n, m in FACTOR_META.items()}

# ── 1. Load + compute factors ONCE ───────────────────────────────────────────
DATA_START = "2020-07-01"     # broad coverage begins here
BT_START   = "2021-07-01"     # 252d momentum valid from here
BT_END     = "2026-06-30"

print(f"[{time.strftime('%H:%M:%S')}] loading store {DATA_START}..{BT_END} ...")
hist_pl = store.load(SPY_QQQ_TICKERS, DATA_START, BT_END).collect()
print(f"  rows={hist_pl.height} symbols={hist_pl['symbol'].n_unique()}")
print(f"[{time.strftime('%H:%M:%S')}] computing factors ...")
fdf = compute_all_factors(hist_pl.lazy()).collect().to_pandas()
fdf["timestamp"] = pd.to_datetime(fdf["timestamp"])

# fwd 1d return for P&L
fdf = fdf.sort_values(["symbol", "timestamp"])
fdf["fwd_ret"] = fdf.groupby("symbol")["close"].shift(-1) / fdf["close"] - 1.0

close_pivot = fdf.pivot_table(index="timestamp", columns="symbol", values="close")
fwd_pivot   = fdf.pivot_table(index="timestamp", columns="symbol", values="fwd_ret")
FAC_PIVOT = {}
for name, col in COL.items():
    if col in fdf.columns:
        FAC_PIVOT[name] = fdf.pivot_table(index="timestamp", columns="symbol", values=col)

ALL_DATES = list(close_pivot.index)
ALL_DATES = [d for d in ALL_DATES if d >= pd.Timestamp(BT_START)]
print(f"[{time.strftime('%H:%M:%S')}] backtest dates: {len(ALL_DATES)} "
      f"({ALL_DATES[0].date()}..{ALL_DATES[-1].date()})")

# ── 2. Parity-honest simulator over precomputed pivots ───────────────────────
def simulate(sleeves, leverage, top_n, lock_rules, rebalance_days, dates,
             borrow_rate=0.0, track_contrib=False):
    """Replicates engine.run_strategy daily loop using combined_target_weights.
    Returns dict with equity Series (+ optional per-symbol/per-year contribution)."""
    state, target_w = {}, {}
    eq = [1.0]
    idx = [dates[0]]
    contrib = {}              # symbol -> additive return contribution
    year_ret = {}             # year -> additive contribution
    holdings_log = []
    for i, date in enumerate(dates[:-1]):
        if i % rebalance_days == 0:
            fv = {f: FAC_PIVOT[f].loc[date].dropna() for f in FAC_PIVOT if date in FAC_PIVOT[f].index}
            price = close_pivot.loc[date].dropna() if date in close_pivot.index else pd.Series(dtype=float)
            target_w, state = combined_target_weights(
                sleeves, fv, price, state, top_n, lock_rules, leverage)
            if track_contrib:
                holdings_log.append((date, dict(target_w)))
        if date not in fwd_pivot.index or not target_w:
            eq.append(eq[-1]); idx.append(dates[i + 1]); continue
        fwd_row = fwd_pivot.loc[date]
        port_ret = 0.0
        for s, w in target_w.items():
            r = fwd_row.get(s, 0.0)
            r = 0.0 if (r is None or pd.isna(r)) else float(r)
            port_ret += w * r
            if track_contrib and w != 0.0 and r != 0.0:
                contrib[s] = contrib.get(s, 0.0) + w * r
                y = date.year
                year_ret[y] = year_ret.get(y, 0.0) + w * r
        gross = sum(target_w.values())
        daily_borrow = max(gross - 1.0, 0.0) * borrow_rate / 252.0
        denom = 1.0 + port_ret
        if denom > 0:
            target_w = {s: w * (1.0 + (0.0 if pd.isna(fwd_row.get(s, 0.0)) else float(fwd_row.get(s, 0.0)))) / denom
                        for s, w in target_w.items()}
        eq.append(eq[-1] * (1.0 + port_ret - daily_borrow))
        idx.append(dates[i + 1])
    equity = pd.Series(eq, index=pd.Index(idx, name="date"))
    out = {"equity": equity}
    if track_contrib:
        out["contrib"] = contrib
        out["year_ret"] = year_ret
        out["holdings"] = holdings_log
    return out


def metrics(equity):
    if len(equity) < 5:
        return dict(cagr=0, sharpe=0, maxdd=0, vol=0, calmar=0, final=1.0)
    rets = equity.pct_change().dropna()
    yrs = len(equity) / 252.0
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / yrs) - 1 if equity.iloc[0] > 0 else 0
    vol = rets.std() * np.sqrt(252)
    sharpe = (rets.mean() / rets.std() * np.sqrt(252)) if rets.std() > 0 else 0
    dd = (equity / equity.cummax() - 1).min()
    calmar = cagr / abs(dd) if dd < 0 else 0
    return dict(cagr=cagr, sharpe=sharpe, maxdd=dd, vol=vol, calmar=calmar,
                final=float(equity.iloc[-1]))


# ── 3. Windows: regimes + rolling ────────────────────────────────────────────
def dates_between(a, b):
    return [d for d in ALL_DATES if pd.Timestamp(a) <= d <= pd.Timestamp(b)]

REGIME_WINDOWS = {
    "2021H2_bull":   ("2021-07-01", "2021-12-31"),
    "2022_bear":     ("2022-01-01", "2022-12-31"),
    "2023_airally":  ("2023-01-01", "2023-12-31"),
    "2024_bull":     ("2024-01-01", "2024-12-31"),
    "2025_full":     ("2025-01-01", "2025-12-31"),
    "2026_ytd":      ("2026-01-01", "2026-06-30"),
}
# rolling 1y windows stepping 6 months
ROLL_WINDOWS = {}
start = pd.Timestamp("2021-07-01")
while start + pd.DateOffset(years=1) <= pd.Timestamp(BT_END):
    end = start + pd.DateOffset(years=1)
    ROLL_WINDOWS[f"roll_{start.date()}"] = (str(start.date()), str(end.date()))
    start += pd.DateOffset(months=6)

LIQ_MM = 0.25  # Reg-T maintenance margin
def liq_dd(L):
    return (1 - LIQ_MM * L) / (1 - LIQ_MM)

# ── 4. Candidate builders ────────────────────────────────────────────────────
def single(factors, weights):
    return [{"name": "Core", "alloc": 1.0, "factors": factors, "weights": weights, "winner_lock": False}]

def core_sat(core_f, core_w, sat_f, sat_w, core_alloc, sat_lock):
    return [
        {"name": "Core", "alloc": core_alloc, "factors": core_f, "weights": core_w, "winner_lock": False},
        {"name": "Satellite", "alloc": round(1 - core_alloc, 3), "factors": sat_f, "weights": sat_w, "winner_lock": sat_lock},
    ]

V15S = (["Momentum 12-1", "Pullback 5d"], {"Momentum 12-1": 0.70, "Pullback 5d": 0.30})
ACCEL = (["v1.5S Score", "Rank Acceleration"], {"v1.5S Score": 0.80, "Rank Acceleration": 0.20})

CORE_CANDIDATES = {
    "v15s_70_30":        single(*V15S),
    "accel_80_20":       single(*ACCEL),
    "mom121_only":       single(["Momentum 12-1"], {"Momentum 12-1": 1.0}),
    "tsmom_only":        single(["Momentum"], {"Momentum": 1.0}),
    "ema200_only":       single(["EMA200 Distance"], {"EMA200 Distance": 1.0}),
    "mom_ema":           single(["Momentum 12-1", "EMA200 Distance"], {"Momentum 12-1": 0.6, "EMA200 Distance": 0.4}),
    "mom_pull_ema":      single(["Momentum 12-1", "Pullback 5d", "EMA200 Distance"],
                                {"Momentum 12-1": 0.5, "Pullback 5d": 0.2, "EMA200 Distance": 0.3}),
    "unicorn":           single(["Unicorn Edge"], {"Unicorn Edge": 1.0}),
    "mom_unicorn":       single(["Momentum 12-1", "Unicorn Edge"], {"Momentum 12-1": 0.6, "Unicorn Edge": 0.4}),
    "defensive":         single(["Reversion", "Volatility", "Drift-Reversion"],
                                {"Reversion": 0.4, "Volatility": 0.3, "Drift-Reversion": 0.3}),
    "mom_lowvol":        single(["Momentum 12-1", "Volatility"], {"Momentum 12-1": 0.7, "Volatility": 0.3}),
    "mom_pull_lowvol":   single(["Momentum 12-1", "Pullback 5d", "Volatility"],
                                {"Momentum 12-1": 0.55, "Pullback 5d": 0.20, "Volatility": 0.25}),
}

def run_candidate(sleeves, leverage, top_n, lock_rules, reb):
    """Return per-window metrics + aggregate robustness stats."""
    res = {}
    # full period
    full = metrics(simulate(sleeves, leverage, top_n, lock_rules, reb, ALL_DATES)["equity"])
    res["full"] = full
    win_metrics = {}
    for wname, (a, b) in {**REGIME_WINDOWS, **ROLL_WINDOWS}.items():
        ds = dates_between(a, b)
        if len(ds) < 20:
            continue
        win_metrics[wname] = metrics(simulate(sleeves, leverage, top_n, lock_rules, reb, ds)["equity"])
    res["windows"] = win_metrics
    sharpes = [m["sharpe"] for m in win_metrics.values()]
    dds = [m["maxdd"] for m in win_metrics.values()]
    cagrs = [m["cagr"] for m in win_metrics.values()]
    res["agg"] = dict(
        min_sharpe=min(sharpes) if sharpes else 0,
        median_sharpe=float(np.median(sharpes)) if sharpes else 0,
        worst_dd=min(dds) if dds else 0,                 # most negative
        median_cagr=float(np.median(cagrs)) if cagrs else 0,
        full_sharpe=full["sharpe"], full_cagr=full["cagr"], full_dd=full["maxdd"],
    )
    return res


def _row(label, leverage, top_n, reb, lock, agg):
    return dict(label=label, leverage=leverage, top_n=top_n, reb=reb,
                lock=("on" if lock else "off"),
                full_cagr=round(agg["full_cagr"], 4), full_sharpe=round(agg["full_sharpe"], 3),
                full_dd=round(agg["full_dd"], 4), min_sharpe=round(agg["min_sharpe"], 3),
                median_sharpe=round(agg["median_sharpe"], 3), worst_dd=round(agg["worst_dd"], 4),
                median_cagr=round(agg["median_cagr"], 4))


def main():
    t0 = time.time()
    summary = {"liq_thresholds": {str(L): round(liq_dd(L), 4) for L in (1.0, 1.25, 1.5, 1.75, 2.0)}}
    print("\nLiquidation (爆倉) equity-DD thresholds @ Reg-T 25% maint margin:")
    for L, v in summary["liq_thresholds"].items():
        print(f"  {L}x -> MaxDD must stay above -{v*100:.1f}% (we require a buffer)")

    # ── STAGE A: best CORE factor combo (single sleeve, 1x, top20, weekly) ────
    print(f"\n[{time.strftime('%H:%M:%S')}] STAGE A: core factor combos (1x top20 weekly)")
    A_rows = []
    for name, sleeves in CORE_CANDIDATES.items():
        r = run_candidate(sleeves, 1.0, 20, {}, 5)
        row = _row(name, 1.0, 20, 5, False, r["agg"]); A_rows.append(row)
        print(f"  {name:18s} cagr={row['full_cagr']*100:6.1f}% sharpe={row['full_sharpe']:.2f} "
              f"dd={row['full_dd']*100:6.1f}% | minSharpe={row['min_sharpe']:.2f} worstDD={row['worst_dd']*100:6.1f}%")
    pd.DataFrame(A_rows).to_csv(OUT / "stageA_cores.csv", index=False)
    # rank cores by robustness: min_sharpe then median_sharpe
    A_sorted = sorted(A_rows, key=lambda x: (x["min_sharpe"], x["median_sharpe"]), reverse=True)
    top_cores = [r["label"] for r in A_sorted[:4]]
    summary["stageA_top_cores"] = top_cores
    print(f"  -> top cores: {top_cores}")

    # ── STAGE B: structure (sleeve split + winner-lock) on top cores, 1x ──────
    print(f"\n[{time.strftime('%H:%M:%S')}] STAGE B: structures on top cores (1x)")
    B_rows, B_structs = [], {}
    LOCK = {"profit_lock": 0.30, "max_weight": 0.15, "lock_rank": 10}
    for core_name in top_cores:
        core_sleeves = CORE_CANDIDATES[core_name]
        cf, cw = core_sleeves[0]["factors"], core_sleeves[0]["weights"]
        # B1: the core alone (already have, but re-list for comparison)
        cands = {f"{core_name}|single": core_sleeves}
        # B2/B3: core + accel satellite, lock on/off, splits 70/30 & 80/20
        for alloc in (0.70, 0.80):
            cands[f"{core_name}|sat_accel_{int(alloc*100)}_lock"] = core_sat(cf, cw, *ACCEL, alloc, True)
            cands[f"{core_name}|sat_accel_{int(alloc*100)}_nolock"] = core_sat(cf, cw, *ACCEL, alloc, False)
        # B4: core + low-vol defensive satellite (regime hedge), lock off
        DEF = (["Volatility", "Reversion"], {"Volatility": 0.6, "Reversion": 0.4})
        cands[f"{core_name}|sat_def_70"] = core_sat(cf, cw, *DEF, 0.70, False)
        for label, sl in cands.items():
            r = run_candidate(sl, 1.0, 20, LOCK, 5)
            row = _row(label, 1.0, 20, 5, "lock" in label, r["agg"]); B_rows.append(row)
            B_structs[label] = sl
            print(f"  {label:34s} cagr={row['full_cagr']*100:6.1f}% sharpe={row['full_sharpe']:.2f} "
                  f"dd={row['full_dd']*100:6.1f}% | minSharpe={row['min_sharpe']:.2f} worstDD={row['worst_dd']*100:6.1f}%")
    pd.DataFrame(B_rows).to_csv(OUT / "stageB_structures.csv", index=False)
    B_sorted = sorted(B_rows, key=lambda x: (x["min_sharpe"], x["median_sharpe"]), reverse=True)
    top_structs = [r["label"] for r in B_sorted[:5]]
    summary["stageB_top_structs"] = top_structs
    print(f"  -> top structures: {top_structs}")

    # ── STAGE C: leverage / top_n / rebalance sweep + margin constraint ───────
    print(f"\n[{time.strftime('%H:%M:%S')}] STAGE C: leverage/top_n/rebalance sweep (margin-constrained)")
    C_rows = []
    LEVS = [1.0, 1.25, 1.5, 1.75, 2.0]
    TOPNS = [15, 20, 30]
    REBS = [5, 10, 21]
    SAFETY = 0.80   # require worst-window MaxDD < SAFETY * liq threshold
    for label in top_structs:
        sl = B_structs[label]
        has_lock = "lock" in label and "nolock" not in label
        lock = {"profit_lock": 0.30, "max_weight": 0.15, "lock_rank": 10} if has_lock else {}
        for L, tn, rb in itertools.product(LEVS, TOPNS, REBS):
            r = run_candidate(sl, L, tn, lock, rb)
            agg = r["agg"]
            thr = liq_dd(L) * SAFETY
            safe = abs(agg["worst_dd"]) < thr
            row = _row(label, L, tn, rb, has_lock, agg)
            row["liq_thr"] = round(liq_dd(L), 4)
            row["safe_thr"] = round(thr, 4)
            row["margin_safe"] = safe
            C_rows.append(row)
        print(f"  swept {label}")
    Cdf = pd.DataFrame(C_rows)
    Cdf.to_csv(OUT / "stageC_sweep.csv", index=False)

    # ── WINNER selection ──────────────────────────────────────────────────────
    safe = Cdf[Cdf["margin_safe"]].copy()
    # robustness objective: maximize min_sharpe, then median_cagr, then full_sharpe
    safe = safe.sort_values(["min_sharpe", "median_cagr", "full_sharpe"], ascending=False)
    print(f"\n[{time.strftime('%H:%M:%S')}] margin-safe candidates: {len(safe)} / {len(Cdf)}")
    print("\nTOP 15 margin-safe by robustness (min rolling Sharpe):")
    cols = ["label", "leverage", "top_n", "reb", "lock", "full_cagr", "full_sharpe",
            "full_dd", "min_sharpe", "median_sharpe", "worst_dd", "median_cagr"]
    print(safe[cols].head(15).to_string(index=False))

    if len(safe):
        win = safe.iloc[0].to_dict()
        summary["winner"] = win
        print(f"\n*** WINNER: {win['label']}  {win['leverage']}x top{int(win['top_n'])} reb{int(win['reb'])} "
              f"lock={win['lock']} ***")
        print(f"    full: CAGR {win['full_cagr']*100:.1f}% Sharpe {win['full_sharpe']:.2f} DD {win['full_dd']*100:.1f}%")
        print(f"    robust: minSharpe {win['min_sharpe']:.2f} worstDD {win['worst_dd']*100:.1f}% "
              f"(liq -{win['liq_thr']*100:.1f}%, safe<-{win['safe_thr']*100:.1f}%)")

    with open(OUT / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n[{time.strftime('%H:%M:%S')}] done in {(time.time()-t0)/60:.1f} min. "
          f"Results in {OUT}")


if __name__ == "__main__":
    main()
