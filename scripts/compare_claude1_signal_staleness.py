"""Compare Claude #1 with current versus exactly five-session-stale signals.

This is a diagnostic, not a model-selection run.  Both variants execute on the
same dates, use the same locally available universe, and pay the same costs.
Only the signal observation date changes.  Outputs are JSON, rebalance-level
holdings-overlap CSV, and a concise Markdown report.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Dict, Iterable, Set

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.alpha.factors import STRATEGY_PRESETS
from backend.backtest.engine import BacktestEngine, compute_metrics
from backend.data import store
from backend.data.constituents import SPY_QQQ_TICKERS


def _symbols(value: object) -> Set[str]:
    if value is None:
        return set()
    return {
        item.strip()
        for item in str(value).split(",")
        if item.strip() and not item.strip().startswith("(")
    }


def _holdings_overlap(normal: pd.DataFrame, stale: pd.DataFrame) -> pd.DataFrame:
    if normal.empty or stale.empty:
        return pd.DataFrame()
    normal_map = {
        pd.Timestamp(date): _symbols(row.get("long"))
        for date, row in normal.iterrows()
    }
    rows = []
    for date, row in stale.iterrows():
        execution_date = pd.Timestamp(date)
        if execution_date not in normal_map:
            continue
        current = normal_map[execution_date]
        delayed = _symbols(row.get("long"))
        union = current | delayed
        shared = current & delayed
        rows.append(
            {
                "execution_date": str(execution_date.date()),
                "stale_signal_date": str(pd.Timestamp(row.get("signal_date")).date()),
                "normal_count": len(current),
                "stale_count": len(delayed),
                "shared_count": len(shared),
                "shared_fraction_of_normal": len(shared) / len(current) if current else 0.0,
                "jaccard": len(shared) / len(union) if union else 1.0,
            }
        )
    return pd.DataFrame(rows)


def _metric_subset(metrics: Dict) -> Dict:
    keys = (
        "final_equity",
        "total_return_pct",
        "cagr_pct",
        "sharpe",
        "max_dd_pct",
        "total_gross_turnover",
        "total_one_way_turnover",
        "annualized_gross_turnover",
        "total_commission",
        "total_slippage",
        "total_regulatory_fees",
        "total_transaction_cost",
        "total_cost_pct_initial",
        "total_days",
    )
    return {key: metrics.get(key) for key in keys}


def _pct(value: float) -> str:
    return f"{value:.2f}%"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2021-07-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--capital", type=float, default=100_000.0)
    parser.add_argument("--commission-bps", type=float, default=0.0)
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    parser.add_argument("--regulatory-sell-bps", type=float, default=0.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "docs" / "experiments" / "claude1_signal_staleness",
    )
    args = parser.parse_args()

    manifest = store.get_manifest()
    symbols = [symbol for symbol in SPY_QQQ_TICKERS if symbol in manifest]
    excluded = sorted(set(SPY_QQQ_TICKERS) - set(symbols))
    if not symbols:
        raise RuntimeError("No local SPY+QQQ constituent data is available")
    end = args.end or max(str(manifest[symbol]["last_date"]) for symbol in symbols)

    preset = STRATEGY_PRESETS["Claude #1"]
    engine = BacktestEngine(initial_capital=args.capital)
    prepared = engine.fetch_and_prepare_data(symbols, args.start, end)
    if prepared.empty:
        raise RuntimeError("No prepared factor data is available for the requested interval")

    # Incremental sync writes symbol files one by one.  Never let a partially
    # written latest cross-section become a synthetic rebalance date.
    raw_session_counts = prepared.groupby("timestamp")["symbol"].nunique().sort_index()
    expected_cross_section = float(raw_session_counts.tail(60).median())
    completeness_floor = max(1, int(expected_cross_section * 0.95))
    complete_sessions = raw_session_counts[raw_session_counts >= completeness_floor]
    if complete_sessions.empty:
        raise RuntimeError("No session meets the 95% cross-section completeness floor")
    raw_latest_session = pd.Timestamp(raw_session_counts.index[-1])
    evaluation_end = pd.Timestamp(complete_sessions.index[-1])
    prepared = prepared[prepared["timestamp"] <= evaluation_end].copy()

    def run(
        delay: int,
        *,
        commission_bps: float | None = None,
        slippage_bps: float | None = None,
        regulatory_sell_bps: float | None = None,
    ):
        local_engine = BacktestEngine(initial_capital=args.capital)
        local_engine.fetch_and_prepare_data = lambda *_a, **_kw: prepared.copy()
        return local_engine.run_strategy(
            symbols=symbols,
            start_date=args.start,
            end_date=end,
            sleeves=preset["sleeves"],
            leverage=float(preset["leverage"]),
            top_n=int(preset["top_n"]),
            lock_rules=preset.get("winner_lock", {}),
            rebalance_days=5,
            commission_bps=args.commission_bps if commission_bps is None else commission_bps,
            slippage_bps=args.slippage_bps if slippage_bps is None else slippage_bps,
            regulatory_sell_bps=(
                args.regulatory_sell_bps
                if regulatory_sell_bps is None else regulatory_sell_bps
            ),
            signal_delay_days=delay,
        )

    normal_result, _, normal_holdings = run(0)
    stale_result, _, stale_holdings = run(5)
    normal_metrics = compute_metrics(normal_result, args.capital, holding_period_days=5)
    stale_metrics = compute_metrics(stale_result, args.capital, holding_period_days=5)
    has_modeled_friction = any(
        value > 0
        for value in (args.commission_bps, args.slippage_bps, args.regulatory_sell_bps)
    )
    zero_cost_metrics = None
    if has_modeled_friction:
        zero_cost_result, _, _ = run(
            0, commission_bps=0.0, slippage_bps=0.0, regulatory_sell_bps=0.0
        )
        zero_cost_metrics = compute_metrics(
            zero_cost_result, args.capital, holding_period_days=5
        )
    overlap = _holdings_overlap(normal_holdings, stale_holdings)

    latest_counts = prepared.groupby("timestamp")["symbol"].nunique().sort_index()
    overlap_summary = {
        "matched_rebalances": int(len(overlap)),
        "mean_shared_names": round(float(overlap["shared_count"].mean()), 3) if len(overlap) else None,
        "mean_shared_fraction_pct": round(float(overlap["shared_fraction_of_normal"].mean() * 100), 2) if len(overlap) else None,
        "median_shared_fraction_pct": round(float(overlap["shared_fraction_of_normal"].median() * 100), 2) if len(overlap) else None,
        "mean_jaccard_pct": round(float(overlap["jaccard"].mean() * 100), 2) if len(overlap) else None,
        "minimum_shared_names": int(overlap["shared_count"].min()) if len(overlap) else None,
    }
    payload = {
        "experiment": "Claude #1 current signal versus exactly five trading-session stale signal",
        "generated_from_local_data": True,
        "requested_start": args.start,
        "requested_end": end,
        "actual_first_session": str(pd.Timestamp(prepared["timestamp"].min()).date()),
        "actual_last_session": str(pd.Timestamp(prepared["timestamp"].max()).date()),
        "raw_latest_session_before_completeness_filter": str(raw_latest_session.date()),
        "cross_section_completeness_floor": completeness_floor,
        "partial_latest_session_excluded": bool(raw_latest_session > evaluation_end),
        "universe_requested": len(SPY_QQQ_TICKERS),
        "universe_with_local_files": len(symbols),
        "excluded_missing_files": excluded,
        "latest_session_symbol_count": int(latest_counts.iloc[-1]),
        "assumptions": {
            "capital": args.capital,
            "leverage": preset["leverage"],
            "top_n": preset["top_n"],
            "rebalance_every_trading_sessions": 5,
            "stale_signal_delay_trading_sessions": 5,
            "commission_bps_one_way": args.commission_bps,
            "slippage_bps_one_way": args.slippage_bps,
            "regulatory_sell_bps": args.regulatory_sell_bps,
        },
        "normal": _metric_subset(normal_metrics),
        "stale_5_sessions": _metric_subset(stale_metrics),
        "normal_zero_modeled_friction": (
            _metric_subset(zero_cost_metrics) if zero_cost_metrics is not None else None
        ),
        "normal_cost_drag": (
            {
                "final_equity_difference": round(
                    float(normal_metrics["final_equity"])
                    - float(zero_cost_metrics["final_equity"]), 2
                ),
                "cagr_difference_pct_points": round(
                    float(normal_metrics["cagr_pct"])
                    - float(zero_cost_metrics["cagr_pct"]), 2
                ),
                "sharpe_difference": round(
                    float(normal_metrics["sharpe"])
                    - float(zero_cost_metrics["sharpe"]), 2
                ),
            }
            if zero_cost_metrics is not None else None
        ),
        "stale_minus_normal": {
            key: round(float(stale_metrics[key]) - float(normal_metrics[key]), 4)
            for key in ("final_equity", "cagr_pct", "sharpe", "max_dd_pct", "total_gross_turnover")
        },
        "holdings_overlap": overlap_summary,
        "limitations": [
            "Static present-day constituent list creates survivorship bias.",
            "Daily close-to-close bars cannot reproduce open-auction fills, quoted spreads, partial fills, or TAF per-share caps.",
            "The production backtest's 5-session cadence is not a calendar-Friday scheduler around market holidays.",
            "The stale variant isolates signal age only; it does not simulate missing symbols, corporate-action corruption, or order failures.",
            "Sessions below 95% of the trailing 60-session median symbol count are excluded as partial incremental-sync snapshots.",
        ],
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "results.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    overlap.to_csv(args.output_dir / "holdings_overlap.csv", index=False)

    n = payload["normal"]
    s = payload["stale_5_sessions"]
    zero = payload["normal_zero_modeled_friction"]
    drag = payload["normal_cost_drag"]
    report = f"""# Claude #1 signal-staleness diagnostic

Generated from the local store through {payload['actual_last_session']}. This
is a controlled sensitivity test: execution dates and cost assumptions are the
same, while the challenger always uses the factor snapshot from exactly five
trading sessions earlier.

| Variant | Final equity | CAGR | Sharpe | MaxDD | Gross turnover | Trading cost |
|---|---:|---:|---:|---:|---:|---:|
| Current signal | ${n['final_equity']:,.2f} | {_pct(n['cagr_pct'])} | {n['sharpe']:.2f} | {_pct(n['max_dd_pct'])} | {n['total_gross_turnover']:.2f}x | ${n['total_transaction_cost']:,.2f} |
| Signal delayed 5 sessions | ${s['final_equity']:,.2f} | {_pct(s['cagr_pct'])} | {s['sharpe']:.2f} | {_pct(s['max_dd_pct'])} | {s['total_gross_turnover']:.2f}x | ${s['total_transaction_cost']:,.2f} |

Average same-name overlap was {overlap_summary['mean_shared_fraction_pct']:.2f}%
of the normal Top-20 book ({overlap_summary['mean_shared_names']:.2f} names),
with mean Jaccard overlap {overlap_summary['mean_jaccard_pct']:.2f}% across
{overlap_summary['matched_rebalances']} matched rebalances.

Cost assumptions: {args.commission_bps:g} bps broker commission on buys/sells,
{args.slippage_bps:g} bps spread/slippage on buys/sells, and
{args.regulatory_sell_bps:g} bps blended regulatory fee on sells. Zero Alpaca
broker commission is not treated as zero execution cost.

{f"The same current-signal strategy with all modeled friction set to zero ended at ${zero['final_equity']:,.2f} with {zero['cagr_pct']:.2f}% CAGR. The default friction therefore reduced final equity by ${abs(drag['final_equity_difference']):,.2f} and CAGR by {abs(drag['cagr_difference_pct_points']):.2f} percentage points. Dollar fees and final-equity drag differ because foregone capital no longer compounds." if zero is not None else "No separate zero-friction sensitivity was needed because all configured trading-cost rates were already zero."}

## Limits

""" + "\n".join(f"- {item}" for item in payload["limitations"]) + "\n"
    (args.output_dir / "report.md").write_text(report, encoding="utf-8")

    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
