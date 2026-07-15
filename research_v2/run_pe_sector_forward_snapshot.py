"""Run one conservative forward diagnostic from the 2026-02-13 PE snapshot.

The study deliberately starts at the next common trading-session open.  It
compares the same globally selected low-positive-PE Top-N under two weighting
rules: equal name weights and equal gross budgets across represented sectors.
It cannot test the sector-median market-cap screen because the frozen snapshot
does not contain point-in-time shares or market cap.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import uuid

import numpy as np
import pandas as pd

from research_v2.fundamental_value import (
    PESectorStrategyConfig,
    build_pe_sector_balanced_portfolio,
)
from research_v2.safety import ensure_research_output_path, offline_context


ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent
DEFAULT_SNAPSHOT = WORKSPACE / "data_cache" / "graham_fundamentals.json"
DEFAULT_STORE = WORKSPACE / "data" / "store"
DEFAULT_OUTPUT = ROOT / "runs" / "20260213_pe_sector_forward_v2"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(payload).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _load_snapshot(path: Path, sector_map: dict[str, str]) -> tuple[pd.DataFrame, pd.Timestamp]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    captured_at = pd.Timestamp(payload["date"])
    # The legacy timestamp is timezone-naive.  We therefore do not trade on
    # its calendar day under any timezone interpretation.
    rows = []
    for symbol, fields in payload["data"].items():
        rows.append(
            {
                "symbol": str(symbol),
                "sector": sector_map.get(str(symbol), "Unknown"),
                "pe_ttm": pd.to_numeric(fields.get("trailingPE"), errors="coerce"),
                "market_cap": np.nan,
                "available_at": captured_at.tz_localize("UTC"),
            }
        )
    frame = pd.DataFrame(rows)
    frame = frame.loc[frame["sector"] != "Unknown"].reset_index(drop=True)
    return frame, captured_at


def _load_selected_prices(
    symbols: list[str],
    store: Path,
    captured_at: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp, pd.Timestamp, list[Path]]:
    opens: dict[str, pd.Series] = {}
    closes: dict[str, pd.Series] = {}
    paths: list[Path] = []
    for symbol in symbols:
        path = store / f"{symbol}.parquet"
        if not path.is_file():
            raise FileNotFoundError(f"missing selected price history: {path}")
        data = pd.read_parquet(path, columns=["timestamp", "open", "close"])
        data["timestamp"] = pd.to_datetime(data["timestamp"])
        data = data.drop_duplicates("timestamp", keep="last").set_index("timestamp").sort_index()
        opens[symbol] = pd.to_numeric(data["open"], errors="coerce")
        closes[symbol] = pd.to_numeric(data["close"], errors="coerce")
        paths.append(path)

    open_frame = pd.DataFrame(opens)
    close_frame = pd.DataFrame(closes)
    valid = (
        open_frame.notna().all(axis=1)
        & close_frame.notna().all(axis=1)
        & (open_frame > 0).all(axis=1)
        & (close_frame > 0).all(axis=1)
    )
    common = open_frame.index[valid & (open_frame.index.normalize() > captured_at.normalize())]
    if len(common) < 2:
        raise RuntimeError("fewer than two common sessions after the frozen snapshot")
    entry_at = pd.Timestamp(common.min())
    end_at = pd.Timestamp(common.max())
    common = common[(common >= entry_at) & (common <= end_at)]
    return open_frame.loc[common], close_frame.loc[common], entry_at, end_at, paths


def _net_buy_and_hold_curve(
    open_frame: pd.DataFrame,
    close_frame: pd.DataFrame,
    weights: pd.Series,
    *,
    one_way_cost_bps: float,
    captured_at: pd.Timestamp,
) -> pd.Series:
    entry_prices = open_frame.iloc[0]
    gross = close_frame.div(entry_prices).mul(weights, axis=1).sum(axis=1)
    cost = one_way_cost_bps / 10_000.0
    # Self-financing entry: a unit of initial wealth buys 1/(1+c) of assets.
    net = gross / (1.0 + cost)
    net.iloc[-1] *= 1.0 - cost
    initial_at = captured_at.normalize()
    return pd.concat([pd.Series([1.0], index=[initial_at]), net]).sort_index()


def _load_complete_benchmark_prices(
    symbols: list[str],
    store: Path,
    dates: pd.DatetimeIndex,
) -> tuple[pd.DataFrame, pd.DataFrame, list[Path]]:
    """Load names with complete, strictly positive bars on every study date."""

    opens: dict[str, pd.Series] = {}
    closes: dict[str, pd.Series] = {}
    paths: list[Path] = []
    for symbol in symbols:
        path = store / f"{symbol}.parquet"
        if not path.is_file():
            continue
        paths.append(path)
        data = pd.read_parquet(path, columns=["timestamp", "open", "close"])
        data["timestamp"] = pd.to_datetime(data["timestamp"])
        data = data.drop_duplicates("timestamp", keep="last").set_index("timestamp").sort_index()
        local_open = pd.to_numeric(data["open"], errors="coerce").reindex(dates)
        local_close = pd.to_numeric(data["close"], errors="coerce").reindex(dates)
        if (
            local_open.notna().all()
            and local_close.notna().all()
            and (local_open > 0).all()
            and (local_close > 0).all()
        ):
            opens[symbol] = local_open
            closes[symbol] = local_close
    if not opens:
        raise RuntimeError("no complete positive-PE benchmark price histories")
    return pd.DataFrame(opens), pd.DataFrame(closes), paths


def _metrics(curve: pd.Series) -> dict[str, float | int]:
    returns = curve.pct_change().dropna()
    total = float(curve.iloc[-1] - 1.0)
    periods = max(1, len(returns))
    annualized_return = float((1.0 + total) ** (252.0 / periods) - 1.0) if total > -1 else -1.0
    vol = float(returns.std(ddof=1) * np.sqrt(252)) if len(returns) > 1 else np.nan
    sharpe = float(returns.mean() / returns.std(ddof=1) * np.sqrt(252)) if len(returns) > 1 and returns.std(ddof=1) > 0 else np.nan
    drawdown = curve / curve.cummax() - 1.0
    return {
        "observations": int(len(returns)),
        "total_return": total,
        "annualized_return_short_sample": annualized_return,
        "annualized_volatility": vol,
        "sharpe_zero_rf_short_sample": sharpe,
        "max_drawdown": float(drawdown.min()),
    }


def _verify(output: Path) -> None:
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    success = json.loads((output / "_SUCCESS.json").read_text(encoding="utf-8"))
    if _sha256(output / "manifest.json") != success["manifest_sha256"]:
        raise AssertionError("manifest hash mismatch")
    if _canonical_sha256(manifest["output_sha256"]) != success["output_hashes_sha256"]:
        raise AssertionError("output aggregate mismatch")
    declared = set(manifest["output_sha256"])
    actual = {
        str(path.relative_to(output)).replace("\\", "/")
        for path in output.rglob("*")
        if path.is_file()
    }
    if actual != declared | {"manifest.json", "_SUCCESS.json"}:
        raise AssertionError("published file set mismatch")
    for relative, expected in manifest["output_sha256"].items():
        if _sha256(output / relative) != expected:
            raise AssertionError(f"output hash mismatch: {relative}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--store", type=Path, default=DEFAULT_STORE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--top-n", type=int, default=22)
    parser.add_argument("--one-way-cost-bps", type=float, default=10.0)
    args = parser.parse_args(argv)
    if args.top_n < 1:
        raise ValueError("top-n must be positive")
    if not math.isfinite(args.one_way_cost_bps) or args.one_way_cost_bps < 0:
        raise ValueError("one-way-cost-bps must be finite and non-negative")

    output = ensure_research_output_path(args.output, research_root=ROOT)
    if output.exists():
        raise FileExistsError(f"immutable output exists: {output}")
    staging = ensure_research_output_path(
        output.parent / f".{output.name}.partial-{uuid.uuid4().hex}", research_root=ROOT
    )
    staging.mkdir(parents=True, exist_ok=False)
    started_at = _utc_now()
    try:
        with offline_context():
            from backend.alpha.neutralization import SECTOR_MAP

            snapshot, captured_at = _load_snapshot(args.snapshot, SECTOR_MAP)
            decision_at = captured_at.tz_localize("UTC")
            _, sector_budget = build_pe_sector_balanced_portfolio(
                snapshot,
                PESectorStrategyConfig(
                    top_n=args.top_n,
                    gross_target=1.0,
                    apply_sector_size_filter=False,
                ),
                decision_at=decision_at,
            )
            sector_budget = sector_budget.set_index("symbol")
            selected = (
                sector_budget.assign(_symbol=sector_budget.index)
                .sort_values(["pe_ttm", "_symbol"], kind="mergesort")
                .index.tolist()
            )
            equal_weight = pd.Series(1.0 / len(selected), index=selected, name="equal_weight")
            sector_weight = sector_budget["weight"].reindex(selected).rename("equal_sector_gross_budget")
            if not np.isclose(equal_weight.sum(), 1.0) or not np.isclose(sector_weight.sum(), 1.0):
                raise AssertionError("portfolio weights do not sum to one")
            open_frame, close_frame, entry_at, end_at, price_paths = _load_selected_prices(
                selected, args.store, captured_at
            )
            benchmark_candidates = snapshot.loc[
                np.isfinite(snapshot["pe_ttm"]) & (snapshot["pe_ttm"] > 0), "symbol"
            ].astype(str).tolist()
            benchmark_open, benchmark_close, benchmark_paths = _load_complete_benchmark_prices(
                benchmark_candidates, args.store, close_frame.index
            )
            benchmark_weights = pd.Series(
                1.0 / benchmark_open.shape[1], index=benchmark_open.columns
            )
            raw_curve = _net_buy_and_hold_curve(
                open_frame,
                close_frame,
                equal_weight,
                one_way_cost_bps=args.one_way_cost_bps,
                captured_at=captured_at,
            )
            sector_curve = _net_buy_and_hold_curve(
                open_frame,
                close_frame,
                sector_weight,
                one_way_cost_bps=args.one_way_cost_bps,
                captured_at=captured_at,
            )
            benchmark_curve = _net_buy_and_hold_curve(
                benchmark_open,
                benchmark_close,
                benchmark_weights,
                one_way_cost_bps=args.one_way_cost_bps,
                captured_at=captured_at,
            )

        equity = pd.DataFrame(
            {
                "raw_pe_equal_weight": raw_curve,
                "raw_pe_equal_sector_gross_budget": sector_curve,
                "positive_pe_universe_equal_weight_internal_benchmark": benchmark_curve,
            }
        )
        metric_rows = []
        for variant in equity.columns:
            metric_rows.append({"variant": variant, **_metrics(equity[variant].dropna())})
        metrics = pd.DataFrame(metric_rows)
        holdings = sector_budget.reset_index()[["symbol", "sector", "pe_ttm"]]
        holdings["raw_pe_equal_weight"] = holdings["symbol"].map(equal_weight)
        holdings["raw_pe_equal_sector_gross_budget"] = holdings["symbol"].map(sector_weight)
        holdings["entry_open"] = holdings["symbol"].map(open_frame.iloc[0])
        holdings["end_close"] = holdings["symbol"].map(close_frame.iloc[-1])
        holdings["price_return_ex_dividends"] = holdings["end_close"] / holdings["entry_open"] - 1.0
        sector_exposure = (
            holdings.groupby("sector")[["raw_pe_equal_weight", "raw_pe_equal_sector_gross_budget"]]
            .sum()
            .reset_index()
        )
        cost_rows = []
        for cost_bps in (0.0, 5.0, 10.0, 20.0):
            for variant, weights in (
                ("raw_pe_equal_weight", equal_weight),
                ("raw_pe_equal_sector_gross_budget", sector_weight),
            ):
                curve = _net_buy_and_hold_curve(
                    open_frame,
                    close_frame,
                    weights,
                    one_way_cost_bps=cost_bps,
                    captured_at=captured_at,
                )
                cost_rows.append(
                    {
                        "one_way_cost_bps": cost_bps,
                        "variant": variant,
                        "total_return": float(curve.iloc[-1] - 1.0),
                    }
                )
        cost_sensitivity = pd.DataFrame(cost_rows)

        source_paths = [
            Path(__file__).resolve(),
            ROOT / "fundamental_value.py",
            ROOT / "safety.py",
            WORKSPACE / "backend" / "alpha" / "neutralization.py",
            args.snapshot.resolve(),
            *price_paths,
            *benchmark_paths,
        ]
        # A selected name is also in the benchmark; de-duplicate before hashing.
        source_paths = list(dict.fromkeys(path.resolve() for path in source_paths))
        source_hashes = {
            str(path.resolve().relative_to(WORKSPACE)).replace("\\", "/"): _sha256(path)
            for path in source_paths
        }
        summary = {
            "study": "pe_snapshot_one_shot_forward_v2",
            "supersedes": "20260213_pe_sector_forward_v1",
            "supersession_reason": (
                "adds complete-positive-PE internal benchmark, cost sensitivity, and "
                "self-financing transaction-cost accounting"
            ),
            "started_at_utc": started_at,
            "completed_at_utc": _utc_now(),
            "snapshot_captured_at_naive": str(captured_at),
            "execution_rule": "first common trading-session open strictly after snapshot calendar day",
            "entry_at": str(entry_at),
            "end_at": str(end_at),
            "top_n": args.top_n,
            "selected_symbols": selected,
            "represented_sectors": int(holdings["sector"].nunique()),
            "internal_benchmark_names": int(benchmark_open.shape[1]),
            "one_way_cost_bps": args.one_way_cost_bps,
            "historical_backtest": False,
            "one_shot_forward_diagnostic": True,
            "production_mutations": [],
            "market_cap_filter_tested": False,
            "market_cap_filter_reason": "frozen 2026-02-13 snapshot has no point-in-time market cap or shares",
            "limitations": [
                "one signal date only",
                "current/static universe and sector map may introduce survivorship/classification bias",
                "local adjusted OHLCV was not independently certified as a total-return index",
                "legacy snapshot timestamp has no timezone",
                "snapshot has no vendor, query, or filing-availability provenance",
                "internal equal-weight opportunity-set benchmark is not a market benchmark",
                "constant 10 bps each side; no symbol-specific market impact",
            ],
        }
        holdings.to_csv(staging / "holdings.csv", index=False)
        sector_exposure.to_csv(staging / "sector_exposure.csv", index=False)
        equity.rename_axis("timestamp").to_csv(staging / "equity.csv")
        metrics.to_csv(staging / "metrics.csv", index=False)
        cost_sensitivity.to_csv(staging / "cost_sensitivity.csv", index=False)
        _write_json(staging / "summary.json", summary)
        lines = [
            "# PE snapshot one-shot forward diagnostic",
            "",
            "This is not a recurring historical backtest. It uses one frozen PE snapshot, "
            "enters on the next common session open, and holds through the last common close.",
            "The selected symbols are identical under both variants; only weights differ.",
            "",
            f"- Snapshot: {captured_at}; entry: {entry_at}; end: {end_at}.",
            f"- Top-N: {args.top_n}; represented sectors: {holdings['sector'].nunique()}.",
            f"- Internal complete positive-PE benchmark names: {benchmark_open.shape[1]}; "
            "this is not SPY or another market benchmark.",
            f"- Transaction-cost assumption: {args.one_way_cost_bps:.1f} bps on entry and exit.",
            "- The sector-median market-cap screen is not tested because the snapshot has no PIT market cap.",
            "",
            "## Results",
            "",
            "| Variant | Total return | Ann. return (short sample) | Ann. vol | Sharpe (0% rf) | MaxDD |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for row in metrics.itertuples(index=False):
            lines.append(
                f"| {row.variant} | {row.total_return:.2%} | "
                f"{row.annualized_return_short_sample:.2%} | {row.annualized_volatility:.2%} | "
                f"{row.sharpe_zero_rf_short_sample:.2f} | {row.max_drawdown:.2%} |"
            )
        lines.extend(
            [
                "",
                "Do not generalize this single favorable or unfavorable path. A defensible weekly "
                "test still requires a multi-date point-in-time fundamentals panel.",
                "",
            ]
        )
        (staging / "report.md").write_text("\n".join(lines), encoding="utf-8")

        if source_hashes != {
            str(path.resolve().relative_to(WORKSPACE)).replace("\\", "/"): _sha256(path)
            for path in source_paths
        }:
            raise AssertionError("source inputs changed during study")
        output_hashes = {
            str(path.relative_to(staging)).replace("\\", "/"): _sha256(path)
            for path in sorted(staging.rglob("*"))
            if path.is_file()
        }
        manifest = {**summary, "source_sha256": source_hashes, "output_sha256": output_hashes}
        _write_json(staging / "manifest.json", manifest)
        _write_json(
            staging / "_SUCCESS.json",
            {
                "completed_at_utc": summary["completed_at_utc"],
                "immutable_by_runner": True,
                "manifest_sha256": _sha256(staging / "manifest.json"),
                "output_hashes_sha256": _canonical_sha256(output_hashes),
                "historical_backtest": False,
            },
        )
        _verify(staging)
        os.replace(staging, output)
        _verify(output)
        print(json.dumps({"event": "pe_sector_forward_completed", "output": str(output)}))
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
