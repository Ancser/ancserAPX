"""Fetch or replay and test a PE/market-cap cross-section without touching daily.

This is a current-snapshot sanity study, not a historical backtest.  Yahoo
fields have no historical filing-availability trail, so the snapshot may be
used only at or after its recorded retrieval timestamps and for future shadow
tracking.  Historical performance requires a separate PIT fundamentals store.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import time
import uuid

import numpy as np
import pandas as pd

from research_v2.fundamental_value import (
    PESectorStrategyConfig,
    build_pe_sector_balanced_portfolio,
    joint_sector_size_residual,
    sector_relative_market_cap_filter,
    size_filter_sector_diagnostics,
    validate_fundamental_snapshot,
)


ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent
DEFAULT_OUTPUT = ROOT / "runs" / "20260714_pe_sector_current_v3"


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


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _finite(value: object) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _fetch_symbol(symbol: str, retries: int) -> dict[str, object]:
    import yfinance as yf

    error = None
    for attempt in range(retries + 1):
        try:
            info = yf.Ticker(symbol).get_info()
            return {
                "symbol": symbol,
                "pe_ttm": _finite(info.get("trailingPE")),
                "market_cap": _finite(info.get("marketCap")),
                "trailing_eps": _finite(info.get("trailingEps")),
                "shares_outstanding": _finite(info.get("sharesOutstanding")),
                "price": _finite(info.get("currentPrice") or info.get("regularMarketPrice")),
                "currency": info.get("currency"),
                "exchange": info.get("exchange"),
                "quote_type": info.get("quoteType"),
                "available_at": _utc_now(),
                "source": "Yahoo Finance via yfinance get_info",
                "fetch_status": "ok",
                "fetch_error": None,
            }
        except Exception as exc:  # network/provider errors are recorded, never imputed
            error = f"{type(exc).__name__}: {exc}"
            if attempt < retries:
                time.sleep(min(2.0 ** attempt, 4.0))
    return {
        "symbol": symbol,
        "pe_ttm": None,
        "market_cap": None,
        "trailing_eps": None,
        "shares_outstanding": None,
        "price": None,
        "currency": None,
        "exchange": None,
        "quote_type": None,
        "available_at": _utc_now(),
        "source": "Yahoo Finance via yfinance get_info",
        "fetch_status": "error",
        "fetch_error": error,
    }


def _fetch_snapshot(symbols: list[str], workers: int, retries: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_fetch_symbol, symbol, retries): symbol for symbol in symbols}
        for completed, future in enumerate(as_completed(futures), start=1):
            rows.append(future.result())
            if completed % 25 == 0 or completed == len(futures):
                print(
                    json.dumps(
                        {
                            "event": "pe_snapshot_progress",
                            "completed": completed,
                            "total": len(futures),
                        }
                    ),
                    flush=True,
                )
    return pd.DataFrame(rows).sort_values("symbol", kind="mergesort").reset_index(drop=True)


def _equal_weight_low_pe(frame: pd.DataFrame, mask: pd.Series, top_n: int, variant: str) -> pd.DataFrame:
    candidates = frame.loc[mask & np.isfinite(frame["pe_ttm"]) & (frame["pe_ttm"] > 0)].copy()
    chosen = candidates.sort_values(["pe_ttm", "symbol"], kind="mergesort").head(top_n)
    chosen["weight"] = 1.0 / len(chosen) if len(chosen) else np.nan
    chosen["sector_weight"] = chosen.groupby("sector")["weight"].transform("sum")
    chosen["variant"] = variant
    return chosen


def _portfolio_diagnostics(portfolios: pd.DataFrame) -> list[dict[str, object]]:
    rows = []
    for variant, group in portfolios.groupby("variant", sort=True):
        sector = group.groupby("sector")["weight"].sum()
        gross = float(group["weight"].abs().sum())
        shares = sector.abs() / gross if gross > 0 else sector * np.nan
        hhi = float(np.square(shares).sum()) if len(shares) else np.nan
        rows.append(
            {
                "variant": variant,
                "names": len(group),
                "sectors": int(group["sector"].nunique()),
                "gross": gross,
                "median_pe": float(group["pe_ttm"].median()),
                "median_market_cap": float(group["market_cap"].median()),
                "max_sector_share_of_gross": float(shares.max()) if len(shares) else np.nan,
                "sector_hhi": hhi,
                "effective_sectors": 1.0 / hhi if hhi > 0 else np.nan,
            }
        )
    return rows


def _legacy_report(summary: dict[str, object], diagnostics: pd.DataFrame) -> str:
    lines = [
        "# PE 100% 排序 + Sector-aware Size Filter 當前截面研究",
        "",
        f"完成時間：{summary['completed_at_utc']}",
        "",
        "## 結論邊界",
        "",
        "這是當前截面 sanity check，不是歷史回測。Yahoo current fields 沒有歷史 filing-available timestamps，不可倒灌到過去。Main daily 未修改。",
        "",
        f"- Live universe：{summary['universe_names']}；成功抓取：{summary['fetch_ok']}。",
        f"- Positive PE coverage：{summary['positive_pe_names']}；positive market-cap coverage：{summary['positive_market_cap_names']}；兩者同時有效：{summary['joint_valid_names']}。",
        "- 指定策略：先剔除各 sector 市值低於 contemporaneous median 的股票，再由 positive trailing PE 100% 全局排序選 Top-N，最後只對已入選 sector 做等 gross budget。",
        "- Equal-sector gross budget 不是 covariance／volatility risk parity，也不會為了湊齊行業而改寫 PE Top-N。",
        "- Sector-median filter 是 hard eligibility rule，不是 joint regression；joint OLS challenger 另行輸出。",
        "",
        "## 組合比較",
        "",
        "| 版本 | 名稱數 | 行業數 | 中位PE | 中位市值 | 最大行業占gross | 有效行業數 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in diagnostics.itertuples(index=False):
        lines.append(
            f"| {row.variant} | {int(row.names)} | {int(row.sectors)} | {row.median_pe:.2f} | "
            f"{row.median_market_cap:,.0f} | {row.max_sector_share_of_gross:.2%} | {row.effective_sectors:.2f} |"
        )
    lines.extend(
        [
            "",
            "## 下一步",
            "",
            "取得帶 filing/availability timestamp 的 PIT EPS、shares、market cap 與歷史 sector effective dates 後，才能做 weekly as-of join、成本回測與 joint OLS `earnings_yield ~ z(log_market_cap) + sector dummies`。在此之前只能 shadow-forward。",
            "",
        ]
    )
    return "\n".join(lines)


def _report(summary: dict[str, object], diagnostics: pd.DataFrame) -> str:
    """Render the audited v2 report using encoding-stable text."""

    lines = [
        "# PE-only ranking + sector-aware size screen: current snapshot study",
        "",
        f"Completed at: {summary['completed_at_utc']}",
        "",
        "## Interpretation boundary",
        "",
        "This is a current cross-sectional sanity check, not a historical backtest. "
        "Current Yahoo fields do not provide historical filing-availability timestamps "
        "and therefore cannot be backfilled. The production daily path is unchanged.",
        "",
        f"- Live universe: {summary['universe_names']}; successful fetches: {summary['fetch_ok']}.",
        f"- Positive PE coverage: {summary['positive_pe_names']}; positive market-cap "
        f"coverage: {summary['positive_market_cap_names']}; both valid: {summary['joint_valid_names']}.",
        "- Requested strategy: remove names below their contemporaneous sector median "
        "market cap; rank the survivors globally using positive trailing PE only; then "
        "assign equal gross budgets only to sectors represented in that PE Top-N.",
        "- Equal-sector gross budgeting is not covariance/volatility risk parity and "
        "does not force an otherwise unselected sector into the PE Top-N.",
        "- The sector-median screen is a hard eligibility rule, not a joint regression. "
        "A joint-OLS challenger is exported separately.",
        "- v1 is invalidated by audit findings in its baseline labels, sector-quota "
        "selection, and residual index alignment. v3 supersedes v1 and the otherwise "
        "numerically correct v2 by adding stricter point-in-time and identifier guards.",
        "",
        "## Portfolio comparison",
        "",
        "| Variant | Names | Sectors | Median PE | Median market cap | Max sector / gross | Effective sectors |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in diagnostics.itertuples(index=False):
        lines.append(
            f"| {row.variant} | {int(row.names)} | {int(row.sectors)} | {row.median_pe:.2f} | "
            f"{row.median_market_cap:,.0f} | {row.max_sector_share_of_gross:.2%} | "
            f"{row.effective_sectors:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Next validation step",
            "",
            "A historical weekly, cost-aware backtest requires point-in-time EPS, shares, "
            "market cap, filing availability, and sector effective dates. Until that store "
            "exists, this strategy can only be shadow-forward tracked.",
            "",
        ]
    )
    return "\n".join(lines)


def _verify(output: Path) -> None:
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    success = json.loads((output / "_SUCCESS.json").read_text(encoding="utf-8"))
    if _sha256(output / "manifest.json") != success["manifest_sha256"]:
        raise AssertionError("manifest hash mismatch")
    if _canonical_sha256(manifest["output_sha256"]) != success["output_hashes_sha256"]:
        raise AssertionError("output hash aggregate mismatch")
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
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--top-n", type=int, default=22)
    parser.add_argument(
        "--input-snapshot",
        type=Path,
        help="Replay an immutable previously fetched snapshot instead of refetching provider fields",
    )
    args = parser.parse_args(argv)
    if not 1 <= args.workers <= 8:
        raise ValueError("workers must be in [1, 8]")

    from backend.alpha.neutralization import SECTOR_MAP
    from research_v2.safety import ensure_research_output_path

    output = ensure_research_output_path(args.output, research_root=ROOT)
    if output.exists():
        raise FileExistsError(f"immutable output exists: {output}")
    staging = ensure_research_output_path(
        output.parent / f".{output.name}.partial-{uuid.uuid4().hex}", research_root=ROOT
    )
    staging.mkdir(parents=True, exist_ok=False)
    started_at = _utc_now()
    config_path = WORKSPACE / "config" / "live_strategy.json"
    source_paths = [
        Path(__file__).resolve(),
        ROOT / "fundamental_value.py",
        ROOT / "safety.py",
        WORKSPACE / "backend" / "alpha" / "neutralization.py",
        config_path,
    ]
    if args.input_snapshot is not None:
        replay_path = args.input_snapshot.resolve()
        if not replay_path.is_file():
            raise FileNotFoundError(f"input snapshot does not exist: {replay_path}")
        source_paths.append(replay_path)
    source_hashes = {str(path.relative_to(WORKSPACE)): _sha256(path) for path in source_paths}
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        symbols = sorted(set(map(str, config.get("universe", ()))))
        if not symbols:
            raise ValueError("live config has no universe")
        missing_sector = [symbol for symbol in symbols if symbol not in SECTOR_MAP]
        if missing_sector:
            raise ValueError(f"missing sector mapping: {missing_sector[:10]}")
        if args.input_snapshot is None:
            fetched = _fetch_snapshot(symbols, args.workers, args.retries)
            snapshot_mode = "provider_fetch"
        else:
            fetched = pd.read_csv(args.input_snapshot)
            snapshot_mode = "frozen_replay"
            replay_symbols = set(fetched["symbol"].astype(str))
            if replay_symbols != set(symbols):
                raise ValueError(
                    "replayed snapshot universe differs from live-config universe: "
                    f"missing={sorted(set(symbols) - replay_symbols)[:10]}, "
                    f"extra={sorted(replay_symbols - set(symbols))[:10]}"
                )
        fetched["sector"] = fetched["symbol"].map(SECTOR_MAP)
        completed_at = _utc_now()
        validated = validate_fundamental_snapshot(fetched, decision_at=completed_at)
        positive_pe = np.isfinite(validated["pe_ttm"]) & (validated["pe_ttm"] > 0)
        positive_cap = np.isfinite(validated["market_cap"]) & (validated["market_cap"] > 0)
        joint = positive_pe & positive_cap
        if int(joint.sum()) < max(100, int(0.50 * len(symbols))):
            raise RuntimeError(f"insufficient PE/market-cap coverage: {int(joint.sum())}/{len(symbols)}")

        screened = sector_relative_market_cap_filter(validated, quantile=0.50, min_sector_names=4)
        raw = _equal_weight_low_pe(
            validated,
            pd.Series(True, index=validated.index),
            args.top_n,
            "raw_pe_equal_weight",
        )
        size_only = _equal_weight_low_pe(
            screened,
            screened["sector_size_eligible"],
            args.top_n,
            "raw_pe_sector_median_size_equal_weight",
        )
        _, sector_only = build_pe_sector_balanced_portfolio(
            validated,
            PESectorStrategyConfig(
                top_n=args.top_n,
                gross_target=1.0,
                apply_sector_size_filter=False,
            ),
            decision_at=completed_at,
        )
        sector_only["variant"] = "raw_pe_equal_sector_gross_budget"
        _, requested = build_pe_sector_balanced_portfolio(
            validated,
            PESectorStrategyConfig(
                top_n=args.top_n,
                gross_target=1.0,
                sector_market_cap_quantile=0.50,
                min_sector_names=4,
            ),
            decision_at=completed_at,
        )
        requested["variant"] = "raw_pe_sector_median_size_equal_sector_gross_budget"

        # Executable definitions of the labels above. Weighting is allowed to
        # change weights only; it must never turn into implicit sector quotas.
        if set(raw["symbol"]) != set(sector_only["symbol"]):
            raise AssertionError("equal-sector budget changed raw-PE Top-N selection")
        if set(size_only["symbol"]) != set(requested["symbol"]):
            raise AssertionError("equal-sector budget changed size-filtered PE Top-N selection")

        residual = joint_sector_size_residual(validated)
        joint_frame = validated.copy()
        joint_frame["joint_value_residual"] = residual
        joint_candidates = joint_frame.loc[joint & np.isfinite(residual)].nlargest(
            args.top_n, "joint_value_residual"
        ).copy()
        joint_candidates["weight"] = (
            1.0 / len(joint_candidates) if len(joint_candidates) else np.nan
        )
        joint_candidates["sector_weight"] = joint_candidates.groupby("sector")["weight"].transform("sum")
        joint_candidates["variant"] = "joint_ols_value_residual_equal_weight"

        portfolios = pd.concat(
            [raw, size_only, sector_only, requested, joint_candidates],
            ignore_index=True,
            sort=False,
        )
        diagnostics = pd.DataFrame(_portfolio_diagnostics(portfolios))
        filter_diagnostics = size_filter_sector_diagnostics(validated, quantile=0.50)
        raw_names = set(raw["symbol"])
        requested_names = set(requested["symbol"])
        summary = {
            "study": "current_pe_sector_size_sanity_v3",
            "supersedes": [
                "20260714_pe_sector_current_v1",
                "20260714_pe_sector_current_v2",
            ],
            "supersession_reason": (
                "v1 invalidated by audit: baseline labels, sector-quota selection, "
                "and joint-residual index alignment; v3 adds fail-closed point-in-time "
                "and null-identifier guards to the numerically correct v2"
            ),
            "started_at_utc": started_at,
            "completed_at_utc": completed_at,
            "historical_backtest": False,
            "production_mutations": [],
            "snapshot_mode": snapshot_mode,
            "universe_names": len(symbols),
            "fetch_ok": int((fetched["fetch_status"] == "ok").sum()),
            "positive_pe_names": int(positive_pe.sum()),
            "positive_market_cap_names": int(positive_cap.sum()),
            "joint_valid_names": int(joint.sum()),
            "requested_top_n": args.top_n,
            "requested_overlap_with_raw": len(raw_names & requested_names),
            "requested_jaccard_with_raw": len(raw_names & requested_names) / len(raw_names | requested_names),
            "selection_invariants": {
                "raw_equals_raw_equal_sector_gross_budget": True,
                "size_filtered_equals_size_filtered_equal_sector_gross_budget": True,
            },
            "source_limit": "current provider fields; no historical filing-availability trail",
        }
        validated.to_csv(staging / "fundamental_snapshot.csv", index=False)
        screened.to_csv(staging / "sector_size_screen.csv", index=False)
        filter_diagnostics.to_csv(staging / "filter_sector_diagnostics.csv", index=False)
        portfolios.to_csv(staging / "portfolio_variants.csv", index=False)
        diagnostics.to_csv(staging / "portfolio_diagnostics.csv", index=False)
        _write_json(staging / "summary.json", summary)
        (staging / "report.md").write_text(_report(summary, diagnostics), encoding="utf-8")

        final_source_hashes = {str(path.relative_to(WORKSPACE)): _sha256(path) for path in source_paths}
        if final_source_hashes != source_hashes:
            raise AssertionError("study source changed during current snapshot fetch")
        output_hashes = {
            str(path.relative_to(staging)).replace("\\", "/"): _sha256(path)
            for path in sorted(staging.rglob("*"))
            if path.is_file()
        }
        manifest = {
            **summary,
            "source_sha256": source_hashes,
            "source_sha256_aggregate": _canonical_sha256(source_hashes),
            "provider": {
                "library": "yfinance",
                "fields": [
                    "trailingPE",
                    "marketCap",
                    "trailingEps",
                    "sharesOutstanding",
                    "currentPrice/regularMarketPrice",
                ],
            },
            "output_sha256": output_hashes,
        }
        _write_json(staging / "manifest.json", manifest)
        _write_json(
            staging / "_SUCCESS.json",
            {
                "completed_at_utc": completed_at,
                "immutable_by_runner": True,
                "manifest_sha256": _sha256(staging / "manifest.json"),
                "output_hashes_sha256": _canonical_sha256(output_hashes),
                "historical_backtest": False,
            },
        )
        _verify(staging)
        os.replace(staging, output)
        _verify(output)
        print(json.dumps({"event": "pe_sector_current_completed", "output": str(output)}), flush=True)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
