"""Build and freeze a two-snapshot delta-PE research signal.

The output is a mechanics/data-quality study plus a forward shadow portfolio.
Returns between the two snapshots are exported only as contemporaneous
attribution and are never treated as predictive performance.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import uuid

import numpy as np
import pandas as pd

from research_v2.delta_pe import (
    DeltaPEConfig,
    build_delta_pe_portfolio,
    compute_delta_pe_features,
)
from research_v2.safety import ensure_research_output_path, offline_context


ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent
DEFAULT_PREVIOUS = WORKSPACE / "data_cache" / "graham_fundamentals.json"
DEFAULT_CURRENT = ROOT / "runs" / "20260714_pe_sector_current_v3" / "fundamental_snapshot.csv"
DEFAULT_STORE = WORKSPACE / "data" / "store"
DEFAULT_OUTPUT = ROOT / "runs" / "20260714_delta_pe_study_v2"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(data).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _load_previous(path: Path, sector_map: dict[str, str]) -> tuple[pd.DataFrame, pd.Timestamp]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    captured = pd.Timestamp(payload["date"])
    available = captured.tz_localize("UTC")
    rows = []
    for symbol, fields in payload["data"].items():
        rows.append(
            {
                "symbol": str(symbol),
                "sector": sector_map.get(str(symbol), "Unknown"),
                "pe_ttm": fields.get("trailingPE"),
                "trailing_eps": fields.get("trailingEps"),
                "market_cap": np.nan,
                "available_at": available,
            }
        )
    return pd.DataFrame(rows), captured


def _load_contemporaneous_returns(
    symbols: list[str],
    store: Path,
    start_date: str,
    end_date: str,
) -> tuple[pd.Series, list[Path]]:
    returns: dict[str, float] = {}
    paths: list[Path] = []
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    for symbol in symbols:
        path = store / f"{symbol}.parquet"
        if not path.is_file():
            continue
        paths.append(path)
        bars = pd.read_parquet(path, columns=["timestamp", "open", "close"])
        bars["timestamp"] = pd.to_datetime(bars["timestamp"])
        bars = bars.drop_duplicates("timestamp", keep="last").set_index("timestamp").sort_index()
        entry = bars.loc[bars.index.normalize() == start, "open"].dropna()
        exit_ = bars.loc[bars.index.normalize() == end, "close"].dropna()
        if len(entry) and len(exit_) and float(entry.iloc[0]) > 0 and float(exit_.iloc[-1]) > 0:
            returns[symbol] = float(exit_.iloc[-1] / entry.iloc[0] - 1.0)
    return pd.Series(returns, name="contemporaneous_adjusted_price_return"), paths


def _rank_correlation(x: pd.Series, y: pd.Series) -> float:
    joined = pd.concat([x, y], axis=1).dropna()
    if len(joined) < 3:
        return np.nan
    return float(joined.iloc[:, 0].rank().corr(joined.iloc[:, 1].rank()))


def _portfolio_diagnostics(portfolios: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variant, group in portfolios.groupby("variant", sort=True):
        sector = group.groupby("sector")["weight"].sum()
        shares = sector.abs() / group["weight"].abs().sum()
        hhi = float(np.square(shares).sum())
        valid_return = group.dropna(subset=["contemporaneous_adjusted_price_return"])
        weighted_return = (
            float(
                np.average(
                    valid_return["contemporaneous_adjusted_price_return"],
                    weights=valid_return["weight"].abs(),
                )
            )
            if len(valid_return)
            else np.nan
        )
        eps_simple = (np.exp(group["delta_log_eps"]) - 1.0).dropna()
        rows.append(
            {
                "variant": variant,
                "names": len(group),
                "sectors": int(group["sector"].nunique()),
                "max_sector_share": float(shares.max()),
                "effective_sectors": 1.0 / hhi if hhi > 0 else np.nan,
                "median_pe_change": float(np.expm1(group["delta_log_pe"].median())),
                "median_reported_eps_change": float(np.expm1(group["delta_log_eps"].median())),
                "reported_eps_coverage": float(len(eps_simple) / len(group)),
                "reported_eps_decline_fraction": float((eps_simple < 0).mean()) if len(eps_simple) else np.nan,
                "reported_eps_decline_over_20pct_fraction": float((eps_simple < -0.20).mean()) if len(eps_simple) else np.nan,
                "contemporaneous_return_coverage": float(len(valid_return) / len(group)),
                "contemporaneous_weighted_price_return_nonpredictive": weighted_return,
            }
        )
    return pd.DataFrame(rows)


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
    parser.add_argument("--previous", type=Path, default=DEFAULT_PREVIOUS)
    parser.add_argument("--current", type=Path, default=DEFAULT_CURRENT)
    parser.add_argument("--store", type=Path, default=DEFAULT_STORE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--top-n", type=int, default=22)
    parser.add_argument("--min-sector-names", type=int, default=15)
    parser.add_argument("--min-sector-retention", type=float, default=0.70)
    parser.add_argument("--eps-floor", type=float, default=-0.10)
    parser.add_argument("--attribution-start", default="2026-02-17")
    parser.add_argument("--attribution-end", default="2026-07-10")
    args = parser.parse_args(argv)
    if not -1.0 < args.eps_floor:
        raise ValueError("eps-floor must be greater than -1")

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

            previous, previous_timestamp = _load_previous(args.previous, SECTOR_MAP)
            current = pd.read_csv(args.current)
            current["available_at"] = pd.to_datetime(current["available_at"], utc=True)
            decision_at = current["available_at"].max()
            config = DeltaPEConfig(
                top_n=args.top_n,
                min_sector_names=args.min_sector_names,
                eps_log_floor=float(np.log1p(args.eps_floor)),
                min_sector_matched_retention=args.min_sector_retention,
            )
            features = compute_delta_pe_features(
                previous,
                current,
                decision_at=decision_at,
                config=config,
            )
            attribution, price_paths = _load_contemporaneous_returns(
                features["symbol"].tolist(),
                args.store,
                args.attribution_start,
                args.attribution_end,
            )
            features = features.merge(
                attribution.rename_axis("symbol").reset_index(), on="symbol", how="left"
            )

        portfolio_specs = [
            ("delta_log_pe", False),
            ("relative_delta_log_pe", False),
            ("literal_surge_and_score", False),
            ("surge_and_score", False),
            ("reported_eps_guarded_literal_surge_score", False),
            ("reported_eps_guarded_surge_score", False),
            ("reported_eps_guarded_literal_surge_score", True),
        ]
        portfolios = []
        for score, equal_sector in portfolio_specs:
            portfolio = build_delta_pe_portfolio(
                features,
                score_column=score,
                config=config,
                equal_sector_gross=equal_sector,
            )
            portfolios.append(portfolio)
        portfolio_frame = pd.concat(portfolios, ignore_index=True, sort=False)
        diagnostics = _portfolio_diagnostics(portfolio_frame)

        sectors = (
            features.groupby("sector", sort=True)
            .agg(
                matched_names=("delta_log_pe", "count"),
                overlap_names=("sector_overlap_names", "first"),
                matched_pe_retention=("sector_matched_pe_retention", "first"),
                median_delta_log_pe=("sector_delta_log_pe", "first"),
                positive_delta_pe_breadth=("sector_positive_delta_pe_breadth", "first"),
                sector_delta_pe_rank=("sector_delta_pe_rank", "first"),
                median_reported_delta_log_eps=("sector_delta_log_eps", "first"),
            )
            .reset_index()
        )
        sectors["median_pe_change"] = np.expm1(sectors["median_delta_log_pe"])
        sectors["median_reported_eps_change"] = np.expm1(
            sectors["median_reported_delta_log_eps"]
        )
        correlations = []
        for score in (
            "delta_log_pe",
            "relative_delta_log_pe",
            "literal_surge_and_score_unfiltered",
            "surge_and_score_unfiltered",
            "reported_eps_quality_delta_log_pe",
        ):
            correlations.append(
                {
                    "score": score,
                    "contemporaneous_spearman_nonpredictive": _rank_correlation(
                        features[score], features["contemporaneous_adjusted_price_return"]
                    ),
                    "observations": int(
                        features[[score, "contemporaneous_adjusted_price_return"]]
                        .dropna()
                        .shape[0]
                    ),
                }
            )
        correlation_frame = pd.DataFrame(correlations)
        transitions = (
            features.groupby("pe_state_transition")["symbol"]
            .count()
            .rename("names")
            .reset_index()
        )

        valid_delta = int(features["delta_log_pe"].notna().sum())
        valid_eps_delta = int(features["delta_log_eps"].notna().sum())
        guarded_literal_names = int(
            features["reported_eps_guarded_literal_surge_score"].notna().sum()
        )
        guarded_literal_sectors = int(
            features.loc[
                features["reported_eps_guarded_literal_surge_score"].notna(), "sector"
            ].nunique()
        )
        guarded_relative_names = int(
            features["reported_eps_guarded_surge_score"].notna().sum()
        )
        guarded_relative_sectors = int(
            features.loc[
                features["reported_eps_guarded_surge_score"].notna(), "sector"
            ].nunique()
        )
        current_available_min = str(current["available_at"].min())
        current_available_max = str(current["available_at"].max())
        summary = {
            "study": "delta_pe_two_snapshot_mechanics_v2",
            "supersedes": "20260714_delta_pe_study_v1",
            "supersession_reason": (
                "adds the user's literal stock-plus-sector AND separately from the robust "
                "stock-relative AND, plus sector retention and diagnostic coverage guards"
            ),
            "started_at_utc": started_at,
            "completed_at_utc": _utc_now(),
            "previous_snapshot_timestamp_naive": str(previous_timestamp),
            "current_available_at_min": current_available_min,
            "current_available_at_max": current_available_max,
            "overlap_names": int(len(features)),
            "valid_positive_pe_delta_names": valid_delta,
            "valid_positive_pe_and_eps_delta_names": valid_eps_delta,
            "reported_eps_guarded_literal_surge_names": guarded_literal_names,
            "reported_eps_guarded_literal_surge_sectors": guarded_literal_sectors,
            "reported_eps_guarded_relative_surge_names": guarded_relative_names,
            "reported_eps_guarded_relative_surge_sectors": guarded_relative_sectors,
            "sector_delta_definition": "median matched constituent log(PE_current / PE_previous)",
            "literal_surge_score_definition": "min(sector percentile rank, global stock delta-PE percentile rank)",
            "relative_surge_score_definition": "min(sector percentile rank, within-sector relative percentile rank)",
            "eps_simple_floor": args.eps_floor,
            "min_sector_matched_retention": args.min_sector_retention,
            "historical_backtest": False,
            "predictive_performance_claimed": False,
            "contemporaneous_attribution_only": True,
            "suddenness_tested": False,
            "forward_returns_available": False,
            "shadow_entry_rule": "first tradable open strictly after current snapshot availability",
            "production_mutations": [],
            "limitations": [
                "only two snapshots approximately five months apart; sudden change is not measurable",
                "legacy timestamp has no timezone, vendor, query, or filing-availability provenance",
                "current static sector map is not a historical point-in-time classification",
                "reported EPS changes are not certified on a corporate-action-consistent share basis",
                "sector coverage differs materially across sectors",
                "contemporaneous price attribution contains the price component of delta PE and is not predictive evidence",
            ],
        }

        features.to_csv(staging / "delta_pe_features.csv", index=False)
        sectors.to_csv(staging / "sector_summary.csv", index=False)
        portfolio_frame.to_csv(staging / "shadow_portfolio_variants.csv", index=False)
        diagnostics.to_csv(staging / "portfolio_diagnostics.csv", index=False)
        correlation_frame.to_csv(staging / "contemporaneous_correlations_nonpredictive.csv", index=False)
        transitions.to_csv(staging / "pe_state_transitions.csv", index=False)
        _write_json(staging / "summary.json", summary)

        lines = [
            "# Delta-PE two-snapshot mechanics and shadow study",
            "",
            "This is not a predictive backtest. The July snapshot is required to know the "
            "delta-PE score, so February-to-July returns are contemporaneous attribution only.",
            "",
            f"- Overlap: {len(features)}; valid positive PE/EPS deltas: {valid_eps_delta}.",
            f"- Current snapshot availability: {current_available_min} to {current_available_max}.",
            f"- EPS-guarded literal stock+sector AND: {guarded_literal_names} candidates across {guarded_literal_sectors} sectors.",
            f"- EPS-guarded robust stock-relative+sector AND: {guarded_relative_names} candidates across {guarded_relative_sectors} sectors.",
            "- Sector delta is the median matched constituent log-PE change; arithmetic mean PE is not used.",
            "- 'Sudden' is not tested: two snapshots are about five months apart.",
            "",
            "## Sector changes",
            "",
            "| Sector | Matched / overlap | Retention | Median PE change | Positive breadth | Reported EPS change |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for row in sectors.sort_values("median_pe_change", ascending=False).itertuples(index=False):
            lines.append(
                f"| {row.sector} | {int(row.matched_names)} / {int(row.overlap_names)} | "
                f"{row.matched_pe_retention:.2%} | {row.median_pe_change:.2%} | "
                f"{row.positive_delta_pe_breadth:.2%} | {row.median_reported_eps_change:.2%} |"
            )
        lines.extend(
            [
                "",
                "## Portfolio mechanics",
                "",
                "| Variant | Names | Sectors | Max sector | Median PE change | Median reported EPS change | EPS <-20% |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in diagnostics.itertuples(index=False):
            lines.append(
                f"| {row.variant} | {int(row.names)} | {int(row.sectors)} | "
                f"{row.max_sector_share:.2%} | {row.median_pe_change:.2%} | "
                f"{row.median_reported_eps_change:.2%} | "
                f"{row.reported_eps_decline_over_20pct_fraction:.2%} |"
            )
        lines.extend(
            [
                "",
                "Both AND portfolios are frozen shadow candidates only. Their earliest "
                "valid execution is the first market open after the July snapshot. A 5/21/63-day "
                "forward evaluation requires future data and repeated PIT snapshots.",
                "",
            ]
        )
        (staging / "report.md").write_text("\n".join(lines), encoding="utf-8")

        source_paths = [
            Path(__file__).resolve(),
            ROOT / "delta_pe.py",
            ROOT / "fundamental_value.py",
            ROOT / "safety.py",
            WORKSPACE / "backend" / "alpha" / "neutralization.py",
            args.previous.resolve(),
            args.current.resolve(),
            *price_paths,
        ]
        source_paths = list(dict.fromkeys(path.resolve() for path in source_paths))
        source_hashes = {
            str(path.relative_to(WORKSPACE)).replace("\\", "/"): _sha256(path)
            for path in source_paths
        }
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
        print(json.dumps({"event": "delta_pe_study_completed", "output": str(output)}))
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
