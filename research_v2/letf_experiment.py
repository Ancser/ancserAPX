"""Execution and diagnostics for the isolated LETF rotation study.

This module deliberately does not know how a rotation score is produced.  It
accepts an already point-in-time ``date -> {ticker: score}`` map, converts a
frozen OHLCV snapshot into :mod:`research_v2.backtest` inputs, and routes every
target through the existing Research-v2 portfolio constructor and event clock.

The separation is important: factor research may change the scores, but it may
not create a second weighting/execution truth or import any live component.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from hashlib import sha256
import json
import math
from pathlib import Path
from pathlib import PurePosixPath
import re
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import polars as pl

from .backtest import (
    BacktestResult,
    MarketBar,
    RiskConfig,
    RiskObservation,
    assert_accounting_identity,
    run_backtest,
)
from .costs import CostConfig
from .metrics import PerformanceMetrics, compute_performance_metrics
from .portfolio import PortfolioConfig
from .snapshot import SnapshotVerificationError, verify_snapshot


REQUIRED_BAR_COLUMNS = (
    "timestamp",
    "symbol",
    "open",
    "high",
    "low",
    "close",
    "volume",
)

SNAPSHOT_PROVENANCE_FIELDS = (
    "provider",
    "feed",
    "adjustment",
    "retrieved_at_utc",
)

SNAPSHOT_DATA_HASH_SCHEME = "research-v2-snapshot-data-sha256-v1"
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


@dataclass(frozen=True)
class LETFExecutionConfig:
    """Frozen execution/risk assumptions for a rotation scenario."""

    initial_capital: float = 100_000.0
    top_k: int = 5
    weighting: str = "inverse_vol"
    gross_target: float = 1.0
    single_name_cap: float = 0.25
    theme_cap: float = 0.25
    rank_buffer: int = 1
    # Disabled in the primary LETF run.  The generic portfolio builder can
    # retain drifted close weights inside a band, which may leave gross a hair
    # above the requested cap.  Turnover control is tested as an ablation only.
    no_trade_band: float = 0.0
    max_adv_participation: float = 0.01
    target_volatility: Optional[float] = 0.15
    risk_vol_lookback: int = 63
    risk_min_observations: int = 30
    trend_filter: bool = True
    breadth_exit: Optional[float] = 0.40
    breadth_enter: Optional[float] = 0.50
    risk_off_multiplier: float = 0.25
    drawdown_steps: Tuple[Tuple[float, float], ...] = (
        (0.10, 0.67),
        (0.20, 0.33),
    )
    risk_target_change_buffer: float = 0.01
    commission_bps: float = 0.0
    extra_friction_bps: float = 10.0
    spread_range_fraction: float = 0.02
    min_spread_bps: float = 1.0
    max_spread_bps: float = 30.0
    impact_coefficient: float = 0.10
    max_impact_bps: float = 250.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.initial_capital) or self.initial_capital <= 0:
            raise ValueError("initial_capital must be finite and positive")
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")
        if self.weighting not in {"equal", "inverse_vol"}:
            raise ValueError("weighting must be equal or inverse_vol")
        if not 0 < self.gross_target <= 1.0:
            raise ValueError("LETF outer gross_target must be in (0, 1]")
        if not 0 < self.single_name_cap <= 1.0:
            raise ValueError("single_name_cap must be in (0, 1]")
        if not 0 < self.theme_cap <= 1.0:
            raise ValueError("theme_cap must be in (0, 1]")
        if not 0 < self.max_adv_participation <= 0.05:
            raise ValueError("max_adv_participation must be in (0, 5%]")
        if not math.isfinite(self.risk_target_change_buffer) or not 0 <= self.risk_target_change_buffer <= 1:
            raise ValueError("risk_target_change_buffer must be in [0, 1]")
        for name in (
            "commission_bps",
            "extra_friction_bps",
            "spread_range_fraction",
            "min_spread_bps",
            "max_spread_bps",
            "impact_coefficient",
            "max_impact_bps",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.max_spread_bps < self.min_spread_bps:
            raise ValueError("max_spread_bps must be >= min_spread_bps")

    def portfolio_config(self) -> PortfolioConfig:
        return PortfolioConfig(
            top_n=self.top_k,
            weighting=self.weighting,
            gross_target=self.gross_target,
            single_name_cap=self.single_name_cap,
            sector_cap=self.theme_cap,
            rank_buffer=self.rank_buffer,
            no_trade_band=self.no_trade_band,
            staggered_tranches=1,
            max_adv_participation=self.max_adv_participation,
        )

    def cost_config(self) -> CostConfig:
        # ``extra_friction_bps`` is deliberately booked in the commission
        # column.  It is an additive one-way stress on top of the lagged spread
        # proxy and square-root impact model, not a claim about broker fees.
        return CostConfig(
            commission_bps=self.commission_bps + self.extra_friction_bps,
            spread_multiplier=1.0,
            min_spread_bps=self.min_spread_bps,
            max_spread_bps=self.max_spread_bps,
            impact_coefficient=self.impact_coefficient,
            max_impact_bps=self.max_impact_bps,
            max_adv_participation=self.max_adv_participation,
            annual_funding_rate=0.0,
            annual_short_borrow_rate=0.0,
            annual_cash_rate=0.0,
        )

    def risk_config(self) -> RiskConfig:
        return RiskConfig(
            target_volatility=self.target_volatility,
            vol_lookback=self.risk_vol_lookback,
            min_vol_observations=self.risk_min_observations,
            drawdown_steps=self.drawdown_steps,
            trend_filter=self.trend_filter,
            breadth_exit=self.breadth_exit,
            breadth_enter=self.breadth_enter,
            risk_off_multiplier=self.risk_off_multiplier,
            target_change_buffer=self.risk_target_change_buffer,
        )


@dataclass(frozen=True)
class LETFMarketContext:
    market: Mapping[pd.Timestamp, Mapping[str, MarketBar]]
    risk_observations: Mapping[pd.Timestamp, RiskObservation]
    themes: Mapping[str, str]
    sessions: Tuple[pd.Timestamp, ...]
    symbols: Tuple[str, ...]
    benchmark_symbol: str
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class ScenarioResult:
    backtest: BacktestResult
    ledger: Tuple[Any, ...]
    metrics: PerformanceMetrics
    first_execution_index: int
    config: LETFExecutionConfig


def _normalise_bar_frame(frame: pd.DataFrame) -> pd.DataFrame:
    missing = set(REQUIRED_BAR_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"bar frame missing columns: {sorted(missing)}")
    data = frame.loc[:, list(REQUIRED_BAR_COLUMNS)].copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True).dt.tz_convert(None).dt.normalize()
    data["symbol"] = data["symbol"].astype(str).str.upper()
    for column in ("open", "high", "low", "close", "volume"):
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.sort_values(["symbol", "timestamp"], kind="mergesort")
    duplicates = data.duplicated(["timestamp", "symbol"], keep=False)
    if duplicates.any():
        sample = data.loc[duplicates, ["timestamp", "symbol"]].head(5).to_dict("records")
        raise ValueError(f"duplicate symbol/session bars: {sample}")
    finite_prices = np.isfinite(data[["open", "high", "low", "close"]]).all(axis=1)
    valid = (
        finite_prices
        & (data[["open", "high", "low", "close"]] > 0).all(axis=1)
        & np.isfinite(data["volume"])
        & (data["volume"] >= 0)
        & (data["high"] >= data[["open", "close", "low"]].max(axis=1))
        & (data["low"] <= data[["open", "close", "high"]].min(axis=1))
    )
    if not valid.all():
        sample = data.loc[~valid, ["timestamp", "symbol"]].head(5).to_dict("records")
        raise ValueError(f"invalid OHLCV rows: {sample}")
    return data.reset_index(drop=True)


def _provenance_sources(payload: Mapping[str, Any], label: str) -> Sequence[Tuple[str, Mapping[str, Any]]]:
    """Return explicitly recorded provenance containers from one JSON object."""

    sources: list[Tuple[str, Mapping[str, Any]]] = []
    for key in ("_provenance", "provenance"):
        if key not in payload:
            continue
        value = payload[key]
        if not isinstance(value, Mapping):
            raise SnapshotVerificationError(f"{label}.{key} must be a JSON object")
        sources.append((f"{label}.{key}", value))
    direct = {field: payload[field] for field in SNAPSHOT_PROVENANCE_FIELDS if field in payload}
    if direct:
        sources.append((label, direct))
    return sources


def _normalise_provenance_value(field: str, value: Any, source: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SnapshotVerificationError(f"{source}.{field} must be a non-empty string")
    recorded = value.strip()
    if field == "retrieved_at_utc":
        try:
            timestamp = pd.Timestamp(recorded)
        except (TypeError, ValueError) as exc:
            raise SnapshotVerificationError(
                f"{source}.retrieved_at_utc must be an ISO-8601 UTC timestamp"
            ) from exc
        if pd.isna(timestamp) or timestamp.tzinfo is None or timestamp.utcoffset().total_seconds() != 0:
            raise SnapshotVerificationError(
                f"{source}.retrieved_at_utc must be an ISO-8601 UTC timestamp"
            )
    return recorded


def _comparison_value(field: str, value: str) -> str:
    if field in {"feed", "adjustment"}:
        return value.upper()
    return value.casefold()


def _verified_snapshot_data_identity(metadata: Mapping[str, Any]) -> Mapping[str, Any]:
    """Content-address the exact Parquet set already verified by ``verify_snapshot``.

    The aggregate commits to each relative path, byte size, and SHA-256 in
    canonical path order.  It intentionally uses only the fingerprints that
    ``verify_snapshot`` has just matched against the physical files; it never
    substitutes a source manifest hash or infers identity from provenance.
    """

    files = metadata.get("files")
    if not isinstance(files, list) or not files:
        raise SnapshotVerificationError(
            "snapshot metadata must contain a non-empty files list"
        )
    declared_count = metadata.get("file_count")
    if isinstance(declared_count, bool) or not isinstance(declared_count, int):
        raise SnapshotVerificationError("snapshot file_count must be an integer")

    canonical_files = []
    seen_paths: set[str] = set()
    for position, item in enumerate(files):
        if not isinstance(item, Mapping):
            raise SnapshotVerificationError(
                f"snapshot files[{position}] must be a JSON object"
            )
        relative_path = item.get("relative_path")
        if not isinstance(relative_path, str) or not relative_path:
            raise SnapshotVerificationError(
                f"snapshot files[{position}].relative_path must be a non-empty string"
            )
        parsed_path = PurePosixPath(relative_path)
        if (
            parsed_path.is_absolute()
            or ".." in parsed_path.parts
            or parsed_path.as_posix() != relative_path
            or parsed_path.suffix.lower() != ".parquet"
        ):
            raise SnapshotVerificationError(
                f"snapshot files[{position}].relative_path is not a canonical Parquet path"
            )
        if relative_path in seen_paths:
            raise SnapshotVerificationError(
                f"snapshot metadata contains duplicate file path: {relative_path}"
            )
        seen_paths.add(relative_path)

        size = item.get("size")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise SnapshotVerificationError(
                f"snapshot files[{position}].size must be a non-negative integer"
            )
        file_sha256 = item.get("sha256")
        if not isinstance(file_sha256, str) or not _SHA256_RE.fullmatch(file_sha256):
            raise SnapshotVerificationError(
                f"snapshot files[{position}].sha256 must be a lowercase SHA-256"
            )
        canonical_files.append(
            {
                "relative_path": relative_path,
                "size": size,
                "sha256": file_sha256,
            }
        )

    if declared_count != len(canonical_files):
        raise SnapshotVerificationError(
            "snapshot file_count does not match verified Parquet fingerprints"
        )
    canonical_files.sort(key=lambda item: item["relative_path"])
    snapshot_id = metadata.get("snapshot_id")
    manifest = metadata.get("manifest")
    format_version = metadata.get("format_version")
    if not isinstance(snapshot_id, str) or not snapshot_id:
        raise SnapshotVerificationError("snapshot metadata is missing snapshot_id")
    if isinstance(format_version, bool) or not isinstance(format_version, int):
        raise SnapshotVerificationError("snapshot metadata has invalid format_version")
    if not isinstance(manifest, Mapping):
        raise SnapshotVerificationError("snapshot metadata is missing manifest identity")
    manifest_sha256 = manifest.get("sha256")
    if not isinstance(manifest_sha256, str) or not _SHA256_RE.fullmatch(
        manifest_sha256
    ):
        raise SnapshotVerificationError("snapshot manifest SHA-256 is invalid")

    # This is byte-for-byte compatible with Research-v2's established
    # ``snapshot_data_sha256`` identity in cli.py.
    aggregate_payload = {
        "format_version": format_version,
        "snapshot_id": snapshot_id,
        "manifest_sha256": manifest_sha256,
        "files": canonical_files,
    }
    encoded = json.dumps(
        aggregate_payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return {
        "snapshot_data_sha256": sha256(encoded).hexdigest(),
        "snapshot_data_file_count": len(canonical_files),
        "snapshot_data_hash_scheme": SNAPSHOT_DATA_HASH_SCHEME,
    }


def load_verified_snapshot_metadata(
    snapshot_dir: str | Path,
    *,
    research_root: str | Path | None = None,
    required_provider: str | None = None,
    required_feed: str | None = None,
    required_adjustment: str | None = None,
    require_retrieved_at_utc: bool = False,
) -> Mapping[str, Any]:
    """Verify a snapshot and return its recorded, never inferred, provenance.

    ``verify_snapshot`` re-hashes the copied manifest and every Parquet file
    before this helper reads provenance.  Provenance may be recorded at the
    top level or under ``_provenance``/``provenance`` in either
    ``snapshot.json`` or ``manifest.json``.  If multiple locations record the
    same field, they must agree.

    Exact requirements are opt-in so generic research snapshots remain
    loadable.  A production-quality caller can fail closed with, for example,
    ``required_feed="SIP"``, ``required_adjustment="ALL"``, and
    ``require_retrieved_at_utc=True``.  Missing fields are never filled from
    the snapshot creation time or command-line arguments.
    """

    directory = Path(snapshot_dir).expanduser().resolve(strict=False)
    root = Path(research_root).expanduser().resolve(strict=False) if research_root else directory.parents[1]
    try:
        metadata = verify_snapshot(directory, research_root=root)
    except SnapshotVerificationError:
        raise
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise SnapshotVerificationError("invalid snapshot metadata schema") from exc
    if not isinstance(metadata, Mapping):
        raise SnapshotVerificationError("snapshot.json must contain a JSON object")

    manifest_path = directory / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SnapshotVerificationError(f"invalid snapshot manifest: {manifest_path}") from exc
    if not isinstance(manifest, Mapping):
        raise SnapshotVerificationError("manifest.json must contain a JSON object")

    provenance: Dict[str, str] = {}
    provenance_source: Dict[str, str] = {}
    sources = (
        *_provenance_sources(metadata, "snapshot.json"),
        *_provenance_sources(manifest, "manifest.json"),
    )
    for source, values in sources:
        for field in SNAPSHOT_PROVENANCE_FIELDS:
            if field not in values:
                continue
            recorded = _normalise_provenance_value(field, values[field], source)
            if field in provenance and _comparison_value(field, provenance[field]) != _comparison_value(field, recorded):
                raise SnapshotVerificationError(
                    f"conflicting {field} provenance in {provenance_source[field]} and {source}"
                )
            provenance[field] = recorded
            provenance_source[field] = source

    requirements = {
        "provider": required_provider,
        "feed": required_feed,
        "adjustment": required_adjustment,
    }
    for field, expected_value in requirements.items():
        if expected_value is None:
            continue
        expected = _normalise_provenance_value(field, expected_value, "requirement")
        actual = provenance.get(field)
        if actual is None:
            raise SnapshotVerificationError(f"snapshot provenance is missing required {field}")
        if _comparison_value(field, actual) != _comparison_value(field, expected):
            raise SnapshotVerificationError(
                f"snapshot provenance {field} is {actual!r}; required {expected!r}"
            )
    if require_retrieved_at_utc and "retrieved_at_utc" not in provenance:
        raise SnapshotVerificationError(
            "snapshot provenance is missing required retrieved_at_utc"
        )

    enriched = dict(metadata)
    enriched["provenance"] = dict(provenance)
    enriched["provenance_sources"] = dict(provenance_source)
    enriched.update(_verified_snapshot_data_identity(metadata))
    return enriched


def load_snapshot_bars(
    snapshot_dir: str | Path,
    symbols: Iterable[str],
    *,
    research_root: str | Path | None = None,
    start: object | None = None,
    end: object | None = None,
    required_provider: str | None = None,
    required_feed: str | None = None,
    required_adjustment: str | None = None,
    require_retrieved_at_utc: bool = False,
) -> Tuple[pd.DataFrame, Mapping[str, Any]]:
    """Load named Parquet files from a verified immutable snapshot."""

    directory = Path(snapshot_dir).expanduser().resolve(strict=False)
    root = Path(research_root).expanduser().resolve(strict=False) if research_root else directory.parents[1]
    metadata = load_verified_snapshot_metadata(
        directory,
        research_root=root,
        required_provider=required_provider,
        required_feed=required_feed,
        required_adjustment=required_adjustment,
        require_retrieved_at_utc=require_retrieved_at_utc,
    )
    requested = tuple(dict.fromkeys(str(symbol).strip().upper() for symbol in symbols if str(symbol).strip()))
    if not requested:
        raise ValueError("symbols cannot be empty")
    frames: list[pl.DataFrame] = []
    missing: list[str] = []
    for symbol in requested:
        path = directory / "store" / f"{symbol}.parquet"
        if not path.is_file():
            missing.append(symbol)
            continue
        frame = pl.read_parquet(path)
        missing_columns = set(REQUIRED_BAR_COLUMNS) - set(frame.columns)
        if missing_columns:
            raise ValueError(f"{symbol} snapshot file missing {sorted(missing_columns)}")
        frames.append(frame.select(REQUIRED_BAR_COLUMNS))
    if missing:
        raise FileNotFoundError(f"snapshot missing symbols: {', '.join(missing)}")
    combined = pl.concat(frames, how="vertical_relaxed").to_pandas()
    data = _normalise_bar_frame(combined)
    if start is not None:
        data = data.loc[data["timestamp"] >= pd.Timestamp(start).normalize()]
    if end is not None:
        data = data.loc[data["timestamp"] <= pd.Timestamp(end).normalize()]
    if data.empty:
        raise ValueError("snapshot date range is empty")
    return data.reset_index(drop=True), metadata


def _rolling_market_inputs(data: pd.DataFrame, config: LETFExecutionConfig) -> pd.DataFrame:
    working = data.copy().sort_values(["symbol", "timestamp"], kind="mergesort")
    working["dollar_volume"] = working["close"] * working["volume"]
    grouped = working.groupby("symbol", sort=False, group_keys=False)
    working["adv20"] = grouped["dollar_volume"].transform(
        lambda values: values.rolling(20, min_periods=1).mean()
    )
    working["daily_return"] = grouped["close"].pct_change(fill_method=None)
    working["vol20"] = grouped["daily_return"].transform(
        lambda values: values.rolling(20, min_periods=2).std(ddof=1)
    ).fillna(0.0)
    range_bps = (
        (working["high"] - working["low"]) / working["close"]
        * 10_000.0
        * config.spread_range_fraction
    )
    working["spread_proxy_bps"] = range_bps.clip(
        lower=config.min_spread_bps,
        upper=config.max_spread_bps,
    )
    if not np.isfinite(working[["adv20", "vol20", "spread_proxy_bps"]]).all(axis=None):
        raise ValueError("non-finite lagged market inputs")
    return working


def build_market_context(
    bars: pd.DataFrame,
    *,
    candidate_symbols: Sequence[str],
    themes: Mapping[str, str],
    benchmark_symbol: str = "SPY",
    breadth: Optional[Mapping[object, float]] = None,
    config: Optional[LETFExecutionConfig] = None,
) -> LETFMarketContext:
    """Create a dynamic (listing-aware) event-engine market context.

    Missing pre-inception rows are allowed.  Missing rows for a held symbol are
    not repaired here; the event engine will raise ``MissingHeldReturnError``.
    """

    execution = config or LETFExecutionConfig()
    data = _normalise_bar_frame(bars)
    candidates = tuple(dict.fromkeys(str(symbol).upper() for symbol in candidate_symbols))
    if not candidates:
        raise ValueError("candidate_symbols cannot be empty")
    benchmark = str(benchmark_symbol).upper()
    needed = set(candidates) | {benchmark}
    data = data.loc[data["symbol"].isin(needed)].copy()
    missing = sorted(needed - set(data["symbol"]))
    if missing:
        raise ValueError(f"market data missing symbols: {missing}")
    missing_themes = sorted(set(candidates) - {str(symbol).upper() for symbol in themes})
    if missing_themes:
        raise ValueError(f"theme map missing candidates: {missing_themes}")
    theme_map = {str(symbol).upper(): str(theme) for symbol, theme in themes.items()}
    theme_map.setdefault(benchmark, f"Benchmark:{benchmark}")

    prepared = _rolling_market_inputs(data, execution)
    benchmark_rows = (
        prepared.loc[prepared["symbol"] == benchmark, ["timestamp", "close"]]
        .drop_duplicates("timestamp")
        .set_index("timestamp")["close"]
        .sort_index()
    )
    benchmark_slow = benchmark_rows.rolling(200, min_periods=200).mean()
    benchmark_fast = benchmark_rows.rolling(100, min_periods=100).mean()
    sessions = tuple(pd.Timestamp(value) for value in sorted(prepared["timestamp"].unique()))
    breadth_map = {
        pd.Timestamp(date).normalize(): float(value)
        for date, value in (breadth or {}).items()
        if value is not None and math.isfinite(float(value))
    }

    market: Dict[pd.Timestamp, Dict[str, MarketBar]] = {}
    observations: Dict[pd.Timestamp, RiskObservation] = {}
    indexed = prepared.set_index(["timestamp", "symbol"], drop=False).sort_index()
    for session in sessions:
        try:
            day = indexed.loc[session]
        except KeyError as exc:
            raise ValueError(f"no bars on {session}") from exc
        if isinstance(day, pd.Series):
            day = day.to_frame().T
        daily_bars: Dict[str, MarketBar] = {}
        betas: Dict[str, float] = {}
        for row in day.itertuples(index=False):
            symbol = str(row.symbol).upper()
            daily_bars[symbol] = MarketBar(
                open=float(row.open),
                close=float(row.close),
                adv_dollars=max(float(row.adv20), 1.0),
                daily_volatility=max(float(row.vol20), 0.0),
                spread_proxy_bps=max(float(row.spread_proxy_bps), 0.0),
                beta=1.0,
            )
            betas[symbol] = 1.0
        market[session] = daily_bars
        close = float(benchmark_rows.get(session, np.nan))
        slow = float(benchmark_slow.get(session, np.nan))
        fast = float(benchmark_fast.get(session, np.nan))
        observations[session] = RiskObservation(
            benchmark_close=close if math.isfinite(close) else None,
            benchmark_slow=slow if math.isfinite(slow) else None,
            benchmark_fast=fast if math.isfinite(fast) else None,
            breadth=breadth_map.get(session),
            betas=betas,
        )

    coverage = (
        prepared.groupby("symbol")["timestamp"]
        .agg(["min", "max", "nunique"])
        .rename(columns={"min": "first_session", "max": "last_session", "nunique": "sessions"})
        .reset_index()
        .to_dict("records")
    )
    return LETFMarketContext(
        market=market,
        risk_observations=observations,
        themes=theme_map,
        sessions=sessions,
        symbols=candidates,
        benchmark_symbol=benchmark,
        metadata={
            "dynamic_listing_membership": True,
            "held_missing_bar_policy": "fail_closed",
            "decision_execution_clock": "close(t) signal -> open(t+1) execution",
            "outer_gross_cap": execution.gross_target,
            "coverage": coverage,
        },
    )


def run_scenario(
    context: LETFMarketContext,
    signals: Mapping[object, Mapping[str, float]],
    *,
    config: Optional[LETFExecutionConfig] = None,
) -> ScenarioResult:
    execution = config or LETFExecutionConfig()
    if execution.breadth_exit is not None or execution.breadth_enter is not None:
        if not any(
            observation.breadth is not None
            for observation in context.risk_observations.values()
        ):
            raise ValueError(
                "breadth thresholds are enabled but no point-in-time breadth observations were supplied"
            )
    normalized_signals = {
        pd.Timestamp(date).normalize(): {
            str(symbol).upper(): float(score)
            for symbol, score in scores.items()
            if math.isfinite(float(score))
        }
        for date, scores in signals.items()
    }
    unknown = sorted(
        {
            symbol
            for scores in normalized_signals.values()
            for symbol in scores
            if symbol not in context.symbols
        }
    )
    if unknown:
        raise ValueError(f"signals contain non-candidate symbols: {unknown}")
    result = run_backtest(
        context.market,
        normalized_signals,
        context.themes,
        execution.portfolio_config(),
        execution.cost_config(),
        execution.risk_config(),
        context.risk_observations,
        initial_capital=execution.initial_capital,
    )
    assert_accounting_identity(result)
    first_execution = next(
        (index for index, row in enumerate(result.ledger) if row.executed_signal_session is not None),
        None,
    )
    if first_execution is None:
        raise ValueError("scenario never executed a signal")
    evaluation_ledger = tuple(result.ledger[first_execution:])
    metrics = compute_performance_metrics(evaluation_ledger)
    # A capacity-clipped execution can remain above the desired gross after
    # mark-to-market drift because the engine correctly refuses to pretend the
    # whole reduction filled.  Preserve that path and surface it as a failed
    # deployment/capacity gate instead of deleting the scenario or silently
    # scaling an unexecutable trade.
    return ScenarioResult(result, evaluation_ledger, metrics, first_execution, execution)


def make_buy_and_hold_signals(
    context: LETFMarketContext,
    symbol: str,
    *,
    signal_session: object,
) -> Mapping[pd.Timestamp, Mapping[str, float]]:
    ticker = str(symbol).upper()
    date = pd.Timestamp(signal_session).normalize()
    if ticker not in context.market.get(date, {}):
        raise ValueError(f"{ticker} has no bar on benchmark signal session {date.date()}")
    return {date: {ticker: 1.0}}


def run_buy_and_hold(
    context: LETFMarketContext,
    symbol: str,
    *,
    signal_session: object,
    extra_friction_bps: float = 10.0,
    target_volatility: Optional[float] = None,
) -> ScenarioResult:
    ticker = str(symbol).upper()
    if ticker not in context.themes:
        raise ValueError(f"benchmark {ticker} is absent from context")
    config = LETFExecutionConfig(
        top_k=1,
        weighting="equal",
        gross_target=1.0,
        single_name_cap=1.0,
        theme_cap=1.0,
        rank_buffer=0,
        no_trade_band=0.0,
        target_volatility=target_volatility,
        trend_filter=False,
        breadth_exit=None,
        breadth_enter=None,
        risk_off_multiplier=1.0,
        drawdown_steps=(),
        extra_friction_bps=extra_friction_bps,
    )
    # A benchmark may be in the context as a gauge rather than a candidate.
    benchmark_context = replace(
        context,
        symbols=tuple(dict.fromkeys((*context.symbols, ticker))),
    )
    return run_scenario(
        benchmark_context,
        make_buy_and_hold_signals(context, ticker, signal_session=signal_session),
        config=config,
    )


def daily_return_series(result: ScenarioResult) -> pd.Series:
    return pd.Series(
        {
            pd.Timestamp(row.session).normalize(): row.ending_equity / row.starting_equity - 1.0
            for row in result.ledger
        },
        dtype=float,
    ).sort_index()


def paired_realized_volatility_control(
    strategy: ScenarioResult,
    benchmark: ScenarioResult,
    *,
    periods_per_year: int = 252,
) -> Tuple[pd.DataFrame, Mapping[str, float | int | str]]:
    """Align returns and scale the benchmark to the strategy's realized risk.

    The scale is estimated on the exact shared evaluation sessions and is used
    only as an ex-post statistical control.  It is not a tradable signal and
    must not be described as a point-in-time target-vol portfolio.
    """

    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive")
    aligned = pd.concat(
        [
            daily_return_series(strategy).rename("strategy"),
            daily_return_series(benchmark).rename("benchmark_unscaled"),
        ],
        axis=1,
        join="inner",
    ).dropna()
    if len(aligned) < 2:
        raise ValueError("at least two aligned return sessions are required")
    strategy_std = float(aligned["strategy"].std(ddof=1))
    benchmark_std = float(aligned["benchmark_unscaled"].std(ddof=1))
    if not math.isfinite(strategy_std) or strategy_std < 0:
        raise ValueError("strategy realized volatility is invalid")
    if not math.isfinite(benchmark_std) or benchmark_std <= 0:
        raise ValueError("benchmark realized volatility must be positive and finite")
    scale = strategy_std / benchmark_std
    aligned["benchmark_control"] = aligned["benchmark_unscaled"] * scale
    if not np.isfinite(aligned.to_numpy(dtype=float)).all():
        raise ValueError("aligned volatility-control returns must be finite")
    if (aligned["benchmark_control"] <= -1.0).any():
        raise ValueError("volatility-control benchmark has a return at or below -100%")
    annualizer = math.sqrt(periods_per_year)
    metadata: Mapping[str, float | int | str] = {
        "comparison_basis": "shared_session_ex_post_realized_volatility_control",
        "aligned_sessions": int(len(aligned)),
        "benchmark_return_scale": float(scale),
        "strategy_annualized_volatility": strategy_std * annualizer,
        "benchmark_unscaled_annualized_volatility": benchmark_std * annualizer,
        "benchmark_control_annualized_volatility": (
            float(aligned["benchmark_control"].std(ddof=1)) * annualizer
        ),
    }
    return aligned, metadata


def paired_moving_block_bootstrap(
    strategy: ScenarioResult,
    benchmark: ScenarioResult,
    *,
    block_length: int = 21,
    repetitions: int = 5_000,
    seed: int = 20260720,
    realized_volatility_control: bool = False,
) -> Mapping[str, float | int | str]:
    """Paired circular moving-block bootstrap of daily excess returns.

    ``realized_volatility_control`` scales benchmark returns on the exact
    shared sample before resampling.  The resulting comparison is explicitly
    ex-post and evaluation-only.
    """

    if block_length < 2 or repetitions < 100:
        raise ValueError("block_length >= 2 and repetitions >= 100 are required")
    control_metadata: Mapping[str, float | int | str] = {}
    if realized_volatility_control:
        aligned, control_metadata = paired_realized_volatility_control(
            strategy, benchmark
        )
        benchmark_column = "benchmark_control"
    else:
        aligned = pd.concat(
            [
                daily_return_series(strategy).rename("strategy"),
                daily_return_series(benchmark).rename("benchmark"),
            ],
            axis=1,
            join="inner",
        ).dropna()
        benchmark_column = "benchmark"
    excess = (aligned["strategy"] - aligned[benchmark_column]).to_numpy(dtype=float)
    n = len(excess)
    if n < block_length:
        raise ValueError("insufficient aligned returns for block bootstrap")
    rng = np.random.default_rng(seed)
    blocks = math.ceil(n / block_length)
    offsets = np.arange(block_length)
    means = np.empty(repetitions, dtype=float)
    for repetition in range(repetitions):
        starts = rng.integers(0, n, size=blocks)
        positions = ((starts[:, None] + offsets[None, :]) % n).reshape(-1)[:n]
        means[repetition] = float(excess[positions].mean())
    return {
        **control_metadata,
        "aligned_sessions": n,
        "mean_daily_excess": float(excess.mean()),
        "annualized_mean_excess": float(excess.mean() * 252.0),
        "ci_2_5_annualized": float(np.quantile(means, 0.025) * 252.0),
        "ci_97_5_annualized": float(np.quantile(means, 0.975) * 252.0),
        "probability_outperform": float(np.mean(means > 0.0)),
        "repetitions": repetitions,
        "block_length": block_length,
        "seed": seed,
    }


def temporal_fold_metrics(
    result: ScenarioResult,
    *,
    minimum_sessions: int = 126,
) -> Sequence[Mapping[str, Any]]:
    """Calendar-year OOS diagnostics without resetting portfolio state."""

    grouped: Dict[int, list[Any]] = {}
    for row in result.ledger:
        grouped.setdefault(pd.Timestamp(row.session).year, []).append(row)
    rows: list[Mapping[str, Any]] = []
    for year, ledger in sorted(grouped.items()):
        if len(ledger) < minimum_sessions:
            continue
        rows.append({"year": year, **compute_performance_metrics(ledger).to_dict()})
    return rows


def cost_sensitivity(
    context: LETFMarketContext,
    signals: Mapping[object, Mapping[str, float]],
    base_config: LETFExecutionConfig,
    *,
    friction_bps: Sequence[float] = (0.0, 5.0, 10.0, 20.0, 40.0),
) -> Sequence[Mapping[str, Any]]:
    rows: list[Mapping[str, Any]] = []
    for value in friction_bps:
        scenario = run_scenario(
            context,
            signals,
            config=replace(base_config, extra_friction_bps=float(value)),
        )
        rows.append(
            {
                "extra_friction_bps": float(value),
                "metrics": scenario.metrics.to_dict(),
            }
        )
    return rows


def scenario_audit_payload(result: ScenarioResult) -> Mapping[str, Any]:
    """JSON-safe core audit fields for immutable run artifacts."""

    executions = sum(1 for row in result.ledger if row.executed_signal_session is not None)
    alpha_signal_days = sum(1 for row in result.ledger if row.signal_generated)
    held_days = sum(1 for row in result.ledger if row.gross_exposure > 1e-12)
    executed_target_gross = [
        sum(abs(float(weight)) for weight in row.executed_target_weights.values())
        for row in result.ledger
        if row.executed_signal_session is not None
    ]
    return {
        "config": asdict(result.config),
        "metrics": result.metrics.to_dict(),
        "alpha_signal_days": alpha_signal_days,
        "executions": executions,
        "occupancy": held_days / len(result.ledger) if result.ledger else 0.0,
        "max_executed_target_gross": max(executed_target_gross, default=0.0),
        "executions_above_one_gross": sum(
            value > 1.0 + 1e-9 for value in executed_target_gross
        ),
        "first_execution_session": str(result.ledger[0].session),
        "last_session": str(result.ledger[-1].session),
        "pending_signal_at_end": (
            None
            if result.backtest.final_state.pending is None
            else str(result.backtest.final_state.pending.signal_session)
        ),
        "accounting_identity": {
            "max_value_error": result.metrics.max_value_identity_error,
            "max_pnl_error": result.metrics.max_pnl_identity_error,
        },
    }


__all__ = [
    "LETFExecutionConfig",
    "LETFMarketContext",
    "ScenarioResult",
    "build_market_context",
    "cost_sensitivity",
    "daily_return_series",
    "load_verified_snapshot_metadata",
    "load_snapshot_bars",
    "make_buy_and_hold_signals",
    "paired_moving_block_bootstrap",
    "paired_realized_volatility_control",
    "run_buy_and_hold",
    "run_scenario",
    "scenario_audit_payload",
    "temporal_fold_metrics",
]
