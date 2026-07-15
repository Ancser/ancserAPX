"""Point-in-time feature and label construction for Research v2.

The research clock is explicit:

    close(t) information -> execute at open(t+1) -> exit at open(t+1+h)

No feature column uses a negative shift.  Negative shifts are confined to the
label block and are guarded by global session indices so a missing bar cannot
silently turn a five-session label into an arbitrary calendar-period return.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import polars as pl


REQUIRED_COLUMNS = (
    "timestamp", "symbol", "open", "high", "low", "close",
    "volume", "vwap", "trade_count",
)

V2_RAW_FEATURES = (
    "mom_6_1_exact_v2",
    "mom_9_1_exact_v2",
    "mom_12_1_exact_v2",
    "reversal_5d_v2",
    "reversal_21d_v2",
    "realized_vol_20d_v2",
    "realized_vol_63d_v2",
    "overnight_gap_v2",
    "intraday_return_v2",
    "range_pct_v2",
    "close_vwap_distance_v2",
    "log_dollar_volume_v2",
    "volume_z_63d_v2",
)

REGIME_FEATURES = (
    "market_proxy_return_v2",
    "market_proxy_vol_20d_v2",
    "cross_section_dispersion_v2",
    "breadth_50d_v2",
    "breadth_200d_v2",
)

BASELINE_SCORE_COLUMNS = (
    "score_production_claude1",
    "score_production_claude3",
    "score_momentum_ensemble_v2",
    "score_momentum_lowvol_v2",
)


@dataclass(frozen=True)
class FeatureBuildResult:
    panel: pl.DataFrame
    feature_columns: Tuple[str, ...]
    report: Dict[str, object]


def _assert_schema(df: pl.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Snapshot is missing required columns: {missing}")


def validate_panel(
    raw: pl.DataFrame,
    *,
    start_date: str,
    end_date: str,
    min_cross_section: int,
    invalid_row_policy: str = "raise",
) -> Tuple[pl.DataFrame, Dict[str, object]]:
    """Validate OHLCV invariants and drop globally sparse sessions.

    Bad prices and duplicate symbol/session rows fail closed.  Sparse sessions
    are reported and removed because a 500-name model trained on a 160-name
    accidental cross-section would learn data-vendor outages as a market state.
    """
    _assert_schema(raw)
    df = raw.select(REQUIRED_COLUMNS).with_columns([
        pl.col("timestamp").cast(pl.Datetime("ns")),
        pl.col("symbol").cast(pl.Utf8),
    ]).sort(["symbol", "timestamp"])

    start = pd.Timestamp(start_date).to_pydatetime()
    end = (pd.Timestamp(end_date) + pd.Timedelta(days=1)).to_pydatetime()
    df = df.filter((pl.col("timestamp") >= start) & (pl.col("timestamp") < end))
    if df.is_empty():
        raise ValueError("Snapshot has no rows inside the configured date range")

    duplicate_rows = (
        df.group_by(["timestamp", "symbol"]).len().filter(pl.col("len") > 1).height
    )
    if duplicate_rows:
        raise ValueError(f"Snapshot contains {duplicate_rows} duplicate symbol/session keys")

    invalid_mask = (
        (pl.col("open") <= 0)
        | (pl.col("high") <= 0)
        | (pl.col("low") <= 0)
        | (pl.col("close") <= 0)
        | (pl.col("volume") < 0)
        | (pl.col("high") < pl.max_horizontal("open", "close", "low"))
        | (pl.col("low") > pl.min_horizontal("open", "close", "high"))
    )
    invalid = df.filter(invalid_mask)
    if invalid.height:
        examples = invalid.select(["timestamp", "symbol", "open", "high", "low", "close"]).head(5).to_dicts()
        if invalid_row_policy == "raise":
            raise ValueError(f"Snapshot contains {invalid.height} invalid OHLCV rows; examples={examples}")
        if invalid_row_policy != "drop":
            raise ValueError("invalid_row_policy must be 'raise' or 'drop'")
        df = df.filter(~invalid_mask)
    else:
        examples = []

    coverage = df.group_by("timestamp").agg(pl.col("symbol").n_unique().alias("symbols")).sort("timestamp")
    sparse = coverage.filter(pl.col("symbols") < int(min_cross_section))
    good_dates = coverage.filter(pl.col("symbols") >= int(min_cross_section)).select("timestamp")
    df = df.join(good_dates, on="timestamp", how="inner")

    sessions = (
        df.select("timestamp").unique().sort("timestamp")
        .with_row_index("session_idx")
    )
    df = df.join(sessions, on="timestamp", how="left").sort(["symbol", "timestamp"])
    report: Dict[str, object] = {
        "input_rows": int(raw.height),
        "validated_rows": int(df.height),
        "symbols": int(df["symbol"].n_unique()),
        "sessions": int(sessions.height),
        "start": str(df["timestamp"].min()),
        "end": str(df["timestamp"].max()),
        "min_cross_section": int(min_cross_section),
        "invalid_row_policy": invalid_row_policy,
        "dropped_invalid_rows": int(invalid.height),
        "invalid_row_examples": [
            {k: str(v) if k == "timestamp" else v for k, v in row.items()}
            for row in examples
        ],
        "dropped_sparse_sessions": [
            {"timestamp": str(r["timestamp"]), "symbols": int(r["symbols"])}
            for r in sparse.to_dicts()
        ],
    }
    return df, report


def _centered_rank_expr(column: str, group: str = "timestamp") -> pl.Expr:
    n = pl.col(column).count().over(group)
    rank = pl.col(column).rank(method="average").over(group)
    return (
        pl.when(n > 1)
        .then((rank - 1.0) / (n - 1.0) - 0.5)
        .otherwise(None)
    )


def _safe_ratio(numerator: pl.Expr, denominator: pl.Expr) -> pl.Expr:
    return pl.when(denominator.abs() > 1e-12).then(numerator / denominator).otherwise(None)


def build_feature_panel(
    raw: pl.DataFrame,
    *,
    start_date: str = "2020-07-27",
    end_date: str = "2026-07-09",
    min_cross_section: int = 350,
    min_symbol_history: int = 252,
    label_horizon: int = 5,
    invalid_row_policy: str = "drop",
) -> FeatureBuildResult:
    if label_horizon < 1:
        raise ValueError("label_horizon must be positive")
    clean, report = validate_panel(
        raw,
        start_date=start_date,
        end_date=end_date,
        min_cross_section=min_cross_section,
        invalid_row_policy=invalid_row_policy,
    )

    # The production factor function is a pure computation and is used only as
    # a versioned legacy baseline.  No store/fetch/execution module is imported.
    from backend.alpha.factors import RUNTIME_FACTOR_META, compute_all_factors

    df = compute_all_factors(clean.lazy()).collect().sort(["symbol", "timestamp"])
    log_volume = pl.col("volume").clip(lower_bound=0).log1p()
    prev_close = pl.col("close").shift(1).over("symbol")
    df = df.with_columns([
        (_safe_ratio(pl.col("close").shift(21).over("symbol"), pl.col("close").shift(126).over("symbol")) - 1.0)
        .alias("mom_6_1_exact_v2"),
        (_safe_ratio(pl.col("close").shift(21).over("symbol"), pl.col("close").shift(189).over("symbol")) - 1.0)
        .alias("mom_9_1_exact_v2"),
        (_safe_ratio(pl.col("close").shift(21).over("symbol"), pl.col("close").shift(252).over("symbol")) - 1.0)
        .alias("mom_12_1_exact_v2"),
        (1.0 - _safe_ratio(pl.col("close"), pl.col("close").shift(5).over("symbol"))).alias("reversal_5d_v2"),
        (1.0 - _safe_ratio(pl.col("close"), pl.col("close").shift(21).over("symbol"))).alias("reversal_21d_v2"),
        pl.col("returns").rolling_std(20).over("symbol").alias("realized_vol_20d_v2"),
        pl.col("returns").rolling_std(63).over("symbol").alias("realized_vol_63d_v2"),
        (_safe_ratio(pl.col("open"), prev_close) - 1.0).alias("overnight_gap_v2"),
        (_safe_ratio(pl.col("close"), pl.col("open")) - 1.0).alias("intraday_return_v2"),
        _safe_ratio(pl.col("high") - pl.col("low"), pl.col("close")).alias("range_pct_v2"),
        (_safe_ratio(pl.col("close"), pl.col("vwap")) - 1.0).alias("close_vwap_distance_v2"),
        (pl.col("close") * pl.col("volume")).clip(lower_bound=0).log1p().alias("log_dollar_volume_v2"),
        (
            (log_volume - log_volume.rolling_mean(63).over("symbol"))
            / (log_volume.rolling_std(63).over("symbol") + 1e-8)
        ).alias("volume_z_63d_v2"),
        (pl.col("close") * pl.col("volume")).rolling_mean(20).over("symbol").alias("adv20_v2"),
        pl.col("close").rolling_mean(50).over("symbol").alias("sma50_v2"),
        pl.col("close").rolling_mean(200).over("symbol").alias("sma200_v2"),
        pl.col("close").cum_count().over("symbol").alias("history_bars"),
    ])

    market = (
        df.group_by("timestamp").agg([
            pl.col("returns").mean().alias("market_proxy_return_v2"),
            pl.col("returns").std().alias("cross_section_dispersion_v2"),
            (pl.col("close") > pl.col("sma50_v2")).mean().alias("breadth_50d_v2"),
            (pl.col("close") > pl.col("sma200_v2")).mean().alias("breadth_200d_v2"),
        ])
        .sort("timestamp")
        .with_columns(
            pl.col("market_proxy_return_v2").rolling_std(20).alias("market_proxy_vol_20d_v2")
        )
    )
    df = df.join(market, on="timestamp", how="left")

    legacy_cols = tuple(dict.fromkeys(
        meta["col"]
        for meta in RUNTIME_FACTOR_META.values()
        if meta.get("col") in df.columns
    ))
    rank_sources = legacy_cols + V2_RAW_FEATURES
    df = df.with_columns([
        _centered_rank_expr(c).alias(f"cs_rank__{c}") for c in rank_sources
    ])

    def oriented(name: str) -> pl.Expr:
        meta = RUNTIME_FACTOR_META[name]
        expr = pl.col(f"cs_rank__{meta['col']}")
        return -expr if bool(meta.get("descending")) else expr

    mom_ens = (
        pl.col("cs_rank__mom_6_1_exact_v2")
        + pl.col("cs_rank__mom_9_1_exact_v2")
        + pl.col("cs_rank__mom_12_1_exact_v2")
    ) / 3.0
    defensive = 0.60 * oriented("Volatility") + 0.40 * oriented("Reversion")
    df = df.with_columns([
        (0.70 * oriented("Momentum") + 0.30 * oriented("Reversion")).alias("score_production_claude1"),
        (0.70 * oriented("Momentum") + 0.30 * defensive).alias("score_production_claude3"),
        mom_ens.alias("score_momentum_ensemble_v2"),
        (0.75 * mom_ens + 0.25 * oriented("Volatility")).alias("score_momentum_lowvol_v2"),
    ])

    # Future values exist only below this line and only in label/audit columns.
    entry_open = pl.col("open").shift(-1).over("symbol")
    exit_open = pl.col("open").shift(-(label_horizon + 1)).over("symbol")
    entry_idx = pl.col("session_idx").shift(-1).over("symbol")
    exit_idx = pl.col("session_idx").shift(-(label_horizon + 1)).over("symbol")
    execution_ts = pl.col("timestamp").shift(-1).over("symbol")
    valid_clock = (
        (entry_idx == pl.col("session_idx") + 1)
        & (exit_idx == pl.col("session_idx") + label_horizon + 1)
    )
    df = df.with_columns([
        pl.when(valid_clock).then(execution_ts).otherwise(None).alias("execution_timestamp"),
        pl.when(valid_clock).then(entry_open).otherwise(None).alias("label_entry_open"),
        pl.when(valid_clock).then(exit_open).otherwise(None).alias("label_exit_open"),
        pl.when(valid_clock)
        .then(_safe_ratio(exit_open, entry_open) - 1.0)
        .otherwise(None)
        .alias(f"label_open_to_open_{label_horizon}d_raw"),
    ])
    raw_label = f"label_open_to_open_{label_horizon}d_raw"
    df = df.with_columns(
        (pl.col(raw_label) - pl.col(raw_label).median().over("timestamp")).alias("label_residual")
    )
    df = df.with_columns(
        _centered_rank_expr("label_residual").alias("label_rank")
    )
    label_count = pl.col("label_rank").count().over("timestamp")
    df = df.with_columns([
        pl.when(label_count > 0).then(1.0 / label_count).otherwise(None).alias("sample_weight"),
        (
            (pl.col("history_bars") >= int(min_symbol_history))
            & pl.col("label_rank").is_not_null()
            & pl.col("execution_timestamp").is_not_null()
        ).alias("model_eligible"),
    ])

    # Turn non-finite floats into nulls so every downstream model handles them
    # explicitly rather than inheriting platform-specific +/-inf behavior.
    float_cols = [c for c, dt in df.schema.items() if dt in (pl.Float32, pl.Float64)]
    df = df.with_columns([
        pl.when(pl.col(c).is_finite()).then(pl.col(c)).otherwise(None).alias(c)
        for c in float_cols
    ])

    feature_columns = tuple(f"cs_rank__{c}" for c in rank_sources) + REGIME_FEATURES
    report.update({
        "label_horizon": int(label_horizon),
        "decision_clock": "close(t) -> open(t+1) execution",
        "label_clock": f"open(t+1) -> open(t+{label_horizon + 1})",
        "legacy_feature_count": len(legacy_cols),
        "model_feature_count": len(feature_columns),
        "eligible_rows": int(df.filter(pl.col("model_eligible")).height),
        "eligible_start": str(df.filter(pl.col("model_eligible"))["timestamp"].min()),
        "eligible_end": str(df.filter(pl.col("model_eligible"))["timestamp"].max()),
        "market_proxy": "equal-weight local-universe return; SPY/QQQ absent from canonical store",
    })
    return FeatureBuildResult(df.sort(["timestamp", "symbol"]), feature_columns, report)


def load_snapshot(path: Path | str) -> pl.DataFrame:
    p = Path(path)
    if p.is_dir():
        store_dir = p / "store" if (p / "store").is_dir() else p
        files = sorted(store_dir.rglob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"No Parquet files under {store_dir}")
        return pl.scan_parquet([str(f) for f in files]).collect()
    if p.is_file():
        return pl.read_parquet(p)
    raise FileNotFoundError(p)


def write_feature_artifacts(
    result: FeatureBuildResult,
    panel_path: Path | str,
    report_path: Path | str,
) -> None:
    """Write generated research artifacts; path confinement belongs to CLI."""
    Path(panel_path).parent.mkdir(parents=True, exist_ok=True)
    result.panel.write_parquet(panel_path, compression="zstd")
    Path(report_path).write_text(
        json.dumps(result.report, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def to_model_frame(result: FeatureBuildResult | pl.DataFrame, feature_columns: Sequence[str] | None = None) -> pd.DataFrame:
    if isinstance(result, FeatureBuildResult):
        df = result.panel
        cols = list(result.feature_columns)
    else:
        df = result
        if feature_columns is None:
            raise ValueError("feature_columns are required when passing a DataFrame")
        cols = list(feature_columns)
    required = ["timestamp", "execution_timestamp", "symbol", "label_rank", "sample_weight", "model_eligible"]
    missing = [c for c in required + cols if c not in df.columns]
    if missing:
        raise ValueError(f"Feature panel missing model columns: {missing}")
    return df.filter(pl.col("model_eligible")).select(required + cols + list(BASELINE_SCORE_COLUMNS)).to_pandas()
