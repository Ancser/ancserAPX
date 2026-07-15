"""Resumable entry point for the definitive Research v2 sequence run.

This module intentionally reads only the frozen Research v2 feature cache and
writes only below ``research_v2/runs``.  It exists separately from the normal
daily runner so a multi-hour GPU job can be stopped and resumed fold by fold
without importing any broker, OMS, scheduler, or mutable production store.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Mapping


ROOT = Path(__file__).resolve().parent
DEFAULT_CACHE = ROOT / "cache" / "canonical_features_h5.parquet"
DEFAULT_OUTPUT = ROOT / "runs" / "20260710_full_v1" / "sequence_full60_all"

# Keep the definitive run comparable with the completed sequence screen.  The
# list contains price/volume-derived ranks plus market-regime state, not future
# data.  A raw-OHLCV end-to-end experiment is a separate model family.
SEQUENCE_FEATURES = (
    "cs_rank__overnight_gap_v2",
    "cs_rank__intraday_return_v2",
    "cs_rank__range_pct_v2",
    "cs_rank__close_vwap_distance_v2",
    "cs_rank__volume_z_63d_v2",
    "cs_rank__realized_vol_20d_v2",
    "cs_rank__realized_vol_63d_v2",
    "cs_rank__mom_6_1_exact_v2",
    "cs_rank__mom_9_1_exact_v2",
    "cs_rank__mom_12_1_exact_v2",
    "market_proxy_return_v2",
    "market_proxy_vol_20d_v2",
    "cross_section_dispersion_v2",
    "breadth_50d_v2",
    "breadth_200d_v2",
)

# Low-level, point-in-time normalization is unavoidable when one network sees
# both a $10 stock and a $1,000 stock.  These channels are derived only from the
# current/previous bar and trailing scale statistics; they deliberately omit
# momentum, RSI, volatility, breadth, and other hand-designed alpha factors.
RAW_SEQUENCE_FEATURES = (
    "raw_log_open_prev_close",
    "raw_log_high_prev_close",
    "raw_log_low_prev_close",
    "raw_log_close_prev_close",
    "raw_log_vwap_prev_close",
    "raw_log_volume_z63",
    "raw_log_trade_count_z63",
    "raw_log_dollar_volume_z63",
    "raw_close_location",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _append_jsonl(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"utc": _utc_now(), **dict(payload)}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True, default=str) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    print(json.dumps(record, sort_keys=True, default=str), flush=True)


def _atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument(
        "--family",
        choices=("engineered", "raw"),
        default="engineered",
        help="engineered factor sequence or normalized low-level OHLCV sequence",
    )
    return parser.parse_args(argv)


def _raw_feature_frame(panel):
    """Create scale-stable OHLCV channels using trailing information only."""

    import numpy as np
    import pandas as pd

    frame = panel.to_pandas().sort_values(
        ["symbol", "timestamp"], kind="mergesort"
    ).reset_index(drop=True)
    groups = frame.groupby("symbol", sort=False, observed=True)
    previous_close = groups["close"].shift(1)
    safe_previous = previous_close.where(previous_close > 0)
    for source, target in (
        ("open", "raw_log_open_prev_close"),
        ("high", "raw_log_high_prev_close"),
        ("low", "raw_log_low_prev_close"),
        ("close", "raw_log_close_prev_close"),
        ("vwap", "raw_log_vwap_prev_close"),
    ):
        ratio = pd.to_numeric(frame[source], errors="coerce") / safe_previous
        frame[target] = np.log(ratio.where(ratio > 0)).clip(-0.50, 0.50)

    raw_scales = {
        "raw_log_volume_z63": np.log1p(
            pd.to_numeric(frame["volume"], errors="coerce").clip(lower=0)
        ),
        "raw_log_trade_count_z63": np.log1p(
            pd.to_numeric(frame["trade_count"], errors="coerce").clip(lower=0)
        ),
        "raw_log_dollar_volume_z63": np.log1p(
            (
                pd.to_numeric(frame["close"], errors="coerce")
                * pd.to_numeric(frame["volume"], errors="coerce")
            ).clip(lower=0)
        ),
    }
    symbol_values = frame["symbol"]
    for target, raw in raw_scales.items():
        by_symbol = raw.groupby(symbol_values, sort=False)
        trailing_mean = by_symbol.transform(
            lambda values: values.rolling(63, min_periods=20).mean()
        )
        trailing_std = by_symbol.transform(
            lambda values: values.rolling(63, min_periods=20).std(ddof=0)
        )
        frame[target] = ((raw - trailing_mean) / trailing_std.replace(0.0, np.nan)).clip(
            -8.0, 8.0
        )

    high = pd.to_numeric(frame["high"], errors="coerce")
    low = pd.to_numeric(frame["low"], errors="coerce")
    close = pd.to_numeric(frame["close"], errors="coerce")
    span = (high - low).where((high - low) > 0)
    frame["raw_close_location"] = ((close - low) / span - 0.5).clip(-0.5, 0.5)
    return frame


def main(argv: list[str] | None = None) -> int:
    args = _arguments(argv)

    # Import the research stack only after entering the process-local offline
    # boundary.  No environment mutation escapes this subprocess.
    from research_v2.safety import ensure_research_output_path, offline_context

    cache = args.cache.expanduser().resolve(strict=True)
    output = ensure_research_output_path(args.output, research_root=ROOT)
    output.mkdir(parents=True, exist_ok=True)
    progress_path = output / "progress.jsonl"

    with offline_context():
        import pandas as pd
        import polars as pl
        import torch

        from research_v2.experiment import complete_case_symbols
        from research_v2.sequence_pipeline import SequencePipelineSettings
        from research_v2.sequence_training import run_checkpointed_sequence_research
        from research_v2.validation import (
            make_purged_walk_forward,
            prediction_diagnostics,
        )

        started = time.perf_counter()
        _append_jsonl(
            progress_path,
            {
                "event": "process_started",
                "pid": os.getpid(),
                "cache": str(cache),
                "cache_sha256": _sha256(cache),
                "output": str(output),
                "torch": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "device": args.device,
            },
        )
        if str(args.device).startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")

        common_required = [
            "timestamp",
            "execution_timestamp",
            "symbol",
            "label_rank",
            "label_residual",
            "sample_weight",
            "model_eligible",
        ]
        family_required = (
            list(SEQUENCE_FEATURES)
            if args.family == "engineered"
            else ["open", "high", "low", "close", "volume", "vwap", "trade_count"]
        )
        required = [*common_required, *family_required]
        lazy = pl.scan_parquet(cache)
        schema_names = lazy.collect_schema().names()
        missing = sorted(set(required) - set(schema_names))
        if missing:
            raise ValueError(f"feature cache is missing sequence columns: {missing}")
        panel_pl = lazy.select(required).collect()
        symbols = complete_case_symbols(panel_pl.select("timestamp", "symbol"))
        panel_pl = panel_pl.filter(pl.col("symbol").is_in(symbols))
        eligible_dates = (
            panel_pl.filter(pl.col("model_eligible"))
            .select("timestamp")
            .unique()
            .sort("timestamp")
            .to_series()
            .to_list()
        )
        if args.family == "engineered":
            feature_columns = SEQUENCE_FEATURES
            frame = panel_pl.select(
                "timestamp",
                "execution_timestamp",
                "symbol",
                "label_rank",
                "label_residual",
                "sample_weight",
                *feature_columns,
            ).to_pandas()
            model_seed = 20260710
        else:
            feature_columns = RAW_SEQUENCE_FEATURES
            raw = _raw_feature_frame(panel_pl)
            frame = raw.loc[
                :,
                [
                    "timestamp",
                    "execution_timestamp",
                    "symbol",
                    "label_rank",
                    "label_residual",
                    "sample_weight",
                    *feature_columns,
                ],
            ].copy()
            model_seed = 20260711
        folds = make_purged_walk_forward(
            eligible_dates,
            train_days=504,
            validation_days=63,
            test_days=63,
            purge_days=5,
            embargo_days=5,
            step_days=63,
            label_horizon=5,
            rolling_train=True,
            selection_end="2025-12-31",
        )
        frame = frame.sort_values(
            ["timestamp", "symbol"], kind="mergesort"
        ).reset_index(drop=True)
        settings = SequencePipelineSettings(
            sequence_length=60,
            gru_hidden_dim=32,
            gru_layers=1,
            transformer_d_model=32,
            transformer_heads=4,
            transformer_layers=2,
            transformer_feedforward=64,
            dropout=0.0,
            epochs=args.epochs,
            batch_size=args.batch_size,
            max_train_samples=None,
            max_parameters=500_000,
            learning_rate=1e-3,
            weight_decay=1e-4,
            patience=args.patience,
            random_seed=model_seed,
            device=args.device,
            ensemble_shrinkage=0.5,
            ensemble_single_model_cap=0.75,
            ensemble_grid_increment=0.25,
            minimum_cross_section=20,
        )
        _append_jsonl(
            progress_path,
            {
                "event": "training_plan_locked",
                "rows": len(frame),
                "symbols": len(symbols),
                "folds": len(folds),
                "family": args.family,
                "features": list(feature_columns),
                "settings": asdict(settings),
                "selection_end": "2025-12-31",
                "lockbox_start": "2026-01-01",
                "endpoint_sampling": "complete_date",
            },
        )

        def progress(event: Mapping[str, object]) -> None:
            _append_jsonl(progress_path, event)

        result = run_checkpointed_sequence_research(
            frame,
            feature_columns,
            folds,
            output_dir=output,
            selection_end="2025-12-31",
            lockbox_start="2026-01-01",
            embargo_days=5,
            settings=settings,
            endpoint_sampling="complete_date",
            research_root=ROOT,
            progress_callback=progress,
        )
        score_columns = (
            "score_gru",
            "score_transformer",
            "score_sequence_equal",
            "score_sequence_locked",
        )
        summary = {
            "elapsed_seconds": time.perf_counter() - started,
            "cache": str(cache),
            "cache_sha256": _sha256(cache),
            "rows": len(frame),
            "complete_case_symbols": len(symbols),
            "family": args.family,
            "features": list(feature_columns),
            "folds": [fold.as_dict() for fold in folds],
            "settings": asdict(settings),
            "locked_settings": result.locked_settings,
            "selection_rows": len(result.selection_predictions),
            "lockbox_rows": len(result.lockbox_predictions),
            "selection_diagnostics": {
                name: prediction_diagnostics(
                    result.selection_predictions,
                    prediction_col=name,
                )
                for name in score_columns
            },
            "lockbox_diagnostics": {
                name: prediction_diagnostics(
                    result.lockbox_predictions,
                    prediction_col=name,
                )
                for name in score_columns
            },
        }
        _atomic_json(output / "summary.json", summary)
        _append_jsonl(
            progress_path,
            {
                "event": "summary_published",
                "summary": str(output / "summary.json"),
                "elapsed_seconds": summary["elapsed_seconds"],
            },
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        # stderr is captured by the background launcher.  Checkpoints completed
        # before the exception remain resumable and are never silently trusted.
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise
