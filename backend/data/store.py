"""
Local Parquet data store.

Layout:
  data/store/<SYMBOL>.parquet   — full OHLCV history per symbol
  data/manifest.json            — {symbol: {last_date, row_count}}

All timestamps stored as naive UTC (timezone-stripped on ingest).
"""

import json
import polars as pl
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Dict, Optional

_ROOT = Path(__file__).resolve().parents[2]  # F:\ancserQuant\ancserAPX
STORE_DIR = _ROOT / "data" / "store"
MANIFEST_PATH = _ROOT / "data" / "manifest.json"


def _ensure_dirs():
    STORE_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------------
# Manifest
# ------------------------------------------------------------------

def get_manifest() -> Dict:
    if not MANIFEST_PATH.exists():
        return {}
    try:
        return json.loads(MANIFEST_PATH.read_text())
    except Exception:
        return {}


def _save_manifest(manifest: Dict):
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))


# ------------------------------------------------------------------
# Write
# ------------------------------------------------------------------

def _strip_tz(df: pl.DataFrame) -> pl.DataFrame:
    """Convert any timezone-aware timestamp column to naive UTC."""
    if "timestamp" in df.columns:
        ts = df["timestamp"]
        if ts.dtype == pl.Datetime("us", "UTC") or (hasattr(ts.dtype, "time_zone") and ts.dtype.time_zone):
            df = df.with_columns(pl.col("timestamp").dt.convert_time_zone("UTC").dt.replace_time_zone(None))
    return df


def save(df: pl.DataFrame, symbol: str):
    """Append-and-deduplicate a symbol's data. Creates file if absent."""
    _ensure_dirs()
    df = _strip_tz(df)
    path = STORE_DIR / f"{symbol}.parquet"
    if path.exists():
        existing = pl.read_parquet(path)
        existing = _strip_tz(existing)
        combined = pl.concat([existing, df]).unique(subset=["timestamp"]).sort("timestamp")
    else:
        combined = df.sort("timestamp")

    combined.write_parquet(path, compression="snappy")

    # Update manifest
    last_ts = combined["timestamp"].max()
    last_date = (
        last_ts.strftime("%Y-%m-%d")
        if isinstance(last_ts, datetime)
        else str(last_ts)[:10]
    )
    manifest = get_manifest()
    manifest[symbol] = {"last_date": last_date, "row_count": len(combined)}
    _save_manifest(manifest)


# ------------------------------------------------------------------
# Read
# ------------------------------------------------------------------

def load(symbols: List[str], start_date: str, end_date: str) -> pl.LazyFrame:
    """Return a LazyFrame for the requested symbols and date window."""
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    frames: List[pl.DataFrame] = []

    for sym in symbols:
        path = STORE_DIR / f"{sym}.parquet"
        if not path.exists():
            continue
        try:
            df = pl.read_parquet(path)
            df = _strip_tz(df)
            # Ensure timestamp column is Datetime (not Date)
            if df["timestamp"].dtype == pl.Date:
                df = df.with_columns(pl.col("timestamp").cast(pl.Datetime("us")))
            df = df.filter(
                (pl.col("timestamp") >= start_dt) & (pl.col("timestamp") <= end_dt)
            )
            if not df.is_empty():
                frames.append(df)
        except Exception as e:
            print(f"[store] Failed reading {sym}: {e}")

    if not frames:
        return pl.LazyFrame()
    return pl.concat(frames).lazy()


def has_symbol(symbol: str) -> bool:
    return (STORE_DIR / f"{symbol}.parquet").exists()


# ------------------------------------------------------------------
# Coverage stats
# ------------------------------------------------------------------

def get_coverage_stats(all_symbols: List[str]) -> Dict:
    manifest = get_manifest()
    covered = [s for s in all_symbols if s in manifest]
    missing = [s for s in all_symbols if s not in manifest]
    dates = [v["last_date"] for v in manifest.values() if "last_date" in v]
    total_rows = sum(v.get("row_count", 0) for v in manifest.values())
    return {
        "total_symbols": len(all_symbols),
        "covered": len(covered),
        "missing_count": len(missing),
        "coverage_pct": round(len(covered) / len(all_symbols) * 100, 1) if all_symbols else 0.0,
        "last_update": max(dates) if dates else None,
        "earliest_date": None,  # Could scan manifest for min
        "total_rows": total_rows,
        "missing_symbols": missing[:30],
    }
