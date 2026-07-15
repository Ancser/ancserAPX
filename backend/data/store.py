"""
Local Parquet data store.

Layout:
  data/store/<SYMBOL>.parquet   — full OHLCV history per symbol
  data/manifest.json            — {symbol: {last_date, row_count}}

All timestamps stored as naive UTC (timezone-stripped on ingest).
"""

import json
import math
import os
import time
import uuid
import polars as pl
from contextlib import contextmanager
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional

_ROOT = Path(__file__).resolve().parents[2]  # F:\ancserQuant\ancserAPX
STORE_DIR = _ROOT / "data" / "store"
MANIFEST_PATH = _ROOT / "data" / "manifest.json"
STORE_LOCK_PATH = _ROOT / "data" / ".store-write.lock"


def _ensure_dirs():
    STORE_DIR.mkdir(parents=True, exist_ok=True)


@contextmanager
def _store_write_lock(timeout_seconds: float = 30.0):
    """Small cross-process lock for parquet + manifest atomic updates.

    The web server and Windows scheduled runner are separate processes. Without
    a lock, simultaneous syncs could overwrite each other's manifest entries or
    read/replace the same symbol file concurrently.
    """
    # Follow a patched/test manifest location while retaining the production
    # constant for normal operation.
    lock_path = (
        MANIFEST_PATH.parent / ".store-write.lock"
        if MANIFEST_PATH.parent != (_ROOT / "data") else STORE_LOCK_PATH
    )
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + timeout_seconds
    fd = None
    while fd is None:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, f"pid={os.getpid()} created={datetime.now(timezone.utc).isoformat()}".encode())
        except FileExistsError:
            try:
                # A crashed process must not block all future live runs forever.
                if time.time() - lock_path.stat().st_mtime > 300:
                    lock_path.unlink(missing_ok=True)
                    continue
            except FileNotFoundError:
                continue
            if time.monotonic() >= deadline:
                raise TimeoutError("Timed out waiting for the market-data store write lock")
            time.sleep(0.05)
    try:
        yield
    finally:
        try:
            os.close(fd)
        finally:
            lock_path.unlink(missing_ok=True)


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
    temp = MANIFEST_PATH.with_name(f"{MANIFEST_PATH.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        temp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        os.replace(temp, MANIFEST_PATH)
    finally:
        temp.unlink(missing_ok=True)


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
    with _store_write_lock():
        if path.exists():
            existing = pl.read_parquet(path)
            existing = _strip_tz(existing)
            combined = pl.concat([existing, df]).unique(subset=["timestamp"]).sort("timestamp")
        else:
            combined = df.sort("timestamp")

        temp_path = path.with_name(f"{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
        try:
            combined.write_parquet(temp_path, compression="snappy")
            os.replace(temp_path, path)
        finally:
            temp_path.unlink(missing_ok=True)

        # Update the manifest while holding the same lock, so another sync
        # cannot lose this symbol's entry between read and atomic replace.
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
    """Return rows in the inclusive *date* window requested by callers.

    Daily Alpaca bars can carry a non-midnight timestamp (for example 04:00).
    Comparing them to ``end_date 00:00`` previously discarded the entire final
    session and made live selection appear one trading day stale.  Use a
    half-open interval ending at midnight of the following day instead.
    """
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_exclusive = datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1)
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
                (pl.col("timestamp") >= start_dt) & (pl.col("timestamp") < end_exclusive)
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


def inspect_physical_data(
    symbols: List[str],
    expected_as_of: str,
    min_history_rows: int = 253,
) -> Dict:
    """Verify the parquet files themselves, independently of the manifest.

    A manifest is only an index and can survive a deleted, truncated, or
    manually replaced parquet file. Live trading therefore uses this slower
    physical inspection after synchronization: every requested symbol must
    contain exactly one valid OHLCV row for ``expected_as_of``. Historical row
    counts are derived from usable close observations in the files, not copied
    from manifest metadata.
    """
    required_columns = {"timestamp", "symbol", "open", "high", "low", "close", "volume"}
    expected = datetime.strptime(str(expected_as_of)[:10], "%Y-%m-%d").date()
    unique_symbols = list(dict.fromkeys(str(symbol).upper() for symbol in symbols if symbol))
    valid_as_of = []
    history_ready = []
    history_rows: Dict[str, int] = {}
    missing_files = []
    invalid_symbols: Dict[str, List[str]] = {}

    for symbol in unique_symbols:
        path = STORE_DIR / f"{symbol}.parquet"
        errors: List[str] = []
        if not path.is_file():
            missing_files.append(symbol)
            invalid_symbols[symbol] = ["parquet file missing"]
            history_rows[symbol] = 0
            continue

        try:
            schema = set(pl.read_parquet_schema(path).names())
        except Exception as exc:
            invalid_symbols[symbol] = [f"parquet schema unreadable: {exc}"]
            history_rows[symbol] = 0
            continue
        missing_columns = sorted(required_columns - schema)
        if missing_columns:
            invalid_symbols[symbol] = [f"missing columns: {', '.join(missing_columns)}"]
            history_rows[symbol] = 0
            continue

        try:
            frame = pl.read_parquet(path, columns=sorted(required_columns))
            frame = _strip_tz(frame).with_columns([
                pl.col("timestamp").cast(pl.Date, strict=False).alias("_session_date"),
                pl.col("symbol").cast(pl.Utf8, strict=False).str.to_uppercase().alias("_symbol_text"),
                pl.col("close").cast(pl.Float64, strict=False).alias("_close_number"),
            ])
        except Exception as exc:
            invalid_symbols[symbol] = [f"parquet data unreadable: {exc}"]
            history_rows[symbol] = 0
            continue

        symbol_rows = frame.filter(pl.col("_symbol_text") == symbol)
        if symbol_rows.height != frame.height:
            errors.append("file contains missing or mismatched symbol values")

        usable_history = symbol_rows.filter(
            pl.col("_session_date").is_not_null()
            & (pl.col("_session_date") <= expected)
            & pl.col("_close_number").is_not_null()
            & pl.col("_close_number").is_finite()
            & (pl.col("_close_number") > 0)
        )
        actual_history_rows = int(usable_history["_session_date"].n_unique())
        history_rows[symbol] = actual_history_rows
        if actual_history_rows >= int(min_history_rows):
            history_ready.append(symbol)

        as_of_rows = symbol_rows.filter(pl.col("_session_date") == expected)
        if as_of_rows.height == 0:
            errors.append(f"no physical row for {expected_as_of}")
        elif as_of_rows.height > 1:
            errors.append(f"duplicate physical rows for {expected_as_of}")
        else:
            row = as_of_rows.row(0, named=True)
            numeric = {}
            for field in ("open", "high", "low", "close", "volume"):
                try:
                    numeric[field] = float(row[field])
                except (TypeError, ValueError):
                    errors.append(f"{field} is not numeric")
                    continue
                if not math.isfinite(numeric[field]):
                    errors.append(f"{field} is not finite")
            for field in ("open", "high", "low", "close"):
                if field in numeric and math.isfinite(numeric[field]) and numeric[field] <= 0:
                    errors.append(f"{field} must be positive")
            if "volume" in numeric and math.isfinite(numeric["volume"]) and numeric["volume"] < 0:
                errors.append("volume must be non-negative")
            if "high" in numeric and "low" in numeric and numeric["high"] < numeric["low"]:
                errors.append("high is below low")
            if all(field in numeric for field in ("open", "high", "low", "close")):
                if numeric["high"] < max(numeric["open"], numeric["close"]):
                    errors.append("high is below open/close")
                if numeric["low"] > min(numeric["open"], numeric["close"]):
                    errors.append("low is above open/close")

        if errors:
            invalid_symbols[symbol] = errors
        else:
            valid_as_of.append(symbol)

    return {
        "passed": len(valid_as_of) == len(unique_symbols),
        "expected_as_of": str(expected_as_of)[:10],
        "required_count": len(unique_symbols),
        "valid_as_of_count": len(valid_as_of),
        "valid_as_of_symbols": valid_as_of,
        "missing_files": missing_files,
        "invalid_symbols": invalid_symbols,
        "history_rows_by_symbol": history_rows,
        "history_ready_symbols": history_ready,
        "min_history_rows": int(min_history_rows),
    }


# ------------------------------------------------------------------
# Coverage stats
# ------------------------------------------------------------------

def get_coverage_stats(all_symbols: List[str]) -> Dict:
    manifest = get_manifest()
    covered = [s for s in all_symbols if s in manifest]
    missing = [s for s in all_symbols if s not in manifest]
    # Status must describe the selected universe, not unrelated symbols that
    # happen to exist in the shared store.
    dates = [manifest[s]["last_date"] for s in covered if "last_date" in manifest[s]]
    total_rows = sum(manifest[s].get("row_count", 0) for s in covered)
    latest = max(dates) if dates else None
    fresh = [s for s in covered if latest and manifest[s].get("last_date") == latest]
    stale = [s for s in covered if s not in fresh]
    return {
        "total_symbols": len(all_symbols),
        "covered": len(covered),
        "missing_count": len(missing),
        "coverage_pct": round(len(covered) / len(all_symbols) * 100, 1) if all_symbols else 0.0,
        "last_update": latest,
        "oldest_last_update": min(dates) if dates else None,
        "fresh_count": len(fresh),
        "fresh_pct": round(len(fresh) / len(all_symbols) * 100, 1) if all_symbols else 0.0,
        "stale_symbols": stale[:30],
        "earliest_date": None,  # Could scan parquet history if needed.
        "total_rows": total_rows,
        "missing_symbols": missing[:30],
    }
