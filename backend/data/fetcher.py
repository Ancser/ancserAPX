"""
Data fetcher: initial 10-year bulk fetch + daily incremental sync.

Usage:
    from backend.data.fetcher import fetch_bulk, fetch_incremental

    # Initial load (one-time, triggered from UI)
    fetch_bulk(SP500_TICKERS, callback=ws_log)

    # Daily sync (triggered on server start-up and by scheduler)
    fetch_incremental(SP500_TICKERS, callback=ws_log)
"""

import polars as pl
import logging
from datetime import datetime
from typing import List, Dict, Callable, Optional

from backend.data import store
from backend.data.alpaca_adapter import AlpacaAdapter

logger = logging.getLogger("backend.fetcher")

CHUNK_SIZE = 50          # symbols per API call (Alpaca limit ~100, keep 50 to be safe)
TEN_YEARS_START = "2015-01-01"


def _today() -> str:
    return datetime.now().strftime("%Y-%m-%d")


def fetch_bulk(
    symbols: List[str],
    start_date: str = TEN_YEARS_START,
    end_date: Optional[str] = None,
    account_name: str = "Main",
    callback: Optional[Callable[[str, str], None]] = None,
) -> Dict:
    """
    Fetch `symbols` in chunks and persist to local Parquet store.
    `callback(level, message)` is called for progress updates.
    """
    adapter = AlpacaAdapter(account_name)
    if not end_date:
        end_date = _today()

    chunks = [symbols[i : i + CHUNK_SIZE] for i in range(0, len(symbols), CHUNK_SIZE)]
    success, failed = [], []

    def _log(level: str, msg: str):
        logger.info(msg)
        if callback:
            callback(level, msg)

    _log("info", f"Starting bulk fetch: {len(symbols)} symbols, {len(chunks)} chunks, {start_date}→{end_date}")

    for idx, chunk in enumerate(chunks, 1):
        _log("info", f"[{idx}/{len(chunks)}] Fetching {len(chunk)} symbols...")
        try:
            df = adapter.fetch_history(chunk, start_date, end_date).collect()
            if df.is_empty():
                _log("warn", f"[{idx}/{len(chunks)}] No data returned.")
                failed.extend(chunk)
                continue
            for sym in df["symbol"].cast(pl.Utf8).unique().to_list():
                sym_df = df.filter(pl.col("symbol").cast(pl.Utf8) == sym)
                store.save(sym_df, sym)
                success.append(sym)
            _log("info", f"[{idx}/{len(chunks)}] Stored {len(df['symbol'].unique())} symbols.")
        except Exception as e:
            _log("error", f"[{idx}/{len(chunks)}] Chunk failed: {e}")
            failed.extend(chunk)

    _log(
        "success",
        f"Bulk fetch done — {len(success)} stored, {len(failed)} failed.",
    )
    return {"success": success, "failed": failed}


def fetch_incremental(
    symbols: Optional[List[str]] = None,
    account_name: str = "Main",
    callback: Optional[Callable[[str, str], None]] = None,
) -> Dict:
    """
    Fetch only data that is missing or outdated in the local store.
    Groups symbols by their required start date for efficient batching.
    """
    from backend.data.constituents import SP500_TICKERS
    if symbols is None:
        symbols = SP500_TICKERS

    manifest = store.get_manifest()
    today = _today()

    # Determine which symbols need updating and from what date
    needs_update: Dict[str, str] = {}  # symbol → fetch_from_date
    for sym in symbols:
        if sym not in manifest:
            needs_update[sym] = TEN_YEARS_START
        else:
            last = manifest[sym].get("last_date", TEN_YEARS_START)
            if last < today:
                needs_update[sym] = last  # fetch from last stored date (will dedup on save)

    if not needs_update:
        if callback:
            callback("info", "Local store is up to date — no incremental fetch needed.")
        return {"updated": 0, "failed": 0}

    def _log(level: str, msg: str):
        logger.info(msg)
        if callback:
            callback(level, msg)

    _log("info", f"Incremental fetch: {len(needs_update)} symbols need updating.")

    adapter = AlpacaAdapter(account_name)
    sym_list = list(needs_update.keys())
    # Use the earliest start date across the batch for simplicity
    # (store.save() deduplicates, so overlapping data is safe)
    start_date = min(needs_update.values())

    chunks = [sym_list[i : i + CHUNK_SIZE] for i in range(0, len(sym_list), CHUNK_SIZE)]
    updated, failed = [], []

    for idx, chunk in enumerate(chunks, 1):
        _log("info", f"[{idx}/{len(chunks)}] Incremental update for {len(chunk)} symbols...")
        try:
            df = adapter.fetch_history(chunk, start_date, today).collect()
            if df.is_empty():
                failed.extend(chunk)
                continue
            for sym in df["symbol"].cast(pl.Utf8).unique().to_list():
                sym_df = df.filter(pl.col("symbol").cast(pl.Utf8) == sym)
                store.save(sym_df, sym)
                updated.append(sym)
        except Exception as e:
            _log("error", f"Incremental chunk {idx} failed: {e}")
            failed.extend(chunk)

    _log("success", f"Incremental done — {len(updated)} updated, {len(failed)} failed.")
    return {"updated": len(updated), "failed": len(failed)}
