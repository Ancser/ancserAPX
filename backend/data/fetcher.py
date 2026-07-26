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
from datetime import date, datetime, time, timedelta, timezone
from typing import List, Dict, Callable, Optional
from zoneinfo import ZoneInfo

from backend.data import store
from backend.data.alpaca_adapter import AlpacaAdapter

logger = logging.getLogger("backend.fetcher")

CHUNK_SIZE = 50          # symbols per API call (Alpaca limit ~100, keep 50 to be safe)
TEN_YEARS_START = "2015-01-01"


def _today() -> str:
    """Current New York market date, independent of the host timezone."""
    return datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def expected_completed_session(adapter: AlpacaAdapter, now: Optional[datetime] = None) -> str:
    """Return the latest NYSE session whose daily bar should be complete.

    The normal live run is 09:35 New York time, so today's still-forming session
    is deliberately excluded.  A manual run after 16:15 ET may use today's bar.
    The Alpaca calendar is authoritative; inability to obtain it is a data-safety
    failure rather than a reason to guess around exchange holidays.
    """
    ny = ZoneInfo("America/New_York")
    if now is None:
        now_ny = datetime.now(ny)
    elif now.tzinfo is None:
        now_ny = now.replace(tzinfo=ny)
    else:
        now_ny = now.astimezone(ny)

    today = now_ny.date()
    start = today - timedelta(days=14)
    sessions = adapter.get_trading_days(start, today)
    completed = [d for d in sessions if d < today]
    if today in sessions and now_ny.time() >= time(16, 15):
        completed.append(today)
    if not completed:
        raise RuntimeError("Unable to determine the latest completed NYSE session")
    return max(completed).isoformat()


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
    required_as_of: Optional[str] = None,
    end_date: Optional[str] = None,
    adapter: Optional[AlpacaAdapter] = None,
) -> Dict:
    """
    Fetch only data that is missing or outdated in the local store.
    Groups symbols by their required start date for efficient batching.
    """
    from backend.data.constituents import SP500_TICKERS
    if symbols is None:
        symbols = SP500_TICKERS

    manifest = store.get_manifest()
    today = end_date or _today()
    freshness_date = required_as_of or today

    # Determine which symbols need updating and from what date
    needs_update: Dict[str, str] = {}  # symbol → fetch_from_date
    for sym in symbols:
        if sym not in manifest:
            needs_update[sym] = TEN_YEARS_START
        else:
            last = manifest[sym].get("last_date", TEN_YEARS_START)
            if last < freshness_date:
                needs_update[sym] = last  # fetch from last stored date (will dedup on save)

    if not needs_update:
        if callback:
            callback("info", "Local store is up to date — no incremental fetch needed.")
        return {
            "requested": 0,
            "updated": 0,
            "failed": 0,
            "updated_symbols": [],
            "failed_symbols": [],
        }

    def _log(level: str, msg: str):
        logger.info(msg)
        if callback:
            callback(level, msg)

    _log("info", f"Incremental fetch: {len(needs_update)} symbols need updating.")

    adapter = adapter or AlpacaAdapter(account_name)
    # Group by required start date. A newly-added/missing ticker may need ten
    # years, while the other 500 names need one day; using the global earliest
    # date made the pre-trade sync unnecessarily download the full history for
    # every symbol and could miss the 09:30 open.
    by_start: Dict[str, List[str]] = {}
    for symbol, start in needs_update.items():
        by_start.setdefault(start, []).append(symbol)
    jobs = []
    for start_date, group in sorted(by_start.items()):
        for offset in range(0, len(group), CHUNK_SIZE):
            jobs.append((start_date, group[offset:offset + CHUNK_SIZE]))
    updated, failed = [], []

    for idx, (start_date, chunk) in enumerate(jobs, 1):
        _log(
            "info",
            f"[{idx}/{len(jobs)}] Incremental update for {len(chunk)} symbols from {start_date}...",
        )
        try:
            df = adapter.fetch_history(chunk, start_date, today).collect()
            if df.is_empty():
                failed.extend(chunk)
                continue
            returned = set(df["symbol"].cast(pl.Utf8).unique().to_list())
            for sym in returned:
                sym_df = df.filter(pl.col("symbol").cast(pl.Utf8) == sym)
                store.save(sym_df, sym)
                updated.append(sym)
            # A successful HTTP response can still omit halted, renamed, or
            # unauthorized symbols.  Treat those omissions as explicit failures.
            failed.extend(sym for sym in chunk if sym not in returned)
        except Exception as e:
            _log("error", f"Incremental chunk {idx} failed: {e}")
            failed.extend(chunk)

    _log("success", f"Incremental done — {len(updated)} updated, {len(failed)} failed.")
    updated = sorted(set(updated))
    failed = sorted(set(failed) - set(updated))
    return {
        "requested": len(needs_update),
        "updated": len(updated),
        "failed": len(failed),
        "updated_symbols": updated,
        "failed_symbols": failed,
    }


def sync_and_validate_live_data(
    symbols: List[str],
    account_name: str = "Main",
    config: Optional[Dict] = None,
    callback: Optional[Callable[[str, str], None]] = None,
    adapter: Optional[AlpacaAdapter] = None,
    now: Optional[datetime] = None,
) -> Dict:
    """Synchronize and fail-closed unless the local store is safe to trade.

    The broker asset master first turns the configured universe into an
    active/tradable/fractionable effective universe. Data validation then applies
    strictly to every effective symbol. QQQ and SPY remain separate mandatory
    risk gauges and are never silently removed by target-universe eligibility.
    """
    started_at = _utc_now()
    config = config or {}
    quality_cfg = config.get("data_quality", {}) or {}
    configured_universe = list(dict.fromkeys(
        str(s).strip().upper() for s in symbols if str(s).strip()
    ))
    adapter = adapter or AlpacaAdapter(account_name)

    report: Dict = {
        "account": account_name,
        "started_at": started_at,
        "completed_at": None,
        "passed": False,
        "configured_universe": configured_universe,
        "configured_universe_count": len(configured_universe),
        "effective_universe": [],
        "effective_universe_count": 0,
        "excluded_assets": {},
        "universe_count": 0,
        "sync_symbol_count": 0,
        "gauge_symbols": ["QQQ", "SPY"],
        "errors": [],
    }
    if not configured_universe:
        report["errors"].append("Configured universe is empty")
        report["completed_at"] = _utc_now()
        return report

    try:
        expected = expected_completed_session(adapter, now=now)
        report["expected_as_of"] = expected
    except Exception as exc:
        report["errors"].append(f"NYSE calendar unavailable: {exc}")
        report["completed_at"] = _utc_now()
        return report

    # Resolve eligibility in one broker bulk request. The current OMS uses
    # two-decimal share quantities, so fractionability is a live requirement.
    # Any malformed/failed asset-master response blocks the run; only explicit
    # inactive, not-tradable, non-fractionable, or not-found results are excluded.
    try:
        eligibility = adapter.get_asset_eligibility(
            configured_universe, require_fractionable=True
        )
    except Exception as exc:
        report["asset_eligibility"] = {
            "status": "failed",
            "source": "alpaca_get_all_assets",
            "error": str(exc),
        }
        report["errors"].append(f"Broker asset eligibility unavailable: {exc}")
        report["completed_at"] = _utc_now()
        return report

    universe = list(dict.fromkeys(
        str(symbol).strip().upper()
        for symbol in (eligibility.get("effective_symbols") or [])
        if str(symbol).strip()
    ))
    excluded_assets = {
        str(symbol).strip().upper(): details
        for symbol, details in (eligibility.get("excluded_assets") or {}).items()
        if str(symbol).strip()
    }
    configured_set = set(configured_universe)
    effective_set = set(universe)
    excluded_set = set(excluded_assets)
    contract_errors = []
    if eligibility.get("status") != "passed":
        contract_errors.append("asset eligibility status is not passed")
    if effective_set & excluded_set:
        contract_errors.append("symbols appear in both effective and excluded sets")
    unclassified = sorted(configured_set - effective_set - excluded_set)
    unexpected = sorted((effective_set | excluded_set) - configured_set)
    if unclassified:
        contract_errors.append("unclassified configured symbols: " + ", ".join(unclassified))
    if unexpected:
        contract_errors.append("unconfigured symbols in eligibility result: " + ", ".join(unexpected))
    missing_reasons = sorted(
        symbol for symbol, details in excluded_assets.items()
        if not isinstance(details, dict) or not details.get("reason")
    )
    if missing_reasons:
        contract_errors.append("excluded symbols missing reasons: " + ", ".join(missing_reasons))
    if contract_errors:
        report["asset_eligibility"] = eligibility
        report["errors"].append(
            "Broker asset eligibility response is incomplete: " + "; ".join(contract_errors)
        )
        report["completed_at"] = _utc_now()
        return report

    report.update({
        "asset_eligibility": eligibility,
        "effective_universe": universe,
        "effective_universe_count": len(universe),
        "excluded_assets": excluded_assets,
        "excluded_asset_count": len(excluded_assets),
        "universe_count": len(universe),
    })
    if not universe:
        report["errors"].append("No broker-eligible symbols remain in the effective universe")
        report["completed_at"] = _utc_now()
        return report

    gauges = [s for s in ("QQQ", "SPY") if s not in universe]
    sync_symbols = universe + gauges
    report["sync_symbol_count"] = len(sync_symbols)

    # Alpaca bar request end is exclusive. Fetch through the day after the
    # required session, which is normally the current NY date at 09:35 ET.
    fetch_end = (date.fromisoformat(expected) + timedelta(days=1)).isoformat()
    try:
        report["sync"] = fetch_incremental(
            sync_symbols,
            account_name=account_name,
            callback=callback,
            required_as_of=expected,
            end_date=fetch_end,
            adapter=adapter,
        )
    except Exception as exc:
        report["sync"] = {"requested": len(sync_symbols), "updated": 0, "failed": len(sync_symbols)}
        report["errors"].append(f"Incremental sync failed: {exc}")

    manifest = store.get_manifest()

    def _last(sym: str) -> Optional[str]:
        value = manifest.get(sym, {}).get("last_date")
        return str(value)[:10] if value else None

    covered = [s for s in universe if _last(s)]
    fresh = [s for s in universe if _last(s) and _last(s) >= expected]
    stale = [s for s in universe if s not in fresh]
    future = [s for s in universe if _last(s) and _last(s) > expected]
    rows_required = max(1, int(quality_cfg.get("min_history_rows", 253)))
    history_ready = [
        s for s in universe
        if int(manifest.get(s, {}).get("row_count", 0) or 0) >= rows_required
    ]
    gauge_dates = {s: _last(s) for s in ("QQQ", "SPY")}
    gauge_fresh = {s: bool(ds and ds >= expected) for s, ds in gauge_dates.items()}
    gauge_rows = {
        s: int(manifest.get(s, {}).get("row_count", 0) or 0) for s in ("QQQ", "SPY")
    }

    def _pct(n: int) -> float:
        return round(n / len(universe) * 100.0, 4) if universe else 0.0

    coverage_pct = _pct(len(covered))
    fresh_pct = _pct(len(fresh))
    history_pct = _pct(len(history_ready))
    # Live trading is unconditionally strict. Config may document a requested
    # threshold, but it cannot weaken physical/fresh coverage below 100%.
    requested_coverage = float(quality_cfg.get("min_coverage_pct", 100.0))
    requested_fresh = float(quality_cfg.get("min_fresh_coverage_pct", 100.0))
    min_coverage = 100.0
    min_fresh = 100.0
    min_history = float(quality_cfg.get("min_history_coverage_pct", 90.0))
    risk_cfg = config.get("risk_management", {}) or {}
    regime_mode = str(
        risk_cfg.get("regime_mode", "cash" if config.get("ema_kill_switch", False) else "off")
    ).lower()
    # Keep both gauges physically ready even before a regime switch is enabled;
    # changing a risk setting must never turn missing data into fail-open risk.
    gauges_required = True

    report.update({
        "store_as_of": max((_last(s) for s in universe if _last(s)), default=None),
        "coverage_pct": coverage_pct,
        "fresh_coverage_pct": fresh_pct,
        "history_coverage_pct": history_pct,
        "min_history_rows": rows_required,
        "covered_count": len(covered),
        "fresh_count": len(fresh),
        "history_ready_count": len(history_ready),
        "missing_symbols": [s for s in universe if s not in covered],
        "stale_symbols": stale,
        "future_symbols": future,
        "gauge_dates": gauge_dates,
        "gauge_fresh": gauge_fresh,
        "gauge_rows": gauge_rows,
        "gauges_required": gauges_required,
        "thresholds": {
            "min_coverage_pct": min_coverage,
            "min_fresh_coverage_pct": min_fresh,
            "min_history_coverage_pct": min_history,
        },
        "requested_thresholds": {
            "min_coverage_pct": requested_coverage,
            "min_fresh_coverage_pct": requested_fresh,
        },
    })

    if coverage_pct < min_coverage:
        report["errors"].append(f"Universe coverage {coverage_pct:.2f}% < {min_coverage:.2f}%")
    if fresh_pct < min_fresh:
        report["errors"].append(f"Fresh coverage {fresh_pct:.2f}% < {min_fresh:.2f}% for {expected}")
    if future:
        report["errors"].append(
            f"Store contains {len(future)} symbols newer than completed session {expected}"
        )
    if history_pct < min_history:
        report["errors"].append(f"History coverage {history_pct:.2f}% < {min_history:.2f}%")
    if not all(gauge_fresh.values()):
        missing_gauges = [s for s, ok in gauge_fresh.items() if not ok]
        report["errors"].append(f"Required gauge data stale/missing: {', '.join(missing_gauges)}")
    if any(rows < 220 for rows in gauge_rows.values()):
        short_gauges = [s for s, rows in gauge_rows.items() if rows < 220]
        report["errors"].append(f"Required gauge history <220 rows: {', '.join(short_gauges)}")

    # Manifest checks are necessary but not sufficient. Once metadata passes,
    # inspect every parquet and the exact completed-session row before trading.
    report["physical_validation"] = {"status": "not_run_manifest_failed"}
    if not report["errors"]:
        try:
            physical = store.inspect_physical_data(
                sync_symbols,
                expected_as_of=expected,
                min_history_rows=rows_required,
            )
        except Exception as exc:
            physical = {
                "passed": False,
                "status": "inspection_failed",
                "error": str(exc),
                "valid_as_of_symbols": [],
                "history_ready_symbols": [],
                "history_rows_by_symbol": {},
            }
            report["errors"].append(f"Physical parquet inspection failed: {exc}")
        report["physical_validation"] = physical
        valid_physical = set(physical.get("valid_as_of_symbols", []))
        history_physical = set(physical.get("history_ready_symbols", []))
        physical_coverage = _pct(sum(1 for s in universe if s in valid_physical))
        physical_history_pct = _pct(sum(1 for s in universe if s in history_physical))
        report["physical_coverage_pct"] = physical_coverage
        report["physical_history_coverage_pct"] = physical_history_pct
        report["history_coverage_pct"] = physical_history_pct
        report["history_ready_count"] = sum(1 for s in universe if s in history_physical)
        if physical_coverage < 100.0:
            report["errors"].append(
                f"Physical completed-session coverage {physical_coverage:.2f}% < 100.00%"
            )
        invalid_gauges = [s for s in ("QQQ", "SPY") if s not in valid_physical]
        if invalid_gauges:
            report["errors"].append(
                f"Required gauge parquet invalid: {', '.join(invalid_gauges)}"
            )
        gauge_physical_rows = physical.get("history_rows_by_symbol", {})
        short_physical_gauges = [
            s for s in ("QQQ", "SPY") if int(gauge_physical_rows.get(s, 0) or 0) < 220
        ]
        if short_physical_gauges:
            report["errors"].append(
                f"Required gauge physical history <220 rows: {', '.join(short_physical_gauges)}"
            )
        if physical_history_pct < min_history:
            report["errors"].append(
                f"Physical history coverage {physical_history_pct:.2f}% < {min_history:.2f}%"
            )

    report["passed"] = not report["errors"]
    report["completed_at"] = _utc_now()
    return report
