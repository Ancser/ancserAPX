"""
AncserEventLoop — daily rebalance scheduler.
Runs as a standalone process: python -m backend.execution.scheduler [--run-once] [--force]
"""

import argparse, hashlib, json, logging, os, subprocess, time, uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Optional
from zoneinfo import ZoneInfo
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.executors.pool import ThreadPoolExecutor

from backend.data.alpaca_adapter import AlpacaAdapter
from backend.utils.accounts import get_configured_accounts

logger = logging.getLogger("backend.scheduler")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)

DAILY_LOCK_DIR = "logs"
PID_FILE = "logs/ancser_daemon.pid"
MARKET_TZ = ZoneInfo("America/New_York")
EXECUTION_LOCK_STALE_SECONDS = 2 * 60 * 60


def _market_now() -> datetime:
    return datetime.now(MARKET_TZ)


def _market_date():
    return _market_now().date()


def _pid_running(pid: int) -> bool:
    try:
        result = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                                capture_output=True, text=True)
        return str(pid) in result.stdout
    except Exception:
        return False


def _check_single_instance() -> bool:
    if os.path.exists(PID_FILE):
        try:
            old = int(open(PID_FILE).read().strip())
            if _pid_running(old):
                logger.warning(f"Another instance running (PID {old}). Exiting.")
                return True
        except Exception:
            pass
    os.makedirs(os.path.dirname(PID_FILE), exist_ok=True)
    open(PID_FILE, "w").write(str(os.getpid()))
    return False


def _remove_pid():
    try:
        os.remove(PID_FILE)
    except Exception:
        pass


def _is_locked(account: str) -> bool:
    path = f"{DAILY_LOCK_DIR}/daily_lock_{account}.json"
    if not os.path.exists(path):
        return False
    try:
        data = json.load(open(path))
        return data.get("date") == _market_date().isoformat()
    except Exception:
        return False


def _write_lock(account: str):
    os.makedirs(DAILY_LOCK_DIR, exist_ok=True)
    json.dump(
        {"date": _market_date().isoformat(), "at": datetime.now(timezone.utc).isoformat()},
        open(f"{DAILY_LOCK_DIR}/daily_lock_{account}.json", "w"),
    )


def _last_rebalance_date(account: str):
    """Read the date of the last actual rebalance, or None."""
    path = f"{DAILY_LOCK_DIR}/last_rebalance_{account}.json"
    if not os.path.exists(path):
        return None
    try:
        data = json.load(open(path))
        ds = data.get("rebalance_date") or data.get("date")
        return datetime.strptime(ds, "%Y-%m-%d").date() if ds else None
    except Exception:
        return None


def _last_trading_day_of_week(today, account: str, target_dow: int = 4):
    """The day we *want* to rebalance this calendar week = the last NYSE trading
    day on/before the configured weekday (default Friday). If Friday is a market
    holiday (e.g. Fri 2026-07-03, July 4th observed) this returns the Thursday.

    Uses Alpaca's official calendar; falls back to the plain target weekday
    (Friday) if the calendar API is unavailable.
    """
    monday = today - timedelta(days=today.weekday())          # Mon of this week
    target_day = monday + timedelta(days=target_dow)          # e.g. Friday
    try:
        adapter = AlpacaAdapter(account)
        days = adapter.get_trading_days(monday, target_day)   # Mon..Fri trading days
        days = [d for d in days if d <= target_day]
        if days:
            return max(days)                                  # last trading day <= Friday
    except Exception as e:
        logger.warning(f"calendar lookup failed ({e}); falling back to weekday {target_dow}")
    return target_day


def _is_nyse_session_today(account: str) -> bool:
    today = _market_date()
    try:
        return today in AlpacaAdapter(account).get_trading_days(today, today)
    except Exception as exc:
        logger.error(f"{account}: unable to verify today's NYSE session: {exc}")
        return False


def _should_rebalance_today(config: dict, account: str, force: bool = False) -> bool:
    """
    Decide whether today's trigger should actually rebalance.

    The schedule fires every weekday (and the user also runs it daily), but a
    WEEKLY strategy must trade at most once per week. Policy:
      - force            -> bypass cadence, but never the NYSE-session safety gate
      - frequency=daily  -> every NYSE session
      - frequency=weekly -> rebalance ONLY on this week's rebalance day, which is
                            the last NYSE trading day on/before the configured
                            weekday (default Friday; Thursday if Friday is a
                            holiday) AND only if we haven't already rebalanced
                            this calendar week.
      - first run ever   -> rebalance immediately to establish the position.
      - long-term failsafe: if the bot was down and it's been >= max_stale_days
                            (default 10) since the last rebalance, rebalance on
                            the next run so the book never drifts stale.

    Note: the weekly rule is the PRIMARY one. The stale failsafe threshold is
    deliberately > 8 days so a normal (or holiday-shortened) week never triggers
    an off-Friday catch-up — that was the old bug that anchored rebalances to
    Monday after an off-cycle first run.
    """
    today = _market_date()
    if not _is_nyse_session_today(account):
        # Calendar uncertainty is not permission to queue market orders on a
        # weekend or exchange holiday.
        return False
    if force:
        return True
    freq = str(config.get("rebalance_frequency", "weekly")).lower()
    if freq == "daily":
        return True
    last = _last_rebalance_date(account)

    # First run ever — establish the position immediately, any day.
    if last is None:
        return True

    if freq == "custom_days":
        target_days = max(1, int(config.get("rebalance_days", 5) or 5))
        try:
            start = last + timedelta(days=1)
            days = AlpacaAdapter(account).get_trading_days(start, today)
            return len(days) >= target_days
        except Exception as e:
            logger.warning(f"{account}: custom_days calendar lookup failed ({e}); using calendar-day fallback.")
            return (today - last).days >= max(target_days, int(target_days * 1.4))

    target_dow = int(config.get("rebalance_weekday", 4))      # 0=Mon..4=Fri
    rebalance_day = _last_trading_day_of_week(today, account, target_dow)
    week_start = today - timedelta(days=today.weekday())       # Monday of this week

    # PRIMARY: it's this week's rebalance day and we haven't traded this week.
    if today == rebalance_day and last < week_start:
        return True

    # FAILSAFE: bot was down long enough that we missed a whole cycle.
    max_stale = int(config.get("max_stale_days", 10))
    if (today - last).days >= max_stale:
        return True

    return False


def _configured_factor_weights(config: Dict) -> Dict[str, float]:
    """Flatten either sleeve weights or legacy factor_weights for audit/UI."""
    sleeves = config.get("sleeves") or []
    if sleeves:
        totals: Dict[str, float] = {}
        for sleeve in sleeves:
            allocation = float(sleeve.get("alloc", 0) or 0)
            factors = sleeve.get("factors") or []
            weights = sleeve.get("weights") or []
            for index, factor in enumerate(factors):
                if isinstance(weights, dict):
                    weight = weights.get(factor, 0.0)
                else:
                    weight = weights[index] if index < len(weights) else 0.0
                totals[str(factor)] = totals.get(str(factor), 0.0) + allocation * float(weight)
        return totals
    explicit = config.get("factor_weights") or {}
    if explicit:
        return {str(k): float(v) for k, v in explicit.items()}
    factors = config.get("active_factors") or []
    return {str(f): 1.0 / len(factors) for f in factors} if factors else {}


def _effective_runtime_config(config: Dict, sync_report: Dict) -> Dict:
    """Build the immutable-per-run config authorized by asset eligibility."""
    configured = list(dict.fromkeys(
        str(symbol).strip().upper()
        for symbol in (config.get("universe") or [])
        if str(symbol).strip()
    ))
    effective = list(dict.fromkeys(
        str(symbol).strip().upper()
        for symbol in (sync_report.get("effective_universe") or [])
        if str(symbol).strip()
    ))
    if not effective:
        raise RuntimeError("Data sync did not provide a non-empty effective universe")
    unexpected = sorted(set(effective) - set(configured))
    if unexpected:
        raise RuntimeError(
            "Data sync effective universe contains unconfigured symbols: "
            + ", ".join(unexpected)
        )
    runtime = dict(config)
    runtime["universe"] = effective
    runtime["_configured_universe"] = configured
    return runtime


def _read_last_target(account: str) -> Dict:
    path = Path(DAILY_LOCK_DIR) / f"last_rebalance_{account}.json"
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    except Exception:
        return {}


def _stop_flag_path(account: str) -> Path:
    return Path(DAILY_LOCK_DIR) / f"stop_{account}.flag"


def _execution_lock_path(account: str) -> Path:
    # Account display names are case-insensitive in the UI/config layer. Do not
    # allow "Main" and "main" to acquire independent mutation locks.
    canonical_account = str(account).strip().casefold()
    digest = hashlib.sha256(canonical_account.encode("utf-8")).hexdigest()[:12]
    return Path(DAILY_LOCK_DIR) / f"execution_lock_{digest}.json"


def _try_acquire_account_execution_lock(account: str, run_id: str) -> Dict:
    """Atomically acquire one live mutation slot for an account.

    This lock is separate from the CLI daemon PID file: web workers and the
    scheduled process have different lifetimes but must not cancel/resubmit the
    same account concurrently.
    """
    path = _execution_lock_path(account)
    path.parent.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex
    payload = {
        "account": account,
        "run_id": run_id,
        "token": token,
        "pid": os.getpid(),
        "acquired_at": datetime.now(timezone.utc).isoformat(),
    }

    for _ in range(2):
        try:
            fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            try:
                os.write(fd, json.dumps(payload).encode("utf-8"))
                os.fsync(fd)
            finally:
                os.close(fd)
            return {"acquired": True, "path": str(path), **payload}
        except FileExistsError:
            try:
                observed_text = path.read_text(encoding="utf-8")
            except Exception:
                observed_text = ""
            try:
                owner = json.loads(observed_text)
            except Exception:
                owner = {}
            try:
                age = max(0.0, time.time() - path.stat().st_mtime)
            except FileNotFoundError:
                continue
            owner_pid = int(owner.get("pid", 0) or 0)
            # Same PID means another web thread is active, not a stale lock.
            owner_alive = owner_pid == os.getpid() or (owner_pid > 0 and _pid_running(owner_pid))
            # Never steal a lock from a live process, even if execution takes
            # unusually long. Safety is preferable to an automatic duplicate
            # order batch; an operator can inspect the audited owner/run id.
            if owner_alive:
                return {
                    "acquired": False,
                    "path": str(path),
                    "reason": "account execution already in progress",
                    "owner": owner,
                    "age_seconds": round(age, 3),
                }

            # There is an unavoidable instant between O_EXCL creating the file
            # and the owner writing/fsyncing its JSON. Missing metadata on a new
            # lock is therefore contention, never evidence that it is stale.
            if (not owner or owner_pid <= 0) and age <= EXECUTION_LOCK_STALE_SECONDS:
                return {
                    "acquired": False,
                    "path": str(path),
                    "reason": "account execution lock is initializing or unreadable",
                    "owner": owner,
                    "age_seconds": round(age, 3),
                }

            # Serialize stale-lock reclamation. Without this guard two waiting
            # processes could both inspect the old token, and the slower one
            # could unlink the faster process's newly acquired lock.
            cleanup_path = path.with_name(f"{path.name}.cleanup")
            try:
                cleanup_fd = os.open(
                    str(cleanup_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY
                )
            except FileExistsError:
                return {
                    "acquired": False,
                    "path": str(path),
                    "reason": "stale execution lock cleanup already in progress",
                    "owner": owner,
                    "age_seconds": round(age, 3),
                }
            try:
                try:
                    current_text = path.read_text(encoding="utf-8")
                except FileNotFoundError:
                    continue
                if current_text != observed_text:
                    continue
                try:
                    path.unlink()
                except FileNotFoundError:
                    continue
                except OSError as exc:
                    return {
                        "acquired": False,
                        "path": str(path),
                        "reason": f"stale execution lock could not be removed: {exc}",
                        "owner": owner,
                    }
            finally:
                os.close(cleanup_fd)
                cleanup_path.unlink(missing_ok=True)
    return {"acquired": False, "path": str(path), "reason": "execution lock contention"}


def _release_account_execution_lock(lock: Dict) -> None:
    if not lock.get("acquired"):
        return
    path = Path(lock["path"])
    try:
        current = json.loads(path.read_text(encoding="utf-8"))
        if current.get("token") != lock.get("token"):
            return
        path.unlink(missing_ok=True)
    except FileNotFoundError:
        return
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        # Never delete a lock whose ownership token cannot be verified, and do
        # not mask the execution result merely because cleanup needs review.
        logger.error(f"Could not safely release account execution lock {path}: {exc}")


def _risk_settings(config: Dict) -> Dict:
    risk_cfg = config.get("risk_management", {}) or {}
    mode = str(
        risk_cfg.get("regime_mode", "cash" if config.get("ema_kill_switch", False) else "off")
    ).lower()
    return {
        "regime_mode": mode,
        "risk_off_leverage": float(
            risk_cfg.get("risk_off_leverage", min(float(config.get("leverage", 1.0)), 1.0))
        ),
        "volatility_throttle": bool(risk_cfg.get("volatility_throttle", False)),
        "vol_target_pct": float(risk_cfg.get("vol_target_pct", 0.25)),
        "vol_lookback": max(5, int(risk_cfg.get("vol_lookback", 20))),
    }


def _risk_settings_active(config: Dict) -> bool:
    settings = _risk_settings(config)
    return settings["regime_mode"] in {"cash", "throttle"} or settings["volatility_throttle"]


def _evaluate_daily_risk(account: str, config: Dict, as_of_date: str) -> Dict:
    """Stateful 200EMA exit / 20EMA re-entry plus daily vol leverage."""
    import numpy as np
    import pandas as pd
    from backend.data import store
    from backend.execution.tracker import LiveTracker

    settings = _risk_settings(config)
    active = _risk_settings_active(config)
    base_leverage = float(config.get("leverage", 1.0) or 1.0)
    result = {
        **settings,
        "active": active,
        "as_of_date": as_of_date,
        "base_leverage": base_leverage,
        "desired_leverage": base_leverage,
        "in_market": True,
        "transition": None,
        "realized_vol": None,
    }
    if not active:
        return result

    end = str(as_of_date)[:10]
    start = (datetime.strptime(end, "%Y-%m-%d") - timedelta(days=450)).strftime("%Y-%m-%d")
    state_path = Path(DAILY_LOCK_DIR) / f"risk_state_{account}.json"
    try:
        prior = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
    except Exception:
        prior = {}

    mode = settings["regime_mode"]
    # Backtest starts in-market and immediately applies the 200EMA exit rule;
    # use the same initialization instead of inferring regime from holdings.
    in_market = bool(prior.get("in_market", True))
    transition = None
    gauge_symbol = None
    gauge_close = ema_slow = ema_fast = None
    if mode in {"cash", "throttle"}:
        gauge_pl = store.load(["QQQ", "SPY"], start, end).collect()
        if gauge_pl.is_empty():
            raise RuntimeError("Daily risk gauge data is empty")
        gauge = gauge_pl.to_pandas()
        gauge["timestamp"] = pd.to_datetime(gauge["timestamp"])
        gauge_pivot = gauge.pivot(index="timestamp", columns="symbol", values="close")
        gauge_symbol = "QQQ" if "QQQ" in gauge_pivot else ("SPY" if "SPY" in gauge_pivot else None)
        if gauge_symbol is None:
            raise RuntimeError("Daily risk gauge QQQ/SPY is missing")
        series = gauge_pivot[gauge_symbol].dropna()
        if len(series) < 220:
            raise RuntimeError(f"Daily risk gauge {gauge_symbol} has only {len(series)} rows")
        gauge_close = float(series.iloc[-1])
        ema_slow = float(series.ewm(span=200, adjust=False).mean().iloc[-1])
        ema_fast = float(series.ewm(span=20, adjust=False).mean().iloc[-1])

        # Apply a session transition once. Re-running the task on the same as-of
        # date is idempotent and keeps the persisted regime state.
        if prior.get("last_checked_as_of") != end or prior.get("regime_mode") != mode:
            if in_market and gauge_close < ema_slow:
                in_market = False
                transition = "entered_risk_off"
            elif (not in_market) and gauge_close > ema_fast:
                in_market = True
                transition = "reentered_risk_on"

    desired = base_leverage
    if mode == "cash" and not in_market:
        desired = 0.0
    elif mode == "throttle" and not in_market:
        desired = min(desired, settings["risk_off_leverage"])

    if settings["volatility_throttle"]:
        universe = config.get("universe", [])
        hist_pl = store.load(universe, start, end).collect()
        if hist_pl.is_empty():
            raise RuntimeError("Daily volatility throttle has no universe data")
        hist = hist_pl.to_pandas()
        hist["timestamp"] = pd.to_datetime(hist["timestamp"])
        closes = hist.pivot(index="timestamp", columns="symbol", values="close")
        market_ret = closes.pct_change().mean(axis=1).dropna()
        lookback = settings["vol_lookback"]
        if len(market_ret) < lookback:
            raise RuntimeError("Daily volatility throttle has insufficient history")
        realized_vol = float(market_ret.tail(lookback).std(ddof=1) * np.sqrt(252))
        if not np.isfinite(realized_vol):
            raise RuntimeError("Daily volatility estimate is not finite")
        result["realized_vol"] = realized_vol
        if realized_vol > settings["vol_target_pct"] and realized_vol > 0:
            desired = min(
                desired,
                base_leverage * settings["vol_target_pct"] / realized_vol,
            )

    result.update({
        "desired_leverage": max(0.0, float(desired)),
        "in_market": in_market,
        "transition": transition,
        "gauge_symbol": gauge_symbol,
        "gauge_close": gauge_close,
        "ema_slow_200": ema_slow,
        "ema_fast_20": ema_fast,
        "last_checked_as_of": end,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    })
    state_path.parent.mkdir(parents=True, exist_ok=True)
    LiveTracker._write_json_atomic(str(state_path), result)
    return result


def _merge_broker_history(path: str, rows: list, key: str) -> list:
    """Merge snapshots so pending orders later become filled without losing history."""
    existing = []
    target = Path(path)
    try:
        if target.exists():
            existing = json.loads(target.read_text(encoding="utf-8"))
    except Exception:
        existing = []
    merged = {}
    anonymous = []
    for row in existing + rows:
        identity = str(row.get(key, "") or "")
        if identity:
            merged[identity] = row
        else:
            anonymous.append(row)
    result = list(merged.values()) + anonymous
    target.parent.mkdir(parents=True, exist_ok=True)
    from backend.execution.tracker import LiveTracker
    LiveTracker._write_json_atomic(str(target), result)
    return result


def record_broker_observation(
    account: str,
    config: Optional[Dict] = None,
    *,
    run_id: Optional[str] = None,
    adapter: Optional[AlpacaAdapter] = None,
) -> Dict:
    """Persist actual equity/positions and broker order/fill lifecycle every day."""
    from backend.execution.tracker import LiveTracker

    config = config or {}
    run_id = run_id or uuid.uuid4().hex
    tracker = LiveTracker(account)
    adapter = adapter or AlpacaAdapter(account)
    account_state = adapter.get_account()
    equity = float(account_state.get("equity", 0) or 0)
    if equity <= 0:
        tracker.record_event(
            "broker_observation", "failed", run_id=run_id,
            details={"reason": "equity unavailable", "account_snapshot": account_state},
        )
        return {"error": "Broker account/equity unavailable", "account": account_state}

    positions = adapter.get_positions()
    account_state["unrealized_pnl"] = round(
        sum(float(p.get("unrealized_pl", 0) or 0) for p in positions), 2
    )
    actual_allocations = {
        p["symbol"]: float(p.get("market_value", 0) or 0) / equity
        for p in positions
        if p.get("symbol")
    }
    broker_orders = adapter.get_orders(limit=500)
    broker_fills = adapter.get_activities(limit=500)
    order_history = _merge_broker_history(
        f"logs/broker_orders_{account}.json", broker_orders, "id"
    )
    fill_history = _merge_broker_history(
        f"logs/broker_fills_{account}.json", broker_fills, "id"
    )
    last_target = _read_last_target(account)
    target_allocations = {
        symbol: float(detail.get("weight", 0) or 0)
        for symbol, detail in (last_target.get("positions") or {}).items()
    }
    if not target_allocations:
        target_allocations = actual_allocations

    tracker.record_event(
        "broker_order_snapshot", "recorded", run_id=run_id,
        details={
            "latest_count": len(broker_orders),
            "historical_count": len(order_history),
            "status_counts": {
                status: sum(1 for row in order_history if row.get("status") == status)
                for status in sorted({str(row.get("status", "unknown")) for row in order_history})
            },
        },
    )
    tracker.record_event(
        "broker_fill_snapshot", "recorded", run_id=run_id,
        details={"latest_count": len(broker_fills), "historical_count": len(fill_history)},
    )
    state = tracker.record_daily_state(
        date_str=_market_date().isoformat(),
        equity=equity,
        day_pnl=None,
        total_pnl_pct=None,
        allocations=target_allocations,
        actual_allocations=actual_allocations,
        factors=config.get("active_factors", []),
        factor_weights=(
            last_target.get("factor_weights") or _configured_factor_weights(config)
        ),
        target_scalar=sum(abs(w) for w in target_allocations.values()),
        account_snapshot=account_state,
        as_of_date=last_target.get("as_of_date"),
        order_summary={
            "broker_orders_seen": len(order_history),
            "broker_fills_seen": len(fill_history),
        },
        run_id=run_id,
    )
    return {
        "account": account_state,
        "positions": positions,
        "orders": broker_orders,
        "fills": broker_fills,
        "tracker_state": state,
    }


def _execute_account_rebalance_locked(
    account: str,
    config: Dict,
    *,
    force: bool = False,
    run_id: Optional[str] = None,
) -> Dict:
    """One safe live pipeline shared by scheduled and manual execution.

    Strict order: incremental sync -> quality gate -> target calculation ->
    as-of check -> order submission -> snapshots/tracker.  Any data failure is
    fail-closed and therefore cannot reach OMS.
    """
    from backend.data.fetcher import sync_and_validate_live_data
    from backend.execution.oms import OrderManagementSystem
    from backend.execution.strategy import LiveStrategy
    from backend.execution.tracker import LiveTracker

    run_id = run_id or uuid.uuid4().hex
    tracker = LiveTracker(account)
    if _stop_flag_path(account).exists():
        tracker.record_event(
            "rebalance_run", "blocked", run_id=run_id,
            details={"stage": "stop_flag", "path": str(_stop_flag_path(account))},
        )
        return {"error": "Live trading stop flag is active", "stage": "stop_flag", "run_id": run_id}
    if not _is_nyse_session_today(account):
        tracker.record_event(
            "rebalance_run", "blocked", run_id=run_id,
            details={"stage": "market_calendar", "market_date": _market_date().isoformat()},
        )
        return {"error": "Today is not a verified NYSE session", "stage": "market_calendar",
                "run_id": run_id}
    tracker.record_event(
        "rebalance_run", "started", run_id=run_id,
        details={"force": bool(force), "market_time": _market_now().isoformat()},
    )

    sync_report = sync_and_validate_live_data(
        config.get("universe", []), account_name=account, config=config
    )
    tracker.record_event(
        "data_sync", "passed" if sync_report.get("passed") else "blocked",
        run_id=run_id, details=sync_report,
    )
    if not sync_report.get("passed"):
        tracker.record_event(
            "rebalance_run", "blocked", run_id=run_id,
            details={"stage": "data_quality", "data_sync": sync_report},
        )
        return {
            "error": "Pre-trade data quality gate failed",
            "stage": "data_quality",
            "run_id": run_id,
            "data_sync": sync_report,
        }

    try:
        runtime_config = _effective_runtime_config(config, sync_report)
    except Exception as exc:
        tracker.record_event(
            "target_calculation", "blocked", run_id=run_id,
            details={"stage": "asset_eligibility", "error": str(exc), "data_sync": sync_report},
        )
        return {"error": str(exc), "stage": "asset_eligibility", "run_id": run_id,
                "data_sync": sync_report}

    try:
        risk_state = _evaluate_daily_risk(
            account, runtime_config, sync_report["expected_as_of"]
        )
    except Exception as exc:
        tracker.record_event(
            "daily_risk", "blocked", run_id=run_id, details={"error": str(exc)}
        )
        return {"error": str(exc), "stage": "daily_risk", "run_id": run_id,
                "data_sync": sync_report}
    runtime_config["_risk_in_market_override"] = risk_state.get("in_market", True)
    strategy_result = LiveStrategy(account).calculate_targets(runtime_config)
    if "error" in strategy_result:
        tracker.record_event(
            "target_calculation", "failed", run_id=run_id,
            details={"error": strategy_result["error"], "data_sync": sync_report},
        )
        return {**strategy_result, "stage": "target_calculation", "run_id": run_id,
                "data_sync": sync_report}

    expected_as_of = str(sync_report.get("expected_as_of") or "")[:10]
    calculation_as_of = str(strategy_result.get("as_of_date") or "")[:10]
    if not calculation_as_of or calculation_as_of != expected_as_of:
        details = {
            "expected_as_of": expected_as_of,
            "calculation_as_of": calculation_as_of,
            "reason": "Target calculation did not consume the synchronized session",
        }
        tracker.record_event("target_calculation", "blocked", run_id=run_id, details=details)
        return {
            "error": details["reason"], "stage": "as_of_gate", "run_id": run_id,
            "data_sync": sync_report, **details,
        }

    allocations = strategy_result.get("allocations") or {}
    effective_symbols = set(runtime_config["universe"])
    unauthorized_targets = sorted(
        str(symbol) for symbol in allocations
        if str(symbol).strip().upper() not in effective_symbols
    )
    if unauthorized_targets:
        details = {
            "stage": "effective_universe_gate",
            "reason": "Target calculation returned broker-ineligible symbols",
            "unauthorized_targets": unauthorized_targets,
            "effective_universe": sorted(effective_symbols),
        }
        tracker.record_event("target_calculation", "blocked", run_id=run_id, details=details)
        return {
            "error": details["reason"], "stage": "effective_universe_gate",
            "run_id": run_id, "data_sync": sync_report,
            "unauthorized_targets": unauthorized_targets,
        }
    target_details = {
        "as_of_date": calculation_as_of,
        "expected_as_of": expected_as_of,
        "allocation_count": len(allocations),
        "target_gross": sum(abs(float(w)) for w in allocations.values()),
        "target_net": sum(float(w) for w in allocations.values()),
        "allocations": allocations,
        "factor_weights": strategy_result.get("factor_weights") or _configured_factor_weights(config),
        "vol_metrics": strategy_result.get("vol_metrics", {}),
        "daily_risk": risk_state,
        "effective_universe_count": len(effective_symbols),
    }
    tracker.record_event("target_calculation", "passed", run_id=run_id, details=target_details)

    # The Windows task starts at 09:25 ET, but a slow asset/data sync must not
    # turn a pre-open rebalance into an unintended after-open market-order batch.
    # Explicit manual force bypasses cadence/window only; all data/as-of gates
    # above still apply.
    if not force and not _inside_scheduled_window():
        details = {
            "stage": "preopen_window",
            "reason": "Pre-open execution window elapsed before OMS",
            "market_time": _market_now().isoformat(),
        }
        tracker.record_event("rebalance_run", "blocked", run_id=run_id, details=details)
        return {
            "error": details["reason"], "stage": "preopen_window", "run_id": run_id,
            "data_sync": sync_report, "market_time": details["market_time"],
        }

    oms = OrderManagementSystem(account)
    try:
        submitted = oms.generate_and_execute_orders(
            allocations,
            runtime_config,
            run_id=run_id,
            audit_context={
                "as_of_date": calculation_as_of,
                "data_sync": sync_report,
                "factor_weights": target_details["factor_weights"],
            },
        )
    except Exception as exc:
        tracker.record_event(
            "rebalance_run", "failed", run_id=run_id,
            details={"stage": "orders", "error": str(exc)},
        )
        return {"error": str(exc), "stage": "orders", "run_id": run_id,
                "data_sync": sync_report}

    # A partial submission has already changed broker state. Lock the run to
    # prevent an automatic duplicate batch; audit exposes it for manual review.
    _write_lock(account)
    observation = record_broker_observation(account, config, run_id=run_id)
    status = oms.last_summary.get("status", "submitted")
    tracker.record_event(
        "rebalance_run", status, run_id=run_id,
        details={"order_summary": oms.last_summary, "as_of_date": calculation_as_of},
    )
    return {
        "success": status != "failed",
        "status": status,
        "run_id": run_id,
        "orders": submitted,
        "order_summary": oms.last_summary,
        "allocations": allocations,
        "as_of_date": calculation_as_of,
        "data_sync": sync_report,
        "broker_observation": observation,
    }


def _weights_from_snapshot(snapshot: Dict) -> Dict[str, float]:
    return {
        symbol: float(detail.get("weight", 0) or 0)
        for symbol, detail in (snapshot.get("positions") or {}).items()
    }


def _latest_target_snapshot(account: str) -> Dict:
    rebalance = Path(DAILY_LOCK_DIR) / f"last_rebalance_{account}.json"
    overlay = Path(DAILY_LOCK_DIR) / f"last_risk_overlay_{account}.json"
    candidates = [p for p in (rebalance, overlay) if p.exists()]
    if not candidates:
        return {}
    latest = max(candidates, key=lambda p: p.stat().st_mtime_ns)
    try:
        return json.loads(latest.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _execute_daily_risk_overlay_locked(
    account: str,
    config: Dict,
    *,
    run_id: Optional[str] = None,
) -> Dict:
    """Apply active risk controls off cadence without blindly reselecting alpha."""
    from backend.data.fetcher import sync_and_validate_live_data
    from backend.execution.oms import OrderManagementSystem
    from backend.execution.strategy import LiveStrategy
    from backend.execution.tracker import LiveTracker

    run_id = run_id or uuid.uuid4().hex
    tracker = LiveTracker(account)
    if _stop_flag_path(account).exists():
        tracker.record_event("daily_risk", "blocked", run_id=run_id,
                             details={"stage": "stop_flag"})
        return {"error": "Live trading stop flag is active", "stage": "stop_flag"}
    if not _is_nyse_session_today(account):
        tracker.record_event("daily_risk", "blocked", run_id=run_id,
                             details={"stage": "market_calendar"})
        return {"error": "Today is not a verified NYSE session", "stage": "market_calendar"}
    if not _risk_settings_active(config):
        return {"active": False, "status": "not_configured", "run_id": run_id}

    sync_report = sync_and_validate_live_data(
        config.get("universe", []), account_name=account, config=config
    )
    tracker.record_event(
        "data_sync", "passed" if sync_report.get("passed") else "blocked",
        run_id=run_id, details={**sync_report, "purpose": "daily_risk"},
    )
    if not sync_report.get("passed"):
        return {"error": "Daily-risk data quality gate failed", "stage": "data_quality",
                "data_sync": sync_report, "run_id": run_id}

    try:
        runtime_config = _effective_runtime_config(config, sync_report)
    except Exception as exc:
        tracker.record_event(
            "daily_risk", "blocked", run_id=run_id,
            details={"stage": "asset_eligibility", "error": str(exc)},
        )
        return {"error": str(exc), "stage": "asset_eligibility",
                "data_sync": sync_report, "run_id": run_id}

    try:
        risk_state = _evaluate_daily_risk(
            account, runtime_config, sync_report["expected_as_of"]
        )
    except Exception as exc:
        tracker.record_event("daily_risk", "blocked", run_id=run_id,
                             details={"error": str(exc)})
        return {"error": str(exc), "stage": "daily_risk", "run_id": run_id}

    desired = float(risk_state["desired_leverage"])
    # Use actual broker weights, not yesterday's target snapshot. This preserves
    # natural holding drift and retries a partial/failed liquidation next day.
    adapter = AlpacaAdapter(account)
    account_state = adapter.get_account()
    equity = float(account_state.get("equity", 0) or 0)
    if equity <= 0:
        return {"error": "Broker account/equity unavailable", "stage": "risk_positions",
                "run_id": run_id}
    positions = adapter.get_positions()
    broker_market_value = abs(float(account_state.get("long_market_value", 0) or 0)) + abs(
        float(account_state.get("short_market_value", 0) or 0)
    )
    if broker_market_value > 1.0 and not positions:
        return {"error": "Broker positions unavailable", "stage": "risk_positions",
                "run_id": run_id}
    current_weights = {
        p["symbol"]: float(p.get("market_value", 0) or 0) / equity
        for p in positions if p.get("symbol")
    }
    needs_fresh_alpha = bool(risk_state.get("transition")) and desired > 0
    if desired > 0 and not current_weights:
        needs_fresh_alpha = True

    calculation_as_of = sync_report["expected_as_of"]
    decision_factor_weights = (
        _latest_target_snapshot(account).get("factor_weights")
        or _configured_factor_weights(config)
    )
    if desired <= 0:
        target_weights = {}
    elif needs_fresh_alpha:
        # Re-entry/regime transitions deliberately refresh alpha, matching the
        # backtest. Disable the overlay inside LiveStrategy for this raw base
        # calculation, then apply the stateful desired gross here.
        selection_config = dict(runtime_config)
        selection_risk = dict(runtime_config.get("risk_management", {}) or {})
        selection_risk["regime_mode"] = "off"
        selection_risk["volatility_throttle"] = False
        selection_config["risk_management"] = selection_risk
        selection_config["ema_kill_switch"] = False
        selection_config["_risk_in_market_override"] = True
        selection = LiveStrategy(account).calculate_targets(selection_config)
        if "error" in selection:
            return {**selection, "stage": "risk_reentry_alpha", "run_id": run_id}
        calculation_as_of = str(selection.get("as_of_date") or "")[:10]
        if calculation_as_of != str(sync_report["expected_as_of"]):
            return {"error": "Risk re-entry alpha did not use the completed session", "stage": "as_of_gate",
                    "run_id": run_id}
        decision_factor_weights = (
            selection.get("factor_weights") or decision_factor_weights
        )
        raw = selection.get("allocations") or {}
        gross = sum(abs(float(w)) for w in raw.values())
        target_weights = {
            symbol: float(weight) * desired / gross for symbol, weight in raw.items()
        } if gross > 0 else {}
    else:
        gross = sum(abs(float(w)) for w in current_weights.values())
        target_weights = {
            symbol: float(weight) * desired / gross for symbol, weight in current_weights.items()
        } if gross > 0 else {}

    unauthorized_targets = sorted(
        str(symbol) for symbol in target_weights
        if str(symbol).strip().upper() not in set(runtime_config["universe"])
    )
    if unauthorized_targets:
        tracker.record_event(
            "daily_risk", "blocked", run_id=run_id,
            details={
                "stage": "effective_universe_gate",
                "reason": "Daily-risk target contains broker-ineligible symbols",
                "unauthorized_targets": unauthorized_targets,
            },
        )
        return {
            "error": "Daily-risk target contains broker-ineligible symbols",
            "stage": "effective_universe_gate",
            "unauthorized_targets": unauthorized_targets,
            "data_sync": sync_report,
            "run_id": run_id,
        }

    changed = any(
        abs(float(target_weights.get(symbol, 0)) - float(current_weights.get(symbol, 0))) > 1e-8
        for symbol in set(target_weights) | set(current_weights)
    )
    tracker.record_event(
        "daily_risk", "change_required" if changed else "unchanged", run_id=run_id,
        details={
            **risk_state,
            "current_gross": sum(abs(w) for w in current_weights.values()),
            "target_gross": sum(abs(w) for w in target_weights.values()),
            "reselected_alpha": needs_fresh_alpha,
        },
    )
    if not changed:
        return {"active": True, "status": "unchanged", "run_id": run_id,
                "risk_state": risk_state, "data_sync": sync_report}

    if not _inside_scheduled_window():
        details = {
            "stage": "preopen_window",
            "reason": "Pre-open execution window elapsed before daily-risk OMS",
            "market_time": _market_now().isoformat(),
        }
        tracker.record_event("daily_risk", "blocked", run_id=run_id, details=details)
        return {
            "error": details["reason"], "stage": "preopen_window", "run_id": run_id,
            "risk_state": risk_state, "data_sync": sync_report,
            "market_time": details["market_time"],
        }

    oms = OrderManagementSystem(account)
    try:
        submitted = oms.generate_and_execute_orders(
            target_weights,
            runtime_config,
            run_id=run_id,
            audit_context={
                "as_of_date": calculation_as_of,
                "data_sync": sync_report,
                "snapshot_kind": "risk_overlay",
                "risk_state": risk_state,
                "factor_weights": decision_factor_weights,
            },
        )
    except Exception as exc:
        return {"error": str(exc), "stage": "risk_orders", "run_id": run_id,
                "risk_state": risk_state}
    observation = record_broker_observation(account, config, run_id=run_id)
    return {
        "active": True,
        "status": oms.last_summary.get("status", "submitted"),
        "run_id": run_id,
        "orders": submitted,
        "order_summary": oms.last_summary,
        "risk_state": risk_state,
        "data_sync": sync_report,
        "broker_observation": observation,
    }


def _execution_lock_blocked_result(account: str, run_id: str, lock: Dict, event_type: str) -> Dict:
    from backend.execution.tracker import LiveTracker

    details = {
        "stage": "execution_lock",
        "reason": lock.get("reason", "account execution already in progress"),
        "lock_path": lock.get("path"),
        "owner": lock.get("owner", {}),
        "age_seconds": lock.get("age_seconds"),
    }
    LiveTracker(account).record_event(
        event_type, "blocked", run_id=run_id, details=details
    )
    return {
        "error": "Another live execution is already in progress for this account",
        "stage": "execution_lock",
        "run_id": run_id,
        "execution_lock": details,
    }


def execute_account_rebalance(
    account: str,
    config: Dict,
    *,
    force: bool = False,
    run_id: Optional[str] = None,
) -> Dict:
    """Account-serialized public rebalance entry for web and scheduler."""
    run_id = run_id or uuid.uuid4().hex
    lock = _try_acquire_account_execution_lock(account, run_id)
    if not lock.get("acquired"):
        return _execution_lock_blocked_result(account, run_id, lock, "rebalance_run")
    try:
        return _execute_account_rebalance_locked(
            account, config, force=force, run_id=run_id
        )
    finally:
        _release_account_execution_lock(lock)


def execute_daily_risk_overlay(
    account: str,
    config: Dict,
    *,
    run_id: Optional[str] = None,
) -> Dict:
    """Account-serialized public daily-risk entry for the scheduler."""
    run_id = run_id or uuid.uuid4().hex
    lock = _try_acquire_account_execution_lock(account, run_id)
    if not lock.get("acquired"):
        return _execution_lock_blocked_result(account, run_id, lock, "daily_risk")
    try:
        return _execute_daily_risk_overlay_locked(account, config, run_id=run_id)
    finally:
        _release_account_execution_lock(lock)


class AncserEventLoop:
    def __init__(self):
        self.scheduler = BackgroundScheduler(executors={"default": ThreadPoolExecutor(2)})
        self.running = False

    def rebalance_check(self, force: bool = False):
        accounts = get_configured_accounts()
        for account in accounts:
            cfg_path = f"config/live_strategy_{account}.json" if account != "Main" else "config/live_strategy.json"
            if not os.path.exists(cfg_path):
                logger.warning(f"No config for {account}, skipping.")
                continue

            try:
                config = json.load(open(cfg_path, encoding="utf-8"))
                run_id = uuid.uuid4().hex

                # A weekly strategy still needs daily P&L and order lifecycle
                # observations. This is read-only broker reconciliation.
                try:
                    record_broker_observation(account, config, run_id=run_id)
                except Exception as observation_error:
                    logger.error(f"{account} broker observation failed: {observation_error}")

                if _is_locked(account) and not force:
                    logger.info(f"{account} already rebalanced today.")
                    continue

                if not _should_rebalance_today(config, account, force):
                    freq = str(config.get("rebalance_frequency", "weekly")).lower()
                    if _risk_settings_active(config):
                        risk_result = execute_daily_risk_overlay(
                            account, config, run_id=run_id
                        )
                        if "error" in risk_result:
                            logger.error(
                                f"{account} daily risk blocked at {risk_result.get('stage')}: "
                                f"{risk_result['error']}"
                            )
                        else:
                            logger.info(
                                f"{account}: {freq} alpha not due; daily risk "
                                f"{risk_result.get('status')}"
                            )
                    else:
                        logger.info(f"{account}: {freq} rebalance - not due; account state recorded.")
                    continue

                result = execute_account_rebalance(account, config, force=force, run_id=run_id)
                if "error" in result:
                    logger.error(
                        f"{account} rebalance blocked at {result.get('stage')}: {result['error']}"
                    )
                    continue
                logger.info(
                    f"{account} rebalance {result.get('status')}: "
                    f"{len(result.get('orders', []))} orders, as-of {result.get('as_of_date')}"
                )

            except Exception as e:
                logger.error(f"Rebalance failed for {account}: {e}")
                import traceback; traceback.print_exc()

    def start(self):
        logger.info("Starting AncserEventLoop...")
        # Rebalance at 09:25 ET (market open minus 5 minutes) Mon-Fri.
        # The New York timezone follows DST independently of the host timezone.
        self.scheduler.add_job(
            self.rebalance_check, "cron",
            day_of_week="mon-fri", hour=9, minute=25,
            timezone="America/New_York", id="rebalance",
        )
        # Daily data sync at 20:00 ET (after market close)
        self.scheduler.add_job(
            lambda: __import__("backend.data.fetcher", fromlist=["fetch_incremental"]).fetch_incremental(),
            "cron", day_of_week="mon-fri", hour=20, minute=0,
            timezone="America/New_York", id="data_sync",
        )
        self.scheduler.start()
        self.running = True
        logger.info("Scheduler armed; no immediate startup trade. Next check is 09:25 ET.")
        try:
            while True:
                time.sleep(5)
        except (KeyboardInterrupt, SystemExit):
            self.stop()

    def stop(self):
        self.scheduler.shutdown()
        self.running = False
        _remove_pid()


def run_once(force: bool = False):
    AncserEventLoop().rebalance_check(force=force)


def _inside_scheduled_window(now: Optional[datetime] = None) -> bool:
    """Allow the intended pre-open window; reject late catch-up executions."""
    if now is None:
        current = _market_now()
    elif now.tzinfo is None:
        current = now.replace(tzinfo=MARKET_TZ)
    else:
        current = now.astimezone(MARKET_TZ)
    if current.weekday() >= 5:
        return False
    minute = current.hour * 60 + current.minute
    return 9 * 60 + 20 <= minute <= 9 * 60 + 29


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-once", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--scheduled", action="store_true",
        help="Require the current time to be within 09:20-09:29 America/New_York",
    )
    args = parser.parse_args()
    if args.run_once:
        if _check_single_instance():
            exit(0)
        try:
            if args.scheduled and not _inside_scheduled_window():
                logger.error("Scheduled run is outside the 09:25 ET safety window; no trading attempted.")
            else:
                run_once(force=args.force)
        finally:
            _remove_pid()
    else:
        if _check_single_instance():
            exit(0)
        AncserEventLoop().start()
