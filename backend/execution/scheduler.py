"""
AncserEventLoop — daily rebalance scheduler.
Runs as a standalone process: python -m backend.execution.scheduler [--run-once] [--force]
"""

import argparse, json, logging, os, subprocess, time
from datetime import datetime, timedelta
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
        return data.get("date") == datetime.now().strftime("%Y-%m-%d")
    except Exception:
        return False


def _write_lock(account: str):
    os.makedirs(DAILY_LOCK_DIR, exist_ok=True)
    json.dump({"date": datetime.now().strftime("%Y-%m-%d"), "at": datetime.now().isoformat()},
              open(f"{DAILY_LOCK_DIR}/daily_lock_{account}.json", "w"))


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


def _should_rebalance_today(config: dict, account: str, force: bool = False) -> bool:
    """
    Decide whether today's trigger should actually rebalance.

    The schedule fires every weekday (and the user also runs it daily), but a
    WEEKLY strategy must trade at most once per week. Policy:
      - force            -> always
      - frequency=daily  -> always
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
    if force:
        return True
    freq = str(config.get("rebalance_frequency", "weekly")).lower()
    if freq == "daily":
        return True

    today = datetime.now().date()
    last = _last_rebalance_date(account)

    # First run ever — establish the position immediately, any day.
    if last is None:
        return True

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
            if _is_locked(account) and not force:
                logger.info(f"{account} already rebalanced today.")
                continue

            try:
                config = json.load(open(cfg_path))

                if not _should_rebalance_today(config, account, force):
                    freq = str(config.get("rebalance_frequency", "weekly")).lower()
                    logger.info(f"{account}: {freq} rebalance — not due today, skipping.")
                    continue

                from backend.execution.strategy import LiveStrategy
                res = LiveStrategy(account).calculate_targets(config)
                if "error" in res:
                    logger.error(f"{account} strategy error: {res['error']}")
                    continue

                from backend.execution.oms import OrderManagementSystem
                OrderManagementSystem(account).generate_and_execute_orders(res["allocations"], config)

                # Track
                try:
                    from backend.execution.tracker import LiveTracker
                    adapter = AlpacaAdapter(account)
                    acct = adapter.get_account()
                    LiveTracker(account).record_daily_state(
                        date_str=datetime.now().strftime("%Y-%m-%d"),
                        equity=float(acct.get("equity", 0)),
                        day_pnl=0.0,
                        total_pnl_pct=0.0,
                        allocations=res["allocations"],
                        factors=config.get("active_factors", []),
                        target_scalar=res.get("vol_metrics", {}).get("final_scalar", 1.0),
                    )
                except Exception as te:
                    logger.error(f"Tracker error: {te}")

                _write_lock(account)
                logger.info(f"{account} rebalance complete.")

                # Incremental data sync after trading
                try:
                    from backend.data.fetcher import fetch_incremental
                    fetch_incremental(config.get("universe", []), account_name=account)
                except Exception as fe:
                    logger.warning(f"Data sync after rebalance failed: {fe}")

            except Exception as e:
                logger.error(f"Rebalance failed for {account}: {e}")
                import traceback; traceback.print_exc()

    def start(self):
        logger.info("Starting AncserEventLoop...")
        # Rebalance at 09:35 ET (market open + 5 min) Mon–Fri
        self.scheduler.add_job(
            self.rebalance_check, "cron",
            day_of_week="mon-fri", hour=9, minute=35,
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
        logger.info("Running initial rebalance check...")
        self.rebalance_check()
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-once", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.run_once:
        run_once(force=args.force)
    else:
        if _check_single_instance():
            exit(0)
        AncserEventLoop().start()
