"""
ancserAPX FastAPI server.

Serves the HTML/CSS/JS frontend and exposes REST + WebSocket API.

Start:  uvicorn frontend.server:app --host 0.0.0.0 --port 8080 --reload
"""

import asyncio
import json
import logging
import os
import sys
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ── Path bootstrap (allow running from repo root) ─────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
load_dotenv(_ROOT / ".env")

from backend.data.constituents import SP500_TICKERS, NASDAQ100_TICKERS, SPY_QQQ_TICKERS, UNIVERSE_PRESETS
from backend.alpha.factors import (
    ALL_FACTORS, FACTOR_PRESETS, FACTOR_WEIGHT_PRESETS, PRESET_DEFAULTS,
    STRATEGY_PRESETS, SECONDARY_FACTORS, PRIMARY_FACTORS,
)
from backend.utils.accounts import get_configured_accounts

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("ancserAPX.server")

# ── WebSocket connection pool ─────────────────────────────────────────────────
_ws_clients: Set[WebSocket] = set()


async def ws_broadcast(level: str, msg: str):
    """Push a log entry to all connected WebSocket clients."""
    payload = json.dumps({"type": "log", "level": level, "msg": msg})
    dead = set()
    for ws in _ws_clients:
        try:
            await ws.send_text(payload)
        except Exception:
            dead.add(ws)
    _ws_clients.difference_update(dead)


def ws_log_sync(level: str, msg: str):
    """Thread-safe wrapper to broadcast from sync code (fetcher, backtest)."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            asyncio.run_coroutine_threadsafe(ws_broadcast(level, msg), loop)
    except Exception:
        pass


# ── Startup: incremental data sync ───────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("ancserAPX server starting…")
    # Async incremental data sync on startup (non-blocking)
    asyncio.create_task(_background_data_sync())
    yield
    logger.info("ancserAPX server stopping.")


async def _background_data_sync():
    """Run incremental fetch in background thread on startup."""
    await asyncio.sleep(3)  # wait for server to fully start
    def _run():
        try:
            from backend.data.fetcher import fetch_incremental
            fetch_incremental(callback=ws_log_sync)
        except Exception as e:
            logger.warning(f"Startup data sync failed: {e}")
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _run)


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="ancserAPX", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_STATIC = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(_STATIC)), name="static")


# ── HTML root ─────────────────────────────────────────────────────────────────
@app.get("/")
async def root():
    return FileResponse(str(_STATIC / "ancserAPX.html"))


# ── WebSocket ─────────────────────────────────────────────────────────────────
@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    _ws_clients.add(ws)
    try:
        while True:
            data = await ws.receive_text()
            # Echo control messages back if needed
            if data == "ping":
                await ws.send_text(json.dumps({"type": "pong"}))
    except WebSocketDisconnect:
        pass
    finally:
        _ws_clients.discard(ws)


# ── Config ─────────────────────────────────────────────────────────────────────
@app.get("/config")
async def get_config():
    accounts = []
    try:
        accounts = get_configured_accounts()
    except Exception:
        pass
    return {
        "factors": ALL_FACTORS,
        "primary_factors": PRIMARY_FACTORS,
        "secondary_factors": SECONDARY_FACTORS,
        "factor_presets": FACTOR_PRESETS,
        "factor_weight_presets": FACTOR_WEIGHT_PRESETS,
        "preset_defaults": PRESET_DEFAULTS,
        "strategy_presets": STRATEGY_PRESETS,
        "universes": {
            "spy_qqq": SPY_QQQ_TICKERS,
        },
        "accounts": accounts if accounts else ["Main"],
        "data_feed": os.getenv("APCA_DATA_FEED", "iex").upper(),
    }


# ── Data status ───────────────────────────────────────────────────────────────
@app.get("/data/status")
async def data_status():
    from backend.data.store import get_coverage_stats
    return get_coverage_stats(SP500_TICKERS)


# ── Data fetch (triggers async bulk/incremental fetch) ───────────────────────
class FetchRequest(BaseModel):
    mode: str = "incremental"   # "bulk" | "incremental"
    symbols: Optional[List[str]] = None
    start_date: Optional[str] = None
    account: str = "Main"


@app.post("/data/fetch")
async def trigger_fetch(req: FetchRequest):
    symbols = req.symbols or SP500_TICKERS

    def _run():
        from backend.data.fetcher import fetch_bulk, fetch_incremental
        if req.mode == "bulk":
            sd = req.start_date or "2015-01-01"
            return fetch_bulk(symbols, start_date=sd, account_name=req.account, callback=ws_log_sync)
        else:
            return fetch_incremental(symbols=symbols, account_name=req.account, callback=ws_log_sync)

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, _run)
    return result


# ── Backtest ──────────────────────────────────────────────────────────────────
class BacktestRequest(BaseModel):
    symbols: Optional[List[str]] = None
    universe: str = "spy_qqq"     # single default: SPY + QQQ constituents
    start_date: str = "2020-01-01"
    end_date: Optional[str] = None
    active_factors: List[str] = ["Momentum", "Reversion", "EMA200 Distance"]
    capital: float = 100_000.0
    leverage: float = 1.0
    use_mwu: bool = False
    top_n: int = 30
    neutralize_sector: bool = False
    factor_weights: Optional[Dict[str, float]] = None
    winner_lock: Optional[Dict[str, float]] = None  # secondary winner-lock rules
    strategy_preset: Optional[str] = None   # e.g. "Claude #1" → sleeve/lock run
    # Risk management (daily check, off the weekly cadence)
    ema_kill_switch: bool = False           # liquidate <200EMA, re-enter >20EMA


@app.post("/backtest/run")
async def run_backtest(req: BacktestRequest):
    # Resolve symbols
    if req.symbols:
        symbols = req.symbols
    else:
        symbols = UNIVERSE_PRESETS.get(req.universe, SPY_QQQ_TICKERS)

    end_date = req.end_date or datetime.now().strftime("%Y-%m-%d")

    def _run():
        from backend.backtest.engine import BacktestEngine, compute_metrics, _compute_benchmark_curve

        def _notify(level, msg):
            ws_log_sync(level, msg)

        _notify("info", f"Backtest starting: {len(symbols)} symbols, {req.start_date}→{end_date}")
        _notify("info", f"Factors: {', '.join(req.active_factors)}")

        engine = BacktestEngine(initial_capital=req.capital)

        risk_on = req.ema_kill_switch
        if risk_on:
            _notify("info", "Risk mgmt: 200EMA kill-switch")

        sp = STRATEGY_PRESETS.get(req.strategy_preset) if req.strategy_preset else None
        if sp:
            _notify("info", f"Strategy preset: {sp.get('label', req.strategy_preset)} "
                            f"(leverage {sp.get('leverage', 1.0)}x, "
                            f"{len(sp.get('sleeves', []))} sleeves)")
            res_df, w_df, h_df = engine.run_strategy(
                symbols=symbols,
                start_date=req.start_date,
                end_date=end_date,
                sleeves=sp["sleeves"],
                leverage=float(sp.get("leverage", req.leverage)),
                top_n=int(sp.get("top_n", req.top_n)),
                lock_rules=sp.get("winner_lock", {}),
                ema_kill_switch=req.ema_kill_switch,
            )
        elif risk_on:
            # Risk overlays only exist in the weekly-hold path, so route custom
            # factors through run_strategy as a single sleeve when risk is on.
            sleeve = {
                "name": "Custom",
                "alloc": 1.0,
                "factors": req.active_factors,
                "weights": req.factor_weights or {},
                "winner_lock": False,
            }
            res_df, w_df, h_df = engine.run_strategy(
                symbols=symbols,
                start_date=req.start_date,
                end_date=end_date,
                sleeves=[sleeve],
                leverage=req.leverage,
                top_n=req.top_n,
                lock_rules={},
                ema_kill_switch=req.ema_kill_switch,
            )
        else:
            res_df, w_df, h_df = engine.run(
                symbols=symbols,
                start_date=req.start_date,
                end_date=end_date,
                active_factors=req.active_factors,
                leverage=req.leverage,
                use_mwu=req.use_mwu,
                use_vol_target=False,
                vol_target_pct=0.20,
                strategy_mode="long_only",
                top_n=req.top_n,
                neutralize_sector=req.neutralize_sector,
                factor_weights=req.factor_weights,
            )

        if res_df.empty:
            return {"error": "No backtest results — check data availability."}

        metrics = compute_metrics(res_df, req.capital)
        _notify("success", f"Backtest done. CAGR={metrics.get('cagr_pct',0):.1f}% Sharpe={metrics.get('sharpe',0):.2f}")

        # Equity curve
        res_df_reset = res_df.reset_index()
        equity_curve = [
            {"date": str(r["date"])[:10], "value": round(r["equity"], 2)}
            for _, r in res_df_reset.iterrows()
        ]

        # QQQ benchmark (buy-and-hold, same starting funding as the strategy)
        benchmark_curve = _compute_benchmark_curve(req.start_date, end_date, req.capital, symbol="QQQ")

        # Holdings log (last 60 rows)
        holdings = []
        if not h_df.empty:
            h_reset = h_df.reset_index().tail(60)
            for _, row in h_reset.iterrows():
                holdings.append({
                    "date": str(row.get("date", ""))[:10],
                    "long": str(row.get("long", "")),
                    "short": str(row.get("short", "")),
                })

        # Factor weights
        weights_out = []
        if not w_df.empty:
            w_reset = w_df.reset_index().tail(60)
            for _, row in w_reset.iterrows():
                entry = {"date": str(row.get("date", ""))[:10]}
                for f in req.active_factors:
                    entry[f] = round(float(row.get(f, 0)), 4) if f in row else 0.0
                weights_out.append(entry)

        return {
            "metrics": metrics,
            "equity_curve": equity_curve,
            "benchmark_curve": benchmark_curve,
            "benchmark": "QQQ",
            "spy_curve": benchmark_curve,  # backwards-compat alias
            "holdings": holdings,
            "factor_weights": weights_out,
            "params": {
                "symbols_count": len(symbols),
                "start_date": req.start_date,
                "end_date": end_date,
                "factors": req.active_factors,
                "leverage": float(sp.get("leverage", req.leverage)) if sp else req.leverage,
            },
        }

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, _run)
    if "error" in result:
        return JSONResponse(status_code=400, content=result)
    return result


# ── Live / Dashboard ──────────────────────────────────────────────────────────
@app.get("/live/status")
async def live_status(account: str = "Main"):
    def _run():
        from backend.data.alpaca_adapter import AlpacaAdapter
        try:
            adapter = AlpacaAdapter(account)
            acct = adapter.get_account()
            positions = adapter.get_positions()
            clock = adapter.get_clock()

            # Read saved config
            cfg_path = f"config/live_strategy_{account}.json" if account != "Main" else "config/live_strategy.json"
            live_config = {}
            if os.path.exists(cfg_path):
                try:
                    live_config = json.load(open(cfg_path))
                except Exception:
                    pass

            # Last rebalance
            snap_path = f"logs/last_rebalance_{account}.json"
            last_rebalance = None
            if os.path.exists(snap_path):
                try:
                    snap = json.load(open(snap_path))
                    last_rebalance = snap.get("rebalance_date")
                except Exception:
                    pass

            return {
                "account": acct,
                "positions": positions,
                "market": clock,
                "live_config": live_config,
                "last_rebalance": last_rebalance,
            }
        except Exception as e:
            return {"error": str(e)}

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _run)


@app.get("/live/equity")
async def live_equity(account: str = "Main", period: str = "1Y"):
    def _run():
        from backend.data.alpaca_adapter import AlpacaAdapter
        try:
            adapter = AlpacaAdapter(account)
            hist = adapter.get_portfolio_history(period=period, timeframe="1D")
            if hist.empty:
                return {"equity_curve": []}
            hist = hist.reset_index()
            result = [
                {"date": str(row["timestamp"])[:10], "value": round(float(row["equity"]), 2)}
                for _, row in hist.iterrows()
                if not pd.isna(row["equity"]) and float(row["equity"]) > 0
            ]
            return {"equity_curve": result}
        except Exception as e:
            return {"error": str(e), "equity_curve": []}

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _run)


@app.get("/live/activities")
async def live_activities(account: str = "Main", limit: int = 100):
    def _run():
        from backend.data.alpaca_adapter import AlpacaAdapter
        try:
            return AlpacaAdapter(account).get_activities(limit=limit)
        except Exception as e:
            return []

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _run)


# ── Live config save ──────────────────────────────────────────────────────────
class LiveConfigRequest(BaseModel):
    account: str = "Main"
    active_factors: List[str]
    universe: str = "spy_qqq"
    symbols: Optional[List[str]] = None
    leverage: float = 1.0
    use_mwu: bool = False
    use_vol_target: bool = False
    vol_target_pct: float = 0.20
    strategy_mode: str = "long_only"
    top_n: int = 30
    neutralize_sector: bool = False
    rebalance_frequency: str = "weekly"
    rebalance_weekday: int = 4  # 0=Mon ... 4=Fri


@app.post("/live/save")
async def live_save(req: LiveConfigRequest):
    symbols = req.symbols or UNIVERSE_PRESETS.get(req.universe, SPY_QQQ_TICKERS)
    config = {
        "active_factors": req.active_factors,
        "universe": symbols,
        "leverage": req.leverage,
        "use_mwu": req.use_mwu,
        "use_vol_target": req.use_vol_target,
        "vol_target": req.vol_target_pct,
        "strategy_mode": req.strategy_mode,
        "top_n": req.top_n,
        "neutralize_sector": req.neutralize_sector,
        "rebalance_frequency": req.rebalance_frequency,
        "rebalance_weekday": req.rebalance_weekday,
        "saved_at": datetime.now().isoformat(),
    }
    cfg_path = f"config/live_strategy_{req.account}.json" if req.account != "Main" else "config/live_strategy.json"
    os.makedirs("config", exist_ok=True)
    with open(cfg_path, "w") as f:
        json.dump(config, f, indent=2)
    await ws_broadcast("success", f"Live config saved for account [{req.account}]")
    return {"saved": True, "path": cfg_path}


# ── Apply a backtested strategy to the live config ────────────────────────────
class ApplyLiveRequest(BaseModel):
    account: str = "Main"
    strategy_preset: Optional[str] = None       # e.g. "Claude #1"
    # fallback fields (used when no strategy_preset is given)
    active_factors: Optional[List[str]] = None
    universe: str = "spy_qqq"
    leverage: float = 1.0
    top_n: int = 20
    factor_weights: Optional[Dict[str, float]] = None
    use_mwu: bool = False
    neutralize_sector: bool = False
    winner_lock: Optional[Dict[str, float]] = None  # secondary winner-lock rules
    # Risk management (applies to both preset and custom)
    ema_kill_switch: bool = False


@app.post("/live/apply")
async def live_apply(req: ApplyLiveRequest):
    """Write the currently-selected backtest strategy into the live config so the
    next daily/weekly rebalance executes it. Logs exactly what was applied."""
    sp = STRATEGY_PRESETS.get(req.strategy_preset) if req.strategy_preset else None
    # Universe is a backtest-only knob — live always trades the full SPY+QQQ set.
    symbols = SPY_QQQ_TICKERS

    if sp:
        # union of all sleeve factors (so the live executor has the full set)
        union_factors, merged_w = [], {}
        for sl in sp.get("sleeves", []):
            for f in sl.get("factors", []):
                if f not in union_factors:
                    union_factors.append(f)
            for f, wv in (sl.get("weights") or {}).items():
                merged_w[f] = merged_w.get(f, 0.0) + float(wv) * float(sl.get("alloc", 0.0))
        tw = sum(merged_w.values()) or 1.0
        merged_w = {f: round(v / tw, 4) for f, v in merged_w.items()}
        config = {
            "strategy_preset": req.strategy_preset,
            "strategy_label": sp.get("label", req.strategy_preset),
            "active_factors": union_factors,
            "factor_weights": merged_w,
            "universe": symbols,
            "leverage": float(sp.get("leverage", 1.0)),
            "top_n": int(sp.get("top_n", req.top_n)),
            "sleeves": sp.get("sleeves", []),
            "winner_lock": sp.get("winner_lock", {}),
            "rebalance_frequency": sp.get("rebalance_frequency", "weekly"),
            "rebalance_weekday": 4,
            "use_mwu": False,
            "use_vol_target": False,
            "strategy_mode": "long_only",
            "ema_kill_switch": req.ema_kill_switch,
            "saved_at": datetime.now().isoformat(),
        }
        desc = (f"{sp.get('label', req.strategy_preset)} | "
                f"leverage {config['leverage']}x | top{config['top_n']} | "
                f"sleeves " + ", ".join(f"{s['name']} {int(s['alloc']*100)}%"
                                        + ("+lock" if s.get("winner_lock") else "")
                                        for s in sp.get("sleeves", []))
                + (" | 200EMA-kill" if req.ema_kill_switch else ""))
    else:
        config = {
            "strategy_preset": None,
            "active_factors": req.active_factors or [],
            "factor_weights": req.factor_weights or {},
            "universe": symbols,
            "leverage": req.leverage,
            "top_n": req.top_n,
            "use_mwu": req.use_mwu,
            "neutralize_sector": req.neutralize_sector,
            "winner_lock": req.winner_lock or {},
            "rebalance_frequency": "weekly",
            "rebalance_weekday": 4,
            "use_vol_target": False,
            "strategy_mode": "long_only",
            "ema_kill_switch": req.ema_kill_switch,
            "saved_at": datetime.now().isoformat(),
        }
        desc = (f"custom | leverage {req.leverage}x | top{req.top_n} | "
                f"factors {', '.join(req.active_factors or [])}"
                + (" | MWU" if req.use_mwu else "")
                + (" | 200EMA-kill" if req.ema_kill_switch else ""))

    cfg_path = f"config/live_strategy_{req.account}.json" if req.account != "Main" else "config/live_strategy.json"
    os.makedirs("config", exist_ok=True)
    with open(cfg_path, "w") as f:
        json.dump(config, f, indent=2)

    await ws_broadcast("success", f"✓ APPLIED LIVE [{req.account}]: {desc}")
    await ws_broadcast("info", f"This strategy takes effect on the NEXT daily rebalance "
                               f"({config['rebalance_frequency']}, "
                               f"{len(symbols)} symbols). Config written to {cfg_path}.")
    return {"applied": True, "path": cfg_path, "config": config, "description": desc}


# ── Force rebalance ───────────────────────────────────────────────────────────
class ExecuteRequest(BaseModel):
    account: str = "Main"
    force: bool = True


@app.post("/live/execute")
async def live_execute(req: ExecuteRequest):
    def _run():
        cfg_path = f"config/live_strategy_{req.account}.json" if req.account != "Main" else "config/live_strategy.json"
        if not os.path.exists(cfg_path):
            return {"error": "No live config saved. Save config first."}
        try:
            ws_log_sync("info", f"Force rebalance triggered for [{req.account}]...")
            config = json.load(open(cfg_path))
            from backend.execution.strategy import LiveStrategy
            res = LiveStrategy(req.account).calculate_targets(config)
            if "error" in res:
                ws_log_sync("error", f"Strategy error: {res['error']}")
                return res
            from backend.execution.oms import OrderManagementSystem
            orders = OrderManagementSystem(req.account).generate_and_execute_orders(
                res["allocations"], config)
            ws_log_sync("success", f"Rebalance complete: {len(orders)} orders for [{req.account}]")
            return {"orders": len(orders), "allocations": len(res["allocations"])}
        except Exception as e:
            ws_log_sync("error", str(e))
            return {"error": str(e)}

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _run)


# ── Stop live ─────────────────────────────────────────────────────────────────
@app.post("/live/stop")
async def live_stop(account: str = "Main"):
    stop_path = f"logs/stop_{account}.flag"
    os.makedirs("logs", exist_ok=True)
    with open(stop_path, "w") as f:
        f.write(datetime.now().isoformat())
    await ws_broadcast("warn", f"Live trading STOPPED for [{account}]. Flag written.")
    return {"stopped": True}


# ── Live preview (calculate targets without executing) ───────────────────────
@app.get("/live/preview")
async def live_preview(account: str = "Main"):
    def _run():
        cfg_path = f"config/live_strategy_{account}.json" if account != "Main" else "config/live_strategy.json"
        if not os.path.exists(cfg_path):
            return {"error": "No live config saved."}
        config = json.load(open(cfg_path))
        from backend.execution.strategy import LiveStrategy
        return LiveStrategy(account).calculate_targets(config)

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _run)
