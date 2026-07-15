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

from backend.data.constituents import SPY_QQQ_TICKERS, UNIVERSE_PRESETS
from backend.alpha.factors import (
    ALL_FACTORS, FACTOR_PRESETS, FACTOR_WEIGHT_PRESETS, PRESET_DEFAULTS,
    STRATEGY_PRESETS, SECONDARY_FACTORS, PRIMARY_FACTORS,
)
from backend.alpha.models import DEFAULT_MODEL_ID, list_models, require_model
from backend.utils.accounts import (
    get_account_paper,
    get_configured_account_details,
)
from backend.utils.performance import fifo_realized_pnl

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


def _effective_factor_weights(config: Dict[str, Any]) -> Dict[str, float]:
    """Return the weights that the live scoring path actually consumes."""
    sleeves = config.get("sleeves") or []
    if sleeves:
        merged: Dict[str, float] = {}
        for sleeve in sleeves:
            allocation = float(sleeve.get("alloc", 0.0) or 0.0)
            for factor, weight in (sleeve.get("weights") or {}).items():
                merged[str(factor)] = merged.get(str(factor), 0.0) + allocation * float(weight)
        total = sum(abs(value) for value in merged.values())
        return {key: value / total for key, value in merged.items()} if total > 0 else merged
    weights = {str(k): float(v) for k, v in (config.get("factor_weights") or {}).items()}
    total = sum(abs(value) for value in weights.values())
    return {key: value / total for key, value in weights.items()} if total > 0 else weights


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
            # Warm the same broad stock universe and market gauges required by
            # the live pre-trade gate. The live runner still performs its own
            # authoritative sync/physical validation immediately before orders.
            startup_symbols = list(dict.fromkeys([*SPY_QQQ_TICKERS, "QQQ", "SPY"]))
            fetch_incremental(symbols=startup_symbols, callback=ws_log_sync)
        except Exception as e:
            logger.warning(f"Startup data sync failed: {e}")
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _run)


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="ancserAPX", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_STATIC = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(_STATIC)), name="static")

_BACKTEST_RUNS_DIR = _ROOT / "logs" / "backtest_runs"
_LEGACY_BACKTEST_DIR = _ROOT / "logs" / "backtest"


def _persist_backtest_run(result: Dict[str, Any], req: "BacktestRequest") -> Dict[str, Any]:
    """Persist complete chart/results data so reloads do not erase research history."""
    now = datetime.now()
    run_id = now.strftime("%Y%m%dT%H%M%S%f")
    preset = req.ui_preset_label or req.strategy_preset or "Custom"
    end_label = req.end_date or now.strftime("%Y-%m-%d")
    label = f"{preset} · {req.model_id} · {req.start_date}→{end_label}"
    enriched = dict(result)
    enriched.update({
        "run_id": run_id,
        "created_at": now.isoformat(),
        "label": label,
    })
    payload = {
        "run_id": run_id,
        "created_at": enriched["created_at"],
        "label": label,
        "request": req.model_dump() if hasattr(req, "model_dump") else req.dict(),
        "result": enriched,
    }
    _BACKTEST_RUNS_DIR.mkdir(parents=True, exist_ok=True)
    target = _BACKTEST_RUNS_DIR / f"{run_id}.json"
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    temporary.replace(target)
    return enriched


def _legacy_backtest_record(path: Path) -> Optional[Dict[str, Any]]:
    """Adapt historical ancser backtest summaries for the unified history UI."""
    try:
        legacy = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning(f"Skipping unreadable legacy backtest {path.name}: {exc}")
        return None
    created_at = str(
        legacy.get("timestamp")
        or datetime.fromtimestamp(path.stat().st_mtime).isoformat()
    )
    run_type = str(legacy.get("run_type") or "legacy")
    config = legacy.get("config") or {}
    stats = legacy.get("stats") or {}
    strategy_stats = stats.get("strategy") if isinstance(stats, dict) else None
    metrics: Dict[str, Any] = {}
    if isinstance(strategy_stats, dict):
        metrics = {
            "cagr_pct": float(strategy_stats.get("cagr", 0) or 0) * 100,
            "sharpe": float(strategy_stats.get("sharpe", 0) or 0),
            "max_dd_pct": float(strategy_stats.get("max_dd", 0) or 0) * 100,
            "calmar": float(strategy_stats.get("calmar", 0) or 0),
            "win_rate_pct": float(strategy_stats.get("win_rate", 0) or 0) * 100,
        }
    summary = legacy.get("equity_summary") or {}
    equity_curve: List[Dict[str, Any]] = []
    if all(summary.get(key) is not None for key in ("start_date", "end_date", "start_value", "end_value")):
        equity_curve = [
            {"date": str(summary["start_date"])[:10], "value": float(summary["start_value"])},
            {"date": str(summary["end_date"])[:10], "value": float(summary["end_value"])},
        ]
    enabled = [
        name for name, detail in (config.get("factors") or {}).items()
        if isinstance(detail, dict) and detail.get("enabled")
    ]
    run_id = f"legacy-{path.stem}"
    label = f"Legacy {run_type} · {'+'.join(enabled) or 'saved model'}"
    result = {
        "run_id": run_id,
        "created_at": created_at,
        "label": label,
        "metrics": metrics,
        "equity_curve": equity_curve,
        "benchmark_curve": [],
        "holdings": [],
        "factor_weights": [],
        "params": config,
        "legacy_summary_only": True,
        "legacy_payload": legacy,
    }
    return {
        "run_id": run_id,
        "created_at": created_at,
        "label": label,
        "kind": "backtest",
        "request": config,
        "result": result,
    }


def _tracker_history_records(path: Path) -> List[Dict[str, Any]]:
    """Expose durable live tracker observations in the unified result history."""
    try:
        history = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning(f"Skipping unreadable live tracker {path.name}: {exc}")
        return []
    if not isinstance(history, list):
        return []
    account = path.stem.removeprefix("tracker_")
    records: List[Dict[str, Any]] = []
    curve: List[Dict[str, Any]] = []
    for index, row in enumerate(history):
        if not isinstance(row, dict):
            continue
        date_value = str(row.get("date") or row.get("recorded_at") or "")[:10]
        equity = float(row.get("equity", 0) or 0)
        if not date_value or equity <= 0:
            continue
        curve.append({"date": date_value, "value": equity})
        created_at = str(row.get("recorded_at") or row.get("timestamp") or date_value)
        run_id = f"live-{account}-{row.get('record_id') or index}"
        factor_weights = row.get("factor_weights") or {}
        live_config = {
            "active_factors": row.get("factors") or list(factor_weights),
            "factor_weights": factor_weights,
        }
        snapshot = {
            "account_name": account,
            "captured_at": created_at,
            "status": {
                "account_name": account,
                "account": row.get("account_snapshot") or {"equity": equity},
                "positions": [],
                "live_config": live_config,
                "effective_factor_weights": factor_weights,
                "tracker_history": history[: index + 1],
            },
            "equity_curve": list(curve),
            "performance": {
                "final_pnl": None,
                "day_pnl": row.get("day_pnl"),
                "total_pnl_pct": row.get("total_pnl_pct"),
                "note": "Historical tracker snapshot; realized gain requires broker fill history.",
            },
            "orders": [],
            "activities": [],
            "local_audit": [],
            "tracker_record": row,
        }
        records.append({
            "run_id": run_id,
            "created_at": created_at,
            "label": f"{account} · {date_value} · tracked equity ${equity:,.2f}",
            "kind": "live",
            "result": snapshot,
        })
    return records


# ── HTML root ─────────────────────────────────────────────────────────────────
@app.get("/")
async def root():
    resp = FileResponse(str(_STATIC / "ancserAPX.html"))
    resp.headers["Cache-Control"] = "no-store, max-age=0"
    return resp


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
    account_details = []
    try:
        account_details = get_configured_account_details()
    except Exception:
        pass
    accounts = [a["name"] for a in account_details]
    return {
        "factors": ALL_FACTORS,
        "primary_factors": PRIMARY_FACTORS,
        "secondary_factors": SECONDARY_FACTORS,
        "factor_presets": FACTOR_PRESETS,
        "factor_weight_presets": FACTOR_WEIGHT_PRESETS,
        "preset_defaults": PRESET_DEFAULTS,
        "strategy_presets": STRATEGY_PRESETS,
        "models": list_models(),
        "universes": {
            "spy_qqq": SPY_QQQ_TICKERS,
        },
        "accounts": accounts if accounts else ["Main"],
        "account_details": account_details if account_details else [{
            "name": "Main",
            "key_env": "APCA_API_KEY_ID",
            "secret_env": "APCA_API_SECRET_KEY",
            "paper": True,
            "mode": "paper",
        }],
        "data_feed": os.getenv("APCA_DATA_FEED", "iex").upper(),
    }


# ── Data status ───────────────────────────────────────────────────────────────
@app.get("/data/status")
async def data_status(universe: str = "spy_qqq", symbols: Optional[str] = None):
    from backend.data.store import get_coverage_stats

    if symbols:
        resolved = [s.strip().upper() for s in symbols.split(",") if s.strip()]
    else:
        if universe not in UNIVERSE_PRESETS:
            return JSONResponse(status_code=400, content={
                "error": f"Unknown universe '{universe}'."
            })
        resolved = UNIVERSE_PRESETS[universe]
    if not resolved:
        return JSONResponse(status_code=400, content={"error": "Universe is empty."})
    return get_coverage_stats(resolved)


# ── Data fetch (triggers async bulk/incremental fetch) ───────────────────────
class FetchRequest(BaseModel):
    mode: str = "incremental"   # "bulk" | "incremental"
    universe: str = "spy_qqq"
    symbols: Optional[List[str]] = None
    start_date: Optional[str] = None
    account: str = "Main"


@app.post("/data/fetch")
async def trigger_fetch(req: FetchRequest):
    if req.symbols:
        symbols = [s.strip().upper() for s in req.symbols if s.strip()]
    else:
        if req.universe not in UNIVERSE_PRESETS:
            return JSONResponse(status_code=400, content={
                "error": f"Unknown universe '{req.universe}'."
            })
        symbols = UNIVERSE_PRESETS[req.universe]
    if not symbols:
        return JSONResponse(status_code=400, content={"error": "Universe is empty."})

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
    model_id: str = DEFAULT_MODEL_ID
    ui_preset_label: Optional[str] = None
    symbols: Optional[List[str]] = None
    universe: str = "spy_qqq"     # single default: SPY + QQQ constituents
    start_date: str = "2020-01-01"
    end_date: Optional[str] = None
    active_factors: List[str] = ["Momentum", "Reversion", "EMA200 Distance"]
    capital: float = 100_000.0
    leverage: float = 1.0
    use_mwu: bool = False
    top_n: int = 30
    holding_period_days: int = 5
    neutralize_sector: bool = False
    factor_weights: Optional[Dict[str, float]] = None
    winner_lock: Optional[Dict[str, float]] = None  # secondary winner-lock rules
    strategy_preset: Optional[str] = None   # e.g. "Claude #1" → sleeve/lock run
    # Risk management (daily check, off the weekly cadence)
    ema_kill_switch: bool = False           # liquidate <200EMA, re-enter >20EMA
    risk_management: Optional[Dict[str, Any]] = None
    # Execution-cost assumptions.  Alpaca's broker commission for US stocks is
    # normally zero; spread/slippage remains a separate economic cost.
    commission_bps: float = 0.0
    slippage_bps: float = 5.0
    regulatory_sell_bps: float = 0.0
    signal_delay_days: int = 0


@app.post("/backtest/run")
async def run_backtest(req: BacktestRequest):
    try:
        model = require_model(req.model_id)
    except ValueError as exc:
        return JSONResponse(status_code=400, content={"error": str(exc)})
    if bool(model.get("uses_factors", False)) and not req.active_factors:
        return JSONResponse(status_code=400, content={
            "error": f"Model '{req.model_id}' requires at least one factor."
        })
    for field_name in ("commission_bps", "slippage_bps", "regulatory_sell_bps"):
        if float(getattr(req, field_name)) < 0:
            return JSONResponse(status_code=400, content={
                "error": f"{field_name} must be non-negative."
            })
    if int(req.signal_delay_days) < 0:
        return JSONResponse(status_code=400, content={
            "error": "signal_delay_days must be non-negative."
        })
    if int(req.signal_delay_days) and (req.use_mwu or req.neutralize_sector):
        return JSONResponse(status_code=400, content={
            "error": "signal_delay_days is not implemented for the legacy MWU/neutralized path."
        })
    # Resolve symbols. Reject unknown UI values instead of silently running a
    # different universe than the user selected.
    if req.symbols:
        symbols = req.symbols
    else:
        if req.universe not in UNIVERSE_PRESETS:
            return JSONResponse(status_code=400, content={
                "error": f"Unknown universe '{req.universe}'."
            })
        symbols = UNIVERSE_PRESETS[req.universe]

    end_date = req.end_date or datetime.now().strftime("%Y-%m-%d")

    def _run():
        from backend.backtest.engine import (
            BacktestEngine,
            compute_benchmark_relative_metrics,
            compute_metrics,
            _compute_benchmark_curve,
        )

        def _notify(level, msg):
            ws_log_sync(level, msg)

        hold_days = max(1, int(req.holding_period_days or 5))

        _notify("info", f"Backtest starting: {len(symbols)} symbols, {req.start_date}→{end_date}")
        _notify("info", f"Factors: {', '.join(req.active_factors)}")
        _notify("info", f"Holding period: {hold_days} trading days")

        engine = BacktestEngine(initial_capital=req.capital)

        risk_cfg = req.risk_management or {}
        risk_on = req.ema_kill_switch or any([
            str(risk_cfg.get("regime_mode", "off")).lower() != "off",
            bool(risk_cfg.get("volatility_throttle", False)),
            bool(risk_cfg.get("liquidity_filter", False)),
            bool(risk_cfg.get("crowding_shock_guard", False)),
            bool(risk_cfg.get("sector_balance", False)),
        ])
        if risk_on and (req.use_mwu or req.neutralize_sector):
            return {
                "error": (
                    "MWU/legacy sector-neutralization combined with the daily risk "
                    "overlay is not implemented in the shared backtest path; no setting "
                    "was silently ignored. Disable one side of the combination."
                )
            }
        if risk_on:
            _notify("info", f"Risk mgmt: {risk_cfg or {'ema_kill_switch': req.ema_kill_switch}}")

        sp = STRATEGY_PRESETS.get(req.strategy_preset) if req.strategy_preset else None
        if req.strategy_preset and sp is None:
            return {"error": f"Unknown strategy_preset '{req.strategy_preset}'."}
        if sp:
            _notify("info", f"Strategy preset: {sp.get('label', req.strategy_preset)} "
                            f"(leverage {req.leverage}x, "
                            f"{len(sp.get('sleeves', []))} sleeves)")
            res_df, w_df, h_df = engine.run_strategy(
                symbols=symbols,
                start_date=req.start_date,
                end_date=end_date,
                sleeves=sp["sleeves"],
                leverage=req.leverage,
                top_n=req.top_n,
                lock_rules=sp.get("winner_lock", {}),
                rebalance_days=hold_days,
                ema_kill_switch=req.ema_kill_switch,
                risk_management=risk_cfg,
                commission_bps=req.commission_bps,
                slippage_bps=req.slippage_bps,
                regulatory_sell_bps=req.regulatory_sell_bps,
                signal_delay_days=req.signal_delay_days,
            )
        elif risk_on or (not req.use_mwu and not req.neutralize_sector):
            # Live-parity path for normal custom factors: build one sleeve and
            # hold/drift its weights exactly like live execution. MWU and sector
            # neutralization stay on the legacy simulation path because they are
            # not implemented in the live sleeve constructor.
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
                rebalance_days=hold_days,
                ema_kill_switch=req.ema_kill_switch,
                risk_management=risk_cfg,
                commission_bps=req.commission_bps,
                slippage_bps=req.slippage_bps,
                regulatory_sell_bps=req.regulatory_sell_bps,
                signal_delay_days=req.signal_delay_days,
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
                rebalance_days=hold_days,
                commission_bps=req.commission_bps,
                slippage_bps=req.slippage_bps,
                regulatory_sell_bps=req.regulatory_sell_bps,
            )

        if res_df.empty:
            return {"error": "No backtest results — check data availability."}

        metrics = compute_metrics(res_df, req.capital, holding_period_days=hold_days)
        _notify("success", f"Backtest done. CAGR={metrics.get('cagr_pct',0):.1f}% Sharpe={metrics.get('sharpe',0):.2f}")

        # Equity curve
        res_df_reset = res_df.reset_index()
        equity_curve = [
            {"date": str(r["date"])[:10], "value": round(r["equity"], 2)}
            for _, r in res_df_reset.iterrows()
        ]

        # QQQ benchmark (buy-and-hold, same starting funding as the strategy)
        benchmark_curve = _compute_benchmark_curve(req.start_date, end_date, req.capital, symbol="QQQ")
        benchmark_metrics = compute_benchmark_relative_metrics(res_df, benchmark_curve)

        # Latest gross sector exposure makes the concentration guard auditable.
        from backend.alpha.neutralization import SECTOR_MAP
        latest_target_weights = getattr(engine, "last_target_weights", {}) or {}
        gross = sum(abs(float(weight)) for weight in latest_target_weights.values())
        sector_exposure: Dict[str, float] = {}
        if gross > 0:
            for symbol, weight in latest_target_weights.items():
                sector = SECTOR_MAP.get(symbol, "Unknown")
                sector_exposure[sector] = sector_exposure.get(sector, 0.0) + abs(float(weight)) / gross
            sector_exposure = {
                sector: round(exposure * 100, 2)
                for sector, exposure in sorted(
                    sector_exposure.items(), key=lambda item: item[1], reverse=True
                )
            }

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
            "benchmark_metrics": benchmark_metrics,
            "sector_exposure": sector_exposure,
            "equity_curve": equity_curve,
            "benchmark_curve": benchmark_curve,
            "benchmark": "QQQ",
            "spy_curve": benchmark_curve,  # backwards-compat alias
            "holdings": holdings,
            "factor_weights": weights_out,
            "params": {
                "model_id": req.model_id,
                "symbols_count": len(symbols),
                "start_date": req.start_date,
                "end_date": end_date,
                "factors": req.active_factors,
                # The UI may deliberately override a preset's default leverage.
                # Report the value actually passed to the engine, not the preset
                # default, so saved history and the chart remain auditable.
                "leverage": req.leverage,
                "holding_period_days": hold_days,
                "commission_bps": req.commission_bps,
                "slippage_bps": req.slippage_bps,
                "regulatory_sell_bps": req.regulatory_sell_bps,
                "signal_delay_days": req.signal_delay_days,
            },
        }

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, _run)
    if "error" in result:
        return JSONResponse(status_code=400, content=result)
    try:
        result = _persist_backtest_run(result, req)
    except Exception as exc:
        logger.warning(f"Could not persist backtest result: {exc}")
        result["persistence_warning"] = str(exc)
    return result


@app.get("/backtest/history")
async def backtest_history(limit: int = 100):
    """Return complete saved backtests, newest first, for chart restoration."""
    capped = max(1, min(int(limit), 500))
    runs: List[Dict[str, Any]] = []
    if _BACKTEST_RUNS_DIR.exists():
        for path in _BACKTEST_RUNS_DIR.glob("*.json"):
            try:
                runs.append(json.loads(path.read_text(encoding="utf-8")))
            except Exception as exc:
                logger.warning(f"Skipping unreadable backtest history {path.name}: {exc}")
    if _LEGACY_BACKTEST_DIR.exists():
        for path in _LEGACY_BACKTEST_DIR.glob("*.json"):
            record = _legacy_backtest_record(path)
            if record:
                runs.append(record)
    logs_dir = _ROOT / "logs"
    if logs_dir.exists():
        for path in logs_dir.glob("tracker_*.json"):
            runs.extend(_tracker_history_records(path))
    runs.sort(
        key=lambda run: str(run.get("created_at") or run.get("timestamp") or ""),
        reverse=True,
    )
    return {"runs": runs[:capped]}


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
            last_rebalance_snapshot: Dict[str, Any] = {}
            if os.path.exists(snap_path):
                try:
                    snap = json.load(open(snap_path))
                    last_rebalance = snap.get("rebalance_date")
                    last_rebalance_snapshot = snap
                except Exception:
                    pass

            tracker_path = _ROOT / "logs" / f"tracker_{account}.json"
            tracker_history: List[Dict[str, Any]] = []
            if tracker_path.exists():
                try:
                    tracker_history = json.loads(tracker_path.read_text(encoding="utf-8"))
                except Exception:
                    pass

            configured_factor_weights = _effective_factor_weights(live_config)
            effective_factor_weights = configured_factor_weights
            if live_config.get("use_mwu"):
                # MWU is calculated at decision time. Prefer the durable values
                # from the last executed target; static config weights are only
                # the starting state and must not be presented as current.
                executed_weights = last_rebalance_snapshot.get("factor_weights") or {}
                if not executed_weights:
                    for row in reversed(tracker_history):
                        if row.get("factor_weights"):
                            executed_weights = row["factor_weights"]
                            break
                if executed_weights:
                    effective_factor_weights = executed_weights

            return {
                "account": acct,
                "account_name": account,
                "account_mode": "paper" if adapter.paper else "live",
                "positions": positions,
                "market": clock,
                "live_config": live_config,
                "configured_factor_weights": configured_factor_weights,
                "effective_factor_weights": effective_factor_weights,
                "last_rebalance": last_rebalance,
                "last_rebalance_snapshot": last_rebalance_snapshot,
                "tracker_history": tracker_history,
                "trading_stopped": (_ROOT / "logs" / f"stop_{account}.flag").exists(),
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


@app.get("/live/orders")
async def live_orders(account: str = "Main", limit: int = 500):
    """Return broker order state plus the local submission audit when available."""
    def _run():
        from backend.data.alpaca_adapter import AlpacaAdapter
        capped = max(1, min(limit, 5000))
        latest_orders = AlpacaAdapter(account).get_orders(limit=capped)

        # The broker endpoint can be page/limit bounded. Merge it with the
        # scheduler's durable lifecycle snapshots so reopening the site does
        # not erase older pending/processed orders.
        persisted_path = _ROOT / "logs" / f"broker_orders_{account}.json"
        persisted_orders: List[Dict[str, Any]] = []
        if persisted_path.exists():
            try:
                payload = json.loads(persisted_path.read_text(encoding="utf-8"))
                persisted_orders = payload if isinstance(payload, list) else []
            except Exception as exc:
                logger.warning(f"Could not read persisted broker orders for {account}: {exc}")
        by_id: Dict[str, Dict[str, Any]] = {}
        anonymous: List[Dict[str, Any]] = []
        for row in persisted_orders + latest_orders:
            identity = str(row.get("id") or row.get("broker_order_id") or "")
            if identity:
                by_id[identity] = row
            else:
                anonymous.append(row)
        broker_orders = list(by_id.values()) + anonymous
        broker_orders.sort(
            key=lambda row: str(
                row.get("updated_at") or row.get("filled_at") or row.get("created_at") or ""
            ),
            reverse=True,
        )

        # Append-only forensic events include planned/submitted/failed orders,
        # data gates and broker reconciliation. Return the newest requested
        # records while keeping the JSONL file itself complete on disk.
        audit_path = _ROOT / "logs" / f"live_audit_{account}.jsonl"
        local_audit: List[Dict[str, Any]] = []
        if audit_path.exists():
            try:
                for line in audit_path.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        local_audit.append(json.loads(line))
            except Exception as exc:
                logger.warning(f"Could not read live audit for {account}: {exc}")
        return {
            "orders": broker_orders[:capped],
            "local_audit": local_audit[-capped:][::-1],
        }

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _run)


@app.get("/live/performance")
async def live_performance(account: str = "Main", fill_limit: int = 5000):
    """Return auditable realised and unrealised live performance components."""
    def _run():
        from backend.data.alpaca_adapter import AlpacaAdapter
        adapter = AlpacaAdapter(account)
        positions = adapter.get_positions()
        capped = max(1, min(fill_limit, 20000))
        latest_fills = adapter.get_activities(limit=capped)
        persisted_path = _ROOT / "logs" / f"broker_fills_{account}.json"
        persisted_fills: List[Dict[str, Any]] = []
        if persisted_path.exists():
            try:
                payload = json.loads(persisted_path.read_text(encoding="utf-8"))
                persisted_fills = payload if isinstance(payload, list) else []
            except Exception as exc:
                logger.warning(f"Could not read persisted fills for {account}: {exc}")
        by_id: Dict[str, Dict[str, Any]] = {}
        anonymous: List[Dict[str, Any]] = []
        for row in persisted_fills + latest_fills:
            identity = str(row.get("id") or "")
            if identity:
                by_id[identity] = row
            else:
                anonymous.append(row)
        fills = list(by_id.values()) + anonymous
        realised = fifo_realized_pnl(fills)
        unrealised = round(sum(float(p.get("unrealized_pl", 0.0) or 0.0) for p in positions), 2)
        times = sorted(str(row.get("time") or row.get("date") or "") for row in fills if row)
        return {
            **realised,
            "unrealized_pnl": unrealised,
            # The requested FINAL P&L is realised gain, not open-position mark-to-market.
            "final_pnl": realised["realized_pnl"],
            "fill_count": len(fills),
            "scope_start": times[0] if times else None,
            "scope_end": times[-1] if times else None,
            "fees_included": False,
            "note": (
                "Gross FIFO realised gain reconstructed from available Alpaca fills; "
                "regulatory fees, transfers, dividends and unmatched pre-history lots are separate."
            ),
        }

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _run)


# ── Live config save ──────────────────────────────────────────────────────────
class LiveConfigRequest(BaseModel):
    account: str = "Main"
    model_id: str = DEFAULT_MODEL_ID
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
    rebalance_days: int = 5
    rebalance_weekday: int = 4  # 0=Mon ... 4=Fri
    risk_management: Optional[Dict[str, Any]] = None


@app.post("/live/save")
async def live_save(req: LiveConfigRequest):
    try:
        model = require_model(req.model_id)
    except ValueError as exc:
        return JSONResponse(status_code=400, content={"error": str(exc)})
    if bool(model.get("uses_factors", False)) and not req.active_factors:
        return JSONResponse(status_code=400, content={
            "error": f"Model '{req.model_id}' requires at least one factor."
        })
    requested_risk = req.risk_management or {}
    risk_active = any([
        str(requested_risk.get("regime_mode", "off")).lower() != "off",
        bool(requested_risk.get("volatility_throttle", False)),
        bool(requested_risk.get("liquidity_filter", False)),
        bool(requested_risk.get("crowding_shock_guard", False)),
        bool(requested_risk.get("sector_balance", False)),
    ])
    if req.use_mwu and risk_active:
        return JSONResponse(status_code=400, content={
            "error": "MWU plus daily risk is not yet supported by shared backtest/live validation."
        })
    if req.symbols:
        symbols = req.symbols
    else:
        if req.universe not in UNIVERSE_PRESETS:
            return JSONResponse(status_code=400, content={
                "error": f"Unknown universe '{req.universe}'."
            })
        symbols = UNIVERSE_PRESETS[req.universe]
    account_mode = "paper" if get_account_paper(req.account) else "live"
    config = {
        "account": req.account,
        "account_mode": account_mode,
        "model_id": req.model_id,
        "active_factors": req.active_factors,
        "universe_id": req.universe,
        "universe": symbols,
        "leverage": req.leverage,
        "use_mwu": req.use_mwu,
        "use_vol_target": req.use_vol_target,
        "vol_target": req.vol_target_pct,
        "strategy_mode": req.strategy_mode,
        "top_n": req.top_n,
        "neutralize_sector": req.neutralize_sector,
        "risk_management": req.risk_management or {},
        "rebalance_frequency": req.rebalance_frequency,
        "rebalance_days": max(1, int(req.rebalance_days or 5)),
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
    model_id: str = DEFAULT_MODEL_ID
    strategy_preset: Optional[str] = None       # e.g. "Claude #1"
    # fallback fields (used when no strategy_preset is given)
    active_factors: Optional[List[str]] = None
    universe: str = "spy_qqq"
    symbols: Optional[List[str]] = None
    leverage: float = 1.0
    top_n: int = 20
    factor_weights: Optional[Dict[str, float]] = None
    use_mwu: bool = False
    neutralize_sector: bool = False
    winner_lock: Optional[Dict[str, float]] = None  # secondary winner-lock rules
    rebalance_days: int = 5
    # Risk management (applies to both preset and custom)
    ema_kill_switch: bool = False
    risk_management: Optional[Dict[str, Any]] = None


@app.post("/live/apply")
async def live_apply(req: ApplyLiveRequest):
    """Write the currently-selected backtest strategy into the live config so the
    next daily/weekly rebalance executes it. Logs exactly what was applied."""
    try:
        model = require_model(req.model_id)
    except ValueError as exc:
        return JSONResponse(status_code=400, content={"error": str(exc)})
    if bool(model.get("uses_factors", False)) and not req.strategy_preset and not req.active_factors:
        return JSONResponse(status_code=400, content={
            "error": f"Model '{req.model_id}' requires at least one factor."
        })
    requested_risk = req.risk_management or {}
    risk_active = req.ema_kill_switch or any([
        str(requested_risk.get("regime_mode", "off")).lower() != "off",
        bool(requested_risk.get("volatility_throttle", False)),
        bool(requested_risk.get("liquidity_filter", False)),
        bool(requested_risk.get("crowding_shock_guard", False)),
        bool(requested_risk.get("sector_balance", False)),
    ])
    if req.use_mwu and risk_active:
        return JSONResponse(status_code=400, content={
            "error": (
                "MWU plus the daily risk overlay is not yet supported by the shared "
                "backtest/live validation path; live config was not changed."
            )
        })
    sp = STRATEGY_PRESETS.get(req.strategy_preset) if req.strategy_preset else None
    if req.strategy_preset and sp is None:
        return JSONResponse(status_code=400, content={
            "error": f"Unknown strategy_preset '{req.strategy_preset}'."
        })
    account_mode = "paper" if get_account_paper(req.account) else "live"
    # Resolve the same named/custom universe used by backtest; live must not
    # silently substitute the full SPY+QQQ set for the user's selection.
    if req.symbols:
        symbols = req.symbols
    else:
        if req.universe not in UNIVERSE_PRESETS:
            return JSONResponse(status_code=400, content={
                "error": f"Unknown universe '{req.universe}'."
            })
        symbols = UNIVERSE_PRESETS[req.universe]
    rebalance_days = max(1, int(req.rebalance_days or (sp.get("rebalance_days", 5) if sp else 5)))
    rebalance_frequency = "daily" if rebalance_days <= 1 else ("weekly" if rebalance_days == 5 else "custom_days")

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
            "account": req.account,
            "account_mode": account_mode,
            "model_id": req.model_id,
            "strategy_preset": req.strategy_preset,
            "strategy_label": sp.get("label", req.strategy_preset),
            "active_factors": union_factors,
            "factor_weights": merged_w,
            "universe_id": req.universe,
            "universe": symbols,
            "leverage": req.leverage,
            "top_n": req.top_n,
            "sleeves": sp.get("sleeves", []),
            "winner_lock": sp.get("winner_lock", {}),
            "rebalance_frequency": rebalance_frequency,
            "rebalance_days": rebalance_days,
            "rebalance_weekday": 4,
            "use_mwu": False,
            "use_vol_target": False,
            "strategy_mode": "long_only",
            "ema_kill_switch": req.ema_kill_switch,
            "risk_management": req.risk_management or {},
            "saved_at": datetime.now().isoformat(),
        }
        desc = (f"{sp.get('label', req.strategy_preset)} | "
                f"leverage {config['leverage']}x | top{config['top_n']} | "
                f"hold {rebalance_days}d | "
                f"sleeves " + ", ".join(f"{s['name']} {int(s['alloc']*100)}%"
                                        + ("+lock" if s.get("winner_lock") else "")
                                        for s in sp.get("sleeves", []))
                + (" | 200EMA-kill" if req.ema_kill_switch else ""))
    else:
        config = {
            "account": req.account,
            "account_mode": account_mode,
            "model_id": req.model_id,
            "strategy_preset": None,
            "active_factors": req.active_factors or [],
            "factor_weights": req.factor_weights or {},
            "universe_id": req.universe,
            "universe": symbols,
            "leverage": req.leverage,
            "top_n": req.top_n,
            "use_mwu": req.use_mwu,
            "neutralize_sector": req.neutralize_sector,
            "winner_lock": req.winner_lock or {},
            "rebalance_frequency": rebalance_frequency,
            "rebalance_days": rebalance_days,
            "rebalance_weekday": 4,
            "use_vol_target": False,
            "strategy_mode": "long_only",
            "ema_kill_switch": req.ema_kill_switch,
            "risk_management": req.risk_management or {},
            "saved_at": datetime.now().isoformat(),
        }
        desc = (f"custom | leverage {req.leverage}x | top{req.top_n} | "
                f"hold {rebalance_days}d | "
                f"factors {', '.join(req.active_factors or [])}"
                + (" | MWU" if req.use_mwu else "")
                + (" | 200EMA-kill" if req.ema_kill_switch else ""))

    cfg_path = f"config/live_strategy_{req.account}.json" if req.account != "Main" else "config/live_strategy.json"
    os.makedirs("config", exist_ok=True)
    with open(cfg_path, "w") as f:
        json.dump(config, f, indent=2)

    # Applying a strategy is an explicit, confirmed resume action in the UI.
    # A mere page refresh or config read never clears the sticky stop flag.
    stop_path = _ROOT / "logs" / f"stop_{req.account}.flag"
    resumed = stop_path.exists()
    if resumed:
        stop_path.unlink()

    await ws_broadcast("success", f"✓ APPLIED LIVE [{req.account}/{account_mode}]: {desc}")
    await ws_broadcast("info", f"This strategy takes effect on the NEXT daily rebalance "
                               f"({config['rebalance_frequency']}, "
                               f"{len(symbols)} symbols). Config written to {cfg_path}.")
    return {
        "applied": True,
        "resumed": resumed,
        "path": cfg_path,
        "config": config,
        "description": desc,
    }


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
            # Manual and scheduled execution deliberately share one fail-closed
            # pipeline: sync -> freshness/coverage gate -> targets -> as-of gate
            # -> OMS. The website cannot bypass pre-trade data validation.
            from backend.execution.scheduler import execute_account_rebalance
            result = execute_account_rebalance(req.account, config, force=req.force)
            if "error" in result:
                ws_log_sync(
                    "error",
                    f"Rebalance blocked at {result.get('stage', 'unknown')}: {result['error']}",
                )
                return result
            ws_log_sync(
                "success",
                f"Rebalance {result.get('status')}: {len(result.get('orders', []))} "
                f"orders for [{req.account}] using {result.get('as_of_date')}",
            )
            return result
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


@app.post("/live/start")
async def live_start(account: str = "Main"):
    """Explicitly clear a sticky stop flag without placing an order."""
    stop_path = _ROOT / "logs" / f"stop_{account}.flag"
    was_stopped = stop_path.exists()
    if was_stopped:
        stop_path.unlink()
    await ws_broadcast("info", f"Live trading RESUMED for [{account}]; no order was placed.")
    return {"started": True, "was_stopped": was_stopped, "orders_placed": 0}


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
