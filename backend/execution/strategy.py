"""
LiveStrategy — compute target portfolio weights from current factor scores.
Reads from local store for factor computation; uses Alpaca for latest prices.
Ported from ancserAPX with data source changed to local store.
"""

import json
import pandas as pd
import numpy as np
import polars as pl
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List

from backend.data.alpaca_adapter import AlpacaAdapter
from backend.data import store
from backend.alpha.factors import compute_all_factors, RUNTIME_FACTOR_META
from backend.alpha.mwu import MWUEngine
from backend.alpha.portfolio import combined_target_weights, sector_balanced_weights
from backend.alpha.models import DEFAULT_MODEL_ID, require_model

# Winner-lock state persisted across rebalances so the live account keeps the
# same entry prices / locked flags the backtest would have. PARITY-CRITICAL.
_STATE_DIR = Path(__file__).resolve().parents[2] / "logs"


def _winner_lock_state_path(account: str) -> Path:
    return _STATE_DIR / f"winner_lock_state_{account}.json"


def _load_winner_lock_state(account: str) -> Dict[str, Dict]:
    p = _winner_lock_state_path(account)
    try:
        if p.exists():
            return json.loads(p.read_text())
    except Exception:
        pass
    return {}


def _save_winner_lock_state(account: str, state: Dict[str, Dict]) -> None:
    try:
        _STATE_DIR.mkdir(parents=True, exist_ok=True)
        _winner_lock_state_path(account).write_text(json.dumps(state, indent=2))
    except Exception:
        pass


def resolve_factor_weights(factors: List[str], explicit: Dict[str, float] | None = None) -> Dict[str, float]:
    """Return the exact normalized weights used by the custom live factor model."""
    if not factors:
        return {}
    supplied = explicit or {}
    weights = {f: max(0.0, float(supplied.get(f, 0.0))) for f in factors}
    total = sum(weights.values())
    if total <= 0:
        return {f: 1.0 / len(factors) for f in factors}
    return {f: weight / total for f, weight in weights.items()}


class LiveStrategy:
    def __init__(self, account_name: str = "Main"):
        self.account_name = account_name
        self.alpaca = AlpacaAdapter(account_name)

    def calculate_targets(self, config: Dict) -> Dict:
        try:
            model = require_model(config.get("model_id", DEFAULT_MODEL_ID))
        except ValueError as exc:
            return {"error": str(exc)}
        universe = config.get("universe", [])
        factors = config.get("active_factors", [])
        if not universe or (bool(model.get("uses_factors", False)) and not factors):
            return {"error": "Universe or Factors empty"}

        use_vol_target = config.get("use_vol_target", False)
        vol_target = config.get("vol_target", 0.20)
        leverage_cap = config.get("leverage", 1.0)
        top_n = config.get("top_n", 30)
        vol_window = 20
        strategy_mode = config.get("strategy_mode", "long_only")

        # Need ~400 days of history for Momentum (252-day window + buffer)
        end_dt = datetime.now()
        start_dt = end_dt - timedelta(days=450)
        start_str = start_dt.strftime("%Y-%m-%d")
        end_str = end_dt.strftime("%Y-%m-%d")

        try:
            # Load from local store; fall back to Alpaca for missing
            hist_pl = store.load(universe, start_str, end_str).collect()
            if hist_pl.is_empty():
                # Fallback: fetch from Alpaca
                hist_pl = self.alpaca.fetch_history(universe, start_str, end_str).collect()
            if hist_pl.is_empty():
                return {"error": "No historical data available"}

            factor_df_pl = compute_all_factors(hist_pl.lazy()).collect()
            factor_df = factor_df_pl.to_pandas()
            factor_df["timestamp"] = pd.to_datetime(factor_df["timestamp"])

            hist = hist_pl.to_pandas()
            hist["timestamp"] = pd.to_datetime(hist["timestamp"])
            closes = hist.pivot(index="timestamp", columns="symbol", values="close")
            risk_cfg = config.get("risk_management", {}) or {}
            regime_mode = str(
                risk_cfg.get("regime_mode", "cash" if config.get("ema_kill_switch", False) else "off")
            ).lower()
            risk_off_leverage = float(risk_cfg.get("risk_off_leverage", min(leverage_cap, 1.0)))
            volatility_throttle = bool(risk_cfg.get("volatility_throttle", False))
            risk_vol_target = float(risk_cfg.get("vol_target_pct", 0.25))
            risk_vol_lookback = int(risk_cfg.get("vol_lookback", 20))
            liquidity_filter = bool(risk_cfg.get("liquidity_filter", False))
            min_price = float(risk_cfg.get("min_price", 5.0))
            min_avg_dollar_vol = float(risk_cfg.get("min_avg_dollar_vol", 20_000_000.0))
            crowding_shock_guard = bool(risk_cfg.get("crowding_shock_guard", False))
            max_avg_range_pct = float(risk_cfg.get("max_avg_range_pct", 0.12))
            sector_balance = bool(risk_cfg.get("sector_balance", False))

            eligible_symbols = set(closes.columns)
            if liquidity_filter or crowding_shock_guard:
                risk_hist = hist.copy()
                risk_hist["_dollar_vol"] = risk_hist["close"] * risk_hist["volume"]
                risk_hist["_range_pct"] = (risk_hist["high"] - risk_hist["low"]) / risk_hist["close"]
                latest_prices = closes.iloc[-1].dropna()
                if liquidity_filter:
                    dvol = risk_hist.pivot(index="timestamp", columns="symbol", values="_dollar_vol")
                    avg_dvol = dvol.rolling(20, min_periods=5).mean().iloc[-1]
                    eligible_symbols &= set(latest_prices[latest_prices >= min_price].index)
                    eligible_symbols &= set(avg_dvol[avg_dvol >= min_avg_dollar_vol].index)
                if crowding_shock_guard:
                    ranges = risk_hist.pivot(index="timestamp", columns="symbol", values="_range_pct")
                    avg_range = ranges.rolling(20, min_periods=5).mean().iloc[-1]
                    eligible_symbols &= set(avg_range[avg_range <= max_avg_range_pct].index)

            def _market_in_market() -> bool:
                if regime_mode not in {"cash", "throttle"}:
                    return True
                override = config.get("_risk_in_market_override")
                if override is not None:
                    return bool(override)
                try:
                    gauge_pl = store.load(["QQQ", "SPY"], start_str, end_str).collect()
                    if gauge_pl.is_empty():
                        raise RuntimeError("EMA/regime risk gauge data is empty")
                    g = gauge_pl.to_pandas()
                    g["timestamp"] = pd.to_datetime(g["timestamp"])
                    gp = g.pivot(index="timestamp", columns="symbol", values="close")
                    sym = "QQQ" if "QQQ" in gp.columns else ("SPY" if "SPY" in gp.columns else None)
                    if sym is None:
                        raise RuntimeError("EMA/regime risk gauge QQQ/SPY is missing")
                    s = gp[sym].dropna()
                    if len(s) < 220:
                        raise RuntimeError(f"EMA/regime risk gauge {sym} has only {len(s)} rows")
                    ema_slow = s.ewm(span=200, adjust=False).mean().iloc[-1]
                    return bool(s.iloc[-1] >= ema_slow)
                except Exception as exc:
                    raise RuntimeError(f"EMA/regime risk guard unavailable: {exc}") from exc

            def _risk_adjusted_leverage() -> float:
                lev = float(leverage_cap)
                in_market = _market_in_market()
                if regime_mode == "cash" and not in_market:
                    return 0.0
                if regime_mode == "throttle" and not in_market:
                    lev = min(lev, risk_off_leverage)
                if volatility_throttle:
                    rets = closes.pct_change().mean(axis=1).dropna()
                    if len(rets) >= max(5, risk_vol_lookback):
                        rv = float(rets.tail(risk_vol_lookback).std() * np.sqrt(252))
                        if rv > risk_vol_target and rv > 0:
                            lev = min(lev, float(leverage_cap) * risk_vol_target / rv)
                return max(0.0, lev)

            # Volatility targeting
            target_scalar = leverage_cap
            vol_metrics: Dict = {}
            if use_vol_target and closes.shape[0] >= vol_window + 1:
                daily_rets = closes.pct_change().dropna(how="all")
                if daily_rets.shape[0] >= vol_window:
                    port_ret = daily_rets.mean(axis=1)
                    recent = port_ret.iloc[-vol_window:].values
                    realized_vol = np.std(recent, ddof=1) * np.sqrt(252)
                    if realized_vol > 0.001:
                        raw_scalar = vol_target / realized_vol
                        target_scalar = min(leverage_cap, raw_scalar)
                    else:
                        raw_scalar = leverage_cap
                    vol_metrics = {
                        "current_vol": round(realized_vol, 4),
                        "target_vol": vol_target,
                        "final_scalar": round(target_scalar, 4),
                    }
            risk_leverage = _risk_adjusted_leverage()
            target_scalar = min(target_scalar, risk_leverage)
            if risk_leverage <= 0:
                return {
                    "allocations": {},
                    "vol_metrics": {"risk_leverage": 0.0, "risk_mode": regime_mode},
                    "latest_prices": closes.iloc[-1].to_dict(),
                    "factor_scores": {},
                    "factor_weights": {},
                    "as_of_date": str(closes.index.max())[:10],
                }

            col_map = {
                name: meta["col"] for name, meta in RUNTIME_FACTOR_META.items()
            }
            descending_factors = {
                name
                for name, meta in RUNTIME_FACTOR_META.items()
                if meta["descending"]
            }

            latest_date = factor_df["timestamp"].max()
            latest_data = factor_df[factor_df["timestamp"] == latest_date].set_index("symbol")

            # ─── PARITY PATH: sleeve / winner-lock strategy ──────────────────
            # When the config carries a `sleeves` list (e.g. the "Claude #1"
            # preset applied via /live/apply), build the SAME inputs the
            # backtest feeds to combined_target_weights() so the live targets
            # are identical by construction. See SKILL.md.
            sleeves = config.get("sleeves")
            if sleeves:
                # raw factor Series (indexed by symbol) for every factor any
                # sleeve references, keyed by factor NAME (portfolio.py expects
                # names, not column ids).
                factor_values: Dict[str, pd.Series] = {}
                for sl in sleeves:
                    for f in sl.get("factors", []):
                        if f in factor_values:
                            continue
                        col = col_map.get(f)
                        if col and col in latest_data.columns:
                            factor_values[f] = latest_data[col].reindex(closes.columns).dropna().loc[
                                lambda s: s.index.isin(eligible_symbols)
                            ]
                price = closes.iloc[-1].dropna()
                price = price.loc[price.index.isin(eligible_symbols)]
                lock_rules = config.get("winner_lock", {}) or {}
                prior_state = _load_winner_lock_state(self.account_name)
                target_w, new_state = combined_target_weights(
                    sleeves=sleeves,
                    factor_values=factor_values,
                    price=price,
                    state=prior_state,
                    top_n=top_n,
                    lock_rules=lock_rules,
                    leverage=target_scalar,
                    sector_balance=sector_balance,
                )
                _save_winner_lock_state(self.account_name, new_state)
                return {
                    "allocations": target_w,
                    "vol_metrics": {"risk_leverage": round(target_scalar, 4), "risk_mode": regime_mode},
                    "latest_prices": closes.iloc[-1].to_dict(),
                    "factor_scores": {},
                    "factor_weights": {},
                    "as_of_date": str(latest_date)[:10],
                }
            # ─────────────────────────────────────────────────────────────────

            scores = pd.Series(0.0, index=closes.columns)

            # MWU
            use_mwu = config.get("use_mwu", False)
            factor_weights = resolve_factor_weights(factors, config.get("factor_weights"))

            if use_mwu:
                mwu = MWUEngine(factors)
                fwd_df = factor_df_pl.with_columns([
                    (pl.col("close").shift(-1).over("symbol") / pl.col("close") - 1).alias("fwd_ret")
                ]).to_pandas()
                fwd_df["timestamp"] = pd.to_datetime(fwd_df["timestamp"])
                hist_dates = sorted(fwd_df["timestamp"].unique())
                fwd_pivot = fwd_df.pivot_table(index="timestamp", columns="symbol", values="fwd_ret")
                fpivots = {
                    f: fwd_df.pivot_table(index="timestamp", columns="symbol", values=col)
                    for f, col in col_map.items()
                    if f in factors and col in fwd_df.columns
                }
                for i, date in enumerate(hist_dates[:-1]):
                    if i >= 2:
                        prev = hist_dates[i - 1]
                        day_ics = {}
                        if prev in fwd_pivot.index:
                            fwd_row = fwd_pivot.loc[prev].dropna()
                            for f in factors:
                                if f in fpivots and prev in fpivots[f].index:
                                    frow = fpivots[f].loc[prev].dropna()
                                    common = frow.index.intersection(fwd_row.index)
                                    if len(common) > 5:
                                        corr = frow[common].corr(fwd_row[common], method="spearman")
                                        day_ics[f] = 0.0 if np.isnan(corr) else corr
                        mwu.update(date, day_ics)
                factor_weights = mwu.weights.copy()

            # Sector neutralization
            neutralize = config.get("neutralize_sector", False)
            if neutralize:
                from backend.alpha.neutralization import SECTOR_MAP
                sector_s = pd.Series({s: SECTOR_MAP.get(s, "Unknown") for s in closes.columns})

            for f in factors:
                col = col_map.get(f)
                if not col or col not in latest_data.columns:
                    continue
                vals = latest_data[col].reindex(closes.columns)
                if neutralize:
                    tmp = pd.DataFrame({"val": vals, "sector": sector_s})
                    vals = vals - tmp.groupby("sector")["val"].transform("mean")
                ascending = f not in descending_factors
                scores += vals.rank(pct=True, ascending=ascending).fillna(0.5) * factor_weights[f]

            scores = scores.loc[scores.index.isin(eligible_symbols)]
            if scores.empty:
                return {"error": "No eligible symbols after risk filters"}

            if strategy_mode == "long_short":
                n_side = min(top_n, max(1, len(scores) // 2))
                if sector_balance:
                    long_weights = sector_balanced_weights(scores, n_side)
                    short_weights = sector_balanced_weights(-scores, n_side)
                    allocations = {s: w * target_scalar for s, w in long_weights.items()}
                    allocations.update({s: -w * target_scalar for s, w in short_weights.items()})
                else:
                    top_stocks = scores.nlargest(n_side)
                    bottom_stocks = scores.nsmallest(n_side)
                    allocations = {s: (1.0 / n_side) * target_scalar for s in top_stocks.index}
                    allocations.update({s: -(1.0 / n_side) * target_scalar for s in bottom_stocks.index})
            else:
                if sector_balance:
                    allocations = {
                        s: w * target_scalar
                        for s, w in sector_balanced_weights(scores, top_n).items()
                    }
                else:
                    top_stocks = scores.nlargest(min(top_n, len(scores)))
                    target_weight = (1.0 / len(top_stocks)) * target_scalar
                    allocations = {s: target_weight for s in top_stocks.index}

            return {
                "allocations": allocations,
                "vol_metrics": vol_metrics,
                "latest_prices": closes.iloc[-1].to_dict(),
                "factor_scores": scores.to_dict(),
                "factor_weights": factor_weights,
                "as_of_date": str(latest_date)[:10],
            }

        except Exception as e:
            import traceback
            traceback.print_exc()
            return {"error": str(e)}
