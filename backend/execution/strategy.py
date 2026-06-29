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
from typing import Dict

from backend.data.alpaca_adapter import AlpacaAdapter
from backend.data import store
from backend.alpha.factors import compute_all_factors, FACTOR_META
from backend.alpha.mwu import MWUEngine
from backend.alpha.portfolio import combined_target_weights

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


class LiveStrategy:
    def __init__(self, account_name: str = "Main"):
        self.account_name = account_name
        self.alpaca = AlpacaAdapter(account_name)

    def calculate_targets(self, config: Dict) -> Dict:
        universe = config.get("universe", [])
        factors = config.get("active_factors", [])
        if not universe or not factors:
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

            col_map = {name: meta["col"] for name, meta in FACTOR_META.items()}
            descending_factors = {name for name, meta in FACTOR_META.items() if meta["descending"]}

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
                            factor_values[f] = latest_data[col].reindex(closes.columns).dropna()
                price = closes.iloc[-1].dropna()
                lock_rules = config.get("winner_lock", {}) or {}
                prior_state = _load_winner_lock_state(self.account_name)
                target_w, new_state = combined_target_weights(
                    sleeves=sleeves,
                    factor_values=factor_values,
                    price=price,
                    state=prior_state,
                    top_n=top_n,
                    lock_rules=lock_rules,
                    leverage=leverage_cap,
                )
                _save_winner_lock_state(self.account_name, new_state)
                return {
                    "allocations": target_w,
                    "vol_metrics": {},
                    "latest_prices": closes.iloc[-1].to_dict(),
                    "factor_scores": {},
                    "factor_weights": {},
                    "as_of_date": str(latest_date)[:10],
                }
            # ─────────────────────────────────────────────────────────────────

            scores = pd.Series(0.0, index=closes.columns)

            # MWU
            use_mwu = config.get("use_mwu", False)
            weight_per = 1.0 / len(factors)
            factor_weights = {f: weight_per for f in factors}

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

            if strategy_mode == "long_short":
                n_side = min(top_n, max(1, len(scores) // 2))
                top_stocks = scores.nlargest(n_side)
                bottom_stocks = scores.nsmallest(n_side)
                allocations = {s: (1.0 / n_side) * target_scalar for s in top_stocks.index}
                allocations.update({s: -(1.0 / n_side) * target_scalar for s in bottom_stocks.index})
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
