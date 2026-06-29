"""
Alpha factor library — ported from ancserFX with no logic changes.
All time-series operations use .over("symbol") to prevent cross-symbol contamination.
"""

import polars as pl
import numpy as np


# ── 1. Basic Momentum & Reversion ────────────────────────────────────────────

def ts_momentum(window: int = 252) -> pl.Expr:
    return (pl.col("close") / pl.col("close").shift(window).over("symbol")) - 1


# ── v1.5S core factors (SeikiChan/alpaca-live-trading) ───────────────────────
# 12-1 momentum = 12-month return excluding the most recent month (~21 trading
# days). The classic cross-sectional momentum signal: strong medium-term trend
# minus the short-term mean-reversion window.

def momentum_12_1(long_window: int = 252, skip_window: int = 21) -> pl.Expr:
    long_ret = (pl.col("close") / pl.col("close").shift(long_window).over("symbol")) - 1
    skip_ret = (pl.col("close") / pl.col("close").shift(skip_window).over("symbol")) - 1
    return long_ret - skip_ret


# 5-day pullback (short-term reversal proxy). Negated 5-day return so that a
# larger recent drop scores higher — buy the dip inside an uptrend.

def pullback_5d(window: int = 5) -> pl.Expr:
    return -((pl.col("close") / pl.col("close").shift(window).over("symbol")) - 1)


def rsi(period: int = 14) -> pl.Expr:
    delta = pl.col("close").diff().over("symbol")
    gain = delta.clip(lower_bound=0)
    loss = -delta.clip(upper_bound=0)
    avg_gain = gain.rolling_mean(window_size=period).over("symbol")
    avg_loss = loss.rolling_mean(window_size=period).over("symbol")
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


# ── 2. Higher Moments ─────────────────────────────────────────────────────────

def realized_skew(window: int = 60) -> pl.Expr:
    return pl.col("returns").rolling_skew(window_size=window).over("symbol")


def idiosyncratic_vol(window: int = 20) -> pl.Expr:
    return pl.col("returns").rolling_std(window_size=window).over("symbol")


def max_daily_return(window: int = 20) -> pl.Expr:
    return pl.col("returns").rolling_max(window_size=window).over("symbol")


# ── 3. Microstructure ─────────────────────────────────────────────────────────

def amihud_illiquidity(window: int = 20) -> pl.Expr:
    term = pl.col("returns").abs() / (pl.col("close") * pl.col("volume"))
    return term.rolling_mean(window_size=window).over("symbol")


def spread_proxy() -> pl.Expr:
    return (pl.col("high") - pl.col("low")) / pl.col("close")


# ── 4. Alpha 101 ──────────────────────────────────────────────────────────────

def alpha_006(window: int = 10) -> pl.Expr:
    return -1 * pl.rolling_corr(pl.col("open"), pl.col("volume"), window_size=window).over("symbol")


def alpha_012() -> pl.Expr:
    return (
        pl.col("volume").diff().over("symbol").sign()
        * (-1 * pl.col("close").diff().over("symbol"))
    )


# ── 5. EMA200 Distance ────────────────────────────────────────────────────────

def ema200_distance() -> pl.Expr:
    ema = pl.col("close").ewm_mean(span=200, ignore_nulls=True).over("symbol")
    return (pl.col("close") - ema) / ema


# ── Feature pipeline ──────────────────────────────────────────────────────────

def compute_features(df: pl.LazyFrame) -> pl.LazyFrame:
    return df.with_columns([
        pl.col("close").pct_change().over("symbol").alias("returns"),
        ((pl.col("close") / pl.col("close").shift(20).over("symbol")) - 1).alias("return_1m"),
    ])


def compute_all_factors(df: pl.LazyFrame) -> pl.LazyFrame:
    df = compute_features(df)

    df = df.with_columns([
        ts_momentum(252).alias("factor_ts_mom"),
        momentum_12_1(252, 21).alias("factor_mom_12_1"),
        pullback_5d(5).alias("factor_pull_5d"),
        rsi(14).alias("factor_rsi"),
        realized_skew(60).alias("factor_skew"),
        idiosyncratic_vol(20).alias("factor_ivol"),
        max_daily_return(20).alias("factor_max"),
        amihud_illiquidity(20).alias("factor_amihud"),
        spread_proxy().alias("factor_spread"),
        alpha_006().alias("factor_alpha006"),
        alpha_012().alias("factor_alpha012"),
        ema200_distance().alias("factor_ema200_distance"),
    ])

    # ── Drift Regime ──────────────────────────────────────────────────────────
    df = df.with_columns([
        (pl.col("returns") > 0).cast(pl.Int32).alias("is_positive_day")
    ])
    df = df.with_columns([
        (pl.col("is_positive_day").rolling_sum(window_size=63).over("symbol") / 63)
        .alias("positive_day_ratio")
    ])
    df = df.with_columns([
        (pl.col("positive_day_ratio") > 0.60).alias("in_drift_regime")
    ])

    # Drift-filtered RSI
    df = df.with_columns([
        pl.when(~pl.col("in_drift_regime"))
        .then(pl.col("factor_rsi"))
        .otherwise(50.0)
        .alias("factor_rsi_filtered")
    ])

    # ── Unicorn Edge (Singha 2025) ────────────────────────────────────────────
    df = df.with_columns([
        (1.0 / pl.col("close")).alias("_ue_inv_price"),
        (-(pl.col("close") / pl.col("close").shift(10).over("symbol") - 1.0)).alias("_ue_ret10d_neg"),
    ])
    df = df.with_columns([
        (pl.col("_ue_inv_price").rank().over("timestamp") /
         pl.col("_ue_inv_price").count().over("timestamp")).alias("_ue_value_cs"),
        ((pl.col("_ue_ret10d_neg") - pl.col("_ue_ret10d_neg").mean().over("timestamp")) /
         (pl.col("_ue_ret10d_neg").std().over("timestamp") + 1e-8)).alias("_ue_reversal_cs"),
    ])
    df = df.with_columns([
        (0.7 * pl.col("_ue_value_cs") + 0.3 * pl.col("_ue_reversal_cs")).alias("_ue_base"),
    ])
    df = df.with_columns([
        (pl.col("_ue_base") * pl.col("in_drift_regime").cast(pl.Float64)).alias("factor_unicorn_edge"),
    ])

    # ── v1.5S composite score + Rank Acceleration (SeikiChan) ──────────────────
    # base_score = 0.70·z(mom_12_1) + 0.30·z(pull_5d)  (cross-sectional z-scores)
    df = df.with_columns([
        ((pl.col("factor_mom_12_1") - pl.col("factor_mom_12_1").mean().over("timestamp")) /
         (pl.col("factor_mom_12_1").std().over("timestamp") + 1e-8)).alias("_z_mom_12_1"),
        ((pl.col("factor_pull_5d") - pl.col("factor_pull_5d").mean().over("timestamp")) /
         (pl.col("factor_pull_5d").std().over("timestamp") + 1e-8)).alias("_z_pull_5d"),
    ])
    df = df.with_columns([
        (0.70 * pl.col("_z_mom_12_1") + 0.30 * pl.col("_z_pull_5d")).alias("factor_v15s_score"),
    ])
    # rank_pct: percentile rank of base score each day (0 = top). Higher score ⇒
    # smaller rank_pct. rank_acceleration = rank_pct(21d ago) − rank_pct(now);
    # positive ⇒ the name climbed toward the top over ~1 trading month.
    df = df.with_columns([
        (pl.col("factor_v15s_score").rank(method="average", descending=True).over("timestamp") /
         pl.col("factor_v15s_score").count().over("timestamp")).alias("_v15s_rank_pct"),
    ])
    df = df.with_columns([
        (pl.col("_v15s_rank_pct").shift(21).over("symbol") - pl.col("_v15s_rank_pct"))
        .alias("factor_rank_accel"),
    ])

    # ── Sector Rank (sector-rotation / volume-acceleration) ────────────────────
    # "When a sector's total market dollar-volume is increasing, rank the fastest
    # sector highest." Each stock inherits the 21-day growth rate of its GICS
    # sector's aggregate dollar volume (Σ close·volume across the sector). A
    # sector whose trading volume is accelerating (capital rotating IN — e.g. a
    # semiconductor/AI take-off) lifts every name in it. Higher ⇒ hotter sector.
    from backend.alpha.neutralization import SECTOR_MAP
    df = df.with_columns([
        pl.col("symbol").replace_strict(SECTOR_MAP, default="Unknown").alias("_sector"),
        (pl.col("close") * pl.col("volume")).alias("_dollar_vol"),
    ])
    df = df.with_columns([
        pl.col("_dollar_vol").sum().over(["timestamp", "_sector"]).alias("_sector_dvol"),
    ])
    df = df.with_columns([
        (pl.col("_sector_dvol") / pl.col("_sector_dvol").shift(21).over("symbol") - 1.0)
        .alias("factor_sector_rank"),
    ])

    return df


# ── Factor metadata ───────────────────────────────────────────────────────────

FACTOR_META = {
    "Momentum":       {"col": "factor_ts_mom",        "descending": False},
    "Momentum 12-1":  {"col": "factor_mom_12_1",       "descending": False},
    "Pullback 5d":    {"col": "factor_pull_5d",        "descending": False},
    "Reversion":      {"col": "factor_rsi",            "descending": True},
    "Skew":           {"col": "factor_skew",           "descending": False},
    "Microstructure": {"col": "factor_amihud",         "descending": True},
    "Alpha 101":      {"col": "factor_alpha006",       "descending": False},
    "Volatility":     {"col": "factor_ivol",           "descending": True},
    "Drift-Reversion":{"col": "factor_rsi_filtered",   "descending": True},
    "Unicorn Edge":   {"col": "factor_unicorn_edge",   "descending": False},
    "EMA200 Distance":{"col": "factor_ema200_distance","descending": False},
    "Rank Acceleration":{"col": "factor_rank_accel",    "descending": False},
    "Sector Rank":    {"col": "factor_sector_rank",     "descending": False},
}
# NOTE: factor_v15s_score is still COMPUTED in compute_all_factors (Rank
# Acceleration is derived from it) but is no longer exposed as a selectable
# factor — the "v1.5S Score" entry was removed per the strategy redesign.

ALL_FACTORS = list(FACTOR_META.keys())

# Secondary (二级) factors — overlay/booster signals layered on top of a primary
# factor model. Rank Acceleration (momentum-of-rank) and Sector Rank (sector
# volume rotation) describe *which names are heating up* rather than a standalone
# alpha, so they live under the "Secondary 二级" subhead in the UI.
SECONDARY_FACTORS = ["Rank Acceleration", "Sector Rank"]
PRIMARY_FACTORS = [f for f in ALL_FACTORS if f not in SECONDARY_FACTORS]

FACTOR_PRESETS = {
    "Baseline 70/30": ["Momentum", "Reversion"],
    "Sector Rotation": ["Momentum", "Sector Rank"],
    "v1.5S 70/30 Top20": ["Momentum 12-1", "Pullback 5d"],
    "Balanced":       ["Momentum", "Reversion", "EMA200 Distance"],
    "Momentum-Heavy": ["Momentum", "Unicorn Edge", "EMA200 Distance"],
    "Defensive":      ["Reversion", "Volatility", "Drift-Reversion"],
    "Alpha":          ["Alpha 101", "Microstructure", "Skew"],
    "Full":           ALL_FACTORS,
}

# Fixed factor weighting per preset. When a preset appears here the backtest
# uses these static weights instead of equal/MWU weights.
FACTOR_WEIGHT_PRESETS = {
    "Baseline 70/30": {"Momentum": 0.70, "Reversion": 0.30},
    "Sector Rotation": {"Momentum": 0.60, "Sector Rank": 0.40},
    "v1.5S 70/30 Top20": {"Momentum 12-1": 0.70, "Pullback 5d": 0.30},
}

# Recommended defaults that ship with each preset (applied by the frontend).
PRESET_DEFAULTS = {
    "Baseline 70/30": {
        "top_n": 20,
        "universe": "spy_qqq",
        "factor_weights": {"Momentum": 0.70, "Reversion": 0.30},
    },
    "Sector Rotation": {
        "top_n": 20,
        "universe": "spy_qqq",
        "factor_weights": {"Momentum": 0.60, "Sector Rank": 0.40},
    },
    "v1.5S 70/30 Top20": {
        "top_n": 20,
        "universe": "spy_qqq",
        "factor_weights": {"Momentum 12-1": 0.70, "Pullback 5d": 0.30},
    },
}


# ── Full strategy presets (sleeves + leverage + winner-lock) ──────────────────
# Unlike FACTOR_PRESETS (which only pick a factor combo for a single portfolio),
# a STRATEGY preset describes a complete tradeable strategy: capital split across
# multiple sleeves, portfolio leverage, and an optional winner-lock overlay on a
# sleeve. The engine's run_strategy() consumes these.
#
# "Claude #1" is the BASELINE: a single-sleeve classic factor blend, no leverage
# tricks, no winner-lock — the clean reference every experiment is measured
# against.
#   • One sleeve (100% capital): composite score = 0.70·rank(Momentum) +
#     0.30·rank(Reversion). Trend-following core with a mean-reversion ballast.
#   • top20, weekly rebalance, NO winner-lock.
#   • 1.5x portfolio leverage.
STRATEGY_PRESETS = {
    "Claude #1": {
        "label": "Claude #1 — Baseline 70/30 Momentum + Reversion @1.5x",
        "leverage": 1.5,
        "top_n": 20,
        "universe": "spy_qqq",
        "rebalance_frequency": "weekly",
        "sleeves": [
            {
                "name": "Core",
                "alloc": 1.0,
                "factors": ["Momentum", "Reversion"],
                "weights": {"Momentum": 0.70, "Reversion": 0.30},
                "winner_lock": False,
            },
        ],
        "winner_lock": {},
    },

    # "Claude #2" is the SECTOR-ROTATION experiment: blend trend with a sector
    # volume-acceleration overlay so the book tilts toward whichever GICS sector
    # is heating up (capital rotating in — e.g. a semiconductor/AI take-off).
    #   • One sleeve (100% capital): score = 0.60·rank(Momentum) +
    #     0.40·rank(Sector Rank). Sector Rank = 21-day growth of the stock's
    #     sector aggregate dollar volume.
    #   • top20, weekly rebalance, NO winner-lock, 1.5x leverage.
    "Claude #2": {
        "label": "Claude #2 — Sector Rotation (Momentum 60 + Sector Rank 40) @1.5x",
        "leverage": 1.5,
        "top_n": 20,
        "universe": "spy_qqq",
        "rebalance_frequency": "weekly",
        "sleeves": [
            {
                "name": "Core",
                "alloc": 1.0,
                "factors": ["Momentum", "Sector Rank"],
                "weights": {"Momentum": 0.60, "Sector Rank": 0.40},
                "winner_lock": False,
            },
        ],
        "winner_lock": {},
    },

    # "Claude #3" is the winner of the overnight rolling-window robustness search
    # (scripts/preset_search.py). It was selected for the HIGHEST worst-case
    # rolling-window Sharpe across every market regime 2021-2026, under a hard
    # margin/liquidation (爆倉) constraint, then run at the leverage that keeps the
    # worst drawdown far below the liquidation line.
    #
    #   • Core sleeve (70%): 12-month time-series Momentum (factor_ts_mom).
    #     Momentum is regime-adaptive — it rotated INTO energy in the 2022 bear
    #     (DVN/APA/COP) and into AI/semis/crypto 2023-2026 (NVDA/APP/PLTR/MSTR).
    #   • Defensive sleeve (30%): low idiosyncratic-vol + oversold RSI. A
    #     mean-reverting, low-beta hedge that cushions the momentum core during
    #     leadership reversals (late-2021 growth top, 2022 rotation).
    #   • top15, weekly rebalance, NO winner-lock (search showed lock added
    #     nothing here).
    #   • 1.5x leverage — worst rolling-window MaxDD ≈ -38%, vs the 1.5x
    #     liquidation drawdown of -83% (Reg-T 25% maint margin): a >2x safety
    #     buffer. You'd need a ~-55% underlying crash (worse than 2008) to risk
    #     a margin call.
    #
    # Validated robustness (live-path simulator, weekly):
    #   full 2021-07..2026-06  CAGR ~73%  Sharpe ~1.59  MaxDD -38.9%
    #   per-regime Sharpe: 2021H2 1.17 | 2022 bear 0.52 (+14% gross, while QQQ
    #     -33%) | 2023 1.46 | 2024 2.43 | 2025 1.47 | 2026 3.00
    #   weakest window: roll_2021-07 Sharpe 0.33 (the late-2021 growth top — a
    #     momentum turning point, the strategy's one structural soft spot).
    "Claude #3": {
        "label": "Claude #3 — TS-Momentum 70 + Defensive 30 @1.5x (robust)",
        "leverage": 1.5,
        "top_n": 15,
        "universe": "spy_qqq",
        "rebalance_frequency": "weekly",
        "sleeves": [
            {
                "name": "Core",
                "alloc": 0.70,
                "factors": ["Momentum"],
                "weights": {"Momentum": 1.0},
                "winner_lock": False,
            },
            {
                "name": "Defensive",
                "alloc": 0.30,
                "factors": ["Volatility", "Reversion"],
                "weights": {"Volatility": 0.6, "Reversion": 0.4},
                "winner_lock": False,
            },
        ],
        "winner_lock": {"profit_lock": 0.30, "max_weight": 0.15, "lock_rank": 10},
    },
}
