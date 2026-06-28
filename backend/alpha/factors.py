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
    "v1.5S Score":    {"col": "factor_v15s_score",      "descending": False},
    "Rank Acceleration":{"col": "factor_rank_accel",    "descending": False},
}

ALL_FACTORS = list(FACTOR_META.keys())

FACTOR_PRESETS = {
    "v1.5S 70/30 Top20": ["Momentum 12-1", "Pullback 5d"],
    "v1.5S RankAccel Top20": ["v1.5S Score", "Rank Acceleration"],
    "Balanced":       ["Momentum", "Reversion", "EMA200 Distance"],
    "Momentum-Heavy": ["Momentum", "Unicorn Edge", "EMA200 Distance"],
    "Defensive":      ["Reversion", "Volatility", "Drift-Reversion"],
    "Alpha":          ["Alpha 101", "Microstructure", "Skew"],
    "Full":           ALL_FACTORS,
}

# Fixed factor weighting per preset. When a preset appears here the backtest
# uses these static weights instead of equal/MWU weights. This reproduces the
# SeikiChan v1.5S "70/30" score = 0.70·z(12-1 mom) + 0.30·z(5d pullback).
FACTOR_WEIGHT_PRESETS = {
    "v1.5S 70/30 Top20": {"Momentum 12-1": 0.70, "Pullback 5d": 0.30},
    # score_rank_accel = 0.80·base_score + 0.20·z(rank_acceleration)
    "v1.5S RankAccel Top20": {"v1.5S Score": 0.80, "Rank Acceleration": 0.20},
}

# Recommended defaults that ship with each preset (applied by the frontend).
PRESET_DEFAULTS = {
    "v1.5S 70/30 Top20": {
        "top_n": 20,
        "universe": "spy_qqq",
        "factor_weights": {"Momentum 12-1": 0.70, "Pullback 5d": 0.30},
    },
    "v1.5S RankAccel Top20": {
        "top_n": 20,
        "universe": "spy_qqq",
        "factor_weights": {"v1.5S Score": 0.80, "Rank Acceleration": 0.20},
    },
}


# ── Full strategy presets (sleeves + leverage + winner-lock) ──────────────────
# Unlike FACTOR_PRESETS (which only pick a factor combo for a single portfolio),
# a STRATEGY preset describes a complete tradeable strategy: capital split across
# multiple sleeves, portfolio leverage, and an optional winner-lock overlay on a
# sleeve. The engine's run_strategy() consumes these.
#
# "Claude #1" reproduces the SeikiChan primary/secondary design that tested best
# in our comparison:
#   • Core sleeve  (70% capital): v1.5S 70/30 top20  — the proven base.
#   • Satellite    (30% capital): RankAccel top20 + winner-lock — let winners run.
#   • 1.5x portfolio leverage — the return/drawdown sweet spot (vs 1x and 2x).
STRATEGY_PRESETS = {
    "Claude #1": {
        "label": "Claude #1 — Sleeve 70/30 + RankAccel WinnerLock @1.5x",
        "leverage": 1.5,
        "top_n": 20,
        "universe": "spy_qqq",
        "rebalance_frequency": "weekly",
        "sleeves": [
            {
                "name": "Core",
                "alloc": 0.70,
                "factors": ["Momentum 12-1", "Pullback 5d"],
                "weights": {"Momentum 12-1": 0.70, "Pullback 5d": 0.30},
                "winner_lock": False,
            },
            {
                "name": "Satellite",
                "alloc": 0.30,
                "factors": ["v1.5S Score", "Rank Acceleration"],
                "weights": {"v1.5S Score": 0.80, "Rank Acceleration": 0.20},
                "winner_lock": True,
            },
        ],
        # Winner-lock rules (applied to sleeves with winner_lock=True):
        #   profit_lock — lock a name once it is up this much since entry
        #   max_weight  — cap each locked name at this portfolio weight
        #   lock_rank   — a name is only eligible to lock while ranked this high
        "winner_lock": {"profit_lock": 0.30, "max_weight": 0.15, "lock_rank": 10},
    },
}
