"""Pure point-in-time fundamental value strategy helpers.

This module deliberately performs no data download and is not imported by the
production daily path.  Callers must supply a snapshot whose ``available_at``
timestamp is no later than the portfolio decision time.  A current snapshot
can be used for a current cross-section or future shadow decisions, but never
backfilled into history.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {
    "symbol",
    "sector",
    "pe_ttm",
    "market_cap",
    "available_at",
}


@dataclass(frozen=True)
class PESectorStrategyConfig:
    """Configuration for PE-only selection with equal sector gross budgets."""

    top_n: int = 22
    gross_target: float = 1.0
    sector_market_cap_quantile: float = 0.50
    min_sector_names: int = 4
    require_positive_pe: bool = True
    apply_sector_size_filter: bool = True

    def __post_init__(self) -> None:
        if self.top_n < 1:
            raise ValueError("top_n must be positive")
        if not np.isfinite(self.gross_target) or self.gross_target <= 0:
            raise ValueError("gross_target must be finite and positive")
        if not 0.0 <= self.sector_market_cap_quantile <= 1.0:
            raise ValueError("sector_market_cap_quantile must be in [0, 1]")
        if self.min_sector_names < 2:
            raise ValueError("min_sector_names must be at least two")


def validate_fundamental_snapshot(
    snapshot: pd.DataFrame,
    *,
    decision_at: object | None = None,
) -> pd.DataFrame:
    """Validate and canonicalize one cross-sectional fundamental snapshot.

    Missing/non-positive PE and market-cap observations remain in the returned
    frame so diagnostics can report them; eligibility functions exclude them.
    Duplicate symbols and look-ahead timestamps fail closed.
    """

    missing = REQUIRED_COLUMNS - set(snapshot.columns)
    if missing:
        raise ValueError(f"fundamental snapshot missing columns: {sorted(missing)}")
    frame = snapshot.copy()
    if frame["symbol"].isna().any() or frame["sector"].isna().any():
        raise ValueError("symbol and sector must be non-null")
    frame["symbol"] = frame["symbol"].astype(str).str.strip()
    frame["sector"] = frame["sector"].astype(str).str.strip()
    frame["pe_ttm"] = pd.to_numeric(frame["pe_ttm"], errors="coerce")
    frame["market_cap"] = pd.to_numeric(frame["market_cap"], errors="coerce")
    frame["available_at"] = pd.to_datetime(frame["available_at"], utc=True, errors="coerce")
    invalid_symbol = frame["symbol"].str.lower().isin({"", "nan", "none", "<na>"})
    invalid_sector = frame["sector"].str.lower().isin({"", "nan", "none", "<na>"})
    if invalid_symbol.any() or invalid_sector.any():
        raise ValueError("symbol and sector must be non-empty")
    if frame["symbol"].duplicated().any():
        duplicates = sorted(frame.loc[frame["symbol"].duplicated(False), "symbol"].unique())
        raise ValueError(f"duplicate fundamental symbols: {duplicates[:10]}")
    if frame["available_at"].isna().any():
        raise ValueError("available_at must be a valid timestamp for every symbol")
    if decision_at is not None:
        decision = pd.Timestamp(decision_at)
        decision = decision.tz_localize("UTC") if decision.tzinfo is None else decision.tz_convert("UTC")
        future = frame["available_at"] > decision
        if future.any():
            offenders = sorted(frame.loc[future, "symbol"].unique())
            raise ValueError(f"look-ahead fundamentals are not available at decision time: {offenders[:10]}")
    return frame


def sector_relative_market_cap_filter(
    snapshot: pd.DataFrame,
    *,
    quantile: float = 0.50,
    min_sector_names: int = 4,
) -> pd.DataFrame:
    """Apply a hard market-cap threshold separately inside each sector.

    This is an eligibility rule, not regression neutralization.  A stock passes
    when its market cap is at or above its sector's contemporaneous quantile.
    Sectors with too few valid observations fail closed.
    """

    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be in [0, 1]")
    if min_sector_names < 2:
        raise ValueError("min_sector_names must be at least two")
    frame = validate_fundamental_snapshot(snapshot)
    valid_cap = np.isfinite(frame["market_cap"]) & (frame["market_cap"] > 0)
    counts = frame.loc[valid_cap].groupby("sector")["market_cap"].transform("count")
    thresholds = frame.loc[valid_cap].groupby("sector")["market_cap"].transform(
        lambda values: values.quantile(quantile)
    )
    frame["sector_market_cap_count"] = 0
    frame.loc[valid_cap, "sector_market_cap_count"] = counts.astype(int)
    frame["sector_market_cap_threshold"] = np.nan
    frame.loc[valid_cap, "sector_market_cap_threshold"] = thresholds
    frame["market_cap_to_sector_threshold"] = (
        frame["market_cap"] / frame["sector_market_cap_threshold"]
    )
    frame["sector_size_eligible"] = (
        valid_cap
        & (frame["sector_market_cap_count"] >= min_sector_names)
        & (frame["market_cap"] >= frame["sector_market_cap_threshold"])
    )
    return frame


def global_market_cap_filter(
    snapshot: pd.DataFrame,
    *,
    quantile: float = 0.50,
) -> pd.DataFrame:
    """Diagnostic global-cap filter used to measure sector selection bias."""

    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be in [0, 1]")
    frame = validate_fundamental_snapshot(snapshot)
    valid = np.isfinite(frame["market_cap"]) & (frame["market_cap"] > 0)
    threshold = float(frame.loc[valid, "market_cap"].quantile(quantile)) if valid.any() else np.nan
    frame["global_market_cap_threshold"] = threshold
    frame["global_size_eligible"] = valid & (frame["market_cap"] >= threshold)
    return frame


def build_pe_sector_balanced_portfolio(
    snapshot: pd.DataFrame,
    config: PESectorStrategyConfig | None = None,
    *,
    decision_at: object,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build a PE-only portfolio after sector-relative market-cap filtering.

    Lower positive trailing PE is better and *strictly* determines the global
    Top-N.  Sector treatment happens only after selection: every represented
    sector receives equal gross budget, then names inside the sector split that
    budget equally.  This is equal sector gross exposure, not covariance-based
    risk parity, and it never forces an otherwise unselected sector into Top-N.
    """

    config = config or PESectorStrategyConfig()
    point_in_time = validate_fundamental_snapshot(snapshot, decision_at=decision_at)
    if config.apply_sector_size_filter:
        screened = sector_relative_market_cap_filter(
            point_in_time,
            quantile=config.sector_market_cap_quantile,
            min_sector_names=config.min_sector_names,
        )
    else:
        screened = point_in_time
        screened["sector_market_cap_count"] = np.nan
        screened["sector_market_cap_threshold"] = np.nan
        screened["market_cap_to_sector_threshold"] = np.nan
        screened["sector_size_eligible"] = True
    valid_pe = np.isfinite(screened["pe_ttm"])
    if config.require_positive_pe:
        valid_pe &= screened["pe_ttm"] > 0
    screened["pe_eligible"] = valid_pe
    candidates = screened.loc[screened["sector_size_eligible"] & valid_pe].copy()
    if candidates.empty:
        return screened, pd.DataFrame(
            columns=["symbol", "sector", "pe_ttm", "market_cap", "pe_score", "weight"]
        )
    candidates["pe_score"] = candidates["pe_ttm"].rank(
        pct=True, ascending=False, method="average"
    )
    candidates["within_sector_pe_percentile"] = candidates.groupby("sector")["pe_ttm"].rank(
        pct=True, ascending=False, method="average"
    )
    portfolio = candidates.sort_values(
        ["pe_ttm", "symbol"], ascending=[True, True], kind="mergesort"
    ).head(config.top_n).copy()
    sector_counts = portfolio.groupby("sector")["symbol"].transform("count")
    represented = int(portfolio["sector"].nunique())
    portfolio["weight"] = config.gross_target / represented / sector_counts
    portfolio["sector_weight"] = config.gross_target / represented
    portfolio["rank"] = portfolio["pe_score"].rank(ascending=False, method="first").astype(int)
    portfolio = portfolio.sort_values(["sector", "pe_ttm", "symbol"], kind="mergesort")
    return screened, portfolio.reset_index(drop=True)


def joint_sector_size_residual(
    snapshot: pd.DataFrame,
    *,
    value_column: str = "earnings_yield",
) -> pd.Series:
    """Residualize a value score jointly on log market cap and sector dummies.

    This is the actual additive joint OLS alternative to the hard sector-median
    filter.  It changes the signal itself and therefore is not a 100% raw-PE
    strategy.  Returned residuals preserve the input index; invalid rows are
    NaN and no imputation is performed.
    """

    frame = validate_fundamental_snapshot(snapshot)
    if value_column == "earnings_yield" and value_column not in frame:
        frame[value_column] = np.where(frame["pe_ttm"] > 0, 1.0 / frame["pe_ttm"], np.nan)
    if value_column not in frame:
        raise ValueError(f"missing value column: {value_column}")
    value = pd.to_numeric(frame[value_column], errors="coerce")
    valid = (
        np.isfinite(value)
        & np.isfinite(frame["market_cap"])
        & (frame["market_cap"] > 0)
    )
    result = pd.Series(np.nan, index=frame.index, dtype=float, name=f"{value_column}_joint_residual")
    local = frame.loc[valid, ["sector", "market_cap"]].copy()
    y = value.loc[valid].to_numpy(float)
    if len(local) < 4 or local["sector"].nunique() < 2:
        return result
    log_cap = np.log(local["market_cap"].to_numpy(float))
    std = float(np.std(log_cap, ddof=0))
    if std <= 0 or not np.isfinite(std):
        return result
    z_cap = (log_cap - float(np.mean(log_cap))) / std
    dummies = pd.get_dummies(local["sector"], drop_first=True, dtype=float).to_numpy(float)
    design = np.column_stack([np.ones(len(local)), z_cap, dummies])
    coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
    result.loc[valid] = y - design @ coefficients
    return result


def size_filter_sector_diagnostics(snapshot: pd.DataFrame, *, quantile: float = 0.50) -> pd.DataFrame:
    """Compare per-sector retention for global versus sector-relative filters."""

    sector = sector_relative_market_cap_filter(snapshot, quantile=quantile)
    global_ = global_market_cap_filter(snapshot, quantile=quantile)
    joined = sector[["symbol", "sector", "sector_size_eligible"]].merge(
        global_[["symbol", "global_size_eligible"]], on="symbol", validate="one_to_one"
    )
    return (
        joined.groupby("sector", sort=True)
        .agg(
            universe_names=("symbol", "size"),
            sector_relative_pass=("sector_size_eligible", "sum"),
            global_pass=("global_size_eligible", "sum"),
        )
        .assign(
            sector_relative_retention=lambda x: x["sector_relative_pass"] / x["universe_names"],
            global_retention=lambda x: x["global_pass"] / x["universe_names"],
        )
        .reset_index()
    )


__all__ = [
    "PESectorStrategyConfig",
    "build_pe_sector_balanced_portfolio",
    "global_market_cap_filter",
    "joint_sector_size_residual",
    "sector_relative_market_cap_filter",
    "size_filter_sector_diagnostics",
    "validate_fundamental_snapshot",
]
