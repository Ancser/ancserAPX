"""Point-in-time delta-PE features for research and forward shadowing.

An expanding P/E multiple can reflect either price re-rating or collapsing
earnings.  This module therefore exposes the naive multiple change, a robust
sector co-re-rating score, a sector-relative residual, and an EPS-quality
guard.  It performs no I/O and is not imported by the production daily path.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from research_v2.fundamental_value import validate_fundamental_snapshot


@dataclass(frozen=True)
class DeltaPEConfig:
    top_n: int = 22
    min_sector_names: int = 15
    eps_log_floor: float = float(np.log(0.90))
    min_sector_positive_breadth: float = 0.50
    min_sector_matched_retention: float = 0.70
    gross_target: float = 1.0

    def __post_init__(self) -> None:
        if self.top_n < 1:
            raise ValueError("top_n must be positive")
        if self.min_sector_names < 2:
            raise ValueError("min_sector_names must be at least two")
        if not np.isfinite(self.eps_log_floor):
            raise ValueError("eps_log_floor must be finite")
        if not 0.0 <= self.min_sector_positive_breadth <= 1.0:
            raise ValueError("min_sector_positive_breadth must be in [0, 1]")
        if not 0.0 <= self.min_sector_matched_retention <= 1.0:
            raise ValueError("min_sector_matched_retention must be in [0, 1]")
        if not np.isfinite(self.gross_target) or self.gross_target <= 0:
            raise ValueError("gross_target must be finite and positive")


def compute_delta_pe_features(
    previous: pd.DataFrame,
    current: pd.DataFrame,
    *,
    decision_at: object,
    config: DeltaPEConfig | None = None,
) -> pd.DataFrame:
    """Build delta-PE features from two available fundamental snapshots.

    Sector PE change is the median constituent log-PE change, not a weighted
    aggregate sector P/E.  A true aggregate requires historical point-in-time
    market caps and earnings, which are not assumed here.
    """

    config = config or DeltaPEConfig()
    if "trailing_eps" not in previous or "trailing_eps" not in current:
        raise ValueError("both snapshots require trailing_eps")
    old = validate_fundamental_snapshot(previous, decision_at=decision_at).rename(
        columns={
            "sector": "sector_previous",
            "pe_ttm": "pe_previous",
            "trailing_eps": "eps_previous",
            "available_at": "previous_available_at",
        }
    )
    new = validate_fundamental_snapshot(current, decision_at=decision_at).rename(
        columns={
            "sector": "sector",
            "pe_ttm": "pe_current",
            "trailing_eps": "eps_current",
            "available_at": "current_available_at",
        }
    )
    keep_old = [
        "symbol",
        "sector_previous",
        "pe_previous",
        "eps_previous",
        "previous_available_at",
    ]
    keep_new = [
        "symbol",
        "sector",
        "pe_current",
        "eps_current",
        "market_cap",
        "current_available_at",
    ]
    panel = old[keep_old].merge(new[keep_new], on="symbol", how="inner", validate="one_to_one")
    if panel.empty:
        raise ValueError("snapshots have no overlapping symbols")
    if (panel["previous_available_at"] >= panel["current_available_at"]).any():
        raise ValueError("previous snapshot must be available strictly before current snapshot")

    for column in ("pe_previous", "pe_current", "eps_previous", "eps_current"):
        panel[column] = pd.to_numeric(panel[column], errors="coerce")
    valid_pe = (
        np.isfinite(panel["pe_previous"])
        & np.isfinite(panel["pe_current"])
        & (panel["pe_previous"] > 0)
        & (panel["pe_current"] > 0)
    )
    valid_eps = (
        valid_pe
        & np.isfinite(panel["eps_previous"])
        & np.isfinite(panel["eps_current"])
        & (panel["eps_previous"] > 0)
        & (panel["eps_current"] > 0)
    )
    previous_positive_pe = np.isfinite(panel["pe_previous"]) & (panel["pe_previous"] > 0)
    current_positive_pe = np.isfinite(panel["pe_current"]) & (panel["pe_current"] > 0)
    panel["pe_state_transition"] = np.select(
        [
            previous_positive_pe & current_positive_pe,
            previous_positive_pe & ~current_positive_pe,
            ~previous_positive_pe & current_positive_pe,
        ],
        ["positive_to_positive", "positive_to_missing_or_nonpositive", "missing_or_nonpositive_to_positive"],
        default="missing_or_nonpositive_both",
    )
    panel["delta_log_pe"] = np.nan
    panel.loc[valid_pe, "delta_log_pe"] = np.log(
        panel.loc[valid_pe, "pe_current"] / panel.loc[valid_pe, "pe_previous"]
    )
    panel["delta_log_eps"] = np.nan
    panel.loc[valid_eps, "delta_log_eps"] = np.log(
        panel.loc[valid_eps, "eps_current"] / panel.loc[valid_eps, "eps_previous"]
    )
    panel["implied_delta_log_price"] = panel["delta_log_pe"] + panel["delta_log_eps"]
    # Diagnostic only until EPS is put on a corporate-action-consistent share
    # basis. When reported EPS falls, claw back its mechanical PE inflation.
    panel["reported_eps_quality_delta_log_pe"] = panel["delta_log_pe"] + np.minimum(
        panel["delta_log_eps"], 0.0
    )

    pe_local = panel.loc[valid_pe, ["sector", "delta_log_pe"]]
    sector_overlap_count = panel.groupby("sector")["symbol"].count()
    sector_count = pe_local.groupby("sector")["delta_log_pe"].count()
    sector_matched_retention = sector_count / sector_overlap_count
    sector_delta_pe = pe_local.groupby("sector")["delta_log_pe"].median()
    sector_positive_breadth = pe_local.groupby("sector")["delta_log_pe"].apply(
        lambda values: float((values > 0).mean())
    )
    sector_delta_eps = panel.loc[valid_eps].groupby("sector")["delta_log_eps"].median()
    qualified_sectors = sector_count[
        (sector_count >= config.min_sector_names)
        & (sector_matched_retention >= config.min_sector_matched_retention)
    ].index
    sector_rank = sector_delta_pe.loc[sector_delta_pe.index.intersection(qualified_sectors)].rank(
        pct=True, method="average"
    )
    panel["sector_valid_names"] = panel["sector"].map(sector_count).fillna(0).astype(int)
    panel["sector_overlap_names"] = panel["sector"].map(sector_overlap_count).fillna(0).astype(int)
    panel["sector_matched_pe_retention"] = panel["sector"].map(sector_matched_retention)
    panel["sector_delta_log_pe"] = panel["sector"].map(sector_delta_pe)
    panel["sector_positive_delta_pe_breadth"] = panel["sector"].map(
        sector_positive_breadth
    )
    panel["sector_delta_log_eps"] = panel["sector"].map(sector_delta_eps)
    panel["stock_delta_pe_rank"] = panel["delta_log_pe"].rank(pct=True, method="average")
    panel["sector_delta_pe_rank"] = panel["sector"].map(sector_rank)
    panel["relative_delta_log_pe"] = panel["delta_log_pe"] - panel["sector_delta_log_pe"]
    panel["relative_delta_pe_rank"] = panel["relative_delta_log_pe"].rank(
        pct=True, method="average"
    )
    panel["within_sector_relative_delta_pe_rank"] = panel.groupby("sector")[
        "relative_delta_log_pe"
    ].rank(pct=True, method="average")
    # The minimum implements a true AND: a strong sector cannot compensate for
    # weak stock-relative re-rating, or vice versa.
    panel["surge_and_score_unfiltered"] = np.minimum(
        panel["sector_delta_pe_rank"], panel["within_sector_relative_delta_pe_rank"]
    )
    panel["literal_surge_and_score_unfiltered"] = np.minimum(
        panel["sector_delta_pe_rank"], panel["stock_delta_pe_rank"]
    )
    panel["literal_surge_and_eligible"] = (
        valid_pe
        & (panel["sector_valid_names"] >= config.min_sector_names)
        & (panel["sector_matched_pe_retention"] >= config.min_sector_matched_retention)
        & (panel["sector_delta_log_pe"] > 0)
        & (
            panel["sector_positive_delta_pe_breadth"]
            >= config.min_sector_positive_breadth
        )
        & (panel["delta_log_pe"] > 0)
    )
    panel["literal_surge_and_score"] = panel[
        "literal_surge_and_score_unfiltered"
    ].where(panel["literal_surge_and_eligible"])
    panel["surge_and_eligible"] = (
        valid_pe
        & (panel["sector_valid_names"] >= config.min_sector_names)
        & (panel["sector_matched_pe_retention"] >= config.min_sector_matched_retention)
        & (panel["sector_delta_log_pe"] > 0)
        & (
            panel["sector_positive_delta_pe_breadth"]
            >= config.min_sector_positive_breadth
        )
        & (panel["relative_delta_log_pe"] > 0)
    )
    panel["surge_and_score"] = panel["surge_and_score_unfiltered"].where(
        panel["surge_and_eligible"]
    )
    panel["reported_eps_quality_eligible"] = (
        valid_eps
        & (panel["sector_valid_names"] >= config.min_sector_names)
        & (panel["sector_matched_pe_retention"] >= config.min_sector_matched_retention)
        & (panel["delta_log_eps"] >= config.eps_log_floor)
        & (panel["sector_delta_log_eps"] >= config.eps_log_floor)
        & (panel["implied_delta_log_price"] > 0)
    )
    panel["reported_eps_guarded_surge_eligible"] = (
        panel["surge_and_eligible"]
        & panel["reported_eps_quality_eligible"]
        & (panel["sector_delta_log_pe"] > 0)
    )
    panel["reported_eps_guarded_surge_score"] = panel["surge_and_score_unfiltered"].where(
        panel["reported_eps_guarded_surge_eligible"]
    )
    panel["reported_eps_guarded_literal_surge_eligible"] = (
        panel["literal_surge_and_eligible"] & panel["reported_eps_quality_eligible"]
    )
    panel["reported_eps_guarded_literal_surge_score"] = panel[
        "literal_surge_and_score_unfiltered"
    ].where(panel["reported_eps_guarded_literal_surge_eligible"])
    panel["reported_eps_guard_is_corporate_action_certified"] = False
    # Stable factor-like aliases for downstream research pipelines. None are
    # production-approved; the quality alias still uses uncertified EPS deltas.
    panel["factor_delta_log_pe_raw"] = panel["delta_log_pe"]
    panel["factor_delta_log_pe_sector"] = panel["sector_delta_log_pe"]
    panel["factor_delta_log_pe_relative"] = panel["relative_delta_log_pe"]
    panel["factor_delta_pe_surge_literal"] = panel["literal_surge_and_score"]
    panel["factor_delta_pe_surge_relative"] = panel["surge_and_score"]
    panel["factor_delta_pe_quality_reported_eps"] = panel[
        "reported_eps_guarded_literal_surge_score"
    ]
    panel["sector_changed"] = panel["sector_previous"] != panel["sector"]
    return panel.sort_values("symbol", kind="mergesort").reset_index(drop=True)


def build_delta_pe_portfolio(
    features: pd.DataFrame,
    *,
    score_column: str,
    config: DeltaPEConfig | None = None,
    equal_sector_gross: bool = False,
) -> pd.DataFrame:
    """Select the score Top-N without ever forcing sectors into selection."""

    config = config or DeltaPEConfig()
    if score_column not in features:
        raise ValueError(f"missing score column: {score_column}")
    candidates = features.loc[np.isfinite(pd.to_numeric(features[score_column], errors="coerce"))].copy()
    selected = candidates.sort_values(
        [score_column, "symbol"], ascending=[False, True], kind="mergesort"
    ).head(config.top_n)
    if selected.empty:
        selected["weight"] = pd.Series(dtype=float)
        return selected
    if equal_sector_gross:
        counts = selected.groupby("sector")["symbol"].transform("count")
        selected["weight"] = config.gross_target / selected["sector"].nunique() / counts
    else:
        selected["weight"] = config.gross_target / len(selected)
    selected["variant"] = score_column + (
        "__equal_sector_gross" if equal_sector_gross else "__equal_weight"
    )
    return selected.reset_index(drop=True)


__all__ = [
    "DeltaPEConfig",
    "build_delta_pe_portfolio",
    "compute_delta_pe_features",
]
