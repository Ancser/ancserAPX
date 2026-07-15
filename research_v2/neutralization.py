"""Pure, date-local neutralization tools for isolated Research v2 studies.

Neutralization is deliberately separated from portfolio constraints.  These
functions remove broad-sector structure from a cross-sectional *signal*; they
do not promise a sector-neutral long-only portfolio.  The latter must be
measured from realised holdings and, when desired, enforced with portfolio
caps.

Every transformation is fit independently inside one decision date.  No
future return, later date, production state, or broker module is consulted.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Literal, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

from .validation import newey_west_mean_stats


NeutralizationMethod = Literal[
    "none",
    "sector_residual",
    "sector_zscore",
    "within_sector_rank",
]
UnknownPolicy = Literal["error", "exclude", "passthrough_global"]


@dataclass(frozen=True)
class NeutralizationSpec:
    """Pre-declared rules for one date-wise cross-sectional transform."""

    method: NeutralizationMethod = "sector_residual"
    min_sector_names: int = 10
    unknown_policy: UnknownPolicy = "error"
    final_cross_section_rank: bool = True

    def __post_init__(self) -> None:
        if self.method not in {
            "none",
            "sector_residual",
            "sector_zscore",
            "within_sector_rank",
        }:
            raise ValueError(f"unsupported neutralization method: {self.method}")
        if self.min_sector_names < 2:
            raise ValueError("min_sector_names must be at least 2")
        if self.unknown_policy not in {"error", "exclude", "passthrough_global"}:
            raise ValueError(f"unsupported unknown policy: {self.unknown_policy}")


def _centered_rank(values: pd.Series, groups: Sequence[pd.Series]) -> pd.Series:
    """Average-tie percentile rank spanning [-0.5, 0.5] within groups."""

    keys = [pd.Series(group, index=values.index) for group in groups]
    grouped = values.groupby(keys, sort=False, observed=True, dropna=False)
    counts = grouped.transform("count")
    ranks = grouped.rank(method="average", na_option="keep")
    out = (ranks - 1.0) / (counts - 1.0) - 0.5
    singleton = values.notna() & counts.eq(1)
    out.loc[singleton] = 0.0
    return out.astype(float)


def _key_fingerprint(frame: pd.DataFrame, date_col: str, symbol_col: str) -> str:
    import hashlib

    keys = frame.loc[:, [date_col, symbol_col]].copy()
    keys[date_col] = pd.to_datetime(keys[date_col]).astype("int64")
    hashed = pd.util.hash_pandas_object(keys, index=False).to_numpy(np.uint64)
    return hashlib.sha256(hashed.tobytes()).hexdigest()


def attach_sector_snapshot(
    frame: pd.DataFrame,
    sector_snapshot: Mapping[str, str],
    *,
    symbol_col: str = "symbol",
    sector_col: str = "sector",
) -> pd.DataFrame:
    """Return a copy with immutable research-sector labels attached."""

    if symbol_col not in frame:
        raise ValueError(f"frame is missing {symbol_col!r}")
    out = frame.copy()
    symbols = out[symbol_col].astype(str)
    out[symbol_col] = symbols
    out[sector_col] = symbols.map({str(k): str(v) for k, v in sector_snapshot.items()})
    return out


def neutralize_cross_sections(
    frame: pd.DataFrame,
    value_columns: Sequence[str],
    sector_snapshot: Mapping[str, str],
    *,
    spec: NeutralizationSpec = NeutralizationSpec(),
    date_col: str = "timestamp",
    symbol_col: str = "symbol",
    sector_col: str = "sector",
    output_prefix: str = "neutral__",
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Transform values independently inside each decision-date cross-section.

    ``sector_residual`` is exactly the residual from an OLS regression on an
    intercept and sector dummies: it is the value minus its date/sector mean.
    ``sector_zscore`` additionally normalises date/sector dispersion, while
    ``within_sector_rank`` replaces distances with robust within-sector ranks.

    The input is never modified.  Output keys and row order are identical.
    """

    columns = tuple(dict.fromkeys(map(str, value_columns)))
    required = {date_col, symbol_col, *columns}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"neutralization frame missing columns: {sorted(missing)}")
    if not columns:
        raise ValueError("at least one value column is required")
    if frame.duplicated([date_col, symbol_col]).any():
        raise ValueError("neutralization keys must be unique")

    out = attach_sector_snapshot(
        frame, sector_snapshot, symbol_col=symbol_col, sector_col=sector_col
    )
    out[date_col] = pd.to_datetime(out[date_col])
    key_hash_before = _key_fingerprint(out, date_col, symbol_col)
    unknown = out[sector_col].isna() | out[sector_col].eq("") | out[sector_col].eq("Unknown")
    unknown_symbols = sorted(out.loc[unknown, symbol_col].astype(str).unique())
    if unknown_symbols and spec.unknown_policy == "error":
        raise ValueError(
            "sector snapshot has unknown symbols: " + ", ".join(unknown_symbols[:12])
        )

    known = ~unknown
    structural_sizes = (
        out.loc[known]
        .groupby([date_col, sector_col], observed=True, sort=False)[symbol_col]
        .size()
    )
    structural_small = structural_sizes.lt(spec.min_sector_names)
    if bool(structural_small.any()):
        examples = [
            {date_col: str(index[0]), sector_col: str(index[1]), "names": int(value)}
            for index, value in structural_sizes.loc[structural_small].head(8).items()
        ]
        raise ValueError(
            f"date/sector groups below min_sector_names={spec.min_sector_names}: {examples}"
        )
    if unknown_symbols and spec.unknown_policy == "passthrough_global" and not spec.final_cross_section_rank:
        raise ValueError(
            "passthrough_global requires final_cross_section_rank=True so unknown and known values share a rank scale"
        )

    per_column: Dict[str, object] = {}
    group_keys = [out[date_col], out[sector_col]]
    for column in columns:
        raw = pd.to_numeric(out[column], errors="coerce").astype(float)
        if np.isinf(raw.to_numpy()).any():
            raise ValueError(f"{column} contains infinite values")
        transformed = raw.copy()
        eligible = known & raw.notna()
        valid_counts = (
            raw.where(known)
            .groupby(group_keys, sort=False, observed=True, dropna=False)
            .count()
            .reindex(structural_sizes.index, fill_value=0)
        )
        invalid_groups = valid_counts.lt(spec.min_sector_names)
        if bool(invalid_groups.any()):
            examples = [
                {date_col: str(index[0]), sector_col: str(index[1]), "valid": int(value)}
                for index, value in valid_counts.loc[invalid_groups].head(8).items()
            ]
            raise ValueError(
                f"{column} has date/sector groups below min valid names={spec.min_sector_names}: {examples}"
            )
        raw_global_rank = _centered_rank(raw, [out[date_col]])

        if spec.method == "none":
            working = raw
        elif spec.method == "within_sector_rank":
            working = pd.Series(np.nan, index=out.index, dtype=float)
            working.loc[eligible] = _centered_rank(
                raw.loc[eligible],
                [out.loc[eligible, date_col], out.loc[eligible, sector_col]],
            )
        else:
            means = raw.where(eligible).groupby(
                group_keys, sort=False, observed=True, dropna=False
            ).transform("mean")
            residual = raw - means
            working = residual.where(eligible)
            if spec.method == "sector_zscore":
                variance = residual.pow(2).where(eligible).groupby(
                    group_keys, sort=False, observed=True, dropna=False
                ).transform("mean")
                scale = np.sqrt(variance)
                zero_scale = (
                    scale.where(eligible)
                    .groupby(group_keys, sort=False, observed=True, dropna=False)
                    .min()
                    .reindex(structural_sizes.index)
                )
                if bool((zero_scale.isna() | zero_scale.le(1e-12)).any()):
                    bad = zero_scale.loc[zero_scale.isna() | zero_scale.le(1e-12)].head(8)
                    examples = [
                        {date_col: str(index[0]), sector_col: str(index[1]), "scale": float(value) if pd.notna(value) else None}
                        for index, value in bad.items()
                    ]
                    raise ValueError(f"{column} has zero-variance date/sector groups: {examples}")
                working = residual / scale.where(scale > 1e-12)

        if unknown_symbols:
            if spec.unknown_policy == "exclude":
                working.loc[unknown] = np.nan
            elif spec.unknown_policy == "passthrough_global":
                # Do not mix raw units (for example RSI 0-100) with residual,
                # z-score, or rank units.  Unknown names are excluded from the
                # known-sector calibration and later receive their own raw
                # global percentile rank on the common [-0.5, 0.5] scale.
                if spec.method != "none":
                    working.loc[unknown] = np.nan

        pre_rank = working.copy()
        if spec.final_cross_section_rank:
            transformed = _centered_rank(working, [out[date_col]])
            if unknown_symbols and spec.unknown_policy == "passthrough_global" and spec.method != "none":
                transformed.loc[unknown] = raw_global_rank.loc[unknown]
        else:
            transformed = working
        output = f"{output_prefix}{column}"
        out[output] = transformed

        sector_means = (
            pre_rank.loc[known]
            .groupby(
                [out.loc[known, date_col], out.loc[known, sector_col]],
                sort=False,
                observed=True,
            )
            .mean()
        )
        per_column[column] = {
            "input_non_missing": int(raw.notna().sum()),
            "output_non_missing": int(transformed.notna().sum()),
            "minimum_valid_date_sector_names": int(valid_counts.min()),
            "pre_final_rank_max_abs_sector_mean": (
                float(sector_means.abs().max()) if len(sector_means) else None
            ),
        }

    key_hash_after = _key_fingerprint(out, date_col, symbol_col)
    if key_hash_after != key_hash_before or len(out) != len(frame):
        raise AssertionError("neutralization changed input keys or row count")

    known_group_sizes = (
        out.loc[known]
        .groupby([date_col, sector_col], observed=True, sort=False)[symbol_col]
        .size()
    )
    audit: Dict[str, object] = {
        "spec": asdict(spec),
        "rows": int(len(out)),
        "dates": int(out[date_col].nunique()),
        "symbols": int(out[symbol_col].nunique()),
        "sectors": int(out.loc[known, sector_col].nunique()),
        "unknown_rows": int(unknown.sum()),
        "unknown_symbols": unknown_symbols,
        "minimum_date_sector_names": (
            int(known_group_sizes.min()) if len(known_group_sizes) else None
        ),
        "maximum_date_sector_names": (
            int(known_group_sizes.max()) if len(known_group_sizes) else None
        ),
        "input_output_key_sha256": key_hash_before,
        "columns": per_column,
    }
    return out, audit


def build_claude1_factor_variants(
    frame: pd.DataFrame,
    sector_snapshot: Mapping[str, str],
    *,
    momentum_col: str = "factor_ts_mom",
    reversion_col: str = "factor_rsi",
    methods: Sequence[NeutralizationMethod] = (
        "none",
        "sector_residual",
        "sector_zscore",
        "within_sector_rank",
    ),
    min_sector_names: int = 10,
    unknown_policy: UnknownPolicy = "error",
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Build factorwise Claude #1 variants with fixed 70/30 orientation."""

    result = frame.copy()
    audits: Dict[str, object] = {}
    for method in methods:
        transformed, audit = neutralize_cross_sections(
            frame,
            [momentum_col, reversion_col],
            sector_snapshot,
            spec=NeutralizationSpec(
                method=method,
                min_sector_names=min_sector_names,
                unknown_policy=unknown_policy,
                final_cross_section_rank=True,
            ),
            output_prefix=f"{method}__",
        )
        # Higher momentum is desirable; lower RSI is desirable.
        score = (
            0.70 * transformed[f"{method}__{momentum_col}"]
            - 0.30 * transformed[f"{method}__{reversion_col}"]
        )
        result[f"score_claude1_factorwise__{method}"] = score.to_numpy(float)
        audits[method] = audit
    return result, audits


def sector_exposure_diagnostics(
    frame: pd.DataFrame,
    *,
    score_col: str,
    date_col: str = "timestamp",
    sector_col: str = "sector",
) -> Tuple[Dict[str, float], pd.DataFrame]:
    """Measure how much of a score's daily variation is explained by sector."""

    required = {date_col, sector_col, score_col}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"exposure frame missing columns: {sorted(missing)}")
    clean = frame.loc[:, [date_col, sector_col, score_col]].copy()
    clean[date_col] = pd.to_datetime(clean[date_col])
    clean[score_col] = pd.to_numeric(clean[score_col], errors="coerce")
    clean = clean.dropna()
    rows = []
    for date, day in clean.groupby(date_col, sort=True, observed=True):
        values = day[score_col].to_numpy(float)
        grand = float(values.mean())
        total_ss = float(np.square(values - grand).sum())
        grouped = day.groupby(sector_col, observed=True)[score_col]
        means = grouped.mean()
        counts = grouped.size().reindex(means.index)
        between_ss = float((counts * np.square(means - grand)).sum())
        r2 = between_ss / total_ss if total_ss > 1e-18 else 0.0
        rows.append(
            {
                "timestamp": pd.Timestamp(date),
                "sector_r2": r2,
                "mean_abs_sector_mean": float(means.abs().mean()),
                "max_abs_sector_mean": float(means.abs().max()),
                "sector_mean_std": float(means.std(ddof=0)),
            }
        )
    daily = pd.DataFrame(rows)
    summary = {
        "mean_sector_r2": float(daily["sector_r2"].mean()),
        "median_sector_r2": float(daily["sector_r2"].median()),
        "mean_abs_sector_mean": float(daily["mean_abs_sector_mean"].mean()),
        "mean_max_abs_sector_mean": float(daily["max_abs_sector_mean"].mean()),
        "mean_sector_mean_std": float(daily["sector_mean_std"].mean()),
    }
    return summary, daily


def sector_conditioned_prediction_diagnostics(
    frame: pd.DataFrame,
    *,
    score_col: str,
    label_col: str = "label_rank",
    date_col: str = "timestamp",
    sector_col: str = "sector",
    horizon: int = 5,
) -> Dict[str, float]:
    """Within-sector and between-sector rank-IC diagnostics."""

    required = {date_col, sector_col, score_col, label_col}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"conditioned diagnostics missing columns: {sorted(missing)}")
    work = frame.loc[:, list(required)].copy()
    work[date_col] = pd.to_datetime(work[date_col])
    work["_score_within"] = _centered_rank(
        pd.to_numeric(work[score_col], errors="coerce"),
        [work[date_col], work[sector_col]],
    )
    work["_label_within"] = _centered_rank(
        pd.to_numeric(work[label_col], errors="coerce"),
        [work[date_col], work[sector_col]],
    )
    within = (
        work.groupby(date_col, sort=True, observed=True)
        .apply(
            lambda day: day[["_score_within", "_label_within"]]
            .dropna()
            .corr(method="spearman")
            .iloc[0, 1],
            include_groups=False,
        )
        .dropna()
    )

    sector_means = (
        work.groupby([date_col, sector_col], sort=True, observed=True)[
            [score_col, label_col]
        ]
        .mean()
        .reset_index()
    )
    between = (
        sector_means.groupby(date_col, sort=True, observed=True)
        .apply(
            lambda day: day[[score_col, label_col]]
            .dropna()
            .corr(method="spearman")
            .iloc[0, 1],
            include_groups=False,
        )
        .dropna()
    )
    within_nw = newey_west_mean_stats(
        within.to_numpy(float), max_lag=max(0, horizon - 1)
    )
    between_nw = newey_west_mean_stats(
        between.to_numpy(float), max_lag=max(0, horizon - 1)
    )
    return {
        "mean_within_sector_rank_ic": float(within_nw["mean"]),
        "within_sector_rank_ic_nw_t": float(within_nw["nw_t"]),
        "within_sector_rank_ic_days": int(within_nw["n"]),
        "mean_between_sector_rank_ic": float(between_nw["mean"]),
        "between_sector_rank_ic_nw_t": float(between_nw["nw_t"]),
        "between_sector_rank_ic_days": int(between_nw["n"]),
    }


def top_n_sector_concentration(
    frame: pd.DataFrame,
    *,
    score_col: str,
    top_n: int,
    date_col: str = "timestamp",
    symbol_col: str = "symbol",
    sector_col: str = "sector",
) -> Tuple[Dict[str, float], pd.DataFrame]:
    """Concentration of the score's unconstrained top-N selection."""

    if top_n < 1:
        raise ValueError("top_n must be positive")
    rows = []
    for date, day in frame.groupby(date_col, sort=True, observed=True):
        ranked = day.dropna(subset=[score_col]).sort_values(
            [score_col, symbol_col], ascending=[False, True], kind="mergesort"
        )
        selected = ranked.head(top_n)
        if selected.empty:
            continue
        shares = selected[sector_col].value_counts(normalize=True)
        hhi = float(np.square(shares.to_numpy(float)).sum())
        rows.append(
            {
                "timestamp": pd.Timestamp(date),
                "selected": int(len(selected)),
                "sector_count": int(len(shares)),
                "max_sector_share": float(shares.max()),
                "sector_hhi": hhi,
                "effective_sectors": 1.0 / hhi if hhi > 0 else np.nan,
            }
        )
    daily = pd.DataFrame(rows)
    summary = {
        "mean_max_sector_share": float(daily["max_sector_share"].mean()),
        "p95_max_sector_share": float(daily["max_sector_share"].quantile(0.95)),
        "mean_sector_hhi": float(daily["sector_hhi"].mean()),
        "mean_effective_sectors": float(daily["effective_sectors"].mean()),
        "mean_sector_count": float(daily["sector_count"].mean()),
    }
    return summary, daily


__all__ = [
    "NeutralizationSpec",
    "attach_sector_snapshot",
    "build_claude1_factor_variants",
    "neutralize_cross_sections",
    "sector_conditioned_prediction_diagnostics",
    "sector_exposure_diagnostics",
    "top_n_sector_concentration",
]
