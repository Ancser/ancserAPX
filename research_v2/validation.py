"""Purged walk-forward split construction and prediction diagnostics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PurgedFold:
    fold_id: str
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    validation_start: pd.Timestamp
    validation_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    purge_days: int
    embargo_days: int

    def as_dict(self) -> Dict[str, object]:
        out = asdict(self)
        for key, value in list(out.items()):
            if isinstance(value, pd.Timestamp):
                out[key] = str(value.date())
        return out


def _dates(values: Iterable[object]) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(pd.to_datetime(list(values))).drop_duplicates().sort_values()
    if idx.hasnans:
        raise ValueError("Walk-forward dates contain NaT")
    return idx


def make_purged_walk_forward(
    dates: Iterable[object],
    *,
    train_days: int = 504,
    validation_days: int = 63,
    test_days: int = 63,
    purge_days: int = 5,
    embargo_days: int = 5,
    step_days: int = 63,
    label_horizon: int = 5,
    rolling_train: bool = True,
    selection_end: str | None = None,
) -> List[PurgedFold]:
    """Create chronological train/purge/validation/embargo/test folds.

    The gap before validation prevents training labels from crossing into the
    validation feature period.  The gap before test performs the same role for
    validation/refit labels.  No random split is used anywhere.
    """
    idx = _dates(dates)
    for name, value in {
        "train_days": train_days,
        "validation_days": validation_days,
        "test_days": test_days,
        "step_days": step_days,
    }.items():
        if int(value) < 1:
            raise ValueError(f"{name} must be positive")
    if purge_days < label_horizon or embargo_days < label_horizon:
        raise ValueError("purge_days and embargo_days must be >= label_horizon")

    cutoff = pd.Timestamp(selection_end) if selection_end else None
    folds: List[PurgedFold] = []
    train_end_pos = train_days - 1
    counter = 0
    while train_end_pos < len(idx):
        purge_end = train_end_pos + purge_days
        valid_start = purge_end + 1
        valid_end = valid_start + validation_days - 1
        embargo_end = valid_end + embargo_days
        test_start = embargo_end + 1
        test_end = test_start + test_days - 1
        if test_end >= len(idx):
            break
        if cutoff is not None and idx[test_end] > cutoff:
            break
        train_start = max(0, train_end_pos - train_days + 1) if rolling_train else 0
        fold = PurgedFold(
            fold_id=f"wf_{counter:02d}_{idx[test_start].date()}",
            train_start=idx[train_start],
            train_end=idx[train_end_pos],
            validation_start=idx[valid_start],
            validation_end=idx[valid_end],
            test_start=idx[test_start],
            test_end=idx[test_end],
            purge_days=int(purge_days),
            embargo_days=int(embargo_days),
        )
        validate_fold(fold, idx, label_horizon=label_horizon)
        folds.append(fold)
        counter += 1
        train_end_pos += step_days
    if not folds:
        need = train_days + purge_days + validation_days + embargo_days + test_days
        raise ValueError(f"Not enough dates for one walk-forward fold; need at least {need}, got {len(idx)}")
    return folds


def validate_fold(fold: PurgedFold, dates: Sequence[object], *, label_horizon: int) -> None:
    idx = _dates(dates)
    positions = {d: i for i, d in enumerate(idx)}
    keys = [
        fold.train_start, fold.train_end, fold.validation_start,
        fold.validation_end, fold.test_start, fold.test_end,
    ]
    if any(k not in positions for k in keys):
        raise ValueError(f"Fold {fold.fold_id} uses dates outside the supplied calendar")
    p = [positions[k] for k in keys]
    if not (p[0] <= p[1] < p[2] <= p[3] < p[4] <= p[5]):
        raise ValueError(f"Fold {fold.fold_id} intervals overlap or are out of order")
    if p[2] - p[1] - 1 < label_horizon:
        raise ValueError(f"Fold {fold.fold_id} training purge is shorter than the label horizon")
    if p[4] - p[3] - 1 < label_horizon:
        raise ValueError(f"Fold {fold.fold_id} test embargo is shorter than the label horizon")


def fold_masks(frame: pd.DataFrame, fold: PurgedFold, date_col: str = "timestamp") -> Dict[str, np.ndarray]:
    dates = pd.to_datetime(frame[date_col])
    return {
        "train": ((dates >= fold.train_start) & (dates <= fold.train_end)).to_numpy(),
        "validation": ((dates >= fold.validation_start) & (dates <= fold.validation_end)).to_numpy(),
        "test": ((dates >= fold.test_start) & (dates <= fold.test_end)).to_numpy(),
        # Refit uses all fully-observed data through validation end.  The
        # embargo before test remains untouched.
        "refit": ((dates >= fold.train_start) & (dates <= fold.validation_end)).to_numpy(),
    }


def cross_sectional_rank(
    values: pd.Series,
    dates: pd.Series,
    *,
    center: bool = True,
) -> pd.Series:
    groups = pd.to_datetime(dates)
    ranked = values.groupby(groups).rank(method="average")
    counts = values.notna().groupby(groups).transform("sum").astype(float)
    percentile = (ranked - 1.0) / (counts - 1.0).replace(0.0, np.nan)
    return percentile - 0.5 if center else percentile


def daily_rank_ic(
    frame: pd.DataFrame,
    *,
    prediction_col: str,
    label_col: str = "label_rank",
    date_col: str = "timestamp",
    minimum_names: int = 20,
) -> pd.Series:
    def one(group: pd.DataFrame) -> float:
        pair = group[[prediction_col, label_col]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(pair) < minimum_names or pair[prediction_col].nunique() < 2 or pair[label_col].nunique() < 2:
            return np.nan
        return float(pair[prediction_col].corr(pair[label_col], method="spearman"))

    return frame.groupby(pd.to_datetime(frame[date_col]), sort=True).apply(one, include_groups=False).dropna()


def newey_west_mean_stats(series: pd.Series, max_lag: int = 4) -> Dict[str, float]:
    x = pd.Series(series, dtype=float).replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
    n = len(x)
    if n < 3:
        return {"mean": float(np.mean(x)) if n else np.nan, "nw_se": np.nan, "nw_t": np.nan, "n": n}
    residual = x - x.mean()
    gamma0 = float(np.dot(residual, residual) / n)
    long_run = gamma0
    lag_cap = min(int(max_lag), n - 1)
    for lag in range(1, lag_cap + 1):
        gamma = float(np.dot(residual[lag:], residual[:-lag]) / n)
        weight = 1.0 - lag / (lag_cap + 1.0)
        long_run += 2.0 * weight * gamma
    variance_mean = max(long_run / n, 0.0)
    se = float(np.sqrt(variance_mean))
    if se < 1e-15:
        se = 0.0
    mean = float(x.mean())
    return {"mean": mean, "nw_se": se, "nw_t": mean / se if se > 0 else np.nan, "n": n}


def decile_spread(
    frame: pd.DataFrame,
    *,
    prediction_col: str,
    return_col: str = "label_residual",
    date_col: str = "timestamp",
) -> pd.Series:
    def one(group: pd.DataFrame) -> float:
        pair = group[[prediction_col, return_col]].dropna()
        if len(pair) < 30:
            return np.nan
        ranks = pair[prediction_col].rank(method="average", pct=True)
        top = pair.loc[ranks >= 0.9, return_col].mean()
        bottom = pair.loc[ranks <= 0.1, return_col].mean()
        return float(top - bottom)

    return frame.groupby(pd.to_datetime(frame[date_col]), sort=True).apply(one, include_groups=False).dropna()


def prediction_diagnostics(
    frame: pd.DataFrame,
    *,
    prediction_col: str,
    label_col: str = "label_rank",
    horizon: int = 5,
) -> Dict[str, float]:
    ic = daily_rank_ic(frame, prediction_col=prediction_col, label_col=label_col)
    nw = newey_west_mean_stats(ic, max_lag=max(0, horizon - 1))
    spread = decile_spread(frame, prediction_col=prediction_col)
    return {
        "mean_rank_ic": nw["mean"],
        "rank_ic_nw_t": nw["nw_t"],
        "rank_ic_days": int(nw["n"]),
        "rank_ic_std": float(ic.std(ddof=1)) if len(ic) > 1 else np.nan,
        "mean_decile_spread": float(spread.mean()) if len(spread) else np.nan,
    }
