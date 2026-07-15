"""Deterministic tabular models and rank-safe ensembling for research_v2.

This module is deliberately independent from the production package.  It does
not read data, write artifacts, or import execution code.  Models emit raw
scores; :func:`cross_sectional_percentile_rank` is the only supported bridge
from raw scores to comparable, date-wise research signals.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ArrayLike = Union[np.ndarray, pd.Series, Sequence[float]]


def _as_2d_float(X: ArrayLike) -> np.ndarray:
    arr = np.asarray(X, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"X must be two-dimensional; got shape {arr.shape}")
    # Missing values are meaningful to the tree and imputed by the ridge
    # pipeline.  Infinite values are never meaningful, so normalize them to
    # missing in one deterministic place.
    if np.isinf(arr).any():
        arr = arr.copy()
        arr[np.isinf(arr)] = np.nan
    return arr


def _as_1d_target(y: ArrayLike, expected_rows: int) -> np.ndarray:
    arr = np.asarray(y, dtype=np.float64).reshape(-1)
    if arr.shape[0] != expected_rows:
        raise ValueError(f"y has {arr.shape[0]} rows; expected {expected_rows}")
    if not np.isfinite(arr).all():
        raise ValueError("y must contain only finite values")
    return arr


def _validate_sample_weight(
    sample_weight: Optional[ArrayLike], expected_rows: int
) -> Optional[np.ndarray]:
    if sample_weight is None:
        return None
    weights = np.asarray(sample_weight, dtype=np.float64).reshape(-1)
    if weights.shape[0] != expected_rows:
        raise ValueError(
            f"sample_weight has {weights.shape[0]} rows; expected {expected_rows}"
        )
    if not np.isfinite(weights).all() or np.any(weights < 0):
        raise ValueError("sample_weight must be finite and non-negative")
    if weights.sum() <= 0:
        raise ValueError("sample_weight must have positive total weight")
    return weights


def cross_sectional_percentile_rank(
    scores: ArrayLike,
    dates: Sequence[object],
    *,
    center: bool = True,
) -> np.ndarray:
    """Calibrate scores independently inside each date's cross-section.

    Ranks use ``(rank - 1) / (n - 1)`` so a non-tied cross-section spans the
    complete ``[0, 1]`` interval.  Average ranks are used for ties, missing
    scores remain missing, and a singleton cross-section receives 0.5.  With
    ``center=True`` the result is shifted into ``[-0.5, 0.5]`` and has mean
    zero within every non-empty date.
    """

    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    group_dates = np.asarray(dates)
    if group_dates.ndim != 1 or group_dates.shape[0] != values.shape[0]:
        raise ValueError("dates must be one-dimensional and match scores")
    if pd.isna(group_dates).any():
        raise ValueError("dates must not contain missing values")
    if np.isinf(values).any():
        raise ValueError("scores must not contain infinite values")

    frame = pd.DataFrame({"score": values, "date": group_dates})

    def _rank_one(group: pd.Series) -> pd.Series:
        valid_count = int(group.notna().sum())
        if valid_count == 0:
            return pd.Series(np.nan, index=group.index, dtype=float)
        if valid_count == 1:
            out = pd.Series(np.nan, index=group.index, dtype=float)
            out.loc[group.notna()] = 0.5
            return out
        ranks = group.rank(method="average", na_option="keep")
        return (ranks - 1.0) / float(valid_count - 1)

    calibrated = (
        frame.groupby("date", sort=False, observed=True)["score"]
        .transform(_rank_one)
        .to_numpy(dtype=np.float64)
    )
    if center:
        calibrated = calibrated - 0.5
    return calibrated


class DeterministicRidge:
    """Median-imputed, standardized Ridge with a deterministic solver."""

    def __init__(self, *, alpha: float = 1.0, fit_intercept: bool = True) -> None:
        if alpha < 0:
            raise ValueError("alpha must be non-negative")
        self.alpha = float(alpha)
        self.fit_intercept = bool(fit_intercept)
        self.estimator_: Optional[Pipeline] = None
        self.n_features_in_: Optional[int] = None

    def _new_estimator(self) -> Pipeline:
        return Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(strategy="median", keep_empty_features=True),
                ),
                ("scale", StandardScaler()),
                (
                    "ridge",
                    Ridge(
                        alpha=self.alpha,
                        fit_intercept=self.fit_intercept,
                        solver="lsqr",
                        tol=1e-8,
                    ),
                ),
            ]
        )

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        sample_weight: Optional[ArrayLike] = None,
    ) -> "DeterministicRidge":
        features = _as_2d_float(X)
        target = _as_1d_target(y, features.shape[0])
        weights = _validate_sample_weight(sample_weight, features.shape[0])
        estimator = self._new_estimator()
        fit_kwargs = {} if weights is None else {"ridge__sample_weight": weights}
        estimator.fit(features, target, **fit_kwargs)
        self.estimator_ = estimator
        self.n_features_in_ = features.shape[1]
        return self

    def predict(self, X: ArrayLike) -> np.ndarray:
        if self.estimator_ is None or self.n_features_in_ is None:
            raise RuntimeError("model has not been fitted")
        features = _as_2d_float(X)
        if features.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {features.shape[1]} features; expected {self.n_features_in_}"
            )
        return np.asarray(self.estimator_.predict(features), dtype=np.float64)

    def predict_calibrated(
        self, X: ArrayLike, dates: Sequence[object]
    ) -> np.ndarray:
        return cross_sectional_percentile_rank(self.predict(X), dates, center=True)


class DeterministicHistGradientBoosting:
    """Deterministic sklearn histogram GBDT without a random validation split.

    sklearn's automatic early stopping creates a random validation subset.
    That is invalid for financial time series, so this wrapper always sets
    ``early_stopping=False`` and ``validation_fraction=None``.  Iteration count
    must be selected by a caller-provided chronological validation fold.
    """

    def __init__(
        self,
        *,
        loss: str = "squared_error",
        learning_rate: float = 0.05,
        max_iter: int = 150,
        max_leaf_nodes: int = 15,
        max_depth: Optional[int] = None,
        min_samples_leaf: int = 100,
        l2_regularization: float = 1.0,
        max_features: float = 1.0,
        random_state: int = 42,
        early_stopping: bool = False,
    ) -> None:
        if early_stopping is not False:
            raise ValueError(
                "random early-stopping splits are forbidden; select max_iter "
                "with an external chronological validation fold"
            )
        if max_iter < 1 or max_leaf_nodes < 2 or min_samples_leaf < 1:
            raise ValueError("invalid tree-size parameter")
        self.params = {
            "loss": loss,
            "learning_rate": float(learning_rate),
            "max_iter": int(max_iter),
            "max_leaf_nodes": int(max_leaf_nodes),
            "max_depth": max_depth,
            "min_samples_leaf": int(min_samples_leaf),
            "l2_regularization": float(l2_regularization),
            "max_features": float(max_features),
            "early_stopping": False,
            "validation_fraction": None,
            "random_state": int(random_state),
        }
        self.estimator_: Optional[HistGradientBoostingRegressor] = None
        self.n_features_in_: Optional[int] = None

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        sample_weight: Optional[ArrayLike] = None,
    ) -> "DeterministicHistGradientBoosting":
        features = _as_2d_float(X)
        target = _as_1d_target(y, features.shape[0])
        weights = _validate_sample_weight(sample_weight, features.shape[0])
        estimator = HistGradientBoostingRegressor(**self.params)
        estimator.fit(features, target, sample_weight=weights)
        self.estimator_ = estimator
        self.n_features_in_ = features.shape[1]
        return self

    def predict(self, X: ArrayLike) -> np.ndarray:
        if self.estimator_ is None or self.n_features_in_ is None:
            raise RuntimeError("model has not been fitted")
        features = _as_2d_float(X)
        if features.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {features.shape[1]} features; expected {self.n_features_in_}"
            )
        return np.asarray(self.estimator_.predict(features), dtype=np.float64)

    def predict_calibrated(
        self, X: ArrayLike, dates: Sequence[object]
    ) -> np.ndarray:
        return cross_sectional_percentile_rank(self.predict(X), dates, center=True)


def _project_capped_simplex(weights: np.ndarray, cap: float) -> np.ndarray:
    """Euclidean projection onto ``sum(w)=1, 0<=w<=cap``."""

    count = weights.size
    if count == 0:
        raise ValueError("at least one model is required")
    if not (0 < cap <= 1):
        raise ValueError("max_weight must be in (0, 1]")
    if cap * count < 1.0 - 1e-12:
        raise ValueError(
            f"max_weight={cap} is infeasible for {count} models; "
            f"it must be at least {1.0 / count:.6f}"
        )

    # Find tau such that sum(clip(weights - tau, 0, cap)) == 1.
    lo = float(np.min(weights) - cap)
    hi = float(np.max(weights))
    for _ in range(100):
        tau = (lo + hi) / 2.0
        candidate = np.clip(weights - tau, 0.0, cap)
        if candidate.sum() > 1.0:
            lo = tau
        else:
            hi = tau
    projected = np.clip(weights - (lo + hi) / 2.0, 0.0, cap)
    total = projected.sum()
    if total <= 0:
        raise RuntimeError("failed to project ensemble weights")
    projected /= total
    return projected


def constrain_ensemble_weights(
    weights: ArrayLike,
    *,
    max_weight: float = 1.0,
    shrinkage: float = 0.0,
) -> np.ndarray:
    """Normalize, cap, and shrink non-negative weights toward equal weight."""

    raw = np.asarray(weights, dtype=np.float64).reshape(-1)
    if raw.size == 0 or not np.isfinite(raw).all() or np.any(raw < 0):
        raise ValueError("weights must be a non-empty finite non-negative vector")
    if not 0.0 <= shrinkage <= 1.0:
        raise ValueError("shrinkage must be in [0, 1]")
    normalized = (
        np.full(raw.size, 1.0 / raw.size, dtype=np.float64)
        if raw.sum() == 0
        else raw / raw.sum()
    )
    projected = _project_capped_simplex(normalized, float(max_weight))
    equal = np.full(raw.size, 1.0 / raw.size, dtype=np.float64)
    constrained = (1.0 - shrinkage) * projected + shrinkage * equal
    # Both endpoints are feasible, so their convex combination is feasible.
    constrained /= constrained.sum()
    return constrained


@dataclass(frozen=True)
class EnsembleWeights:
    names: tuple[str, ...]
    values: np.ndarray

    def as_dict(self) -> dict[str, float]:
        return {name: float(value) for name, value in zip(self.names, self.values)}


class CalibratedRankEnsemble:
    """Blend date-wise calibrated model ranks, never raw model outputs."""

    def __init__(
        self,
        weights: Mapping[str, float],
        *,
        max_weight: float = 1.0,
        shrinkage: float = 0.0,
    ) -> None:
        if not weights:
            raise ValueError("at least one named model weight is required")
        names = tuple(str(name) for name in weights)
        if len(set(names)) != len(names):
            raise ValueError("model names must be unique")
        values = constrain_ensemble_weights(
            [weights[name] for name in weights],
            max_weight=max_weight,
            shrinkage=shrinkage,
        )
        self.weights_ = EnsembleWeights(names=names, values=values)

    def combine_calibrated(
        self, calibrated_scores: Mapping[str, ArrayLike]
    ) -> np.ndarray:
        """Combine centered percentile ranks with row-wise missing renormalization."""

        missing = set(self.weights_.names) - set(calibrated_scores)
        extra = set(calibrated_scores) - set(self.weights_.names)
        if missing or extra:
            raise ValueError(
                f"score names must exactly match weights; missing={sorted(missing)}, "
                f"extra={sorted(extra)}"
            )
        columns = [
            np.asarray(calibrated_scores[name], dtype=np.float64).reshape(-1)
            for name in self.weights_.names
        ]
        lengths = {column.shape[0] for column in columns}
        if len(lengths) != 1:
            raise ValueError("all model score vectors must have the same length")
        matrix = np.column_stack(columns)
        finite_values = matrix[np.isfinite(matrix)]
        if finite_values.size and (
            finite_values.min() < -0.500000001
            or finite_values.max() > 0.500000001
        ):
            raise ValueError(
                "combine_calibrated accepts centered percentile ranks only; "
                "use predict_from_raw to calibrate raw scores first"
            )
        if np.isinf(matrix).any():
            raise ValueError("scores must not contain infinite values")

        available = np.isfinite(matrix)
        row_weights = available * self.weights_.values.reshape(1, -1)
        denominator = row_weights.sum(axis=1)
        numerator = np.nansum(matrix * self.weights_.values.reshape(1, -1), axis=1)
        out = np.full(matrix.shape[0], np.nan, dtype=np.float64)
        valid_rows = denominator > 0
        out[valid_rows] = numerator[valid_rows] / denominator[valid_rows]
        return out

    def predict_from_raw(
        self,
        raw_scores: Mapping[str, ArrayLike],
        dates: Sequence[object],
    ) -> np.ndarray:
        """Rank-calibrate every model by date, then blend calibrated scores."""

        calibrated = {
            name: cross_sectional_percentile_rank(raw_scores[name], dates, center=True)
            for name in self.weights_.names
        }
        return self.combine_calibrated(calibrated)

    # Short aliases for callers that prefer estimator-like naming.
    predict = predict_from_raw
    blend_calibrated = combine_calibrated


# Backward-friendly aliases while the research package interface settles.
RidgeWrapper = DeterministicRidge
HistGradientBoostingWrapper = DeterministicHistGradientBoosting


__all__ = [
    "CalibratedRankEnsemble",
    "DeterministicHistGradientBoosting",
    "DeterministicRidge",
    "EnsembleWeights",
    "HistGradientBoostingWrapper",
    "RidgeWrapper",
    "constrain_ensemble_weights",
    "cross_sectional_percentile_rank",
]
