"""Chronological tabular ML research pipeline.

Outer test predictions are produced exactly once per fold.  Hyperparameters and
fold-level ensemble weights are chosen only on each fold's preceding validation
period.  A final lockbox bundle may be fit after the selection OOS period, but
its settings are locked using selection-period evidence before lockbox labels
are inspected.
"""

from __future__ import annotations

from dataclasses import dataclass
import itertools
import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

from .models import (
    CalibratedRankEnsemble,
    DeterministicHistGradientBoosting,
    DeterministicRidge,
    cross_sectional_percentile_rank,
)
from .validation import PurgedFold, daily_rank_ic, fold_masks, prediction_diagnostics


DEFAULT_BASELINES = (
    "score_production_claude1",
    "score_production_claude3",
    "score_momentum_ensemble_v2",
    "score_momentum_lowvol_v2",
)


@dataclass(frozen=True)
class TabularMLResult:
    selection_predictions: pd.DataFrame
    lockbox_predictions: pd.DataFrame
    fold_records: Tuple[Dict[str, object], ...]
    locked_settings: Dict[str, object]


def _finite_training(frame: pd.DataFrame, features: Sequence[str]) -> pd.DataFrame:
    out = frame.loc[frame["label_rank"].notna()].copy()
    if out.empty:
        raise ValueError("Training split has no finite labels")
    values = out.loc[:, features].to_numpy(dtype=float)
    entirely_missing = np.isnan(values).all(axis=0)
    if entirely_missing.any():
        missing = [features[i] for i in np.flatnonzero(entirely_missing)]
        raise ValueError(f"Training split has entirely missing features: {missing}")
    return out


def _mean_ic(frame: pd.DataFrame, prediction_col: str) -> float:
    ic = daily_rank_ic(frame, prediction_col=prediction_col)
    return float(ic.mean()) if len(ic) else -np.inf


def _fit_ridge(alpha: float, train: pd.DataFrame, features: Sequence[str]) -> DeterministicRidge:
    return DeterministicRidge(alpha=float(alpha)).fit(
        train.loc[:, features].to_numpy(dtype=float),
        train["label_rank"].to_numpy(dtype=float),
        train["sample_weight"].to_numpy(dtype=float),
    )


def _fit_gbdt(params: Mapping[str, object], train: pd.DataFrame, features: Sequence[str], seed: int) -> DeterministicHistGradientBoosting:
    clean = dict(params)
    clean["random_state"] = int(seed)
    clean["early_stopping"] = False
    return DeterministicHistGradientBoosting(**clean).fit(
        train.loc[:, features].to_numpy(dtype=float),
        train["label_rank"].to_numpy(dtype=float),
        train["sample_weight"].to_numpy(dtype=float),
    )


def _calibrated(model, frame: pd.DataFrame, features: Sequence[str]) -> np.ndarray:
    raw = model.predict(frame.loc[:, features].to_numpy(dtype=float))
    return cross_sectional_percentile_rank(raw, frame["timestamp"], center=True)


def _choose_baseline(validation: pd.DataFrame, baseline_columns: Sequence[str]) -> Tuple[str, Dict[str, float]]:
    scores: Dict[str, float] = {}
    for name in baseline_columns:
        tmp = validation[["timestamp", "label_rank", name]].rename(columns={name: "prediction"})
        scores[name] = _mean_ic(tmp, "prediction")
    winner = max(scores, key=lambda name: (scores[name], name))
    return winner, scores


def _ensemble_weight_grid(names: Sequence[str], increment: float = 0.25, cap: float = 0.75) -> Iterable[Dict[str, float]]:
    units = int(round(1.0 / increment))
    for parts in itertools.product(range(units + 1), repeat=len(names)):
        if sum(parts) != units:
            continue
        weights = np.asarray(parts, dtype=float) / units
        if weights.max(initial=0.0) > cap + 1e-12:
            continue
        yield {name: float(weight) for name, weight in zip(names, weights)}


def _choose_ensemble(
    validation: pd.DataFrame,
    calibrated: Mapping[str, np.ndarray],
    *,
    shrinkage: float,
    cap: float,
) -> Tuple[CalibratedRankEnsemble, Dict[str, float], float]:
    best = None
    for raw_weights in _ensemble_weight_grid(tuple(calibrated), cap=cap):
        ensemble = CalibratedRankEnsemble(raw_weights, max_weight=cap, shrinkage=shrinkage)
        prediction = ensemble.combine_calibrated(calibrated)
        tmp = validation[["timestamp", "label_rank"]].copy()
        tmp["prediction"] = prediction
        score = _mean_ic(tmp, "prediction")
        actual = ensemble.weights_.as_dict()
        key = (score, -sum(abs(v - 1.0 / len(actual)) for v in actual.values()))
        if best is None or key > best[0]:
            best = (key, ensemble, actual, score)
    if best is None:
        raise RuntimeError("No feasible ensemble weights")
    return best[1], best[2], float(best[3])


def _prediction_shell(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "timestamp", "execution_timestamp", "symbol", "label_rank",
        "label_residual", "sample_weight",
    ] + [c for c in DEFAULT_BASELINES if c in frame.columns]
    return frame.loc[:, columns].copy()


def run_tabular_walk_forward(
    panel: pd.DataFrame,
    feature_columns: Sequence[str],
    folds: Sequence[PurgedFold],
    *,
    ridge_alphas: Sequence[float],
    gbdt_grid: Sequence[Mapping[str, object]],
    random_seed: int,
    ensemble_shrinkage: float = 0.50,
    ensemble_single_model_cap: float = 0.75,
    baseline_columns: Sequence[str] = DEFAULT_BASELINES,
) -> Tuple[pd.DataFrame, Tuple[Dict[str, object], ...]]:
    missing = [c for c in [*feature_columns, *baseline_columns, "label_rank", "sample_weight"] if c not in panel.columns]
    if missing:
        raise ValueError(f"Panel missing walk-forward columns: {missing}")
    if not folds:
        raise ValueError("At least one walk-forward fold is required")

    outputs: List[pd.DataFrame] = []
    records: List[Dict[str, object]] = []
    for fold_number, fold in enumerate(folds):
        masks = fold_masks(panel, fold)
        train = _finite_training(panel.loc[masks["train"]], feature_columns)
        validation = _finite_training(panel.loc[masks["validation"]], feature_columns)
        test = panel.loc[masks["test"] & panel["label_rank"].notna()].copy()
        refit = _finite_training(panel.loc[masks["refit"]], feature_columns)
        if test.empty:
            raise ValueError(f"Fold {fold.fold_id} has no test rows")

        baseline_name, baseline_scores = _choose_baseline(validation, baseline_columns)
        validation_baseline = cross_sectional_percentile_rank(
            validation[baseline_name].to_numpy(dtype=float), validation["timestamp"], center=True
        )

        ridge_trials: List[Dict[str, object]] = []
        best_ridge = None
        for alpha in ridge_alphas:
            model = _fit_ridge(float(alpha), train, feature_columns)
            pred = _calibrated(model, validation, feature_columns)
            tmp = validation[["timestamp", "label_rank"]].copy()
            tmp["prediction"] = pred
            score = _mean_ic(tmp, "prediction")
            ridge_trials.append({"alpha": float(alpha), "validation_rank_ic": score})
            key = (score, -float(alpha))
            if best_ridge is None or key > best_ridge[0]:
                best_ridge = (key, float(alpha), pred)
        assert best_ridge is not None

        gbdt_trials: List[Dict[str, object]] = []
        best_gbdt = None
        for trial_number, params in enumerate(gbdt_grid):
            model = _fit_gbdt(params, train, feature_columns, random_seed + fold_number * 100 + trial_number)
            pred = _calibrated(model, validation, feature_columns)
            tmp = validation[["timestamp", "label_rank"]].copy()
            tmp["prediction"] = pred
            score = _mean_ic(tmp, "prediction")
            record = {"params": dict(params), "validation_rank_ic": score}
            gbdt_trials.append(record)
            complexity = int(params.get("max_leaf_nodes", 999)) * int(params.get("max_iter", 999))
            key = (score, -complexity)
            if best_gbdt is None or key > best_gbdt[0]:
                best_gbdt = (key, dict(params), pred)
        assert best_gbdt is not None

        validation_scores = {
            "baseline": validation_baseline,
            "ridge": best_ridge[2],
            "gbdt": best_gbdt[2],
        }
        ensemble, ensemble_weights, ensemble_ic = _choose_ensemble(
            validation,
            validation_scores,
            shrinkage=ensemble_shrinkage,
            cap=ensemble_single_model_cap,
        )

        ridge_model = _fit_ridge(best_ridge[1], refit, feature_columns)
        gbdt_model = _fit_gbdt(best_gbdt[1], refit, feature_columns, random_seed + fold_number)
        test_baseline = cross_sectional_percentile_rank(
            test[baseline_name].to_numpy(dtype=float), test["timestamp"], center=True
        )
        test_ridge = _calibrated(ridge_model, test, feature_columns)
        test_gbdt = _calibrated(gbdt_model, test, feature_columns)
        test_ensemble = ensemble.combine_calibrated({
            "baseline": test_baseline,
            "ridge": test_ridge,
            "gbdt": test_gbdt,
        })

        output = _prediction_shell(test)
        output["score_baseline_selected"] = test_baseline
        output["score_ridge"] = test_ridge
        output["score_gbdt"] = test_gbdt
        output["score_ensemble"] = test_ensemble
        output["fold_id"] = fold.fold_id
        output["train_end"] = fold.train_end
        outputs.append(output)

        diagnostics = {
            name: prediction_diagnostics(
                output[["timestamp", "label_rank", "label_residual", name]],
                prediction_col=name,
            )
            for name in ["score_baseline_selected", "score_ridge", "score_gbdt", "score_ensemble"]
        }
        records.append({
            "fold": fold.as_dict(),
            "rows": {"train": len(train), "validation": len(validation), "refit": len(refit), "test": len(test)},
            "baseline_name": baseline_name,
            "baseline_validation_scores": baseline_scores,
            "ridge_trials": ridge_trials,
            "selected_ridge_alpha": best_ridge[1],
            "gbdt_trials": gbdt_trials,
            "selected_gbdt_params": best_gbdt[1],
            "ensemble_weights": ensemble_weights,
            "ensemble_validation_rank_ic": ensemble_ic,
            "test_diagnostics": diagnostics,
        })

    result = pd.concat(outputs, ignore_index=True).sort_values(["timestamp", "symbol"])
    if result.duplicated(["timestamp", "symbol"]).any():
        raise RuntimeError("Walk-forward folds produced duplicate OOS predictions")
    return result, tuple(records)


def _select_locked_settings(
    selection_predictions: pd.DataFrame,
    fold_records: Sequence[Mapping[str, object]],
    *,
    ensemble_shrinkage: float,
    ensemble_single_model_cap: float,
) -> Dict[str, object]:
    # Choose model hyperparameters only from the validation decisions already
    # recorded before their respective OOS tests.
    ridge_values = [float(r["selected_ridge_alpha"]) for r in fold_records]
    ridge_alpha = float(pd.Series(ridge_values).mode().sort_values().iloc[0])
    serialized = [json.dumps(r["selected_gbdt_params"], sort_keys=True) for r in fold_records]
    gbdt_serialized = pd.Series(serialized).mode().sort_values().iloc[0]
    gbdt_params = json.loads(gbdt_serialized)

    baseline_scores = {}
    for name in DEFAULT_BASELINES:
        if name in selection_predictions:
            tmp = selection_predictions[["timestamp", "label_rank", name]].rename(columns={name: "prediction"})
            baseline_scores[name] = _mean_ic(tmp, "prediction")
    baseline_name = max(baseline_scores, key=lambda n: (baseline_scores[n], n))

    calibrated = {
        "baseline": cross_sectional_percentile_rank(selection_predictions[baseline_name], selection_predictions["timestamp"], center=True),
        "ridge": selection_predictions["score_ridge"].to_numpy(dtype=float),
        "gbdt": selection_predictions["score_gbdt"].to_numpy(dtype=float),
    }
    ensemble, weights, ic = _choose_ensemble(
        selection_predictions,
        calibrated,
        shrinkage=ensemble_shrinkage,
        cap=ensemble_single_model_cap,
    )
    return {
        "ridge_alpha": ridge_alpha,
        "gbdt_params": gbdt_params,
        "baseline_name": baseline_name,
        "baseline_selection_scores": baseline_scores,
        "ensemble_weights": weights,
        "ensemble_selection_rank_ic": ic,
    }


def fit_locked_lockbox(
    panel: pd.DataFrame,
    feature_columns: Sequence[str],
    *,
    selection_predictions: pd.DataFrame,
    fold_records: Sequence[Mapping[str, object]],
    selection_end: str,
    lockbox_start: str,
    embargo_days: int,
    random_seed: int,
    ensemble_shrinkage: float,
    ensemble_single_model_cap: float,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    settings = _select_locked_settings(
        selection_predictions,
        fold_records,
        ensemble_shrinkage=ensemble_shrinkage,
        ensemble_single_model_cap=ensemble_single_model_cap,
    )
    dates = pd.DatetimeIndex(pd.to_datetime(panel["timestamp"]).drop_duplicates().sort_values())
    lockbox_start_ts = pd.Timestamp(lockbox_start)
    lock_pos = int(dates.searchsorted(lockbox_start_ts, side="left"))
    train_end_pos = lock_pos - int(embargo_days) - 1
    if train_end_pos < 0:
        raise ValueError("Lockbox leaves no training history after embargo")
    configured_end = pd.Timestamp(selection_end)
    train_end = min(dates[train_end_pos], configured_end)

    train = _finite_training(panel.loc[pd.to_datetime(panel["timestamp"]) <= train_end], feature_columns)
    lockbox = panel.loc[
        (pd.to_datetime(panel["timestamp"]) >= lockbox_start_ts)
        & panel["label_rank"].notna()
    ].copy()
    if lockbox.empty:
        raise ValueError("No fully-labelled rows exist in the lockbox period")

    ridge = _fit_ridge(float(settings["ridge_alpha"]), train, feature_columns)
    gbdt = _fit_gbdt(settings["gbdt_params"], train, feature_columns, random_seed)
    baseline_name = str(settings["baseline_name"])
    scores = {
        "baseline": cross_sectional_percentile_rank(lockbox[baseline_name], lockbox["timestamp"], center=True),
        "ridge": _calibrated(ridge, lockbox, feature_columns),
        "gbdt": _calibrated(gbdt, lockbox, feature_columns),
    }
    ensemble = CalibratedRankEnsemble(
        settings["ensemble_weights"],
        max_weight=ensemble_single_model_cap,
        shrinkage=0.0,  # settings are already constrained/shrunk
    )
    output = _prediction_shell(lockbox)
    output["score_baseline_selected"] = scores["baseline"]
    output["score_ridge"] = scores["ridge"]
    output["score_gbdt"] = scores["gbdt"]
    output["score_ensemble"] = ensemble.combine_calibrated(scores)
    output["fold_id"] = "LOCKBOX"
    output["train_end"] = train_end
    settings = dict(settings)
    settings.update({
        "lockbox_train_end": str(train_end.date()),
        "lockbox_start": str(pd.Timestamp(lockbox_start).date()),
        "lockbox_rows": int(len(output)),
    })
    return output.sort_values(["timestamp", "symbol"]), settings


def run_tabular_research(
    panel: pd.DataFrame,
    feature_columns: Sequence[str],
    folds: Sequence[PurgedFold],
    *,
    ridge_alphas: Sequence[float],
    gbdt_grid: Sequence[Mapping[str, object]],
    random_seed: int,
    selection_end: str,
    lockbox_start: str,
    embargo_days: int,
    ensemble_shrinkage: float = 0.50,
    ensemble_single_model_cap: float = 0.75,
) -> TabularMLResult:
    selection, records = run_tabular_walk_forward(
        panel,
        feature_columns,
        folds,
        ridge_alphas=ridge_alphas,
        gbdt_grid=gbdt_grid,
        random_seed=random_seed,
        ensemble_shrinkage=ensemble_shrinkage,
        ensemble_single_model_cap=ensemble_single_model_cap,
    )
    lockbox, settings = fit_locked_lockbox(
        panel,
        feature_columns,
        selection_predictions=selection,
        fold_records=records,
        selection_end=selection_end,
        lockbox_start=lockbox_start,
        embargo_days=embargo_days,
        random_seed=random_seed,
        ensemble_shrinkage=ensemble_shrinkage,
        ensemble_single_model_cap=ensemble_single_model_cap,
    )
    return TabularMLResult(selection, lockbox, records, settings)


def write_ml_artifacts(result: TabularMLResult, directory: Path | str) -> None:
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    result.selection_predictions.to_parquet(out / "selection_oos_predictions.parquet", index=False)
    result.lockbox_predictions.to_parquet(out / "lockbox_predictions.parquet", index=False)
    (out / "fold_records.json").write_text(json.dumps(result.fold_records, indent=2, default=str), encoding="utf-8")
    (out / "locked_settings.json").write_text(json.dumps(result.locked_settings, indent=2, default=str), encoding="utf-8")
