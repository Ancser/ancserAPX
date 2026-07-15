"""Purged walk-forward OOS pipeline for compact sequence models.

Every sample is anchored to a close(t) row.  A sequence may use earlier rows
from the same symbol, but never a row after its endpoint.  Train, validation,
test, and lockbox endpoints are explicit whitelists passed to
``LazySequenceDataset``; test and lockbox datasets do not carry targets into
the model prediction path.

This module is offline research code.  It has no production imports, I/O, or
side effects beyond in-memory model training requested by the caller.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch

from .models import CalibratedRankEnsemble, cross_sectional_percentile_rank
from .sequence import (
    GRUSequenceRegressor,
    LazySequenceDataset,
    TransformerSequenceRegressor,
    count_trainable_parameters,
)
from .validation import PurgedFold, daily_rank_ic


@dataclass(frozen=True)
class SequencePipelineSettings:
    sequence_length: int = 60
    gru_hidden_dim: int = 32
    gru_layers: int = 1
    transformer_d_model: int = 32
    transformer_heads: int = 4
    transformer_layers: int = 2
    transformer_feedforward: int = 64
    dropout: float = 0.0
    epochs: int = 4
    batch_size: int = 256
    max_train_samples: Optional[int] = 120_000
    max_parameters: int = 500_000
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    patience: Optional[int] = None
    random_seed: int = 42
    device: Union[str, torch.device, None] = "auto"
    ensemble_shrinkage: float = 0.50
    ensemble_single_model_cap: float = 0.75
    ensemble_grid_increment: float = 0.25
    minimum_cross_section: int = 20

    def __post_init__(self) -> None:
        positive = {
            "sequence_length": self.sequence_length,
            "gru_hidden_dim": self.gru_hidden_dim,
            "gru_layers": self.gru_layers,
            "transformer_d_model": self.transformer_d_model,
            "transformer_heads": self.transformer_heads,
            "transformer_layers": self.transformer_layers,
            "transformer_feedforward": self.transformer_feedforward,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "max_parameters": self.max_parameters,
            "minimum_cross_section": self.minimum_cross_section,
        }
        if any(int(value) < 1 for value in positive.values()):
            raise ValueError(f"sequence settings must be positive: {positive}")
        if self.max_train_samples is not None and self.max_train_samples < 1:
            raise ValueError("max_train_samples must be positive or None")
        if self.transformer_d_model % self.transformer_heads != 0:
            raise ValueError("transformer_d_model must be divisible by heads")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if not 0.0 <= self.ensemble_shrinkage <= 1.0:
            raise ValueError("ensemble_shrinkage must be in [0, 1]")
        if not 0.5 <= self.ensemble_single_model_cap <= 1.0:
            raise ValueError("two-model ensemble cap must be in [0.5, 1]")
        if not 0.0 < self.ensemble_grid_increment <= 1.0:
            raise ValueError("ensemble_grid_increment must be in (0, 1]")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("invalid optimizer settings")


@dataclass(frozen=True)
class SequenceFoldDatasets:
    panel: pd.DataFrame
    train: LazySequenceDataset
    validation: LazySequenceDataset
    test: LazySequenceDataset


@dataclass(frozen=True)
class SequenceResearchResult:
    selection_predictions: pd.DataFrame
    lockbox_predictions: pd.DataFrame
    fold_records: Tuple[Dict[str, object], ...]
    locked_settings: Dict[str, object]


@dataclass(frozen=True)
class _PreparedPanel:
    frame: pd.DataFrame
    features: np.ndarray
    targets: np.ndarray
    groups: np.ndarray


def deterministic_cap_endpoint_indices(
    endpoint_indices: Sequence[int],
    max_samples: Optional[int],
    *,
    seed: int,
) -> np.ndarray:
    """Deterministically cap endpoints without inspecting features or labels."""

    indices = np.asarray(endpoint_indices)
    if indices.ndim != 1:
        raise ValueError("endpoint_indices must be a one-dimensional integer vector")
    if indices.size == 0:
        indices = np.empty(0, dtype=np.int64)
    elif not np.issubdtype(indices.dtype, np.integer):
        raise ValueError("endpoint_indices must be a one-dimensional integer vector")
    else:
        indices = indices.astype(np.int64, copy=False)
    if np.unique(indices).size != indices.size:
        raise ValueError("endpoint_indices must be unique")
    indices = np.sort(indices)
    if max_samples is None or indices.size <= int(max_samples):
        return indices.copy()
    if int(max_samples) < 1:
        raise ValueError("max_samples must be positive or None")
    rng = np.random.default_rng(int(seed))
    selected = rng.choice(indices, size=int(max_samples), replace=False)
    return np.sort(selected.astype(np.int64, copy=False))


def _prepare_panel(
    panel: pd.DataFrame,
    feature_columns: Sequence[str],
) -> _PreparedPanel:
    features = tuple(feature_columns)
    if not features or len(set(features)) != len(features):
        raise ValueError("feature_columns must be non-empty and unique")
    forbidden = [
        name
        for name in features
        if name.startswith("label_")
        or name in {"timestamp", "symbol", "execution_timestamp"}
    ]
    if forbidden:
        raise ValueError(f"sequence features contain future/reserved columns: {forbidden}")
    required = ["timestamp", "symbol", "label_rank", *features]
    missing = [name for name in required if name not in panel.columns]
    if missing:
        raise ValueError(f"panel is missing sequence columns: {missing}")

    frame = panel.copy(deep=True)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="raise")
    if frame["timestamp"].isna().any() or frame["symbol"].isna().any():
        raise ValueError("timestamp and symbol must not be missing")
    frame["_source_row"] = np.arange(len(frame), dtype=np.int64)
    frame = frame.sort_values(
        ["symbol", "timestamp"], kind="mergesort"
    ).reset_index(drop=True)
    if frame.duplicated(["timestamp", "symbol"]).any():
        raise ValueError("panel contains duplicate timestamp/symbol rows")
    frame["_endpoint_row"] = np.arange(len(frame), dtype=np.int64)

    feature_values = frame.loc[:, features].to_numpy(dtype=np.float32)
    if np.isnan(feature_values).all(axis=0).any():
        bad = [features[i] for i in np.flatnonzero(np.isnan(feature_values).all(axis=0))]
        raise ValueError(f"sequence features are entirely missing: {bad}")
    target_values = pd.to_numeric(frame["label_rank"], errors="coerce").to_numpy(
        dtype=np.float32
    )
    return _PreparedPanel(
        frame=frame,
        features=np.ascontiguousarray(feature_values),
        targets=np.ascontiguousarray(target_values),
        groups=frame["symbol"].to_numpy(copy=True),
    )


def _allowed_indices(
    prepared: _PreparedPanel,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> np.ndarray:
    dates = prepared.frame["timestamp"]
    eligible = (
        (dates >= pd.Timestamp(start))
        & (dates <= pd.Timestamp(end))
        & np.isfinite(prepared.targets)
    )
    return np.flatnonzero(eligible.to_numpy()).astype(np.int64)


def _assert_lookback_only(
    prepared: _PreparedPanel,
    dataset: LazySequenceDataset,
) -> None:
    endpoints = dataset.end_indices
    if endpoints.size == 0:
        return
    starts = endpoints - dataset.sequence_length + 1
    dates = prepared.frame["timestamp"].to_numpy()
    symbols = prepared.frame["symbol"].to_numpy()
    if np.any(dates[starts] > dates[endpoints]):
        raise RuntimeError("sequence contains a timestamp after its endpoint")
    if np.any(symbols[starts] != symbols[endpoints]):
        raise RuntimeError("sequence crosses a symbol boundary")


def _datasets_from_fold(
    prepared: _PreparedPanel,
    fold: PurgedFold,
    *,
    sequence_length: int,
    max_train_samples: Optional[int],
    seed: int,
) -> SequenceFoldDatasets:
    natural_endpoints = LazySequenceDataset(
        prepared.features,
        targets=None,
        groups=prepared.groups,
        sequence_length=sequence_length,
    ).end_indices
    train_candidates = np.intersect1d(
        _allowed_indices(prepared, fold.train_start, fold.train_end),
        natural_endpoints,
        assume_unique=True,
    )
    train_indices = deterministic_cap_endpoint_indices(
        train_candidates,
        max_train_samples,
        seed=seed,
    )
    validation_indices = np.intersect1d(
        _allowed_indices(prepared, fold.validation_start, fold.validation_end),
        natural_endpoints,
        assume_unique=True,
    )
    test_indices = np.intersect1d(
        _allowed_indices(prepared, fold.test_start, fold.test_end),
        natural_endpoints,
        assume_unique=True,
    )

    if set(train_indices) & set(validation_indices):
        raise RuntimeError("train and validation endpoints overlap")
    if set(train_indices) & set(test_indices):
        raise RuntimeError("train and test endpoints overlap")
    if set(validation_indices) & set(test_indices):
        raise RuntimeError("validation and test endpoints overlap")

    common = {
        "features": prepared.features,
        "groups": prepared.groups,
        "sequence_length": int(sequence_length),
    }
    train = LazySequenceDataset(
        targets=prepared.targets,
        allowed_endpoint_indices=train_indices,
        **common,
    )
    validation = LazySequenceDataset(
        targets=prepared.targets,
        allowed_endpoint_indices=validation_indices,
        **common,
    )
    # OOS labels are deliberately absent from the model prediction dataset.
    test = LazySequenceDataset(
        targets=None,
        allowed_endpoint_indices=test_indices,
        **common,
    )
    for name, dataset in {
        "train": train,
        "validation": validation,
        "test": test,
    }.items():
        if len(dataset) == 0:
            raise ValueError(
                f"Fold {fold.fold_id} has no {name} endpoints after sequence warmup"
            )
        _assert_lookback_only(prepared, dataset)
    return SequenceFoldDatasets(prepared.frame, train, validation, test)


def build_fold_sequence_datasets(
    panel: pd.DataFrame,
    feature_columns: Sequence[str],
    fold: PurgedFold,
    *,
    sequence_length: int,
    max_train_samples: Optional[int] = None,
    random_seed: int = 42,
) -> SequenceFoldDatasets:
    """Build auditable endpoint-disjoint datasets for one ``PurgedFold``."""

    prepared = _prepare_panel(panel, feature_columns)
    return _datasets_from_fold(
        prepared,
        fold,
        sequence_length=sequence_length,
        max_train_samples=max_train_samples,
        seed=random_seed,
    )


def _fit_model_pair(
    datasets: SequenceFoldDatasets,
    feature_count: int,
    settings: SequencePipelineSettings,
    *,
    seed: int,
):
    gru = GRUSequenceRegressor(
        feature_count,
        hidden_dim=settings.gru_hidden_dim,
        num_layers=settings.gru_layers,
        dropout=settings.dropout,
        max_parameters=settings.max_parameters,
        seed=seed,
        device=settings.device,
        learning_rate=settings.learning_rate,
        weight_decay=settings.weight_decay,
    )
    transformer = TransformerSequenceRegressor(
        feature_count,
        d_model=settings.transformer_d_model,
        nhead=settings.transformer_heads,
        num_layers=settings.transformer_layers,
        dim_feedforward=settings.transformer_feedforward,
        dropout=settings.dropout,
        max_sequence_length=settings.sequence_length,
        max_parameters=settings.max_parameters,
        seed=seed + 1,
        device=settings.device,
        learning_rate=settings.learning_rate,
        weight_decay=settings.weight_decay,
    )
    fit_kwargs = {
        "epochs": settings.epochs,
        "batch_size": settings.batch_size,
        # Validation is always the explicit later chronological interval.
        "validation_dataset": datasets.validation,
        "patience": settings.patience,
        "shuffle": True,
        "num_workers": 0,
    }
    gru.fit(datasets.train, **fit_kwargs)
    transformer.fit(datasets.train, **fit_kwargs)
    return gru, transformer


def _prediction_shell(
    frame: pd.DataFrame,
    endpoints: np.ndarray,
) -> pd.DataFrame:
    optional = [
        name
        for name in [
            "execution_timestamp",
            "label_residual",
            "sample_weight",
        ]
        if name in frame.columns
    ]
    output = frame.iloc[endpoints][
        ["timestamp", "symbol", "label_rank", *optional, "_source_row", "_endpoint_row"]
    ].copy()
    output = output.rename(
        columns={"_source_row": "source_row", "_endpoint_row": "endpoint_row"}
    )
    output["sequence_end_timestamp"] = output["timestamp"]
    return output.reset_index(drop=True)


def _score_pair(
    gru: GRUSequenceRegressor,
    transformer: TransformerSequenceRegressor,
    datasets: SequenceFoldDatasets,
    *,
    batch_size: int,
) -> pd.DataFrame:
    endpoints = datasets.test.end_indices
    output = _prediction_shell(datasets.panel, endpoints)
    gru_raw = gru.predict(datasets.test, batch_size=batch_size)
    transformer_raw = transformer.predict(datasets.test, batch_size=batch_size)
    if len(gru_raw) != len(output) or len(transformer_raw) != len(output):
        raise RuntimeError("prediction count does not match sequence endpoints")
    output["score_gru"] = cross_sectional_percentile_rank(
        gru_raw, output["timestamp"], center=True
    )
    output["score_transformer"] = cross_sectional_percentile_rank(
        transformer_raw, output["timestamp"], center=True
    )
    equal = CalibratedRankEnsemble({"gru": 1.0, "transformer": 1.0})
    output["score_sequence_equal"] = equal.combine_calibrated(
        {
            "gru": output["score_gru"].to_numpy(dtype=float),
            "transformer": output["score_transformer"].to_numpy(dtype=float),
        }
    )
    return output


def run_sequence_walk_forward(
    panel: pd.DataFrame,
    feature_columns: Sequence[str],
    folds: Sequence[PurgedFold],
    *,
    settings: Optional[SequencePipelineSettings] = None,
) -> Tuple[pd.DataFrame, Tuple[Dict[str, object], ...]]:
    """Train GRU/Transformer per fold and emit unique, calibrated OOS scores."""

    if not folds:
        raise ValueError("at least one PurgedFold is required")
    cfg = settings or SequencePipelineSettings()
    prepared = _prepare_panel(panel, feature_columns)
    ordered_folds = tuple(sorted(folds, key=lambda item: (item.test_start, item.fold_id)))
    outputs = []
    records = []
    for fold_number, fold in enumerate(ordered_folds):
        fold_seed = cfg.random_seed + fold_number * 1000
        datasets = _datasets_from_fold(
            prepared,
            fold,
            sequence_length=cfg.sequence_length,
            max_train_samples=cfg.max_train_samples,
            seed=fold_seed,
        )
        gru, transformer = _fit_model_pair(
            datasets,
            len(feature_columns),
            cfg,
            seed=fold_seed,
        )
        output = _score_pair(
            gru,
            transformer,
            datasets,
            batch_size=cfg.batch_size,
        )
        output["fold_id"] = fold.fold_id
        output["train_end"] = fold.train_end
        output["validation_end"] = fold.validation_end
        outputs.append(output)
        records.append(
            {
                "fold": fold.as_dict(),
                "seed": fold_seed,
                "endpoint_rows": {
                    "train": int(len(datasets.train)),
                    "validation": int(len(datasets.validation)),
                    "test": int(len(datasets.test)),
                },
                "train_cap": cfg.max_train_samples,
                "gru_parameters": count_trainable_parameters(gru.model),
                "transformer_parameters": count_trainable_parameters(
                    transformer.model
                ),
                "gru_history": tuple(dict(row) for row in gru.history_),
                "transformer_history": tuple(
                    dict(row) for row in transformer.history_
                ),
            }
        )

    result = pd.concat(outputs, ignore_index=True).sort_values(
        ["timestamp", "symbol"], kind="mergesort"
    )
    if result.duplicated(["timestamp", "symbol"]).any():
        raise RuntimeError("walk-forward folds produced duplicate OOS predictions")
    if (pd.to_datetime(result["sequence_end_timestamp"]) > pd.to_datetime(result["timestamp"])).any():
        raise RuntimeError("a sequence prediction is anchored after its signal date")
    return result.reset_index(drop=True), tuple(records)


def _mean_rank_ic(
    frame: pd.DataFrame,
    prediction_col: str,
    *,
    minimum_names: int,
) -> float:
    values = daily_rank_ic(
        frame,
        prediction_col=prediction_col,
        label_col="label_rank",
        minimum_names=minimum_names,
    )
    return float(values.mean()) if len(values) else -np.inf


def _lock_settings_from_selection(
    selection_predictions: pd.DataFrame,
    *,
    settings: SequencePipelineSettings,
    selection_lock_end: pd.Timestamp,
) -> Dict[str, object]:
    eligible = selection_predictions.loc[
        pd.to_datetime(selection_predictions["timestamp"]) <= selection_lock_end
    ].copy()
    if eligible.empty:
        raise ValueError("no selection OOS predictions precede the lockbox embargo")

    raw_model_scores = {
        "gru": _mean_rank_ic(
            eligible, "score_gru", minimum_names=settings.minimum_cross_section
        ),
        "transformer": _mean_rank_ic(
            eligible,
            "score_transformer",
            minimum_names=settings.minimum_cross_section,
        ),
    }
    step = settings.ensemble_grid_increment
    grid = np.arange(0.0, 1.0 + step / 2.0, step)
    grid = np.unique(np.r_[grid[grid <= 1.0 + 1e-12], 1.0])
    best = None
    calibrated = {
        "gru": eligible["score_gru"].to_numpy(dtype=float),
        "transformer": eligible["score_transformer"].to_numpy(dtype=float),
    }
    for gru_weight in grid:
        ensemble = CalibratedRankEnsemble(
            {"gru": float(gru_weight), "transformer": float(1.0 - gru_weight)},
            max_weight=settings.ensemble_single_model_cap,
            shrinkage=settings.ensemble_shrinkage,
        )
        trial = eligible[["timestamp", "label_rank"]].copy()
        trial["prediction"] = ensemble.combine_calibrated(calibrated)
        score = _mean_rank_ic(
            trial,
            "prediction",
            minimum_names=settings.minimum_cross_section,
        )
        actual = ensemble.weights_.as_dict()
        distance = sum(abs(value - 0.5) for value in actual.values())
        key = (score, -distance, actual["gru"])
        if best is None or key > best[0]:
            best = (key, actual, score)
    if best is None:
        raise RuntimeError("no feasible sequence ensemble weights")
    return {
        "pipeline": {
            key: (str(value) if isinstance(value, torch.device) else value)
            for key, value in asdict(settings).items()
        },
        "selection_lock_end": str(selection_lock_end.date()),
        "selection_rows_used": int(len(eligible)),
        "model_selection_rank_ic": raw_model_scores,
        "ensemble_weights": best[1],
        "ensemble_selection_rank_ic": float(best[2]),
    }


def _calendar_count(
    calendar: pd.DatetimeIndex,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> int:
    return int(((calendar >= pd.Timestamp(start)) & (calendar <= pd.Timestamp(end))).sum())


def _lockbox_fold(
    prepared: _PreparedPanel,
    folds: Sequence[PurgedFold],
    *,
    selection_lock_end: pd.Timestamp,
    lockbox_start: pd.Timestamp,
    embargo_days: int,
) -> PurgedFold:
    calendar = pd.DatetimeIndex(
        prepared.frame["timestamp"].drop_duplicates().sort_values()
    )
    lock_position = int(calendar.searchsorted(lockbox_start, side="left"))
    if lock_position >= len(calendar):
        raise ValueError("lockbox_start is after the panel calendar")
    embargo_validation_end = lock_position - int(embargo_days) - 1
    if embargo_validation_end < 0:
        raise ValueError("lockbox embargo leaves no preceding validation history")
    configured_end_position = int(
        calendar.searchsorted(selection_lock_end, side="right") - 1
    )
    validation_end_position = min(
        embargo_validation_end, configured_end_position
    )
    if validation_end_position < 0:
        raise ValueError("selection_lock_end is before the panel calendar")

    last_fold = max(folds, key=lambda item: item.test_start)
    validation_days = _calendar_count(
        calendar, last_fold.validation_start, last_fold.validation_end
    )
    train_days = _calendar_count(
        calendar, last_fold.train_start, last_fold.train_end
    )
    validation_start_position = validation_end_position - validation_days + 1
    train_end_position = validation_start_position - int(last_fold.purge_days) - 1
    train_start_position = train_end_position - train_days + 1
    if train_start_position < 0 or train_end_position < train_start_position:
        raise ValueError("lockbox split has insufficient train/validation history")
    return PurgedFold(
        fold_id="LOCKBOX",
        train_start=calendar[train_start_position],
        train_end=calendar[train_end_position],
        validation_start=calendar[validation_start_position],
        validation_end=calendar[validation_end_position],
        test_start=calendar[lock_position],
        test_end=calendar[-1],
        purge_days=int(last_fold.purge_days),
        embargo_days=int(embargo_days),
    )


def fit_sequence_lockbox(
    panel: pd.DataFrame,
    feature_columns: Sequence[str],
    folds: Sequence[PurgedFold],
    *,
    locked_settings: Mapping[str, object],
    selection_lock_end: pd.Timestamp,
    lockbox_start: str,
    embargo_days: int,
    settings: SequencePipelineSettings,
) -> pd.DataFrame:
    """Fit fixed models before the lockbox and apply pre-locked weights."""

    prepared = _prepare_panel(panel, feature_columns)
    fold = _lockbox_fold(
        prepared,
        folds,
        selection_lock_end=selection_lock_end,
        lockbox_start=pd.Timestamp(lockbox_start),
        embargo_days=embargo_days,
    )
    datasets = _datasets_from_fold(
        prepared,
        fold,
        sequence_length=settings.sequence_length,
        max_train_samples=settings.max_train_samples,
        seed=settings.random_seed + 900_000,
    )
    gru, transformer = _fit_model_pair(
        datasets,
        len(feature_columns),
        settings,
        seed=settings.random_seed + 900_000,
    )
    output = _score_pair(
        gru,
        transformer,
        datasets,
        batch_size=settings.batch_size,
    )
    ensemble = CalibratedRankEnsemble(
        locked_settings["ensemble_weights"],  # type: ignore[arg-type]
        max_weight=settings.ensemble_single_model_cap,
        shrinkage=0.0,
    )
    output["score_sequence_locked"] = ensemble.combine_calibrated(
        {
            "gru": output["score_gru"].to_numpy(dtype=float),
            "transformer": output["score_transformer"].to_numpy(dtype=float),
        }
    )
    output["fold_id"] = "LOCKBOX"
    output["train_end"] = fold.train_end
    output["validation_end"] = fold.validation_end
    return output.sort_values(["timestamp", "symbol"], kind="mergesort").reset_index(
        drop=True
    )


def run_sequence_research(
    panel: pd.DataFrame,
    feature_columns: Sequence[str],
    folds: Sequence[PurgedFold],
    *,
    selection_end: str,
    lockbox_start: str,
    embargo_days: int,
    settings: Optional[SequencePipelineSettings] = None,
) -> SequenceResearchResult:
    """Run selection OOS, lock settings, then evaluate the untouched lockbox."""

    if embargo_days < 1:
        raise ValueError("embargo_days must be positive")
    if folds and embargo_days < max(int(fold.embargo_days) for fold in folds):
        raise ValueError(
            "lockbox embargo_days must be at least the walk-forward fold embargo"
        )
    if pd.Timestamp(lockbox_start) <= pd.Timestamp(selection_end):
        raise ValueError("lockbox_start must be later than selection_end")
    cfg = settings or SequencePipelineSettings()
    selection, records = run_sequence_walk_forward(
        panel, feature_columns, folds, settings=cfg
    )

    calendar = pd.DatetimeIndex(pd.to_datetime(panel["timestamp"]).drop_duplicates().sort_values())
    lock_position = int(calendar.searchsorted(pd.Timestamp(lockbox_start), side="left"))
    lock_cut_position = lock_position - int(embargo_days) - 1
    if lock_cut_position < 0:
        raise ValueError("lockbox embargo leaves no selection history")
    selection_lock_end = min(
        pd.Timestamp(selection_end), calendar[lock_cut_position]
    )
    locked = _lock_settings_from_selection(
        selection,
        settings=cfg,
        selection_lock_end=selection_lock_end,
    )
    locked_ensemble = CalibratedRankEnsemble(
        locked["ensemble_weights"],  # type: ignore[arg-type]
        max_weight=cfg.ensemble_single_model_cap,
        shrinkage=0.0,
    )
    selection = selection.copy()
    selection["score_sequence_locked"] = locked_ensemble.combine_calibrated(
        {
            "gru": selection["score_gru"].to_numpy(dtype=float),
            "transformer": selection["score_transformer"].to_numpy(dtype=float),
        }
    )
    lockbox = fit_sequence_lockbox(
        panel,
        feature_columns,
        folds,
        locked_settings=locked,
        selection_lock_end=selection_lock_end,
        lockbox_start=lockbox_start,
        embargo_days=embargo_days,
        settings=cfg,
    )
    locked = dict(locked)
    locked.update(
        {
            "lockbox_start": str(pd.Timestamp(lockbox_start).date()),
            "lockbox_rows": int(len(lockbox)),
            "lockbox_train_end": str(pd.Timestamp(lockbox["train_end"].iloc[0]).date()),
            "lockbox_validation_end": str(
                pd.Timestamp(lockbox["validation_end"].iloc[0]).date()
            ),
        }
    )
    if selection.duplicated(["timestamp", "symbol"]).any():
        raise RuntimeError("selection OOS predictions are not unique")
    if lockbox.duplicated(["timestamp", "symbol"]).any():
        raise RuntimeError("lockbox predictions are not unique")
    return SequenceResearchResult(selection, lockbox, records, locked)


__all__ = [
    "SequenceFoldDatasets",
    "SequencePipelineSettings",
    "SequenceResearchResult",
    "build_fold_sequence_datasets",
    "deterministic_cap_endpoint_indices",
    "fit_sequence_lockbox",
    "run_sequence_research",
    "run_sequence_walk_forward",
]
