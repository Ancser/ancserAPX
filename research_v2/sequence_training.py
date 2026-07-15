"""Checkpointed, resumable orchestration for offline sequence research.

The model and split semantics remain in :mod:`research_v2.sequence_pipeline`.
This module adds only long-run operational guarantees: every fold is published
by a same-directory atomic rename, every payload is SHA-256 verified before it
is reused, and the lockbox is not fitted until all selection folds and the
selection-only ensemble lock have completed.

No production module, broker credential, mutable data store, or live runner is
imported here.  Callers must provide an already frozen in-memory feature panel.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
from typing import Any, Callable, Dict, Literal, Mapping, Optional, Sequence, Tuple
import uuid

import numpy as np
import pandas as pd
import torch

from .models import CalibratedRankEnsemble
from .safety import RESEARCH_ROOT, UnsafeResearchPath, ensure_research_output_path
from .sequence import LazySequenceDataset, count_trainable_parameters
from .sequence_pipeline import (
    SequenceFoldDatasets,
    SequencePipelineSettings,
    SequenceResearchResult,
    _allowed_indices,
    _assert_lookback_only,
    _fit_model_pair,
    _lock_settings_from_selection,
    _lockbox_fold,
    _prepare_panel,
    _score_pair,
    deterministic_cap_endpoint_indices,
)
from .validation import PurgedFold


CHECKPOINT_FORMAT_VERSION = 1
_CHUNK_SIZE = 1024 * 1024
_CHECKPOINT_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,159}\Z")
EndpointSampling = Literal["row", "complete_date"]
ProgressCallback = Callable[[Mapping[str, object]], None]


class SequenceCheckpointError(RuntimeError):
    """Base error for sequence-training checkpoint failures."""


class SequenceCheckpointIntegrityError(SequenceCheckpointError):
    """A completed checkpoint is incomplete, stale, or has been modified."""


@dataclass(frozen=True)
class CheckpointedSelectionResult:
    predictions: pd.DataFrame
    fold_records: Tuple[Dict[str, object], ...]
    locked_settings: Dict[str, object]
    checkpoint_manifest_sha256: str


@dataclass(frozen=True)
class _LoadedCheckpoint:
    payloads: Dict[str, object]
    manifest_sha256: str


def _emit(
    callback: Optional[ProgressCallback], event: str, **details: object
) -> None:
    if callback is not None:
        callback({"event": event, **details})


def _jsonable(value: Any) -> Any:
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        _jsonable(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def _fingerprint_file(path: Path) -> Dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise SequenceCheckpointIntegrityError(
            f"checkpoint payload is missing, invalid, or a symlink: {path}"
        )
    before = path.stat()
    digest = _sha256_file(path)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (
        after.st_size,
        after.st_mtime_ns,
    ):
        raise SequenceCheckpointIntegrityError(
            f"checkpoint payload changed while hashing: {path}"
        )
    return {"size": int(after.st_size), "sha256": digest}


def _write_json(path: Path, payload: object) -> None:
    encoded = json.dumps(
        _jsonable(payload), indent=2, sort_keys=True, allow_nan=False
    ).encode("utf-8") + b"\n"
    with path.open("xb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())


def _write_parquet(path: Path, frame: pd.DataFrame) -> None:
    frame.to_parquet(path, index=False)
    # pandas closes the file before returning.  An explicit fsync makes the
    # publication contract independent of filesystem write-back timing.
    # Windows does not permit fsync on a read-only descriptor.
    with path.open("r+b") as handle:
        handle.flush()
        os.fsync(handle.fileno())


def _validate_checkpoint_name(name: str) -> str:
    if not _CHECKPOINT_NAME.fullmatch(name) or name in {".", ".."}:
        raise UnsafeResearchPath(f"invalid checkpoint name: {name!r}")
    return name


def _safe_remove_staging(path: Path, parent: Path, root: Path) -> None:
    checked = ensure_research_output_path(path, research_root=root)
    if checked.parent != parent or not checked.name.startswith("."):
        raise UnsafeResearchPath(f"refusing to remove non-staging path: {checked}")
    if checked.exists():
        shutil.rmtree(checked)


def _payload_reader(path: Path) -> object:
    if path.suffix == ".parquet":
        try:
            return pd.read_parquet(path)
        except Exception as exc:
            raise SequenceCheckpointIntegrityError(
                f"cannot read checkpoint parquet: {path}"
            ) from exc
    if path.suffix == ".json":
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise SequenceCheckpointIntegrityError(
                f"cannot read checkpoint JSON: {path}"
            ) from exc
    raise SequenceCheckpointIntegrityError(
        f"unsupported checkpoint payload type: {path.name}"
    )


def _load_checkpoint(
    directory: Path,
    *,
    checkpoint_type: str,
    identity: Mapping[str, object],
) -> _LoadedCheckpoint:
    if directory.is_symlink() or not directory.is_dir():
        raise SequenceCheckpointIntegrityError(
            f"checkpoint directory is missing or invalid: {directory}"
        )
    success_path = directory / "_SUCCESS"
    manifest_path = directory / "manifest.json"
    try:
        success = json.loads(success_path.read_text(encoding="utf-8"))
        manifest_bytes = manifest_path.read_bytes()
        manifest = json.loads(manifest_bytes.decode("utf-8"))
    except Exception as exc:
        raise SequenceCheckpointIntegrityError(
            f"checkpoint completion metadata is invalid: {directory}"
        ) from exc

    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    if success != {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "checkpoint_type": checkpoint_type,
        "manifest_sha256": manifest_sha256,
    }:
        raise SequenceCheckpointIntegrityError(
            f"checkpoint _SUCCESS does not authenticate manifest: {directory}"
        )
    if manifest.get("format_version") != CHECKPOINT_FORMAT_VERSION:
        raise SequenceCheckpointIntegrityError("unsupported checkpoint format")
    if manifest.get("checkpoint_type") != checkpoint_type:
        raise SequenceCheckpointIntegrityError("checkpoint type mismatch")
    expected_identity = _jsonable(identity)
    if manifest.get("identity") != expected_identity:
        raise SequenceCheckpointIntegrityError(
            f"checkpoint identity does not match this run: {directory}"
        )

    file_records = manifest.get("files")
    if not isinstance(file_records, dict) or not file_records:
        raise SequenceCheckpointIntegrityError("checkpoint manifest has no files")
    expected_names = set(file_records) | {"manifest.json", "_SUCCESS"}
    actual_names = {path.name for path in directory.iterdir()}
    if actual_names != expected_names:
        raise SequenceCheckpointIntegrityError(
            f"checkpoint file set mismatch: {directory}"
        )

    payloads: Dict[str, object] = {}
    for name, expected in sorted(file_records.items()):
        if Path(name).name != name or name in {"manifest.json", "_SUCCESS"}:
            raise SequenceCheckpointIntegrityError(
                f"unsafe checkpoint payload name: {name!r}"
            )
        path = directory / name
        actual = _fingerprint_file(path)
        if actual != expected:
            raise SequenceCheckpointIntegrityError(
                f"checkpoint payload fingerprint mismatch: {path}"
            )
        payloads[name] = _payload_reader(path)
    return _LoadedCheckpoint(payloads, manifest_sha256)


def _publish_checkpoint(
    directory: Path,
    *,
    checkpoint_type: str,
    identity: Mapping[str, object],
    parquet_payloads: Mapping[str, pd.DataFrame],
    json_payloads: Mapping[str, object],
    research_root: Path,
) -> _LoadedCheckpoint:
    parent = ensure_research_output_path(directory.parent, research_root=research_root)
    final = ensure_research_output_path(directory, research_root=research_root)
    parent.mkdir(parents=True, exist_ok=True)
    if final.exists():
        return _load_checkpoint(
            final, checkpoint_type=checkpoint_type, identity=identity
        )

    staging = ensure_research_output_path(
        parent / f".{final.name}.tmp-{uuid.uuid4().hex}",
        research_root=research_root,
    )
    staging.mkdir(parents=False, exist_ok=False)
    try:
        names = set(parquet_payloads) | set(json_payloads)
        if len(names) != len(parquet_payloads) + len(json_payloads):
            raise ValueError("checkpoint payload names must be unique")
        for name in names:
            if Path(name).name != name or name in {"manifest.json", "_SUCCESS"}:
                raise ValueError(f"invalid checkpoint payload name: {name!r}")

        for name, frame in parquet_payloads.items():
            _write_parquet(staging / name, frame)
        for name, payload in json_payloads.items():
            _write_json(staging / name, payload)

        fingerprints = {
            name: _fingerprint_file(staging / name) for name in sorted(names)
        }
        manifest = {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "checkpoint_type": checkpoint_type,
            "identity": _jsonable(identity),
            "files": fingerprints,
        }
        _write_json(staging / "manifest.json", manifest)
        manifest_sha256 = _sha256_file(staging / "manifest.json")
        _write_json(
            staging / "_SUCCESS",
            {
                "format_version": CHECKPOINT_FORMAT_VERSION,
                "checkpoint_type": checkpoint_type,
                "manifest_sha256": manifest_sha256,
            },
        )

        # A same-parent rename is the sole publication step.  On Windows it
        # refuses to replace a checkpoint created concurrently.
        if final.exists():
            raise SequenceCheckpointError(
                f"checkpoint appeared concurrently: {final}"
            )
        os.rename(staging, final)
        return _load_checkpoint(
            final, checkpoint_type=checkpoint_type, identity=identity
        )
    finally:
        if staging.exists():
            _safe_remove_staging(staging, parent, research_root)


def _settings_payload(settings: SequencePipelineSettings) -> Dict[str, object]:
    return _jsonable(asdict(settings))


def _fold_payload(fold: PurgedFold) -> Dict[str, object]:
    return {
        "fold_id": fold.fold_id,
        "train_start": pd.Timestamp(fold.train_start).isoformat(),
        "train_end": pd.Timestamp(fold.train_end).isoformat(),
        "validation_start": pd.Timestamp(fold.validation_start).isoformat(),
        "validation_end": pd.Timestamp(fold.validation_end).isoformat(),
        "test_start": pd.Timestamp(fold.test_start).isoformat(),
        "test_end": pd.Timestamp(fold.test_end).isoformat(),
        "purge_days": int(fold.purge_days),
        "embargo_days": int(fold.embargo_days),
    }


def _input_prefix_sha256(
    prepared: object,
    feature_columns: Sequence[str],
    *,
    end: pd.Timestamp,
) -> str:
    frame = prepared.frame  # type: ignore[attr-defined]
    optional = [
        name
        for name in ("execution_timestamp", "label_residual", "sample_weight")
        if name in frame.columns
    ]
    columns = [
        "timestamp",
        "symbol",
        "label_rank",
        *feature_columns,
        *optional,
        "_source_row",
        "_endpoint_row",
    ]
    relevant = frame.loc[
        pd.to_datetime(frame["timestamp"]) <= pd.Timestamp(end), columns
    ]
    row_hashes = pd.util.hash_pandas_object(
        relevant, index=False, categorize=True
    ).to_numpy(dtype="uint64", copy=False)
    digest = hashlib.sha256()
    digest.update(_canonical_json({
        "columns": columns,
        "dtypes": [str(relevant[column].dtype) for column in columns],
        "rows": int(len(relevant)),
    }))
    digest.update(np.ascontiguousarray(row_hashes).tobytes())
    return digest.hexdigest()


def _complete_date_cap(
    prepared: object,
    candidates: np.ndarray,
    max_samples: Optional[int],
    *,
    seed: int,
) -> Tuple[np.ndarray, Dict[str, object]]:
    candidates = np.sort(np.asarray(candidates, dtype=np.int64))
    if max_samples is None or candidates.size <= int(max_samples):
        return candidates, {
            "mode": "complete_date",
            "candidate_rows": int(candidates.size),
            "selected_rows": int(candidates.size),
            "selected_dates": None,
            "complete_cross_section": None,
        }
    if int(max_samples) < 1:
        raise ValueError("max_train_samples must be positive or None")

    frame = prepared.frame  # type: ignore[attr-defined]
    dates = pd.DatetimeIndex(pd.to_datetime(frame.iloc[candidates]["timestamp"]))
    unique_dates, counts = np.unique(dates.to_numpy(), return_counts=True)
    complete_count = int(counts.max())
    complete_dates = unique_dates[counts == complete_count]
    date_budget = int(max_samples) // complete_count
    if date_budget < 1:
        raise ValueError(
            "max_train_samples is smaller than one complete training date "
            f"({max_samples} < {complete_count})"
        )
    selected_date_count = min(int(len(complete_dates)), date_budget)
    rng = np.random.default_rng(int(seed))
    # One date per chronological stratum gives every part of the rolling train
    # window representation while retaining every name on each chosen date.
    strata = np.array_split(np.arange(len(complete_dates)), selected_date_count)
    chosen_positions = np.array(
        [int(rng.choice(stratum)) for stratum in strata], dtype=np.int64
    )
    chosen_dates = np.sort(complete_dates[chosen_positions])
    selected = candidates[np.isin(dates.to_numpy(), chosen_dates)]
    if selected.size > int(max_samples):
        raise RuntimeError("complete-date sampling exceeded its endpoint cap")
    return np.sort(selected), {
        "mode": "complete_date",
        "candidate_rows": int(candidates.size),
        "selected_rows": int(selected.size),
        "selected_dates": int(len(chosen_dates)),
        "complete_cross_section": complete_count,
    }


def _datasets_for_fold(
    prepared: object,
    fold: PurgedFold,
    *,
    settings: SequencePipelineSettings,
    seed: int,
    endpoint_sampling: EndpointSampling,
) -> Tuple[SequenceFoldDatasets, Dict[str, object]]:
    if endpoint_sampling not in {"row", "complete_date"}:
        raise ValueError(
            "endpoint_sampling must be either 'row' or 'complete_date'"
        )
    natural = LazySequenceDataset(
        prepared.features,  # type: ignore[attr-defined]
        targets=None,
        groups=prepared.groups,  # type: ignore[attr-defined]
        sequence_length=settings.sequence_length,
    ).end_indices
    train_candidates = np.intersect1d(
        _allowed_indices(prepared, fold.train_start, fold.train_end),
        natural,
        assume_unique=True,
    )
    if endpoint_sampling == "row":
        train_indices = deterministic_cap_endpoint_indices(
            train_candidates, settings.max_train_samples, seed=seed
        )
        sampling = {
            "mode": "row",
            "candidate_rows": int(len(train_candidates)),
            "selected_rows": int(len(train_indices)),
            "selected_dates": None,
            "complete_cross_section": None,
        }
    else:
        train_indices, sampling = _complete_date_cap(
            prepared,
            train_candidates,
            settings.max_train_samples,
            seed=seed,
        )

    validation_indices = np.intersect1d(
        _allowed_indices(prepared, fold.validation_start, fold.validation_end),
        natural,
        assume_unique=True,
    )
    test_indices = np.intersect1d(
        _allowed_indices(prepared, fold.test_start, fold.test_end),
        natural,
        assume_unique=True,
    )
    endpoint_sets = [set(train_indices), set(validation_indices), set(test_indices)]
    if not endpoint_sets[0].isdisjoint(endpoint_sets[1]):
        raise RuntimeError("train and validation endpoints overlap")
    if not endpoint_sets[0].isdisjoint(endpoint_sets[2]):
        raise RuntimeError("train and test endpoints overlap")
    if not endpoint_sets[1].isdisjoint(endpoint_sets[2]):
        raise RuntimeError("validation and test endpoints overlap")

    common = {
        "features": prepared.features,  # type: ignore[attr-defined]
        "groups": prepared.groups,  # type: ignore[attr-defined]
        "sequence_length": settings.sequence_length,
    }
    train = LazySequenceDataset(
        targets=prepared.targets,  # type: ignore[attr-defined]
        allowed_endpoint_indices=train_indices,
        **common,
    )
    validation = LazySequenceDataset(
        targets=prepared.targets,  # type: ignore[attr-defined]
        allowed_endpoint_indices=validation_indices,
        **common,
    )
    test = LazySequenceDataset(
        targets=None, allowed_endpoint_indices=test_indices, **common
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
    return (
        SequenceFoldDatasets(prepared.frame, train, validation, test),  # type: ignore[attr-defined]
        sampling,
    )


def _fold_identity(
    prepared: object,
    feature_columns: Sequence[str],
    fold: PurgedFold,
    *,
    settings: SequencePipelineSettings,
    seed: int,
    endpoint_sampling: EndpointSampling,
    checkpoint_type: str,
    extra: Optional[Mapping[str, object]] = None,
) -> Dict[str, object]:
    return {
        "checkpoint_type": checkpoint_type,
        "fold": _fold_payload(fold),
        "feature_columns": list(feature_columns),
        "settings": _settings_payload(settings),
        "seed": int(seed),
        "endpoint_sampling": endpoint_sampling,
        "input_prefix_sha256": _input_prefix_sha256(
            prepared, feature_columns, end=fold.test_end
        ),
        "extra": _jsonable(extra or {}),
    }


def _record_from_models(
    fold: PurgedFold,
    *,
    seed: int,
    datasets: SequenceFoldDatasets,
    settings: SequencePipelineSettings,
    sampling: Mapping[str, object],
    gru: object,
    transformer: object,
) -> Dict[str, object]:
    return {
        "fold": fold.as_dict(),
        "seed": int(seed),
        "endpoint_rows": {
            "train": int(len(datasets.train)),
            "validation": int(len(datasets.validation)),
            "test": int(len(datasets.test)),
        },
        "train_cap": settings.max_train_samples,
        "endpoint_sampling": dict(sampling),
        "gru_parameters": count_trainable_parameters(gru.model),  # type: ignore[attr-defined]
        "transformer_parameters": count_trainable_parameters(
            transformer.model  # type: ignore[attr-defined]
        ),
        "gru_history": [dict(row) for row in gru.history_],  # type: ignore[attr-defined]
        "transformer_history": [
            dict(row) for row in transformer.history_  # type: ignore[attr-defined]
        ],
    }


def _train_or_resume_fold(
    prepared: object,
    feature_columns: Sequence[str],
    fold: PurgedFold,
    *,
    fold_number: int,
    settings: SequencePipelineSettings,
    endpoint_sampling: EndpointSampling,
    folds_dir: Path,
    research_root: Path,
    progress_callback: Optional[ProgressCallback],
) -> Tuple[pd.DataFrame, Dict[str, object], str]:
    fold_name = _validate_checkpoint_name(fold.fold_id)
    directory = ensure_research_output_path(
        folds_dir / fold_name, research_root=research_root
    )
    seed = settings.random_seed + fold_number * 1000
    identity = _fold_identity(
        prepared,
        feature_columns,
        fold,
        settings=settings,
        seed=seed,
        endpoint_sampling=endpoint_sampling,
        checkpoint_type="selection_fold",
    )
    if directory.exists():
        loaded = _load_checkpoint(
            directory, checkpoint_type="selection_fold", identity=identity
        )
        _emit(
            progress_callback,
            "fold_resumed",
            fold_id=fold.fold_id,
            fold_number=fold_number,
        )
        return (
            loaded.payloads["predictions.parquet"],  # type: ignore[return-value]
            loaded.payloads["history.json"],  # type: ignore[return-value]
            loaded.manifest_sha256,
        )

    datasets, sampling = _datasets_for_fold(
        prepared,
        fold,
        settings=settings,
        seed=seed,
        endpoint_sampling=endpoint_sampling,
    )
    _emit(
        progress_callback,
        "fold_started",
        fold_id=fold.fold_id,
        fold_number=fold_number,
        train_rows=len(datasets.train),
        validation_rows=len(datasets.validation),
        test_rows=len(datasets.test),
    )
    gru, transformer = _fit_model_pair(
        datasets, len(feature_columns), settings, seed=seed
    )
    output = _score_pair(
        gru, transformer, datasets, batch_size=settings.batch_size
    )
    output["fold_id"] = fold.fold_id
    output["train_end"] = fold.train_end
    output["validation_end"] = fold.validation_end
    record = _record_from_models(
        fold,
        seed=seed,
        datasets=datasets,
        settings=settings,
        sampling=sampling,
        gru=gru,
        transformer=transformer,
    )
    loaded = _publish_checkpoint(
        directory,
        checkpoint_type="selection_fold",
        identity=identity,
        parquet_payloads={"predictions.parquet": output},
        json_payloads={
            "history.json": record,
            "settings.json": {
                "pipeline": _settings_payload(settings),
                "feature_columns": list(feature_columns),
                "endpoint_sampling": endpoint_sampling,
            },
        },
        research_root=research_root,
    )
    _emit(
        progress_callback,
        "fold_completed",
        fold_id=fold.fold_id,
        fold_number=fold_number,
    )
    return output, record, loaded.manifest_sha256


def _selection_lock_end(
    panel: pd.DataFrame,
    *,
    selection_end: str,
    lockbox_start: str,
    embargo_days: int,
) -> pd.Timestamp:
    calendar = pd.DatetimeIndex(
        pd.to_datetime(panel["timestamp"]).drop_duplicates().sort_values()
    )
    lock_position = int(
        calendar.searchsorted(pd.Timestamp(lockbox_start), side="left")
    )
    lock_cut_position = lock_position - int(embargo_days) - 1
    if lock_position >= len(calendar) or lock_cut_position < 0:
        raise ValueError("lockbox embargo leaves no selection history")
    return min(pd.Timestamp(selection_end), calendar[lock_cut_position])


def _run_selection(
    prepared: object,
    feature_columns: Sequence[str],
    folds: Sequence[PurgedFold],
    *,
    output_dir: Path,
    selection_end: str,
    lockbox_start: str,
    embargo_days: int,
    settings: SequencePipelineSettings,
    endpoint_sampling: EndpointSampling,
    research_root: Path,
    progress_callback: Optional[ProgressCallback],
) -> CheckpointedSelectionResult:
    if not folds:
        raise ValueError("at least one PurgedFold is required")
    ordered = tuple(
        sorted(folds, key=lambda item: (item.test_start, item.fold_id))
    )
    folds_dir = ensure_research_output_path(
        output_dir / "folds", research_root=research_root
    )
    outputs = []
    records = []
    fold_manifests = []
    for number, fold in enumerate(ordered):
        output, record, manifest_sha256 = _train_or_resume_fold(
            prepared,
            feature_columns,
            fold,
            fold_number=number,
            settings=settings,
            endpoint_sampling=endpoint_sampling,
            folds_dir=folds_dir,
            research_root=research_root,
            progress_callback=progress_callback,
        )
        outputs.append(output)
        records.append(record)
        fold_manifests.append(
            {"fold_id": fold.fold_id, "manifest_sha256": manifest_sha256}
        )

    selection = pd.concat(outputs, ignore_index=True).sort_values(
        ["timestamp", "symbol"], kind="mergesort"
    ).reset_index(drop=True)
    if selection.duplicated(["timestamp", "symbol"]).any():
        raise RuntimeError("walk-forward folds produced duplicate OOS predictions")
    if (
        pd.to_datetime(selection["sequence_end_timestamp"])
        > pd.to_datetime(selection["timestamp"])
    ).any():
        raise RuntimeError("a sequence prediction is anchored after its signal date")

    lock_end = _selection_lock_end(
        prepared.frame,  # type: ignore[attr-defined]
        selection_end=selection_end,
        lockbox_start=lockbox_start,
        embargo_days=embargo_days,
    )
    identity = {
        "checkpoint_type": "selection",
        "feature_columns": list(feature_columns),
        "settings": _settings_payload(settings),
        "endpoint_sampling": endpoint_sampling,
        "selection_end": str(selection_end),
        "lockbox_start": str(lockbox_start),
        "embargo_days": int(embargo_days),
        "selection_lock_end": lock_end.isoformat(),
        "fold_manifests": fold_manifests,
    }
    directory = ensure_research_output_path(
        output_dir / "selection", research_root=research_root
    )
    if directory.exists():
        loaded = _load_checkpoint(
            directory, checkpoint_type="selection", identity=identity
        )
        _emit(progress_callback, "selection_resumed", rows=len(selection))
        return CheckpointedSelectionResult(
            loaded.payloads["predictions.parquet"],  # type: ignore[arg-type]
            tuple(loaded.payloads["history.json"]),  # type: ignore[arg-type]
            dict(loaded.payloads["settings.json"]),  # type: ignore[arg-type]
            loaded.manifest_sha256,
        )

    locked = _lock_settings_from_selection(
        selection, settings=settings, selection_lock_end=lock_end
    )
    ensemble = CalibratedRankEnsemble(
        locked["ensemble_weights"],  # type: ignore[arg-type]
        max_weight=settings.ensemble_single_model_cap,
        shrinkage=0.0,
    )
    selection = selection.copy()
    selection["score_sequence_locked"] = ensemble.combine_calibrated(
        {
            "gru": selection["score_gru"].to_numpy(dtype=float),
            "transformer": selection["score_transformer"].to_numpy(dtype=float),
        }
    )
    loaded = _publish_checkpoint(
        directory,
        checkpoint_type="selection",
        identity=identity,
        parquet_payloads={"predictions.parquet": selection},
        json_payloads={
            "history.json": records,
            "settings.json": locked,
        },
        research_root=research_root,
    )
    _emit(progress_callback, "selection_completed", rows=len(selection))
    return CheckpointedSelectionResult(
        selection, tuple(records), locked, loaded.manifest_sha256
    )


def run_checkpointed_sequence_selection(
    panel: pd.DataFrame,
    feature_columns: Sequence[str],
    folds: Sequence[PurgedFold],
    *,
    output_dir: Path | str,
    selection_end: str,
    lockbox_start: str,
    embargo_days: int,
    settings: Optional[SequencePipelineSettings] = None,
    endpoint_sampling: EndpointSampling = "row",
    research_root: Path | str = RESEARCH_ROOT,
    progress_callback: Optional[ProgressCallback] = None,
) -> CheckpointedSelectionResult:
    """Train or resume selection folds, then lock ensemble settings once."""

    if embargo_days < 1:
        raise ValueError("embargo_days must be positive")
    if folds and embargo_days < max(int(fold.embargo_days) for fold in folds):
        raise ValueError(
            "lockbox embargo_days must be at least the walk-forward fold embargo"
        )
    if pd.Timestamp(lockbox_start) <= pd.Timestamp(selection_end):
        raise ValueError("lockbox_start must be later than selection_end")
    cfg = settings or SequencePipelineSettings()
    root = Path(research_root).expanduser().resolve(strict=False)
    ensure_research_output_path(root, research_root=root)
    output = ensure_research_output_path(output_dir, research_root=root)
    output.mkdir(parents=True, exist_ok=True)
    prepared = _prepare_panel(panel, feature_columns)
    return _run_selection(
        prepared,
        feature_columns,
        folds,
        output_dir=output,
        selection_end=selection_end,
        lockbox_start=lockbox_start,
        embargo_days=embargo_days,
        settings=cfg,
        endpoint_sampling=endpoint_sampling,
        research_root=root,
        progress_callback=progress_callback,
    )


def _train_or_resume_lockbox(
    prepared: object,
    feature_columns: Sequence[str],
    folds: Sequence[PurgedFold],
    selection: CheckpointedSelectionResult,
    *,
    output_dir: Path,
    lockbox_start: str,
    embargo_days: int,
    settings: SequencePipelineSettings,
    endpoint_sampling: EndpointSampling,
    research_root: Path,
    progress_callback: Optional[ProgressCallback],
) -> Tuple[pd.DataFrame, Dict[str, object], str]:
    selection_lock_end = pd.Timestamp(
        selection.locked_settings["selection_lock_end"]
    )
    fold = _lockbox_fold(
        prepared,
        folds,
        selection_lock_end=selection_lock_end,
        lockbox_start=pd.Timestamp(lockbox_start),
        embargo_days=embargo_days,
    )
    seed = settings.random_seed + 900_000
    identity = _fold_identity(
        prepared,
        feature_columns,
        fold,
        settings=settings,
        seed=seed,
        endpoint_sampling=endpoint_sampling,
        checkpoint_type="lockbox",
        extra={
            "selection_manifest_sha256": selection.checkpoint_manifest_sha256,
            "ensemble_weights": selection.locked_settings["ensemble_weights"],
        },
    )
    directory = ensure_research_output_path(
        output_dir / "lockbox", research_root=research_root
    )
    if directory.exists():
        loaded = _load_checkpoint(
            directory, checkpoint_type="lockbox", identity=identity
        )
        _emit(progress_callback, "lockbox_resumed")
        return (
            loaded.payloads["predictions.parquet"],  # type: ignore[return-value]
            loaded.payloads["history.json"],  # type: ignore[return-value]
            loaded.manifest_sha256,
        )

    datasets, sampling = _datasets_for_fold(
        prepared,
        fold,
        settings=settings,
        seed=seed,
        endpoint_sampling=endpoint_sampling,
    )
    _emit(
        progress_callback,
        "lockbox_started",
        train_rows=len(datasets.train),
        validation_rows=len(datasets.validation),
        test_rows=len(datasets.test),
    )
    gru, transformer = _fit_model_pair(
        datasets, len(feature_columns), settings, seed=seed
    )
    output = _score_pair(
        gru, transformer, datasets, batch_size=settings.batch_size
    )
    ensemble = CalibratedRankEnsemble(
        selection.locked_settings["ensemble_weights"],  # type: ignore[arg-type]
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
    output = output.sort_values(
        ["timestamp", "symbol"], kind="mergesort"
    ).reset_index(drop=True)
    record = _record_from_models(
        fold,
        seed=seed,
        datasets=datasets,
        settings=settings,
        sampling=sampling,
        gru=gru,
        transformer=transformer,
    )
    loaded = _publish_checkpoint(
        directory,
        checkpoint_type="lockbox",
        identity=identity,
        parquet_payloads={"predictions.parquet": output},
        json_payloads={
            "history.json": record,
            "settings.json": {
                "pipeline": _settings_payload(settings),
                "feature_columns": list(feature_columns),
                "endpoint_sampling": endpoint_sampling,
                "locked_selection_settings": selection.locked_settings,
            },
        },
        research_root=research_root,
    )
    _emit(progress_callback, "lockbox_completed", rows=len(output))
    return output, record, loaded.manifest_sha256


def _write_root_success(
    output_dir: Path,
    payload: Mapping[str, object],
) -> None:
    path = output_dir / "_SUCCESS"
    normalized = _jsonable(payload)
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise SequenceCheckpointIntegrityError(
                f"invalid root _SUCCESS: {path}"
            ) from exc
        if existing != normalized:
            raise SequenceCheckpointIntegrityError(
                "root _SUCCESS does not match verified child checkpoints"
            )
        return
    temporary = output_dir / f"._SUCCESS.tmp-{uuid.uuid4().hex}"
    try:
        _write_json(temporary, normalized)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def run_checkpointed_sequence_research(
    panel: pd.DataFrame,
    feature_columns: Sequence[str],
    folds: Sequence[PurgedFold],
    *,
    output_dir: Path | str,
    selection_end: str,
    lockbox_start: str,
    embargo_days: int,
    settings: Optional[SequencePipelineSettings] = None,
    endpoint_sampling: EndpointSampling = "row",
    research_root: Path | str = RESEARCH_ROOT,
    progress_callback: Optional[ProgressCallback] = None,
) -> SequenceResearchResult:
    """Run resumable selection training, lock settings, then fit the lockbox."""

    if embargo_days < 1:
        raise ValueError("embargo_days must be positive")
    if not folds:
        raise ValueError("at least one PurgedFold is required")
    if embargo_days < max(int(fold.embargo_days) for fold in folds):
        raise ValueError(
            "lockbox embargo_days must be at least the walk-forward fold embargo"
        )
    if pd.Timestamp(lockbox_start) <= pd.Timestamp(selection_end):
        raise ValueError("lockbox_start must be later than selection_end")
    if endpoint_sampling not in {"row", "complete_date"}:
        raise ValueError(
            "endpoint_sampling must be either 'row' or 'complete_date'"
        )

    cfg = settings or SequencePipelineSettings()
    root = Path(research_root).expanduser().resolve(strict=False)
    ensure_research_output_path(root, research_root=root)
    output = ensure_research_output_path(output_dir, research_root=root)
    output.mkdir(parents=True, exist_ok=True)
    prepared = _prepare_panel(panel, feature_columns)
    selection = _run_selection(
        prepared,
        feature_columns,
        folds,
        output_dir=output,
        selection_end=selection_end,
        lockbox_start=lockbox_start,
        embargo_days=embargo_days,
        settings=cfg,
        endpoint_sampling=endpoint_sampling,
        research_root=root,
        progress_callback=progress_callback,
    )
    lockbox, lockbox_record, lockbox_manifest = _train_or_resume_lockbox(
        prepared,
        feature_columns,
        folds,
        selection,
        output_dir=output,
        lockbox_start=lockbox_start,
        embargo_days=embargo_days,
        settings=cfg,
        endpoint_sampling=endpoint_sampling,
        research_root=root,
        progress_callback=progress_callback,
    )
    if lockbox.duplicated(["timestamp", "symbol"]).any():
        raise RuntimeError("lockbox predictions are not unique")

    locked = dict(selection.locked_settings)
    locked.update(
        {
            "lockbox_start": str(pd.Timestamp(lockbox_start).date()),
            "lockbox_rows": int(len(lockbox)),
            "lockbox_train_end": str(
                pd.Timestamp(lockbox["train_end"].iloc[0]).date()
            ),
            "lockbox_validation_end": str(
                pd.Timestamp(lockbox["validation_end"].iloc[0]).date()
            ),
        }
    )
    root_success = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "checkpoint_type": "sequence_research",
        "selection_manifest_sha256": selection.checkpoint_manifest_sha256,
        "lockbox_manifest_sha256": lockbox_manifest,
        "locked_settings_sha256": hashlib.sha256(
            _canonical_json(locked)
        ).hexdigest(),
    }
    _write_root_success(output, root_success)
    _emit(
        progress_callback,
        "run_completed",
        selection_rows=len(selection.predictions),
        lockbox_rows=len(lockbox),
    )
    # The public pipeline result intentionally contains selection fold records
    # only.  Lockbox history remains independently verified in its checkpoint.
    del lockbox_record
    return SequenceResearchResult(
        selection.predictions,
        lockbox,
        selection.fold_records,
        locked,
    )


__all__ = [
    "CHECKPOINT_FORMAT_VERSION",
    "CheckpointedSelectionResult",
    "EndpointSampling",
    "ProgressCallback",
    "SequenceCheckpointError",
    "SequenceCheckpointIntegrityError",
    "run_checkpointed_sequence_research",
    "run_checkpointed_sequence_selection",
]
