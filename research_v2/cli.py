"""Reproducible, offline command line entry point for Research v2.

Importing this module performs no filesystem reads/writes, data loading, model
initialisation, or environment mutation.  Heavy research modules are imported
only after :func:`main` has entered ``offline_context``.

Every published stage is immutable and carries a ``_SUCCESS.json`` manifest.
The manifest binds the artifact to the canonical snapshot, full resolved
configuration, code, dependency versions, deterministic seeds, stage
parameters, inputs, and every output byte.  A resumed run skips a stage only
after re-hashing and verifying all of those outputs.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from importlib import metadata as importlib_metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import sys
from typing import Any, Callable, Mapping, Sequence
import uuid


# Constants are deliberately path objects only; no path is resolved or probed
# at import time.  ``canonical`` is an alias resolved at command execution to a
# literal ``snapshots/canonical`` or the lexicographically latest
# ``snapshots/canonical-*`` directory.
MODULE_DIR = Path(__file__).parent
DEFAULT_RESEARCH_ROOT = MODULE_DIR
DEFAULT_CONFIG = "default_config.json"
DEFAULT_SNAPSHOT = "canonical"
DEFAULT_RUN_ID = "canonical"
ARTIFACT_FORMAT_VERSION = 1
RUN_FORMAT_VERSION = 1
SUCCESS_FILE = "_SUCCESS.json"
_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_DEPENDENCIES = (
    "numpy",
    "pandas",
    "polars",
    "pyarrow",
    "scikit-learn",
    "torch",
)


class CLIError(RuntimeError):
    """A safe, user-facing CLI failure."""


class ArtifactVerificationError(CLIError):
    """An artifact is incomplete, mismatched, or was modified after publish."""


@dataclass(frozen=True)
class RunContext:
    research_root: Path
    snapshot_dir: Path
    snapshot_metadata: Mapping[str, Any]
    snapshot_data_sha256: str
    config_path: Path | None
    config_file_sha256: str | None
    config: Any
    config_sha256: str
    run_id: str
    run_dir: Path
    code_sha256: str
    versions: Mapping[str, str]
    seeds: Mapping[str, Any]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    # Avoid importing NumPy merely for serialization.  Scalars expose item().
    if type(value).__module__.startswith("numpy") and hasattr(value, "item"):
        return _json_safe(value.item())
    return value


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        _json_safe(value),
        sort_keys=True,
        separators=(",", ":"),
        default=str,
        allow_nan=False,
    ).encode("utf-8")


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(_json_safe(value), indent=2, sort_keys=True, default=str, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_id(value: str, label: str) -> str:
    if not _ID_RE.fullmatch(value) or value in {".", ".."}:
        raise CLIError(f"invalid {label}: {value!r}")
    return value


def _resolved_research_root(value: str | os.PathLike[str]) -> Path:
    # The safety module owns the authoritative containment check.
    from .safety import ensure_research_output_path

    root = Path(value).expanduser().resolve(strict=False)
    return ensure_research_output_path(root, research_root=root)


def _safe_output(path: Path | str, root: Path) -> Path:
    from .safety import ensure_research_output_path

    return ensure_research_output_path(path, research_root=root)


def _atomic_write(path: Path, payload: bytes, *, root: Path) -> None:
    destination = _safe_output(path, root)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = _safe_output(
        destination.parent / f".{destination.name}.partial-{uuid.uuid4().hex}",
        root,
    )
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_write_json(path: Path, value: Any, *, root: Path) -> None:
    _atomic_write(path, _json_bytes(value), root=root)


def _dependency_versions() -> dict[str, str]:
    versions = {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
    }
    for distribution in _DEPENDENCIES:
        try:
            versions[distribution] = importlib_metadata.version(distribution)
        except importlib_metadata.PackageNotFoundError:
            versions[distribution] = "not-installed"
    return versions


def _code_hash() -> str:
    """Hash Research v2 source without importing any model module."""

    digest = sha256()
    files = sorted(MODULE_DIR.glob("*.py"), key=lambda item: item.name)
    if not files:
        raise CLIError(f"no Research v2 source files found under {MODULE_DIR}")
    for path in files:
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(_sha256_file(path)))
    return digest.hexdigest()


def _snapshot_data_hash(metadata: Mapping[str, Any]) -> str:
    files = []
    for raw in metadata.get("files", []):
        files.append(
            {
                "relative_path": str(raw["relative_path"]),
                "size": int(raw["size"]),
                "sha256": str(raw["sha256"]),
            }
        )
    manifest = metadata.get("manifest", {})
    payload = {
        "format_version": metadata.get("format_version"),
        "snapshot_id": metadata.get("snapshot_id"),
        "manifest_sha256": manifest.get("sha256"),
        "files": sorted(files, key=lambda item: item["relative_path"]),
    }
    return _sha256_bytes(_canonical_json(payload))


def _resolve_snapshot(root: Path, token: str) -> tuple[Path, Mapping[str, Any]]:
    from .snapshot import verify_snapshot

    snapshots_dir = _safe_output("snapshots", root)
    candidate_token = str(token)
    raw = Path(candidate_token).expanduser()
    if raw.is_absolute():
        candidate = _safe_output(raw, root)
    elif candidate_token == DEFAULT_SNAPSHOT:
        literal = _safe_output(snapshots_dir / DEFAULT_SNAPSHOT, root)
        if literal.is_dir():
            candidate = literal
        else:
            matches = sorted(
                (
                    path
                    for path in snapshots_dir.glob("canonical-*")
                    if path.is_dir() and not path.name.startswith(".")
                ),
                key=lambda item: item.name,
            )
            if not matches:
                raise CLIError(
                    "canonical snapshot is missing; run `python -m research_v2.cli "
                    "snapshot` first or pass --snapshot"
                )
            candidate = _safe_output(matches[-1], root)
    elif len(raw.parts) > 1:
        # A relative path is interpreted from the current working directory;
        # verification below still requires it to be a direct child of this
        # research root's snapshots directory.
        candidate = _safe_output(raw.resolve(strict=False), root)
    else:
        _validate_id(candidate_token, "snapshot id")
        candidate = _safe_output(snapshots_dir / candidate_token, root)
    metadata = verify_snapshot(candidate, research_root=root)
    return candidate, metadata


def _resolve_config(root: Path, token: str) -> tuple[Path | None, Any, str | None]:
    from .config import load_config

    raw = Path(token).expanduser()
    if raw.is_absolute():
        path = raw.resolve(strict=False)
    else:
        candidates = (root / raw, Path.cwd() / raw, MODULE_DIR / raw)
        path = next((candidate for candidate in candidates if candidate.is_file()), candidates[0])
        path = path.resolve(strict=False)
    if not path.is_file():
        raise CLIError(f"research config is missing: {path}")
    config = load_config(path)
    return path, config, _sha256_file(path)


def _seed_manifest(config: Any) -> dict[str, Any]:
    base = int(config.models.random_seed)
    return {
        "base_random_seed": base,
        "tabular_fold_seed_formula": "base + fold_number * 100 + trial_number",
        "tabular_refit_seed_formula": "base + fold_number",
        "sequence_fold_seed_formula": "base + fold_number * 1000",
        "sequence_lockbox_seed_formula": "base + 900000",
        "torch_deterministic": True,
        "python_hash_seed": os.environ.get("PYTHONHASHSEED", "not-set-before-process-start"),
    }


def _run_identity(context: RunContext) -> dict[str, Any]:
    return {
        "run_id": context.run_id,
        "snapshot_id": context.snapshot_metadata["snapshot_id"],
        "snapshot_data_sha256": context.snapshot_data_sha256,
        "config_sha256": context.config_sha256,
    }


def _prepare_context(args: argparse.Namespace, *, require_existing: bool = False) -> RunContext:
    root = _resolved_research_root(args.research_root)
    snapshot_dir, snapshot_metadata = _resolve_snapshot(root, args.snapshot)
    config_path, config, config_file_hash = _resolve_config(root, args.config)
    config_hash = str(config.fingerprint())
    data_hash = _snapshot_data_hash(snapshot_metadata)
    run_id = _validate_id(str(args.run_id), "run id")
    run_dir = _safe_output(Path("runs") / run_id, root)
    code_hash = _code_hash()
    versions = _dependency_versions()
    seeds = _seed_manifest(config)
    context = RunContext(
        research_root=root,
        snapshot_dir=snapshot_dir,
        snapshot_metadata=snapshot_metadata,
        snapshot_data_sha256=data_hash,
        config_path=config_path,
        config_file_sha256=config_file_hash,
        config=config,
        config_sha256=config_hash,
        run_id=run_id,
        run_dir=run_dir,
        code_sha256=code_hash,
        versions=versions,
        seeds=seeds,
    )

    manifest_path = _safe_output(run_dir / "run.json", root)
    config_copy = _safe_output(run_dir / "config.json", root)
    if manifest_path.is_file():
        try:
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise ArtifactVerificationError(f"invalid run manifest: {manifest_path}") from exc
        if existing.get("format_version") != RUN_FORMAT_VERSION:
            raise ArtifactVerificationError("unsupported run manifest format")
        expected = _run_identity(context)
        mismatched = {
            key: (existing.get(key), value)
            for key, value in expected.items()
            if existing.get(key) != value
        }
        if mismatched:
            raise ArtifactVerificationError(
                f"run id {run_id!r} belongs to different immutable inputs: {mismatched}"
            )
        if not config_copy.is_file() or _sha256_bytes(config_copy.read_bytes()) != _sha256_bytes(
            _json_bytes(config.as_dict())
        ):
            raise ArtifactVerificationError("run config copy is missing or modified")
        return context

    if require_existing:
        raise ArtifactVerificationError(f"run does not exist: {run_dir}")
    if run_dir.exists() and any(run_dir.iterdir()):
        raise ArtifactVerificationError(
            f"non-empty run directory has no run manifest: {run_dir}"
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "format_version": RUN_FORMAT_VERSION,
        **_run_identity(context),
        "created_at_utc": _utc_now(),
        "config_source": str(config_path) if config_path is not None else None,
        "config_file_sha256": config_file_hash,
        "config": config.as_dict(),
        "code_sha256_at_run_creation": code_hash,
        "versions_at_run_creation": versions,
        "seeds": seeds,
        "offline_only": True,
    }
    _atomic_write_json(config_copy, config.as_dict(), root=root)
    _atomic_write_json(manifest_path, manifest, root=root)
    return context


def _relative_files(directory: Path) -> list[Path]:
    files: list[Path] = []
    for path in sorted(directory.rglob("*"), key=lambda item: item.as_posix()):
        if path.is_symlink():
            raise ArtifactVerificationError(f"artifact may not contain symlinks: {path}")
        if path.is_file() and path.name != SUCCESS_FILE:
            files.append(path)
    return files


def _file_manifest(directory: Path) -> dict[str, dict[str, Any]]:
    return {
        path.relative_to(directory).as_posix(): {
            "size": int(path.stat().st_size),
            "sha256": _sha256_file(path),
        }
        for path in _relative_files(directory)
    }


def verify_stage(
    directory: Path | str,
    *,
    research_root: Path | str,
    expected: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    """Re-hash a completed stage and return its success manifest."""

    root = _resolved_research_root(research_root)
    stage_dir = _safe_output(directory, root)
    marker_path = _safe_output(stage_dir / SUCCESS_FILE, root)
    if not stage_dir.is_dir() or not marker_path.is_file():
        raise ArtifactVerificationError(f"stage is incomplete or missing: {stage_dir}")
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ArtifactVerificationError(f"invalid success marker: {marker_path}") from exc
    if marker.get("format_version") != ARTIFACT_FORMAT_VERSION:
        raise ArtifactVerificationError(f"unsupported artifact format: {stage_dir}")
    expected_stage = str(expected.get("stage")) if expected and "stage" in expected else stage_dir.name
    if marker.get("status") != "complete" or marker.get("stage") != expected_stage:
        raise ArtifactVerificationError(f"invalid stage identity/status: {stage_dir}")
    if expected:
        mismatch = {
            key: (marker.get(key), value)
            for key, value in expected.items()
            if marker.get(key) != value
        }
        if mismatch:
            raise ArtifactVerificationError(
                f"stage {stage_dir.name!r} belongs to different inputs/parameters: {mismatch}"
            )
    parameters = marker.get("parameters")
    inputs = marker.get("inputs")
    if not isinstance(parameters, dict) or marker.get("parameters_sha256") != _parameters_hash(
        parameters
    ):
        raise ArtifactVerificationError(f"stage parameter manifest changed: {stage_dir}")
    if not isinstance(inputs, dict) or marker.get("inputs_sha256") != _parameters_hash(inputs):
        raise ArtifactVerificationError(f"stage input manifest changed: {stage_dir}")
    recorded = marker.get("files")
    if not isinstance(recorded, dict) or not recorded:
        raise ArtifactVerificationError(f"stage records no output files: {stage_dir}")
    actual = _file_manifest(stage_dir)
    if set(recorded) != set(actual):
        raise ArtifactVerificationError(
            f"artifact file set changed for {stage_dir}; "
            f"added={sorted(set(actual) - set(recorded))[:8]}, "
            f"removed={sorted(set(recorded) - set(actual))[:8]}"
        )
    changed = [name for name in actual if actual[name] != recorded[name]]
    if changed:
        raise ArtifactVerificationError(
            f"artifact files changed for {stage_dir}: {changed[:8]}"
        )
    return marker


def _parameters_hash(parameters: Mapping[str, Any]) -> str:
    return _sha256_bytes(_canonical_json(parameters))


def _stage_expected(
    context: RunContext,
    stage: str,
    parameters: Mapping[str, Any],
    inputs: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    return {
        "stage": stage,
        **_run_identity(context),
        "code_sha256": context.code_sha256,
        "parameters_sha256": _parameters_hash(parameters),
        "inputs_sha256": _parameters_hash(dict(sorted((inputs or {}).items()))),
    }


def _remove_private_staging(staging: Path, destination: Path, root: Path) -> None:
    checked = _safe_output(staging, root)
    if checked.parent != destination.parent or not checked.name.startswith(
        f".{destination.name}.partial-"
    ):
        raise CLIError(f"refusing to remove non-staging path: {checked}")
    if checked.exists():
        shutil.rmtree(checked)


def _publish_stage(
    context: RunContext,
    stage: str,
    *,
    parameters: Mapping[str, Any],
    inputs: Mapping[str, str],
    resume: bool,
    writer: Callable[[Path], Mapping[str, Any] | None],
) -> dict[str, Any]:
    _validate_id(stage, "stage")
    destination = _safe_output(context.run_dir / stage, context.research_root)
    expected = _stage_expected(context, stage, parameters, inputs)
    if destination.exists():
        marker = verify_stage(
            destination,
            research_root=context.research_root,
            expected=expected,
        )
        if not resume:
            raise CLIError(
                f"verified stage already exists and --no-resume was requested: {destination}"
            )
        return {
            "stage": stage,
            "status": "skipped_verified",
            "path": str(destination),
            "artifact_sha256": _sha256_file(destination / SUCCESS_FILE),
            "files": len(marker["files"]),
        }

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = _safe_output(
        destination.parent / f".{destination.name}.partial-{uuid.uuid4().hex}",
        context.research_root,
    )
    code_before = _code_hash()
    if code_before != context.code_sha256:
        raise CLIError(
            "Research v2 source changed after the run context was prepared; "
            "start a fresh command so provenance can be recomputed"
        )
    staging.mkdir(parents=False, exist_ok=False)
    try:
        details = dict(writer(staging) or {})
        files = _file_manifest(staging)
        if not files:
            raise CLIError(f"stage {stage!r} produced no files")
        code_after = _code_hash()
        if code_after != code_before:
            raise CLIError(
                f"Research v2 source changed while stage {stage!r} was running; "
                "the unpublished result was discarded"
            )
        marker = {
            "format_version": ARTIFACT_FORMAT_VERSION,
            "status": "complete",
            **expected,
            "completed_at_utc": _utc_now(),
            "parameters": dict(parameters),
            "inputs": dict(sorted(inputs.items())),
            "versions": dict(context.versions),
            "seeds": dict(context.seeds),
            "details": details,
            "files": files,
        }
        _atomic_write_json(staging / SUCCESS_FILE, marker, root=context.research_root)
        verify_stage(
            staging,
            research_root=context.research_root,
            expected=expected,
        )
        if destination.exists():
            raise CLIError(f"stage appeared concurrently: {destination}")
        os.rename(staging, destination)
        final = verify_stage(
            destination,
            research_root=context.research_root,
            expected=expected,
        )
        return {
            "stage": stage,
            "status": "completed",
            "path": str(destination),
            "artifact_sha256": _sha256_file(destination / SUCCESS_FILE),
            "files": len(final["files"]),
        }
    finally:
        if staging.exists():
            _remove_private_staging(staging, destination, context.research_root)


def _feature_parameters(context: RunContext) -> dict[str, Any]:
    return dict(context.config.data.__dict__)


def _ensure_features(context: RunContext, *, resume: bool) -> dict[str, Any]:
    parameters = _feature_parameters(context)

    def write(directory: Path) -> Mapping[str, Any]:
        from .features import build_feature_panel, load_snapshot, write_feature_artifacts

        raw = load_snapshot(context.snapshot_dir)
        result = build_feature_panel(
            raw,
            start_date=context.config.data.start_date,
            end_date=context.config.data.end_date,
            min_cross_section=context.config.data.min_cross_section,
            min_symbol_history=context.config.data.min_symbol_history,
            label_horizon=context.config.data.label_horizon,
            invalid_row_policy=context.config.data.invalid_row_policy,
        )
        panel_path = _safe_output(directory / "panel.parquet", context.research_root)
        report_path = _safe_output(directory / "report.json", context.research_root)
        write_feature_artifacts(result, panel_path, report_path)
        _atomic_write_json(
            directory / "feature_columns.json",
            list(result.feature_columns),
            root=context.research_root,
        )
        return {
            "rows": int(result.panel.height),
            "features": len(result.feature_columns),
            "eligible_rows": int(result.report.get("eligible_rows", 0)),
        }

    return _publish_stage(
        context,
        "features",
        parameters=parameters,
        inputs={"snapshot_data_sha256": context.snapshot_data_sha256},
        resume=resume,
        writer=write,
    )


def _load_feature_panel(context: RunContext):
    import polars as pl

    directory = _safe_output(context.run_dir / "features", context.research_root)
    marker = verify_stage(
        directory,
        research_root=context.research_root,
        expected=_stage_expected(
            context,
            "features",
            _feature_parameters(context),
            {"snapshot_data_sha256": context.snapshot_data_sha256},
        ),
    )
    panel = pl.read_parquet(directory / "panel.parquet")
    try:
        columns = tuple(
            json.loads((directory / "feature_columns.json").read_text(encoding="utf-8"))
        )
    except Exception as exc:
        raise ArtifactVerificationError("invalid feature_columns.json") from exc
    if not columns or len(set(columns)) != len(columns):
        raise ArtifactVerificationError("feature column list is empty or duplicated")
    return panel, columns, marker


def _complete_case_panel(context: RunContext):
    import polars as pl

    from .experiment import complete_case_symbols

    panel, feature_columns, marker = _load_feature_panel(context)
    symbols = complete_case_symbols(panel)
    filtered = panel.filter(pl.col("symbol").cast(pl.Utf8).is_in(list(symbols)))
    return filtered, feature_columns, symbols, marker


def _tabular_model_frame(panel: Any, feature_columns: Sequence[str]):
    import polars as pl

    from .features import BASELINE_SCORE_COLUMNS

    columns = [
        "timestamp",
        "execution_timestamp",
        "symbol",
        "label_rank",
        "label_residual",
        "sample_weight",
        "model_eligible",
        *feature_columns,
        *BASELINE_SCORE_COLUMNS,
    ]
    missing = [name for name in columns if name not in panel.columns]
    if missing:
        raise ArtifactVerificationError(f"feature panel lacks model columns: {missing}")
    return (
        panel.filter(pl.col("model_eligible"))
        .select(columns)
        .to_pandas()
        .sort_values(["timestamp", "symbol"], kind="mergesort")
        .reset_index(drop=True)
    )


def _walk_forward_folds(frame: Any, context: RunContext):
    from .validation import make_purged_walk_forward

    cfg = context.config.walk_forward
    dates = frame["timestamp"].drop_duplicates().sort_values()
    return make_purged_walk_forward(
        dates,
        train_days=cfg.train_days,
        validation_days=cfg.validation_days,
        test_days=cfg.test_days,
        purge_days=cfg.purge_days,
        embargo_days=cfg.embargo_days,
        step_days=cfg.step_days,
        label_horizon=context.config.data.label_horizon,
        rolling_train=cfg.rolling_train,
        selection_end=cfg.selection_end,
    )


def _tabular_parameters(context: RunContext) -> dict[str, Any]:
    return {
        "walk_forward": dict(context.config.walk_forward.__dict__),
        "models": {
            "random_seed": context.config.models.random_seed,
            "ridge_alphas": list(context.config.models.ridge_alphas),
            "gbdt_grid": list(context.config.models.gbdt_grid),
            "ensemble_shrinkage": context.config.models.ensemble_shrinkage,
            "ensemble_single_model_cap": context.config.models.ensemble_single_model_cap,
        },
        "universe": "complete-case snapshot symbols",
    }


def _diagnostics(frame: Any, columns: Sequence[str]) -> dict[str, Any]:
    from .validation import prediction_diagnostics

    output: dict[str, Any] = {}
    for name in columns:
        if name not in frame.columns:
            continue
        required = ["timestamp", "label_rank", "label_residual", name]
        output[name] = prediction_diagnostics(
            frame.loc[:, required], prediction_col=name
        )
    return output


def _ensure_tabular(context: RunContext, *, resume: bool) -> dict[str, Any]:
    feature_status = _ensure_features(context, resume=resume)
    parameters = _tabular_parameters(context)
    feature_marker = context.run_dir / "features" / SUCCESS_FILE

    def write(directory: Path) -> Mapping[str, Any]:
        from .ml_pipeline import run_tabular_research, write_ml_artifacts

        panel, feature_columns, symbols, _ = _complete_case_panel(context)
        frame = _tabular_model_frame(panel, feature_columns)
        folds = _walk_forward_folds(frame, context)
        cfg = context.config
        result = run_tabular_research(
            frame,
            feature_columns,
            folds,
            ridge_alphas=cfg.models.ridge_alphas,
            gbdt_grid=cfg.models.gbdt_grid,
            random_seed=cfg.models.random_seed,
            selection_end=cfg.walk_forward.selection_end,
            lockbox_start=cfg.walk_forward.lockbox_start,
            embargo_days=cfg.walk_forward.embargo_days,
            ensemble_shrinkage=cfg.models.ensemble_shrinkage,
            ensemble_single_model_cap=cfg.models.ensemble_single_model_cap,
        )
        write_ml_artifacts(result, directory)
        # Re-emit JSON through the CLI's strict sanitizer/atomic writer so a
        # rare undefined diagnostic becomes null rather than non-standard NaN.
        _atomic_write_json(
            directory / "fold_records.json",
            list(result.fold_records),
            root=context.research_root,
        )
        _atomic_write_json(
            directory / "locked_settings.json",
            result.locked_settings,
            root=context.research_root,
        )
        _atomic_write_json(
            directory / "folds.json",
            [fold.as_dict() for fold in folds],
            root=context.research_root,
        )
        _atomic_write_json(
            directory / "feature_columns.json",
            list(feature_columns),
            root=context.research_root,
        )
        score_columns = [
            name
            for name in result.selection_predictions.columns
            if name.startswith("score_")
        ]
        summary = {
            "complete_case_symbols": len(symbols),
            "feature_count": len(feature_columns),
            "fold_count": len(folds),
            "selection_rows": len(result.selection_predictions),
            "lockbox_rows": len(result.lockbox_predictions),
            "locked_settings": result.locked_settings,
            "selection_diagnostics": _diagnostics(
                result.selection_predictions, score_columns
            ),
            "lockbox_diagnostics": _diagnostics(
                result.lockbox_predictions, score_columns
            ),
        }
        _atomic_write_json(directory / "summary.json", summary, root=context.research_root)
        return {
            "complete_case_symbols": len(symbols),
            "folds": len(folds),
            "selection_rows": len(result.selection_predictions),
            "lockbox_rows": len(result.lockbox_predictions),
        }

    result = _publish_stage(
        context,
        "tabular",
        parameters=parameters,
        inputs={
            "features_success_sha256": _sha256_file(feature_marker),
        },
        resume=resume,
        writer=write,
    )
    result["prerequisite_features"] = feature_status["status"]
    return result


def _sequence_parameters(context: RunContext, device: str) -> dict[str, Any]:
    cfg = context.config.models
    return {
        "walk_forward": dict(context.config.walk_forward.__dict__),
        "sequence": {
            "sequence_length": cfg.sequence_length,
            "hidden_size": cfg.sequence_hidden_size,
            "epochs": cfg.sequence_epochs,
            "batch_size": cfg.sequence_batch_size,
            "max_train_samples": cfg.sequence_max_train_samples,
            "random_seed": cfg.random_seed,
            "ensemble_shrinkage": cfg.ensemble_shrinkage,
            "ensemble_single_model_cap": cfg.ensemble_single_model_cap,
            "device": device,
        },
        "universe": "complete-case snapshot symbols",
    }


def _ensure_sequence(
    context: RunContext,
    *,
    resume: bool,
    device: str,
) -> dict[str, Any]:
    feature_status = _ensure_features(context, resume=resume)
    parameters = _sequence_parameters(context, device)
    feature_marker = context.run_dir / "features" / SUCCESS_FILE

    def write(directory: Path) -> Mapping[str, Any]:
        from dataclasses import asdict

        from .sequence_pipeline import SequencePipelineSettings, run_sequence_research

        panel, feature_columns, symbols, _ = _complete_case_panel(context)
        # Sequence construction needs warm-up rows, so retain the full panel;
        # endpoint eligibility is governed by finite point-in-time labels.
        frame = panel.select(
            [
                "timestamp",
                "execution_timestamp",
                "symbol",
                "label_rank",
                "label_residual",
                "sample_weight",
                *feature_columns,
            ]
        ).to_pandas()
        folds = _walk_forward_folds(_tabular_model_frame(panel, feature_columns), context)
        model_cfg = context.config.models
        settings = SequencePipelineSettings(
            sequence_length=model_cfg.sequence_length,
            gru_hidden_dim=model_cfg.sequence_hidden_size,
            transformer_d_model=model_cfg.sequence_hidden_size,
            transformer_feedforward=model_cfg.sequence_hidden_size * 2,
            epochs=model_cfg.sequence_epochs,
            batch_size=model_cfg.sequence_batch_size,
            max_train_samples=model_cfg.sequence_max_train_samples,
            random_seed=model_cfg.random_seed,
            device=device,
            ensemble_shrinkage=model_cfg.ensemble_shrinkage,
            ensemble_single_model_cap=model_cfg.ensemble_single_model_cap,
        )
        result = run_sequence_research(
            frame,
            feature_columns,
            folds,
            selection_end=context.config.walk_forward.selection_end,
            lockbox_start=context.config.walk_forward.lockbox_start,
            embargo_days=context.config.walk_forward.embargo_days,
            settings=settings,
        )
        result.selection_predictions.to_parquet(
            directory / "selection_oos_predictions.parquet", index=False
        )
        result.lockbox_predictions.to_parquet(
            directory / "lockbox_predictions.parquet", index=False
        )
        _atomic_write_json(
            directory / "fold_records.json",
            list(result.fold_records),
            root=context.research_root,
        )
        _atomic_write_json(
            directory / "locked_settings.json",
            result.locked_settings,
            root=context.research_root,
        )
        _atomic_write_json(
            directory / "settings.json", asdict(settings), root=context.research_root
        )
        _atomic_write_json(
            directory / "folds.json",
            [fold.as_dict() for fold in folds],
            root=context.research_root,
        )
        _atomic_write_json(
            directory / "feature_columns.json",
            list(feature_columns),
            root=context.research_root,
        )
        scores = [
            name
            for name in result.selection_predictions.columns
            if name.startswith("score_")
        ]
        summary = {
            "complete_case_symbols": len(symbols),
            "feature_count": len(feature_columns),
            "fold_count": len(folds),
            "selection_rows": len(result.selection_predictions),
            "lockbox_rows": len(result.lockbox_predictions),
            "locked_settings": result.locked_settings,
            "selection_diagnostics": _diagnostics(
                result.selection_predictions, scores
            ),
            "lockbox_diagnostics": _diagnostics(result.lockbox_predictions, scores),
        }
        _atomic_write_json(directory / "summary.json", summary, root=context.research_root)
        return {
            "complete_case_symbols": len(symbols),
            "folds": len(folds),
            "selection_rows": len(result.selection_predictions),
            "lockbox_rows": len(result.lockbox_predictions),
            "device": device,
        }

    result = _publish_stage(
        context,
        "sequence",
        parameters=parameters,
        inputs={"features_success_sha256": _sha256_file(feature_marker)},
        resume=resume,
        writer=write,
    )
    result["prerequisite_features"] = feature_status["status"]
    return result


def _engine_configs(context: RunContext):
    from .backtest import RiskConfig
    from .costs import CostConfig
    from .portfolio import PortfolioConfig

    p = context.config.portfolio
    c = context.config.costs
    r = context.config.risk
    portfolio = PortfolioConfig(
        top_n=p.top_n,
        weighting=p.weighting,
        gross_target=p.leverage,
        single_name_cap=p.max_single_weight,
        sector_cap=p.max_sector_weight,
        rank_buffer=p.rank_buffer,
        no_trade_band=p.minimum_weight_change,
        staggered_tranches=p.staggered_tranches,
        max_adv_participation=c.max_adv_participation,
    )
    # Fixed slippage is represented as commission-like friction so it is added
    # on every traded dollar instead of incorrectly replacing the spread proxy.
    cost = CostConfig(
        commission_bps=c.commission_bps + c.fixed_slippage_bps,
        spread_multiplier=1.0,
        min_spread_bps=0.0,
        max_spread_bps=c.max_half_spread_bps,
        impact_coefficient=c.impact_coefficient,
        max_impact_bps=c.max_impact_bps,
        max_adv_participation=c.max_adv_participation,
        annual_funding_rate=c.annual_funding_rate,
    )
    risk = RiskConfig(
        target_volatility=r.target_volatility,
        vol_lookback=r.volatility_lookback,
        min_vol_observations=min(20, r.volatility_lookback),
        drawdown_steps=(
            (abs(r.drawdown_level_1), r.drawdown_multiplier_1),
            (abs(r.drawdown_level_2), r.drawdown_multiplier_2),
        ),
        max_abs_beta=r.beta_cap,
        breadth_exit=r.breadth_exit_below,
        breadth_enter=r.breadth_reenter_above,
        risk_off_multiplier=r.risk_off_multiplier,
        crowding_threshold=r.crowding_correlation_limit,
        crowding_multiplier=r.crowding_multiplier,
        target_change_buffer=p.minimum_weight_change,
    )
    return portfolio, cost, risk


def _search_parameters(context: RunContext, device: str) -> dict[str, Any]:
    search = context.config.search
    cadences = [
        [int(days), 1] for days in search.rebalance_grid
    ] + [
        [1, int(days)] for days in search.rebalance_grid if int(days) > 1
    ]
    return {
        "search": dict(search.__dict__),
        "portfolio": dict(context.config.portfolio.__dict__),
        "costs": dict(context.config.costs.__dict__),
        "risk": dict(context.config.risk.__dict__),
        "cadence_grid": cadences,
        "selection_cost_bps": 10.0,
        "requires": ["tabular", "sequence"],
        "sequence_device": device,
    }


def _merge_predictions(tabular: Path, sequence: Path):
    import pandas as pd

    tab = pd.read_parquet(tabular)
    seq = pd.read_parquet(sequence)
    sequence_scores = [name for name in seq.columns if name.startswith("score_")]
    right = seq.loc[:, ["timestamp", "symbol", *sequence_scores]].copy()
    if right.duplicated(["timestamp", "symbol"]).any():
        raise ArtifactVerificationError("sequence predictions contain duplicate keys")
    merged = tab.merge(
        right,
        on=["timestamp", "symbol"],
        how="inner",
        validate="one_to_one",
        suffixes=("", "_sequence_duplicate"),
    )
    duplicate_scores = [name for name in merged if name.endswith("_sequence_duplicate")]
    if duplicate_scores:
        raise ArtifactVerificationError(
            f"tabular/sequence score names overlap: {duplicate_scores}"
        )
    if merged.empty:
        raise ArtifactVerificationError("tabular and sequence predictions do not overlap")
    return merged.sort_values(["timestamp", "symbol"], kind="mergesort").reset_index(
        drop=True
    )


def _ensure_search(
    context: RunContext,
    *,
    resume: bool,
    device: str,
) -> dict[str, Any]:
    tabular_status = _ensure_tabular(context, resume=resume)
    sequence_status = _ensure_sequence(context, resume=resume, device=device)
    parameters = _search_parameters(context, device)
    tabular_marker = context.run_dir / "tabular" / SUCCESS_FILE
    sequence_marker = context.run_dir / "sequence" / SUCCESS_FILE

    def write(directory: Path) -> Mapping[str, Any]:
        from .experiment import build_market_context
        from .search import Cadence, SearchPolicy, run_staged_search, write_search_artifacts

        panel, _, symbols, _ = _complete_case_panel(context)
        selection = _merge_predictions(
            context.run_dir / "tabular" / "selection_oos_predictions.parquet",
            context.run_dir / "sequence" / "selection_oos_predictions.parquet",
        )
        lockbox = _merge_predictions(
            context.run_dir / "tabular" / "lockbox_predictions.parquet",
            context.run_dir / "sequence" / "lockbox_predictions.parquet",
        )
        score_columns = sorted(
            name
            for name in selection.columns
            if name.startswith("score_")
            and name in lockbox.columns
            and selection[name].notna().all()
            and lockbox[name].notna().all()
        )
        if not score_columns:
            raise ArtifactVerificationError("no complete common score columns for search")
        start = min(selection["timestamp"].min(), lockbox["timestamp"].min())
        end = panel["timestamp"].max()
        context_data = build_market_context(
            panel,
            symbols=symbols,
            start=start,
            end=end,
            beta_lookback=context.config.risk.beta_lookback,
        )
        base_portfolio, base_cost, base_risk = _engine_configs(context)
        search_cfg = context.config.search
        cadences = tuple(
            [Cadence(f"every_{days}d", int(days), 1) for days in search_cfg.rebalance_grid]
            + [
                Cadence(f"daily_{days}_tranches", 1, int(days))
                for days in search_cfg.rebalance_grid
                if int(days) > 1
            ]
        )
        policy = SearchPolicy(
            require_positive_worst_fold=search_cfg.require_positive_worst_fold,
            max_drawdown_limit=search_cfg.max_drawdown_limit,
        )
        result = run_staged_search(
            context_data,
            selection,
            lockbox,
            score_columns=score_columns,
            base_portfolio=base_portfolio,
            base_cost=base_cost,
            base_risk=base_risk,
            top_n_grid=search_cfg.top_n_grid,
            cadence_grid=cadences,
            weighting_grid=search_cfg.weighting_grid,
            leverage_grid=search_cfg.leverage_grid,
            cost_sensitivity_bps=search_cfg.cost_sensitivity_bps,
            selection_cost_bps=10.0,
            base_rebalance_days=context.config.portfolio.rebalance_days,
            policy=policy,
            initial_capital=search_cfg.initial_capital,
        )
        write_search_artifacts(
            result,
            directory,
            research_root=context.research_root,
        )
        _atomic_write_json(
            directory / "search_result.json", result.to_dict(), root=context.research_root
        )
        return {
            "selection_rows": len(selection),
            "lockbox_rows": len(lockbox),
            "score_columns": score_columns,
            "complete_case_symbols": len(symbols),
            "champion_candidate_id": result.champion.candidate.candidate_id,
        }

    result = _publish_stage(
        context,
        "search",
        parameters=parameters,
        inputs={
            "tabular_success_sha256": _sha256_file(tabular_marker),
            "sequence_success_sha256": _sha256_file(sequence_marker),
        },
        resume=resume,
        writer=write,
    )
    result["prerequisite_tabular"] = tabular_status["status"]
    result["prerequisite_sequence"] = sequence_status["status"]
    return result


def _ensure_all_marker(
    context: RunContext,
    *,
    resume: bool,
    device: str,
) -> dict[str, Any]:
    parameters = {"pipeline": ["features", "tabular", "sequence", "search"], "device": device}
    markers = {
        stage: context.run_dir / stage / SUCCESS_FILE
        for stage in parameters["pipeline"]
    }

    def write(directory: Path) -> Mapping[str, Any]:
        payload = {
            "run_id": context.run_id,
            "snapshot_data_sha256": context.snapshot_data_sha256,
            "config_sha256": context.config_sha256,
            "stages": {
                stage: _sha256_file(marker) for stage, marker in markers.items()
            },
        }
        _atomic_write_json(directory / "pipeline.json", payload, root=context.research_root)
        return {"stages": list(markers)}

    return _publish_stage(
        context,
        "all",
        parameters=parameters,
        inputs={f"{stage}_success_sha256": _sha256_file(path) for stage, path in markers.items()},
        resume=resume,
        writer=write,
    )


def _snapshot_command(args: argparse.Namespace) -> Mapping[str, Any]:
    from .snapshot import create_snapshot, verify_snapshot

    root = _resolved_research_root(args.research_root)
    snapshots_dir = _safe_output("snapshots", root)
    snapshot_id = _validate_id(args.snapshot_id, "snapshot id")
    destination = _safe_output(snapshots_dir / snapshot_id, root)
    if destination.exists():
        metadata = verify_snapshot(destination, research_root=root)
        if not args.resume:
            raise CLIError(
                f"verified snapshot already exists and --no-resume was requested: {destination}"
            )
        return {
            "command": "snapshot",
            "status": "skipped_verified",
            "path": str(destination),
            "snapshot_id": metadata["snapshot_id"],
            "data_sha256": _snapshot_data_hash(metadata),
        }
    keyword: dict[str, Any] = {
        "research_root": root,
        "snapshot_id": snapshot_id,
    }
    if args.store_dir is not None:
        keyword["store_dir"] = args.store_dir
    if args.manifest_path is not None:
        keyword["manifest_path"] = args.manifest_path
    completed = create_snapshot(**keyword)
    metadata = verify_snapshot(completed, research_root=root)
    return {
        "command": "snapshot",
        "status": "completed",
        "path": str(completed),
        "snapshot_id": metadata["snapshot_id"],
        "data_sha256": _snapshot_data_hash(metadata),
    }


def _verify_command(args: argparse.Namespace) -> Mapping[str, Any]:
    selected = args.stage
    if selected == "snapshot":
        root = _resolved_research_root(args.research_root)
        _, metadata = _resolve_snapshot(root, args.snapshot)
        return {
            "command": "verify",
            "status": "verified",
            "snapshot_id": metadata["snapshot_id"],
            "data_sha256": _snapshot_data_hash(metadata),
        }
    context = _prepare_context(args, require_existing=True)
    if selected == "run":
        return {
            "command": "verify",
            "status": "verified",
            "run_id": context.run_id,
            "snapshot_id": context.snapshot_metadata["snapshot_id"],
            "config_sha256": context.config_sha256,
            "data_sha256": context.snapshot_data_sha256,
        }
    stages = ["features", "tabular", "sequence", "search", "all"] if selected == "all" else [selected]
    verified: dict[str, Any] = {}
    marker_cache: dict[str, Mapping[str, Any]] = {}

    def verify_with_dependencies(stage: str) -> Mapping[str, Any]:
        if stage in marker_cache:
            return marker_cache[stage]
        dependencies = {
            "features": (),
            "tabular": ("features",),
            "sequence": ("features",),
            "search": ("tabular", "sequence"),
            "all": ("features", "tabular", "sequence", "search"),
        }[stage]
        for name in dependencies:
            verify_with_dependencies(name)
        directory = _safe_output(context.run_dir / stage, context.research_root)
        marker = verify_stage(directory, research_root=context.research_root)
        identity = _run_identity(context)
        mismatch = {
            key: (marker.get(key), value)
            for key, value in identity.items()
            if marker.get(key) != value
        }
        if mismatch:
            raise ArtifactVerificationError(
                f"stage {stage!r} does not belong to this run: {mismatch}"
            )
        if stage == "features":
            expected_inputs = {"snapshot_data_sha256": context.snapshot_data_sha256}
        elif stage in {"tabular", "sequence"}:
            expected_inputs = {
                "features_success_sha256": _sha256_file(
                    context.run_dir / "features" / SUCCESS_FILE
                )
            }
        elif stage == "search":
            expected_inputs = {
                "tabular_success_sha256": _sha256_file(
                    context.run_dir / "tabular" / SUCCESS_FILE
                ),
                "sequence_success_sha256": _sha256_file(
                    context.run_dir / "sequence" / SUCCESS_FILE
                ),
            }
        else:
            expected_inputs = {
                f"{name}_success_sha256": _sha256_file(
                    context.run_dir / name / SUCCESS_FILE
                )
                for name in dependencies
            }
        if marker.get("inputs") != expected_inputs:
            raise ArtifactVerificationError(
                f"stage {stage!r} no longer points to the current verified upstream artifacts"
            )
        marker_cache[stage] = marker
        return marker

    for stage in stages:
        directory = _safe_output(context.run_dir / stage, context.research_root)
        marker = verify_with_dependencies(stage)
        verified[stage] = {
            "files": len(marker["files"]),
            "completed_at_utc": marker["completed_at_utc"],
            "success_sha256": _sha256_file(directory / SUCCESS_FILE),
        }
    return {
        "command": "verify",
        "status": "verified",
        "run_id": context.run_id,
        "snapshot_id": context.snapshot_metadata["snapshot_id"],
        "config_sha256": context.config_sha256,
        "data_sha256": context.snapshot_data_sha256,
        "stages": verified,
    }


def _dispatch(args: argparse.Namespace) -> Mapping[str, Any]:
    if os.environ.get("ANCSER_RESEARCH_OFFLINE") != "1":
        raise CLIError("research CLI must run inside offline_context")
    if args.command == "snapshot":
        return _snapshot_command(args)
    if args.command == "verify":
        return _verify_command(args)

    context = _prepare_context(args)
    resume = bool(args.resume)
    if args.command == "features":
        result = _ensure_features(context, resume=resume)
    elif args.command == "tabular":
        result = _ensure_tabular(context, resume=resume)
    elif args.command == "sequence":
        result = _ensure_sequence(context, resume=resume, device=args.device)
    elif args.command == "search":
        result = _ensure_search(context, resume=resume, device=args.device)
    elif args.command == "all":
        search = _ensure_search(context, resume=resume, device=args.device)
        result = _ensure_all_marker(context, resume=resume, device=args.device)
        result["prerequisite_search"] = search["status"]
    else:  # pragma: no cover - argparse enforces the choices
        raise CLIError(f"unknown command: {args.command}")
    return {
        "command": args.command,
        "run_id": context.run_id,
        "snapshot_id": context.snapshot_metadata["snapshot_id"],
        "config_sha256": context.config_sha256,
        "data_sha256": context.snapshot_data_sha256,
        **result,
    }


def _add_resume_flags(parser: argparse.ArgumentParser) -> None:
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--resume",
        dest="resume",
        action="store_true",
        default=True,
        help="verify and skip already-complete immutable artifacts (default)",
    )
    group.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="fail if the requested snapshot/stage already exists",
    )


def _add_run_arguments(parser: argparse.ArgumentParser, *, device: bool = False) -> None:
    parser.add_argument(
        "--research-root",
        default=str(DEFAULT_RESEARCH_ROOT),
        help="isolated directory named research_v2 (outputs cannot escape it)",
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="research JSON config")
    parser.add_argument(
        "--snapshot",
        default=DEFAULT_SNAPSHOT,
        help="verified snapshot id/path; canonical resolves the latest canonical-* snapshot",
    )
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID, help="immutable research run id")
    if device:
        parser.add_argument(
            "--device",
            default="auto",
            help="sequence device: auto, cpu, cuda, cuda:0, ...",
        )
    _add_resume_flags(parser)


def build_parser() -> argparse.ArgumentParser:
    """Build the parser without reading config, data, environment, or disk."""

    parser = argparse.ArgumentParser(
        prog="python -m research_v2.cli",
        description="Offline, resumable and auditable ancserAPX Research v2 pipeline",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    snapshot = commands.add_parser("snapshot", help="create/verify a physical data snapshot")
    snapshot.add_argument("--research-root", default=str(DEFAULT_RESEARCH_ROOT))
    snapshot.add_argument("--snapshot-id", default=DEFAULT_SNAPSHOT)
    snapshot.add_argument("--store-dir", default=None)
    snapshot.add_argument("--manifest-path", default=None)
    _add_resume_flags(snapshot)

    features = commands.add_parser("features", help="build point-in-time features and labels")
    _add_run_arguments(features)

    tabular = commands.add_parser("tabular", help="train Ridge/GBDT/rank ensemble walk-forward")
    _add_run_arguments(tabular)

    sequence = commands.add_parser("sequence", help="train GRU/Transformer walk-forward")
    _add_run_arguments(sequence, device=True)

    search = commands.add_parser("search", help="run frozen-score net-cost portfolio search")
    _add_run_arguments(search, device=True)

    all_command = commands.add_parser("all", help="run every stage with verified resume")
    _add_run_arguments(all_command, device=True)

    verify = commands.add_parser("verify", help="re-hash immutable snapshot/run artifacts")
    _add_run_arguments(verify)
    verify.add_argument(
        "--stage",
        choices=("snapshot", "run", "features", "tabular", "sequence", "search", "all"),
        default="all",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Execute one command.  Returns a process exit status for tests/scripts."""

    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    from .safety import offline_context

    try:
        with offline_context():
            result = _dispatch(args)
    except Exception as exc:
        print(f"research_v2: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True, default=str, allow_nan=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
