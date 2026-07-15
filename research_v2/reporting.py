"""Auditable final reports for isolated Research v2 runs.

The reporter is intentionally a read-only consumer of completed research
artifacts.  It does not import production modules, train a model, run a
backtest, or change a live configuration.  The only writes are the final JSON
and Markdown files, both confined to ``research_v2/runs``.

Three evidence layers are kept separate throughout the report:

* selection OOS signal diagnostics used for model/configuration selection;
* an already-opened confirmation period, which must not be reused as a fresh
  lockbox after its results have been inspected;
* net-of-cost portfolio results produced by the staged search.

Missing artifacts are reported as unavailable rather than silently treated as
successful experiments.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from numbers import Integral, Real
import os
from pathlib import Path
from typing import Any

from .safety import RESEARCH_ROOT, UnsafeResearchPath, ensure_research_output_path


REPORT_SCHEMA_VERSION = "1.0"
EXPECTED_COST_SENSITIVITY_BPS = (0.0, 5.0, 10.0, 20.0)


class ResearchReportError(RuntimeError):
    """A required report input is malformed or unsafe to read."""


@dataclass(frozen=True)
class ReportPaths:
    """Files written by :func:`generate_research_report`."""

    json: Path
    markdown: Path


def _runs_root(research_root: str | Path) -> Path:
    root = ensure_research_output_path(research_root, research_root=research_root)
    return ensure_research_output_path(root / "runs", research_root=root)


def _ensure_runs_path(path: str | Path, *, research_root: str | Path) -> Path:
    """Require a resolved path to remain below ``research_v2/runs``."""

    root = ensure_research_output_path(research_root, research_root=research_root)
    runs = _runs_root(root)
    checked = ensure_research_output_path(path, research_root=root)
    if checked != runs and runs not in checked.parents:
        raise UnsafeResearchPath(f"research report path must be under {runs}: {checked}")
    return checked


def _safe_artifact(path: Path, *, run_dir: Path) -> Path:
    checked = path.expanduser().resolve(strict=False)
    if checked != run_dir and run_dir not in checked.parents:
        raise ResearchReportError(f"artifact escapes run directory {run_dir}: {checked}")
    return checked


def _load_json(path: Path, *, run_dir: Path, required: bool = False) -> Any:
    checked = _safe_artifact(path, run_dir=run_dir)
    if not checked.is_file():
        if required:
            raise ResearchReportError(f"required artifact is missing: {checked}")
        return None
    try:
        return json.loads(checked.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - exact decoder errors vary
        raise ResearchReportError(f"invalid JSON artifact: {checked}") from exc


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (datetime,)):
        return value.isoformat()
    return str(value)


def _as_mapping(value: Any, *, label: str) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ResearchReportError(f"{label} must contain a JSON object")
    return {str(key): item for key, item in value.items()}


def _optional_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _optional_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _summary_counts(summary: Mapping[str, Any] | None) -> dict[str, int | None]:
    source = summary or {}
    features = source.get("feature_count", source.get("features"))
    if isinstance(features, Sequence) and not isinstance(features, (str, bytes)):
        feature_count = len(features)
    else:
        feature_count = _optional_int(features)
    return {
        "symbols": _optional_int(
            source.get("complete_case_symbols", source.get("complete_symbols"))
        ),
        "features": feature_count,
        "folds": _optional_int(source.get("fold_count", source.get("folds"))),
        "selection_rows": _optional_int(source.get("selection_rows")),
        "confirmation_rows": _optional_int(source.get("lockbox_rows")),
    }


def _model_assessment(
    diagnostics: Mapping[str, Any] | None,
    *,
    source: str,
    period: str,
) -> list[dict[str, Any]]:
    if not diagnostics:
        return []
    output: list[dict[str, Any]] = []
    for model, raw in sorted(diagnostics.items()):
        if not isinstance(raw, Mapping):
            continue
        mean_ic = _optional_float(raw.get("mean_rank_ic"))
        nw_t = _optional_float(raw.get("rank_ic_nw_t"))
        if period == "selection_oos":
            if mean_ic is not None and nw_t is not None and mean_ic > 0 and nw_t >= 1.96:
                assessment = "effective"
                rationale = "positive selection-OOS rank IC with Newey-West t >= 1.96"
            elif mean_ic is not None and mean_ic <= 0:
                assessment = "ineffective"
                rationale = "non-positive selection-OOS rank IC in this run"
            else:
                assessment = "inconclusive"
                rationale = "positive evidence does not clear the pre-declared t >= 1.96 reporting rule"
        else:
            # Confirmation results are never promoted to a selection verdict.
            assessment = "confirmation_only"
            rationale = "already-opened confirmation evidence; not reusable for selection"
        output.append(
            {
                "model_id": f"{source}:{model}",
                "source": source,
                "model": str(model),
                "period": period,
                "assessment": assessment,
                "rationale": rationale,
                "mean_rank_ic": mean_ic,
                "rank_ic_nw_t": nw_t,
                "rank_ic_days": _optional_int(raw.get("rank_ic_days")),
                "rank_ic_std": _optional_float(raw.get("rank_ic_std")),
                "mean_decile_spread": _optional_float(raw.get("mean_decile_spread")),
            }
        )
    return output


def _confirmation_comparisons(
    selection: Sequence[Mapping[str, Any]],
    confirmation: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    selection_by_id = {str(item["model_id"]): item for item in selection}
    output: list[dict[str, Any]] = []
    for item in confirmation:
        model_id = str(item["model_id"])
        prior = selection_by_id.get(model_id)
        selection_ic = _optional_float(prior.get("mean_rank_ic")) if prior else None
        confirmation_ic = _optional_float(item.get("mean_rank_ic"))
        delta = (
            confirmation_ic - selection_ic
            if confirmation_ic is not None and selection_ic is not None
            else None
        )
        retention = (
            confirmation_ic / selection_ic
            if confirmation_ic is not None
            and selection_ic is not None
            and abs(selection_ic) > 1e-15
            else None
        )
        if confirmation_ic is None or selection_ic is None:
            stability = "unavailable"
        elif selection_ic * confirmation_ic < 0:
            stability = "sign_flip"
        elif abs(confirmation_ic) >= abs(selection_ic):
            stability = "held_or_improved"
        else:
            stability = "weakened"
        output.append(
            {
                **dict(item),
                "selection_mean_rank_ic": selection_ic,
                "confirmation_ic_change": delta,
                "confirmation_ic_retention": retention,
                "stability": stability,
            }
        )
    return output


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_checkpoint_marker(marker_path: Path, *, run_dir: Path) -> dict[str, Any]:
    """Read and minimally authenticate a sequence checkpoint marker."""

    checked = _safe_artifact(marker_path, run_dir=run_dir)
    relative = str(checked.relative_to(run_dir)).replace("\\", "/")
    result: dict[str, Any] = {
        "path": relative,
        "valid": False,
        "checkpoint_type": None,
        "reason": None,
    }
    try:
        success = json.loads(checked.read_text(encoding="utf-8"))
        if not isinstance(success, Mapping):
            raise ValueError("marker is not an object")
        result["checkpoint_type"] = success.get("checkpoint_type", success.get("stage"))
        manifest_path = checked.parent / "manifest.json"
        if manifest_path.is_file():
            manifest_path = _safe_artifact(manifest_path, run_dir=run_dir)
            manifest_bytes = manifest_path.read_bytes()
            expected = success.get("manifest_sha256")
            actual = hashlib.sha256(manifest_bytes).hexdigest()
            if expected != actual:
                raise ValueError("marker does not authenticate manifest.json")
            manifest = json.loads(manifest_bytes.decode("utf-8"))
            if not isinstance(manifest, Mapping):
                raise ValueError("manifest is not an object")
            if manifest.get("checkpoint_type") != success.get("checkpoint_type"):
                raise ValueError("checkpoint type mismatch")
            files = manifest.get("files")
            if not isinstance(files, Mapping) or not files:
                raise ValueError("manifest has no payload files")
            for name, expected_fingerprint in files.items():
                if Path(str(name)).name != str(name):
                    raise ValueError("unsafe checkpoint payload name")
                payload = _safe_artifact(checked.parent / str(name), run_dir=run_dir)
                if not payload.is_file() or not isinstance(expected_fingerprint, Mapping):
                    raise ValueError(f"missing or invalid checkpoint payload: {name}")
                if _optional_int(expected_fingerprint.get("size")) != payload.stat().st_size:
                    raise ValueError(f"checkpoint payload size mismatch: {name}")
                if str(expected_fingerprint.get("sha256")) != _sha256(payload):
                    raise ValueError(f"checkpoint payload hash mismatch: {name}")
        elif not result["checkpoint_type"]:
            raise ValueError("completion marker has no checkpoint identity")
        result["valid"] = True
    except Exception as exc:
        result["reason"] = str(exc)
    return result


def _sequence_evidence(run_dir: Path) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    directories = [
        path
        for path in sorted(run_dir.iterdir())
        if path.is_dir() and path.name.lower().startswith("sequence")
    ]
    summary: dict[str, Any] | None = None
    markers: list[dict[str, Any]] = []
    for directory in directories:
        candidate = _load_json(directory / "summary.json", run_dir=run_dir)
        if candidate is not None and summary is None:
            summary = _as_mapping(candidate, label="sequence summary")
        for marker in sorted(directory.rglob("_SUCCESS")):
            markers.append(_verify_checkpoint_marker(marker, run_dir=run_dir))

    valid = [item for item in markers if item["valid"]]
    invalid = [item for item in markers if not item["valid"]]
    types = Counter(str(item.get("checkpoint_type") or "unknown") for item in valid)
    if summary is not None and not invalid:
        status = "complete_summary_available"
    elif summary is not None:
        status = "summary_with_invalid_checkpoints"
    elif valid and not invalid:
        status = "verified_checkpoints_only"
    elif invalid:
        status = "invalid_or_incomplete_checkpoints"
    else:
        status = "not_available"
    checkpoint_report = {
        "status": status,
        "markers_found": len(markers),
        "valid_markers": len(valid),
        "invalid_markers": len(invalid),
        "checkpoint_types": dict(sorted(types.items())),
        "markers": markers,
        "statement": (
            "Sequence model effectiveness is unknown until a valid summary contains OOS diagnostics."
            if summary is None
            else "Sequence diagnostics are reported separately from checkpoint completion."
        ),
    }
    return summary, checkpoint_report


def _search_payload(
    run_dir: Path,
    search_result: Any | None,
) -> dict[str, Any] | None:
    if search_result is None:
        raw = _load_json(run_dir / "search" / "search_summary.json", run_dir=run_dir)
        if raw is None:
            champion = _load_json(run_dir / "search" / "champion.json", run_dir=run_dir)
            raw = {"champion": champion} if champion is not None else None
    elif isinstance(search_result, (str, Path)):
        raw = _load_json(Path(search_result), run_dir=run_dir, required=True)
    elif isinstance(search_result, Mapping):
        raw = dict(search_result)
    elif hasattr(search_result, "to_dict"):
        raw = search_result.to_dict()
    else:
        raise ResearchReportError("search_result must be a SearchResult, mapping, or confined JSON path")
    return _as_mapping(raw, label="search result")


def _candidate_identity_without_cost(candidate: Mapping[str, Any]) -> str:
    copy = _json_safe(candidate)
    assert isinstance(copy, dict)
    copy.pop("candidate_id", None)
    copy.pop("extra_cost_bps", None)
    return json.dumps(copy, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _metric_subset(raw: Any) -> dict[str, float | int | None]:
    values = raw if isinstance(raw, Mapping) else {}
    names = (
        "total_return",
        "cagr",
        "annualized_volatility",
        "sharpe",
        "sortino",
        "calmar",
        "max_drawdown",
        "total_gross_turnover",
        "total_one_way_turnover",
        "total_cost",
        "cost_to_initial_equity",
        "max_adv_participation",
    )
    output: dict[str, float | int | None] = {}
    for name in names:
        value = values.get(name)
        output[name] = _optional_int(value) if name == "periods" else _optional_float(value)
    return output


def _cost_curve(search: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    champion = search.get("champion") if search else None
    champion_candidate = champion.get("candidate") if isinstance(champion, Mapping) else None
    target_identity = (
        _candidate_identity_without_cost(champion_candidate)
        if isinstance(champion_candidate, Mapping)
        else None
    )
    records = search.get("stage_c", []) if search else []
    matching: dict[float, Mapping[str, Any]] = {}
    if target_identity is not None and isinstance(records, Sequence):
        for raw in records:
            if not isinstance(raw, Mapping) or not isinstance(raw.get("candidate"), Mapping):
                continue
            candidate = raw["candidate"]
            if _candidate_identity_without_cost(candidate) != target_identity:
                continue
            bps = _optional_float(candidate.get("extra_cost_bps"))
            if bps is not None:
                matching[bps] = raw
    output: list[dict[str, Any]] = []
    for bps in EXPECTED_COST_SENSITIVITY_BPS:
        raw = next(
            (record for key, record in matching.items() if math.isclose(key, bps)),
            None,
        )
        output.append(
            {
                "extra_cost_bps": bps,
                "available": raw is not None,
                "objective_value": _optional_float(raw.get("objective_value")) if raw else None,
                "eligible": bool(raw.get("eligible")) if raw else None,
                "metrics": _metric_subset(raw.get("metrics")) if raw else {},
                "cost_breakdown": _json_safe(raw.get("cost_breakdown", {})) if raw else {},
                "interpretation": "additional fixed friction on top of base spread/impact/funding",
            }
        )
    return output


def _stress_evidence(run_dir: Path, search: Mapping[str, Any] | None) -> dict[str, Any]:
    embedded = search.get("stress") if search else None
    if embedded is None and search and isinstance(search.get("audit"), Mapping):
        embedded = search["audit"].get("stress")
    if embedded is None:
        for path in (
            run_dir / "search" / "stress_summary.json",
            run_dir / "search" / "stress.json",
            run_dir / "stress" / "summary.json",
        ):
            embedded = _load_json(path, run_dir=run_dir)
            if embedded is not None:
                break
    if embedded is None:
        return {
            "status": "not_available",
            "scenarios": [],
            "statement": "No verified stress-window artifact was found; no stress robustness claim is made.",
        }
    if isinstance(embedded, Mapping):
        scenarios = embedded.get("scenarios", embedded.get("results", embedded.get("windows", [])))
        if not isinstance(scenarios, Sequence) or isinstance(scenarios, (str, bytes)):
            scenarios = [dict(embedded)]
    elif isinstance(embedded, Sequence) and not isinstance(embedded, (str, bytes)):
        scenarios = list(embedded)
    else:
        raise ResearchReportError("stress artifact must be an object or array")
    return {
        "status": "available",
        "scenarios": _json_safe(scenarios),
        "statement": "Stress results are diagnostics, not an independent untouched holdout.",
    }


def _data_biases(feature_report: Mapping[str, Any] | None) -> list[dict[str, str]]:
    report = feature_report or {}
    market_proxy = str(report.get("market_proxy", "not documented"))
    return [
        {
            "id": "survivorship_bias",
            "severity": "high",
            "status": "known_limitation",
            "statement": "The current local universe is projected backward; delisted and historical constituents are not fully point-in-time.",
        },
        {
            "id": "benchmark_and_sector_coverage",
            "severity": "medium",
            "status": "known_limitation",
            "statement": f"Benchmark/sector coverage is incomplete. Market proxy recorded by the feature stage: {market_proxy}",
        },
        {
            "id": "daily_fill_approximation",
            "severity": "medium",
            "status": "known_limitation",
            "statement": "Daily bars approximate next-open execution and cannot reconstruct an exact 09:35 fill or intraday market impact.",
        },
        {
            "id": "cost_calibration",
            "severity": "high",
            "status": "known_limitation",
            "statement": "Spread and square-root impact remain proxies until calibrated against actual order/fill records.",
        },
        {
            "id": "opened_confirmation_period",
            "severity": "high",
            "status": "information_boundary",
            "statement": "The reported confirmation period has already been inspected and cannot be called an untouched lockbox during further tuning.",
        },
    ]


def build_research_report(
    run_dir: str | Path,
    *,
    research_root: str | Path = RESEARCH_ROOT,
    search_result: Any | None = None,
    production_deployed: bool = False,
) -> dict[str, Any]:
    """Build a JSON-safe report from a confined Research v2 run directory.

    ``production_deployed=True`` is deliberately rejected: this offline
    reporter has no authority or evidence to attest a production deployment.
    """

    if production_deployed:
        raise ResearchReportError(
            "the offline research reporter cannot attest or mark a production deployment"
        )
    root = ensure_research_output_path(research_root, research_root=research_root)
    run = _ensure_runs_path(run_dir, research_root=root)
    if not run.is_dir():
        raise ResearchReportError(f"research run directory does not exist: {run}")

    feature_report = _as_mapping(
        _load_json(run / "features" / "report.json", run_dir=run),
        label="feature report",
    )
    tabular = _as_mapping(
        _load_json(run / "tabular" / "summary.json", run_dir=run),
        label="tabular summary",
    )
    sequence, checkpoints = _sequence_evidence(run)
    search = _search_payload(run, search_result)

    selection_models = [
        *_model_assessment(
            tabular.get("selection_diagnostics") if tabular else None,
            source="tabular",
            period="selection_oos",
        ),
        *_model_assessment(
            sequence.get("selection_diagnostics") if sequence else None,
            source="sequence",
            period="selection_oos",
        ),
    ]
    confirmation_models = [
        *_model_assessment(
            tabular.get("lockbox_diagnostics") if tabular else None,
            source="tabular",
            period="previously_opened_confirmation",
        ),
        *_model_assessment(
            sequence.get("lockbox_diagnostics") if sequence else None,
            source="sequence",
            period="previously_opened_confirmation",
        ),
    ]
    confirmation_models = _confirmation_comparisons(selection_models, confirmation_models)
    classifications = {
        label: [item["model_id"] for item in selection_models if item["assessment"] == label]
        for label in ("effective", "ineffective", "inconclusive")
    }

    champion = search.get("champion") if search else None
    if not isinstance(champion, Mapping):
        champion = None
    lockbox = search.get("lockbox") if search else None
    if not isinstance(lockbox, Mapping):
        lockbox = None
    audit = search.get("audit") if search and isinstance(search.get("audit"), Mapping) else {}
    net_cost_status = "available" if champion is not None else "not_available"

    run_metadata = _load_json(run / "run.json", run_dir=run)
    report: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "report_type": "offline_research_evidence_not_deployment",
        "run": {
            "run_id": run.name,
            "run_dir": str(run),
            "metadata": _json_safe(run_metadata) if isinstance(run_metadata, Mapping) else {},
        },
        "evidence_boundaries": {
            "selection_oos": {
                "role": "model and portfolio selection evidence",
                "used_for_selection": True,
                "status": "available" if selection_models else "not_available",
            },
            "previously_opened_confirmation": {
                "role": "confirmation only; never reused for tuning or selection",
                "used_for_selection": False,
                "status": "available" if confirmation_models or lockbox else "not_available",
                "previously_opened": True,
            },
            "net_cost_portfolio": {
                "role": "tradable portfolio evidence after modeled execution/funding costs",
                "used_for_selection": True,
                "status": net_cost_status,
                "continuous_oos_state": bool(champion.get("continuous_oos_state")) if champion else None,
            },
        },
        "data": {
            "feature_report_status": "available" if feature_report else "not_available",
            "feature_report": _json_safe(feature_report or {}),
            "biases_and_limitations": _data_biases(feature_report),
        },
        "models": {
            "assessment_rule": {
                "effective": "selection-OOS mean rank IC > 0 and Newey-West t >= 1.96",
                "ineffective": "selection-OOS mean rank IC <= 0 in this run",
                "inconclusive": "all other finite or missing cases",
                "note": "This reporting label is evidence classification, not proof of permanent alpha.",
            },
            "tabular_counts": _summary_counts(tabular),
            "sequence_counts": _summary_counts(sequence),
            "selection_oos": selection_models,
            "previously_opened_confirmation": confirmation_models,
            "classifications": classifications,
            "tabular_locked_settings": _json_safe(tabular.get("locked_settings", {})) if tabular else {},
            "sequence_locked_settings": _json_safe(sequence.get("locked_settings", {})) if sequence else {},
            "sequence_checkpoints": checkpoints,
        },
        "net_cost_portfolio": {
            "status": net_cost_status,
            "selection_only_champion": bool(audit.get("selection_only_champion")) if audit else None,
            "cost_interpretation": audit.get(
                "cost_interpretation",
                "additional fixed friction on top of modeled spread, impact, and funding",
            ),
            "champion": {
                "candidate": _json_safe(champion.get("candidate", {})),
                "metrics": _metric_subset(champion.get("metrics")),
                "cost_breakdown": _json_safe(champion.get("cost_breakdown", {})),
                "objective_value": _optional_float(champion.get("objective_value")),
                "eligible": bool(champion.get("eligible")),
                "gate_failures": _json_safe(champion.get("gate_failures", [])),
            } if champion else None,
            "cost_sensitivity": _cost_curve(search),
            "worst_fold": _json_safe(champion.get("worst_metrics", {})) if champion else {},
            "stress": _stress_evidence(run, search),
            "previously_opened_confirmation": {
                "status": "available" if lockbox else "not_available",
                "metrics": _metric_subset(lockbox.get("metrics")) if lockbox else {},
                "objective_value": _optional_float(lockbox.get("objective_value")) if lockbox else None,
                "candidate_frozen_before_access": bool(audit.get("lockbox_accessed_after_champion_lock")) if audit else None,
            },
        },
        "production": {
            "deployed": False,
            "daily_run_changed": False,
            "statement": "Research-only result. No model, champion configuration, risk overlay, scheduler, OMS, broker, or daily-run setting was deployed or changed by this report.",
            "promotion_requirements": [
                "fresh shadow/paper forward evidence",
                "real-fill cost calibration and capacity review",
                "explicit production code review and deployment approval",
            ],
        },
    }
    return _json_safe(report)


def _fmt_number(value: Any, digits: int = 4) -> str:
    number = _optional_float(value)
    return "N/A" if number is None else f"{number:.{digits}f}"


def _fmt_percent(value: Any, digits: int = 2) -> str:
    number = _optional_float(value)
    return "N/A" if number is None else f"{number * 100:.{digits}f}%"


def _cell(value: Any) -> str:
    return str(value if value is not None else "N/A").replace("|", "\\|").replace("\n", " ")


def render_research_report_markdown(report: Mapping[str, Any]) -> str:
    """Render the canonical JSON report as a concise Chinese Markdown report."""

    run = report.get("run", {})
    boundaries = report.get("evidence_boundaries", {})
    models = report.get("models", {})
    portfolio = report.get("net_cost_portfolio", {})
    lines = [
        "# ancserAPX Research v2 最終可驗證研究報告",
        "",
        f"- Run ID：`{_cell(run.get('run_id'))}`",
        f"- 產生時間（UTC）：`{_cell(report.get('generated_at_utc'))}`",
        "- 性質：離線研究證據，不是部署批准。",
        "",
        "> 重要：確認期結果已被查看，只能作 previously-opened confirmation；後續調參不得再稱它為 untouched lockbox。",
        "",
        "## 證據邊界",
        "",
        "| 層級 | 用途 | 用於選擇 | 狀態 |",
        "|---|---|---:|---|",
    ]
    for name in ("selection_oos", "previously_opened_confirmation", "net_cost_portfolio"):
        item = boundaries.get(name, {})
        lines.append(
            f"| `{name}` | {_cell(item.get('role'))} | "
            f"{'是' if item.get('used_for_selection') else '否'} | {_cell(item.get('status'))} |"
        )

    data = report.get("data", {})
    feature = data.get("feature_report", {})
    lines.extend(
        [
            "",
            "## 資料與時鐘",
            "",
            f"- Feature report：`{_cell(data.get('feature_report_status'))}`",
            f"- 資料列／有效列：`{_cell(feature.get('validated_rows'))}` / `{_cell(feature.get('eligible_rows'))}`",
            f"- 股票／交易日：`{_cell(feature.get('symbols'))}` / `{_cell(feature.get('sessions'))}`",
            f"- 決策時鐘：{_cell(feature.get('decision_clock'))}",
            f"- 標籤時鐘：{_cell(feature.get('label_clock'))}",
            f"- 移除錯價列：`{_cell(feature.get('dropped_invalid_rows'))}`；移除稀疏交易日：`{len(feature.get('dropped_sparse_sessions', [])) if isinstance(feature.get('dropped_sparse_sessions'), list) else 'N/A'}`",
            "",
            "已知偏差／限制：",
            "",
        ]
    )
    for bias in data.get("biases_and_limitations", []):
        lines.append(
            f"- **{_cell(bias.get('id'))} [{_cell(bias.get('severity'))}]**：{_cell(bias.get('statement'))}"
        )

    lines.extend(
        [
            "",
            "## Selection OOS 模型證據",
            "",
            "判定規則：IC > 0 且 Newey-West t ≥ 1.96 才標為 `effective`；IC ≤ 0 標為 `ineffective`；其他為 `inconclusive`。這不是永久 alpha 的證明。",
            "",
            "| 模型 | 來源 | Rank IC | NW t | 天數 | 判定 |",
            "|---|---|---:|---:|---:|---|",
        ]
    )
    selection = models.get("selection_oos", [])
    if selection:
        for item in selection:
            lines.append(
                f"| `{_cell(item.get('model'))}` | {_cell(item.get('source'))} | "
                f"{_fmt_number(item.get('mean_rank_ic'))} | {_fmt_number(item.get('rank_ic_nw_t'), 2)} | "
                f"{_cell(item.get('rank_ic_days'))} | **{_cell(item.get('assessment'))}** |"
            )
    else:
        lines.append("| N/A | N/A | N/A | N/A | N/A | 沒有可驗證 summary |")

    lines.extend(
        [
            "",
            "## Previously-opened confirmation（不參與選擇）",
            "",
            "| 模型 | Rank IC | NW t | 相對 Selection | 穩定性 |",
            "|---|---:|---:|---:|---|",
        ]
    )
    confirmation = models.get("previously_opened_confirmation", [])
    if confirmation:
        for item in confirmation:
            lines.append(
                f"| `{_cell(item.get('model'))}` | {_fmt_number(item.get('mean_rank_ic'))} | "
                f"{_fmt_number(item.get('rank_ic_nw_t'), 2)} | {_fmt_number(item.get('confirmation_ic_change'))} | "
                f"{_cell(item.get('stability'))} |"
            )
    else:
        lines.append("| N/A | N/A | N/A | N/A | 沒有可驗證確認期診斷 |")

    checkpoints = models.get("sequence_checkpoints", {})
    lines.extend(
        [
            "",
            "## GRU／Transformer 與 checkpoint",
            "",
            f"- 狀態：`{_cell(checkpoints.get('status'))}`",
            f"- marker：{_cell(checkpoints.get('valid_markers'))} valid / {_cell(checkpoints.get('invalid_markers'))} invalid",
            f"- 說明：{_cell(checkpoints.get('statement'))}",
            "",
            "## 扣成本組合與 champion",
            "",
            f"- 狀態：`{_cell(portfolio.get('status'))}`",
            f"- Champion 僅由 selection 決定：`{_cell(portfolio.get('selection_only_champion'))}`",
        ]
    )
    champion = portfolio.get("champion")
    if champion:
        metrics = champion.get("metrics", {})
        lines.extend(
            [
                f"- Selection net CAGR / Sharpe / MaxDD：{_fmt_percent(metrics.get('cagr'))} / {_fmt_number(metrics.get('sharpe'), 2)} / {_fmt_percent(metrics.get('max_drawdown'))}",
                f"- Total cost / one-way turnover：{_fmt_number(metrics.get('total_cost'), 2)} / {_fmt_number(metrics.get('total_one_way_turnover'), 2)}",
                "",
                "Champion 完整配置：",
                "",
                "```json",
                json.dumps(champion.get("candidate", {}), indent=2, sort_keys=True, ensure_ascii=False),
                "```",
            ]
        )
    else:
        lines.append("- 尚無可驗證 net-cost champion；不宣稱已找到最佳策略。")

    lines.extend(
        [
            "",
            "### 成本敏感度（額外 fixed friction）",
            "",
            "| bps | 可用 | CAGR | Sharpe | MaxDD | Total cost |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for item in portfolio.get("cost_sensitivity", []):
        metrics = item.get("metrics", {})
        lines.append(
            f"| {_fmt_number(item.get('extra_cost_bps'), 0)} | {'是' if item.get('available') else '否'} | "
            f"{_fmt_percent(metrics.get('cagr'))} | {_fmt_number(metrics.get('sharpe'), 2)} | "
            f"{_fmt_percent(metrics.get('max_drawdown'))} | {_fmt_number(metrics.get('total_cost'), 2)} |"
        )

    worst = portfolio.get("worst_fold", {})
    lines.extend(
        [
            "",
            "### Worst fold",
            "",
            f"- 目標最差 fold：`{_cell(worst.get('objective_fold'))}`，{_cell(worst.get('objective_name'))} = `{_fmt_number(worst.get('objective_value'))}`",
            f"- 最低報酬 fold：`{_cell(worst.get('return_fold'))}`，total return = `{_fmt_percent(worst.get('minimum_total_return'))}`",
            f"- 最大回撤最差 fold：`{_cell(worst.get('drawdown_fold'))}`，MaxDD = `{_fmt_percent(worst.get('worst_max_drawdown'))}`",
            "",
            "### Stress",
            "",
        ]
    )
    stress = portfolio.get("stress", {})
    lines.append(f"- 狀態：`{_cell(stress.get('status'))}`。{_cell(stress.get('statement'))}")
    for scenario in stress.get("scenarios", []):
        if isinstance(scenario, Mapping):
            name = scenario.get("name", scenario.get("scenario", scenario.get("window", "unnamed")))
            lines.append(f"- `{_cell(name)}`：`{_cell(json.dumps(_json_safe(scenario), sort_keys=True, ensure_ascii=False))}`")
        else:
            lines.append(f"- {_cell(scenario)}")

    confirmation_portfolio = portfolio.get("previously_opened_confirmation", {})
    confirmation_metrics = confirmation_portfolio.get("metrics", {})
    production = report.get("production", {})
    lines.extend(
        [
            "",
            "### Champion 的已開封確認期",
            "",
            f"- 狀態：`{_cell(confirmation_portfolio.get('status'))}`；champion 在讀取前已鎖定：`{_cell(confirmation_portfolio.get('candidate_frozen_before_access'))}`",
            f"- Confirmation CAGR / Sharpe / MaxDD：{_fmt_percent(confirmation_metrics.get('cagr'))} / {_fmt_number(confirmation_metrics.get('sharpe'), 2)} / {_fmt_percent(confirmation_metrics.get('max_drawdown'))}",
            "",
            "## Production 聲明",
            "",
            f"> **未部署。** {_cell(production.get('statement'))}",
            "",
            "升級前仍須：",
            "",
        ]
    )
    for requirement in production.get("promotion_requirements", []):
        lines.append(f"- {_cell(requirement)}")
    return "\n".join(lines).rstrip() + "\n"


def _atomic_write(path: Path, content: str) -> None:
    temporary = path.parent / f".{path.name}.tmp-{os.getpid()}"
    try:
        temporary.write_text(content, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_research_report(
    report: Mapping[str, Any],
    output_dir: str | Path,
    *,
    research_root: str | Path = RESEARCH_ROOT,
) -> ReportPaths:
    """Write canonical JSON/Markdown under ``research_v2/runs`` only."""

    directory = _ensure_runs_path(output_dir, research_root=research_root)
    if directory.exists() and (directory.is_symlink() or not directory.is_dir()):
        raise UnsafeResearchPath(f"report output is not a safe directory: {directory}")
    directory.mkdir(parents=True, exist_ok=True)
    json_path = _ensure_runs_path(directory / "final_report.json", research_root=research_root)
    markdown_path = _ensure_runs_path(directory / "final_report.md", research_root=research_root)
    payload = json.dumps(_json_safe(report), indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
    markdown = render_research_report_markdown(report)
    _atomic_write(json_path, payload)
    _atomic_write(markdown_path, markdown)
    return ReportPaths(json=json_path, markdown=markdown_path)


def generate_research_report(
    run_dir: str | Path,
    *,
    output_dir: str | Path | None = None,
    research_root: str | Path = RESEARCH_ROOT,
    search_result: Any | None = None,
    production_deployed: bool = False,
) -> ReportPaths:
    """Build and persist the final report without running any experiment."""

    root = ensure_research_output_path(research_root, research_root=research_root)
    run = _ensure_runs_path(run_dir, research_root=root)
    report = build_research_report(
        run,
        research_root=root,
        search_result=search_result,
        production_deployed=production_deployed,
    )
    destination = run / "report" if output_dir is None else output_dir
    return write_research_report(report, destination, research_root=root)


__all__ = [
    "EXPECTED_COST_SENSITIVITY_BPS",
    "REPORT_SCHEMA_VERSION",
    "ReportPaths",
    "ResearchReportError",
    "build_research_report",
    "generate_research_report",
    "render_research_report_markdown",
    "write_research_report",
]
