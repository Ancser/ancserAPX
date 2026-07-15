import copy
import hashlib
import json
from pathlib import Path

import pytest

from research_v2.reporting import (
    ResearchReportError,
    build_research_report,
    generate_research_report,
)
from research_v2.safety import UnsafeResearchPath


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _metrics(*, cagr: float, sharpe: float, drawdown: float, cost: float) -> dict:
    return {
        "total_return": cagr * 2.0,
        "cagr": cagr,
        "annualized_volatility": 0.16,
        "sharpe": sharpe,
        "sortino": sharpe * 1.25,
        "calmar": cagr / abs(drawdown),
        "max_drawdown": drawdown,
        "total_gross_turnover": 8.0,
        "total_one_way_turnover": 4.0,
        "total_cost": cost,
        "cost_to_initial_equity": cost / 100_000.0,
        "max_adv_participation": 0.02,
    }


def _candidate(extra_cost_bps: float) -> dict:
    return {
        "candidate_id": f"candidate-{extra_cost_bps:g}",
        "score_column": "score_ensemble",
        "risk_variant": "full",
        "cadence": {
            "name": "daily_5_tranches",
            "rebalance_days": 1,
            "staggered_tranches": 5,
        },
        "extra_cost_bps": extra_cost_bps,
        "rebalance_offset": 0,
        "portfolio_config": {
            "top_n": 15,
            "weighting": "inverse_vol",
            "gross_target": 1.0,
            "sector_cap": 0.35,
            "single_name_cap": 0.10,
            "staggered_tranches": 5,
        },
        "risk_config": {
            "annual_vol_target": 0.15,
            "drawdown_steps": [[0.10, 0.75], [0.20, 0.50]],
            "beta_cap": 1.25,
        },
    }


def _evaluation(extra_cost_bps: float) -> dict:
    return {
        "stage": "C_cost_leverage",
        "label": f"cost={extra_cost_bps:g}",
        "candidate": _candidate(extra_cost_bps),
        "metrics": _metrics(
            cagr=0.24 - extra_cost_bps / 1000.0,
            sharpe=1.60 - extra_cost_bps / 100.0,
            drawdown=-0.14 - extra_cost_bps / 10000.0,
            cost=500.0 + extra_cost_bps * 20.0,
        ),
        "fold_metrics": {
            "F1": _metrics(cagr=0.30, sharpe=1.8, drawdown=-0.10, cost=200.0),
            "F2": _metrics(cagr=0.05, sharpe=0.3, drawdown=-0.14, cost=300.0),
        },
        "worst_metrics": {
            "objective_name": "sharpe",
            "objective_fold": "F2",
            "objective_value": 0.3,
            "return_fold": "F2",
            "minimum_total_return": 0.02,
            "drawdown_fold": "F2",
            "worst_max_drawdown": -0.14,
        },
        "cost_breakdown": {
            "extra_fixed_friction_bps": extra_cost_bps,
            "commission": 100.0,
            "spread": 150.0,
            "market_impact": 200.0,
            "funding": 50.0,
            "total_cost": 500.0 + extra_cost_bps * 20.0,
        },
        "objective_value": 1.60 - extra_cost_bps / 100.0,
        "eligible": True,
        "gate_failures": [],
        "ledger_rows": 200,
        "continuous_oos_state": True,
    }


def _make_full_run(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "research_v2"
    run = root / "runs" / "synthetic-full"
    run.mkdir(parents=True)
    _write_json(run / "run.json", {"snapshot_id": "snapshot-test", "config_sha256": "abc"})
    _write_json(
        run / "features" / "report.json",
        {
            "validated_rows": 20_000,
            "eligible_rows": 12_000,
            "symbols": 100,
            "sessions": 200,
            "decision_clock": "close(t) -> open(t+1) execution",
            "label_clock": "open(t+1) -> open(t+6)",
            "dropped_invalid_rows": 1,
            "dropped_sparse_sessions": [{"timestamp": "2022-01-03", "symbols": 20}],
            "market_proxy": "equal-weight local-universe return",
        },
    )
    diagnostics = {
        "score_ridge": {
            "mean_rank_ic": 0.031,
            "rank_ic_nw_t": 2.40,
            "rank_ic_days": 250,
            "rank_ic_std": 0.18,
            "mean_decile_spread": 0.004,
        },
        "score_gbdt": {
            "mean_rank_ic": 0.010,
            "rank_ic_nw_t": 0.40,
            "rank_ic_days": 250,
            "rank_ic_std": 0.17,
            "mean_decile_spread": 0.001,
        },
        "score_failed": {
            "mean_rank_ic": -0.005,
            "rank_ic_nw_t": -0.50,
            "rank_ic_days": 250,
            "rank_ic_std": 0.16,
            "mean_decile_spread": -0.001,
        },
    }
    confirmation = copy.deepcopy(diagnostics)
    confirmation["score_ridge"]["mean_rank_ic"] = 0.025
    confirmation["score_gbdt"]["mean_rank_ic"] = -0.002
    _write_json(
        run / "tabular" / "summary.json",
        {
            "complete_case_symbols": 100,
            "feature_count": 35,
            "fold_count": 4,
            "selection_rows": 10000,
            "lockbox_rows": 2000,
            "selection_diagnostics": diagnostics,
            "lockbox_diagnostics": confirmation,
            "locked_settings": {"ridge_alpha": 100.0},
        },
    )
    _write_json(
        run / "sequence" / "summary.json",
        {
            "complete_case_symbols": 100,
            "feature_count": 35,
            "fold_count": 4,
            "selection_rows": 10000,
            "lockbox_rows": 2000,
            "selection_diagnostics": {
                "score_gru": {
                    "mean_rank_ic": 0.020,
                    "rank_ic_nw_t": 2.05,
                    "rank_ic_days": 250,
                }
            },
            "lockbox_diagnostics": {
                "score_gru": {
                    "mean_rank_ic": 0.015,
                    "rank_ic_nw_t": 1.10,
                    "rank_ic_days": 50,
                }
            },
            "locked_settings": {"ensemble_weights": {"gru": 1.0}},
        },
    )

    checkpoint = run / "sequence" / "folds" / "F1"
    checkpoint.mkdir(parents=True)
    payload = checkpoint / "history.json"
    payload.write_text('{"loss": [1.0, 0.8]}', encoding="utf-8")
    manifest = {
        "format_version": 1,
        "checkpoint_type": "selection_fold",
        "identity": {"fold": "F1"},
        "files": {
            "history.json": {
                "size": payload.stat().st_size,
                "sha256": hashlib.sha256(payload.read_bytes()).hexdigest(),
            }
        },
    }
    manifest_path = checkpoint / "manifest.json"
    _write_json(manifest_path, manifest)
    _write_json(
        checkpoint / "_SUCCESS",
        {
            "format_version": 1,
            "checkpoint_type": "selection_fold",
            "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        },
    )
    _write_json(
        run / "sequence" / "_SUCCESS",
        {
            "format_version": 1,
            "checkpoint_type": "sequence_research",
            "selection_manifest_sha256": "selection",
            "lockbox_manifest_sha256": "lockbox",
        },
    )

    curve = [_evaluation(bps) for bps in (0.0, 5.0, 10.0, 20.0)]
    champion = copy.deepcopy(curve[2])
    champion["stage"] = "C_champion"
    lockbox = copy.deepcopy(champion)
    lockbox["stage"] = "LOCKBOX"
    lockbox["metrics"] = _metrics(cagr=0.12, sharpe=0.9, drawdown=-0.17, cost=150.0)
    search = {
        "stage_a": [],
        "stage_b": [],
        "stage_c": curve,
        "champion": champion,
        "lockbox": lockbox,
        "neighborhood": [],
        "offset_sensitivity": [],
        "audit": {
            "selection_only_champion": True,
            "lockbox_accessed_after_champion_lock": True,
            "cost_interpretation": "additional fixed friction over modeled costs",
        },
        "stress": {
            "scenarios": [
                {"name": "2022-bear", "total_return": -0.03, "max_drawdown": -0.08}
            ]
        },
    }
    _write_json(run / "search" / "search_summary.json", search)
    return root, run


def test_final_report_separates_evidence_and_reports_all_required_audits(tmp_path):
    root, run = _make_full_run(tmp_path)
    paths = generate_research_report(run, research_root=root)

    assert paths.json == run / "report" / "final_report.json"
    assert paths.markdown == run / "report" / "final_report.md"
    payload = json.loads(paths.json.read_text(encoding="utf-8"))
    boundaries = payload["evidence_boundaries"]
    assert boundaries["selection_oos"]["used_for_selection"] is True
    assert boundaries["previously_opened_confirmation"]["used_for_selection"] is False
    assert boundaries["previously_opened_confirmation"]["previously_opened"] is True
    assert boundaries["net_cost_portfolio"]["status"] == "available"

    classified = payload["models"]["classifications"]
    assert "tabular:score_ridge" in classified["effective"]
    assert "sequence:score_gru" in classified["effective"]
    assert "tabular:score_failed" in classified["ineffective"]
    assert "tabular:score_gbdt" in classified["inconclusive"]
    gbdt_confirmation = next(
        item
        for item in payload["models"]["previously_opened_confirmation"]
        if item["model_id"] == "tabular:score_gbdt"
    )
    assert gbdt_confirmation["stability"] == "sign_flip"

    checkpoints = payload["models"]["sequence_checkpoints"]
    assert checkpoints["status"] == "complete_summary_available"
    assert checkpoints["valid_markers"] == 2
    assert checkpoints["invalid_markers"] == 0

    portfolio = payload["net_cost_portfolio"]
    assert [item["extra_cost_bps"] for item in portfolio["cost_sensitivity"]] == [
        0.0,
        5.0,
        10.0,
        20.0,
    ]
    assert all(item["available"] for item in portfolio["cost_sensitivity"])
    assert portfolio["worst_fold"]["objective_fold"] == "F2"
    assert portfolio["stress"]["status"] == "available"
    assert portfolio["champion"]["candidate"]["portfolio_config"]["weighting"] == "inverse_vol"
    assert payload["production"]["deployed"] is False
    assert payload["production"]["daily_run_changed"] is False

    markdown = paths.markdown.read_text(encoding="utf-8")
    for required in (
        "Selection OOS",
        "Previously-opened confirmation",
        "成本敏感度",
        "Worst fold",
        "Stress",
        "未部署",
        "survivorship_bias",
    ):
        assert required in markdown


def test_missing_experiments_are_unknown_not_success(tmp_path):
    root = tmp_path / "research_v2"
    run = root / "runs" / "incomplete"
    run.mkdir(parents=True)

    report = build_research_report(run, research_root=root)

    assert report["evidence_boundaries"]["selection_oos"]["status"] == "not_available"
    assert report["models"]["sequence_checkpoints"]["status"] == "not_available"
    assert report["net_cost_portfolio"]["status"] == "not_available"
    assert report["net_cost_portfolio"]["stress"]["status"] == "not_available"
    assert report["net_cost_portfolio"]["champion"] is None
    assert all(
        item["available"] is False
        for item in report["net_cost_portfolio"]["cost_sensitivity"]
    )
    assert report["production"]["deployed"] is False


def test_report_writes_cannot_escape_research_runs(tmp_path):
    root = tmp_path / "research_v2"
    run = root / "runs" / "safe"
    run.mkdir(parents=True)

    with pytest.raises(UnsafeResearchPath, match="under"):
        generate_research_report(
            run,
            output_dir=root / "reports-outside-runs",
            research_root=root,
        )
    with pytest.raises(ResearchReportError, match="cannot attest"):
        build_research_report(run, research_root=root, production_deployed=True)

