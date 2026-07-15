import json
from pathlib import Path

from frontend.server import _legacy_backtest_record, _tracker_history_records


def test_legacy_backtest_summary_is_visible_in_unified_history(tmp_path: Path):
    path = tmp_path / "backtest_old.json"
    path.write_text(json.dumps({
        "timestamp": "2026-02-13T20:00:00",
        "run_type": "main",
        "config": {
            "top_n": 10,
            "factors": {"momentum": {"enabled": True, "weight": 0.7}},
        },
        "stats": {
            "strategy": {
                "cagr": 0.2, "sharpe": 1.1, "max_dd": -0.15,
                "calmar": 1.33, "win_rate": 0.52,
            },
        },
        "equity_summary": {
            "start_date": "2021-01-01", "end_date": "2026-01-01",
            "start_value": 1.0, "end_value": 2.0,
        },
    }), encoding="utf-8")

    record = _legacy_backtest_record(path)
    assert record is not None
    assert record["result"]["metrics"]["cagr_pct"] == 20.0
    assert len(record["result"]["equity_curve"]) == 2
    assert "momentum" in record["label"]


def test_live_tracker_rows_become_reloadable_live_history(tmp_path: Path):
    path = tmp_path / "tracker_Main.json"
    path.write_text(json.dumps([
        {
            "record_id": "one", "date": "2026-07-13",
            "recorded_at": "2026-07-13T20:00:00+00:00",
            "equity": 2000, "day_pnl": 10,
            "factor_weights": {"Momentum": 1.0},
            "factors": ["Momentum"],
        },
        {
            "record_id": "two", "date": "2026-07-14",
            "recorded_at": "2026-07-14T20:00:00+00:00",
            "equity": 2025, "day_pnl": 25,
        },
    ]), encoding="utf-8")

    records = _tracker_history_records(path)
    assert [record["kind"] for record in records] == ["live", "live"]
    assert records[-1]["result"]["account_name"] == "Main"
    assert records[-1]["result"]["equity_curve"][-1] == {
        "date": "2026-07-14", "value": 2025.0,
    }

