from __future__ import annotations

import pandas as pd
import pytest

from research_v2.run_neutralization_study_v2 import (
    LEGACY_PARITY_FIELDS,
    _assert_legacy_parity,
    _legacy_sector_snapshot,
    _weekly_last_session_signal_map,
    _weekly_pre_rebalance_signal_map,
)


def test_weekly_schedule_uses_last_available_session_not_every_fifth_row():
    dates = pd.to_datetime(
        [
            "2025-01-06",
            "2025-01-07",
            "2025-01-08",
            "2025-01-09",
            "2025-01-10",
            "2025-01-13",
            "2025-01-14",
            "2025-01-15",
            "2025-01-16",  # Friday holiday: Thursday is the last session.
        ]
    )
    frame = pd.DataFrame(
        {
            "timestamp": dates,
            "symbol": ["A"] * len(dates),
            "score": range(len(dates)),
        }
    )
    signals = _weekly_last_session_signal_map(
        frame, score_column="score", eligible_symbols=["A"]
    )
    assert tuple(signals) == (
        pd.Timestamp("2025-01-10"),
        pd.Timestamp("2025-01-16"),
    )


def test_live_weekly_schedule_uses_prior_close_before_rebalance_session():
    dates = pd.to_datetime(
        [
            "2025-01-06",
            "2025-01-07",
            "2025-01-08",
            "2025-01-09",
            "2025-01-10",
            "2025-01-13",
            "2025-01-14",
            "2025-01-15",
            "2025-01-16",  # Friday holiday: scheduled rebalance is Thursday.
            "2025-01-20",  # proves the prior ISO week is complete
        ]
    )
    frame = pd.DataFrame(
        {
            "timestamp": dates,
            "symbol": ["A"] * len(dates),
            "score": range(len(dates)),
        }
    )
    signals = _weekly_pre_rebalance_signal_map(
        frame, score_column="score", eligible_symbols=["A"], market_sessions=dates
    )
    # Thursday close -> Friday open in week one; Wednesday close -> Thursday
    # open when Friday is a holiday in week two.
    assert tuple(signals) == (
        pd.Timestamp("2025-01-09"),
        pd.Timestamp("2025-01-15"),
    )


def test_live_weekly_schedule_skips_stale_or_truncated_week_signals():
    sessions = pd.to_datetime(
        [
            "2025-01-06", "2025-01-07", "2025-01-08", "2025-01-09", "2025-01-10",
            "2025-01-13", "2025-01-14", "2025-01-15", "2025-01-16", "2025-01-17",
            "2025-01-20", "2025-01-21",  # terminal partial week
        ]
    )
    # Week two has no Thursday score, so a Friday execution must not reuse a
    # stale score.  The terminal Tuesday is not treated as that week's Friday.
    prediction_dates = pd.to_datetime(
        [
            "2025-01-06", "2025-01-07", "2025-01-08", "2025-01-09", "2025-01-10",
            "2025-01-17", "2025-01-20", "2025-01-21",
        ]
    )
    frame = pd.DataFrame(
        {"timestamp": prediction_dates, "symbol": "A", "score": range(len(prediction_dates))}
    )
    signals = _weekly_pre_rebalance_signal_map(
        frame,
        score_column="score",
        eligible_symbols=["A"],
        market_sessions=sessions,
    )
    assert tuple(signals) == (pd.Timestamp("2025-01-09"),)


def test_live_weekly_schedule_conservatively_skips_terminal_thursday():
    sessions = pd.to_datetime(
        ["2025-01-20", "2025-01-21", "2025-01-22", "2025-01-23"]
    )
    frame = pd.DataFrame(
        {"timestamp": sessions, "symbol": "A", "score": range(len(sessions))}
    )
    with pytest.raises(ValueError, match="generated no signals"):
        _weekly_pre_rebalance_signal_map(
            frame,
            score_column="score",
            eligible_symbols=["A"],
            market_sessions=sessions,
        )


def test_legacy_sector_snapshot_reconstructs_frozen_search_coverage():
    # Include every key needed to exercise known and unknown mapping without
    # importing the mutable production module into the test process.
    symbols = ["AAPL", "ABNB", "MSFT", "NVDA"]
    snapshot, audit = _legacy_sector_snapshot(symbols)
    assert snapshot["AAPL"] == "Technology"
    assert snapshot["ABNB"] == "Unknown"
    assert audit["runtime_entries"] == 382
    # The 356/124 hard guard applies when called with the actual 480-name OOS
    # universe, so a small fixture legitimately reports its own count.


def test_legacy_parity_guard_fails_on_any_material_drift():
    expected = {
        field: (504 if field == "periods" else float(index + 1))
        for index, field in enumerate(LEGACY_PARITY_FIELDS)
    }
    row = {
        "period": "selection",
        "scenario": "hybrid50_legacy_champion_parity",
        "variant": "raw",
        "offset": 0,
        "extra_friction_bps": 10.0,
        **expected,
    }
    payload = {
        "candidate": {"candidate_id": "frozen"},
        "metrics": expected,
    }
    result = _assert_legacy_parity(pd.DataFrame([row]), payload)
    assert result["passed"] is True

    drifted = dict(row)
    drifted["final_equity"] += 1.0
    with pytest.raises(AssertionError, match="final_equity"):
        _assert_legacy_parity(pd.DataFrame([drifted]), payload)
