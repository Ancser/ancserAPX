from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from research_v2.letf_universe import SEED30_REGISTRY
from research_v2.run_letf_rotation import (
    _eligibility_matrix,
    _paired_return_comparison,
    _snapshot_marker_fields,
    _source_hashes,
)


def _instrument(ticker: str):
    return next(item for item in SEED30_REGISTRY if item.ticker == ticker)


def test_vectorized_runner_eligibility_respects_reused_ticker_identity() -> None:
    fngu = _instrument("FNGU")
    sessions = pd.DatetimeIndex(
        pd.to_datetime(
            ["2025-06-20", "2025-06-23", "2025-06-24", "2025-06-25", "2025-06-26"]
        )
    )
    # A current-symbol-normalized provider may return FNGU rows from before the
    # actual 24 June rename.  They cannot stand in for the FNGB ticker that was
    # tradable then, so the FNGU warm-up restarts at the alias boundary.
    bars = pd.DataFrame({"timestamp": sessions, "symbol": "FNGU"})
    matrix = _eligibility_matrix(bars, sessions, (fngu,), 2)

    assert not matrix.at[pd.Timestamp("2025-06-23"), "FNGU"]
    assert not matrix.at[pd.Timestamp("2025-06-24"), "FNGU"]
    assert matrix.at[pd.Timestamp("2025-06-25"), "FNGU"]


def test_vectorized_runner_eligibility_is_future_invariant_and_contiguous() -> None:
    upro = _instrument("UPRO")
    past = pd.bdate_range("2024-01-02", periods=6)
    observed = pd.DataFrame({"timestamp": past, "symbol": "UPRO"})
    base = _eligibility_matrix(observed, past, (upro,), 3)

    future_sessions = past.append(pd.bdate_range("2030-01-02", periods=3))
    future_bars = pd.concat(
        [
            observed,
            pd.DataFrame(
                {"timestamp": future_sessions[-3:], "symbol": "UPRO"}
            ),
        ],
        ignore_index=True,
    )
    mutated = _eligibility_matrix(future_bars, future_sessions, (upro,), 3)
    pd.testing.assert_series_equal(
        base["UPRO"], mutated.loc[past, "UPRO"], check_freq=False
    )

    missing = observed.loc[observed["timestamp"] != past[-2]]
    broken = _eligibility_matrix(missing, past, (upro,), 3)
    assert not broken.at[past[-1], "UPRO"]


def _scenario(dates: list[str], returns: list[float]):
    rows = []
    for date, value in zip(dates, returns):
        rows.append(
            SimpleNamespace(
                session=pd.Timestamp(date),
                starting_equity=100.0,
                ending_equity=100.0 * (1.0 + value),
            )
        )
    return SimpleNamespace(ledger=tuple(rows))


def test_paired_comparison_uses_only_common_sessions() -> None:
    benchmark = _scenario(
        ["2026-01-02", "2026-01-05", "2026-01-06"],
        [0.50, 0.08, 0.12],
    )
    strategy = _scenario(
        ["2026-01-05", "2026-01-06"],
        [0.03, 0.05],
    )

    result = _paired_return_comparison(strategy, benchmark)

    assert result["paired_sessions"] == 2
    assert result["paired_strategy_total_return"] == pytest.approx(0.0815)
    assert result["spy_realized_vol_control_return_scale"] == pytest.approx(0.5)
    assert result["paired_spy_realized_vol_control_total_return"] == pytest.approx(0.1024)
    assert result["total_return_minus_spy_realized_vol_control"] == pytest.approx(-0.0209)
    assert result["paired_strategy_annualized_volatility"] == pytest.approx(
        result["paired_spy_control_annualized_volatility"]
    )


def test_reproducibility_source_manifest_covers_formal_engine() -> None:
    hashes = _source_hashes()

    assert "research_v2/run_letf_rotation.py" in hashes
    assert "research_v2/letf_universe.py" in hashes
    assert "research_v2/backtest.py" in hashes
    assert "research_v2/snapshot.py" in hashes
    assert "research_v2/safety.py" in hashes
    assert all(len(value) == 64 for value in hashes.values())


def test_success_marker_identity_binds_verified_snapshot_parquet_aggregate() -> None:
    metadata = {
        "snapshot_id": "verified-snapshot",
        "manifest": {"sha256": "1" * 64},
        "snapshot_data_sha256": "2" * 64,
        "snapshot_data_file_count": 58,
        "snapshot_data_hash_scheme": "research-v2-snapshot-data-sha256-v1",
    }

    fields = _snapshot_marker_fields(metadata)

    assert fields == {
        "snapshot_id": "verified-snapshot",
        "snapshot_manifest_sha256": "1" * 64,
        "snapshot_data_sha256": "2" * 64,
        "snapshot_data_file_count": 58,
        "snapshot_data_hash_scheme": "research-v2-snapshot-data-sha256-v1",
    }
    with pytest.raises(ValueError, match="snapshot data SHA-256"):
        _snapshot_marker_fields(
            {key: value for key, value in metadata.items() if key != "snapshot_data_sha256"}
        )
