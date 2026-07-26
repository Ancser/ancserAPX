from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pytest

from research_v2.backtest import MissingHeldReturnError
from research_v2.letf_experiment import (
    LETFExecutionConfig,
    build_market_context,
    cost_sensitivity,
    load_verified_snapshot_metadata,
    paired_moving_block_bootstrap,
    paired_realized_volatility_control,
    run_scenario,
)
from research_v2.snapshot import (
    SnapshotVerificationError,
    create_snapshot,
)


def _bars(*, periods: int = 340, missing: tuple[str, int] | None = None) -> pd.DataFrame:
    sessions = pd.bdate_range("2023-01-02", periods=periods)
    rows = []
    for number, symbol in enumerate(("A", "B", "SPY")):
        returns = np.full(periods, 0.0003 + number * 0.0001)
        close = 100.0 * np.cumprod(1.0 + returns)
        for index, (session, price) in enumerate(zip(sessions, close)):
            if missing == (symbol, index):
                continue
            rows.append(
                {
                    "timestamp": session,
                    "symbol": symbol,
                    "open": price * 0.999,
                    "high": price * 1.002,
                    "low": price * 0.998,
                    "close": price,
                    "volume": 2_000_000.0,
                }
            )
    return pd.DataFrame(rows)


def _config(**updates) -> LETFExecutionConfig:
    base = LETFExecutionConfig(
        top_k=2,
        weighting="equal",
        single_name_cap=0.5,
        theme_cap=0.5,
        target_volatility=None,
        trend_filter=False,
        breadth_exit=None,
        breadth_enter=None,
        drawdown_steps=(),
        extra_friction_bps=0.0,
        min_spread_bps=0.0,
        impact_coefficient=0.0,
    )
    return replace(base, **updates)


def _context(bars: pd.DataFrame, config: LETFExecutionConfig):
    sessions = pd.DatetimeIndex(sorted(pd.to_datetime(bars["timestamp"]).unique()))
    return build_market_context(
        bars,
        candidate_symbols=("A", "B"),
        themes={"A": "ThemeA", "B": "ThemeB"},
        breadth={session: 1.0 for session in sessions},
        config=config,
    )


def _snapshot(
    tmp_path: Path,
    *,
    snapshot_id: str,
    provenance: object = None,
    include_provenance: bool = True,
) -> tuple[Path, Path]:
    source = tmp_path / f"source-{snapshot_id}"
    store = source / "store"
    store.mkdir(parents=True)
    pl.DataFrame(
        {
            "timestamp": [pd.Timestamp("2026-01-02")],
            "symbol": ["SPY"],
            "open": [100.0],
            "high": [101.0],
            "low": [99.0],
            "close": [100.5],
            "volume": [1_000_000.0],
        }
    ).write_parquet(store / "SPY.parquet")
    manifest: dict[str, object] = {
        "SPY": {"last_date": "2026-01-02", "row_count": 1},
    }
    if include_provenance:
        manifest["_provenance"] = provenance
    manifest_path = source / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    research_root = tmp_path / "research_v2"
    snapshot = create_snapshot(
        store_dir=store,
        manifest_path=manifest_path,
        research_root=research_root,
        snapshot_id=snapshot_id,
    )
    return snapshot, research_root


def _sip_provenance(**updates: object) -> dict[str, object]:
    values: dict[str, object] = {
        "provider": "Alpaca Markets",
        "feed": "SIP",
        "adjustment": "ALL",
        "retrieved_at_utc": "2026-07-21T06:20:00+00:00",
    }
    values.update(updates)
    return values


def test_verified_snapshot_provenance_accepts_recorded_sip_all(tmp_path: Path) -> None:
    snapshot, root = _snapshot(
        tmp_path,
        snapshot_id="sip-all",
        provenance=_sip_provenance(),
    )

    metadata = load_verified_snapshot_metadata(
        snapshot,
        research_root=root,
        required_provider="Alpaca Markets",
        required_feed="SIP",
        required_adjustment="ALL",
        require_retrieved_at_utc=True,
    )

    assert metadata["snapshot_id"] == "sip-all"
    assert metadata["manifest"]["sha256"]
    assert metadata["provenance"] == _sip_provenance()
    assert metadata["provenance_sources"]["feed"] == "manifest.json._provenance"
    canonical_files = [
        {
            "relative_path": item["relative_path"],
            "size": item["size"],
            "sha256": item["sha256"],
        }
        for item in sorted(
            metadata["files"], key=lambda item: item["relative_path"]
        )
    ]
    expected_data_sha256 = sha256(
        json.dumps(
            {
                "format_version": metadata["format_version"],
                "snapshot_id": metadata["snapshot_id"],
                "manifest_sha256": metadata["manifest"]["sha256"],
                "files": canonical_files,
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()
    assert metadata["snapshot_data_sha256"] == expected_data_sha256
    assert metadata["snapshot_data_file_count"] == 1
    assert (
        metadata["snapshot_data_hash_scheme"]
        == "research-v2-snapshot-data-sha256-v1"
    )


def test_verified_snapshot_provenance_rejects_iex_when_sip_is_required(tmp_path: Path) -> None:
    snapshot, root = _snapshot(
        tmp_path,
        snapshot_id="iex",
        provenance=_sip_provenance(feed="IEX"),
    )

    with pytest.raises(SnapshotVerificationError, match="feed.*IEX.*SIP"):
        load_verified_snapshot_metadata(
            snapshot,
            research_root=root,
            required_feed="SIP",
            required_adjustment="ALL",
            require_retrieved_at_utc=True,
        )


def test_verified_snapshot_provenance_rejects_missing_or_invalid_schema(tmp_path: Path) -> None:
    missing, root = _snapshot(
        tmp_path,
        snapshot_id="missing-provenance",
        include_provenance=False,
    )
    with pytest.raises(SnapshotVerificationError, match="missing required feed"):
        load_verified_snapshot_metadata(
            missing,
            research_root=root,
            required_feed="SIP",
            required_adjustment="ALL",
            require_retrieved_at_utc=True,
        )

    invalid, root = _snapshot(
        tmp_path,
        snapshot_id="invalid-provenance",
        provenance=[],
    )
    with pytest.raises(SnapshotVerificationError, match="_provenance must be a JSON object"):
        load_verified_snapshot_metadata(invalid, research_root=root)


def test_verified_snapshot_metadata_detects_manifest_tampering(tmp_path: Path) -> None:
    snapshot, root = _snapshot(
        tmp_path,
        snapshot_id="tampered",
        provenance=_sip_provenance(),
    )
    manifest_path = snapshot / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["_provenance"]["feed"] = "IEX"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(SnapshotVerificationError, match="manifest fingerprint mismatch"):
        load_verified_snapshot_metadata(
            snapshot,
            research_root=root,
            required_feed="SIP",
            required_adjustment="ALL",
        )


def test_verified_snapshot_metadata_detects_parquet_and_fingerprint_tampering(
    tmp_path: Path,
) -> None:
    snapshot, root = _snapshot(
        tmp_path,
        snapshot_id="tampered-parquet",
        provenance=_sip_provenance(),
    )
    with (snapshot / "store" / "SPY.parquet").open("ab") as handle:
        handle.write(b"tamper")

    with pytest.raises(SnapshotVerificationError, match="store files changed"):
        load_verified_snapshot_metadata(snapshot, research_root=root)

    metadata_snapshot, metadata_root = _snapshot(
        tmp_path,
        snapshot_id="tampered-fingerprint",
        provenance=_sip_provenance(),
    )
    metadata_path = metadata_snapshot / "snapshot.json"
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    payload["files"][0]["sha256"] = "0" * 64
    metadata_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(SnapshotVerificationError, match="store files changed"):
        load_verified_snapshot_metadata(metadata_snapshot, research_root=metadata_root)


def test_close_signal_executes_only_at_following_open_and_target_gross_is_capped() -> None:
    config = _config()
    context = _context(_bars(), config)
    signal_session = context.sessions[220]
    scenario = run_scenario(
        context,
        {signal_session: {"A": 2.0, "B": 1.0}},
        config=config,
    )

    first = scenario.ledger[0]
    assert first.session == context.sessions[221]
    assert first.executed_signal_session == signal_session
    assert sum(first.executed_target_weights.values()) == pytest.approx(1.0)
    assert all(row.executed_signal_session != signal_session for row in scenario.backtest.ledger[:221])


def test_missing_held_bar_fails_closed_instead_of_becoming_zero_return() -> None:
    config = _config()
    bars = _bars(missing=("A", 225))
    context = _context(bars, config)
    signal_session = pd.bdate_range("2023-01-02", periods=340)[220]

    with pytest.raises(MissingHeldReturnError, match="held symbol A"):
        run_scenario(context, {signal_session: {"A": 1.0}}, config=config)


def test_cost_stress_is_monotone_and_bootstrap_is_paired() -> None:
    config = _config()
    context = _context(_bars(), config)
    signals = {
        context.sessions[index]: {"A": 2.0, "B": 1.0}
        for index in range(220, len(context.sessions) - 1, 5)
    }
    rows = cost_sensitivity(
        context,
        signals,
        config,
        friction_bps=(0.0, 20.0, 40.0),
    )
    finals = [row["metrics"]["final_equity"] for row in rows]
    assert finals[0] > finals[1] > finals[2]

    scenario = run_scenario(context, signals, config=config)
    bootstrap = paired_moving_block_bootstrap(
        scenario,
        scenario,
        block_length=5,
        repetitions=100,
        seed=7,
        realized_volatility_control=True,
    )
    assert bootstrap["annualized_mean_excess"] == pytest.approx(0.0)
    assert bootstrap["probability_outperform"] == 0.0
    assert bootstrap["benchmark_return_scale"] == pytest.approx(1.0)
    control, metadata = paired_realized_volatility_control(scenario, scenario)
    assert metadata["comparison_basis"] == (
        "shared_session_ex_post_realized_volatility_control"
    )
    assert control["benchmark_control"].equals(control["strategy"])


def test_enabled_breadth_layer_requires_point_in_time_breadth() -> None:
    config = _config(breadth_exit=0.4, breadth_enter=0.5)
    bars = _bars()
    context = build_market_context(
        bars,
        candidate_symbols=("A", "B"),
        themes={"A": "ThemeA", "B": "ThemeB"},
        config=config,
    )
    with pytest.raises(ValueError, match="no point-in-time breadth"):
        run_scenario(
            context,
            {context.sessions[220]: {"A": 1.0}},
            config=config,
        )
