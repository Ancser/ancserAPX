from dataclasses import FrozenInstanceError
from datetime import date, timedelta
import json
from pathlib import Path

import pytest
import pandas as pd

from research_v2.letf_universe import (
    INDEX_REGIMES,
    SEED30_REGISTRY,
    SEED30_TICKERS,
    TICKER_REGIMES,
    has_contiguous_global_warmup,
    instrument_for_ticker,
    pit_eligible_tickers,
    proxy_at,
    registry_records,
    target_leverage_at,
    tradable_ticker_at,
)
from research_v2.run_letf_rotation import _eligibility_matrix


EXPECTED_SEED30 = (
    "FNGU", "DPST", "SOXL", "KORU", "ERX", "TNA", "AGQ", "UTSL",
    "GUSH", "CWEB", "EURL", "EDC", "RETL", "BOIL", "TECL", "DFEN",
    "CURE", "YINN", "DRN", "FAS", "TQQQ", "LABU", "DUSL", "NAIL",
    "NUGT", "UPRO", "TPOR", "SPXL", "WANT", "MIDU",
)


def _sessions(start: str, count: int):
    first = date.fromisoformat(start)
    return tuple(first + timedelta(days=index) for index in range(count))


def test_registry_is_exact_frozen_image_seed_and_has_required_schema():
    assert SEED30_TICKERS == EXPECTED_SEED30
    assert tuple(item.ticker for item in SEED30_REGISTRY) == EXPECTED_SEED30
    assert len(SEED30_REGISTRY) == len({item.instrument_id for item in SEED30_REGISTRY}) == 30
    assert "TPOR" in SEED30_TICKERS
    assert "TMF" not in SEED30_TICKERS

    expected_fields = {
        "instrument_id", "ticker", "valid_from", "valid_to", "theme",
        "macro_bucket", "target_leverage", "structure", "proxy", "issuer",
        "core_vs_diagnostic", "source_url", "identity_valid_from",
    }
    records = registry_records()
    assert len(records) == 30
    assert all(set(record) == expected_fields for record in records)
    assert all(record["source_url"].startswith("https://") for record in records)

    expected_proxies = {
        "FNGU": "QQQ", "DPST": "KRE", "SOXL": "SOXX", "KORU": "EWY",
        "ERX": "XLE", "TNA": "IWM", "AGQ": "SLV", "UTSL": "XLU",
        "GUSH": "XOP", "CWEB": "KWEB", "EURL": "VGK", "EDC": "EEM",
        "RETL": "XRT", "BOIL": "UNG", "TECL": "XLK", "DFEN": "ITA",
        "CURE": "XLV", "YINN": "FXI", "DRN": "VNQ", "FAS": "XLF",
        "TQQQ": "QQQ", "LABU": "XBI", "DUSL": "XLI", "NAIL": "ITB",
        "NUGT": "GDX", "UPRO": "SPY", "TPOR": "IYT", "SPXL": "SPY",
        "WANT": "XLY", "MIDU": "MDY",
    }
    assert {item.ticker: item.proxy for item in SEED30_REGISTRY} == expected_proxies
    themes = {item.ticker: item.theme for item in SEED30_REGISTRY}
    assert {themes[ticker] for ticker in ("FNGU", "SOXL", "TECL", "TQQQ")} == {"Technology"}
    assert {themes[ticker] for ticker in ("DPST", "FAS")} == {"Financials"}
    assert {themes[ticker] for ticker in ("ERX", "GUSH")} == {"Energy"}
    assert {themes[ticker] for ticker in ("CWEB", "YINN")} == {"China"}
    assert {themes[ticker] for ticker in ("UPRO", "SPXL")} == {"US large cap"}
    assert {themes[ticker] for ticker in ("RETL", "WANT")} == {"Consumer cyclical"}

    with pytest.raises(FrozenInstanceError):
        SEED30_REGISTRY[0].ticker = "TMF"


def test_fngu_current_identity_never_stitches_old_same_ticker_rows():
    fngu = next(item for item in SEED30_REGISTRY if item.ticker == "FNGU")
    assert fngu.instrument_id == "FNGU_BMO_ETN_20250220"
    assert fngu.identity_valid_from == "2025-02-20"
    assert fngu.valid_from == "2025-06-24"
    assert instrument_for_ticker("FNGU", "2025-02-19") is None
    assert instrument_for_ticker("fngu", "2025-02-20") is None
    assert instrument_for_ticker("fngb", "2025-02-20") == fngu
    assert instrument_for_ticker("FNGB", "2025-06-23") == fngu
    assert instrument_for_ticker("FNGB", "2025-06-24") is None
    assert instrument_for_ticker("FNGU", "2025-06-23") is None
    assert instrument_for_ticker("FNGU", "2025-06-24") == fngu
    assert tradable_ticker_at(fngu, "2025-02-19") is None
    assert tradable_ticker_at(fngu, "2025-02-20") == "FNGB"
    assert tradable_ticker_at(fngu, "2025-06-23") == "FNGB"
    assert tradable_ticker_at(fngu, "2025-06-24") == "FNGU"
    assert target_leverage_at(fngu, "2025-02-20") == 3.0
    assert proxy_at(fngu, "2025-02-20") == "QQQ"
    assert [(item.ticker, item.valid_from, item.valid_to) for item in TICKER_REGIMES[fngu.instrument_id]] == [
        ("FNGB", "2025-02-20", "2025-06-23"),
        ("FNGU", "2025-06-24", None),
    ]

    sessions = (
        date(2025, 2, 18),
        date(2025, 2, 19),
        date(2025, 2, 20),
        date(2025, 2, 21),
        date(2025, 2, 24),
    )
    # A provider may normalize the new note's FNGB rows to its current FNGU
    # symbol.  Ticker-keyed PIT data must reject that backwards projection.
    normalized_current_symbol_only = {"FNGU": sessions}
    assert "FNGU" not in pit_eligible_tickers(
        "2025-02-24", sessions, normalized_current_symbol_only, warmup_sessions=3,
    )
    # The same identity is eligible when the historically correct alias exists.
    assert "FNGU" in pit_eligible_tickers(
        "2025-02-24", sessions, {"FNGB": sessions}, warmup_sessions=3,
    )


def test_fngu_alias_boundary_requires_both_tickers_and_ignores_future_rows():
    sessions = tuple(pd.to_datetime([
        "2025-06-20", "2025-06-23", "2025-06-24", "2025-06-25",
    ]).date)

    # A two-session warm-up legitimately crosses the rename only when both
    # ticker regimes are supplied.
    assert "FNGU" in pit_eligible_tickers(
        "2025-06-24",
        sessions,
        {"FNGB": sessions[:2], "FNGU": sessions[2:]},
        warmup_sessions=2,
    )
    assert "FNGU" not in pit_eligible_tickers(
        "2025-06-24",
        sessions,
        {"FNGU": sessions},
        warmup_sessions=2,
    )

    # Mutating or relabelling rows after the decision cutoff cannot change it.
    cutoff_sessions = sessions[:2]
    before = pit_eligible_tickers(
        "2025-06-23",
        sessions,
        {"FNGB": cutoff_sessions, "FNGU": sessions[2:]},
        warmup_sessions=2,
    )
    after = pit_eligible_tickers(
        "2025-06-23",
        sessions + (date(2035, 1, 2),),
        {"FNGB": cutoff_sessions, "FNGU": sessions + (date(2035, 1, 2),)},
        warmup_sessions=2,
    )
    assert before == after
    assert "FNGU" in before


def test_runner_fails_closed_before_fngu_current_ticker_start_without_fngb_bars():
    fngu = next(item for item in SEED30_REGISTRY if item.ticker == "FNGU")
    sessions = pd.to_datetime([
        "2025-06-20", "2025-06-23", "2025-06-24", "2025-06-25",
    ])
    # Simulate a vendor file normalized entirely to today's FNGU symbol.  The
    # runner only consumes current-symbol bars, so valid_from must keep all
    # pre-rename sessions out and rebuild its warm-up after the boundary.
    bars = pd.DataFrame({"symbol": "FNGU", "timestamp": sessions})
    eligibility = _eligibility_matrix(bars, sessions, (fngu,), warmup_sessions=2)
    assert eligibility["FNGU"].tolist() == [False, False, False, True]


def test_warmup_requires_every_trailing_global_session_without_forward_fill():
    upro = next(item for item in SEED30_REGISTRY if item.ticker == "UPRO")
    sessions = _sessions("2024-01-02", 8)
    assert has_contiguous_global_warmup(
        upro, sessions[-1], sessions, sessions[-5:], warmup_sessions=5,
    )
    missing_middle = tuple(session for session in sessions[-5:] if session != sessions[-3])
    assert not has_contiguous_global_warmup(
        upro, sessions[-1], sessions, missing_middle, warmup_sessions=5,
    )


def test_pit_eligibility_is_invariant_to_future_session_mutation():
    cutoff = date(2024, 1, 8)
    past = _sessions("2024-01-02", 7)
    future_a = (date(2024, 1, 9), date(2024, 1, 10))
    future_b = (date(2035, 6, 1), date(2040, 1, 1))
    observed = {ticker: past for ticker in EXPECTED_SEED30}

    first = pit_eligible_tickers(
        cutoff,
        past + future_a,
        {ticker: values + future_a for ticker, values in observed.items()},
        warmup_sessions=5,
    )
    second = pit_eligible_tickers(
        cutoff,
        past + future_b,
        {ticker: values + future_b for ticker, values in observed.items()},
        warmup_sessions=5,
    )
    assert first == second
    assert "FNGU" not in first  # its current identity does not yet exist.
    assert "UPRO" in first


def test_leverage_and_index_regimes_are_resolved_point_in_time():
    erx = next(item for item in SEED30_REGISTRY if item.ticker == "ERX")
    soxl = next(item for item in SEED30_REGISTRY if item.ticker == "SOXL")
    assert target_leverage_at(erx, "2020-03-31") == 3.0
    assert target_leverage_at(erx, "2020-04-01") == 2.0
    assert target_leverage_at(erx, "2026-01-02") == erx.target_leverage
    assert proxy_at(soxl, "2021-08-24") == "SOXX"
    assert proxy_at(soxl, "2021-08-25") == "SOXX"
    regimes = INDEX_REGIMES[soxl.instrument_id]
    assert [(item.valid_from, item.valid_to) for item in regimes] == [
        ("2010-03-11", "2021-08-24"),
        ("2021-08-25", None),
    ]
    assert regimes[0].source_url != regimes[1].source_url
    for ticker, boundary in (
        ("FAS", "2022-08-01"),
        ("TPOR", "2022-08-01"),
        ("NUGT", "2025-09-19"),
    ):
        instrument = next(item for item in SEED30_REGISTRY if item.ticker == ticker)
        event_regimes = INDEX_REGIMES[instrument.instrument_id]
        assert len(event_regimes) == 2
        assert event_regimes[1].valid_from == boundary
        assert proxy_at(instrument, boundary) == instrument.proxy


def test_seed30_config_is_isolated_frozen_and_requires_data_provenance():
    path = Path(__file__).resolve().parents[1] / "research_v2" / "configs" / "letf_rotation_seed30.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    assert config["research_only"] is True
    assert config["live_config_impact"] == "none"
    assert tuple(config["universe"]["tickers"]) == EXPECTED_SEED30
    assert config["universe"]["frozen"] is True
    assert config["universe"]["warmup_sessions"] == 252
    assert config["portfolio"]["gross_target_grid"] == [0.75, 1.0]
    assert config["portfolio"]["external_margin_leverage_allowed"] is False
    assert config["acceptance"]["require_positive_worst_fold"] is True

    required = config["data_audit"]["required_before_run"]
    assert required == [
        "provider", "feed", "adjustment", "retrieved_at_utc", "snapshot_id",
        "source_manifest_sha256",
    ]
    assert all(config["data_audit"][field] is None for field in required)
    assert config["data_audit"]["ticker_identity_stitching"] == "forbidden"
    assert config["data_audit"]["allow_missing_held_bars"] is False
