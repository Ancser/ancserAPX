"""Frozen, point-in-time identities for the research-only Seed-30 LETF universe.

This module is deliberately pure.  It does not import the production universe,
the live configuration, a broker adapter, or a data store.  A ticker is not an
instrument identity: the same ticker can be re-issued with a different security
behind it.  ``instrument_id`` and inclusive validity windows are therefore the
keys used by the point-in-time layer.

Eligibility is stricter than "a row exists".  An instrument needs one bar on
every trailing *global market session* in the requested warm-up window.  Missing
bars are never forward-filled and future rows are ignored, which makes an
eligibility decision invariant to future data mutations.  ``ticker`` is the
current Seed-30 label; ticker regimes preserve the symbol that was actually
tradable on each historical session.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime
from types import MappingProxyType
from typing import Iterable, Mapping, Optional, Sequence, Tuple, Union


DateLike = Union[str, date, datetime]


@dataclass(frozen=True)
class LETFInstrument:
    instrument_id: str
    ticker: str
    valid_from: str
    valid_to: Optional[str]
    theme: str
    macro_bucket: str
    target_leverage: float
    structure: str
    proxy: str
    issuer: str
    core_vs_diagnostic: str
    source_url: str
    # Usually identical to valid_from.  It differs when the current Seed-30
    # ticker began after the security itself (the 2025 FNGB -> FNGU rename).
    identity_valid_from: Optional[str] = None


@dataclass(frozen=True)
class TickerRegime:
    instrument_id: str
    ticker: str
    valid_from: str
    valid_to: Optional[str]
    source_url: str


@dataclass(frozen=True)
class LeverageRegime:
    instrument_id: str
    valid_from: str
    valid_to: Optional[str]
    target_leverage: float
    source_url: str


@dataclass(frozen=True)
class IndexRegime:
    instrument_id: str
    valid_from: str
    valid_to: Optional[str]
    proxy: str
    source_url: str


_DIREXION = "https://www.direxion.com/product/"
_PROSHARES = "https://www.proshares.com/our-etfs/leveraged-and-inverse/"


SEED30_TICKERS: Tuple[str, ...] = (
    "FNGU", "DPST", "SOXL", "KORU", "ERX", "TNA", "AGQ", "UTSL",
    "GUSH", "CWEB", "EURL", "EDC", "RETL", "BOIL", "TECL", "DFEN",
    "CURE", "YINN", "DRN", "FAS", "TQQQ", "LABU", "DUSL", "NAIL",
    "NUGT", "UPRO", "TPOR", "SPXL", "WANT", "MIDU",
)


SEED30_REGISTRY: Tuple[LETFInstrument, ...] = (
    LETFInstrument(
        "FNGU_BMO_ETN_20250220", "FNGU", "2025-06-24", None,
        "Technology", "US Equity", 3.0, "ETN",
        "QQQ", "Bank of Montreal (MicroSectors)", "diagnostic",
        "https://microsectors.com/fang/",
        "2025-02-20",
    ),
    LETFInstrument(
        "DPST_DIREXION_ETF_20150819", "DPST", "2015-08-19", None,
        "Financials", "US Equity", 3.0, "ETF",
        "KRE", "Direxion", "core",
        _DIREXION + "daily-regional-banks-bull-3x-etf",
    ),
    LETFInstrument(
        "SOXL_DIREXION_ETF_20100311", "SOXL", "2010-03-11", None,
        "Technology", "US Equity", 3.0, "ETF",
        "SOXX", "Direxion", "core",
        _DIREXION + "daily-semiconductor-bull-bear-3x-etfs",
    ),
    LETFInstrument(
        "KORU_DIREXION_ETF_20130410", "KORU", "2013-04-10", None,
        "South Korea", "International Equity", 3.0, "ETF",
        "EWY", "Direxion", "diagnostic",
        _DIREXION + "daily-msci-south-korea-bull-3x-etf",
    ),
    LETFInstrument(
        "ERX_DIREXION_ETF_20081106", "ERX", "2008-11-06", None,
        "Energy", "US Equity", 2.0, "ETF",
        "XLE", "Direxion", "core",
        _DIREXION + "daily-energy-bull-bear-2x-etfs",
    ),
    LETFInstrument(
        "TNA_DIREXION_ETF_20081105", "TNA", "2008-11-05", None,
        "US small cap", "US Equity", 3.0, "ETF",
        "IWM", "Direxion", "core",
        _DIREXION + "daily-small-cap-bull-bear-3x-etfs",
    ),
    LETFInstrument(
        "AGQ_PROSHARES_ETF_20081201", "AGQ", "2008-12-01", None,
        "Silver", "Commodity", 2.0, "ETF",
        "SLV", "ProShares", "diagnostic",
        _PROSHARES + "agq",
    ),
    LETFInstrument(
        "UTSL_DIREXION_ETF_20170503", "UTSL", "2017-05-03", None,
        "Utilities", "US Equity", 3.0, "ETF",
        "XLU", "Direxion", "core",
        _DIREXION + "daily-utilities-bull-3x-etf",
    ),
    LETFInstrument(
        "GUSH_DIREXION_ETF_20150528", "GUSH", "2015-05-28", None,
        "Energy", "US Equity", 2.0, "ETF",
        "XOP",
        "Direxion", "core",
        _DIREXION + "daily-sp-oil-gas-exp-prod-bull-bear-2x-etfs",
    ),
    LETFInstrument(
        "CWEB_DIREXION_ETF_20161102", "CWEB", "2016-11-02", None,
        "China", "International Equity", 2.0, "ETF",
        "KWEB", "Direxion", "diagnostic",
        _DIREXION + "daily-csi-china-internet-index-bull-2x-etf",
    ),
    LETFInstrument(
        "EURL_DIREXION_ETF_20140122", "EURL", "2014-01-22", None,
        "Developed Europe", "International Equity", 3.0, "ETF",
        "VGK", "Direxion", "diagnostic",
        _DIREXION + "daily-ftse-europe-bull-3x-etf",
    ),
    LETFInstrument(
        "EDC_DIREXION_ETF_20081217", "EDC", "2008-12-17", None,
        "Emerging markets", "International Equity", 3.0, "ETF",
        "EEM", "Direxion", "diagnostic",
        _DIREXION + "daily-msci-emerging-markets-bull-bear-3x-etfs",
    ),
    LETFInstrument(
        "RETL_DIREXION_ETF_20100714", "RETL", "2010-07-14", None,
        "Consumer cyclical", "US Equity", 3.0, "ETF",
        "XRT", "Direxion", "core",
        _DIREXION + "daily-retail-bull-3x-etf",
    ),
    LETFInstrument(
        "BOIL_PROSHARES_ETF_20111004", "BOIL", "2011-10-04", None,
        "Natural gas", "Commodity", 2.0, "ETF",
        "UNG", "ProShares", "diagnostic",
        _PROSHARES + "boil",
    ),
    LETFInstrument(
        "TECL_DIREXION_ETF_20081217", "TECL", "2008-12-17", None,
        "Technology", "US Equity", 3.0, "ETF",
        "XLK", "Direxion", "core",
        _DIREXION + "daily-technology-bull-bear-3x-etfs",
    ),
    LETFInstrument(
        "DFEN_DIREXION_ETF_20170503", "DFEN", "2017-05-03", None,
        "Aerospace and defense", "US Equity", 3.0, "ETF",
        "ITA", "Direxion", "core",
        _DIREXION + "daily-aerospace-defense-bull-3x-etf",
    ),
    LETFInstrument(
        "CURE_DIREXION_ETF_20110615", "CURE", "2011-06-15", None,
        "Health care", "US Equity", 3.0, "ETF",
        "XLV", "Direxion", "core",
        _DIREXION + "daily-healthcare-bull-3x-etf",
    ),
    LETFInstrument(
        "YINN_DIREXION_ETF_20091203", "YINN", "2009-12-03", None,
        "China", "International Equity", 3.0, "ETF",
        "FXI", "Direxion", "diagnostic",
        _DIREXION + "daily-ftse-china-bull-bear-3x-etfs",
    ),
    LETFInstrument(
        "DRN_DIREXION_ETF_20090716", "DRN", "2009-07-16", None,
        "Real estate", "US Equity", 3.0, "ETF",
        "VNQ", "Direxion", "core",
        _DIREXION + "daily-real-estate-bull-bear-3x-etfs",
    ),
    LETFInstrument(
        "FAS_DIREXION_ETF_20081106", "FAS", "2008-11-06", None,
        "Financials", "US Equity", 3.0, "ETF",
        "XLF", "Direxion", "core",
        _DIREXION + "daily-financial-bull-bear-3x-etfs",
    ),
    LETFInstrument(
        "TQQQ_PROSHARES_ETF_20100209", "TQQQ", "2010-02-09", None,
        "Technology", "US Equity", 3.0, "ETF",
        "QQQ", "ProShares", "core",
        _PROSHARES + "tqqq",
    ),
    LETFInstrument(
        "LABU_DIREXION_ETF_20150528", "LABU", "2015-05-28", None,
        "Biotechnology", "US Equity", 3.0, "ETF",
        "XBI", "Direxion", "core",
        _DIREXION + "daily-sp-biotech-bull-bear-3x-etfs",
    ),
    LETFInstrument(
        "DUSL_DIREXION_ETF_20170503", "DUSL", "2017-05-03", None,
        "Industrials", "US Equity", 3.0, "ETF",
        "XLI", "Direxion", "core",
        _DIREXION + "daily-industrials-bull-3x-etf",
    ),
    LETFInstrument(
        "NAIL_DIREXION_ETF_20150819", "NAIL", "2015-08-19", None,
        "Homebuilders and supplies", "US Equity", 3.0, "ETF",
        "ITB", "Direxion", "core",
        _DIREXION + "daily-homebuilders-supplies-bull-3x-etf",
    ),
    LETFInstrument(
        "NUGT_DIREXION_ETF_20101208", "NUGT", "2010-12-08", None,
        "Gold miners", "Commodity Equity", 2.0, "ETF",
        "GDX", "Direxion", "diagnostic",
        _DIREXION + "daily-gold-miners-bull-bear-2x-etfs",
    ),
    LETFInstrument(
        "UPRO_PROSHARES_ETF_20090623", "UPRO", "2009-06-23", None,
        "US large cap", "US Equity", 3.0, "ETF",
        "SPY", "ProShares", "core",
        _PROSHARES + "upro",
    ),
    LETFInstrument(
        "TPOR_DIREXION_ETF_20170503", "TPOR", "2017-05-03", None,
        "Transportation", "US Equity", 3.0, "ETF",
        "IYT", "Direxion", "core",
        _DIREXION + "daily-transportation-bull-3x-etf",
    ),
    LETFInstrument(
        "SPXL_DIREXION_ETF_20081105", "SPXL", "2008-11-05", None,
        "US large cap", "US Equity", 3.0, "ETF",
        "SPY", "Direxion", "core",
        _DIREXION + "daily-sp-500-bull-bear-3x-etfs",
    ),
    LETFInstrument(
        "WANT_DIREXION_ETF_20181129", "WANT", "2018-11-29", None,
        "Consumer cyclical", "US Equity", 3.0, "ETF",
        "XLY", "Direxion", "core",
        _DIREXION + "daily-consumer-discretionary-bull-3x-etf",
    ),
    LETFInstrument(
        "MIDU_DIREXION_ETF_20090108", "MIDU", "2009-01-08", None,
        "US mid cap", "US Equity", 3.0, "ETF",
        "MDY", "Direxion", "core",
        _DIREXION + "daily-mid-cap-bull-bear-3x-etfs",
    ),
)


def _as_date(value: DateLike) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value[:10])
        except ValueError as exc:
            raise ValueError(f"invalid ISO date: {value!r}") from exc
    raise TypeError(f"unsupported date value: {type(value).__name__}")


def _inside(value: date, valid_from: str, valid_to: Optional[str]) -> bool:
    return _as_date(valid_from) <= value and (
        valid_to is None or value <= _as_date(valid_to)
    )


def _identity_valid_from(instrument: LETFInstrument) -> str:
    return instrument.identity_valid_from or instrument.valid_from


def _inside_identity(value: date, instrument: LETFInstrument) -> bool:
    return _inside(value, _identity_valid_from(instrument), instrument.valid_to)


_FNGU_LAUNCH_SOURCE = (
    "https://microsectors.com/insights/"
    "bmo-announces-upcoming-redemption-and-ticker-symbol-change-for-"
    "microsectorstm-fang-index-3x-leveraged-etns-ticker-fngu-as-well-as-"
    "the-launch-of-a-new-microsectorstm-exchange-traded-n/"
)
_FNGU_RENAME_SOURCE = (
    "https://microsectors.com/insights/"
    "bmo-announces-upcoming-ticker-symbol-change-for-microsectors-fang-"
    "3x-leveraged-etns/"
)


def _ticker_regimes() -> Mapping[str, Tuple[TickerRegime, ...]]:
    result: dict[str, Tuple[TickerRegime, ...]] = {}
    for instrument in SEED30_REGISTRY:
        if instrument.instrument_id == "FNGU_BMO_ETN_20250220":
            result[instrument.instrument_id] = (
                TickerRegime(
                    instrument.instrument_id,
                    "FNGB",
                    "2025-02-20",
                    "2025-06-23",
                    _FNGU_LAUNCH_SOURCE,
                ),
                TickerRegime(
                    instrument.instrument_id,
                    "FNGU",
                    "2025-06-24",
                    None,
                    _FNGU_RENAME_SOURCE,
                ),
            )
        else:
            result[instrument.instrument_id] = (
                TickerRegime(
                    instrument.instrument_id,
                    instrument.ticker,
                    instrument.valid_from,
                    instrument.valid_to,
                    instrument.source_url,
                ),
            )
    return MappingProxyType(result)


TICKER_REGIMES: Mapping[str, Tuple[TickerRegime, ...]] = _ticker_regimes()


def _instrument_map_by_ticker() -> Mapping[str, Tuple[LETFInstrument, ...]]:
    grouped: dict[str, list[LETFInstrument]] = {}
    for instrument in SEED30_REGISTRY:
        for regime in TICKER_REGIMES[instrument.instrument_id]:
            grouped.setdefault(regime.ticker, []).append(instrument)
    return MappingProxyType({
        ticker: tuple(sorted(items, key=_identity_valid_from))
        for ticker, items in grouped.items()
    })


INSTRUMENTS_BY_ID: Mapping[str, LETFInstrument] = MappingProxyType({
    item.instrument_id: item for item in SEED30_REGISTRY
})
INSTRUMENTS_BY_TICKER: Mapping[str, Tuple[LETFInstrument, ...]] = (
    _instrument_map_by_ticker()
)


_LEVERAGE_OVERRIDES: Mapping[str, Tuple[LeverageRegime, ...]] = {
    # Direxion made these 3X -> 2X changes effective after the 2020-03-31
    # close.  Explicit regimes prevent a current product label from being
    # projected backwards through a materially different objective.
    "ERX_DIREXION_ETF_20081106": (
        LeverageRegime(
            "ERX_DIREXION_ETF_20081106", "2008-11-06", "2020-03-31", 3.0,
            "https://www.direxion.com/uploads/Change-in-Investment-Objectives-and-Strategies-of-Ten-Daily-Leveraged-and-Daily-Inverse-Leveraged-Funds.pdf",
        ),
        LeverageRegime(
            "ERX_DIREXION_ETF_20081106", "2020-04-01", None, 2.0,
            _DIREXION + "daily-energy-bull-bear-2x-etfs",
        ),
    ),
    "GUSH_DIREXION_ETF_20150528": (
        LeverageRegime(
            "GUSH_DIREXION_ETF_20150528", "2015-05-28", "2020-03-31", 3.0,
            "https://www.direxion.com/uploads/Change-in-Investment-Objectives-and-Strategies-of-Ten-Daily-Leveraged-and-Daily-Inverse-Leveraged-Funds.pdf",
        ),
        LeverageRegime(
            "GUSH_DIREXION_ETF_20150528", "2020-04-01", None, 2.0,
            _DIREXION + "daily-sp-oil-gas-exp-prod-bull-bear-2x-etfs",
        ),
    ),
    "NUGT_DIREXION_ETF_20101208": (
        LeverageRegime(
            "NUGT_DIREXION_ETF_20101208", "2010-12-08", "2020-03-31", 3.0,
            "https://www.direxion.com/uploads/Change-in-Investment-Objectives-and-Strategies-of-Ten-Daily-Leveraged-and-Daily-Inverse-Leveraged-Funds.pdf",
        ),
        LeverageRegime(
            "NUGT_DIREXION_ETF_20101208", "2020-04-01", None, 2.0,
            _DIREXION + "daily-gold-miners-bull-bear-2x-etfs",
        ),
    ),
}


def _leverage_regimes() -> Mapping[str, Tuple[LeverageRegime, ...]]:
    result: dict[str, Tuple[LeverageRegime, ...]] = {}
    for instrument in SEED30_REGISTRY:
        result[instrument.instrument_id] = _LEVERAGE_OVERRIDES.get(
            instrument.instrument_id,
            (
                LeverageRegime(
                    instrument.instrument_id,
                    _identity_valid_from(instrument),
                    instrument.valid_to,
                    instrument.target_leverage,
                    instrument.source_url,
                ),
            ),
        )
    return MappingProxyType(result)


LEVERAGE_REGIMES: Mapping[str, Tuple[LeverageRegime, ...]] = _leverage_regimes()


_INDEX_OVERRIDES: Mapping[str, Tuple[IndexRegime, ...]] = {
    "SOXL_DIREXION_ETF_20100311": (
        IndexRegime(
            "SOXL_DIREXION_ETF_20100311", "2010-03-11", "2021-08-24",
            "SOXX",
            "https://www.direxion.com/uploads/Direxion-Changes-Index-for-Semiconductor-ETFs.pdf",
        ),
        IndexRegime(
            "SOXL_DIREXION_ETF_20100311", "2021-08-25", None,
            "SOXX",
            _DIREXION + "daily-semiconductor-bull-bear-3x-etfs",
        ),
    ),
    "FAS_DIREXION_ETF_20081106": (
        IndexRegime(
            "FAS_DIREXION_ETF_20081106", "2008-11-06", "2022-07-31",
            "XLF",
            "https://www.direxion.com/uploads/Direxion-Changes-Index-for-Financial-and-Transportation-Leveraged-ETFs-FAS-FAZ-TPOR.pdf",
        ),
        IndexRegime(
            "FAS_DIREXION_ETF_20081106", "2022-08-01", None,
            "XLF", _DIREXION + "daily-financial-bull-bear-3x-etfs",
        ),
    ),
    "TPOR_DIREXION_ETF_20170503": (
        IndexRegime(
            "TPOR_DIREXION_ETF_20170503", "2017-05-03", "2022-07-31",
            "IYT",
            "https://www.direxion.com/uploads/Direxion-Changes-Index-for-Financial-and-Transportation-Leveraged-ETFs-FAS-FAZ-TPOR.pdf",
        ),
        IndexRegime(
            "TPOR_DIREXION_ETF_20170503", "2022-08-01", None,
            "IYT", _DIREXION + "daily-transportation-bull-3x-etf",
        ),
    ),
    "NUGT_DIREXION_ETF_20101208": (
        IndexRegime(
            "NUGT_DIREXION_ETF_20101208", "2010-12-08", "2025-09-18",
            "GDX", "https://www.direxion.com/press-release/new-index-nugt-dust",
        ),
        IndexRegime(
            "NUGT_DIREXION_ETF_20101208", "2025-09-19", None,
            "GDX", _DIREXION + "daily-gold-miners-bull-bear-2x-etfs",
        ),
    ),
}


def _index_regimes() -> Mapping[str, Tuple[IndexRegime, ...]]:
    result: dict[str, Tuple[IndexRegime, ...]] = {}
    for instrument in SEED30_REGISTRY:
        result[instrument.instrument_id] = _INDEX_OVERRIDES.get(
            instrument.instrument_id,
            (
                IndexRegime(
                    instrument.instrument_id,
                    _identity_valid_from(instrument),
                    instrument.valid_to,
                    instrument.proxy,
                    instrument.source_url,
                ),
            ),
        )
    return MappingProxyType(result)


INDEX_REGIMES: Mapping[str, Tuple[IndexRegime, ...]] = _index_regimes()


def _resolve_instrument(
    instrument_or_id: Union[LETFInstrument, str],
) -> LETFInstrument:
    if isinstance(instrument_or_id, LETFInstrument):
        return instrument_or_id
    try:
        return INSTRUMENTS_BY_ID[str(instrument_or_id)]
    except KeyError as exc:
        raise KeyError(f"unknown LETF instrument_id: {instrument_or_id!r}") from exc


def instrument_for_ticker(ticker: str, as_of: DateLike) -> Optional[LETFInstrument]:
    """Return the identity traded under ``ticker`` at ``as_of``.

    This resolves historical aliases, so the new 2025 BMO note is ``FNGB``
    before 2025-06-24 and ``FNGU`` from that date onward.
    """

    value = _as_date(as_of)
    normalized = str(ticker).upper()
    candidates = []
    for item in INSTRUMENTS_BY_TICKER.get(normalized, ()):
        if any(
            regime.ticker == normalized
            and _inside(value, regime.valid_from, regime.valid_to)
            for regime in TICKER_REGIMES[item.instrument_id]
        ):
            candidates.append(item)
    if len(candidates) > 1:
        raise ValueError(f"overlapping identities for ticker {ticker!r} at {value}")
    return candidates[0] if candidates else None


def tradable_ticker_at(
    instrument_or_id: Union[LETFInstrument, str], as_of: DateLike
) -> Optional[str]:
    """Return the symbol actually tradable for an identity at ``as_of``."""

    instrument = _resolve_instrument(instrument_or_id)
    value = _as_date(as_of)
    if not _inside_identity(value, instrument):
        return None
    matches = [
        regime for regime in TICKER_REGIMES[instrument.instrument_id]
        if _inside(value, regime.valid_from, regime.valid_to)
    ]
    if len(matches) > 1:
        raise ValueError(f"overlapping ticker regimes for {instrument.instrument_id}")
    return matches[0].ticker if matches else None


def target_leverage_at(
    instrument_or_id: Union[LETFInstrument, str], as_of: DateLike
) -> Optional[float]:
    instrument = _resolve_instrument(instrument_or_id)
    value = _as_date(as_of)
    if not _inside_identity(value, instrument):
        return None
    matches = [
        regime for regime in LEVERAGE_REGIMES[instrument.instrument_id]
        if _inside(value, regime.valid_from, regime.valid_to)
    ]
    if len(matches) > 1:
        raise ValueError(f"overlapping leverage regimes for {instrument.instrument_id}")
    return float(matches[0].target_leverage) if matches else None


def proxy_at(
    instrument_or_id: Union[LETFInstrument, str], as_of: DateLike
) -> Optional[str]:
    instrument = _resolve_instrument(instrument_or_id)
    value = _as_date(as_of)
    if not _inside_identity(value, instrument):
        return None
    matches = [
        regime for regime in INDEX_REGIMES[instrument.instrument_id]
        if _inside(value, regime.valid_from, regime.valid_to)
    ]
    if len(matches) > 1:
        raise ValueError(f"overlapping index regimes for {instrument.instrument_id}")
    return str(matches[0].proxy) if matches else None


def has_contiguous_global_warmup(
    instrument: LETFInstrument,
    as_of: DateLike,
    global_sessions: Sequence[DateLike],
    observed_sessions: Iterable[DateLike],
    warmup_sessions: int = 252,
) -> bool:
    """Require bars on every trailing global session known at ``as_of``.

    Only global and observed sessions on or before ``as_of`` are inspected.
    Rows before the instrument identity's ``valid_from`` are discarded even if
    they carry the same ticker, which is essential for the 2025 FNGU identity.
    """

    if int(warmup_sessions) <= 0:
        raise ValueError("warmup_sessions must be positive")
    cutoff = _as_date(as_of)
    if not _inside_identity(cutoff, instrument):
        return False

    known_global = sorted({
        _as_date(session) for session in global_sessions
        if _as_date(session) <= cutoff
    })
    if len(known_global) < int(warmup_sessions):
        return False
    trailing = tuple(known_global[-int(warmup_sessions):])
    if not all(_inside_identity(session, instrument) for session in trailing):
        return False

    known_observed = {
        _as_date(session) for session in observed_sessions
        if _as_date(session) <= cutoff
        and _inside_identity(_as_date(session), instrument)
    }
    return all(session in known_observed for session in trailing)


def _identity_observed_sessions(
    instrument: LETFInstrument,
    observed_sessions_by_instrument: Mapping[str, Iterable[DateLike]],
) -> Tuple[date, ...]:
    """Resolve observations without projecting a current ticker backwards.

    An explicit ``instrument_id`` key is already identity-qualified.  Otherwise
    each ticker-keyed row is accepted only inside that ticker's own regime.  In
    particular, provider-normalized pre-rename rows under an ``FNGU`` key cannot
    stand in for the sessions that were actually published as ``FNGB``.
    """

    identity_rows = observed_sessions_by_instrument.get(instrument.instrument_id)
    if identity_rows is not None:
        return tuple(_as_date(session) for session in identity_rows)

    observed: set[date] = set()
    for regime in TICKER_REGIMES[instrument.instrument_id]:
        for session in observed_sessions_by_instrument.get(regime.ticker, ()):
            value = _as_date(session)
            if _inside(value, regime.valid_from, regime.valid_to):
                observed.add(value)
    return tuple(sorted(observed))


def pit_eligible_instruments(
    as_of: DateLike,
    global_sessions: Sequence[DateLike],
    observed_sessions_by_instrument: Mapping[str, Iterable[DateLike]],
    warmup_sessions: int = 252,
    registry: Sequence[LETFInstrument] = SEED30_REGISTRY,
) -> Tuple[LETFInstrument, ...]:
    """Return active identities with a contiguous point-in-time warm-up.

    ``instrument_id`` keys are preferred.  Ticker-key fallbacks are resolved by
    their historical regimes.  This both joins a legitimate rename when both
    aliases are present and rejects current-symbol-normalized history when the
    historical alias is absent.
    """

    eligible = []
    for instrument in registry:
        observed = _identity_observed_sessions(
            instrument, observed_sessions_by_instrument
        )
        if (
            target_leverage_at(instrument, as_of) is not None
            and proxy_at(instrument, as_of) is not None
            and has_contiguous_global_warmup(
                instrument,
                as_of,
                global_sessions,
                observed,
                warmup_sessions=warmup_sessions,
            )
        ):
            eligible.append(instrument)
    return tuple(eligible)


def pit_eligible_tickers(
    as_of: DateLike,
    global_sessions: Sequence[DateLike],
    observed_sessions_by_instrument: Mapping[str, Iterable[DateLike]],
    warmup_sessions: int = 252,
    registry: Sequence[LETFInstrument] = SEED30_REGISTRY,
) -> Tuple[str, ...]:
    return tuple(
        item.ticker
        for item in pit_eligible_instruments(
            as_of,
            global_sessions,
            observed_sessions_by_instrument,
            warmup_sessions=warmup_sessions,
            registry=registry,
        )
    )


def registry_records() -> Tuple[dict, ...]:
    """Return detached JSON-serialisable registry records for artifacts."""

    return tuple(asdict(item) for item in SEED30_REGISTRY)


def _validate_registry() -> None:
    if tuple(item.ticker for item in SEED30_REGISTRY) != SEED30_TICKERS:
        raise ValueError("Seed-30 registry order differs from the frozen ticker list")
    if len(set(SEED30_TICKERS)) != 30:
        raise ValueError("Seed-30 tickers must be unique")
    if len(INSTRUMENTS_BY_ID) != len(SEED30_REGISTRY):
        raise ValueError("instrument_id values must be unique")
    for instrument in SEED30_REGISTRY:
        if instrument.ticker != instrument.ticker.upper():
            raise ValueError(f"ticker is not uppercase: {instrument.ticker}")
        if _as_date(_identity_valid_from(instrument)) > _as_date(instrument.valid_from):
            raise ValueError(
                f"identity begins after current ticker for {instrument.instrument_id}"
            )
        if instrument.structure not in {"ETF", "ETN"}:
            raise ValueError(f"unsupported structure for {instrument.instrument_id}")
        if instrument.core_vs_diagnostic not in {"core", "diagnostic"}:
            raise ValueError(f"unsupported research role for {instrument.instrument_id}")
        if instrument.target_leverage <= 0:
            raise ValueError(f"non-positive target leverage for {instrument.instrument_id}")
        if not instrument.source_url.startswith("https://"):
            raise ValueError(f"non-HTTPS source for {instrument.instrument_id}")
        if instrument.valid_to is not None and _as_date(instrument.valid_to) < _as_date(instrument.valid_from):
            raise ValueError(f"reversed validity window for {instrument.instrument_id}")
        ticker_regimes = TICKER_REGIMES[instrument.instrument_id]
        if ticker_regimes[0].valid_from != _identity_valid_from(instrument):
            raise ValueError(
                f"ticker history does not start with identity {instrument.instrument_id}"
            )
        if not any(
            regime.ticker == instrument.ticker
            and regime.valid_from == instrument.valid_from
            for regime in ticker_regimes
        ):
            raise ValueError(
                f"current ticker start is missing for {instrument.instrument_id}"
            )
        prior_to: Optional[date] = None
        for regime in ticker_regimes:
            if regime.ticker != regime.ticker.upper():
                raise ValueError(f"ticker regime is not uppercase: {regime.ticker}")
            if not regime.source_url.startswith("https://"):
                raise ValueError(
                    f"non-HTTPS ticker source for {instrument.instrument_id}"
                )
            start = _as_date(regime.valid_from)
            if prior_to is not None and start <= prior_to:
                raise ValueError(
                    f"overlapping ticker regimes for {instrument.instrument_id}"
                )
            prior_to = _as_date(regime.valid_to) if regime.valid_to else None


_validate_registry()


__all__ = [
    "INDEX_REGIMES",
    "INSTRUMENTS_BY_ID",
    "INSTRUMENTS_BY_TICKER",
    "LEVERAGE_REGIMES",
    "SEED30_REGISTRY",
    "SEED30_TICKERS",
    "TICKER_REGIMES",
    "IndexRegime",
    "LETFInstrument",
    "LeverageRegime",
    "TickerRegime",
    "has_contiguous_global_warmup",
    "instrument_for_ticker",
    "pit_eligible_instruments",
    "pit_eligible_tickers",
    "proxy_at",
    "registry_records",
    "target_leverage_at",
    "tradable_ticker_at",
]
