"""Pure point-in-time signal and selection primitives for LETF rotation.

This module deliberately owns neither data acquisition nor portfolio sizing.
Callers provide an explicit global session calendar, long-form pandas bars,
point-in-time eligibility, and immutable universe metadata.  The output is a
score/selection audit only; it never contains target weights.

Information convention
----------------------
All calculations for ``session=t`` use rows whose normalized timestamp is at
or before ``t``.  Lookbacks are positions in the supplied *global* session
calendar, never "the previous available row" for an individual symbol.  A
missing row therefore makes the affected candidate unavailable instead of
silently shifting a lookback or forward-filling a price.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


class RotationDataError(ValueError):
    """Raised when point-in-time rotation inputs are ambiguous or invalid."""


class HeldDataUnavailableError(RotationDataError):
    """Raised rather than silently dropping a held instrument with no data."""


@dataclass(frozen=True)
class LETFMember:
    """Immutable product-to-theme metadata supplied by the caller."""

    symbol: str
    proxy_symbol: str
    theme: str
    macro_bucket: str

    def __post_init__(self) -> None:
        for name in ("symbol", "proxy_symbol", "theme", "macro_bucket"):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"{name} must be non-empty")


@dataclass(frozen=True)
class RotationConfig:
    """Pre-declared signal, cadence, and diversification settings."""

    cadence_sessions: int = 5
    rebalance_offset: int = 0
    top_k: int = 5
    long_lookback: int = 126
    medium_lookback: int = 63
    skip_sessions: int = 5
    acceleration_short: int = 21
    trend_lookback: int = 200
    absolute_proxy_gate: bool = True
    volatility_lookback: int = 63
    risk_adjust_momentum: bool = True
    volatility_floor: float = 1e-8
    signal_weights: Tuple[float, float, float] = (0.45, 0.35, 0.20)
    correlation_lookback: int = 126
    min_correlation_observations: int = 63
    max_abs_correlation: float = 0.70
    max_per_theme: int = 1
    max_per_macro: int = 2

    def __post_init__(self) -> None:
        positive_ints = {
            "cadence_sessions": self.cadence_sessions,
            "top_k": self.top_k,
            "long_lookback": self.long_lookback,
            "medium_lookback": self.medium_lookback,
            "skip_sessions": self.skip_sessions,
            "acceleration_short": self.acceleration_short,
            "trend_lookback": self.trend_lookback,
            "volatility_lookback": self.volatility_lookback,
            "correlation_lookback": self.correlation_lookback,
            "min_correlation_observations": self.min_correlation_observations,
            "max_per_theme": self.max_per_theme,
            "max_per_macro": self.max_per_macro,
        }
        for name, value in positive_ints.items():
            if not isinstance(value, (int, np.integer)) or int(value) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not 0 <= int(self.rebalance_offset) < int(self.cadence_sessions):
            raise ValueError("rebalance_offset must be inside cadence_sessions")
        if not (
            self.long_lookback > self.medium_lookback > self.acceleration_short
            and self.medium_lookback > self.skip_sessions
        ):
            raise ValueError(
                "lookbacks must satisfy long > medium > acceleration_short and medium > skip"
            )
        if self.min_correlation_observations > self.correlation_lookback:
            raise ValueError(
                "min_correlation_observations cannot exceed correlation_lookback"
            )
        if self.volatility_lookback <= 1:
            raise ValueError("volatility_lookback must exceed one")
        if not isinstance(self.absolute_proxy_gate, (bool, np.bool_)):
            raise ValueError("absolute_proxy_gate must be boolean")
        if not isinstance(self.risk_adjust_momentum, (bool, np.bool_)):
            raise ValueError("risk_adjust_momentum must be boolean")
        if not math.isfinite(float(self.volatility_floor)) or self.volatility_floor <= 0:
            raise ValueError("volatility_floor must be finite and positive")
        if not math.isfinite(float(self.max_abs_correlation)) or not 0 <= float(
            self.max_abs_correlation
        ) <= 1:
            raise ValueError("max_abs_correlation must be in [0, 1]")
        if len(self.signal_weights) != 3:
            raise ValueError("signal_weights must contain long, medium, acceleration weights")
        weights = tuple(float(value) for value in self.signal_weights)
        if any(not math.isfinite(value) or value < 0 for value in weights):
            raise ValueError("signal_weights must be finite and non-negative")
        if not math.isclose(sum(weights), 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("signal_weights must sum to one")


@dataclass(frozen=True)
class SignalComponents:
    """Raw and cross-sectional signal components for one instrument."""

    proxy_symbol: str
    raw_m126_5: float
    raw_m63_5: float
    m126_5: float
    m63_5: float
    acceleration: float
    trailing_volatility: float
    proxy_sma: float
    absolute_return_63d: float
    m126_5_percentile: float
    m63_5_percentile: float
    acceleration_percentile: float
    score: float


@dataclass(frozen=True)
class ScoreSnapshot:
    session: pd.Timestamp
    scores: Mapping[str, float]
    components: Mapping[str, SignalComponents]
    rejections: Mapping[str, str]


@dataclass(frozen=True)
class SelectionAudit:
    symbol: str
    accepted: bool
    reason: str
    score: Optional[float]
    details: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class SelectorResult:
    session: pd.Timestamp
    selected: Tuple[str, ...]
    audits: Tuple[SelectionAudit, ...]
    cash_slots: int


@dataclass(frozen=True)
class RotationDecision:
    session: pd.Timestamp
    rebalance_due: bool
    scores: Mapping[str, float]
    components: Mapping[str, SignalComponents]
    selected: Tuple[str, ...]
    audits: Tuple[SelectionAudit, ...]
    cash_slots: int


def _session(value: object) -> pd.Timestamp:
    try:
        stamp = pd.Timestamp(value)
    except Exception as exc:  # pragma: no cover - pandas supplies exact detail
        raise RotationDataError(f"invalid session {value!r}") from exc
    if pd.isna(stamp):
        raise RotationDataError("session cannot be NaT")
    if stamp.tzinfo is not None:
        stamp = stamp.tz_convert("UTC").tz_localize(None)
    return stamp.normalize()


def _calendar(sessions: Sequence[object] | pd.DatetimeIndex) -> pd.DatetimeIndex:
    if isinstance(sessions, pd.DatetimeIndex):
        index = sessions
        if index.tz is not None:
            index = index.tz_convert("UTC").tz_localize(None)
        # ``normalize`` is vectorized and preserves an already-normalized
        # immutable calendar.  This fast path matters when hundreds of PIT
        # decisions share the same frozen session index.
        index = index.normalize()
    else:
        stamps = [_session(value) for value in sessions]
        index = pd.DatetimeIndex(stamps)
    if len(index) == 0:
        raise RotationDataError("global session calendar is empty")
    if index.has_duplicates:
        raise RotationDataError("global session calendar contains duplicates")
    if not index.is_monotonic_increasing:
        raise RotationDataError("global session calendar must be strictly increasing")
    return index


def _session_position(
    session: object,
    sessions: Sequence[object] | pd.DatetimeIndex,
) -> tuple[pd.Timestamp, pd.DatetimeIndex, int]:
    stamp = _session(session)
    calendar = _calendar(sessions)
    matches = np.flatnonzero(calendar == stamp)
    if len(matches) != 1:
        raise RotationDataError(f"decision session {stamp.date()} is not in the global calendar")
    return stamp, calendar, int(matches[0])


def _member_map(
    universe: Sequence[LETFMember] | pd.DataFrame | Mapping[str, object],
) -> dict[str, LETFMember]:
    members: list[LETFMember] = []
    if isinstance(universe, pd.DataFrame):
        required = {"symbol", "proxy_symbol", "theme", "macro_bucket"}
        missing = required - set(universe.columns)
        if missing:
            raise RotationDataError(f"universe metadata missing columns: {sorted(missing)}")
        members = [
            LETFMember(
                symbol=str(row.symbol).strip().upper(),
                proxy_symbol=str(row.proxy_symbol).strip().upper(),
                theme=str(row.theme).strip(),
                macro_bucket=str(row.macro_bucket).strip(),
            )
            for row in universe.loc[:, sorted(required)].itertuples(index=False)
        ]
    elif isinstance(universe, Mapping):
        for key, value in universe.items():
            if isinstance(value, LETFMember):
                member = value
            elif isinstance(value, Mapping):
                member = LETFMember(
                    symbol=str(value.get("symbol", key)),
                    proxy_symbol=str(value["proxy_symbol"]),
                    theme=str(value["theme"]),
                    macro_bucket=str(value["macro_bucket"]),
                )
            else:
                raise RotationDataError("universe mapping values must be LETFMember or mappings")
            members.append(member)
    else:
        members = list(universe)
        if any(not isinstance(member, LETFMember) for member in members):
            raise RotationDataError("universe sequence must contain LETFMember values")

    result: dict[str, LETFMember] = {}
    for raw in members:
        member = LETFMember(
            symbol=str(raw.symbol).strip().upper(),
            proxy_symbol=str(raw.proxy_symbol).strip().upper(),
            theme=str(raw.theme).strip(),
            macro_bucket=str(raw.macro_bucket).strip(),
        )
        if member.symbol in result:
            raise RotationDataError(f"duplicate universe symbol {member.symbol}")
        result[member.symbol] = member
    if not result:
        raise RotationDataError("universe metadata is empty")
    return result


def _normalised_bar_sessions(values: pd.Series) -> pd.Series:
    try:
        result = pd.to_datetime(values, errors="raise", utc=True)
    except Exception as exc:
        raise RotationDataError("bar timestamps are invalid") from exc
    return result.dt.tz_convert(None).dt.normalize()


def prepare_close_panel(bars: pd.DataFrame) -> pd.DataFrame:
    """Validate and pre-pivot long-form closes for repeated pure decisions.

    The returned wide frame is an immutable-by-convention research input.  It
    contains no computed feature or future value; it only caches parsing and
    pivoting that would otherwise be repeated for every robustness scenario.
    """

    required = {"timestamp", "symbol", "close"}
    missing = required - set(bars.columns)
    if missing:
        raise RotationDataError(f"bars missing columns: {sorted(missing)}")
    work = bars.loc[:, ["timestamp", "symbol", "close"]].copy()
    work["_session"] = _normalised_bar_sessions(work["timestamp"])
    work["_symbol"] = work["symbol"].astype(str).str.strip().str.upper()
    if work.duplicated(["_session", "_symbol"]).any():
        duplicate = work.loc[
            work.duplicated(["_session", "_symbol"], keep=False),
            ["_session", "_symbol"],
        ].iloc[0]
        raise RotationDataError(
            f"duplicate bar for {duplicate['_symbol']} on "
            f"{pd.Timestamp(duplicate['_session']).date()}"
        )
    work["close"] = pd.to_numeric(work["close"], errors="coerce")
    invalid = ~np.isfinite(work["close"].to_numpy(dtype=float)) | (work["close"] <= 0)
    work.loc[invalid, "close"] = np.nan
    panel = (
        work.pivot(index="_session", columns="_symbol", values="close")
        .sort_index()
        .astype(float)
    )
    panel.attrs["letf_normalized_close_panel"] = True
    return panel


def _close_matrix(
    bars: pd.DataFrame,
    *,
    window: pd.DatetimeIndex,
    symbols: Iterable[str],
) -> pd.DataFrame:
    # Batch runners may pre-pivot a frozen close panel once.  Supporting that
    # representation avoids reparsing and filtering the same long-form frame
    # for every offset, ablation, and leave-one-out decision.  The exact same
    # global-session reindex and invalid-price semantics still apply.
    if isinstance(bars.index, pd.DatetimeIndex) and not {
        "timestamp", "symbol", "close"
    }.issubset(bars.columns):
        if bars.attrs.get("letf_normalized_close_panel") is True:
            panel = bars
        else:
            panel = bars.copy()
            normalized_index = pd.DatetimeIndex([_session(value) for value in panel.index])
            if normalized_index.has_duplicates:
                raise RotationDataError("wide close panel contains duplicate sessions")
            panel.index = normalized_index
            panel.columns = [str(column).strip().upper() for column in panel.columns]
            if pd.Index(panel.columns).has_duplicates:
                raise RotationDataError("wide close panel contains duplicate symbols")
        wanted = tuple(dict.fromkeys(str(symbol).strip().upper() for symbol in symbols))
        if not wanted:
            return pd.DataFrame(index=window)
        matrix = panel.reindex(index=window, columns=list(wanted)).apply(
            pd.to_numeric, errors="coerce"
        )
        values = matrix.to_numpy(dtype=float)
        matrix.iloc[:, :] = np.where(np.isfinite(values) & (values > 0), values, np.nan)
        return matrix.astype(float)

    required = {"timestamp", "symbol", "close"}
    missing = required - set(bars.columns)
    if missing:
        raise RotationDataError(f"bars missing columns: {sorted(missing)}")
    wanted = tuple(dict.fromkeys(str(symbol).strip().upper() for symbol in symbols))
    if not wanted:
        return pd.DataFrame(index=window)

    # Timestamp conversion does not inspect prices.  Price validation happens
    # only inside the requested historical window, so future price mutations
    # cannot invalidate an already-computed decision.
    work = bars.loc[:, ["timestamp", "symbol", "close"]].copy()
    work["_session"] = _normalised_bar_sessions(work["timestamp"])
    work["_symbol"] = work["symbol"].astype(str).str.strip().str.upper()
    work = work.loc[
        work["_session"].isin(window) & work["_symbol"].isin(wanted),
        ["_session", "_symbol", "close"],
    ]
    if work.duplicated(["_session", "_symbol"]).any():
        duplicate = work.loc[
            work.duplicated(["_session", "_symbol"], keep=False),
            ["_session", "_symbol"],
        ].iloc[0]
        raise RotationDataError(
            f"duplicate bar for {duplicate['_symbol']} on "
            f"{pd.Timestamp(duplicate['_session']).date()}"
        )
    work["close"] = pd.to_numeric(work["close"], errors="coerce")
    invalid = ~np.isfinite(work["close"].to_numpy(dtype=float)) | (work["close"] <= 0)
    work.loc[invalid, "close"] = np.nan
    return (
        work.pivot(index="_session", columns="_symbol", values="close")
        .reindex(index=window, columns=list(wanted))
        .astype(float)
    )


def _eligibility_at(
    eligibility: pd.DataFrame | Mapping[str, object],
    *,
    session: pd.Timestamp,
    symbols: Iterable[str],
) -> tuple[set[str], dict[str, str]]:
    wanted = tuple(sorted(set(str(symbol).strip().upper() for symbol in symbols)))
    values: dict[str, bool] = {}
    present: set[str] = set()
    if isinstance(eligibility, pd.DataFrame):
        required = {"timestamp", "symbol", "eligible"}
        missing = required - set(eligibility.columns)
        if missing:
            raise RotationDataError(f"eligibility missing columns: {sorted(missing)}")
        frame = eligibility.loc[:, ["timestamp", "symbol", "eligible"]].copy()
        frame["_session"] = _normalised_bar_sessions(frame["timestamp"])
        frame["_symbol"] = frame["symbol"].astype(str).str.strip().str.upper()
        frame = frame.loc[
            (frame["_session"] == session) & frame["_symbol"].isin(wanted)
        ]
        if frame.duplicated("_symbol").any():
            symbol = str(frame.loc[frame.duplicated("_symbol", keep=False), "_symbol"].iloc[0])
            raise RotationDataError(
                f"duplicate eligibility for {symbol} on {session.date()}"
            )
        for symbol, value in zip(frame["_symbol"], frame["eligible"]):
            if not isinstance(value, (bool, np.bool_)):
                raise RotationDataError(
                    f"eligibility for {symbol} must be boolean"
                )
            present.add(str(symbol))
            values[str(symbol)] = bool(value)
    elif isinstance(eligibility, Mapping):
        for raw_symbol, value in eligibility.items():
            symbol = str(raw_symbol).strip().upper()
            if symbol not in wanted:
                continue
            if not isinstance(value, (bool, np.bool_)):
                raise RotationDataError(f"eligibility for {symbol} must be boolean")
            present.add(symbol)
            values[symbol] = bool(value)
    else:
        raise RotationDataError("eligibility must be an exact-session DataFrame or mapping")

    eligible = {symbol for symbol in wanted if values.get(symbol) is True}
    rejected = {
        symbol: (
            "eligibility_missing_for_session"
            if symbol not in present
            else "not_eligible"
        )
        for symbol in wanted
        if symbol not in eligible
    }
    return eligible, rejected


def _percentiles(values: Mapping[str, float]) -> dict[str, float]:
    if not values:
        return {}
    series = pd.Series(values, dtype=float).sort_index()
    ranks = series.rank(method="average")
    if len(series) == 1:
        return {str(series.index[0]): 0.5}
    scaled = (ranks - 1.0) / float(len(series) - 1)
    return {str(key): float(value) for key, value in scaled.items()}


def is_rebalance_session(
    session: object,
    sessions: Sequence[object] | pd.DatetimeIndex,
    config: RotationConfig = RotationConfig(),
) -> bool:
    """Return cadence membership using the explicit global session index."""

    _stamp, _calendar_index, position = _session_position(session, sessions)
    return (
        position >= config.rebalance_offset
        and (position - config.rebalance_offset) % config.cadence_sessions == 0
    )


def compute_proxy_first_scores(
    *,
    session: object,
    sessions: Sequence[object] | pd.DatetimeIndex,
    proxy_bars: pd.DataFrame,
    universe: Sequence[LETFMember] | pd.DataFrame | Mapping[str, object],
    eligible_symbols: Iterable[str],
    config: RotationConfig = RotationConfig(),
) -> ScoreSnapshot:
    """Compute proxy-first close(t) scores without reading future prices.

    Cross-sectional percentiles are computed over unique proxies so duplicate
    products tracking the same exposure do not receive extra influence merely
    because multiple wrappers exist in the seed universe.
    """

    stamp, calendar, position = _session_position(session, sessions)
    members = _member_map(universe)
    eligible = tuple(sorted(set(str(symbol).strip().upper() for symbol in eligible_symbols)))
    unknown = sorted(set(eligible) - set(members))
    if unknown:
        raise RotationDataError(f"eligible symbols absent from universe: {unknown}")

    required_history = max(
        config.long_lookback,
        config.trend_lookback,
        config.volatility_lookback,
    )
    if position < required_history:
        return ScoreSnapshot(
            session=stamp,
            scores={},
            components={},
            rejections={symbol: "insufficient_global_history" for symbol in eligible},
        )

    window = calendar[position - required_history : position + 1]
    proxies = tuple(sorted({members[symbol].proxy_symbol for symbol in eligible}))
    matrix = _close_matrix(proxy_bars, window=window, symbols=proxies)
    # adjusted long, adjusted medium, acceleration, raw long, raw medium,
    # trailing volatility, slow trend, absolute medium-horizon return
    raw_by_proxy: dict[str, tuple[float, float, float, float, float, float, float, float]] = {}
    proxy_rejections: dict[str, str] = {}
    for proxy in proxies:
        values = matrix[proxy].to_numpy(dtype=float)
        if len(values) != required_history + 1 or not np.isfinite(values).all():
            proxy_rejections[proxy] = "missing_contiguous_proxy_history"
            continue
        p_now = values[-1]
        p_skip = values[-1 - config.skip_sessions]
        p_long = values[-1 - config.long_lookback]
        p_medium = values[-1 - config.medium_lookback]
        p_short = values[-1 - config.acceleration_short]
        raw_long = math.log(p_skip / p_long)
        raw_medium = math.log(p_skip / p_medium)
        recent_slope = math.log(p_now / p_short) / config.acceleration_short
        prior_slope = math.log(p_short / p_medium) / (
            config.medium_lookback - config.acceleration_short
        )
        acceleration = recent_slope - prior_slope
        volatility_prices = values[-(config.volatility_lookback + 1) :]
        volatility_returns = np.diff(np.log(volatility_prices))
        trailing_volatility = float(np.std(volatility_returns, ddof=1))
        proxy_sma = float(np.mean(values[-config.trend_lookback :]))
        absolute_return = math.log(p_now / p_medium)
        diagnostics = (
            raw_long,
            raw_medium,
            acceleration,
            trailing_volatility,
            proxy_sma,
            absolute_return,
        )
        if not all(math.isfinite(value) for value in diagnostics):
            proxy_rejections[proxy] = "invalid_proxy_signal"
            continue
        if config.absolute_proxy_gate and not (
            p_now > proxy_sma and absolute_return > 0.0
        ):
            proxy_rejections[proxy] = "absolute_proxy_gate"
            continue
        if config.risk_adjust_momentum:
            if trailing_volatility <= config.volatility_floor:
                proxy_rejections[proxy] = "invalid_proxy_volatility"
                continue
            m_long = raw_long / trailing_volatility
            m_medium = raw_medium / trailing_volatility
        else:
            m_long = raw_long
            m_medium = raw_medium
        raw_by_proxy[proxy] = (
            m_long,
            m_medium,
            acceleration,
            raw_long,
            raw_medium,
            trailing_volatility,
            proxy_sma,
            absolute_return,
        )

    long_pct = _percentiles({proxy: values[0] for proxy, values in raw_by_proxy.items()})
    medium_pct = _percentiles({proxy: values[1] for proxy, values in raw_by_proxy.items()})
    acceleration_pct = _percentiles({proxy: values[2] for proxy, values in raw_by_proxy.items()})
    weights = tuple(float(value) for value in config.signal_weights)
    scores: dict[str, float] = {}
    components: dict[str, SignalComponents] = {}
    rejections: dict[str, str] = {}
    for symbol in eligible:
        proxy = members[symbol].proxy_symbol
        if proxy in proxy_rejections or proxy not in raw_by_proxy:
            rejections[symbol] = proxy_rejections.get(proxy, "invalid_proxy_signal")
            continue
        raw = raw_by_proxy[proxy]
        score = (
            weights[0] * long_pct[proxy]
            + weights[1] * medium_pct[proxy]
            + weights[2] * acceleration_pct[proxy]
        )
        scores[symbol] = float(score)
        components[symbol] = SignalComponents(
            proxy_symbol=proxy,
            raw_m126_5=float(raw[3]),
            raw_m63_5=float(raw[4]),
            m126_5=float(raw[0]),
            m63_5=float(raw[1]),
            acceleration=float(raw[2]),
            trailing_volatility=float(raw[5]),
            proxy_sma=float(raw[6]),
            absolute_return_63d=float(raw[7]),
            m126_5_percentile=float(long_pct[proxy]),
            m63_5_percentile=float(medium_pct[proxy]),
            acceleration_percentile=float(acceleration_pct[proxy]),
            score=float(score),
        )
    return ScoreSnapshot(stamp, scores, components, rejections)


def select_rotation_candidates(
    *,
    session: object,
    sessions: Sequence[object] | pd.DatetimeIndex,
    product_bars: pd.DataFrame,
    universe: Sequence[LETFMember] | pd.DataFrame | Mapping[str, object],
    scores: Mapping[str, float],
    config: RotationConfig = RotationConfig(),
) -> SelectorResult:
    """Greedily select score-ranked products under group/correlation limits."""

    stamp, calendar, position = _session_position(session, sessions)
    members = _member_map(universe)
    clean_scores: dict[str, float] = {}
    for raw_symbol, raw_score in scores.items():
        symbol = str(raw_symbol).strip().upper()
        if symbol not in members:
            raise RotationDataError(f"score symbol {symbol} is absent from universe")
        score = float(raw_score)
        if not math.isfinite(score):
            raise RotationDataError(f"score for {symbol} must be finite")
        clean_scores[symbol] = score

    ordered = sorted(clean_scores, key=lambda symbol: (-clean_scores[symbol], symbol))
    if not ordered:
        return SelectorResult(stamp, (), (), config.top_k)
    if position < config.correlation_lookback:
        audits = tuple(
            SelectionAudit(
                symbol=symbol,
                accepted=False,
                reason="insufficient_global_correlation_history",
                score=clean_scores[symbol],
            )
            for symbol in ordered
        )
        return SelectorResult(stamp, (), audits, config.top_k)

    window = calendar[position - config.correlation_lookback : position + 1]
    matrix = _close_matrix(product_bars, window=window, symbols=ordered)
    valid_history = {
        symbol: bool(np.isfinite(matrix[symbol].to_numpy(dtype=float)).all())
        for symbol in ordered
    }
    returns = matrix.pct_change(fill_method=None).iloc[1:]
    selected: list[str] = []
    audits: list[SelectionAudit] = []
    theme_counts: dict[str, int] = {}
    macro_counts: dict[str, int] = {}

    for symbol in ordered:
        score = clean_scores[symbol]
        member = members[symbol]
        if len(selected) >= config.top_k:
            audits.append(SelectionAudit(symbol, False, "rank_below_top_k", score))
            continue
        if not valid_history[symbol]:
            audits.append(
                SelectionAudit(symbol, False, "missing_contiguous_product_history", score)
            )
            continue
        if theme_counts.get(member.theme, 0) >= config.max_per_theme:
            audits.append(
                SelectionAudit(
                    symbol,
                    False,
                    "theme_cap",
                    score,
                    {"theme": member.theme, "limit": config.max_per_theme},
                )
            )
            continue
        if macro_counts.get(member.macro_bucket, 0) >= config.max_per_macro:
            audits.append(
                SelectionAudit(
                    symbol,
                    False,
                    "macro_cap",
                    score,
                    {"macro_bucket": member.macro_bucket, "limit": config.max_per_macro},
                )
            )
            continue

        blocker: Optional[str] = None
        blocker_correlation: Optional[float] = None
        unavailable_with: Optional[str] = None
        for incumbent in selected:
            pair = returns.loc[:, [symbol, incumbent]].dropna()
            if len(pair) < config.min_correlation_observations:
                unavailable_with = incumbent
                break
            correlation = float(pair[symbol].corr(pair[incumbent]))
            if not math.isfinite(correlation):
                unavailable_with = incumbent
                break
            if abs(correlation) > config.max_abs_correlation + 1e-12:
                blocker = incumbent
                blocker_correlation = correlation
                break
        if unavailable_with is not None:
            audits.append(
                SelectionAudit(
                    symbol,
                    False,
                    "correlation_unavailable",
                    score,
                    {"with": unavailable_with},
                )
            )
            continue
        if blocker is not None:
            audits.append(
                SelectionAudit(
                    symbol,
                    False,
                    "correlation_cap",
                    score,
                    {
                        "with": blocker,
                        "correlation": blocker_correlation,
                        "absolute_limit": config.max_abs_correlation,
                    },
                )
            )
            continue

        selected.append(symbol)
        theme_counts[member.theme] = theme_counts.get(member.theme, 0) + 1
        macro_counts[member.macro_bucket] = macro_counts.get(member.macro_bucket, 0) + 1
        audits.append(SelectionAudit(symbol, True, "selected", score))

    return SelectorResult(
        session=stamp,
        selected=tuple(selected),
        audits=tuple(audits),
        cash_slots=max(config.top_k - len(selected), 0),
    )


def evaluate_rotation(
    *,
    session: object,
    sessions: Sequence[object] | pd.DatetimeIndex,
    product_bars: pd.DataFrame,
    proxy_bars: pd.DataFrame,
    universe: Sequence[LETFMember] | pd.DataFrame | Mapping[str, object],
    eligibility: pd.DataFrame | Mapping[str, object],
    held_symbols: Iterable[str] = (),
    config: RotationConfig = RotationConfig(),
) -> RotationDecision:
    """Evaluate one close without I/O, forward-fill, sizing, or live state."""

    stamp, calendar, _position = _session_position(session, sessions)
    members = _member_map(universe)
    eligible, eligibility_rejections = _eligibility_at(
        eligibility,
        session=stamp,
        symbols=members,
    )

    held = tuple(sorted(set(str(symbol).strip().upper() for symbol in held_symbols)))
    unknown_held = sorted(set(held) - set(members))
    if unknown_held:
        raise HeldDataUnavailableError(f"held symbols absent from universe: {unknown_held}")
    unavailable_held = sorted(set(held) - eligible)
    if unavailable_held:
        raise HeldDataUnavailableError(
            "held symbols lack exact-session eligibility: " + ", ".join(unavailable_held)
        )
    if held:
        held_close = _close_matrix(product_bars, window=pd.DatetimeIndex([stamp]), symbols=held)
        missing_held = [
            symbol
            for symbol in held
            if not math.isfinite(float(held_close.at[stamp, symbol]))
        ]
        if missing_held:
            raise HeldDataUnavailableError(
                "held symbols lack an exact-session close: " + ", ".join(missing_held)
            )

    if not is_rebalance_session(stamp, calendar, config):
        return RotationDecision(stamp, False, {}, {}, (), (), 0)

    snapshot = compute_proxy_first_scores(
        session=stamp,
        sessions=calendar,
        proxy_bars=proxy_bars,
        universe=tuple(members.values()),
        eligible_symbols=eligible,
        config=config,
    )
    selection = select_rotation_candidates(
        session=stamp,
        sessions=calendar,
        product_bars=product_bars,
        universe=tuple(members.values()),
        scores=snapshot.scores,
        config=config,
    )

    pre_audits = [
        SelectionAudit(symbol, False, reason, None)
        for symbol, reason in sorted(eligibility_rejections.items())
    ]
    pre_audits.extend(
        SelectionAudit(symbol, False, reason, None)
        for symbol, reason in sorted(snapshot.rejections.items())
    )
    return RotationDecision(
        session=stamp,
        rebalance_due=True,
        scores=dict(snapshot.scores),
        components=dict(snapshot.components),
        selected=selection.selected,
        audits=tuple(pre_audits) + selection.audits,
        cash_slots=selection.cash_slots,
    )


__all__ = [
    "HeldDataUnavailableError",
    "LETFMember",
    "RotationConfig",
    "RotationDataError",
    "RotationDecision",
    "ScoreSnapshot",
    "SelectionAudit",
    "SelectorResult",
    "SignalComponents",
    "compute_proxy_first_scores",
    "evaluate_rotation",
    "is_rebalance_session",
    "prepare_close_panel",
    "select_rotation_candidates",
]
