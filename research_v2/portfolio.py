"""Pure portfolio construction for Research v2.

The module intentionally has no data-store, network or live-trading imports.
It turns an already point-in-time cross-sectional score into a desired target,
then applies turnover controls and an executable ADV participation limit.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, List, Mapping, Optional, Sequence, Tuple


_EPS = 1e-12


def _gross(weights: Mapping[str, float]) -> float:
    return sum(abs(float(value)) for value in weights.values())


def _clean_long_weights(weights: Mapping[str, float]) -> Dict[str, float]:
    clean: Dict[str, float] = {}
    for symbol, raw in weights.items():
        value = float(raw)
        if not math.isfinite(value):
            raise ValueError(f"non-finite weight for {symbol}")
        if value < -_EPS:
            raise ValueError("Research v2 portfolio construction is long-only")
        if value > _EPS:
            clean[str(symbol)] = value
    return clean


@dataclass(frozen=True)
class PortfolioConfig:
    top_n: int = 20
    weighting: str = "equal"  # equal | inverse_vol
    gross_target: float = 1.0
    single_name_cap: float = 0.10
    sector_cap: float = 0.35
    inverse_vol_floor: float = 0.005
    rank_buffer: int = 0
    no_trade_band: float = 0.0
    staggered_tranches: int = 1
    max_adv_participation: Optional[float] = 0.05
    unknown_sector: str = "Unknown"

    def __post_init__(self) -> None:
        if self.top_n <= 0:
            raise ValueError("top_n must be positive")
        if self.weighting not in {"equal", "inverse_vol"}:
            raise ValueError("weighting must be 'equal' or 'inverse_vol'")
        finite_non_negative = {
            "gross_target": self.gross_target,
            "single_name_cap": self.single_name_cap,
            "sector_cap": self.sector_cap,
            "inverse_vol_floor": self.inverse_vol_floor,
            "no_trade_band": self.no_trade_band,
        }
        for name, value in finite_non_negative.items():
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.gross_target <= 0:
            raise ValueError("gross_target must be positive")
        if self.single_name_cap <= 0 or self.sector_cap <= 0:
            raise ValueError("single_name_cap and sector_cap must be positive")
        if self.inverse_vol_floor <= 0:
            raise ValueError("inverse_vol_floor must be positive")
        if self.rank_buffer < 0:
            raise ValueError("rank_buffer must be non-negative")
        if self.staggered_tranches <= 0:
            raise ValueError("staggered_tranches must be positive")
        if self.max_adv_participation is not None:
            if not math.isfinite(self.max_adv_participation) or not 0 < self.max_adv_participation <= 1:
                raise ValueError("max_adv_participation must be in (0, 1]")


@dataclass(frozen=True)
class PortfolioReason:
    layer: str
    triggered: bool
    message: str
    before_gross: float
    after_gross: float


@dataclass(frozen=True)
class StaggerState:
    """Explicit sub-portfolio state for true staggered rebalancing.

    Each rebalance replaces exactly one tranche with the latest full desired
    book.  The aggregate target is the arithmetic mean of all tranches.
    """

    tranches: Tuple[Mapping[str, float], ...] = ()
    next_index: int = 0


@dataclass(frozen=True)
class PortfolioDecision:
    selected: Tuple[str, ...]
    unconstrained_weights: Mapping[str, float]
    desired_weights: Mapping[str, float]
    target_weights: Mapping[str, float]
    stagger_state: StaggerState
    reasons: Tuple[PortfolioReason, ...]


def select_with_rank_buffer(
    scores: Mapping[str, float],
    current_weights: Mapping[str, float],
    top_n: int,
    rank_buffer: int = 0,
) -> Tuple[str, ...]:
    """Select entrants in ``top_n`` and retain incumbents through the buffer.

    The holding count may temporarily rise to ``top_n + rank_buffer``.  This is
    the usual entry/exit-buffer behaviour and, importantly, never allows a
    buffered incumbent to crowd a new top-ranked entrant out of the book.
    """

    ranked = []
    for symbol, raw in scores.items():
        score = float(raw)
        if math.isfinite(score):
            ranked.append((str(symbol), score))
    ranked.sort(key=lambda item: (-item[1], item[0]))
    ordered = [symbol for symbol, _ in ranked]
    if not ordered:
        return ()

    rank = {symbol: index + 1 for index, symbol in enumerate(ordered)}
    held = _clean_long_weights(current_weights)
    selected: List[str] = list(ordered[:top_n])
    cutoff = top_n + max(0, rank_buffer)
    retained = sorted(
        (
            symbol
            for symbol in held
            if symbol not in selected and rank.get(symbol, 10**9) <= cutoff
        ),
        key=lambda symbol: (rank[symbol], symbol),
    )
    selected.extend(retained)
    return tuple(selected)


def _waterfill(
    preferences: Mapping[str, float],
    capacities: Mapping[str, float],
    total: float,
) -> Dict[str, float]:
    """Proportionally allocate ``total`` without exceeding per-key capacities."""

    result = {key: 0.0 for key in preferences}
    active = {
        key
        for key in preferences
        if float(preferences[key]) > 0 and float(capacities.get(key, 0.0)) > _EPS
    }
    remaining = min(float(total), sum(max(float(capacities.get(k, 0.0)), 0.0) for k in active))

    while active and remaining > _EPS:
        pref_sum = sum(max(float(preferences[key]), 0.0) for key in active)
        if pref_sum <= _EPS:
            pref_sum = float(len(active))
            pref = {key: 1.0 for key in active}
        else:
            pref = {key: max(float(preferences[key]), 0.0) for key in active}

        capped: List[str] = []
        proposals: Dict[str, float] = {}
        for key in sorted(active):
            proposal = remaining * pref[key] / pref_sum
            capacity_left = max(float(capacities[key]) - result[key], 0.0)
            proposals[key] = proposal
            if proposal >= capacity_left - _EPS:
                capped.append(key)

        if not capped:
            for key, proposal in proposals.items():
                result[key] += proposal
            remaining = 0.0
            break

        for key in capped:
            capacity_left = max(float(capacities[key]) - result[key], 0.0)
            result[key] += capacity_left
            remaining -= capacity_left
            active.remove(key)

    return {key: value for key, value in result.items() if value > _EPS}


def apply_position_and_sector_caps(
    preferences: Mapping[str, float],
    sectors: Mapping[str, str],
    gross_target: float,
    single_name_cap: float,
    sector_cap: float,
    unknown_sector: str = "Unknown",
) -> Dict[str, float]:
    """Allocate a long-only book subject to hard single-name and sector caps."""

    clean_pref: Dict[str, float] = {}
    grouped: Dict[str, List[str]] = {}
    for symbol, raw in preferences.items():
        value = float(raw)
        if not math.isfinite(value) or value <= 0:
            continue
        symbol = str(symbol)
        clean_pref[symbol] = value
        grouped.setdefault(str(sectors.get(symbol, unknown_sector)), []).append(symbol)
    if not clean_pref:
        return {}

    sector_preferences = {
        sector: sum(clean_pref[symbol] for symbol in symbols)
        for sector, symbols in grouped.items()
    }
    sector_capacities = {
        sector: min(float(sector_cap), len(symbols) * float(single_name_cap))
        for sector, symbols in grouped.items()
    }
    sector_targets = _waterfill(sector_preferences, sector_capacities, float(gross_target))

    result: Dict[str, float] = {}
    for sector, sector_target in sector_targets.items():
        symbols = grouped[sector]
        inside = _waterfill(
            {symbol: clean_pref[symbol] for symbol in symbols},
            {symbol: float(single_name_cap) for symbol in symbols},
            sector_target,
        )
        result.update(inside)
    return result


def apply_no_trade_band(
    current_weights: Mapping[str, float],
    target_weights: Mapping[str, float],
    band: float,
) -> Tuple[Dict[str, float], Tuple[str, ...]]:
    """Keep current weights where the proposed absolute change is too small."""

    current = _clean_long_weights(current_weights)
    target = _clean_long_weights(target_weights)
    if band <= 0:
        return dict(target), ()
    held = []
    result: Dict[str, float] = {}
    for symbol in sorted(set(current) | set(target)):
        old = current.get(symbol, 0.0)
        new = target.get(symbol, 0.0)
        if abs(new - old) < band:
            value = old
            if abs(new - old) > _EPS:
                held.append(symbol)
        else:
            value = new
        if value > _EPS:
            result[symbol] = value
    return result, tuple(held)


def update_staggered_tranches(
    desired_weights: Mapping[str, float],
    current_weights: Mapping[str, float],
    state: Optional[StaggerState],
    tranche_count: int,
) -> Tuple[Dict[str, float], StaggerState, int]:
    """Replace one explicit tranche and return its aggregate target."""

    desired = _clean_long_weights(desired_weights)
    current = _clean_long_weights(current_weights)
    if tranche_count <= 1:
        next_state = StaggerState(tranches=(dict(desired),), next_index=0)
        return dict(desired), next_state, 0

    if state is None or len(state.tranches) != tranche_count:
        tranches: List[Dict[str, float]] = [dict(current) for _ in range(tranche_count)]
        index = 0
    else:
        tranches = [dict(_clean_long_weights(tranche)) for tranche in state.tranches]
        index = int(state.next_index) % tranche_count

    tranches[index] = dict(desired)
    aggregate: Dict[str, float] = {}
    for tranche in tranches:
        for symbol, weight in tranche.items():
            aggregate[symbol] = aggregate.get(symbol, 0.0) + weight / tranche_count
    aggregate = {symbol: weight for symbol, weight in aggregate.items() if weight > _EPS}
    next_state = StaggerState(
        tranches=tuple(dict(tranche) for tranche in tranches),
        next_index=(index + 1) % tranche_count,
    )
    return aggregate, next_state, index


def apply_adv_participation_cap(
    current_weights: Mapping[str, float],
    target_weights: Mapping[str, float],
    equity: float,
    adv_dollars: Mapping[str, float],
    max_participation: Optional[float],
) -> Tuple[Dict[str, float], Mapping[str, float], Tuple[str, ...]]:
    """Clip each trade so dollar notional does not exceed the ADV budget."""

    current = _clean_long_weights(current_weights)
    target = _clean_long_weights(target_weights)
    if max_participation is None:
        return dict(target), {}, ()
    equity = float(equity)
    if not math.isfinite(equity) or equity <= 0:
        raise ValueError("equity must be finite and positive")
    if not 0 < max_participation <= 1:
        raise ValueError("max_participation must be in (0, 1]")

    result: Dict[str, float] = {}
    participation: Dict[str, float] = {}
    clipped: List[str] = []
    for symbol in sorted(set(current) | set(target)):
        old = current.get(symbol, 0.0)
        proposed = target.get(symbol, 0.0)
        delta = proposed - old
        if abs(delta) <= _EPS:
            if proposed > _EPS:
                result[symbol] = proposed
            continue
        if symbol not in adv_dollars:
            raise KeyError(f"missing lagged ADV for traded symbol {symbol}")
        adv = float(adv_dollars[symbol])
        if not math.isfinite(adv) or adv <= 0:
            raise ValueError(f"invalid ADV for {symbol}")
        max_delta = max_participation * adv / equity
        executed_delta = max(-max_delta, min(max_delta, delta))
        value = old + executed_delta
        participation[symbol] = abs(executed_delta) * equity / adv
        if abs(executed_delta - delta) > _EPS:
            clipped.append(symbol)
        if value > _EPS:
            result[symbol] = value
    return result, participation, tuple(clipped)


def construct_portfolio(
    scores: Mapping[str, float],
    current_weights: Mapping[str, float],
    volatility: Mapping[str, float],
    sectors: Mapping[str, str],
    adv_dollars: Mapping[str, float],
    equity: float,
    config: PortfolioConfig,
    stagger_state: Optional[StaggerState] = None,
) -> PortfolioDecision:
    """Build a deterministic, capacity-aware long-only target portfolio."""

    current = _clean_long_weights(current_weights)
    selected = select_with_rank_buffer(
        scores,
        current,
        top_n=config.top_n,
        rank_buffer=config.rank_buffer,
    )
    reasons: List[PortfolioReason] = []
    if not selected:
        empty_state = stagger_state or StaggerState()
        return PortfolioDecision((), {}, {}, {}, empty_state, ())

    if config.weighting == "equal":
        preferences = {symbol: 1.0 for symbol in selected}
    else:
        preferences = {}
        for symbol in selected:
            if symbol not in volatility:
                raise KeyError(f"missing point-in-time volatility for {symbol}")
            value = float(volatility[symbol])
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"invalid volatility for {symbol}")
            preferences[symbol] = 1.0 / max(value, config.inverse_vol_floor)

    pref_total = sum(preferences.values())
    unconstrained = {
        symbol: config.gross_target * preference / pref_total
        for symbol, preference in preferences.items()
    }
    constrained = apply_position_and_sector_caps(
        preferences,
        sectors,
        gross_target=config.gross_target,
        single_name_cap=config.single_name_cap,
        sector_cap=config.sector_cap,
        unknown_sector=config.unknown_sector,
    )
    constrained_gross = _gross(constrained)
    caps_triggered = any(
        abs(constrained.get(symbol, 0.0) - unconstrained.get(symbol, 0.0)) > 1e-10
        for symbol in set(constrained) | set(unconstrained)
    )
    reasons.append(
        PortfolioReason(
            layer="position_sector_caps",
            triggered=caps_triggered,
            message=(
                "single-name/sector constraints redistributed the desired book"
                if caps_triggered
                else "single-name and sector constraints inactive"
            ),
            before_gross=_gross(unconstrained),
            after_gross=constrained_gross,
        )
    )
    if constrained_gross < config.gross_target - 1e-9:
        reasons.append(
            PortfolioReason(
                layer="constraint_feasibility",
                triggered=True,
                message="requested gross is infeasible under the configured caps; residual remains cash",
                before_gross=config.gross_target,
                after_gross=constrained_gross,
            )
        )

    no_trade, buffered = apply_no_trade_band(current, constrained, config.no_trade_band)
    reasons.append(
        PortfolioReason(
            layer="no_trade_band",
            triggered=bool(buffered),
            message=(
                f"kept {len(buffered)} symbols inside the no-trade band"
                if buffered
                else "no proposed changes were suppressed by the no-trade band"
            ),
            before_gross=constrained_gross,
            after_gross=_gross(no_trade),
        )
    )

    staggered, new_stagger_state, replaced_index = update_staggered_tranches(
        no_trade,
        current,
        stagger_state,
        config.staggered_tranches,
    )
    reasons.append(
        PortfolioReason(
            layer="staggered_tranches",
            triggered=config.staggered_tranches > 1,
            message=(
                f"replaced tranche {replaced_index + 1}/{config.staggered_tranches}"
                if config.staggered_tranches > 1
                else "staggering disabled"
            ),
            before_gross=_gross(no_trade),
            after_gross=_gross(staggered),
        )
    )

    executable, _participation, clipped = apply_adv_participation_cap(
        current,
        staggered,
        equity,
        adv_dollars,
        config.max_adv_participation,
    )
    reasons.append(
        PortfolioReason(
            layer="adv_participation",
            triggered=bool(clipped),
            message=(
                f"clipped {len(clipped)} trades to the ADV participation budget"
                if clipped
                else "all proposed trades fit the ADV participation budget"
            ),
            before_gross=_gross(staggered),
            after_gross=_gross(executable),
        )
    )

    return PortfolioDecision(
        selected=selected,
        unconstrained_weights=unconstrained,
        desired_weights=no_trade,
        target_weights=executable,
        stagger_state=new_stagger_state,
        reasons=tuple(reasons),
    )
