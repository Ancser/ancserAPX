"""Pure, cost-aware execution primitives for the isolated Research v2 engine.

All market inputs passed here must be observable at the signal close.  In
particular, an order generated at close(t) must use the ADV, volatility and
spread proxy known at close(t), even though it is executed at open(t+1).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Dict, Mapping, Optional, Tuple


_BPS = 10_000.0
_EPS = 1e-12


@dataclass(frozen=True)
class LiquiditySnapshot:
    """Lagged, point-in-time liquidity inputs for one symbol.

    ``spread_proxy_bps`` is the estimated *one-way half-spread* paid by the
    strategy.  It is deliberately named a proxy: a daily high/low range is not
    a quoted spread and must be calibrated before being supplied here.
    """

    adv_dollars: float
    daily_volatility: float
    spread_proxy_bps: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.adv_dollars) or self.adv_dollars <= 0:
            raise ValueError("adv_dollars must be finite and positive")
        if not math.isfinite(self.daily_volatility) or self.daily_volatility < 0:
            raise ValueError("daily_volatility must be finite and non-negative")
        if not math.isfinite(self.spread_proxy_bps) or self.spread_proxy_bps < 0:
            raise ValueError("spread_proxy_bps must be finite and non-negative")


@dataclass(frozen=True)
class CostConfig:
    """Execution and financing assumptions.

    Market impact follows a square-root participation model::

        impact_bps = impact_coefficient * daily_vol * sqrt(notional / ADV) * 1e4

    ``max_adv_participation`` is checked again at execution time.  Portfolio
    construction normally clips the trade first, but the second check catches
    overnight equity moves and stale/inconsistent liquidity inputs.
    """

    commission_bps: float = 0.0
    spread_multiplier: float = 1.0
    min_spread_bps: float = 0.0
    max_spread_bps: float = 100.0
    impact_coefficient: float = 0.10
    max_impact_bps: float = 250.0
    max_adv_participation: Optional[float] = 0.05
    annual_funding_rate: float = 0.0
    annual_short_borrow_rate: float = 0.0
    annual_cash_rate: float = 0.0
    periods_per_year: int = 252
    participation_tolerance: float = 1e-9

    def __post_init__(self) -> None:
        non_negative = {
            "commission_bps": self.commission_bps,
            "spread_multiplier": self.spread_multiplier,
            "min_spread_bps": self.min_spread_bps,
            "max_spread_bps": self.max_spread_bps,
            "impact_coefficient": self.impact_coefficient,
            "max_impact_bps": self.max_impact_bps,
            "annual_funding_rate": self.annual_funding_rate,
            "annual_short_borrow_rate": self.annual_short_borrow_rate,
            "annual_cash_rate": self.annual_cash_rate,
            "participation_tolerance": self.participation_tolerance,
        }
        for name, value in non_negative.items():
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.max_spread_bps < self.min_spread_bps:
            raise ValueError("max_spread_bps must be >= min_spread_bps")
        if self.max_adv_participation is not None:
            if not math.isfinite(self.max_adv_participation) or not 0 < self.max_adv_participation <= 1:
                raise ValueError("max_adv_participation must be in (0, 1]")
        if self.periods_per_year <= 0:
            raise ValueError("periods_per_year must be positive")


@dataclass(frozen=True)
class TradeCost:
    symbol: str
    delta_weight: float
    notional: float
    adv_participation: float
    commission_bps: float
    spread_bps: float
    impact_bps: float
    commission: float
    spread: float
    impact: float

    @property
    def total(self) -> float:
        return self.commission + self.spread + self.impact


@dataclass(frozen=True)
class FundingBreakdown:
    borrowed_cash: float = 0.0
    short_notional: float = 0.0
    idle_cash: float = 0.0
    margin_interest: float = 0.0
    short_borrow: float = 0.0
    cash_interest_credit: float = 0.0

    @property
    def total(self) -> float:
        """Net financing cost; cash interest may make this negative."""

        return self.margin_interest + self.short_borrow - self.cash_interest_credit


@dataclass(frozen=True)
class CostBreakdown:
    trades: Tuple[TradeCost, ...] = ()
    traded_notional: float = 0.0
    gross_turnover: float = 0.0
    one_way_turnover: float = 0.0
    commission: float = 0.0
    spread: float = 0.0
    impact: float = 0.0
    funding: float = 0.0
    max_adv_participation: float = 0.0

    @property
    def execution_total(self) -> float:
        return self.commission + self.spread + self.impact

    @property
    def total(self) -> float:
        return self.execution_total + self.funding

    def with_funding(self, funding: FundingBreakdown | float) -> "CostBreakdown":
        value = funding.total if isinstance(funding, FundingBreakdown) else float(funding)
        return replace(self, funding=value)


class AdvParticipationError(ValueError):
    """Raised when a proposed execution exceeds the configured ADV limit."""


def _clean_weights(weights: Mapping[str, float]) -> Dict[str, float]:
    clean: Dict[str, float] = {}
    for symbol, raw in weights.items():
        value = float(raw)
        if not math.isfinite(value):
            raise ValueError(f"non-finite weight for {symbol}")
        if abs(value) > _EPS:
            clean[str(symbol)] = value
    return clean


def trade_weight_deltas(
    pretrade_weights: Mapping[str, float],
    target_weights: Mapping[str, float],
) -> Dict[str, float]:
    """Return signed target-minus-pretrade weight changes."""

    before = _clean_weights(pretrade_weights)
    after = _clean_weights(target_weights)
    symbols = sorted(set(before) | set(after))
    return {
        symbol: after.get(symbol, 0.0) - before.get(symbol, 0.0)
        for symbol in symbols
        if abs(after.get(symbol, 0.0) - before.get(symbol, 0.0)) > _EPS
    }


def turnover_from_deltas(deltas: Mapping[str, float]) -> Tuple[float, float]:
    """Return ``(gross traded weight, conventional one-way turnover)``."""

    gross = sum(abs(float(value)) for value in deltas.values())
    return gross, 0.5 * gross


def estimate_execution_costs(
    pretrade_weights: Mapping[str, float],
    target_weights: Mapping[str, float],
    equity: float,
    liquidity: Mapping[str, LiquiditySnapshot],
    config: CostConfig,
    *,
    enforce_adv_limit: bool = True,
) -> CostBreakdown:
    """Estimate one-way execution costs from lagged market information."""

    equity = float(equity)
    if not math.isfinite(equity) or equity <= 0:
        raise ValueError("equity must be finite and positive")

    deltas = trade_weight_deltas(pretrade_weights, target_weights)
    gross_turnover, one_way_turnover = turnover_from_deltas(deltas)
    trades = []

    for symbol, delta_weight in deltas.items():
        if symbol not in liquidity:
            raise KeyError(f"missing lagged liquidity for traded symbol {symbol}")
        point = liquidity[symbol]
        notional = equity * abs(delta_weight)
        participation = notional / point.adv_dollars
        if (
            enforce_adv_limit
            and config.max_adv_participation is not None
            and participation > config.max_adv_participation + config.participation_tolerance
        ):
            raise AdvParticipationError(
                f"{symbol} participation {participation:.6f} exceeds "
                f"limit {config.max_adv_participation:.6f}"
            )

        spread_bps = min(
            config.max_spread_bps,
            max(config.min_spread_bps, point.spread_proxy_bps * config.spread_multiplier),
        )
        impact_bps = min(
            config.max_impact_bps,
            config.impact_coefficient
            * point.daily_volatility
            * math.sqrt(max(participation, 0.0))
            * _BPS,
        )
        commission = notional * config.commission_bps / _BPS
        spread = notional * spread_bps / _BPS
        impact = notional * impact_bps / _BPS
        trades.append(
            TradeCost(
                symbol=symbol,
                delta_weight=delta_weight,
                notional=notional,
                adv_participation=participation,
                commission_bps=config.commission_bps,
                spread_bps=spread_bps,
                impact_bps=impact_bps,
                commission=commission,
                spread=spread,
                impact=impact,
            )
        )

    return CostBreakdown(
        trades=tuple(trades),
        traded_notional=sum(item.notional for item in trades),
        gross_turnover=gross_turnover,
        one_way_turnover=one_way_turnover,
        commission=sum(item.commission for item in trades),
        spread=sum(item.spread for item in trades),
        impact=sum(item.impact for item in trades),
        max_adv_participation=max((item.adv_participation for item in trades), default=0.0),
    )


def estimate_daily_funding(
    weights: Mapping[str, float],
    equity: float,
    config: CostConfig,
) -> FundingBreakdown:
    """Return one period of margin, short-borrow and cash carry.

    Cash borrowing depends on *net* exposure; a market-neutral long/short book
    instead pays the separately reported short-borrow charge.
    """

    clean = _clean_weights(weights)
    equity = float(equity)
    if not math.isfinite(equity) or equity <= 0:
        raise ValueError("equity must be finite and positive")

    net = sum(clean.values())
    short_weight = sum(max(-weight, 0.0) for weight in clean.values())
    borrowed_cash = max(net - 1.0, 0.0) * equity
    idle_cash = max(1.0 - net, 0.0) * equity
    short_notional = short_weight * equity
    periods = float(config.periods_per_year)

    return FundingBreakdown(
        borrowed_cash=borrowed_cash,
        short_notional=short_notional,
        idle_cash=idle_cash,
        margin_interest=borrowed_cash * config.annual_funding_rate / periods,
        short_borrow=short_notional * config.annual_short_borrow_rate / periods,
        cash_interest_credit=idle_cash * config.annual_cash_rate / periods,
    )

