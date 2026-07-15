"""Event-driven, point-in-time Research v2 backtest engine.

The clock is deliberately explicit:

* information through close(t) may create a target;
* that target can only execute at open(t+1);
* pre-existing holdings earn the overnight close(t)-to-open(t+1) move;
* post-trade holdings earn open(t+1)-to-close(t+1);
* execution and funding costs are deducted from cash and reconciled daily.

The engine is an offline pure-state transition: it performs no file, network,
broker or global-state access.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from research_v2.costs import (
    CostBreakdown,
    CostConfig,
    FundingBreakdown,
    LiquiditySnapshot,
    estimate_daily_funding,
    estimate_execution_costs,
)
from research_v2.portfolio import (
    PortfolioConfig,
    PortfolioReason,
    StaggerState,
    apply_adv_participation_cap,
    construct_portfolio,
)


_EPS = 1e-12


class MissingHeldReturnError(RuntimeError):
    """A held or scheduled-to-trade symbol lacks an executable daily bar."""


class AccountingError(RuntimeError):
    """The cash/position ledger failed its accounting identity."""


@dataclass(frozen=True)
class MarketBar:
    """One session's market data plus close(t)-observable risk inputs."""

    open: float
    close: float
    adv_dollars: float
    daily_volatility: float
    spread_proxy_bps: float = 0.0
    beta: float = 1.0

    def __post_init__(self) -> None:
        positive = {
            "open": self.open,
            "close": self.close,
            "adv_dollars": self.adv_dollars,
        }
        for name, value in positive.items():
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        non_negative = {
            "daily_volatility": self.daily_volatility,
            "spread_proxy_bps": self.spread_proxy_bps,
        }
        for name, value in non_negative.items():
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if not math.isfinite(self.beta):
            raise ValueError("beta must be finite")

    def liquidity(self) -> LiquiditySnapshot:
        return LiquiditySnapshot(
            adv_dollars=self.adv_dollars,
            daily_volatility=self.daily_volatility,
            spread_proxy_bps=self.spread_proxy_bps,
        )


@dataclass(frozen=True)
class RiskObservation:
    """Signals known at a decision close, never at the following execution."""

    benchmark_close: Optional[float] = None
    benchmark_slow: Optional[float] = None
    benchmark_fast: Optional[float] = None
    breadth: Optional[float] = None
    crowding_score: Optional[float] = None
    betas: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class RiskConfig:
    target_volatility: Optional[float] = None
    vol_lookback: int = 20
    min_vol_observations: int = 10
    annualization: int = 252
    drawdown_steps: Tuple[Tuple[float, float], ...] = ()
    max_abs_beta: Optional[float] = None
    trend_filter: bool = False
    breadth_exit: Optional[float] = None
    breadth_enter: Optional[float] = None
    risk_off_multiplier: float = 0.0
    crowding_threshold: Optional[float] = None
    crowding_multiplier: float = 0.5
    target_change_buffer: float = 0.0

    def __post_init__(self) -> None:
        if self.target_volatility is not None:
            if not math.isfinite(self.target_volatility) or self.target_volatility <= 0:
                raise ValueError("target_volatility must be finite and positive")
        if self.vol_lookback <= 1 or self.min_vol_observations <= 1:
            raise ValueError("vol lookback and minimum observations must exceed one")
        if self.min_vol_observations > self.vol_lookback:
            raise ValueError("min_vol_observations cannot exceed vol_lookback")
        if self.annualization <= 0:
            raise ValueError("annualization must be positive")
        if self.max_abs_beta is not None:
            if not math.isfinite(self.max_abs_beta) or self.max_abs_beta <= 0:
                raise ValueError("max_abs_beta must be finite and positive")
        for name, value in {
            "risk_off_multiplier": self.risk_off_multiplier,
            "crowding_multiplier": self.crowding_multiplier,
            "target_change_buffer": self.target_change_buffer,
        }.items():
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.risk_off_multiplier > 1 or self.crowding_multiplier > 1:
            raise ValueError("risk multipliers cannot exceed one")
        if self.breadth_exit is not None and not 0 <= self.breadth_exit <= 1:
            raise ValueError("breadth_exit must be in [0, 1]")
        if self.breadth_enter is not None and not 0 <= self.breadth_enter <= 1:
            raise ValueError("breadth_enter must be in [0, 1]")
        if (
            self.breadth_exit is not None
            and self.breadth_enter is not None
            and self.breadth_enter < self.breadth_exit
        ):
            raise ValueError("breadth_enter must be >= breadth_exit for hysteresis")
        if self.crowding_threshold is not None and not math.isfinite(self.crowding_threshold):
            raise ValueError("crowding_threshold must be finite")
        for threshold, multiplier in self.drawdown_steps:
            if not 0 <= threshold < 1 or not 0 <= multiplier <= 1:
                raise ValueError("drawdown steps require threshold in [0,1) and multiplier in [0,1]")


@dataclass(frozen=True)
class RiskState:
    in_market: bool = True
    peak_equity: float = 0.0


@dataclass(frozen=True)
class RiskLayerDecision:
    layer: str
    triggered: bool
    reason: str
    before_gross: float
    after_gross: float
    observed: Optional[float] = None
    threshold: Optional[float] = None


@dataclass(frozen=True)
class RiskDecision:
    weights: Mapping[str, float]
    state: RiskState
    layers: Tuple[RiskLayerDecision, ...]


@dataclass(frozen=True)
class PendingTarget:
    signal_session: Any
    weights: Mapping[str, float]
    liquidity: Mapping[str, LiquiditySnapshot]
    portfolio_reasons: Tuple[PortfolioReason, ...] = ()
    risk_layers: Tuple[RiskLayerDecision, ...] = ()


@dataclass(frozen=True)
class LedgerRow:
    session: Any
    signal_generated: bool
    executed_signal_session: Optional[Any]
    starting_equity: float
    equity_at_open: float
    overnight_pnl: float
    intraday_pnl: float
    ending_equity: float
    cash: float
    market_value: float
    gross_exposure: float
    net_exposure: float
    traded_notional: float
    gross_turnover: float
    one_way_turnover: float
    commission: float
    spread: float
    impact: float
    funding: float
    total_cost: float
    max_adv_participation: float
    positions: Mapping[str, float]
    executed_target_weights: Mapping[str, float]
    portfolio_reasons: Tuple[PortfolioReason, ...]
    risk_layers: Tuple[RiskLayerDecision, ...]
    value_identity_error: float
    pnl_identity_error: float


@dataclass(frozen=True)
class EngineState:
    cash: float
    shares: Mapping[str, float]
    last_close_prices: Mapping[str, float]
    alpha_target: Mapping[str, float]
    risk_target: Mapping[str, float]
    pending: Optional[PendingTarget]
    stagger_state: StaggerState
    risk_state: RiskState
    returns_history: Tuple[Mapping[str, float], ...]
    equity: float


@dataclass(frozen=True)
class BacktestResult:
    ledger: Tuple[LedgerRow, ...]
    final_state: EngineState

    @property
    def equity_curve(self) -> Tuple[Tuple[Any, float], ...]:
        return tuple((row.session, row.ending_equity) for row in self.ledger)


def _gross(weights: Mapping[str, float]) -> float:
    return sum(abs(float(weight)) for weight in weights.values())


def _net(weights: Mapping[str, float]) -> float:
    return sum(float(weight) for weight in weights.values())


def _scale(weights: Mapping[str, float], scalar: float) -> Dict[str, float]:
    return {
        symbol: float(weight) * float(scalar)
        for symbol, weight in weights.items()
        if abs(float(weight) * float(scalar)) > _EPS
    }


def _weights_differ(a: Mapping[str, float], b: Mapping[str, float], tolerance: float) -> bool:
    max_difference = max(
        (abs(float(a.get(symbol, 0.0)) - float(b.get(symbol, 0.0))) for symbol in set(a) | set(b)),
        default=0.0,
    )
    return max_difference > max(tolerance, _EPS)


def target_weight_realized_volatility(
    weights: Mapping[str, float],
    returns_history: Sequence[Mapping[str, float]],
    lookback: int = 20,
    min_observations: int = 10,
    annualization: int = 252,
) -> Optional[float]:
    """Annualized realized volatility of the *proposed target weights*."""

    active = {symbol: float(weight) for symbol, weight in weights.items() if abs(float(weight)) > _EPS}
    if not active:
        return 0.0
    observations: List[float] = []
    for row in list(returns_history)[-lookback:]:
        if all(symbol in row and math.isfinite(float(row[symbol])) for symbol in active):
            observations.append(sum(weight * float(row[symbol]) for symbol, weight in active.items()))
    if len(observations) < min_observations:
        return None
    return float(np.std(np.asarray(observations, dtype=float), ddof=1) * math.sqrt(annualization))


def target_weight_pairwise_crowding(
    weights: Mapping[str, float],
    returns_history: Sequence[Mapping[str, float]],
    lookback: int = 20,
    min_observations: int = 10,
) -> Optional[float]:
    """Weighted mean pairwise correlation among proposed holdings.

    This is a holdings-concentration proxy, not a claim about external fund
    positioning.  A caller may provide a richer point-in-time crowding score in
    :class:`RiskObservation`; otherwise this transparent OHLCV-only fallback is
    used.
    """

    active = {
        symbol: abs(float(weight))
        for symbol, weight in weights.items()
        if abs(float(weight)) > _EPS
    }
    symbols = sorted(active)
    if len(symbols) < 2:
        return 0.0
    complete_rows = [
        [float(row[symbol]) for symbol in symbols]
        for row in list(returns_history)[-lookback:]
        if all(symbol in row and math.isfinite(float(row[symbol])) for symbol in symbols)
    ]
    if len(complete_rows) < min_observations:
        return None
    matrix = np.asarray(complete_rows, dtype=float)
    correlations = np.corrcoef(matrix, rowvar=False)
    weighted_sum = 0.0
    pair_weight = 0.0
    for left in range(len(symbols)):
        for right in range(left + 1, len(symbols)):
            correlation = float(correlations[left, right])
            if not math.isfinite(correlation):
                continue
            weight = active[symbols[left]] * active[symbols[right]]
            weighted_sum += weight * correlation
            pair_weight += weight
    return weighted_sum / pair_weight if pair_weight > _EPS else None


def _regime_transition(
    state: RiskState,
    observation: RiskObservation,
    config: RiskConfig,
) -> Tuple[bool, str]:
    trend_available = config.trend_filter and all(
        value is not None and math.isfinite(float(value))
        for value in (observation.benchmark_close, observation.benchmark_slow, observation.benchmark_fast)
    )
    breadth_enabled = config.breadth_exit is not None or config.breadth_enter is not None
    breadth_available = observation.breadth is not None and math.isfinite(float(observation.breadth))

    if state.in_market:
        trend_exit = bool(
            trend_available
            and float(observation.benchmark_close) < float(observation.benchmark_slow)
        )
        breadth_exit = bool(
            config.breadth_exit is not None
            and breadth_available
            and float(observation.breadth) < config.breadth_exit
        )
        if trend_exit or breadth_exit:
            triggers = []
            if trend_exit:
                triggers.append("benchmark below slow trend")
            if breadth_exit:
                triggers.append("breadth below exit threshold")
            return False, "; ".join(triggers)
        return True, "risk-on state retained"

    configured_checks: List[bool] = []
    descriptions: List[str] = []
    if trend_available:
        configured_checks.append(float(observation.benchmark_close) > float(observation.benchmark_fast))
        descriptions.append("benchmark above fast recovery trend")
    if breadth_enabled:
        threshold = config.breadth_enter if config.breadth_enter is not None else config.breadth_exit
        configured_checks.append(bool(breadth_available and float(observation.breadth) > float(threshold)))
        descriptions.append("breadth above re-entry threshold")
    if configured_checks and all(configured_checks):
        return True, "; ".join(descriptions)
    return False, "risk-off hysteresis retained until all re-entry conditions pass"


def apply_risk_overlays(
    target_weights: Mapping[str, float],
    returns_history: Sequence[Mapping[str, float]],
    equity: float,
    state: RiskState,
    observation: Optional[RiskObservation],
    config: RiskConfig,
) -> RiskDecision:
    """Apply independently auditable risk layers in a fixed order."""

    observation = observation or RiskObservation()
    equity = float(equity)
    if not math.isfinite(equity) or equity <= 0:
        raise ValueError("equity must be finite and positive")
    peak = max(float(state.peak_equity), equity)
    working = {symbol: float(weight) for symbol, weight in target_weights.items() if abs(float(weight)) > _EPS}
    layers: List[RiskLayerDecision] = []

    if config.target_volatility is not None:
        before = _gross(working)
        realized = target_weight_realized_volatility(
            working,
            returns_history,
            lookback=config.vol_lookback,
            min_observations=config.min_vol_observations,
            annualization=config.annualization,
        )
        if realized is None:
            layers.append(
                RiskLayerDecision(
                    "target_weight_volatility",
                    False,
                    "insufficient complete point-in-time return observations; exposure unchanged",
                    before,
                    before,
                    None,
                    config.target_volatility,
                )
            )
        else:
            scalar = min(1.0, config.target_volatility / realized) if realized > _EPS else 1.0
            working = _scale(working, scalar)
            layers.append(
                RiskLayerDecision(
                    "target_weight_volatility",
                    scalar < 1.0 - _EPS,
                    (
                        f"scaled gross by {scalar:.6f} to target realized portfolio volatility"
                        if scalar < 1.0 - _EPS
                        else "target-weight realized volatility is within budget"
                    ),
                    before,
                    _gross(working),
                    realized,
                    config.target_volatility,
                )
            )

    if config.drawdown_steps:
        before = _gross(working)
        drawdown = max(0.0, 1.0 - equity / peak) if peak > 0 else 0.0
        multiplier = 1.0
        triggered_thresholds = []
        for threshold, candidate in sorted(config.drawdown_steps):
            if drawdown + _EPS >= threshold:
                multiplier = min(multiplier, candidate)
                triggered_thresholds.append(threshold)
        working = _scale(working, multiplier)
        layers.append(
            RiskLayerDecision(
                "drawdown_governor",
                multiplier < 1.0 - _EPS,
                (
                    f"drawdown crossed {max(triggered_thresholds):.2%}; exposure multiplier {multiplier:.4f}"
                    if triggered_thresholds
                    else "drawdown remains below all governor steps"
                ),
                before,
                _gross(working),
                drawdown,
                min((step[0] for step in config.drawdown_steps), default=None),
            )
        )

    if config.max_abs_beta is not None:
        before = _gross(working)
        missing = [symbol for symbol in working if symbol not in observation.betas]
        if missing:
            raise KeyError(f"missing point-in-time beta proxy for {', '.join(sorted(missing))}")
        beta = sum(float(weight) * float(observation.betas[symbol]) for symbol, weight in working.items())
        scalar = min(1.0, config.max_abs_beta / abs(beta)) if abs(beta) > _EPS else 1.0
        working = _scale(working, scalar)
        layers.append(
            RiskLayerDecision(
                "beta_proxy",
                scalar < 1.0 - _EPS,
                (
                    f"scaled gross by {scalar:.6f} to enforce beta proxy cap"
                    if scalar < 1.0 - _EPS
                    else "target beta proxy is within budget"
                ),
                before,
                _gross(working),
                beta,
                config.max_abs_beta,
            )
        )

    # Supplying a rich observation must not silently enable a risk layer.  The
    # old behaviour made a configuration labelled "none" trade a slow-trend
    # cash overlay whenever callers passed benchmark context.
    regime_enabled = (
        config.trend_filter
        or config.breadth_exit is not None
        or config.breadth_enter is not None
    )
    in_market = state.in_market
    regime_reason = "regime overlay disabled"
    if regime_enabled:
        in_market, regime_reason = _regime_transition(
            RiskState(in_market=state.in_market, peak_equity=peak), observation, config
        )
        before = _gross(working)
        if not in_market:
            working = _scale(working, config.risk_off_multiplier)
        layers.append(
            RiskLayerDecision(
                "breadth_regime_hysteresis",
                not in_market,
                regime_reason,
                before,
                _gross(working),
                observation.breadth,
                config.breadth_exit if in_market else config.breadth_enter,
            )
        )

    if config.crowding_threshold is not None:
        before = _gross(working)
        score = observation.crowding_score
        score_source = "supplied point-in-time"
        if score is None:
            score = target_weight_pairwise_crowding(
                working,
                returns_history,
                lookback=config.vol_lookback,
                min_observations=config.min_vol_observations,
            )
            score_source = "holdings pairwise-correlation proxy"
        triggered = bool(score is not None and math.isfinite(float(score)) and float(score) > config.crowding_threshold)
        if triggered:
            working = _scale(working, config.crowding_multiplier)
            reason = (
                f"{score_source} crowding score {float(score):.6f} exceeds threshold; "
                f"exposure multiplier {config.crowding_multiplier:.4f}"
            )
        elif score is None:
            reason = "crowding input unavailable; exposure unchanged and layer reported"
        else:
            reason = f"{score_source} crowding score is within budget"
        layers.append(
            RiskLayerDecision(
                "crowding",
                triggered,
                reason,
                before,
                _gross(working),
                None if score is None else float(score),
                config.crowding_threshold,
            )
        )

    return RiskDecision(
        weights=working,
        state=RiskState(in_market=in_market, peak_equity=peak),
        layers=tuple(layers),
    )


def _weights_from_positions(
    shares: Mapping[str, float],
    prices: Mapping[str, float],
    equity: float,
) -> Dict[str, float]:
    if equity <= 0:
        raise ValueError("equity must be positive")
    result = {}
    for symbol, quantity in shares.items():
        if abs(quantity) <= _EPS:
            continue
        if symbol not in prices:
            raise MissingHeldReturnError(f"held symbol {symbol} has no price")
        result[symbol] = float(quantity) * float(prices[symbol]) / equity
    return result


def _liquidity_for_symbols(
    bars: Mapping[str, MarketBar],
    symbols: Sequence[str],
) -> Dict[str, LiquiditySnapshot]:
    result: Dict[str, LiquiditySnapshot] = {}
    for symbol in sorted(set(symbols)):
        if symbol not in bars:
            raise MissingHeldReturnError(
                f"symbol {symbol} needed by the close decision has no close(t) bar"
            )
        result[symbol] = bars[symbol].liquidity()
    return result


def run_backtest(
    market: Mapping[Any, Mapping[str, MarketBar]],
    signals: Mapping[Any, Mapping[str, float]],
    sectors: Mapping[str, str],
    portfolio_config: PortfolioConfig,
    cost_config: CostConfig,
    risk_config: Optional[RiskConfig] = None,
    risk_observations: Optional[Mapping[Any, RiskObservation]] = None,
    *,
    initial_capital: float = 100_000.0,
    accounting_tolerance: float = 1e-8,
) -> BacktestResult:
    """Run a deterministic close-signal/next-open execution simulation."""

    risk_config = risk_config or RiskConfig()
    risk_observations = risk_observations or {}
    initial_capital = float(initial_capital)
    if not math.isfinite(initial_capital) or initial_capital <= 0:
        raise ValueError("initial_capital must be finite and positive")
    sessions = sorted(market)
    if not sessions:
        raise ValueError("market is empty")

    cash = initial_capital
    shares: Dict[str, float] = {}
    last_close_prices: Dict[str, float] = {}
    alpha_target: Dict[str, float] = {}
    last_risk_target: Dict[str, float] = {}
    pending: Optional[PendingTarget] = None
    stagger_state = StaggerState()
    risk_state = RiskState(in_market=True, peak_equity=initial_capital)
    returns_history: List[Mapping[str, float]] = []
    ledger: List[LedgerRow] = []
    previous_equity = initial_capital

    for session in sessions:
        bars = market[session]
        if bars is None:
            raise ValueError(f"market[{session!r}] is None")

        # A missing held bar is not a zero return.  Stop before any accounting
        # mutation so the data defect cannot be hidden in the equity curve.
        for symbol, quantity in shares.items():
            if abs(quantity) > _EPS and symbol not in bars:
                raise MissingHeldReturnError(
                    f"held symbol {symbol} has no bar on session {session!r}"
                )

        starting_equity = previous_equity
        open_prices = {symbol: bar.open for symbol, bar in bars.items()}
        close_prices = {symbol: bar.close for symbol, bar in bars.items()}
        equity_at_open = cash + sum(
            quantity * open_prices[symbol]
            for symbol, quantity in shares.items()
            if abs(quantity) > _EPS
        )
        if equity_at_open <= 0:
            raise AccountingError(f"non-positive equity at open on {session!r}")
        overnight_pnl = equity_at_open - starting_equity

        execution_cost = CostBreakdown()
        executed_signal_session = None
        executed_target: Dict[str, float] = {}
        execution_portfolio_reasons: Tuple[PortfolioReason, ...] = ()
        execution_risk_layers: Tuple[RiskLayerDecision, ...] = ()

        if pending is not None:
            executed_signal_session = pending.signal_session
            execution_portfolio_reasons = pending.portfolio_reasons
            execution_risk_layers = pending.risk_layers
            for symbol in pending.weights:
                if symbol not in bars:
                    raise MissingHeldReturnError(
                        f"target symbol {symbol} from close({pending.signal_session!r}) "
                        f"has no open on {session!r}"
                    )

            pretrade_weights = _weights_from_positions(shares, open_prices, equity_at_open)
            lagged_adv = {
                symbol: point.adv_dollars for symbol, point in pending.liquidity.items()
            }
            executed_target, _participation, clipped = apply_adv_participation_cap(
                pretrade_weights,
                pending.weights,
                equity_at_open,
                lagged_adv,
                cost_config.max_adv_participation,
            )
            if clipped:
                execution_portfolio_reasons = execution_portfolio_reasons + (
                    PortfolioReason(
                        layer="execution_adv_recheck",
                        triggered=True,
                        message=(
                            f"overnight repricing required execution-time ADV clipping for "
                            f"{', '.join(clipped)}"
                        ),
                        before_gross=_gross(pending.weights),
                        after_gross=_gross(executed_target),
                    ),
                )

            execution_cost = estimate_execution_costs(
                pretrade_weights,
                executed_target,
                equity_at_open,
                pending.liquidity,
                cost_config,
                enforce_adv_limit=True,
            )

            all_symbols = sorted(set(shares) | set(executed_target))
            new_shares: Dict[str, float] = {}
            trade_cash_flow = 0.0
            for symbol in all_symbols:
                if symbol not in open_prices:
                    raise MissingHeldReturnError(
                        f"symbol {symbol} cannot be valued for execution on {session!r}"
                    )
                old_quantity = shares.get(symbol, 0.0)
                target_quantity = executed_target.get(symbol, 0.0) * equity_at_open / open_prices[symbol]
                delta_quantity = target_quantity - old_quantity
                trade_cash_flow += delta_quantity * open_prices[symbol]
                if abs(target_quantity) > _EPS:
                    new_shares[symbol] = target_quantity
            cash -= trade_cash_flow
            cash -= execution_cost.execution_total
            shares = new_shares
            pending = None

        market_value_before_funding = sum(
            quantity * close_prices[symbol]
            for symbol, quantity in shares.items()
            if abs(quantity) > _EPS
        )
        equity_before_funding = cash + market_value_before_funding
        if equity_before_funding <= 0:
            raise AccountingError(f"non-positive equity before funding on {session!r}")
        close_weights_before_funding = _weights_from_positions(
            shares, close_prices, equity_before_funding
        )
        funding_breakdown: FundingBreakdown = estimate_daily_funding(
            close_weights_before_funding,
            equity_before_funding,
            cost_config,
        )
        cash -= funding_breakdown.total
        ending_equity = cash + market_value_before_funding
        if ending_equity <= 0:
            raise AccountingError(f"non-positive ending equity on {session!r}")
        intraday_pnl = equity_before_funding - (
            equity_at_open - execution_cost.execution_total
        )

        close_weights = _weights_from_positions(shares, close_prices, ending_equity)
        market_value = sum(quantity * close_prices[symbol] for symbol, quantity in shares.items())
        value_identity_error = ending_equity - (cash + market_value)
        pnl_identity_error = ending_equity - (
            starting_equity
            + overnight_pnl
            + intraday_pnl
            - execution_cost.execution_total
            - funding_breakdown.total
        )
        scale = max(abs(ending_equity), 1.0)
        if (
            abs(value_identity_error) > accounting_tolerance * scale
            or abs(pnl_identity_error) > accounting_tolerance * scale
        ):
            raise AccountingError(
                f"accounting identity failed on {session!r}: "
                f"value={value_identity_error}, pnl={pnl_identity_error}"
            )

        # Close-to-close asset returns become observable now and may be used by
        # this close's risk decision.  Only consecutive observations are kept;
        # a missing unheld bar is never converted into a multi-day one-day return.
        return_row = {
            symbol: close_prices[symbol] / last_close_prices[symbol] - 1.0
            for symbol in set(close_prices) & set(last_close_prices)
        }
        if return_row:
            returns_history.append(return_row)
            max_history = max(risk_config.vol_lookback * 3, risk_config.min_vol_observations)
            if len(returns_history) > max_history:
                returns_history = returns_history[-max_history:]
        last_close_prices = dict(close_prices)

        signal_generated = session in signals
        close_portfolio_reasons: Tuple[PortfolioReason, ...] = ()
        if signal_generated:
            scores = signals[session]
            volatility = {symbol: bar.daily_volatility for symbol, bar in bars.items()}
            adv = {symbol: bar.adv_dollars for symbol, bar in bars.items()}
            decision = construct_portfolio(
                scores=scores,
                current_weights=close_weights,
                volatility=volatility,
                sectors=sectors,
                adv_dollars=adv,
                equity=ending_equity,
                config=portfolio_config,
                stagger_state=stagger_state,
            )
            alpha_target = dict(decision.target_weights)
            stagger_state = decision.stagger_state
            close_portfolio_reasons = decision.reasons

        supplied_observation = risk_observations.get(session)
        if supplied_observation is None:
            observation = RiskObservation(betas={symbol: bar.beta for symbol, bar in bars.items()})
        elif supplied_observation.betas:
            observation = supplied_observation
        else:
            observation = replace(
                supplied_observation,
                betas={symbol: bar.beta for symbol, bar in bars.items()},
            )

        risk_decision = apply_risk_overlays(
            alpha_target,
            returns_history,
            ending_equity,
            risk_state,
            observation,
            risk_config,
        )
        risk_state = risk_decision.state
        proposed_risk_target = dict(risk_decision.weights)
        risk_changed = _weights_differ(
            proposed_risk_target,
            last_risk_target,
            risk_config.target_change_buffer,
        )

        if signal_generated or risk_changed:
            needed_symbols = tuple(set(close_weights) | set(proposed_risk_target))
            liquidity = _liquidity_for_symbols(bars, needed_symbols)
            pending = PendingTarget(
                signal_session=session,
                weights=proposed_risk_target,
                liquidity=liquidity,
                portfolio_reasons=close_portfolio_reasons,
                risk_layers=risk_decision.layers,
            )
            last_risk_target = proposed_risk_target

        daily_cost = execution_cost.with_funding(funding_breakdown)
        ledger.append(
            LedgerRow(
                session=session,
                signal_generated=signal_generated,
                executed_signal_session=executed_signal_session,
                starting_equity=starting_equity,
                equity_at_open=equity_at_open,
                overnight_pnl=overnight_pnl,
                intraday_pnl=intraday_pnl,
                ending_equity=ending_equity,
                cash=cash,
                market_value=market_value,
                gross_exposure=_gross(close_weights),
                net_exposure=_net(close_weights),
                traded_notional=execution_cost.traded_notional,
                gross_turnover=execution_cost.gross_turnover,
                one_way_turnover=execution_cost.one_way_turnover,
                commission=execution_cost.commission,
                spread=execution_cost.spread,
                impact=execution_cost.impact,
                funding=funding_breakdown.total,
                total_cost=daily_cost.total,
                max_adv_participation=execution_cost.max_adv_participation,
                positions=dict(shares),
                executed_target_weights=dict(executed_target),
                portfolio_reasons=execution_portfolio_reasons,
                risk_layers=execution_risk_layers,
                value_identity_error=value_identity_error,
                pnl_identity_error=pnl_identity_error,
            )
        )
        previous_equity = ending_equity

    final_state = EngineState(
        cash=cash,
        shares=dict(shares),
        last_close_prices=dict(last_close_prices),
        alpha_target=dict(alpha_target),
        risk_target=dict(last_risk_target),
        pending=pending,
        stagger_state=stagger_state,
        risk_state=risk_state,
        returns_history=tuple(dict(row) for row in returns_history),
        equity=previous_equity,
    )
    return BacktestResult(ledger=tuple(ledger), final_state=final_state)


def assert_accounting_identity(
    result: BacktestResult,
    *,
    relative_tolerance: float = 1e-9,
) -> None:
    """Raise when any recorded daily accounting identity is inconsistent."""

    for row in result.ledger:
        scale = max(abs(row.ending_equity), 1.0)
        if abs(row.value_identity_error) > relative_tolerance * scale:
            raise AssertionError(
                f"value identity failed on {row.session!r}: {row.value_identity_error}"
            )
        if abs(row.pnl_identity_error) > relative_tolerance * scale:
            raise AssertionError(
                f"PnL identity failed on {row.session!r}: {row.pnl_identity_error}"
            )
