"""Unrounded performance and audit metrics for a Research v2 ledger."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class PerformanceMetrics:
    initial_equity: float
    final_equity: float
    total_return: float
    cagr: float
    annualized_volatility: float
    sharpe: float
    sortino: float
    max_drawdown: float
    calmar: float
    win_rate: float
    periods: int
    total_gross_turnover: float
    total_one_way_turnover: float
    total_traded_notional: float
    commission: float
    spread: float
    impact: float
    funding: float
    total_cost: float
    cost_to_initial_equity: float
    average_gross_exposure: float
    average_net_exposure: float
    max_adv_participation: float
    max_value_identity_error: float
    max_pnl_identity_error: float

    def to_dict(self) -> Mapping[str, float | int]:
        return asdict(self)


def _read(row: Any, name: str) -> float:
    if isinstance(row, Mapping):
        return float(row[name])
    return float(getattr(row, name))


def compute_performance_metrics(
    ledger: Sequence[Any],
    *,
    periods_per_year: int = 252,
    annual_risk_free_rate: float = 0.0,
) -> PerformanceMetrics:
    """Compute net-of-cost metrics directly from daily accounting rows."""

    if not ledger:
        raise ValueError("ledger is empty")
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive")
    if not math.isfinite(annual_risk_free_rate):
        raise ValueError("annual_risk_free_rate must be finite")

    initial = _read(ledger[0], "starting_equity")
    final = _read(ledger[-1], "ending_equity")
    if initial <= 0 or final <= 0:
        raise ValueError("equity must stay positive")

    daily_returns = np.asarray(
        [
            _read(row, "ending_equity") / _read(row, "starting_equity") - 1.0
            for row in ledger
        ],
        dtype=float,
    )
    periods = len(daily_returns)
    total_return = final / initial - 1.0
    cagr = (final / initial) ** (periods_per_year / periods) - 1.0
    annualized_volatility = (
        float(np.std(daily_returns, ddof=1) * math.sqrt(periods_per_year))
        if periods > 1
        else 0.0
    )
    daily_rf = (1.0 + annual_risk_free_rate) ** (1.0 / periods_per_year) - 1.0
    excess = daily_returns - daily_rf
    excess_std = float(np.std(excess, ddof=1)) if periods > 1 else 0.0
    sharpe = (
        float(np.mean(excess) / excess_std * math.sqrt(periods_per_year))
        if excess_std > 0
        else 0.0
    )
    downside = np.minimum(excess, 0.0)
    downside_deviation = float(np.sqrt(np.mean(np.square(downside))))
    sortino = (
        float(np.mean(excess) / downside_deviation * math.sqrt(periods_per_year))
        if downside_deviation > 0
        else 0.0
    )

    equity = np.asarray([_read(row, "ending_equity") for row in ledger], dtype=float)
    equity_with_initial = np.concatenate(([initial], equity))
    peaks = np.maximum.accumulate(equity_with_initial)
    drawdowns = equity_with_initial / peaks - 1.0
    max_drawdown = float(np.min(drawdowns))
    calmar = cagr / abs(max_drawdown) if max_drawdown < 0 else 0.0

    def total(name: str) -> float:
        return float(sum(_read(row, name) for row in ledger))

    commission = total("commission")
    spread = total("spread")
    impact = total("impact")
    funding = total("funding")
    total_cost = commission + spread + impact + funding

    return PerformanceMetrics(
        initial_equity=initial,
        final_equity=final,
        total_return=total_return,
        cagr=cagr,
        annualized_volatility=annualized_volatility,
        sharpe=sharpe,
        sortino=sortino,
        max_drawdown=max_drawdown,
        calmar=calmar,
        win_rate=float(np.mean(daily_returns > 0)),
        periods=periods,
        total_gross_turnover=total("gross_turnover"),
        total_one_way_turnover=total("one_way_turnover"),
        total_traded_notional=total("traded_notional"),
        commission=commission,
        spread=spread,
        impact=impact,
        funding=funding,
        total_cost=total_cost,
        cost_to_initial_equity=total_cost / initial,
        average_gross_exposure=float(np.mean([_read(row, "gross_exposure") for row in ledger])),
        average_net_exposure=float(np.mean([_read(row, "net_exposure") for row in ledger])),
        max_adv_participation=max(_read(row, "max_adv_participation") for row in ledger),
        max_value_identity_error=max(abs(_read(row, "value_identity_error")) for row in ledger),
        max_pnl_identity_error=max(abs(_read(row, "pnl_identity_error")) for row in ledger),
    )

