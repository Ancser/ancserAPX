"""Staged, cost-aware portfolio search for frozen Research v2 predictions.

The search is deliberately downstream of model training.  It accepts frozen
out-of-sample scores and uses the existing event-driven backtester; it never
imports production data, scheduling, broker, or execution modules.

The information boundary is explicit:

* Stage A selects a score on the *selection OOS* period.
* Stage B selects construction, cadence, weighting, and risk overlays.
* Stage C selects leverage at a pre-declared fixed-friction assumption while
  also reporting the complete 0/5/10/20 bps sensitivity curve.
* The resulting configuration is frozen before the lockbox frame is inspected.

Each candidate is run once over the complete chronological OOS interval.  Fold
statistics are slices of that one continuous cash/position ledger, so holdings,
cost basis, staggered tranches, drawdown state, and risk history are never reset
at fold boundaries.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from hashlib import sha256
import csv
import json
import math
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

from .backtest import RiskConfig
from .costs import CostConfig
from .experiment import EvaluationResult, MarketContext, evaluate_strategy, make_signal_map
from .metrics import PerformanceMetrics, compute_performance_metrics
from .portfolio import PortfolioConfig
from .safety import RESEARCH_ROOT, ensure_research_output_path


ProgressCallback = Callable[[Mapping[str, object]], None]


@dataclass(frozen=True)
class Cadence:
    """A close-signal schedule and its explicit staggered sub-portfolios."""

    name: str
    rebalance_days: int
    staggered_tranches: int = 1

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("cadence name cannot be empty")
        if self.rebalance_days <= 0 or self.staggered_tranches <= 0:
            raise ValueError("cadence days and tranches must be positive")


@dataclass(frozen=True)
class SearchPolicy:
    """Pre-declared selection objective and portfolio-level acceptance gates."""

    objective: str = "sharpe"
    require_positive_worst_fold: bool = False
    max_drawdown_limit: float = -0.35

    def __post_init__(self) -> None:
        allowed = {"sharpe", "sortino", "calmar", "cagr", "total_return"}
        if self.objective not in allowed:
            raise ValueError(f"objective must be one of {sorted(allowed)}")
        if not math.isfinite(self.max_drawdown_limit):
            raise ValueError("max_drawdown_limit must be finite")
        if not -1.0 <= self.max_drawdown_limit <= 0.0:
            raise ValueError("max_drawdown_limit must be in [-1, 0]")


@dataclass(frozen=True)
class SearchCandidate:
    """All inputs needed to reproduce one portfolio simulation."""

    score_column: str
    portfolio_config: PortfolioConfig
    risk_config: RiskConfig
    risk_variant: str
    cadence: Cadence
    extra_cost_bps: float
    rebalance_offset: int = 0

    def __post_init__(self) -> None:
        if not self.score_column:
            raise ValueError("score_column cannot be empty")
        if not self.risk_variant:
            raise ValueError("risk_variant cannot be empty")
        if not math.isfinite(self.extra_cost_bps) or self.extra_cost_bps < 0:
            raise ValueError("extra_cost_bps must be finite and non-negative")
        if not 0 <= self.rebalance_offset < self.cadence.rebalance_days:
            raise ValueError("rebalance_offset is outside the cadence")
        if self.portfolio_config.staggered_tranches != self.cadence.staggered_tranches:
            raise ValueError("portfolio config and cadence disagree on tranche count")

    @property
    def candidate_id(self) -> str:
        payload = json.dumps(self.to_dict(include_id=False), sort_keys=True, separators=(",", ":"))
        return sha256(payload.encode("utf-8")).hexdigest()[:16]

    def to_dict(self, *, include_id: bool = True) -> Dict[str, object]:
        result: Dict[str, object] = {
            "score_column": self.score_column,
            "risk_variant": self.risk_variant,
            "cadence": asdict(self.cadence),
            "extra_cost_bps": self.extra_cost_bps,
            "rebalance_offset": self.rebalance_offset,
            "portfolio_config": asdict(self.portfolio_config),
            "risk_config": asdict(self.risk_config),
        }
        if include_id:
            result["candidate_id"] = self.candidate_id
        return result


@dataclass(frozen=True)
class SearchEvaluation:
    """Compact audit result; the large daily ledger is intentionally omitted."""

    stage: str
    label: str
    candidate: SearchCandidate
    metrics: Mapping[str, float | int]
    fold_metrics: Mapping[str, Mapping[str, float | int]]
    worst_metrics: Mapping[str, object]
    cost_breakdown: Mapping[str, float]
    objective_value: float
    eligible: bool
    gate_failures: Tuple[str, ...]
    ledger_rows: int
    continuous_oos_state: bool = True

    def to_dict(self) -> Dict[str, object]:
        return {
            "stage": self.stage,
            "label": self.label,
            "candidate": self.candidate.to_dict(),
            "metrics": dict(self.metrics),
            "fold_metrics": {key: dict(value) for key, value in self.fold_metrics.items()},
            "worst_metrics": dict(self.worst_metrics),
            "cost_breakdown": dict(self.cost_breakdown),
            "objective_value": self.objective_value,
            "eligible": self.eligible,
            "gate_failures": list(self.gate_failures),
            "ledger_rows": self.ledger_rows,
            "continuous_oos_state": self.continuous_oos_state,
        }


@dataclass(frozen=True)
class SearchResult:
    stage_a: Tuple[SearchEvaluation, ...]
    stage_b: Tuple[SearchEvaluation, ...]
    stage_c: Tuple[SearchEvaluation, ...]
    champion: SearchEvaluation
    lockbox: Optional[SearchEvaluation]
    neighborhood: Tuple[SearchEvaluation, ...]
    offset_sensitivity: Tuple[SearchEvaluation, ...]
    audit: Mapping[str, object]

    def to_dict(self) -> Dict[str, object]:
        payload = {
            "stage_a": [item.to_dict() for item in self.stage_a],
            "stage_b": [item.to_dict() for item in self.stage_b],
            "stage_c": [item.to_dict() for item in self.stage_c],
            "champion": self.champion.to_dict(),
            "lockbox": self.lockbox.to_dict() if self.lockbox is not None else None,
            "neighborhood": [item.to_dict() for item in self.neighborhood],
            "offset_sensitivity": [item.to_dict() for item in self.offset_sensitivity],
            "audit": dict(self.audit),
        }
        return _json_safe(payload)


def _json_safe(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def _emit(progress: Optional[ProgressCallback], **event: object) -> None:
    if progress is not None:
        progress(dict(event))


def _normalise_cadences(values: Sequence[Cadence | Tuple[int, int] | int]) -> Tuple[Cadence, ...]:
    result: List[Cadence] = []
    for raw in values:
        if isinstance(raw, Cadence):
            value = raw
        elif isinstance(raw, int):
            value = Cadence(f"every_{raw}d", raw, 1)
        else:
            days, tranches = raw
            name = "daily_5_tranches" if days == 1 and tranches == 5 else f"every_{days}d_{tranches}t"
            value = Cadence(name, int(days), int(tranches))
        if value not in result:
            result.append(value)
    if not result:
        raise ValueError("cadence grid cannot be empty")
    return tuple(result)


def _validate_selection_predictions(frame: pd.DataFrame, score_columns: Sequence[str]) -> pd.DataFrame:
    required = {"timestamp", "symbol", "fold_id", *score_columns}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"selection predictions missing columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("selection predictions are empty")
    clean = frame.copy()
    clean["timestamp"] = pd.to_datetime(clean["timestamp"])
    clean["symbol"] = clean["symbol"].astype(str)
    clean["fold_id"] = clean["fold_id"].astype(str)
    if clean.duplicated(["timestamp", "symbol"]).any():
        raise ValueError("selection predictions contain duplicate timestamp/symbol rows")
    if (clean["fold_id"] == "LOCKBOX").any():
        raise ValueError("selection predictions cannot contain LOCKBOX rows")
    _fold_bounds(clean)
    return clean.sort_values(["timestamp", "symbol"]).reset_index(drop=True)


def _validate_lockbox_predictions(
    frame: pd.DataFrame,
    *,
    score_column: str,
    selection_end: pd.Timestamp,
) -> pd.DataFrame:
    required = {"timestamp", "symbol", score_column}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"lockbox predictions missing columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("lockbox predictions are empty")
    clean = frame.copy()
    clean["timestamp"] = pd.to_datetime(clean["timestamp"])
    clean["symbol"] = clean["symbol"].astype(str)
    if clean.duplicated(["timestamp", "symbol"]).any():
        raise ValueError("lockbox predictions contain duplicate timestamp/symbol rows")
    if clean["timestamp"].min() <= selection_end:
        raise ValueError("lockbox must start strictly after the selection OOS period")
    clean["fold_id"] = "LOCKBOX"
    return clean.sort_values(["timestamp", "symbol"]).reset_index(drop=True)


def _fold_bounds(frame: pd.DataFrame) -> Tuple[Tuple[str, pd.Timestamp, pd.Timestamp], ...]:
    date_fold = frame[["timestamp", "fold_id"]].drop_duplicates()
    if date_fold.duplicated("timestamp").any():
        raise ValueError("one OOS date is assigned to multiple folds")
    groups = []
    for fold_id, part in date_fold.groupby("fold_id", sort=False):
        groups.append((str(fold_id), pd.Timestamp(part["timestamp"].min()), pd.Timestamp(part["timestamp"].max())))
    groups.sort(key=lambda item: (item[1], item[0]))
    for previous, current in zip(groups, groups[1:]):
        if current[1] <= previous[2]:
            raise ValueError("fold date ranges overlap or interleave")
    return tuple(groups)


def _slice_context(context: MarketContext, start: pd.Timestamp, end: pd.Timestamp) -> MarketContext:
    sessions = tuple(
        session for session in context.sessions
        if start <= pd.Timestamp(session) <= end
    )
    if not sessions:
        raise ValueError(f"market context has no sessions in [{start}, {end}]")
    available = {pd.Timestamp(session) for session in sessions}
    if start not in available or end not in available:
        raise ValueError("prediction boundary is not an executable market session")
    metadata = dict(context.metadata)
    metadata.update({"search_slice_start": str(start), "search_slice_end": str(end)})
    return MarketContext(
        market={session: context.market[session] for session in sessions},
        full_risk_observations={
            session: context.full_risk_observations[session]
            for session in sessions if session in context.full_risk_observations
        },
        beta_only_observations={
            session: context.beta_only_observations[session]
            for session in sessions if session in context.beta_only_observations
        },
        sectors=context.sectors,
        sessions=sessions,
        symbols=context.symbols,
        metadata=metadata,
    )


def _metrics_dict(metrics: PerformanceMetrics) -> Dict[str, float | int]:
    return dict(metrics.to_dict())


def _cost_breakdown(metrics: PerformanceMetrics, extra_cost_bps: float) -> Dict[str, float]:
    return {
        "extra_fixed_friction_bps": float(extra_cost_bps),
        "commission": metrics.commission,
        "spread": metrics.spread,
        "market_impact": metrics.impact,
        "funding": metrics.funding,
        "execution_cost": metrics.commission + metrics.spread + metrics.impact,
        "total_cost": metrics.total_cost,
        "cost_to_initial_equity": metrics.cost_to_initial_equity,
        "gross_turnover": metrics.total_gross_turnover,
        "one_way_turnover": metrics.total_one_way_turnover,
        "traded_notional": metrics.total_traded_notional,
        "max_adv_participation": metrics.max_adv_participation,
    }


def _worst_fold_summary(
    fold_metrics: Mapping[str, Mapping[str, float | int]], objective: str
) -> Dict[str, object]:
    if not fold_metrics:
        raise ValueError("fold metrics are empty")
    objective_fold = min(fold_metrics, key=lambda key: float(fold_metrics[key][objective]))
    return_fold = min(fold_metrics, key=lambda key: float(fold_metrics[key]["total_return"]))
    drawdown_fold = min(fold_metrics, key=lambda key: float(fold_metrics[key]["max_drawdown"]))
    return {
        "objective_name": objective,
        "objective_fold": objective_fold,
        "objective_value": float(fold_metrics[objective_fold][objective]),
        "objective_fold_metrics": dict(fold_metrics[objective_fold]),
        "return_fold": return_fold,
        "minimum_total_return": float(fold_metrics[return_fold]["total_return"]),
        "drawdown_fold": drawdown_fold,
        "worst_max_drawdown": float(fold_metrics[drawdown_fold]["max_drawdown"]),
    }


def _evaluate_candidate(
    context: MarketContext,
    signal_frame: pd.DataFrame,
    report_frame: pd.DataFrame,
    *,
    candidate: SearchCandidate,
    base_cost: CostConfig,
    policy: SearchPolicy,
    initial_capital: float,
    stage: str,
    label: str,
) -> SearchEvaluation:
    signal_start = pd.Timestamp(signal_frame["timestamp"].min())
    signal_end = pd.Timestamp(signal_frame["timestamp"].max())
    sliced = _slice_context(context, signal_start, signal_end)

    # Standardise the temporary column because pandas.itertuples sanitises names
    # containing spaces/punctuation, while make_signal_map intentionally accepts
    # any valid DataFrame column name.
    scores = signal_frame[["timestamp", "symbol", candidate.score_column]].copy()
    scores["search_score"] = scores[candidate.score_column]
    signals = make_signal_map(
        scores,
        score_column="search_score",
        eligible_symbols=sliced.symbols,
        rebalance_days=candidate.cadence.rebalance_days,
        offset=candidate.rebalance_offset,
        liquidate_at_end=False,
    )
    if not signals:
        raise ValueError(f"candidate {candidate.candidate_id} generated no signals")

    cost = replace(
        base_cost,
        commission_bps=base_cost.commission_bps + candidate.extra_cost_bps,
    )
    evaluation: EvaluationResult = evaluate_strategy(
        sliced,
        signals,
        portfolio_config=candidate.portfolio_config,
        cost_config=cost,
        risk_config=candidate.risk_config,
        use_full_risk_observations=True,
        initial_capital=initial_capital,
    )

    bounds = _fold_bounds(report_frame)
    report_rows = []
    per_fold: Dict[str, Dict[str, float | int]] = {}
    for fold_id, start, end in bounds:
        rows = [
            row for row in evaluation.result.ledger
            if start <= pd.Timestamp(row.session) <= end
        ]
        if not rows:
            raise ValueError(f"fold {fold_id} has no executable ledger rows")
        report_rows.extend(rows)
        per_fold[fold_id] = _metrics_dict(compute_performance_metrics(rows))
    # Bounds are non-overlapping and ordered, but sort explicitly for audit-safe
    # aggregation if caller fold identifiers were not lexicographic.
    report_rows.sort(key=lambda row: pd.Timestamp(row.session))
    metrics_obj = compute_performance_metrics(report_rows)
    metrics = _metrics_dict(metrics_obj)
    worst = _worst_fold_summary(per_fold, policy.objective)

    failures: List[str] = []
    if float(metrics["max_drawdown"]) < policy.max_drawdown_limit:
        failures.append(
            f"max_drawdown {metrics['max_drawdown']:.6f} below limit {policy.max_drawdown_limit:.6f}"
        )
    if policy.require_positive_worst_fold and float(worst["minimum_total_return"]) <= 0:
        failures.append(
            f"worst fold total return {worst['minimum_total_return']:.6f} is not positive"
        )
    return SearchEvaluation(
        stage=stage,
        label=label,
        candidate=candidate,
        metrics=metrics,
        fold_metrics=per_fold,
        worst_metrics=worst,
        cost_breakdown=_cost_breakdown(metrics_obj, candidate.extra_cost_bps),
        objective_value=float(metrics[policy.objective]),
        eligible=not failures,
        gate_failures=tuple(failures),
        ledger_rows=len(report_rows),
    )


def _choose(records: Sequence[SearchEvaluation], policy: SearchPolicy) -> SearchEvaluation:
    if not records:
        raise ValueError("cannot select from an empty result set")
    ordered = sorted(records, key=lambda item: item.candidate.candidate_id)
    eligible = [item for item in ordered if item.eligible]
    pool = eligible or ordered
    return max(
        pool,
        key=lambda item: (
            item.objective_value,
            float(item.worst_metrics["objective_value"]),
            float(item.metrics["cagr"]),
            float(item.metrics["max_drawdown"]),
            -float(item.metrics["total_cost"]),
        ),
    )


def _candidate_from(
    *,
    score_column: str,
    portfolio: PortfolioConfig,
    risk: RiskConfig,
    risk_variant: str,
    cadence: Cadence,
    extra_cost_bps: float,
    rebalance_offset: int = 0,
) -> SearchCandidate:
    configured = replace(portfolio, staggered_tranches=cadence.staggered_tranches)
    return SearchCandidate(
        score_column=score_column,
        portfolio_config=configured,
        risk_config=risk,
        risk_variant=risk_variant,
        cadence=cadence,
        extra_cost_bps=float(extra_cost_bps),
        rebalance_offset=rebalance_offset,
    )


def _run_records(
    *,
    stage: str,
    candidates: Sequence[Tuple[str, SearchCandidate]],
    context: MarketContext,
    signal_frame: pd.DataFrame,
    report_frame: pd.DataFrame,
    base_cost: CostConfig,
    policy: SearchPolicy,
    initial_capital: float,
    progress: Optional[ProgressCallback],
) -> Tuple[SearchEvaluation, ...]:
    _emit(progress, event="stage_started", stage=stage, total=len(candidates))
    records: List[SearchEvaluation] = []
    for index, (label, candidate) in enumerate(candidates, start=1):
        record = _evaluate_candidate(
            context,
            signal_frame,
            report_frame,
            candidate=candidate,
            base_cost=base_cost,
            policy=policy,
            initial_capital=initial_capital,
            stage=stage,
            label=label,
        )
        records.append(record)
        _emit(
            progress,
            event="candidate_completed",
            stage=stage,
            completed=index,
            total=len(candidates),
            candidate_id=candidate.candidate_id,
            objective_value=record.objective_value,
            eligible=record.eligible,
        )
    _emit(progress, event="stage_completed", stage=stage, total=len(candidates))
    return tuple(records)


def _default_risk_variants(base: RiskConfig) -> Mapping[str, RiskConfig]:
    unmanaged = RiskConfig(target_change_buffer=base.target_change_buffer)
    return {"none": unmanaged, "full": base}


def _neighbour_candidates(
    champion: SearchCandidate,
    *,
    score_columns: Sequence[str],
    top_n_grid: Sequence[int],
    cadences: Sequence[Cadence],
    weighting_grid: Sequence[str],
    risk_variants: Mapping[str, RiskConfig],
    leverage_grid: Sequence[float],
) -> Tuple[Tuple[str, SearchCandidate], ...]:
    choices: List[Tuple[str, SearchCandidate]] = []
    p = champion.portfolio_config
    for score in score_columns:
        if score != champion.score_column:
            choices.append((f"score_column={score}", replace(champion, score_column=score)))
    for top_n in top_n_grid:
        if top_n != p.top_n:
            choices.append((f"top_n={top_n}", replace(champion, portfolio_config=replace(p, top_n=int(top_n)))))
    for cadence in cadences:
        if cadence != champion.cadence:
            choices.append((
                f"cadence={cadence.name}",
                replace(
                    champion,
                    cadence=cadence,
                    rebalance_offset=0,
                    portfolio_config=replace(p, staggered_tranches=cadence.staggered_tranches),
                ),
            ))
    for weighting in weighting_grid:
        if weighting != p.weighting:
            choices.append((
                f"weighting={weighting}",
                replace(champion, portfolio_config=replace(p, weighting=str(weighting))),
            ))
    for name, risk in risk_variants.items():
        if name != champion.risk_variant or risk != champion.risk_config:
            choices.append((
                f"risk_variant={name}",
                replace(champion, risk_variant=str(name), risk_config=risk),
            ))
    for leverage in leverage_grid:
        if not math.isclose(float(leverage), p.gross_target):
            choices.append((
                f"gross_target={float(leverage):g}",
                replace(champion, portfolio_config=replace(p, gross_target=float(leverage))),
            ))

    unique: Dict[str, Tuple[str, SearchCandidate]] = {}
    for label, candidate in choices:
        if candidate.candidate_id != champion.candidate_id:
            unique.setdefault(candidate.candidate_id, (label, candidate))
    return tuple(unique.values())


def run_staged_search(
    context: MarketContext,
    selection_predictions: pd.DataFrame,
    lockbox_predictions: Optional[pd.DataFrame],
    *,
    score_columns: Sequence[str],
    base_portfolio: PortfolioConfig,
    base_cost: CostConfig,
    base_risk: RiskConfig,
    top_n_grid: Sequence[int] = (15, 20, 30),
    cadence_grid: Sequence[Cadence | Tuple[int, int] | int] = (
        Cadence("weekly", 5, 1),
        Cadence("daily_5_tranches", 1, 5),
        Cadence("biweekly", 10, 1),
        Cadence("monthly", 21, 1),
    ),
    weighting_grid: Sequence[str] = ("equal", "inverse_vol"),
    risk_variants: Optional[Mapping[str, RiskConfig]] = None,
    leverage_grid: Sequence[float] = (0.75, 1.0, 1.25, 1.5),
    cost_sensitivity_bps: Sequence[float] = (0.0, 5.0, 10.0, 20.0),
    selection_cost_bps: float = 10.0,
    base_rebalance_days: int = 5,
    policy: Optional[SearchPolicy] = None,
    initial_capital: float = 100_000.0,
    progress: Optional[ProgressCallback] = None,
) -> SearchResult:
    """Run the staged search and, only after locking, evaluate the lockbox.

    ``selection_cost_bps`` is the pre-declared Stage A/B/C selection assumption.
    Sensitivities are additional fixed friction on top of the supplied spread,
    impact, funding, and base commission settings.
    """

    policy = policy or SearchPolicy()
    scores = tuple(dict.fromkeys(map(str, score_columns)))
    if not scores:
        raise ValueError("score_columns cannot be empty")
    if initial_capital <= 0 or not math.isfinite(initial_capital):
        raise ValueError("initial_capital must be finite and positive")
    if base_rebalance_days <= 0:
        raise ValueError("base_rebalance_days must be positive")
    top_ns = tuple(dict.fromkeys(int(value) for value in top_n_grid))
    if not top_ns or any(value <= 0 for value in top_ns):
        raise ValueError("top_n_grid must contain positive values")
    weights = tuple(dict.fromkeys(map(str, weighting_grid)))
    if not weights or any(value not in {"equal", "inverse_vol"} for value in weights):
        raise ValueError("weighting_grid contains an unsupported method")
    leverages = tuple(dict.fromkeys(float(value) for value in leverage_grid))
    if not leverages or any(not math.isfinite(value) or value <= 0 for value in leverages):
        raise ValueError("leverage_grid must contain finite positive values")
    sensitivity = tuple(dict.fromkeys(float(value) for value in cost_sensitivity_bps))
    if any(not math.isfinite(value) or value < 0 for value in sensitivity):
        raise ValueError("cost sensitivity values must be finite and non-negative")
    if not math.isfinite(selection_cost_bps) or selection_cost_bps < 0:
        raise ValueError("selection_cost_bps must be finite and non-negative")
    # These four points are a required regression curve, not optional tuning
    # suggestions.  Callers may add harsher assumptions but cannot silently
    # remove the standard curve.
    for required_bps in (0.0, 5.0, 10.0, 20.0, float(selection_cost_bps)):
        if not any(math.isclose(value, required_bps) for value in sensitivity):
            sensitivity = sensitivity + (required_bps,)
    cadences = _normalise_cadences(cadence_grid)
    if not any(item.rebalance_days == 1 and item.staggered_tranches == 5 for item in cadences):
        cadences = cadences + (Cadence("daily_5_tranches", 1, 5),)
    risks = dict(_default_risk_variants(base_risk) if risk_variants is None else risk_variants)
    if not risks or any(not str(name) for name in risks):
        raise ValueError("risk_variants cannot be empty and must have names")
    if any(not isinstance(value, RiskConfig) for value in risks.values()):
        raise TypeError("each risk variant must be a backtest.RiskConfig")

    # This is the only frame inspected before configuration lock.
    selection = _validate_selection_predictions(selection_predictions, scores)
    selection_start = pd.Timestamp(selection["timestamp"].min())
    selection_end = pd.Timestamp(selection["timestamp"].max())
    _emit(
        progress,
        event="search_started",
        selection_start=selection_start.isoformat(),
        selection_end=selection_end.isoformat(),
        folds=[item[0] for item in _fold_bounds(selection)],
    )

    base_cadence = Cadence(
        f"base_every_{base_rebalance_days}d",
        base_rebalance_days,
        base_portfolio.staggered_tranches,
    )
    stage_a_candidates = [
        (
            f"score={score}",
            _candidate_from(
                score_column=score,
                portfolio=base_portfolio,
                risk=base_risk,
                risk_variant="base",
                cadence=base_cadence,
                extra_cost_bps=selection_cost_bps,
            ),
        )
        for score in scores
    ]
    stage_a = _run_records(
        stage="A_score",
        candidates=stage_a_candidates,
        context=context,
        signal_frame=selection,
        report_frame=selection,
        base_cost=base_cost,
        policy=policy,
        initial_capital=initial_capital,
        progress=progress,
    )
    stage_a_winner = _choose(stage_a, policy)

    stage_b_candidates: List[Tuple[str, SearchCandidate]] = []
    for top_n in top_ns:
        for cadence in cadences:
            for weighting in weights:
                for risk_name, risk in risks.items():
                    portfolio = replace(
                        base_portfolio,
                        top_n=top_n,
                        weighting=weighting,
                        staggered_tranches=cadence.staggered_tranches,
                    )
                    label = (
                        f"top_n={top_n};cadence={cadence.name};"
                        f"weighting={weighting};risk={risk_name}"
                    )
                    stage_b_candidates.append((
                        label,
                        _candidate_from(
                            score_column=stage_a_winner.candidate.score_column,
                            portfolio=portfolio,
                            risk=risk,
                            risk_variant=str(risk_name),
                            cadence=cadence,
                            extra_cost_bps=selection_cost_bps,
                        ),
                    ))
    stage_b = _run_records(
        stage="B_portfolio",
        candidates=stage_b_candidates,
        context=context,
        signal_frame=selection,
        report_frame=selection,
        base_cost=base_cost,
        policy=policy,
        initial_capital=initial_capital,
        progress=progress,
    )
    stage_b_winner = _choose(stage_b, policy)

    stage_c_candidates: List[Tuple[str, SearchCandidate]] = []
    for leverage in leverages:
        for extra_bps in sensitivity:
            portfolio = replace(stage_b_winner.candidate.portfolio_config, gross_target=leverage)
            candidate = replace(
                stage_b_winner.candidate,
                portfolio_config=portfolio,
                extra_cost_bps=extra_bps,
            )
            stage_c_candidates.append((
                f"gross_target={leverage:g};extra_cost_bps={extra_bps:g}",
                candidate,
            ))
    stage_c = _run_records(
        stage="C_leverage_cost",
        candidates=stage_c_candidates,
        context=context,
        signal_frame=selection,
        report_frame=selection,
        base_cost=base_cost,
        policy=policy,
        initial_capital=initial_capital,
        progress=progress,
    )
    primary_cost_records = [
        item for item in stage_c
        if math.isclose(item.candidate.extra_cost_bps, selection_cost_bps)
    ]
    champion = _choose(primary_cost_records, policy)

    neighbours = _neighbour_candidates(
        champion.candidate,
        score_columns=scores,
        top_n_grid=top_ns,
        cadences=cadences,
        weighting_grid=weights,
        risk_variants=risks,
        leverage_grid=leverages,
    )
    neighborhood = _run_records(
        stage="N_neighborhood",
        candidates=neighbours,
        context=context,
        signal_frame=selection,
        report_frame=selection,
        base_cost=base_cost,
        policy=policy,
        initial_capital=initial_capital,
        progress=progress,
    ) if neighbours else ()

    offset_candidates = [
        (
            f"rebalance_offset={offset}",
            replace(champion.candidate, rebalance_offset=offset),
        )
        for offset in range(champion.candidate.cadence.rebalance_days)
    ]
    offset_sensitivity = _run_records(
        stage="O_rebalance_offset",
        candidates=offset_candidates,
        context=context,
        signal_frame=selection,
        report_frame=selection,
        base_cost=base_cost,
        policy=policy,
        initial_capital=initial_capital,
        progress=progress,
    )

    # Critical boundary: no lockbox columns, dates, or values were touched above.
    _emit(
        progress,
        event="champion_locked",
        candidate_id=champion.candidate.candidate_id,
        selection_objective=champion.objective_value,
    )

    lockbox_evaluation: Optional[SearchEvaluation] = None
    if lockbox_predictions is not None:
        lockbox = _validate_lockbox_predictions(
            lockbox_predictions,
            score_column=champion.candidate.score_column,
            selection_end=selection_end,
        )
        # Preserve the actual selection-end holdings and all engine state into
        # lockbox by running one combined chronology, then report lockbox rows.
        signal_columns = ["timestamp", "symbol", "fold_id", champion.candidate.score_column]
        combined = pd.concat(
            [selection[signal_columns], lockbox[signal_columns]],
            ignore_index=True,
        ).sort_values(["timestamp", "symbol"]).reset_index(drop=True)
        lockbox_evaluation = _evaluate_candidate(
            context,
            combined,
            lockbox,
            candidate=champion.candidate,
            base_cost=base_cost,
            policy=policy,
            initial_capital=initial_capital,
            stage="LOCKBOX",
            label="frozen_champion",
        )
        _emit(
            progress,
            event="lockbox_completed",
            candidate_id=champion.candidate.candidate_id,
            objective_value=lockbox_evaluation.objective_value,
        )

    audit = {
        "selection_only_champion": True,
        "lockbox_accessed_after_champion_lock": lockbox_predictions is not None,
        "fold_state_reset": False,
        "backtest_runs_per_candidate": 1,
        "selection_start": selection_start.isoformat(),
        "selection_end": selection_end.isoformat(),
        "selection_folds": [item[0] for item in _fold_bounds(selection)],
        "selection_objective": policy.objective,
        "selection_extra_cost_bps": selection_cost_bps,
        "cost_sensitivity_bps": list(sensitivity),
        "cost_interpretation": "additional fixed friction on top of base commission, spread, impact, and funding",
        "champion_passed_gates": champion.eligible,
        "champion_candidate_id": champion.candidate.candidate_id,
        "stage_a_winner_id": stage_a_winner.candidate.candidate_id,
        "stage_b_winner_id": stage_b_winner.candidate.candidate_id,
        "daily_five_tranche_tested": any(
            candidate.cadence.rebalance_days == 1
            and candidate.cadence.staggered_tranches == 5
            for _, candidate in stage_b_candidates
        ),
    }
    result = SearchResult(
        stage_a=stage_a,
        stage_b=stage_b,
        stage_c=stage_c,
        champion=champion,
        lockbox=lockbox_evaluation,
        neighborhood=tuple(neighborhood),
        offset_sensitivity=offset_sensitivity,
        audit=audit,
    )
    _emit(progress, event="search_completed", candidate_id=champion.candidate.candidate_id)
    return result


def _flat_evaluation(record: SearchEvaluation) -> Dict[str, object]:
    candidate = record.candidate
    row: Dict[str, object] = {
        "stage": record.stage,
        "label": record.label,
        "candidate_id": candidate.candidate_id,
        "score_column": candidate.score_column,
        "top_n": candidate.portfolio_config.top_n,
        "weighting": candidate.portfolio_config.weighting,
        "gross_target": candidate.portfolio_config.gross_target,
        "risk_variant": candidate.risk_variant,
        "cadence": candidate.cadence.name,
        "rebalance_days": candidate.cadence.rebalance_days,
        "staggered_tranches": candidate.cadence.staggered_tranches,
        "rebalance_offset": candidate.rebalance_offset,
        "extra_cost_bps": candidate.extra_cost_bps,
        "objective_value": record.objective_value,
        "eligible": record.eligible,
        "gate_failures": " | ".join(record.gate_failures),
    }
    row.update({f"metric_{key}": value for key, value in record.metrics.items()})
    row.update({f"cost_{key}": value for key, value in record.cost_breakdown.items()})
    return row


def _write_csv(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    materialised = list(rows)
    if not materialised:
        path.write_text("", encoding="utf-8")
        return
    columns = sorted({str(key) for row in materialised for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(materialised)


def write_search_artifacts(
    result: SearchResult,
    output_dir: str | Path,
    *,
    research_root: str | Path = RESEARCH_ROOT,
) -> Mapping[str, Path]:
    """Persist JSON/CSV audit artifacts under the confined research root."""

    directory = ensure_research_output_path(output_dir, research_root=research_root)
    directory.mkdir(parents=True, exist_ok=True)
    paths = {
        "summary": directory / "search_summary.json",
        "champion": directory / "champion.json",
        "stage_a": directory / "stage_a.csv",
        "stage_b": directory / "stage_b.csv",
        "stage_c": directory / "stage_c.csv",
        "neighborhood": directory / "configuration_neighborhood.csv",
        "offset_sensitivity": directory / "rebalance_offset_sensitivity.csv",
        "fold_metrics": directory / "fold_metrics.csv",
    }
    paths["summary"].write_text(
        json.dumps(result.to_dict(), indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    paths["champion"].write_text(
        json.dumps(_json_safe(result.champion.to_dict()), indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    _write_csv(paths["stage_a"], (_flat_evaluation(item) for item in result.stage_a))
    _write_csv(paths["stage_b"], (_flat_evaluation(item) for item in result.stage_b))
    _write_csv(paths["stage_c"], (_flat_evaluation(item) for item in result.stage_c))
    _write_csv(paths["neighborhood"], (_flat_evaluation(item) for item in result.neighborhood))
    _write_csv(
        paths["offset_sensitivity"],
        (_flat_evaluation(item) for item in result.offset_sensitivity),
    )
    all_records = (
        list(result.stage_a) + list(result.stage_b) + list(result.stage_c)
        + list(result.neighborhood) + list(result.offset_sensitivity)
        + ([result.lockbox] if result.lockbox is not None else [])
    )
    fold_rows = []
    for record in all_records:
        for fold_id, metrics in record.fold_metrics.items():
            row = {
                "stage": record.stage,
                "candidate_id": record.candidate.candidate_id,
                "fold_id": fold_id,
            }
            row.update(metrics)
            fold_rows.append(row)
    _write_csv(paths["fold_metrics"], fold_rows)
    return paths


__all__ = [
    "Cadence",
    "ProgressCallback",
    "SearchCandidate",
    "SearchEvaluation",
    "SearchPolicy",
    "SearchResult",
    "run_staged_search",
    "write_search_artifacts",
]
