"""Configuration primitives for the isolated Research v2 pipeline.

Importing this module is side-effect free.  In particular it never reads the
production data store, environment credentials, live configuration, or logs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Dict, List


@dataclass(frozen=True)
class DataConfig:
    start_date: str = "2020-07-27"
    end_date: str = "2026-07-09"
    min_cross_section: int = 350
    min_symbol_history: int = 252
    label_horizon: int = 5
    invalid_row_policy: str = "drop"
    canonical_timezone: str = "America/New_York"


@dataclass(frozen=True)
class WalkForwardConfig:
    train_days: int = 504
    validation_days: int = 63
    test_days: int = 63
    purge_days: int = 5
    embargo_days: int = 5
    step_days: int = 63
    rolling_train: bool = True
    selection_end: str = "2025-12-31"
    lockbox_start: str = "2026-01-01"


@dataclass(frozen=True)
class CostConfig:
    commission_bps: float = 0.0
    fixed_slippage_bps: float = 5.0
    range_to_half_spread: float = 0.05
    max_half_spread_bps: float = 30.0
    impact_coefficient: float = 0.10
    max_impact_bps: float = 50.0
    max_adv_participation: float = 0.02
    annual_funding_rate: float = 0.055
    minimum_trade_notional: float = 25.0


@dataclass(frozen=True)
class PortfolioConfig:
    top_n: int = 20
    leverage: float = 1.0
    weighting: str = "inverse_vol"
    max_single_weight: float = 0.10
    max_sector_weight: float = 0.30
    rank_buffer: int = 5
    minimum_weight_change: float = 0.0025
    rebalance_days: int = 5
    staggered_tranches: int = 1


@dataclass(frozen=True)
class RiskConfig:
    target_volatility: float = 0.18
    volatility_lookback: int = 63
    covariance_shrinkage: float = 0.50
    beta_cap: float = 1.25
    beta_lookback: int = 126
    breadth_reduce_below: float = 0.40
    breadth_exit_below: float = 0.25
    breadth_reenter_above: float = 0.45
    risk_off_multiplier: float = 0.50
    drawdown_level_1: float = -0.10
    drawdown_level_2: float = -0.20
    drawdown_multiplier_1: float = 0.67
    drawdown_multiplier_2: float = 0.33
    drawdown_recovery_buffer: float = 0.03
    crowding_correlation_limit: float = 0.70
    crowding_multiplier: float = 0.70


@dataclass(frozen=True)
class ModelConfig:
    random_seed: int = 20260710
    ridge_alphas: List[float] = field(default_factory=lambda: [1.0, 10.0, 100.0])
    gbdt_grid: List[Dict[str, Any]] = field(default_factory=lambda: [
        {
            "learning_rate": 0.05,
            "max_iter": 150,
            "max_leaf_nodes": 15,
            "min_samples_leaf": 500,
            "l2_regularization": 5.0,
        },
        {
            "learning_rate": 0.03,
            "max_iter": 250,
            "max_leaf_nodes": 15,
            "min_samples_leaf": 300,
            "l2_regularization": 10.0,
        },
        {
            "learning_rate": 0.04,
            "max_iter": 200,
            "max_leaf_nodes": 31,
            "min_samples_leaf": 750,
            "l2_regularization": 10.0,
        },
    ])
    sequence_length: int = 60
    sequence_hidden_size: int = 32
    sequence_epochs: int = 4
    sequence_batch_size: int = 256
    sequence_max_train_samples: int = 120_000
    ensemble_shrinkage: float = 0.50
    ensemble_single_model_cap: float = 0.70


@dataclass(frozen=True)
class SearchConfig:
    initial_capital: float = 100_000.0
    cost_sensitivity_bps: List[float] = field(default_factory=lambda: [0.0, 5.0, 10.0, 20.0])
    top_n_grid: List[int] = field(default_factory=lambda: [15, 20, 30])
    leverage_grid: List[float] = field(default_factory=lambda: [0.75, 1.0, 1.25, 1.5])
    rebalance_grid: List[int] = field(default_factory=lambda: [5, 10, 21])
    weighting_grid: List[str] = field(default_factory=lambda: ["equal", "inverse_vol"])
    require_positive_worst_fold: bool = False
    max_drawdown_limit: float = -0.35


@dataclass(frozen=True)
class ResearchConfig:
    name: str = "ancserAPX_research_v2"
    data: DataConfig = field(default_factory=DataConfig)
    walk_forward: WalkForwardConfig = field(default_factory=WalkForwardConfig)
    costs: CostConfig = field(default_factory=CostConfig)
    portfolio: PortfolioConfig = field(default_factory=PortfolioConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    models: ModelConfig = field(default_factory=ModelConfig)
    search: SearchConfig = field(default_factory=SearchConfig)

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def fingerprint(self) -> str:
        payload = json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"))
        return sha256(payload.encode("utf-8")).hexdigest()


def _construct(data: Dict[str, Any]) -> ResearchConfig:
    """Construct a config while rejecting unknown top-level sections."""
    allowed = {"name", "data", "walk_forward", "costs", "portfolio", "risk", "models", "search"}
    unknown = set(data) - allowed
    if unknown:
        raise ValueError(f"Unknown research config sections: {sorted(unknown)}")
    return ResearchConfig(
        name=data.get("name", ResearchConfig.name),
        data=DataConfig(**data.get("data", {})),
        walk_forward=WalkForwardConfig(**data.get("walk_forward", {})),
        costs=CostConfig(**data.get("costs", {})),
        portfolio=PortfolioConfig(**data.get("portfolio", {})),
        risk=RiskConfig(**data.get("risk", {})),
        models=ModelConfig(**data.get("models", {})),
        search=SearchConfig(**data.get("search", {})),
    )


def load_config(path: Path | str | None = None) -> ResearchConfig:
    if path is None:
        return ResearchConfig()
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Research config must be a JSON object")
    return _construct(payload)


def save_config(config: ResearchConfig, path: Path | str) -> None:
    """Write only to a caller-approved research path.

    Path confinement is enforced by :mod:`research_v2.safety` in the CLI.  This
    helper deliberately has no production path knowledge so it remains pure and
    easy to test.
    """
    Path(path).write_text(
        json.dumps(config.as_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
