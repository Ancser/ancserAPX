from __future__ import annotations

import math
import unittest

from research_v2.backtest import (
    MarketBar,
    MissingHeldReturnError,
    RiskConfig,
    RiskObservation,
    RiskState,
    apply_risk_overlays,
    assert_accounting_identity,
    run_backtest,
)
from research_v2.costs import (
    AdvParticipationError,
    CostConfig,
    LiquiditySnapshot,
    estimate_daily_funding,
    estimate_execution_costs,
)
from research_v2.metrics import compute_performance_metrics
from research_v2.portfolio import (
    PortfolioConfig,
    StaggerState,
    construct_portfolio,
    select_with_rank_buffer,
    update_staggered_tranches,
)


def bar(
    open_price: float,
    close_price: float,
    *,
    adv: float = 1_000_000_000.0,
    vol: float = 0.02,
    spread_bps: float = 0.0,
    beta: float = 1.0,
) -> MarketBar:
    return MarketBar(
        open=open_price,
        close=close_price,
        adv_dollars=adv,
        daily_volatility=vol,
        spread_proxy_bps=spread_bps,
        beta=beta,
    )


class CostModelTests(unittest.TestCase):
    def test_cost_components_turnover_participation_and_funding(self) -> None:
        liquidity = {
            "A": LiquiditySnapshot(
                adv_dollars=100_000.0,
                daily_volatility=0.02,
                spread_proxy_bps=2.0,
            )
        }
        config = CostConfig(
            commission_bps=1.0,
            impact_coefficient=0.10,
            max_adv_participation=0.20,
            annual_funding_rate=0.10,
        )
        costs = estimate_execution_costs({}, {"A": 0.10}, 100_000.0, liquidity, config)

        expected_impact_bps = 0.10 * 0.02 * math.sqrt(0.10) * 10_000.0
        self.assertAlmostEqual(costs.gross_turnover, 0.10)
        self.assertAlmostEqual(costs.one_way_turnover, 0.05)
        self.assertAlmostEqual(costs.traded_notional, 10_000.0)
        self.assertAlmostEqual(costs.commission, 1.0)
        self.assertAlmostEqual(costs.spread, 2.0)
        self.assertAlmostEqual(costs.impact, 10_000.0 * expected_impact_bps / 10_000.0)
        self.assertAlmostEqual(costs.max_adv_participation, 0.10)

        funding = estimate_daily_funding({"A": 1.50}, 100_000.0, config)
        self.assertAlmostEqual(funding.borrowed_cash, 50_000.0)
        self.assertAlmostEqual(funding.margin_interest, 50_000.0 * 0.10 / 252.0)

    def test_adv_limit_is_rechecked_by_cost_model(self) -> None:
        liquidity = {"A": LiquiditySnapshot(100_000.0, 0.02, 1.0)}
        with self.assertRaises(AdvParticipationError):
            estimate_execution_costs(
                {},
                {"A": 0.10},
                100_000.0,
                liquidity,
                CostConfig(max_adv_participation=0.05),
            )


class PortfolioConstructionTests(unittest.TestCase):
    def test_rank_buffer_never_crowds_out_a_new_top_ranked_name(self) -> None:
        selected = select_with_rank_buffer(
            scores={"NEW": 4.0, "A": 3.0, "B": 2.0, "C": 1.0},
            current_weights={"A": 0.5, "B": 0.5},
            top_n=2,
            rank_buffer=1,
        )
        self.assertIn("NEW", selected)
        self.assertEqual(selected, ("NEW", "A", "B"))

    def test_inverse_vol_respects_single_and_sector_caps(self) -> None:
        decision = construct_portfolio(
            scores={"A": 4.0, "B": 3.0, "C": 2.0, "D": 1.0},
            current_weights={},
            volatility={"A": 0.01, "B": 0.01, "C": 0.10, "D": 0.10},
            sectors={"A": "Tech", "B": "Tech", "C": "Health", "D": "Health"},
            adv_dollars={"A": 1e9, "B": 1e9, "C": 1e9, "D": 1e9},
            equity=100_000.0,
            config=PortfolioConfig(
                top_n=4,
                weighting="inverse_vol",
                gross_target=1.0,
                single_name_cap=0.35,
                sector_cap=0.55,
                max_adv_participation=None,
            ),
        )
        weights = decision.target_weights
        self.assertAlmostEqual(sum(weights.values()), 1.0)
        self.assertLessEqual(max(weights.values()), 0.35 + 1e-12)
        self.assertLessEqual(weights["A"] + weights["B"], 0.55 + 1e-12)
        self.assertTrue(any(reason.layer == "position_sector_caps" and reason.triggered for reason in decision.reasons))

    def test_no_trade_band_keeps_small_changes(self) -> None:
        decision = construct_portfolio(
            scores={"A": 2.0, "B": 1.0},
            current_weights={"A": 0.51, "B": 0.49},
            volatility={},
            sectors={"A": "One", "B": "Two"},
            adv_dollars={"A": 1e9, "B": 1e9},
            equity=100_000.0,
            config=PortfolioConfig(
                top_n=2,
                gross_target=1.0,
                single_name_cap=0.60,
                sector_cap=1.0,
                no_trade_band=0.02,
                max_adv_participation=None,
            ),
        )
        self.assertAlmostEqual(decision.target_weights["A"], 0.51)
        self.assertAlmostEqual(decision.target_weights["B"], 0.49)
        self.assertTrue(any(reason.layer == "no_trade_band" and reason.triggered for reason in decision.reasons))

    def test_staggering_replaces_one_explicit_tranche(self) -> None:
        first, state, index = update_staggered_tranches(
            {"A": 1.0}, {}, None, tranche_count=2
        )
        self.assertEqual(index, 0)
        self.assertAlmostEqual(first["A"], 0.50)
        second, state, index = update_staggered_tranches(
            {"B": 1.0}, first, state, tranche_count=2
        )
        self.assertEqual(index, 1)
        self.assertAlmostEqual(second["A"], 0.50)
        self.assertAlmostEqual(second["B"], 0.50)
        self.assertIsInstance(state, StaggerState)


class RiskOverlayTests(unittest.TestCase):
    def test_crowding_falls_back_to_target_holdings_pairwise_correlation(self) -> None:
        history = tuple(
            {"A": value, "B": value * 2.0}
            for value in (0.01, -0.02, 0.03, -0.01, 0.02, -0.03)
        )
        decision = apply_risk_overlays(
            {"A": 0.5, "B": 0.5},
            history,
            equity=100.0,
            state=RiskState(True, 100.0),
            observation=RiskObservation(betas={"A": 1.0, "B": 1.0}),
            config=RiskConfig(
                vol_lookback=6,
                min_vol_observations=5,
                crowding_threshold=0.80,
                crowding_multiplier=0.50,
            ),
        )
        layer = next(item for item in decision.layers if item.layer == "crowding")
        self.assertTrue(layer.triggered)
        self.assertAlmostEqual(layer.observed, 1.0)
        self.assertAlmostEqual(sum(decision.weights.values()), 0.50)

    def test_every_risk_layer_is_auditable(self) -> None:
        history = tuple(
            {"A": 0.04 if index % 2 == 0 else -0.04}
            for index in range(12)
        )
        config = RiskConfig(
            target_volatility=0.20,
            vol_lookback=12,
            min_vol_observations=10,
            drawdown_steps=((0.10, 0.75), (0.20, 0.50)),
            max_abs_beta=0.10,
            trend_filter=True,
            breadth_exit=0.40,
            breadth_enter=0.60,
            risk_off_multiplier=0.20,
            crowding_threshold=0.80,
            crowding_multiplier=0.50,
        )
        decision = apply_risk_overlays(
            {"A": 1.0},
            history,
            equity=80.0,
            state=RiskState(in_market=True, peak_equity=100.0),
            observation=RiskObservation(
                benchmark_close=90.0,
                benchmark_slow=100.0,
                benchmark_fast=95.0,
                breadth=0.20,
                crowding_score=0.90,
                betas={"A": 2.0},
            ),
            config=config,
        )
        by_name = {layer.layer: layer for layer in decision.layers}
        self.assertTrue(by_name["target_weight_volatility"].triggered)
        self.assertTrue(by_name["drawdown_governor"].triggered)
        self.assertTrue(by_name["beta_proxy"].triggered)
        self.assertTrue(by_name["breadth_regime_hysteresis"].triggered)
        self.assertTrue(by_name["crowding"].triggered)
        self.assertFalse(decision.state.in_market)
        self.assertLess(sum(decision.weights.values()), 0.01)
        self.assertTrue(all(layer.reason for layer in decision.layers))

    def test_regime_reentry_requires_fast_trend_and_breadth(self) -> None:
        config = RiskConfig(
            trend_filter=True,
            breadth_exit=0.40,
            breadth_enter=0.60,
            risk_off_multiplier=0.25,
        )
        exit_decision = apply_risk_overlays(
            {"A": 1.0},
            (),
            100.0,
            RiskState(True, 100.0),
            RiskObservation(90.0, 100.0, 95.0, 0.30, betas={"A": 1.0}),
            config,
        )
        self.assertFalse(exit_decision.state.in_market)
        self.assertAlmostEqual(sum(exit_decision.weights.values()), 0.25)

        wait_decision = apply_risk_overlays(
            {"A": 1.0},
            (),
            100.0,
            exit_decision.state,
            RiskObservation(98.0, 100.0, 100.0, 0.55, betas={"A": 1.0}),
            config,
        )
        self.assertFalse(wait_decision.state.in_market)

        enter_decision = apply_risk_overlays(
            {"A": 1.0},
            (),
            100.0,
            wait_decision.state,
            RiskObservation(105.0, 100.0, 100.0, 0.70, betas={"A": 1.0}),
            config,
        )
        self.assertTrue(enter_decision.state.in_market)
        self.assertAlmostEqual(sum(enter_decision.weights.values()), 1.0)

    def test_rich_benchmark_observation_does_not_silently_enable_trend(self) -> None:
        decision = apply_risk_overlays(
            {"A": 1.0},
            (),
            100.0,
            RiskState(True, 100.0),
            RiskObservation(
                benchmark_close=90.0,
                benchmark_slow=100.0,
                benchmark_fast=95.0,
                breadth=0.20,
                betas={"A": 1.0},
            ),
            RiskConfig(),
        )
        self.assertTrue(decision.state.in_market)
        self.assertAlmostEqual(sum(decision.weights.values()), 1.0)
        self.assertFalse(
            any(layer.layer == "breadth_regime_hysteresis" for layer in decision.layers)
        )


class EventEngineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.portfolio_config = PortfolioConfig(
            top_n=1,
            gross_target=1.0,
            single_name_cap=1.0,
            sector_cap=1.0,
            max_adv_participation=1.0,
        )
        self.zero_cost = CostConfig(
            impact_coefficient=0.0,
            max_adv_participation=1.0,
        )

    def test_close_signal_executes_next_open_not_same_close(self) -> None:
        market = {
            "2026-01-01": {"A": bar(100.0, 100.0)},
            "2026-01-02": {"A": bar(110.0, 121.0)},
            "2026-01-03": {"A": bar(121.0, 121.0)},
        }
        result = run_backtest(
            market,
            signals={"2026-01-01": {"A": 1.0}},
            sectors={"A": "One"},
            portfolio_config=self.portfolio_config,
            cost_config=self.zero_cost,
            initial_capital=1_000.0,
        )
        first, execution, final = result.ledger
        self.assertEqual(first.ending_equity, 1_000.0)
        self.assertIsNone(first.executed_signal_session)
        self.assertEqual(execution.executed_signal_session, "2026-01-01")
        self.assertAlmostEqual(execution.overnight_pnl, 0.0)
        self.assertAlmostEqual(execution.intraday_pnl, 100.0)
        self.assertAlmostEqual(execution.ending_equity, 1_100.0)
        self.assertAlmostEqual(final.ending_equity, 1_100.0)
        assert_accounting_identity(result)

    def test_execution_costs_flow_through_cash_and_identity(self) -> None:
        market = {
            "2026-01-01": {"A": bar(100.0, 100.0, adv=1_000_000.0, spread_bps=2.0)},
            "2026-01-02": {"A": bar(100.0, 100.0, adv=1_000_000.0, spread_bps=2.0)},
        }
        result = run_backtest(
            market,
            signals={"2026-01-01": {"A": 1.0}},
            sectors={"A": "One"},
            portfolio_config=PortfolioConfig(
                top_n=1,
                gross_target=1.0,
                single_name_cap=1.0,
                sector_cap=1.0,
                max_adv_participation=0.20,
            ),
            cost_config=CostConfig(
                commission_bps=1.0,
                impact_coefficient=0.0,
                max_adv_participation=0.20,
            ),
            initial_capital=100_000.0,
        )
        execution = result.ledger[1]
        self.assertAlmostEqual(execution.gross_turnover, 1.0)
        self.assertAlmostEqual(execution.one_way_turnover, 0.5)
        self.assertAlmostEqual(execution.commission, 10.0)
        self.assertAlmostEqual(execution.spread, 20.0)
        self.assertAlmostEqual(execution.ending_equity, 99_970.0)
        assert_accounting_identity(result)

    def test_missing_held_return_fails_fast(self) -> None:
        market = {
            "2026-01-01": {"A": bar(100.0, 100.0)},
            "2026-01-02": {"A": bar(100.0, 100.0)},
            "2026-01-03": {},
        }
        with self.assertRaises(MissingHeldReturnError):
            run_backtest(
                market,
                signals={"2026-01-01": {"A": 1.0}},
                sectors={"A": "One"},
                portfolio_config=self.portfolio_config,
                cost_config=self.zero_cost,
                initial_capital=1_000.0,
            )

    def test_metrics_are_net_of_cost_and_include_audit_fields(self) -> None:
        market = {
            "2026-01-01": {"A": bar(100.0, 100.0)},
            "2026-01-02": {"A": bar(100.0, 110.0)},
            "2026-01-03": {"A": bar(110.0, 110.0)},
        }
        result = run_backtest(
            market,
            signals={"2026-01-01": {"A": 1.0}},
            sectors={"A": "One"},
            portfolio_config=self.portfolio_config,
            cost_config=self.zero_cost,
            initial_capital=1_000.0,
        )
        metrics = compute_performance_metrics(result.ledger)
        self.assertAlmostEqual(metrics.total_return, 0.10)
        self.assertEqual(metrics.periods, 3)
        self.assertAlmostEqual(metrics.total_gross_turnover, 1.0)
        self.assertAlmostEqual(metrics.total_cost, 0.0)
        self.assertLess(metrics.max_value_identity_error, 1e-10)
        self.assertLess(metrics.max_pnl_identity_error, 1e-10)


if __name__ == "__main__":
    unittest.main()
