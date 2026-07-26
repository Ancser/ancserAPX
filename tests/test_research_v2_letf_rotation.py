import dataclasses
import unittest

import numpy as np
import pandas as pd

from research_v2.letf_rotation import (
    HeldDataUnavailableError,
    LETFMember,
    RotationConfig,
    compute_proxy_first_scores,
    evaluate_rotation,
    is_rebalance_session,
    prepare_close_panel,
    select_rotation_candidates,
)


def _calendar(count=260):
    return pd.bdate_range("2025-01-02", periods=count)


def _long_bars(sessions, closes):
    rows = []
    for symbol, values in closes.items():
        for timestamp, close in zip(sessions, values):
            rows.append(
                {
                    "timestamp": timestamp,
                    "symbol": symbol,
                    "open": float(close),
                    "high": float(close) * 1.001,
                    "low": float(close) * 0.999,
                    "close": float(close),
                    "volume": 1_000_000.0,
                }
            )
    return pd.DataFrame(rows)


def _proxy_prices(sessions, symbols):
    position = np.arange(len(sessions), dtype=float)
    result = {}
    for index, symbol in enumerate(symbols):
        drift = 0.0003 + index * 0.00012
        curvature = index * 0.0000002
        log_price = (
            np.log(100.0)
            + drift * position
            + curvature * position**2
            + 0.003 * np.sin(position / 7.0 + index)
        )
        result[symbol] = np.exp(log_price)
    return result


def _independent_product_prices(sessions, symbols, seed=1234):
    rng = np.random.default_rng(seed)
    result = {}
    for index, symbol in enumerate(symbols):
        daily = rng.normal(0.0004 + index * 0.00002, 0.012, len(sessions))
        result[symbol] = 100.0 * np.cumprod(1.0 + daily)
    return result


class LETFRotationTests(unittest.TestCase):
    def test_prepared_close_panels_are_exactly_signal_and_selection_equivalent(self):
        sessions = _calendar()
        decision_session = sessions[225]
        symbols = ["A", "B", "C", "D", "E", "F"]
        proxies = [f"P_{symbol}" for symbol in symbols]
        universe = [
            LETFMember(symbol, proxy, f"theme_{symbol}", f"macro_{index % 3}")
            for index, (symbol, proxy) in enumerate(zip(symbols, proxies))
        ]
        product_bars = _long_bars(
            sessions, _independent_product_prices(sessions, symbols)
        )
        proxy_bars = _long_bars(sessions, _proxy_prices(sessions, proxies))
        kwargs = {
            "session": decision_session,
            "sessions": sessions,
            "universe": universe,
            "eligibility": {symbol: True for symbol in symbols},
        }

        long_form = evaluate_rotation(
            product_bars=product_bars,
            proxy_bars=proxy_bars,
            **kwargs,
        )
        prepared = evaluate_rotation(
            product_bars=prepare_close_panel(product_bars),
            proxy_bars=prepare_close_panel(proxy_bars),
            **kwargs,
        )

        self.assertEqual(long_form, prepared)

    def test_future_price_mutation_does_not_change_past_signal_or_selection(self):
        sessions = _calendar()
        decision_position = 225
        decision_session = sessions[decision_position]
        symbols = ["A", "B", "C", "D", "E", "F", "G"]
        proxies = [f"P_{symbol}" for symbol in symbols]
        universe = [
            LETFMember(symbol, proxy, f"theme_{symbol}", f"macro_{index % 4}")
            for index, (symbol, proxy) in enumerate(zip(symbols, proxies))
        ]
        proxy_bars = _long_bars(sessions, _proxy_prices(sessions, proxies))
        product_bars = _long_bars(
            sessions, _independent_product_prices(sessions, symbols)
        )
        eligibility = {symbol: True for symbol in symbols}

        before = evaluate_rotation(
            session=decision_session,
            sessions=sessions,
            product_bars=product_bars,
            proxy_bars=proxy_bars,
            universe=universe,
            eligibility=eligibility,
        )
        self.assertTrue(before.scores)

        future = sessions > decision_session
        changed_proxy = proxy_bars.copy()
        changed_product = product_bars.copy()
        changed_proxy.loc[changed_proxy["timestamp"].isin(sessions[future]), "close"] = -999.0
        changed_product.loc[
            changed_product["timestamp"].isin(sessions[future]), "close"
        ] = 1e12
        after = evaluate_rotation(
            session=decision_session,
            sessions=sessions,
            product_bars=changed_product,
            proxy_bars=changed_proxy,
            universe=universe,
            eligibility=eligibility,
        )

        self.assertEqual(before.scores, after.scores)
        self.assertEqual(before.components, after.components)
        self.assertEqual(before.selected, after.selected)
        self.assertEqual(before.audits, after.audits)

    def test_selector_enforces_theme_macro_and_absolute_correlation_caps(self):
        sessions = _calendar(140)
        session = sessions[-1]
        symbols = ["A", "C", "D", "E", "B", "F", "G", "H"]
        universe = [
            LETFMember("A", "PA", "theme_a", "macro_1"),
            LETFMember("C", "PC", "theme_a", "macro_2"),
            LETFMember("D", "PD", "theme_d", "macro_1"),
            LETFMember("E", "PE", "theme_e", "macro_1"),
            LETFMember("B", "PB", "theme_b", "macro_2"),
            LETFMember("F", "PF", "theme_f", "macro_2"),
            LETFMember("G", "PG", "theme_g", "macro_3"),
            LETFMember("H", "PH", "theme_h", "macro_4"),
        ]
        prices = _independent_product_prices(sessions, symbols, seed=77)
        prices["B"] = prices["A"].copy()
        product_bars = _long_bars(sessions, prices)
        scores = {
            "A": 0.99,
            "C": 0.98,
            "D": 0.97,
            "E": 0.96,
            "B": 0.95,
            "F": 0.94,
            "G": 0.93,
            "H": 0.92,
        }

        result = select_rotation_candidates(
            session=session,
            sessions=sessions,
            product_bars=product_bars,
            universe=universe,
            scores=scores,
        )
        reasons = {audit.symbol: audit.reason for audit in result.audits}

        self.assertEqual(result.selected, ("A", "D", "F", "G", "H"))
        self.assertEqual(reasons["C"], "theme_cap")
        self.assertEqual(reasons["E"], "macro_cap")
        self.assertEqual(reasons["B"], "correlation_cap")
        self.assertEqual(result.cash_slots, 0)

    def test_constraints_leave_cash_instead_of_relaxing_limits(self):
        sessions = _calendar(140)
        symbols = ["A", "B", "C"]
        universe = [
            LETFMember(symbol, f"P{symbol}", "one_theme", f"macro_{symbol}")
            for symbol in symbols
        ]
        product_bars = _long_bars(
            sessions, _independent_product_prices(sessions, symbols)
        )
        result = select_rotation_candidates(
            session=sessions[-1],
            sessions=sessions,
            product_bars=product_bars,
            universe=universe,
            scores={"A": 3.0, "B": 2.0, "C": 1.0},
        )

        self.assertEqual(result.selected, ("A",))
        self.assertEqual(result.cash_slots, 4)
        self.assertEqual(
            [audit.reason for audit in result.audits if not audit.accepted],
            ["theme_cap", "theme_cap"],
        )

    def test_cadence_uses_global_session_position_and_offset(self):
        sessions = _calendar(30)
        config = RotationConfig(cadence_sessions=5, rebalance_offset=2)

        self.assertFalse(is_rebalance_session(sessions[1], sessions, config))
        self.assertTrue(is_rebalance_session(sessions[2], sessions, config))
        self.assertTrue(is_rebalance_session(sessions[7], sessions, config))
        self.assertFalse(is_rebalance_session(sessions[8], sessions, config))

    def test_missing_proxy_session_is_rejected_not_shifted_or_filled(self):
        sessions = _calendar(240)
        session = sessions[220]
        universe = [
            LETFMember("A", "PA", "theme_a", "macro_a"),
            LETFMember("B", "PB", "theme_b", "macro_b"),
        ]
        proxy_bars = _long_bars(sessions, _proxy_prices(sessions, ["PA", "PB"]))
        missing_session = sessions[220 - 63]
        proxy_bars = proxy_bars.loc[
            ~(
                (proxy_bars["symbol"] == "PB")
                & (proxy_bars["timestamp"] == missing_session)
            )
        ]

        snapshot = compute_proxy_first_scores(
            session=session,
            sessions=sessions,
            proxy_bars=proxy_bars,
            universe=universe,
            eligible_symbols={"A", "B"},
        )

        self.assertIn("A", snapshot.scores)
        self.assertNotIn("B", snapshot.scores)
        self.assertEqual(
            snapshot.rejections["B"], "missing_contiguous_proxy_history"
        )

    def test_absolute_proxy_gate_and_volatility_adjustment_are_auditable(self):
        sessions = _calendar(240)
        session = sessions[220]
        position = np.arange(len(sessions), dtype=float)
        up = 100.0 * np.exp(
            0.0007 * position + 0.004 * np.sin(position / 8.0)
        )
        down = 100.0 * np.exp(
            -0.0007 * position + 0.004 * np.sin(position / 8.0 + 1.0)
        )
        universe = [
            LETFMember("UP", "PUP", "up", "equity"),
            LETFMember("DOWN", "PDOWN", "down", "equity"),
        ]
        proxy_bars = _long_bars(sessions, {"PUP": up, "PDOWN": down})

        guarded = compute_proxy_first_scores(
            session=session,
            sessions=sessions,
            proxy_bars=proxy_bars,
            universe=universe,
            eligible_symbols={"UP", "DOWN"},
        )
        self.assertIn("UP", guarded.scores)
        self.assertEqual(guarded.rejections["DOWN"], "absolute_proxy_gate")
        component = guarded.components["UP"]
        self.assertAlmostEqual(
            component.m126_5,
            component.raw_m126_5 / component.trailing_volatility,
            places=10,
        )
        self.assertAlmostEqual(
            component.m63_5,
            component.raw_m63_5 / component.trailing_volatility,
            places=10,
        )
        self.assertGreater(up[220], component.proxy_sma)
        self.assertGreater(component.absolute_return_63d, 0.0)

        ablation = compute_proxy_first_scores(
            session=session,
            sessions=sessions,
            proxy_bars=proxy_bars,
            universe=universe,
            eligible_symbols={"UP", "DOWN"},
            config=RotationConfig(
                absolute_proxy_gate=False,
                risk_adjust_momentum=False,
            ),
        )
        self.assertEqual(set(ablation.scores), {"UP", "DOWN"})
        self.assertAlmostEqual(
            ablation.components["UP"].m126_5,
            ablation.components["UP"].raw_m126_5,
            places=12,
        )

    def test_exact_session_eligibility_is_not_forward_filled(self):
        sessions = _calendar(240)
        session = sessions[220]
        symbols = ["A", "B"]
        universe = [
            LETFMember(symbol, f"P{symbol}", f"theme_{symbol}", f"macro_{symbol}")
            for symbol in symbols
        ]
        product_bars = _long_bars(
            sessions, _independent_product_prices(sessions, symbols)
        )
        proxy_bars = _long_bars(
            sessions, _proxy_prices(sessions, ["PA", "PB"])
        )
        eligibility = pd.DataFrame(
            {
                "timestamp": [sessions[219], sessions[219]],
                "symbol": symbols,
                "eligible": [True, True],
            }
        )

        decision = evaluate_rotation(
            session=session,
            sessions=sessions,
            product_bars=product_bars,
            proxy_bars=proxy_bars,
            universe=universe,
            eligibility=eligibility,
        )

        self.assertTrue(decision.rebalance_due)
        self.assertEqual(decision.scores, {})
        self.assertEqual(decision.selected, ())
        self.assertEqual(decision.cash_slots, 5)
        self.assertEqual(
            {audit.symbol: audit.reason for audit in decision.audits},
            {
                "A": "eligibility_missing_for_session",
                "B": "eligibility_missing_for_session",
            },
        )

    def test_missing_held_close_fails_closed(self):
        sessions = _calendar(240)
        session = sessions[220]
        universe = [LETFMember("A", "PA", "theme_a", "macro_a")]
        product_bars = _long_bars(
            sessions, _independent_product_prices(sessions, ["A"])
        )
        product_bars = product_bars.loc[
            ~(
                (product_bars["symbol"] == "A")
                & (product_bars["timestamp"] == session)
            )
        ]
        proxy_bars = _long_bars(sessions, _proxy_prices(sessions, ["PA"]))

        with self.assertRaises(HeldDataUnavailableError):
            evaluate_rotation(
                session=session,
                sessions=sessions,
                product_bars=product_bars,
                proxy_bars=proxy_bars,
                universe=universe,
                eligibility={"A": True},
                held_symbols={"A"},
            )

    def test_public_results_have_no_weight_or_gross_target(self):
        sessions = _calendar(240)
        session = sessions[220]
        universe = [LETFMember("A", "PA", "theme_a", "macro_a")]
        product_bars = _long_bars(
            sessions, _independent_product_prices(sessions, ["A"])
        )
        proxy_bars = _long_bars(sessions, _proxy_prices(sessions, ["PA"]))
        decision = evaluate_rotation(
            session=session,
            sessions=sessions,
            product_bars=product_bars,
            proxy_bars=proxy_bars,
            universe=universe,
            eligibility=pd.DataFrame(
                {
                    "timestamp": [session],
                    "symbol": ["A"],
                    "eligible": [True],
                }
            ),
        )

        self.assertEqual(decision.selected, ("A",))
        self.assertNotIn("gross_target", {field.name for field in dataclasses.fields(RotationConfig)})
        self.assertNotIn("target_weights", {field.name for field in dataclasses.fields(decision)})


if __name__ == "__main__":
    unittest.main()
