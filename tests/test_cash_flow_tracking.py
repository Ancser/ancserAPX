import json
import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch

from backend.data.alpaca_adapter import AlpacaAdapter
from backend.execution import scheduler
from backend.execution.tracker import LiveTracker


class AlpacaCashActivityTests(unittest.TestCase):
    def test_trans_activities_are_paginated_and_normalized(self):
        first_page = [
            {
                "id": f"activity-{index:03d}",
                "activity_type": "CSD",
                "date": "2026-07-17",
                # Sign is normalized from the activity type, not trusted here.
                "net_amount": "-10.00",
            }
            for index in range(100)
        ]
        second_page = [{
            "id": "activity-100",
            "activity_type": "CSW",
            "date": "2026-07-18",
            "transaction_time": "2026-07-18T14:30:00Z",
            "net_amount": "5.25",
        }]
        calls = []

        class TradingClient:
            def get(self, path, params):
                calls.append((path, dict(params)))
                return first_page if len(calls) == 1 else second_page

        adapter = AlpacaAdapter.__new__(AlpacaAdapter)
        adapter.trading_client = TradingClient()
        result = adapter.get_cash_activities(
            limit=101, after="2026-07-01", until="2026-07-20"
        )

        self.assertEqual(len(result), 101)
        self.assertEqual(calls[0][0], "/account/activities/TRANS")
        self.assertEqual(calls[0][1]["page_size"], 100)
        self.assertEqual(calls[0][1]["direction"], "asc")
        self.assertEqual(calls[0][1]["after"], "2026-07-01")
        self.assertEqual(calls[1][1]["page_token"], "activity-099")
        self.assertEqual(calls[1][1]["page_size"], 1)
        self.assertEqual(result[0]["amount"], 10.0)
        self.assertEqual(result[0]["net_amount"], 10.0)
        self.assertEqual(result[0]["direction"], "deposit")
        self.assertEqual(result[-1]["amount"], -5.25)
        self.assertEqual(result[-1]["date"], "2026-07-18")
        self.assertEqual(result[-1]["time"], "2026-07-18T14:30:00+00:00")
        self.assertEqual(result[-1]["time_precision"], "timestamp")
        self.assertEqual(result[0]["time_precision"], "date_only")
        self.assertEqual(result[-1]["direction"], "withdrawal")

    def test_cash_activity_api_error_is_not_masked_as_empty_data(self):
        class TradingClient:
            def get(self, path, params):
                raise ConnectionError("broker unavailable")

        adapter = AlpacaAdapter.__new__(AlpacaAdapter)
        adapter.trading_client = TradingClient()
        with self.assertRaisesRegex(RuntimeError, "cash activity request failed"):
            adapter.get_cash_activities()

    def test_unexpected_trans_schema_fails_closed(self):
        adapter = AlpacaAdapter.__new__(AlpacaAdapter)
        adapter.trading_client = type("TradingClient", (), {
            "get": lambda self, path, params: [{
                "id": "fee-1",
                "activity_type": "FEE",
                "date": "2026-07-17",
                "net_amount": "-1",
            }],
        })()
        with self.assertRaisesRegex(RuntimeError, "unsupported type"):
            adapter.get_cash_activities()


class CashFlowAdjustedTrackerTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        root = Path(self.temp_dir.name)
        self.tracker = LiveTracker("CashFlowUnit")
        self.tracker.history_path = str(root / "tracker.json")
        self.tracker.audit_path = str(root / "audit.jsonl")
        self.baseline = {"date": "2026-07-15", "equity": 2000.0}

    def tearDown(self):
        self.temp_dir.cleanup()

    def _record(self, date, equity, cash_activities):
        return self.tracker.record_daily_state(
            date_str=date,
            equity=equity,
            day_pnl=None,
            total_pnl_pct=None,
            allocations={"SPY": 1.0},
            factors=["Momentum"],
            cash_activities=cash_activities,
            performance_baseline=self.baseline,
        )

    def test_deposit_is_excluded_from_pnl_and_linked_return(self):
        first = self._record("2026-07-16", 1800.0, [])
        deposit = [{
            "id": "deposit-1",
            "activity_type": "CSD",
            "date": "2026-07-17",
            "net_amount": "1000.00",
        }]
        second = self._record("2026-07-17", 2800.0, deposit)

        self.assertAlmostEqual(first["cash_flow_adjusted_linked_return"], -0.1)
        self.assertEqual(second["day_pnl"], 0.0)
        self.assertEqual(second["net_cash_flow"], 1000.0)
        self.assertEqual(second["cash_flow_activity_ids"], ["deposit-1"])
        self.assertAlmostEqual(second["day_pnl_pct"], 0.0)
        self.assertAlmostEqual(second["total_pnl_pct"], -0.1)
        self.assertAlmostEqual(second["cash_flow_adjusted_linked_return"], -0.1)
        self.assertEqual(second["total_pnl_basis"], "cash_flow_adjusted_linked_return")
        self.assertTrue(second["cash_flow_adjusted"])
        self.assertEqual(second["performance_baseline"], self.baseline)
        self.assertTrue(second["return_calculation"]["is_estimate"])
        self.assertEqual(
            second["return_calculation"]["cash_activity_timestamp_precision"], "date_only"
        )
        self.assertEqual(second["observation_pnl"], second["day_pnl"])
        self.assertEqual(
            second["day_pnl_basis"], "tracker_observation_period_legacy_alias"
        )

    def test_cash_flow_on_baseline_date_fails_closed_when_timing_is_ambiguous(self):
        with self.assertRaisesRegex(ValueError, "baseline date is timing-ambiguous"):
            self._record("2026-07-16", 3000.0, [{
                "id": "same-day-after-baseline",
                "activity_type": "CSD",
                "date": "2026-07-15",
                "amount": 1000.0,
            }])

    def test_broker_calendar_day_pnl_is_separate_from_observation_pnl(self):
        self._record("2026-07-16", 1800.0, [])
        row = self.tracker.record_daily_state(
            date_str="2026-07-17",
            equity=2833.03,
            day_pnl=None,
            total_pnl_pct=None,
            allocations={"SPY": 1.0},
            factors=["Momentum"],
            account_snapshot={"last_equity": 2828.01},
            cash_activities=[],
            performance_baseline=self.baseline,
        )

        self.assertEqual(row["broker_calendar_day_cash_adjusted_pnl"], 5.02)
        self.assertEqual(row["broker_calendar_day_external_cash_flow"], 0.0)
        self.assertNotEqual(row["broker_calendar_day_cash_adjusted_pnl"], row["day_pnl"])

    def test_same_day_retry_does_not_apply_deposit_twice(self):
        deposit = [{
            "id": "deposit-1",
            "activity_type": "CSD",
            "date": "2026-07-17",
            "amount": 1000.0,
        }]
        first = self._record("2026-07-17", 3000.0, deposit)
        retry = self._record("2026-07-17", 3000.0, deposit)

        self.assertEqual(first["net_cash_flow"], 1000.0)
        self.assertEqual(first["day_pnl"], 0.0)
        self.assertEqual(retry["net_cash_flow"], 0.0)
        self.assertEqual(retry["cash_flow_activity_ids"], [])
        self.assertEqual(retry["day_pnl"], 0.0)
        self.assertAlmostEqual(retry["total_pnl_pct"], 0.0)
        self.assertEqual(len(self.tracker.get_history()), 2)

    def test_newly_visible_same_day_deposit_attaches_to_later_observation(self):
        before_deposit = self._record("2026-07-17", 1800.0, [])
        after_deposit = self._record("2026-07-17", 2800.0, [{
            "id": "late-deposit",
            "activity_type": "CSD",
            "date": "2026-07-17",
            "amount": 1000.0,
        }])

        self.assertAlmostEqual(before_deposit["total_pnl_pct"], -0.1)
        self.assertEqual(after_deposit["net_cash_flow"], 1000.0)
        self.assertEqual(after_deposit["day_pnl"], 0.0)
        self.assertAlmostEqual(after_deposit["total_pnl_pct"], -0.1)

    def test_backfilled_historical_deposit_attaches_to_first_same_day_observation(self):
        self._record("2026-07-16", 1800.0, [])
        self._record("2026-07-17", 2800.0, [])
        self._record("2026-07-17", 2810.0, [])
        row = self._record("2026-07-20", 2810.0, [{
            "id": "historical-deposit",
            "activity_type": "CSD",
            "date": "2026-07-17",
            "amount": 1000.0,
        }])

        # -10% to 7/16, zero cash-adjusted return on the first 7/17 snapshot,
        # then a 10/2800 gain.  Assigning the deposit to the retry would create
        # a spurious +55.6% followed by -35.4% pair instead.
        expected = (0.9 * (2810.0 / 2800.0)) - 1.0
        self.assertAlmostEqual(row["total_pnl_pct"], expected, places=6)

    def test_strategy_baseline_starts_at_first_rebalance_after_saved_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "rebalance_history_Main.json").write_text(json.dumps([
                {"rebalance_date": "2026-06-28", "equity": 4038.44},
                {"rebalance_date": "2026-07-09", "equity": 2000.0},
                {"rebalance_date": "2026-07-17", "equity": 2774.78},
            ]), encoding="utf-8")
            with patch.object(scheduler, "DAILY_LOCK_DIR", tmp):
                baseline = scheduler._strategy_performance_baseline(
                    "Main", {"saved_at": "2026-07-08T19:05:31"}
                )

        self.assertEqual(baseline, {"date": "2026-07-09", "equity": 2000.0})

    def test_strategy_baseline_rejects_partial_rebalance(self):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "rebalance_history_Main.json").write_text(json.dumps([
                {"rebalance_date": "2026-07-17", "equity": 2774.78,
                 "order_summary": {"status": "partial"}},
            ]), encoding="utf-8")
            with patch.object(scheduler, "DAILY_LOCK_DIR", tmp):
                baseline = scheduler._strategy_performance_baseline(
                    "Main", {"saved_at": "2026-07-08T19:05:31"}
                )

        self.assertIsNone(baseline)

    def test_strategy_baseline_rejects_different_saved_strategy_epoch(self):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "rebalance_history_Main.json").write_text(json.dumps([
                {"rebalance_date": "2026-07-17", "equity": 2774.78,
                 "order_summary": {"status": "completed"},
                 "strategy_config": {"saved_at": "2026-07-01T12:00:00"}},
            ]), encoding="utf-8")
            with patch.object(scheduler, "DAILY_LOCK_DIR", tmp):
                baseline = scheduler._strategy_performance_baseline(
                    "Main", {"saved_at": "2026-07-08T19:05:31"}
                )

        self.assertIsNone(baseline)

    def test_withdrawal_positive_input_is_normalized_negative(self):
        self._record("2026-07-16", 2200.0, [])
        withdrawal = [{
            "id": "withdrawal-1",
            "activity_type": "CSW",
            "date": "2026-07-17",
            "amount": 500.0,
        }]
        row = self._record("2026-07-17", 1700.0, withdrawal)

        self.assertEqual(row["net_cash_flow"], -500.0)
        self.assertEqual(row["day_pnl"], 0.0)
        self.assertAlmostEqual(row["total_pnl_pct"], 0.1)

    def test_legacy_call_keeps_unadjusted_behavior(self):
        Path(self.tracker.history_path).write_text(json.dumps([{
            "date": "2026-07-15",
            "equity": 100.0,
        }]), encoding="utf-8")
        row = self.tracker.record_daily_state(
            "2026-07-16",
            110.0,
            None,
            None,
            {"SPY": 1.0},
            ["Momentum"],
            account_snapshot={"last_equity": 105.0},
        )

        self.assertEqual(row["day_pnl"], 5.0)
        self.assertAlmostEqual(row["total_pnl_pct"], 0.1)
        self.assertEqual(
            row["total_pnl_basis"],
            "first_tracked_equity_unadjusted_for_cash_flows",
        )
        self.assertNotIn("cash_flow_adjusted_linked_return", row)


class SchedulerCashFlowIntegrationTests(unittest.TestCase):
    def test_broker_observation_passes_transfer_ledger_and_strategy_baseline(self):
        captured = {}

        class Adapter:
            def get_account(self):
                return {"equity": 3000.0, "last_equity": 2000.0}

            def get_positions(self):
                return []

            def get_orders(self, limit=500):
                return []

            def get_activities(self, limit=500):
                return []

            def get_cash_activities(self, limit=500):
                return [{
                    "id": "deposit-1", "activity_type": "CSD",
                    "date": "2026-07-17", "amount": 1000.0,
                }]

        class Tracker:
            def __init__(self, account):
                self.account = account

            def record_event(self, *args, **kwargs):
                pass

            def record_daily_state(self, **kwargs):
                captured.update(kwargs)
                return kwargs

        baseline = {"date": "2026-07-09", "equity": 2000.0}
        with patch("backend.execution.tracker.LiveTracker", Tracker), \
                patch.object(scheduler, "_merge_broker_history",
                             side_effect=lambda path, rows, key: rows), \
                patch.object(scheduler, "_strategy_performance_baseline",
                             return_value=baseline), \
                patch.object(scheduler, "_market_date",
                             return_value=date(2026, 7, 20)):
            result = scheduler.record_broker_observation(
                "Main", {"active_factors": ["Momentum"]}, adapter=Adapter()
            )

        self.assertEqual(captured["cash_activities"][0]["id"], "deposit-1")
        self.assertEqual(captured["performance_baseline"], baseline)
        self.assertEqual(result["cash_activity_error"], None)

    def test_cash_ledger_failure_does_not_append_false_unadjusted_return(self):
        events = []

        class Adapter:
            def get_account(self):
                return {"equity": 3000.0, "last_equity": 2000.0}

            def get_positions(self):
                return []

            def get_orders(self, limit=500):
                return []

            def get_activities(self, limit=500):
                return []

            def get_cash_activities(self, limit=500):
                raise RuntimeError("TRANS unavailable")

        class Tracker:
            def __init__(self, account):
                pass

            def record_event(self, event_type, status, **kwargs):
                events.append((event_type, status, kwargs.get("details")))

            def record_daily_state(self, **kwargs):
                raise AssertionError("must not append an unadjusted performance row")

        baseline = {"date": "2026-07-09", "equity": 2000.0}
        with patch("backend.execution.tracker.LiveTracker", Tracker), \
                patch.object(scheduler, "_merge_broker_history",
                             side_effect=lambda path, rows, key: rows), \
                patch.object(scheduler, "_strategy_performance_baseline",
                             return_value=baseline), \
                patch.object(scheduler, "_market_date", return_value=date(2026, 7, 20)):
            result = scheduler.record_broker_observation(
                "Main", {"active_factors": ["Momentum"]}, adapter=Adapter()
            )

        self.assertIn("TRANS unavailable", result["cash_activity_error"])
        self.assertFalse(result["tracker_state"]["performance_available"])
        self.assertTrue(any(
            event_type == "account_state" and status == "performance_blocked"
            for event_type, status, _ in events
        ))


if __name__ == "__main__":
    unittest.main()
