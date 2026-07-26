import json
import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import polars as pl

from backend.data import fetcher, store
from backend.data.alpaca_adapter import AlpacaAdapter
from backend.execution import scheduler
from backend.execution.oms import OrderManagementSystem
from backend.execution.tracker import LiveTracker


class FakeCalendarAdapter:
    def get_trading_days(self, start, end):
        return [datetime(2026, 7, 10).date(), datetime(2026, 7, 13).date(),
                datetime(2026, 7, 14).date()]

    def get_asset_eligibility(self, symbols, *, require_fractionable=True):
        effective = list(dict.fromkeys(str(symbol).upper() for symbol in symbols))
        return {
            "status": "passed",
            "source": "test_bulk_assets",
            "bulk_request_count": 1,
            "require_fractionable": require_fractionable,
            "configured_count": len(effective),
            "effective_count": len(effective),
            "effective_symbols": effective,
            "excluded_count": 0,
            "excluded_assets": {},
        }


class AssetEligibilityTests(unittest.TestCase):
    @staticmethod
    def _asset(symbol, status="active", tradable=True, fractionable=True):
        return SimpleNamespace(
            symbol=symbol,
            status=status,
            tradable=tradable,
            fractionable=fractionable,
            asset_class="us_equity",
            exchange="NASDAQ",
        )

    def test_bulk_asset_master_classifies_without_per_symbol_requests(self):
        calls = []

        class NotFoundError(Exception):
            status_code = 404

        class TradingClient:
            def get_all_assets(self, asset_filter):
                calls.append("bulk")
                return [
                    AssetEligibilityTests._asset("QQQ"),
                    AssetEligibilityTests._asset("SPY"),
                    AssetEligibilityTests._asset("AAA"),
                    AssetEligibilityTests._asset("BLOCKED", tradable=False),
                    AssetEligibilityTests._asset("WHOLE", fractionable=False),
                ]

            def get_asset(self, symbol):
                calls.append(f"lookup:{symbol}")
                if symbol == "OLD":
                    return AssetEligibilityTests._asset(
                        "OLD", status="inactive", tradable=False
                    )
                raise NotFoundError("404 asset not found")

        adapter = AlpacaAdapter.__new__(AlpacaAdapter)
        adapter.trading_client = TradingClient()
        report = adapter.get_asset_eligibility(
            ["AAA", "OLD", "BLOCKED", "WHOLE", "BK"]
        )

        self.assertEqual(calls, ["bulk", "lookup:OLD", "lookup:BK"])
        self.assertEqual(report["effective_symbols"], ["AAA"])
        self.assertEqual(report["excluded_assets"]["OLD"]["reason"], "inactive")
        self.assertEqual(report["excluded_assets"]["BLOCKED"]["reason"], "not_tradable")
        self.assertEqual(report["excluded_assets"]["WHOLE"]["reason"], "not_fractionable")
        self.assertEqual(report["excluded_assets"]["BK"]["reason"], "not_found")

    def test_unknown_active_asset_flag_fails_closed(self):
        class TradingClient:
            def get_all_assets(self, asset_filter):
                return [
                    AssetEligibilityTests._asset("QQQ"),
                    AssetEligibilityTests._asset("SPY"),
                    AssetEligibilityTests._asset("AAA", tradable=None),
                ]

        adapter = AlpacaAdapter.__new__(AlpacaAdapter)
        adapter.trading_client = TradingClient()
        with self.assertRaisesRegex(RuntimeError, "unknown tradable flag"):
            adapter.get_asset_eligibility(["AAA"])

    def test_non_us_equity_is_explicitly_excluded(self):
        class TradingClient:
            def get_all_assets(self, asset_filter):
                return [
                    AssetEligibilityTests._asset("QQQ"),
                    AssetEligibilityTests._asset("SPY"),
                    SimpleNamespace(
                        symbol="ODD", status="active", tradable=True,
                        fractionable=True, asset_class="crypto", exchange="CRYPTO",
                    ),
                ]

        adapter = AlpacaAdapter.__new__(AlpacaAdapter)
        adapter.trading_client = TradingClient()
        report = adapter.get_asset_eligibility(["ODD"])
        self.assertEqual(
            report["excluded_assets"]["ODD"]["reason"], "unsupported_asset_class"
        )


class LiveDataSafetyTests(unittest.TestCase):
    @staticmethod
    def _write_valid_physical(root: Path, symbol: str, expected: datetime, rows: int = 253):
        root.mkdir(parents=True, exist_ok=True)
        dates = [expected - timedelta(days=rows - 1 - i) for i in range(rows)]
        pl.DataFrame({
            "timestamp": dates,
            "symbol": [symbol] * rows,
            "open": [10.0] * rows,
            "high": [11.0] * rows,
            "low": [9.0] * rows,
            "close": [10.5] * rows,
            "volume": [1000.0] * rows,
        }).write_parquet(root / f"{symbol}.parquet")

    def test_store_load_includes_non_midnight_rows_on_end_date(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch.object(store, "STORE_DIR", root / "store"), \
                    patch.object(store, "MANIFEST_PATH", root / "manifest.json"):
                frame = pl.DataFrame({
                    "timestamp": [datetime(2026, 7, 13, 4, 0)],
                    "symbol": ["AAA"],
                    "open": [10.0], "high": [11.0], "low": [9.0],
                    "close": [10.5], "volume": [1000.0], "vwap": [10.4],
                    "trade_count": [10],
                })
                store.save(frame, "AAA")
                loaded = store.load(["AAA"], "2026-07-13", "2026-07-13").collect()
                self.assertEqual(loaded.height, 1)
                self.assertEqual(loaded["timestamp"][0], datetime(2026, 7, 13, 4, 0))

    def test_expected_session_excludes_current_premarket_session(self):
        now = datetime(2026, 7, 14, 9, 25, tzinfo=timezone.utc).astimezone(
            scheduler.MARKET_TZ
        )
        # Build the intended New York wall clock explicitly.
        now = datetime(2026, 7, 14, 9, 25, tzinfo=scheduler.MARKET_TZ)
        self.assertEqual(
            fetcher.expected_completed_session(FakeCalendarAdapter(), now=now),
            "2026-07-13",
        )

    def test_physical_inspection_rejects_manifest_false_positive(self):
        manifest = {
            symbol: {"last_date": "2026-07-13", "row_count": 300}
            for symbol in ("AAA", "QQQ", "SPY")
        }
        with tempfile.TemporaryDirectory() as tmp:
            physical_root = Path(tmp) / "store"
            self._write_valid_physical(physical_root, "QQQ", datetime(2026, 7, 13))
            self._write_valid_physical(physical_root, "SPY", datetime(2026, 7, 13))
            # AAA intentionally has fresh manifest metadata but no parquet.
            with patch.object(store, "STORE_DIR", physical_root), \
                    patch.object(fetcher.store, "get_manifest", return_value=manifest), \
                    patch.object(fetcher, "fetch_incremental", return_value={
                        "requested": 0, "updated": 0, "failed": 0,
                        "updated_symbols": [], "failed_symbols": [],
                    }):
                report = fetcher.sync_and_validate_live_data(
                    ["AAA"], adapter=FakeCalendarAdapter(),
                    now=datetime(2026, 7, 14, 9, 25, tzinfo=scheduler.MARKET_TZ),
                )
        self.assertFalse(report["passed"])
        self.assertEqual(report["physical_coverage_pct"], 0.0)
        self.assertIn("AAA", report["physical_validation"]["missing_files"])

    def test_physical_inspection_rejects_invalid_completed_close(self):
        with tempfile.TemporaryDirectory() as tmp:
            physical_root = Path(tmp) / "store"
            self._write_valid_physical(physical_root, "AAA", datetime(2026, 7, 13))
            frame = pl.read_parquet(physical_root / "AAA.parquet").with_columns(
                pl.when(pl.col("timestamp") == datetime(2026, 7, 13))
                .then(float("nan"))
                .otherwise(pl.col("close"))
                .alias("close")
            )
            frame.write_parquet(physical_root / "AAA.parquet")
            with patch.object(store, "STORE_DIR", physical_root):
                result = store.inspect_physical_data(["AAA"], "2026-07-13")
        self.assertFalse(result["passed"])
        self.assertTrue(any("close is not finite" in error
                            for error in result["invalid_symbols"]["AAA"]))

    def test_live_config_cannot_lower_coverage_or_freshness_below_100(self):
        manifest = {
            symbol: {"last_date": "2026-07-13", "row_count": 300}
            for symbol in ("AAA", "QQQ", "SPY")
        }
        with tempfile.TemporaryDirectory() as tmp:
            physical_root = Path(tmp) / "store"
            for symbol in ("AAA", "QQQ", "SPY"):
                self._write_valid_physical(physical_root, symbol, datetime(2026, 7, 13))
            with patch.object(store, "STORE_DIR", physical_root), \
                    patch.object(fetcher.store, "get_manifest", return_value=manifest), \
                    patch.object(fetcher, "fetch_incremental", return_value={
                        "requested": 0, "updated": 0, "failed": 0,
                        "updated_symbols": [], "failed_symbols": [],
                    }):
                report = fetcher.sync_and_validate_live_data(
                    ["AAA"],
                    config={"data_quality": {
                        "min_coverage_pct": 0, "min_fresh_coverage_pct": 1,
                    }},
                    adapter=FakeCalendarAdapter(),
                    now=datetime(2026, 7, 14, 9, 25, tzinfo=scheduler.MARKET_TZ),
                )
        self.assertTrue(report["passed"])
        self.assertEqual(report["thresholds"]["min_coverage_pct"], 100.0)
        self.assertEqual(report["thresholds"]["min_fresh_coverage_pct"], 100.0)
        self.assertEqual(report["requested_thresholds"]["min_coverage_pct"], 0.0)

    def test_inactive_asset_is_audited_and_removed_before_sync(self):
        class EligibilityAdapter(FakeCalendarAdapter):
            def get_asset_eligibility(self, symbols, *, require_fractionable=True):
                return {
                    "status": "passed",
                    "source": "test_bulk_assets",
                    "bulk_request_count": 1,
                    "require_fractionable": require_fractionable,
                    "configured_count": 2,
                    "effective_count": 1,
                    "effective_symbols": ["AAA"],
                    "excluded_count": 1,
                    "excluded_assets": {
                        "OLD": {"reason": "inactive", "status": "inactive"},
                    },
                }

        manifest = {
            symbol: {"last_date": "2026-07-13", "row_count": 300}
            for symbol in ("AAA", "QQQ", "SPY")
        }
        with tempfile.TemporaryDirectory() as tmp:
            physical_root = Path(tmp) / "store"
            for symbol in ("AAA", "QQQ", "SPY"):
                self._write_valid_physical(physical_root, symbol, datetime(2026, 7, 13))
            with patch.object(store, "STORE_DIR", physical_root), \
                    patch.object(fetcher.store, "get_manifest", return_value=manifest), \
                    patch.object(fetcher, "fetch_incremental", return_value={
                        "requested": 0, "updated": 0, "failed": 0,
                        "updated_symbols": [], "failed_symbols": [],
                    }) as incremental:
                report = fetcher.sync_and_validate_live_data(
                    ["AAA", "OLD"], adapter=EligibilityAdapter(),
                    now=datetime(2026, 7, 14, 9, 25, tzinfo=scheduler.MARKET_TZ),
                )

        self.assertTrue(report["passed"])
        self.assertEqual(report["configured_universe"], ["AAA", "OLD"])
        self.assertEqual(report["effective_universe"], ["AAA"])
        self.assertEqual(report["excluded_assets"]["OLD"]["reason"], "inactive")
        self.assertEqual(incremental.call_args.args[0], ["AAA", "QQQ", "SPY"])

    def test_asset_api_failure_blocks_before_any_data_sync(self):
        class FailedEligibilityAdapter(FakeCalendarAdapter):
            def get_asset_eligibility(self, symbols, *, require_fractionable=True):
                raise RuntimeError("asset service timeout")

        with patch.object(fetcher, "fetch_incremental") as incremental:
            report = fetcher.sync_and_validate_live_data(
                ["AAA"], adapter=FailedEligibilityAdapter(),
                now=datetime(2026, 7, 14, 9, 25, tzinfo=scheduler.MARKET_TZ),
            )
        self.assertFalse(report["passed"])
        self.assertIn("asset service timeout", report["errors"][0])
        incremental.assert_not_called()

    def test_unclassified_asset_result_fails_closed_before_sync(self):
        class IncompleteEligibilityAdapter(FakeCalendarAdapter):
            def get_asset_eligibility(self, symbols, *, require_fractionable=True):
                return {
                    "status": "passed",
                    "effective_symbols": ["AAA"],
                    "excluded_assets": {},
                }

        with patch.object(fetcher, "fetch_incremental") as incremental:
            report = fetcher.sync_and_validate_live_data(
                ["AAA", "MISSING"], adapter=IncompleteEligibilityAdapter(),
                now=datetime(2026, 7, 14, 9, 25, tzinfo=scheduler.MARKET_TZ),
            )
        self.assertFalse(report["passed"])
        self.assertIn("unclassified configured symbols: MISSING", report["errors"][0])
        incremental.assert_not_called()

    def test_partial_cross_section_fails_freshness_gate(self):
        universe = [f"S{i:02d}" for i in range(20)]
        manifest = {
            symbol: {"last_date": "2026-07-13" if i < 18 else "2026-07-10", "row_count": 300}
            for i, symbol in enumerate(universe)
        }
        manifest.update({
            "QQQ": {"last_date": "2026-07-13", "row_count": 300},
            "SPY": {"last_date": "2026-07-13", "row_count": 300},
        })
        with patch.object(fetcher.store, "get_manifest", return_value=manifest), \
                patch.object(fetcher, "fetch_incremental", return_value={
                    "requested": 2, "updated": 0, "failed": 2,
                    "updated_symbols": [], "failed_symbols": universe[-2:],
                }):
            report = fetcher.sync_and_validate_live_data(
                universe,
                config={"data_quality": {"min_fresh_coverage_pct": 95}},
                adapter=FakeCalendarAdapter(),
                now=datetime(2026, 7, 14, 9, 25, tzinfo=scheduler.MARKET_TZ),
            )
        self.assertFalse(report["passed"])
        self.assertEqual(report["fresh_coverage_pct"], 90.0)

    def test_default_gate_rejects_even_one_stale_configured_symbol(self):
        universe = [f"S{i:02d}" for i in range(20)]
        manifest = {
            symbol: {
                "last_date": "2026-07-10" if i == 19 else "2026-07-13",
                "row_count": 300,
            }
            for i, symbol in enumerate(universe)
        }
        manifest.update({
            "QQQ": {"last_date": "2026-07-13", "row_count": 300},
            "SPY": {"last_date": "2026-07-13", "row_count": 300},
        })
        with patch.object(fetcher.store, "get_manifest", return_value=manifest), \
                patch.object(fetcher, "fetch_incremental", return_value={
                    "requested": 1, "updated": 0, "failed": 1,
                    "updated_symbols": [], "failed_symbols": [universe[-1]],
                }):
            report = fetcher.sync_and_validate_live_data(
                universe,
                adapter=FakeCalendarAdapter(),
                now=datetime(2026, 7, 14, 9, 25, tzinfo=scheduler.MARKET_TZ),
            )
        self.assertFalse(report["passed"])
        self.assertEqual(report["thresholds"]["min_fresh_coverage_pct"], 100.0)
        self.assertIn(universe[-1], report["effective_universe"])
        self.assertEqual(report["excluded_assets"], {})
        self.assertIn(universe[-1], report["stale_symbols"])

    def test_regime_guard_requires_fresh_long_gauge_history(self):
        manifest = {
            "AAA": {"last_date": "2026-07-13", "row_count": 300},
            "QQQ": {"last_date": "2026-07-13", "row_count": 100},
            "SPY": {"last_date": "2026-07-13", "row_count": 300},
        }
        with patch.object(fetcher.store, "get_manifest", return_value=manifest), \
                patch.object(fetcher, "fetch_incremental", return_value={
                    "requested": 0, "updated": 0, "failed": 0,
                    "updated_symbols": [], "failed_symbols": [],
                }):
            report = fetcher.sync_and_validate_live_data(
                ["AAA"],
                config={"risk_management": {"regime_mode": "cash"}},
                adapter=FakeCalendarAdapter(),
                now=datetime(2026, 7, 14, 9, 25, tzinfo=scheduler.MARKET_TZ),
            )
        self.assertFalse(report["passed"])
        self.assertTrue(any("<220 rows" in error for error in report["errors"]))

    def test_incremental_sync_does_not_refetch_full_history_for_fresh_names(self):
        calls = []

        class Adapter:
            def fetch_history(self, symbols, start, end):
                calls.append((tuple(symbols), start, end))
                return pl.DataFrame({
                    "timestamp": [datetime(2026, 7, 13)] * len(symbols),
                    "symbol": symbols,
                    "close": [1.0] * len(symbols),
                }).lazy()

        manifest = {"FRESH": {"last_date": "2026-07-10", "row_count": 300}}
        with patch.object(fetcher.store, "get_manifest", return_value=manifest), \
                patch.object(fetcher.store, "save"):
            fetcher.fetch_incremental(
                ["FRESH", "NEW"], required_as_of="2026-07-13",
                end_date="2026-07-14", adapter=Adapter(),
            )
        self.assertIn((("FRESH",), "2026-07-10", "2026-07-14"), calls)
        self.assertIn((("NEW",), fetcher.TEN_YEARS_START, "2026-07-14"), calls)


class TrackerAndAccountTests(unittest.TestCase):
    def test_tracker_computes_real_day_pnl_total_return_and_gross(self):
        with tempfile.TemporaryDirectory() as tmp:
            tracker = LiveTracker("Unit")
            tracker.history_path = str(Path(tmp) / "tracker.json")
            tracker.audit_path = str(Path(tmp) / "audit.jsonl")
            Path(tracker.history_path).write_text(json.dumps([{
                "date": "2026-07-10", "equity": 100.0,
            }]), encoding="utf-8")
            row = tracker.record_daily_state(
                "2026-07-13", 110.0, None, None,
                {"AAA": 0.75, "BBB": 0.75}, ["Momentum"],
                account_snapshot={"last_equity": 105.0},
            )
            self.assertEqual(row["day_pnl"], 5.0)
            self.assertAlmostEqual(row["total_pnl_pct"], 0.1)
            self.assertAlmostEqual(row["gross_exposure"], 1.5)
            self.assertEqual(len(tracker.get_audit_events()), 1)

    def test_adapter_preserves_last_equity_and_portfolio_value(self):
        account = SimpleNamespace(
            equity="110", last_equity="105", portfolio_value="110",
            buying_power="50", cash="10", long_market_value="150",
            short_market_value="0", initial_margin="70", maintenance_margin="50",
            daytrade_count=None, status=SimpleNamespace(value="ACTIVE"), currency="USD",
        )
        adapter = AlpacaAdapter.__new__(AlpacaAdapter)
        adapter.trading_client = SimpleNamespace(get_account=lambda: account)
        result = adapter.get_account()
        self.assertEqual(result["last_equity"], 105.0)
        self.assertEqual(result["portfolio_value"], 110.0)
        self.assertEqual(result["daytrade_count"], 0)

    def test_broker_history_merges_fill_by_activity_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "fills.json")
            scheduler._merge_broker_history(path, [{"id": "F1", "price": 10}], "id")
            merged = scheduler._merge_broker_history(path, [{"id": "F1", "price": 11}], "id")
            self.assertEqual(merged, [{"id": "F1", "price": 11}])

    def test_oms_account_failure_cannot_cancel_or_liquidate(self):
        calls = []

        class Adapter:
            def get_account(self):
                return {"equity": 0.0, "status": "Error"}

            def cancel_all_orders(self):
                calls.append("cancel")

        class Tracker:
            def record_event(self, *args, **kwargs):
                calls.append("audit")

        oms = OrderManagementSystem.__new__(OrderManagementSystem)
        oms.account_name = "Unit"
        oms.alpaca = Adapter()
        oms.tracker = Tracker()
        oms.last_summary = {}
        with self.assertRaises(RuntimeError):
            oms.generate_and_execute_orders({"AAA": 1.0})
        self.assertNotIn("cancel", calls)

    def test_oms_cancel_failure_blocks_replacement_batch(self):
        calls = []

        class Adapter:
            def get_account(self):
                return {
                    "equity": 100.0, "status": "ACTIVE",
                    "long_market_value": 0.0, "short_market_value": 0.0,
                }

            def get_positions(self):
                return []

            def get_latest_prices(self, symbols):
                return {"AAA": 10.0}

            def cancel_all_orders(self):
                calls.append("cancel")
                raise RuntimeError("broker cancel unavailable")

            def submit_order(self, *args, **kwargs):
                calls.append("submit")

        class Tracker:
            def record_event(self, event_type, status, **kwargs):
                calls.append((event_type, status))

        oms = OrderManagementSystem.__new__(OrderManagementSystem)
        oms.account_name = "Unit"
        oms.alpaca = Adapter()
        oms.tracker = Tracker()
        oms.last_summary = {}
        with self.assertRaisesRegex(RuntimeError, "replacement batch was not submitted"):
            oms.generate_and_execute_orders({"AAA": 1.0})
        self.assertIn("cancel", calls)
        self.assertNotIn("submit", calls)
        self.assertIn(("open_orders", "cancel_failed"), calls)


class LastRebalanceRecoveryTests(unittest.TestCase):
    @staticmethod
    def _write(path: Path, value):
        path.write_text(json.dumps(value), encoding="utf-8")

    def test_partial_latest_falls_back_to_legacy_completed_rebalance(self):
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(scheduler, "DAILY_LOCK_DIR", tmp):
            root = Path(tmp)
            self._write(root / "last_rebalance_Main.json", {
                "rebalance_date": "2026-07-17",
                "snapshot_kind": "rebalance",
                "order_summary": {"status": "partial"},
            })
            self._write(root / "rebalance_history_Main.json", [
                {
                    "rebalance_date": "2026-07-09",
                    "positions": {"AAA": {"weight": 1.0}},
                },
                {
                    "rebalance_date": "2026-07-17",
                    "snapshot_kind": "rebalance",
                    "order_summary": {"status": "partial"},
                },
            ])

            self.assertEqual(
                scheduler._last_rebalance_date("Main"),
                datetime(2026, 7, 9).date(),
            )

    def test_valid_completed_latest_snapshot_is_authoritative(self):
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(scheduler, "DAILY_LOCK_DIR", tmp):
            root = Path(tmp)
            self._write(root / "last_rebalance_Main.json", {
                "rebalance_date": "2026-07-17",
                "snapshot_kind": "rebalance",
                "order_summary": {"status": "completed"},
            })
            # A malformed/future history row must not displace the valid,
            # terminal snapshot used by cadence.
            self._write(root / "rebalance_history_Main.json", [
                {
                    "rebalance_date": "2026-07-18",
                    "snapshot_kind": "rebalance",
                    "order_summary": {"status": "completed"},
                },
            ])

            self.assertEqual(
                scheduler._last_rebalance_date("Main"),
                datetime(2026, 7, 17).date(),
            )

    def test_history_uses_newest_eligible_and_ignores_non_rebalances(self):
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(scheduler, "DAILY_LOCK_DIR", tmp):
            root = Path(tmp)
            self._write(root / "last_rebalance_Main.json", {
                "rebalance_date": "2026-07-20",
                "snapshot_kind": "rebalance",
                "order_summary": {"status": "pending"},
            })
            self._write(root / "rebalance_history_Main.json", [
                {"rebalance_date": "2026-07-09"},
                {
                    "rebalance_date": "2026-07-16",
                    "snapshot_kind": "risk_overlay",
                    "order_summary": {"status": "completed"},
                },
                {
                    "rebalance_date": "2026-07-15",
                    "snapshot_kind": "rebalance",
                    "order_summary": {"status": "completed"},
                },
                {
                    "rebalance_date": "2026-07-17",
                    "snapshot_kind": "rebalance",
                    "order_summary": {"status": "failed"},
                },
            ])

            self.assertEqual(
                scheduler._last_rebalance_date("Main"),
                datetime(2026, 7, 15).date(),
            )

    def test_all_partial_history_returns_none(self):
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(scheduler, "DAILY_LOCK_DIR", tmp):
            root = Path(tmp)
            self._write(root / "last_rebalance_Main.json", {
                "rebalance_date": "2026-07-17",
                "snapshot_kind": "rebalance",
                "order_summary": {"status": "partial"},
            })
            self._write(root / "rebalance_history_Main.json", [
                {
                    "rebalance_date": "2026-07-10",
                    "snapshot_kind": "rebalance",
                    "order_summary": {"status": "pending"},
                },
                {
                    "rebalance_date": "2026-07-17",
                    "snapshot_kind": "rebalance",
                    "order_summary": {"status": "partial"},
                },
            ])

            self.assertIsNone(scheduler._last_rebalance_date("Main"))

    def test_malformed_files_fail_safe(self):
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(scheduler, "DAILY_LOCK_DIR", tmp):
            root = Path(tmp)
            (root / "last_rebalance_Main.json").write_text(
                "{incomplete", encoding="utf-8"
            )
            self._write(root / "rebalance_history_Main.json", {
                "rebalance_date": "2026-07-09"
            })

            self.assertIsNone(scheduler._last_rebalance_date("Main"))


class MarginEligibilitySafetyTests(unittest.TestCase):
    def test_below_2000_caps_requested_outer_gross_at_one(self):
        weights, audit = scheduler._apply_margin_eligibility_cap(
            {"AAA": 0.75, "BBB": 0.75},
            {
                "equity": 1868.42, "multiplier": 2, "status": "ACTIVE",
                "account_blocked": False, "trading_blocked": False,
            },
        )

        self.assertAlmostEqual(sum(abs(value) for value in weights.values()), 1.0)
        self.assertTrue(audit["triggered"])
        self.assertIn("equity_below_2000_margin_minimum", audit["reasons"])

    def test_cash_only_broker_multiplier_caps_even_above_2000(self):
        weights, audit = scheduler._apply_margin_eligibility_cap(
            {"AAA": 1.5},
            {
                "equity": 5000.0, "multiplier": 1, "status": "ACTIVE",
                "account_blocked": False, "trading_blocked": False,
            },
        )

        self.assertEqual(weights, {"AAA": 1.0})
        self.assertIn("broker_multiplier_is_cash_only", audit["reasons"])

    def test_margin_eligible_account_preserves_requested_target(self):
        weights, audit = scheduler._apply_margin_eligibility_cap(
            {"AAA": 0.75, "BBB": 0.75},
            {
                "equity": 5000.0, "multiplier": 2, "status": "ACTIVE",
                "account_blocked": False, "trading_blocked": False,
            },
        )

        self.assertEqual(weights, {"AAA": 0.75, "BBB": 0.75})
        self.assertFalse(audit["triggered"])

    def test_missing_margin_fields_blocks_leveraged_target(self):
        with self.assertRaisesRegex(RuntimeError, "eligibility fields"):
            scheduler._apply_margin_eligibility_cap(
                {"AAA": 1.5}, {
                    "equity": 5000.0, "status": "ACTIVE",
                    "account_blocked": False, "trading_blocked": False,
                }
            )

    def test_invalid_margin_multiplier_blocks_leveraged_target(self):
        for multiplier in (float("nan"), True):
            with self.subTest(multiplier=multiplier), self.assertRaisesRegex(
                RuntimeError, "eligibility fields are invalid"
            ):
                scheduler._apply_margin_eligibility_cap(
                    {"AAA": 1.5}, {
                        "equity": 5000.0, "multiplier": multiplier,
                        "status": "ACTIVE", "account_blocked": False,
                        "trading_blocked": False,
                    }
                )

    def test_nan_target_weight_is_hard_blocked(self):
        with self.assertRaisesRegex(RuntimeError, "weight.*finite"):
            scheduler._apply_margin_eligibility_cap(
                {"AAA": float("nan")},
                {"equity": 5000.0, "status": "ACTIVE"},
            )

    def test_nonfinite_requested_gross_is_hard_blocked(self):
        with self.assertRaisesRegex(RuntimeError, "gross must be finite"):
            scheduler._apply_margin_eligibility_cap(
                {"AAA": 1e308, "BBB": 1e308},
                {"equity": 5000.0, "status": "ACTIVE"},
            )

    def test_nan_equity_is_hard_blocked_at_one_x(self):
        with self.assertRaisesRegex(RuntimeError, "equity must be finite"):
            scheduler._apply_margin_eligibility_cap(
                {"AAA": 1.0},
                {"equity": float("nan"), "status": "ACTIVE"},
            )

    def test_non_active_status_is_hard_blocked(self):
        with self.assertRaisesRegex(RuntimeError, "not ACTIVE.*REJECTED"):
            scheduler._apply_margin_eligibility_cap(
                {"AAA": 1.0},
                {"equity": 5000.0, "status": "REJECTED"},
            )

    def test_broker_block_flags_are_hard_blocks_at_or_below_one_x(self):
        for flag in ("account_blocked", "trading_blocked"):
            with self.subTest(flag=flag), self.assertRaisesRegex(
                RuntimeError, "account or trading is blocked"
            ):
                scheduler._apply_margin_eligibility_cap(
                    {"AAA": 0.75},
                    {
                        "equity": 5000.0, "status": "ACTIVE",
                        flag: True,
                    },
                )

    def test_active_unblocked_one_x_does_not_require_margin_multiplier(self):
        weights, audit = scheduler._apply_margin_eligibility_cap(
            {"AAA": 0.6, "BBB": 0.4},
            {
                "equity": 5000.0, "status": "ACTIVE",
                "account_blocked": False, "trading_blocked": False,
            },
        )

        self.assertEqual(weights, {"AAA": 0.6, "BBB": 0.4})
        self.assertFalse(audit["triggered"])
        self.assertIsNone(audit["multiplier"])

class SafeExecutionPipelineTests(unittest.TestCase):
    @staticmethod
    def _tracker(events):
        class Tracker:
            def __init__(self, account):
                self.account = account

            def record_event(self, event_type, status, **kwargs):
                events.append((event_type, status))
                return {}
        return Tracker

    def test_account_execution_lock_blocks_all_same_account_mutations(self):
        events = []
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(scheduler, "DAILY_LOCK_DIR", tmp), \
                patch("backend.execution.tracker.LiveTracker", self._tracker(events)), \
                patch.object(scheduler, "_execute_account_rebalance_locked") as rebalance, \
                patch.object(scheduler, "_execute_daily_risk_overlay_locked") as daily_risk:
            held = scheduler._try_acquire_account_execution_lock("Main", "owner-run")
            self.assertTrue(held["acquired"])
            try:
                rebalance_result = scheduler.execute_account_rebalance(
                    "Main", {"universe": ["AAA"]}, run_id="rebalance-contender"
                )
                risk_result = scheduler.execute_daily_risk_overlay(
                    "Main", {"universe": ["AAA"]}, run_id="risk-contender"
                )
            finally:
                scheduler._release_account_execution_lock(held)

        self.assertEqual(rebalance_result["stage"], "execution_lock")
        self.assertEqual(risk_result["stage"], "execution_lock")
        rebalance.assert_not_called()
        daily_risk.assert_not_called()
        self.assertIn(("rebalance_run", "blocked"), events)
        self.assertIn(("daily_risk", "blocked"), events)

    def test_account_execution_lock_releases_after_inner_exception(self):
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(scheduler, "DAILY_LOCK_DIR", tmp), \
                patch.object(
                    scheduler,
                    "_execute_account_rebalance_locked",
                    side_effect=RuntimeError("simulated failure"),
                ):
            with self.assertRaisesRegex(RuntimeError, "simulated failure"):
                scheduler.execute_account_rebalance(
                    "Main", {"universe": ["AAA"]}, run_id="failed-run"
                )
            self.assertFalse(scheduler._execution_lock_path("Main").exists())
            next_lock = scheduler._try_acquire_account_execution_lock("Main", "next-run")
            self.assertTrue(next_lock["acquired"])
            scheduler._release_account_execution_lock(next_lock)

    def test_account_execution_lock_is_scoped_per_account(self):
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(scheduler, "DAILY_LOCK_DIR", tmp):
            main_lock = scheduler._try_acquire_account_execution_lock("Main", "main-run")
            alt_lock = scheduler._try_acquire_account_execution_lock("Alt", "alt-run")
            try:
                self.assertTrue(main_lock["acquired"])
                self.assertTrue(alt_lock["acquired"])
                self.assertNotEqual(main_lock["path"], alt_lock["path"])
            finally:
                scheduler._release_account_execution_lock(main_lock)
                scheduler._release_account_execution_lock(alt_lock)

    def test_account_execution_lock_is_case_insensitive(self):
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(scheduler, "DAILY_LOCK_DIR", tmp):
            held = scheduler._try_acquire_account_execution_lock("Main", "owner-run")
            try:
                contender = scheduler._try_acquire_account_execution_lock(
                    " main ", "contender-run"
                )
                self.assertFalse(contender["acquired"])
                self.assertEqual(held["path"], contender["path"])
            finally:
                scheduler._release_account_execution_lock(held)

    def test_new_empty_execution_lock_is_treated_as_active(self):
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(scheduler, "DAILY_LOCK_DIR", tmp):
            path = scheduler._execution_lock_path("Main")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"")
            contender = scheduler._try_acquire_account_execution_lock(
                "Main", "contender-run"
            )
            self.assertFalse(contender["acquired"])
            self.assertTrue(path.exists())
            self.assertIn("initializing", contender["reason"])

    def test_old_unreadable_execution_lock_can_be_safely_reclaimed(self):
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(scheduler, "DAILY_LOCK_DIR", tmp):
            path = scheduler._execution_lock_path("Main")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{incomplete", encoding="utf-8")
            old = datetime.now().timestamp() - scheduler.EXECUTION_LOCK_STALE_SECONDS - 10
            os.utime(path, (old, old))
            acquired = scheduler._try_acquire_account_execution_lock(
                "Main", "recovery-run"
            )
            try:
                self.assertTrue(acquired["acquired"])
                self.assertFalse(Path(f"{path}.cleanup").exists())
            finally:
                scheduler._release_account_execution_lock(acquired)

    def test_old_lock_is_never_stolen_from_a_live_owner(self):
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(scheduler, "DAILY_LOCK_DIR", tmp):
            held = scheduler._try_acquire_account_execution_lock("Main", "long-run")
            path = Path(held["path"])
            old = datetime.now().timestamp() - scheduler.EXECUTION_LOCK_STALE_SECONDS - 10
            os.utime(path, (old, old))
            try:
                contender = scheduler._try_acquire_account_execution_lock(
                    "Main", "contender-run"
                )
                self.assertFalse(contender["acquired"])
                self.assertEqual(contender["owner"]["run_id"], "long-run")
            finally:
                scheduler._release_account_execution_lock(held)

    def test_pipeline_syncs_and_checks_asof_before_oms(self):
        calls = []
        events = []
        observed = {}

        def sync(*args, **kwargs):
            calls.append("sync")
            return {
                "passed": True,
                "expected_as_of": "2026-07-13",
                "effective_universe": ["AAA"],
            }

        class Strategy:
            def __init__(self, account):
                pass

            def calculate_targets(self, config):
                calls.append("targets")
                observed["strategy_universe"] = list(config["universe"])
                return {"allocations": {"AAA": 1.0}, "as_of_date": "2026-07-13",
                        "factor_weights": {}, "vol_metrics": {}}

        class OMS:
            def __init__(self, account):
                self.last_summary = {"status": "completed", "failed": 0}

            def generate_and_execute_orders(self, *args, **kwargs):
                calls.append("oms")
                observed["oms_universe"] = list(args[1]["universe"])
                return []

        account_adapter = SimpleNamespace(get_account=lambda: {
            "equity": 5000.0, "status": "ACTIVE",
            "account_blocked": False, "trading_blocked": False,
        })

        with tempfile.TemporaryDirectory() as tmp, \
                patch("backend.data.fetcher.sync_and_validate_live_data", side_effect=sync), \
                patch("backend.execution.strategy.LiveStrategy", Strategy), \
                patch("backend.execution.oms.OrderManagementSystem", OMS), \
                patch("backend.execution.tracker.LiveTracker", self._tracker(events)), \
                patch.object(scheduler, "AlpacaAdapter", return_value=account_adapter), \
                patch.object(scheduler, "_evaluate_daily_risk", side_effect=lambda *a: calls.append("risk") or {"in_market": True}), \
                patch.object(scheduler, "_write_lock", side_effect=lambda *a: calls.append("lock")), \
                patch.object(scheduler, "record_broker_observation", side_effect=lambda *a, **k: calls.append("observe") or {}), \
                patch.object(scheduler, "_is_nyse_session_today", return_value=True), \
                patch.object(scheduler, "_inside_scheduled_window", return_value=True), \
                patch.object(scheduler, "_stop_flag_path", return_value=Path(tmp) / "absent"):
            result = scheduler.execute_account_rebalance(
                "Main", {"universe": ["AAA", "OLD"], "active_factors": ["Momentum"]}
            )
        self.assertTrue(result["success"])
        self.assertEqual(calls, ["sync", "risk", "targets", "oms", "lock", "observe"])
        self.assertEqual(observed["strategy_universe"], ["AAA"])
        self.assertEqual(observed["oms_universe"], ["AAA"])

    def test_one_x_rebalance_checks_account_hard_block_before_oms(self):
        events = []

        class Strategy:
            def __init__(self, account):
                pass

            def calculate_targets(self, config):
                return {
                    "allocations": {"AAA": 1.0},
                    "as_of_date": "2026-07-13",
                    "factor_weights": {},
                    "vol_metrics": {},
                }

        account_adapter = SimpleNamespace(get_account=lambda: {
            "equity": 5000.0, "status": "ACTIVE",
            "account_blocked": True, "trading_blocked": False,
        })
        with tempfile.TemporaryDirectory() as tmp, \
                patch("backend.data.fetcher.sync_and_validate_live_data", return_value={
                    "passed": True, "expected_as_of": "2026-07-13",
                    "effective_universe": ["AAA"],
                }), \
                patch("backend.execution.strategy.LiveStrategy", Strategy), \
                patch("backend.execution.oms.OrderManagementSystem") as oms, \
                patch("backend.execution.tracker.LiveTracker", self._tracker(events)), \
                patch.object(scheduler, "AlpacaAdapter", return_value=account_adapter), \
                patch.object(scheduler, "_evaluate_daily_risk", return_value={"in_market": True}), \
                patch.object(scheduler, "_is_nyse_session_today", return_value=True), \
                patch.object(scheduler, "_stop_flag_path", return_value=Path(tmp) / "absent"):
            result = scheduler.execute_account_rebalance(
                "Main", {"universe": ["AAA"]}
            )

        self.assertEqual(result["stage"], "margin_eligibility")
        self.assertIn("account or trading is blocked", result["error"])
        oms.assert_not_called()

    def test_target_outside_effective_universe_never_reaches_oms(self):
        events = []

        class Strategy:
            def __init__(self, account):
                pass

            def calculate_targets(self, config):
                return {
                    "allocations": {"OLD": 1.0},
                    "as_of_date": "2026-07-13",
                    "factor_weights": {},
                    "vol_metrics": {},
                }

        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(scheduler, "DAILY_LOCK_DIR", tmp), \
                patch("backend.data.fetcher.sync_and_validate_live_data", return_value={
                    "passed": True,
                    "expected_as_of": "2026-07-13",
                    "effective_universe": ["AAA"],
                }), \
                patch("backend.execution.strategy.LiveStrategy", Strategy), \
                patch("backend.execution.oms.OrderManagementSystem") as oms, \
                patch("backend.execution.tracker.LiveTracker", self._tracker(events)), \
                patch.object(scheduler, "_evaluate_daily_risk", return_value={"in_market": True}), \
                patch.object(scheduler, "_is_nyse_session_today", return_value=True), \
                patch.object(scheduler, "_stop_flag_path", return_value=Path(tmp) / "absent"):
            result = scheduler.execute_account_rebalance(
                "Main", {"universe": ["AAA", "OLD"]}
            )

        self.assertEqual(result["stage"], "effective_universe_gate")
        self.assertEqual(result["unauthorized_targets"], ["OLD"])
        oms.assert_not_called()

    def test_elapsed_execution_window_blocks_scheduled_but_force_can_bypass(self):
        events = []
        oms_calls = []

        class Strategy:
            def __init__(self, account):
                pass

            def calculate_targets(self, config):
                return {
                    "allocations": {"AAA": 1.0},
                    "as_of_date": "2026-07-13",
                    "factor_weights": {},
                    "vol_metrics": {},
                }

        class OMS:
            def __init__(self, account):
                self.last_summary = {"status": "completed", "failed": 0}

            def generate_and_execute_orders(self, *args, **kwargs):
                oms_calls.append("submitted")
                return []

        sync_report = {
            "passed": True,
            "expected_as_of": "2026-07-13",
            "effective_universe": ["AAA"],
        }
        account_adapter = SimpleNamespace(get_account=lambda: {
            "equity": 5000.0, "status": "ACTIVE",
            "account_blocked": False, "trading_blocked": False,
        })
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(scheduler, "DAILY_LOCK_DIR", tmp), \
                patch("backend.data.fetcher.sync_and_validate_live_data", return_value=sync_report), \
                patch("backend.execution.strategy.LiveStrategy", Strategy), \
                patch("backend.execution.oms.OrderManagementSystem", OMS), \
                patch("backend.execution.tracker.LiveTracker", self._tracker(events)), \
                patch.object(scheduler, "AlpacaAdapter", return_value=account_adapter), \
                patch.object(scheduler, "_evaluate_daily_risk", return_value={"in_market": True}), \
                patch.object(scheduler, "_inside_scheduled_window", return_value=False), \
                patch.object(scheduler, "_is_nyse_session_today", return_value=True), \
                patch.object(scheduler, "_stop_flag_path", return_value=Path(tmp) / "absent"), \
                patch.object(scheduler, "_write_lock"), \
                patch.object(scheduler, "record_broker_observation", return_value={}):
            scheduled = scheduler.execute_account_rebalance(
                "Main", {"universe": ["AAA"]}, force=False
            )
            forced = scheduler.execute_account_rebalance(
                "Main", {"universe": ["AAA"]}, force=True
            )

        self.assertEqual(scheduled["stage"], "execution_window")
        self.assertTrue(forced["success"])
        self.assertEqual(oms_calls, ["submitted"])

    def test_failed_sync_never_calculates_or_orders(self):
        events = []
        with tempfile.TemporaryDirectory() as tmp, \
                patch("backend.data.fetcher.sync_and_validate_live_data", return_value={
                    "passed": False, "errors": ["stale"],
                }), \
                patch("backend.execution.tracker.LiveTracker", self._tracker(events)), \
                patch.object(scheduler, "_is_nyse_session_today", return_value=True), \
                patch.object(scheduler, "_stop_flag_path", return_value=Path(tmp) / "absent"), \
                patch("backend.execution.strategy.LiveStrategy") as strategy, \
                patch("backend.execution.oms.OrderManagementSystem") as oms:
            result = scheduler.execute_account_rebalance("Main", {"universe": ["AAA"]})
        self.assertEqual(result["stage"], "data_quality")
        strategy.assert_not_called()
        oms.assert_not_called()

    def test_stop_flag_blocks_before_sync(self):
        events = []
        with tempfile.TemporaryDirectory() as tmp:
            stop = Path(tmp) / "stop.flag"
            stop.write_text("stop")
            with patch("backend.execution.tracker.LiveTracker", self._tracker(events)), \
                    patch.object(scheduler, "_stop_flag_path", return_value=stop), \
                    patch("backend.data.fetcher.sync_and_validate_live_data") as sync:
                result = scheduler.execute_account_rebalance("Main", {"universe": ["AAA"]})
            self.assertEqual(result["stage"], "stop_flag")
            sync.assert_not_called()

    def test_claude_sleeve_factor_weights_dict(self):
        weights = scheduler._configured_factor_weights({
            "sleeves": [{
                "alloc": 1.0,
                "factors": ["Momentum", "Reversion"],
                "weights": {"Momentum": 0.7, "Reversion": 0.3},
            }]
        })
        self.assertEqual(weights, {"Momentum": 0.7, "Reversion": 0.3})

    def test_scheduled_window_is_new_york_time(self):
        self.assertTrue(scheduler._inside_scheduled_window(
            datetime(2026, 7, 14, 9, 35, tzinfo=scheduler.MARKET_TZ)
        ))
        self.assertFalse(scheduler._inside_scheduled_window(
            datetime(2026, 7, 14, 9, 29, tzinfo=scheduler.MARKET_TZ)
        ))
        self.assertFalse(scheduler._inside_scheduled_window(
            datetime(2026, 7, 14, 9, 45, tzinfo=scheduler.MARKET_TZ)
        ))

    def test_daily_risk_scales_actual_drifted_broker_weights(self):
        events = []
        captured = {}
        risk_universe = []

        class Adapter:
            def __init__(self, account):
                pass

            def get_account(self):
                return {
                    "equity": 100.0, "long_market_value": 100.0,
                    "short_market_value": 0.0, "status": "ACTIVE",
                    "account_blocked": False, "trading_blocked": False,
                }

            def get_positions(self):
                return [
                    {"symbol": "AAA", "market_value": 60.0},
                    {"symbol": "BBB", "market_value": 40.0},
                ]

        class OMS:
            def __init__(self, account):
                self.last_summary = {"status": "completed"}

            def generate_and_execute_orders(self, weights, *args, **kwargs):
                captured.update(weights)
                return []

        risk_state = {
            "active": True, "desired_leverage": 0.75,
            "transition": None, "in_market": False,
        }
        def evaluate_risk(account, runtime_config, as_of):
            risk_universe.extend(runtime_config["universe"])
            return risk_state

        with tempfile.TemporaryDirectory() as tmp, \
                patch("backend.data.fetcher.sync_and_validate_live_data", return_value={
                    "passed": True, "expected_as_of": "2026-07-13",
                    "effective_universe": ["AAA", "BBB"],
                }), \
                patch.object(scheduler, "_evaluate_daily_risk", side_effect=evaluate_risk), \
                patch.object(scheduler, "AlpacaAdapter", Adapter), \
                patch("backend.execution.oms.OrderManagementSystem", OMS), \
                patch("backend.execution.tracker.LiveTracker", self._tracker(events)), \
                patch.object(scheduler, "record_broker_observation", return_value={}), \
                patch.object(scheduler, "_is_nyse_session_today", return_value=True), \
                patch.object(scheduler, "_inside_scheduled_window", return_value=True), \
                patch.object(scheduler, "_stop_flag_path", return_value=Path(tmp) / "absent"):
            result = scheduler.execute_daily_risk_overlay(
                "Main", {"universe": ["AAA", "BBB", "OLD"], "leverage": 1.5,
                         "risk_management": {"regime_mode": "throttle"}}
            )
        self.assertEqual(result["status"], "completed")
        self.assertAlmostEqual(captured["AAA"], 0.45)
        self.assertAlmostEqual(captured["BBB"], 0.30)
        self.assertEqual(risk_universe, ["AAA", "BBB"])

    def test_daily_risk_nan_position_weight_never_reaches_oms(self):
        events = []

        class Adapter:
            def __init__(self, account):
                pass

            def get_account(self):
                return {
                    "equity": 100.0, "long_market_value": 100.0,
                    "short_market_value": 0.0, "status": "ACTIVE",
                    "account_blocked": False, "trading_blocked": False,
                }

            def get_positions(self):
                return [{"symbol": "AAA", "market_value": float("nan")}]

        with tempfile.TemporaryDirectory() as tmp, \
                patch("backend.data.fetcher.sync_and_validate_live_data", return_value={
                    "passed": True, "expected_as_of": "2026-07-13",
                    "effective_universe": ["AAA"],
                }), \
                patch.object(scheduler, "_evaluate_daily_risk", return_value={
                    "active": True, "desired_leverage": 0.75,
                    "transition": None, "in_market": False,
                }), \
                patch.object(scheduler, "AlpacaAdapter", Adapter), \
                patch("backend.execution.oms.OrderManagementSystem") as oms, \
                patch("backend.execution.tracker.LiveTracker", self._tracker(events)), \
                patch.object(scheduler, "_is_nyse_session_today", return_value=True), \
                patch.object(scheduler, "_stop_flag_path", return_value=Path(tmp) / "absent"):
            result = scheduler.execute_daily_risk_overlay(
                "Main", {"universe": ["AAA"], "leverage": 1.0,
                         "risk_management": {"regime_mode": "throttle"}}
            )

        self.assertEqual(result["stage"], "risk_positions")
        self.assertIn("weight", result["error"])
        oms.assert_not_called()

    def test_daily_risk_elapsed_execution_window_never_reaches_oms(self):
        events = []

        class Adapter:
            def __init__(self, account):
                pass

            def get_account(self):
                return {
                    "equity": 100.0, "long_market_value": 100.0,
                    "short_market_value": 0.0, "status": "ACTIVE",
                    "account_blocked": False, "trading_blocked": False,
                }

            def get_positions(self):
                return [{"symbol": "AAA", "market_value": 100.0}]

        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(scheduler, "DAILY_LOCK_DIR", tmp), \
                patch("backend.data.fetcher.sync_and_validate_live_data", return_value={
                    "passed": True, "expected_as_of": "2026-07-13",
                    "effective_universe": ["AAA"],
                }), \
                patch.object(scheduler, "_evaluate_daily_risk", return_value={
                    "active": True, "desired_leverage": 0.5,
                    "transition": None, "in_market": False,
                }), \
                patch.object(scheduler, "AlpacaAdapter", Adapter), \
                patch("backend.execution.oms.OrderManagementSystem") as oms, \
                patch("backend.execution.tracker.LiveTracker", self._tracker(events)), \
                patch.object(scheduler, "_inside_scheduled_window", return_value=False), \
                patch.object(scheduler, "_is_nyse_session_today", return_value=True), \
                patch.object(scheduler, "_stop_flag_path", return_value=Path(tmp) / "absent"):
            result = scheduler.execute_daily_risk_overlay(
                "Main", {"universe": ["AAA"], "leverage": 1.0,
                         "risk_management": {"regime_mode": "throttle"}}
            )

        self.assertEqual(result["stage"], "execution_window")
        oms.assert_not_called()

    def test_stateful_regime_exits_at_200ema_and_reenters_at_20ema(self):
        base = datetime(2025, 11, 25)
        first_values = [100.0] * 209 + [50.0] * 21
        second_values = first_values + [60.0]

        def gauge_frame(values):
            return pl.DataFrame({
                "timestamp": [base + timedelta(days=i) for i in range(len(values))],
                "symbol": ["QQQ"] * len(values),
                "close": values,
            }).lazy()

        class Store:
            call = 0

            @staticmethod
            def load(symbols, start, end):
                Store.call += 1
                return gauge_frame(first_values if Store.call == 1 else second_values)

        config = {
            "universe": ["AAA"], "leverage": 1.5,
            "risk_management": {"regime_mode": "cash"},
        }
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(scheduler, "DAILY_LOCK_DIR", tmp), \
                patch("backend.data.store.load", side_effect=Store.load):
            exited = scheduler._evaluate_daily_risk("Main", config, "2026-07-13")
            reentered = scheduler._evaluate_daily_risk("Main", config, "2026-07-14")
        self.assertEqual(exited["transition"], "entered_risk_off")
        self.assertEqual(exited["desired_leverage"], 0.0)
        self.assertEqual(reentered["transition"], "reentered_risk_on")
        self.assertEqual(reentered["desired_leverage"], 1.5)
        self.assertLess(reentered["gauge_close"], reentered["ema_slow_200"])
        self.assertGreater(reentered["gauge_close"], reentered["ema_fast_20"])


if __name__ == "__main__":
    unittest.main()
