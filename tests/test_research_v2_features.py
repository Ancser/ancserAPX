import unittest
from datetime import datetime, timedelta

import numpy as np
import polars as pl

from research_v2.features import build_feature_panel, validate_panel


def synthetic_panel(symbols=6, days=330):
    rows = []
    start = datetime(2020, 1, 1)
    for j in range(days):
        ts = start + timedelta(days=j)
        for i in range(symbols):
            close = 50.0 + i + j * (0.02 + i * 0.001)
            rows.append({
                "timestamp": ts,
                "symbol": f"S{i}",
                "open": close * 0.999,
                "high": close * 1.01,
                "low": close * 0.99,
                "close": close,
                "volume": 1_000_000.0 + i * 1000,
                "vwap": close * 0.9995,
                "trade_count": 1000,
            })
    return pl.DataFrame(rows)


class FeatureTests(unittest.TestCase):
    def test_sparse_session_removed(self):
        raw = synthetic_panel(symbols=6, days=10)
        bad_day = raw["timestamp"].unique().sort()[5]
        raw = raw.filter(~((pl.col("timestamp") == bad_day) & (pl.col("symbol") != "S0")))
        clean, report = validate_panel(
            raw,
            start_date="2020-01-01",
            end_date="2020-12-31",
            min_cross_section=4,
        )
        self.assertNotIn(bad_day, clean["timestamp"].unique().to_list())
        self.assertEqual(len(report["dropped_sparse_sessions"]), 1)

    def test_signal_execution_clock_and_no_feature_future_shift(self):
        result = build_feature_panel(
            synthetic_panel(),
            start_date="2020-01-01",
            end_date="2021-12-31",
            min_cross_section=4,
            min_symbol_history=252,
            label_horizon=5,
        )
        eligible = result.panel.filter(pl.col("model_eligible"))
        self.assertGreater(eligible.height, 0)
        delta = (
            eligible.select([
                pl.col("timestamp"), pl.col("execution_timestamp")
            ]).with_columns(
                (pl.col("execution_timestamp") > pl.col("timestamp")).alias("after")
            )["after"].all()
        )
        self.assertTrue(delta)
        self.assertTrue(np.isclose(eligible.group_by("timestamp").agg(pl.col("sample_weight").sum())["sample_weight"].median(), 1.0))

    def test_future_price_mutation_does_not_change_past_features(self):
        raw = synthetic_panel()
        result_a = build_feature_panel(raw, start_date="2020-01-01", end_date="2021-12-31", min_cross_section=4)
        cutoff = raw["timestamp"].unique().sort()[280]
        mutated = raw.with_columns([
            pl.when(pl.col("timestamp") > cutoff).then(pl.col(c) * 3).otherwise(pl.col(c)).alias(c)
            for c in ["open", "high", "low", "close", "vwap"]
        ])
        result_b = build_feature_panel(mutated, start_date="2020-01-01", end_date="2021-12-31", min_cross_section=4)
        cols = ["timestamp", "symbol"] + list(result_a.feature_columns)
        a = result_a.panel.filter(pl.col("timestamp") <= cutoff).select(cols)
        b = result_b.panel.filter(pl.col("timestamp") <= cutoff).select(cols)
        self.assertTrue(a.equals(b, null_equal=True))


if __name__ == "__main__":
    unittest.main()
