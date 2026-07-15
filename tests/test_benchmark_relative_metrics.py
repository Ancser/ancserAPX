import unittest

import numpy as np
import pandas as pd

from backend.backtest.engine import compute_benchmark_relative_metrics


class BenchmarkRelativeMetricsTests(unittest.TestCase):
    @staticmethod
    def _inputs(strategy_returns, benchmark_returns):
        dates = pd.bdate_range("2020-01-02", periods=len(strategy_returns) + 1)
        strategy_equity = 100.0 * np.cumprod(np.r_[1.0, 1.0 + np.asarray(strategy_returns)])
        benchmark_equity = 100.0 * np.cumprod(np.r_[1.0, 1.0 + np.asarray(benchmark_returns)])
        result = pd.DataFrame({"equity": strategy_equity}, index=dates)
        curve = [
            {"date": date.strftime("%Y-%m-%d"), "value": float(value)}
            for date, value in zip(dates, benchmark_equity)
        ]
        return result, curve

    def test_identical_strategy_has_unit_beta_and_no_active_return(self):
        returns = np.tile([0.01, -0.006, 0.004, -0.002], 200)
        result, curve = self._inputs(returns, returns)

        metrics = compute_benchmark_relative_metrics(result, curve)

        self.assertAlmostEqual(metrics["beta"], 1.0, places=3)
        self.assertAlmostEqual(metrics["alpha_pct_annual"], 0.0, places=2)
        self.assertAlmostEqual(metrics["excess_cagr_pct"], 0.0, places=2)
        self.assertAlmostEqual(metrics["upside_capture_pct"], 100.0, places=2)
        self.assertAlmostEqual(metrics["downside_capture_pct"], 100.0, places=2)
        self.assertIsNone(metrics["information_ratio"])

    def test_constant_daily_alpha_is_detected_and_wins_rolling_windows(self):
        benchmark = np.tile([0.008, -0.006, 0.003, -0.002], 250)
        strategy = benchmark + 0.0002
        result, curve = self._inputs(strategy, benchmark)

        metrics = compute_benchmark_relative_metrics(result, curve)

        self.assertAlmostEqual(metrics["beta"], 1.0, places=2)
        self.assertGreater(metrics["alpha_pct_annual"], 4.9)
        self.assertGreater(metrics["excess_cagr_pct"], 0.0)
        self.assertEqual(metrics["rolling_1y_win_rate_pct"], 100.0)
        self.assertEqual(metrics["rolling_3y_win_rate_pct"], 100.0)
        self.assertGreater(metrics["latest_1y_excess_pct"], 0.0)
        self.assertGreater(metrics["latest_3y_excess_pct"], 0.0)


if __name__ == "__main__":
    unittest.main()
