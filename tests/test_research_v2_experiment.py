import unittest

import pandas as pd

from research_v2.experiment import make_signal_map


class ExperimentTests(unittest.TestCase):
    def test_signal_cadence_and_liquidation(self):
        dates = pd.bdate_range("2024-01-01", periods=6)
        frame = pd.DataFrame([
            {"timestamp": date, "symbol": symbol, "score": score}
            for date in dates for symbol, score in [("A", 1.0), ("B", 0.0)]
        ])
        signals = make_signal_map(
            frame,
            score_column="score",
            eligible_symbols=["A", "B"],
            rebalance_days=2,
            liquidate_at_end=True,
        )
        self.assertIn(dates[0], signals)
        self.assertIn(dates[2], signals)
        self.assertEqual(signals[dates[-1]], {})


if __name__ == "__main__":
    unittest.main()
