import unittest

import numpy as np
import pandas as pd

from research_v2.validation import (
    cross_sectional_rank,
    make_purged_walk_forward,
    newey_west_mean_stats,
)


class ValidationTests(unittest.TestCase):
    def test_fold_gaps_and_order(self):
        dates = pd.bdate_range("2020-01-01", periods=900)
        folds = make_purged_walk_forward(
            dates,
            train_days=504,
            validation_days=63,
            test_days=63,
            purge_days=5,
            embargo_days=5,
            step_days=63,
            label_horizon=5,
        )
        self.assertGreaterEqual(len(folds), 4)
        pos = {d: i for i, d in enumerate(dates)}
        for fold in folds:
            self.assertEqual(pos[fold.validation_start] - pos[fold.train_end] - 1, 5)
            self.assertEqual(pos[fold.test_start] - pos[fold.validation_end] - 1, 5)

    def test_short_purge_rejected(self):
        with self.assertRaisesRegex(ValueError, "label_horizon"):
            make_purged_walk_forward(pd.bdate_range("2020", periods=900), purge_days=4)

    def test_cross_sectional_rank_is_date_local(self):
        values = pd.Series([1.0, 2.0, 100.0, 200.0])
        dates = pd.Series(["2020-01-01", "2020-01-01", "2020-01-02", "2020-01-02"])
        rank = cross_sectional_rank(values, dates)
        self.assertTrue(np.allclose(rank.to_numpy(), [-0.5, 0.5, -0.5, 0.5]))

    def test_newey_west_constant_is_stable(self):
        stats = newey_west_mean_stats(pd.Series([0.02] * 20), max_lag=4)
        self.assertAlmostEqual(stats["mean"], 0.02)
        self.assertEqual(stats["nw_se"], 0.0)


if __name__ == "__main__":
    unittest.main()
