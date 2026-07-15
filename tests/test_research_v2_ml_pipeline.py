import unittest

import numpy as np
import pandas as pd

from research_v2.ml_pipeline import run_tabular_research
from research_v2.validation import make_purged_walk_forward


class MLPipelineTests(unittest.TestCase):
    def test_oos_predictions_are_unique_and_lockbox_is_later(self):
        rng = np.random.default_rng(7)
        dates = pd.bdate_range("2020-01-01", periods=120)
        rows = []
        for d in dates:
            for i in range(25):
                x1 = rng.normal()
                x2 = rng.normal()
                y = np.tanh(x1) + 0.1 * x2 + rng.normal(scale=0.2)
                rows.append({
                    "timestamp": d,
                    "execution_timestamp": d + pd.offsets.BDay(1),
                    "symbol": f"S{i:02d}",
                    "f1": x1,
                    "f2": x2,
                    "label_rank": y,
                    "label_residual": y,
                    "sample_weight": 1 / 25,
                    "score_production_claude1": x1,
                    "score_production_claude3": x1 + x2,
                    "score_momentum_ensemble_v2": x2,
                    "score_momentum_lowvol_v2": x1 - x2,
                })
        panel = pd.DataFrame(rows)
        folds = make_purged_walk_forward(
            dates[:90], train_days=30, validation_days=10, test_days=10,
            purge_days=2, embargo_days=2, step_days=10, label_horizon=2,
        )
        result = run_tabular_research(
            panel, ["f1", "f2"], folds,
            ridge_alphas=[1.0],
            gbdt_grid=[{"max_iter": 20, "max_leaf_nodes": 7, "min_samples_leaf": 20, "l2_regularization": 1.0}],
            random_seed=7,
            selection_end=str(dates[89].date()),
            lockbox_start=str(dates[90].date()),
            embargo_days=2,
            ensemble_shrinkage=0.5,
            ensemble_single_model_cap=0.75,
        )
        self.assertFalse(result.selection_predictions.duplicated(["timestamp", "symbol"]).any())
        self.assertTrue((result.lockbox_predictions["timestamp"] >= dates[90]).all())
        self.assertTrue((result.selection_predictions["fold_id"] != "LOCKBOX").all())


if __name__ == "__main__":
    unittest.main()
