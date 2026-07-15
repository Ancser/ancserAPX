import unittest

import pandas as pd

from backend.alpha.neutralization import SECTOR_MAP
from backend.alpha.portfolio import (
    combined_target_weights,
    sector_balanced_symbols,
    sector_balanced_weights,
)


class SectorBalanceTests(unittest.TestCase):
    def setUp(self):
        self.scores = pd.Series({
            "NVDA": 12.0,
            "AAPL": 11.0,
            "MSFT": 10.0,
            "AMD": 9.0,
            "AVGO": 8.0,
            "ORCL": 7.0,
            "XOM": 6.0,
            "CVX": 5.0,
            "LLY": 4.0,
            "JNJ": 3.0,
            "JPM": 2.0,
            "BAC": 1.0,
        })

    def test_selection_uses_near_equal_sector_slots(self):
        selected = sector_balanced_symbols(self.scores, 8)
        counts = {}
        for symbol in selected:
            sector = SECTOR_MAP[symbol]
            counts[sector] = counts.get(sector, 0) + 1

        self.assertEqual(len(selected), 8)
        self.assertEqual(counts, {
            "Technology": 2,
            "Energy": 2,
            "Healthcare": 2,
            "Financials": 2,
        })

    def test_weights_equalize_sector_budgets(self):
        weights = sector_balanced_weights(self.scores, 8)
        exposure = {}
        for symbol, weight in weights.items():
            sector = SECTOR_MAP[symbol]
            exposure[sector] = exposure.get(sector, 0.0) + weight

        self.assertAlmostEqual(sum(weights.values()), 1.0)
        for value in exposure.values():
            self.assertAlmostEqual(value, 0.25)

    def test_combined_portfolio_preserves_requested_leverage(self):
        sleeves = [{
            "name": "Core",
            "alloc": 1.0,
            "factors": ["Momentum"],
            "weights": {"Momentum": 1.0},
            "winner_lock": False,
        }]
        weights, _ = combined_target_weights(
            sleeves=sleeves,
            factor_values={"Momentum": self.scores},
            price=pd.Series(100.0, index=self.scores.index),
            state={},
            top_n=8,
            lock_rules={},
            leverage=1.5,
            sector_balance=True,
        )

        self.assertAlmostEqual(sum(weights.values()), 1.5)
        exposure = {}
        for symbol, weight in weights.items():
            sector = SECTOR_MAP[symbol]
            exposure[sector] = exposure.get(sector, 0.0) + weight
        for value in exposure.values():
            self.assertAlmostEqual(value, 0.375)


if __name__ == "__main__":
    unittest.main()
