"""
Multiplicative Weights Update (MWU) engine — ported from ancserFX unchanged.
"""

import numpy as np
import polars as pl
from typing import Dict, List


class MWUEngine:
    def __init__(
        self,
        factor_names: List[str],
        learning_rate: float = 0.1,
        window: int = 252,
        min_weight: float = 0.05,
        max_weight: float = 0.60,
    ):
        self.factor_names = factor_names
        self.learning_rate = learning_rate
        self.window = window
        self.min_weight = min_weight
        self.max_weight = max_weight
        n = len(factor_names)
        self.weights = {f: 1.0 / n for f in factor_names}
        self.history: List[Dict] = []
        self.ic_history: List[Dict] = []

    def update(self, date, factor_ics: Dict[str, float]) -> Dict[str, float]:
        current = self.weights.copy()
        new_weights: Dict[str, float] = {}
        total = 0.0

        self.ic_history.append({"date": date, **factor_ics})

        for f in self.factor_names:
            if f not in current:
                continue
            w_t = current[f]
            if f in factor_ics:
                ic = factor_ics[f]
                mult = (1.0 + self.learning_rate * abs(ic)) if ic > 0 else (1.0 - self.learning_rate * abs(ic))
                mult = max(0.1, mult)
                w_next = w_t * mult
            else:
                w_next = w_t
            new_weights[f] = w_next
            total += w_next

        # Normalize
        if total > 0:
            for f in new_weights:
                new_weights[f] /= total
        else:
            n = len(new_weights)
            new_weights = {f: 1.0 / n for f in new_weights}

        # Bound
        bounded = {f: max(self.min_weight, min(self.max_weight, w)) for f, w in new_weights.items()}
        total_b = sum(bounded.values())
        if total_b > 0:
            bounded = {f: w / total_b for f, w in bounded.items()}

        self.weights = bounded
        self.history.append({"date": str(date), **self.weights})
        return self.weights

    def get_history_df(self) -> pl.DataFrame:
        if not self.history:
            return pl.DataFrame()
        return pl.DataFrame(self.history)
