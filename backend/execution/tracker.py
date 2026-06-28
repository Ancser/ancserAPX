"""Daily state tracker — logs equity, P&L, and allocations to JSON."""

import json, os
from datetime import datetime
from typing import Dict, Optional


class LiveTracker:
    def __init__(self, account_name: str = "Main"):
        self.account_name = account_name
        self.history_path = f"logs/tracker_{account_name}.json"

    def record_daily_state(
        self,
        date_str: str,
        equity: float,
        day_pnl: float,
        total_pnl_pct: float,
        allocations: Dict[str, float],
        factors: list,
        target_scalar: float = 1.0,
    ):
        os.makedirs("logs", exist_ok=True)
        history = []
        if os.path.exists(self.history_path):
            try:
                history = json.load(open(self.history_path))
            except Exception:
                pass

        record = {
            "date": date_str,
            "recorded_at": datetime.now().isoformat(),
            "equity": round(equity, 2),
            "day_pnl": round(day_pnl, 2),
            "total_pnl_pct": round(total_pnl_pct, 4),
            "allocations": {k: round(v, 4) for k, v in allocations.items()},
            "factors": factors,
            "vol_scalar": round(target_scalar, 4),
        }

        history = [h for h in history if h.get("date") != date_str]
        history.append(record)

        with open(self.history_path, "w") as f:
            json.dump(history, f, indent=2)

    def get_history(self):
        if not os.path.exists(self.history_path):
            return []
        try:
            return json.load(open(self.history_path))
        except Exception:
            return []
