"""Persistent live-state and append-only audit tracking.

``tracker_<account>.json`` remains a convenient snapshot history for the UI.
``live_audit_<account>.jsonl`` is the forensic event stream: sync, calculation,
orders, fills/errors, and account observations are appended and never replaced.
"""

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


class LiveTracker:
    def __init__(self, account_name: str = "Main"):
        self.account_name = account_name
        self.history_path = f"logs/tracker_{account_name}.json"
        self.audit_path = f"logs/live_audit_{account_name}.jsonl"

    @staticmethod
    def _write_json_atomic(path: str, value: Any) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temp = target.with_suffix(target.suffix + ".tmp")
        temp.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")
        os.replace(temp, target)

    def record_event(
        self,
        event_type: str,
        status: str,
        *,
        run_id: Optional[str] = None,
        details: Optional[Dict] = None,
    ) -> Dict:
        """Append one durable, timestamped live event and return it."""
        os.makedirs("logs", exist_ok=True)
        event = {
            "event_id": uuid.uuid4().hex,
            "run_id": run_id,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "account": self.account_name,
            "event_type": str(event_type),
            "status": str(status),
            "details": details or {},
        }
        with open(self.audit_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, default=str, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        return event

    @staticmethod
    def _baseline_equity(history: list, fallback: float) -> float:
        for row in history:
            try:
                value = float(row.get("equity", 0))
                if value > 0:
                    return value
            except Exception:
                continue
        return fallback

    def record_daily_state(
        self,
        date_str: str,
        equity: float,
        day_pnl: Optional[float],
        total_pnl_pct: Optional[float],
        allocations: Dict[str, float],
        factors: list,
        target_scalar: float = 1.0,
        account_snapshot: Optional[Dict] = None,
        factor_weights: Optional[Dict[str, float]] = None,
        as_of_date: Optional[str] = None,
        data_sync: Optional[Dict] = None,
        order_summary: Optional[Dict] = None,
        actual_allocations: Optional[Dict[str, float]] = None,
        run_id: Optional[str] = None,
    ) -> Dict:
        os.makedirs("logs", exist_ok=True)
        history = []
        if os.path.exists(self.history_path):
            try:
                with open(self.history_path, encoding="utf-8") as handle:
                    history = json.load(handle)
            except Exception:
                pass

        account_snapshot = account_snapshot or {}
        last_equity = float(account_snapshot.get("last_equity", 0) or 0)
        if day_pnl is None:
            day_pnl = equity - last_equity if last_equity > 0 else 0.0
        day_pnl_pct = day_pnl / last_equity if last_equity > 0 else 0.0
        baseline_equity = self._baseline_equity(history, equity)
        if total_pnl_pct is None:
            total_pnl_pct = equity / baseline_equity - 1.0 if baseline_equity > 0 else 0.0

        gross_exposure = sum(abs(float(w)) for w in allocations.values())
        net_exposure = sum(float(w) for w in allocations.values())
        actual_gross = sum(abs(float(w)) for w in (actual_allocations or {}).values())
        actual_net = sum(float(w) for w in (actual_allocations or {}).values())
        record = {
            "record_id": uuid.uuid4().hex,
            "run_id": run_id,
            "date": date_str,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "equity": round(equity, 2),
            "last_equity": round(last_equity, 2),
            "day_pnl": round(day_pnl, 2),
            "day_pnl_pct": round(day_pnl_pct, 6),
            "total_pnl_pct": round(total_pnl_pct, 6),
            "baseline_equity": round(baseline_equity, 2),
            "total_pnl_basis": "first_tracked_equity_unadjusted_for_cash_flows",
            "allocations": {k: round(float(v), 6) for k, v in allocations.items()},
            "allocation_type": "target",
            "actual_allocations": {
                k: round(float(v), 6) for k, v in (actual_allocations or {}).items()
            },
            "factors": factors,
            "factor_weights": {
                k: round(float(v), 6) for k, v in (factor_weights or {}).items()
            },
            "vol_scalar": round(target_scalar, 6),
            "gross_exposure": round(gross_exposure, 6),
            "net_exposure": round(net_exposure, 6),
            "actual_gross_exposure": round(actual_gross, 6),
            "actual_net_exposure": round(actual_net, 6),
            "as_of_date": as_of_date,
            "account_snapshot": account_snapshot,
            "data_sync": data_sync,
            "order_summary": order_summary,
        }

        # Same-day retries/observations are evidence, so never replace them.
        history.append(record)
        self._write_json_atomic(self.history_path, history)
        self.record_event("account_state", "recorded", run_id=run_id, details=record)
        return record

    def get_history(self):
        if not os.path.exists(self.history_path):
            return []
        try:
            with open(self.history_path, encoding="utf-8") as handle:
                return json.load(handle)
        except Exception:
            return []

    def get_audit_events(self, limit: Optional[int] = None) -> list:
        if not os.path.exists(self.audit_path):
            return []
        events = []
        try:
            with open(self.audit_path, encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        events.append(json.loads(line))
        except Exception:
            return events[-limit:] if limit else events
        return events[-limit:] if limit else events
