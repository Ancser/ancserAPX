"""Persistent live-state and append-only audit tracking.

``tracker_<account>.json`` remains a convenient snapshot history for the UI.
``live_audit_<account>.jsonl`` is the forensic event stream: sync, calculation,
orders, fills/errors, and account observations are appended and never replaced.
"""

import json
import math
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

    @staticmethod
    def _iso_date(value: Any, field_name: str) -> str:
        if isinstance(value, datetime):
            return value.date().isoformat()
        raw = str(value or "").strip()
        try:
            return datetime.strptime(raw[:10], "%Y-%m-%d").date().isoformat()
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field_name} must contain a YYYY-MM-DD date") from exc

    @classmethod
    def _normalize_performance_baseline(
        cls,
        history: list,
        current_date: str,
        current_equity: float,
        performance_baseline: Optional[Dict],
    ) -> Dict[str, Any]:
        if performance_baseline is None:
            for row in history:
                try:
                    row_equity = float(row.get("equity", 0))
                    row_date = cls._iso_date(row.get("date"), "tracked baseline date")
                except (AttributeError, TypeError, ValueError):
                    continue
                if math.isfinite(row_equity) and row_equity > 0:
                    return {"date": row_date, "equity": row_equity}
            return {"date": current_date, "equity": current_equity}

        if not isinstance(performance_baseline, dict):
            raise ValueError("performance_baseline must be a mapping with date and equity")
        baseline_date = cls._iso_date(
            performance_baseline.get("date"), "performance_baseline.date"
        )
        try:
            baseline_equity = float(performance_baseline.get("equity"))
        except (TypeError, ValueError) as exc:
            raise ValueError("performance_baseline.equity must be numeric") from exc
        if not math.isfinite(baseline_equity) or baseline_equity <= 0:
            raise ValueError("performance_baseline.equity must be positive and finite")
        if baseline_date > current_date:
            raise ValueError("performance_baseline.date cannot be after the observation date")
        return {"date": baseline_date, "equity": baseline_equity}

    @classmethod
    def _normalize_cash_activities(cls, cash_activities: list) -> list:
        if not isinstance(cash_activities, list):
            raise ValueError("cash_activities must be a list")

        normalized = []
        anonymous_counts: Dict[str, int] = {}
        by_id: Dict[str, Dict] = {}
        for activity in cash_activities:
            if not isinstance(activity, dict):
                raise ValueError("each cash activity must be a mapping")
            activity_date = cls._iso_date(
                activity.get("date") or activity.get("time"), "cash activity date"
            )
            raw_amount = activity.get("amount")
            if raw_amount is None:
                raw_amount = activity.get("net_amount")
            try:
                amount = float(raw_amount)
            except (TypeError, ValueError) as exc:
                raise ValueError("cash activity amount/net_amount must be numeric") from exc
            if not math.isfinite(amount):
                raise ValueError("cash activity amount/net_amount must be finite")

            activity_type = str(activity.get("activity_type") or "").strip().upper()
            if activity_type == "CSD":
                amount = abs(amount)
            elif activity_type == "CSW":
                amount = -abs(amount)
            elif activity_type:
                raise ValueError(
                    f"unsupported cash activity type {activity_type!r}; expected CSD or CSW"
                )
            elif amount > 0:
                activity_type = "CSD"
            elif amount < 0:
                activity_type = "CSW"
            else:
                raise ValueError("zero cash activity without CSD/CSW type is ambiguous")

            activity_id = str(activity.get("id") or "").strip()
            if not activity_id:
                base = "|".join([
                    activity_type,
                    activity_date,
                    str(activity.get("time") or ""),
                    f"{amount:.10f}",
                ])
                occurrence = anonymous_counts.get(base, 0)
                anonymous_counts[base] = occurrence + 1
                activity_id = f"anonymous:{base}:{occurrence}"

            activity_time = str(activity.get("time") or activity_date)
            time_precision = str(activity.get("time_precision") or "").strip().lower()
            if not time_precision:
                time_precision = "timestamp" if "T" in activity_time else "date_only"
            if time_precision not in {"timestamp", "date_only"}:
                raise ValueError("cash activity time_precision must be timestamp or date_only")
            row = {
                "id": activity_id,
                "activity_type": activity_type,
                "date": activity_date,
                "time": activity_time,
                "time_precision": time_precision,
                "amount": amount,
            }
            duplicate = by_id.get(activity_id)
            if duplicate is not None:
                if duplicate != row:
                    raise ValueError(
                        f"cash activity id {activity_id!r} has conflicting values"
                    )
                continue
            by_id[activity_id] = row
            normalized.append(row)
        return sorted(normalized, key=lambda row: (row["date"], row["time"], row["id"]))

    @classmethod
    def _cash_flow_adjusted_performance(
        cls,
        history: list,
        date_str: str,
        equity: float,
        cash_activities: list,
        performance_baseline: Optional[Dict],
    ) -> Dict[str, Any]:
        """Link observation-period returns after removing external cash flows."""
        current_date = cls._iso_date(date_str, "observation date")
        try:
            current_equity = float(equity)
        except (TypeError, ValueError) as exc:
            raise ValueError("equity must be numeric") from exc
        if not math.isfinite(current_equity) or current_equity < 0:
            raise ValueError("equity must be non-negative and finite")

        baseline = cls._normalize_performance_baseline(
            history, current_date, current_equity, performance_baseline
        )
        baseline_date = baseline["date"]
        normalized_flows = cls._normalize_cash_activities(cash_activities)
        baseline_day_flows = [
            row for row in normalized_flows if row["date"] == baseline_date
        ]
        if baseline_day_flows:
            raise ValueError(
                "cash activity on the performance baseline date is timing-ambiguous; "
                "choose a baseline after that date or supply an independently reconciled baseline"
            )
        flows = [
            row for row in normalized_flows
            if baseline_date < row["date"] <= current_date
        ]
        flow_by_id = {row["id"]: row for row in flows}
        return_is_estimate = bool(flows)
        timing_precision = (
            "timestamp"
            if flows and all(row["time_precision"] == "timestamp" for row in flows)
            else "date_only"
            if flows
            else "not_applicable"
        )

        observations = []
        for sequence, row in enumerate(history):
            if not isinstance(row, dict):
                continue
            try:
                observation_date = cls._iso_date(row.get("date"), "tracked observation date")
                observation_equity = float(row.get("equity"))
            except (TypeError, ValueError):
                continue
            if (
                observation_date <= baseline_date
                or observation_date > current_date
                or not math.isfinite(observation_equity)
                or observation_equity < 0
            ):
                continue
            observations.append({
                "date": observation_date,
                "equity": observation_equity,
                "sequence": sequence,
                "row": row,
                "is_current": False,
            })
        observations.append({
            "date": current_date,
            "equity": current_equity,
            "sequence": len(history),
            "row": {},
            "is_current": True,
        })
        observations.sort(key=lambda row: (row["date"], row["sequence"]))

        # Preserve the original observation assignment once an activity has
        # been recorded.  This is what makes a same-day retry idempotent.
        assigned: Dict[str, int] = {}
        for observation_index, observation in enumerate(observations):
            for activity_id in observation["row"].get("cash_flow_activity_ids", []) or []:
                activity_id = str(activity_id)
                if activity_id not in flow_by_id:
                    continue
                previous_index = assigned.get(activity_id)
                if previous_index is not None and previous_index != observation_index:
                    raise ValueError(
                        f"cash activity {activity_id!r} was assigned to multiple observations"
                    )
                if flow_by_id[activity_id]["date"] > observation["date"]:
                    raise ValueError(
                        f"cash activity {activity_id!r} postdates its recorded observation"
                    )
                assigned[activity_id] = observation_index

        # A newly discovered flow dated today attaches to the latest observation:
        # it may have posted after an earlier same-day snapshot.  A historical
        # flow attaches to the first snapshot on/after its date instead.  This
        # matters when old, unadjusted tracker rows are migrated: a deposit that
        # was already present in the first broker snapshot must not be applied to
        # a later same-day retry and create two artificial return periods.
        for flow in flows:
            if flow["id"] in assigned:
                continue
            candidate_dates = [
                observation["date"]
                for observation in observations
                if observation["date"] >= flow["date"]
            ]
            if not candidate_dates:
                continue
            first_date = min(candidate_dates)
            eligible_indices = [
                index
                for index, observation in enumerate(observations)
                if observation["date"] == first_date
            ]
            assigned[flow["id"]] = (
                max(eligible_indices)
                if first_date == current_date
                else min(eligible_indices)
            )

        flows_by_observation: Dict[int, list] = {}
        for activity_id, observation_index in assigned.items():
            flows_by_observation.setdefault(observation_index, []).append(
                flow_by_id[activity_id]
            )

        previous_equity = float(baseline["equity"])
        linked_growth = 1.0
        cumulative_cash_flow = 0.0
        current_result: Optional[Dict[str, Any]] = None
        for observation_index, observation in enumerate(observations):
            if previous_equity <= 0:
                raise ValueError(
                    "cannot link returns after an observation with zero equity"
                )
            observation_flows = sorted(
                flows_by_observation.get(observation_index, []),
                key=lambda row: (row["date"], row["time"], row["id"]),
            )
            net_cash_flow = sum(row["amount"] for row in observation_flows)
            adjusted_pnl = observation["equity"] - previous_equity - net_cash_flow
            period_return = adjusted_pnl / previous_equity
            linked_growth *= 1.0 + period_return
            cumulative_cash_flow += net_cash_flow
            if observation["is_current"]:
                current_result = {
                    "previous_equity": previous_equity,
                    "observation_pnl": adjusted_pnl,
                    "observation_return": period_return,
                    "day_pnl": adjusted_pnl,
                    "period_return": period_return,
                    "linked_return": linked_growth - 1.0,
                    "net_cash_flow": net_cash_flow,
                    "cumulative_cash_flow": cumulative_cash_flow,
                    "cash_flow_activity_ids": [row["id"] for row in observation_flows],
                    "cash_flow_activities": [
                        {
                            "id": row["id"],
                            "activity_type": row["activity_type"],
                            "date": row["date"],
                            "time": row["time"],
                            "time_precision": row["time_precision"],
                            "amount": round(row["amount"], 2),
                        }
                        for row in observation_flows
                    ],
                    "performance_baseline": {
                        "date": baseline_date,
                        "equity": round(float(baseline["equity"]), 2),
                    },
                    "return_is_estimate": return_is_estimate,
                    "cash_activity_timestamp_precision": timing_precision,
                    "cash_flow_timing_assumption": (
                        "end_of_observation_period" if flows else "not_applicable"
                    ),
                }
            previous_equity = observation["equity"]

        if current_result is None:
            raise RuntimeError("current observation was not included in return calculation")
        return current_result

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
        cash_activities: Optional[list] = None,
        performance_baseline: Optional[Dict] = None,
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
        cash_flow_performance = None
        if cash_activities is not None or performance_baseline is not None:
            cash_flow_performance = self._cash_flow_adjusted_performance(
                history=history,
                date_str=date_str,
                equity=equity,
                cash_activities=cash_activities or [],
                performance_baseline=performance_baseline,
            )
            last_equity = cash_flow_performance["previous_equity"]
            day_pnl = cash_flow_performance["day_pnl"]
            day_pnl_pct = cash_flow_performance["period_return"]
            total_pnl_pct = cash_flow_performance["linked_return"]
            baseline_equity = cash_flow_performance["performance_baseline"]["equity"]
        else:
            last_equity = float(account_snapshot.get("last_equity", 0) or 0)
            if day_pnl is None:
                day_pnl = equity - last_equity if last_equity > 0 else 0.0
            day_pnl_pct = day_pnl / last_equity if last_equity > 0 else 0.0
            baseline_equity = self._baseline_equity(history, equity)
            if total_pnl_pct is None:
                total_pnl_pct = equity / baseline_equity - 1.0 if baseline_equity > 0 else 0.0

        # ``day_pnl`` is retained as a compatibility alias for the change since
        # the preceding tracker observation.  It is not necessarily Alpaca's
        # calendar-day P&L when multiple observations exist on one day.  When
        # the broker's prior-close equity and complete cash ledger are present,
        # expose the cash-adjusted calendar-day dollar P&L separately.
        broker_calendar_day_pnl = None
        broker_calendar_day_cash_flow = None
        if cash_activities is not None:
            try:
                broker_last_equity = float(account_snapshot.get("last_equity"))
            except (TypeError, ValueError):
                broker_last_equity = 0.0
            if math.isfinite(broker_last_equity) and broker_last_equity > 0:
                current_date = self._iso_date(date_str, "observation date")
                normalized_cash = self._normalize_cash_activities(cash_activities)
                broker_calendar_day_cash_flow = sum(
                    row["amount"] for row in normalized_cash
                    if row["date"] == current_date
                )
                broker_calendar_day_pnl = (
                    float(equity)
                    - broker_last_equity
                    - broker_calendar_day_cash_flow
                )

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
            "observation_period_start_equity": round(last_equity, 2),
            "observation_pnl": round(day_pnl, 2),
            "observation_return": round(day_pnl_pct, 6),
            "day_pnl_basis": (
                "tracker_observation_period_legacy_alias"
                if cash_flow_performance is not None
                else "legacy_or_broker_last_equity"
            ),
            "total_pnl_pct": round(total_pnl_pct, 6),
            "baseline_equity": round(baseline_equity, 2),
            "total_pnl_basis": (
                "cash_flow_adjusted_linked_return"
                if cash_flow_performance is not None
                else "first_tracked_equity_unadjusted_for_cash_flows"
            ),
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

        if broker_calendar_day_pnl is not None:
            record.update({
                "broker_calendar_day_cash_adjusted_pnl": round(
                    broker_calendar_day_pnl, 2
                ),
                "broker_calendar_day_external_cash_flow": round(
                    broker_calendar_day_cash_flow or 0.0, 2
                ),
                "broker_calendar_day_pnl_basis": (
                    "equity_minus_broker_last_equity_minus_same_day_external_cash_flow"
                ),
            })

        if cash_flow_performance is not None:
            record.update({
                "cash_flow_adjusted": True,
                "cash_flow_adjustment": round(
                    cash_flow_performance["net_cash_flow"], 2
                ),
                "net_cash_flow": round(cash_flow_performance["net_cash_flow"], 2),
                "net_cash_flow_since_baseline": round(
                    cash_flow_performance["cumulative_cash_flow"], 2
                ),
                "cash_flow_activity_ids": cash_flow_performance[
                    "cash_flow_activity_ids"
                ],
                "cash_flow_activities": cash_flow_performance[
                    "cash_flow_activities"
                ],
                "cash_flow_adjusted_day_pnl": round(
                    cash_flow_performance["day_pnl"], 2
                ),
                "cash_flow_adjusted_observation_pnl": round(
                    cash_flow_performance["observation_pnl"], 2
                ),
                "cash_flow_adjusted_period_return": round(
                    cash_flow_performance["period_return"], 6
                ),
                "cash_flow_adjusted_linked_return": round(
                    cash_flow_performance["linked_return"], 6
                ),
                "performance_baseline": cash_flow_performance[
                    "performance_baseline"
                ],
                "return_calculation": {
                    "method": "cash_flow_adjusted_linked_return",
                    "period_formula": (
                        "(ending_equity - starting_equity - net_cash_flow) "
                        "/ starting_equity"
                    ),
                    "cash_flow_timing": cash_flow_performance[
                        "cash_flow_timing_assumption"
                    ],
                    "cash_activity_timestamp_precision": cash_flow_performance[
                        "cash_activity_timestamp_precision"
                    ],
                    "valuation_observed_at_cash_flow": False,
                    "is_estimate": cash_flow_performance["return_is_estimate"],
                    "estimate_reason": (
                        "No account valuation immediately before each external cash flow"
                        if cash_flow_performance["return_is_estimate"]
                        else None
                    ),
                },
            })

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
