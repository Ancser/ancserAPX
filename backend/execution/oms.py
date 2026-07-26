"""Order Management System with durable order/audit snapshots."""

import hashlib
import json
import logging
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from backend.data.alpaca_adapter import AlpacaAdapter
from backend.execution.tracker import LiveTracker

logger = logging.getLogger("backend.oms")


def _order_response(order) -> Dict:
    """Extract stable broker fields without depending on one Alpaca SDK version."""
    if order is None:
        return {}

    def _value(name):
        value = getattr(order, name, None)
        return value.value if hasattr(value, "value") else value

    return {
        "broker_order_id": str(_value("id")) if _value("id") is not None else None,
        "broker_status": str(_value("status")) if _value("status") is not None else "submitted",
        "filled_qty": float(_value("filled_qty") or 0),
        "filled_avg_price": float(_value("filled_avg_price")) if _value("filled_avg_price") else None,
        "created_at": str(_value("created_at")) if _value("created_at") else None,
        "updated_at": str(_value("updated_at")) if _value("updated_at") else None,
    }


class OrderManagementSystem:
    # The scheduled run now starts after the open, so normal market orders
    # should fill in seconds.  Keep the waits bounded and fail closed: a buy
    # phase must never rely on proceeds from a sell that is merely accepted.
    SELL_FILL_TIMEOUT_SECONDS = 90.0
    BUY_FILL_TIMEOUT_SECONDS = 90.0
    CANCEL_CONFIRM_TIMEOUT_SECONDS = 15.0
    OPEN_ORDER_CANCEL_TIMEOUT_SECONDS = 15.0
    ORDER_POLL_INTERVAL_SECONDS = 0.5
    BUYING_POWER_PRICE_BUFFER = 1.03

    def __init__(self, account_name: str = "Main"):
        self.account_name = account_name
        self.alpaca = AlpacaAdapter(account_name)
        self.tracker = LiveTracker(account_name)
        self.last_summary: Dict = {}

    def _build_order_plan(
        self,
        target_weights: Dict[str, float],
        equity: float,
        positions: List[Dict],
    ) -> Tuple[List[Dict], List[Dict], Dict[str, float]]:
        """Build deltas from a fresh broker snapshot.

        This helper is deliberately called again after the sell barrier.  The
        second plan therefore uses the buying power, equity, positions, and
        prices that actually exist after sells fill instead of stale pre-sell
        quantities.
        """
        current_qtys = {p["symbol"]: float(p["qty"]) for p in positions}
        current_prices = {p["symbol"]: float(p["current_price"]) for p in positions}

        all_symbols = set(current_qtys) | set(target_weights)
        missing = [s for s in all_symbols if s not in current_prices]
        if missing:
            current_prices.update(self.alpaca.get_latest_prices(missing))

        orders = []
        skipped = []
        for sym in sorted(all_symbols):
            current_qty = current_qtys.get(sym, 0.0)
            target_pct = float(target_weights.get(sym, 0.0) or 0.0)
            price = float(current_prices.get(sym, 0.0) or 0.0)
            if price <= 0:
                skipped.append({"symbol": sym, "reason": "missing_latest_price"})
                continue

            if target_pct == 0.0:
                if current_qty <= 0:
                    continue
                orders.append({
                    "client_event_id": uuid.uuid4().hex,
                    "symbol": sym, "side": "sell", "qty": current_qty,
                    "price": price, "target_qty": 0.0, "target_weight": 0.0,
                })
            else:
                target_qty = round((equity * abs(target_pct)) / price, 2)
                diff_qty = round(target_qty - current_qty, 2)
                if abs(diff_qty) * price < 10.0 or diff_qty == 0:
                    continue
                orders.append({
                    "client_event_id": uuid.uuid4().hex,
                    "symbol": sym,
                    "side": "buy" if diff_qty > 0 else "sell",
                    "qty": abs(diff_qty),
                    "price": price,
                    "target_qty": target_qty,
                    "target_weight": target_pct,
                })
        return orders, skipped, current_prices

    @staticmethod
    def _broker_status(value) -> str:
        if hasattr(value, "value"):
            value = value.value
        return str(value or "unknown").strip().lower()

    @staticmethod
    def _broker_market_value(account: Dict) -> float:
        return abs(float(account.get("long_market_value", 0) or 0)) + abs(
            float(account.get("short_market_value", 0) or 0)
        )

    def _positions_are_consistent(self, account: Dict, positions: List[Dict]) -> bool:
        """An empty position response is valid only when broker market value is flat.

        ``AlpacaAdapter.get_positions`` intentionally returns ``[]`` on an API
        error for display callers.  The OMS therefore must cross-check account
        market value before treating an empty list as an actual flat book.
        """
        return bool(positions) or self._broker_market_value(account) <= 1.0

    def _wait_for_open_orders_to_clear(self, *, run_id: str) -> None:
        """Confirm bulk-canceled orders are no longer live before replacements."""
        deadline = time.monotonic() + max(
            0.0, float(self.OPEN_ORDER_CANCEL_TIMEOUT_SECONDS)
        )
        first_poll = True
        last_error = None
        last_open_orders = []
        while first_poll or time.monotonic() < deadline:
            first_poll = False
            try:
                last_open_orders = self.alpaca.get_open_orders_strict(limit=500)
                last_error = None
                if not last_open_orders:
                    self.tracker.record_event(
                        "open_orders", "cancel_confirmed", run_id=run_id,
                        details={"remaining": 0},
                    )
                    return
            except Exception as exc:
                last_error = str(exc)
            if time.monotonic() < deadline:
                time.sleep(max(0.0, float(self.ORDER_POLL_INTERVAL_SECONDS)))

        details = {
            "remaining": last_open_orders,
            "last_error": last_error,
        }
        self.tracker.record_event(
            "open_orders", "cancel_unconfirmed", run_id=run_id, details=details
        )
        raise RuntimeError(
            "Could not confirm all existing broker orders were terminally canceled; "
            "replacement batch was not submitted"
        )

    def _wait_for_fills(
        self,
        submitted_orders: List[Dict],
        *,
        phase: str,
        timeout_seconds: float,
        run_id: str,
    ) -> Tuple[List[Dict], List[Dict]]:
        """Wait until every submitted order is terminal-filled.

        Accepted/new/partially-filled orders remain pending.  Rejected,
        canceled, expired, and timed-out orders fail the barrier.  A broker
        lookup error is retried only within the bounded deadline and can never
        be interpreted as a fill.
        """
        if not submitted_orders:
            return [], []

        terminal_failures = {
            "canceled", "expired", "rejected", "suspended", "stopped",
            "done_for_day", "calculated",
        }
        pending = {}
        filled = []
        failures = []
        for order in submitted_orders:
            broker_id = str(order.get("broker_order_id") or "")
            if not broker_id:
                failures.append({
                    **order,
                    "phase": f"{phase}_fill",
                    "submission_status": "unknown",
                    "error": "broker response did not include an order id",
                })
            else:
                pending[broker_id] = order

        deadline = time.monotonic() + max(0.0, float(timeout_seconds))
        last_errors: Dict[str, str] = {}
        while pending:
            for broker_id, order in list(pending.items()):
                try:
                    response = self.alpaca.get_order_by_id(broker_id)
                    status = self._broker_status(getattr(response, "status", None))
                    snapshot = {**order, **_order_response(response), "terminal_status": status}
                    if status == "filled":
                        expected_qty = float(order.get("qty", 0) or 0)
                        filled_qty = float(snapshot.get("filled_qty", 0) or 0)
                        if filled_qty + 1e-8 < expected_qty:
                            last_errors[broker_id] = (
                                f"filled status reported only {filled_qty} of {expected_qty}"
                            )
                            continue
                        filled.append(snapshot)
                        pending.pop(broker_id, None)
                    elif status in terminal_failures:
                        failures.append({
                            **snapshot,
                            "phase": f"{phase}_fill",
                            "submission_status": "failed",
                            "error": f"broker order reached terminal status {status}",
                        })
                        pending.pop(broker_id, None)
                except Exception as exc:
                    last_errors[broker_id] = str(exc)

            if not pending or time.monotonic() >= deadline:
                break
            time.sleep(max(0.0, float(self.ORDER_POLL_INTERVAL_SECONDS)))

        # Never return while a timed-out order is knowingly live. Request
        # cancellation, then poll for a terminal state. It may race to filled;
        # that is still authoritative and is handled as a fill. Anything whose
        # terminal state cannot be confirmed remains pending_unknown and blocks
        # every subsequent phase.
        if pending:
            cancel_errors: Dict[str, str] = {}
            for broker_id in list(pending):
                try:
                    self.alpaca.cancel_order_by_id(broker_id)
                    self.tracker.record_event(
                        "order_cancel", "requested", run_id=run_id,
                        details={"phase": phase, "broker_order_id": broker_id},
                    )
                except Exception as exc:
                    cancel_errors[broker_id] = str(exc)
                    self.tracker.record_event(
                        "order_cancel", "request_failed", run_id=run_id,
                        details={
                            "phase": phase, "broker_order_id": broker_id,
                            "error": str(exc),
                        },
                    )

            cancel_deadline = time.monotonic() + max(
                0.0, float(self.CANCEL_CONFIRM_TIMEOUT_SECONDS)
            )
            first_cancel_poll = True
            while pending and (first_cancel_poll or time.monotonic() < cancel_deadline):
                first_cancel_poll = False
                for broker_id, order in list(pending.items()):
                    try:
                        response = self.alpaca.get_order_by_id(broker_id)
                        status = self._broker_status(getattr(response, "status", None))
                        snapshot = {
                            **order, **_order_response(response),
                            "terminal_status": status,
                        }
                        if status == "filled":
                            expected_qty = float(order.get("qty", 0) or 0)
                            filled_qty = float(snapshot.get("filled_qty", 0) or 0)
                            if filled_qty + 1e-8 >= expected_qty:
                                filled.append(snapshot)
                                pending.pop(broker_id, None)
                        elif status in terminal_failures:
                            failures.append({
                                **snapshot,
                                "phase": f"{phase}_fill",
                                "submission_status": "canceled_after_timeout",
                                "error": (
                                    f"order timed out and reached terminal status {status}"
                                ),
                            })
                            pending.pop(broker_id, None)
                    except Exception as exc:
                        last_errors[broker_id] = str(exc)
                if pending and time.monotonic() < cancel_deadline:
                    time.sleep(max(0.0, float(self.ORDER_POLL_INTERVAL_SECONDS)))

            for broker_id, order in pending.items():
                reasons = [
                    f"order did not fill within {float(timeout_seconds):.1f} seconds",
                    "terminal cancellation could not be confirmed",
                ]
                if cancel_errors.get(broker_id):
                    reasons.append(f"cancel request failed: {cancel_errors[broker_id]}")
                if last_errors.get(broker_id):
                    reasons.append(f"last broker lookup: {last_errors[broker_id]}")
                failures.append({
                    **order,
                    "phase": f"{phase}_fill",
                    "submission_status": "pending_unknown",
                    "error": "; ".join(reasons),
                })

        self.tracker.record_event(
            "order_fill_barrier",
            "passed" if not failures else "blocked",
            run_id=run_id,
            details={
                "phase": phase,
                "submitted": len(submitted_orders),
                "filled": len(filled),
                "failed": len(failures),
                "failures": failures,
            },
        )
        return filled, failures

    def _submit_phase(
        self,
        orders: List[Dict],
        *,
        phase: str,
        run_id: str,
    ) -> Tuple[List[Dict], List[Dict]]:
        executed = []
        failures = []
        for order in orders:
            submitted_at = datetime.now(timezone.utc).isoformat()
            try:
                if order["side"] == "sell" and order["target_qty"] == 0.0:
                    response = self.alpaca.trading_client.close_position(
                        symbol_or_asset_id=order["symbol"]
                    )
                else:
                    response = self.alpaca.submit_order(
                        order["symbol"], order["qty"], order["side"]
                    )
                submitted = {
                    **order,
                    **_order_response(response),
                    "phase": phase,
                    "submitted_at": submitted_at,
                    "submission_status": "submitted",
                }
                executed.append(submitted)
                self.tracker.record_event(
                    "order", "submitted", run_id=run_id, details=submitted
                )
                logger.info(f"{order['side'].upper()} {order['symbol']} qty={order['qty']:.2f}")
            except Exception as exc:
                failed = {
                    **order,
                    "phase": phase,
                    "submitted_at": submitted_at,
                    "submission_status": "failed",
                    "error": str(exc),
                }
                failures.append(failed)
                self.tracker.record_event("order", "failed", run_id=run_id, details=failed)
                logger.error(f"{order['side'].upper()} {order['symbol']} failed: {exc}")
        return executed, failures

    def generate_and_execute_orders(
        self,
        target_weights: dict,
        strategy_config: dict = None,
        *,
        run_id: Optional[str] = None,
        audit_context: Optional[Dict] = None,
    ) -> list:
        run_id = run_id or uuid.uuid4().hex
        audit_context = audit_context or {}

        # Account lookup must succeed before cancelling anything or converting a
        # transient API failure (equity=0) into an accidental full liquidation.
        acct = self.alpaca.get_account()
        equity = float(acct.get("equity", 0.0) or 0.0)
        if equity <= 0 or str(acct.get("status", "")).lower() == "error":
            self.tracker.record_event(
                "orders", "blocked", run_id=run_id,
                details={"reason": "Broker account/equity unavailable", "account": acct},
            )
            raise RuntimeError("Broker account unavailable or equity is non-positive; no orders submitted")
        logger.info(f"Account equity: ${equity:,.2f}")

        positions = self.alpaca.get_positions()
        if not self._positions_are_consistent(acct, positions):
            self.tracker.record_event(
                "orders", "blocked", run_id=run_id,
                details={"reason": "Broker positions unavailable while account has market value"},
            )
            raise RuntimeError("Broker positions unavailable; no orders submitted")
        orders, skipped, current_prices = self._build_order_plan(
            target_weights, equity, positions
        )

        self.tracker.record_event(
            "order_plan", "planned", run_id=run_id,
            details={
                "equity": equity,
                "target_gross": sum(abs(float(w)) for w in target_weights.values()),
                "target_net": sum(float(w) for w in target_weights.values()),
                "orders": orders,
                "skipped": skipped,
                **audit_context,
            },
        )
        if skipped:
            self.tracker.record_event(
                "order_plan", "blocked", run_id=run_id,
                details={"reason": "One or more symbols have no executable price", "skipped": skipped},
            )
            raise RuntimeError("Order preflight failed: one or more symbols have no latest price")

        # Mutating broker state starts only after the complete batch passes
        # account, position, and price preflight. Bulk cancellation is followed
        # by a strict open-order barrier; a surviving/unknown prior order can
        # never overlap the replacement batch.
        has_open_orders = False
        if not orders:
            try:
                has_open_orders = bool(self.alpaca.get_open_orders_strict(limit=500))
            except Exception as exc:
                self.tracker.record_event(
                    "open_orders", "lookup_failed", run_id=run_id,
                    details={"error": str(exc)},
                )
                raise RuntimeError(
                    "Could not verify existing broker orders; no replacement batch submitted"
                ) from exc
        if orders or has_open_orders:
            try:
                self.alpaca.cancel_all_orders()
                self.tracker.record_event("open_orders", "cancel_requested", run_id=run_id)
                self._wait_for_open_orders_to_clear(run_id=run_id)
            except Exception as exc:
                self.tracker.record_event(
                    "open_orders", "cancel_failed", run_id=run_id,
                    details={"error": str(exc)},
                )
                raise RuntimeError(
                    "Could not confirm cancellation of existing broker orders; "
                    "replacement batch was not submitted"
                ) from exc

            # A prior order may have raced to a fill while cancellation was in
            # flight. Rebuild the entire initial plan from the confirmed broker
            # state before submitting the sell phase.
            acct = self.alpaca.get_account()
            equity = float(acct.get("equity", 0.0) or 0.0)
            if equity <= 0 or str(acct.get("status", "")).lower() == "error":
                raise RuntimeError(
                    "Broker account unavailable after open-order cancellation; "
                    "replacement batch was not submitted"
                )
            positions = self.alpaca.get_positions()
            if not self._positions_are_consistent(acct, positions):
                raise RuntimeError(
                    "Broker positions unavailable after open-order cancellation; "
                    "replacement batch was not submitted"
                )
            orders, skipped, replanned_prices = self._build_order_plan(
                target_weights, equity, positions
            )
            current_prices.update(replanned_prices)
            if skipped:
                raise RuntimeError(
                    "Order preflight failed after open-order cancellation: "
                    "one or more symbols have no latest price"
                )
            self.tracker.record_event(
                "order_plan", "replanned_after_cancel", run_id=run_id,
                details={"equity": equity, "orders": orders},
            )

        sell_orders = [o for o in orders if o["side"] == "sell"]
        executed = []
        failures = []

        submitted_sells, sell_submit_failures = self._submit_phase(
            sell_orders, phase="sell", run_id=run_id
        )
        executed.extend(submitted_sells)
        failures.extend(sell_submit_failures)

        if submitted_sells:
            filled_sells, sell_fill_failures = self._wait_for_fills(
                submitted_sells,
                phase="sell",
                timeout_seconds=self.SELL_FILL_TIMEOUT_SECONDS,
                run_id=run_id,
            )
            filled_by_id = {
                row.get("broker_order_id"): row for row in filled_sells
                if row.get("broker_order_id")
            }
            executed = [filled_by_id.get(row.get("broker_order_id"), row) for row in executed]
            failures.extend(sell_fill_failures)

        buy_orders = []
        if not failures:
            # Sell proceeds are usable only after every sell is confirmed filled.
            refreshed_account = self.alpaca.get_account()
            refreshed_equity = float(refreshed_account.get("equity", 0.0) or 0.0)
            if refreshed_equity <= 0:
                failures.append({
                    "phase": "buy_preflight",
                    "submission_status": "blocked",
                    "error": "post-sell broker equity is unavailable",
                })
            else:
                refreshed_positions = self.alpaca.get_positions()
                if not self._positions_are_consistent(
                    refreshed_account, refreshed_positions
                ):
                    failures.append({
                        "phase": "buy_preflight",
                        "submission_status": "blocked",
                        "error": (
                            "post-sell positions are unavailable while the account "
                            "still reports market value"
                        ),
                    })
                else:
                    refreshed_orders, refreshed_skipped, refreshed_prices = self._build_order_plan(
                        target_weights, refreshed_equity, refreshed_positions
                    )
                    current_prices.update(refreshed_prices)
                    if refreshed_skipped:
                        failures.extend({
                            **row,
                            "phase": "buy_preflight",
                            "submission_status": "blocked",
                            "error": row.get("reason", "unexecutable symbol"),
                        } for row in refreshed_skipped)
                    remaining_sells = [o for o in refreshed_orders if o["side"] == "sell"]
                    if remaining_sells:
                        failures.append({
                            "phase": "buy_preflight",
                            "submission_status": "blocked",
                            "error": "material sell deltas remain after the sell fill barrier",
                            "orders": remaining_sells,
                        })
                    buy_orders = [o for o in refreshed_orders if o["side"] == "buy"]

                    required_buying_power = sum(
                        float(order["qty"]) * float(order["price"]) for order in buy_orders
                    ) * self.BUYING_POWER_PRICE_BUFFER
                    available_buying_power = float(
                        refreshed_account.get("buying_power", 0.0) or 0.0
                    )
                    if required_buying_power > available_buying_power + 1e-6:
                        failures.append({
                            "phase": "buy_preflight",
                            "submission_status": "blocked",
                            "error": "post-sell buying power is insufficient for the target batch",
                            "required_buying_power": round(required_buying_power, 2),
                            "available_buying_power": round(available_buying_power, 2),
                        })

        if not failures and buy_orders:
            submitted_buys, buy_submit_failures = self._submit_phase(
                buy_orders, phase="buy", run_id=run_id
            )
            executed.extend(submitted_buys)
            failures.extend(buy_submit_failures)
            if submitted_buys:
                filled_buys, buy_fill_failures = self._wait_for_fills(
                    submitted_buys,
                    phase="buy",
                    timeout_seconds=self.BUY_FILL_TIMEOUT_SECONDS,
                    run_id=run_id,
                )
                filled_by_id = {
                    row.get("broker_order_id"): row for row in filled_buys
                    if row.get("broker_order_id")
                }
                executed = [filled_by_id.get(row.get("broker_order_id"), row) for row in executed]
                failures.extend(buy_fill_failures)

        residual_orders = []
        residual_skipped = []
        if not failures:
            final_account = self.alpaca.get_account()
            final_equity = float(final_account.get("equity", 0.0) or 0.0)
            if final_equity <= 0:
                failures.append({
                    "phase": "reconciliation",
                    "submission_status": "blocked",
                    "error": "final broker equity is unavailable",
                })
            else:
                final_positions = self.alpaca.get_positions()
                if not self._positions_are_consistent(final_account, final_positions):
                    failures.append({
                        "phase": "reconciliation",
                        "submission_status": "blocked",
                        "error": (
                            "final positions are unavailable while the account "
                            "still reports market value"
                        ),
                    })
                else:
                    residual_orders, residual_skipped, final_prices = self._build_order_plan(
                        target_weights, final_equity, final_positions
                    )
                    current_prices.update(final_prices)
                    if residual_orders or residual_skipped:
                        failures.append({
                            "phase": "reconciliation",
                            "submission_status": "partial",
                            "error": "material target deltas remain after fills",
                            "orders": residual_orders,
                            "skipped": residual_skipped,
                        })

        status = "completed" if not failures else ("partial" if executed else "failed")
        self.last_summary = {
            "run_id": run_id,
            "planned": len(orders),
            "submitted": len(executed),
            "failed": len(failures),
            "skipped": len(skipped),
            "status": status,
            "orders": executed,
            "failures": failures,
            "skipped_orders": skipped,
            "residual_orders": residual_orders,
            "residual_skipped": residual_skipped,
        }
        self._save_snapshot(
            equity,
            current_prices,
            target_weights,
            strategy_config,
            run_id=run_id,
            audit_context=audit_context,
            order_summary=self.last_summary,
        )
        self.tracker.record_event(
            "order_batch", self.last_summary["status"], run_id=run_id,
            details=self.last_summary,
        )
        return executed

    def _save_snapshot(
        self,
        equity,
        prices,
        weights,
        config,
        *,
        run_id: str,
        audit_context: Dict,
        order_summary: Dict,
    ):
        try:
            snapshot_kind = str(audit_context.get("snapshot_kind", "rebalance"))
            if snapshot_kind == "risk_overlay":
                snap_path = f"logs/last_risk_overlay_{self.account_name}.json"
                hist_path = f"logs/risk_overlay_history_{self.account_name}.json"
            else:
                snap_path = f"logs/last_rebalance_{self.account_name}.json"
                hist_path = f"logs/rebalance_history_{self.account_name}.json"
            os.makedirs("logs", exist_ok=True)
            config = config or {}
            config_json = json.dumps(config, sort_keys=True, default=str)
            snap = {
                "snapshot_id": uuid.uuid4().hex,
                "run_id": run_id,
                "rebalance_date": datetime.now().strftime("%Y-%m-%d"),
                "rebalance_time": datetime.now(timezone.utc).isoformat(),
                "account": self.account_name,
                "snapshot_kind": snapshot_kind,
                "equity": equity,
                "as_of_date": audit_context.get("as_of_date"),
                "data_sync": audit_context.get("data_sync"),
                # For adaptive models (for example MWU), the weights used for
                # this decision can differ from the static config. Persist the
                # calculated values so the dashboard/audit trail shows reality.
                "factor_weights": audit_context.get("factor_weights") or {},
                "target_gross": round(sum(abs(float(w)) for w in weights.values()), 6),
                "target_net": round(sum(float(w) for w in weights.values()), 6),
                "positions": {
                    sym: {
                        "weight": round(float(w), 6),
                        "price": round(float(prices.get(sym, 0) or 0), 4),
                        "value": round(equity * abs(float(w)), 2),
                    }
                    for sym, w in weights.items()
                },
                "strategy_config": config,
                "strategy_config_sha256": hashlib.sha256(config_json.encode("utf-8")).hexdigest(),
                "order_summary": order_summary,
            }
            if order_summary.get("status") == "completed":
                LiveTracker._write_json_atomic(snap_path, snap)

            pending_path = Path("logs") / f"pending_rebalance_{self.account_name}.json"
            if snapshot_kind == "rebalance":
                if order_summary.get("status") == "completed":
                    pending_path.unlink(missing_ok=True)
                else:
                    LiveTracker._write_json_atomic(str(pending_path), snap)

            history = []
            if os.path.exists(hist_path):
                try:
                    history = json.load(open(hist_path, encoding="utf-8"))
                except Exception:
                    pass
            # Do not erase same-day retries; run_id/snapshot_id disambiguate them.
            history.append(snap)
            LiveTracker._write_json_atomic(hist_path, history)
        except Exception as exc:
            logger.warning(f"Snapshot save failed: {exc}")
