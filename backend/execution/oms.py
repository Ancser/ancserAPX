"""Order Management System with durable order/audit snapshots."""

import hashlib
import json
import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Dict, Optional

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
    def __init__(self, account_name: str = "Main"):
        self.account_name = account_name
        self.alpaca = AlpacaAdapter(account_name)
        self.tracker = LiveTracker(account_name)
        self.last_summary: Dict = {}

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
        broker_market_value = abs(float(acct.get("long_market_value", 0) or 0)) + abs(
            float(acct.get("short_market_value", 0) or 0)
        )
        if broker_market_value > 1.0 and not positions:
            self.tracker.record_event(
                "orders", "blocked", run_id=run_id,
                details={"reason": "Broker positions unavailable while account has market value"},
            )
            raise RuntimeError("Broker positions unavailable; no orders submitted")
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
        # account, position, and price preflight.
        if orders:
            try:
                self.alpaca.cancel_all_orders()
                self.tracker.record_event("open_orders", "cancel_requested", run_id=run_id)
            except Exception as exc:
                self.tracker.record_event(
                    "open_orders", "cancel_failed", run_id=run_id,
                    details={"error": str(exc)},
                )
                raise RuntimeError(
                    "Could not confirm cancellation of existing broker orders; "
                    "replacement batch was not submitted"
                ) from exc

        sell_orders = [o for o in orders if o["side"] == "sell"]
        buy_orders = [o for o in orders if o["side"] == "buy"]
        executed = []
        failures = []

        for order in sell_orders + buy_orders:
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
                    "submitted_at": submitted_at,
                    "submission_status": "failed",
                    "error": str(exc),
                }
                failures.append(failed)
                self.tracker.record_event("order", "failed", run_id=run_id, details=failed)
                logger.error(f"{order['side'].upper()} {order['symbol']} failed: {exc}")

        self.last_summary = {
            "run_id": run_id,
            "planned": len(orders),
            "submitted": len(executed),
            "failed": len(failures),
            "skipped": len(skipped),
            "status": "submitted" if not failures else ("partial" if executed else "failed"),
            "orders": executed,
            "failures": failures,
            "skipped_orders": skipped,
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
            if order_summary.get("status") != "failed":
                LiveTracker._write_json_atomic(snap_path, snap)

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
