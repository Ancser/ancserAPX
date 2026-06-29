"""Order Management System — ported from ancserAPX unchanged."""

import json, logging, os
from datetime import datetime
from backend.data.alpaca_adapter import AlpacaAdapter

logger = logging.getLogger("backend.oms")


class OrderManagementSystem:
    def __init__(self, account_name: str = "Main"):
        self.account_name = account_name
        self.alpaca = AlpacaAdapter(account_name)

    def generate_and_execute_orders(self, target_weights: dict, strategy_config: dict = None) -> list:
        self.alpaca.cancel_all_orders()

        acct = self.alpaca.get_account()
        equity = float(acct.get("equity", 0.0))
        logger.info(f"Account equity: ${equity:,.2f}")

        positions = self.alpaca.get_positions()
        current_qtys = {p["symbol"]: float(p["qty"]) for p in positions}
        current_prices = {p["symbol"]: float(p["current_price"]) for p in positions}

        all_symbols = set(current_qtys) | set(target_weights)
        missing = [s for s in all_symbols if s not in current_prices]
        if missing:
            current_prices.update(self.alpaca.get_latest_prices(missing))

        orders = []
        for sym in all_symbols:
            current_qty = current_qtys.get(sym, 0.0)
            target_pct = target_weights.get(sym, 0.0)
            price = current_prices.get(sym, 0.0)
            if price <= 0:
                continue

            if target_pct == 0.0:
                if current_qty <= 0:
                    continue
                orders.append({"symbol": sym, "side": "sell", "qty": current_qty,
                                "price": price, "target_qty": 0.0})
            else:
                target_qty = round((equity * abs(target_pct)) / price, 2)
                diff_qty = round(target_qty - current_qty, 2)
                if abs(diff_qty) * price < 10.0 or diff_qty == 0:
                    continue
                orders.append({
                    "symbol": sym,
                    "side": "buy" if diff_qty > 0 else "sell",
                    "qty": abs(diff_qty),
                    "price": price,
                    "target_qty": target_qty,
                })

        sell_orders = [o for o in orders if o["side"] == "sell"]
        buy_orders  = [o for o in orders if o["side"] == "buy"]
        executed = []

        for order in sell_orders:
            try:
                if order["target_qty"] == 0.0:
                    self.alpaca.trading_client.close_position(symbol_or_asset_id=order["symbol"])
                else:
                    self.alpaca.submit_order(order["symbol"], order["qty"], "sell")
                executed.append(order)
                logger.info(f"SELL {order['symbol']} qty={order['qty']:.2f}")
            except Exception as e:
                logger.error(f"SELL {order['symbol']} failed: {e}")

        for order in buy_orders:
            try:
                self.alpaca.submit_order(order["symbol"], order["qty"], "buy")
                executed.append(order)
                logger.info(f"BUY {order['symbol']} qty={order['qty']:.2f}")
            except Exception as e:
                logger.error(f"BUY {order['symbol']} failed: {e}")

        # Save snapshot
        self._save_snapshot(equity, current_prices, target_weights, strategy_config)
        return executed

    def _save_snapshot(self, equity, prices, weights, config):
        try:
            snap_path = f"logs/last_rebalance_{self.account_name}.json"
            hist_path = f"logs/rebalance_history_{self.account_name}.json"
            os.makedirs("logs", exist_ok=True)
            snap = {
                "rebalance_date": datetime.now().strftime("%Y-%m-%d"),
                "rebalance_time": datetime.now().isoformat(),
                "account": self.account_name,
                "equity": equity,
                "positions": {
                    sym: {
                        "weight": round(w, 4),
                        "price": round(prices.get(sym, 0), 4),
                        "value": round(equity * abs(w), 2),
                    }
                    for sym, w in weights.items()
                },
            }
            if config:
                snap["strategy_config"] = {k: config[k] for k in
                    ["active_factors", "use_mwu", "leverage", "use_vol_target", "strategy_mode"]
                    if k in config}
            with open(snap_path, "w") as f:
                json.dump(snap, f, indent=2)

            history = []
            if os.path.exists(hist_path):
                try:
                    history = json.load(open(hist_path))
                except Exception:
                    pass
            history = [h for h in history if h.get("rebalance_date") != snap["rebalance_date"]]
            history.append(snap)
            with open(hist_path, "w") as f:
                json.dump(history, f, indent=2)
        except Exception as e:
            logger.warning(f"Snapshot save failed: {e}")
