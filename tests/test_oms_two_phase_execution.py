from types import SimpleNamespace

from backend.execution.oms import OrderManagementSystem


def _order(order_id, status, qty, filled_qty=0.0, price=None):
    return SimpleNamespace(
        id=order_id,
        status=status,
        filled_qty=filled_qty,
        filled_avg_price=price,
        created_at=None,
        updated_at=None,
    )


class _Tracker:
    def __init__(self):
        self.events = []

    def record_event(self, event_type, status, **kwargs):
        self.events.append((event_type, status, kwargs.get("details", {})))


class _TwoPhaseAdapter:
    def __init__(self, *, post_sell_buying_power=200.0, sell_status="filled"):
        self.trading_client = self
        self.post_sell_buying_power = post_sell_buying_power
        self.sell_status = sell_status
        self.sell_filled = False
        self.buy_filled = False
        self.open_orders = []
        self.calls = []

    def get_account(self):
        return {
            "equity": 100.0,
            "status": "ACTIVE",
            "buying_power": self.post_sell_buying_power if self.sell_filled else 0.0,
            "long_market_value": 0.0 if self.sell_filled else 100.0,
            "short_market_value": 0.0,
        }

    def get_positions(self):
        if self.buy_filled:
            return [{
                "symbol": "NEW", "qty": 10.0, "current_price": 10.0,
                "market_value": 100.0,
            }]
        if self.sell_filled:
            return []
        return [{
            "symbol": "OLD", "qty": 10.0, "current_price": 10.0,
            "market_value": 100.0,
        }]

    def get_latest_prices(self, symbols):
        return {symbol: 10.0 for symbol in symbols}

    def cancel_all_orders(self):
        self.calls.append("cancel_all")

    def get_open_orders_strict(self, limit=500):
        self.calls.append("poll_open_orders")
        return list(self.open_orders)

    def close_position(self, symbol_or_asset_id):
        self.calls.append(f"submit_sell:{symbol_or_asset_id}")
        return _order("SELL-1", "accepted", 10.0)

    def submit_order(self, symbol, qty, side):
        assert self.sell_filled, "buy was submitted before the sell fill barrier"
        self.calls.append(f"submit_{side}:{symbol}:{qty}")
        return _order("BUY-1", "accepted", qty)

    def get_order_by_id(self, order_id):
        self.calls.append(f"poll:{order_id}")
        if order_id == "SELL-1":
            if self.sell_status == "filled":
                self.sell_filled = True
                return _order(order_id, "filled", 10.0, filled_qty=10.0, price=10.0)
            return _order(order_id, self.sell_status, 10.0)
        self.buy_filled = True
        return _order(order_id, "filled", 10.0, filled_qty=10.0, price=10.0)

    def cancel_order_by_id(self, order_id):
        self.calls.append(f"cancel:{order_id}")
        if order_id == "SELL-1":
            self.sell_status = "canceled"


def _oms(adapter):
    oms = OrderManagementSystem.__new__(OrderManagementSystem)
    oms.account_name = "Unit"
    oms.alpaca = adapter
    oms.tracker = _Tracker()
    oms.last_summary = {}
    oms.ORDER_POLL_INTERVAL_SECONDS = 0.0
    oms.SELL_FILL_TIMEOUT_SECONDS = 0.02
    oms.BUY_FILL_TIMEOUT_SECONDS = 0.02
    oms.CANCEL_CONFIRM_TIMEOUT_SECONDS = 0.0
    oms.OPEN_ORDER_CANCEL_TIMEOUT_SECONDS = 0.0
    oms._save_snapshot = lambda *args, **kwargs: None
    return oms


def test_sells_fill_before_buys_and_post_sell_state_is_replanned():
    adapter = _TwoPhaseAdapter()
    oms = _oms(adapter)

    oms.generate_and_execute_orders({"NEW": 1.0}, run_id="two-phase")

    assert oms.last_summary["status"] == "completed"
    assert oms.last_summary["failed"] == 0
    assert adapter.calls.index("poll:SELL-1") < adapter.calls.index("submit_buy:NEW:10.0")
    assert adapter.buy_filled
    barriers = [event for event in oms.tracker.events if event[0] == "order_fill_barrier"]
    assert [(event[1], event[2]["phase"]) for event in barriers] == [
        ("passed", "sell"), ("passed", "buy")
    ]


def test_unfilled_sell_blocks_every_buy():
    adapter = _TwoPhaseAdapter(sell_status="accepted")
    oms = _oms(adapter)
    oms.SELL_FILL_TIMEOUT_SECONDS = 0.0

    oms.generate_and_execute_orders({"NEW": 1.0}, run_id="sell-timeout")

    assert oms.last_summary["status"] == "partial"
    assert oms.last_summary["failed"] >= 1
    assert not any(call.startswith("submit_buy") for call in adapter.calls)
    assert any(
        failure.get("phase") == "sell_fill"
        and failure.get("submission_status") == "canceled_after_timeout"
        for failure in oms.last_summary["failures"]
    )
    assert "cancel:SELL-1" in adapter.calls


def test_post_sell_buying_power_shortfall_blocks_the_buy_batch():
    adapter = _TwoPhaseAdapter(post_sell_buying_power=35.84)
    oms = _oms(adapter)

    oms.generate_and_execute_orders({"NEW": 1.0}, run_id="bp-shortfall")

    assert oms.last_summary["status"] == "partial"
    assert not any(call.startswith("submit_buy") for call in adapter.calls)
    failure = next(
        item for item in oms.last_summary["failures"]
        if item.get("phase") == "buy_preflight"
    )
    assert failure["available_buying_power"] == 35.84
    assert failure["required_buying_power"] == 103.0


def test_empty_post_sell_positions_with_market_value_blocks_duplicate_buys():
    class Adapter(_TwoPhaseAdapter):
        def get_account(self):
            account = super().get_account()
            if self.sell_filled:
                account["long_market_value"] = 50.0
            return account

        def get_positions(self):
            if self.sell_filled:
                # Simulate AlpacaAdapter masking a transient API error as [].
                return []
            return [
                {"symbol": "OLD", "qty": 5.0, "current_price": 10.0,
                 "market_value": 50.0},
                {"symbol": "KEEP", "qty": 5.0, "current_price": 10.0,
                 "market_value": 50.0},
            ]

    adapter = Adapter()
    oms = _oms(adapter)

    oms.generate_and_execute_orders({"KEEP": 0.5}, run_id="empty-post-sell")

    assert oms.last_summary["status"] == "partial"
    assert not any(call.startswith("submit_buy") for call in adapter.calls)
    assert any(
        failure.get("phase") == "buy_preflight"
        and "positions are unavailable" in failure.get("error", "")
        for failure in oms.last_summary["failures"]
    )


def test_unconfirmed_preexisting_order_blocks_all_replacements():
    adapter = _TwoPhaseAdapter()
    adapter.open_orders = [{"id": "OLD-OPEN", "symbol": "NEW", "status": "new"}]
    oms = _oms(adapter)

    try:
        oms.generate_and_execute_orders({"NEW": 1.0}, run_id="open-order-barrier")
    except RuntimeError as exc:
        assert "replacement batch was not submitted" in str(exc)
    else:
        raise AssertionError("unconfirmed open order should block replacements")

    assert not any(call.startswith("submit_sell") for call in adapter.calls)
    assert not any(call.startswith("submit_buy") for call in adapter.calls)
