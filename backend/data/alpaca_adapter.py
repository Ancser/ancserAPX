from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed, Adjustment
from alpaca.trading.client import TradingClient
from alpaca.data.historical import StockHistoricalDataClient
import polars as pl
import pandas as pd
from dotenv import load_dotenv
import os
from datetime import datetime
from typing import List, Dict, Optional

load_dotenv()

MARKET_DATA_SCHEMA = {
    "timestamp": pl.Datetime,
    "symbol": pl.Categorical,
    "open": pl.Float32,
    "high": pl.Float32,
    "low": pl.Float32,
    "close": pl.Float32,
    "volume": pl.Float32,
    "vwap": pl.Float32,
    "trade_count": pl.UInt32,
}


class AlpacaAdapter:
    def __init__(self, account_name: str = "Main", paper: bool = True):
        self.account_name = account_name
        self.paper = paper

        if account_name == "Main":
            self.api_key = os.getenv("APCA_API_KEY_ID")
            self.secret_key = os.getenv("APCA_API_SECRET_KEY")
        else:
            upper = account_name.upper()
            self.api_key = os.getenv(f"APCA_API_KEY_ID_{upper}") or os.getenv("APCA_API_KEY_ID")
            self.secret_key = (
                os.getenv(f"APCA_API_SECRET_KEY_{upper}")
                or os.getenv(f"APCA_API_SECRET_{upper}")
                or os.getenv("APCA_API_SECRET_KEY")
            )

        if not self.api_key or not self.secret_key:
            raise ValueError(f"Alpaca API keys missing for account '{account_name}'. Check .env file.")

        # Use IEX by default (free). SIP requires paid Alpaca data subscription.
        # Override with APCA_DATA_FEED=sip in .env for 10-year depth.
        feed_env = os.getenv("APCA_DATA_FEED", "iex").lower()
        self._feed = DataFeed.SIP if feed_env == "sip" else DataFeed.IEX

        self.data_client = StockHistoricalDataClient(self.api_key, self.secret_key)
        self.trading_client = TradingClient(self.api_key, self.secret_key, paper=self.paper)

    # ------------------------------------------------------------------
    # Market data
    # ------------------------------------------------------------------

    def fetch_history(self, symbols: List[str], start_date: str, end_date: Optional[str] = None) -> pl.LazyFrame:
        if not end_date:
            end_date = datetime.now().strftime("%Y-%m-%d")
        req = StockBarsRequest(
            symbol_or_symbols=symbols,
            timeframe=TimeFrame.Day,
            start=datetime.strptime(start_date, "%Y-%m-%d"),
            end=datetime.strptime(end_date, "%Y-%m-%d"),
            adjustment=Adjustment.ALL,
            feed=self._feed,
        )
        try:
            bars = self.data_client.get_stock_bars(req).df
        except Exception as e:
            print(f"[AlpacaAdapter] fetch_history error: {e}")
            return pl.LazyFrame()

        if bars.empty:
            return pl.LazyFrame()

        bars = bars.reset_index()
        df = pl.from_pandas(bars)
        df = df.with_columns([
            pl.col("symbol").cast(pl.Categorical),
            pl.col("open").cast(pl.Float32),
            pl.col("high").cast(pl.Float32),
            pl.col("low").cast(pl.Float32),
            pl.col("close").cast(pl.Float32),
            pl.col("volume").cast(pl.Float32),
            pl.col("vwap").cast(pl.Float32).fill_null(pl.col("close").cast(pl.Float32)),
            pl.col("trade_count").cast(pl.UInt32).fill_null(0),
        ])
        # Keep only schema columns that exist
        schema_cols = [c for c in MARKET_DATA_SCHEMA.keys() if c in df.columns]
        return df.select(schema_cols).lazy()

    def get_latest_prices(self, symbols: List[str]) -> Dict[str, float]:
        from alpaca.data.requests import StockLatestTradeRequest
        try:
            req = StockLatestTradeRequest(symbol_or_symbols=symbols)
            trades = self.data_client.get_stock_latest_trade(req)
            return {sym: float(t.price) for sym, t in trades.items()}
        except Exception as e:
            print(f"[AlpacaAdapter] get_latest_prices error: {e}")
            return {}

    # ------------------------------------------------------------------
    # Account / trading
    # ------------------------------------------------------------------

    def get_account(self) -> Dict:
        try:
            a = self.trading_client.get_account()
            return {
                "equity": float(a.equity),
                "buying_power": float(a.buying_power),
                "cash": float(a.cash),
                "daytrade_count": int(a.daytrade_count),
                "status": str(a.status.value) if hasattr(a.status, "value") else str(a.status),
                "currency": a.currency,
            }
        except Exception as e:
            print(f"[AlpacaAdapter] get_account error: {e}")
            return {"equity": 0.0, "buying_power": 0.0, "status": "Error"}

    def get_positions(self) -> List[Dict]:
        try:
            return [
                {
                    "symbol": p.symbol,
                    "qty": float(p.qty),
                    "market_value": float(p.market_value),
                    "avg_entry": float(p.avg_entry_price),
                    "current_price": float(p.current_price),
                    "unrealized_pl": float(p.unrealized_pl),
                    "pl_pct": float(p.unrealized_plpc) * 100,
                }
                for p in self.trading_client.get_all_positions()
            ]
        except Exception as e:
            print(f"[AlpacaAdapter] get_positions error: {e}")
            return []

    def get_portfolio_history(self, period: str = "1M", timeframe: str = "1D") -> pd.DataFrame:
        from alpaca.trading.requests import GetPortfolioHistoryRequest
        try:
            req = GetPortfolioHistoryRequest(period=period, timeframe=timeframe, extended_hours=True)
            hist = self.trading_client.get_portfolio_history(req)
            if not hist or not hasattr(hist, "timestamp"):
                return pd.DataFrame()
            df = pd.DataFrame({
                "timestamp": hist.timestamp,
                "equity": hist.equity,
                "profit_loss": hist.profit_loss,
                "profit_loss_pct": hist.profit_loss_pct,
            })
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
            return df.set_index("timestamp")
        except Exception as e:
            print(f"[AlpacaAdapter] get_portfolio_history error: {e}")
            return pd.DataFrame()

    def get_orders(self, limit: int = 20) -> List[Dict]:
        from alpaca.trading.requests import GetOrdersRequest
        from alpaca.trading.enums import QueryOrderStatus
        try:
            req = GetOrdersRequest(status=QueryOrderStatus.ALL, limit=limit, nested=True)
            orders = self.trading_client.get_orders(filter=req)
            return [
                {
                    "id": str(o.id),
                    "symbol": o.symbol,
                    "qty": float(o.qty) if o.qty else 0.0,
                    "side": o.side.value,
                    "status": o.status.value,
                    "filled_avg_price": float(o.filled_avg_price) if o.filled_avg_price else None,
                    "created_at": str(o.created_at),
                }
                for o in orders
            ]
        except Exception as e:
            print(f"[AlpacaAdapter] get_orders error: {e}")
            return []

    def get_activities(self, limit: int = 200) -> List[Dict]:
        try:
            all_acts = []
            page_token = None
            while len(all_acts) < limit:
                params = {"page_size": min(100, limit)}
                if page_token:
                    params["page_token"] = page_token
                page = self.trading_client.get("/account/activities/FILL", params)
                if not page:
                    break
                all_acts.extend(page)
                if len(page) < params["page_size"]:
                    break
                last_id = page[-1].get("id", "")
                if not last_id:
                    break
                page_token = last_id
            result = []
            for a in all_acts[:limit]:
                txn = a.get("transaction_time", "")
                try:
                    dt = datetime.fromisoformat(txn.replace("Z", "+00:00"))
                    date_str = dt.strftime("%Y-%m-%d")
                    time_str = dt.isoformat()
                except Exception:
                    date_str, time_str = txn[:10], txn
                qty = float(a.get("qty", 0))
                price = float(a.get("price", 0))
                result.append({
                    "date": date_str, "time": time_str,
                    "symbol": a.get("symbol", ""), "side": a.get("side", ""),
                    "qty": qty, "price": price, "value": round(qty * price, 2),
                })
            return result
        except Exception as e:
            print(f"[AlpacaAdapter] get_activities error: {e}")
            return []

    def get_clock(self) -> Dict:
        try:
            c = self.trading_client.get_clock()
            return {
                "is_open": c.is_open,
                "next_open": str(c.next_open),
                "next_close": str(c.next_close),
            }
        except Exception as e:
            print(f"[AlpacaAdapter] get_clock error: {e}")
            return {"is_open": False}

    def get_trading_days(self, start, end) -> List:
        """Return the list of NYSE trading dates (datetime.date) in [start, end]
        per Alpaca's official market calendar (handles holidays & half-days).
        `start`/`end` may be date or 'YYYY-MM-DD' strings."""
        from datetime import date as _date
        from alpaca.trading.requests import GetCalendarRequest

        def _to_date(x):
            if isinstance(x, str):
                return datetime.strptime(x[:10], "%Y-%m-%d").date()
            if isinstance(x, datetime):
                return x.date()
            return x  # already a date

        cal = self.trading_client.get_calendar(
            GetCalendarRequest(start=_to_date(start), end=_to_date(end))
        )
        out = []
        for c in cal:
            d = getattr(c, "date", None)
            if d is None:
                continue
            out.append(_to_date(d))
        return sorted(out)

    def submit_order(self, symbol: str, qty: float, side: str, notional: float = None):
        from alpaca.trading.requests import MarketOrderRequest
        from alpaca.trading.enums import OrderSide, TimeInForce
        side_enum = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL
        params = {"symbol": symbol, "side": side_enum, "time_in_force": TimeInForce.DAY}
        if notional and notional > 0:
            params["notional"] = round(notional, 2)
        else:
            params["qty"] = qty
        order = self.trading_client.submit_order(order_data=MarketOrderRequest(**params))
        return order

    def cancel_all_orders(self):
        try:
            self.trading_client.cancel_orders()
        except Exception as e:
            print(f"[AlpacaAdapter] cancel_all_orders error: {e}")
