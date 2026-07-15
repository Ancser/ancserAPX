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

from backend.utils.accounts import get_account_credentials, get_account_paper

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
    def __init__(self, account_name: str = "Main", paper: Optional[bool] = None):
        self.account_name = account_name
        self.paper = get_account_paper(account_name) if paper is None else paper

        creds = get_account_credentials(account_name)
        self.api_key = creds["api_key"]
        self.secret_key = creds["secret_key"]

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
            # Daily bars are keyed by trading date throughout the local store.
            # Strip Alpaca's UTC timezone so fetched fallback rows can be
            # concatenated with the store's tz-naive timestamp column.
            pl.col("timestamp").dt.replace_time_zone(None),
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

    def get_asset_eligibility(
        self,
        symbols: List[str],
        *,
        require_fractionable: bool = True,
    ) -> Dict:
        """Resolve configured symbols through one active bulk call plus misses.

        Alpaca's observed Trading API response can omit inactive assets even
        when status is not supplied. Request active US equities explicitly, then
        query only configured symbols absent from that bulk response. In a normal
        S&P/QQQ universe this is one bulk request plus a handful of delisted or
        renamed candidates, never one request per configured symbol.

        The OMS submits quantities rounded to two decimal shares, so live target
        candidates must be fractionable. Existing positions are handled later by
        the OMS liquidation path; this check prevents opening a new fractional
        target that the broker will reject.
        """
        configured = list(dict.fromkeys(
            str(symbol).strip().upper() for symbol in symbols if str(symbol).strip()
        ))
        if not configured:
            raise RuntimeError("Asset eligibility cannot validate an empty universe")

        try:
            from alpaca.trading.enums import AssetClass, AssetStatus
            from alpaca.trading.requests import GetAssetsRequest

            asset_filter = GetAssetsRequest(
                status=AssetStatus.ACTIVE,
                asset_class=AssetClass.US_EQUITY,
            )
            assets = self.trading_client.get_all_assets(asset_filter)
        except Exception as exc:
            raise RuntimeError(f"Alpaca asset master request failed: {exc}") from exc
        if not isinstance(assets, list) or not assets:
            raise RuntimeError("Alpaca asset master returned an empty or invalid response")

        def _field(asset, name):
            return asset.get(name) if isinstance(asset, dict) else getattr(asset, name, None)

        def _text(value) -> str:
            if hasattr(value, "value"):
                value = value.value
            return str(value).strip().lower() if value is not None else ""

        def _record(asset) -> Dict:
            return {
                "status": _text(_field(asset, "status")),
                "tradable": _field(asset, "tradable"),
                "fractionable": _field(asset, "fractionable"),
                "asset_class": _text(_field(asset, "asset_class") or _field(asset, "class")),
                "exchange": _text(_field(asset, "exchange")),
            }

        configured_set = set(configured)
        asset_map: Dict[str, Dict] = {}
        duplicate_symbols = set()
        returned_symbols = set()
        for asset in assets:
            symbol = str(_field(asset, "symbol") or "").strip().upper()
            if not symbol:
                continue
            returned_symbols.add(symbol)
            if symbol not in configured_set:
                continue
            record = _record(asset)
            if symbol in asset_map and asset_map[symbol] != record:
                duplicate_symbols.add(symbol)
            asset_map[symbol] = record

        # These liquid US-equity sentinels should always exist in a complete
        # asset-master response. Their absence is treated as a truncated or
        # semantically incompatible API response, not as permission to exclude
        # configured symbols.
        missing_sentinels = sorted({"QQQ", "SPY"} - returned_symbols)
        if missing_sentinels:
            raise RuntimeError(
                "Alpaca asset master appears incomplete; missing sentinel assets: "
                + ", ".join(missing_sentinels)
            )
        if duplicate_symbols:
            raise RuntimeError(
                "Alpaca asset master returned conflicting duplicate symbols: "
                + ", ".join(sorted(duplicate_symbols))
            )

        missing_candidates = [symbol for symbol in configured if symbol not in asset_map]
        max_candidate_lookups = 25
        if len(missing_candidates) > max_candidate_lookups:
            raise RuntimeError(
                "Too many configured symbols are absent from the active asset master "
                f"({len(missing_candidates)} > {max_candidate_lookups}); refusing to infer exclusions"
            )

        explicit_not_found = set()
        lookup_count = 0
        for requested_symbol in missing_candidates:
            lookup_count += 1
            try:
                asset = self.trading_client.get_asset(requested_symbol)
            except Exception as exc:
                status_code = (
                    getattr(exc, "status_code", None)
                    or getattr(exc, "status", None)
                    or getattr(exc, "http_status", None)
                )
                code = getattr(exc, "code", None)
                try:
                    is_not_found = int(status_code or code or 0) == 404
                except (TypeError, ValueError):
                    is_not_found = False
                if not is_not_found:
                    message = str(exc).lower()
                    is_not_found = "404" in message and "not found" in message
                if is_not_found:
                    explicit_not_found.add(requested_symbol)
                    continue
                raise RuntimeError(
                    f"Alpaca asset lookup for {requested_symbol} failed ambiguously: {exc}"
                ) from exc

            returned_symbol = str(_field(asset, "symbol") or "").strip().upper()
            if returned_symbol != requested_symbol:
                raise RuntimeError(
                    f"Alpaca asset lookup for {requested_symbol} returned {returned_symbol or 'no symbol'}"
                )
            asset_map[requested_symbol] = _record(asset)

        effective = []
        excluded: Dict[str, Dict] = {}
        for symbol in configured:
            record = asset_map.get(symbol)
            if record is None:
                if symbol not in explicit_not_found:
                    raise RuntimeError(
                        f"Alpaca asset eligibility for {symbol} is unresolved"
                    )
                excluded[symbol] = {"reason": "not_found"}
                continue
            status = record["status"]
            if status not in {"active", "inactive"}:
                raise RuntimeError(
                    f"Alpaca asset eligibility for {symbol} has unknown status {status!r}"
                )
            if status != "active":
                excluded[symbol] = {"reason": "inactive", **record}
                continue
            if not record["asset_class"]:
                raise RuntimeError(
                    f"Alpaca asset eligibility for {symbol} has unknown asset class"
                )
            if record["asset_class"] != "us_equity":
                excluded[symbol] = {"reason": "unsupported_asset_class", **record}
                continue
            if not isinstance(record["tradable"], bool):
                raise RuntimeError(
                    f"Alpaca asset eligibility for {symbol} has unknown tradable flag"
                )
            if not record["tradable"]:
                excluded[symbol] = {"reason": "not_tradable", **record}
                continue
            if require_fractionable:
                if not isinstance(record["fractionable"], bool):
                    raise RuntimeError(
                        f"Alpaca asset eligibility for {symbol} has unknown fractionable flag"
                    )
                if not record["fractionable"]:
                    excluded[symbol] = {"reason": "not_fractionable", **record}
                    continue
            effective.append(symbol)

        return {
            "status": "passed",
            "source": "alpaca_get_all_assets",
            "bulk_request_count": 1,
            "candidate_lookup_count": lookup_count,
            "request_count": 1 + lookup_count,
            "asset_master_count": len(assets),
            "require_fractionable": bool(require_fractionable),
            "configured_count": len(configured),
            "effective_count": len(effective),
            "effective_symbols": effective,
            "excluded_count": len(excluded),
            "excluded_assets": excluded,
        }

    # ------------------------------------------------------------------
    # Account / trading
    # ------------------------------------------------------------------

    def get_account(self) -> Dict:
        try:
            a = self.trading_client.get_account()
            # daytrade_count can be None on some Alpaca account types; int(None) crashes
            # and previously made OMS think equity=0 (skip buys / force full liquidations).
            dtc = a.daytrade_count
            return {
                "equity": float(a.equity),
                "last_equity": float(a.last_equity or 0),
                "portfolio_value": float(a.portfolio_value or a.equity or 0),
                "buying_power": float(a.buying_power),
                "cash": float(a.cash),
                "long_market_value": float(a.long_market_value or 0),
                "short_market_value": float(a.short_market_value or 0),
                "initial_margin": float(a.initial_margin or 0),
                "maintenance_margin": float(a.maintenance_margin or 0),
                "daytrade_count": int(dtc) if dtc is not None else 0,
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
                    "filled_qty": float(o.filled_qty) if o.filled_qty else 0.0,
                    "side": o.side.value,
                    "status": o.status.value,
                    "filled_avg_price": float(o.filled_avg_price) if o.filled_avg_price else None,
                    "created_at": str(o.created_at),
                    "updated_at": str(o.updated_at) if o.updated_at else None,
                    "filled_at": str(o.filled_at) if o.filled_at else None,
                    "order_type": o.type.value if hasattr(o.type, "value") else str(o.type),
                    "time_in_force": (
                        o.time_in_force.value
                        if hasattr(o.time_in_force, "value") else str(o.time_in_force)
                    ),
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
                    "id": str(a.get("id", "")),
                    "date": date_str, "time": time_str,
                    "symbol": a.get("symbol", ""), "side": a.get("side", ""),
                    "qty": qty, "price": price, "value": round(qty * price, 2),
                    "order_id": str(a.get("order_id", "")),
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
        # Cancellation is part of the OMS mutation boundary. A failure must
        # propagate so the caller cannot submit a replacement batch on top of
        # broker orders whose state is unknown.
        return self.trading_client.cancel_orders()
