"""Pure helpers for auditable performance reporting."""

from __future__ import annotations

from typing import Any, Dict, List


def fifo_realized_pnl(activities: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Reconstruct gross realised P&L from available fills using FIFO lots."""
    ordered = sorted(activities, key=lambda row: str(row.get("time") or row.get("date") or ""))
    lots: Dict[str, List[List[float]]] = {}
    realised_by_symbol: Dict[str, float] = {}
    unmatched_sells: Dict[str, float] = {}
    for fill in ordered:
        symbol = str(fill.get("symbol", "")).upper()
        side = str(fill.get("side", "")).lower()
        qty = max(0.0, float(fill.get("qty", 0.0) or 0.0))
        price = max(0.0, float(fill.get("price", 0.0) or 0.0))
        if not symbol or qty <= 0 or price <= 0:
            continue
        if side == "buy":
            lots.setdefault(symbol, []).append([qty, price])
            continue
        if side != "sell":
            continue
        remaining = qty
        queue = lots.setdefault(symbol, [])
        while remaining > 1e-10 and queue:
            lot_qty, lot_price = queue[0]
            matched = min(remaining, lot_qty)
            realised_by_symbol[symbol] = realised_by_symbol.get(symbol, 0.0) + matched * (price - lot_price)
            remaining -= matched
            lot_qty -= matched
            if lot_qty <= 1e-10:
                queue.pop(0)
            else:
                queue[0][0] = lot_qty
        if remaining > 1e-10:
            unmatched_sells[symbol] = unmatched_sells.get(symbol, 0.0) + remaining
    return {
        "realized_pnl": round(sum(realised_by_symbol.values()), 2),
        "realized_by_symbol": {k: round(v, 2) for k, v in sorted(realised_by_symbol.items())},
        "unmatched_sell_qty": {k: round(v, 6) for k, v in sorted(unmatched_sells.items())},
    }

