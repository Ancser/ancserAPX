"""
Shared portfolio-construction logic.

⚠️  PARITY-CRITICAL MODULE — see SKILL.md.
This is the SINGLE source of truth for turning factor data into target weights.
BOTH the backtest engine (backend/backtest/engine.py :: run_strategy) and the
live executor (backend/execution/strategy.py :: calculate_targets) call these
functions, so a backtest and the live account that ran the same preset produce
identical decisions. DO NOT re-implement any of this logic anywhere else — if
you change a rule here, both backtest and live change together, which is the
entire point.

Every function is pure (no I/O, no Alpaca, no store): inputs are pandas Series
indexed by symbol for ONE rebalance date, plus prior winner-lock state.
"""

from typing import Dict, List, Tuple
import pandas as pd

from backend.alpha.factors import RUNTIME_FACTOR_META
from backend.alpha.neutralization import SECTOR_MAP

_COL = {n: m["col"] for n, m in RUNTIME_FACTOR_META.items()}
_DESCENDING = {n for n, m in RUNTIME_FACTOR_META.items() if m["descending"]}


def composite_score(
    factor_values: Dict[str, pd.Series],
    factors: List[str],
    weights: Dict[str, float],
) -> pd.Series:
    """Cross-sectional weighted percentile-rank score for one date.

    factor_values[f] = raw factor Series (indexed by symbol) for the date.
    Weights are renormalised over the factors actually present. Returns a score
    Series indexed by symbol (higher = better)."""
    valid = [f for f in factors if f in factor_values and factor_values[f] is not None]
    if not valid:
        return pd.Series(dtype=float)
    w = {f: float(weights.get(f, 0.0)) for f in valid}
    tot = sum(w.values())
    if tot <= 0:
        w = {f: 1.0 / len(valid) for f in valid}
    else:
        w = {f: v / tot for f, v in w.items()}

    score = None
    for f in valid:
        ascending = f not in _DESCENDING
        ranked = factor_values[f].rank(pct=True, ascending=ascending).fillna(0.5) * w[f]
        score = ranked if score is None else score.add(ranked, fill_value=0.0)
    return score if score is not None else pd.Series(dtype=float)


def core_sleeve_weights(score: pd.Series, top_n: int) -> Dict[str, float]:
    """Equal-weight long-only top_n. Within-sleeve weights sum to 1.0."""
    if score is None or score.empty:
        return {}
    top = score.nlargest(min(top_n, len(score))).index.tolist()
    if not top:
        return {}
    w = 1.0 / len(top)
    return {s: w for s in top}


def sector_balanced_symbols(score: pd.Series, top_n: int) -> List[str]:
    """Pick top names with near-equal representation across known sectors."""
    if score is None or score.empty:
        return []
    ranked = score.dropna().sort_values(ascending=False)
    grouped: Dict[str, List[str]] = {}
    for symbol in ranked.index:
        sector = SECTOR_MAP.get(str(symbol), "Unknown")
        if sector != "Unknown":
            grouped.setdefault(sector, []).append(symbol)

    # Sector balancing is not defensible with fewer than two mapped sectors.
    if len(grouped) < 2:
        return ranked.head(min(top_n, len(ranked))).index.tolist()

    selected: List[str] = []
    limit = min(max(1, int(top_n)), sum(len(names) for names in grouped.values()))
    offsets = {sector: 0 for sector in grouped}
    while len(selected) < limit:
        available = [
            sector for sector, names in grouped.items()
            if offsets[sector] < len(names)
        ]
        if not available:
            break
        # Each round gives every available sector one slot. Sectors whose next
        # candidate scores highest receive any final partial-round slots.
        available.sort(
            key=lambda sector: float(ranked.loc[grouped[sector][offsets[sector]]]),
            reverse=True,
        )
        for sector in available:
            selected.append(grouped[sector][offsets[sector]])
            offsets[sector] += 1
            if len(selected) >= limit:
                break
    return selected


def sector_balanced_weights(score: pd.Series, top_n: int) -> Dict[str, float]:
    """Equal-weight sectors, then equal-weight names inside each sector."""
    selected = sector_balanced_symbols(score, top_n)
    if not selected:
        return {}
    groups: Dict[str, List[str]] = {}
    for symbol in selected:
        groups.setdefault(SECTOR_MAP.get(str(symbol), "Unknown"), []).append(symbol)
    sector_weight = 1.0 / len(groups)
    return {
        symbol: sector_weight / len(symbols)
        for symbols in groups.values()
        for symbol in symbols
    }


def equalize_sector_exposure(weights: Dict[str, float]) -> Dict[str, float]:
    """Preserve gross exposure while equalizing represented sector budgets."""
    if not weights:
        return {}
    groups: Dict[str, List[str]] = {}
    for symbol, weight in weights.items():
        if abs(float(weight)) > 0:
            groups.setdefault(SECTOR_MAP.get(str(symbol), "Unknown"), []).append(symbol)
    if len(groups) < 2:
        return dict(weights)
    gross = sum(abs(float(weight)) for weight in weights.values())
    target_sector_gross = gross / len(groups)
    adjusted: Dict[str, float] = {}
    for symbols in groups.values():
        sector_gross = sum(abs(float(weights[symbol])) for symbol in symbols)
        if sector_gross <= 0:
            continue
        scale = target_sector_gross / sector_gross
        for symbol in symbols:
            adjusted[symbol] = float(weights[symbol]) * scale
    return adjusted


def satellite_weights(
    score: pd.Series,
    price: pd.Series,
    prior: Dict[str, Dict],
    top_n: int,
    lock_rules: Dict[str, float],
    sector_balance: bool = False,
) -> Tuple[Dict[str, float], Dict[str, Dict]]:
    """Winner-lock sleeve.

    State (prior / returned `new`): {symbol: {"entry": entry_price, "locked": bool}}.
    A held name becomes locked once its return since entry (price/entry - 1) is
    >= profit_lock while ranked within lock_rank. Locked names are retained
    across rebalances (even if they drop out of top_n, as long as still inside
    top_n*2) and capped at max_weight; remaining capital is split equally among
    the unlocked top_n names. Within-sleeve weights sum to ~1.0."""
    profit_lock = float(lock_rules.get("profit_lock", 0.30))
    max_weight = float(lock_rules.get("max_weight", 0.15))
    lock_rank = int(lock_rules.get("lock_rank", 10))

    prior = {k: dict(v) for k, v in (prior or {}).items()}
    if score is None or score.empty:
        return {}, prior

    ranked = score.sort_values(ascending=False)
    rank_of = {s: i + 1 for i, s in enumerate(ranked.index)}
    top = (
        sector_balanced_symbols(score, top_n)
        if sector_balance else ranked.head(min(top_n, len(ranked))).index.tolist()
    )

    def _px(sym):
        try:
            v = float(price.get(sym)) if sym in price.index else 0.0
        except Exception:
            v = 0.0
        return 0.0 if (v is None or pd.isna(v)) else v

    # Promote eligible held winners to locked (by realised return since entry).
    for sym, p in prior.items():
        entry = float(p.get("entry", 0.0) or 0.0)
        px = _px(sym)
        if entry > 0 and px > 0:
            ret = px / entry - 1.0
            if (not p.get("locked")) and ret >= profit_lock and rank_of.get(sym, 10**9) <= lock_rank:
                p["locked"] = True

    keep_locked = [s for s, p in prior.items()
                   if p.get("locked") and rank_of.get(s, 10**9) <= top_n * 2]
    target = list(dict.fromkeys(keep_locked + top))

    new: Dict[str, Dict] = {}
    for s in target:
        if s in prior:
            new[s] = prior[s]
        else:
            new[s] = {"entry": _px(s), "locked": False}

    locked = [s for s in target if new[s].get("locked")]
    unlocked = [s for s in target if not new[s].get("locked")]
    locked_w = {s: min(max_weight, 1.0 / max(len(target), 1)) for s in locked}
    used = sum(locked_w.values())
    per = (1.0 - used) / len(unlocked) if unlocked else 0.0
    weights = {s: locked_w.get(s, per) for s in target}
    return weights, new


def combined_target_weights(
    sleeves: List[Dict],
    factor_values: Dict[str, pd.Series],
    price: pd.Series,
    state: Dict[str, Dict],
    top_n: int,
    lock_rules: Dict[str, float],
    leverage: float,
    sector_balance: bool = False,
) -> Tuple[Dict[str, float], Dict[str, Dict]]:
    """Blend all sleeves into one combined target-weight dict (already scaled by
    `leverage`, so the weights sum to ~`leverage`). `state` keys winner-lock
    holdings by sleeve name. Returns (target_weights, new_state).

    This is the function that guarantees backtest/live parity: feed it the same
    factor_values + price + state and it returns the same weights regardless of
    caller."""
    lock_rules = lock_rules or {}
    combined: Dict[str, float] = {}
    new_state = {k: dict(v) for k, v in (state or {}).items()}

    for sl in sleeves:
        alloc = float(sl.get("alloc", 0.0))
        if alloc <= 0:
            continue
        name = sl.get("name", "sleeve")
        sl_top = int(sl.get("top_n", top_n))
        score = composite_score(factor_values, sl.get("factors", []), sl.get("weights", {}))
        if sl.get("winner_lock"):
            w, st = satellite_weights(
                score, price, new_state.get(name, {}), sl_top, lock_rules,
                sector_balance=sector_balance,
            )
            new_state[name] = st
        else:
            w = (
                sector_balanced_weights(score, sl_top)
                if sector_balance else core_sleeve_weights(score, sl_top)
            )
        for s, wv in w.items():
            combined[s] = combined.get(s, 0.0) + alloc * wv * leverage

    if sector_balance:
        combined = equalize_sector_exposure(combined)
    return combined, new_state
