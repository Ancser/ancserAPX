from backend.alpha.factors import (
    ALL_FACTORS,
    FACTOR_PRESETS,
    LEGACY_FACTOR_META,
    SECONDARY_FACTORS,
    STRATEGY_PRESETS,
)


REMOVED_FACTORS = {
    "Momentum 12-1",
    "KDJ Reversal Entry",
    "KDJ PD50 RSI Entry",
    "Short Squeeze Proxy",
    "Unicorn Edge",
    "Skew",
    "Alpha 101",
    "Drift-Reversion",
    "Rank Acceleration",
    "Sector Rank",
}


def test_unvalidated_factors_are_not_selectable():
    assert REMOVED_FACTORS.isdisjoint(ALL_FACTORS)
    assert REMOVED_FACTORS.issubset(LEGACY_FACTOR_META)
    assert SECONDARY_FACTORS == []


def test_presets_only_reference_selectable_factors():
    selectable = set(ALL_FACTORS)
    assert all(set(factors) <= selectable for factors in FACTOR_PRESETS.values())
    for strategy in STRATEGY_PRESETS.values():
        for sleeve in strategy.get("sleeves", []):
            assert set(sleeve.get("factors", [])) <= selectable


def test_removed_strategy_presets_are_not_offered():
    assert "Claude #2" not in STRATEGY_PRESETS
    assert "Short Squeeze #1" not in STRATEGY_PRESETS
