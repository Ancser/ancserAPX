from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research_v2.neutralization import (
    NeutralizationSpec,
    build_claude1_factor_variants,
    neutralize_cross_sections,
    sector_exposure_diagnostics,
    top_n_sector_concentration,
)


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2025-01-02"] * 6 + ["2025-01-03"] * 6
            ),
            "symbol": list("ABCDEF") * 2,
            "value": [1.0, 2.0, 7.0, 10.0, 14.0, 18.0, 2.0, 5.0, 8.0, 20.0, 22.0, 30.0],
            "factor_ts_mom": [1.0, 2.0, 3.0, 6.0, 5.0, 4.0] * 2,
            "factor_rsi": [30.0, 40.0, 50.0, 70.0, 60.0, 80.0] * 2,
        }
    )


SECTORS = {"A": "S1", "B": "S1", "C": "S1", "D": "S2", "E": "S2", "F": "S2"}


def test_sector_residual_is_pure_date_local_and_matches_dummy_ols():
    source = _frame()
    before = source.copy(deep=True)
    transformed, audit = neutralize_cross_sections(
        source,
        ["value"],
        SECTORS,
        spec=NeutralizationSpec(
            method="sector_residual",
            min_sector_names=3,
            final_cross_section_rank=False,
        ),
    )

    pd.testing.assert_frame_equal(source, before)
    assert audit["rows"] == len(source)
    assert audit["unknown_rows"] == 0
    assert audit["minimum_date_sector_names"] == 3
    means = transformed.groupby(["timestamp", "sector"])["neutral__value"].mean()
    np.testing.assert_allclose(means, 0.0, atol=1e-12)

    first = transformed.loc[transformed["timestamp"] == pd.Timestamp("2025-01-02")]
    x = pd.get_dummies(first["sector"], drop_first=True, dtype=float)
    x.insert(0, "intercept", 1.0)
    y = first["value"].to_numpy(float)
    beta = np.linalg.lstsq(x.to_numpy(float), y, rcond=None)[0]
    ols_residual = y - x.to_numpy(float) @ beta
    np.testing.assert_allclose(first["neutral__value"], ols_residual, atol=1e-12)

    changed = source.copy()
    changed.loc[changed["timestamp"] == pd.Timestamp("2025-01-03"), "value"] *= 1000
    changed_result, _ = neutralize_cross_sections(
        changed,
        ["value"],
        SECTORS,
        spec=NeutralizationSpec(
            method="sector_residual", min_sector_names=3, final_cross_section_rank=False
        ),
    )
    np.testing.assert_allclose(
        transformed.loc[transformed["timestamp"] == pd.Timestamp("2025-01-02"), "neutral__value"],
        changed_result.loc[changed_result["timestamp"] == pd.Timestamp("2025-01-02"), "neutral__value"],
    )


def test_sector_zscore_and_rank_are_centered_inside_every_sector():
    source = _frame()
    for method in ("sector_zscore", "within_sector_rank"):
        transformed, _ = neutralize_cross_sections(
            source,
            ["value"],
            SECTORS,
            spec=NeutralizationSpec(
                method=method, min_sector_names=3, final_cross_section_rank=False
            ),
        )
        groups = transformed.groupby(["timestamp", "sector"])["neutral__value"]
        np.testing.assert_allclose(groups.mean(), 0.0, atol=1e-12)
        if method == "sector_zscore":
            np.testing.assert_allclose(
                groups.apply(lambda values: np.sqrt(np.mean(np.square(values)))),
                1.0,
                atol=1e-12,
            )
        else:
            for values in groups:
                np.testing.assert_allclose(sorted(values[1]), [-0.5, 0.0, 0.5])


def test_unknown_and_too_small_sector_fail_closed():
    source = _frame()
    missing = dict(SECTORS)
    missing.pop("F")
    with pytest.raises(ValueError, match="unknown symbols"):
        neutralize_cross_sections(
            source,
            ["value"],
            missing,
            spec=NeutralizationSpec(method="sector_residual", min_sector_names=2),
        )
    with pytest.raises(ValueError, match="below min_sector_names"):
        neutralize_cross_sections(
            source,
            ["value"],
            SECTORS,
            spec=NeutralizationSpec(method="sector_residual", min_sector_names=4),
        )


def test_valid_count_zero_variance_and_unknown_rank_scale_fail_safely():
    source = _frame().iloc[:6].copy()
    missing_value = source.copy()
    missing_value.loc[missing_value["symbol"] == "C", "value"] = np.nan
    with pytest.raises(ValueError, match="min valid names"):
        neutralize_cross_sections(
            missing_value,
            ["value"],
            SECTORS,
            spec=NeutralizationSpec(method="sector_residual", min_sector_names=3),
        )

    constant = source.copy()
    constant.loc[constant["symbol"].isin(["A", "B", "C"]), "value"] = 1.0
    with pytest.raises(ValueError, match="zero-variance"):
        neutralize_cross_sections(
            constant,
            ["value"],
            SECTORS,
            spec=NeutralizationSpec(method="sector_zscore", min_sector_names=3),
        )

    with_unknown = pd.concat(
        [
            source,
            pd.DataFrame(
                {
                    "timestamp": [pd.Timestamp("2025-01-02")],
                    "symbol": ["U"],
                    "value": [100.0],
                    "factor_ts_mom": [100.0],
                    "factor_rsi": [100.0],
                }
            ),
        ],
        ignore_index=True,
    )
    transformed, _ = neutralize_cross_sections(
        with_unknown,
        ["value"],
        SECTORS,
        spec=NeutralizationSpec(
            method="sector_residual",
            min_sector_names=3,
            unknown_policy="passthrough_global",
            final_cross_section_rank=True,
        ),
    )
    assert transformed.loc[transformed["symbol"] == "U", "neutral__value"].iloc[0] == pytest.approx(0.5)
    assert transformed.loc[transformed["symbol"] != "U", "neutral__value"].between(-0.5, 0.5).all()
    with pytest.raises(ValueError, match="requires final_cross_section_rank"):
        neutralize_cross_sections(
            with_unknown,
            ["value"],
            SECTORS,
            spec=NeutralizationSpec(
                method="sector_residual",
                min_sector_names=3,
                unknown_policy="passthrough_global",
                final_cross_section_rank=False,
            ),
        )


def test_claude1_raw_variant_has_expected_orientation_and_formula():
    source = _frame()
    result, audits = build_claude1_factor_variants(
        source,
        SECTORS,
        methods=("none", "sector_residual"),
        min_sector_names=3,
    )
    day = source["timestamp"] == pd.Timestamp("2025-01-02")
    mom = source.loc[day, "factor_ts_mom"].rank(method="average")
    rsi = source.loc[day, "factor_rsi"].rank(method="average")
    mom = (mom - 1) / (len(mom) - 1) - 0.5
    rsi = (rsi - 1) / (len(rsi) - 1) - 0.5
    expected = 0.70 * mom - 0.30 * rsi
    np.testing.assert_allclose(
        result.loc[day, "score_claude1_factorwise__none"], expected
    )
    assert set(audits) == {"none", "sector_residual"}


def test_exposure_and_top_n_concentration_detect_sector_bias():
    source = _frame().iloc[:6].copy()
    source["sector"] = source["symbol"].map(SECTORS)
    summary, daily = sector_exposure_diagnostics(source, score_col="value")
    assert len(daily) == 1
    assert 0.0 < summary["mean_sector_r2"] <= 1.0

    concentration, selected = top_n_sector_concentration(
        source, score_col="value", top_n=3
    )
    assert len(selected) == 1
    assert concentration["mean_max_sector_share"] == pytest.approx(1.0)
    assert concentration["mean_effective_sectors"] == pytest.approx(1.0)
