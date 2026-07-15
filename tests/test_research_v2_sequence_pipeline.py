from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research_v2.models import CalibratedRankEnsemble
from research_v2.sequence import LazySequenceDataset
from research_v2.sequence_pipeline import (
    SequencePipelineSettings,
    build_fold_sequence_datasets,
    run_sequence_research,
    run_sequence_walk_forward,
)
from research_v2.validation import make_purged_walk_forward


def _panel(seed: int = 11):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-01", periods=30)
    rows = []
    for date_number, date in enumerate(dates):
        for symbol_number in range(4):
            f1 = np.sin(date_number / 5.0) + symbol_number * 0.15
            f2 = rng.normal(scale=0.10)
            label = np.tanh(f1) + 0.2 * f2
            rows.append(
                {
                    "timestamp": date,
                    "execution_timestamp": date + pd.offsets.BDay(1),
                    "symbol": f"S{symbol_number}",
                    "f1": f1,
                    "f2": f2,
                    "label_rank": label,
                    "label_residual": label,
                    "sample_weight": 0.25,
                }
            )
    panel = pd.DataFrame(rows)
    folds = make_purged_walk_forward(
        dates[:22],
        train_days=6,
        validation_days=4,
        test_days=4,
        purge_days=1,
        embargo_days=1,
        step_days=4,
        label_horizon=1,
    )
    return panel, dates, folds


def _settings():
    return SequencePipelineSettings(
        sequence_length=3,
        gru_hidden_dim=4,
        transformer_d_model=4,
        transformer_heads=1,
        transformer_layers=1,
        transformer_feedforward=8,
        epochs=1,
        batch_size=16,
        max_train_samples=16,
        max_parameters=10_000,
        learning_rate=0.01,
        weight_decay=0.0,
        random_seed=37,
        device="cpu",
        minimum_cross_section=3,
    )


def test_lazy_dataset_accepts_explicit_endpoint_mask_or_indices():
    features = np.arange(24, dtype=np.float32).reshape(12, 2)
    groups = np.repeat(["A", "B"], 6)
    mask = np.zeros(12, dtype=bool)
    mask[[2, 4, 8, 11]] = True

    masked = LazySequenceDataset(
        features,
        groups=groups,
        sequence_length=3,
        allowed_endpoint_mask=mask,
    )
    indexed = LazySequenceDataset(
        features,
        groups=groups,
        sequence_length=3,
        allowed_endpoint_indices=[2, 4, 8, 11],
    )
    np.testing.assert_array_equal(masked.end_indices, [2, 4, 8, 11])
    np.testing.assert_array_equal(indexed.end_indices, masked.end_indices)
    with pytest.raises(ValueError, match="mutually exclusive"):
        LazySequenceDataset(
            features,
            groups=groups,
            sequence_length=3,
            allowed_endpoint_mask=mask,
            allowed_endpoint_indices=[2],
        )


def test_fold_datasets_are_disjoint_capped_and_lookback_only():
    panel, _, folds = _panel()
    first = build_fold_sequence_datasets(
        panel,
        ["f1", "f2"],
        folds[0],
        sequence_length=3,
        max_train_samples=10,
        random_seed=5,
    )
    repeated = build_fold_sequence_datasets(
        panel,
        ["f1", "f2"],
        folds[0],
        sequence_length=3,
        max_train_samples=10,
        random_seed=5,
    )

    np.testing.assert_array_equal(first.train.end_indices, repeated.train.end_indices)
    assert len(first.train) == 10
    assert not first.test.has_targets
    endpoint_sets = [
        set(first.train.end_indices),
        set(first.validation.end_indices),
        set(first.test.end_indices),
    ]
    assert endpoint_sets[0].isdisjoint(endpoint_sets[1])
    assert endpoint_sets[0].isdisjoint(endpoint_sets[2])
    assert endpoint_sets[1].isdisjoint(endpoint_sets[2])

    test_dates = first.panel.iloc[first.test.end_indices]["timestamp"]
    assert test_dates.between(folds[0].test_start, folds[0].test_end).all()
    endpoint = int(first.test.end_indices[0])
    sequence = first.test[0].numpy()
    expected = first.panel.iloc[endpoint - 2 : endpoint + 1][["f1", "f2"]].to_numpy(
        dtype=np.float32
    )
    np.testing.assert_array_equal(sequence, expected)
    assert first.panel.iloc[endpoint - 2]["timestamp"] <= first.panel.iloc[endpoint]["timestamp"]


def test_future_feature_mutation_cannot_change_earlier_oos_predictions_or_indices():
    panel, _, folds = _panel()
    early_fold = [folds[0]]
    settings = _settings()
    original, _ = run_sequence_walk_forward(
        panel, ["f1", "f2"], early_fold, settings=settings
    )

    mutated = panel.copy()
    future = pd.to_datetime(mutated["timestamp"]) > folds[0].test_end
    mutated.loc[future, ["f1", "f2"]] += 100_000.0
    changed, _ = run_sequence_walk_forward(
        mutated, ["f1", "f2"], early_fold, settings=settings
    )

    columns = [
        "timestamp",
        "symbol",
        "endpoint_row",
        "source_row",
        "score_gru",
        "score_transformer",
        "score_sequence_equal",
    ]
    pd.testing.assert_frame_equal(
        original[columns].reset_index(drop=True),
        changed[columns].reset_index(drop=True),
        check_exact=True,
    )


def test_full_sequence_research_has_unique_oos_and_prelocked_lockbox_settings():
    panel, dates, folds = _panel()
    settings = _settings()
    result = run_sequence_research(
        panel,
        ["f1", "f2"],
        folds,
        selection_end=str(dates[21].date()),
        lockbox_start=str(dates[22].date()),
        embargo_days=1,
        settings=settings,
    )

    selection = result.selection_predictions
    lockbox = result.lockbox_predictions
    assert not selection.duplicated(["timestamp", "symbol"]).any()
    assert not lockbox.duplicated(["timestamp", "symbol"]).any()
    assert (selection["fold_id"] != "LOCKBOX").all()
    assert (lockbox["fold_id"] == "LOCKBOX").all()
    assert (pd.to_datetime(lockbox["timestamp"]) >= dates[22]).all()
    assert (
        pd.Timestamp(result.locked_settings["selection_lock_end"])
        < pd.Timestamp(result.locked_settings["lockbox_start"])
    )
    assert result.locked_settings["selection_rows_used"] > 0

    for column in ["score_gru", "score_transformer"]:
        means = selection.groupby("timestamp")[column].mean()
        np.testing.assert_allclose(means, 0.0, atol=1e-12)
        assert selection[column].between(-0.5, 0.5).all()

    weights = result.locked_settings["ensemble_weights"]
    assert sum(weights.values()) == pytest.approx(1.0)
    assert min(weights.values()) >= 0.0
    assert max(weights.values()) <= settings.ensemble_single_model_cap + 1e-12
    ensemble = CalibratedRankEnsemble(
        weights,
        max_weight=settings.ensemble_single_model_cap,
        shrinkage=0.0,
    )
    expected = ensemble.combine_calibrated(
        {
            "gru": lockbox["score_gru"].to_numpy(dtype=float),
            "transformer": lockbox["score_transformer"].to_numpy(dtype=float),
        }
    )
    np.testing.assert_allclose(lockbox["score_sequence_locked"], expected)
