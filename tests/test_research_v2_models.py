from __future__ import annotations

import numpy as np
import pytest
import torch

from research_v2.models import (
    CalibratedRankEnsemble,
    DeterministicHistGradientBoosting,
    DeterministicRidge,
    constrain_ensemble_weights,
    cross_sectional_percentile_rank,
)
from research_v2.sequence import (
    GRUSequenceRegressor,
    LazySequenceDataset,
    SmallGRU,
    SmallTransformer,
    TransformerSequenceRegressor,
    count_trainable_parameters,
)


def _tabular_data(seed: int = 7):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(240, 6))
    X[::17, 2] = np.nan
    y = 0.6 * np.nan_to_num(X[:, 0]) - 0.25 * np.nan_to_num(X[:, 1])
    y += 0.1 * rng.normal(size=len(y))
    return X, y


def test_cross_sectional_percentile_rank_is_datewise_and_centered():
    scores = np.array([10.0, 30.0, 20.0, 2.0, 2.0, 8.0, np.nan])
    dates = np.array(["d1", "d1", "d1", "d2", "d2", "d2", "d2"])
    ranked = cross_sectional_percentile_rank(scores, dates)

    np.testing.assert_allclose(ranked[:3], [-0.5, 0.5, 0.0])
    np.testing.assert_allclose(ranked[3:6], [-0.25, -0.25, 0.5])
    assert np.isnan(ranked[6])
    assert ranked.shape == scores.shape
    assert np.nanmean(ranked[:3]) == pytest.approx(0.0)
    assert np.nanmean(ranked[3:]) == pytest.approx(0.0)


def test_deterministic_ridge_shape_and_repeatability():
    X, y = _tabular_data()
    first = DeterministicRidge(alpha=2.0).fit(X, y)
    second = DeterministicRidge(alpha=2.0).fit(X, y)

    prediction_1 = first.predict(X)
    prediction_2 = second.predict(X)
    assert prediction_1.shape == (len(X),)
    np.testing.assert_array_equal(prediction_1, prediction_2)


def test_hist_gbdt_is_deterministic_and_forbids_random_early_stopping():
    X, y = _tabular_data()
    kwargs = dict(
        max_iter=35,
        max_leaf_nodes=7,
        min_samples_leaf=8,
        random_state=19,
    )
    first = DeterministicHistGradientBoosting(**kwargs).fit(X, y)
    second = DeterministicHistGradientBoosting(**kwargs).fit(X, y)

    np.testing.assert_array_equal(first.predict(X), second.predict(X))
    assert first.estimator_.early_stopping is False
    assert first.estimator_.validation_fraction is None
    with pytest.raises(ValueError, match="early-stopping"):
        DeterministicHistGradientBoosting(early_stopping=True)


def test_ensemble_caps_shrinks_and_never_raw_averages():
    unshrunk = constrain_ensemble_weights(
        [0.95, 0.05, 0.0], max_weight=0.6, shrinkage=0.0
    )
    shrunk = constrain_ensemble_weights(
        [0.95, 0.05, 0.0], max_weight=0.6, shrinkage=0.5
    )
    equal = np.full(3, 1 / 3)
    assert unshrunk.sum() == pytest.approx(1.0)
    assert np.all(unshrunk >= 0)
    assert unshrunk.max() <= 0.6 + 1e-12
    assert np.linalg.norm(shrunk - equal) < np.linalg.norm(unshrunk - equal)

    raw = {
        "large_scale": np.array([0.0, 100.0, 200.0]),
        "small_scale": np.array([0.2, 0.1, 0.0]),
    }
    ensemble = CalibratedRankEnsemble(
        {"large_scale": 1.0, "small_scale": 1.0}, max_weight=0.6
    )
    prediction = ensemble.predict_from_raw(raw, ["same"] * 3)
    np.testing.assert_allclose(prediction, 0.0)
    assert not np.allclose(prediction, np.mean(np.column_stack(list(raw.values())), axis=1))
    with pytest.raises(ValueError, match="centered percentile ranks"):
        ensemble.combine_calibrated(raw)


def _sequence_data():
    rng = np.random.default_rng(31)
    features = rng.normal(size=(28, 3)).astype(np.float32)
    groups = np.repeat(["A", "B"], 14)
    targets = (0.5 * features[:, 0] - 0.2 * features[:, 1]).astype(np.float32)
    return features, groups, targets


def test_lazy_sequence_dataset_shapes_and_group_boundaries():
    features = np.arange(24, dtype=np.float32).reshape(12, 2)
    groups = np.repeat(["A", "B"], 6)
    targets = np.arange(12, dtype=np.float32)
    dataset = LazySequenceDataset(
        features, targets, groups=groups, sequence_length=3
    )

    assert dataset.storage_shape == features.shape
    assert len(dataset) == 8
    first_x, first_y = dataset[0]
    second_group_x, second_group_y = dataset[4]
    assert first_x.shape == (3, 2)
    np.testing.assert_array_equal(first_x.numpy(), features[:3])
    assert first_y.item() == 2.0
    np.testing.assert_array_equal(second_group_x.numpy(), features[6:9])
    assert second_group_y.item() == 8.0


def test_small_sequence_models_shape_and_parameter_limit():
    inputs = torch.zeros(5, 7, 3)
    gru = SmallGRU(3, hidden_dim=8, max_parameters=10_000)
    transformer = SmallTransformer(
        3,
        d_model=8,
        nhead=2,
        num_layers=1,
        dim_feedforward=16,
        max_sequence_length=7,
        max_parameters=10_000,
    )
    assert gru(inputs).shape == (5,)
    assert transformer(inputs).shape == (5,)
    assert count_trainable_parameters(gru) <= 10_000
    assert count_trainable_parameters(transformer) <= 10_000
    with pytest.raises(ValueError, match="exceeding limit"):
        SmallGRU(3, hidden_dim=8, max_parameters=10)


def test_gru_fit_predict_is_deterministic_on_cpu():
    features, groups, targets = _sequence_data()
    dataset = LazySequenceDataset(
        features, targets, groups=groups, sequence_length=4
    )
    kwargs = dict(
        input_dim=3,
        hidden_dim=8,
        seed=101,
        device="cpu",
        learning_rate=0.01,
        weight_decay=0.0,
    )
    first = GRUSequenceRegressor(**kwargs).fit(
        dataset, epochs=2, batch_size=6
    )
    second = GRUSequenceRegressor(**kwargs).fit(
        dataset, epochs=2, batch_size=6
    )
    prediction_1 = first.predict(dataset, batch_size=5)
    prediction_2 = second.predict(dataset, batch_size=5)
    assert prediction_1.shape == (len(dataset),)
    np.testing.assert_allclose(prediction_1, prediction_2, rtol=0, atol=0)


def test_transformer_fit_predict_runs_on_cpu_without_random_validation_split():
    features, groups, targets = _sequence_data()
    dataset = LazySequenceDataset(
        features, targets, groups=groups, sequence_length=4
    )
    model = TransformerSequenceRegressor(
        3,
        d_model=8,
        nhead=2,
        num_layers=1,
        dim_feedforward=16,
        max_sequence_length=4,
        seed=9,
        device="cpu",
        learning_rate=0.005,
        weight_decay=0.0,
    )
    with pytest.raises(ValueError, match="validation_dataset"):
        model.fit(dataset, epochs=1, patience=1)
    model.fit(dataset, epochs=1, batch_size=8)
    prediction = model.predict(dataset, batch_size=8)
    assert prediction.shape == (len(dataset),)
    assert np.isfinite(prediction).all()
