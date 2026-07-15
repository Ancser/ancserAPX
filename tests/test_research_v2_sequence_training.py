from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

import research_v2.sequence_training as training
from research_v2.sequence_pipeline import SequencePipelineSettings
from research_v2.validation import make_purged_walk_forward


def _panel_and_folds(seed: int = 17):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-01", periods=30)
    rows = []
    for date_number, date in enumerate(dates):
        for symbol_number in range(4):
            f1 = np.sin(date_number / 4.0) + symbol_number * 0.11
            f2 = rng.normal(scale=0.08)
            label = np.tanh(f1) + 0.15 * f2
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
        dates[:16],
        train_days=6,
        validation_days=4,
        test_days=4,
        purge_days=1,
        embargo_days=1,
        step_days=4,
        label_horizon=1,
    )
    assert len(folds) == 1
    return panel, dates, folds


def _settings(max_train_samples=8):
    return SequencePipelineSettings(
        sequence_length=3,
        gru_hidden_dim=4,
        transformer_d_model=4,
        transformer_heads=1,
        transformer_layers=1,
        transformer_feedforward=8,
        epochs=1,
        batch_size=16,
        max_train_samples=max_train_samples,
        max_parameters=10_000,
        learning_rate=0.01,
        weight_decay=0.0,
        random_seed=101,
        device="cpu",
        minimum_cross_section=3,
    )


def _root(tmp_path):
    root = tmp_path / "research_v2"
    root.mkdir()
    return root


def test_full_runner_checkpoints_complete_dates_and_resumes_without_fit(
    tmp_path, monkeypatch
):
    panel, dates, folds = _panel_and_folds()
    root = _root(tmp_path)
    events = []
    kwargs = dict(
        output_dir="runs/resume",
        selection_end=str(dates[21].date()),
        lockbox_start=str(dates[22].date()),
        embargo_days=1,
        settings=_settings(),
        endpoint_sampling="complete_date",
        research_root=root,
    )
    first = training.run_checkpointed_sequence_research(
        panel,
        ["f1", "f2"],
        folds,
        progress_callback=events.append,
        **kwargs,
    )

    run_dir = root / "runs" / "resume"
    fold_dir = run_dir / "folds" / folds[0].fold_id
    for directory in (fold_dir, run_dir / "selection", run_dir / "lockbox"):
        assert {path.name for path in directory.iterdir()} == {
            "predictions.parquet",
            "history.json",
            "settings.json",
            "manifest.json",
            "_SUCCESS",
        }
    assert (run_dir / "_SUCCESS").is_file()
    sampling = first.fold_records[0]["endpoint_sampling"]
    assert sampling["mode"] == "complete_date"
    assert sampling["selected_rows"] == 8
    assert sampling["selected_dates"] == 2
    assert sampling["complete_cross_section"] == 4
    assert {event["event"] for event in events} >= {
        "fold_started",
        "fold_completed",
        "selection_completed",
        "lockbox_started",
        "lockbox_completed",
        "run_completed",
    }

    def unexpected_fit(*args, **kwargs):
        raise AssertionError("a verified checkpoint must skip model fitting")

    monkeypatch.setattr(training, "_fit_model_pair", unexpected_fit)
    resumed_events = []
    second = training.run_checkpointed_sequence_research(
        panel,
        ["f1", "f2"],
        folds,
        progress_callback=resumed_events.append,
        **kwargs,
    )
    pd.testing.assert_frame_equal(
        first.selection_predictions.reset_index(drop=True),
        second.selection_predictions.reset_index(drop=True),
    )
    pd.testing.assert_frame_equal(
        first.lockbox_predictions.reset_index(drop=True),
        second.lockbox_predictions.reset_index(drop=True),
    )
    assert {event["event"] for event in resumed_events} >= {
        "fold_resumed",
        "selection_resumed",
        "lockbox_resumed",
        "run_completed",
    }


def test_checkpoint_tamper_is_rejected_before_retraining(tmp_path, monkeypatch):
    panel, dates, folds = _panel_and_folds()
    root = _root(tmp_path)
    kwargs = dict(
        output_dir="runs/tamper",
        selection_end=str(dates[21].date()),
        lockbox_start=str(dates[22].date()),
        embargo_days=1,
        settings=_settings(),
        endpoint_sampling="row",
        research_root=root,
    )
    training.run_checkpointed_sequence_selection(
        panel, ["f1", "f2"], folds, **kwargs
    )
    prediction_path = (
        root
        / "runs"
        / "tamper"
        / "folds"
        / folds[0].fold_id
        / "predictions.parquet"
    )
    with prediction_path.open("ab") as handle:
        handle.write(b"tampered")

    def unexpected_fit(*args, **kwargs):
        raise AssertionError("integrity failure must happen before fitting")

    monkeypatch.setattr(training, "_fit_model_pair", unexpected_fit)
    with pytest.raises(
        training.SequenceCheckpointIntegrityError,
        match="fingerprint mismatch",
    ):
        training.run_checkpointed_sequence_selection(
            panel, ["f1", "f2"], folds, **kwargs
        )


def test_future_mutation_keeps_earlier_fold_checkpoint_valid(tmp_path, monkeypatch):
    panel, dates, folds = _panel_and_folds()
    root = _root(tmp_path)
    events = []
    kwargs = dict(
        output_dir="runs/future-safe",
        selection_end=str(dates[21].date()),
        lockbox_start=str(dates[22].date()),
        embargo_days=1,
        settings=_settings(),
        endpoint_sampling="complete_date",
        research_root=root,
    )
    original = training.run_checkpointed_sequence_selection(
        panel,
        ["f1", "f2"],
        folds,
        progress_callback=events.append,
        **kwargs,
    )

    mutated = panel.copy(deep=True)
    future = pd.to_datetime(mutated["timestamp"]) > folds[0].test_end
    mutated.loc[future, ["f1", "f2"]] += 100_000.0

    def unexpected_fit(*args, **kwargs):
        raise AssertionError("future-only values must not invalidate an early fold")

    monkeypatch.setattr(training, "_fit_model_pair", unexpected_fit)
    resumed_events = []
    resumed = training.run_checkpointed_sequence_selection(
        mutated,
        ["f1", "f2"],
        folds,
        progress_callback=resumed_events.append,
        **kwargs,
    )
    pd.testing.assert_frame_equal(original.predictions, resumed.predictions)
    assert {event["event"] for event in resumed_events} >= {
        "fold_resumed",
        "selection_resumed",
    }


def test_none_train_cap_uses_every_natural_training_endpoint(tmp_path):
    panel, dates, folds = _panel_and_folds()
    root = _root(tmp_path)
    result = training.run_checkpointed_sequence_selection(
        panel,
        ["f1", "f2"],
        folds,
        output_dir="runs/no-cap",
        selection_end=str(dates[21].date()),
        lockbox_start=str(dates[22].date()),
        embargo_days=1,
        settings=replace(_settings(), max_train_samples=None),
        endpoint_sampling="complete_date",
        research_root=root,
    )
    # Six training dates minus the two-date sequence warmup, four symbols/day.
    assert result.fold_records[0]["endpoint_rows"]["train"] == 16
    assert result.fold_records[0]["endpoint_sampling"]["selected_dates"] is None
