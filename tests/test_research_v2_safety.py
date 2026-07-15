from __future__ import annotations

import importlib
import json
import os
from pathlib import Path

import pytest

from research_v2 import snapshot as snapshot_module
from research_v2.safety import (
    UnsafeResearchImport,
    UnsafeResearchPath,
    ensure_research_output_path,
    offline_context,
)
from research_v2.snapshot import (
    SnapshotSourceChanged,
    SnapshotVerificationError,
    create_snapshot,
    verify_snapshot,
)


def _source_tree(tmp_path: Path) -> tuple[Path, Path, Path]:
    source = tmp_path / "production_fixture"
    store = source / "data" / "store"
    store.mkdir(parents=True)
    (store / "A.parquet").write_bytes(b"parquet-fixture-a")
    (store / "nested").mkdir()
    (store / "nested" / "B.parquet").write_bytes(b"parquet-fixture-b")
    manifest = source / "data" / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "A": {"last_date": "2026-07-09", "row_count": 1},
                "B": {"last_date": "2026-07-09", "row_count": 1},
            }
        ),
        encoding="utf-8",
    )
    return source, store, manifest


def test_output_paths_cannot_escape_research_root(tmp_path: Path) -> None:
    root = tmp_path / "research_v2"
    assert ensure_research_output_path("runs/x.json", research_root=root) == (
        root / "runs" / "x.json"
    ).resolve()

    with pytest.raises(UnsafeResearchPath):
        ensure_research_output_path("../production.json", research_root=root)
    with pytest.raises(UnsafeResearchPath):
        ensure_research_output_path(tmp_path / "outside.json", research_root=root)
    with pytest.raises(UnsafeResearchPath):
        ensure_research_output_path("x", research_root=tmp_path / "wrong_name")


def test_offline_context_clears_and_restores_only_local_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("APCA_API_KEY_ID", "live-key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "live-secret")
    monkeypatch.setenv("PAPER_TRADING_TEST", "true")
    monkeypatch.setenv("UNRELATED_SETTING", "kept")

    with offline_context(extra_blocked=("research_v2_forbidden_probe",)):
        assert "APCA_API_KEY_ID" not in os.environ
        assert "ALPACA_SECRET_KEY" not in os.environ
        assert "PAPER_TRADING_TEST" not in os.environ
        assert os.environ["UNRELATED_SETTING"] == "kept"
        assert os.environ["ANCSER_RESEARCH_OFFLINE"] == "1"
        os.environ["APCA_CREATED_INSIDE_CONTEXT"] = "must-not-leak"
        with pytest.raises(UnsafeResearchImport):
            importlib.import_module("research_v2_forbidden_probe.child")

    assert os.environ["APCA_API_KEY_ID"] == "live-key"
    assert os.environ["ALPACA_SECRET_KEY"] == "live-secret"
    assert os.environ["PAPER_TRADING_TEST"] == "true"
    assert "APCA_CREATED_INSIDE_CONTEXT" not in os.environ
    assert "ANCSER_RESEARCH_OFFLINE" not in os.environ


def test_snapshot_is_a_verified_physical_atomic_copy(tmp_path: Path) -> None:
    _, store, manifest = _source_tree(tmp_path)
    root = tmp_path / "research_v2"

    completed = create_snapshot(
        store_dir=store,
        manifest_path=manifest,
        research_root=root,
        snapshot_id="fixture-001",
    )

    assert completed == root / "snapshots" / "fixture-001"
    assert (completed / "store" / "A.parquet").read_bytes() == b"parquet-fixture-a"
    assert (completed / "store" / "nested" / "B.parquet").read_bytes() == b"parquet-fixture-b"
    assert not os.path.samefile(store / "A.parquet", completed / "store" / "A.parquet")
    assert not list((root / "snapshots").glob(".*.tmp-*"))

    metadata = verify_snapshot(completed, research_root=root)
    assert metadata["snapshot_id"] == "fixture-001"
    assert metadata["file_count"] == 2
    assert len(metadata["manifest"]["sha256"]) == 64


def test_snapshot_rejects_traversal_id_before_writing(tmp_path: Path) -> None:
    _, store, manifest = _source_tree(tmp_path)
    root = tmp_path / "research_v2"
    with pytest.raises(UnsafeResearchPath):
        create_snapshot(
            store_dir=store,
            manifest_path=manifest,
            research_root=root,
            snapshot_id="../escape",
        )
    assert not (tmp_path / "escape").exists()


def test_source_mutation_aborts_without_partial_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, store, manifest = _source_tree(tmp_path)
    root = tmp_path / "research_v2"
    original_copy = snapshot_module._copy_file
    mutated = False

    def copy_then_mutate(source: Path, destination: Path):
        nonlocal mutated
        result = original_copy(source, destination)
        if source.name == "A.parquet" and not mutated:
            source.write_bytes(source.read_bytes() + b"-changed")
            mutated = True
        return result

    monkeypatch.setattr(snapshot_module, "_copy_file", copy_then_mutate)
    with pytest.raises(SnapshotSourceChanged):
        create_snapshot(
            store_dir=store,
            manifest_path=manifest,
            research_root=root,
            snapshot_id="must-abort",
        )

    snapshots = root / "snapshots"
    assert not (snapshots / "must-abort").exists()
    assert not list(snapshots.glob(".*.tmp-*"))


def test_manifest_mutation_aborts_without_partial_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, store, manifest = _source_tree(tmp_path)
    root = tmp_path / "research_v2"
    original_copy = snapshot_module._copy_file
    mutated = False

    def copy_then_mutate(source: Path, destination: Path):
        nonlocal mutated
        result = original_copy(source, destination)
        if source == manifest and not mutated:
            source.write_text('{"changed": true}', encoding="utf-8")
            mutated = True
        return result

    monkeypatch.setattr(snapshot_module, "_copy_file", copy_then_mutate)
    with pytest.raises(SnapshotSourceChanged):
        create_snapshot(
            store_dir=store,
            manifest_path=manifest,
            research_root=root,
            snapshot_id="manifest-must-abort",
        )

    snapshots = root / "snapshots"
    assert not (snapshots / "manifest-must-abort").exists()
    assert not list(snapshots.glob(".*.tmp-*"))


def test_verify_snapshot_detects_tampering(tmp_path: Path) -> None:
    _, store, manifest = _source_tree(tmp_path)
    root = tmp_path / "research_v2"
    completed = create_snapshot(
        store_dir=store,
        manifest_path=manifest,
        research_root=root,
        snapshot_id="tamper-check",
    )
    (completed / "store" / "A.parquet").write_bytes(b"tampered")
    with pytest.raises(SnapshotVerificationError, match="changed|fingerprint"):
        verify_snapshot(completed, research_root=root)
