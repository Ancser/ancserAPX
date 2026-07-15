from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from research_v2 import cli
from research_v2.safety import UnsafeResearchPath, offline_context
from research_v2.snapshot import create_snapshot


def _research_root(tmp_path: Path) -> Path:
    root = tmp_path / "research_v2"
    root.mkdir()
    return root


def _snapshot_fixture(tmp_path: Path, root: Path, snapshot_id: str = "canonical") -> Path:
    source = tmp_path / "source"
    store = source / "store"
    store.mkdir(parents=True)
    (store / "AAA.parquet").write_bytes(b"physical-parquet-fixture")
    manifest = source / "manifest.json"
    manifest.write_text('{"fixture": true}\n', encoding="utf-8")
    return create_snapshot(
        store_dir=store,
        manifest_path=manifest,
        research_root=root,
        snapshot_id=snapshot_id,
    )


def _context(root: Path) -> cli.RunContext:
    run_dir = root / "runs" / "fixture"
    run_dir.mkdir(parents=True)
    return cli.RunContext(
        research_root=root,
        snapshot_dir=root / "snapshots" / "canonical",
        snapshot_metadata={"snapshot_id": "canonical"},
        snapshot_data_sha256="d" * 64,
        config_path=None,
        config_file_sha256=None,
        config=SimpleNamespace(),
        config_sha256="c" * 64,
        run_id="fixture",
        run_dir=run_dir,
        code_sha256=cli._code_hash(),
        versions={"python": "fixture"},
        seeds={"base_random_seed": 7},
    )


def test_cli_import_does_not_import_dataframes_or_model_runtimes() -> None:
    command = [
        sys.executable,
        "-c",
        (
            "import json,sys; import research_v2.cli; "
            "print(json.dumps({name: name in sys.modules for name in "
            "['numpy','pandas','polars','sklearn','torch']}))"
        ),
    ]
    completed = subprocess.run(
        command,
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(completed.stdout) == {
        "numpy": False,
        "pandas": False,
        "polars": False,
        "sklearn": False,
        "torch": False,
    }


def test_parser_exposes_all_commands_and_canonical_defaults() -> None:
    parser = cli.build_parser()
    expected = {"snapshot", "features", "tabular", "sequence", "search", "all", "verify"}
    choices = next(
        action.choices
        for action in parser._actions
        if isinstance(action, argparse._SubParsersAction)
    )
    assert set(choices) == expected

    args = parser.parse_args(["all"])
    assert args.snapshot == "canonical"
    assert args.config == "default_config.json"
    assert args.run_id == "canonical"
    assert args.resume is True
    assert args.device == "auto"


def test_main_enters_offline_context_and_restores_environment(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("APCA_API_KEY_ID", "must-return")

    def probe(args: argparse.Namespace):
        assert args.command == "features"
        assert os.environ["ANCSER_RESEARCH_OFFLINE"] == "1"
        assert "APCA_API_KEY_ID" not in os.environ
        return {"offline": True}

    monkeypatch.setattr(cli, "_dispatch", probe)
    assert cli.main(["features"]) == 0
    assert json.loads(capsys.readouterr().out) == {"offline": True}
    assert os.environ["APCA_API_KEY_ID"] == "must-return"
    assert "ANCSER_RESEARCH_OFFLINE" not in os.environ


def test_atomic_stage_publish_verified_resume_and_tamper_detection(tmp_path: Path) -> None:
    root = _research_root(tmp_path)
    context = _context(root)
    writes = 0

    def writer(directory: Path):
        nonlocal writes
        writes += 1
        (directory / "artifact.txt").write_text("immutable", encoding="utf-8")
        return {"rows": 1}

    with offline_context():
        first = cli._publish_stage(
            context,
            "tabular",
            parameters={"alpha": 1.0},
            inputs={"features_success_sha256": "f" * 64},
            resume=True,
            writer=writer,
        )
        second = cli._publish_stage(
            context,
            "tabular",
            parameters={"alpha": 1.0},
            inputs={"features_success_sha256": "f" * 64},
            resume=True,
            writer=writer,
        )

    assert first["status"] == "completed"
    assert second["status"] == "skipped_verified"
    assert writes == 1
    stage = context.run_dir / "tabular"
    marker = cli.verify_stage(stage, research_root=root)
    assert marker["details"] == {"rows": 1}
    assert marker["versions"] == {"python": "fixture"}
    assert marker["seeds"] == {"base_random_seed": 7}

    (stage / "artifact.txt").write_text("tampered", encoding="utf-8")
    with pytest.raises(cli.ArtifactVerificationError, match="changed"):
        cli.verify_stage(stage, research_root=root)


def test_resume_rejects_changed_upstream_hash_or_parameters(tmp_path: Path) -> None:
    root = _research_root(tmp_path)
    context = _context(root)

    def writer(directory: Path):
        (directory / "artifact.json").write_text("{}", encoding="utf-8")
        return None

    with offline_context():
        cli._publish_stage(
            context,
            "sequence",
            parameters={"device": "cpu"},
            inputs={"features_success_sha256": "a" * 64},
            resume=True,
            writer=writer,
        )
        with pytest.raises(cli.ArtifactVerificationError, match="different"):
            cli._publish_stage(
                context,
                "sequence",
                parameters={"device": "cpu"},
                inputs={"features_success_sha256": "b" * 64},
                resume=True,
                writer=writer,
            )
        with pytest.raises(cli.ArtifactVerificationError, match="different"):
            cli._publish_stage(
                context,
                "sequence",
                parameters={"device": "cuda"},
                inputs={"features_success_sha256": "a" * 64},
                resume=True,
                writer=writer,
            )


def test_atomic_writer_cannot_escape_research_root(tmp_path: Path) -> None:
    root = _research_root(tmp_path)
    with pytest.raises(UnsafeResearchPath):
        cli._atomic_write_json(
            root.parent / "escaped.json",
            {"forbidden": True},
            root=root,
        )
    assert not (root.parent / "escaped.json").exists()


def test_prepare_context_persists_hashes_versions_seeds_and_is_immutable(
    tmp_path: Path,
) -> None:
    root = _research_root(tmp_path)
    _snapshot_fixture(tmp_path, root)
    config_source = Path(cli.MODULE_DIR) / "default_config.json"
    (root / "default_config.json").write_bytes(config_source.read_bytes())
    args = cli.build_parser().parse_args(
        [
            "features",
            "--research-root",
            str(root),
            "--snapshot",
            "canonical",
            "--run-id",
            "audit-001",
        ]
    )
    with offline_context():
        context = cli._prepare_context(args)
        resumed = cli._prepare_context(args, require_existing=True)

    assert resumed.config_sha256 == context.config_sha256
    manifest = json.loads((context.run_dir / "run.json").read_text(encoding="utf-8"))
    assert manifest["snapshot_data_sha256"] == context.snapshot_data_sha256
    assert manifest["config_sha256"] == context.config_sha256
    assert manifest["config_file_sha256"] == cli._sha256_file(root / "default_config.json")
    assert manifest["versions_at_run_creation"]["python"]
    assert manifest["seeds"]["base_random_seed"] == 20260710
    assert manifest["offline_only"] is True

    changed = json.loads((root / "default_config.json").read_text(encoding="utf-8"))
    changed["data"]["label_horizon"] = 10
    (root / "default_config.json").write_text(json.dumps(changed), encoding="utf-8")
    with offline_context(), pytest.raises(cli.ArtifactVerificationError, match="different"):
        cli._prepare_context(args, require_existing=True)


def test_verify_snapshot_command_does_not_require_a_run(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = _research_root(tmp_path)
    _snapshot_fixture(tmp_path, root, snapshot_id="canonical-20990101")
    status = cli.main(
        [
            "verify",
            "--research-root",
            str(root),
            "--snapshot",
            "canonical",
            "--stage",
            "snapshot",
        ]
    )
    assert status == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "verified"
    assert payload["snapshot_id"] == "canonical-20990101"
    assert len(payload["data_sha256"]) == 64
    assert not (root / "runs").exists()


def test_incomplete_stage_is_never_treated_as_resumable(tmp_path: Path) -> None:
    root = _research_root(tmp_path)
    context = _context(root)
    incomplete = context.run_dir / "search"
    incomplete.mkdir()
    (incomplete / "partial.csv").write_text("not complete", encoding="utf-8")

    with offline_context(), pytest.raises(
        cli.ArtifactVerificationError, match="incomplete"
    ):
        cli._publish_stage(
            context,
            "search",
            parameters={},
            inputs={},
            resume=True,
            writer=lambda _: None,
        )
