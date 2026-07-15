"""Verified, physically copied snapshots of the mutable production data store."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable

from .safety import RESEARCH_ROOT, UnsafeResearchPath, ensure_research_output_path


PROJECT_ROOT = RESEARCH_ROOT.parent
DEFAULT_STORE_DIR = PROJECT_ROOT / "data" / "store"
DEFAULT_MANIFEST_PATH = PROJECT_ROOT / "data" / "manifest.json"
SNAPSHOT_FORMAT_VERSION = 1
_CHUNK_SIZE = 1024 * 1024
_SNAPSHOT_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")


class SnapshotError(RuntimeError):
    """Base error for snapshot creation or verification failures."""


class SnapshotSourceChanged(SnapshotError):
    """The production source changed while a snapshot was being made."""


class SnapshotVerificationError(SnapshotError):
    """A copied snapshot does not match its recorded fingerprints."""


@dataclass(frozen=True)
class FileFingerprint:
    relative_path: str
    size: int
    mtime_ns: int
    sha256: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def _fingerprint(path: Path, relative_path: str) -> FileFingerprint:
    if path.is_symlink():
        raise SnapshotError(f"snapshot source/copy may not be a symlink: {path}")
    if not path.is_file():
        raise SnapshotError(f"snapshot file is missing or not regular: {path}")

    before = path.stat()
    sha256 = _sha256(path)
    after = path.stat()
    before_state = (before.st_size, before.st_mtime_ns)
    after_state = (after.st_size, after.st_mtime_ns)
    if before_state != after_state:
        raise SnapshotSourceChanged(f"file changed while hashing: {path}")
    return FileFingerprint(relative_path, after.st_size, after.st_mtime_ns, sha256)


def _store_files(store_dir: Path) -> list[Path]:
    if store_dir.is_symlink() or not store_dir.is_dir():
        raise SnapshotError(f"store directory is missing, invalid, or a symlink: {store_dir}")

    files: list[Path] = []
    for path in sorted(store_dir.rglob("*"), key=lambda item: item.as_posix()):
        if path.is_symlink():
            raise SnapshotError(f"store contains a symlink: {path}")
        if path.is_file():
            if path.suffix.lower() != ".parquet":
                raise SnapshotError(f"unexpected non-Parquet file in store: {path}")
            files.append(path)
    if not files:
        raise SnapshotError(f"store contains no Parquet files: {store_dir}")
    return files


def _fingerprint_store(store_dir: Path) -> Dict[str, FileFingerprint]:
    fingerprints: Dict[str, FileFingerprint] = {}
    for path in _store_files(store_dir):
        relative = path.relative_to(store_dir).as_posix()
        fingerprints[relative] = _fingerprint(path, relative)
    return fingerprints


def _state(fp: FileFingerprint) -> tuple[int, int, str]:
    return fp.size, fp.mtime_ns, fp.sha256


def _assert_source_unchanged(
    before: Dict[str, FileFingerprint],
    after: Dict[str, FileFingerprint],
) -> None:
    if set(before) != set(after):
        added = sorted(set(after) - set(before))
        removed = sorted(set(before) - set(after))
        raise SnapshotSourceChanged(
            f"store file set changed; added={added[:5]}, removed={removed[:5]}"
        )
    changed = [name for name in before if _state(before[name]) != _state(after[name])]
    if changed:
        raise SnapshotSourceChanged(
            f"store files changed during snapshot: {', '.join(changed[:8])}"
        )


def _copy_file(source: Path, destination: Path) -> FileFingerprint:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination, follow_symlinks=False)
    try:
        if os.path.samefile(source, destination):
            raise SnapshotVerificationError(
                f"snapshot must be a physical copy, not the source file: {source}"
            )
    except FileNotFoundError as exc:
        raise SnapshotSourceChanged(f"source disappeared while copying: {source}") from exc
    return _fingerprint(destination, destination.name)


def _copy_and_verify(
    source: Path,
    destination: Path,
    expected: FileFingerprint,
) -> None:
    copied = _copy_file(source, destination)
    if _state(copied) == _state(expected):
        return

    try:
        current_source = _fingerprint(source, expected.relative_path)
    except SnapshotError as exc:
        raise SnapshotSourceChanged(f"source changed while copying: {source}") from exc
    if _state(current_source) != _state(expected):
        raise SnapshotSourceChanged(f"source changed while copying: {source}")
    raise SnapshotVerificationError(f"copy failed fingerprint verification: {destination}")


def _validate_snapshot_id(snapshot_id: str) -> str:
    if not _SNAPSHOT_ID_RE.fullmatch(snapshot_id) or snapshot_id in {".", ".."}:
        raise UnsafeResearchPath(f"invalid snapshot id: {snapshot_id!r}")
    return snapshot_id


def _default_snapshot_id(manifest_hash: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{timestamp}-{manifest_hash[:12]}-{uuid.uuid4().hex[:8]}"


def _safe_remove_staging(path: Path, snapshots_dir: Path, research_root: Path) -> None:
    checked = ensure_research_output_path(path, research_root=research_root)
    if checked.parent != snapshots_dir or not checked.name.startswith("."):
        raise UnsafeResearchPath(f"refusing to remove non-staging path: {checked}")
    if checked.exists():
        shutil.rmtree(checked)


def _write_metadata(path: Path, payload: dict) -> None:
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    with path.open("xb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())


def create_snapshot(
    *,
    store_dir: os.PathLike[str] | str = DEFAULT_STORE_DIR,
    manifest_path: os.PathLike[str] | str = DEFAULT_MANIFEST_PATH,
    research_root: os.PathLike[str] | str = RESEARCH_ROOT,
    snapshot_id: str | None = None,
) -> Path:
    """Create a verified snapshot and publish it atomically.

    Source files and the manifest are fingerprinted before copying and again
    afterwards using size, nanosecond mtime, and SHA-256.  Copies are independently
    fingerprinted.  Until every check succeeds, all output lives in a hidden
    staging directory; a same-volume directory rename is the only publication
    step, so consumers never see a partial snapshot.
    """

    source_store = Path(store_dir).expanduser().resolve(strict=False)
    source_manifest = Path(manifest_path).expanduser().resolve(strict=False)
    root = Path(research_root).expanduser().resolve(strict=False)
    # Validate the root before creating even its first directory.
    ensure_research_output_path(root, research_root=root)
    snapshots_dir = ensure_research_output_path("snapshots", research_root=root)

    manifest_before = _fingerprint(source_manifest, "manifest.json")
    store_before = _fingerprint_store(source_store)
    chosen_id = _validate_snapshot_id(snapshot_id or _default_snapshot_id(manifest_before.sha256))
    final_dir = ensure_research_output_path(snapshots_dir / chosen_id, research_root=root)
    staging_dir = ensure_research_output_path(
        snapshots_dir / f".{chosen_id}.tmp-{uuid.uuid4().hex}", research_root=root
    )

    snapshots_dir.mkdir(parents=True, exist_ok=True)
    if final_dir.exists():
        raise SnapshotError(f"snapshot already exists: {final_dir}")
    staging_dir.mkdir(parents=False, exist_ok=False)

    try:
        staging_store = ensure_research_output_path(staging_dir / "store", research_root=root)
        staging_store.mkdir(parents=False, exist_ok=False)

        for relative, expected in store_before.items():
            source = source_store / Path(relative)
            destination = ensure_research_output_path(
                staging_store / Path(relative), research_root=root
            )
            _copy_and_verify(source, destination, expected)

        copied_manifest = ensure_research_output_path(
            staging_dir / "manifest.json", research_root=root
        )
        _copy_and_verify(source_manifest, copied_manifest, manifest_before)

        # Re-hash the mutable source only after every physical copy is complete.
        store_after = _fingerprint_store(source_store)
        manifest_after = _fingerprint(source_manifest, "manifest.json")
        _assert_source_unchanged(store_before, store_after)
        if _state(manifest_before) != _state(manifest_after):
            raise SnapshotSourceChanged("data/manifest.json changed during snapshot")

        # Re-scan the staged store so a successful result is independently
        # verifiable rather than trusting the copy loop.
        copied_store = _fingerprint_store(staging_store)
        _assert_source_unchanged(store_before, copied_store)
        copied_manifest_fp = _fingerprint(copied_manifest, "manifest.json")
        if _state(copied_manifest_fp) != _state(manifest_before):
            raise SnapshotVerificationError("copied manifest fingerprint mismatch")

        metadata = {
            "format_version": SNAPSHOT_FORMAT_VERSION,
            "snapshot_id": chosen_id,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_store": str(source_store),
            "source_manifest": str(source_manifest),
            "manifest": asdict(manifest_before),
            "file_count": len(store_before),
            "files": [asdict(store_before[name]) for name in sorted(store_before)],
        }
        metadata_path = ensure_research_output_path(
            staging_dir / "snapshot.json", research_root=root
        )
        _write_metadata(metadata_path, metadata)

        # staging_dir and final_dir share a parent/volume.  os.rename refuses to
        # overwrite an existing directory on Windows, preserving prior snapshots.
        if final_dir.exists():
            raise SnapshotError(f"snapshot appeared concurrently: {final_dir}")
        os.rename(staging_dir, final_dir)
        return final_dir
    finally:
        if staging_dir.exists():
            _safe_remove_staging(staging_dir, snapshots_dir, root)


def _fingerprints_from_metadata(items: Iterable[dict]) -> Dict[str, FileFingerprint]:
    result: Dict[str, FileFingerprint] = {}
    for item in items:
        fp = FileFingerprint(
            relative_path=str(item["relative_path"]),
            size=int(item["size"]),
            mtime_ns=int(item["mtime_ns"]),
            sha256=str(item["sha256"]),
        )
        result[fp.relative_path] = fp
    return result


def verify_snapshot(
    snapshot_dir: os.PathLike[str] | str,
    *,
    research_root: os.PathLike[str] | str = RESEARCH_ROOT,
) -> dict:
    """Re-hash a completed snapshot and return its recorded metadata."""

    root = Path(research_root).expanduser().resolve(strict=False)
    directory = ensure_research_output_path(snapshot_dir, research_root=root)
    if directory.parent != ensure_research_output_path("snapshots", research_root=root):
        raise UnsafeResearchPath(f"not a completed snapshot directory: {directory}")
    if directory.name.startswith(".") or not directory.is_dir():
        raise SnapshotVerificationError(f"snapshot is incomplete or missing: {directory}")

    metadata_path = directory / "snapshot.json"
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SnapshotVerificationError(f"invalid snapshot metadata: {metadata_path}") from exc
    if metadata.get("format_version") != SNAPSHOT_FORMAT_VERSION:
        raise SnapshotVerificationError("unsupported snapshot format version")
    if metadata.get("snapshot_id") != directory.name:
        raise SnapshotVerificationError("snapshot id does not match directory name")

    expected_store = _fingerprints_from_metadata(metadata.get("files", []))
    actual_store = _fingerprint_store(directory / "store")
    try:
        _assert_source_unchanged(expected_store, actual_store)
    except SnapshotSourceChanged as exc:
        raise SnapshotVerificationError(str(exc)) from exc

    expected_manifest = FileFingerprint(**metadata["manifest"])
    actual_manifest = _fingerprint(directory / "manifest.json", "manifest.json")
    if _state(expected_manifest) != _state(actual_manifest):
        raise SnapshotVerificationError("manifest fingerprint mismatch")
    if metadata.get("file_count") != len(actual_store):
        raise SnapshotVerificationError("snapshot file count mismatch")
    return metadata
