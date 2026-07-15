"""Isolated, offline-only research infrastructure for ancserAPX.

Nothing in this package imports the live scheduler, OMS, Alpaca adapter, or the
production data store module.  Research code should enter :func:`offline_context`
and read only from a completed snapshot created by :func:`create_snapshot`.
"""

from .safety import (
    APCA_ENV_PREFIXES,
    BLOCKED_IMPORT_PREFIXES,
    RESEARCH_ROOT,
    UnsafeResearchImport,
    UnsafeResearchPath,
    ensure_research_output_path,
    offline_context,
)
from .snapshot import (
    DEFAULT_MANIFEST_PATH,
    DEFAULT_STORE_DIR,
    SnapshotError,
    SnapshotSourceChanged,
    SnapshotVerificationError,
    create_snapshot,
    verify_snapshot,
)

__all__ = [
    "APCA_ENV_PREFIXES",
    "BLOCKED_IMPORT_PREFIXES",
    "DEFAULT_MANIFEST_PATH",
    "DEFAULT_STORE_DIR",
    "RESEARCH_ROOT",
    "SnapshotError",
    "SnapshotSourceChanged",
    "SnapshotVerificationError",
    "UnsafeResearchImport",
    "UnsafeResearchPath",
    "create_snapshot",
    "ensure_research_output_path",
    "offline_context",
    "verify_snapshot",
]
