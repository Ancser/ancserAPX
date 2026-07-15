"""Safety boundaries for research code.

The safeguards here are intentionally process-local.  ``offline_context``
temporarily removes broker credentials from *this* Python process and restores
them exactly on exit; it cannot alter the environment of the web server,
scheduler, or any other OS process.
"""

from __future__ import annotations

import os
import sys
import threading
from contextlib import contextmanager
from importlib.abc import MetaPathFinder
from pathlib import Path
from typing import Iterable, Iterator, Mapping


RESEARCH_ROOT = Path(__file__).resolve().parent

# Import prefixes that can read mutable production data, reach Alpaca, write
# live state, or submit orders.  Pure alpha modules are deliberately not
# blocked so factor math can be reused after explicit parity tests.
BLOCKED_IMPORT_PREFIXES = (
    "alpaca",
    "alpaca_trade_api",
    "backend.backtest.engine",
    "backend.data.alpaca_adapter",
    "backend.data.fetcher",
    "backend.data.store",
    "backend.execution",
    "frontend.server",
)

# Alpaca's official variables plus common legacy aliases.  Matching is
# case-insensitive.
APCA_ENV_PREFIXES = ("APCA_", "ALPACA_", "PAPER_TRADING")
_OFFLINE_FLAG = "ANCSER_RESEARCH_OFFLINE"
_OFFLINE_LOCK = threading.RLock()


class UnsafeResearchPath(ValueError):
    """Raised when research code attempts to write outside research_v2."""


class UnsafeResearchImport(ImportError):
    """Raised before a live/production module can be imported in research."""


def _normalise_prefixes(prefixes: Iterable[str]) -> tuple[str, ...]:
    cleaned = {str(prefix).strip().strip(".") for prefix in prefixes}
    return tuple(sorted(prefix for prefix in cleaned if prefix))


def _matches_import_prefix(module_name: str, prefixes: Iterable[str]) -> bool:
    return any(
        module_name == prefix or module_name.startswith(prefix + ".")
        for prefix in prefixes
    )


def _is_broker_env(name: str) -> bool:
    upper = name.upper()
    return any(upper == prefix or upper.startswith(prefix) for prefix in APCA_ENV_PREFIXES)


def ensure_research_output_path(
    path: os.PathLike[str] | str,
    *,
    research_root: os.PathLike[str] | str = RESEARCH_ROOT,
) -> Path:
    """Resolve and validate an output path under a ``research_v2`` directory.

    Relative paths are interpreted relative to ``research_root``.  Existing
    symlinks are resolved before the containment check, preventing a symlink or
    ``..`` component from escaping the research tree.  A configurable root is
    accepted solely so tests and isolated workers can use their own temporary
    ``research_v2`` tree.
    """

    root = Path(research_root).expanduser().resolve(strict=False)
    if root.name != "research_v2":
        raise UnsafeResearchPath(
            f"research root must be a directory named 'research_v2': {root}"
        )

    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    candidate = candidate.resolve(strict=False)

    if candidate != root and root not in candidate.parents:
        raise UnsafeResearchPath(
            f"research output escapes {root}: {candidate}"
        )
    return candidate


class _BlockedImportFinder(MetaPathFinder):
    def __init__(self, prefixes: Iterable[str]):
        self.prefixes = _normalise_prefixes(prefixes)

    def find_spec(self, fullname: str, path=None, target=None):  # noqa: ANN001
        if _matches_import_prefix(fullname, self.prefixes):
            raise UnsafeResearchImport(
                f"offline research blocked import of '{fullname}'"
            )
        return None


@contextmanager
def offline_context(
    *,
    extra_blocked: Iterable[str] = (),
    environ: Mapping[str, str] | None = None,
) -> Iterator[None]:
    """Temporarily remove broker credentials and block production imports.

    The default uses ``os.environ`` and restores all matching variables in a
    ``finally`` block.  Variables created inside the context with an APCA,
    ALPACA, or PAPER_TRADING prefix are removed on exit as well.  Passing an
    alternate mutable mapping is supported for isolated tests.

    This context is intended for a single-threaded research CLI/subprocess.
    Environment variables and ``sys.meta_path`` are process state, so a lock
    serializes nested/concurrent uses inside this module.  Other OS processes,
    including the daily runner, are never affected.
    """

    env = os.environ if environ is None else environ
    if not hasattr(env, "pop") or not hasattr(env, "__setitem__"):
        raise TypeError("environ must be a mutable mapping")

    prefixes = _normalise_prefixes((*BLOCKED_IMPORT_PREFIXES, *extra_blocked))
    blocker = _BlockedImportFinder(prefixes)

    with _OFFLINE_LOCK:
        preloaded = sorted(
            name
            for name, module in sys.modules.items()
            if module is not None and _matches_import_prefix(name, prefixes)
        )
        if preloaded:
            raise UnsafeResearchImport(
                "blocked production modules were already loaded before entering "
                f"offline_context: {', '.join(preloaded[:8])}"
            )

        saved_broker = {key: value for key, value in env.items() if _is_broker_env(key)}
        flag_existed = _OFFLINE_FLAG in env
        saved_flag = env.get(_OFFLINE_FLAG)

        for key in list(env):
            if _is_broker_env(key):
                env.pop(key, None)
        env[_OFFLINE_FLAG] = "1"
        sys.meta_path.insert(0, blocker)

        try:
            yield
        finally:
            # Remove by identity so an unrelated finder cannot be removed.
            sys.meta_path[:] = [finder for finder in sys.meta_path if finder is not blocker]
            for key in list(env):
                if _is_broker_env(key):
                    env.pop(key, None)
            for key, value in saved_broker.items():
                env[key] = value

            if flag_existed:
                env[_OFFLINE_FLAG] = saved_flag  # type: ignore[assignment]
            else:
                env.pop(_OFFLINE_FLAG, None)
