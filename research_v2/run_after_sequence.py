"""Wait for the definitive sequence checkpoint, then start the frozen search."""

from __future__ import annotations

import argparse
import ctypes
import json
import os
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parent
DEFAULT_RUN = ROOT / "runs" / "20260710_full_v1"


def _process_active(pid: int) -> bool:
    if os.name == "nt":
        process_query_limited_information = 0x1000
        still_active = 259
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(process_query_limited_information, False, pid)
        if not handle:
            return False
        try:
            code = ctypes.c_ulong()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                return False
            return int(code.value) == still_active
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence-pid", type=int, required=True)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--action", choices=("search", "raw"), default="search")
    args = parser.parse_args(argv)
    if args.poll_seconds <= 0:
        raise ValueError("poll-seconds must be positive")

    run_dir = args.run_dir.expanduser().resolve(strict=True)
    sequence = run_dir / "sequence_full60_all"
    marker = sequence / "_SUCCESS"
    summary = sequence / "summary.json"
    while not (marker.is_file() and summary.is_file()):
        if not _process_active(args.sequence_pid):
            stderr = sequence / "stderr.log"
            detail = stderr.read_text(encoding="utf-8", errors="replace")[-4000:] if stderr.is_file() else ""
            raise RuntimeError(
                "sequence process exited before verified completion"
                + (f": {detail}" if detail else "")
            )
        print(
            json.dumps(
                {
                    "event": "waiting_for_sequence",
                    "sequence_pid": args.sequence_pid,
                    "marker": marker.is_file(),
                    "summary": summary.is_file(),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        time.sleep(args.poll_seconds)

    if args.action == "search":
        from research_v2.run_full_search import main as run_search

        return run_search(["--run-dir", str(run_dir)])

    from research_v2.run_full_sequence import main as run_sequence

    return run_sequence(
        [
            "--family",
            "raw",
            "--output",
            str(run_dir / "sequence_raw60_all"),
            "--device",
            "cuda",
            "--epochs",
            "12",
            "--batch-size",
            "256",
            "--patience",
            "3",
        ]
    )


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise
