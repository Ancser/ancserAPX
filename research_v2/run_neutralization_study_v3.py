"""Run the immutable v3 sector-neutralization study.

The implementation remains in the hardened v2 module so historical tests can
keep importing its pure helpers.  V3 adds the audited Main weekly clock:
prior completed daily close -> last trading day on/before Friday open proxy.
"""

from __future__ import annotations

from research_v2.run_neutralization_study_v2 import main


if __name__ == "__main__":
    raise SystemExit(main())
