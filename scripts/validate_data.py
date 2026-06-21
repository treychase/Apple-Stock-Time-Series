"""CLI entrypoint for the CI data-validation job.

Loads price history (live API, or bundled sample with STOCKTS_USE_SAMPLE=1),
runs the validation suite, prints the report, and exits non-zero on failure so
the GitHub Actions job goes red.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stockts.data import get_history  # noqa: E402
from stockts.universe import default_universe  # noqa: E402
from stockts.validation import validate_history  # noqa: E402


def main() -> int:
    tickers = default_universe()
    print(f"validating history for {len(tickers)} tickers...")
    history = get_history(tickers, period="1y")
    report = validate_history(history)
    print(report)
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
