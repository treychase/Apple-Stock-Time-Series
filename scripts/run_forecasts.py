"""CLI: forecast the whole universe one week ahead and write a JSON artifact.

The dashboard reads ``data/forecasts.json`` when present so it can render
instantly without re-running the sampler.  In CI / offline mode set
``STOCKTS_USE_SAMPLE=1`` to use the bundled sample data.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stockts.data import get_history  # noqa: E402
from stockts.forecast import forecast_universe  # noqa: E402
from stockts.universe import default_universe  # noqa: E402

OUT = ROOT / "data" / "forecasts.json"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--horizon", type=int, default=5, help="trading days ahead")
    ap.add_argument("--level", type=float, default=0.95, help="credible level")
    ap.add_argument("--period", default="2y", help="yfinance history period")
    ap.add_argument("--limit", type=int, default=0, help="cap number of tickers (0=all)")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    tickers = default_universe()
    if args.limit:
        tickers = tickers[: args.limit]

    print(f"loading history for {len(tickers)} tickers...")
    history = get_history(tickers, period=args.period)
    print(f"history: {history.shape[0]} rows x {history.shape[1]} tickers")

    ft = forecast_universe(history, horizon=args.horizon, level=args.level)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "horizon": ft.horizon,
        "level": ft.level,
        "forecasts": ft.table.to_dict(orient="records"),
        "top5": ft.top5.to_dict(orient="records"),
        "bottom5": ft.bottom5.to_dict(orient="records"),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(f"wrote {out} ({len(payload['forecasts'])} forecasts)")
    if not ft.top5.empty:
        print("\nTop 5 projected gainers (1 week):")
        print(ft.top5[["ticker", "last_price", "forecast", "expected_return"]].to_string(index=False))
        print("\nBottom 5 projected (1 week):")
        print(ft.bottom5[["ticker", "last_price", "forecast", "expected_return"]].to_string(index=False))


if __name__ == "__main__":
    main()
