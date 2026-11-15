#!/usr/bin/env python3
"""BTC cent-forecast entry.

Reads every ``*.in`` case file from the input directory.  Each case is a
JSON object describing one market snapshot: 4320 one-minute closes whose
last value is the checkpoint close.  For every case we emit one forecast
of the price ``horizon_minutes`` (240) minutes after the checkpoint.

Method
------
For a horizon ``h`` the training signal already inside the series is the
set of overlapping h-minute changes ``d_i = P[i+h] - P[i]``.  The point
forecast that minimises expected absolute error (the competition scores
in cents, i.e. an L1-style loss) is the checkpoint close plus the MEDIAN
of those changes.  We refine the raw full-window median by blending it
50/50 with the median of the changes drawn only from the most recent
24 hours, which lets the forecast lean toward the current regime without
abandoning the stability of the full 72-hour window.

Pure stdlib, fully deterministic, runs in well under a second per case.
"""

import json
import math
import os
import statistics
import sys

DEFAULT_HORIZON = 240
RECENT_WINDOW = 1440  # one day of one-minute bars


def horizon_changes(prices, horizon):
    """All overlapping horizon-minute price changes in the series."""
    n = len(prices)
    return [prices[i + horizon] - prices[i] for i in range(n - horizon)]


def forecast_price(prices, horizon):
    """Checkpoint close plus the blended median horizon change."""
    prices = [p for p in prices if math.isfinite(p)]
    if not prices:
        return None
    last = float(prices[-1])
    if len(prices) <= horizon:
        return last
    full = horizon_changes(prices, horizon)
    med_full = statistics.median(full)
    recent = prices[-RECENT_WINDOW:] if len(prices) > RECENT_WINDOW else prices
    if len(recent) > horizon:
        med_recent = statistics.median(horizon_changes(recent, horizon))
        # Lean on the recent regime only when it agrees with the full
        # window about direction; on a sign clash the long window wins.
        w_recent = 0.5 if med_full * med_recent >= 0 else 0.25
        drift = (1.0 - w_recent) * med_full + w_recent * med_recent
    else:
        drift = med_full
    price = last + drift
    if not math.isfinite(price) or price <= 0:
        return last
    return price


def load_case(path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def case_forecast(case):
    case_id = case["case"]
    horizon = int(case.get("horizon_minutes") or DEFAULT_HORIZON)
    markets = case.get("markets") or []
    prices = []
    if markets:
        prices = markets[0].get("prices") or []
    price = forecast_price([float(p) for p in prices], horizon)
    if price is None:
        price = 0.0
    return {"case": case_id, "price": price}


def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    os.makedirs(out_dir, exist_ok=True)
    names = sorted(n for n in os.listdir(in_dir) if n.endswith(".in"))
    for name in names:
        case = load_case(os.path.join(in_dir, name))
        result = {"forecasts": [case_forecast(case)]}
        out_name = "{}.out".format(case["case"])
        with open(os.path.join(out_dir, out_name), "w", encoding="utf-8") as fh:
            json.dump(result, fh, separators=(",", ":"))
            fh.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
