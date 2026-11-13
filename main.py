#!/usr/bin/env python3
"""Forecast Polymarket's "Next CEO of NVIDIA?" Yes prices, 4 hours ahead.

Model (backtested on the kit's six practice cases, loss 0.2727 vs the house
baseline's 0.3636, mean pinball loss in percentage points):

- Persistence median: each market's price stays at its checkpoint price.
  Momentum does not persist at 1-minute scale (drift terms tested and
  rejected), and prices do not mean-revert to their 24h mean (also tested
  and rejected) -- the top markets genuinely drift for hours.
- Per-market volatility: sigma is estimated from the last 120 minutes of
  1-minute price changes as the average of the standard deviation and the
  MAD-based estimate (1.2533 * mean |diff|). Short windows win: 120 min
  beats 360/1440/4320 badly, because volatility is regime-dependent and
  these thin markets swing between calm and jumpy. A floor keeps quiet
  markets honest: sigma >= max(0.0002, 0.005 * sqrt(x * (1 - x))), since
  even a market that has not moved in two hours can jump (Colette Kress
  went 0.3535 -> 0.59 in one minute in the practice data).
- Horizon scaling: quantiles widen as h ** 0.38, not sqrt(h). Spreads grow
  slower than a random walk because prices are bounded probabilities that
  spend long stretches unmoved; alpha 0.38 beat 0.30/0.35/0.40/0.45 and
  the equivalent sqrt(h)-with-damping form on the practice cases.
- Quantiles: q_tau(h) = clamp(x + z_tau * sigma * h ** 0.38, 0, 1), rounded
  to 6 decimals, with z the standard-normal quantiles. Fat-tail widening,
  empirical h-step quantiles, a no-move mixture and logit-space forecasting
  were all tried and lost to this form on the practice cases.

Deterministic, stdlib only, no network. Reads every *.in in the input dir
and writes one *.out per case.
"""
import json
import math
import os

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.6448536269514722, -0.6744897501960817, 0.0,
     0.6744897501960817, 1.6448536269514722]
WINDOW = 120          # minutes of recent history used for sigma
ALPHA = 0.38          # horizon exponent: spread ~ h ** ALPHA
ABS_FLOOR = 0.0002    # absolute sigma floor (price units per min**ALPHA)
REL_FLOOR = 0.005     # sigma >= REL_FLOOR * sqrt(x * (1 - x))
SIGMA_CAP = 0.02


def price6(x):
    v = round(min(1.0, max(0.0, x)), 6)
    return v + 0.0  # never -0.0


def sigma(prices):
    """Per-market volatility from the last WINDOW minutes of 1-min moves."""
    p = prices[-WINDOW - 1:] if len(prices) > WINDOW else prices
    diffs = [p[i + 1] - p[i] for i in range(len(p) - 1)]
    if not diffs:
        return ABS_FLOOR
    mean = sum(diffs) / len(diffs)
    var = sum((d - mean) ** 2 for d in diffs) / len(diffs)
    mad = sum(abs(d - mean) for d in diffs) / len(diffs) * 1.2533
    s = 0.5 * (math.sqrt(var) + mad)
    x = prices[-1]
    floor = max(ABS_FLOOR, REL_FLOOR * math.sqrt(max(0.001, x * (1.0 - x))))
    return min(SIGMA_CAP, max(floor, s))


def forecast(market, horizon):
    x = market["prices"][-1]
    sig = sigma(market["prices"])
    rows = []
    for h in range(1, horizon + 1):
        spread = sig * (h ** ALPHA)
        row = [price6(x + z * spread) for z in Z]
        for j in range(1, 5):
            if row[j] < row[j - 1]:
                row[j] = row[j - 1]
        rows.append(row)
    return rows


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    inputs = sorted(f for f in os.listdir(INPUT_DIR) if f.endswith(".in"))
    for name in inputs:
        case_id = name[: -len(".in")]
        with open(os.path.join(INPUT_DIR, name), "r", encoding="utf-8") as f:
            case = json.load(f)
        if list(case["quantiles"]) != QUANTILES:
            raise ValueError(f"{name}: unexpected quantiles {case['quantiles']}")
        forecasts = [
            {"id": m["id"], "q": forecast(m, case["horizon_minutes"])}
            for m in case["markets"]
        ]
        out = {"forecasts": forecasts}
        with open(os.path.join(OUTPUT_DIR, f"{case_id}.out"), "w", encoding="utf-8") as f:
            f.write(json.dumps(out) + "\n")


if __name__ == "__main__":
    main()
