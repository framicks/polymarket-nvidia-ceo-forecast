# Polymarket NVIDIA CEO Forecaster

Forecast Polymarket's "Next CEO of NVIDIA?" Yes prices minute by minute, 4 hours ahead, for the make.fun `polymarket-nvidia-ceo-forecast-01` bounty.

## Entry

`main.py` (Python 3.13, stdlib only, deterministic) reads each case's `*.in` file from the input directory and writes one `*.out` forecast file per case: for each market and each of the 240 minutes after the checkpoint, the five quantiles (0.05, 0.25, 0.5, 0.75, 0.95) of that minute's Yes price.

## Model

Persistence median with per-market volatility and sub-diffusive spread growth:

- **Median:** the price at the checkpoint. Momentum and mean reversion were both tested on the practice cases and rejected — 1-minute momentum does not persist, and prices do not revert to their 24h mean; the leading markets genuinely drift for hours.
- **Volatility:** sigma from the last 120 minutes of 1-minute moves, as the average of the standard deviation and the MAD-based estimate. Short windows win decisively (120 min beats 360/1440/4320) because volatility is regime-dependent in these thin markets. Floored at `max(0.0002, 0.005 * sqrt(x * (1 - x)))` so a market that has sat still for two hours is still given jump room — Colette Kress went 0.3535 → 0.59 in a single minute in the practice data.
- **Horizon:** quantiles widen as `h ** 0.38`, not `sqrt(h)` — spreads grow slower than a random walk because prices are bounded probabilities that spend long stretches unmoved. Alpha 0.38 beat 0.30–0.45 on the practice cases.
- **Quantiles:** `q = clamp(x + z_tau * sigma * h ** 0.38, 0, 1)`, standard-normal z, rounded to 6 decimals.

## Backtest

Scored against the real Polymarket price history (fetched from the CLOB `prices-history` endpoint, 1-minute grid rebuilt exactly as the house builds it — verified 0 mismatches against the case files' own prices):

| model | practice loss (mean pinball, pp) |
|---|---|
| house baseline (persistence, σ=0.0025·√h) | 0.363589 |
| **this entry** | **0.272717** |

Per-case: p00 0.4969, p01 0.3023, p02 0.3836, p03 0.2287, p04 0.1912, p05 0.1077. The model also beat the baseline on every leave-one-case-out subset (0.24–0.32 vs 0.33–0.39), so the edge is not one lucky case.

Alternatives tried and rejected on the same backtest: drift/momentum terms, mean reversion to the 24h mean, longer sigma windows, EWMA volatility, fat-tail widening, a no-move mixture model, empirical h-step quantiles, and logit-space forecasting.
