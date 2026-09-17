# breeden-litzenberger

Extracts the market-implied risk-neutral probability density of an
underlying's price at expiration from a single-expiration snapshot of listed
option prices, via Breeden & Litzenberger (1978).

Given a ticker and expiration, this library fetches the option chain, cleans
and selects liquid out-of-the-money quotes, inverts them to implied
volatilities, fits an arbitrage-checked SVI smile (Gatheral 2004; Gatheral &
Jacquier 2014), and differentiates the resulting smooth call-price curve
twice to recover a full probability density — along with the diagnostics
needed to trust (or distrust) the result: non-negativity, normalization,
forward-price consistency, and desk-convention smile-shape metrics (ATM vol,
25-delta risk reversal, 25-delta butterfly).

See [`docs/theory.md`](docs/theory.md) for why this is numerically hard and
how each step addresses it — that's the real content of this project.

## Install

```bash
pip install -e ".[dev]"
```

## Quickstart

```python
from datetime import date
from breeden_litzenberger import extract

report = extract("SPY", date(2026, 12, 18))
print(report.summary())
```

```bash
bl extract --ticker SPY --expiration 2026-12-18
```

See [`docs/usage.md`](docs/usage.md) for overrides, plotting, error handling,
and the full CLI, and
[`specs/001-rnd-extraction/quickstart.md`](specs/001-rnd-extraction/quickstart.md)
for runnable end-to-end validation scenarios.

## Development

```bash
pytest
mypy --strict breeden_litzenberger/
```

The test suite includes a ground-truth validation (`tests/test_density.py`):
the full pipeline run on option prices synthesized from a known
constant-volatility Black-Scholes model must recover the closed-form
lognormal density to within 0.1% relative error on its moments — the test
that actually proves the pipeline is mathematically correct, per
constitution Principle III.

## Scope

Single underlying, single expiration, single fixed-timestamp snapshot per
call. Explicitly out of scope: multi-expiration term structure / calendar
arbitrage across maturities, live/streaming data, and any connection to a
trading or execution system.
