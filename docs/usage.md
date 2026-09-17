# Usage

See [theory.md](theory.md) for what this library computes and why; this page
is about how to call it. For runnable end-to-end scenarios (including
determinism checks and failure handling), see
[`specs/001-rnd-extraction/quickstart.md`](../specs/001-rnd-extraction/quickstart.md).

## Install

```bash
pip install -e ".[dev]"
```

## Quick example

```python
from datetime import date
from breeden_litzenberger import extract

report = extract("SPY", date(2026, 12, 18))
print(report.summary())
```

`summary()` prints the fitted SVI parameters, fit quality, arbitrage status,
density diagnostics, moments, and smile-shape metrics, ending in a
plain-language verdict. For structured data instead:

```python
data = report.to_dict()
```

Overrides (risk-free rate, dividend yield, liquidity floor) are keyword-only:

```python
from breeden_litzenberger import LiquidityFloor

report = extract(
    "SPY",
    date(2026, 12, 18),
    risk_free_rate=0.045,
    dividend_yield=0.013,
    liquidity_floor=LiquidityFloor(min_open_interest=10, min_volume=10),
)
```

## Plotting

```python
from breeden_litzenberger.plotting import plot_smile, plot_density

data = report.plot_data()
plot_smile(data).savefig("smile.png")
plot_density(data).savefig("density.png")
```

`plot_data()` raises `NoPlotDataError` if no smile was fitted (verdict
`INSUFFICIENT_LIQUID_DATA`) — check `report.verdict` first if that's possible
for your inputs.

## Handling failures

`extract()` raises a descriptive exception — no retry, no silent fallback —
when the underlying data fetch fails (network error, invalid ticker, no data
for that expiration) or when the requested expiration isn't strictly in the
future. It does **not** raise for market-quality problems (illiquid smile,
detected arbitrage, diagnostic threshold exceeded): those are reported as a
`Verdict` on the returned report instead.

```python
try:
    report = extract("NOT_A_REAL_TICKER", date(2026, 12, 18))
except Exception as exc:
    print(f"fetch failed: {exc}")
```

## CLI

```bash
bl extract --ticker SPY --expiration 2026-12-18
bl extract --ticker SPY --expiration 2026-12-18 --format json
bl extract --ticker SPY --expiration 2026-12-18 --rate 0.045 --dividend-yield 0.013 \
    --min-open-interest 10 --min-volume 10
```

Exit code is non-zero only when `extract()` itself raises (see above); an
unfavorable `Verdict` (e.g. arbitrage detected) is still a successful run and
exits 0 with the report printed.
