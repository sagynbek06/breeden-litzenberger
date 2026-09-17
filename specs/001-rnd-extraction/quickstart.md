# Quickstart: Risk-Neutral Density Extraction Library

Validation guide proving the feature works end-to-end. See `contracts/public_api.md`
for the full interface and `data-model.md` for field-level detail.

## Prerequisites

- Python 3.11+
- Network access (for the real-ticker scenario only; the ground-truth scenario needs
  none)

## Setup

```bash
pip install -e ".[dev]"
```

Installs the package plus `pytest`, `hypothesis`, and `mypy` (dev dependencies).

## Scenario 1 — Ground truth proves the pipeline is correct (User Story 2)

This MUST pass before Scenario 2 is meaningful — it is this project's proof of
correctness (constitution Principle III).

```bash
pytest tests/test_density.py -v
```

**Expected outcome**: all tests pass. The synthetic-lognormal round-trip test asserts
the extracted density's mean and variance are within 0.1% relative error of the known
analytic values, and the density curve matches to 3 decimal places (spec SC-001).

## Scenario 2 — Extract a report from a real option chain (User Story 1)

```python
from datetime import date
from breeden_litzenberger import extract

report = extract("SPY", date(2026, 12, 18))
print(report.summary())
```

**Expected outcome**: a printed summary containing the fitted SVI parameters, fit
quality, arbitrage status, density moments, the forward-price sanity check, and the
smile-shape metrics, ending in a plain-language verdict line (spec FR-013, SC-002,
SC-003).

```python
assert report.diagnostics is not None  # unless Verdict.INSUFFICIENT_LIQUID_DATA
assert report.verdict.name in {
    "WELL_FIT_ARBITRAGE_FREE", "ARBITRAGE_DETECTED",
    "DIAGNOSTIC_WARNING", "INSUFFICIENT_LIQUID_DATA",
}
```

## Scenario 3 — Determinism (spec FR-015, SC-005)

```python
report_a = extract("SPY", date(2026, 12, 18))
report_b = extract("SPY", date(2026, 12, 18))
assert report_a.to_dict() == report_b.to_dict()
```

**Expected outcome**: identical dicts — same market snapshot in, bit-identical
report out, since SVI calibration takes a fixed default initial guess and no
unseeded randomness exists anywhere in the pipeline.

**Verified nuance**: this scenario makes two genuinely separate live fetches,
each capturing its own `fetched_at` timestamp. The pipeline's own
determinism (identical *inputs* → identical output) is unconditionally
guaranteed and unit-tested against a fixed in-memory fixture
(`tests/test_report_integration.py::test_identical_inputs_produce_identical_reports`),
immune to live market movement. Two live calls a few seconds apart on the
*same calendar day* were confirmed to produce identical `to_dict()` output
in practice, because `T` is computed from `fetched_at.date()` (calendar
days), not the full timestamp — but this scenario's pass/fail also depends
on yfinance returning unchanged quotes between the two calls, which is not
itself something this library controls.

## Scenario 4 — Plot-ready data without extra computation (User Story 3, SC-006)

```python
plot_data = report.plot_data()
import matplotlib.pyplot as plt
plt.plot(plot_data.smile_fitted_strikes, plot_data.smile_fitted_ivs)
plt.scatter(plot_data.smile_strikes, plot_data.smile_market_ivs)
```

**Expected outcome**: the fitted smile curve visually passes through/near the raw
market IV points with no additional calculation by the caller.

## Scenario 5 — Data-fetch failure is explicit, not silent (spec FR-001, Edge Cases)

```python
try:
    extract("NOT_A_REAL_TICKER_XYZ", date(2026, 12, 18))
except Exception as exc:
    print(f"Raised as expected: {exc}")
```

**Expected outcome**: a descriptive exception, not an empty or malformed report.

## Scenario 6 — CLI convenience wrapper

```bash
bl extract --ticker SPY --expiration 2026-12-18 --format text
bl extract --ticker SPY --expiration 2026-12-18 --format json
```

**Expected outcome**: the same content as Scenario 2's `summary()`/`to_dict()`,
rendered as text or JSON respectively; non-zero exit code only on a raised exception
(Scenario 5), not on an unfavorable `Verdict`.

## Static gates

```bash
mypy --strict breeden_litzenberger/
```

**Expected outcome**: zero errors (constitution: API & Engineering Standards).
