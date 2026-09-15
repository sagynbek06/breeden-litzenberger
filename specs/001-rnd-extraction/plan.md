# Implementation Plan: Risk-Neutral Density Extraction Library

**Branch**: `001-rnd-extraction` | **Date**: 2026-09-15 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `specs/001-rnd-extraction/spec.md`

## Summary

Build `breeden_litzenberger`, a Python library that turns a single-expiration option
chain snapshot (fetched via yfinance) into a market-implied risk-neutral probability
density, following the constitution's mandated pipeline: clean and select liquid
OTM quotes → invert to implied vol (Brent's method) → fit a raw SVI smile in
log-forward-moneyness/total-variance space (least-squares, arbitrage-constrained) →
reconstruct a smooth Black-Scholes call-price curve from the fitted smile →
differentiate that smooth curve twice (Breeden & Litzenberger, 1978) to get the
density. Every result ships with mandatory arbitrage/normalization/forward-price
diagnostics and a plain-language verdict, and is validated pre-release against a
synthetic lognormal ground truth to the tolerances fixed during `/speckit.clarify`
(0.1% relative error on mean/variance, 3-decimal-place density match).

## Technical Context

**Language/Version**: Python 3.11+

**Primary Dependencies**: numpy, scipy (`optimize.brentq` for implied-vol inversion,
`optimize.least_squares` for SVI calibration, `stats` for moment cross-checks),
pandas (isolated to the yfinance I/O boundary only), yfinance, matplotlib (plot-data
consumers, not required at import time for core math), typer (CLI convenience layer)

**Storage**: N/A — stateless, single-call library; no persistence layer (per spec
Assumptions)

**Testing**: pytest (reference-value, ground-truth, and integration tests),
hypothesis (property-based tests on the BL convexity/monotonicity invariants),
mypy --strict as a static correctness gate

**Target Platform**: Cross-platform Python library (Linux/macOS/Windows); no
OS-specific dependency

**Project Type**: Single library package with a thin CLI wrapper (not a service)

**Performance Goals**: No hard SLA (single-call library, not a live service, per
clarify session). Informal target: one extraction, excluding network I/O, completes
in low single-digit seconds for a typical chain (tens to a few hundred strikes),
since SVI calibration is a 5-parameter nonlinear least-squares fit, not a large
optimization problem.

**Constraints**: No network access inside `core/` (yfinance/pandas isolated to
`data/`); bit-identical outputs for identical inputs, no unseeded randomness in any
calibration step (Constitution V); public package surface small and typed, internal
helpers underscore-prefixed (Constitution: API & Engineering Standards); every
implemented formula cited in its docstring and in `docs/theory.md` (Constitution I).

**Scale/Scope**: One ticker, one expiration, one fixed-timestamp snapshot per call;
typical option chains of ~20–300 strikes; no multi-expiration, no streaming, no
persistence (spec FR-016, FR-017).

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Gate | Status |
|---|---|---|
| I. Cited Provenance for Every Formula | Every formula module (`black_scholes.py`, `smile.py`, `density.py`) carries a docstring citation; `docs/theory.md` derives all four formulas (BS, BL, raw SVI, Gatheral-Jacquier) in the author's own words | PASS — planned explicitly in Project Structure and `docs/theory.md` scope below |
| II. Smooth-Then-Differentiate, Never Raw | `density.py` differentiates only the smooth call-price curve reconstructed from the **fitted SVI smile**, never raw market points; raw prices only ever reach `black_scholes.py`'s implied-vol inversion | PASS — enforced by module boundary: `density.py` takes a `FittedSmile`, never an `OptionChainSnapshot` |
| III. Ground-Truth Validation Before Real Data | `tests/test_density.py` runs the full pipeline on synthetic lognormal-generated prices and asserts the tolerances fixed in spec SC-001 | PASS — named as THE key test in both spec and this plan's input |
| IV. No Lookahead Bias | One `OptionChainSnapshot` (one ticker, one expiration, one fetch timestamp) flows through the whole pipeline per call; the fetch timestamp is captured once and reused for every downstream time-to-expiration calculation, never re-derived | PASS — see `data-model.md` `OptionChainSnapshot.fetched_at` |
| V. Determinism | SVI calibration uses `scipy.optimize.least_squares` (deterministic, gradient-based) with a fixed default initial guess; no stochastic global optimizer is used in v1, so there is no randomness to seed | PASS — see research.md decision on calibration strategy and its deterministic multi-start fallback |
| VI. Arbitrage-Awareness as First-Class Output | `smile.py` reports the Gatheral-Jacquier butterfly condition; `density.py`/`moments.py` report non-negativity, normalization error, and the forward-price check — all as populated fields on `ExtractionReport`, never swallowed | PASS — see `data-model.md` `FittedSmile`/`DensityDiagnostics` |
| VII. Numerical Stability Over Cleverness | `optimize.brentq` for IV inversion, `optimize.least_squares` for SVI, both scipy primitives; all smile math in log-forward-moneyness/total-variance space | PASS |
| VIII. Documentation Explains Why | `docs/theory.md` scoped as a technical-blog-post derivation of why raw differentiation fails and why each step exists; module docstrings required project-wide (enforced at task level) | PASS |
| API & Engineering Standards | Small typed public surface re-exported from `breeden_litzenberger/__init__.py` (`extract`, `ExtractionReport`); internal helpers underscore-prefixed within each module | PASS — see research.md decision on public surface |
| Git & Release Workflow | Work proceeds on feature branch `001-rnd-extraction`; commits follow Conventional Commits; tag `v0.1.0` on completion of this feature | PASS — process note, enforced during `/speckit.implement` |

No violations requiring justification. Complexity Tracking table omitted (empty).

## Project Structure

### Documentation (this feature)

```text
specs/001-rnd-extraction/
├── plan.md              # This file
├── research.md           # Phase 0 output
├── data-model.md          # Phase 1 output
├── quickstart.md           # Phase 1 output
├── contracts/
│   └── public_api.md        # Phase 1 output — the library's public contract
└── tasks.md                  # Phase 2 output (/speckit-tasks — not created here)
```

### Source Code (repository root)

```text
breeden_litzenberger/
├── __init__.py              # curated public exports: extract(), ExtractionReport, ...
├── data/
│   ├── yfinance_loader.py     # option chain snapshot fetch, spot price (pandas/yfinance live here only)
│   └── rates.py                # risk-free rate / dividend yield helpers
├── core/
│   ├── black_scholes.py         # BS pricing, vega, implied-vol inversion (brentq)
│   ├── smile.py                   # raw SVI parameterization, calibration, Gatheral-Jacquier check
│   ├── density.py                   # Breeden-Litzenberger extraction + non-negativity/normalization checks
│   ├── moments.py                     # RND mean/variance/skew/kurtosis, forward-price sanity check
│   └── smile_metrics.py                 # ATM vol, 25-delta risk reversal, 25-delta butterfly
├── report.py                              # top-level orchestration + ExtractionReport dataclass
├── plotting.py                              # matplotlib: smile fit, RND vs lognormal benchmark
└── cli.py                                     # typer CLI (`bl extract ...`), thin wrapper over report.py

tests/
├── test_black_scholes.py      # reference-value + implied-vol round-trip tests (rtol=1e-6)
├── test_smile_svi.py           # SVI calibration + Gatheral-Jacquier arbitrage checks on synthetic smiles
├── test_density.py              # THE ground-truth synthetic-lognormal round-trip test
├── test_moments.py
├── test_smile_metrics.py
└── test_property_based.py         # hypothesis: convexity → non-negative density, b=0 → symmetric
                                     # lognormal, variance monotonicity, forward-price check

docs/
├── theory.md    # formula derivations + citations (portfolio centerpiece)
└── usage.md

README.md
pyproject.toml
```

**Structure Decision**: Single-package library (no frontend/backend split, no mobile
target). `data/` is the only layer allowed to import pandas/yfinance; `core/` is pure
numpy/scipy and independently unit-testable with zero network dependency, per the
plan input's explicit constraint. `report.py` is the seam where `data/` outputs are
converted into the plain dataclasses `core/` consumes, and where the single top-level
public operation (FR-013) is assembled. `cli.py` and `plotting.py` are thin consumers
of `report.py`'s output, not alternate code paths.

## Complexity Tracking

*No Constitution Check violations — table intentionally empty.*
