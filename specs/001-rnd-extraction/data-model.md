# Phase 1 Data Model: Risk-Neutral Density Extraction Library

All types below are plain `dataclass`es (or `Enum`s) over `numpy`/builtin types only.
Per the constitution and this plan's `core/`-vs-`data/` boundary, **no pandas
DataFrame or yfinance object crosses out of `data/`** — `data/yfinance_loader.py` and
`data/rates.py` are responsible for converting their raw pandas/yfinance results into
the dataclasses below before anything reaches `report.py` or `core/`.

## OptionQuote

One listed quote at one strike, for one option type.

| Field | Type | Notes |
|---|---|---|
| `strike` | `float` | Must be `> 0` |
| `option_type` | `Literal["call", "put"]` | |
| `bid` | `float` | `>= 0` |
| `ask` | `float` | `>= 0` |
| `last_price` | `float` | Reference only; not used for mid-price math |
| `open_interest` | `int` | `>= 0` |
| `volume` | `int` | `>= 0` |

Derived (not stored, computed on demand): `mid_price = (bid + ask) / 2`.

## OptionChainSnapshot

The full input to one extraction (spec FR-001, FR-016; constitution Principle IV).

| Field | Type | Notes |
|---|---|---|
| `ticker` | `str` | |
| `expiration` | `date` | Target expiration; validated `> fetched_at.date()` (FR-004) |
| `fetched_at` | `datetime` | Captured once at fetch time; the library's sole source of "now" (research.md §7) |
| `spot_price` | `float` | `> 0` |
| `quotes` | `list[OptionQuote]` | Both calls and puts, all available strikes |

## RateInputs

Risk-free rate and dividend yield actually used, with provenance (spec FR-002,
FR-003 — "never silently assumed").

| Field | Type | Notes |
|---|---|---|
| `risk_free_rate` | `float` | Continuously compounded, annualized |
| `risk_free_rate_source` | `Literal["treasury_proxy", "override"]` | |
| `dividend_yield` | `float` | Annualized |
| `dividend_yield_source` | `Literal["trailing_yield", "zero_fallback", "override"]` | `zero_fallback` MUST always be visible when it occurs |

## ExcludedQuote

A quote dropped during cleaning or IV inversion, with the reason (spec FR-005,
FR-007 — exclusions are logged, never silent).

| Field | Type | Notes |
|---|---|---|
| `strike` | `float` | |
| `option_type` | `Literal["call", "put"]` | |
| `reason` | `Literal["zero_bid", "crossed_market", "below_liquidity_floor", "not_otm", "below_intrinsic_value", "zero_price", "iv_bracket_failure"]` | `"not_otm"` is retained in the type for completeness but is no longer produced in practice: OTM selection now happens at fetch time (`data/yfinance_loader.py::fetch_otm_chain`), so a non-OTM quote is simply never fetched rather than fetched-then-excluded. |

## CleanOTMPoint

A retained quote after liquidity/crossed-market/OTM selection (spec FR-005, FR-006).

| Field | Type | Notes |
|---|---|---|
| `strike` | `float` | |
| `option_type` | `Literal["call", "put"]` | |
| `mid_price` | `float` | `> 0` |

## ImpliedVolPoint

A `CleanOTMPoint` after Black-Scholes inversion (spec FR-007), in both native and
SVI-fitting coordinates (constitution Principle VII).

| Field | Type | Notes |
|---|---|---|
| `strike` | `float` | |
| `implied_vol` | `float` | Annualized; within the documented sane bracket (e.g. 1%–500%) |
| `log_forward_moneyness` | `float` | `k = ln(strike / forward_price)` |
| `total_variance` | `float` | `w = implied_vol² * T` |

## SVIParams

Raw SVI parameterization (Gatheral 2004), fit per expiration (spec FR-008).

| Field | Type | Notes |
|---|---|---|
| `a` | `float` | Level |
| `b` | `float` | `>= 0` |
| `rho` | `float` | `-1 < rho < 1` |
| `m` | `float` | Horizontal shift |
| `sigma` | `float` | `> 0`, curvature |

## FittedSmile

The calibrated smile plus its own diagnostics (spec FR-008, FR-009).

| Field | Type | Notes |
|---|---|---|
| `params` | `SVIParams` | |
| `forward_price` | `float` | `F = S0 * exp((r - q) * T)` |
| `fit_rmse_variance` | `float` | RMSE in total-variance space against retained `ImpliedVolPoint`s |
| `butterfly_arbitrage_free` | `bool` | Gatheral-Jacquier (2014) sufficient condition result |
| `retained_points` | `list[ImpliedVolPoint]` | What was actually fit, for plotting/audit |
| `excluded_points` | `list[ExcludedQuote]` | What was dropped and why |

## DensityGrid

The extracted density itself (spec FR-010), a dense strike grid beyond the quoted
range.

| Field | Type | Notes |
|---|---|---|
| `strikes` | `np.ndarray[float]` | Dense grid, extends beyond quoted strikes |
| `density` | `np.ndarray[float]` | `f(K)` at each grid strike |

## DensityDiagnostics

Mandatory, always-populated validation output (spec FR-011; constitution
Principle VI; clarify session thresholds).

| Field | Type | Notes |
|---|---|---|
| `is_non_negative` | `bool` | `False` ⇒ static arbitrage violation |
| `min_density` | `float` | For audit even when non-negative |
| `integral` | `float` | Should be ≈ 1 |
| `normalization_error` | `float` | `abs(integral - 1)`; flagged if `> 1%` (SC-004) |
| `forward_price_theoretical` | `float` | `F = S0 * exp((r - q) * T)` |
| `forward_price_realized` | `float` | Mean of the extracted density |
| `forward_price_deviation` | `float` | Relative deviation; flagged if `> 1%` (SC-004) |
| `flagged` | `bool` | `True` if any of the above thresholds trip, or `not is_non_negative`, or `not butterfly_arbitrage_free` |

## DensityMoments

(spec FR-011.)

| Field | Type | Notes |
|---|---|---|
| `mean` | `float` | Should ≈ `forward_price_theoretical` |
| `variance` | `float` | |
| `skewness` | `float` | |
| `excess_kurtosis` | `float` | |

## SmileShapeMetrics

Desk-convention descriptors (spec FR-012).

| Field | Type | Notes |
|---|---|---|
| `atm_implied_vol` | `float` | |
| `risk_reversal_25d` | `float` | 25-delta call vol − 25-delta put vol |
| `butterfly_25d` | `float` | (25-delta call vol + 25-delta put vol)/2 − ATM vol |

## Verdict

Plain-language outcome classification (spec FR-013).

| Value | Meaning |
|---|---|
| `WELL_FIT_ARBITRAGE_FREE` | Smile fit converged, no-arbitrage holds, diagnostics within tolerance |
| `ARBITRAGE_DETECTED` | Butterfly-arbitrage condition failed at fitted parameters |
| `DIAGNOSTIC_WARNING` | Fit converged and is arbitrage-free but normalization or forward-price deviation exceeds 1% |
| `INSUFFICIENT_LIQUID_DATA` | Too few retained `CleanOTMPoint`s to calibrate a smile |

## ExtractionReport

The single combined result (spec FR-013, FR-014) — the library's public contract
object (see `contracts/public_api.md`).

| Field | Type | Notes |
|---|---|---|
| `ticker` | `str` | Echoed input |
| `expiration` | `date` | Echoed input |
| `snapshot` | `OptionChainSnapshot` | For audit/reproducibility |
| `rates` | `RateInputs` | Echoed, with provenance |
| `fitted_smile` | `FittedSmile \| None` | `None` only when verdict is `INSUFFICIENT_LIQUID_DATA` |
| `density_grid` | `DensityGrid \| None` | `None` only when verdict is `INSUFFICIENT_LIQUID_DATA` |
| `diagnostics` | `DensityDiagnostics \| None` | `None` only when verdict is `INSUFFICIENT_LIQUID_DATA` |
| `moments` | `DensityMoments \| None` | `None` only when verdict is `INSUFFICIENT_LIQUID_DATA` |
| `smile_shape_metrics` | `SmileShapeMetrics \| None` | `None` only when verdict is `INSUFFICIENT_LIQUID_DATA` |
| `verdict` | `Verdict` | Always populated |
| `verdict_message` | `str` | Human-readable rendering of `verdict` plus the specific reason |

Methods (behavior, not stored fields): `to_dict() -> dict`, `summary() -> str`
(human-readable text render, FR-014a), `plot_data() -> PlotData` (FR-014c —
fitted-smile curve, raw retained IV points, extracted density curve, lognormal
benchmark density at the same forward/ATM vol; see `contracts/public_api.md`).

## Validation Rules Summary (cross-reference to spec)

- `OptionQuote` exclusion rules (zero bid, crossed market, liquidity floor) → FR-005,
  recorded as `ExcludedQuote(reason=...)`.
- OTM selection rule (puts below forward, calls above forward) → FR-006.
- IV inversion exclusion rules (below intrinsic, zero price, no bracket root) → FR-007,
  recorded as `ExcludedQuote(reason=...)`, never coerced to a placeholder.
- SVI parameter domain (`b >= 0`, `|rho| < 1`, `sigma > 0`) → FR-008, enforced as
  `least_squares` bounds (research.md §2).
- `DensityDiagnostics.flagged` thresholds (1% normalization error, 1% forward-price
  deviation) → SC-004, fixed in `/speckit.clarify`.
- Ground-truth tolerance (0.1% relative error on mean/variance, 3-decimal density
  match) → SC-001, fixed in `/speckit.clarify`; enforced in `tests/test_density.py`,
  not on `ExtractionReport` itself (it's a test assertion, not a runtime field).
