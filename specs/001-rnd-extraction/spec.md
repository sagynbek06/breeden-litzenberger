# Feature Specification: Risk-Neutral Density Extraction Library

**Feature Branch**: `001-rnd-extraction`

**Created**: 2026-09-15

**Status**: Draft

**Input**: User description: "Build a Python library called `breeden-litzenberger` that extracts the market-implied risk-neutral probability density function of an underlying's price at expiration, from a snapshot of listed option prices, via Breeden & Litzenberger (1978)."

## Clarifications

### Session 2026-09-15

- Q: What numerical tolerance should the ground-truth validation (SC-001) require between the extracted density and the known closed-form lognormal density? → A: Very strict — relative error < 0.1% on mean/variance, density curve matches to 3 decimal places.
- Q: What should the library do when the yfinance data fetch itself fails or is unavailable (network error, rate limiting, or no data for a valid ticker/expiration)? → A: Raise a descriptive exception immediately; no retry, no fallback — caller decides what to do.
- Q: Should the "large error" threshold for flagging density-normalization and forward-price diagnostics on real market data be a fixed, spec-level number, or left as a configurable/implementation-defined constant? → A: Fix it now — normalization error > 1% (density integral deviates from 1 by more than 1%) or forward-price deviation > 1% triggers a flag.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Extract a risk-neutral density report from a live option chain (Priority: P1)

A user supplies a ticker symbol and a target expiration date. The library fetches the
matching option chain snapshot and spot price, cleans and selects the liquid
out-of-the-money quotes, fits a smooth arbitrage-aware volatility smile, and returns a
single structured report containing the extracted risk-neutral density, its diagnostics
(non-negativity, normalization, forward-price sanity check), and its moments (mean,
variance, skewness, excess kurtosis).

**Why this priority**: This is the entire reason the library exists — everything else is
in service of producing this one trustworthy result from a single function call.

**Independent Test**: Call the top-level extraction function with a real, liquid ticker and
a near-term expiration; verify a report is returned with a fitted smile, a density grid,
all diagnostics populated, and a plain-language verdict string.

**Acceptance Scenarios**:

1. **Given** a liquid ticker with a valid future expiration date, **When** the user requests
   an extraction, **Then** the library returns a report containing fitted SVI parameters,
   fit quality, arbitrage diagnostics, the density grid, its moments, the forward-price
   sanity check, and smile-shape metrics.
2. **Given** the same ticker and expiration are extracted twice with identical inputs
   (rate, dividend yield, liquidity floor, and the same market snapshot data), **When**
   the user compares the two reports, **Then** the results are identical.
3. **Given** a ticker/expiration pair with too few liquid OTM quotes to calibrate a smile,
   **When** the user requests an extraction, **Then** the library returns a report whose
   verdict clearly states the smile could not be fit, rather than a misleading or partial
   numeric result.

---

### User Story 2 - Validate the pipeline against a known distribution (Priority: P1)

A user (or the library's own test suite) runs the full extraction pipeline — implied
volatility inversion, smile fitting, and density extraction — on option prices generated
from a known, closed-form lognormal distribution instead of real market data, in order to
confirm the pipeline recovers that known distribution to a tight tolerance before it is
ever trusted on real data.

**Why this priority**: Establishes correctness before the library is applied to anything
real; without this, no other result can be trusted.

**Independent Test**: Generate synthetic option prices from a constant-volatility
Black-Scholes model, run the full extraction pipeline, and compare the extracted density
to the closed-form lognormal density at the same parameters.

**Acceptance Scenarios**:

1. **Given** synthetic option prices generated from a known constant-volatility lognormal
   model, **When** the full extraction pipeline is run on them, **Then** the extracted
   density's mean and variance match the closed-form analytic density within 0.1%
   relative error, and the density curve matches to 3 decimal places.

---

### User Story 3 - Inspect smile shape and export data for plotting (Priority: P2)

A user who already has an extraction report wants the desk-convention smile-shape
numbers (ATM implied volatility, 25-delta risk reversal, 25-delta butterfly) and the
underlying data series needed to plot the fitted smile against the raw implied-vol points,
and the extracted density against a lognormal benchmark at the same forward and ATM vol.

**Why this priority**: These are the numbers and visuals a reviewer or the user
themselves uses to sanity-check and present the result; they are not required to produce
the core density, so this is independently valuable but secondary to Story 1.

**Independent Test**: From a completed extraction report, retrieve the smile-shape
metrics and the plotting data series, and confirm both are present and internally
consistent with the fitted smile in the same report.

**Acceptance Scenarios**:

1. **Given** a completed extraction report, **When** the user requests the smile-shape
   metrics, **Then** ATM implied volatility, 25-delta risk reversal, and 25-delta
   butterfly are all returned as numeric values.
2. **Given** a completed extraction report, **When** the user requests plotting data,
   **Then** the library returns the fitted smile curve, the raw retained implied-vol
   points, the extracted density curve, and a lognormal benchmark density at the same
   forward and ATM vol — all as plain data, not rendered images.

---

### Edge Cases

- What happens when the requested expiration has no listed options, or the ticker is
  invalid/delisted? The library MUST raise a descriptive exception rather than
  returning an empty or malformed result (see also the data-fetch-failure case below).
- What happens when every available quote at a strike is illiquid (fails the liquidity
  floor) or crossed? The strike MUST be excluded, and if this leaves too few points to
  calibrate a smile, the report MUST say so explicitly (see User Story 1, Scenario 3).
- What happens when an option's mid price is below its intrinsic value (a data error or
  extreme illiquidity), so no implied volatility can reproduce it? The quote MUST be
  excluded with a logged reason, never silently coerced to a placeholder volatility.
- What happens when the SVI calibration converges to parameters that violate the
  no-butterfly-arbitrage condition? The report MUST flag this explicitly as a diagnostic,
  not hide it or silently re-fit without disclosure.
- What happens when the extracted density integrates to a value noticeably different
  from 1, or has any negative region? Both MUST be surfaced as flagged diagnostics, never
  silently renormalized or clipped without disclosure.
- What happens when the dividend yield cannot be determined for the underlying? The
  library MUST fall back to zero but MUST make that fallback visible in the report output,
  never assume it silently.
- What happens when the target expiration date has already passed relative to the
  snapshot being used? The library MUST reject this input rather than computing a
  negative or zero time-to-expiration.
- What happens when the underlying data fetch fails or is unavailable (network error,
  rate limiting, or no data returned for a valid ticker/expiration)? The library MUST
  raise a descriptive exception immediately, with no automatic retry or silent
  fallback; the caller decides how to handle it.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST accept a ticker symbol and a target expiration date and fetch,
  for that single expiration, the full snapshot of listed call and put option quotes
  (across all available strikes) together with the underlying's current spot price. If
  this fetch fails or returns no usable data (network error, rate limiting, or an
  invalid/delisted ticker or expiration), the system MUST raise a descriptive exception
  immediately, with no automatic retry or silent fallback.
- **FR-002**: System MUST determine a risk-free rate for the matching tenor, defaulting to
  a short-dated Treasury yield proxy when not supplied, and MUST allow the caller to
  override this value.
- **FR-003**: System MUST determine a dividend yield estimate for the underlying,
  defaulting to its trailing dividend yield when available and to zero otherwise, MUST
  allow the caller to override this value, and MUST always report which value was used
  and whether it was a zero fallback.
- **FR-004**: System MUST compute time-to-expiration in years using a documented,
  explicit calendar-day convention, and MUST reject expiration dates that are not
  strictly in the future relative to the snapshot.
- **FR-005**: System MUST compute a mid price for every quote and MUST exclude quotes
  with a zero bid, a crossed market (bid greater than ask), or open interest and volume
  both below a configurable liquidity floor.
- **FR-006**: System MUST assemble one clean implied-volatility smile per expiration
  using out-of-the-money puts for strikes below the forward price and out-of-the-money
  calls for strikes above the forward price, and MUST document this American-as-European
  approximation explicitly rather than leaving it implicit.
- **FR-007**: System MUST invert each retained quote to an implied volatility via a
  bounded root-finding search over a documented sane volatility range, and MUST exclude
  or explicitly reject (with a stated reason) any quote priced below intrinsic value,
  priced at zero, or for which no volatility in that range reproduces the market price.
  Placeholder or default volatility values MUST never be silently substituted.
- **FR-008**: System MUST fit a single smooth, arbitrage-aware volatility smile per
  expiration to the retained implied-volatility points, in log-forward-moneyness and
  total-variance space, subject to documented parameter domain constraints.
- **FR-009**: System MUST report, for every fitted smile: a fit-quality measure, and
  whether the fitted smile satisfies the sufficient condition for absence of static
  butterfly arbitrage.
- **FR-010**: System MUST extract a risk-neutral probability density from the fitted,
  smooth smile — never from raw, unsmoothed market prices — across a strike range wide
  enough to extend beyond the quoted strikes.
- **FR-011**: System MUST report, for every extracted density, whether it is
  non-negative everywhere, how closely it integrates to 1 (with the normalization error
  reported rather than silently corrected), its mean compared against the theoretical
  forward price, and its variance, skewness, and excess kurtosis.
- **FR-012**: System MUST compute, from the fitted smile, at-the-money implied
  volatility, the 25-delta risk reversal, and the 25-delta butterfly.
- **FR-013**: System MUST expose a single top-level operation that accepts a ticker,
  expiration, and optional overrides (rate, dividend yield, liquidity floor) and returns
  one structured report containing the fitted smile parameters, fit quality, arbitrage
  diagnostics, the density and its moments, the forward-price sanity check, the
  smile-shape metrics, and a plain-language verdict summarizing the outcome (e.g.
  well-fit and arbitrage-free, arbitrage detected, or insufficient liquid data).
- **FR-014**: System MUST make every report available as structured data, as a
  human-readable text summary, and as the data series needed to plot the fitted smile
  against the raw retained implied-vol points and the extracted density against a
  lognormal benchmark at the same forward price and ATM volatility.
- **FR-015**: System MUST produce identical outputs when given identical inputs (same
  snapshot data, same overrides), with no unseeded randomness in any calibration step.
- **FR-016**: System MUST operate on exactly one fixed-timestamp snapshot for one
  underlying and one expiration per invocation, and MUST NOT combine quotes from
  different snapshots or expirations within a single extraction.
- **FR-017**: System's scope MUST exclude multi-expiration term structure or
  calendar-arbitrage analysis across maturities, live/streaming data feeds, connections
  to any trading or execution system, and any position-sizing logic.

### Key Entities

- **Option Chain Snapshot**: The full set of listed call and put quotes for one
  underlying and one expiration at one point in time, plus the underlying's spot price.
- **Option Quote**: A single strike's bid, ask, last price, open interest, and volume,
  for either a call or a put.
- **Clean OTM Point**: A retained, liquid, non-crossed quote selected as out-of-the-money
  relative to the forward price, reduced to a mid price.
- **Implied Volatility Point**: A clean OTM point after inversion, expressed as strike (or
  log-forward-moneyness) paired with an implied volatility (or total variance).
- **Fitted Smile**: The calibrated volatility smile for one expiration, its parameters,
  fit-quality measure, and its no-butterfly-arbitrage status.
- **Extracted Density**: The risk-neutral probability density over terminal underlying
  price implied by the fitted smile, represented as a grid, together with its
  non-negativity status, normalization error, and moments.
- **Smile-Shape Metrics**: ATM implied volatility, 25-delta risk reversal, and 25-delta
  butterfly for one fitted smile.
- **Extraction Report**: The single combined result of one extraction — inputs used,
  fitted smile, extracted density, all diagnostics, smile-shape metrics, and verdict —
  renderable as structured data, text, or plot-ready series.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: When run on option prices synthesized from a known constant-volatility
  lognormal model, the extracted density's mean and variance match the closed-form
  analytic density to within 0.1% relative error, and the extracted density curve
  matches the analytic density curve to 3 decimal places, on every run.
- **SC-002**: For a liquid real-world ticker and near-term expiration, a user obtains a
  complete report — fitted smile, density, all diagnostics, smile-shape metrics, and a
  verdict — from a single request, with no manual data wrangling.
- **SC-003**: Every extraction, successful or not, produces an explicit, human-readable
  verdict describing the outcome; a user is never left with a numeric result whose
  trustworthiness is unstated.
- **SC-004**: When a fitted smile violates the no-arbitrage condition, or the extracted
  density is negative anywhere, or its normalization error (density integral deviating
  from 1) or forward-price deviation exceeds 1%, this is flagged in the report 100% of
  the time — never silently passed through.
- **SC-005**: Two extractions run with identical inputs produce identical numeric
  results, every time.
- **SC-006**: A user can obtain the data needed to plot the fitted smile and the
  extracted density without performing any additional calculation beyond calling the
  single top-level operation.

## Assumptions

- A configurable liquidity floor is required by the feature but no numeric default is
  specified by the business request; a conservative default (excluding quotes with both
  zero open interest and zero volume) is assumed and remains fully overridable.
- The short-dated Treasury yield proxy for the default risk-free rate is a specific data
  source selection left to the implementation phase; this spec only requires that a
  sensible default exists and is overridable.
- yfinance is assumed to be an available and sufficient data source for option chains,
  spot price, and trailing dividend yield for the tickers this library targets; no
  fallback data source is in scope.
- The library targets standalone, single-call, single-snapshot use (e.g. notebook or
  script usage, or as a dependency imported by another codebase); no persistence layer,
  web service, or scheduled/streaming execution mode is in scope.
- "Renderable as structured data, text, and plot-ready series" (FR-014) refers to the
  shape and content of the output, not any specific plotting library or file format;
  actual chart rendering is a caller concern, not this library's responsibility.
