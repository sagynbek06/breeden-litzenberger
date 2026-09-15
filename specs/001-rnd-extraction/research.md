# Phase 0 Research: Risk-Neutral Density Extraction Library

All items below were genuinely open after `/speckit.specify` + `/speckit.clarify`
(spec intentionally left them as implementation-phase decisions) or were introduced
by this plan's stack input and need a documented rationale before Phase 1 design.

## 1. Breeden-Litzenberger differentiation method

**Decision**: Central finite differences on the dense strike grid of Black-Scholes
call prices reconstructed from the fitted SVI curve — not an analytic second
derivative.

**Rationale**: The spec explicitly permits either, conditioned on documenting the
choice and its error characteristics (constitution Principle VII, spec FR-010).
Because differentiation happens on the *smooth, fitted* SVI-implied price curve
(never raw prices — Principle II), central differences are numerically stable here
in a way they would not be on raw market points; the grid can be made arbitrarily
dense at negligible cost, and the error is a well-understood O(h²) truncation term,
tunable and independent of market noise. A closed-form second derivative of the
SVI-implied BS price w.r.t. strike is algebraically available but couples the BS
chain rule to the SVI parameterization's own derivatives, adding implementation and
review surface without a numerical-accuracy benefit at this grid density.

**Alternatives considered**: Analytic second derivative via the BS/SVI chain rule —
rejected for v1 as unnecessary complexity; may be revisited if profiling shows
finite-difference grid density is a bottleneck (unlikely given the informal
performance target).

**Documented error characteristic** (to go in `docs/theory.md` and the `density.py`
docstring): central difference error is O(h²) in the grid spacing h; h is chosen
small enough to keep truncation error negligible relative to the SC-001 tolerance
(0.1% relative error on mean/variance) but large enough to stay above float64
cancellation noise — verified directly by the ground-truth test, which is sensitive
to both failure modes.

**Grid range** (resolves `/speckit.analyze` finding A1 — FR-010's "wide enough to
extend beyond the quoted strikes" was otherwise unquantified): the dense grid spans
log-forward-moneyness `k ∈ [-8, +8] * sqrt(atm_total_variance)` (i.e. ±8 standard
deviations of the fitted smile's own ATM total variance), converted to strikes via
`K = F * exp(k)`, with a minimum of 2000 grid points. Rationale: this captures
>99.999% of the lognormal tail mass at typical equity volatilities (so the SC-001/
SC-004 normalization checks aren't starved by a truncated grid), while staying
inside the region where raw SVI's linear-in-variance wing behavior is well-behaved,
rather than extrapolating arbitrarily far past where the Gatheral-Jacquier check was
evaluated. A fixed *absolute* strike range (e.g. spot ± 50%) was rejected because it
doesn't scale with the fitted smile's own ATM vol/tenor — too narrow for high-vol or
long-dated names, needlessly wide for low-vol/short-dated ones.

## 2. SVI calibration strategy and determinism

**Decision**: `scipy.optimize.least_squares` (Trust Region Reflective, supports
box bounds) minimizing squared error in total-variance space, with a fixed default
initial guess derived analytically from the retained points (`a` from the minimum
observed total variance, `b=0.1`, `rho=0.0`, `m=0.0`, `sigma=0.1`), subject to bounds
`b >= 0`, `-1 < rho < 1`, `sigma > 0`. If this converges to a fit that fails the
Gatheral-Jacquier no-butterfly-arbitrage condition, retry from a small, fixed
(non-random) grid of alternate initial guesses (e.g. varying `rho` over
`{-0.5, 0, 0.5}`) and keep the best arbitrage-free result; if none is arbitrage-free,
report the best fit with the arbitrage flag set (per Principle VI — never hide it).

**Rationale**: `least_squares` is a scipy optimization primitive (Principle VII),
deterministic given fixed inputs (Principle V) — since no stochastic global search is
used, there is no randomness to seed, and multi-start retries use a fixed
deterministic sequence rather than random restarts, preserving bit-identical output
for identical input.

**Alternatives considered**: `scipy.optimize.minimize` (L-BFGS-B) — equivalent
choice, `least_squares` preferred because the objective is naturally a sum of squared
residuals; `scipy.optimize.differential_evolution` or other global/stochastic
optimizers — rejected for v1, since they would require an explicit seed parameter
end-to-end (constitution requires this *if* used) and add complexity not justified
unless local convergence proves insufficient in practice.

## 3. Risk-free rate default source

**Decision**: `^IRX` (13-week / 3-month Treasury bill discount yield) via yfinance,
converted from a discount-yield quote to a continuously compounded rate before use
in Black-Scholes.

**Rationale**: Matches the spec's "short-dated Treasury yield proxy" requirement
(FR-002) and is tenor-appropriate for the near-dated equity/index options this
library targets; always overridable per FR-002.

**Alternatives considered**: `^FVX` (5-year) — rejected as tenor-mismatched for
typical near-term option expirations this library is built for.

## 4. Dividend yield default source

**Decision**: yfinance `Ticker.info['trailingAnnualDividendYield']`, falling back to
`Ticker.info['dividendYield']`, falling back to `0.0` if neither is present or is
`None`.

**Rationale**: Matches FR-003's "trailing dividend yield if available, else 0"
requirement; the fallback chain and the resulting value/source are both surfaced on
`ExtractionReport` per FR-003 and the corresponding edge case.

## 5. Public API surface shape

**Decision**: `breeden_litzenberger/__init__.py` re-exports exactly: `extract`
(the FR-013 top-level operation), `ExtractionReport`, `Verdict` (the enum backing
the verdict string), and `LiquidityFloor` (the override type for FR-005's
configurable floor). All other names — including every name inside `core/` and
`data/` — are reachable only via explicit submodule import and are not part of the
supported contract; functions and classes not intended for reuse outside their
module are underscore-prefixed.

**Rationale**: Satisfies "public API surface is small and typed" (constitution: API
& Engineering Standards) while still allowing the documented future consumer (a
separate Greeks Dashboard project, per spec context) to reach into `core/` submodules
directly if it needs lower-level pieces (e.g. `core.smile`) — that is an explicit,
deliberate submodule import, not part of the small curated top-level surface.

## 6. CI

**Decision**: GitHub Actions workflow running `pytest` and `mypy --strict` on every
push and pull request.

**Rationale**: Matches the plan input's requirement that `mypy --strict` gate the
project in CI; GitHub Actions is the natural choice absent any other CI system
referenced in this repository.

## 7. Time-to-expiration and "no lookahead" boundary

**Decision**: The option chain snapshot's fetch timestamp (captured once, in
`data/yfinance_loader.py`, at the moment of the fetch) is the single source of "now"
for the entire extraction. `T = (expiration_date - fetched_at.date()).days / 365.25`
is computed once in `report.py` from that stored timestamp and passed down
explicitly; no downstream module calls a clock.

**Rationale**: Directly implements spec FR-004's calendar-day convention and
constitution Principle IV (no notion of "now" beyond what's passed in); makes the
snapshot's `fetched_at` field the auditable record of what "now" meant for that
extraction, satisfying the constitution's own definition of the boundary.

## 8. 25-delta strike location for smile-shape metrics

**Decision** (resolves `/speckit.analyze` finding U1 — FR-012 named the metrics but
not how a "25-delta strike" is located on the smile): solve for the strike
`K_delta` such that the Black-Scholes delta at `(K_delta, SVI-implied vol at
K_delta)` equals `+0.25` (call) or `-0.25` (put), via `scipy.optimize.brentq`
bracketed over the fitted smile's *calibrated* log-moneyness domain — i.e. the span
of the retained `ImpliedVolPoint`s, not the wide extrapolated grid from §1.

**Rationale**: structurally the same kind of inversion as FR-007's implied-vol
inversion (a function monotonic in strike, within the calibrated region), so it
gets the same treatment per constitution Principle VII: a scipy root-finding
primitive, not a hand-rolled iterative scheme. Bracketing to the calibrated domain
(rather than the full extrapolated density grid) avoids reporting a 25-delta point
in a region the SVI fit was never validated against.

**Alternatives considered**: a closed-form delta-to-moneyness approximation
(e.g. assuming a flat smile) — rejected, since it would reintroduce exactly the
kind of raw-approximation shortcut constitution Principle II warns against; a full
root-find against the actual fitted smile is barely more expensive and stays
consistent with the rest of the pipeline.
