# Theory: From Option Prices to a Risk-Neutral Density

This document derives, in my own words, every formula this library implements,
and explains the numerical reasoning behind the order they're applied in. If
you only read one section, read [§1](#1-the-core-problem-why-you-cant-just-differentiate-market-prices) —
it's the reason this library exists at all.

## 1. The core problem: why you can't just differentiate market prices

Breeden & Litzenberger (1978) showed something remarkable: the market's entire
risk-neutral probability distribution for where an asset will end up at
expiration is already encoded in today's option prices, and you can extract it
with nothing more than a second derivative. Concretely, for a European call
price `C(K)` as a function of strike `K`, with risk-free rate `r` and time to
expiration `T`:

```
f(K) = e^{rT} * d^2 C(K) / dK^2
```

`f(K)` is a genuine probability density — it's non-negative, it integrates to
1, and its mean equals the forward price. The problem is entirely in the
phrase "as a function of strike." You don't have `C(K)` as a smooth function.
You have `C` sampled at a handful of discrete strikes — typically a few dozen
listed contracts, spaced $2.50, $5, or $10 apart depending on the underlying —
each with its own bid-ask spread and quoting noise on top of the "true" price.

Numerical differentiation amplifies noise, and it amplifies it worse the
higher the derivative order. A first derivative estimated from noisy, unevenly
spaced points is already sensitive to that noise; a *second* derivative from
the same data is dramatically more sensitive, because you're differencing
differences. Try it directly on 15-20 real listed strikes and the resulting
"density" is a jagged, often negative mess that has nothing to do with the
market's actual view — you're mostly extracting quantization and bid-ask
noise, not signal.

**This library never does that.** The one rule everything else in this
document serves is: *never differentiate raw market points.* Instead, it
takes a three-step detour:

1. Convert each raw price to an **implied volatility** (§2) — this alone
   doesn't fix the noise problem, but it moves into a coordinate system
   (volatility, not price) where the *next* step is tractable.
2. Fit a **smooth, arbitrage-constrained curve** through those implied
   vols — the raw SVI parameterization (§3), checked against a real
   arbitrage condition (§4). This is where the noise actually gets
   filtered out: a 5-parameter curve fit to 20-40 points cannot chase
   every individual bid-ask wiggle, by construction.
3. Convert the *fitted, smooth* curve back to a call-price function and
   differentiate **that** (§5). Because the input to this differentiation
   is now an analytic, infinitely-differentiable function instead of a
   handful of noisy points, the numerical differentiation step that was
   catastrophic in step 0 becomes routine.

Every other design decision in this codebase — why SVI, why check for
arbitrage before extracting a density, why validate against a known
distribution before touching real data — is downstream of this one problem.

## 2. Black-Scholes pricing and implied volatility

Reference: Black, F. & Scholes, M. (1973), "The Pricing of Options and
Corporate Liabilities," *Journal of Political Economy* 81(3), 637-654. The
continuous dividend-yield extension is the standard Merton (1973) adjustment.

```
d1 = [ln(S/K) + (r - q + 0.5*sigma^2)*T] / (sigma*sqrt(T))
d2 = d1 - sigma*sqrt(T)

C(K) = S*e^{-qT}*N(d1) - K*e^{-rT}*N(d2)      (call)
P(K) = K*e^{-rT}*N(-d2) - S*e^{-qT}*N(-d1)    (put)
```

`S` is spot, `K` is strike, `r` the risk-free rate, `q` the dividend yield,
`sigma` the volatility, `T` time to expiration in years, and `N` the standard
normal CDF. This library never *uses* this formula in the direction shown
above during extraction — it uses it inverted. Given a market price and every
other input, there's a unique `sigma` (implied volatility) that reproduces it,
because `C` is strictly increasing in `sigma` (its derivative, vega, is always
positive). That uniqueness is exactly what makes root-finding for it
well-posed: `implied_volatility()` brackets `sigma` in `[1%, 500%]` and uses
Brent's method (`scipy.optimize.brentq`) rather than a hand-rolled
Newton-Raphson, because vega collapses toward zero far from the money — a
gradient-based method can stall or diverge exactly where market noise is
worst, while Brent's method (bisection's guaranteed convergence, combined with
a faster interpolation step when it's safe) never leaves the economically sane
bracket.

Quotes that can't be inverted at all — priced below intrinsic value, priced at
zero, or with no volatility in the bracket that reproduces the price — are
excluded with a recorded reason, never silently coerced to a placeholder.
Garbage in that step would become garbage everywhere downstream, at exactly
the resolution (the smooth fitted curve) meant to filter garbage *out*.

**Two subtleties in "priced below intrinsic value" worth stating explicitly,
both found by actually generating a wide range of synthetic prices and
inverting them, not by inspection.** First, the correct lower bound for a
*European* option is the **discounted-forward** intrinsic value —
`max(S*e^{-qT} - K*e^{-rT}, 0)` for calls, `max(K*e^{-rT} - S*e^{-qT}, 0)` for
puts — not the undiscounted `max(S-K,0)`/`max(K-S,0)` bound that's valid for
*American* early exercise. At a meaningful rate and tenor these two bounds
diverge materially; using the American bound would incorrectly reject
legitimate European quotes (verified: a real, unremarkable European put price
at spot=50, strike=51.68, T=1.5y, r=4.7% sat below its naive American bound
of 1.68 while its true price, 1.627, was perfectly valid — the correct
discounted-forward bound there is actually 0). Second, for a deep
in-the-money option with very little time value left, `price()` and the
intrinsic-value formula are two independently-computed expressions that can
differ by a few floating-point ULPs even when mathematically identical; the
comparison carries a small (`1e-9` relative), explicitly documented tolerance
for exactly this, verified against a real case where the gap was `~1.6e-16`
relative — machine epsilon, not a mispricing. Deep enough in the money,
`vega -> 0` and implied volatility becomes numerically unrecoverable by any
method regardless of tolerance — that's not a bug to fix, it's the same
"vega collapses away from the money" fact mentioned above, and it's part of
why this library restricts itself to out-of-the-money quotes in the first
place (§6).

## 3. The raw SVI smile

Reference: Gatheral, J. (2004), "A Parsimonious Arbitrage-Free Implied
Volatility Parameterization with Application to the Valuation of Volatility
Derivatives."

Rather than fit strike vs. implied vol directly, this library works in
**log-forward-moneyness** and **total implied variance**:

```
k = ln(K / F)              (F = forward price)
w(k) = sigma(k)^2 * T
```

This is the standard practitioner convention, and it isn't arbitrary: the
no-arbitrage machinery in §4 is only checkable in closed form in these
coordinates, and `w(k)` behaves far better numerically than `sigma(k)` does —
`w` is what naturally interpolates and extrapolates smoothly across a smile
that flattens near the money and steepens in the wings.

The raw SVI parameterization models one expiration's entire smile with five
numbers:

```
w(k) = a + b * ( rho * (k - m) + sqrt((k - m)^2 + sigma^2) )
```

with domain `a ∈ R`, `b >= 0`, `|rho| < 1`, `m ∈ R`, `sigma > 0`. Each
parameter has a direct, visual meaning: `a` is the overall variance level,
`b` controls how steep both wings are, `rho` tilts the smile (skew), `m`
shifts it left/right, and `sigma` controls the ATM curvature. This library
fits all five by nonlinear least-squares (`scipy.optimize.least_squares`)
against the retained implied-vol points, with a fixed default initial guess
and a small, fixed retry sequence — deterministic throughout, so identical
inputs always produce identical output (no stochastic optimizer, no seed
needed because there is no randomness to seed).

This is the noise-filtering step referred to in §1: five parameters fit
across 20-40 points cannot reproduce every individual point's bid-ask noise.
It reproduces the *shape* of the smile and discards the rest — which is
precisely what makes the density extraction in §5 numerically stable.

## 4. No static arbitrage: Gatheral & Jacquier's g(k) condition

Reference: Gatheral, J. & Jacquier, A. (2014), "Arbitrage-free SVI volatility
surfaces" (arXiv:1204.0646).

A fitted smile isn't automatically sane. Because a European call price is a
convex, decreasing function of strike in any arbitrage-free market, its
second derivative must be non-negative everywhere — that's exactly the
density from §5's Breeden-Litzenberger formula. If the fitted smile is
badly shaped, the implied "density" can go negative somewhere: a real
violation of no-arbitrage (a *butterfly spread* — long one wing, short two at
the middle, long the other wing — would have negative cost), not just an
ugly curve.

Gatheral & Jacquier prove (their Lemma 2.2) that a slice is free of butterfly
arbitrage **if and only if**

```
g(k) := (1 - k*w'(k) / (2*w(k)))^2 - (w'(k)^2 / 4) * (1/w(k) + 1/4) + w''(k)/2 >= 0
```

for every real `k`, together with the boundary condition
`lim_{k -> +inf} d+(k) = -inf` (call prices must actually decay to zero as
strikes go to infinity, not just have a non-negative second derivative
locally). This library evaluates `g(k)` on a dense grid spanning ±8 standard
deviations of the smile's own ATM total variance, and separately verified
(numerically, both sides of the boundary) that the asymptotic condition
reduces to the closed form `b*(1+rho) < 2` for raw SVI.

**A note on why this is a grid evaluation and not a single algebraic
inequality**, because it's the kind of subtlety this project is built to get
right rather than paper over: Gatheral & Jacquier's own Theorem 4.2 *does*
give a closed-form, purely algebraic sufficient condition —
`b*(1+|rho|) < 2` and `b^2*(1+|rho|) <= theta` — but only for their **SSVI**
surface construction, in which the raw-SVI shift parameter `m` is not free;
it's fixed by `rho` and `sigma` (`m = -rho*sigma/sqrt(1-rho^2)`) as part of
that construction. This library fits `m` independently, to track the
market's actual observed skew location rather than a mathematically
convenient constraint. During development, applying Theorem 4.2's condition
directly to an independently-fitted `m` produced a real false positive — a
parameter set with a genuinely negative `g(k)` (real butterfly arbitrage)
that the closed-form check nonetheless called arbitrage-free — confirmed by
computing `g(k)` directly, not merely suspected. Gatheral & Jacquier
themselves note (§3.4 of the paper) that no general closed-form condition on
unconstrained raw SVI is known to exist; the grid evaluation of the exact
Lemma 2.2 condition is the correct tool for that case, and it's what this
library uses.

The result carries a **margin**, not just a boolean: the minimum value of
`g(k)` found on the grid. A large positive margin means comfortably
arbitrage-free; a value near zero means marginal; negative means a real
violation, and the calibration retry sequence will attempt a different
initial guess before reporting it.

## 5. Breeden-Litzenberger density extraction

Reference: Breeden, D.T. & Litzenberger, R.H. (1978), "Prices of
State-Contingent Claims Implicit in Option Prices," *Journal of Business*
51(4), 621-651.

The convexity argument behind the formula in §1 is a static replication one.
Consider a butterfly spread centered at strike `K` with wing width `h`: long
one call at `K-h`, short two calls at `K`, long one call at `K+h`. Its payoff
is zero everywhere except a triangular spike of height `h` at `K`. As `h -> 0`,
that payoff converges (after dividing by `h^2` to normalize its area to 1) to
a Dirac delta at `K` — exactly the payoff of an Arrow-Debreu security paying
$1 if the terminal price lands at `K`. The butterfly's cost is
`C(K-h) - 2*C(K) + C(K+h)`, a discrete second difference; dividing by `h^2`
and taking `h -> 0` turns that discrete difference into `d^2C/dK^2`, and
discounting the state price back to a probability density gives the `e^{rT}`
factor. No-arbitrage pricing theory says today's price of a claim paying $1
at outcome `K` *is* (up to discounting) the risk-neutral probability of that
outcome — which is exactly what §4 checked before we got here.

This library reconstructs a dense grid of Black-Scholes call prices from the
*fitted SVI curve* (never raw quotes — that's the whole point of §1-§4), then
differentiates that smooth curve via **central finite differences**:

```
f(K) ≈ e^{rT} * [C(K-h) - 2*C(K) + C(K+h)] / h^2
```

**Why finite differences rather than an analytic second derivative**: an
analytic derivative of the SVI-implied Black-Scholes price is algebraically
available (chain rule through `w(k)`'s own derivatives), but adds real
implementation and review surface for no accuracy benefit at this grid
density — the input curve is already smooth by construction (§1-§3), so
central differences are numerically stable here in a way they categorically
are not on raw market points. The error characteristic is the standard
`O(h^2)` truncation term; `h` is chosen small enough to keep that error
negligible relative to this project's ground-truth tolerance (§7) but large
enough to stay above floating-point cancellation noise — both failure modes
are directly, empirically checked by that same ground-truth test.

The extraction grid spans `k ∈ [-8, +8] * sqrt(ATM total variance)` — wide
enough to capture over 99.999% of the tail mass at realistic equity
volatilities, without extrapolating so far into the wings that the fitted
SVI curve's linear-in-variance asymptotic behavior (itself only checked by §4
out to the grid boundary) becomes speculative.

**Every extracted density ships with mandatory diagnostics**, never silently
discarded: is it non-negative everywhere (allowing a tight, empirically
characterized floating-point noise floor around exactly zero); does it
integrate to 1; does its mean match the theoretical forward price
`F = S0 * e^{(r-q)T}` — the single most important sanity check in the whole
pipeline, since it's a completely independent identity from anything used to
fit the smile.

## 6. The American-vs-European approximation

Every formula above assumes **European** exercise (exercise only at
expiration). Real listed U.S. equity and index options are typically
**American** (exercise any time up to expiration), and yfinance's option
chain data is exactly that: real, tradeable, American-style contracts.

American options carry an early-exercise premium relative to their
European counterparts — it's never optimal to exercise a call early absent
dividends, but it can be optimal for puts, and for calls when a dividend is
imminent, so American option prices can exceed the Black-Scholes-implied
European price for in-the-money contracts.

This library sidesteps the problem rather than modeling it: it only ever
uses **out-of-the-money** quotes — OTM puts for strikes below the forward,
OTM calls above it. The early-exercise premium on an OTM option is
economically small (there's little value to exercising early into a position
that's currently worth less than intrinsic, because there's no intrinsic
value to capture), so treating OTM American quotes as approximately European
introduces a much smaller, second-order error than either (a) using ITM
quotes directly and ignoring the premium entirely, or (b) building a full
American pricing model (binomial trees, or a free-boundary PDE solver) just
for this correction. This is a deliberate, disclosed approximation, not an
oversight: it's the standard convention practitioners use precisely because
it's small and empirically well-behaved, not because it's exact.

## 7. Ground-truth validation

Everything above is a chain of individually well-established results. The
only way to know the *chain* — this specific pipeline, this specific
codebase — is implemented correctly is to run it on an input where the
right answer is known in closed form, and check the output matches.

This library generates synthetic option prices from a constant-volatility
Black-Scholes model — a model whose true risk-neutral density is, by
construction, the closed-form lognormal distribution — and runs the *entire*
pipeline (implied-vol inversion → SVI fit → Breeden-Litzenberger extraction)
on those synthetic prices. The extracted density's mean and variance must
match the analytic lognormal values to within 0.1% relative error, and the
density curve itself must match to 3 decimal places, across multiple
vol/tenor scenarios. In practice this pipeline reproduces the known analytic
density to within `1e-4`–`1e-5` relative error on the moments — two to three
orders of magnitude inside the required tolerance.

This is this project's equivalent of a textbook's worked example, and it is
the test that actually proves the pipeline is correct. Every other test in
this codebase is regression coverage layered on top of it.
