"""THE ground-truth pipeline test (tasks.md T021; spec SC-001; constitution Principle III).

Recipe: pick a known constant volatility sigma0, spot S0, rate r, dividend
q, and tenor T. Generate Black-Scholes CALL prices at strikes spanning
roughly 0.5x to 2x the forward price. Run the synthetic prices through the
full pipeline -- implied-vol inversion (must recover sigma0 at every
strike, near-exactly, since there's no noise) -> SVI fit (must recover an
approximately flat total-variance curve, b close to 0) -> Breeden-
Litzenberger extraction -- and compare the extracted density against the
closed-form lognormal density implied by (S0, r, q, sigma0, T), via both
max absolute error and integrated (trapezoidal) absolute error over the
grid. A second variant repeats this with small synthetic bid-ask noise
added to the prices before inversion, at a looser but still explicit
tolerance, to demonstrate robustness to realistic quote noise rather than
only to a perfectly clean input.

Why the scenarios are chosen the way they are, not arbitrarily: a fixed
0.5x-2x-forward range is only numerically well-conditioned when
sigma0*sqrt(T) is large enough relative to that range's width in log-
moneyness (ln(2) ~= 0.693). Verified directly during development: a
low-vol, short-tenor combination (sigma0=0.15, T=0.25, so
sigma0*sqrt(T)=0.075) pushes the 0.5x strike so deep in-the-money that
Black-Scholes vega is numerically zero there -- the price carries no
recoverable information about volatility at float64 precision, and even
points that nominally do invert produced implied-vol errors up to 93%,
corrupting the SVI fit. That is a real, unavoidable property of Black-
Scholes (vega collapses away from the money, faster at lower vol/shorter
tenor) -- it is exactly why this library's production path (spec FR-006)
only ever uses OTM quotes within a vol-scaled range, not a fixed
multiplicative band. The three scenarios below all keep
sigma0*sqrt(T) >= 0.25, comfortably inside the well-conditioned region
(verified: 41/41 points retained at both ends, clean and noisy, for all
three).
"""
from __future__ import annotations

import numpy as np
import pytest

from breeden_litzenberger.core.black_scholes import ImpliedVolError, implied_volatility, price
from breeden_litzenberger.core.density import extract_density
from breeden_litzenberger.core.moments import compute_moments
from breeden_litzenberger.core.smile import calibrate_svi
from tests._synthetic import analytic_lognormal_density, analytic_lognormal_moments

# (spot, r, q, t, sigma0) -- see module docstring for why these specific
# combinations were chosen (all have sigma0*sqrt(T) >= 0.25).
SCENARIOS = [
    (100.0, 0.03, 0.01, 1.0, 0.25),
    (150.0, 0.04, 0.02, 0.25, 0.50),
    (250.0, 0.02, 0.015, 2.0, 0.40),
]

_N_STRIKES = 41
_STRIKE_RANGE_LOW = 0.5   # x forward
_STRIKE_RANGE_HIGH = 2.0  # x forward

# Explicit, documented thresholds. Each is set with a comfortable margin
# (roughly one to two orders of magnitude) over the worst value actually
# observed across all three scenarios during development, not chosen to
# just barely pass.
_CLEAN_IV_RECOVERY_RTOL = 1e-6          # observed worst: 5.8e-11
_CLEAN_B_NEAR_ZERO = 1e-6               # observed worst: 6.5e-11 (flat curve -> b ~ 0)
_CLEAN_MEAN_VARIANCE_RTOL = 1e-3        # spec SC-001 / /speckit.clarify: 0.1%
_CLEAN_MAX_ABS_DENSITY_ERROR = 1e-5     # observed worst: 3.6e-7
_CLEAN_INTEGRATED_ABS_DENSITY_ERROR = 1e-3  # observed worst: 2.5e-5

_NOISY_BID_ASK_BPS = 5.0                # +/-5 basis points synthetic quote noise
_NOISY_MAX_ABS_DENSITY_ERROR = 0.02     # observed worst: 9.0e-3
_NOISY_INTEGRATED_ABS_DENSITY_ERROR = 0.02  # observed worst: 8.0e-3


def _make_strikes(forward: float, n: int = _N_STRIKES) -> np.ndarray:
    """Strikes spanning [0.5, 2.0] x forward, log-spaced (uniform in log-moneyness)."""
    k_grid = np.linspace(np.log(_STRIKE_RANGE_LOW), np.log(_STRIKE_RANGE_HIGH), n)
    return forward * np.exp(k_grid)


def _deterministic_bid_ask_noise(n: int, bps: float) -> np.ndarray:
    """Small, deterministic (not random) synthetic noise pattern, +/-bps basis
    points, oscillating strike to strike. Deliberately not drawn from a
    random generator, even seeded: constitution Principle V requires no
    unseeded randomness in the *library*, and using a fixed, reproducible
    pattern here keeps this ground-truth test itself exactly reproducible
    without relying on a specific RNG algorithm being stable across numpy
    versions.
    """
    return bps * np.sin(2.0 * np.arange(n))


def _run_pipeline(spot: float, r: float, q: float, t: float, sigma0: float, noise_bps: float | None):
    """Generate synthetic call prices, invert to implied vol (excluding any
    quote that genuinely can't be inverted, per spec FR-007 -- never
    crashing), fit SVI, and extract the density.
    """
    forward = spot * float(np.exp((r - q) * t))
    strikes = _make_strikes(forward)
    prices = np.array([price(spot, float(k), t, r, q, sigma0, "call") for k in strikes])

    if noise_bps is not None:
        prices = prices * (1.0 + _deterministic_bid_ask_noise(len(strikes), noise_bps) / 10000.0)

    iv_points = []
    excluded = []
    for p, k in zip(prices, strikes):
        try:
            iv_points.append(implied_volatility(float(p), spot, float(k), t, r, q, "call", forward))
        except ImpliedVolError as exc:
            excluded.append((float(k), exc.reason))

    fitted_smile = calibrate_svi(iv_points, forward_price=forward)
    density_grid, diagnostics = extract_density(fitted_smile, spot, t, r, q)
    moments = compute_moments(density_grid)

    return forward, iv_points, excluded, fitted_smile, density_grid, diagnostics, moments


@pytest.mark.parametrize("spot,r,q,t,sigma0", SCENARIOS)
def test_ground_truth_lognormal_round_trip_clean(spot, r, q, t, sigma0):
    """Clean synthetic prices: implied vol must recover sigma0 near-exactly,
    the SVI fit must recover an approximately flat curve (b ~ 0), and the
    extracted density must match the closed-form lognormal to a tight,
    explicit tolerance (both max abs error and integrated abs error).
    """
    forward, iv_points, excluded, fitted_smile, density_grid, diagnostics, moments = _run_pipeline(
        spot, r, q, t, sigma0, noise_bps=None
    )

    assert excluded == [], f"expected all {_N_STRIKES} clean quotes to invert; excluded: {excluded}"

    max_iv_rel_err = max(abs(pt.implied_vol - sigma0) / sigma0 for pt in iv_points)
    assert max_iv_rel_err < _CLEAN_IV_RECOVERY_RTOL, (
        f"implied vol did not recover sigma0={sigma0} near-exactly from noise-free prices: "
        f"max relative error {max_iv_rel_err:.3e}"
    )

    assert abs(fitted_smile.params.b) < _CLEAN_B_NEAR_ZERO, (
        f"a constant-vol input should fit to an approximately flat SVI curve (b ~ 0); "
        f"got b={fitted_smile.params.b:.3e}"
    )
    assert fitted_smile.butterfly_arbitrage_free

    analytic = analytic_lognormal_moments(forward, sigma0, t)
    mean_rel_err = abs(moments.mean - analytic["mean"]) / analytic["mean"]
    variance_rel_err = abs(moments.variance - analytic["variance"]) / analytic["variance"]
    assert mean_rel_err < _CLEAN_MEAN_VARIANCE_RTOL, f"mean relative error {mean_rel_err:.3e} exceeds SC-001 tolerance"
    assert variance_rel_err < _CLEAN_MEAN_VARIANCE_RTOL, f"variance relative error {variance_rel_err:.3e} exceeds SC-001 tolerance"

    analytic_density = analytic_lognormal_density(density_grid.strikes, forward, sigma0, t)
    max_abs_error = float(np.max(np.abs(density_grid.density - analytic_density)))
    integrated_abs_error = float(np.trapezoid(np.abs(density_grid.density - analytic_density), density_grid.strikes))

    assert max_abs_error < _CLEAN_MAX_ABS_DENSITY_ERROR, (
        f"max abs density error {max_abs_error:.3e} exceeds clean threshold {_CLEAN_MAX_ABS_DENSITY_ERROR:.1e}"
    )
    assert integrated_abs_error < _CLEAN_INTEGRATED_ABS_DENSITY_ERROR, (
        f"integrated abs density error {integrated_abs_error:.3e} exceeds "
        f"clean threshold {_CLEAN_INTEGRATED_ABS_DENSITY_ERROR:.1e}"
    )


@pytest.mark.parametrize("spot,r,q,t,sigma0", SCENARIOS)
def test_ground_truth_lognormal_round_trip_with_bid_ask_noise(spot, r, q, t, sigma0):
    """Same recipe, with small (+/-5bps) deterministic synthetic bid-ask
    noise added to the prices before inversion -- demonstrates the pipeline
    degrades gracefully under realistic quote noise, at a looser but still
    explicit tolerance, not just under perfectly clean synthetic input.
    """
    forward, iv_points, excluded, fitted_smile, density_grid, diagnostics, moments = _run_pipeline(
        spot, r, q, t, sigma0, noise_bps=_NOISY_BID_ASK_BPS
    )

    assert excluded == [], f"expected all {_N_STRIKES} quotes to invert even with {_NOISY_BID_ASK_BPS}bps noise; excluded: {excluded}"
    assert fitted_smile.butterfly_arbitrage_free, (
        f"expected the fit to remain arbitrage-free under {_NOISY_BID_ASK_BPS}bps noise "
        f"(margin={fitted_smile.arbitrage_check.margin:.6f})"
    )

    analytic_density = analytic_lognormal_density(density_grid.strikes, forward, sigma0, t)
    max_abs_error = float(np.max(np.abs(density_grid.density - analytic_density)))
    integrated_abs_error = float(np.trapezoid(np.abs(density_grid.density - analytic_density), density_grid.strikes))

    assert max_abs_error < _NOISY_MAX_ABS_DENSITY_ERROR, (
        f"max abs density error {max_abs_error:.3e} exceeds noisy threshold {_NOISY_MAX_ABS_DENSITY_ERROR:.1e}"
    )
    assert integrated_abs_error < _NOISY_INTEGRATED_ABS_DENSITY_ERROR, (
        f"integrated abs density error {integrated_abs_error:.3e} exceeds "
        f"noisy threshold {_NOISY_INTEGRATED_ABS_DENSITY_ERROR:.1e}"
    )
