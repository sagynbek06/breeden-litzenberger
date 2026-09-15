"""Breeden-Litzenberger risk-neutral density extraction from a fitted smile.

Formula source: Breeden, D.T. & Litzenberger, R.H. (1978), "Prices of
State-Contingent Claims Implicit in Option Prices", Journal of Business
51(4), 621-651. The central result used here:

    f(K) = exp(r*T) * d^2 C(K) / dK^2

i.e. the risk-neutral density at terminal price K is (up to a discount
factor) the second strike-derivative of the European call-price function.
The argument is a static-replication one: a butterfly spread centered at K
with infinitesimal wing width pays off (up to normalization) exactly the
Arrow-Debreu / state price at K, and its cost is the discrete second
difference of adjacent call prices; taking the width to zero turns that
discrete difference into the second derivative above.

Why this module differentiates a *fitted* curve, never raw market prices
(constitution Principle II): a small set of discrete, noisy market quotes
has no meaningful second derivative -- differencing them twice amplifies
quote noise catastrophically. This module's only input is a FittedSmile,
whose SVI curve is smooth and arbitrage-checked *before* it ever reaches
here; density() is not reachable from raw market data at all, by
construction of this module's signature.

Differentiation method: central finite differences on a dense strike grid
(research.md §1), not an analytic second derivative -- chosen because
differentiating the already-smooth SVI-implied price curve makes finite
differences numerically stable (the error is a well-characterized O(h^2)
truncation term, not market noise), while an analytic derivative would add
SVI-chain-rule complexity without an accuracy benefit at this grid density.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import exp, sqrt

import numpy as np

from breeden_litzenberger.core.black_scholes import price
from breeden_litzenberger.core.smile import FittedSmile

# Density-extraction grid (research.md §1 "Grid range" -- resolves
# /speckit.analyze finding A1): +/-8 standard deviations of the fitted
# smile's own ATM total variance in log-forward-moneyness, minimum 2000
# points, converted to strikes via K = F*exp(k).
_GRID_STD_DEVS = 8.0
_GRID_MIN_POINTS = 2000

# Diagnostic flag thresholds fixed in /speckit.clarify (spec SC-004).
_NORMALIZATION_ERROR_THRESHOLD = 0.01
_FORWARD_PRICE_DEVIATION_THRESHOLD = 0.01

# Floor below which a negative density value is treated as central-finite-
# -difference floating-point noise rather than a genuine static-arbitrage
# violation. Empirically, double-differencing (np.gradient applied twice)
# on the 2000-point grid above produces deep-tail noise on the order of
# 1e-12 to 1e-10 even for a provably arbitrage-free, ground-truth-validated
# smile (see specs/001-rnd-extraction/tasks.md T021 development notes) --
# two orders of magnitude below this floor, which stays far tighter than
# any density magnitude a real mispriced smile would produce.
_NON_NEGATIVITY_NOISE_FLOOR = -1e-8


@dataclass(frozen=True)
class DensityGrid:
    """The extracted risk-neutral density (data-model.md)."""

    strikes: np.ndarray
    density: np.ndarray


@dataclass(frozen=True)
class DensityDiagnostics:
    """Mandatory validation output, always populated (spec FR-011, constitution Principle VI)."""

    is_non_negative: bool
    min_density: float
    integral: float
    normalization_error: float
    forward_price_theoretical: float
    forward_price_realized: float
    forward_price_deviation: float
    flagged: bool


def _call_price_curve(strikes: np.ndarray, smile: FittedSmile, spot: float, t: float, r: float, q: float) -> np.ndarray:
    """C(K) reconstructed from the fitted SVI curve (never from raw quotes)."""
    forward = smile.forward_price
    k = np.log(strikes / forward)
    w = smile.params.total_variance(k)
    sigma = np.sqrt(np.maximum(w, 1e-12) / t)
    return np.array([price(spot, float(K), t, r, q, float(s), "call") for K, s in zip(strikes, sigma)])


def extract_density(smile: FittedSmile, spot: float, t: float, r: float, q: float) -> tuple[DensityGrid, DensityDiagnostics]:
    """Reconstruct a dense call-price curve from ``smile`` and differentiate it twice.

    f(K) = exp(r*T) * d^2 C/dK^2 (Breeden & Litzenberger 1978), via central
    finite differences on the grid described in research.md §1.
    """
    atm_w = float(smile.params.total_variance(np.array([0.0]))[0])
    half_range_k = _GRID_STD_DEVS * sqrt(max(atm_w, 1e-8))
    k_grid = np.linspace(-half_range_k, half_range_k, _GRID_MIN_POINTS)
    strikes = smile.forward_price * np.exp(k_grid)

    call_prices = _call_price_curve(strikes, smile, spot, t, r, q)

    # Central finite difference on the (generally non-uniform-in-K, uniform-
    # in-k) grid: convert to a uniform-in-K local second derivative via
    # numpy.gradient applied twice, which handles the non-uniform spacing
    # from the log-grid correctly.
    first_derivative = np.gradient(call_prices, strikes)
    second_derivative = np.gradient(first_derivative, strikes)
    density = exp(r * t) * second_derivative

    integral = float(np.trapezoid(density, strikes))
    is_non_negative = bool(np.min(density) >= _NON_NEGATIVITY_NOISE_FLOOR)
    min_density = float(np.min(density))

    forward_theoretical = spot * exp((r - q) * t)
    forward_realized = float(np.trapezoid(strikes * density, strikes))
    forward_deviation = abs(forward_realized - forward_theoretical) / forward_theoretical
    normalization_error = abs(integral - 1.0)

    flagged = (
        not is_non_negative
        or normalization_error > _NORMALIZATION_ERROR_THRESHOLD
        or forward_deviation > _FORWARD_PRICE_DEVIATION_THRESHOLD
        or not smile.butterfly_arbitrage_free
    )

    grid = DensityGrid(strikes=strikes, density=density)
    diagnostics = DensityDiagnostics(
        is_non_negative=is_non_negative,
        min_density=min_density,
        integral=integral,
        normalization_error=normalization_error,
        forward_price_theoretical=forward_theoretical,
        forward_price_realized=forward_realized,
        forward_price_deviation=forward_deviation,
        flagged=flagged,
    )
    return grid, diagnostics
