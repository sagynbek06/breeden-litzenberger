"""Raw SVI smile calibration and its no-butterfly-arbitrage diagnostic.

Formula sources:
  * Gatheral, J. (2004), "A Parsimonious Arbitrage-Free Implied Volatility
    Parameterization with Application to the Valuation of Volatility
    Derivatives" -- the raw SVI parameterization w(k) itself (eq. below).
  * Gatheral, J. & Jacquier, A. (2014), "Arbitrage-free SVI volatility
    surfaces" (arXiv:1204.0646) -- Lemma 2.2 / eq. (2.1), the g(k) function
    whose non-negativity on the whole real line is *necessary and
    sufficient* for a slice to be free of butterfly (static) arbitrage.

Why fitting happens here, in (k, w) space, rather than in (strike, vol)
space: raw SVI's own no-arbitrage machinery (g(k) below) is only tractable
in log-forward-moneyness / total-variance coordinates -- this is the
"standard practitioner convention" constitution Principle VII requires, and
is what makes the arbitrage check below checkable in closed form at all.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from math import sqrt
from typing import Any

import numpy as np
from scipy.optimize import least_squares

from breeden_litzenberger.core.black_scholes import ImpliedVolPoint

# Fixed default initial guess and deterministic alternate-guess retry
# sequence (research.md §2): no randomness anywhere, so calibration is
# bit-identical for identical inputs (constitution Principle V).
_DEFAULT_B = 0.1
_DEFAULT_SIGMA = 0.1
_RHO_RETRY_SEQUENCE = (0.0, -0.5, 0.5)

# Bounds enforcing the raw SVI parameter domain (Gatheral & Jacquier 2014,
# eq. 3.1): b >= 0, |rho| < 1, sigma > 0. rho is bounded strictly inside
# (-1, 1) since the open interval itself isn't representable to a bounded
# optimizer.
_RHO_BOUND = 0.999
_SIGMA_MIN = 1e-6

# Grid used to evaluate g(k) for the arbitrage check (research.md §1 "Grid
# range" -- reused here since it is the same domain the density extraction
# itself will later be evaluated on).
_ARBITRAGE_CHECK_STD_DEVS = 8.0
_ARBITRAGE_CHECK_POINTS = 1000
_G_TOLERANCE = -1e-8  # allow tiny floating-point noise around g(k) == 0


@dataclass(frozen=True)
class SVIParams:
    """Raw SVI parameter set (Gatheral 2004, eq. 3.1 in Gatheral & Jacquier 2014)."""

    a: float
    b: float
    rho: float
    m: float
    sigma: float

    def total_variance(self, k: np.ndarray) -> np.ndarray:
        """w(k) = a + b*(rho*(k-m) + sqrt((k-m)^2 + sigma^2))."""
        u = k - self.m
        return self.a + self.b * (self.rho * u + np.sqrt(u * u + self.sigma * self.sigma))


@dataclass(frozen=True)
class FittedSmile:
    """The calibrated smile plus its own diagnostics (data-model.md)."""

    params: SVIParams
    forward_price: float
    fit_rmse_variance: float
    butterfly_arbitrage_free: bool
    retained_points: list[ImpliedVolPoint] = field(default_factory=list)
    excluded_points: list[Any] = field(default_factory=list)


def _w_derivatives(k: np.ndarray, params: SVIParams) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """w(k), w'(k), w''(k) for raw SVI, differentiated analytically.

    w(k)   = a + b*(rho*u + s),                 u = k - m, s = sqrt(u^2 + sigma^2)
    w'(k)  = b*(rho + u/s)
    w''(k) = b*sigma^2 / s^3
    """
    u = k - params.m
    s = np.sqrt(u * u + params.sigma * params.sigma)
    w = params.a + params.b * (params.rho * u + s)
    dw = params.b * (params.rho + u / s)
    d2w = params.b * params.sigma * params.sigma / (s ** 3)
    return w, dw, d2w


def _g(k: np.ndarray, params: SVIParams) -> np.ndarray:
    """g(k) from Gatheral & Jacquier (2014), eq. (2.1).

    g(k) := (1 - k*w'(k)/(2*w(k)))^2 - (w'(k)^2/4)*(1/w(k) + 1/4) + w''(k)/2

    A slice is free of butterfly arbitrage iff g(k) >= 0 for all real k (and
    a call-price decay condition at k -> +inf, checked separately below).
    """
    w, dw, d2w = _w_derivatives(k, params)
    term1 = (1.0 - k * dw / (2.0 * w)) ** 2
    term2 = (dw * dw / 4.0) * (1.0 / w + 0.25)
    term3 = d2w / 2.0
    return np.asarray(term1 - term2 + term3)


def _check_butterfly_arbitrage_free(params: SVIParams) -> bool:
    """Evaluate g(k) >= 0 over a dense grid (Lemma 2.2 is a for-all-k condition,
    so it is checked numerically, the standard practice for raw SVI which has
    no simpler closed-form parameter-only equivalent -- see research.md).
    """
    atm_total_variance = params.total_variance(np.array([0.0]))[0]
    half_range = _ARBITRAGE_CHECK_STD_DEVS * sqrt(max(atm_total_variance, 1e-8))
    k_grid = np.linspace(-half_range, half_range, _ARBITRAGE_CHECK_POINTS)
    g_values = _g(k_grid, params)
    if np.min(g_values) < _G_TOLERANCE:
        return False

    # Asymptotic call-price-decay condition (Lemma 2.2): lim_{k->+inf} d+(k) = -inf.
    # For raw SVI, w(k) ~ b*(1+rho)*k as k->+inf (from the sqrt((k-m)^2+sigma^2)
    # term linearizing), so d+(k) = -k/sqrt(w(k)) + sqrt(w(k))/2 grows as
    # sqrt(k) * (0.5*c - 1/c) with c = sqrt(b*(1+rho)): this diverges to -inf
    # iff 0.5*c^2 < 1, i.e. b*(1+rho) < 2 -- verified numerically (both sides
    # of the boundary) during development, and consistent with the paper's
    # own reference to Roger Lee's moment formula bounding the wing slope.
    return bool(params.b * (1.0 + params.rho) < 2.0)


def _residuals(x: np.ndarray, k: np.ndarray, w_market: np.ndarray) -> np.ndarray:
    params = SVIParams(a=x[0], b=x[1], rho=x[2], m=x[3], sigma=x[4])
    return np.asarray(params.total_variance(k) - w_market)


def calibrate(points: list[ImpliedVolPoint], forward_price: float) -> FittedSmile:
    """Fit raw SVI to retained implied-vol points via least-squares (spec FR-008).

    Uses scipy.optimize.least_squares (constitution Principle VII: a scipy
    primitive, not a hand-rolled optimizer) with the fixed default initial
    guess, retried from a small fixed (non-random) alternate-guess sequence
    if the first fit is not arbitrage-free (research.md §2) -- deterministic
    throughout, so no seed is needed (constitution Principle V).
    """
    k = np.array([p.log_forward_moneyness for p in points])
    w_market = np.array([p.total_variance for p in points])

    a0 = float(np.min(w_market))
    lower = [-np.inf, 0.0, -_RHO_BOUND, -np.inf, _SIGMA_MIN]
    upper = [np.inf, np.inf, _RHO_BOUND, np.inf, np.inf]

    best_params: SVIParams | None = None
    best_rmse = np.inf
    best_arbitrage_free = False

    for rho0 in _RHO_RETRY_SEQUENCE:
        x0 = [a0, _DEFAULT_B, rho0, 0.0, _DEFAULT_SIGMA]
        result = least_squares(_residuals, x0, bounds=(lower, upper), args=(k, w_market))
        params = SVIParams(a=result.x[0], b=result.x[1], rho=result.x[2], m=result.x[3], sigma=result.x[4])
        rmse = float(np.sqrt(np.mean(result.fun ** 2)))
        arbitrage_free = _check_butterfly_arbitrage_free(params)

        if best_params is None or (arbitrage_free and not best_arbitrage_free) or (
            arbitrage_free == best_arbitrage_free and rmse < best_rmse
        ):
            best_params, best_rmse, best_arbitrage_free = params, rmse, arbitrage_free

        if arbitrage_free:
            break  # first arbitrage-free fit in the fixed retry order wins

    assert best_params is not None
    return FittedSmile(
        params=best_params,
        forward_price=forward_price,
        fit_rmse_variance=best_rmse,
        butterfly_arbitrage_free=best_arbitrage_free,
        retained_points=list(points),
        excluded_points=[],
    )
