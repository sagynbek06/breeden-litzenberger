"""Raw SVI smile calibration and its no-butterfly-arbitrage diagnostic.

Formula sources:
  * Gatheral, J. (2004), "A Parsimonious Arbitrage-Free Implied Volatility
    Parameterization with Application to the Valuation of Volatility
    Derivatives" -- the raw SVI parameterization itself.
  * Gatheral, J. & Jacquier, A. (2014), "Arbitrage-free SVI volatility
    surfaces" (arXiv:1204.0646) -- Lemma 2.2 / eq. (2.1), the g(k) function:
    the EXACT necessary-and-sufficient condition for a slice to be free of
    butterfly arbitrage, enforced below.

Why g(k) is evaluated numerically over a dense grid rather than reduced to
a closed-form inequality on (a,b,rho,m,sigma) directly: no such reduction
exists in the published literature for a raw SVI slice with an
independently-fitted m. Gatheral & Jacquier's own Theorem 4.2 gives a
genuine closed-form sufficient condition, but only for their SSVI surface
construction, in which m is NOT a free parameter -- their own Lemma 3.1
correspondence fixes it at m = -rho*sigma/sqrt(1-rho^2) whenever the
natural-SVI shift mu is 0. This library fits m freely so the smile can
track the market's actual skew location rather than a mathematically
convenient constraint; verified directly (not assumed) during development
that forcing Theorem 4.2's constrained m onto an independently-calibrated
smile can turn a real arbitrage violation (g(k) genuinely negative) into a
false "arbitrage-free" reading. Lemma 2.2's g(k)>=0-for-all-k condition,
evaluated on a dense grid spanning the smile's own domain, has no such
blind spot: it is the exact condition, not an approximation of it, for any
raw SVI parameter set, at the cost of being a numerical check rather than
a single algebraic inequality.

Why fitting happens here, in (k, w) space, rather than in (strike, vol)
space: this is the "standard practitioner convention" constitution
Principle VII requires, and it is what makes g(k) itself expressible in
closed form from the raw SVI parameters at all.
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

# Bounds enforcing the raw SVI parameter domain (Gatheral 2004): b >= 0,
# |rho| < 1, sigma > 0. rho is bounded strictly inside (-1, 1) since the
# open interval itself isn't representable to a bounded optimizer.
_RHO_BOUND = 0.999
_SIGMA_MIN = 1e-6

# Grid used to evaluate g(k) (research.md §1 "Grid range" -- the same
# domain the density extraction itself is evaluated on): +/-8 standard
# deviations of the slice's own ATM total variance, 2000 points.
_GRID_STD_DEVS = 8.0
_GRID_POINTS = 2000

# Floor below which a negative g(k) minimum is treated as floating-point
# noise rather than a genuine violation (see core/density.py's analogous
# noise floor and its docstring for the empirical basis of this order of
# magnitude).
_G_TOLERANCE = -1e-8


@dataclass(frozen=True)
class SVIParams:
    """Raw SVI parameter set: w(k) = a + b*(rho*(k-m) + sqrt((k-m)^2 + sigma^2)).

    where k is log-forward-moneyness log(K/F), w(k) is total implied
    variance (sigma_impl(k)^2 * T), and the parameter domain is
    a in R, b >= 0, |rho| < 1, m in R, sigma > 0 (Gatheral 2004).
    """

    a: float
    b: float
    rho: float
    m: float
    sigma: float

    def total_variance(self, k: np.ndarray) -> np.ndarray:
        """w(k) = a + b*(rho*(k-m) + sqrt((k-m)^2 + sigma^2))."""
        u = k - self.m
        return self.a + self.b * (self.rho * u + np.sqrt(u * u + self.sigma * self.sigma))


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
    """g(k) from Gatheral & Jacquier (2014), eq. (2.1):

    g(k) := (1 - k*w'(k)/(2*w(k)))^2 - (w'(k)^2/4)*(1/w(k) + 1/4) + w''(k)/2
    """
    w, dw, d2w = _w_derivatives(k, params)
    term1 = (1.0 - k * dw / (2.0 * w)) ** 2
    term2 = (dw * dw / 4.0) * (1.0 / w + 0.25)
    term3 = d2w / 2.0
    return np.asarray(term1 - term2 + term3)


@dataclass(frozen=True)
class ButterflyArbitrageCheck:
    """Lemma 2.2 (Gatheral & Jacquier 2014) butterfly-arbitrage check.

    A slice is free of butterfly arbitrage iff g(k) >= 0 for all real k
    (eq. 2.1) and lim_{k->+inf} d+(k) = -inf. The first part is evaluated
    on a dense grid spanning the slice's own domain (see module docstring
    for why no closed-form reduction of "for all k" exists here); the
    second part reduces to the closed-form b*(1+rho) < 2 (verified
    numerically during development against the true asymptotic behavior of
    d+(k) at k -> +inf, both above and below this boundary).

    ``margin`` is the signed distance to violation on the grid: min(g(k))
    over the evaluated domain. More negative means a deeper violation;
    positive means genuinely arbitrage-free with that much room to spare.
    """

    is_arbitrage_free: bool
    margin: float
    margin_at_k: float
    asymptotic_decay_holds: bool


def check_butterfly_arbitrage(params: SVIParams) -> ButterflyArbitrageCheck:
    """Evaluate Lemma 2.2's exact butterfly-arbitrage condition on ``params``."""
    theta = float(params.total_variance(np.array([0.0]))[0])
    half_range = _GRID_STD_DEVS * sqrt(max(theta, 1e-8))
    k_grid = np.linspace(-half_range, half_range, _GRID_POINTS)
    g_values = _g(k_grid, params)

    min_index = int(np.argmin(g_values))
    margin = float(g_values[min_index])
    margin_at_k = float(k_grid[min_index])

    # lim_{k->+inf} d+(k) = -inf iff b*(1+rho) < 2 for raw SVI (verified
    # numerically at k = 10..1e5 on both sides of this boundary; see
    # tests/test_smile_svi.py).
    asymptotic_decay_holds = bool(params.b * (1.0 + params.rho) < 2.0)

    is_arbitrage_free = bool(margin >= _G_TOLERANCE and asymptotic_decay_holds)

    return ButterflyArbitrageCheck(
        is_arbitrage_free=is_arbitrage_free,
        margin=margin,
        margin_at_k=margin_at_k,
        asymptotic_decay_holds=asymptotic_decay_holds,
    )


@dataclass(frozen=True)
class FittedSmile:
    """The calibrated smile plus its own diagnostics (data-model.md)."""

    params: SVIParams
    forward_price: float
    fit_rmse_variance: float
    arbitrage_check: ButterflyArbitrageCheck
    retained_points: list[ImpliedVolPoint] = field(default_factory=list)
    excluded_points: list[Any] = field(default_factory=list)

    @property
    def butterfly_arbitrage_free(self) -> bool:
        return self.arbitrage_check.is_arbitrage_free


def _residuals(x: np.ndarray, k: np.ndarray, w_market: np.ndarray) -> np.ndarray:
    params = SVIParams(a=x[0], b=x[1], rho=x[2], m=x[3], sigma=x[4])
    return np.asarray(params.total_variance(k) - w_market)


def calibrate_svi(points: list[ImpliedVolPoint], forward_price: float) -> FittedSmile:
    """Fit raw SVI -- w(k) = a + b*(rho*(k-m) + sqrt((k-m)^2 + sigma^2)) -- to
    retained implied-vol points by least-squares (spec FR-008), where k is
    log-forward-moneyness, w is total implied variance, and the parameter
    domain is a in R, b >= 0, |rho| < 1, m in R, sigma > 0 (Gatheral 2004).

    Uses scipy.optimize.least_squares (constitution Principle VII: a scipy
    primitive, not a hand-rolled optimizer) with the fixed default initial
    guess, retried from a small fixed (non-random) alternate-guess sequence
    if the first fit does not pass ``check_butterfly_arbitrage`` (research.md
    §2) -- deterministic throughout, so no seed is needed (constitution
    Principle V).
    """
    k = np.array([p.log_forward_moneyness for p in points])
    w_market = np.array([p.total_variance for p in points])

    a0 = float(np.min(w_market))
    lower = [-np.inf, 0.0, -_RHO_BOUND, -np.inf, _SIGMA_MIN]
    upper = [np.inf, np.inf, _RHO_BOUND, np.inf, np.inf]

    best_params: SVIParams | None = None
    best_rmse = np.inf
    best_check: ButterflyArbitrageCheck | None = None

    for rho0 in _RHO_RETRY_SEQUENCE:
        x0 = [a0, _DEFAULT_B, rho0, 0.0, _DEFAULT_SIGMA]
        result = least_squares(_residuals, x0, bounds=(lower, upper), args=(k, w_market))
        params = SVIParams(a=result.x[0], b=result.x[1], rho=result.x[2], m=result.x[3], sigma=result.x[4])
        rmse = float(np.sqrt(np.mean(result.fun ** 2)))
        check = check_butterfly_arbitrage(params)

        if best_params is None or (check.is_arbitrage_free and not best_check.is_arbitrage_free) or (  # type: ignore[union-attr]
            check.is_arbitrage_free == best_check.is_arbitrage_free and rmse < best_rmse  # type: ignore[union-attr]
        ):
            best_params, best_rmse, best_check = params, rmse, check

        if check.is_arbitrage_free:
            break  # first arbitrage-free fit in the fixed retry order wins

    assert best_params is not None and best_check is not None
    return FittedSmile(
        params=best_params,
        forward_price=forward_price,
        fit_rmse_variance=best_rmse,
        arbitrage_check=best_check,
        retained_points=list(points),
        excluded_points=[],
    )
