"""Moment-computation accuracy vs. known closed-form lognormal moments (tasks.md T011).

For S_T lognormal under Q with E[S_T] = F and Var(ln S_T) = sigma^2*T, the
closed-form moments are the standard lognormal-distribution formulas (e.g.
Johnson, Kotz & Balakrishnan, "Continuous Univariate Distributions" Vol 1,
Ch. 14): mean=F, and the variance/skewness/excess-kurtosis formulas below in
terms of v = sigma^2*T.
"""
from __future__ import annotations

import numpy as np
import pytest

from breeden_litzenberger.core.density import (
    FORWARD_PRICE_RELATIVE_ERROR_THRESHOLD,
    DensityGrid,
    extract_density,
)
from breeden_litzenberger.core.moments import compute_moments, forward_price_check
from breeden_litzenberger.core.smile import FittedSmile, SVIParams, check_butterfly_arbitrage

CASES = [
    # forward, sigma, t
    (100.0, 0.20, 1.0),
    (250.0, 0.35, 0.5),
    (50.0, 0.15, 2.0),
    (100.0, 0.60, 0.1),
]


def _lognormal_grid(forward: float, sigma: float, t: float, n: int = 20000, std_devs: float = 10.0) -> DensityGrid:
    v = sigma * sigma * t
    mu = np.log(forward) - 0.5 * v
    lo = forward * np.exp(-std_devs * np.sqrt(v))
    hi = forward * np.exp(std_devs * np.sqrt(v))
    strikes = np.linspace(lo, hi, n)
    density = (1.0 / (strikes * sigma * np.sqrt(2 * np.pi * t))) * np.exp(
        -((np.log(strikes) - mu) ** 2) / (2 * v)
    )
    return DensityGrid(strikes=strikes, density=density)


def _closed_form_moments(forward: float, sigma: float, t: float) -> tuple[float, float, float, float]:
    v = sigma * sigma * t
    mean = forward
    variance = (np.exp(v) - 1) * forward ** 2
    skewness = (np.exp(v) + 2) * np.sqrt(np.exp(v) - 1)
    excess_kurtosis = np.exp(4 * v) + 2 * np.exp(3 * v) + 3 * np.exp(2 * v) - 6
    return mean, variance, skewness, excess_kurtosis


@pytest.mark.parametrize("forward,sigma,t", CASES)
def test_moments_match_closed_form_lognormal(forward, sigma, t):
    grid = _lognormal_grid(forward, sigma, t)
    got = compute_moments(grid)
    want_mean, want_var, want_skew, want_kurt = _closed_form_moments(forward, sigma, t)

    assert got.mean == pytest.approx(want_mean, rel=1e-4)
    assert got.variance == pytest.approx(want_var, rel=1e-3)
    assert got.skewness == pytest.approx(want_skew, rel=2e-2)
    assert got.excess_kurtosis == pytest.approx(want_kurt, rel=5e-2)


# --- forward_price_check ----------------------------------------------------

_S0, _R, _Q, _T = 98.0, 0.03, 0.01, 1.0
_FORWARD = _S0 * float(np.exp((_R - _Q) * _T))


def test_forward_price_check_passes_for_a_density_whose_mean_is_the_forward():
    grid = _lognormal_grid(_FORWARD, 0.25, _T)

    check = forward_price_check(grid.density, grid.strikes, _S0, _R, _Q, _T)

    assert check.passed
    assert check.theoretical_forward == pytest.approx(_FORWARD, rel=1e-12)
    assert check.realized_mean == pytest.approx(_FORWARD, rel=1e-4)
    assert check.relative_error < 1e-4
    assert check.threshold == FORWARD_PRICE_RELATIVE_ERROR_THRESHOLD


def test_forward_price_check_reports_absolute_and_relative_error_consistently():
    grid = _lognormal_grid(_FORWARD, 0.25, _T)

    check = forward_price_check(grid.density * 1.05, grid.strikes, _S0, _R, _Q, _T)

    # Scaling the density by 1.05 scales its integral of K*f(K), i.e. the mean, by 1.05.
    assert check.realized_mean == pytest.approx(1.05 * _FORWARD, rel=1e-4)
    assert check.absolute_error == pytest.approx(abs(check.realized_mean - _FORWARD), rel=1e-12)
    assert check.relative_error == pytest.approx(check.absolute_error / _FORWARD, rel=1e-12)
    assert check.relative_error == pytest.approx(0.05, rel=1e-2)


def test_forward_price_check_fails_when_mean_deviates_beyond_the_threshold():
    grid = _lognormal_grid(_FORWARD, 0.25, _T)

    too_high = forward_price_check(grid.density * 1.05, grid.strikes, _S0, _R, _Q, _T)
    too_low = forward_price_check(grid.density * 0.95, grid.strikes, _S0, _R, _Q, _T)

    assert not too_high.passed
    assert not too_low.passed  # symmetric: absolute deviation, not signed


def test_forward_price_check_threshold_is_explicit_and_inclusive():
    grid = _lognormal_grid(_FORWARD, 0.25, _T)
    density = grid.density * 1.05
    rel = forward_price_check(density, grid.strikes, _S0, _R, _Q, _T).relative_error

    assert forward_price_check(density, grid.strikes, _S0, _R, _Q, _T, threshold=0.10).passed
    assert forward_price_check(density, grid.strikes, _S0, _R, _Q, _T, threshold=rel).passed  # boundary: <=
    assert not forward_price_check(density, grid.strikes, _S0, _R, _Q, _T, threshold=rel * 0.999).passed


def test_forward_price_check_fails_safe_on_nan_density():
    grid = _lognormal_grid(_FORWARD, 0.25, _T)
    density = grid.density.copy()
    density[10] = np.nan

    check = forward_price_check(density, grid.strikes, _S0, _R, _Q, _T)

    assert np.isnan(check.relative_error)
    assert not check.passed  # a NaN must never read as "within tolerance"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"strikes": np.linspace(50.0, 150.0, 9)},  # length mismatch with the 10-point density below
        {"T": 0.0},
        {"T": -1.0},
        {"S0": 0.0},
    ],
)
def test_forward_price_check_rejects_invalid_inputs(kwargs):
    args = {"rnd_grid": np.ones(10), "strikes": np.linspace(50.0, 150.0, 10), "S0": _S0, "r": _R, "q": _Q, "T": _T}
    args.update(kwargs)
    with pytest.raises(ValueError):
        forward_price_check(**args)


def test_forward_price_check_needs_at_least_two_grid_points():
    with pytest.raises(ValueError):
        forward_price_check(np.array([1.0]), np.array([100.0]), _S0, _R, _Q, _T)


def test_forward_price_check_agrees_with_extract_density_diagnostics():
    # The standalone check and the diagnostics inside extract_density must
    # tell the same story: same realized mean, same pass/fail verdict.
    params = SVIParams(a=0.0625, b=0.0, rho=0.0, m=0.0, sigma=0.1)  # flat smile, 25% vol at T=1
    smile = FittedSmile(
        params=params, forward_price=_FORWARD, fit_rmse_variance=0.0,
        arbitrage_check=check_butterfly_arbitrage(params), retained_points=[], excluded_points=[],
    )
    grid, diagnostics = extract_density(smile, _S0, _T, _R, _Q)

    check = forward_price_check(grid.density, grid.strikes, _S0, _R, _Q, _T)

    assert check.realized_mean == diagnostics.forward_price_realized
    assert check.theoretical_forward == diagnostics.forward_price_theoretical
    assert check.relative_error == pytest.approx(diagnostics.forward_price_deviation, rel=1e-12)
    assert check.passed == (diagnostics.forward_price_deviation <= FORWARD_PRICE_RELATIVE_ERROR_THRESHOLD)
