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

from breeden_litzenberger.core.density import DensityGrid
from breeden_litzenberger.core.moments import compute_moments

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
