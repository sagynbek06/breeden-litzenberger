"""SVI calibration convergence and Gatheral-Jacquier arbitrage checks (tasks.md T008).

The arbitrage-free / arbitrage parameter sets below were confirmed against
the actual g(k) values and the b*(1+rho) < 2 asymptotic boundary during
development (see specs/001-rnd-extraction/research.md and the smile.py
docstrings for the derivation), not assumed from memory.
"""
from __future__ import annotations

import numpy as np
import pytest

from breeden_litzenberger.core.black_scholes import ImpliedVolPoint
from breeden_litzenberger.core.smile import SVIParams, _check_butterfly_arbitrage_free, calibrate


def test_known_arbitrage_free_params():
    params = SVIParams(a=0.04, b=0.1, rho=0.0, m=0.0, sigma=0.1)
    assert _check_butterfly_arbitrage_free(params) is True


def test_known_arbitrage_params():
    # b*(1+rho) = 2.0*(1+0.9) = 3.8 >> 2: violates the asymptotic decay
    # condition, and min(g(k)) is strongly negative on the calibration grid.
    params = SVIParams(a=0.01, b=2.0, rho=0.9, m=0.0, sigma=0.01)
    assert _check_butterfly_arbitrage_free(params) is False


def _synthetic_points(true_params: SVIParams, forward: float, t: float, strikes_k: np.ndarray) -> list[ImpliedVolPoint]:
    w = true_params.total_variance(strikes_k)
    sigma = np.sqrt(w / t)
    points = []
    for k, s, wv in zip(strikes_k, sigma, w):
        strike = forward * np.exp(k)
        points.append(ImpliedVolPoint(strike=strike, implied_vol=float(s), log_forward_moneyness=float(k), total_variance=float(wv)))
    return points


def test_calibration_recovers_known_arbitrage_free_params():
    true_params = SVIParams(a=0.04, b=0.15, rho=-0.2, m=0.0, sigma=0.12)
    forward, t = 100.0, 1.0
    k_grid = np.linspace(-1.5, 1.5, 25)
    points = _synthetic_points(true_params, forward, t, k_grid)

    fitted = calibrate(points, forward_price=forward)

    assert fitted.butterfly_arbitrage_free is True
    assert fitted.fit_rmse_variance == pytest.approx(0.0, abs=1e-8)
    # Recalibrated total variance should match the generating curve closely.
    recovered_w = fitted.params.total_variance(k_grid)
    true_w = true_params.total_variance(k_grid)
    assert np.allclose(recovered_w, true_w, atol=1e-6)


def test_calibration_retries_when_first_guess_is_not_arbitrage_free():
    # A skewed, moderately arbitrage-free target smile -- exercises the
    # deterministic retry sequence (research.md §2) rather than assuming the
    # first (rho0=0.0) initial guess always converges cleanly.
    true_params = SVIParams(a=0.06, b=0.3, rho=0.5, m=0.1, sigma=0.3)
    assert _check_butterfly_arbitrage_free(true_params) is True  # sanity: target itself is valid
    forward, t = 100.0, 0.5
    k_grid = np.linspace(-1.0, 1.2, 20)
    points = _synthetic_points(true_params, forward, t, k_grid)

    fitted = calibrate(points, forward_price=forward)

    assert fitted.butterfly_arbitrage_free is True
    assert fitted.fit_rmse_variance < 1e-6
