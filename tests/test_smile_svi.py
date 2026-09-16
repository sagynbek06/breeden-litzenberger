"""SVI calibration convergence and Gatheral-Jacquier arbitrage checks (tasks.md T008).

Includes an independent re-derivation of g(k) (Gatheral & Jacquier 2014, eq.
2.1), kept only in this test file, cross-checked against smile.py's own
computation -- catching implementation bugs (e.g. a sign error) that a test
which only calls into smile.py's own g(k) could never detect.
"""
from __future__ import annotations

import numpy as np
import pytest

from breeden_litzenberger.core.black_scholes import ImpliedVolPoint
from breeden_litzenberger.core.smile import SVIParams, calibrate_svi, check_butterfly_arbitrage


def _independent_g(k: np.ndarray, p: SVIParams) -> np.ndarray:
    """Independent re-derivation of g(k) (Gatheral & Jacquier 2014, eq. 2.1)."""
    u = k - p.m
    s = np.sqrt(u * u + p.sigma * p.sigma)
    w = p.a + p.b * (p.rho * u + s)
    dw = p.b * (p.rho + u / s)
    d2w = p.b * p.sigma * p.sigma / (s ** 3)
    return (1.0 - k * dw / (2.0 * w)) ** 2 - (dw * dw / 4.0) * (1.0 / w + 0.25) + d2w / 2.0


@pytest.mark.parametrize(
    "a,b,rho,m,sigma",
    [
        (0.04, 0.1, 0.0, 0.0, 0.1),
        (0.01, 2.0, 0.9, 0.0, 0.01),
        (0.05, 0.4, -0.3, 0.0, 0.2),
        (0.02, 0.1, -0.8, -0.2, 0.05),  # the case that exposed the Theorem 4.2 mistranslation
        (0.06, 0.3, 0.5, 0.1, 0.3),
    ],
)
def test_g_matches_independent_reference(a, b, rho, m, sigma):
    params = SVIParams(a=a, b=b, rho=rho, m=m, sigma=sigma)
    k_grid = np.linspace(-2.0, 2.0, 500)
    from breeden_litzenberger.core.smile import _g  # internal, for cross-check only

    got = _g(k_grid, params)
    want = _independent_g(k_grid, params)
    np.testing.assert_allclose(got, want, rtol=1e-10, atol=1e-12)


def test_known_arbitrage_free_params():
    params = SVIParams(a=0.04, b=0.1, rho=0.0, m=0.0, sigma=0.1)
    check = check_butterfly_arbitrage(params)
    assert check.is_arbitrage_free is True
    assert check.margin > 0


def test_known_arbitrage_params():
    # b*(1+rho) = 2.0*(1+0.9) = 3.8 >> 2: the wing is so steep the density
    # goes negative -- g(k) is strongly negative on the grid.
    params = SVIParams(a=0.01, b=2.0, rho=0.9, m=0.0, sigma=0.01)
    check = check_butterfly_arbitrage(params)
    assert check.is_arbitrage_free is False
    assert check.margin < 0


def test_nonzero_m_with_negative_rho_is_correctly_flagged():
    """Regression test: this exact parameter set was where an earlier,
    mistranslated closed-form check (Gatheral & Jacquier 2014 Theorem 4.2,
    which assumes m is constrained by rho/sigma, not independently fitted)
    produced a false "arbitrage-free" reading. The true g(k) minimum here is
    approximately -0.00097 -- a genuine, non-noise violation, not floating-
    point error (see core/smile.py's _G_TOLERANCE, several orders of
    magnitude tighter). The exact Lemma 2.2 check must catch it.
    """
    params = SVIParams(a=0.02, b=0.1, rho=-0.8, m=-0.2, sigma=0.05)
    check = check_butterfly_arbitrage(params)
    assert check.is_arbitrage_free is False
    assert check.margin < -1e-5


def test_asymptotic_decay_condition_matches_true_limiting_behavior():
    # Empirically verified boundary: b*(1+rho) < 2 => d+(k) -> -inf (fine);
    # b*(1+rho) > 2 => d+(k) -> +inf (violates Lemma 2.2's boundary clause).
    fine = SVIParams(a=0.04, b=0.1, rho=0.0, m=0.0, sigma=0.1)  # b*(1+rho)=0.1
    bad = SVIParams(a=0.01, b=2.0, rho=0.9, m=0.0, sigma=0.01)  # b*(1+rho)=3.8
    assert check_butterfly_arbitrage(fine).asymptotic_decay_holds is True
    assert check_butterfly_arbitrage(bad).asymptotic_decay_holds is False


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

    fitted = calibrate_svi(points, forward_price=forward)

    assert fitted.butterfly_arbitrage_free is True
    assert fitted.fit_rmse_variance == pytest.approx(0.0, abs=1e-8)
    recovered_w = fitted.params.total_variance(k_grid)
    true_w = true_params.total_variance(k_grid)
    assert np.allclose(recovered_w, true_w, atol=1e-6)


def test_calibration_retries_when_first_guess_is_not_arbitrage_free():
    true_params = SVIParams(a=0.06, b=0.3, rho=0.5, m=0.1, sigma=0.3)
    assert check_butterfly_arbitrage(true_params).is_arbitrage_free is True  # sanity: target itself is valid
    forward, t = 100.0, 0.5
    k_grid = np.linspace(-1.0, 1.2, 20)
    points = _synthetic_points(true_params, forward, t, k_grid)

    fitted = calibrate_svi(points, forward_price=forward)

    assert fitted.butterfly_arbitrage_free is True
    assert fitted.fit_rmse_variance < 1e-6
