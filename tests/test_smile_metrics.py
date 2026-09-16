"""Smile-shape metric tests against hand-computable reference cases (tasks.md T024)."""
from __future__ import annotations

import numpy as np
import pytest

from breeden_litzenberger.core.black_scholes import ImpliedVolPoint
from breeden_litzenberger.core.smile import FittedSmile, SVIParams, check_butterfly_arbitrage
from breeden_litzenberger.core.smile_metrics import compute_smile_shape_metrics


def _make_smile(params: SVIParams, forward: float, t: float, k_range: float = 1.2, n: int = 41) -> FittedSmile:
    k_grid = np.linspace(-k_range, k_range, n)
    w = params.total_variance(k_grid)
    points = [
        ImpliedVolPoint(
            strike=float(forward * np.exp(k)),
            implied_vol=float(np.sqrt(wv / t)),
            log_forward_moneyness=float(k),
            total_variance=float(wv),
        )
        for k, wv in zip(k_grid, w)
    ]
    return FittedSmile(
        params=params,
        forward_price=forward,
        fit_rmse_variance=0.0,
        arbitrage_check=check_butterfly_arbitrage(params),
        retained_points=points,
        excluded_points=[],
    )


def test_atm_implied_vol_matches_hand_computed_w0():
    params = SVIParams(a=0.04, b=0.15, rho=-0.2, m=0.1, sigma=0.15)
    t = 1.0
    forward = 100.0
    smile = _make_smile(params, forward, t)

    metrics = compute_smile_shape_metrics(smile, spot=forward, t=t, r=0.03, q=0.03)

    w0 = float(params.total_variance(np.array([0.0]))[0])
    assert metrics.atm_implied_vol == pytest.approx((w0 / t) ** 0.5, rel=1e-9)


def test_symmetric_smile_risk_reversal_vanishes_in_low_vol_limit():
    # rho=0, m=0 gives a smile symmetric in k (w(k) == w(-k)). This does NOT
    # make risk_reversal_25d exactly zero, even with r=q (forward == spot):
    # Black-Scholes d1 = -k/(sigma*sqrt(T)) + 0.5*sigma*sqrt(T) has a drift
    # term that is itself asymmetric between the call and put sides, so
    # "symmetric in vol" and "symmetric in delta-space" are genuinely
    # different things (verified numerically during development: the
    # residual is ~0.018 at a 25% vol level and shrinks toward 0 as vol
    # shrinks, confirming it is this known BS-convexity effect and not a
    # bug in the delta-strike search). What IS true is that the residual
    # vanishes in the low-vol limit, which this test checks directly.
    forward, t = 100.0, 1.0
    low_vol_params = SVIParams(a=0.04 * 0.03 ** 2, b=0.15 * 0.03 ** 2, rho=0.0, m=0.0, sigma=0.15)
    smile = _make_smile(low_vol_params, forward, t)

    metrics = compute_smile_shape_metrics(smile, spot=forward, t=t, r=0.03, q=0.03)

    assert metrics.risk_reversal_25d == pytest.approx(0.0, abs=1e-5)
    assert metrics.butterfly_25d == pytest.approx(0.0, abs=1e-5)


def test_negative_skew_produces_negative_risk_reversal():
    # Equity-typical negative skew (rho<0): OTM puts trade at higher implied
    # vol than OTM calls, so risk_reversal_25d = call_vol - put_vol < 0.
    params = SVIParams(a=0.04, b=0.2, rho=-0.4, m=0.0, sigma=0.15)
    t = 1.0
    forward = 100.0
    smile = _make_smile(params, forward, t)

    metrics = compute_smile_shape_metrics(smile, spot=forward, t=t, r=0.03, q=0.03)

    assert metrics.risk_reversal_25d < 0.0


def test_butterfly_is_nonnegative_for_a_convex_smile():
    # A raw SVI smile is always convex in k (constitution/smile.py docstring
    # references Gatheral 2004's convexity note), so wing vols should sit
    # above the ATM vol -- butterfly_25d should be positive.
    params = SVIParams(a=0.04, b=0.2, rho=-0.1, m=0.0, sigma=0.15)
    t = 1.0
    forward = 100.0
    smile = _make_smile(params, forward, t)

    metrics = compute_smile_shape_metrics(smile, spot=forward, t=t, r=0.03, q=0.03)

    assert metrics.butterfly_25d > 0.0
