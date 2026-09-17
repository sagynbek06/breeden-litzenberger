"""Property-based tests (tasks.md T022) exercising the invariants a correct
Breeden-Litzenberger pipeline must satisfy, across randomly generated inputs
rather than the fixed scenarios in test_density.py.
"""
from __future__ import annotations

import numpy as np
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from breeden_litzenberger.core.black_scholes import implied_volatility
from breeden_litzenberger.core.density import extract_density
from breeden_litzenberger.core.moments import compute_moments
from breeden_litzenberger.core.smile import FittedSmile, SVIParams, calibrate_svi, check_butterfly_arbitrage
from tests._synthetic import analytic_lognormal_density, generate_synthetic_chain

_SLOW_SETTINGS = settings(max_examples=20, deadline=None, suppress_health_check=[HealthCheck.filter_too_much])

# Conservative SVI parameter ranges chosen so most draws are arbitrage-free
# by construction (per the Theorem 4.2 intuition: keep b*(1+|rho|) well
# under 2); remaining non-conforming draws are filtered via `assume`.
_svi_params_strategy = st.builds(
    SVIParams,
    a=st.floats(min_value=0.02, max_value=0.15, allow_nan=False),
    b=st.floats(min_value=0.02, max_value=0.4, allow_nan=False),
    rho=st.floats(min_value=-0.85, max_value=0.85, allow_nan=False),
    m=st.floats(min_value=-0.2, max_value=0.2, allow_nan=False),
    sigma=st.floats(min_value=0.08, max_value=0.4, allow_nan=False),
)


@given(params=_svi_params_strategy, forward=st.floats(min_value=50.0, max_value=300.0), t=st.floats(min_value=0.1, max_value=2.0))
@_SLOW_SETTINGS
def test_arbitrage_free_smile_always_yields_non_negative_density(params, forward, t):
    """(a) The mathematical premise of Breeden-Litzenberger itself: a smile
    that produces a convex, arbitrage-free call-price curve must yield a
    non-negative density. We generate arbitrage-free SVI parameter sets
    (via check_butterfly_arbitrage, the exact Lemma 2.2 condition) and
    verify extract_density's non-negativity diagnostic holds.
    """
    check = check_butterfly_arbitrage(params)
    assume(check.is_arbitrage_free)
    theta = float(params.total_variance(np.array([0.0]))[0])
    assume(theta > 1e-6)

    smile = FittedSmile(
        params=params, forward_price=forward, fit_rmse_variance=0.0,
        arbitrage_check=check, retained_points=[], excluded_points=[],
    )
    # r=q=0 so spot == forward exactly, matching smile.forward_price
    # (extract_density requires this consistency -- see InconsistentForwardError,
    # added after this test class of mistake was actually caught during
    # development of test (b) below).
    _, diagnostics = extract_density(smile, spot=forward, t=t, r=0.0, q=0.0)

    assert diagnostics.is_non_negative


@given(
    a=st.floats(min_value=0.01, max_value=0.3, allow_nan=False),
    forward=st.floats(min_value=50.0, max_value=300.0),
    t=st.floats(min_value=0.1, max_value=2.0),
)
@_SLOW_SETTINGS
def test_flat_smile_reduces_to_analytic_lognormal(a, forward, t):
    """(b) b=0 (flat total variance, zero skew) collapses raw SVI to a
    constant-vol Black-Scholes smile -- the extracted density must match
    the closed-form lognormal exactly (this is the ground-truth scenario
    with zero smile-fitting noise, since we build the FittedSmile directly
    rather than calibrating it).
    """
    params = SVIParams(a=a, b=0.0, rho=0.0, m=0.0, sigma=0.1)  # b=0 -> rho, m, sigma are inert
    check = check_butterfly_arbitrage(params)
    smile = FittedSmile(
        params=params, forward_price=forward, fit_rmse_variance=0.0,
        arbitrage_check=check, retained_points=[], excluded_points=[],
    )
    # r=q=0 so spot == forward exactly, matching smile.forward_price (see
    # InconsistentForwardError in core/density.py -- found via this exact test:
    # a nonzero (r-q) here made spot*exp((r-q)*T) diverge from forward_price,
    # producing a real ~9% density error at some strikes that had nothing to
    # do with the extraction math itself).
    grid, diagnostics = extract_density(smile, spot=forward, t=t, r=0.0, q=0.0)

    true_vol = (a / t) ** 0.5
    analytic = analytic_lognormal_density(grid.strikes, forward, true_vol, t)

    np.testing.assert_allclose(grid.density, analytic, atol=1e-3)
    assert diagnostics.is_non_negative


@given(
    a_low=st.floats(min_value=0.02, max_value=0.08, allow_nan=False),
    a_delta=st.floats(min_value=0.01, max_value=0.1, allow_nan=False),
    forward=st.floats(min_value=50.0, max_value=300.0),
    t=st.floats(min_value=0.2, max_value=1.5),
)
@_SLOW_SETTINGS
def test_variance_increases_monotonically_with_atm_total_variance(a_low, a_delta, forward, t):
    """(c) Holding the smile's shape fixed (b, rho, m, sigma) and increasing
    only `a` (which increases the ATM total variance one-for-one) must
    increase the extracted density's variance.
    """
    b, rho, m, sigma = 0.15, -0.1, 0.0, 0.15
    a_high = a_low + a_delta

    def variance_for(a: float) -> float:
        params = SVIParams(a=a, b=b, rho=rho, m=m, sigma=sigma)
        check = check_butterfly_arbitrage(params)
        assume(check.is_arbitrage_free)
        smile = FittedSmile(
            params=params, forward_price=forward, fit_rmse_variance=0.0,
            arbitrage_check=check, retained_points=[], excluded_points=[],
        )
        grid, _ = extract_density(smile, spot=forward, t=t, r=0.0, q=0.0)  # r=q=0: spot == forward
        return compute_moments(grid).variance

    variance_low = variance_for(a_low)
    variance_high = variance_for(a_high)

    assert variance_high > variance_low


@given(
    spot=st.floats(min_value=50.0, max_value=300.0),
    r=st.floats(min_value=0.0, max_value=0.06),
    q=st.floats(min_value=0.0, max_value=0.03),
    t=st.floats(min_value=0.2, max_value=1.5),
    true_vol=st.floats(min_value=0.1, max_value=0.5),
)
@_SLOW_SETTINGS
def test_forward_price_sanity_check_passes_for_synthetic_ground_truth(spot, r, q, t, true_vol):
    """(d) With no market noise, the full pipeline's forward-price sanity
    check (realized density mean vs. theoretical forward) must pass to a
    tight tolerance across a wide range of market parameters, not just the
    3 fixed scenarios in test_density.py.
    """
    chain = generate_synthetic_chain(spot, r, q, t, true_vol)
    iv_points = [
        implied_volatility(quote.market_price, chain.spot, quote.strike, t, r, q, quote.option_type, chain.forward)
        for quote in chain.quotes
    ]
    fitted_smile = calibrate_svi(iv_points, forward_price=chain.forward)
    _, diagnostics = extract_density(fitted_smile, spot, t, r, q)

    assert diagnostics.forward_price_deviation < 0.01
