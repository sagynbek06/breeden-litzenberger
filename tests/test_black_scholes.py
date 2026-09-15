"""Reference-value and round-trip tests for core/black_scholes.py (tasks.md T006).

Two independent reference checks, deliberately not just "call the function and
trust it": (1) put-call parity, a closed-form identity the pricer must satisfy
exactly regardless of its own internals; (2) an independently re-derived
Black-Scholes formula using scipy.stats.norm (a different code path from the
module's own math.erf-based normal CDF), so a bug shared between the two
would have to be a bug in the *shared* Black-Scholes formula itself, not in
one particular implementation of the normal CDF.
"""
from __future__ import annotations

import math

import pytest
from scipy.stats import norm

from breeden_litzenberger.core import black_scholes as bs

CASES = [
    # spot, strike, t, r, q, sigma
    (100.0, 100.0, 1.0, 0.05, 0.00, 0.20),
    (100.0, 110.0, 0.5, 0.03, 0.01, 0.35),
    (50.0, 40.0, 2.0, 0.01, 0.00, 0.15),
    (250.0, 260.0, 0.1, 0.045, 0.02, 0.60),
    (100.0, 100.0, 1.0, 0.00, 0.00, 0.20),
]


def _reference_price(spot, strike, t, r, q, sigma, option_type):
    """Independent re-derivation of Black-Scholes-Merton, via scipy.stats.norm."""
    d1 = (math.log(spot / strike) + (r - q + 0.5 * sigma * sigma) * t) / (sigma * math.sqrt(t))
    d2 = d1 - sigma * math.sqrt(t)
    if option_type == "call":
        return spot * math.exp(-q * t) * norm.cdf(d1) - strike * math.exp(-r * t) * norm.cdf(d2)
    return strike * math.exp(-r * t) * norm.cdf(-d2) - spot * math.exp(-q * t) * norm.cdf(-d1)


@pytest.mark.parametrize("spot,strike,t,r,q,sigma", CASES)
@pytest.mark.parametrize("option_type", ["call", "put"])
def test_price_matches_independent_reference(spot, strike, t, r, q, sigma, option_type):
    got = bs.price(spot, strike, t, r, q, sigma, option_type)
    want = _reference_price(spot, strike, t, r, q, sigma, option_type)
    assert got == pytest.approx(want, rel=1e-12, abs=1e-12)


@pytest.mark.parametrize("spot,strike,t,r,q,sigma", CASES)
def test_put_call_parity(spot, strike, t, r, q, sigma):
    call = bs.price(spot, strike, t, r, q, sigma, "call")
    put = bs.price(spot, strike, t, r, q, sigma, "put")
    parity_rhs = spot * math.exp(-q * t) - strike * math.exp(-r * t)
    assert (call - put) == pytest.approx(parity_rhs, rel=1e-12, abs=1e-10)


def test_atm_call_equals_put_when_r_and_q_are_zero():
    # Classic reference relation: at S=K with r=q=0, C-P = S-K = 0, so C == P exactly.
    call = bs.price(100.0, 100.0, 1.0, 0.0, 0.0, 0.2, "call")
    put = bs.price(100.0, 100.0, 1.0, 0.0, 0.0, 0.2, "put")
    assert call == pytest.approx(put, rel=1e-12)


@pytest.mark.parametrize("spot,strike,t,r,q,true_sigma", CASES)
@pytest.mark.parametrize("option_type", ["call", "put"])
def test_implied_vol_round_trip(spot, strike, t, r, q, true_sigma, option_type):
    """price -> implied_vol -> price round trip, rtol=1e-6 (tasks.md T006)."""
    market_price = bs.price(spot, strike, t, r, q, true_sigma, option_type)
    forward = spot * math.exp((r - q) * t)
    point = bs.implied_volatility(market_price, spot, strike, t, r, q, option_type, forward)
    assert point.implied_vol == pytest.approx(true_sigma, rel=1e-6)
    round_trip_price = bs.price(spot, strike, t, r, q, point.implied_vol, option_type)
    assert round_trip_price == pytest.approx(market_price, rel=1e-6)


def test_below_intrinsic_value_raises():
    with pytest.raises(bs.ImpliedVolError) as exc_info:
        bs.implied_volatility(0.5, 100.0, 80.0, 1.0, 0.0, 0.0, "call", 100.0)
    assert exc_info.value.reason == "below_intrinsic_value"


def test_zero_price_raises():
    with pytest.raises(bs.ImpliedVolError) as exc_info:
        bs.implied_volatility(0.0, 100.0, 100.0, 1.0, 0.0, 0.0, "call", 100.0)
    assert exc_info.value.reason == "zero_price"


def test_iv_bracket_failure_raises():
    # A price no volatility in [1%, 500%] can reproduce: above the max-vol price.
    absurd_price = bs.price(100.0, 100.0, 1.0, 0.0, 0.0, bs.MAX_VOL, "call") + 50.0
    with pytest.raises(bs.ImpliedVolError) as exc_info:
        bs.implied_volatility(absurd_price, 100.0, 100.0, 1.0, 0.0, 0.0, "call", 100.0)
    assert exc_info.value.reason == "iv_bracket_failure"
