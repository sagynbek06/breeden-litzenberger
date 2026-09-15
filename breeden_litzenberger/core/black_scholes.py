"""Black-Scholes(-Merton) option pricing and implied-volatility inversion.

Formula source: Black, F. & Scholes, M. (1973), "The Pricing of Options and
Corporate Liabilities", Journal of Political Economy 81(3), 637-654. The
continuous dividend-yield term ``q`` in ``d1``/``d2`` below is the standard
Merton (1973) extension for a continuously-compounded payout.

Why this module exists and why inversion is bounded, not hand-rolled: implied
volatility has no closed-form inverse, and price(sigma) is monotonic but flat
in its tails (vega -> 0 far from the money), which makes unconstrained
Newton-Raphson root-finding prone to overshoot or divergence on exactly the
kind of noisy, wide-strike-range market quotes this library ingests. Brent's
method (``scipy.optimize.brentq``) combines bisection's guaranteed
convergence within a bracket with superlinear convergence speed, and never
leaves the economically sane volatility bracket -- consistent with
constitution Principle VII (numerical stability over cleverness).
"""
from __future__ import annotations

from dataclasses import dataclass
from math import erf, exp, log, pi, sqrt
from typing import Literal

from scipy.optimize import brentq

OptionType = Literal["call", "put"]

# Sane annualized-volatility bracket for implied-vol inversion (spec FR-007,
# research.md task T005): 1% to 500%.
MIN_VOL = 0.01
MAX_VOL = 5.00


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def _norm_pdf(x: float) -> float:
    return exp(-0.5 * x * x) / sqrt(2.0 * pi)


def _d1_d2(spot: float, strike: float, t: float, r: float, q: float, sigma: float) -> tuple[float, float]:
    if t <= 0:
        raise ValueError(f"time-to-expiration must be positive, got t={t}")
    if sigma <= 0:
        raise ValueError(f"volatility must be positive, got sigma={sigma}")
    sqrt_t = sqrt(t)
    d1 = (log(spot / strike) + (r - q + 0.5 * sigma * sigma) * t) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t
    return d1, d2


def price(spot: float, strike: float, t: float, r: float, q: float, sigma: float, option_type: OptionType) -> float:
    """Black-Scholes(-Merton) price of a European option (Black & Scholes 1973)."""
    d1, d2 = _d1_d2(spot, strike, t, r, q, sigma)
    if option_type == "call":
        return spot * exp(-q * t) * _norm_cdf(d1) - strike * exp(-r * t) * _norm_cdf(d2)
    return strike * exp(-r * t) * _norm_cdf(-d2) - spot * exp(-q * t) * _norm_cdf(-d1)


def vega(spot: float, strike: float, t: float, r: float, q: float, sigma: float) -> float:
    """Sensitivity of price to volatility; the same formula for calls and puts."""
    d1, _ = _d1_d2(spot, strike, t, r, q, sigma)
    return spot * exp(-q * t) * _norm_pdf(d1) * sqrt(t)


def delta(spot: float, strike: float, t: float, r: float, q: float, sigma: float, option_type: OptionType) -> float:
    """Black-Scholes delta; used by core/smile_metrics.py's 25-delta strike search (research.md §8)."""
    d1, _ = _d1_d2(spot, strike, t, r, q, sigma)
    if option_type == "call":
        return exp(-q * t) * _norm_cdf(d1)
    return -exp(-q * t) * _norm_cdf(-d1)


def _intrinsic_value(spot: float, strike: float, option_type: OptionType) -> float:
    if option_type == "call":
        return max(spot - strike, 0.0)
    return max(strike - spot, 0.0)


@dataclass(frozen=True)
class ImpliedVolPoint:
    """A clean OTM point after Black-Scholes inversion (data-model.md)."""

    strike: float
    implied_vol: float
    log_forward_moneyness: float
    total_variance: float


class ImpliedVolError(ValueError):
    """A quote could not be inverted to an implied volatility (spec FR-007).

    ``reason`` matches one of data-model.md's ``ExcludedQuote.reason`` values:
    ``"below_intrinsic_value"``, ``"zero_price"``, or ``"iv_bracket_failure"``.
    Raised, never silently coerced to a placeholder volatility.
    """

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason


def implied_volatility(
    market_price: float,
    spot: float,
    strike: float,
    t: float,
    r: float,
    q: float,
    option_type: OptionType,
    forward_price: float,
) -> ImpliedVolPoint:
    """Invert the Black-Scholes formula for volatility via Brent's method.

    Bounded root-find over ``[MIN_VOL, MAX_VOL]`` (spec FR-007; constitution
    Principle VII). Degenerate cases raise ``ImpliedVolError`` rather than
    returning a placeholder.
    """
    if market_price <= 0:
        raise ImpliedVolError("zero_price", f"non-positive market price {market_price!r} at strike {strike}")

    intrinsic = _intrinsic_value(spot, strike, option_type)
    if market_price < intrinsic:
        raise ImpliedVolError(
            "below_intrinsic_value",
            f"price {market_price} is below intrinsic value {intrinsic} for {option_type} strike {strike}",
        )

    def _objective(sigma: float) -> float:
        return price(spot, strike, t, r, q, sigma, option_type) - market_price

    lo, hi = _objective(MIN_VOL), _objective(MAX_VOL)
    if lo * hi > 0:
        raise ImpliedVolError(
            "iv_bracket_failure",
            f"no volatility in [{MIN_VOL:.0%}, {MAX_VOL:.0%}] reproduces price "
            f"{market_price} at strike {strike} (bracket values: {lo!r}, {hi!r})",
        )

    sigma = brentq(_objective, MIN_VOL, MAX_VOL, xtol=1e-10, rtol=1e-12)

    k = log(strike / forward_price)
    w = sigma * sigma * t
    return ImpliedVolPoint(strike=strike, implied_vol=sigma, log_forward_moneyness=k, total_variance=w)
