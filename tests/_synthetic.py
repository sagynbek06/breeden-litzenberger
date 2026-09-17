"""Synthetic ground-truth option chain generator (tasks.md T020).

Generates OTM option prices from a KNOWN constant-volatility Black-Scholes
model -- a model whose true risk-neutral density is the closed-form
lognormal distribution -- so the full pipeline (implied-vol inversion ->
SVI fit -> Breeden-Litzenberger extraction) can be validated against a known
answer (constitution Principle III) before it is ever pointed at real
market data. This is test-support code, not part of the public library.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from breeden_litzenberger.core.black_scholes import OptionType, price


@dataclass(frozen=True)
class SyntheticQuote:
    strike: float
    option_type: OptionType
    market_price: float


@dataclass(frozen=True)
class SyntheticChain:
    spot: float
    r: float
    q: float
    t: float
    true_vol: float
    forward: float
    quotes: list[SyntheticQuote]


def generate_synthetic_chain(
    spot: float,
    r: float,
    q: float,
    t: float,
    true_vol: float,
    n_strikes: int = 41,
    std_devs: float = 6.0,
) -> SyntheticChain:
    """Price an OTM strike grid (puts below forward, calls above) under constant-vol BS.

    The log-moneyness range scales with the smile's own total-variance
    (std_devs * true_vol * sqrt(t)), not a fixed absolute range: a fixed
    range appropriate for a high-vol/long-tenor scenario would place a
    low-vol/short-tenor scenario's deepest strikes so far out of the money
    that their true Black-Scholes price underflows to exactly 0.0 in
    float64 -- a real quoting desk would never list such a strike either.
    """
    forward = spot * float(np.exp((r - q) * t))
    k_range = std_devs * true_vol * np.sqrt(t)
    k_grid = np.linspace(-k_range, k_range, n_strikes)
    strikes = forward * np.exp(k_grid)

    quotes = []
    for k, strike in zip(k_grid, strikes):
        option_type: OptionType = "put" if k < 0 else "call"
        p = price(spot, float(strike), t, r, q, true_vol, option_type)
        quotes.append(SyntheticQuote(strike=float(strike), option_type=option_type, market_price=p))

    return SyntheticChain(spot=spot, r=r, q=q, t=t, true_vol=true_vol, forward=forward, quotes=quotes)


def analytic_lognormal_moments(forward: float, true_vol: float, t: float) -> dict[str, float]:
    """Closed-form moments of the lognormal density this chain's prices imply."""
    v = true_vol * true_vol * t
    return {
        "mean": forward,
        "variance": (np.exp(v) - 1) * forward ** 2,
        "skewness": (np.exp(v) + 2) * np.sqrt(np.exp(v) - 1),
        "excess_kurtosis": np.exp(4 * v) + 2 * np.exp(3 * v) + 3 * np.exp(2 * v) - 6,
    }


def analytic_lognormal_density(strikes: np.ndarray, forward: float, true_vol: float, t: float) -> np.ndarray:
    """f(K) for the closed-form lognormal RND at the given strikes."""
    v = true_vol * true_vol * t
    mu = np.log(forward) - 0.5 * v
    return (1.0 / (strikes * true_vol * np.sqrt(2 * np.pi * t))) * np.exp(
        -((np.log(strikes) - mu) ** 2) / (2 * v)
    )
