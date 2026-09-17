"""Desk-convention smile-shape descriptors (spec FR-012).

ATM implied vol, 25-delta risk reversal, and 25-delta butterfly, computed
by locating the strike where Black-Scholes delta equals the target on the
fitted SVI smile (research.md §8), via a bounded root-find bracketed to
the smile's own calibrated domain -- the same treatment as every other
inversion in this pipeline (constitution Principle VII), and restricted to
the calibrated region rather than the wider density-extraction grid so a
25-delta point is never reported in a region the SVI fit was never
validated against.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import numpy as np
from scipy.optimize import brentq

from breeden_litzenberger.core.black_scholes import OptionType
from breeden_litzenberger.core.black_scholes import delta as bs_delta
from breeden_litzenberger.core.smile import FittedSmile

_TARGET_DELTA = 0.25


@dataclass(frozen=True)
class SmileShapeMetrics:
    """ATM implied vol, 25-delta risk reversal, 25-delta butterfly (data-model.md)."""

    atm_implied_vol: float
    risk_reversal_25d: float
    butterfly_25d: float


def _smile_implied_vol_at_strike(smile: FittedSmile, strike: float, t: float) -> float:
    k = np.log(strike / smile.forward_price)
    w = float(smile.params.total_variance(np.array([k]))[0])
    return sqrt(max(w, 1e-12) / t)


def _find_delta_strike(
    smile: FittedSmile,
    spot: float,
    t: float,
    r: float,
    q: float,
    target_delta: float,
    option_type: OptionType,
    lo: float,
    hi: float,
) -> float:
    """Solve for K such that BS delta(K, smile-implied vol at K) == target_delta,
    bracketed to [lo, hi] within the fitted smile's own calibrated domain.
    """

    def _objective(strike: float) -> float:
        sigma = _smile_implied_vol_at_strike(smile, strike, t)
        return bs_delta(spot, strike, t, r, q, sigma, option_type) - target_delta

    lo_val, hi_val = _objective(lo), _objective(hi)
    if lo_val * hi_val > 0:
        # Target delta not bracketed within the calibrated domain (a very
        # flat or very skewed smile) -- fall back to whichever boundary
        # strike's delta is closest to the target, rather than raising.
        return lo if abs(lo_val) < abs(hi_val) else hi

    return brentq(_objective, lo, hi, xtol=1e-8)


def compute_smile_shape_metrics(smile: FittedSmile, spot: float, t: float, r: float, q: float) -> SmileShapeMetrics:
    """ATM vol, 25-delta risk reversal (call vol - put vol), and 25-delta
    butterfly ((call vol + put vol)/2 - ATM vol) from a fitted smile (FR-012).
    """
    atm_vol = _smile_implied_vol_at_strike(smile, smile.forward_price, t)

    retained_strikes = [p.strike for p in smile.retained_points]
    min_strike, max_strike, forward = min(retained_strikes), max(retained_strikes), smile.forward_price

    call_strike = _find_delta_strike(smile, spot, t, r, q, _TARGET_DELTA, "call", forward, max_strike)
    put_strike = _find_delta_strike(smile, spot, t, r, q, -_TARGET_DELTA, "put", min_strike, forward)

    call_vol = _smile_implied_vol_at_strike(smile, call_strike, t)
    put_vol = _smile_implied_vol_at_strike(smile, put_strike, t)

    return SmileShapeMetrics(
        atm_implied_vol=atm_vol,
        risk_reversal_25d=call_vol - put_vol,
        butterfly_25d=(call_vol + put_vol) / 2.0 - atm_vol,
    )
