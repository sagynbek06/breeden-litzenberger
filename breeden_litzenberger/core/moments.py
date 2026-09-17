"""Moments of the extracted risk-neutral density (spec FR-011).

Standard statistical moments computed by numerical integration over a
DensityGrid: mean, variance, skewness, and excess kurtosis. The theoretical
forward price F = S0*exp((r-q)*T) is the risk-neutral pricing identity this
library uses as its single most important sanity check (spec SC-004) --
compared against the density's own realized mean in core/density.py.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import exp

import numpy as np

from breeden_litzenberger.core.density import DensityGrid


@dataclass(frozen=True)
class DensityMoments:
    """Mean/variance/skewness/excess kurtosis of an extracted density (data-model.md)."""

    mean: float
    variance: float
    skewness: float
    excess_kurtosis: float


def theoretical_forward_price(spot: float, t: float, r: float, q: float) -> float:
    """F = S0 * exp((r - q) * T)."""
    return spot * exp((r - q) * t)


def compute_moments(grid: DensityGrid) -> DensityMoments:
    """Mean/variance/skewness/excess kurtosis via trapezoidal integration over ``grid``."""
    strikes, density = grid.strikes, grid.density

    mean = float(np.trapezoid(strikes * density, strikes))
    central = strikes - mean
    variance = float(np.trapezoid(central ** 2 * density, strikes))
    std = np.sqrt(variance)
    skewness = float(np.trapezoid(central ** 3 * density, strikes)) / (std ** 3)
    kurtosis = float(np.trapezoid(central ** 4 * density, strikes)) / (variance ** 2)
    excess_kurtosis = kurtosis - 3.0

    return DensityMoments(mean=mean, variance=variance, skewness=skewness, excess_kurtosis=excess_kurtosis)
