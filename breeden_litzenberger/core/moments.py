"""Moments of the extracted risk-neutral density (spec FR-011).

Standard statistical moments computed by numerical integration over a
DensityGrid: mean, variance, skewness, and excess kurtosis. The theoretical
forward price F = S0*exp((r-q)*T) is the risk-neutral pricing identity this
library uses as its single most important sanity check (spec SC-004) --
forward_price_check below compares it against the density's own mean.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import exp

import numpy as np

from breeden_litzenberger.core.density import FORWARD_PRICE_RELATIVE_ERROR_THRESHOLD, DensityGrid


@dataclass(frozen=True)
class DensityMoments:
    """Mean/variance/skewness/excess kurtosis of an extracted density (data-model.md)."""

    mean: float
    variance: float
    skewness: float
    excess_kurtosis: float


@dataclass(frozen=True)
class ForwardPriceCheck:
    """Result of comparing the extracted density's mean to the theoretical forward.

    ``passed`` is ``relative_error <= threshold``; a NaN relative error
    (e.g. from a NaN density) therefore fails, never passes by accident.
    """

    theoretical_forward: float
    realized_mean: float
    absolute_error: float
    relative_error: float
    threshold: float
    passed: bool


def theoretical_forward_price(spot: float, t: float, r: float, q: float) -> float:
    """F = S0 * exp((r - q) * T)."""
    return spot * exp((r - q) * t)


def forward_price_check(
    rnd_grid: np.ndarray,
    strikes: np.ndarray,
    S0: float,
    r: float,
    q: float,
    T: float,
    threshold: float = FORWARD_PRICE_RELATIVE_ERROR_THRESHOLD,
) -> ForwardPriceCheck:
    """Compare the risk-neutral mean of an extracted density to the theoretical forward.

    Under risk-neutral pricing the discounted underlying is a martingale, so
    E_Q[S_T] must equal F = S0 * exp((r - q) * T). That identity is
    independent of everything used to fit the smile, which makes the
    density's own mean -- the trapezoidal integral of K * f(K) over
    ``strikes`` -- the most important end-to-end sanity check in the
    pipeline: a wrong density, a bad rate/dividend input, or a truncated
    grid all show up here.

    ``rnd_grid`` holds the density values f(K) at each entry of ``strikes``.
    The mean is integrated from ``rnd_grid`` exactly as given, without
    renormalizing, so a density that does not integrate to 1 shows up as a
    forward-price error as well (it is reported separately in
    ``DensityDiagnostics.normalization_error``).

    Returns both the absolute error ``|mean - F|`` and the relative error
    ``|mean - F| / F``; ``passed`` compares the relative error against
    ``threshold`` (default: the 1% fixed in /speckit.clarify, SC-004).
    """
    rnd_grid = np.asarray(rnd_grid, dtype=float)
    strikes = np.asarray(strikes, dtype=float)
    if rnd_grid.ndim != 1 or rnd_grid.shape != strikes.shape:
        raise ValueError(
            f"rnd_grid and strikes must be 1-D arrays of the same length, "
            f"got shapes {rnd_grid.shape} and {strikes.shape}"
        )
    if strikes.size < 2:
        raise ValueError("need at least 2 grid points to integrate")
    if T <= 0:
        raise ValueError(f"T must be positive, got {T!r}")

    theoretical = theoretical_forward_price(S0, T, r, q)
    if not theoretical > 0:
        raise ValueError(f"theoretical forward must be positive, got {theoretical!r} for S0={S0!r}")

    realized = float(np.trapezoid(strikes * rnd_grid, strikes))
    absolute_error = abs(realized - theoretical)
    relative_error = absolute_error / theoretical

    return ForwardPriceCheck(
        theoretical_forward=theoretical,
        realized_mean=realized,
        absolute_error=absolute_error,
        relative_error=relative_error,
        threshold=threshold,
        passed=bool(relative_error <= threshold),
    )


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
