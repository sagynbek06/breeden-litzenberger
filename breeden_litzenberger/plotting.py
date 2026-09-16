"""Matplotlib convenience helpers over PlotData (spec Assumptions).

Rendering an actual chart is a caller concern, not part of the required
contract (report.ExtractionReport.plot_data() returns plain data) -- this
module is a thin, optional convenience layer on top of it.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from breeden_litzenberger.report import PlotData


def plot_smile(data: PlotData, title: str = "Fitted SVI smile vs. market implied vol") -> Figure:
    """Fitted SVI smile curve against the raw retained implied-vol points (spec FR-014c)."""
    fig, ax = plt.subplots()
    ax.plot(data.smile_fitted_strikes, data.smile_fitted_ivs, label="Fitted SVI smile")
    ax.scatter(data.smile_strikes, data.smile_market_ivs, s=20, color="black", label="Market implied vol", zorder=3)
    ax.set_xlabel("Strike")
    ax.set_ylabel("Implied volatility")
    ax.set_title(title)
    ax.legend()
    return fig


def plot_density(data: PlotData, title: str = "Extracted risk-neutral density vs. lognormal benchmark") -> Figure:
    """Extracted RND against a lognormal benchmark at the same forward/ATM vol (spec FR-014c)."""
    fig, ax = plt.subplots()
    ax.plot(data.density_strikes, data.density_values, label="Extracted RND")
    ax.plot(data.density_strikes, data.lognormal_benchmark_density, linestyle="--", label="Lognormal benchmark")
    ax.set_xlabel("Terminal price")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend()
    return fig
