"""User Story 3 acceptance tests (tasks.md T028): smile-shape metrics and
plot-ready data from a completed extraction report.
"""
from __future__ import annotations

import numpy as np
import pytest
from matplotlib.figure import Figure

from breeden_litzenberger.plotting import plot_density, plot_smile
from breeden_litzenberger.report import Verdict, build_report
from tests.test_report_integration import _make_rates, _make_snapshot


def _completed_report():
    snapshot = _make_snapshot(spot=100.0, r=0.03, q=0.01, t_years=1.0, true_vol=0.25)
    rates = _make_rates(0.03, 0.01)
    return build_report(snapshot, rates)


def test_smile_shape_metrics_are_numeric_on_a_completed_report():
    report = _completed_report()
    assert report.verdict == Verdict.WELL_FIT_ARBITRAGE_FREE
    metrics = report.smile_shape_metrics
    assert metrics is not None
    assert isinstance(metrics.atm_implied_vol, float) and metrics.atm_implied_vol > 0
    assert isinstance(metrics.risk_reversal_25d, float)
    assert isinstance(metrics.butterfly_25d, float)


def test_plot_data_series_are_internally_consistent():
    report = _completed_report()
    data = report.plot_data()

    assert len(data.smile_strikes) == len(data.smile_market_ivs) == len(report.fitted_smile.retained_points)
    assert len(data.smile_fitted_strikes) == len(data.smile_fitted_ivs)
    assert len(data.density_strikes) == len(data.density_values) == len(data.lognormal_benchmark_density)
    assert np.array_equal(data.density_strikes, report.density_grid.strikes)
    assert np.array_equal(data.density_values, report.density_grid.density)
    # The fitted smile curve should span (at least) the retained market strikes.
    assert data.smile_fitted_strikes.min() <= data.smile_strikes.min()
    assert data.smile_fitted_strikes.max() >= data.smile_strikes.max()
    # The lognormal benchmark should itself integrate to ~1 (it's a real pdf).
    integral = np.trapezoid(data.lognormal_benchmark_density, data.density_strikes)
    assert integral == pytest.approx(1.0, abs=0.01)


def test_plotting_helpers_produce_figures():
    report = _completed_report()
    data = report.plot_data()

    smile_fig = plot_smile(data)
    density_fig = plot_density(data)

    assert isinstance(smile_fig, Figure)
    assert isinstance(density_fig, Figure)
