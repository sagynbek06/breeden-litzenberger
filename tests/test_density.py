"""THE ground-truth pipeline test (tasks.md T021; spec SC-001; constitution Principle III).

Runs the FULL pipeline -- implied-vol inversion -> SVI fit -> Breeden-
Litzenberger extraction -- on option prices synthesized from a KNOWN
constant-volatility Black-Scholes model, and asserts the extracted density
matches the closed-form analytic lognormal density to the tolerance fixed
in /speckit.clarify: 0.1% relative error on mean/variance, density curve
matching to 3 decimal places. This is this project's reference-paper
worked example -- the one test that actually proves the pipeline is
correct. Everything else is regression coverage on top of it.
"""
from __future__ import annotations

import numpy as np
import pytest

from breeden_litzenberger.core.black_scholes import implied_volatility
from breeden_litzenberger.core.density import extract_density
from breeden_litzenberger.core.moments import compute_moments
from breeden_litzenberger.core.smile import calibrate
from tests._synthetic import analytic_lognormal_density, analytic_lognormal_moments, generate_synthetic_chain

SCENARIOS = [
    # spot, r, q, t, true_vol -- spans low/high vol and short/long tenor
    (100.0, 0.03, 0.01, 1.0, 0.25),
    (100.0, 0.05, 0.00, 0.25, 0.15),
    (250.0, 0.02, 0.015, 2.0, 0.40),
]


def _run_full_pipeline(spot, r, q, t, true_vol):
    chain = generate_synthetic_chain(spot, r, q, t, true_vol)

    iv_points = [
        implied_volatility(quote.market_price, chain.spot, quote.strike, t, r, q, quote.option_type, chain.forward)
        for quote in chain.quotes
    ]

    fitted_smile = calibrate(iv_points, forward_price=chain.forward)
    density_grid, diagnostics = extract_density(fitted_smile, spot, t, r, q)
    moments = compute_moments(density_grid)

    return chain, fitted_smile, density_grid, diagnostics, moments


@pytest.mark.parametrize("spot,r,q,t,true_vol", SCENARIOS)
def test_ground_truth_lognormal_round_trip(spot, r, q, t, true_vol):
    chain, fitted_smile, density_grid, diagnostics, moments = _run_full_pipeline(spot, r, q, t, true_vol)
    analytic = analytic_lognormal_moments(chain.forward, true_vol, t)

    mean_rel_err = abs(moments.mean - analytic["mean"]) / analytic["mean"]
    variance_rel_err = abs(moments.variance - analytic["variance"]) / analytic["variance"]

    assert fitted_smile.butterfly_arbitrage_free
    assert mean_rel_err < 0.001, f"mean relative error {mean_rel_err:.6f} exceeds 0.1% tolerance (SC-001)"
    assert variance_rel_err < 0.001, f"variance relative error {variance_rel_err:.6f} exceeds 0.1% tolerance (SC-001)"

    analytic_density = analytic_lognormal_density(density_grid.strikes, chain.forward, true_vol, t)
    np.testing.assert_allclose(
        density_grid.density,
        analytic_density,
        atol=5e-4,  # "matches to 3 decimal places" (SC-001, fixed in /speckit.clarify)
        err_msg="extracted density does not match analytic lognormal density to 3 decimal places",
    )
