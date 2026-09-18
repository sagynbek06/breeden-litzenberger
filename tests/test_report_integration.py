"""Deterministic report-orchestration integration tests (tasks.md T019).

Uses ``build_report()`` -- the pure, no-network half of ``extract()`` -- with
a fixture ``OptionChainSnapshot``, so User Story 1's orchestration logic
(liquidity cleaning, IV inversion, SVI fit, density extraction, verdict
determination) is tested end-to-end without depending on live yfinance data
or network availability in CI. OTM selection itself now happens earlier, in
data/yfinance_loader.py's fetch_otm_chain, so these fixtures are built
already OTM-consistent by construction (put strikes below the fixture's own
forward, call strikes above it) rather than exercising an OTM filter here.
"""
from __future__ import annotations

import dataclasses
from datetime import datetime, timedelta

import numpy as np
import pytest

from breeden_litzenberger import report as report_module
from breeden_litzenberger.core.black_scholes import price
from breeden_litzenberger.core.smile import ButterflyArbitrageCheck
from breeden_litzenberger.data.rates import RateInputs
from breeden_litzenberger.data.yfinance_loader import OptionChainSnapshot, OptionQuote
from breeden_litzenberger.report import LiquidityFloor, Verdict, build_report


def _make_snapshot(spot: float, r: float, q: float, t_years: float, true_vol: float, n_strikes: int = 25) -> OptionChainSnapshot:
    """A liquid, internally-consistent fixture chain: real Black-Scholes
    prices at a constant vol, so this exercises the full pipeline the same
    way tests/test_density.py's ground truth does, but through report.py's
    orchestration layer instead of calling core/ directly.
    """
    fetched_at = datetime.now()
    expiration = fetched_at.date() + timedelta(days=round(t_years * 365.25))
    forward = spot * float(np.exp((r - q) * t_years))

    k_grid = np.linspace(-6.0 * true_vol * (t_years ** 0.5), 6.0 * true_vol * (t_years ** 0.5), n_strikes)
    quotes = []
    for k in k_grid:
        strike = forward * float(np.exp(k))
        option_type = "put" if k < 0 else "call"
        fair_price = price(spot, strike, t_years, r, q, true_vol, option_type)
        quotes.append(
            OptionQuote(
                strike=strike,
                option_type=option_type,
                bid=fair_price * 0.99,
                ask=fair_price * 1.01,
                last_price=fair_price,
                open_interest=100,
                volume=50,
            )
        )
    return OptionChainSnapshot(ticker="TEST", expiration=expiration, fetched_at=fetched_at, spot_price=spot, quotes=quotes)


def _make_rates(r: float, q: float) -> RateInputs:
    return RateInputs(
        risk_free_rate=r,
        risk_free_rate_source="override",
        dividend_yield=q,
        dividend_yield_source="override",
    )


def test_liquid_fixture_produces_a_complete_well_fit_report():
    snapshot = _make_snapshot(spot=100.0, r=0.03, q=0.01, t_years=1.0, true_vol=0.25)
    rates = _make_rates(0.03, 0.01)

    report = build_report(snapshot, rates)

    assert report.verdict == Verdict.WELL_FIT_ARBITRAGE_FREE
    assert report.fitted_smile is not None
    assert report.fitted_smile.butterfly_arbitrage_free
    assert report.density_grid is not None
    assert report.diagnostics is not None
    assert not report.diagnostics.flagged
    assert report.moments is not None
    # Sanity: extracted mean should be close to the theoretical forward price.
    assert report.moments.mean == pytest.approx(report.diagnostics.forward_price_theoretical, rel=1e-3)


def test_identical_inputs_produce_identical_reports():
    snapshot = _make_snapshot(spot=100.0, r=0.03, q=0.01, t_years=1.0, true_vol=0.25)
    rates = _make_rates(0.03, 0.01)

    report_a = build_report(snapshot, rates)
    report_b = build_report(snapshot, rates)

    assert report_a.to_dict() == report_b.to_dict()


def test_too_few_liquid_points_yields_insufficient_liquid_data_verdict():
    fetched_at = datetime.now()
    expiration = fetched_at.date() + timedelta(days=30)
    # Only 2 quotes survive cleaning -- below the 5-parameter SVI fit minimum.
    quotes = [
        OptionQuote(strike=90.0, option_type="put", bid=1.0, ask=1.2, last_price=1.1, open_interest=10, volume=5),
        OptionQuote(strike=110.0, option_type="call", bid=0.8, ask=1.0, last_price=0.9, open_interest=10, volume=5),
    ]
    snapshot = OptionChainSnapshot(ticker="ILLIQUID", expiration=expiration, fetched_at=fetched_at, spot_price=100.0, quotes=quotes)
    rates = _make_rates(0.03, 0.0)

    report = build_report(snapshot, rates)

    assert report.verdict == Verdict.INSUFFICIENT_LIQUID_DATA
    assert report.fitted_smile is None
    assert report.density_grid is None
    assert "liquid OTM quotes" in report.verdict_message


def test_nan_bid_or_ask_is_excluded_not_propagated():
    """Regression test: a real SPY chain fetch during development produced a
    quote with bid=NaN (an illiquid strike with no posted bid). Since
    `math.nan <= 0` and `math.nan > x` are both False in Python, a naive
    `bid <= 0` check silently let it through to become a NaN mid price,
    which then crashed scipy.optimize.brentq several steps downstream with
    "function value is NaN". Cleaning must catch this explicitly.
    """
    snapshot = _make_snapshot(spot=100.0, r=0.03, q=0.0, t_years=0.5, true_vol=0.2)
    quotes = list(snapshot.quotes)
    bad = quotes[0]
    quotes[0] = OptionQuote(bad.strike, bad.option_type, float("nan"), bad.ask, bad.last_price, bad.open_interest, bad.volume)
    snapshot = OptionChainSnapshot(
        ticker=snapshot.ticker, expiration=snapshot.expiration, fetched_at=snapshot.fetched_at,
        spot_price=snapshot.spot_price, quotes=quotes,
    )
    rates = _make_rates(0.03, 0.0)

    report = build_report(snapshot, rates)  # must not raise

    assert report.fitted_smile is not None
    excluded_reasons = {(e.strike, e.reason) for e in report.fitted_smile.excluded_points}
    assert (bad.strike, "zero_bid") in excluded_reasons


_FORWARD_FAILURE_PREFIX = "forward-price check failed: extracted RND mean deviates from theoretical forward by "


def _patch_forward_check_to_see_a_biased_density(monkeypatch, factor: float) -> None:
    """Make report.py's forward_price_check see a density whose mean is off by
    ``factor`` (the density is scaled, so its K*f(K) integral scales too),
    while extract_density's own diagnostics stay untouched. This is a genuine
    numerical failure of the check on real pipeline output, not a stubbed
    result -- it just simulates the failure mode the check exists to catch.
    """
    real = report_module.forward_price_check

    def biased(rnd_grid, strikes, *rest):
        return real(rnd_grid * factor, strikes, *rest)

    monkeypatch.setattr(report_module, "forward_price_check", biased)


def _patch_calibration_to_report_arbitrage(monkeypatch) -> None:
    real = report_module.calibrate_svi

    def with_arbitrage(points, forward_price):
        fitted = real(points, forward_price)
        violated = ButterflyArbitrageCheck(
            is_arbitrage_free=False, margin=-0.1, margin_at_k=0.3, asymptotic_decay_holds=True
        )
        return dataclasses.replace(fitted, arbitrage_check=violated)

    monkeypatch.setattr(report_module, "calibrate_svi", with_arbitrage)


def test_forward_price_check_is_threaded_through_and_reported_as_passed():
    snapshot = _make_snapshot(spot=100.0, r=0.03, q=0.01, t_years=1.0, true_vol=0.25)

    report = build_report(snapshot, _make_rates(0.03, 0.01))

    check = report.forward_price_check
    assert check is not None and check.passed
    assert check.realized_mean == report.diagnostics.forward_price_realized
    assert check.relative_error == pytest.approx(check.absolute_error / check.theoretical_forward, rel=1e-12)
    assert "forward-price check passed" in report.verdict_message
    assert "failed" not in report.verdict_message
    assert "Forward-price check: PASSED" in report.summary()
    assert report.to_dict()["forward_price_check"]["passed"] is True


def test_failed_forward_price_check_is_stated_explicitly_in_the_verdict(monkeypatch):
    snapshot = _make_snapshot(spot=100.0, r=0.03, q=0.01, t_years=1.0, true_vol=0.25)
    _patch_forward_check_to_see_a_biased_density(monkeypatch, factor=1.05)

    report = build_report(snapshot, _make_rates(0.03, 0.01))

    # The rest of the pipeline is perfectly healthy -- the density diagnostics
    # and the arbitrage check don't flag anything -- yet the verdict must
    # still say the forward-price check failed.
    assert report.fitted_smile.butterfly_arbitrage_free
    assert not report.diagnostics.flagged
    assert report.verdict == Verdict.DIAGNOSTIC_WARNING
    assert _FORWARD_FAILURE_PREFIX + "5.00%" in report.verdict_message
    assert report.forward_price_check is not None and not report.forward_price_check.passed
    assert "Forward-price check: FAILED" in report.summary()
    assert _FORWARD_FAILURE_PREFIX in report.summary()  # the Verdict line carries it, not just the detail line
    assert report.to_dict()["forward_price_check"]["passed"] is False
    assert _FORWARD_FAILURE_PREFIX in report.to_dict()["verdict_message"]


def test_forward_price_failure_is_not_swallowed_by_an_arbitrage_verdict(monkeypatch):
    snapshot = _make_snapshot(spot=100.0, r=0.03, q=0.01, t_years=1.0, true_vol=0.25)
    _patch_calibration_to_report_arbitrage(monkeypatch)
    _patch_forward_check_to_see_a_biased_density(monkeypatch, factor=1.05)

    report = build_report(snapshot, _make_rates(0.03, 0.01))

    assert report.verdict == Verdict.ARBITRAGE_DETECTED
    assert "Gatheral-Jacquier" in report.verdict_message
    assert _FORWARD_FAILURE_PREFIX + "5.00%" in report.verdict_message


def test_arbitrage_verdict_does_not_claim_a_forward_failure_that_did_not_happen(monkeypatch):
    snapshot = _make_snapshot(spot=100.0, r=0.03, q=0.01, t_years=1.0, true_vol=0.25)
    _patch_calibration_to_report_arbitrage(monkeypatch)

    report = build_report(snapshot, _make_rates(0.03, 0.01))

    assert report.verdict == Verdict.ARBITRAGE_DETECTED
    assert "forward-price check" not in report.verdict_message
    assert report.forward_price_check is not None and report.forward_price_check.passed


def test_no_forward_price_check_when_no_density_was_extracted():
    fetched_at = datetime.now()
    quotes = [OptionQuote(strike=90.0, option_type="put", bid=1.0, ask=1.2, last_price=1.1, open_interest=10, volume=5)]
    snapshot = OptionChainSnapshot(
        ticker="ILLIQUID", expiration=fetched_at.date() + timedelta(days=30),
        fetched_at=fetched_at, spot_price=100.0, quotes=quotes,
    )

    report = build_report(snapshot, _make_rates(0.03, 0.0))

    assert report.verdict == Verdict.INSUFFICIENT_LIQUID_DATA
    assert report.forward_price_check is None
    assert report.to_dict()["forward_price_check"] is None
    assert "Forward-price check" not in report.summary()


def test_liquidity_floor_excludes_thin_quotes():
    snapshot = _make_snapshot(spot=100.0, r=0.02, q=0.0, t_years=0.5, true_vol=0.2)
    # Thin out every other quote's open interest/volume below a floor of 60.
    thinned_quotes = [
        q if i % 2 == 0 else OptionQuote(q.strike, q.option_type, q.bid, q.ask, q.last_price, 5, 5)
        for i, q in enumerate(snapshot.quotes)
    ]
    thin_snapshot = OptionChainSnapshot(
        ticker=snapshot.ticker, expiration=snapshot.expiration, fetched_at=snapshot.fetched_at,
        spot_price=snapshot.spot_price, quotes=thinned_quotes,
    )
    rates = _make_rates(0.02, 0.0)

    report = build_report(thin_snapshot, rates, LiquidityFloor(min_open_interest=60, min_volume=60))

    assert report.fitted_smile is not None
    excluded_reasons = {e.reason for e in report.fitted_smile.excluded_points}
    assert "below_liquidity_floor" in excluded_reasons
