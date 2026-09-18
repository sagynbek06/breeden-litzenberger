"""Top-level orchestration: single-call, single-snapshot RND extraction (spec FR-013).

This is the seam between data/ (pandas/yfinance) and core/ (pure
numpy/scipy): an OptionChainSnapshot's OptionQuote list -- already
OTM-selected by data/yfinance_loader.py's fetch_otm_chain -- is converted
into plain CleanOTMPoint/ExcludedQuote records here (liquidity/crossed-
market cleaning only) before anything reaches core/, and this is where the
single top-level public operation (extract()) is assembled from core/'s
independently-tested pieces (plan.md Structure Decision).
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from datetime import date
from enum import Enum
from typing import Any

import numpy as np

from breeden_litzenberger.core.black_scholes import ImpliedVolError, implied_volatility
from breeden_litzenberger.core.density import (
    NORMALIZATION_ERROR_THRESHOLD,
    DensityDiagnostics,
    DensityGrid,
    extract_density,
)
from breeden_litzenberger.core.moments import (
    DensityMoments,
    ForwardPriceCheck,
    compute_moments,
    forward_price_check,
)
from breeden_litzenberger.core.smile import FittedSmile, calibrate_svi
from breeden_litzenberger.core.smile_metrics import SmileShapeMetrics, compute_smile_shape_metrics
from breeden_litzenberger.data.rates import RateInputs, resolve_rate_inputs
from breeden_litzenberger.data.yfinance_loader import (
    OptionChainSnapshot,
    OptionType,
    fetch_snapshot,
    time_to_expiration,
)

# Minimum retained points to calibrate a 5-parameter SVI smile at all.
# Below this the fit would be underdetermined, not just noisy -- spec
# leaves the exact threshold as an implementation choice (Assumptions).
_MIN_POINTS_FOR_SMILE_FIT = 5


@dataclass(frozen=True)
class LiquidityFloor:
    """Configurable liquidity exclusion rule (spec FR-005; default per spec.md Assumptions)."""

    min_open_interest: int = 0
    min_volume: int = 0


@dataclass(frozen=True)
class ExcludedQuote:
    """A quote dropped during cleaning or IV inversion, with the reason
    (spec FR-005, FR-007) -- exclusions are logged, never silent.
    """

    strike: float
    option_type: OptionType
    reason: str


@dataclass(frozen=True)
class CleanOTMPoint:
    """A retained quote after liquidity/crossed-market cleaning (spec FR-005);
    OTM selection (spec FR-006) already happened at fetch time -- see
    data/yfinance_loader.py's fetch_otm_chain.
    """

    strike: float
    option_type: OptionType
    mid_price: float


class Verdict(Enum):
    """Plain-language outcome classification (spec FR-013)."""

    WELL_FIT_ARBITRAGE_FREE = "well-fit and arbitrage-free"
    ARBITRAGE_DETECTED = "butterfly arbitrage detected at fitted parameters"
    DIAGNOSTIC_WARNING = "fit converged but a diagnostic threshold was exceeded"
    INSUFFICIENT_LIQUID_DATA = "insufficient liquid strikes to fit a smile"


class NoPlotDataError(ValueError):
    """``plot_data()`` called on a report with no fitted smile (spec Edge Cases)."""


@dataclass(frozen=True)
class PlotData:
    """The series needed to plot the fitted smile and the extracted density
    (spec FR-014c; contracts/public_api.md) -- plain data, not a rendered
    chart; rendering is the caller's concern (spec Assumptions).
    """

    smile_strikes: np.ndarray
    smile_market_ivs: np.ndarray
    smile_fitted_strikes: np.ndarray
    smile_fitted_ivs: np.ndarray
    density_strikes: np.ndarray
    density_values: np.ndarray
    lognormal_benchmark_density: np.ndarray


@dataclass(frozen=True)
class ExtractionReport:
    """The single combined result of one extraction (spec FR-013, FR-014)."""

    ticker: str
    expiration: date
    snapshot: OptionChainSnapshot
    rates: RateInputs
    fitted_smile: FittedSmile | None
    density_grid: DensityGrid | None
    diagnostics: DensityDiagnostics | None
    moments: DensityMoments | None
    forward_price_check: ForwardPriceCheck | None  # None only when no density was extracted
    smile_shape_metrics: SmileShapeMetrics | None
    verdict: Verdict
    verdict_message: str

    def to_dict(self) -> dict[str, Any]:
        """Structured-data rendering (spec FR-014a)."""
        return {
            "ticker": self.ticker,
            "expiration": self.expiration.isoformat(),
            "rates": asdict(self.rates),
            "fitted_smile": None
            if self.fitted_smile is None
            else {
                "params": asdict(self.fitted_smile.params),
                "forward_price": self.fitted_smile.forward_price,
                "fit_rmse_variance": self.fitted_smile.fit_rmse_variance,
                "butterfly_arbitrage_free": self.fitted_smile.butterfly_arbitrage_free,
                "arbitrage_margin": self.fitted_smile.arbitrage_check.margin,
            },
            "diagnostics": None if self.diagnostics is None else asdict(self.diagnostics),
            "moments": None if self.moments is None else asdict(self.moments),
            "forward_price_check": None if self.forward_price_check is None else asdict(self.forward_price_check),
            "smile_shape_metrics": None if self.smile_shape_metrics is None else asdict(self.smile_shape_metrics),
            "density_grid": None
            if self.density_grid is None
            else {
                "strikes": self.density_grid.strikes.tolist(),
                "density": self.density_grid.density.tolist(),
            },
            "verdict": self.verdict.name,
            "verdict_message": self.verdict_message,
        }

    def summary(self) -> str:
        """Human-readable text rendering (spec FR-014b)."""
        lines = [
            f"Risk-Neutral Density Extraction Report: {self.ticker} exp {self.expiration.isoformat()}",
            f"Verdict: {self.verdict.value} -- {self.verdict_message}",
            "",
            f"Inputs: spot={self.snapshot.spot_price:.4f}  "
            f"r={self.rates.risk_free_rate:.4%} ({self.rates.risk_free_rate_source})  "
            f"q={self.rates.dividend_yield:.4%} ({self.rates.dividend_yield_source})",
        ]
        if self.fitted_smile is not None:
            p = self.fitted_smile.params
            lines += [
                "",
                f"Fitted SVI: a={p.a:.6f} b={p.b:.6f} rho={p.rho:.4f} m={p.m:.4f} sigma={p.sigma:.4f}",
                f"Fit RMSE (variance space): {self.fitted_smile.fit_rmse_variance:.3e}",
                f"Butterfly-arbitrage-free: {self.fitted_smile.butterfly_arbitrage_free} "
                f"(margin={self.fitted_smile.arbitrage_check.margin:.6f})",
            ]
        if self.diagnostics is not None:
            d = self.diagnostics
            lines += [
                "",
                f"Density non-negative: {d.is_non_negative}  integral={d.integral:.6f} "
                f"(normalization_error={d.normalization_error:.4%})",
            ]
        if self.forward_price_check is not None:
            c = self.forward_price_check
            lines += [
                f"Forward-price check: {'PASSED' if c.passed else 'FAILED'} -- "
                f"RND mean={c.realized_mean:.4f} vs theoretical forward={c.theoretical_forward:.4f} "
                f"(abs error={c.absolute_error:.4f}, rel error={c.relative_error:.4%}, "
                f"limit={c.threshold:.4%})",
            ]
        if self.moments is not None:
            m = self.moments
            lines += [
                "",
                f"Moments: mean={m.mean:.4f} variance={m.variance:.4f} "
                f"skew={m.skewness:.4f} excess_kurtosis={m.excess_kurtosis:.4f}",
            ]
        if self.smile_shape_metrics is not None:
            s = self.smile_shape_metrics
            lines += [
                "",
                f"Smile shape: ATM vol={s.atm_implied_vol:.4%}  "
                f"25d risk reversal={s.risk_reversal_25d:.4%}  25d butterfly={s.butterfly_25d:.4%}",
            ]
        return "\n".join(lines)

    def plot_data(self) -> PlotData:
        """The series needed to plot the fitted smile vs. raw IV points, and
        the extracted density vs. a lognormal benchmark (spec FR-014c).
        """
        if self.fitted_smile is None or self.density_grid is None or self.smile_shape_metrics is None:
            raise NoPlotDataError(
                f"no plot data available for verdict={self.verdict.name}: "
                f"a smile must have been fitted first"
            )

        t = time_to_expiration(self.snapshot.expiration, self.snapshot.fetched_at)
        smile = self.fitted_smile

        smile_strikes = np.array([p.strike for p in smile.retained_points])
        smile_market_ivs = np.array([p.implied_vol for p in smile.retained_points])

        fitted_strikes = np.linspace(float(smile_strikes.min()), float(smile_strikes.max()), 200)
        k_fitted = np.log(fitted_strikes / smile.forward_price)
        w_fitted = smile.params.total_variance(k_fitted)
        fitted_ivs = np.sqrt(np.maximum(w_fitted, 1e-12) / t)

        atm_vol = self.smile_shape_metrics.atm_implied_vol
        density_strikes = self.density_grid.strikes
        v = atm_vol * atm_vol * t
        mu = np.log(smile.forward_price) - 0.5 * v
        lognormal_benchmark = (1.0 / (density_strikes * atm_vol * np.sqrt(2 * np.pi * t))) * np.exp(
            -((np.log(density_strikes) - mu) ** 2) / (2 * v)
        )

        return PlotData(
            smile_strikes=smile_strikes,
            smile_market_ivs=smile_market_ivs,
            smile_fitted_strikes=fitted_strikes,
            smile_fitted_ivs=fitted_ivs,
            density_strikes=density_strikes,
            density_values=self.density_grid.density,
            lognormal_benchmark_density=lognormal_benchmark,
        )


def _forward_check_failure_message(check: ForwardPriceCheck) -> str:
    return (
        f"forward-price check failed: extracted RND mean deviates from theoretical forward "
        f"by {check.relative_error:.2%} (limit {check.threshold:.2%})"
    )


def _compute_forward_price(spot: float, r: float, q: float, t: float) -> float:
    return float(spot * np.exp((r - q) * t))


def _clean_quotes(
    snapshot: OptionChainSnapshot, liquidity_floor: LiquidityFloor
) -> tuple[list[CleanOTMPoint], list[ExcludedQuote]]:
    """FR-005 liquidity/crossed-market cleaning. OTM selection (FR-006)
    already happened in data/yfinance_loader.py's fetch_otm_chain -- every
    quote in ``snapshot.quotes`` is already known to be OTM relative to the
    forward price computed at fetch time, so there is nothing left to tag
    "not_otm" here; a non-OTM quote is simply never fetched in the first
    place, not excluded after the fact.
    """
    retained: list[CleanOTMPoint] = []
    excluded: list[ExcludedQuote] = []

    for quote in snapshot.quotes:
        # NaN bid/ask (real yfinance quotes, e.g. an illiquid strike with no
        # posted bid) must be checked explicitly: `math.nan <= 0` and
        # `math.nan > x` both evaluate to False in Python, so a plain
        # `quote.bid <= 0` check silently lets a NaN bid through to become a
        # NaN mid price a few steps downstream (verified against a real SPY
        # chain during development -- see tasks.md T014 notes).
        if math.isnan(quote.bid) or math.isnan(quote.ask) or quote.bid <= 0:
            excluded.append(ExcludedQuote(quote.strike, quote.option_type, "zero_bid"))
            continue
        if quote.bid > quote.ask:
            excluded.append(ExcludedQuote(quote.strike, quote.option_type, "crossed_market"))
            continue
        if quote.open_interest < liquidity_floor.min_open_interest and quote.volume < liquidity_floor.min_volume:
            excluded.append(ExcludedQuote(quote.strike, quote.option_type, "below_liquidity_floor"))
            continue

        retained.append(CleanOTMPoint(strike=quote.strike, option_type=quote.option_type, mid_price=quote.mid_price))

    return retained, excluded


def extract(
    ticker: str,
    expiration: date,
    *,
    risk_free_rate: float | None = None,
    dividend_yield: float | None = None,
    liquidity_floor: LiquidityFloor | None = None,
) -> ExtractionReport:
    """The single top-level operation (spec FR-013). See contracts/public_api.md.

    Rates are resolved before the chain is fetched, not after: fetch_snapshot
    needs (r, q) up front to compute the forward price its OTM stitching is
    anchored to (data/yfinance_loader.py's fetch_otm_chain).
    """
    rates = resolve_rate_inputs(ticker, risk_free_rate, dividend_yield)
    snapshot = fetch_snapshot(ticker, expiration, rates.risk_free_rate, rates.dividend_yield)
    floor = liquidity_floor if liquidity_floor is not None else LiquidityFloor()
    return build_report(snapshot, rates, floor)


def build_report(
    snapshot: OptionChainSnapshot,
    rates: RateInputs,
    liquidity_floor: LiquidityFloor | None = None,
) -> ExtractionReport:
    """Everything after the data fetch -- a pure function of (snapshot, rates,
    liquidity_floor) with no network access, so it can be exercised
    deterministically without mocking yfinance (tests/test_report_integration.py).
    ``extract()`` is a thin wrapper: fetch, then delegate here.
    """
    floor = liquidity_floor if liquidity_floor is not None else LiquidityFloor()
    ticker, expiration = snapshot.ticker, snapshot.expiration

    t = time_to_expiration(snapshot.expiration, snapshot.fetched_at)
    forward_price = _compute_forward_price(snapshot.spot_price, rates.risk_free_rate, rates.dividend_yield, t)

    clean_points, excluded = _clean_quotes(snapshot, floor)

    iv_points = []
    for point in clean_points:
        try:
            iv_points.append(
                implied_volatility(
                    point.mid_price,
                    snapshot.spot_price,
                    point.strike,
                    t,
                    rates.risk_free_rate,
                    rates.dividend_yield,
                    point.option_type,
                    forward_price,
                )
            )
        except ImpliedVolError as exc:
            excluded.append(ExcludedQuote(point.strike, point.option_type, exc.reason))

    if len(iv_points) < _MIN_POINTS_FOR_SMILE_FIT:
        return ExtractionReport(
            ticker=ticker,
            expiration=expiration,
            snapshot=snapshot,
            rates=rates,
            fitted_smile=None,
            density_grid=None,
            diagnostics=None,
            moments=None,
            forward_price_check=None,
            smile_shape_metrics=None,
            verdict=Verdict.INSUFFICIENT_LIQUID_DATA,
            verdict_message=(
                f"only {len(iv_points)} liquid OTM quotes survived cleaning; at least "
                f"{_MIN_POINTS_FOR_SMILE_FIT} are required to calibrate a 5-parameter SVI smile"
            ),
        )

    calibrated = calibrate_svi(iv_points, forward_price)
    fitted_smile = FittedSmile(
        params=calibrated.params,
        forward_price=calibrated.forward_price,
        fit_rmse_variance=calibrated.fit_rmse_variance,
        arbitrage_check=calibrated.arbitrage_check,
        retained_points=calibrated.retained_points,
        excluded_points=excluded,  # cleaning-stage + IV-inversion-stage exclusions, merged
    )

    density_grid, diagnostics = extract_density(
        fitted_smile, snapshot.spot_price, t, rates.risk_free_rate, rates.dividend_yield
    )
    moments = compute_moments(density_grid)
    smile_shape_metrics = compute_smile_shape_metrics(
        fitted_smile, snapshot.spot_price, t, rates.risk_free_rate, rates.dividend_yield
    )

    fwd_check = forward_price_check(
        density_grid.density,
        density_grid.strikes,
        snapshot.spot_price,
        rates.risk_free_rate,
        rates.dividend_yield,
        t,
    )

    # Every failed check is reported, not just the first: a forward-price
    # failure must never be dropped from the verdict just because a more
    # prominent problem (e.g. detected arbitrage) was also present, nor
    # because the rest of the pipeline ran without raising.
    problems: list[str] = []
    if not fitted_smile.butterfly_arbitrage_free:
        problems.append(
            f"SVI calibration converged but the fitted smile violates the Gatheral-Jacquier "
            f"no-butterfly-arbitrage condition (margin={fitted_smile.arbitrage_check.margin:.6f} "
            f"at k={fitted_smile.arbitrage_check.margin_at_k:.4f})"
        )
    if not diagnostics.is_non_negative:
        problems.append(f"extracted density is negative somewhere (min={diagnostics.min_density:.3e})")
    if diagnostics.normalization_error > NORMALIZATION_ERROR_THRESHOLD:
        problems.append(
            f"extracted density integrates to {diagnostics.integral:.4f}, not 1 "
            f"(normalization error {diagnostics.normalization_error:.2%})"
        )
    if not fwd_check.passed:
        problems.append(_forward_check_failure_message(fwd_check))

    if not fitted_smile.butterfly_arbitrage_free:
        verdict = Verdict.ARBITRAGE_DETECTED
        verdict_message = "; ".join(problems)
    elif problems:
        verdict = Verdict.DIAGNOSTIC_WARNING
        verdict_message = "smile is arbitrage-free, but " + "; ".join(problems)
    else:
        verdict = Verdict.WELL_FIT_ARBITRAGE_FREE
        verdict_message = (
            "smile is arbitrage-free and well-fit; diagnostics within tolerance; forward-price check passed "
            f"(extracted RND mean deviates from theoretical forward by {fwd_check.relative_error:.2%})"
        )

    return ExtractionReport(
        ticker=ticker,
        expiration=expiration,
        snapshot=snapshot,
        rates=rates,
        fitted_smile=fitted_smile,
        density_grid=density_grid,
        diagnostics=diagnostics,
        moments=moments,
        forward_price_check=fwd_check,
        smile_shape_metrics=smile_shape_metrics,
        verdict=verdict,
        verdict_message=verdict_message,
    )
