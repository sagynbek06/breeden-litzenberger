"""Risk-free rate and dividend-yield defaults (spec FR-002, FR-003; research.md §3-4).

yfinance is isolated to this module (and data/yfinance_loader.py) only --
core/ never imports it (constitution: no network access inside core/).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import yfinance as yf  # type: ignore[import-untyped]

from breeden_litzenberger.data.yfinance_loader import DataFetchError

RateSource = Literal["treasury_proxy", "override"]
DividendSource = Literal["trailing_yield", "zero_fallback", "override"]

_TREASURY_PROXY_TICKER = "^IRX"
_TREASURY_PROXY_TENOR_DAYS = 91  # 13-week T-bill


@dataclass(frozen=True)
class RateInputs:
    """Risk-free rate and dividend yield actually used, with provenance
    (data-model.md) -- FR-002/FR-003's "never silently assumed" requirement.
    """

    risk_free_rate: float
    risk_free_rate_source: RateSource
    dividend_yield: float
    dividend_yield_source: DividendSource


def _discount_yield_to_continuous_rate(discount_yield: float, days: int = _TREASURY_PROXY_TENOR_DAYS) -> float:
    """Convert a T-bill discount-yield quote to a continuously compounded rate.

    Discount-yield convention: Price = 100 * (1 - discount_yield * days/360).
    The continuously compounded equivalent over the bill's own tenor is
    r = -ln(Price/100) / (days/365), using the same calendar-day/365
    convention as T elsewhere in this library (spec FR-004).
    """
    price = 100.0 * (1.0 - discount_yield * days / 360.0)
    t_years = days / 365.0
    return -math.log(price / 100.0) / t_years


def _default_risk_free_rate() -> float:
    """^IRX (13-week Treasury proxy), converted to a continuous rate (research.md §3)."""
    try:
        quote_percent = float(yf.Ticker(_TREASURY_PROXY_TICKER).fast_info["lastPrice"])
    except Exception as exc:
        raise DataFetchError(f"failed to fetch risk-free rate proxy {_TREASURY_PROXY_TICKER!r}: {exc}") from exc
    return _discount_yield_to_continuous_rate(quote_percent / 100.0)


def _default_dividend_yield(ticker: str) -> tuple[float, DividendSource]:
    """trailingAnnualDividendYield -> dividendYield -> 0.0 (research.md §4)."""
    try:
        info = yf.Ticker(ticker).info
    except Exception as exc:
        raise DataFetchError(f"failed to fetch dividend yield info for ticker={ticker!r}: {exc}") from exc

    trailing = info.get("trailingAnnualDividendYield")
    if trailing is not None:
        return float(trailing), "trailing_yield"
    fallback = info.get("dividendYield")
    if fallback is not None:
        return float(fallback), "trailing_yield"
    return 0.0, "zero_fallback"


def resolve_rate_inputs(
    ticker: str,
    override_rate: float | None,
    override_dividend_yield: float | None,
) -> RateInputs:
    """Resolve the risk-free rate and dividend yield to use for one extraction,
    applying caller overrides (spec FR-002, FR-003) and always reporting the
    provenance of each value.
    """
    rate_source: RateSource
    if override_rate is not None:
        risk_free_rate, rate_source = override_rate, "override"
    else:
        risk_free_rate, rate_source = _default_risk_free_rate(), "treasury_proxy"

    dividend_source: DividendSource
    if override_dividend_yield is not None:
        dividend_yield, dividend_source = override_dividend_yield, "override"
    else:
        dividend_yield, dividend_source = _default_dividend_yield(ticker)

    return RateInputs(
        risk_free_rate=risk_free_rate,
        risk_free_rate_source=rate_source,
        dividend_yield=dividend_yield,
        dividend_yield_source=dividend_source,
    )
