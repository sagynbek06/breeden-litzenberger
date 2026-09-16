"""Live option chain snapshot fetch via yfinance (spec FR-001).

yfinance and pandas are isolated to this module (and data/rates.py) only --
core/ never imports either, per plan.md's Structure Decision, so the pure
numpy/scipy math is independently unit-testable with zero network
dependency (constitution: no network access inside core/).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Literal

import yfinance as yf  # type: ignore[import-untyped]

OptionType = Literal["call", "put"]


@dataclass(frozen=True)
class OptionQuote:
    """A single listed quote at one strike, for one option type (data-model.md)."""

    strike: float
    option_type: OptionType
    bid: float
    ask: float
    last_price: float
    open_interest: int
    volume: int

    @property
    def mid_price(self) -> float:
        return (self.bid + self.ask) / 2.0


@dataclass(frozen=True)
class OptionChainSnapshot:
    """The full input to one extraction (spec FR-001, FR-016).

    ``fetched_at`` is captured once, here, at fetch time -- the library's
    sole source of "now" for the whole extraction (research.md §7;
    constitution Principle IV).
    """

    ticker: str
    expiration: date
    fetched_at: datetime
    spot_price: float
    quotes: list[OptionQuote]


class DataFetchError(RuntimeError):
    """The underlying data fetch failed or returned no usable data (spec
    FR-001, Edge Cases). Raised immediately -- no retry, no silent
    fallback (per /speckit.clarify).
    """


def fetch_snapshot(ticker: str, expiration: date) -> OptionChainSnapshot:
    """Fetch the option chain snapshot and spot price for ``ticker``/``expiration``."""
    fetched_at = datetime.now()
    try:
        yf_ticker = yf.Ticker(ticker)
        chain = yf_ticker.option_chain(expiration.isoformat())
        spot_price = float(yf_ticker.fast_info["lastPrice"])
    except DataFetchError:
        raise
    except Exception as exc:
        raise DataFetchError(
            f"failed to fetch option chain for ticker={ticker!r} expiration={expiration!r}: {exc}"
        ) from exc

    calls_df, puts_df = chain.calls, chain.puts
    if calls_df.empty and puts_df.empty:
        raise DataFetchError(f"no option quotes returned for ticker={ticker!r} expiration={expiration!r}")

    quotes: list[OptionQuote] = []
    for _, row in calls_df.iterrows():
        quotes.append(_row_to_quote(row, "call"))
    for _, row in puts_df.iterrows():
        quotes.append(_row_to_quote(row, "put"))

    return OptionChainSnapshot(
        ticker=ticker, expiration=expiration, fetched_at=fetched_at, spot_price=spot_price, quotes=quotes
    )


def _row_to_quote(row: Any, option_type: OptionType) -> OptionQuote:
    return OptionQuote(
        strike=float(row["strike"]),
        option_type=option_type,
        bid=float(row["bid"]),
        ask=float(row["ask"]),
        last_price=float(row["lastPrice"]),
        open_interest=_int_or_zero(row["openInterest"]),
        volume=_int_or_zero(row["volume"]),
    )


def _int_or_zero(value: Any) -> int:
    """yfinance returns NaN for open interest/volume on illiquid strikes;
    treated as 0 here so the liquidity-floor exclusion (FR-005) applies
    naturally rather than propagating NaN downstream.
    """
    return 0 if value != value else int(value)  # `value != value` is the NaN check
