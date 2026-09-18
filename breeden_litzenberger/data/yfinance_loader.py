"""Live option chain snapshot fetch via yfinance (spec FR-001, FR-006).

yfinance and pandas are isolated to this module (and data/rates.py) only --
core/ never imports either, per plan.md's Structure Decision, so the pure
numpy/scipy math is independently unit-testable with zero network
dependency (constitution: no network access inside core/).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Literal

import pandas as pd
import yfinance as yf  # type: ignore[import-untyped]

OptionType = Literal["call", "put"]

_DAYS_PER_YEAR = 365.25  # spec FR-004's explicit calendar-day convention

# Keys under DataFrame.attrs on fetch_otm_chain()'s return value.
_ATTR_SPOT = "spot"
_ATTR_FETCHED_AT = "fetched_at"
_ATTR_FORWARD = "forward"
_ATTR_T = "t"


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
    constitution Principle IV). ``quotes`` is already restricted to the
    clean OTM smile (spec FR-006) -- see ``fetch_otm_chain``.
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


class InvalidExpirationError(ValueError):
    """``expiration`` is not strictly in the future relative to the
    snapshot's fetch time (spec FR-004). Defined here, not in report.py:
    T (and therefore this check) is now needed at fetch time, to compute
    the forward price ``fetch_otm_chain`` stitches the smile around.
    """


def time_to_expiration(expiration: date, fetched_at: datetime) -> float:
    """T = (expiration - fetched_at.date()).days / 365.25 (spec FR-004).

    A pure function of two already-fixed timestamps -- computing T more
    than once from the *same* (expiration, fetched_at) pair (as both this
    module and report.py do) is not a new capture of "now", just a
    deterministic re-derivation of it (constitution Principle IV is about
    never calling a clock more than once per extraction, not about calling
    this formula more than once).
    """
    days = (expiration - fetched_at.date()).days
    if days <= 0:
        raise InvalidExpirationError(
            f"expiration {expiration} is not strictly in the future relative to "
            f"the snapshot fetched at {fetched_at}"
        )
    return days / _DAYS_PER_YEAR


def fetch_otm_chain(ticker: str, expiration: date, risk_free_rate: float, dividend_yield: float) -> pd.DataFrame:
    """Fetch one option chain snapshot and stitch together a single clean
    out-of-the-money smile (spec FR-001, FR-006).

    Fetches calls and puts as separate DataFrames via
    ``yfinance.Ticker.option_chain``, computes the forward price estimate
    ``F = S0 * exp((r - q) * T)`` from the fetched spot price, the given
    risk-free rate / dividend yield -- whichever the caller supplied,
    override or already-resolved default (data/rates.py); this function
    doesn't care which -- and T derived from this fetch's own timestamp.
    It then retains only OTM put quotes (``strike < F``) and OTM call
    quotes (``strike >= F``), dropping everything else, and returns one
    combined DataFrame labeled with an ``"option_type"`` column.

    yfinance equity options are American-style (exercisable any time up to
    expiration), while this library's entire pipeline -- Black-Scholes
    pricing, implied-vol inversion, and Breeden-Litzenberger extraction --
    assumes European exercise (exercisable only at expiration). An
    American option carries an early-exercise premium the European
    formulas don't price in. Restricting to out-of-the-money quotes is the
    standard practitioner approximation that makes treating them as
    European reasonable anyway: an OTM option has no intrinsic value to
    capture by exercising early, so that premium is negligible, whereas
    the in-the-money quotes this function drops entirely are exactly where
    it would not be. This is a deliberate, disclosed approximation, not an
    oversight -- see docs/theory.md §6 for the full discussion, including
    a note on genuinely European-style data sources as a future extension.

    Spot price, the fetch timestamp, the computed forward, and T are
    attached to the returned DataFrame's ``.attrs`` (keys ``"spot"``,
    ``"fetched_at"``, ``"forward"``, ``"t"``) so a caller needing them --
    ``fetch_snapshot`` below -- can reuse this fetch's own values instead
    of fetching or deriving them a second time.
    """
    fetched_at = datetime.now()
    try:
        yf_ticker = yf.Ticker(ticker)
        chain = yf_ticker.option_chain(expiration.isoformat())
        spot = float(yf_ticker.fast_info["lastPrice"])
    except Exception as exc:
        raise DataFetchError(
            f"failed to fetch option chain for ticker={ticker!r} expiration={expiration!r}: {exc}"
        ) from exc

    calls_df, puts_df = chain.calls, chain.puts
    if calls_df.empty and puts_df.empty:
        raise DataFetchError(f"no option quotes returned for ticker={ticker!r} expiration={expiration!r}")

    t = time_to_expiration(expiration, fetched_at)
    forward = spot * math.exp((risk_free_rate - dividend_yield) * t)

    calls_df = calls_df.assign(option_type="call")
    puts_df = puts_df.assign(option_type="put")
    otm_calls = calls_df[calls_df["strike"] >= forward]
    otm_puts = puts_df[puts_df["strike"] < forward]

    combined: pd.DataFrame = (
        pd.concat([otm_puts, otm_calls], ignore_index=True)
        .sort_values("strike", kind="stable")
        .reset_index(drop=True)
    )
    combined.attrs[_ATTR_SPOT] = spot
    combined.attrs[_ATTR_FETCHED_AT] = fetched_at
    combined.attrs[_ATTR_FORWARD] = forward
    combined.attrs[_ATTR_T] = t
    return combined


def fetch_snapshot(ticker: str, expiration: date, risk_free_rate: float, dividend_yield: float) -> OptionChainSnapshot:
    """Fetch the option chain snapshot for ``ticker``/``expiration``, already
    restricted to the clean OTM smile -- see ``fetch_otm_chain``.
    """
    combined = fetch_otm_chain(ticker, expiration, risk_free_rate, dividend_yield)
    quotes = [_row_to_quote(row, row["option_type"]) for _, row in combined.iterrows()]
    return OptionChainSnapshot(
        ticker=ticker,
        expiration=expiration,
        fetched_at=combined.attrs[_ATTR_FETCHED_AT],
        spot_price=combined.attrs[_ATTR_SPOT],
        quotes=quotes,
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
