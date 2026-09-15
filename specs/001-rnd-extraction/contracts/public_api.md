# Public API Contract: breeden_litzenberger

This is the supported surface (research.md §5). Everything else in the package is
reachable via explicit submodule import but is not covered by this contract or by
the compatibility expectations that follow from it.

## `breeden_litzenberger.extract`

```python
def extract(
    ticker: str,
    expiration: date,
    *,
    risk_free_rate: float | None = None,
    dividend_yield: float | None = None,
    liquidity_floor: LiquidityFloor | None = None,
) -> ExtractionReport: ...
```

The single top-level operation (spec FR-013). Fetches one option chain snapshot for
`ticker`/`expiration`, runs the full pipeline (clean → IV invert → SVI fit → BL
extract → validate), and returns one `ExtractionReport`.

- `risk_free_rate`: overrides the Treasury-proxy default (spec FR-002).
- `dividend_yield`: overrides the trailing-yield/zero-fallback default (spec FR-003).
- `liquidity_floor`: overrides the default `LiquidityFloor(min_open_interest=0,
  min_volume=0)` exclusion rule (spec FR-005, assumption in spec.md).

**Raises**: a descriptive exception (not a silent empty/malformed result) when:
- the underlying data fetch fails or returns no data for `ticker`/`expiration`
  (network error, rate limiting, invalid/delisted ticker) — spec FR-001, Edge Cases.
- `expiration` is not strictly in the future relative to the snapshot's fetch time —
  spec FR-004, Edge Cases.

**Never raises** for market-quality problems that are instead reported as diagnostics
on the returned `ExtractionReport` (illiquid smile, arbitrage detected, normalization
or forward-price deviation) — those produce a populated `Verdict`, per spec FR-013.

## `LiquidityFloor`

```python
@dataclass(frozen=True)
class LiquidityFloor:
    min_open_interest: int = 0
    min_volume: int = 0
```

A quote is excluded when **both** `open_interest < min_open_interest` and
`volume < min_volume` (spec FR-005; default per spec.md Assumptions).

## `ExtractionReport`

See `data-model.md` for the full field list. Contract-relevant methods:

```python
class ExtractionReport:
    def to_dict(self) -> dict[str, Any]: ...
    def summary(self) -> str: ...
    def plot_data(self) -> PlotData: ...
```

- `to_dict()`: structured-data rendering (spec FR-014a) — every field in
  `data-model.md`'s `ExtractionReport`, nested dataclasses converted to nested dicts,
  `numpy` arrays converted to lists.
- `summary()`: human-readable text rendering (spec FR-014b) — verdict message,
  fitted SVI parameters, fit quality, arbitrage status, density moments, forward-price
  check, smile-shape metrics, one section each, in that order.
- `plot_data()`: returns a `PlotData` value (below) — the series needed to plot the
  fitted smile against raw IV points, and the extracted density against a lognormal
  benchmark (spec FR-014c). Returns data only; rendering an actual chart is the
  caller's responsibility (spec Assumptions) — `plotting.py`'s matplotlib helpers are
  a convenience built on top of this, not part of the required contract.

```python
@dataclass(frozen=True)
class PlotData:
    smile_strikes: np.ndarray           # retained ImpliedVolPoint strikes
    smile_market_ivs: np.ndarray        # retained ImpliedVolPoint implied vols
    smile_fitted_strikes: np.ndarray    # dense grid across the fitted smile's domain
    smile_fitted_ivs: np.ndarray        # SVI-implied vol at each fitted_strikes point
    density_strikes: np.ndarray         # == DensityGrid.strikes
    density_values: np.ndarray          # == DensityGrid.density
    lognormal_benchmark_density: np.ndarray  # lognormal pdf at same forward/ATM vol
```

## `Verdict`

`Enum` with values `WELL_FIT_ARBITRAGE_FREE`, `ARBITRAGE_DETECTED`,
`DIAGNOSTIC_WARNING`, `INSUFFICIENT_LIQUID_DATA` (see `data-model.md`).

## CLI contract (`breeden_litzenberger.cli`, entry point `bl`)

Thin wrapper over `extract()` — not an alternate code path (plan.md Structure
Decision).

```text
bl extract --ticker SPY --expiration 2026-12-18
           [--rate 0.045] [--dividend-yield 0.013]
           [--min-open-interest 10] [--min-volume 10]
           [--format text|json]
```

- Default `--format text` prints `ExtractionReport.summary()`.
- `--format json` prints `json.dumps(ExtractionReport.to_dict())`.
- Exit code non-zero and a stderr message on any raised exception from `extract()`
  (data-fetch failure, invalid expiration); exit code 0 with the report printed
  regardless of `Verdict` value otherwise (an `ARBITRAGE_DETECTED` verdict is a
  successful run that found a problem, not a CLI failure).
