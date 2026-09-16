---

description: "Task list for feature implementation"
---

# Tasks: Risk-Neutral Density Extraction Library

**Input**: Design documents from `specs/001-rnd-extraction/`

**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/public_api.md, quickstart.md

**Tests**: Included — the plan explicitly specifies a three-tier pytest/hypothesis test suite as part of the deliverable, and User Story 2's entire purpose is a mandatory ground-truth test (constitution Principle III).

**Organization**: Tasks are grouped by user story (spec.md: US1 "Extract a report from a live option chain", US2 "Validate against a known distribution", US3 "Inspect smile shape and plot data") to enable independent implementation and testing.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependency on an incomplete task)
- **[Story]**: US1 / US2 / US3 — omitted for Setup, Foundational, and Polish tasks
- File paths are exact, per `plan.md`'s Project Structure

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Repository/package scaffolding, no feature logic yet.

- [X] T001 Create the package skeleton per `plan.md` Project Structure: `breeden_litzenberger/__init__.py`, `breeden_litzenberger/data/__init__.py`, `breeden_litzenberger/core/__init__.py`, `tests/__init__.py`, `docs/` directory, empty `README.md`
- [X] T002 Create `pyproject.toml`: package metadata, runtime dependencies (`numpy`, `scipy`, `pandas`, `yfinance`, `matplotlib`, `typer`) and a `dev` extra (`pytest`, `hypothesis`, `mypy`), installable via `pip install -e ".[dev]"` (quickstart.md Setup)
- [X] T003 [P] Configure `mypy --strict` scoped to `breeden_litzenberger/` in `pyproject.toml` (or `mypy.ini`)
- [X] T004 [P] Configure `.github/workflows/ci.yml`: run `pytest` and `mypy --strict` on push and pull request (research.md §6)

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: The pure numpy/scipy math pipeline (`core/`) that every user story is built on. No pandas/yfinance types may appear here (plan.md Structure Decision).

**⚠️ CRITICAL**: No user story task may begin until this phase is complete.

- [X] T005 [P] Implement `breeden_litzenberger/core/black_scholes.py`: `price()`, `vega()`, and `implied_volatility()` via `scipy.optimize.brentq` bounded to the sane annualized-vol bracket [1%, 500%]; the `ImpliedVolPoint` dataclass (`strike: float`, `implied_vol: float`, `log_forward_moneyness: float`, `total_variance: float`); degenerate cases (price below intrinsic value, zero-price quote, no volatility in bracket reproduces the price) must be raised or returned as an exclusion with reason `"below_intrinsic_value"` / `"zero_price"` / `"iv_bracket_failure"` respectively — never silently coerced to a placeholder vol (FR-007). Module docstring cites Black & Scholes (1973) per constitution Principle I.
- [X] T006 [P] Implement `tests/test_black_scholes.py`: hard-coded reference-value Black-Scholes price tests, and price→implied-vol→price round-trip tests at several strike/vol/tenor combinations, asserting `rtol=1e-6`
- [X] T007 [P] Implement `breeden_litzenberger/core/smile.py`: `SVIParams` dataclass (`a`, `b>=0`, `-1<rho<1`, `m`, `sigma>0`) and `FittedSmile` dataclass (`params`, `forward_price`, `fit_rmse_variance`, `butterfly_arbitrage_free: bool`, `retained_points: list[ImpliedVolPoint]`, `excluded_points: list[ExcludedQuote]`); raw SVI `w(k) = a + b*(rho*(k-m) + sqrt((k-m)**2 + sigma**2))`; `calibrate()` via `scipy.optimize.least_squares` with those bounds, the fixed default initial guess, and the fixed deterministic alternate-guess retry sequence from research.md §2 (no randomness — constitution Principle V); Gatheral-Jacquier (2014) sufficient condition for absence of butterfly arbitrage. Module docstring cites Gatheral (2004) and Gatheral & Jacquier (2014).
- [X] T008 [P] Implement `tests/test_smile_svi.py`: SVI calibration convergence tests, and Gatheral-Jacquier no-butterfly-arbitrage condition checks against synthetic smiles built from both a known-arbitrage-free and a known-arbitrage parameter set
- [X] T009 Implement `breeden_litzenberger/core/density.py`: `DensityGrid` dataclass (`strikes: np.ndarray`, `density: np.ndarray`) and `DensityDiagnostics` dataclass (`is_non_negative`, `min_density`, `integral`, `normalization_error` flagged if `>1%`, `forward_price_theoretical`, `forward_price_realized`, `forward_price_deviation` flagged if `>1%`, `flagged: bool`); reconstruct a dense Black-Scholes call-price grid from a `FittedSmile`'s SVI curve spanning log-forward-moneyness `k ∈ [-8, +8] * sqrt(atm_total_variance)` (`K = F*exp(k)`, minimum 2000 grid points — research.md §1 "Grid range"), then extract `f(K) = exp(r*T) * d²C/dK²` via central finite differences per research.md §1 — **never** differentiates raw market points (constitution Principle II). Module docstring cites Breeden & Litzenberger (1978), explains the convexity argument, and documents the O(h²) central-difference error characteristic (constitution Principles I, VIII). (Depends on: T007 for `FittedSmile`.)
- [X] T010 [P] Implement `breeden_litzenberger/core/moments.py`: `DensityMoments` dataclass (`mean`, `variance`, `skewness`, `excess_kurtosis`), computed by numerical integration over a `DensityGrid`; theoretical forward price `F = S0 * exp((r - q) * T)` (FR-011)
- [X] T011 [P] Implement `tests/test_moments.py`: moment-computation accuracy tests against known closed-form lognormal moments at several volatility/tenor combinations

**Checkpoint**: Foundational math pipeline is implemented and unit-tested. User story work may now begin.

---

## Phase 3: User Story 1 - Extract a risk-neutral density report from a live option chain (Priority: P1) 🎯 MVP

**Goal**: A user gives a ticker + expiration and gets back one structured report — fitted smile, density, all diagnostics, moments, forward-price check, and a verdict.

**Independent Test**: Call the top-level extraction function with a real, liquid ticker and a near-term expiration; verify a report is returned with a fitted smile, a density grid, all diagnostics populated, and a plain-language verdict string.

- [X] T012 [P] [US1] Implement `breeden_litzenberger/data/yfinance_loader.py`: `OptionQuote` dataclass (`strike>0`, `option_type: Literal["call","put"]`, `bid>=0`, `ask>=0`, `last_price`, `open_interest>=0`, `volume>=0`) and `OptionChainSnapshot` dataclass (`ticker`, `expiration: date`, `fetched_at: datetime` captured once at fetch time, `spot_price>0`, `quotes: list[OptionQuote]`); `fetch_snapshot(ticker, expiration) -> OptionChainSnapshot`; raise a descriptive exception immediately — no retry, no silent fallback — on network error, rate limiting, or no data for the ticker/expiration (FR-001, Edge Cases, research.md §7)
- [X] T013 [P] [US1] Implement `breeden_litzenberger/data/rates.py`: `RateInputs` dataclass (`risk_free_rate`, `risk_free_rate_source: Literal["treasury_proxy","override"]`, `dividend_yield`, `dividend_yield_source: Literal["trailing_yield","zero_fallback","override"]`); default risk-free rate from the `^IRX` Treasury proxy converted to continuous compounding (research.md §3); default dividend yield from `trailingAnnualDividendYield` → `dividendYield` → `0.0` (research.md §4); both values overridable by the caller (FR-002, FR-003)
- [X] T014 [US1] Implement cleaning/OTM-selection in `breeden_litzenberger/report.py`: `LiquidityFloor` dataclass (`min_open_interest: int = 0`, `min_volume: int = 0`); `ExcludedQuote` dataclass (`strike`, `option_type`, `reason`) and `CleanOTMPoint` dataclass (`strike`, `option_type`, `mid_price>0`); drop quotes with zero bid, a crossed market (`bid > ask`), or `open_interest < floor.min_open_interest AND volume < floor.min_volume`, recording an `ExcludedQuote` for each; select OTM puts below the forward price and OTM calls above it (FR-005, FR-006). (Depends on: T012 for `OptionChainSnapshot`.)
- [X] T015 [US1] Implement time-to-expiration and input validation in `breeden_litzenberger/report.py`: `T = (expiration - fetched_at.date()).days / 365.25`; raise a descriptive exception if `expiration` is not strictly in the future relative to `fetched_at` (FR-004, Edge Cases). (Depends on: T012.)
- [X] T016 [US1] Implement `Verdict` enum (`WELL_FIT_ARBITRAGE_FREE`, `ARBITRAGE_DETECTED`, `DIAGNOSTIC_WARNING`, `INSUFFICIENT_LIQUID_DATA`) and `ExtractionReport` dataclass in `breeden_litzenberger/report.py` per data-model.md, with `to_dict()` and `summary()` methods per contracts/public_api.md (FR-013, FR-014a, FR-014b)
- [X] T017 [US1] Implement `extract()` orchestration in `breeden_litzenberger/report.py`: wire `data/yfinance_loader.py` (T012) → cleaning/OTM selection (T014) → `core/black_scholes.py` implied-vol inversion (collecting further `ExcludedQuote` entries) → `core/smile.py` calibration (T007) → `core/density.py` extraction (T009) → `core/moments.py` (T010); determine `Verdict`: `INSUFFICIENT_LIQUID_DATA` when too few retained points to fit a smile, `ARBITRAGE_DETECTED` when the butterfly check fails, `DIAGNOSTIC_WARNING` when arbitrage-free but `diagnostics.flagged`, else `WELL_FIT_ARBITRAGE_FREE`; signature `extract(ticker, expiration, *, risk_free_rate=None, dividend_yield=None, liquidity_floor=None) -> ExtractionReport` per contracts/public_api.md (FR-013, FR-015, FR-016). (Depends on: T013, T014, T015, T016, T009.)
- [X] T018 [P] [US1] Implement `breeden_litzenberger/__init__.py`: curated public exports `extract`, `ExtractionReport`, `Verdict`, `LiquidityFloor` per research.md §5. (Depends on: T017.)
- [X] T019 [P] [US1] Implement `tests/test_report_integration.py`: build a deterministic fixture `OptionChainSnapshot` (no network), run `extract()`'s orchestration end-to-end and assert (a) all report sections are populated for a liquid fixture, (b) two calls with identical inputs produce an identical `to_dict()` (FR-015, SC-005), (c) a fixture with too few liquid OTM quotes yields `Verdict.INSUFFICIENT_LIQUID_DATA` (spec US1 Acceptance Scenarios 1–3). (Depends on: T017.)

**Checkpoint**: User Story 1 is fully functional and independently testable — this is the MVP.

---

## Phase 4: User Story 2 - Validate the pipeline against a known distribution (Priority: P1)

**Goal**: Prove the full pipeline recovers a known closed-form lognormal density before it is ever trusted on real data (constitution Principle III).

**Independent Test**: Generate synthetic option prices from a constant-volatility Black-Scholes model, run the full extraction pipeline, and compare the extracted density to the closed-form lognormal density at the same parameters.

- [X] T020 [P] [US2] Implement `tests/_synthetic.py` test helper: generate a synthetic OTM option chain (calls above forward, puts below) by pricing a strike grid under a known constant Black-Scholes volatility via `core/black_scholes.py` (T005), returning both the synthetic quotes and the closed-form lognormal density/moments they imply, for use as ground truth
- [X] T021 [US2] Implement `tests/test_density.py`: run the full pipeline (implied-vol inversion → SVI fit → Breeden-Litzenberger extraction) on T020's synthetic chain, bypassing `data/yfinance_loader.py` entirely, and assert the extracted density's mean and variance are within **0.1% relative error** of the closed-form lognormal values and the density curve matches the analytic curve to **3 decimal places** (spec SC-001). This is THE test that proves the pipeline is mathematically correct — everything else is regression coverage on top of it (constitution Principle III). (Depends on: T009, T020.)
- [ ] T022 [P] [US2] Implement `tests/test_property_based.py` using `hypothesis`: (a) a manufactured strictly convex call-price curve in strike always yields a non-negative extracted density; (b) an SVI fit with `b=0` (flat variance, zero skew) reduces the extracted density to a symmetric lognormal; (c) extracted risk-neutral variance is monotonically increasing in fitted ATM total variance, smile shape held fixed; (d) the forward-price sanity check passes to a tight tolerance when the synthetic ground-truth inputs are used directly with no market noise. (Depends on: T009, T007.)

**Checkpoint**: Pipeline correctness is proven against ground truth. User Story 1's live-data results can now be trusted, per constitution Principle III.

---

## Phase 5: User Story 3 - Inspect smile shape and export data for plotting (Priority: P2)

**Goal**: From a completed report, get the desk-convention smile-shape numbers and the data series needed to plot the fit and the density.

**Independent Test**: From a completed extraction report, retrieve the smile-shape metrics and the plotting data series, and confirm both are present and internally consistent with the fitted smile in the same report.

- [ ] T023 [P] [US3] Implement `breeden_litzenberger/core/smile_metrics.py`: `SmileShapeMetrics` dataclass (`atm_implied_vol`, `risk_reversal_25d`, `butterfly_25d`); locate each 25-delta strike via `scipy.optimize.brentq` solving for BS delta = ±0.25 bracketed within the fitted smile's calibrated log-moneyness domain (research.md §8), then compute `atm_implied_vol`, `risk_reversal_25d` (25-delta call vol − 25-delta put vol), and `butterfly_25d` ((25-delta call vol + 25-delta put vol)/2 − ATM vol) from a `FittedSmile` (FR-012)
- [ ] T024 [P] [US3] Implement `tests/test_smile_metrics.py`: verify ATM vol / risk-reversal / butterfly values against hand-computed references for known SVI parameter sets (e.g. a symmetric smile should yield `risk_reversal_25d ≈ 0`)
- [ ] T025 [US3] Wire `SmileShapeMetrics` into `extract()` and `ExtractionReport` in `breeden_litzenberger/report.py` (FR-013). (Depends on: T017, T023.)
- [ ] T026 [US3] Implement `ExtractionReport.plot_data() -> PlotData` in `breeden_litzenberger/report.py` per contracts/public_api.md: retained `ImpliedVolPoint` strikes/market IVs, a dense fitted-smile strike/IV curve, `DensityGrid` strikes/values, and a lognormal benchmark density at the same forward price and ATM vol (FR-014c). (Depends on: T017, T025.)
- [ ] T027 [P] [US3] Implement `breeden_litzenberger/plotting.py`: matplotlib helpers consuming `PlotData` to render the fitted-smile-vs-market-IV chart and the extracted-density-vs-lognormal-benchmark chart (spec Assumptions — caller convenience, not part of the required contract). (Depends on: T026.)
- [ ] T028 [US3] Implement `tests/test_smile_metrics_and_plotting.py`: from a completed fixture `ExtractionReport`, assert `smile_shape_metrics` returns numeric ATM-vol/risk-reversal/butterfly values and `plot_data()` returns internally consistent series (spec US3 Acceptance Scenarios 1–2). (Depends on: T026.)

**Checkpoint**: All three user stories are independently functional.

---

## Phase 6: Polish & Cross-Cutting Concerns

- [ ] T029 [P] Write `docs/theory.md`: derive, in the author's own words, with formulas in LaTeX-style fenced code blocks followed by a plain-language explanation of every symbol — (a) why raw price differentiation is numerically unstable and what SVI buys instead, (b) the Breeden-Litzenberger formula and its convexity argument, (c) the raw SVI parameterization and the Gatheral-Jacquier no-butterfly-arbitrage condition, (d) the American-vs-European / OTM-only approximation and why it's necessary with yfinance data specifically (constitution Principles I, II, VIII)
- [ ] T030 [P] Write `docs/usage.md`: install instructions, an `extract()` quick example, and CLI usage, referencing quickstart.md's scenarios
- [ ] T031 [P] Write `README.md`: project overview, install command, quickstart snippet, link to `docs/theory.md`
- [ ] T032 Implement `breeden_litzenberger/cli.py`: `typer` command `bl extract --ticker --expiration [--rate] [--dividend-yield] [--min-open-interest] [--min-volume] [--format text|json]` wrapping `extract()` per contracts/public_api.md's CLI contract; non-zero exit code and a stderr message only on a raised exception, exit 0 regardless of `Verdict` value otherwise. (Depends on: T017, T025, T026.)
- [ ] T033 [P] Cross-check every formula-bearing module's docstring (`core/black_scholes.py`, `core/smile.py`, `core/density.py`) against `docs/theory.md` for citation consistency (constitution Principle I). (Depends on: T029.)
- [ ] T034 Run `mypy --strict` across `breeden_litzenberger/` and fix every reported error (constitution: API & Engineering Standards)
- [ ] T035 Execute quickstart.md Scenarios 1–6 end-to-end and fix any discrepancies found
- [ ] T036 Tag `v0.1.0` per constitution Git & Release Workflow once all phases pass

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies — start immediately.
- **Foundational (Phase 2)**: Depends on Setup. **Blocks all user stories.**
- **User Story 1 (Phase 3)**: Depends on Foundational (needs `core/black_scholes.py`, `core/smile.py`, `core/density.py`, `core/moments.py`). No dependency on US2 or US3.
- **User Story 2 (Phase 4)**: Depends on Foundational only (`core/black_scholes.py`, `core/smile.py`, `core/density.py`) — does **not** depend on US1's `data/` layer, since it uses synthetic prices. Can run in parallel with Phase 3.
- **User Story 3 (Phase 5)**: Depends on Foundational (`core/smile.py`) and on US1's `report.py` orchestration (T017) to have somewhere to wire `SmileShapeMetrics`/`plot_data()` into. Effectively starts after T017.
- **Polish (Phase 6)**: Depends on the user stories it touches (T032 needs T017/T025/T026; T029/T033 are independent of code and can start once Foundational's formulas are stable).

### Within Each Phase

- Foundational: `black_scholes.py` (T005) and `smile.py` (T007) are independent of each other and parallelizable; `density.py` (T009) depends on `smile.py`'s `FittedSmile`; `moments.py` (T010) is independent.
- US1: `yfinance_loader.py` (T012) and `rates.py` (T013) are parallelizable; cleaning (T014) and T-validation (T015) both depend on T012; `extract()` (T017) depends on everything before it in the phase.
- US2: the synthetic-chain helper (T020) must exist before the test that uses it (T021); property-based tests (T022) only need Foundational.
- US3: `smile_metrics.py` (T023) is independent of US1; wiring it in (T025) and `plot_data()` (T026) both need US1's `extract()` (T017).

### Parallel Opportunities

- Setup: T003, T004 in parallel after T001/T002.
- Foundational: T005, T006, T007, T008, T010, T011 can all run in parallel (T009 waits on T007).
- US1: T012, T013 in parallel; T018, T019 in parallel once T017 lands.
- US2: T022 can run in parallel with T020/T021 (different files, same Foundational dependency only).
- Phase 3 (US1) and Phase 4 (US2) can be staffed in parallel once Foundational is done — neither depends on the other.
- Polish: T029, T030, T031, T033 in parallel; T032 depends on US1/US3 code.

---

## Parallel Example: Foundational Phase

```bash
# After Setup completes, launch the independent foundational modules together:
Task: "Implement core/black_scholes.py per T005"
Task: "Implement core/smile.py per T007"
Task: "Implement core/moments.py per T010"
# core/density.py (T009) starts once core/smile.py (T007) lands.
```

## Parallel Example: User Story 1

```bash
Task: "Implement data/yfinance_loader.py per T012"
Task: "Implement data/rates.py per T013"
# T014 (cleaning) and T015 (T validation) start once T012 lands.
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup.
2. Complete Phase 2: Foundational — this alone is what proves the math is right; do not skip T005–T011 to get to US1 faster.
3. Complete Phase 3: User Story 1.
4. **STOP and VALIDATE**: run `tests/test_report_integration.py`; manually run quickstart.md Scenario 2 against a real ticker.
5. This is a usable MVP: `extract("SPY", date(...))` returns a full report.

### Incremental Delivery

1. Setup + Foundational → foundation ready, math pipeline unit-tested.
2. Add User Story 1 → MVP: live extraction works end-to-end.
3. Add User Story 2 → the MVP's correctness is now proven against ground truth (constitution Principle III) — do this before trusting US1's output on anything real.
4. Add User Story 3 → smile-shape metrics and plot data available.
5. Polish → docs/theory.md (the portfolio centerpiece), CLI, README, static checks, tag `v0.1.0`.

### Parallel Team Strategy

Once Foundational (Phase 2) is done:
- Developer A: User Story 1 (Phase 3)
- Developer B: User Story 2 (Phase 4) — independent of A, since it only needs `core/`
- Developer C: starts User Story 3's `smile_metrics.py` (T023/T024, independent of US1), then waits on Developer A's T017 to do T025/T026

---

## Notes

- [P] tasks touch different files and have no incomplete-task dependency.
- Every dataclass field constraint quoted in a task description above is taken verbatim from `data-model.md`; do not relax them during implementation.
- `core/` must never import `pandas` or `yfinance` (plan.md Structure Decision) — if a Foundational or US2/US3 task seems to need either, that is a design smell, not a shortcut to take.
- Commit after each task or logical group, on branch `001-rnd-extraction`, using Conventional Commits (constitution: Git & Release Workflow).
- Stop at any checkpoint to validate a story independently before moving on.
