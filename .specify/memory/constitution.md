<!--
Sync Impact Report
- Version change: TEMPLATE (unratified) → 1.0.0
- Modified principles: n/a (initial ratification)
  - Template's 5 generic placeholder principles replaced with 8 concrete,
    project-specific principles (Cited Provenance; Smooth-Then-Differentiate;
    Ground-Truth Validation; No Lookahead Bias; Determinism; Arbitrage-Awareness;
    Numerical Stability Over Cleverness; Documentation Explains Why Not Just What)
- Added sections:
  - "API & Engineering Standards" (template SECTION_2 slot)
  - "Git & Release Workflow" (template SECTION_3 slot)
- Removed sections: none
- Deferred / TODO placeholders: none
- Templates requiring follow-up: none reviewed as part of this command (out of
  scope per constitution command's scope guard — dependent templates/commands
  read this file at runtime and are not modified here)
-->

# Breeden-Litzenberger RND Extraction Constitution

## Core Principles

### I. Cited Provenance for Every Formula
Every formula implemented MUST be traceable to a named, cited academic source, both
in the implementing module's docstring and in `docs/theory.md`. Primary references:
Breeden & Litzenberger, "Prices of State-Contingent Claims Implicit in Option Prices"
(Journal of Business, 1978); Black & Scholes (1973) for the pricing/implied-vol
layer; Gatheral, "A Parsimonious Arbitrage-Free Implied Volatility Parameterization"
(2004) for the raw SVI smile model; Gatheral & Jacquier, "Arbitrage-Free SVI
Volatility Surfaces" (2014) for the no-static-arbitrage conditions on the fitted
smile.
Rationale: this library's entire value is numerical correctness under a famously
ill-conditioned operation; an uncited formula cannot be checked against the
literature it claims to implement.

### II. Smooth-Then-Differentiate, Never Raw
The library MUST NOT differentiate raw, discrete market prices directly. Extraction
MUST always follow this order: (1) invert market prices to implied volatilities,
(2) fit a smooth, arbitrage-constrained SVI model to the implied volatility smile,
(3) convert the fitted smile back to a continuous call-price function via
Black-Scholes, (4) differentiate that smooth function per Breeden-Litzenberger. Any
deviation from this order (e.g. spline-smoothing raw prices instead of the vol
smile) MUST be justified in `docs/theory.md`, not merely implemented.
Rationale: differentiating a small set of discrete, noisy market prices twice is
numerically unstable; smoothing in vol space rather than price space is the
standard practitioner fix and must never be silently bypassed.

### III. Ground-Truth Validation Before Real Data
Before any real market data is touched, the pipeline MUST be validated against
synthetic option prices generated from a known closed-form distribution
(constant-vol Black-Scholes, i.e. a lognormal risk-neutral density). The full
pipeline (implied vol → smile fit → Breeden-Litzenberger extraction) MUST be run on
these synthetic prices, and the extracted density MUST be asserted to match the
known analytic lognormal density to tight tolerance.
Rationale: this is this project's equivalent of a reference-paper worked example —
the test that actually proves the pipeline is correct. Everything downstream is
regression coverage layered on top of it.

### IV. No Lookahead Bias
The library operates on a single fixed-timestamp snapshot of option quotes for one
underlying and one expiration. It MUST NOT mix quotes across timestamps and MUST
have no notion of "now" beyond what is explicitly passed in by the caller.

### V. Determinism
Identical inputs MUST produce bit-identical outputs. The SVI calibration optimizer
MUST take an explicit seed and a fixed default initial guess. No unseeded
randomness is permitted anywhere in the pipeline.

### VI. Arbitrage-Awareness as a First-Class Output
For any fitted smile, the library MUST be able to report — as diagnostics reported
alongside the result, never silently discarded or silently "fixed" without
flagging — whether the smile admits static arbitrage (calendar or butterfly),
whether the extracted density is non-negative everywhere, whether it integrates to
1, and whether its risk-neutral mean matches the theoretical forward price within
tolerance.

### VII. Numerical Stability Over Cleverness
Implied-vol inversion MUST use scipy.optimize primitives (Brent's method with sane
bracket bounds) rather than hand-rolled root-finders, and SVI calibration MUST
likewise prefer scipy.optimize primitives over hand-rolled optimizers. All SVI
fitting and no-arbitrage checks MUST work in log-forward-moneyness and
total-variance space — the standard practitioner convention that is what makes the
SVI no-arbitrage conditions checkable at all.

### VIII. Documentation Explains Why, Not Just What
Every module's docstrings MUST explain why a step exists — e.g. why the smoothing
step exists, why raw differentiation fails — not just what the code does. The
codebase is written to be read by someone reviewing it as a portfolio artifact, and
this WHY-focused documentation is the single most interview-relevant explanation in
the whole repo.

## API & Engineering Standards

The public API surface MUST remain small and fully typed. Internal helpers MUST be
underscore-prefixed and treated as private, non-public implementation detail.

## Git & Release Workflow

Each spec-kit feature MUST be developed on its own feature branch. Commit messages
MUST follow Conventional Commits. Each completed phase MUST be tagged
`v0.<phase>.0`.

## Governance

This constitution supersedes all other engineering practices and conventions in
this repository. Amendments require: (1) a documented rationale for the change,
(2) a version bump per the versioning policy below, and (3) a check of whether
dependent templates or commands that consult this constitution need to be updated
to stay consistent (tracked in each amendment's Sync Impact Report).

All PRs and code reviews MUST verify compliance with these principles. Principles
II (Smooth-Then-Differentiate), III (Ground-Truth Validation), V (Determinism), and
VI (Arbitrage-Awareness) are the numerical-correctness core of this project and are
treated as non-negotiable; any deviation from Principle II specifically MUST be
justified in `docs/theory.md` before merge.

**Versioning policy**: MAJOR — backward-incompatible governance or principle
removal/redefinition; MINOR — a new principle added or materially expanded
guidance; PATCH — clarifications, wording, or typo fixes that do not change
substantive meaning.

**Version**: 1.0.0 | **Ratified**: 2026-09-15 | **Last Amended**: 2026-09-15
