"""`bl` CLI: a thin wrapper over breeden_litzenberger.extract() -- not the
primary integration path (plan.md Structure Decision); see
contracts/public_api.md for the CLI contract.
"""
from __future__ import annotations

import json
from datetime import date

import typer

from breeden_litzenberger import extract as extract_rnd
from breeden_litzenberger.report import LiquidityFloor

app = typer.Typer()


@app.callback()
def _callback() -> None:
    """breeden_litzenberger: market-implied risk-neutral density extraction."""


@app.command()
def extract(
    ticker: str = typer.Option(..., "--ticker"),
    expiration: str = typer.Option(..., "--expiration", help="YYYY-MM-DD"),
    rate: float | None = typer.Option(None, "--rate", help="Override the default risk-free rate"),
    dividend_yield: float | None = typer.Option(None, "--dividend-yield", help="Override the default dividend yield"),
    min_open_interest: int = typer.Option(0, "--min-open-interest"),
    min_volume: int = typer.Option(0, "--min-volume"),
    output_format: str = typer.Option("text", "--format", help="text|json"),
) -> None:
    """Extract a risk-neutral density report for TICKER at EXPIRATION."""
    expiration_date = date.fromisoformat(expiration)
    floor = LiquidityFloor(min_open_interest=min_open_interest, min_volume=min_volume)

    try:
        report = extract_rnd(
            ticker,
            expiration_date,
            risk_free_rate=rate,
            dividend_yield=dividend_yield,
            liquidity_floor=floor,
        )
    except Exception as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    if output_format == "json":
        typer.echo(json.dumps(report.to_dict(), indent=2))
    else:
        typer.echo(report.summary())


if __name__ == "__main__":
    app()
