from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from black_scholes.data import enrich_option_frame, load_option_csv, summarize_greeks
from black_scholes.pricing import Greeks, option_price

app = typer.Typer(help="Black-Scholes option analytics for option datasets.")
console = Console()


@app.command()
def example() -> None:
    """Print a quick Black-Scholes example using sample values."""
    price = option_price(
        S=100.0,
        K=100.0,
        T=1.0,
        r=0.05,
        q=0.0,
        volatility=0.2,
        option="call",
    )
    greeks = Greeks(S=100.0, K=100.0, T=1.0, r=0.05, volatility=0.2, option="call")
    console.print(f"Call option price: {price:.6f}")
    console.print(f"Greeks: {greeks.calculate_all()}")


@app.command()
def analyze(
    csv_path: Path, option: str = typer.Option("C", help="Option side: C or P")
) -> None:
    """Load a CSV file, enrich the option data, and summarize the selected option side."""
    df = load_option_csv(csv_path)
    df = enrich_option_frame(df)
    summary = summarize_greeks(df, option)
    console.print(summary.head(10))


@app.command()
def price(
    s: float = typer.Option(100.0, help="Spot price"),
    k: float = typer.Option(100.0, help="Strike"),
    t: float = typer.Option(1.0, help="Time to maturity"),
    r: float = typer.Option(0.05, help="Risk-free rate"),
    q: float = typer.Option(0.0, help="Dividend yield"),
    volatility: float = typer.Option(0.2, help="Volatility"),
    option: str = typer.Option("call", help="Option type: call or put"),
) -> None:
    """Compute the option price directly from the Black-Scholes model."""
    value = option_price(S=s, K=k, T=t, r=r, q=q, volatility=volatility, option=option)
    console.print(f"{option.title()} price: {value:.6f}")


if __name__ == "__main__":
    app()
