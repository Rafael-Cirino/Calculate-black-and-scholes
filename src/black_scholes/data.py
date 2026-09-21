from __future__ import annotations

from pathlib import Path

import polars as pl

from black_scholes.pricing import Greeks


def load_option_csv(path: str | Path) -> pl.DataFrame:
    """Load the option dataset and normalize a few column names used by the project."""
    df = pl.read_csv(str(path))
    renamed = {
        "best_bid_amount": "bid_amount",
        "best_ask_amount": "ask_amount",
        "index_price": "underlying_avg",
    }
    return df.rename(renamed)


def enrich_option_frame(df: pl.DataFrame) -> pl.DataFrame:
    """Add bid/ask values derived from the raw option prices."""
    if "best_bid_price" in df.columns and "best_ask_price" in df.columns:
        df = df.with_columns(
            (pl.col("best_bid_price") * pl.col("underlying_avg")).alias("bid"),
            (pl.col("best_ask_price") * pl.col("underlying_avg")).alias("ask"),
        )
    return df


def summarize_greeks(df: pl.DataFrame, option: str) -> pl.DataFrame:
    """Return a small summary table for the selected option type."""
    kind = option.upper()
    subset = df.filter(pl.col("kind") == kind)
    return subset.select(
        [
            "kind",
            "strike",
            "underlying_avg",
            "dte",
            "bid",
            "ask",
        ]
    )


def calc_implied_volatility_and_greeks(df: pl.DataFrame, option: str) -> pl.DataFrame:
    """Solve implied volatility per row and attach it plus all Greeks for the selected option side."""
    kind_letter = option.upper()
    option_name = "call" if kind_letter == "C" else "put"

    subset = df.filter(pl.col("kind") == kind_letter).with_columns(
        ((pl.col("bid") + pl.col("ask")) / 2).alias("avg_prize")
    )

    rows = subset.to_dicts()
    vol_init = 0.5
    ivs, deltas, vegas, thetas, gammas, rhos = [], [], [], [], [], []

    for row in rows:
        gr = Greeks(
            row["underlying_avg"],
            row["strike"],
            (row["dte"] + 0.5) / 366,
            0.0,
            vol_init,
            option=option_name,
        )
        iv = gr.implied_volatility(row["avg_prize"], vol_init)
        vol_init = iv

        greeks_at_iv = Greeks(
            row["underlying_avg"],
            row["strike"],
            (row["dte"] + 1) / 366,
            0.0,
            iv,
            option=option_name,
        ).calculate_all()

        ivs.append(round(iv * 100, 2))
        deltas.append(greeks_at_iv["delta"])
        vegas.append(greeks_at_iv["vega"] / 100)
        thetas.append(greeks_at_iv["theta"] / 4)
        gammas.append(greeks_at_iv["gamma"])
        rhos.append(greeks_at_iv["rho"] * 100)

    return subset.with_columns(
        [
            pl.Series("implied_vol", ivs),
            pl.Series("delta", deltas),
            pl.Series("vega", vegas),
            pl.Series("theta", thetas),
            pl.Series("gamma", gammas),
            pl.Series("rho", rhos),
        ]
    )
