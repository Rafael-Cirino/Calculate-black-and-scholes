from __future__ import annotations

from pathlib import Path

import polars as pl


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
