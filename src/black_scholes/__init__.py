"""Black-Scholes option analytics package."""

from .pricing import Greeks, option_price

__all__ = ["Greeks", "option_price"]
