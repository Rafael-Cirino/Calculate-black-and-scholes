import math

import numpy as np

from black_scholes.pricing import Greeks, option_price


def test_call_option_price_is_positive():
    price = option_price(
        S=100.0, K=100.0, T=1.0, r=0.05, q=0.0, volatility=0.2, option="call"
    )
    assert price > 0
    assert math.isfinite(price)


def test_put_option_price_is_positive():
    price = option_price(
        S=100.0, K=100.0, T=1.0, r=0.05, q=0.0, volatility=0.2, option="put"
    )
    assert price > 0
    assert math.isfinite(price)


def test_delta_and_gamma_are_finite_for_call():
    gs = Greeks(S=100.0, K=100.0, T=1.0, r=0.05, volatility=0.2, option="call")
    values = gs.calculate_all()
    assert all(math.isfinite(v) for v in values.values())
    assert np.isfinite(values["delta"])
    assert np.isfinite(values["gamma"])


def test_implied_volatility_matches_price():
    gs = Greeks(S=100.0, K=100.0, T=1.0, r=0.05, volatility=0.2, option="call")
    market_price = gs.option_price(0.2)
    implied = gs.implied_volatility(market_price, volatility_init=0.1)
    assert math.isfinite(implied)
    assert 0.01 <= implied <= 1.0
