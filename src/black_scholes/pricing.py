from __future__ import annotations

import numpy as np
from scipy import optimize
from scipy.stats import norm

N = norm.cdf
N_prime = norm.pdf


def option_price(S, K, T, r, q, volatility, option="call"):
    """Price a vanilla option using Black-Scholes."""
    if T <= 0:
        raise ValueError("Time to maturity T must be greater than zero.")
    if volatility <= 0:
        raise ValueError("Volatility must be greater than zero.")

    d1 = (np.log(S / K) + (r - q + 0.5 * volatility**2) * T) / (volatility * np.sqrt(T))
    d2 = (np.log(S / K) + (r - q - 0.5 * volatility**2) * T) / (volatility * np.sqrt(T))

    option_name = option.lower()
    if option_name in {"call", "callvanilla"}:
        return S * np.exp(-q * T) * N(d1) - K * np.exp(-r * T) * N(d2)
    if option_name in {"put", "putvanilla"}:
        return K * np.exp(-r * T) * N(-d2) - S * np.exp(-q * T) * N(-d1)
    raise ValueError(f"Unsupported option type: {option!r}")


class Greeks:
    """Black-Scholes Greeks and implied-volatility helpers."""

    def __init__(self, S, K, T, r, volatility, option=""):
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.q = 0.0
        self.volatility = volatility
        self.option = option.lower() if isinstance(option, str) else ""

        self.d1 = (
            np.log(self.S / self.K)
            + (self.r - self.q + 0.5 * self.volatility**2) * self.T
        ) / (self.volatility * np.sqrt(self.T))
        self.d2 = (
            np.log(self.S / self.K)
            + (self.r - self.q - 0.5 * self.volatility**2) * self.T
        ) / (self.volatility * np.sqrt(self.T))

    def option_price(self, volatility):
        return option_price(
            S=self.S,
            K=self.K,
            T=self.T,
            r=self.r,
            q=self.q,
            volatility=volatility,
            option=self.option,
        )

    def delta(self):
        sign = -1 if self.option == "put" else 1
        return -1 * sign * N(sign * self.d1)

    def vega(self):
        return self.S * N_prime(self.d1) * np.sqrt(self.T)

    def theta(self):
        if self.option == "call":
            return 0.01 * (
                -(self.S * N_prime(self.d1) * self.volatility) / (2 * np.sqrt(self.T))
                - self.r * self.K * np.exp(-self.r * self.T) * N(self.d2)
            )
        if self.option == "put":
            return 0.01 * (
                -(self.S * N_prime(self.d1) * self.volatility) / (2 * np.sqrt(self.T))
                + self.r * self.K * np.exp(-self.r * self.T) * N(-self.d2)
            )
        raise ValueError(f"Unsupported option type: {self.option!r}")

    def gamma(self):
        return N_prime(self.d1) / (self.S * self.volatility * np.sqrt(self.T))

    def rho(self):
        if self.option == "call":
            return 0.01 * (self.K * self.T * np.exp(-self.r * self.T) * N(self.d2))
        if self.option == "put":
            return 0.01 * (-self.K * self.T * np.exp(-self.r * self.T) * N(-self.d2))
        raise ValueError(f"Unsupported option type: {self.option!r}")

    def calculate_all(self):
        return {
            "delta": self.delta(),
            "vega": self.vega(),
            "theta": self.theta(),
            "gamma": self.gamma(),
            "rho": self.rho(),
        }

    def mark_iv(self, x):
        diff = self.option_price(x) - self.C
        return diff / Greeks(self.S, self.K, self.T, self.r, x, self.option).vega()

    def secant(self, C, a, b):
        self.C = C
        return optimize.brentq(self.mark_iv, a, b, xtol=0.005, maxiter=30)

    def method_newton(self, x, tol=0.005, max_iterations=20):
        for _ in range(max_iterations):
            diff = self.option_price(x) - self.C
            if abs(diff) < tol:
                break
            x = (
                x
                - diff
                / Greeks(self.S, self.K, self.T, self.r, x, option=self.option).vega()
            )
        return x

    def implied_volatility(self, C, volatility_init, tol=0.005, max_iterations=20):
        self.C = C

        if np.isnan(volatility_init) or np.isinf(volatility_init):
            volatility_init = 1.0

        vol_more = volatility_init
        vol_less = volatility_init

        while vol_more <= 2.0:
            volatility = optimize.newton(
                self.mark_iv,
                vol_more,
                tol=tol,
                disp=False,
                maxiter=max_iterations,
            )
            if not np.isnan(volatility):
                return float(volatility)
            vol_more += 0.2

        while vol_less > 0.0:
            volatility = optimize.newton(
                self.mark_iv,
                vol_less,
                tol=tol,
                disp=False,
                maxiter=max_iterations,
            )
            if not np.isnan(volatility):
                return float(volatility)
            vol_less -= 0.2

        return float(self.secant(C, 2.0, 5.0))
