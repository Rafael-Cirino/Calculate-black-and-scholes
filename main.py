import numpy as np
from scipy.stats import norm
from scipy import optimize

N_prime = norm.pdf
N = norm.cdf

import pandas as pd
import numpy as np
import scipy.stats as ss

pd.options.mode.chained_assignment = None
np.warnings.filterwarnings("ignore")

NAME_CSV = "exemplo.zip"


def calc_vol(df_vol, option):
    """Calculate volatility of option and greeks

    Arguments:
        df_vol {Dataframe} -- Dataframe containing options and bid, ask, underlying, index_close, strike, kind and dte
        option {string} -- Kind of option to calculate

    Returns:
        Dataframe -- Dataframe containing option with greeks
    """
    kind = "call" if option == "C" else "put"

    df_vol = df_vol[(df_vol["kind"] == option)]
    df_vol["avg_prize"] = df_vol[["bid", "ask"]].mean(axis=1)
    line_df = len(df_vol)
    list_vol, count = [], 0
    vol_init = 0.5
    for i, row in df_vol.iterrows():
        gr = greeks(
            row["underlying_avg"],
            row["strike"],
            (row["dte"] + 0.5) / 366,
            0,
            vol_init,
            option=kind,
        )
        vol = gr.implied_volatility(row["avg_prize"], vol_init)

        list_vol.append(round(vol * 100, 2))
        vol_init = vol

        count += 1
        print(f"BS progress {kind}: {(count/line_df)*100:.1f} %", end="\r", flush=True)

    df_vol["new_mark_iv"] = pd.Series(list_vol).values
    print("")

    bs_greeks = greeks(
        df_vol["underlying_avg"],
        df_vol["strike"],
        (df_vol["dte"] + 1) / 366,
        0,
        df_vol["new_mark_iv"] / 100,
        option=kind,
    )
    greeks_value = bs_greeks.calculate_all()

    df_vol["delta"] = greeks_value["delta"]
    df_vol["vega"] = greeks_value["vega"] / 100
    df_vol["theta"] = greeks_value["theta"] / 4
    df_vol["gamma"] = greeks_value["gamma"]
    df_vol["rho"] = greeks_value["rho"] * 100

    if option == "P":
        df_vol["payoff_short"] = np.minimum(
            df_vol["index_close"] - df_vol["strike"], 0
        )  # put
    elif option == "C":
        df_vol["payoff_short"] = np.minimum(
            df_vol["strike"] - df_vol["index_close"], 0
        )  # call

    df_vol["pl_short"] = df_vol["payoff_short"] + df_vol["bid"]

    return df_vol


def option_price(S, K, T, r, q, volatility, option="call"):
    """Option price with Black Scholes
    Params:
        S: spot price
        K: strike price
        T: time to maturity
        r: interest rate
        q: rate of continuous dividend paying asset
        volatility(decimal): volatility of underlying asset
    """

    d1 = (np.log(S / K) + (r - q + 0.5 * volatility**2) * T) / (
        volatility * np.sqrt(T)
    )
    d2 = (np.log(S / K) + (r - q - 0.5 * volatility**2) * T) / (
        volatility * np.sqrt(T)
    )

    if option.lower() in ["call", "callvanilla"]:
        price = S * np.exp(-q * T) * ss.norm.cdf(d1) - K * np.exp(-r * T) * ss.norm.cdf(
            d2
        )
    elif option.lower() in ["put", "putvanilla"]:
        price = K * np.exp(-r * T) * ss.norm.cdf(-d2) - S * np.exp(
            -q * T
        ) * ss.norm.cdf(-d1)

    return price


class greeks:
    def __init__(self, S, K, T, r, volatility, option=""):
        """init greeks

        Arguments:
            S {float} -- spot price
            K {float} -- strike price
            T {float} -- time to maturity
            r {float} -- interest rate
            volatility {float} -- volatility of underlying asset (decimal)
            option {str} -- option type
        """

        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.q = 0
        self.volatility = volatility
        self.option = option

        self.d1 = (np.log(S / K) + (r + 0.5 * volatility**2) * T) / (
            volatility * np.sqrt(T)
        )
        self.d2 = (np.log(S / K) + (r - 0.5 * volatility**2) * T) / (
            volatility * np.sqrt(T)
        )

    def option_price(self, volatility):
        """Option price with Black Scholes
        Params:
            S: spot price
            K: strike price
            T: time to maturity
            r: interest rate
            q: rate of continuous dividend paying asset
            volatility(decimal): volatility of underlying asset
        """

        d1 = (
            np.log(self.S / self.K) + (self.r - self.q + 0.5 * volatility**2) * self.T
        ) / (volatility * np.sqrt(self.T))
        d2 = (
            np.log(self.S / self.K) + (self.r - self.q - 0.5 * volatility**2) * self.T
        ) / (volatility * np.sqrt(self.T))

        if self.option.lower() in ["call", "callvanilla"]:
            price = self.S * np.exp(-self.q * self.T) * ss.norm.cdf(
                d1
            ) - self.K * np.exp(-self.r * self.T) * ss.norm.cdf(d2)
        elif self.option.lower() in ["put", "putvanilla"]:
            price = self.K * np.exp(-self.r * self.T) * ss.norm.cdf(
                -d2
            ) - self.S * np.exp(-self.q * self.T) * ss.norm.cdf(-d1)

        return price

    def delta(self):
        """Calculate greeks delta with black and scholes - Average uncertainty = 0.5%

        Returns:
            float -- greeks delta
        """

        sign = -1 if self.option == "put" else 1
        return -1 * sign * ss.norm.cdf(sign * self.d1)

    def vega(self):
        """Calculate greeks vega with black and scholes - Average uncertainty = 3.94%

        Returns:
            float -- greeks vega
        """

        return self.S * norm.pdf(self.d1) * np.sqrt(self.T)

    def theta(self):
        """Calculate greeks theta with black and scholes - Average uncertainty = 10.87%

        Returns:
            float -- greeks theta
        """
        if self.option == "call":
            return 0.01 * (
                -(self.S * norm.pdf(self.d1) * self.volatility) / (2 * np.sqrt(self.T))
                - self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(self.d2)
            )
        elif self.option == "put":
            return 0.01 * (
                -(self.S * norm.pdf(self.d1) * self.volatility) / (2 * np.sqrt(self.T))
                + self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(-self.d2)
            )

    def gamma(self):
        """Calculate greeks gamma with black and scholes - Average uncertainty = 0.9%

        Returns:
            float -- greeks gamma
        """

        return norm.pdf(self.d1) / (self.S * self.volatility * np.sqrt(self.T))

    def rho(self):
        """Calculate greeks rho with black and scholes

        Returns:
            float -- greeks rho
        """

        if self.option == "call":
            return 0.01 * (
                self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(self.d2)
            )
        elif self.option == "put":
            return 0.01 * (
                -self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(-self.d2)
            )

    def calculate_all(self):
        """Calculate all greeks

        Returns:
            dict -- Dictionary containing all the greeks
        """
        dict_return = {}

        dict_return["delta"] = self.delta()
        dict_return["vega"] = self.vega()
        dict_return["theta"] = self.theta()
        dict_return["gamma"] = self.gamma()
        dict_return["rho"] = self.rho()

        return dict_return

    def mark_iv(self, x):
        """Calculate market iv

        Arguments:
            x {float} -- Estimated implied volatility  (decimal)

        Returns:
            float -- Implied volatility  (decimal)
        """
        diff = self.option_price(x) - self.C

        return diff / greeks(self.S, self.K, self.T, self.r, x, self.option).vega()

    def secant(self, C, a, b):
        """Calculate mark_iv by secant method

        Arguments:
            C {float} -- Observed call price
            a {float} -- More value possible
            b {float} -- Less value possible

        Returns:
            float -- Implied volatility  (decimal)
        """

        self.C = C
        volatility = optimize.brentq(
            self.mark_iv, a, b, disp=False, xtol=0.005, maxiter=30
        )

        return volatility

    def method_newton(self, x, tol=0.005, max_iterations=20):
        """Calculate mark_iv by newton method

        Arguments:
            x {float} -- Estimated implied volatility  (decimal)

        Keyword Arguments:
            tol {float} -- error tolerance in result (default: {0.0001})
            max_iterations {int} -- max iterations to update vol (default: {100})

        Returns:
            float -- Implied volatility  (decimal)
        """
        for i in range(max_iterations):
            ### calculate difference between blackscholes price and market price with
            ### iteratively updated volality estimate
            diff = self.option_price(x) - self.C

            ###break if difference is less than specified tolerance level
            if abs(diff) < tol:
                break

            ### use newton rapshon to update the estimate
            x = (
                x
                - diff
                / greeks(self.S, self.K, self.T, self.r, x, option=self.option).vega()
            )

        return x

    def implied_volatility(self, C, volatility_init, tol=0.005, max_iterations=20):
        """Calculate implied volatility - Average uncertainty = 2.3%

        Arguments:
            C {float} -- Observed price
            volatility_init {float} -- Estimated initial volatility

        Keyword Arguments:
            tol {float} -- error tolerance in result (default: {0.0001})
            max_iterations {int} -- max iterations to update vol (default: {100})

        Returns:
            float -- Implied volatility  (decimal)
        """
        self.C = C

        if np.isnan(volatility_init) or np.isinf(volatility_init):
            volatility_init = 1
            vol_more = volatility_init
            vol_less = volatility_init
        else:
            vol_more = volatility_init
            vol_less = volatility_init

        while vol_more <= 2:
            volatility = vol_more
            # volatility = self.method_newton(volatility)
            volatility = optimize.newton(
                self.mark_iv, vol_more, tol=tol, disp=False, maxiter=max_iterations
            )

            if np.isnan(volatility):
                vol_more += 0.2
            else:
                return volatility

        while vol_less > 0:
            volatility = vol_less
            # volatility = self.method_newton(volatility)
            volatility = optimize.newton(
                self.mark_iv, vol_less, tol=tol, disp=False, maxiter=max_iterations
            )

            if np.isnan(volatility):
                vol_less -= 0.2
            else:
                return volatility

        return self.secant(C, 2, 5)


if __name__ == "__main__":
    df = pd.read_csv(NAME_CSV)

    df.rename(
        columns={
            "best_bid_amount": "bid_amount",
            "best_ask_amount": "ask_amount",
            "index_price": "underlying_avg",
        },
        inplace=True,
    )

    df["bid"] = df["best_bid_price"] * df["underlying_avg"]
    df["ask"] = df["best_ask_price"] * df["underlying_avg"]

    for kind in ["C", "P"]:
        df_option = df[df["kind"] == kind]
        greeks_value = calc_vol(df, kind)

        for g in ["delta", "vega", "theta", "gamma"]:
            df_option[g] = greeks_value[g]

            print(
                f"{g}: ",
                abs(
                    round(
                        (
                            (
                                (abs(df_option[f"greeks_{g}"]) - abs(df_option[g]))
                                / df_option[f"{g}"]
                            ).mean()
                            * 100
                        ),
                        2,
                    )
                ),
            )

        print()
