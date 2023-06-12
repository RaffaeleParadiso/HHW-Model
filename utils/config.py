import enum

import numpy as np


class OptionType(enum.Enum):
    CALL = 1.0
    PUT = -1.0


CP = OptionType.CALL  # Option type (CALL/PUT)

NPaths = 10000  # n° of paths for MC Euler and AES
NSteps = 999  # n° of time steps for MC Euler and AES
lambd = 0.05  # mean reversion rate velocity for the interest rate process
eta = 0.005  # volatility of the interest rate process
S0 = 100.0  # initial stock price
T = 0.5  # time until maturity (years)
r = 0.07  # initial interest rate

# K = np.linspace(.01,2*S0*np.exp(r*T),50)
K = np.arange(60, 141, 1)  # OTM/ATM/ITM options
K = np.array(K).reshape([len(K), 1])  # array of strike prices
S = np.arange(60, 140, 0.1)
S = np.array(S).reshape([len(S), 1])  # array of prices


def P0T(T): return np.exp(
    -r * T
)  # ZCB curve with maturity T (usually obtained from the market)


gamma = 0.0571  # volatility of the volatility process
vbar = 0.0398  # long-run mean value for the volatility process
v0 = 0.0175  # initial volatility value
rhoxr = 0.2  # correlation between log stocks and interest rate
rhoxv = -0.5711  # correlation between log stocks and volatility
kappa = 1.5768  # mean reversion rate velocity for the volatility process

# Settings for the COS method
# n° of contribute (N terms) in the cosine expansion for the characteristic function
N = 150000
L = 8  # truncation domain [-L*\sqrt(\tau), +L*\sqrt(\tau)]  (\tau = T - t_0)
