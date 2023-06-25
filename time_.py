import time

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
from scipy.stats import norm

from utils.config import *
from utils.HHW_AES import GeneratePathsHestonHW_AES  # almost exact simulation
from utils.HHW_CHF import ChFH1HWModel, Chi_Psi  # characteristic function
from utils.HHW_MC import HHW_Euler  # standard euler mode


def OptionPriceFromMonteCarlo(CP, S, K, M):
    """
    S is a vector of Monte Carlo samples at T
    """
    result = np.zeros([len(K), 1])
    if CP == OptionType.CALL:
        for idx, k in enumerate(K):
            result[idx] = np.mean(1.0 / M * np.maximum(S - k, 0.0))
    elif CP == OptionType.PUT:
        for idx, k in enumerate(K):
            result[idx] = np.mean(1.0 / M * np.maximum(k - S, 0.0))
    return result


def CallPutCoefficients(CP, a, b, k):
    """
    Determine coefficients chi and psi for call/put prices
    """
    if CP == OptionType.CALL:
        c = 0.0
        d = b
        coef = Chi_Psi(a, b, c, d, k)
        Chi_k = coef["chi"]
        Psi_k = coef["psi"]
        if a < b and b < 0.0:
            H_k = np.zeros([len(k), 1])
        else:
            H_k = 2.0 / (b - a) * (Chi_k - Psi_k)
    elif CP == OptionType.PUT:
        c = a
        d = 0.0
        coef = Chi_Psi(a, b, c, d, k)
        Chi_k = coef["chi"]
        Psi_k = coef["psi"]
        H_k = 2.0 / (b - a) * (-Chi_k + Psi_k)
    return H_k


def OptionPriceFromCOS(cf, CP, S0, tau, K, N, L, P0T):
    """
    Calculate the price for the list of strikes provided with the COS method.
    Parameters
    ----------
    cf : lambda function
        Characteristic function.
    CP : class option type
        CALL for call and PUT for put.
    S0 : float
        Initial stock price.
    tau : float
        Time to maturity.
    K : ndarray of float
        Array of strikes.
    N : int
        Number of expansion terms.
    L : int
        Size of truncation domain (L=8 or L=10).
    P0T : lambda function
        Zero-coupon bond for maturity T.

    Returns
    -------
    value : ndarray of float
        Array of prices.
    """
    if K is not np.array:
        K = np.array(K).reshape([len(K), 1])
    i = complex(0.0, 1.0)
    x0 = np.log(S0 / K)
    # Truncation domain
    a = -L * np.sqrt(tau)
    b = +L * np.sqrt(tau)
    # Summation from k = 0 to k=N-1
    k = np.linspace(0, N - 1, N).reshape([N, 1])
    u = k * np.pi / (b - a)
    U_k = CallPutCoefficients(OptionType.PUT, a, b, k)
    mat = np.exp(i * np.outer((x0 - a), u))
    temp = cf(u) * U_k
    temp[0] = 0.5 * temp[0]
    value = K * np.real(mat.dot(temp))
    if CP == OptionType.CALL:
        # Put-call parity for call options
        value = value + S0 - K * P0T
    return value


def BS(CP, S_0: float, K: float, sigma: float, t, T: float, r: float) -> float:
    """ Black-Scholes formula for European options
    Parameters
    ----------
    CP : class option type
        CALL for call and PUT for put.
    S_0 : float
        Initial stock price.
    K : ndarray of float
        Array of strikes.
    sigma : float
        volatility of the stock price (constant)
    t : float
        time
    T : float
        time to maturity
    r : float
        risk-free rate of interest (constant)

    Returns
    -------
    value : ndarray of float
        Array of prices.
    """
    if K is np.array:
        K = np.array(K).reshape([len(K), 1])
    d1 = (np.log(S_0 / K) + (r + 0.5 * sigma**2.0) * (T - t)) / (
        sigma * np.sqrt(T - t)
    )
    d2 = d1 - sigma * np.sqrt(T - t)
    if CP == OptionType.CALL:
        value = st.norm.cdf(d1) * S_0 - st.norm.cdf(d2) * \
            K * np.exp(-r * (T - t))
    elif CP == OptionType.PUT:
        value = st.norm.cdf(-d2) * K * np.exp(-r * (T - t)
                                              ) - st.norm.cdf(-d1) * S_0
    return value


if __name__ == "__main__":

    set_params = (P0T, T, kappa, gamma, rhoxr, rhoxv, vbar, v0, lambd, eta)

    np.random.seed(1)
    time_euler = []
    for i in range(1000):
        print(i)
        init = time.time()
        paths, _ = HHW_Euler(NPaths, NSteps, S0, set_params)
        S_n = paths["S"]
        M_t_n = paths["M_t"]
        valueOptMC = OptionPriceFromMonteCarlo(CP, S_n[:, -1], K, M_t_n[:, -1])
        end = time.time()
        time_euler.append(end-init)


    time_COS = []
    for i in range(1000):
        init = time.time()
        cf2 = ChFH1HWModel(set_params)
        valCOS = OptionPriceFromCOS(cf2, CP, S0, T, K, N, L, P0T(T))
        end = time.time()
        time_COS.append(end-init)



    print(np.mean(time_euler))
    print(np.std(time_euler))
    print(np.mean(time_COS))
    print(np.std(time_COS))



    # plt.figure()
    # plt.plot(K, valueOptMC, label="HHW - Euler")
    # plt.plot(K, valCOS, "--r", label="HHW - COS")
    # plt.legend()
    # plt.xlabel("Strike K")
    # plt.ylabel("Option Value")
    # plt.xticks(np.arange(min(K), max(K)+1, 10))
    # plt.grid()
    # plt.show()

