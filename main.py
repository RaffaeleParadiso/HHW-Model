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
    EULER = 1
    AES = 0
    COS = 1
    BLACK = 1

    FIGURE = 1
    SAVE = 1

    print("=" * 60)

    if EULER:
        set_params = (P0T, T, kappa, gamma, rhoxr, rhoxv, vbar, v0, lambd, eta)
        start = time.time()
        np.random.seed(1)
        paths = HHW_Euler(NPaths, NSteps, S0, set_params)
        S_n = paths["S"]
        M_t_n = paths["M_t"]
        valueOptMC = OptionPriceFromMonteCarlo(CP, S_n[:, -1], K, M_t_n[:, -1])
        if SAVE:
            np.savetxt(f"raw_data/MC_{T}_{CP}.txt", valueOptMC, fmt="%.4f")
        print(
            f"Time elapsed for Euler MC  : {(time.time() - start):.3f} seconds for {len(K)} strikes"
        )
        print(
            f"Price MC_Euler Martingala  : {(np.mean(S_n[:,-1]/M_t_n[:,-1])):.2f}")
        print("=" * 60)

    if AES:
        set_params = (P0T, T, kappa, gamma, rhoxr, rhoxv, vbar, v0, lambd, eta)
        start = time.time()
        np.random.seed(1)
        pathsExact = GeneratePathsHestonHW_AES(NPaths, NSteps, S0, set_params)
        S_ex = pathsExact["S"]
        M_t_ex = pathsExact["M_t"]
        valueOptMC_ex = OptionPriceFromMonteCarlo(
            CP, S_ex[:, -1], K, M_t_ex[:, -1])
        if SAVE:
            np.savetxt("AES.txt", valueOptMC_ex, fmt="%.4f")
        print(
            f"Time elapsed for AES MC    : {(time.time() - start):.3f} seconds for {len(K)} strikes"
        )
        print(
            f"Price MC_AES Martingala    : {(np.mean(S_ex[:,-1]/M_t_ex[:,-1])):.2f}")
        print("=" * 60)

    if COS:
        set_params = (P0T, T, kappa, gamma, rhoxr, rhoxv, vbar, v0, lambd, eta)
        start = time.time()
        cf2 = ChFH1HWModel(set_params)
        # u=np.array([1.0,2.0,3.0])
        # print(cf2(u))
        valCOS = OptionPriceFromCOS(cf2, CP, S0, T, K, N, L, P0T(T))
        if SAVE:
            np.savetxt(f"raw_data/COS_{T}_{CP}.txt", valCOS, fmt="%.4f")
        print(
            f"Time elapsed for COS Method: {(time.time() - start):.3f} seconds for {len(K)} strikes"
        )
        print("=" * 60)

    if BLACK:
        value_BS = BS(CP, S0, K, v0, 0, T, r)

    if FIGURE:
        plt.figure()
        plt.title(f"Option Price vs Strike Price for T={T} years", fontsize=10)
        if EULER:
            plt.plot(K, valueOptMC, label="HHW - Euler")
        if AES:
            plt.plot(K, valueOptMC_ex, ".k", label="HHW - AES")
        if COS:
            plt.plot(K, valCOS, "--r", label="HHW - COS")
        if BLACK:
            plt.plot(K, value_BS, "--g", label="Black-Scholes")
        plt.legend()
        plt.xlabel("Strike K")
        plt.ylabel("Option Value")
        plt.grid()
        if SAVE:
            plt.savefig(f"figure/HHW_{T}y_{CP}.png", bbox_inches="tight")
        plt.show()

    nx = 0
    if nx:
        K = 100
        r = 0.1
        T = 1
        sigma = 0.3
        N = norm.cdf

        def BS_CALL(S, K, T, r, sigma):
            d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
            return S * N(d1) - K * np.exp(-r*T) * N(d2)

        def BS_PUT(S, K, T, r, sigma):
            d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
            return K*np.exp(-r*T)*N(-d2) - S*N(-d1)

        S = np.arange(60, 140, 5)

        calls = [BS(OptionType.CALL, s, K, sigma, 0, T, r) for s in S]
        puts = [BS(OptionType.PUT, s, K, sigma, 0, T, r) for s in S]
        plt.figure()
        plt.plot(S, calls, label='Call Value')
        plt.plot(S, puts, label='Put Value')
        plt.xlabel('$S_0$')
        plt.ylabel(' Value')
        plt.legend()
        plt.show()

        variyng_S = []
        for element in S:
            set_params = (P0T, T, kappa, gamma, rhoxr, rhoxv, vbar, v0, lambd, eta)
            np.random.seed(1)
            paths = HHW_Euler(NPaths, NSteps, element, set_params)
            S_n = paths["S"]
            M_t_n = paths["M_t"]
            variyng_S.append(OptionPriceFromMonteCarlo(
                CP, S_n[:, -1], [100], M_t_n[:, -1]))

        variyng_S = np.array(variyng_S).reshape(len(S))

        def long_put(S, K, Price):
            # Long Put Payoff = max(Strike Price - Stock Price, 0)     # If we are long a call, we would only elect to call if the current stock price is less than     # the strike price on our option
            P = list(map(lambda x: max(K - x, 0) - Price, S))
            return P

        T4 = long_put(S, 100, 0)
        plt.figure()
        plt.plot(S, puts, label='Puts Value BS')
        plt.plot(S, variyng_S, label='Puts Value HHW')
        plt.plot(S, T4, label='Payoff')
        plt.xlabel('$S_0$')
        plt.ylabel(' Value')
        plt.legend()
        plt.show()

        plt.style.use('ggplot')
        plt.rcParams['xtick.labelsize'] = 14
        plt.rcParams['ytick.labelsize'] = 14
        plt.rcParams['figure.titlesize'] = 18
        plt.rcParams['figure.titleweight'] = 'medium'
        plt.rcParams['lines.linewidth'] = 2.5

        # S = stock underlying # K = strike price # Price = premium paid for option
        def long_call(S, K, Price):
            # Long Call Payoff = max(Stock Price - Strike Price, 0)     # If we are long a call, we would only elect to call if the current stock price is greater than     # the strike price on our option
            P = list(map(lambda x: max(x - K, 0) - Price, S))
            return P

        def long_put(S, K, Price):
            # Long Put Payoff = max(Strike Price - Stock Price, 0)     # If we are long a call, we would only elect to call if the current stock price is less than     # the strike price on our option
            P = list(map(lambda x: max(K - x, 0) - Price, S))
            return P

        S = [t/5 for t in range(0, 1000)]  # Define some series of stock-prices
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 10))
        fig.suptitle('Payoff Functions for Long/Short Put/Calls',
                    fontsize=20, fontweight='bold')
        fig.text(0.5, 0.04, 'Stock/Underlying Price ($)',
                ha='center', fontsize=14, fontweight='bold')
        fig.text(0.08, 0.5, 'Option Payoff ($)', va='center',
                rotation='vertical', fontsize=14, fontweight='bold')

        lc_P = long_call(S, 100, 10)
        lp_P = long_put(S, 100, 10)
        plt.subplot(1, 2, 1)
        plt.plot(S, lc_P, 'r')
        plt.plot(S, lp_P, 'b')
        plt.legend(["Long Call", "Long Put"])

        T2 = long_call(S, 120, 10)
        T4 = long_put(S, 100, 10)
        plt.subplot(1, 2, 2)
        plt.plot(S, T2, 'r')
        plt.plot(S, T4, 'b')
        plt.legend(["Long Call", "Long Put"])
        plt.show()
