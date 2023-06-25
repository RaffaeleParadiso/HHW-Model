""" 
Heston Hull-White MonteCarlo Euler simulation

"""

import sys

import numpy as np


def HHW_Euler(NPaths, NSteps, S0, set_params):
    """
    Generate Paths from Monte Carlo Euler discretization for the Heston Hull White model (HHW)

    Parameters
    ----------
    NoOfPaths : int
        Number of paths for the evolution of the SDE.
    NoOfSteps : int
        Number of time steps for every path.
    S0 : float
        Price value of the underlaying for the SDE with GBM.
    T : float
        Time until maturity for the options (years).
    P0T : function
        Zero Coupon Bond curve with maturity T (obtained from the market).
    kappa : float
    rhoxr : float
    rhoxv : float
    vbar : float
    v0 : float
    lambd : float
    eta : float
    gamma : float

    Returns
    -------
    paths : ndarray
        see dtype parameter above.
    """
    P0T, T, kappa, gamma, rhoxr, rhoxv, vbar, v0, lambd, eta = set_params
    dt = 0.00001
    f_ZERO_T = lambda t: -(np.log(P0T(t + dt)) - np.log(P0T(t - dt))) / (
        2 * dt
    )  # instantaneus forward rate
    r0 = f_ZERO_T(
        0.00001
    )  # The initial interest rate is the forward rate at time t -> 0.
    theta = (
        lambda t: 1.0 / lambd * (f_ZERO_T(t + dt) - f_ZERO_T(t - dt)) / (2.0 * dt)
        + f_ZERO_T(t)
        + eta**2 / (2.0 * lambd**2) * (1.0 - np.exp(-2.0 * lambd * t))
    )

    # Values from normal distribution with mean 0 and variance 1.
    Z1 = np.random.normal(0.0, 1.0, [NPaths, NSteps])
    Z2 = np.random.normal(0.0, 1.0, [NPaths, NSteps])
    Z3 = np.random.normal(0.0, 1.0, [NPaths, NSteps])

    # Wiener process for S(t), R(t) and V(t)
    W1 = np.zeros([NPaths, NSteps + 1])
    W2 = np.zeros([NPaths, NSteps + 1])
    W3 = np.zeros([NPaths, NSteps + 1])

    V = np.zeros([NPaths, NSteps + 1])
    X = np.zeros([NPaths, NSteps + 1])
    R = np.zeros([NPaths, NSteps + 1])
    M_t = np.ones([NPaths, NSteps + 1])

    # Initial values
    R[:, 0] = r0  # initial interest rate
    V[:, 0] = v0  # initial volatility
    X[:, 0] = np.log(S0)  # current stock price

    time = np.zeros([NSteps + 1])
    dt = T / float(
        NSteps
    )  # time steps (defined from the number of steps and the maturity time)
    for i in range(0, NSteps):
        # Making sure that samples from a normal have mean 0 and variance 1 (Standardization)
        if NPaths > 1:
            Z1[:, i] = (Z1[:, i] - np.mean(Z1[:, i])) / np.std(Z1[:, i])
            Z2[:, i] = (Z2[:, i] - np.mean(Z2[:, i])) / np.std(Z2[:, i])
            Z3[:, i] = (Z3[:, i] - np.mean(Z3[:, i])) / np.std(Z3[:, i])

        # Wiener process evolution
        W1[:, i + 1] = W1[:, i] + (dt**0.5) * Z1[:, i]
        W2[:, i + 1] = W2[:, i] + (dt**0.5) * Z2[:, i]
        W3[:, i + 1] = W3[:, i] + (dt**0.5) * Z3[:, i]

        # Truncated boundary condition
        R[:, i + 1] = (
            R[:, i]
            + lambd * (theta(time[i]) - R[:, i]) * dt
            + eta * (W1[:, i + 1] - W1[:, i])
        )

        M_t[:, i + 1] = M_t[:, i] * np.exp(0.5 * (R[:, i + 1] + R[:, i]) * dt)

        V[:, i + 1] = (
            V[:, i]
            + kappa * (vbar - V[:, i]) * dt
            + gamma * np.sqrt(V[:, i]) * (W2[:, i + 1] - W2[:, i])
        )
        V[:, i + 1] = np.maximum(
            V[:, i + 1], 0.0
        )  # Truncated Euler scheme for CIR process

        term1 = (
            rhoxr * (W1[:, i + 1] - W1[:, i])
            + rhoxv * (W2[:, i + 1] - W2[:, i])
            + np.sqrt(1.0 - rhoxr**2 - rhoxv**2) * (W3[:, i + 1] - W3[:, i])
        )

        X[:, i + 1] = (
            X[:, i] + (R[:, i] - 0.5 * V[:, i]) * dt + np.sqrt(V[:, i]) * term1
        )
        time[i + 1] = time[i] + dt

        # Moment matching component, ensure that E(S(T)/M(T)) = S(t_0)/M(t_0) is a martingala
        a = S0 / np.mean(np.exp(X[:, i + 1]) / M_t[:, i + 1])
        X[:, i + 1] = X[:, i + 1] + np.log(a)
        sys.stderr.write("Time step Euler MC         : {0}\r".format(i))
    sys.stderr.write("\n")
    S = np.exp(X)  # Compute exponent
    paths = {"time": time, "S": S, "R": R, "M_t": M_t}
    return paths, V


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


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from matplotlib import pylab
    from pylab import *

    pylab.rcParams["figure.figsize"] = (10, 4)

    from config import *

    FIGURE = True
    SAVE = 1

    np.random.seed(1)

    set_params = (P0T, T, kappa, gamma, rhoxr, rhoxv, vbar, v0, lambd, eta)

    paths, Volat = HHW_Euler(NPaths, NSteps, S0, set_params)

    time_n = paths["time"]
    S_n = paths["S"]
    R_n = paths["R"]
    M_t_n = paths["M_t"]
    # ==============================================================================
    print(f"{np.mean(S_n[:,1]/M_t_n[:,1]):.2f}")
    print(f"{np.mean(S_n[:,-1]/M_t_n[:,-1]):.2f}")
    # ==============================================================================
    valueOptMC = OptionPriceFromMonteCarlo(CP, S_n[:, -1], K, M_t_n[:, -1])
    # ==============================================================================
    if FIGURE:
        plt.figure()
        plt.plot(K, valueOptMC)
        plt.title('European Option Value for different Strikes Price (K)')
        plt.legend(["Euler"])
        plt.xlabel("Strike, K")
        plt.ylabel("EU Option Value")
        plt.grid()
        if SAVE:
            plt.savefig("MC.png", bbox_inches="tight", dpi=600)

        plt.figure()
        plt.title("Stock Price paths")
        plt.xlabel("Time t (years)")
        plt.grid()
        for i in range(0, 50):
            plt.plot(time_n, S_n[i, :])
        if SAVE:
            plt.savefig(f"Euler_{T}_StockPaths.png", bbox_inches="tight", dpi=600)

        plt.figure()
        plt.title("Interest rate paths")
        plt.xlabel("Time t (years)")
        plt.grid()
        for i in range(0, 50):
            plt.plot(time_n, R_n[i, :])
        if SAVE:
            plt.savefig(f"Euler_{T}_IRPaths.png", bbox_inches="tight", dpi=600)

        plt.figure()
        plt.title("Stochastic Volatility paths")
        plt.xlabel("Time t (years)")
        plt.grid()
        for i in range(0, 50):
            plt.plot(time_n, Volat[i, :])
        if SAVE:
            plt.savefig(f"Euler_{T}_VolPaths.png", bbox_inches="tight", dpi=600)

        plt.figure()
        plt.title("Numeraire paths")
        plt.xlabel("Time t (years)")
        plt.grid()
        for i in range(0, 10):
            plt.plot(time_n, M_t_n[i, :])
        plt.show()

    if SAVE:
        np.savetxt("MC_time_n.txt", time_n, fmt="%.4f")
        np.savetxt("MC_S_n.txt", S_n, fmt="%.4f")
        np.savetxt("MC_R_n.txt", R_n, fmt="%.4f")
        np.savetxt("MC_M_t_n.txt", M_t_n, fmt="%.4f")
