"""
Expectation of the square root of the CIR process
"""
import matplotlib.pyplot as plt
import numpy as np
from regex import D


def CIREuler(NPaths, NSteps, T, kappa, v0, vbar, gamma):
    """
    Monte Carlo Euler simulation for the CIR type SDE
    """
    Z = np.random.normal(0.0, 1.0, [NPaths, NSteps])
    W = np.zeros([NPaths, NSteps + 1])
    V = np.zeros([NPaths, NSteps + 1])
    V[:, 0] = v0
    time = np.zeros([NSteps + 1])
    dt = T / float(NSteps)
    for i in range(0, NSteps):
        if NPaths > 1:
            Z[:, i] = (Z[:, i] - np.mean(Z[:, i])) / np.std(Z[:, i])
        W[:, i + 1] = W[:, i] + (dt**0.5) * Z[:, i]
        V[:, i + 1] = (
            V[:, i]
            + kappa * (vbar - V[:, i]) * dt
            + gamma * np.sqrt(V[:, i]) * (W[:, i + 1] - W[:, i])
        )
        V[:, i + 1] = np.maximum(
            V[:, i + 1], 0.0
        )  # Truncation scheme for negative values (maximum)
        time[i + 1] = time[i] + dt
    return time, V


def meanSqrtV_1(kappa, v0, vbar, gamma):
    """
    Result 13.2.1 page 423
    Approximation for the expectation \mathbf{E}[\sqrt{v_t}] after the application of the theta method
    """
    delta = (4.0 * kappa * vbar) / (gamma**2)
    c = lambda t: (gamma**2) * (1.0 - np.exp(-kappa * t)) / (4.0 * kappa)
    kappaBar = (
        lambda t: 4.0
        * kappa
        * v0
        * np.exp(-kappa * t)
        / ((gamma**2) * (1.0 - np.exp(-kappa * t)))
    )
    return lambda t: np.sqrt(
        c(t) * ((kappaBar(t) - 1.0) + delta + delta / (2.0 * (delta + kappaBar(t))))
    )


def meanSqrtV_2(kappa, v0, vbar, gamma):
    """
    Result 13.45 page 424
    Further approximation for the expectation \mathbf{E}[\sqrt{v_t}] after the application of the theta method
    and by matching the value of the limits in 0, +infinity and 1.
    """
    a = np.sqrt(vbar - (gamma**2.0) / (8.0 * kappa))
    b = np.sqrt(v0) - a
    temp = meanSqrtV_1(kappa, v0, vbar, gamma)  # first approx for expectation
    epsilon1 = temp(1)  # value from the first approx in t=1
    c = -np.log(1.0 / b * (epsilon1 - a))
    return lambda t: a + b * np.exp(-c * t)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from matplotlib import pylab
    from pylab import *

    pylab.rcParams["figure.figsize"] = (7, 4)

    NPaths = 10000
    NSteps = 2000
    T = 10.0
    time2 = np.linspace(0, T, 10)

    SAVE = 0

    # Feller Condition 2* kappa * vbar > gamma^2
    Parameters1 = {"kappa": 1.2, "gamma": 0.1, "v0": 0.04, "vbar": 0.03}  # yep
    Parameters2 = {"kappa": 1.2, "gamma": 0.22, "v0": 0.035, "vbar": 0.02}  # nope

    plt.figure()
    kappa = Parameters1["kappa"]
    gamma = Parameters1["gamma"]
    v0 = Parameters1["v0"]
    vbar = Parameters1["vbar"]
    time, V = CIREuler(NPaths, NSteps, T, kappa, v0, vbar, gamma)
    Vsqrt = np.sqrt(V)
    EsqrtV = Vsqrt.mean(axis=0)
    plt.plot(time, EsqrtV, "x", markersize=2, label="Exact")
    approx1 = meanSqrtV_1(kappa, v0, vbar, gamma)
    approx2 = meanSqrtV_2(kappa, v0, vbar, gamma)
    plt.plot(time, approx1(time), "--r", label=f"First Approx")
    plt.plot(time2, approx2(time2), ".k", label=f"Second Approx")
    plt.xlabel("time")
    plt.ylabel(r"$\mathbb{E}[\sqrt{v_t}]$", size=15)
    text = rf"$\mathbb{{E}}[\sqrt{{v_t}}]$ with parameters: $\kappa = {kappa}$, $\gamma = {gamma}$, $v_0 = {v0}$, $\bar{{v}} = {vbar}$"
    plt.title(text)
    plt.legend()
    plt.grid()
    if SAVE:
        plt.savefig("Feller_hold.png", bbox_inches="tight")
    plt.show()

    plt.figure()
    kappa = Parameters2["kappa"]
    gamma = Parameters2["gamma"]
    v0 = Parameters2["v0"]
    vbar = Parameters2["vbar"]
    time, V = CIREuler(NPaths, NSteps, T, kappa, v0, vbar, gamma)
    Vsqrt = np.sqrt(V)
    EsqrtV = Vsqrt.mean(axis=0)
    plt.plot(time, EsqrtV, "x", markersize=2, label="Exact")
    approx1 = meanSqrtV_1(kappa, v0, vbar, gamma)
    approx2 = meanSqrtV_2(kappa, v0, vbar, gamma)
    plt.plot(time, approx1(time), "--r", label=f"First Approx")
    plt.plot(time2, approx2(time2), ".k", label=f"Second Approx")
    plt.xlabel("time")
    plt.ylabel(r"$\mathbb{E}[\sqrt{v_t}]$", size=15)
    text = rf"$\mathbb{{E}}[\sqrt{{v_t}}]$ with parameters: $\kappa = {kappa}$, $\gamma = {gamma}$, $v_0 = {v0}$, $\bar{{v}} = {vbar}$"
    plt.title(text)
    plt.legend()
    plt.grid()
    if SAVE:
        plt.savefig("Feller_not_hold.png", bbox_inches="tight")
    plt.show()
