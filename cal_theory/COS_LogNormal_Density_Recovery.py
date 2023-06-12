"""
Normal and LogNormal density recovery using the COS method
"""
import numpy as np


def COSDensity(cf, x, N, a, b):
    i = complex(0.0, 1.0)
    k = np.linspace(0, N - 1, N)
    u = np.zeros([1, N])
    u = k * np.pi / (b - a)
    F_k = 2.0 / (b - a) * np.real(cf(u) * np.exp(-i * u * a))
    F_k[0] = F_k[0] * 0.5
    f_X = np.matmul(F_k, np.cos(np.outer(u, x - a)))
    return f_X


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import scipy.stats as st
    from matplotlib import pylab
    from pylab import *
    from scipy.stats import lognorm

    pylab.rcParams["figure.figsize"] = (6.5, 4)

    SAVE = 1
    i = complex(0.0, 1.0)
    a = -10
    b = 10
    N = [2**x for x in range(2, 9, 1)]
    mu = 0
    sigma = 0.25
    cF = lambda u: np.exp(i * mu * u - 0.5 * sigma**2 * u**2)
    y = np.linspace(0.05, 5, 1000)

    dist = lognorm([sigma], loc=mu)
    ff = dist.pdf(y)

    plt.figure()
    plt.grid()
    plt.xlabel("y")
    plt.ylabel("$f_Y(y)$")
    error_list0 = []
    for n in N:
        f_Y = 1 / y * COSDensity(cF, np.log(y), n, a, b)
        error = np.max(np.abs(f_Y - ff))
        error_list0.append([n, error])
        print("For {0} expansion terms the error is {1}".format(n, error))
        plt.plot(y, f_Y, label=f"N = {n}")
    plt.plot(y, dist.pdf(y), "x", markersize=2, label="Exact")
    plt.legend()  # ["N=%.0f"%i for i in N])
    plt.title(
        rf"Log Normal Density recover with ${{\mu}}={mu}$ and ${{\sigma}} = {sigma}$"
    )
    if SAVE:
        plt.savefig("Log_COS.png", bbox_inches="tight")
    np.savetxt(
        "error_log_normal_approx_cos_method.txt", error_list0, fmt=["%.0i", "%.18e"]
    )
    plt.show()

    mu = 0.0
    sigma = 1.0
    x = np.linspace(-10.0, 10, 1000)
    f_XExact = st.norm.pdf(x, mu, sigma)

    plt.figure()
    plt.grid()
    plt.xlabel("x")
    plt.ylabel("$f_X(x)$")
    error_list = []
    for n in N:
        f_X = COSDensity(cF, x, n, a, b)
        error = np.max(np.abs(f_X - f_XExact))
        error_list.append([n, error])
        print("For {0} expansion terms the error is {1}".format(n, error))
        plt.plot(x, f_X, label=f"N = {n}")
    plt.plot(x, f_XExact, "x", markersize=2, label="Exact")
    plt.legend()
    plt.title(rf"Normal Density recover with ${{\mu}}={mu}$ and ${{\sigma}} = {sigma}$")
    if SAVE:
        plt.savefig("Normal_COS.png", bbox_inches="tight")
    plt.show()
    np.savetxt("error_normal_approx_cos_method.txt", error_list, fmt=["%.0i", "%.18e"])
