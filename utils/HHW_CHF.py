"""
Characterist function for the Heston-Hull-White Model, obtain from the Riccati System of Equations
and coefficients chi e psi for the call/put price

"""

import numpy as np
import scipy.integrate as integrate
import scipy.special as sp


def meanSqrtV_3(k, v0, vbar, gamma):
    """
    Exact expectation E(sqrt(V(t)))
    """
    delta = (4.0 * k * vbar) / (gamma**2)
    c = lambda t: (gamma**2) * (1.0 - np.exp(-k * t)) / (4.0 * k)
    kbar = (
        lambda t: 4.0
        * k
        * v0
        * np.exp(-k * t)
        / ((gamma**2) * (1.0 - np.exp(-k * t)))
    )
    temp1 = (
        lambda t: np.sqrt(2.0 * c(t))
        * sp.gamma((1.0 + delta) / 2.0)
        / sp.gamma(delta / 2.0)
        * sp.hyp1f1(-0.5, delta / 2.0, -kbar(t) / 2.0)
    )  # hyp1f1 confluent hyper-geometric function
    return temp1


def C_H1HW(u, tau, lambd):
    """
    Solution of the ODE system for the HHW
    """
    i = complex(0.0, 1.0)
    C = (i * u - 1.0) / lambd * (1 - np.exp(-lambd * tau))
    return C


def D_H1HW(u, tau, k, gamma, rhoxv):
    """
    Solution of the ODE system for the HHW
    """
    i = complex(0.0, 1.0)
    D1 = np.sqrt((gamma * rhoxv * i * u - k) ** 2 - gamma**2 * i * u * (i * u - 1))
    g = (k - gamma * rhoxv * i * u - D1) / (k - gamma * rhoxv * i * u + D1)
    D = (
        (1.0 - np.exp(-D1 * tau))
        / ((gamma**2) * (1.0 - g * np.exp(-D1 * tau)))
        * (k - gamma * rhoxv * i * u - D1)
    )
    return D


def A_H1HW(u, tau, P0T, lambd, eta, k, gamma, vbar, v0, rhoxv, rhoxr):
    """
    Solution of the ODE system for the HHW
    """
    i = complex(0.0, 1.0)
    D1 = np.sqrt((gamma * rhoxv * i * u - k) ** 2 - gamma**2 * i * u * (i * u - 1))
    # D1 = np.sqrt(np.power(k-gamma*rhoxv*i*u,2)+(u**2+i*u)*gamma**2)
    g = (k - gamma * rhoxv * i * u - D1) / (k - gamma * rhoxv * i * u + D1)
    dt = 0.0001
    f0T = lambda t: -(np.log(P0T(t + dt)) - np.log(P0T(t - dt))) / (2.0 * dt)
    theta = (
        lambda t: 1.0 / lambd * (f0T(t + dt) - f0T(t - dt)) / (2.0 * dt)
        + f0T(t)
        + eta**2 / (2.0 * lambd**2) * (1.0 - np.exp(-2.0 * lambd * t))
    )
    N = 10000  # Integration within the function I_1 and I_4
    z = np.linspace(0, tau - 1e-10, N)

    # I_1_adj with time-dependent theta {C(u,\tau)*\theta}
    f1 = (1.0 - np.exp(-lambd * z)) * theta(tau - z) 
    value1 = integrate.trapezoid(f1, z)
    I_1_adj = (i * u - 1.0) * value1  
    
    I_2 = tau / (gamma**2.0) * (k - gamma * rhoxv * i * u - D1) - 2.0 / (
        gamma**2.0
    ) * np.log((1.0 - g * np.exp(-D1 * tau)) / (1.0 - g))
    I_3 = (
        1.0
        / (2.0 * np.power(lambd, 3.0))
        * np.power(i + u, 2.0)
        * (
            3.0
            + np.exp(-2.0 * lambd * tau)
            - 4.0 * np.exp(-lambd * tau)
            - 2.0 * lambd * tau
        )
    )
    meanSqrtV = meanSqrtV_3(k, v0, vbar, gamma)
    f2 = meanSqrtV(tau - z) * (1.0 - np.exp(-lambd * z))
    value2 = integrate.trapezoid(f2, z)
    I_4 = -(i * u + u**2) * value2 / lambd
    return I_1_adj + k * vbar * I_2 + 0.5 * eta**2.0 * I_3 + eta * rhoxr * I_4


def ChFH1HWModel(set_params):
    """
    Characteristic function for the H1-HW model without the B(u).
    """
    i = complex(0.0, 1.0)
    P0T, tau, k, gamma, rhoxr, rhoxv, vbar, v0, lambd, eta = set_params
    dt = 0.0001
    f0T = lambda t: -(np.log(P0T(t + dt)) - np.log(P0T(t - dt))) / (2.0 * dt)
    r0 = f0T(0.00001)
    C = lambda u: C_H1HW(u, tau, lambd)
    D = lambda u: D_H1HW(u, tau, k, gamma, rhoxv)
    A = lambda u: A_H1HW(u, tau, P0T, lambd, eta, k, gamma, vbar, v0, rhoxv, rhoxr)
    return lambda u: np.exp(A(u) + C(u) * r0 + D(u) * v0)


def Chi_Psi(a, b, c, d, k):
    """
    Return the values for psi and chi for e^x and 1 on [c,d] \in [a,b]
    """
    psi = np.sin(k * np.pi * (d - a) / (b - a)) - np.sin(k * np.pi * (c - a) / (b - a))
    psi[1:] = psi[1:] * (b - a) / (k[1:] * np.pi)
    psi[0] = d - c
    chi = 1.0 / (1.0 + (k * np.pi / (b - a)) ** 2.0)
    expr1 = np.cos(k * np.pi * (d - a) / (b - a)) * np.exp(d) - np.cos(
        k * np.pi * (c - a) / (b - a)
    ) * np.exp(c)
    expr2 = (k * np.pi / (b - a)) * (
        np.sin(k * np.pi * (d - a) / (b - a))
        - np.sin(k * np.pi * (c - a) / (b - a)) * np.exp(c)
    )
    chi = chi * (expr1 + expr2)
    value = {"chi": chi, "psi": psi}
    return value
