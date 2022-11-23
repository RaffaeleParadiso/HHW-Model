""" Heston Hull-White MonteCarlo Euler simulation
"""
import matplotlib.pyplot as plt
import numpy as np
from configuration import (EUOptionPriceFromMCPathsGeneralizedStochIR, CP, NPaths,
                            NSteps, lambd, eta, S0, T, K, P0T, gamma, vbar, v0,
                            rhoxr, rhoxv, kappa)
np.random.seed(1)

FIGURE = False
SAVE = True

dt = 0.00001
f0T = lambda t: -(np.log(P0T(t+dt))-np.log(P0T(t-dt)))/(2*dt)  # forward rate
r0 = f0T(0.00001) # Initial interest rate is the forward rate at time t -> 0.
theta = lambda t: 1.0/lambd*(f0T(t+dt)-f0T(t-dt))/(2.0*dt)+f0T(t)+eta**2/(2.0*lambd**2)*(1.0-np.exp(-2.0*lambd*t))

# Values from normal distribution with mean 0 and variance 1.
Z1 = np.random.normal(0.0,1.0,[NPaths,NSteps])
Z2 = np.random.normal(0.0,1.0,[NPaths,NSteps])
Z3 = np.random.normal(0.0,1.0,[NPaths,NSteps])

# Wiener process for S(t), R(t) and V(t)
W1 = np.zeros([NPaths, NSteps+1])
W2 = np.zeros([NPaths, NSteps+1])
W3 = np.zeros([NPaths, NSteps+1])

# Initial values
V = np.zeros([NPaths, NSteps+1])
X = np.zeros([NPaths, NSteps+1])
R = np.zeros([NPaths, NSteps+1])
M_t = np.ones([NPaths,NSteps+1])
R[:,0]=r0 # initial interest rate
V[:,0]=v0 # initial volatility
X[:,0]=np.log(S0) # current stock price


time = np.zeros([NSteps+1])
dt = T/float(NSteps) # time steps (defined from the number of steps and the maturity time)
for i in range(0,NSteps):
    if NPaths > 1:     # Normal with mean 0 and variance 1 (standard normal distribution)
        Z1[:,i] = (Z1[:,i]-np.mean(Z1[:,i]))/np.std(Z1[:,i])
        Z2[:,i] = (Z2[:,i]-np.mean(Z2[:,i]))/np.std(Z2[:,i])
        Z3[:,i] = (Z3[:,i]-np.mean(Z3[:,i]))/np.std(Z3[:,i])

    # Wiener process evolution
    W1[:,i+1] = W1[:,i]+np.power(dt,0.5)*Z1[:,i]
    W2[:,i+1] = W2[:,i]+np.power(dt,0.5)*Z2[:,i]
    W3[:,i+1] = W3[:,i]+np.power(dt,0.5)*Z3[:,i]

    # Truncated boundary condition
    R[:,i+1] = R[:,i]+lambd*(theta(time[i])-R[:,i])*dt+eta*(W1[:,i+1]-W1[:,i])
    M_t[:,i+1] = M_t[:,i]*np.exp(0.5*(R[:,i+1]+R[:,i])*dt)
    V[:,i+1] = V[:,i]+kappa*(vbar-V[:,i])*dt+gamma*np.sqrt(V[:,i])*(W2[:,i+1]-W2[:,i])
    V[:,i+1] = np.maximum(V[:,i+1],0.0)
    term1 = rhoxr*(W1[:,i+1]-W1[:,i])+rhoxv*(W2[:,i+1]-W2[:,i])+np.sqrt(1.0-rhoxr**2-rhoxv**2)*(W3[:,i+1]-W3[:,i])
    X[:,i+1] = X[:,i]+(R[:,i]-0.5*V[:,i])*dt+np.sqrt(V[:,i])*term1
    time[i+1] = time[i]+dt
    # Moment matching component, ensure that E(S(T)/M(T)) = S(t_0)/M(t_0) is a martingala
    a = S0 / np.mean(np.exp(X[:,i+1])/M_t[:,i+1])
    X[:,i+1] = X[:,i+1] + np.log(a)
# Compute exponent
S = np.exp(X)
paths = {"time":time,"S":S,"R":R,"M_t":M_t}
#==============================================================================
time_n = paths["time"]
S_n = paths["S"]
R_n = paths["R"]
M_t_n = paths["M_t"]
#==============================================================================
print(np.mean(S_n[:,1]/M_t_n[:,1]))
print(np.mean(S_n[:,-1]/M_t_n[:,-1]))
#==============================================================================
valueOptMC= EUOptionPriceFromMCPathsGeneralizedStochIR(CP,S_n[:,-1],K,M_t_n[:,-1])
#==============================================================================
if FIGURE:
    plt.figure(0)
    plt.plot(K,valueOptMC)
    plt.ylim([0.0,110.0])
    plt.legend(['Euler'])
    plt.grid()
    plt.show()

    plt.figure()
    for i in range(0,5):
        plt.title("Stock Price path")
        plt.plot(time,S_n[i,:])
    plt.show()

    plt.figure()
    for i in range(0,5):
        plt.title("Interest rate paths")
        plt.plot(time,R_n[i,:])
    plt.show()

    plt.figure()
    for i in range(0,10):
        plt.title("Numeraire paths")
        plt.plot(time,M_t_n[i,:])
    plt.show()

if SAVE:
    np.savetxt("time_n.txt", time_n,  fmt='%.4f')
    np.savetxt("S_n.txt", S_n,  fmt='%.4f')
    np.savetxt("R_n.txt", R_n,  fmt='%.4f')
    np.savetxt("M_t_n.txt", M_t_n,  fmt='%.4f')
