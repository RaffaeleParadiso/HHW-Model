"""
Hull-White MonteCarlo Euler simulation

"""

import numpy as np
from tqdm import tqdm

def HWEuler(NPaths,NSteps,T,P0T, lambd, eta):    
    dt = 0.0001    
    f_ZERO_T = lambda t: - (np.log(P0T(t+dt))-np.log(P0T(t-dt)))/(2*dt)
    r0 = f_ZERO_T(dt) # Initial interest rate is forward rate at time t->0
    theta = lambda t: 1.0/lambd * (f_ZERO_T(t+dt)-f_ZERO_T(t-dt))/(2.0*dt) + f_ZERO_T(t) + eta*eta/(2.0*lambd*lambd)*(1.0-np.exp(-2.0*lambd*t))      
    
    # Values from normal distribution with mean 0 and variance 1.
    Z = np.random.normal(0.0,1.0,[NPaths,NSteps])
    
    # Wiener process for R(t)
    W = np.zeros([NPaths, NSteps+1])
    
    R = np.zeros([NPaths, NSteps+1])
    
    # Initial values
    R[:,0] = r0 # initial interest rate

    time = np.zeros([NSteps+1]) 
    dt = T / float(NSteps)
    for i in tqdm(range(0,NSteps)):
        # Making sure that samples from a normal have mean 0 and variance 1 (Standardization)
        if NPaths > 1:
            Z[:,i] = (Z[:,i] - np.mean(Z[:,i])) / np.std(Z[:,i])

        W[:,i+1] = W[:,i] + np.power(dt, 0.5)*Z[:,i]

        R[:,i+1] = R[:,i] + lambd*(theta(time[i]) - R[:,i]) * dt + eta* (W[:,i+1]-W[:,i])
        time[i+1] = time[i] + dt

    paths = {"time":time,"R":R}
    return paths

if __name__ == "__main__":
    
    import matplotlib.pyplot as plt
    from matplotlib import pylab
    from pylab import *
    pylab.rcParams['figure.figsize'] = (13, 4)


    SAVE = False
    NPaths = 1
    NSteps = 20000
    T      = 5.0
    lambd  = 0.5
    eta    = 0.01
    P0T    = lambda T: np.exp(-0.05*T)

    plt.figure() 
    legend = []
    lambdVec = [-0.2, 0.2, 5.0]
    color = ["blue", "red", "green"]
    for idx,lambdl in enumerate(lambdVec):    
        np.random.seed(1)
        print(f"eta = {lambdl}")
        Paths = HWEuler(NPaths,NSteps,T,P0T, lambdl, eta)
        legend.append(f'lambdda={lambdl:.2f}')
        timeGrid = Paths["time"]
        R = Paths["R"]
        plt.plot(timeGrid, np.transpose(R), color=color[idx])   
    plt.grid()
    plt.title(rf"Effect of mean reversion lambda $\lambda$")
    plt.xlabel("time")
    plt.ylabel("R(t)")
    plt.legend(legend)
    if SAVE: plt.savefig("HullWhitelambda.png",bbox_inches='tight')

    plt.figure()    
    legend = []
    etaVec = [0.1, 0.2, 0.3]
    for idx, etat in enumerate(etaVec):
        np.random.seed(1)
        print(f"eta = {etat}")
        Paths = HWEuler(NPaths,NSteps,T,P0T, lambd, etat)
        legend.append(f'Eta={etat:.2f}')
        timeGrid = Paths["time"]
        R = Paths["R"]
        plt.plot(timeGrid, np.transpose(R), color=color[idx])   
    plt.grid()
    plt.title(rf"Effect of the Volatility $\eta$")
    plt.xlabel("time")
    plt.ylabel("R(t)")
    plt.legend(legend, loc='upper left')
    if SAVE: plt.savefig("HullWhiteVolatility.png",bbox_inches='tight')
    plt.show()
