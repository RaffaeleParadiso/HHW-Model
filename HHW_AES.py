""" 
Heston Hull-White MonteCarlo Almost Exact simulation

"""

import sys
import numpy as np


def CIR_Sample(NPaths,kappa,gamma,vbar,s,t,v_s):
    """
    """
    delta = 4.0 *kappa*vbar/gamma/gamma
    c= 1.0/(4.0*kappa)*gamma*gamma*(1.0-np.exp(-kappa*(t-s)))
    kappaBar = 4.0*kappa*v_s*np.exp(-kappa*(t-s))/(gamma*gamma*(1.0-np.exp(-kappa*(t-s))))
    sample = c* np.random.noncentral_chisquare(delta,kappaBar,NPaths)
    return  sample

def GeneratePathsHestonHW_AES(NPaths,NSteps,S0,set_params):
    """
    Generate Paths from Monte Carlo Euler discretization for the Heston Hull White model (HHW)

    Parameters
    ----------
    NPaths : int
        Number of paths for the evolution of the SDE.
    NSteps : int
        Number of time steps for every path.
    S0 : float
        Price value of the underlaying for the SDE with GBM.
    T : float
        Time until maturity for the options, in years.
    P0T : function
        Discounted bond curve.
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
    P0T,T,kappa,gamma,rhoxr,rhoxv,vbar,v0,lambd,eta = set_params
    dt = 0.0001
    f_ZERO_T = lambda t: - (np.log(P0T(t+dt))-np.log(P0T(t-dt)))/(2*dt)
    r0 = f_ZERO_T(dt) # Initial interest rate is forward rate at time t->0
    theta = lambda t: 1.0/lambd * (f_ZERO_T(t+dt)-f_ZERO_T(t-dt))/(2.0*dt) + f_ZERO_T(t) + eta**2/(2.0*lambd**2)*(1.0-np.exp(-2.0*lambd*t))
    
    # Values from normal distribution with mean 0 and variance 1.
    Z1 = np.random.normal(0.0,1.0,[NPaths,NSteps])
    Z2 = np.random.normal(0.0,1.0,[NPaths,NSteps])
    Z3 = np.random.normal(0.0,1.0,[NPaths,NSteps])
    
    # Wiener process for S(t), R(t) and V(t)
    W1 = np.zeros([NPaths, NSteps+1])
    W2 = np.zeros([NPaths, NSteps+1])
    W3 = np.zeros([NPaths, NSteps+1])
    

    V = np.zeros([NPaths, NSteps+1])
    X = np.zeros([NPaths, NSteps+1])
    R = np.zeros([NPaths, NSteps+1])
    M_t = np.ones([NPaths,NSteps+1])
    
    # Initial values
    R[:,0] = r0 # initial interest rate
    V[:,0] = v0 # initial volatility
    X[:,0] = np.log(S0) # current stock price
    
    time = np.zeros([NSteps+1])
    dt = T / float(NSteps)
    for i in range(0,NSteps):
        # Making sure that samples from a normal have mean 0 and variance 1 (Standardization)
        if NPaths > 1:
            Z1[:,i] = (Z1[:,i] - np.mean(Z1[:,i])) / np.std(Z1[:,i])
            Z2[:,i] = (Z2[:,i] - np.mean(Z2[:,i])) / np.std(Z2[:,i])
            Z3[:,i] = (Z3[:,i] - np.mean(Z3[:,i])) / np.std(Z3[:,i])
        
        W1[:,i+1] = W1[:,i] + np.power(dt, 0.5)*Z1[:,i]
        W2[:,i+1] = W2[:,i] + np.power(dt, 0.5)*Z2[:,i]
        W3[:,i+1] = W3[:,i] + np.power(dt, 0.5)*Z3[:,i]
        
        R[:,i+1] = R[:,i] + lambd*(theta(time[i]) - R[:,i]) * dt + eta* (W1[:,i+1]-W1[:,i])
        M_t[:,i+1] = M_t[:,i] * np.exp(0.5*(R[:,i+1] + R[:,i])*dt)
        
        # Exact samples for the variance process
        V[:,i+1] = CIR_Sample(NPaths,kappa,gamma,vbar,0,dt,V[:,i])
        
        k0 = -rhoxv /gamma * kappa*vbar*dt
        k2 = rhoxv/gamma
        k1 = kappa*k2 -0.5
        k3 = np.sqrt(1.0-rhoxr*rhoxr - rhoxv*rhoxv)
        
        X[:,i+1] = X[:,i] + k0 + (k1*dt - k2)*V[:,i] + R[:,i]*dt + k2*V[:,i+1] + np.sqrt(V[:,i]*dt)*(rhoxr*Z1[:,i] + k3 * Z3[:,i])
        time[i+1] = time[i] + dt
        
        # Moment matching component, ensure that E(S(T)/M(T)) = S(t_0)/M(t_0) is a martingala
        a = S0 / np.mean(np.exp(X[:,i+1])/M_t[:,i+1])
        X[:,i+1] = X[:,i+1] + np.log(a)
        sys.stderr.write("Time step AES : {0}\r".format(i))
    sys.stderr.write("\n")
    # Compute exponent
    S = np.exp(X)
    paths = {"time":time,"S":S,"R":R,"M_t":M_t}
    return paths

if __name__ == "__main__":

    import matplotlib.pyplot as plt
    from matplotlib import pylab
    from pylab import *
    pylab.rcParams['figure.figsize'] = (13, 4)

    from main import OptionPriceFromMonteCarlo
    from config import *
    
    FIGURE = True
    SAVE = False

    np.random.seed(1)

    set_params = (P0T,T,kappa,gamma,rhoxr,rhoxv,vbar,v0,lambd,eta)

    pathsExact = GeneratePathsHestonHW_AES(NPaths,NSteps,S0,set_params)


    time_n = pathsExact["time"]
    S_ex = pathsExact["S"]
    R_n = pathsExact["R"]
    M_t_ex = pathsExact["M_t"]
    #==============================================================================
    print(f"{(np.mean(S_ex[:,1]/M_t_ex[:,1])):.2f}")
    print(f"{(np.mean(S_ex[:,-1]/M_t_ex[:,-1])):.2f}")
    #==============================================================================
    valueOptMC_ex = OptionPriceFromMonteCarlo(CP,S_ex[:,-1],K,M_t_ex[:,-1])
    #==============================================================================
    if FIGURE:
        plt.figure()
        plt.plot(K,valueOptMC_ex,'.k')
        plt.legend(['AES'])
        plt.xlabel('Strike, K')
        plt.ylabel('EU Option Value')
        plt.grid()
        if SAVE: plt.savefig("AES.png",bbox_inches='tight')
        plt.show()

        plt.figure()
        for i in range(0,5):
            plt.title("Stock Price path")
            plt.plot(time_n,S_ex[i,:])
        plt.show()

        plt.figure()
        for i in range(0,5):
            plt.title("Interest rate paths")
            plt.plot(time_n,R_n[i,:])
        plt.show()

        plt.figure()
        for i in range(0,10):
            plt.title("Numeraire paths")
            plt.plot(time_n,M_t_ex[i,:])
        plt.show()

    if SAVE:
        np.savetxt("AES_time_ex.txt", time_n,  fmt='%.4f')
        np.savetxt("AES_S_ex.txt", S_ex,  fmt='%.4f')
        np.savetxt("AES_R_ex.txt", R_n,  fmt='%.4f')
        np.savetxt("AES_M_t_ex.txt", M_t_ex,  fmt='%.4f')
