""" Heston Hull-White MonteCarlo Almost Exact simulation
"""
import sys
import matplotlib.pyplot as plt
import numpy as np
from config import *

def CIR_Sample(NPaths,kappa,gamma,vbar,s,t,v_s):
    """
    """
    delta = 4.0 *kappa*vbar/gamma/gamma
    c= 1.0/(4.0*kappa)*gamma*gamma*(1.0-np.exp(-kappa*(t-s)))
    kappaBar = 4.0*kappa*v_s*np.exp(-kappa*(t-s))/(gamma*gamma*(1.0-np.exp(-kappa*(t-s))))
    sample = c* np.random.noncentral_chisquare(delta,kappaBar,NPaths)
    return  sample

def GeneratePathsHestonHW_AES(NPaths,NSteps,P0T,T,S_0,kappa,gamma,rhoxr,rhoxv,vbar,v0,lambd,eta):
    """
    Generate Paths from Monte Carlo Euler discretization for the Heston Hull White model (HHW)

    Parameters
    ----------
    NoOfPaths : int
        Number of paths for the evolution of the SDE.

    NoOfSteps : int
        Number of time steps for every path.

    P0T : function
        Discounted bond curve.

    T : float
        Time until maturity for the options, in years.

    S_0 : float
        Price value of the underlaying for the SDE with GBM.

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
    dt = 0.0001
    f0T = lambda t: - (np.log(P0T(t+dt))-np.log(P0T(t-dt)))/(2*dt)
    # Initial interest rate is forward rate at time t->0
    r0 = f0T(0.00001)
    theta = lambda t: 1.0/lambd * (f0T(t+dt)-f0T(t-dt))/(2.0*dt) + f0T(t) + eta**2/(2.0*lambd**2)*(1.0-np.exp(-2.0*lambd*t))
    
    Z1 = np.random.normal(0.0,1.0,[NPaths,NSteps])
    Z2 = np.random.normal(0.0,1.0,[NPaths,NSteps])
    Z3 = np.random.normal(0.0,1.0,[NPaths,NSteps])
    
    W1 = np.zeros([NPaths, NSteps+1])
    W2 = np.zeros([NPaths, NSteps+1])
    W3 = np.zeros([NPaths, NSteps+1])
    
    V = np.zeros([NPaths, NSteps+1])
    X = np.zeros([NPaths, NSteps+1])
    R = np.zeros([NPaths, NSteps+1])
    M_t = np.ones([NPaths,NSteps+1])
    
    R[:,0]=r0
    V[:,0]=v0
    X[:,0]=np.log(S_0)
    
    time = np.zeros([NSteps+1])
    dt = T / float(NSteps)
    for i in range(0,NSteps):
        # Making sure that samples from a normal have mean 0 and variance 1
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
        # Moment matching component, i.e. ensure that E(S(T)/M(T))= S0
        a = S_0 / np.mean(np.exp(X[:,i+1])/M_t[:,i+1])
        X[:,i+1] = X[:,i+1] + np.log(a)
        time[i+1] = time[i] + dt
        sys.stderr.write("Time step AES : {0}\r".format(i))
    sys.stderr.write("\n")
    # Compute exponent
    S = np.exp(X)
    paths = {"time":time,"S":S,"R":R,"M_t":M_t}
    return paths

if __name__ == "__main__":

    from H1HW import OptionPriceFromMonteCarlo

    np.random.seed(1)
    pathsExact = GeneratePathsHestonHW_AES(NPaths,NSteps,P0T,T,S0,kappa,gamma,rhoxr,rhoxv,vbar,v0,lambd,eta)
    S_ex = pathsExact["S"]
    M_t_ex = pathsExact["M_t"]
    print(np.mean(S_ex[:,1]/M_t_ex[:,1]))
    print(np.mean(S_ex[:,-1]/M_t_ex[:,-1]))
    valueOptMC_ex = OptionPriceFromMonteCarlo(CP,S_ex[:,-1],K,M_t_ex[:,-1])
    #==============================================================================
    plt.figure(1)
    plt.plot(K,valueOptMC_ex,'.k')
    # plt.ylim([0.0,110.0])
    plt.legend(['AES'])
    plt.xlabel('Strike, K')
    plt.ylabel('EU Option Value')
    plt.grid()
    # plt.savefig("img/MC_vs_AES_vs_COS.png",bbox_inches='tight')
    plt.show()
