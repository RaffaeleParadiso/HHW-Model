import numpy as np
import sys

def GeneratePathsHestonHWEuler(NoOfPaths,NoOfSteps,P0T,T,S_0,kappa,gamma,rhoxr,rhoxv,vbar,v0,lambd,eta):   
    """
    Generate Paths from Monte Carlo Euler discretization for the Heston Hull White model (HHW)

    Parameters
    ----------
    NoOfPaths : int
        Number of paths for the evolution of the SDE.

    NoOfSteps : int
        Number of time steps for every path.

    P0T : function
        Zero Coupon Bond curve with maturity T (obtained from the market).

    T : float
        Time until maturity for the options (years).

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
    r0 = f0T(0.00001)  # Initial interest rate is forward rate at time t->0
    theta = lambda t: 1.0/lambd * (f0T(t+dt)-f0T(t-dt))/(2.0*dt) + f0T(t) + eta**2/(2.0*lambd**2)*(1.0-np.exp(-2.0*lambd*t))
    
    Z1 = np.random.normal(0.0,1.0,[NoOfPaths,NoOfSteps])
    Z2 = np.random.normal(0.0,1.0,[NoOfPaths,NoOfSteps])
    Z3 = np.random.normal(0.0,1.0,[NoOfPaths,NoOfSteps])
    
    W1 = np.zeros([NoOfPaths, NoOfSteps+1])
    W2 = np.zeros([NoOfPaths, NoOfSteps+1])
    W3 = np.zeros([NoOfPaths, NoOfSteps+1])
    
    V = np.zeros([NoOfPaths, NoOfSteps+1])
    X = np.zeros([NoOfPaths, NoOfSteps+1])
    R = np.zeros([NoOfPaths, NoOfSteps+1])
    M_t = np.ones([NoOfPaths,NoOfSteps+1])
    
    R[:,0]=r0
    V[:,0]=v0
    X[:,0]=np.log(S_0)
    
    time = np.zeros([NoOfSteps+1])
    dt = T / float(NoOfSteps)
    for i in range(0,NoOfSteps):
        if NoOfPaths > 1: # samples from a normal with mean 0 and variance 1
            Z1[:,i] = (Z1[:,i] - np.mean(Z1[:,i])) / np.std(Z1[:,i])
            Z2[:,i] = (Z2[:,i] - np.mean(Z2[:,i])) / np.std(Z2[:,i])
            Z3[:,i] = (Z3[:,i] - np.mean(Z3[:,i])) / np.std(Z3[:,i])

        W1[:,i+1] = W1[:,i] + np.power(dt, 0.5)*Z1[:,i]
        W2[:,i+1] = W2[:,i] + np.power(dt, 0.5)*Z2[:,i]
        W3[:,i+1] = W3[:,i] + np.power(dt, 0.5)*Z3[:,i]

        # Truncated boundary condition
        R[:,i+1] = R[:,i] + lambd*(theta(time[i]) - R[:,i]) * dt + eta* (W1[:,i+1]-W1[:,i])

        M_t[:,i+1] = M_t[:,i] * np.exp(0.5*(R[:,i+1] + R[:,i])*dt)

        V[:,i+1] = V[:,i] + kappa*(vbar - V[:,i]) * dt + gamma* np.sqrt(V[:,i]) * (W2[:,i+1]-W2[:,i])
        V[:,i+1] = np.maximum(V[:,i+1],0.0)

        term1 = rhoxr * (W1[:,i+1]-W1[:,i]) + rhoxv * (W2[:,i+1]-W2[:,i]) + np.sqrt(1.0-rhoxr*rhoxr-rhoxv*rhoxv)* (W3[:,i+1]-W3[:,i])
        
        X[:,i+1] = X[:,i] + (R[:,i] - 0.5*V[:,i])*dt + np.sqrt(V[:,i])*term1
        time[i+1] = time[i] + dt     

        # Moment matching component, i.e. ensure that E(S(T)/M(T))= S0
        a = S_0 / np.mean(np.exp(X[:,i+1])/M_t[:,i+1])
        X[:,i+1] = X[:,i+1] + np.log(a)

        sys.stderr.write("Time step Euler MC: {0}\r".format(i))
    sys.stderr.write("\n")

    # Compute exponent
    S = np.exp(X)
    paths = {"time":time,"S":S,"R":R,"M_t":M_t}
    return paths