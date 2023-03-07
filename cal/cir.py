"""
Expectation of the square root of the CIR process
"""
import numpy as np
import matplotlib.pyplot as plt

def CIREuler(NPaths,NSteps,T,kappa,v0,vbar,gamma):
    """
    Monte Carlo Euler simulation for the CIR type SDE 
    """
    Z = np.random.normal(0.0,1.0,[NPaths,NSteps])
    W = np.zeros([NPaths, NSteps+1])
    V = np.zeros([NPaths, NSteps+1])
    V[:,0]=v0
    time = np.zeros([NSteps+1])
    dt = T / float(NSteps)
    for i in range(0,NSteps):
        if NPaths > 1:
            Z[:,i] = (Z[:,i] - np.mean(Z[:,i])) / np.std(Z[:,i])
        W[:,i+1] = W[:,i] + (dt**0.5)*Z[:,i]
        V[:,i+1] = V[:,i] + kappa*(vbar - V[:,i]) * dt + gamma* np.sqrt(V[:,i]) * (W[:,i+1]-W[:,i])
        V[:,i+1] = np.maximum(V[:,i+1],0.0)  # Truncation scheme for negative values (maximum)
        time[i+1] = time[i] + dt
    paths = {"time":time,"V":V}
    return paths

def meanSqrtV_1(kappa,v0,vbar,gamma):
    """
    Result 13.2.1 page 423
    Approximation for the expectation \mathbf{E}[\sqrt{v_t}] after the application of the theta method
    """
    delta = (4.0*kappa*vbar)/(gamma**2)
    c = lambda t: (gamma**2)*(1.0-np.exp(-kappa*t))/(4.0*kappa)
    kappaBar = lambda t: 4.0*kappa*v0*np.exp(-kappa*t)/((gamma**2)*(1.0-np.exp(-kappa*t)))
    result = lambda t: np.sqrt(c(t) *((kappaBar(t)-1.0) + delta + delta/(2.0*(delta + kappaBar(t)))))
    return result

def meanSqrtV_2(kappa,v0,vbar,gamma):
    """
    Result 13.45 page 424
    Further approximation for the expectation \mathbf{E}[\sqrt{v_t}] after the application of the theta method
    and by matching the value of the limits in 0, +infinity and 1.
    """
    a = np.sqrt(vbar-(gamma**2.0)/(8.0*kappa))
    b = np.sqrt(v0)-a
    temp = meanSqrtV_1(kappa,v0,vbar,gamma) # first approx for expectation
    epsilon1 = temp(1) # value from the first approx in t=1
    c = -np.log(1.0/b *(epsilon1-a))
    return lambda t: a + b *np.exp(-c*t)

if __name__ == "__main__":

    NPaths = 10000
    NSteps = 200
    T      = 5.0

    Parameters1 ={"kappa":0.8,"gamma":0.1,"v0":0.03,"vbar":0.04}
    # Parameters2 ={"kappa":1.2,"gamma":0.1,"v0":0.035,"vbar":0.02}
    # Parameters3 ={"kappa":1.2,"gamma":0.2,"v0":0.05,"vbar":0.02}
    # Parameters4 ={"kappa":0.8,"gamma":0.25,"v0":0.15,"vbar":0.1}
    # Parameters5 ={"kappa":1.0,"gamma":0.2,"v0":0.11,"vbar":0.06}

    ParV = [Parameters1]#, Parameters2, Parameters3, Parameters4, Parameters5]

    plt.figure()
    for par in ParV:
        kappa   = par["kappa"]
        gamma   = par["gamma"]
        v0      = par["v0"]
        vbar    = par["vbar"]
        PathsVolHeston = CIREuler(NPaths,NSteps,T,kappa,v0,vbar,gamma)
        time    = PathsVolHeston["time"]
        time2   = np.linspace(0.0,T,20)
        V       = PathsVolHeston["V"]
        Vsqrt   = np.sqrt(V)
        EsqrtV  = Vsqrt.mean(axis=0)
        plt.plot(time,EsqrtV,'x',markersize=1)
        approx1 = meanSqrtV_1(kappa,v0,vbar,gamma)
        approx2 = meanSqrtV_2(kappa,v0,vbar,gamma)
        plt.plot(time,approx1(time),'--r', label=f'first approx')
        plt.plot(time2,approx2(time2),'.k', label=f"second approx")
    plt.xlabel('time')
    plt.ylabel(r'$\mathbb{E}[\sqrt{v_t}]$', size=15)
    plt.legend()
    plt.grid()
    plt.show()
