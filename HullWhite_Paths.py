import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib import pylab
from pylab import *
pylab.rcParams['figure.figsize'] = (13, 4)

def HWEuler(N_Paths,N_Steps,T,P0T, lamb, eta):    
    dt = 0.0001    
    f_zero_T = lambda t: - (np.log(P0T(t+dt))-np.log(P0T(t-dt)))/(2*dt)
    r0 = f_zero_T(dt)
    theta = lambda t: 1.0/lamb * (f_zero_T(t+dt)-f_zero_T(t-dt))/(2.0*dt) + f_zero_T(t) + eta*eta/(2.0*lamb*lamb)*(1.0-np.exp(-2.0*lamb*t))      
    Z = np.random.normal(0.0,1.0,[N_Paths,N_Steps])
    W = np.zeros([N_Paths, N_Steps+1])
    R = np.zeros([N_Paths, N_Steps+1])
    R[:,0]=r0
    time = np.zeros([N_Steps+1]) 
    dt = T / float(N_Steps)
    for i in range(0,N_Steps):
        if N_Paths > 1:
            Z[:,i] = (Z[:,i] - np.mean(Z[:,i])) / np.std(Z[:,i])
        W[:,i+1] = W[:,i] + np.power(dt, 0.5)*Z[:,i]
        R[:,i+1] = R[:,i] + lamb*(theta(time[i]) - R[:,i]) * dt + eta* (W[:,i+1]-W[:,i])
        time[i+1] = time[i] + dt
    paths = {"time":time,"R":R}
    return paths

if __name__ == "__main__":
    save_figure = False

    N_Paths = 1
    N_Steps = 2000
    T       = 5.0
    lamb    = 0.5
    eta     = 0.01
    P0T     = lambda T: np.exp(-0.05*T)

    plt.figure() 
    legend = []
    lambVec = [-0.2, 0.2, 5.0]
    color = ["blue", "red", "green"]
    for idx,lamb in enumerate(tqdm(lambVec)):    
        np.random.seed(1)
        Paths = HWEuler(N_Paths,N_Steps,T,P0T, lamb, eta)
        legend.append(f'Lambda={lamb:.2f}')
        timeGrid = Paths["time"]
        R = Paths["R"]
        plt.plot(timeGrid, np.transpose(R), color=color[idx])   
    plt.grid()
    plt.title(rf"Effect of mean reversion Lambda $\lambda$")
    plt.xlabel("time")
    plt.ylabel("R(t)")
    plt.legend(legend)
    if save_figure: plt.savefig("HullWhiteLambda.png",bbox_inches='tight')

    plt.figure()    
    legend = []
    etaVec = [0.1, 0.2, 0.3]
    for idx, etat in enumerate(tqdm(etaVec)):
        np.random.seed(1)
        Paths = HWEuler(N_Paths,N_Steps,T,P0T, lamb, etat)
        legend.append(f'Eta={eta:.2f}')
        timeGrid = Paths["time"]
        R = Paths["R"]
        plt.plot(timeGrid, np.transpose(R), color=color[idx])   
    plt.grid()
    plt.title(rf"Effect of the Volatility $\eta$")
    plt.xlabel("time")
    plt.ylabel("R(t)")
    plt.legend(legend, loc='upper left')
    if save_figure: plt.savefig("HullWhiteVolatility.png",bbox_inches='tight')
    
    plt.show()
