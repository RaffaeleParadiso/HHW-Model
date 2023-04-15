import time

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st

from utils.config import *
from utils.HHW_CHF import ChFH1HWModel, Chi_Psi      # characteristic function
from utils.HHW_AES import GeneratePathsHestonHW_AES  # almost exact simulation
from utils.HHW_MC import HHW_Euler                   # standard euler mode


def OptionPriceFromMonteCarlo(CP,S,K,M):
    """
    S is a vector of Monte Carlo samples at T
    """
    result = np.zeros([len(K),1])
    if CP == OptionType.CALL:
        for (idx,k) in enumerate(K):
            result[idx] = np.mean(1.0/M*np.maximum(S-k,0.0))
    elif CP == OptionType.PUT:
        for (idx,k) in enumerate(K):
            result[idx] = np.mean(1.0/M*np.maximum(k-S,0.0))
    return result

def CallPutCoefficients(CP,a,b,k):
    """
    Determine coefficients chi and psi for call/put prices
    """
    if CP==OptionType.CALL:                  
        c = 0.0
        d = b
        coef  = Chi_Psi(a,b,c,d,k)
        Chi_k = coef["chi"]
        Psi_k = coef["psi"]
        if a < b and b < 0.0:
            H_k = np.zeros([len(k),1])
        else:
            H_k = 2.0 / (b - a) * (Chi_k - Psi_k)  
    elif CP==OptionType.PUT:
        c = a
        d = 0.0
        coef = Chi_Psi(a,b,c,d,k)
        Chi_k = coef["chi"]
        Psi_k = coef["psi"]
        H_k   = 2.0 / (b - a) * (- Chi_k + Psi_k)               
    return H_k

def OptionPriceFromCOS(cf,CP,S0,tau,K,N,L,P0T):
    """
    Calculate the price for the list of strikes provided with the COS method.
    Parameters
    ----------
    cf : lambda function
        Characteristic function.
    CP : class option type
        CALL for call and PUT for put.
    S0 : float
        Initial stock price.
    tau : float
        Time to maturity.
    K : ndarray of float
        Array of strikes.
    N : int
        Number of expansion terms.
    L : int
        Size of truncation domain (L=8 or L=10).
    P0T : lambda function
        Zero-coupon bond for maturity T.

    Returns
    -------
    value : ndarray of float
        Array of prices.
    """
    if K is not np.array: K = np.array(K).reshape([len(K),1])
    i = complex(0.0,1.0)
    x0 = np.log(S0 / K)
    # Truncation domain
    a = - L * np.sqrt(tau)
    b = + L * np.sqrt(tau)
    # Summation from k = 0 to k=N-1
    k = np.linspace(0,N-1,N).reshape([N,1])
    u = k * np.pi / (b - a)
    H_k = CallPutCoefficients(OptionType.PUT,a,b,k)
    mat = np.exp(i * np.outer((x0 - a) , u))
    temp = cf(u) * H_k
    temp[0] = 0.5 * temp[0]
    value = K * np.real(mat.dot(temp))
    if CP == OptionType.CALL:
        # Put-call parity for call options
        value = value + S0 - K * P0T
    return value

def BS(CP ,S_0 : float ,K,sigma,t,T,r) -> float:
    """
    """
    K = np.array(K).reshape([len(K),1])
    d1    = (np.log(S_0 / K) + (r + 0.5 * np.power(sigma,2.0)) 
    * (T-t)) / (sigma * np.sqrt(T-t))
    d2    = d1 - sigma * np.sqrt(T-t)
    if CP == OptionType.CALL:
        value = st.norm.cdf(d1) * S_0 - st.norm.cdf(d2) * K * np.exp(-r * (T-t))
    elif CP == OptionType.PUT:
        value = st.norm.cdf(-d2) * K * np.exp(-r * (T-t)) - st.norm.cdf(-d1)*S_0
    return value

if __name__ == "__main__":

    EULER = True
    AES = True
    COS = True
    
    FIGURE = True
    SAVE = False
    
    print("="*60)

    if EULER:
        set_params = (P0T,T,kappa,gamma,rhoxr,rhoxv,vbar,v0,lambd,eta)
        start = time.time()
        np.random.seed(1)
        paths = HHW_Euler(NPaths,NSteps,S0, set_params)
        S_n = paths["S"]
        M_t_n = paths["M_t"]
        valueOptMC= OptionPriceFromMonteCarlo(CP,S_n[:,-1],K,M_t_n[:,-1])
        if SAVE: np.savetxt("MC.txt", valueOptMC, fmt='%.4f')
        print(f"Time elapsed for Euler MC  : {(time.time() - start):.3f} seconds for {len(K)} strikes")
        print(f"Price MC_Euler Martingala  : {(np.mean(S_n[:,-1]/M_t_n[:,-1])):.2f}")
        print("="*60)

    if AES:
        set_params = (P0T,T,kappa,gamma,rhoxr,rhoxv,vbar,v0,lambd,eta)
        start = time.time()
        np.random.seed(1)
        pathsExact = GeneratePathsHestonHW_AES(NPaths,NSteps,S0,set_params)
        S_ex = pathsExact["S"]
        M_t_ex = pathsExact["M_t"]
        valueOptMC_ex= OptionPriceFromMonteCarlo(CP,S_ex[:,-1],K,M_t_ex[:,-1])
        if SAVE: np.savetxt("AES.txt", valueOptMC_ex, fmt='%.4f')
        print(f"Time elapsed for AES MC    : {(time.time() - start):.3f} seconds for {len(K)} strikes")
        print(f"Price MC_AES Martingala    : {(np.mean(S_ex[:,-1]/M_t_ex[:,-1])):.2f}")
        print("="*60)

    if COS:
        set_params = (P0T,T,kappa,gamma,rhoxr,rhoxv,vbar,v0,lambd,eta)
        start = time.time()
        cf2 = ChFH1HWModel(set_params)
        # u=np.array([1.0,2.0,3.0])
        # print(cf2(u))
        valCOS = OptionPriceFromCOS(cf2, CP, S0, T, K, N, L,P0T(T))
        if SAVE: np.savetxt("COS.txt", valCOS, fmt='%.4f')
        print(f"Time elapsed for COS Method: {(time.time() - start):.3f} seconds for {len(K)} strikes")
        print("="*60)


    value_BS = BS(CP,S0,K,0.05,0,T,r)

    if FIGURE:
        plt.figure()
        plt.plot(K,valueOptMC)
        plt.plot(K,valueOptMC_ex,'.k')
        plt.plot(K,valCOS,'--r')
        plt.plot(K,value_BS, '--g')
        # plt.ylim([0.0,60.0])
        plt.legend(['Euler','AES','COS', 'Black Scholes'])
        plt.xlabel('Strike, K')
        plt.ylabel('EU Option Value')
        plt.grid()
        if SAVE: plt.savefig("MC_vs_AES_vs_COS.png",bbox_inches='tight')
        plt.show()
