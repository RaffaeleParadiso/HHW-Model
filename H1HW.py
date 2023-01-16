import time
import numpy as np
import matplotlib.pyplot as plt
from config import *
from CHF import ChFH1HWModel, CallPutCoefficients # characteristic function
from HHW_AES import GeneratePathsHestonHW_AES     # almost exact simulation
from HHW_MC import GeneratePathsHestonHWEuler     # standard euler mode


def OptionPriceFromMonteCarlo(CP,S,K,M):
    """
    """
    # S is a vector of Monte Carlo samples at T
    result = np.zeros([len(K),1])
    if CP == OptionType.CALL:
        for (idx,k) in enumerate(K):
            result[idx] = np.mean(1.0/M*np.maximum(S-k,0.0))
    elif CP == OptionType.PUT:
        for (idx,k) in enumerate(K):
            result[idx] = np.mean(1.0/M*np.maximum(k-S,0.0))
    return result

def OptionPriceFromCOS(cf,CP,S0,tau,K,N,L,P0T):
    """
    Calculate the price for the list of strikes provided with the COS method.
    Parameters
    ----------
    cf :    
        Characteristic function.
    CP :
        C for call and P for put.
    S0 : 
        Initial stock price.
    tau :  
        Time to maturity.
    K :    
        List of strikes.
    N :    
        Number of expansion terms.
    L :    
        Size of truncation domain (L=8 or L=10).
    P0T : 
        Zero-coupon bond for maturity T.

    Returns
    -------
    value : ndarray
        see dtype parameter above.
    """
    if K is not np.array: K = np.array(K).reshape([len(K),1])   
    i = complex(0.0,1.0) 
    x0 = np.log(S0 / K)   
    # Truncation domain
    a = 0.0 - L * np.sqrt(tau)
    b = 0.0 + L * np.sqrt(tau) 
    # Summation from k = 0 to k=N-1
    k = np.linspace(0,N-1,N).reshape([N,1])  
    u = k * np.pi / (b - a)  
    # Determine coefficients for put prices  
    H_k = CallPutCoefficients(OptionType.PUT,a,b,k)   
    mat = np.exp(i * np.outer((x0 - a) , u))
    temp = cf(u) * H_k 
    temp[0] = 0.5 * temp[0]    
    value = K * np.real(mat.dot(temp))     
    # We use the put-call parity for call options
    if CP == OptionType.CALL:
        value = value + S0 - K * P0T   
    return value


if __name__ == "__main__":

    euler = 1
    aes = 1
    cos = 1
    figure = 1

    if euler:
        start = time.time()
        np.random.seed(1)
        paths = GeneratePathsHestonHWEuler(NPaths,NSteps,P0T,T,S0,kappa,gamma,rhoxr,rhoxv,vbar,v0,lambd,eta)
        S_n = paths["S"]
        M_t_n = paths["M_t"]
        # print(np.mean(S_n[:,-1]/M_t_n[:,-1]))
        valueOptMC= OptionPriceFromMonteCarlo(CP,S_n[:,-1],K,M_t_n[:,-1])
        np.savetxt("MC.txt", valueOptMC, fmt='%.4f')
        print(f"Time elapsed for Euler MC: {time.time() - start} for {len(K)} strikes")

    if aes:
        start = time.time()
        np.random.seed(1)
        pathsExact = GeneratePathsHestonHW_AES(NPaths,NSteps,P0T,T,S0,kappa,gamma,rhoxr,rhoxv,vbar,v0,lambd,eta)
        S_ex = pathsExact["S"]
        M_t_ex = pathsExact["M_t"]
        # print(np.mean(S_ex[:,-1]/M_t_ex[:,-1]))
        valueOptMC_ex= OptionPriceFromMonteCarlo(CP,S_ex[:,-1],K,M_t_ex[:,-1])
        np.savetxt("AES.txt", valueOptMC_ex, fmt='%.4f')
        print(f"Time elapsed for AES MC: {time.time() - start} for {len(K)} strikes")

    if cos:
        start = time.time()
        cf2 = ChFH1HWModel(P0T,lambd,eta,T,kappa,gamma,vbar,v0,rhoxv, rhoxr)
        # u=np.array([1.0,2.0,3.0])
        # print(cf2(u))
        valCOS = OptionPriceFromCOS(cf2, CP, S0, T, K, N, L,P0T(T))
        np.savetxt("COS.txt", valCOS, fmt='%.4f')
        print(f"Time elapsed for COS Method: {time.time() - start} for {len(K)} strikes")
        

    if figure:
        plt.figure(1)
        plt.plot(K,valueOptMC)
        plt.plot(K,valueOptMC_ex,'.k')
        plt.plot(K,valCOS,'--r')
        # plt.ylim([0.0,110.0])
        plt.legend(['Euler','AES','COS'])
        plt.xlabel('Strike, K')
        plt.ylabel('EU Option Value')
        plt.grid()
        # plt.savefig("img/MC_vs_AES_vs_COS.png",bbox_inches='tight')
        plt.show()
