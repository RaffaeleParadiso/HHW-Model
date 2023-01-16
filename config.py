import numpy as np
import enum

class OptionType(enum.Enum):
    CALL = 1.0
    PUT = -1.0

def OptionPriceFromMonteCarlo(CP,S,K,M):
    # S is a vector of Monte Carlo samples at T
    result = np.zeros([len(K),1])
    if CP == OptionType.CALL:
        for (idx,k) in enumerate(K):
            result[idx] = np.mean(1.0/M*np.maximum(S-k,0.0))
    elif CP == OptionType.PUT:
        for (idx,k) in enumerate(K):
            result[idx] = np.mean(1.0/M*np.maximum(k-S,0.0))
    return result

CP  = OptionType.CALL
NPaths = 10000 # #of paths for MC Euler and AES
NSteps = 500   # #of time steps for MC Euler and AES
lambd  = 0.05  #
eta    = 0.005 #
S0     = 100.0 #
T      = 1.5     # Time until maturity (years)
r      = 0.07  # Initial interest rate


# K = np.linspace(.01,2*S0*np.exp(r*T),50)
K = np.arange(50,151,5)
K = np.array(K).reshape([len(K),1]) # Strikes prices (array of)

P0T = lambda T: np.exp(-r*T)  # ZCB curve with maturity T (obtained from the market)

gamma  =  0.0571
vbar   =  0.0398
v0     =  0.0175
rhoxr  =  0.2
rhoxv  = -0.5711
kappa  =  1.5768

# Settings for the COS method
N      = 1500 # #of contribute in the cosine expansion for the characteristic function
L      = 10   # Truncation domain from [-L\sqrt(T), +L\sqrt(T)]
