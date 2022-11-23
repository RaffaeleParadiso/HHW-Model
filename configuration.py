import numpy as np
import enum

class OptionType(enum.Enum):
    CALL = 1.0
    PUT = -1.0

def EUOptionPriceFromMCPathsGeneralizedStochIR(CP,S,K,M):
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
NPaths = 10000
NSteps = 500
lambd  = 0.5
eta    = 0.01
S0     = 100.0
T      = 1    
r      = 0.03

K = np.linspace(.01,2*S0*np.exp(r*T),50)
K = np.array(K).reshape([len(K),1])
P0T = lambda T: np.exp(-r*T)  # We define a ZCB curve (obtained from the market)

gamma  =  0.06
vbar   =  0.05
v0     =  0.02
rhoxr  =  0.5
rhoxv  = -0.7
kappa  =  0.5

# Settings for the COS method
N      = 15000
L      = 20 