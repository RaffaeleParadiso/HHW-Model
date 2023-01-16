import numpy as np
import enum

class OptionType(enum.Enum):
    CALL = 1.0
    PUT = -1.0

CP  = OptionType.CALL
NPaths = 10000 # #of paths for MC Euler and AES
NSteps = 501   # #of time steps for MC Euler and AES
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
