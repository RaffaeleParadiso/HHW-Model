import numpy as np
import enum

class OptionType(enum.Enum):
    CALL = 1.0
    PUT = -1.0

CP  = OptionType.CALL

NPaths = 10000 # n° of paths for MC Euler and AES
NSteps = 1001  # n° of time steps for MC Euler and AES
lambd  = 0.05  # 
eta    = 0.005 #
S0     = 100.0 # Initial stock price
T      = 1   # Time until maturity (years)
r      = 0.07  # Initial interest rate

# K = np.linspace(.01,2*S0*np.exp(r*T),50)
K = np.arange(50,131,1) # OTM/ATM/ITM options
K = np.array(K).reshape([len(K),1]) # Array of strike prices

P0T = lambda T: np.exp(-r*T)  # ZCB curve with maturity T (usually obtained from the market)

gamma  =  0.0571    
vbar   =  0.0398    # long-run mean value for the volatility process
v0     =  0.0175    # initial volatility value
rhoxr  =  0.2       # correlation between log stocks and interest rate
rhoxv  = -0.5711    # correlation between log stocks and volatility
kappa  =  1.5768    # mean reversion rate velocity for the volatility process 

# Settings for the COS method
N      = 150000 # n° of contribute (N terms) in the cosine expansion for the characteristic function
L      = 8   # Truncation domain [-L*\sqrt(\tau), +L*\sqrt(\tau)]  (\tau = T - t_0)
