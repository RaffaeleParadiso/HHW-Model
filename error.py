import time
import numpy as np
import matplotlib.pyplot as plt
from config import *
from HHW_CHF import ChFH1HWModel                      # characteristic function
from HHW_CHF import CallPutCoefficients               # characteristic function
from HHW_AES import GeneratePathsHestonHW_AES         # almost exact simulation
from HHW_MC import HHW_Euler                          # standard euler mode
from main import OptionPriceFromMonteCarlo
from main import OptionPriceFromCOS


np.random.seed(1)
set_params = (P0T,T,kappa,gamma,rhoxr,rhoxv,vbar,v0,lambd,eta)
Nstepss = [101, 201, 301, 401, 501]
value_euler = []
value_AES = []
for NStepss in Nstepss:
    
    paths = HHW_Euler(NPaths,NStepss,S0, set_params)
    S_n = paths["S"]
    M_t_n = paths["M_t"]
    valueOptMC= OptionPriceFromMonteCarlo(CP,S_n[:,-1],K,M_t_n[:,-1])
    value_euler.append(valueOptMC[0])
    pathsExact = GeneratePathsHestonHW_AES(NPaths,NStepss,S0,set_params)
    S_ex = pathsExact["S"]
    M_t_ex = pathsExact["M_t"]
    valueOptMC_ex= OptionPriceFromMonteCarlo(CP,S_ex[:,-1],K,M_t_ex[:,-1])
    value_AES.append(valueOptMC_ex[0])

print(value_euler)
print(value_AES)