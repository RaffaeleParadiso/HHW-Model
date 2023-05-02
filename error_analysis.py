import time

import matplotlib.pyplot as plt
import numpy as np

from utils.config import *
from utils.HHW_AES import GeneratePathsHestonHW_AES  # almost exact simulation
from utils.HHW_CHF import ChFH1HWModel, Chi_Psi  # characteristic function
from utils.HHW_MC import HHW_Euler  # standard euler mode
