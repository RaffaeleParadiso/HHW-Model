"""
Normal and LogNormal density recovery using the COS method
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st

def COSDensity(cf,x,N,a,b):
    i = complex(0.0,1.0)
    k = np.linspace(0,N-1,N)
    u = np.zeros([1,N])
    u = k * np.pi / (b-a)  
    F_k = 2.0 / (b - a) * np.real(cf(u) * np.exp(-i * u * a))
    F_k[0] = F_k[0] * 0.5
    f_X = np.matmul(F_k , np.cos(np.outer(u, x - a )))
    return f_X
    
if __name__ == "__main__":
    i = complex(0.0, 1.0)
    a = -10
    b = 10
    N = [16, 64, 128]
    mu = 0.5
    sigma = 0.2  
    cF = lambda u : np.exp(i * mu * u - 0.5 * np.power(sigma,2.0) * np.power(u,2.0))
    y = np.linspace(0.05,5,1000) 

    plt.figure()
    plt.grid()
    plt.xlabel("y")
    plt.ylabel("$f_Y(y)$")
    for n in N:
        f_Y = 1/y * COSDensity(cF,np.log(y),n,a,b)
        plt.plot(y,f_Y)
    plt.legend(["N=%.0f"%N[0],"N=%.0f"%N[1],"N=%.0f"%N[2]])
    plt.show()

    N = [2**x for x in range(2,7,1)]
    mu = 0.0
    sigma = 1.0
    x = np.linspace(-10.0,10,1000)
    f_XExact = st.norm.pdf(x,mu,sigma)

    plt.figure()
    plt.grid()
    plt.xlabel("x")
    plt.ylabel("$f_X(x)$")
    for n in N:
        f_X = COSDensity(cF,x,n,a,b)
        error = np.max(np.abs(f_X-f_XExact))
        print("For {0} expanansion terms the error is {1}".format(n,error))
        plt.plot(x,f_X)
    plt.show()
