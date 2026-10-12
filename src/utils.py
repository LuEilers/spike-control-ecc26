import numpy as np

# Calculate theoretical upper bounds for the state x

def upperbound(a, eps0, theta, lam, x0, t):
    b = eps0/theta
    alph = a - b
    return np.abs(x0) * np.exp(alph/2*t) + eps0*(1+np.abs(alph+lam)/np.abs(alph))

def upperbound_improved(a, eps0, theta, lam, x0, t):
    b = eps0/theta
    alph = a - b
    return np.abs(x0) * np.exp(alph*t) + eps0*1

