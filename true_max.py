

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def goldstain(x1, x2, x3, x4):
        c1 = 1
        c2 = 1
        return -(53.3108 \
            + 0.184901 * x1 \
            - 5.02914 * x1**3 * 1e-6 \
            + 7.72522 * x1**4 * 1e-8 \
            - 0.0870775 * x2 \
            - 0.106959 * x3 \
            + 7.98772 * x3**3 * 1e-6 \
            + 0.00242482 * x4 \
            + 1.32851 * x4**3 * 1e-6 \
            - 0.00146393 * x1 * x2 \
            - 0.00301588 * x1 * x3 \
            - 0.00272291 * x1 * x4 \
            + 0.0017004 * x2 * x3 \
            + 0.0038428 * x2 * x4 \
            - 0.000198969 * x3 * x4 \
            + 1.86025 * x1 * x2 * x3 * 1e-5 \
            - 1.88719 * x1 * x2 * x4 * 1e-6 \
            + 2.50923 * x1 * x3 * x4 * 1e-5 \
            - 5.62199 * x2 * x3 * x4 * 1e-5)
        
def g_goldstain(x1,x2):
    c1 = 1
    c2 = 1
    return(c1*np.sin((x1/10)**3)+c2*np.cos((x2/20)**2))

x1_val = np.linspace(0,100,1000)
x2_val = np.linspace(0,100,1000)
x3_val = [0, 1, 2]
x4_val = [0, 1, 2]


def goldstain00(x):
    x1, x2 = x
    return goldstain(x1,x2,0,0)
def goldstain01(x):
    x1, x2 = x
    return goldstain(x1,x2,0,1)
def goldstain02(x):
    x1, x2 = x
    return goldstain(x1,x2,0,2)
def goldstain10(x):
    x1, x2 = x
    return goldstain(x1,x2,1,0)
def goldstain11(x):
    x1, x2 = x
    return goldstain(x1,x2,1,1)
def goldstain12(x):
    x1, x2 = x
    return goldstain(x1,x2,1,2)
def goldstain20(x):
    x1, x2 = x
    return goldstain(x1,x2,2,0)
def goldstain21(x):
    x1, x2 = x
    return goldstain(x1,x2,2,1)
def goldstain22(x):
    x1, x2 = x
    return goldstain(x1,x2,2,2)


x0 = [10, 30]
bounds = [(0,100), (0, 100)]
constraints  = [ {'type': 'ineq', 'fun': lambda x: np.sin((x[0]/10)**3)+np.cos((x[1]/20)**2)}]

res00 = minimize(goldstain00, x0, bounds = bounds, constraints = constraints, method = 'L-BFGS-B')
res01 = minimize(goldstain01, x0, bounds = bounds, constraints = constraints, method = 'L-BFGS-B')
res02 = minimize(goldstain02, x0, bounds = bounds, constraints = constraints, method = 'L-BFGS-B')
res10 = minimize(goldstain10, x0, bounds = bounds, constraints = constraints, method = 'L-BFGS-B')
res11 = minimize(goldstain11, x0, bounds = bounds, constraints = constraints, method = 'L-BFGS-B')
res12 = minimize(goldstain12, x0, bounds = bounds, constraints = constraints, method = 'L-BFGS-B')
res20 = minimize(goldstain20, x0, bounds = bounds, constraints = constraints, method = 'L-BFGS-B')
res21 = minimize(goldstain21, x0, bounds = bounds, constraints = constraints, method = 'L-BFGS-B')
res22 = minimize(goldstain22, x0, bounds = bounds, constraints = constraints, method = 'L-BFGS-B')

xmax00 = res00.x
fmax00 = -res00.fun
print('xmax00 ', xmax00)
print('fmax00 ', fmax00)

xmax01 = res01.x
fmax01 = -res01.fun
print('xmax01 ', xmax01)
print('fmax01 ', fmax01)

xmax02 = res02.x
fmax02 = -res02.fun
print('xmax02 ', xmax02)
print('fmax02 ', fmax02)

xmax10 = res10.x
fmax10 = -res10.fun
print('xmax10 ', xmax10)
print('fmax10 ', fmax10)

xmax11 = res11.x
fmax11 = -res11.fun
print('xmax11 ', xmax11)
print('fmax11 ', fmax11)

xmax12 = res12.x
fmax12 = -res12.fun
print('xmax12 ', xmax12)
print('fmax12 ', fmax12)

xmax20 = res20.x
fmax20 = -res20.fun
print('xmax20 ', xmax20)
print('fmax20 ', fmax20)

xmax21 = res21.x
fmax21 = -res21.fun
print('xmax21 ', xmax21)
print('fmax21 ', fmax21)

xmax22 = res22.x
fmax22 = -res22.fun
print('xmax22 ', xmax22)
print('fmax22 ', fmax22)
