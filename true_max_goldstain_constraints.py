import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import pdb


def convert_to_discrete(z1, z2):
    # Definizione delle categorie discrete per x3 e x4
    if z1 == 0 and z2 == 0:
        x3, x4, c1, c2 = 20, 20, 2, 0.5
    elif z1 == 0 and z2 == 1:
        x3, x4, c1, c2 = 20, 50, 2, -1
    elif z1== 0 and z2 == 2:
        x3, x4, c1, c2 = 20, 80, 2, -2
    elif z1 == 1 and z2 == 0:
        x3, x4, c1, c2 = 50, 20, -2, 0.5
    elif z1 == 1 and z2 == 1:
        x3, x4, c1, c2 = 50, 50, -2, -1
    elif z1 == 1 and z2 == 2:
        x3, x4, c1, c2 = 50, 80, -2, -2
    elif z1 == 2 and z2 == 0:
        x3, x4, c1, c2 = 80, 20, 1, 0.5
    elif z1== 2 and z2 == 1:
        x3, x4, c1, c2 = 80, 50, 1, -1
    elif z1 == 2 and z2 == 2:
        x3, x4, c1, c2 = 80, 80, 1, -2
    else:
        raise ValueError("Valori di z1 o z2 fuori intervallo!")
    return x3, x4, c1, c2
    

def goldstain(x1, x2, z1, z2):
    ret = {}
    x3, x4, c1, c2 = convert_to_discrete(z1, z2)
        
    ret['value'] = 53.3108 \
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
        - 5.62199 * x2 * x3 * x4 * 1e-5       
    
    ret['blackbox'] = c1*np.sin((x1/10)**3)+c2*np.cos((x2/20)**2)
    return ret    

x1_val = np.linspace(0, 100, 2000)
x2_val = np.linspace(0, 100, 2000)
X1, X2 = np.meshgrid(x1_val, x2_val)

z1_vals = [0,1,2]
z2_vals = [0,1,2]

fig, axes = plt.subplots(np.size(z1_vals), np.size(z2_vals), figsize=(10, 10))
min_x1 = np.zeros(np.size(z1_vals)*np.size(z2_vals))
min_x2 = np.zeros(np.size(z1_vals)*np.size(z2_vals))
min_z = np.zeros(np.size(z1_vals)*np.size(z2_vals))

for i,z1 in enumerate(z1_vals):
  for j,z2 in enumerate(z2_vals):
    def constraint_function(x):
            x1, x2 = x
            return goldstain(x1, x2, z1, z2)['blackbox']
    def target_function(x):
            x1, x2 = x
            return goldstain(x1, x2, z1, z2)['value']        
    ax = axes[i, j]
    F = np.vectorize(lambda x1, x2: target_function([x1, x2]))(X1, X2)
    Z = np.vectorize(lambda x1, x2: constraint_function([x1, x2]))(X1, X2)
    Z_binary = (Z>0).astype(int)
    min_index = np.argmin(F[(Z>0)])
    min_x1[i*np.size(z1_vals)+j], min_x2[i*np.size(z1_vals)+j], min_z[i*np.size(z1_vals)+j] = (X1[(Z>0)])[min_index], (X2[(Z>0)])[min_index], (F[(Z>0)])[min_index]
    
    contour = ax.contourf(X1, X2, Z_binary, levels=1, cmap="bwr")
    ax.set_title("Constraint Function Values")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")

plt.tight_layout()
cbar = fig.colorbar(contour, ax=axes, orientation="horizontal", fraction=0.05, pad=0.1)
cbar.set_label("Constraint Value")

plt.show()

print(min_x1)
print(min_x2)
print(min_z)

plt.savefig('goldstain_constraint.png')

#optimum x = [94.24712356, 98.14907454, 2, 2] f(x) = 38.11340175

#[70.28514257  8.10405203 78.58929465 77.28864432 89.89494747 85.29264632
 #82.44122061 87.94397199 94.24712356]
#[100.           0.           2.30115058 100.         100.
#   3.2016008    0.           2.60130065  98.14907454]
#[49.17861332 51.43116346 48.27748025 47.06305414 45.91549622 42.9685138
# 43.50824191 41.45802338 38.11340175]

