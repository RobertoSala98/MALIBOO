import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import pdb


   
def h(x1, x2):
    term1 = (15 * x2 - (5 / (4 * np.pi**2)) * (15 * x1 - 5)**2 + (5 / np.pi) * (15 * x1 - 5) - 6)**2
    term2 = 10 * (1 - 1 / (8 * np.pi)) * np.cos(15 * x1 - 5) + 10
    return (term1 + term2 - 54.8104) * (1 / 51.9496)

def branin(x1, x2, z1, z2):
    ret = {}
    if z1 == 0 and z2 == 0:
        ret['value'] = h(x1, x2)
        ret['blackbox'] = x1 * x2 - 0.4
    elif z1 == 0 and z2 == 1:
        ret['value'] = 0.4 * h(x1, x2)
        ret['blackbox'] = 1.5 * x1 * x2 - 0.4
    elif z1 == 1 and z2 == 0:
        ret['value'] = -0.75 * h(x1, x2) + 3.0
        ret['blackbox'] = 1.5 * x1 * x2 - 0.2
    elif z1 == 1 and z2 == 1:
        ret['value'] = -0.5 * h(x1, x2) + 1.4
        ret['blackbox'] = 1.2 * x1 * x2 - 0.3
    else:
        raise ValueError("Error! This function can be evaluated only for z1,z2 in {0, 1}")
    return ret
   

z1_vals = [0, 1]
z2_vals = [0, 1]

for z1 in z1_vals:  
    for z2 in z2_vals: 
        
        def objective_function(x):
            x1, x2 = x
            return branin(x1, x2, z1, z2)['value']
        
        def constraint_function(x):
            x1, x2 = x
            return branin(x1, x2, z1, z2)['blackbox']
        
        nofeasible_initial = False
        initial_guess = [np.random.uniform(0, 1), np.random.uniform(0, 1)]
        if(constraint_function(initial_guess)<0):
            nofeasible_initial=True
        while(nofeasible_initial):
            initial_guess = [np.random.uniform(0, 1), np.random.uniform(0, 1)]
            if(constraint_function(initial_guess)>=0):
                nofeasible_initial=False
                
        constraint = {'type': 'ineq', 'fun': constraint_function}  
        
        result = minimize(objective_function, initial_guess,
                          bounds=[(0, 1), (0, 1)],  method='SLSQP', constraints = constraint)

        # print the current solution
        if result.success:
            print(f"Solution for z1 = {z1}, z2 = {z2}:")
            print(f"x1 = {result.x[0]:.4f}, x2 = {result.x[1]:.4f}")
            print(f"Value of the objective function: {result.fun:.4f}")
            # print the value of the constraint",
            constraint_value = constraint_function(result.x)
            print(f'constraint value: {constraint_value}')
            print()
            
        else:
            print(f"optimization failed for z1 = {z1_val}, z2 = {z2_val}.")
            print()

#Solution for z1 = 0, z2 = 0:
#x1 = 1.0000, x2 = 0.4000
#Value of the objective function: -0.8143
#Valore del vincolo: 1.1309597702791052e-10