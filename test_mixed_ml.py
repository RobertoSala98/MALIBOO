from maliboo.bayesian_optimization import BayesianOptimization as BO
import pandas as pd
import os
import time
import numpy as np
from test import perform_test
import matplotlib.pyplot as plt
from pathlib import Path
import pdb



seed = 42
debug = False


#####################################################
##################  Goldstain  ######################
#####################################################


@perform_test
def test_goldstain_ml(output_path):
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
        
        ret['value'] = -(53.3108 \
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
            
        ret['field'] = "idk"    
        ret['blackbox'] = c1*np.sin((x1/10)**3)+c2*np.cos((x2/20)**2)
        return ret
    
 
    dataset_discrete = pd.DataFrame({'z1': [0, 0, 0, 1, 1, 1, 2, 2, 2], 'z2': [0, 1, 2, 0, 1, 2, 0, 1, 2]})
    
    for i in range(10):
        optimizer = BO(f=goldstain, pbounds={'x1': (0, 100), 'x2': (0, 100), 'z1': (0, 2), 'z2': (0, 2)}, 
                       debug = debug, dataset_discrete = dataset_discrete, 
                       output_path=output_path + '_epsgreedy_probontarget_' + str(i), 
                       true_maximum_value=-38.11)

        optimizer.maximize(init_points=5, n_iter=15, acq='ei', ml_on_bounds=True, ml_on_target=True, epsilon_greedy=True, 
                        acq_info={'ml_target': 'blackbox', 'ml_bounds': (0, float('inf')), 'ml_bounds_type':'indicator',
                                  'ml_bounds_model':'Ridge', 'ml_bounds_alpha':float(0.5), 'ml_target_type':'probability',
                                  'ml_target_model':'Ridge', 'ml_target_alpha':float(0.025), 'ml_target_coeff':[float(1.0), None],
                                  'eps_greedy_random_prob':float(0.1)}
                        )
        
        #plot_regret(output_path=output_path + '_' + str(i) + "/results.csv")




#####################################################
####################  Branin  #######################
#####################################################


@perform_test
def test_branin_ml(output_path):
   
    def h(x1, x2):
        term1 = (15 * x2 - (5 / (4 * np.pi**2)) * (15 * x1 - 5)**2 + (5 / np.pi) * (15 * x1 - 5) - 6)**2
        term2 = 10 * (1 - 1 / (8 * np.pi)) * np.cos(15 * x1 - 5) + 10
        return (term1 + term2 - 54.8104) * (1 / 51.9496)
 
    def branin(x1, x2, z1, z2):
        # the function defined here is actually -branin.
        # Since the optimization we want to solve is min(branin), we define -branin and solve
        # max(-branin)
        ret = {}
        if z1 == 0 and z2 == 0:
            ret['value'] = -h(x1, x2)
            ret['blackbox'] = x1 * x2 - 0.4
        elif z1 == 0 and z2 == 1:
            ret['value'] = -0.4 * h(x1, x2)
            ret['blackbox'] = 1.5 * x1 * x2 - 0.4
        elif z1 == 1 and z2 == 0:
            ret['value'] = -(-0.75 * h(x1, x2) + 3.0)
            ret['blackbox'] = 1.5 * x1 * x2 - 0.2
        elif z1 == 1 and z2 == 1:
            ret['value'] = -(-0.5 * h(x1, x2) + 1.4)
            ret['blackbox'] = 1.2 * x1 * x2 - 0.3
        else:
            raise ValueError("Error! This function can be evaluated only for z1,z2 in {0, 1}")
        return ret
   
    dataset_discrete = pd.DataFrame({'z1': [0, 0, 1, 1], 'z2': [0, 1, 0, 1]})
    
    for i in range(10):
        
        optimizer = BO(f=branin, pbounds={'x1': (0, 1), 'x2': (0, 1), 'z1': (0, 1), 'z2': (0, 1)}, debug = debug, 
                       dataset_discrete = dataset_discrete, output_path=output_path + '_indicator_' + str(i)
                    , true_maximum_value=0.81439)

        
        optimizer.maximize(init_points=5, n_iter=45, acq='ei', ml_on_bounds=True, ml_on_target=True, epsilon_greedy=True,
                        acq_info={'ml_target': 'blackbox', 'ml_bounds': (0, float('inf')), 'ml_bounds_type':'indicator', 
                                  'ml_bounds_model':'Ridge', 'ml_bounds_alpha':float(0.5),  # it was set to 1
                                  'ml_target_type':'probability',
                                  'ml_target_model':'Ridge', 'ml_target_alpha':float(0.025), 'ml_target_coeff':[float(1.0), None],
                                  'eps_greedy_random_prob':float(0.1)}, 
                        )
        

#test_goldstain_ml()
#test_branin_ml()

from plots import plot_average

plot_average('test_branin_ml_indicator', 
             true_opt_value=-0.81439, 
             output_name='branin_avg', 
             optimization_type='min', 
             init_points=5, 
             plot_quantiles=True, 
             avg_label='branin_all_active')
