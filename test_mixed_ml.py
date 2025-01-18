from maliboo.bayesian_optimization import BayesianOptimization as BO
import pandas as pd
import os
import time
import numpy as np
from test import perform_test
import matplotlib.pyplot as plt
from pathlib import Path
from test_mixed.py import plot_regret, compare_regret

def target_func(x1, x2):
    return -x1 ** 2 - (x2 - 1) ** 2 + 2


seed = 42
debug = False


@perform_test
def test_mixed_2d(output_path):
    dataset_discrete = pd.DataFrame({'x2': [1, 2, 3]})
    optimizer = BO(f=target_func, pbounds={'x1': (-2, 4), 'x2': (-3, 3)},
                 random_state=seed, debug = debug, dataset_discrete = dataset_discrete, output_path=output_path,
                 true_maximum_value=2)

    optimizer.maximize(init_points = 2, n_iter = 10, acq = 'ei')

@perform_test
def test_continuos(output_path):
    optimizer = BO(f=target_func, pbounds={'x1': (-2, 4), 'x2': (-3, 3)},
                 random_state=seed, debug = debug, output_path=output_path)
    optimizer.maximize(init_points = 2, n_iter = 10, acq = 'ei')


### Test 1: goldstain. 
#guarda il file goldstain


@perform_test
def test_goldstain_ml(output_path):
    def goldstain(x1, x2, x3, x4):
        c1 = 1
        c2 = 1
        ret = {}
        ret['value'] = 53.3108 \
            + 0.184901 * x1 \
            - 5.02914 * x1**3 * 1e-6 \
            + 7.72522 * x1**4 * 1e-8 \
            + 0.0870775 * x2 \
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
    
    dataset_discrete = pd.DataFrame({'x3': [0, 0, 0, 1, 1, 1, 2, 2, 2], 'x4': [0, 1, 2, 0, 1, 2, 0, 1, 2]})
    optimizer = BO(f=goldstain, pbounds={'x1': (0, 100), 'x2': (0, 100), 'x3': (0, 2), 'x4': (0, 2)},
                 random_state=seed, debug = debug, dataset_discrete = dataset_discrete, output_path=output_path+'_ml'
                 , true_maximum_value=74.4970)

    optimizer.maximize(init_points=2, n_iter=10, acq='ei_ml',
                     acq_info={'ml_target': 'blackbox', 'ml_bounds': (0, inf)})
    plot_regret(output_path=output_path + "/results.csv")

### Test 2: branin

@perform_test
def test_branin_ml(output_path):
    
    def h(x1, x2):
        term1 = (15 * x2 - (5 / (4 * np.pi**2)) * (15 * x1 - 5)**2 + (5 / np.pi) * (15 * x1 - 5) - 6)**2
        term2 = 10 * (1 - 1 / (8 * np.pi)) * np.cos(15 * x1 - 5) + 10
        return (term1 + term2 - 54.8104) * (1 / 51.9496)

    def branin(x1, x2, x3, x4):
        if x3 == 0 and x4 == 0:
            return h(x1, x2)
        elif x3 == 0 and x4 == 1:
            return 0.4 * h(x1, x2)
        elif x3 == 1 and x4 == 0:
            return -0.75 * h(x1, x2) + 3.0
        elif x3 == 1 and x4 == 1:
            return -0.5 * h(x1, x2) + 1.4
        else: 
            raise ValueError("Error! This function can be evaluated only for x3,x4 in {0, 1}")

    dataset_discrete = pd.DataFrame({'x3': [0, 0, 1, 1], 'x4': [0, 1, 0, 1]})
    optimizer = BO(
        f=branin,
        pbounds={'x1': (0, 1), 'x2': (0, 1), 'x3': (0, 1), 'x4': (0, 1)},
        random_state=seed,
        debug=debug,
        dataset_discrete=dataset_discrete,
        output_path=output_path,
        true_maximum_value=4.8344
    )

    optimizer.maximize(init_points=2, n_iter=10, acq='ei')
    plot_regret(output_path=output_path + "/results.csv")


@perform_test
def test_branin_a_10_ml(output_path):
        
    def h(x1, x2):
        term1 = (15 * x2 - (5 / (4 * np.pi**2)) * (15 * x1 - 5)**2 + (5 / np.pi) * (15 * x1 - 5) - 6)**2
        term2 = 10 * (1 - 1 / (8 * np.pi)) * np.cos(15 * x1 - 5) + 10
        return (term1 + term2 - 54.8104) * (1 / 51.9496)

    def branin(x1, x2, x3, x4):
        if x3 == 0 and x4 == 0:
            return h(x1, x2)
        elif x3 == 0 and x4 == 1:
            return 0.4 * h(x1, x2)
        elif x3 == 1 and x4 == 0:
            return -0.75 * h(x1, x2) + 3.0
        elif x3 == 1 and x4 == 1:
            return -0.5 * h(x1, x2) + 1.4
        else: 
            raise ValueError("Error! This function can be evaluated only for x3,x4 in {0, 1}")

    dataset_discrete = pd.DataFrame({'x3': [0, 0, 1, 1], 'x4': [0, 1, 0, 1]})
    optimizer = BO(
        f=branin,
        pbounds={'x1': (0, 1), 'x2': (0, 1), 'x3': (0, 1), 'x4': (0, 1)},
        random_state=seed,
        debug=debug,
        dataset_discrete=dataset_discrete,
        output_path=output_path,
        true_maximum_value=4.8344,
        penalization_alpha=10
    )

    optimizer.maximize(init_points=2, n_iter=10, acq='ei')
    plot_regret(output_path=output_path + "/results.csv")


#test_mixed_2d_ml()
#test_continuos()
test_goldstain_ml()
#test_branin_ml()
#test_branin_a_10_ml()

#compare_regret(['test_mixed_2d','test_goldstain', 'test_branin', 'test_branin_a_10'])

from test import test03_dataset_X

# test03_dataset_X()
