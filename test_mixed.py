from maliboo.bayesian_optimization import BayesianOptimization as BO
import pandas as pd
import os



def target_func(x1, x2):
    return -x1 ** 2 - (x2 - 1) ** 2 + 2


seed = 42
debug = False
dataset_discrete = pd.DataFrame({'x2': [1, 2, 3]})



# it has been added dataset_discrete to the constructor of BO class. This dataset must contain all the combination of
# of the discrete variable. It is used in order to distinguish between discrete and continuos variables. 

optimizer = BO(f=target_func, pbounds={'x1': (-2, 4), 'x2': (-3, 3)},
                 random_state=seed, debug = debug, dataset_discrete = dataset_discrete)

optimizer.maximize(init_points = 2, n_iter = 20, acq = 'ei')



""" optimizer = BO(f=None, pbounds={'x1': (999,2501), 'x2': (1,50)},
                random_state=seed,
                dataset=os.path.join('resources', 'test_xyz.csv'),
                target_column='z', debug=debug)
optimizer.maximize(init_points=5, n_iter=5, acq='eic',
                    acq_info={'eic_bounds': (2_500_000, 2_700_000)})
 """