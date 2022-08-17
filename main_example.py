import argparse
import os
import pandas as pd
import time

from bayes_opt import BayesianOptimization as BO

def target_func(x, y):
    return -x ** 2 - (y - 1) ** 2 + 1

def target_func_1D(y):
    return -y ** 2

def target_func_dict(x, y):
    ret = {}
    ret['value'] = -x ** 2 - (y - 1) ** 2 + 1
    ret['field'] = "idk"
    ret['blackbox'] = 2.0 * x + 3.0 * y
    return ret

# User parameters (debug: ignored if None, otherwise it overrides the command-line argument)
seed = 1
n0 = 2
n_iter = 5
debug = None





# Initialize argument parser
parser = argparse.ArgumentParser(description="Use Bayesian Optimization to maximize a function")
parser.add_argument('-d', "--debug", help="debug mode (ignored if debug is set in the script)", default=False, action="store_true")
args = parser.parse_args()
# Get debug flag
debug = args.debug if debug is None else debug


def perform_test(testfunc):
  print("Starting", testfunc.__name__, "...")
  output_path = os.path.join('outputs', testfunc.__name__)
  start = time.time()
  testfunc(output_path)
  stop = time.time()
  print("Done in", stop-start, "seconds\n\n\n")


def test01_free(output_path):
  optimizer = BO(f=target_func, pbounds={'x': (2, 4), 'y': (-3, 3)},
                 random_state=seed, output_path=output_path, debug=debug)
  optimizer.maximize(init_points=n0, n_iter=n_iter)


def test02_dataset_Xy(output_path):
  optimizer = BO(f=None, pbounds={'x': (999,2501), 'y': (1,50)},
                 random_state=seed,
                 dataset=os.path.join('datasets', 'test_xyz.csv'),
                 target_column='z', output_path=output_path, debug=debug)
  optimizer.maximize(init_points=n0, n_iter=n_iter)


def test03_dataset_X(output_path):
  data = pd.read_csv(os.path.join('datasets', 'test_xyz.csv'))
  optimizer = BO(f=target_func, pbounds={'x': (999,2501), 'y': (1,50)},
                 random_state=seed, dataset=data, output_path=output_path,
                 debug=debug)
  optimizer.maximize(init_points=n0, n_iter=n_iter)
  print("res:", optimizer.res)


def test04_free_ml(output_path):
  optimizer = BO(f=target_func_dict, pbounds={'x': (2, 4), 'y': (-3, 3)},
                 random_state=seed, output_path=output_path, debug=debug)
  optimizer.maximize(init_points=n0, n_iter=n_iter, acq='ei_ml',
                     acq_info={'ml_target': 'blackbox', 'ml_bounds': (2, 8)})


def test05_dataset_Xy_ml(output_path):
  optimizer = BO(f=None, pbounds={'x': (7,73), 'y': (7,73)}, random_state=seed,
                 dataset=os.path.join('datasets', 'test_ml.csv'),
                 target_column='z', output_path=output_path, debug=debug)
  optimizer.maximize(init_points=n0, n_iter=n_iter, acq='ei_ml',
                     acq_info={'ml_target': 'z_pred', 'ml_bounds': (0, 2.2)})


def test06_dataset_X_ml(output_path):
  optimizer = BO(f=target_func_dict, pbounds={'x': (7,73), 'y': (7,73)},
                 random_state=seed,
                 dataset=os.path.join('datasets', 'test_ml.csv'),
                 output_path=output_path, debug=debug)
  optimizer.maximize(init_points=n0, n_iter=n_iter, acq='ei_ml',
                     acq_info={'ml_target': 'z_pred', 'ml_bounds': (0, 2.2)})


def test07_dataset_Xy_queue(output_path):
  optimizer = BO(f=None, pbounds={'y': (1,50)}, random_state=seed,
                 dataset=os.path.join('datasets', 'test_xyz.csv'),
                 target_column='z', output_path=output_path, debug=debug)
  optimizer.maximize(init_points=n0, n_iter=n_iter, memory_queue_len=3)


def test08_dataset_X_queue(output_path):
  optimizer = BO(f=target_func_1D, pbounds={'y': (1,50)}, random_state=seed,
                 dataset=os.path.join('datasets', 'test_xyz.csv'),
                 output_path=output_path, debug=debug)
  optimizer.maximize(init_points=n0, n_iter=n_iter, memory_queue_len=3)


def test09_free_eic_default(output_path):
  optimizer = BO(f=target_func, pbounds={'x': (2, 4), 'y': (-3, 3)},
                 random_state=seed, output_path=output_path, debug=debug)
  optimizer.maximize(init_points=n0, n_iter=n_iter, acq='eic',
                     acq_info={'eic_bounds': (-3.2, -3.0)})


def test10_dataset_Xy_eic_default(output_path):
  optimizer = BO(f=None, pbounds={'x': (999,2501), 'y': (1,50)},
                 random_state=seed,
                 dataset=os.path.join('datasets', 'test_xyz.csv'),
                 target_column='z', output_path=output_path, debug=debug)
  optimizer.maximize(init_points=n0, n_iter=n_iter, acq='eic',
                     acq_info={'eic_bounds': (2_500_000, 2_700_000)})


def test11_free_eic_custom_PQ(output_path):
  def my_P(x):
      return 2.0 * x[:, 0]
  def my_Q(x):
      return 5.0
  optimizer = BO(f=target_func, pbounds={'x': (2, 4), 'y': (-3, 3)},
                 random_state=seed, output_path=output_path, debug=debug)
  optimizer.maximize(init_points=n0, n_iter=n_iter, acq='eic',
                     acq_info={'eic_bounds': (-3.2, -3.0), 'eic_P_func': my_P,
                                                           'eic_Q_func': my_Q})


def test12_free_init_points_tuple(output_path):
  optimizer = BO(f=target_func, pbounds={'x': (2, 4), 'y': (-3, 3)},
                 random_state=seed, output_path=output_path, debug=debug)
  init_points = ({'x': 3.1, 'y': 0.7}, {'x': 2.2, 'y': -2.8})
  optimizer.add_initial_points(init_points)
  optimizer.maximize(init_points=0, n_iter=n_iter)


def test13_dataset_Xy_init_points_dicts(output_path):
  optimizer = BO(f=None, pbounds={'x': (999,2501), 'y': (1,50)},
                 random_state=seed,
                 dataset=os.path.join('datasets', 'test_xyz.csv'),
                 target_column='z', output_path=output_path, debug=debug)
  optimizer.add_initial_points(dict(x=1000, y=6))   # z=971359, idx=197
  optimizer.add_initial_points(dict(x=2500, y=38))  # z=1544285, idx=12
  optimizer.maximize(init_points=0, n_iter=n_iter)


def test14_dataset_X_init_points_df(output_path):
  optimizer = BO(f=target_func, pbounds={'x': (999,2501), 'y': (1,50)},
                 random_state=seed,
                 dataset=os.path.join('datasets', 'test_xyz.csv'),
                 output_path=output_path, debug=debug)
  init_points = pd.DataFrame([dict(x=1000, y=6), dict(x=2500, y=38)])
  optimizer.add_initial_points(init_points)
  optimizer.maximize(init_points=0, n_iter=n_iter)


def test16_free_eic_ml_C(output_path):
  optimizer = BO(f=target_func_dict, pbounds={'x': (2, 4), 'y': (-3, 3)},
                 random_state=seed, output_path=output_path, debug=debug)
  optimizer.maximize(init_points=n0, n_iter=n_iter, acq='eic_ml_C',
                     acq_info={'eic_bounds': (-3.2, -3.0),
                               'ml_target': 'blackbox',
                               'ml_bounds': (2, 8)
                               })


if __name__ == '__main__':
  # perform_test(test01_free)
  # perform_test(test02_dataset_Xy)
  # perform_test(test03_dataset_X)
  # perform_test(test04_free_ml)
  # perform_test(test05_dataset_Xy_ml)
  # perform_test(test06_dataset_X_ml)
  # perform_test(test07_dataset_Xy_queue)
  # perform_test(test08_dataset_X_queue)
  # perform_test(test09_free_eic_default)
  # perform_test(test10_dataset_Xy_eic_default)
  # perform_test(test11_free_eic_custom_PQ)
  # perform_test(test12_free_init_points_tuple)
  # perform_test(test13_dataset_Xy_init_points_dicts)
  # perform_test(test14_dataset_X_init_points_df)
  perform_test(test16_free_eic_ml_C)
