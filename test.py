import argparse
import os
import pandas as pd
import time

from maliboo import BayesianOptimization as BO

def target_func(x1, x2):
    return -x1 ** 2 - (x2 - 1) ** 2 + 1

def target_func_slow(x1, x2):
    time.sleep(2)
    return -x1 ** 2 - (x2 - 1) ** 2 + 1

def target_func_1D(x2):
    return -x2 ** 2

def target_func_dict(x1, x2):
    ret = {}
    ret['value'] = -x1 ** 2 - (x2 - 1) ** 2 + 1
    ret['field'] = "idk"
    ret['blackbox'] = 2.0 * x1 + 3.0 * x2
    return ret

# User parameters (debug: ignored if None, otherwise it overrides the
# command-line argument)
seed = 1
n0 = 2
n_iter = 5
debug = None





# Initialize argument parser
parser = argparse.ArgumentParser(description="Use Bayesian Optimization to "
                                              "maximize a function")
parser.add_argument('-d', "--debug", default=False, action="store_true",
                    help="debug mode (ignored if debug is set in the script)")
args = parser.parse_args()
# Get debug flag
debug = args.debug if debug is None else debug


def perform_test(testfunc):
    def ret():
        print("Starting", testfunc.__name__, "...")
        output_path = os.path.join('outputs', testfunc.__name__)
        start = time.time()
        testfunc(output_path)
        stop = time.time()
        print("Done in", stop-start, "seconds\n\n\n")
    return ret


@perform_test
def test00a_free_complex(output_path):
  def my_P(x):
      return 2.0 * x[:, 0]
  def my_Q(x):
      return 5.0
  optimizer = BO(f=target_func_dict, pbounds={'x1': (2, 4), 'x2': (-3, 3)},
                 random_state=seed, output_path=output_path, debug=debug)
  init_points = ({'x1': 3.1, 'x2': 0.7}, {'x1': 2.2, 'x2': -2.8})
  optimizer.add_initial_points(init_points)
  optimizer.maximize(init_points=0, n_iter=n_iter, acq='eic_ml',
                     memory_queue_len=3,
                     acq_info={'eic_ml_var': 'B', 'eic_bounds': (-3.2, -3.0),
                               'eic_P_func': my_P, 'eic_Q_func': my_Q,
                               'ml_target': 'blackbox', 'ml_bounds': (2, 8),
                               'eic_ml_exp_B': 2.0
                               },
                     stop_crit_info={'hard_stop': False, 'conjunction': 'or',
                                     'ml_bounds_coeff': (0.9, 1.1)
                                    })


@perform_test
def test00b_dataset_Xy_complex(output_path):
  def my_P(x):
      return 2.0 * x[:, 0]
  def my_Q(x):
      return 5.0
  optimizer = BO(f=None, pbounds={'x1': (7,73), 'x2': (7,73)},
                 random_state=seed,
                 dataset=os.path.join('resources', 'test_ml.csv'),
                 target_column='z', output_path=output_path, debug=debug)
  optimizer.add_initial_points(dict(x1=48, x2=48))  # z=220.431, idx=0
  optimizer.add_initial_points(dict(x1=16, x2=8))   # z=146.028, idx=1
  optimizer.maximize(init_points=0, n_iter=n_iter, acq='eic_ml',
                     memory_queue_len=3,
                     acq_info={'eic_ml_var': 'B',
                               'eic_bounds': (0, 2.2),
                               'eic_P_func': my_P, 'eic_Q_func': my_Q,
                               'ml_target': 'z_pred', 'ml_bounds': (0, 2.2),
                               'eic_ml_exp_B': 2.0
                               },
                     stop_crit_info={'hard_stop': True, 'conjunction': 'or',
                                     'ml_bounds_coeff': (0.9, 1.01)
                                    })


@perform_test
def test01_free(output_path):
  optimizer = BO(f=target_func, pbounds={'x1': (2, 4), 'x2': (-3, 3)},
                 random_state=seed, output_path=output_path, debug=debug)
  optimizer.maximize(init_points=n0, n_iter=n_iter)


@perform_test
def test02_dataset_Xy(output_path):
  optimizer = BO(f=None, pbounds={'x1': (999,2501), 'x2': (1,50)},
                 random_state=seed,
                 dataset=os.path.join('resources', 'test_xyz.csv'),
                 target_column='z', output_path=output_path, debug=debug)
  optimizer.maximize(init_points=n0, n_iter=n_iter)


@perform_test
def test03_dataset_X(output_path):
  data = pd.read_csv(os.path.join('resources', 'test_xyz.csv'))
  optimizer = BO(f=target_func, pbounds={'x1': (999,2501), 'x2': (1,50)},
                 random_state=seed, dataset=data, output_path=output_path,
                 debug=debug)
  optimizer.maximize(init_points=n0, n_iter=n_iter)
  print("res:", optimizer.res)


@perform_test
def test04_free_ml(output_path):
  optimizer = BO(f=target_func_dict, pbounds={'x1': (2, 4), 'x2': (-3, 3)},
                 random_state=seed, output_path=output_path, debug=debug)
  optimizer.maximize(init_points=n0, n_iter=n_iter, acq='ei_ml',
                     acq_info={'ml_target': 'blackbox', 'ml_bounds': (2, 8)})


@perform_test
def test05_dataset_Xy_ml(output_path):
  optimizer = BO(f=None, pbounds={'x1': (7,73), 'x2': (7,73)},
                 random_state=seed,
                 dataset=os.path.join('resources', 'test_ml.csv'),
                 target_column='z', output_path=output_path, debug=debug)
  optimizer.maximize(init_points=n0, n_iter=n_iter, acq='ei_ml',
                     acq_info={'ml_target': 'z_pred', 'ml_bounds': (0, 2.2)})


@perform_test
def test06_dataset_X_ml(output_path):
  optimizer = BO(f=target_func_dict, pbounds={'x1': (7,73), 'x2': (7,73)},
                 random_state=seed,
                 dataset=os.path.join('resources', 'test_ml.csv'),
                 output_path=output_path, debug=debug)
  optimizer.maximize(init_points=n0, n_iter=n_iter, acq='ei_ml',
                     acq_info={'ml_target': 'z_pred', 'ml_bounds': (0, 2.2)})


@perform_test
def test07_dataset_Xy_queue(output_path):
  optimizer = BO(f=None, pbounds={'x2': (1,50)}, random_state=seed,
                 dataset=os.path.join('resources', 'test_xyz.csv'),
                 target_column='z', output_path=output_path, debug=debug)
  optimizer.maximize(init_points=n0, n_iter=n_iter, memory_queue_len=3)


@perform_test
def test08_dataset_X_queue(output_path):
  optimizer = BO(f=target_func_1D, pbounds={'x2': (1,50)}, random_state=seed,
                 dataset=os.path.join('resources', 'test_xyz.csv'),
                 output_path=output_path, debug=debug)
  optimizer.maximize(init_points=n0, n_iter=n_iter, memory_queue_len=3)


@perform_test
def test09_free_eic_default(output_path):
  optimizer = BO(f=target_func, pbounds={'x1': (2, 4), 'x2': (-3, 3)},
                 random_state=seed, output_path=output_path, debug=debug)
  optimizer.maximize(init_points=n0, n_iter=n_iter, acq='eic',
                     acq_info={'eic_bounds': (-3.2, -3.0)})


@perform_test
def test10_dataset_Xy_eic_default(output_path):
  optimizer = BO(f=None, pbounds={'x1': (999,2501), 'x2': (1,50)},
                 random_state=seed,
                 dataset=os.path.join('resources', 'test_xyz.csv'),
                 target_column='z', output_path=output_path, debug=debug)
  optimizer.maximize(init_points=n0, n_iter=n_iter, acq='eic',
                     acq_info={'eic_bounds': (2_500_000, 2_700_000)})


@perform_test
def test11_free_eic_custom_PQ(output_path):
  def my_P(x):
      return 2.0 * x[:, 0]
  def my_Q(x):
      return 5.0
  optimizer = BO(f=target_func, pbounds={'x1': (2, 4), 'x2': (-3, 3)},
                 random_state=seed, output_path=output_path, debug=debug)
  optimizer.maximize(init_points=n0, n_iter=n_iter, acq='eic',
                     acq_info={'eic_bounds': (-3.2, -3.0), 'eic_P_func': my_P,
                                                           'eic_Q_func': my_Q})


@perform_test
def test12_free_init_points_tuple(output_path):
  optimizer = BO(f=target_func, pbounds={'x1': (2, 4), 'x2': (-3, 3)},
                 random_state=seed, output_path=output_path, debug=debug)
  init_points = ({'x1': 3.1, 'x2': 0.7}, {'x1': 2.2, 'x2': -2.8})
  optimizer.add_initial_points(init_points)
  optimizer.maximize(init_points=0, n_iter=n_iter)


@perform_test
def test13_dataset_Xy_init_points_dicts(output_path):
  optimizer = BO(f=None, pbounds={'x1': (999,2501), 'x2': (1,50)},
                 random_state=seed,
                 dataset=os.path.join('resources', 'test_xyz.csv'),
                 target_column='z', output_path=output_path, debug=debug)
  optimizer.add_initial_points(dict(x1=1000, x2=6))   # z=971359, idx=197
  optimizer.add_initial_points(dict(x1=2500, x2=38))  # z=1544285, idx=12
  optimizer.maximize(init_points=0, n_iter=n_iter)


@perform_test
def test14_dataset_X_init_points_df(output_path):
  optimizer = BO(f=target_func, pbounds={'x1': (999,2501), 'x2': (1,50)},
                 random_state=seed,
                 dataset=os.path.join('resources', 'test_xyz.csv'),
                 output_path=output_path, debug=debug)
  init_points = pd.DataFrame([dict(x1=1000, x2=6),
                              dict(x1=2500, x2=38)])
  optimizer.add_initial_points(init_points)
  optimizer.maximize(init_points=0, n_iter=n_iter)


@perform_test
def test15_free_eic_ml_B(output_path):
  optimizer = BO(f=target_func_dict, pbounds={'x1': (2, 4), 'x2': (-3, 3)},
                 random_state=seed, output_path=output_path, debug=debug)
  optimizer.maximize(init_points=n0, n_iter=n_iter, acq='eic_ml',
                     acq_info={'eic_ml_var': 'B',
                               'eic_bounds': (-3.2, -3.0),
                               'ml_target': 'blackbox', 'ml_bounds': (2, 8),
                               'eic_ml_exp_B': 2.0
                               })


@perform_test
def test16_free_eic_ml_C(output_path):
  optimizer = BO(f=target_func_dict, pbounds={'x1': (2, 4), 'x2': (-3, 3)},
                 random_state=seed, output_path=output_path, debug=debug)
  optimizer.maximize(init_points=n0, n_iter=n_iter, acq='eic_ml',
                     acq_info={'eic_ml_var': 'C',
                               'eic_bounds': (-3.2, -3.0),
                               'ml_target': 'blackbox', 'ml_bounds': (2, 8)
                               })


@perform_test
def test17_free_eic_ml_D(output_path):
  optimizer = BO(f=target_func_dict, pbounds={'x1': (2, 4), 'x2': (-3, 3)},
                 random_state=seed, output_path=output_path, debug=debug)
  optimizer.maximize(init_points=n0, n_iter=n_iter, acq='eic_ml',
                     acq_info={'eic_ml_var': 'D',
                               'eic_bounds': (-3.2, -3.0),
                               'ml_target': 'blackbox', 'ml_bounds': (2, 8)
                               })


@perform_test
def test18_free_stop_crit_soft(output_path):
  optimizer = BO(f=target_func_dict, pbounds={'x1': (2, 4), 'x2': (-3, 3)},
                 random_state=seed, output_path=output_path, debug=debug)
  optimizer.maximize(init_points=n0, n_iter=n_iter, acq='eic_ml',
                     acq_info={'eic_ml_var': 'C',
                               'eic_bounds': (-3.2, -3.0),
                               'ml_target': 'blackbox',
                               'ml_bounds': (2, 8)
                               },
                     stop_crit_info={'hard_stop': False,
                                     'ml_bounds_coeff': (0.9, None)
                                    })


@perform_test
def test19_dataset_Xy_stop_crit_hard(output_path):
  optimizer = BO(f=None, pbounds={'x1': (7,73), 'x2': (7,73)},
                 random_state=seed,
                 dataset=os.path.join('resources', 'test_ml.csv'),
                 target_column='z', output_path=output_path, debug=debug)
  optimizer.maximize(init_points=n0, n_iter=n_iter, acq='eic_ml',
                     acq_info={'eic_ml_var': 'C',
                               'eic_bounds': (-3.2, -3.0),
                               'ml_target': 'z_pred',
                               'ml_bounds': (0, 2.2)
                               },
                     stop_crit_info={'hard_stop': True, 'conjunction': 'or',
                                     'ml_bounds_coeff': (0.9, None)
                                    })


@perform_test
def test20_dataset_Xy_relaxation(output_path):
  optimizer = BO(f=None, pbounds={'x1': (999,2501), 'x2': (1,50)},
                 random_state=seed,
                 dataset=os.path.join('resources', 'test_xyz.csv'),
                 target_column='z', output_path=output_path, debug=debug)
  optimizer.maximize(init_points=n0, n_iter=n_iter, relaxation=True)


@perform_test
def test21_dataset_X_relaxation(output_path):
  optimizer = BO(f=target_func, pbounds={'x1': (999,2501), 'x2': (1,50)},
                 random_state=seed,
                 dataset=os.path.join('resources', 'test_xyz.csv'),
                 output_path=output_path, debug=debug)
  optimizer.maximize(init_points=n0, n_iter=n_iter, relaxation=True)


@perform_test
def test22_dataset_Xy_relaxation_queue(output_path):
  optimizer = BO(f=None, pbounds={'x2': (1,50)}, random_state=seed,
                 dataset=os.path.join('resources', 'test_xyz.csv'),
                 target_column='z', output_path=output_path, debug=debug)
  optimizer.maximize(init_points=n0, n_iter=n_iter, memory_queue_len=3,
                     relaxation=True)


@perform_test
def test23_dataset_X_relaxation_queue(output_path):
  optimizer = BO(f=target_func_1D, pbounds={'x2': (1,50)}, random_state=seed,
                 dataset=os.path.join('resources', 'test_xyz.csv'),
                 output_path=output_path, debug=debug)
  optimizer.maximize(init_points=n0, n_iter=n_iter, memory_queue_len=3,
                     relaxation=True)


@perform_test
def test24_free_fault_tol(output_path):
  optimizer = BO(f=target_func_slow, pbounds={'x1': (2, 4), 'x2': (-3, 3)},
                 random_state=seed, output_path=output_path, debug=debug)
  optimizer.maximize(init_points=3, n_iter=5)


if __name__ == '__main__':
  test00a_free_complex()
  test00b_dataset_Xy_complex()
  test01_free()
  test02_dataset_Xy()
  test03_dataset_X()
  test04_free_ml()
  test05_dataset_Xy_ml()
  test06_dataset_X_ml()
  test07_dataset_Xy_queue()
  test08_dataset_X_queue()
  test09_free_eic_default()
  test10_dataset_Xy_eic_default()
  test11_free_eic_custom_PQ()
  test12_free_init_points_tuple()
  test13_dataset_Xy_init_points_dicts()
  test14_dataset_X_init_points_df()
  test15_free_eic_ml_B()
  test16_free_eic_ml_C()
  test17_free_eic_ml_D()
  test18_free_stop_crit_soft()
  test19_dataset_Xy_stop_crit_hard()
  test20_dataset_Xy_relaxation()
  test21_dataset_X_relaxation()
  test22_dataset_Xy_relaxation_queue()
  test23_dataset_X_relaxation_queue()
  test24_free_fault_tol()
