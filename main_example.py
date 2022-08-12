import os
import pandas as pd
import time

from bayes_opt import BayesianOptimization as BO

def black_box_function(x, y):
    return -x ** 2 - (y - 1) ** 2 + 1

def black_box_function_1D(y):
    return -y ** 2

def black_box_function_dict(x, y):
    ret = {}
    ret['value'] = -x ** 2 - (y - 1) ** 2 + 1
    ret['field'] = "idk"
    ret['blackbox'] = 2.0 * x + 3.0 * y
    return ret

seed = 1
n0 = 2
n_iter = 5
debug = False


def perform_test(testfunc):
  print("Starting", testfunc.__name__, "...")
  start = time.time()
  testfunc()
  stop = time.time()
  print("Done in", stop-start, "seconds\n\n\n")



def test01_free():
  optimizer = BO(f=black_box_function, pbounds={'x': (2, 4), 'y': (-3, 3)},
                 random_state=seed,
                 output_path=os.path.join('outputs' ,'test01'), debug=debug)
  optimizer.maximize(init_points=n0, n_iter=n_iter)


def test02_dataset_Xy():
  optimizer = BO(f=None, pbounds={'x': (999,2501), 'y': (1,50)},
                 random_state=seed,
                 dataset=os.path.join('datasets', 'test_xyz.csv'),
                 target_column='z',
                 output_path=os.path.join('outputs' ,'test02'), debug=debug)
  optimizer.maximize(init_points=n0, n_iter=n_iter)


def test03_dataset_X():
  data = pd.read_csv(os.path.join('datasets', 'test_xyz.csv'))
  optimizer = BO(f=black_box_function, pbounds={'x': (999,2501), 'y': (1,50)},
                 random_state=seed, dataset=data,
                 output_path=os.path.join('outputs' ,'test03'), debug=debug)
  optimizer.maximize(init_points=n0, n_iter=n_iter)


def test04_free_ml():
  optimizer = BO(f=black_box_function_dict,
                 pbounds={'x': (2, 4), 'y': (-3, 3)}, random_state=seed,
                 output_path=os.path.join('outputs' ,'test04'), debug=debug)
  optimizer.maximize(init_points=n0, n_iter=n_iter, acq='ei_ml',
                     ml_info={'target': 'blackbox', 'bounds': (2, 8)})


def test05_dataset_Xy_ml():
  optimizer = BO(f=None, pbounds={'x': (7,73), 'y': (7,73)}, random_state=seed,
                 dataset=os.path.join('datasets', 'test_ml.csv'),
                 target_column='z',
                 output_path=os.path.join('outputs' ,'test05'), debug=debug)
  optimizer.maximize(init_points=n0, n_iter=n_iter, acq='ei_ml',
                     ml_info={'target': 'z_pred', 'bounds': (0, 2.2)})


def test06_dataset_X_ml():
  optimizer = BO(f=black_box_function_dict, pbounds={'x': (7,73), 'y': (7,73)},
                 random_state=seed,
                 dataset=os.path.join('datasets', 'test_ml.csv'),
                 output_path=os.path.join('outputs' ,'test06'), debug=debug)
  optimizer.maximize(init_points=n0, n_iter=n_iter, acq='ei_ml',
                     ml_info={'target': 'z_pred', 'bounds': (0, 2.2)})


def test07_dataset_Xy_queue():
  optimizer = BO(f=None, pbounds={'y': (1,50)}, random_state=seed,
                 dataset=os.path.join('datasets', 'test_xyz.csv'),
                 target_column='z',
                 output_path=os.path.join('outputs' ,'test07'), debug=debug)
  optimizer.maximize(init_points=n0, n_iter=n_iter, memory_queue_len=3)


def test08_dataset_X_queue():
  optimizer = BO(f=black_box_function_1D, pbounds={'y': (1,50)},
                 random_state=seed,
                 dataset=os.path.join('datasets', 'test_xyz.csv'),
                 output_path=os.path.join('outputs' ,'test08'), debug=debug)
  optimizer.maximize(init_points=n0, n_iter=n_iter, memory_queue_len=3)


def test09_free_eic_default():
  optimizer = BO(f=black_box_function,
                 pbounds={'x': (2, 4), 'y': (-3, 3)}, random_state=seed,
                 output_path=os.path.join('outputs' ,'test09'), debug=debug)
  optimizer.maximize(init_points=n0, n_iter=n_iter, acq='eic',
                     eic_info={'bounds': (-3.2, -3.0)})


def test10_dataset_Xy_eic_default():
  optimizer = BO(f=None, pbounds={'x': (999,2501), 'y': (1,50)},
                 random_state=seed,
                 dataset=os.path.join('datasets', 'test_xyz.csv'),
                 target_column='z',
                 output_path=os.path.join('outputs' ,'test10'), debug=debug)
  optimizer.maximize(init_points=n0, n_iter=n_iter, acq='eic',
                     eic_info={'bounds': (2_500_000, 2_700_000)})


def test11_free_eic_custom_PQ():
  def my_P(x):
      return 2.0 * x[:, 0]
  def my_Q(x):
      return 5.0
  optimizer = BO(f=black_box_function,
                 pbounds={'x': (2, 4), 'y': (-3, 3)}, random_state=seed,
                 output_path=os.path.join('outputs' ,'test09'), debug=debug)
  optimizer.maximize(init_points=n0, n_iter=n_iter, acq='eic',
                     eic_info={'bounds': (-3.2, -3.0), 'P_func': my_P,
                                                       'Q_func': my_Q})


if __name__ == '__main__':
  perform_test(test01_free)
  perform_test(test02_dataset_Xy)
  perform_test(test03_dataset_X)
  perform_test(test04_free_ml)
  perform_test(test05_dataset_Xy_ml)
  perform_test(test06_dataset_X_ml)
  perform_test(test07_dataset_Xy_queue)
  perform_test(test08_dataset_X_queue)
  perform_test(test09_free_eic_default)
  perform_test(test10_dataset_Xy_eic_default)
  perform_test(test11_free_eic_custom_PQ)
