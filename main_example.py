import os
from bayes_opt import BayesianOptimization as BO

def black_box_function(x, y):
    return -x ** 2 - (y - 1) ** 2 + 1

def black_box_function_ml(x, y):
    ret = {}
    ret['value'] = -x ** 2 - (y - 1) ** 2 + 1
    ret['field'] = "idk"
    ret['blackbox'] = 2.0 * x + 3.0 * y
    return ret

seed = 1
n0 = 2
n_iter = 5


def perform_test(testfunc):
  print("Starting", testfunc.__name__, "...")
  testfunc()
  print("Done\n")



def test01_free():
  optimizer = BO(f=black_box_function, pbounds={'x': (2, 4), 'y': (-3, 3)},
                 random_state=seed,
                 output_path=os.path.join('outputs' ,'test01'))
  optimizer.maximize(init_points=n0, n_iter=n_iter)


def test02_dataset_Xy():
  optimizer = BO(f=None, pbounds={'x': (999,2501), 'y': (1,50)},
                 random_state=seed,
                 dataset_path=os.path.join('datasets', 'test_xyz.csv'),
                 target_column='z',
                 output_path=os.path.join('outputs' ,'test02')
                 )
  optimizer.maximize(init_points=n0, n_iter=n_iter)


def test03_dataset_X():
  optimizer = BO(f=black_box_function, pbounds={'x': (999,2501), 'y': (1,50)},
                 random_state=seed,
                 dataset_path=os.path.join('datasets', 'test_xyz.csv'),
                 output_path=os.path.join('outputs' ,'test03'))
  optimizer.maximize(init_points=n0, n_iter=n_iter)


def test04_free_ml():
  optimizer = BO(f=black_box_function_ml, pbounds={'x': (2, 4), 'y': (-3, 3)},
                 random_state=seed,
                 output_path=os.path.join('outputs' ,'test04'))
  optimizer.maximize(init_points=n0, n_iter=n_iter, acq='ei_ml',
                     ml_info={'target': 'blackbox', 'bounds': (2, 8)})


def test05_dataset_Xy_ml():
  optimizer = BO(f=None, pbounds={'x': (7,73), 'y': (7,73)},
                 random_state=seed,
                 dataset_path=os.path.join('datasets', 'test_ml.csv'),
                 target_column='z',
                 output_path=os.path.join('outputs' ,'test05')
                 )
  optimizer.maximize(init_points=n0, n_iter=n_iter, acq='ei_ml',
                     ml_info={'target': 'z_pred', 'bounds': (0, 2.2)})


def test06_dataset_X_ml():
  optimizer = BO(f=black_box_function_ml, pbounds={'x': (7,73), 'y': (7,73)},
                 random_state=seed,
                 dataset_path=os.path.join('datasets', 'test_ml.csv'),
                 output_path=os.path.join('outputs' ,'test06')
                 )
  optimizer.maximize(init_points=n0, n_iter=n_iter, acq='ei_ml',
                     ml_info={'target': 'z_pred', 'bounds': (0, 2.2)})


if __name__ == '__main__':
  perform_test(test01_free)
  perform_test(test02_dataset_Xy)
  perform_test(test03_dataset_X)
  perform_test(test04_free_ml)
  perform_test(test05_dataset_Xy_ml)
  perform_test(test06_dataset_X_ml)
