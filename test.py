import os
from bayes_opt import BayesianOptimization as BO

def black_box_function(x, y):
    return -x ** 2 - (y - 1) ** 2 + 1

seed = 1
n0 = 2
n_iter = 3


def perform_test(testfunc):
  print("Starting", testfunc.__name__, "...")
  testfunc()
  print("Done\n\n")



def test01_free():
  optimizer = BO(f=black_box_function, pbounds={'x': (2, 4), 'y': (-3, 3)},
                 random_state=seed,
                 output_path=os.path.join('outputs' ,'test01'))
  optimizer.maximize(init_points=n0, n_iter=n_iter)


def test02_dataset_Xy():
  optimizer = BO(f=None, pbounds={'x': (999,2501), 'y': (1,50)},
                 random_state=seed,
                 dataset_path=os.path.join('datasets', 'test02.csv'),
                 target_column='z',
                 output_path=os.path.join('outputs' ,'test02')
                 )
  optimizer.maximize(init_points=n0, n_iter=n_iter)


def test03_dataset_X():
  optimizer = BO(f=black_box_function, pbounds={'x': (999,2501), 'y': (1,50)},
                 random_state=seed,
                 dataset_path=os.path.join('datasets', 'test03.csv'),
                 output_path=os.path.join('outputs' ,'test03'))
  optimizer.maximize(init_points=n0, n_iter=n_iter)


if __name__ == '__main__':
  perform_test(test01_free)
  perform_test(test02_dataset_Xy)
  perform_test(test03_dataset_X)
