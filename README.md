<div align="center">
  <img src="https://github.com/brunoguindani/MALIBOO/blob/master/resources/bo_dark_4.png" width="80%" height="80%"><br><br>
</div>

# MALIBOO
MALIBOO (MAchine Learning In Bayesian OptimizatiOn) is a Python library which performs Bayesian Optimization (BO) on black-box functions, or on their tabular representation.
The implemented BO algorithms are integrated with Machine Learning techniques.

This library originates as a fork from the [BayesianOptimization](https://github.com/fmfn/BayesianOptimization) package from Fernando Nogueira ([@fmfn](https://github.com/fmfn)), which was started in 2014.


## Installing
Please `git clone` this repository directly.
You will also need Python 3 and a few dependencies.
The latter can be taken care of by heading too the root directory of this repository and running:
```
pip3 install -r requirements.txt
```


## Tutorial
This library can be used with pure BO, just like the original package:
```python
from maliboo import BayesianOptimization as BO

def target_func(x1, x2):
    return -x1 ** 2 - (x2 - 1) ** 2 + 1

optimizer = BO(f=target_func, pbounds={'x1': (2, 4), 'x2': (-3, 3)},
               random_state=1, output_path='outputs/tutorial', debug=False)
optimizer.maximize(init_points=2, n_iter=5)
```
The output should look something like this:
```
|   iter    |  target   |    x1     |    x2     |
-------------------------------------------------
| 1         | -7.135    | 2.834     | 1.322     |
| 2         | -7.78     | 2.0       | -1.186    |
| 3         | -7.11     | 2.218     | -0.7867   |
| 4         | -12.4     | 3.66      | 0.9608    |
| 5         | -6.999    | 2.23      | -0.7392   |
| 6         | -3.047    | 2.0       | 0.783     |
| 7         | -4.166    | 2.0       | 2.08      |
max: {'target': -3.0471017620190217, 'params': {'x1': 2.0, 'x2': 0.7829705964183161}}
Results successfully saved to outputs/tutorial
=================================================
```
You can also run the [`main_examples.py`](https://github.com/brunoguindani/BayesianOptimization/blob/master/main_example.py) file, which contains over 20 usage examples using different features and techniques.


## File documentation
For further documentation, please also refer to the well-documented [Python files](https://github.com/brunoguindani/BayesianOptimization/tree/master/maliboo) of the project.
