<div align="center">
  <img src="https://github.com/brunoguindani/MALIBOO/blob/master/resources/bo_dark_4.png" width="80%" height="80%"><br><br>
</div>

# MALIBOO
MALIBOO (MAchine Learning In Bayesian OptimizatiOn) is a Python library which performs Bayesian Optimization (BO) on single-output black-box functions, or on their tabular representation.
The implemented BO algorithms are integrated with Machine Learning techniques.

This library originates as a fork from the [BayesianOptimization](https://github.com/fmfn/BayesianOptimization) package from Fernando Nogueira ([@fmfn](https://github.com/fmfn)), which was started in 2014.


## Installing
Please `git clone` this repository directly.
You will also need Python 3 and a few dependencies.
The latter can be taken care of by heading to the root directory of this repository and running:
```
pip3 install -r requirements.txt
```


## Tutorial
This library can be used with pure BO, just like the original package:
```py
from maliboo import BayesianOptimization as BO

def target_func(x1, x2):
    return -x1 ** 2 - (x2 - 1) ** 2 + 1

optimizer = BO(f=target_func, pbounds={'x1': (2, 4), 'x2': (-3, 3)},
               output_path='outputs/tutorial', random_state=1, debug=False)
optimizer.maximize(init_points=2, n_iter=5, acq='ucb')
```
The library is first initialized by the construction of the `optimizer` object, then we call its `maximize()` method to perform the optimization proocedure.

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
You can also run the [`test.py`](https://github.com/brunoguindani/BayesianOptimization/blob/master/test.py) file, which contains over 20 usage examples using different features and techniques.

There are multiple usage modes, as we shall see in the next section.
In all cases, the mandatory arguments are:
* for the constructor, either the `f` and/or the `dataset` argument, depending on the usage mode
* for the constructor, the `pbounds` argument, a dictionary which maps the names of the optimization variables too a tuple containing their lower and upper bound
* for `maximize()`, the `init_points` and the `n_iter` arguments. These are respectively the number of randomly chosen initial points, and the maximum number of iterations of the BO algorithm.


## Usage modes
There are three main ways to use the MALIBOO library:
1) with **free functions**, i.e. which can be coded in Python and return values (either a scalar, or a dictionary with a `'value'` field).
This is the traditional method used in the original package and most other optimization libraries.
The function is given as the `f` argument of the constructor of the library object.
See the [Tutorial](#tutorial) section for such an example.
Note that this does *not* necessarily mean that the optimized function must have an analytic, closed-form expression.
For instance, a valid target function can contain a call to another command-line program (e.g. via the `subprocess` Python module), recover the output of this program, and return appropriate values based on such output.

2) with **free functions having finite domain** (aka $X$*-datset mode*): this is particularly useful for functions which have discrete input parameters.
In this case, in addition to `f`, we must also pass the `dataset` argument, which is the dataset $X$ containing the list of all domain points (either in the form of a `pandas.DataFrame`, or of the path to a `.csv` file).
The optimization process will only consider these points, and will never pick a point which is not included in the dataset.
Example:
```py
from maliboo import BayesianOptimization as BO

def target_func(x1, x2):
    return -x1 ** 2 - (x2 - 1) ** 2 + 1

optimizer = BO(f=target_func, pbounds={'x1': (999,2501), 'x2': (1,50)},
               dataset='resources/test_xyz.csv')
optimizer.maximize(init_points=2, n_iter=5)
```

3) with **tabular logs of simulated executions** (aka $Xy$*-dataset mode*). In this case, no function is given in the `f` argument, therefore no function evaluation ever takes place.
We only pass the `dataset` $Xy$, which in this case must contain both the points of the domain and the target values, whose column is is indicated by the `target_column` argument.
Again, the process will only choose points included in the dataset.
This mode is useful if, for instance, we are studying the converge of multiple algorithms, or versions thereof, but we already have a log of past executions.
This way, we can study the convergence of those algorithms without having to run again the potentially expensive black-box function.
Example:
```py
from maliboo import BayesianOptimization as BO

optimizer = BO(f=None, pbounds={'x1': (999,2501), 'x2': (1,50)},
               dataset='resources/test_xyz.csv', target_column='z')
optimizer.maximize(init_points=2, n_iter=5)
```
Notice the lack of the `f` function and the presence of the `target_column` argument.


## Documentation
Features of the library include:
* new acquisition functions integrated with Machine Learning models
* several classical acquisition functions for "unconstrained" optimization, such as `ucb` (Upper Confidence Bound), `ei` (Expected Improvement), `ei_ml`
* black-box-constrained optimization with appropriate acquisition functions: `eic`, `eic_ml`
* memory queue for discrete features: if activated by calling `maximize()` with `memory_queue_len=q`, a point visited at any iteration will not be sampled again for the next `q` iterations
* termination criteria for the BO algorithm
* `relaxation` mode when a dataset is used (i.e. in modes 2 and 3).
If `maximize()` is called with such option set to `True`, the acquisition function will be maximized over the relaxed real-numbered domain, then the maximizer found will be approximated to the closest point in the dataset (wrt the Euclidean distance).
This means that the point found at the current iteration is the discrete approximation of the solution of a continuous relaxation.
If `False` (which is also the default value), the acquisition function will only be evaluated on the dataset points as usual, therefore an exact maximizer will be found, without any approximation taking place.

For further information, please also refer to the well-documented [Python files](https://github.com/brunoguindani/BayesianOptimization/tree/master/maliboo) of the project.
