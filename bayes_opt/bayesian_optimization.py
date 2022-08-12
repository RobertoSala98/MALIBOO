import itertools
import numpy as np
import os
import pandas as pd
import warnings

from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_percentage_error as mape

from .target_space import TargetSpace
from .event import Events, DEFAULT_EVENTS
from .logger import _get_default_logger
from .util import UtilityFunction, acq_max, ensure_rng

class Queue:
    def __init__(self):
        self._queue = []

    @property
    def empty(self):
        return len(self) == 0

    def __len__(self):
        return len(self._queue)

    def __next__(self):
        if self.empty:
            raise StopIteration("Queue is empty, no more objects to retrieve.")
        obj = self._queue[0]
        self._queue = self._queue[1:]
        return obj

    def next(self):
        return self.__next__()

    def add(self, obj):
        """Add object to end of queue."""
        self._queue.append(obj)


class Observable(object):
    """

    Inspired/Taken from
        https://www.protechtraining.com/blog/post/879#simple-observer
    """

    def __init__(self, events):
        # maps event names to subscribers
        # str -> dict
        self._events = {event: dict() for event in events}

    def get_subscribers(self, event):
        return self._events[event]

    def subscribe(self, event, subscriber, callback=None):
        if callback is None:
            callback = getattr(subscriber, 'update')
        self.get_subscribers(event)[subscriber] = callback

    def unsubscribe(self, event, subscriber):
        del self.get_subscribers(event)[subscriber]

    def dispatch(self, event):
        for _, callback in self.get_subscribers(event).items():
            callback(event, self)


class BayesianOptimization(Observable):
    """
    This class takes the function to optimize as well as the parameters bounds
    in order to find which values for the parameters yield the maximum value
    using bayesian optimization.

    Parameters
    ----------
    f: function, optional (default=None)
        Function to be maximized.

    pbounds: dict, optional (default=None)
        Dictionary with parameters names as keys and a tuple with minimum
        and maximum values.
        It is actually a mandatory parameter: if the default value None is left,
        an error will be raised.

    random_state: int or numpy.random.RandomState, optional (default=None)
        If the value is an integer, it is used as the seed for creating a
        numpy.random.RandomState. Otherwise the random state provieded it is used.
        When set to None, an unseeded random state is generated.

    verbose: int, optional (default=2)
        The level of verbosity.

    bounds_transformer: DomainTransformer, optional (default=None)
        If provided, the transformation is applied to the bounds.

    dataset: str, file handle, or pandas.DataFrame, optional (default=None)
        The dataset specified by the user, or a path/file handle of such file.

    output_path: str, optional (default=None)
        Path to directory in which the results are written. Default value is the working directory.

    target_column: str, optional (default=None)
        Name of the column that will act as the target value of the optimization.
        Only works if dataset is passed.

    debug: bool, optional (default=False)
        Whether or not to print detailed debugging information

    Methods
    -------
    probe()
        Evaluates the function on the given points.
        Can be used to guide the optimizer.

    maximize()
        Tries to find the parameters that yield the maximum value for the
        given function.

    set_bounds()
        Allows changing the lower and upper searching bounds
    """

    def __init__(self, f=None, pbounds=None, random_state=None, verbose=2, bounds_transformer=None,
                 dataset=None, output_path=None, target_column=None, debug=False):

        # Initialize members from arguments
        self._random_state = ensure_rng(random_state)
        self._verbose = verbose
        self._debug = debug
        self._bounds_transformer = bounds_transformer
        self._output_path = os.getcwd() if output_path is None else os.path.join(output_path)

        # Check for coherence among constructor arguments
        if pbounds is None:
            raise ValueError("pbounds must be specified")
        if f is None and target_column is None:
            raise ValueError("Target column must be specified if no target function f is given")
        if f is not None and target_column is not None:
                raise ValueError("Target column cannot be provided if target function f is also provided")
        if target_column is not None and dataset is None:
            raise ValueError("Dataset must be specified for the given target column")

        # Data structure containing the function to be optimized, the bounds of
        # its domain, and a record of the evaluations we have done so far
        self._space = TargetSpace(target_func=f, pbounds=pbounds, random_state=random_state, dataset=dataset,
                                  target_column=target_column, debug=debug)
        self._queue = Queue()

        if self._bounds_transformer:
            try:
                self._bounds_transformer.initialize(self._space)
            except (AttributeError, TypeError):
                raise TypeError('The transformer must be an instance of '
                                'DomainTransformer')

        # Internal GP regressor
        self._gp = GaussianProcessRegressor(
            kernel=Matern(nu=2.5),
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=5,
            random_state=self._random_state,
        )

        super(BayesianOptimization, self).__init__(events=DEFAULT_EVENTS)

        if self._debug: print("BayesianOptimization initialization completed")

    @property
    def space(self):
        return self._space

    @property
    def max(self):
        return self._space.max()

    @property
    def res(self):
        return self._space.res()

    @property
    def dataset(self):
        return self._space.dataset

    def register(self, params, target):
        """Expect observation with known target"""
        self._space.register(params, target)
        self.dispatch(Events.OPTIMIZATION_STEP)

    def probe(self, params, lazy=True):
        """
        Evaluates the function on the given points. Useful to guide the optimizer.

        Parameters
        ----------
        params: dict or list
            The parameters where the optimizer will evaluate the function.

        lazy: bool, optional (default=True)
            If True, the optimizer will evaluate the points when calling
            maximize(). Otherwise it will evaluate it at the moment.
        """
        if lazy:
            # TODO actually, an (index, params) tuple should be passed to add()
            self._queue.add(params)
        else:
            self._space.probe(params)
            self.dispatch(Events.OPTIMIZATION_STEP)

    def suggest(self, utility_function):
        """Most promising point to probe next"""
        if len(self._space) == 0:
            idx, x_rand = self._space.random_sample()
            return idx, self._space.array_to_params(x_rand)

        # Sklearn's GP throws a large number of warnings at times, but
        # we don't really need to see them here.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._gp.fit(self._space.params, self._space.target)
            # If requird, train ML model with all space parameters data collected so far
            if 'ml' in utility_function.kind:
                model = self.get_ml_model(y_name=utility_function.ml_target)
                utility_function.set_ml_model(model)

        if self.dataset is None:
            dataset_acq = None
        else:
            # Flatten memory queue (a list of indexes lists) to one single list
            idxs = list(itertools.chain.from_iterable(self.memory_queue))
            # Create mask to select rows whose index is not included in idxs
            mask = np.ones(self.dataset.shape[0], np.bool)
            mask[idxs] = 0
            # Create dataset to be passed to acq_max()
            dataset_acq = self.dataset.loc[mask, self._space.keys]

        # Find argmax of the acquisition function
        idx, suggestion = acq_max(
            ac=utility_function.utility,
            gp=self._gp,
            y_max=self._space.target.max(),
            bounds=self._space.bounds,
            random_state=self._random_state,
            dataset=dataset_acq,
            debug=self._debug
        )

        if self.dataset is not None:
            self.update_memory_queue(self.dataset[self._space.keys], suggestion)

        return idx, self._space.array_to_params(suggestion)

    def _prime_queue(self, init_points):
        """Make sure there's something in the queue at the very beginning."""
        if self._queue.empty and self._space.empty:
            init_points = max(init_points, 1)

        if self._debug: print("_prime_queue(): initializing", init_points, "random points")

        for _ in range(init_points):
            idx, x_init = self._space.random_sample()
            self._queue.add((idx, x_init))
            if self.dataset is not None:
                self.update_memory_queue(self.dataset[self._space.keys],
                                         self._space.params_to_array(x_init))

    def _prime_subscriptions(self):
        if not any([len(subs) for subs in self._events.values()]):
            _logger = _get_default_logger(self._verbose)
            self.subscribe(Events.OPTIMIZATION_START, _logger)
            self.subscribe(Events.OPTIMIZATION_STEP, _logger)
            self.subscribe(Events.OPTIMIZATION_END, _logger)

    def maximize(self,
                 init_points=5,
                 n_iter=25,
                 acq='ucb',
                 kappa=2.576,
                 kappa_decay=1,
                 kappa_decay_delay=0,
                 xi=0.0,
                 ml_info={},
                 eic_info={},
                 memory_queue_len=0,
                 **gp_params):
        """
        Probes the target space to find the parameters that yield the maximum
        value for the given function.

        Parameters
        ----------
        init_points: int, optional (default=5)
            Number of iterations before the explorations starts the exploration
            for the maximum.

        n_iter: int, optional (default=25)
            Number of iterations where the method attempts to find the maximum
            value.

        acq: str
            The acquisition method used. Among others:
                * 'ucb' stands for the Upper Confidence Bounds method
                * 'ei' is the Expected Improvement method
                * 'poi' is the Probability Of Improvement criterion.

        kappa: float, optional (default=2.576)
            Parameter to indicate how closed are the next parameters sampled.
                Higher value = favors spaces that are least explored.
                Lower value = favors spaces where the regression function is the
                highest.

        kappa_decay: float, optional (default=1)
            `kappa` is multiplied by this factor every iteration.

        kappa_decay_delay: int, optional (default=0)
            Number of iterations that must have passed before applying the decay
            to `kappa`.

        xi: float, optional (default=0.0)
            [unused]

        ml_info: dict, optional (default={})
            Information required for using Machine Learning models. Namely, ml_info['target'] is
            the name of the target quantity and ml_info['bounds'] is a tuple with its lower and
            upper bounds.

        eic_info: dict, optional (default={})
            Information required for using the Expected Improvement with Constraints acquisition.
            EIC assumes that the target function has the form f(x) = P(x) g(x) + Q(x) and is bound
            to the constraint Gmin <= g(x) <= Gmax. Then, eic_info['bounds'] is a tuple with Gmin
            and Gmax, and eic_info['P_func'] and eic_info['Q_func'] are the functions in the
            definition of f. The default values for the latter are P(x) == 1 and Q(x) == 0.

        memory_queue_len: int, optional (default=0)
            Length of FIFO memory queue. If used alongside a dataset, at each iteration,
            values which have already been chosen in the last memory_queue_len iterations
            will not be considered
        """
        # Initialize the memory queue, a list of lists of forbidden indexes for the current iteration
        if self._debug: print("Starting maximize()")
        self.memory_queue_len = memory_queue_len
        self.memory_queue = []

        # Initialize other stuff
        self._prime_subscriptions()
        self.dispatch(Events.OPTIMIZATION_START)
        self._prime_queue(init_points)
        self.set_gp_params(**gp_params)

        util = UtilityFunction(kind=acq,
                               kappa=kappa,
                               xi=xi,
                               kappa_decay=kappa_decay,
                               kappa_decay_delay=kappa_decay_delay,
                               ml_info=ml_info,
                               eic_info=eic_info,
                               debug=self._debug)
        iteration = 0

        while not self._queue.empty or iteration < n_iter:
            # Sample new point from GP
            try:
                idx, x_probe = next(self._queue)
                if self._debug: print("Selected point from queue: index {}, value {}".format(idx, x_probe))
            except StopIteration:
                util.update_params()
                idx, x_probe = self.suggest(util)
                if self._debug: print("Iteration {}, suggested point: index {}, value {}".format(iteration, idx, x_probe))
                iteration += 1

            if x_probe is None:
                raise ValueError("No point found")

            self._space.indexes.append(idx)

            # Register new point
            if self.dataset is None or self._space.target_column is None:
                # No dataset, or dataset for X only: we evaluate the target function directly
                if self._debug: print("No dataset, or dataset for X only: evaluating target function")
                self.probe(x_probe, lazy=False)
            else:
                # Dataset for both X and y: register point entirely from dataset without probe()
                if self._debug: print("Dataset Xy: registering dataset point")
                target_value = self.dataset.loc[idx, self._space.target_column]
                self.register(x_probe, target_value)

        if self._bounds_transformer:
            self.set_bounds(
                self._bounds_transformer.transform(self._space))
        self.save_res_to_csv()
        self.dispatch(Events.OPTIMIZATION_END)

    def save_res_to_csv(self):
        """
        Save results of the optimization to csv files located in results directory
        """
        os.makedirs(self._output_path, exist_ok=True)
        results = pd.DataFrame.from_dict(self.res)
        results['index'] = self._space.indexes
        results.set_index('index', inplace=True)
        results.to_csv(os.path.join(self._output_path, "results.csv"), index=True)

        print("Results successfully saved to " + self._output_path)

    def set_bounds(self, new_bounds):
        """
        Change the lower and upper searching bounds

        Parameters
        ----------
        new_bounds: dict
            A dictionary with the parameter name and its new bounds
        """
        self._space.set_bounds(new_bounds)

    def set_gp_params(self, **params):
        """Set parameters to the internal Gaussian Process Regressor"""
        self._gp.set_params(**params)

    def get_ml_model(self, y_name):
        """
        Returns Machine Learning model trained on current history

        Parameters
        ----------
        y_name: str
            Name of the dataset column that will act as the target of the regression

        Returns
        -------
        model: sklearn.model object
            The trained ML model
        """
        # Build training dataset for the ML model
        X = pd.DataFrame(self._space._params, columns=self._space.keys)
        if self._debug: print("Dataset for ML model has shape", X.shape)
        try:
            y = self._space._target_dict_info[y_name]
        except KeyError:
            if self.dataset is None:
                raise KeyError("Target function return values must have '{}' field".format(y_name))
            elif y_name in self.dataset.columns:
                y = self.dataset.loc[self._space.indexes, y_name]
            else:
                raise KeyError("Dataset must have '{}' column".format(y_name))

        # Initialize and train model
        model = Ridge()
        model.fit(X, y)

        if self._debug:
            try:
                print("Trained ML model:")
                print("Training MAPE =", mape(model.predict(X), y))
                print("Coefficients =", model.coef_)
            except:
                pass
        return model

    def update_memory_queue(self, dataset, x_new):
        """
        Updates self.memory_queue, the list of dataset entries which cannot be selected
        in the current iteration. The list is always no larger than memory_queue_len elements.

        Parameters
        ----------
        dataset: pandas.DataFrame
            The dataset on which to perform filtering

        x_new: numpy.ndarray
            The lasted selected point, which is to be included in the memory queue
        """
        if self.memory_queue_len == 0:
            if self._debug: print("No memory queue to be updated")
            return

        self.memory_queue.append([])
        dataset_vals = dataset.values

        # Place indexes of data equal to x_new in memory queue
        for i in range(dataset_vals.shape[0]):
            if np.array_equal(dataset_vals[i], x_new):
                self.memory_queue[-1].append(i)

        # Remove oldest entry if exceeding max length
        if len(self.memory_queue) > self.memory_queue_len:
            self.memory_queue.pop(0)
            if self._debug: print("Exceeded memory queue length {}, removing first entry".format(self.memory_queue_len))

        if self._debug:
            print("Updated memory queue:", self.memory_queue)
            counts = [len(_) for _ in self.memory_queue]
            print("Counts in memory queue: {} (total: {})".format(counts, sum(counts)))
