import os
import random
import warnings

import numpy.linalg
import pandas as pd

from .target_space import TargetSpace
from .event import Events, DEFAULT_EVENTS
from .logger import _get_default_logger
from .util import UtilityFunction, acq_max, ensure_rng

from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process import GaussianProcessRegressor


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
    f: function, optional(default=None)
        Function to be maximized.

    pbounds: dict, optional(default=None)
        Dictionary with parameters names as keys and a tuple with minimum
        and maximum values.
        It is actually a mandatory parameter: if the default value None is left,
        an error will be raised.

    random_state: int or numpy.random.RandomState, optional(default=None)
        If the value is an integer, it is used as the seed for creating a
        numpy.random.RandomState. Otherwise the random state provieded it is used.
        When set to None, an unseeded random state is generated.

    verbose: int, optional(default=2)
        The level of verbosity.

    bounds_transformer: DomainTransformer, optional(default=None)
        If provided, the transformation is applied to the bounds.

    dataset_path: str, optional(default=None)
            path of the dataset file specified by the user.

    output_path: str, optional(default=None)
            path to directory in which the results are written, if not specified by user it is the working directory

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

    def __init__(self, f=None, pbounds=None, random_state=None, verbose=2,
                 bounds_transformer=None,
                 dataset_path=None, output_path=None, target_column=None):

        if output_path is None:
            self.output_path = os.getcwd()
        else:
            self.output_path = output_path

        if dataset_path is None:
            self._dataset = None
        else:
            self._dataset = pd.read_csv(dataset_path)

        if target_column is None:
            self._target_column = None
        else:
            self._target_column = target_column

        if pbounds is None:
            raise ValueError("pbounds must be specified!")
        self._random_state = ensure_rng(random_state)

        if f is None and target_column is None:
            raise ValueError("target column must be specified if no function is given!")
        elif f is not None and target_column is not None:
            raise Exception("You cannot specify both function and target column, one of them must be None!")

        if target_column is not None and dataset_path is None:
            raise Exception("You must specify a dataset for the given target column!")

        # Data structure containing the function to be optimized, the bounds of
        # its domain, and a record of the evaluations we have done so far
        self._space = TargetSpace(f, pbounds, random_state)

        self._queue = Queue()

        # Internal GP regressor
        self._gp = GaussianProcessRegressor(
            kernel=Matern(nu=2.5),
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=5,
            random_state=self._random_state,
        )

        self._verbose = verbose
        self._bounds_transformer = bounds_transformer

        if self._bounds_transformer:
            try:
                self._bounds_transformer.initialize(self._space)
            except (AttributeError, TypeError):
                raise TypeError('The transformer must be an instance of '
                                'DomainTransformer')

        super(BayesianOptimization, self).__init__(events=DEFAULT_EVENTS)

    @property
    def space(self):
        return self._space

    @property
    def max(self):
        return self._space.max()

    @property
    def res(self):
        return self._space.res()

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

        lazy: bool, optional(default=True)
            If True, the optimizer will evaluate the points when calling
            maximize(). Otherwise it will evaluate it at the moment.
        """
        if lazy:
            self._queue.add(params)
        else:
            self._space.probe(params)
            self.dispatch(Events.OPTIMIZATION_STEP)

    def suggest(self, utility_function):
        """Most promising point to probe next"""
        if len(self._space) == 0:
            return self._space.array_to_params(self._space.random_sample())

        # Sklearn's GP throws a large number of warnings at times, but
        # we don't really need to see them here.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._gp.fit(self._space.params, self._space.target)

        # Finding argmax of the acquisition function.
        suggestion = acq_max(
            ac=utility_function.utility,
            gp=self._gp,
            y_max=self._space.target.max(),
            bounds=self._space.bounds,
            random_state=self._random_state
        )

        return self._space.array_to_params(suggestion)

    def _prime_queue(self, init_points):
        """Make sure there's something in the queue at the very beginning."""
        if self._queue.empty and self._space.empty:
            init_points = max(init_points, 1)

        for _ in range(init_points):
            self._queue.add(self._space.random_sample())

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
                 **gp_params):
        """
        Probes the target space to find the parameters that yield the maximum
        value for the given function.

        Parameters
        ----------
        init_points : int, optional(default=5)
            Number of iterations before the explorations starts the exploration
            for the maximum.

        n_iter: int, optional(default=25)
            Number of iterations where the method attempts to find the maximum
            value.

        acq: {'ucb', 'ei', 'poi'}
            The acquisition method used.
                * 'ucb' stands for the Upper Confidence Bounds method
                * 'ei' is the Expected Improvement method
                * 'poi' is the Probability Of Improvement criterion.

        kappa: float, optional(default=2.576)
            Parameter to indicate how closed are the next parameters sampled.
                Higher value = favors spaces that are least explored.
                Lower value = favors spaces where the regression function is the
                highest.

        kappa_decay: float, optional(default=1)
            `kappa` is multiplied by this factor every iteration.

        kappa_decay_delay: int, optional(default=0)
            Number of iterations that must have passed before applying the decay
            to `kappa`.

        xi: float, optional(default=0.0)
            [unused]

        """
        self._prime_subscriptions()
        self.dispatch(Events.OPTIMIZATION_START)
        self._prime_queue(init_points)
        self.set_gp_params(**gp_params)

        util = UtilityFunction(kind=acq,
                               kappa=kappa,
                               xi=xi,
                               kappa_decay=kappa_decay,
                               kappa_decay_delay=kappa_decay_delay)
        iteration = 0

        # if user specifies a dataset it takes approximated points from it
        if self._dataset is not None:
            exact_x_dict = []
            while not self._queue.empty or iteration < n_iter:
                try:
                    x_probe = next(self._queue)
                except StopIteration:
                    util.update_params()
                    x_probe = self.suggest(util)
                    iteration += 1

                try:
                    exact_x_dict.append(dict(zip(self._space.keys, x_probe.T)))

                except AttributeError:
                    exact_x_dict.append(x_probe)

                approximation = self.get_approximation(self._dataset, x_probe)
                if self._target_column is not None and approximation is not None:

                    self._space.register(approximation["params"], approximation["target"])
                    self.dispatch(Events.OPTIMIZATION_STEP)

                elif approximation is not None:
                    self.probe(approximation, lazy=False)
                else:
                    self.probe(x_probe, lazy=False)

                if self._bounds_transformer:
                    self.set_bounds(
                        self._bounds_transformer.transform(self._space))
            self.dispatch(Events.OPTIMIZATION_END)
            self.save_res_to_csv(True, exact_x=exact_x_dict)

        else:
            while not self._queue.empty or iteration < n_iter:
                try:
                    x_probe = next(self._queue)
                except StopIteration:
                    util.update_params()
                    x_probe = self.suggest(util)
                    iteration += 1

                self.probe(x_probe, lazy=False)

                if self._bounds_transformer:
                    self.set_bounds(
                        self._bounds_transformer.transform(self._space))

            self.dispatch(Events.OPTIMIZATION_END)
            self.save_res_to_csv(False)

    def get_approximation(self, dataset, x_probe):
        """
        Method to get from the dataset passed by the user the nearest point to the x_probe point

        Parameters
        ----------

        dataset: pandas.DataFrame
            dataset specified by the user

        x_probe: dict
            point found by the optimization process

        Returns
        -------
            approximations : dict[]

            approximated x_probe, with corresponding target value, if target column is specified by the user

        """

        try:
            x_array = numpy.array(list(x_probe.values()))

        except AttributeError:
            x_array = x_probe

        min_distance = None
        min_index = None
        approximations = []

        if self._target_column is None:
            for row in dataset.itertuples():

                dataset_tuple = numpy.array(row[1:])

                res = numpy.linalg.norm(x_array - dataset_tuple, 2)

                if min_distance is None:
                    min_distance = res
                    approximations = [self._space.array_to_params(dataset_tuple)]
                elif res == min_distance:
                    approximations.append(self._space.array_to_params(dataset_tuple))
                elif res < min_distance:
                    min_distance = res
                    approximations = [self._space.array_to_params(dataset_tuple)]
            return random.choice(approximations)
        else:
            for row in dataset.loc[:, dataset.columns != self._target_column].itertuples():

                dataset_tuple = numpy.array(row[1:])

                res = numpy.linalg.norm(x_array - dataset_tuple, 2)

                if min_distance is None:
                    min_index = row[0]
                    min_distance = res
                    approximations = [
                        {
                            "target": dataset.iloc[min_index][self._target_column],
                            "params": self._space.array_to_params(dataset_tuple)
                        }]
                elif res == min_distance:

                    min_index = row[0]
                    min_distance = res
                    approximations.append(
                        {
                            "target": dataset.iloc[min_index][self._target_column],
                            "params": self._space.array_to_params(dataset_tuple)
                        })
                elif res < min_distance:
                    min_index = row[0]
                    min_distance = res
                    approximations = [
                        {
                            "target": dataset.iloc[min_index][self._target_column],
                            "params": self._space.array_to_params(dataset_tuple)
                        }]
            return random.choice(approximations)

    def save_res_to_csv(self, is_approximation, exact_x=None):
        """
        A method to save results of the optimization to csv files located in results directory

        Parameters
        ----------

        is_approximation: bool
            true if the user passes a dataset as input
        exact_x : list[dict]
            contains exact x_probe
        """
        if is_approximation:

            try:
                os.makedirs(self.output_path)
            except FileExistsError:
                pass

            approximation_res = pd.DataFrame.from_dict(self.res)
            approximation_res.to_csv(os.path.join(self.output_path, "results.csv"), index=False)

            exact_points = pd.DataFrame.from_dict(exact_x)
            exact_points.to_csv(os.path.join(self.output_path, "results_exact.csv"), index=False)
            print("Results successfully saved to " + self.output_path)

        else:
            try:
                os.makedirs(self.output_path)
            except FileExistsError:
                pass

            exact_res = pd.DataFrame.from_dict(self.res)
            exact_res.to_csv(os.path.join(self.output_path, "results.csv"), index=False)

            print("Results successfully saved to " + self.output_path)

    def set_bounds(self, new_bounds):
        """
        A method that allows changing the lower and upper searching bounds

        Parameters
        ----------
        new_bounds : dict
            A dictionary with the parameter name and its new bounds
        """
        self._space.set_bounds(new_bounds)

    def set_gp_params(self, **params):
        """Set parameters to the internal Gaussian Process Regressor"""
        self._gp.set_params(**params)
