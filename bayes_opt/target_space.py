from numbers import Number
import numpy as np
import pandas as pd
from .util import ensure_rng


def _hashable(x):
    """ ensure that an point is hashable by a python dict """
    return tuple(map(float, x))


class TargetSpace(object):
    """
    Holds the param-space coordinates (X) and target values (Y)
    Allows for constant-time appends while ensuring no duplicates are added

    Example
    -------
    >>> def target_func(p1, p2):
    >>>     return p1 + p2
    >>> pbounds = {'p1': (0, 1), 'p2': (1, 100)}
    >>> space = TargetSpace(target_func, pbounds, random_state=0)
    >>> x = space.random_points(1)[0]
    >>> y = space.register_point(x)
    >>> assert self.max_point()['max_val'] == y
    """

    def __init__(self, target_func=None, pbounds=None, random_state=None,
                 dataset=None, target_column=None):
        """
        Parameters
        ----------
        target_func : function, optional(default=None)
            Target function to be maximized.

        pbounds : dict
            Dictionary with parameters names as keys and a tuple with minimum
            and maximum values.

        random_state : int, RandomState, or None, optional(default=None)
            Optionally specify a seed for a random number generator

        dataset: str, file handle, or pandas.DataFrame, optional(default=None)
            The dataset, if any, which constitutes the optimization domain and possibly
            the list of target values

        target_column: str, optional(default=None)
            Name of the column that will act as the target value of the optimization.
            Only works if dataset is passed.
        """
        if pbounds is None:
            raise ValueError("pbounds must be specified")

        # Get the name of the parameters, aka the optimization columns
        self._keys = sorted(pbounds)

        # Initialize other members
        self.random_state = ensure_rng(random_state)
        self.target_func = target_func
        self.initialize_dataset(dataset, target_column)
        # List of dataset indexes of points, or Nones if no dataset is used
        self._indexes = []

        # Create an array with parameters bounds
        self._bounds = np.array(
            [item[1] for item in sorted(pbounds.items(), key=lambda x: x[0])],
            dtype=float
        )

        # preallocated memory for X and Y points
        self._params = np.empty(shape=(0, self.dim))
        self._target = np.empty(shape=(0))
        self._target_dict_info = pd.DataFrame()
        self._target_dict_key = 'value'

    def __len__(self):
        assert len(self._params) == len(self._target)
        return len(self._target)

    @property
    def empty(self):
        return len(self) == 0

    @property
    def params(self):
        return self._params

    @property
    def target(self):
        return self._target

    @property
    def dim(self):
        return len(self._keys)

    @property
    def keys(self):
        return self._keys

    @property
    def bounds(self):
        return self._bounds

    @property
    def dataset(self):
        return self._dataset

    @property
    def target_column(self):
        return self._target_column

    @property
    def indexes(self):
        return self._indexes

    def params_to_array(self, params):
        try:
            assert set(params) == set(self.keys)
        except AssertionError:
            raise ValueError(
                "Parameters' keys ({}) do ".format(sorted(params)) +
                "not match the expected set of keys ({}).".format(self.keys)
            )
        return np.asarray([params[key] for key in self.keys])

    def array_to_params(self, x):
        try:
            assert len(x) == len(self.keys)
        except AssertionError:
            raise ValueError(
                "Size of array ({}) is different than the ".format(len(x)) +
                "expected number of parameters ({}).".format(len(self.keys))
            )
        return dict(zip(self.keys, x))

    def _as_array(self, x):
        try:
            x = np.asarray(x, dtype=float)
        except TypeError:
            x = self.params_to_array(x)

        x = x.ravel()
        try:
            assert x.size == self.dim
        except AssertionError:
            raise ValueError(
                "Size of array ({}) is different than the ".format(len(x)) +
                "expected number of parameters ({}).".format(len(self.keys))
            )
        return x

    def register(self, params, target):
        """
        Append a point and its target value to the known data.

        Parameters
        ----------
        params : numpy.ndarray
            a single point, with len(x) == self.dim

        target : float
            target function value

        Raises
        ------
        KeyError:
            if the point is not unique

        Notes
        -----
        runs in ammortized constant time

        Example
        -------
        >>> pbounds = {'p1': (0, 1), 'p2': (1, 100)}
        >>> space = TargetSpace(lambda p1, p2: p1 + p2, pbounds)
        >>> len(space)
        0
        >>> x = np.array([0, 0])
        >>> y = 1
        >>> space.add_observation(x, y)
        >>> len(space)
        1
        """
        x = self._as_array(params)
        value, info = self.extract_value_and_info(target)

        self._params = np.concatenate([self._params, x.reshape(1, -1)])
        self._target = np.concatenate([self._target, [value]])
        if info:  # The return value of the target function is a dict
            if self._target_dict_info.empty:
                # Initialize member
                self._target_dict_info = pd.DataFrame(info, index=[0])
            else:
                # Append new point to member
                self._target_dict_info = pd.concat((self._target_dict_info, pd.DataFrame(info, index=[0])), ignore_index=True)
            # print(self._target_dict_info)  # !DEBUG!

    def probe(self, params):
        """
        Evaulates a single point x, to obtain the value y and then records them
        as observations.

        Parameters
        ----------
        x : ndarray
            a single point, with len(x) == self.dim

        Returns
        -------
        y : float
            target function value.
        """
        x = self._as_array(params)

        params = dict(zip(self._keys, x))
        target = self.target_func(**params)
        self.register(x, target)
        ret, _ = self.extract_value_and_info(target)
        return ret

    def random_sample(self):
        """
        Creates random points within the bounds of the space.

        Returns
        ----------
        idx: int or None
            index number of chosen point, or None if no dataset is used
        data: ndarray
            [num x dim] array points with dimensions corresponding to `self._keys`

        Example
        -------
        >>> target_func = lambda p1, p2: p1 + p2
        >>> pbounds = {'p1': (0, 1), 'p2': (1, 100)}
        >>> space = TargetSpace(target_func, pbounds, random_state=0)
        >>> space.random_points(1)
        array([[ 55.33253689,   0.54488318]])
        """
        # TODO: support integer, category, and basic scipy.optimize constraints
        if self.dataset is not None:
            # Recover random row from dataset
            idx = self.random_state.choice(self.dataset.index)
            data = self.dataset.loc[idx, self.keys].to_numpy()
        else:
            idx = None
            data = np.empty((1, self.dim))
            for col, (lower, upper) in enumerate(self._bounds):
                data.T[col] = self.random_state.uniform(lower, upper, size=1)
        return idx, self.array_to_params(data.ravel())

    def max(self):
        """Get maximum target value found and corresponding parameters."""
        try:
            res = {
                'target': self.target.max(),
                'params': dict(
                    zip(self.keys, self.params[self.target.argmax()])
                )
            }
        except ValueError:
            res = {}
        return res

    def res(self):
        """Get all target values found and corresponding parameters."""
        params = [dict(zip(self.keys, p)) for p in self.params]

        return [
            {"target": target} | param
            for target, param in zip(self.target, params)
        ]

    def set_bounds(self, new_bounds):
        """
        A method that allows changing the lower and upper searching bounds

        Parameters
        ----------
        new_bounds : dict
            A dictionary with the parameter name and its new bounds
        """
        for row, key in enumerate(self.keys):
            if key in new_bounds:
                self._bounds[row] = new_bounds[key]

    def extract_value_and_info(self, target):
        """
        Return function numeric value and further information

        The return value of the target function can also be a dictionary. In this case,
        we return separately its 'value' field as the true target value, and we also
        return the whole dictionary separately. Otherwise, if the target function is
        purely numeric, we return an empty information dictionary

        Parameters
        ----------
        target: numeric value or dict
            An object returned by the target function

        Returns
        -------
        target: numeric value
            The actual numeric value
        info: dict
            The full input dictionary, or an empty dictionary if target was purely numeric
        """
        if isinstance(target, Number):
            return target, {}
        elif isinstance(target, dict):
            if self._target_dict_key not in target:
                raise ValueError("If target function is a dictionary, it must "
                                 "contain the '{}' field".format(self._target_dict_key))
            return target[self._target_dict_key], target
        else:
            raise ValueError("Unrecognized return type '{}' in target function".format(type(target)))

    def initialize_dataset(self, dataset=None, target_column=None):
        """
        Checks and loads the dataset as well as other utilities. The dataset loaded in this class by
        this method is constant and will not change throughout the optimization procedure.

        Parameters
        ----------
        dataset: str, file handle, or pandas.DataFrame, optional(default=None)
            The dataset which constitutes the optimization domain, if any.

        target_column: str, optional(default=None)
            Name of the column that will act as the target value of the optimization.
            Only works if dataset is passed.
        """
        if dataset is None:
            self._dataset = None
            return

        if type(dataset) == pd.DataFrame:
            self._dataset = dataset
        else:
            try:
                self._dataset = pd.read_csv(dataset)
            except:
                raise ValueError("Dataset must be a pandas.DataFrame or a (path to a) valid file")

        # Check for banned column names
        banned_columns = ('index', 'params', 'target', 'value')
        for col in banned_columns:
            if col in self._dataset.columns:
                raise ValueError("Column name '{}' is not allowed in a dataset, please change it".format(col))

        # Set target column and check for missing columns
        self._target_column = target_column
        if not hasattr(self, '_keys'):
            raise ValueError("self._keys must be set before initialize_dataset() is called")
        missing_cols = set(self._keys) - set(self._dataset.columns)
        if missing_cols:
            raise ValueError("Columns {} indicated in pbounds are missing "
                             "from the dataset".format(missing_cols))
        if target_column is not None and target_column not in self._dataset:
            raise ValueError("The specified target column '{}' is not present in the dataset".format(target_column))
