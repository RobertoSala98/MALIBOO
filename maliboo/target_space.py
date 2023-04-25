from numbers import Number
import numpy as np
import pandas as pd
from .util import ensure_rng


class TargetSpace(object):
    """
    Holds the param-space coordinates (X) and target values (Y)

    Parameters
    ----------
    target_func: function, optional (default=None)
        Target function to be maximized.

    pbounds: dict
        Dictionary with parameters names as keys and a tuple with minimum
        and maximum values. Mandatory argument.

    random_state: int, RandomState, or None, optional (default=None)
        Optionally specify a seed for a random number generator

    dataset: str, file handle, or pandas.DataFrame, optional (default=None)
        The dataset, if any, which constitutes the optimization domain (X) and possibly
        the list of target values (Y)

    target_column: str, optional (default=None)
        Name of the column that will act as the target value of the optimization.
        Only works if dataset is passed.

    debug: bool, optional (default=False)
        Whether or not to print detailed debugging information
    """
    def __init__(self, target_func=None, pbounds=None, random_state=None,
                 dataset=None, target_column=None, debug=False):
        if pbounds is None:
            raise ValueError("pbounds must be specified")

        self._debug = debug

        # Get the name of the parameters, aka the optimization variables/columns
        self._keys = sorted(pbounds)

        # Create an array with parameters bounds
        self._bounds = np.array(
            [item[1] for item in sorted(pbounds.items(), key=lambda x: x[0])],
            dtype=float
        )

        if self._debug: print("Initializing TargetSpace with bounds:", pbounds)

        # Initialize other members
        self.random_state = ensure_rng(random_state)
        self.target_func = target_func
        self.initialize_dataset(dataset, target_column)

        # preallocated memory for X and Y points
        self._params = pd.DataFrame()
        self._target = np.empty(shape=(0))

        # Other information to be recorded
        self._target_dict_info = pd.DataFrame()
        self._optimization_info = pd.DataFrame()

        if self._debug: print("TargetSpace initialization completed")


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
        return self._params.index


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


    def register(self, params, target, idx=None):
        """
        Append a point and its target value to the known data.

        Parameters
        ----------
        params: numpy.ndarray
            A single point, with x.shape[1] == self.dim

        target: float
            Target function value

        idx: int or None, optional (default=None)
            The dataset index of the point to be registered, or None if no dataset is being used

        Returns
        -------
        value: float
            The registered target value

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
        if self._debug: print("Registering params", params, "with index", idx, "and target value", target)
        value, info = self.extract_value_and_info(target)

        x_df = pd.DataFrame(params.reshape(1, -1), columns=self._keys, index=[idx], dtype=float)
        self._params = pd.concat((self._params, x_df))
        self._target = np.concatenate([self._target, [value]])
        if info:  # The return value of the target function is a dict
            if self._target_dict_info.empty:
                # Initialize member
                self._target_dict_info = pd.DataFrame(info, index=[idx])
            else:
                # Append new point to member
                info_new = pd.DataFrame(info, index=[idx])
                self._target_dict_info = pd.concat((self._target_dict_info, info_new))

        if self._debug: print("Point registered successfully")
        return value


    def register_optimization_info(self, info_new):
        """Register relevant information into self._optimization_info"""
        self._optimization_info = pd.concat((self._optimization_info, info_new))
        if self._debug: print("Registered optimization information:", info_new, sep="\n")


    def probe(self, params, idx=None):
        """
        Evaulates a single point x, to obtain the value y and then records them
        as observations.

        Parameters
        ----------
        params: dict
            A single point, with len(x) == self.dim

        idx: int or None, optional (default=None)
            The dataset index of the point to be probed, or None if no dataset is being used

        Returns
        -------
        target_value: float
            Target function value.
        """
        if self._debug: print("Probing point: index {}, value {}".format(idx, params))
        x = self._as_array(params)

        params = dict(zip(self._keys, x))
        target = self.target_func(**params)
        target_value = self.register(x, target, idx)
        if self._debug: print("Probed target value:", target_value)
        return target_value


    def random_sample(self):
        """
        Creates random points within the bounds of the space.

        Returns
        ----------
        idx: int or None
            The dataset index number of the chosen point, or None if no dataset is being used
        data: numpy.ndarray
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
            if self._debug: print("Randomly sampled dataset point: index {}, value {}".format(idx, data))
        else:
            idx = None
            data = np.empty((1, self.dim))
            for col, (lower, upper) in enumerate(self._bounds):
                data.T[col] = self.random_state.uniform(lower, upper, size=1)
            if self._debug: print("Uniform randomly sampled point: value {}".format(data))
        return idx, self.array_to_params(data.ravel())


    def max(self):
        """Get maximum target value found and corresponding parameters."""
        try:
            res = {
                'target': self.target.max(),
                'params': dict(
                    zip(self.keys, self.params.values[self.target.argmax()])
                )
            }
        except ValueError:
            res = {}
        return res


    def res(self):
        """Get all target values found and corresponding parameters."""
        params = [dict(zip(self.keys, p)) for p in self.params.values]

        return [
            {"target": target} | param
            for target, param in zip(self.target, params)
        ]


    def set_bounds(self, new_bounds):
        """
        A method that allows changing the lower and upper searching bounds

        Parameters
        ----------
        new_bounds: dict
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
        target: float
            The actual numeric value
        info: dict
            The full input dictionary, or an empty dictionary if target was purely numeric
        """
        if isinstance(target, Number):
            if self._debug: print("Extracting info: target", target, "is scalar")
            return target, {}
        elif isinstance(target, dict):
            key = 'value'
            if key not in target:
                raise ValueError("If target function is a dictionary, it must contain the '{}' field".format(key))
            if self._debug: print("Extracting info: target is a dict with value", target[key])
            return target[key], target
        else:
            raise ValueError("Unrecognized return type '{}' in target function".format(type(target)))


    def initialize_dataset(self, dataset=None, target_column=None):
        """
        Checks and loads the dataset as well as other utilities. The dataset loaded in this class by
        this method is constant and will not change throughout the optimization procedure.

        Parameters
        ----------
        dataset: str, file handle, or pandas.DataFrame, optional (default=None)
            The dataset which constitutes the optimization domain, if any.

        target_column: str, optional (default=None)
            Name of the column that will act as the target value of the optimization.
            Only works if dataset is passed.
        """
        if dataset is None:
            self._dataset = None
            if self._debug: print("initialize_dataset(): dataset is None")
            return

        if isinstance(dataset, pd.DataFrame):
            self._dataset = dataset
        else:
            try:
                self._dataset = pd.read_csv(dataset)
            except:
                raise ValueError("Dataset must be a pandas.DataFrame or a (path to a) valid file")

        if self._debug: print("Shape of initialized dataset is", self._dataset.shape)

        # Check for banned column names
        banned_columns = ('index', 'params', 'target', 'value', 'acquisition', 'ml_mape')
        for col in banned_columns:
            if col in self._dataset.columns:
                raise ValueError("Column name '{}' is not allowed in a dataset, please change it".format(col))

        # Check for relevant class members
        for attr in ('_bounds', '_keys'):
            if not hasattr(self, attr):
                raise ValueError("'self.{}' must be set before initialize_dataset() is called".format(attr))

        # Set target column and check for missing columns
        self._target_column = target_column
        missing_cols = set(self._keys) - set(self._dataset.columns)
        if missing_cols:
            raise ValueError("Columns {} indicated in pbounds are missing "
                             "from the dataset".format(missing_cols))
        if target_column is not None and target_column not in self._dataset:
            raise ValueError("The specified target column '{}' is not present in the dataset".format(target_column))

        # Check that bounds are respected by the corresponding dataset columns
        for key, (lb, ub) in zip(self._keys, self._bounds):
            if self.dataset[key].min() < lb or self.dataset[key].max() >= ub:
                raise ValueError("Dataset values for '{}' column are not consistent with bounds".format(key))


    def find_point_in_dataset(self, params):
        """
        Find index of a matching row in the dataset.

        Parameters
        ----------
        params: dict
            The point to be found in the dataset

        Returns
        -------
        idx: int
            The dataset index of the point found
        target_val: float
            Dataset target value associated to the point found
        """
        dataset_vals = self._dataset[self._keys].values
        x = self.params_to_array(params)

        # Find matching rows and choose randomly one of them
        matches = np.where((dataset_vals == x).all(axis=1))[0]
        if len(matches) == 0:
            raise ValueError("{} not found in dataset".format(params))
        idx = self.random_state.choice(matches)
        target_val = self.dataset.loc[idx, self._target_column]
        if self._debug: print("Located {} as data[{}], with target value {}".format(x, idx, target_val))

        return idx, target_val
