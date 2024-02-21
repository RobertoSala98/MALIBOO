import warnings
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
import pandas as pd
from math import exp
from sys import float_info
import os
import csv
import matplotlib.pyplot as plt
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process import GaussianProcessRegressor
from math import sqrt
import random


class CustomRBFKernel(RBF):
    def __init__(self, length_scale=1.0, sigma_2=1.0, **kwargs):
        super().__init__(length_scale=length_scale, **kwargs)
        self.sigma_2 = sigma_2

    def __call__(self, X, Y=None, eval_gradient=False):
        K = super().__call__(X, Y, eval_gradient=eval_gradient)
        if not eval_gradient:
            return self.sigma_2 * K
        else:
            try:
                K, K_gradient = K
                return self.sigma_2 * K, self.sigma_2 * K_gradient
            except TypeError:  # K is not a tuple, indicating no gradient
                return self.sigma_2 * K, None


def min_max_normalize(vector):

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        min_val = min(vector)
        max_val = max(vector)

        if max_val > min_val:
            return [(x - min_val) / (max_val - min_val) for x in vector]
        
        else:
            # should be a vector of ones
            return vector


def acq_max(ac, gp, y_max, bounds, random_state, n_warmup=10000, n_iter=10, dataset=None,
            debug=False, iter_num=0, at_least_one_feasible_found=True, kind='ucb', 
            beta=1.0, l=1.0, old_x=[], old_y=[], sigma_2=1.0, beta_h=1.0, l_h=1.0):
    """
    A function to find the maximum of the acquisition function

    It uses a combination of random sampling (cheap) and the 'L-BFGS-B'
    optimization method. First by sampling `n_warmup` (1e5) points at random,
    and then running L-BFGS-B from `n_iter` (250) random starting points.

    Parameters
    ----------
    ac: function
        The acquisition function object that return its point-wise value

    gp: sklearn.gaussian_process.GaussianProcessRegressor object
        A gaussian process fitted to the relevant data

    y_max: float
        The current maximum known (aka incumbent) value of the target function

    bounds: dict
        The variables bounds to limit the search of the acq max

    random_state: numpy.RandomState object
        Instance of a random number generator

    n_warmup: int, optional (default=10000)
        Number of times to randomly sample the aquisition function

    n_iter: int, optional (default=10)
        Number of times to run scipy.minimize

    dataset: pandas.DataFrame, optional (default=None)
        The (possibly reduced) domain dataset, if any, on which the maximum is to be found

    debug: bool, optional (default=False)
        Whether or not to print detailed debugging information

    Returns
    -------
    x_max: numpy.ndarray
        The arg max of the acquisition function

    idx: int or None
        The dataset index of the arg max of the acquisition function, or None if no dataset is being used

    max_acq: float
        The computed maximum of the acquisition function, namely ac(x_max)
    """
    # Warm up with random points or dataset points
    if debug: print("Starting acq_max()\nIncumbent target: y_max =", y_max)
    if dataset is not None:
        if debug: print("Dataset passed to initial grid has shape", dataset.shape)
        x_tries = dataset.values
    else:
        if debug: print("No dataset, initial grid will be random with shape {}".format((n_warmup, bounds.shape[0])))
        x_tries = random_state.uniform(bounds[:, 0], bounds[:, 1],
                                       size=(n_warmup, bounds.shape[0]))
        
    ys = ac(x_tries, gp=gp, y_max=y_max, iter_num=iter_num, at_least_one_feasible_found=at_least_one_feasible_found)
    if debug: print("Acquisition evaluated successfully on grid")
    idx = ys.argmax()  # this index is relative to the local x_tries values matrix
    x_max = x_tries[idx]
    
    if kind == 'DiscreteBO':

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            if x_max.tolist() in np.array(old_x).tolist():
                
                def f(x):       
                    cand_gp = GaussianProcessRegressor(
                        kernel=CustomRBFKernel(length_scale=x[1], sigma_2=sigma_2),
                        alpha=1e-6,
                        normalize_y=True,
                        n_restarts_optimizer=5,
                        random_state=random_state,
                    )

                    cand_gp.fit(old_x, old_y)

                    x[0] = max(1e-30, x[0])
                    ys = ac(x_tries, gp=cand_gp, y_max=y_max, iter_num=iter_num, at_least_one_feasible_found=at_least_one_feasible_found, beta=x[0])
                    for idx_ in range(len(ys)):
                        if x_tries[idx_].tolist() in np.array(old_x).tolist():
                            ys[idx_] = -np.inf

                    idx = ys.argmax()  # this index is relative to the local x_tries values matrix
                    candidate_x_ = x_tries[idx]

                    return abs(x[0]-beta) + np.linalg.norm(candidate_x_ - x_max) + 1e30*(candidate_x_.tolist() in np.array(old_x).tolist())

                x0 = [random.uniform(max(beta-beta_h,1e-30), beta+beta_h), random.uniform(1e-30, l_h)]
                
                constraints = [{'type': 'ineq', 'fun': lambda x: abs(x[0] - beta)},
                            {'type': 'ineq', 'fun': lambda x: beta_h - abs(x[0] - beta)},
                            {'type': 'ineq', 'fun': lambda x: x[1] - 1e-30},
                            {'type': 'ineq', 'fun': lambda x: l_h - x[1]}]

                result = result = minimize(f, x0, constraints=constraints)       
                optimal_values = result.x
                beta = optimal_values[0]
                l = optimal_values[1]
                print("Updated beta = %s, l = %s" %(beta, l))

                gp = GaussianProcessRegressor(
                    kernel=RBF(length_scale=l),
                    alpha=1e-6,
                    normalize_y=True,
                    n_restarts_optimizer=5,
                    random_state=random_state,
                )

                gp.fit(old_x, old_y)

                ys = ac(x_tries, gp=gp, y_max=y_max, iter_num=iter_num, at_least_one_feasible_found=at_least_one_feasible_found, beta=beta)
                for idx_ in range(len(ys)):
                        if x_tries[idx_].tolist() in np.array(old_x).tolist():
                            ys[idx_] = -np.inf

                idx = ys.argmax()
                x_max = x_tries[idx]

                if x_max.tolist() in np.array(old_x).tolist():
                    import pdb; pdb.set_trace()

    if dataset is not None:
        max_acq = ys[idx]
        # idx becomes the true dataset index of the selected point, rather than being relative to x_tries
        idx = dataset.index[idx]
        if debug: print("End of acq_max(): maximizer of utility is x = data[{}] = {}, with ac(x) = {}".format(idx, x_max, max_acq))
        return x_max, idx, max_acq

    max_acq = ys[idx]
    if debug: print("Best point on initial grid is ac({}) = {}".format(x_max, max_acq))

    # Explore the parameter space more throughly
    x_seeds = random_state.uniform(bounds[:, 0], bounds[:, 1],
                                   size=(n_iter, bounds.shape[0]))

    if debug: print("Calling minimize() with", len(x_seeds), "different starting seeds")

    for x_try in x_seeds:
        # Find the minimum of minus the acquisition function
        res = minimize(lambda x: -ac(x.reshape(1, -1), gp=gp, y_max=y_max),
                       x_try.reshape(1, -1),
                       bounds=bounds,
                       method="L-BFGS-B")

        # See if success
        if not res.success:
            continue
        # Store it if better than previous minimum(maximum).
        if max_acq is None or -np.squeeze(res.fun) >= max_acq:
            x_max = res.x
            max_acq = -np.squeeze(res.fun)

    if debug: print("End of acq_max(): maximizer of utility is ac({}) = {}".format(x_max, max_acq))

    # Clip output to make sure it lies within the bounds. Due to floating
    # point technicalities this is not always the case.
    return np.clip(x_max, bounds[:, 0], bounds[:, 1]), None, max_acq


class UtilityFunction(object):
    """
    An object to compute the acquisition functions.

    See the maximize() function in bayesian_optimization.py for a description of the constructor arguments.
    """
    def __init__(self, kind, kappa, xi, kappa_decay=1, kappa_decay_delay=0, acq_info={}, ml_on_bounds=False, ml_on_target=False, debug=False):

        self._debug = debug
        self.kappa = kappa
        self._kappa_decay = kappa_decay
        self._kappa_decay_delay = kappa_decay_delay
        self.xi = xi
        self.kind = kind
        self._iters_counter = 0

        self._ml_on_bounds = ml_on_bounds
        self._ml_on_target = ml_on_target

        self.initialize_acq_info(acq_info, kind)

        if self._debug: print("UtilityFunction initialization completed")


    def set_acq_info_field(self, acq_info, key_from, key_to=None):
        """
        Set a field from acq_info into the current object if it exists, otherwise raise an error

        Parameters
        ----------
        acq_info: dict
            Dictionary with relevant objects and information for several acquisition functions

        key_from: str
            Name of the field to fetch from acq_info

        key_to: str or None, optional (default=None)
            Name of the member of the current object to set the field to. If None, key_from will be
            used as the member name
        """
        key_to = key_from if key_to is None else key_to
        if self._debug: print("Setting '{}' field of acq_info to '{}' member".format(key_from, key_to))
        if key_from in acq_info:
            self.__setattr__(key_to, acq_info[key_from])
        else:
            raise KeyError("'{}' field is required in acq_info if using '{}' acquisition".format(key_from, self.kind))


    def initialize_acq_info(self, acq_info, kind):
        """Initialize some parameters of the `kind` acquisition function from the `acq_info` dict, if any"""
        if self._debug: print("Initializing UtilityFunction of kind '{}' with acq_info = {}".format(kind, acq_info))

        # For Machine Learning-based acquisitions
        if self._ml_on_bounds:
            self.set_acq_info_field(acq_info, 'ml_target')
            self.set_acq_info_field(acq_info, 'ml_bounds')
            self.set_acq_info_field(acq_info, 'ml_bounds_type')
            self.set_acq_info_field(acq_info, 'ml_bounds_alpha')

        # For Expected Improvement with Constraints-based acquisitions
        if 'eic' in kind:
            self.set_acq_info_field(acq_info, 'eic_bounds')
            # Check for other needed fields, and provide default values if not present
            if 'eic_P_func' not in acq_info:
                if self._debug: print("Using default eic_P_func, P(x) == 1")
                def P_func_default(x):
                    return 1.0
                acq_info['eic_P_func'] = P_func_default
            if 'eic_Q_func' not in acq_info:
                if self._debug: print("Using default eic_Q_func, Q(x) == 0")
                def Q_func_default(x):
                    return 0.0
                acq_info['eic_Q_func'] = Q_func_default

            self.set_acq_info_field(acq_info, 'eic_P_func')
            self.set_acq_info_field(acq_info, 'eic_Q_func')

        # For EIC-ML hybrid acquisitions
        if 'eic' in kind and self._ml_on_bounds:
            self.set_acq_info_field(acq_info, 'eic_ml_var')
            if self.eic_ml_var not in ('B', 'C', 'D'):
                raise ValueError("'eic_ml_var' field must be B/C/D, not {}".format(self.eic_ml_var))

            # Check for other needed field, and provide default value if not present
            if 'eic_ml_exp_B' not in acq_info:
                if self._debug: print("Using default eic_ml_exp_B = 2")
                acq_info['eic_ml_exp_B'] = 2.0
            self.set_acq_info_field(acq_info, 'eic_ml_exp_B')

        if self._ml_on_target:

            self.set_acq_info_field(acq_info, 'ml_target_type')
            self.set_acq_info_field(acq_info, 'ml_target_alpha')

            if self.ml_target_type == 'sum':

                self.set_acq_info_field(acq_info, 'ml_target_gamma_iter0')
                self.set_acq_info_field(acq_info, 'ml_target_gamma_iterN')
                self.set_acq_info_field(acq_info, 'ml_target_gamma_max')

            if self.ml_target_type in ['indicator', 'probability']:
                self.set_acq_info_field(acq_info, 'ml_target_coeff')

    def update_params(self):
        self._iters_counter += 1

        if self._kappa_decay < 1 and self._iters_counter > self._kappa_decay_delay:
            self.kappa *= self._kappa_decay


    def set_ml_model(self, model):
        self.ml_model = model

    
    def set_objective_ml_model(self, model):
        self.objective_ml_model = model


    def utility(self, x, gp, y_max, iter_num, at_least_one_feasible_found, beta=1.0):

        if self.kind == 'no_BO' or not at_least_one_feasible_found:
            res = self._no_BO(x)
        elif self.kind == 'ucb':
            res = self._ucb(x, gp, self.kappa)
        elif self.kind == 'poi':
            res = self._poi(x, gp, y_max, self.xi)
        elif self.kind == 'ei':
            res = self._ei(x, gp, y_max, self.xi)

        elif self.kind == 'DiscreteBO':
            res = self._ucb(x, gp, sqrt(beta))

        elif self.kind == 'eic':

            res = self._eic(x, gp, y_max, self.xi, self.eic_bounds, self.eic_P_func, self.eic_Q_func)

            if self._ml_on_bounds:
                if self.eic_ml_var in ('B', 'D'):
                    lb, ub = self.ml_bounds
                    y_hat = self.ml_model[0].predict(x)
                    coeff = np.exp(-self.eic_ml_exp_B * y_hat)
                    norm_const = self.eic_ml_exp_B * (lb-ub) - 0.5 * self.eic_ml_exp_B ** 2 * (lb ** 2 - ub ** 2)
                    res *= coeff * norm_const
                if self.eic_ml_var in ('C', 'D'):
                    res *= self._consider_ml_on_bounds(x, self.ml_model, self.ml_bounds, self.ml_bounds_type)

        else:
            raise NotImplementedError("The utility function {} has not been implemented.".format(self.kind))
        

        if self._ml_on_bounds and self.kind != 'eic':
            res *= self._consider_ml_on_bounds(x, self.ml_model, self.ml_bounds, self.ml_bounds_type)
        
        if self._ml_on_target:

            parameters = {}

            if self.ml_target_type == 'sum':

                parameters = {'ml_target_gamma_iter0': self.ml_target_gamma_iter0,
                              'ml_target_gamma_iterN': self.ml_target_gamma_iterN,
                              'ml_target_gamma_max': self.ml_target_gamma_max}

            elif self.ml_target_type == 'indicator':

                parameters = {'ml_target_coeff': self.ml_target_coeff}

            res = self._consider_ml_on_target(x, self.objective_ml_model, self.ml_target_type, res, iter_num, y_max, parameters)

        return res
    
    
    @staticmethod
    def _no_BO(x):

        return np.ones(x.shape[0])


    @staticmethod
    def _ucb(x, gp, kappa):
        """Compute Upper Confidence Bound"""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mean, std = gp.predict(x, return_std=True)

        return mean + kappa * std


    @staticmethod
    def _poi(x, gp, y_max, xi):
        """Compute Probability Of Improvement"""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mean, std = gp.predict(x, return_std=True)

        z = (mean - y_max - xi)/std
        return norm.cdf(z)


    @staticmethod
    def _ei(x, gp, y_max, xi):
        """Compute Expected Improvement"""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mean, std = gp.predict(x, return_std=True)
  
        a = (mean - y_max - xi)
        z = a / std
        return a * norm.cdf(z) + std * norm.pdf(z)


    @staticmethod
    def _consider_ml_on_bounds(x, ml_model, ml_bounds, ml_bounds_type):

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            if ml_bounds_type == 'indicator':
                y_hat = ml_model[0].predict(x)
                lb, ub = ml_bounds
                return np.array([lb <= y and y <= ub for y in y_hat])
            
            elif ml_bounds_type == 'probability':
                indicator = ml_model[1].decision_function(x)
                return 1 / (1 + np.exp(-indicator))


    @staticmethod
    def _consider_ml_on_target(x, objective_ml_model, ml_target_type, res, iter_num, y_max, parameters):

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            f_tilde = objective_ml_model[0].predict(x)

            ml_target_type = ml_target_type
            
            if ml_target_type == 'product':
                res *= min_max_normalize(f_tilde)

            elif ml_target_type == 'sum':

                if iter_num > parameters['ml_target_gamma_iter0']:
                    gamma = parameters['ml_target_gamma_max'] * (1 - exp(-5*(iter_num - parameters['ml_target_gamma_iter0'])/(parameters['ml_target_gamma_iterN'] - parameters['ml_target_gamma_iter0'])))
                    res = (1 - gamma) * np.array(min_max_normalize(res)) + gamma * np.array(min_max_normalize(f_tilde))    

            elif ml_target_type == 'probability':
                indicator = objective_ml_model[1].decision_function(x)
                res *= 1 / (1 + np.exp(-indicator))
            
            elif ml_target_type == 'indicator':
                lb_coeff, ub_coeff = parameters['ml_target_coeff']

                for idx in range(len(f_tilde)):

                    if lb_coeff != None:
                        if f_tilde[idx] < lb_coeff*y_max:
                            res[idx] = -float_info.max

                    if ub_coeff != None:
                        if f_tilde[idx] > ub_coeff*y_max:
                            res[idx] = -float_info.max
            
        return res
    

    @staticmethod
    def _eic(x, gp, y_max, xi, bounds, P, Q):
        """
        Compute Expected Improvement with Constraints.

        Given the target function f(x) = P(x) g(x) + Q(x), with P, Q fixed and P >= 0,
        this function multiplies the regular Expected Improvement with the probability
        that Gmin <= g(x) <= Gmax, with Gmin = bounds[0] and Gmax = bounds[1].
        """

        # Compute regular Expected Improvement
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mean, std = gp.predict(x, return_std=True)
        # std = max(std, 1e-10)
        a = (mean - y_max - xi)
        z = a / std
        ei = a * norm.cdf(z) + std * norm.pdf(z)

        # Compute probability of x respecting the constraint
        Gmin, Gmax = bounds
        mean_Gmax = P(x) * Gmax + Q(x)
        mean_Gmin = P(x) * Gmin + Q(x)
        prob_ub = norm.cdf( (mean_Gmax - mean) / std )
        prob_lb = norm.cdf( (mean_Gmin - mean) / std )

        return ei * (prob_ub - prob_lb)



class StoppingCriterion(object):
    """
    An object that represents the algorithm stopping criterion.

    Parameters
    ----------
    conjunction: str 'and' or 'or', optional (default='or')
        Whether to apply AND or OR between the different termination criteria

    hard_stop: bool, optional (default=True)
        Whether to actually stop the optimization procedure once terminated

    ml_bounds_coeff: tuple, optional (default=None)
        Coefficients (lb_coef, ub_coef) to apply to ml_bounds (see _violate_ml_bounds() method).
        Needs an ML-related acquisition function, which has the ml_bounds member

    debug: bool, optional (default=False)
        Whether or not to print detailed debugging information
    """
    def __init__(self, conjunction='or', hard_stop=True, ml_bounds_coeff=None, debug=False):
        if conjunction == 'and':
            self._AND_join = True
        elif conjunction == 'or':
            self._AND_join = False
        else:
            raise ValueError("'conjunction' option for Stopping Criterion must be 'and' or 'or'")
        self._debug = debug
        self._hard_stop = hard_stop
        self._ml_bounds_coeff = ml_bounds_coeff
        if self._debug: print("StoppingCriterion initialization completed")


    def hard_stop(self):
        return self._hard_stop


    def terminate(self, x_point, target, iteration, utility, ml_target_val=None):
        bool_list = []
        # Target value within given bounds
        if self._ml_bounds_coeff is not None:
            try:
                ml_bounds = utility.ml_bounds
            except AttributeError:
                raise ValueError("terminate(): 'ml_bounds_coeff' was initialized, but utility.ml_bounds was not")
            if ml_target_val is None:
                raise ValueError("terminate(): 'ml_bounds_coeff' was initialized, but ml_target_val was not given")
            vi = self._violate_ml_bounds(ml_target_val, ml_bounds)
            bool_list.append(vi)
            if self._debug: print("_violate_ml_bounds() termination criterion:", vi)
        # Do not terminate if there was no stopping criterion required,
        # otherwise reduce list of bools according to the 'and'/'or' conjunction
        if not bool_list:
            if self._debug: print("No termination criteria have been used")
            term = False
        elif self._AND_join:
            term = bool(np.product(bool_list))
        else:
            term = bool(np.sum(bool_list))
        if self._debug: print("Result of termination criteria:", term)
        return term


    def _violate_ml_bounds(self, val, ml_bounds):
        """
        Checks if val is not included in the [lb_coef * lb, ub_coef * ub] interval

        Parameters
        ----------
        val: float
            Target value of the ML model, which will be checked against the interval
        ml_bounds: tuple
            Contains lb and ub. If either is None, that extremity of the interval will not be checked

        Returns
        -------
        ret: bool
            Whether the bounds are violated or not
        """
        lb, ub = ml_bounds
        lb_coef, ub_coef = self._ml_bounds_coeff
        if lb_coef is not None:
            if self._debug: print("Checking lower bound {}*{}={} vs {}".format(lb_coef, lb, lb_coef*lb, val))
            if val < lb_coef*lb:
                return True
        if ub_coef is not None:
            if self._debug: print("Checking upper bound {}*{}={} vs {}".format(ub_coef, ub, ub_coef*ub, val))
            if val > ub_coef*ub:
                return True
        return False


def load_logs(optimizer, logs):
    """Load previous ...

    """
    import json

    if isinstance(logs, str):
        logs = [logs]

    for log in logs:
        with open(log, "r") as j:
            while True:
                try:
                    iteration = next(j)
                except StopIteration:
                    break

                iteration = json.loads(iteration)
                try:
                    optimizer.register(
                        params=iteration["params"],
                        target=iteration["target"],
                    )
                except KeyError:
                    pass

    return optimizer


def ensure_rng(random_state=None):
    """
    Creates a random number generator based on an optional seed. This can be
    an integer or another random state for a seeded rng, or None for an
    unseeded rng.
    """
    if random_state is None:
        random_state = np.random.RandomState()
    elif isinstance(random_state, int):
        random_state = np.random.RandomState(random_state)
    else:
        assert isinstance(random_state, np.random.RandomState)
    return random_state


def evaluate_max(original_dataset, results_dataset, target_name='', bounds={}, print_res=False):

    indices = pd.read_csv(results_dataset)['index'].tolist()
    df = pd.read_csv(original_dataset).loc[indices].sort_values(by=target_name, ascending=False)

    for index, row in df.iterrows():

        constraints_respected = True
        
        for key, value in bounds.items():
            
            lb, ub = value

            if row[key] < lb or row[key] > ub:
                constraints_respected = False
                
        if constraints_respected:
            if print_res:
                print("\nMax feasible:")
                print(row)
            return row[target_name]
        
    return -np.inf


def print_final_results(output_path, real_max, init_points):

    repetitions_dir = os.listdir(output_path)
    repetitions = len(repetitions_dir)

    errors_ = []
    feasible_ = []
   
    for dir in repetitions_dir:

        errors = []
        
        with open(output_path + "/" + dir + "/results.csv") as csvfile:
            reader = csv.reader(csvfile)
            header = next(reader)

            target_index = header.index('target')
            feasibility_index = header.index('feasible')
            
            act_err = np.inf

            idx = 0

            for row in reader:

                if row[feasibility_index] == "True":
                    if (float(row[target_index]) - real_max)/real_max < act_err:
                        act_err = (float(row[target_index]) - real_max)/real_max

                        if act_err < 0:
                            import pdb; pdb.set_trace()
                    
                errors.append(act_err)

                if len(feasible_) < idx:
                    import pdb; pdb.set_trace()
                elif len(feasible_) == idx:
                    feasible_.append(0)

                if act_err < np.inf:
                    feasible_[idx] += 1/repetitions

                idx += 1

        errors_.append(errors)


    values_ = []
    found_first_not_inf = False

    for idx in range(init_points-1, len(feasible_)):

        value = 0
        non_inf = 0

        for rep in range(repetitions):
            
            if errors_[rep][idx] != np.inf:
                value += errors_[rep][idx]
                non_inf += 1

        if non_inf == 0:
            values_.append(np.inf)
        else:
            values_.append(value/non_inf*100)
            if not found_first_not_inf:
                found_first_not_inf = True
                first_not_inf = value/non_inf*100

    plt.plot([ii for ii in range(init_points-1, len(feasible_))], values_, alpha=0.5, color='green', linewidth=2) 
    plt.scatter([ii for ii in range(init_points-1, len(feasible_))], values_, alpha=feasible_[init_points-1:len(feasible_)], color='blue', s=50)

    if found_first_not_inf:
        plt.plot([init_points-0.5, init_points-0.5], [0, first_not_inf], '--', color='red')

    plt.xlabel('step number')
    plt.ylabel('Error [%]')
    plt.savefig(output_path + "/results.png")
    plt.close()
    
    return


class Colours:
    """Print in nice colours."""

    BLUE = '\033[94m'
    BOLD = '\033[1m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    END = '\033[0m'
    GREEN = '\033[92m'
    PURPLE = '\033[95m'
    RED = '\033[91m'
    UNDERLINE = '\033[4m'
    YELLOW = '\033[93m'

    @classmethod
    def _wrap_colour(cls, s, colour):
        return colour + s + cls.END

    @classmethod
    def black(cls, s):
        """Wrap text in black."""
        return cls._wrap_colour(s, cls.END)

    @classmethod
    def blue(cls, s):
        """Wrap text in blue."""
        return cls._wrap_colour(s, cls.BLUE)

    @classmethod
    def bold(cls, s):
        """Wrap text in bold."""
        return cls._wrap_colour(s, cls.BOLD)

    @classmethod
    def cyan(cls, s):
        """Wrap text in cyan."""
        return cls._wrap_colour(s, cls.CYAN)

    @classmethod
    def darkcyan(cls, s):
        """Wrap text in darkcyan."""
        return cls._wrap_colour(s, cls.DARKCYAN)

    @classmethod
    def green(cls, s):
        """Wrap text in green."""
        return cls._wrap_colour(s, cls.GREEN)

    @classmethod
    def purple(cls, s):
        """Wrap text in purple."""
        return cls._wrap_colour(s, cls.PURPLE)

    @classmethod
    def red(cls, s):
        """Wrap text in red."""
        return cls._wrap_colour(s, cls.RED)

    @classmethod
    def underline(cls, s):
        """Wrap text in underline."""
        return cls._wrap_colour(s, cls.UNDERLINE)

    @classmethod
    def yellow(cls, s):
        """Wrap text in yellow."""
        return cls._wrap_colour(s, cls.YELLOW)
