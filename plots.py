from maliboo.bayesian_optimization import BayesianOptimization as BO
import pandas as pd
import os
import time
import numpy as np
from test import perform_test
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from pathlib import Path



seed = 42
debug = False


def plot_regret(output_path, save_fig = True):

    path = Path(output_path)
    test_name = path.parent.name

    if not path.exists():
        raise FileNotFoundError({f'Cannot plot the following test regret since it has not beed perfomed yet: {test_name}'})
    
    test_data = pd.read_csv(output_path)
    n_iterations = len(test_data.index)

    plt.plot(range(n_iterations), test_data["regret"], label = test_name)
    plt.xlabel('iteration')
    plt.ylabel('log(regret)')
    plt.legend()
    plt.yscale('log')
    plt.title(f'{test_name}: regret')
    
    filename = path.parent / (str(path.stem) + '.png')

    if save_fig:
        plt.savefig(filename)
        plt.clf()

def compare_regret(test_names: list[str], test_path: str = 'outputs/', output_name: str = None):
    """
    This function takes as input a list of test and compares them, in term of regret. 
    It may be useful to comapre the result of the same benchmark function, optimized with 
    different penalizations, or different parameters.
    """
    if output_name is None:
        output_name = "comparison.png"

    test_path =  Path(test_path)
    plt.clf()

    for test in test_names:
        plot_regret(test_path / test / 'results.csv', save_fig=False)

    comaprison_path: Path = Path(test_path / 'comparison')
    comaprison_path.mkdir(exist_ok=True)

    plt.title('Regret comparison')
    plt.savefig(comaprison_path / output_name)
    plt.clf()

def plot_average(test_name: str, true_opt_value: float, test_path: str = 'outputs/',output_name: str = None, optimization_type: str = 'max', init_points: int = 0):
    
    if output_name is None:
        output_name = "average.png"
    
    test_path = test_path + test_name
    i = 0
  
    results = []

    while Path(test_path + f"_{i}").exists():
        result_path = Path(test_path + f"_{i}") / "results.csv"
        result = pd.read_csv(result_path)

        results.append(result)
        i += 1

    number_of_test_executions = i
    numeber_of_iterations = len(results[0].index)

    average_maximum_per_iteration = []
 
    alpha = 0.05

    for i in range( numeber_of_iterations ):
        mean = 0
        feasibile_maximums = 0
        for experiment in results:

            observed_values = experiment['target'][:i+1].values
            feasibile_values = experiment['feasible'][:i+1].values
            
            if np.any(feasibile_values):
                feasibile_maximums += 1
                mean += np.max(observed_values[feasibile_values]) #TODO: Maybe we need to avoid computing the mean for the first random points?

        if feasibile_maximums>0:  
            mean = mean / feasibile_maximums
        else:
            mean = np.nan

        average_maximum_per_iteration.append(mean)
    
    
    upper_quantiles = []
    lower_quantiles = []


    for i in range( numeber_of_iterations ):
        
        max_found_at_iteration_i = []

        for experiment in results:
            observed_values = experiment['target'][:i+1].values
            feasibile_values = experiment['feasible'][:i+1].values

            if np.any(feasibile_values):
                experiment_maximum_at_current_iteration = np.max ( observed_values[feasibile_values] )
            else:
                experiment_maximum_at_current_iteration = np.nan

            max_found_at_iteration_i.append(experiment_maximum_at_current_iteration)

        upper_quantiles.append(  np.percentile(max_found_at_iteration_i, (1 - alpha/2)*100 ) )  
        lower_quantiles.append(  np.percentile(max_found_at_iteration_i, (alpha/2)*100     ) )  
        

    max_y_value = true_opt_value
    min_y_value = np.nanmin(lower_quantiles)
    

    if optimization_type == 'min':
        average_maximum_per_iteration = [-x for x in average_maximum_per_iteration]
        upper_quantiles = [-x for x in upper_quantiles]
        lower_quantiles = [-x for x in lower_quantiles]
        max_y_value = np.nanmax(lower_quantiles)
        min_y_value = true_opt_value
    

    plt.plot(range(numeber_of_iterations), average_maximum_per_iteration, label = test_name, marker='o')
    
    if output_name is not None:
        plt.plot(range(numeber_of_iterations), [true_opt_value]*numeber_of_iterations, linestyle='--', label = 'True optimum value')
    
    plt.xlabel('Iteration')
    plt.ylabel('Average optimum value')
    plt.legend() #loc='upper left', bbox_to_anchor=(1, 1)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.fill_between(range(numeber_of_iterations), lower_quantiles, upper_quantiles, color='lightblue', alpha=0.3, label="90% CI")

    #import pdb; pdb.set_trace()
    plt.plot(2*[init_points], [min_y_value, max_y_value], linestyle = '-.', color = 'red')

    if output_name is not None:
        figname = output_name
        plt.savefig(Path('outputs/comparison/') / figname )

goldstain_files = []
branin_files = []
for i in range(10):
  goldstain_files.append('test_goldstain_ml_'+str(i))
  branin_files.append('test_branin_ml_'+str(i))  
#compare_regret(goldstain_files)
#compare_regret(branin_files)
#plot_average(test_name= 'test_goldstain_ml', true_opt_value=38.11, optimization_type='min', output_name='avg_opt_goldstain_ml', init_points=5)
#plot_average(test_name= 'test_branin_ml', true_opt_value=-0.8143, optimization_type='min', output_name='avg_opt_branin_ml', init_points=5)
plot_average(test_name= 'test_branin_ml_indicator', true_opt_value=-0.8143, optimization_type='min', output_name='avg_opt_branin_ml', init_points=5)