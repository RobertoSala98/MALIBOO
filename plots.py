from maliboo.bayesian_optimization import BayesianOptimization as BO
import pandas as pd
import os
import time
import numpy as np
from test import perform_test
import matplotlib.pyplot as plt
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

goldstain_files = []
for i in range(10):
    goldstain_files.append('test_goldstain_ml_'+str(i))
    
compare_regret(goldstain_files)