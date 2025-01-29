from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def compute_regret(test_name: str, true_max: float) -> list[float]:
   
   test_path = Path('outputs') / test_name 

   if not test_path.exists():
      raise FileNotFoundError({f'Cannot plot the following test regret since it has not beed perfomed yet: {test_name}'})

   experiment_data = pd.read_csv(test_path / 'results.csv')
   regret = []

   for idx, _ in enumerate(experiment_data['target']):

      target = experiment_data['target'][:idx+1].values
      feasibility = experiment_data['feasible'][:idx+1].values
      feasible_max_value_so_far = np.max(target[feasibility])
      
      regret.append( np.abs(true_max - feasible_max_value_so_far) )
   return regret

def plot_regret(test_name: str, true_max: float, save_fig = True):

   test_path = Path('outputs') / test_name

   if not test_path.exists():
      raise FileNotFoundError({f'Cannot plot the following test regret since it has not beed perfomed yet: {test_name}'})
   
   test_data = pd.read_csv(test_path / "results.csv")
   
   n_iterations = len(test_data.index)
   regret = compute_regret(test_name, true_max)

   plt.plot(range(n_iterations), regret, label = test_name)
   plt.xlabel('iteration')
   plt.ylabel('log(regret)')
   plt.legend()
   plt.yscale('log')
   plt.title(f'{test_name}: regret')
   
   
   if save_fig:
      plt.savefig(test_path / 'regret.png')
      plt.clf()

def compare_regret(test_names: list[str],  true_maximum_values : list[float], test_path: str = 'outputs/', output_name: str = None):
    
    """
    This function takes as input a list of test and compares them, in term of regret. 
    It may be useful to comapre the result of the same benchmark function, optimized with 
    different penalizations, or different parameters.
    """
    
    if output_name is None:
        output_name = "comparision.png"

    test_path =  Path(test_path)
    plt.clf()

    for idx, test in enumerate(test_names):
        plot_regret( test , save_fig=False, true_max=true_maximum_values[idx])

    comaprison_path: Path = Path(test_path / 'comparison')
    comaprison_path.mkdir(exist_ok=True)

    plt.title('Regret comparison')
    plt.savefig(comaprison_path / output_name)
    plt.clf()


compare_regret(test_names=['test_goldstain'], true_maximum_values=[74.4970], output_name='goldstain_comparison')