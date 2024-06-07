import os
import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt

root_results_path = "outputs"
output_path = "./test.png"
dataset = 'ligen'
threshold = 2.1

configs = [
    {"ml_bounds": "probability", "ml_target": "product", "consider_only_true_max": False, "epsilon_greedy": False, "adaptive_method_kernel": "Matern", "af": "ei"},
    {"ml_bounds": "probability", "ml_target": "product", "consider_only_true_max": True, "epsilon_greedy": False, "adaptive_method_kernel": "Matern", "af": "ei"},
    {"ml_bounds": "probability", "ml_target": "sum", "consider_only_true_max": True, "epsilon_greedy": False, "adaptive_method_kernel": "Matern", "af": "ei"}
]

thresholds = {
    "oscarp": [65, 150, 300, 450, 600],
    "query26": [185000, 205000, 225000, 245000],
    "stereomatch": [6000, 8000, 10000, 12000, 17000, 20000, 40000],
    "ligen": [1.9, 2.0, 2.1, 2.2, 2.45, 2.75]
}

maximum = {
    "oscarp": [-0.174361111, -0.174361111, -0.174361111, -0.174361111, -0.174361111],
    "query26": [-4356300.0, -4079658.0, -4022118.0, -683682.0],
    "stereomatch": [-62810.0, -50776.0, -45525.0, -40196.0, -40196.0, -37412.0, -36791.0],
    "ligen": [-881.258448995863, -567.312555400384, -567.312555400384, -464.349441178703, -424.355184049556, -340.099251789615]
}

initial_points = {
    "oscarp": 9,
    "stereomatch": 3,
    "query26": 3,
    "ligen": 11
}

bounds_constraint = {
    "oscarp": {"total_time": [0, 300]},
    "stereomatch": {"time": [0, 17000]},
    "query26": {"time": [0, 205000]},
    "ligen": {"RMSD_0.75": [0, 2.1]}
}

datasets = {
    "oscarp": "resources/OSCAR-P_combinatorial.csv",
    "stereomatch": "resources/stereomatch.csv",
    "query26": "resources/query26_vm_ram.csv",
    "ligen": "resources/ligen_synth_table.csv"
}

def print_results_multiple_thresholds(root_results_path, output_path, configs, dataset='', threshold=-np.inf, labels=[]):

    init_points = initial_points[dataset]
    real_max = maximum[dataset][thresholds[dataset].index(threshold)]
    bounds = bounds_constraint[dataset]

    colors = ['green', 'deepskyblue', 'black', 'sienna', 'orange', 'blue', 'darkviolet', 'magenta', 'darkgray', 'gold', 'darkred', 'cadetblue']

    for config in configs:

        results_path = "./%s/%s/bounds_%s/target_%s/only_true_max_%s/eps_greedy_%s/adaptive_%s/af_%s/threshold_%s" %(root_results_path, dataset, config["ml_bounds"], config["ml_target"], config["consider_only_true_max"], config["epsilon_greedy"], config["adaptive_method_kernel"], config["af"], threshold)

        repetitions_dir = os.listdir(results_path)
        if "results.png" in repetitions_dir:
            repetitions_dir.remove("results.png")
        repetitions = len(repetitions_dir)

        errors_ = []
        feasible_ = []

        if "is_DiscreteBO" in config and config["is_DiscreteBO"]:
            df = pd.read_csv(dataset)

        for dir in repetitions_dir:

            errors = []
            
            with open(results_path + "/" + dir + "/results.csv") as csvfile:
                reader = csv.reader(csvfile)
                header = next(reader)

                target_index = header.index('target')
                feasibility_index = header.index('feasible')
                
                act_err = np.inf

                idx = 0

                for row in reader:

                    constraints_respected = "True"

                    if "is_DiscreteBO" in config and config["is_DiscreteBO"]:

                        for key, value in bounds.items():

                            lb, ub = value
                            ub = threshold

                            if df[key].values[int(row[0])] < lb or df[key].values[int(row[0])] > ub:
                                constraints_respected = "False"

                    else:
                        constraints_respected = row[feasibility_index]

                    if constraints_respected == "True":

                        if (float(row[target_index]) - real_max)/real_max < act_err:
                            act_err = (float(row[target_index]) - real_max)/real_max

                            if act_err < 0:
                                print("The error computed is negative: " + str(100*act_err) + " %")
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

        plt.plot([ii for ii in range(init_points-1, len(feasible_))], values_, color=colors[configs.index(config)], linewidth=2, label="config: %s" %labels[configs.index(config)]) 
        plt.scatter([ii for ii in range(init_points-1, len(feasible_))], values_, alpha=feasible_[init_points-1:len(feasible_)], color=colors[configs.index(config)], s=20)

        if found_first_not_inf:
            plt.plot([init_points-0.5, init_points-0.5], [0, first_not_inf], '--', color='red')
        
    plt.xlabel('step number')
    plt.ylabel('Error [%]')
    plt.legend(loc="best")
    plt.savefig(output_path)
    plt.close()

    return

print_results_multiple_thresholds(root_results_path, output_path, configs, dataset, threshold, ["config1", "config2", "config3"])