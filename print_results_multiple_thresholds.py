from maliboo.util import print_results_multiple_thresholds

datasets_name = ["oscarp", "stereomatch", "query26", "ligen"]

configurations_1 = [
    {"ml_bounds": "probability", "ml_target": "product", "consider_only_true_max": False, "epsilon_greedy": False, "adaptive_method_kernel": "Matern", "af": "ei"},
    {"ml_bounds": "probability", "ml_target": "product", "consider_only_true_max": True, "epsilon_greedy": False, "adaptive_method_kernel": "Matern", "af": "ei"},
    {"ml_bounds": "probability", "ml_target": "sum", "consider_only_true_max": True, "epsilon_greedy": False, "adaptive_method_kernel": "Matern", "af": "ei"},
    {"ml_bounds": "indicator", "ml_target": "product", "consider_only_true_max": True, "epsilon_greedy": True, "adaptive_method_kernel": "Matern", "af": "ei"},
    {"ml_bounds": "indicator", "ml_target": "sum", "consider_only_true_max": False, "epsilon_greedy": True, "adaptive_method_kernel": "None", "af": "ei"},
    {"ml_bounds": "probability", "ml_target": "sum", "consider_only_true_max": False, "epsilon_greedy": False, "adaptive_method_kernel": "Matern", "af": "ei"},
    {"ml_bounds": "probability", "ml_target": "probability", "consider_only_true_max": False, "epsilon_greedy": False, "adaptive_method_kernel": "Matern", "af": "ei"},
    {"ml_bounds": "indicator", "ml_target": "sum", "consider_only_true_max": False, "epsilon_greedy": True, "adaptive_method_kernel": "Matern", "af": "ei"},
    {"ml_bounds": "probability", "ml_target": "probability", "consider_only_true_max": True, "epsilon_greedy": True, "adaptive_method_kernel": "None", "af": "ei"},
    {"ml_bounds": "probability", "ml_target": "product", "consider_only_true_max": True, "epsilon_greedy": False, "adaptive_method_kernel": "None", "af": "ei"},
    {"ml_bounds": "indicator", "ml_target": "sum", "consider_only_true_max": False, "epsilon_greedy": False, "adaptive_method_kernel": "None", "af": "ei"},      
    {"ml_bounds": "indicator", "ml_target": "indicator", "consider_only_true_max": False, "epsilon_greedy": True, "adaptive_method_kernel": "None", "af": "ei"},      
    {"ml_bounds": "indicator", "ml_target": "sum", "consider_only_true_max": False, "epsilon_greedy": False, "adaptive_method_kernel": "RBF", "af": "ucb"},         
    {"ml_bounds": "indicator", "ml_target": "None", "consider_only_true_max": False, "epsilon_greedy": False, "adaptive_method_kernel": "None", "af": "ei"},                        # maliboo
    {"ml_bounds": "None", "ml_target": "None", "consider_only_true_max": False, "epsilon_greedy": False, "adaptive_method_kernel": "RBF", "af": "ucb", "is_DiscreteBO": True}       # discreteBO
]

configurations_2 = [
    {"ml_bounds": "probability", "ml_target": "product", "consider_only_true_max": False, "epsilon_greedy": False, "adaptive_method_kernel": "Matern", "af": "eic"},
    {"ml_bounds": "probability", "ml_target": "product", "consider_only_true_max": True, "epsilon_greedy": False, "adaptive_method_kernel": "Matern", "af": "eic"},
    {"ml_bounds": "probability", "ml_target": "sum", "consider_only_true_max": True, "epsilon_greedy": False, "adaptive_method_kernel": "Matern", "af": "eic"},
    {"ml_bounds": "indicator", "ml_target": "product", "consider_only_true_max": True, "epsilon_greedy": True, "adaptive_method_kernel": "Matern", "af": "eic"},
    {"ml_bounds": "indicator", "ml_target": "sum", "consider_only_true_max": False, "epsilon_greedy": True, "adaptive_method_kernel": "None", "af": "eic"},
    {"ml_bounds": "probability", "ml_target": "sum", "consider_only_true_max": False, "epsilon_greedy": False, "adaptive_method_kernel": "Matern", "af": "eic"},
    {"ml_bounds": "probability", "ml_target": "probability", "consider_only_true_max": False, "epsilon_greedy": False, "adaptive_method_kernel": "Matern", "af": "eic"},
    {"ml_bounds": "indicator", "ml_target": "sum", "consider_only_true_max": False, "epsilon_greedy": True, "adaptive_method_kernel": "Matern", "af": "eic"},
    {"ml_bounds": "probability", "ml_target": "probability", "consider_only_true_max": True, "epsilon_greedy": True, "adaptive_method_kernel": "None", "af": "eic"},
    {"ml_bounds": "probability", "ml_target": "product", "consider_only_true_max": True, "epsilon_greedy": False, "adaptive_method_kernel": "None", "af": "eic"},
    {"ml_bounds": "indicator", "ml_target": "sum", "consider_only_true_max": False, "epsilon_greedy": False, "adaptive_method_kernel": "None", "af": "eic"},      
    {"ml_bounds": "indicator", "ml_target": "indicator", "consider_only_true_max": False, "epsilon_greedy": True, "adaptive_method_kernel": "None", "af": "eic"},      
    {"ml_bounds": "indicator", "ml_target": "sum", "consider_only_true_max": False, "epsilon_greedy": False, "adaptive_method_kernel": "RBF", "af": "ucb"},         
    {"ml_bounds": "indicator", "ml_target": "None", "consider_only_true_max": False, "epsilon_greedy": False, "adaptive_method_kernel": "None", "af": "eic"},                        # maliboo
    {"ml_bounds": "None", "ml_target": "None", "consider_only_true_max": False, "epsilon_greedy": False, "adaptive_method_kernel": "RBF", "af": "ucb", "is_DiscreteBO": True}       # discreteBO
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

init_points = {
    "oscarp": 9,
    "stereomatch": 3,
    "query26": 3,
    "ligen": 11
}

bounds = {
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

for dataset in datasets_name:

    if dataset in ['oscarp', 'ligen']:
        configurations = configurations_1
    elif dataset in ['stereomatch', 'query26']:
        configurations = configurations_2

    for config in configurations:

        output_path = "./outputs/%s/bounds_%s/target_%s/only_true_max_%s/eps_greedy_%s/adaptive_%s/af_%s" %(dataset, config["ml_bounds"], config["ml_target"], config["consider_only_true_max"], config["epsilon_greedy"], config["adaptive_method_kernel"], config["af"])

        if "is_DiscreteBO" in config:
            is_DBO = config["is_DiscreteBO"]
        else:
            is_DBO = False

        print_results_multiple_thresholds(output_path, maximum[dataset], init_points[dataset], is_DBO=is_DBO, bounds=bounds[dataset], dataset=datasets[dataset], thresholds=thresholds[dataset])
