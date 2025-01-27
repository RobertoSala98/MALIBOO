from main import main as test_config
import csv
from multiprocessing import Pool
import functools
import os
from numpy.random import randint

cores_number = 10
datasets = ["ligen"]

def generate_folder_structure(path):

    subfolders = path.split("/")

    actual_folder = ""

    for elem in subfolders:

        if elem == ".":
            actual_folder = "."

        else:
            if not os.path.exists(actual_folder + "/" + elem):
                os.makedirs(actual_folder + "/" + elem)
            actual_folder = actual_folder + "/" + elem


for dataset in datasets:

    test_file = 'yaml_files/%s.yaml' %dataset

    header = ['ml_bounds', 'ml_target', 'consider_only_true_max', 'epsilon_greedy', 'adaptive_method_kernel', 'af', 'threshold', 'error (%)', 'std_dev (%)', 'error_cleaned (%)', 'std_dev_cleaned (%)', 'feasible_values_found (%)', 'avg_time (sec)']
    data = []

    setting = []
    idx_setting = 0
    
    # Ablation study: all bounds but only one feature at a time
    thresholds = {
        "oscarp": [150, 300, 450, 600],
        "query26": [195000, 205000, 215000, 225000],
        "stereomatch": [8000, 10000, 12000, 17000, 20000, 40000],
        "ligen": [2.0, 2.1, 2.2, 2.45, 2.75],
        "query52": [262887, 334000, 407000, 645000, 2.25e6]
    }

    maximum = {
        "oscarp": [-0.174361111, -0.174361111, -0.174361111, -0.174361111],
        "query26": [-4079658.0, -4079658.0, -4079658.0, -4022118.0],
        "stereomatch": [-50776.0, -45525.0, -40196.0, -40196.0, -37412.0, -36791.0],
        "ligen": [-567.312555400384, -567.312555400384, -464.349441178703, -424.355184049556, -340.099251789615],
        "query52": [-682215, -682215, -682215, -682215, -682215]
    }
    
    configurations_best = [
        {'ml_bounds': 'indicator', 'ml_target': 'probability', 'consider_only_true_max': True, 'epsilon_greedy': True, 'adaptive_method_kernel': 'Matern', 'af': 'ei'}, # REBOLD
        {'ml_bounds': 'indicator', 'ml_target': 'probability', 'consider_only_true_max': True, 'epsilon_greedy': True, 'adaptive_method_kernel': 'Matern', 'af': 'ucb'}, # REBOLD
        {'ml_bounds': 'indicator', 'ml_target': 'probability', 'consider_only_true_max': True, 'epsilon_greedy': True, 'adaptive_method_kernel': 'RBF', 'af': 'ei'}, # REBOLD
        {'ml_bounds': 'indicator', 'ml_target': 'probability', 'consider_only_true_max': True, 'epsilon_greedy': True, 'adaptive_method_kernel': 'RBF', 'af': 'ucb'}, # REBOLD
        {'ml_bounds': 'indicator', 'ml_target': 'probability', 'consider_only_true_max': True, 'epsilon_greedy': True, 'adaptive_method_kernel': 'None', 'af': 'ei'}, # d-MALIBOO
        {'ml_bounds': 'indicator', 'ml_target': 'None', 'consider_only_true_max': True, 'epsilon_greedy': False, 'adaptive_method_kernel': 'None', 'af': 'ei'}, # MALIBOO fixed
        {'ml_bounds': 'indicator', 'ml_target': 'None', 'consider_only_true_max': False, 'epsilon_greedy': False, 'adaptive_method_kernel': 'None', 'af': 'ei'}, # MALIBOO original
        {'ml_bounds': 'None', 'ml_target': 'None', 'consider_only_true_max': False, 'epsilon_greedy': False, 'adaptive_method_kernel': 'RBF', 'af': 'ucb', "is_DiscreteBO": True} # discreteBO
    ]

    seeds = []
    for _ in range(len(thresholds[dataset])):
        seeds_ = randint(0, 1e9, 30)
        seeds.append(seeds_.tolist())

    for config in configurations_best:
        for threshold in thresholds[dataset]:

            ml_bounds = config["ml_bounds"]
            ml_target = config["ml_target"]
            consider_only_true_max = config["consider_only_true_max"]
            epsilon_greedy = config["epsilon_greedy"]
            adaptive_method_kernel = config["adaptive_method_kernel"]
            af = config["af"]

            # Modify yaml file
            with open(test_file, 'r') as file:
                lines = file.readlines()

            output_path = "./outputs/%s/bounds_%s/target_%s/only_true_max_%s/eps_greedy_%s/adaptive_%s/af_%s/threshold_%s" %(dataset, ml_bounds, ml_target, consider_only_true_max, epsilon_greedy, adaptive_method_kernel, af, threshold)
            generate_folder_structure(output_path)

            for idx in range(len(lines)):

                line = lines[idx]

                if "  initial_points_selection_method: " in line:
                    if adaptive_method_kernel == 'None' or 'is_DiscreteBO' in config:
                        lines[idx] = "  initial_points_selection_method: random" + "\n"
                    else:
                        lines[idx] = "  initial_points_selection_method: latin" + "\n"

                if "  seeds: " in line:

                    seeds_ = seeds[thresholds[dataset].index(threshold)]

                    lines[idx] = "  seeds: ["
                    for s_ in seeds_:
                        lines[idx] += str(s_)
                        if s_ != seeds_[-1]:
                            lines[idx] += ", "
                    lines[idx] += "]\n"

                if "  output_path: " in line:
                    lines[idx] = "  output_path: " + output_path + "\n"

                if "  max: " in line:
                    lines[idx] = "  max: " + str(maximum[dataset][thresholds[dataset].index(threshold)]) + "\n"

                if "  acquisition_function: " in line:
                    lines[idx] = "  acquisition_function: " + af + "\n"

                if "  ml_on_bounds: " in line: 

                    if ml_bounds == 'None':
                        lines[idx] = "  ml_on_bounds: " + "False\n"
                    else:
                        lines[idx] = "  ml_on_bounds: " + "True\n"

                if "  ml_on_target: " in line: 

                    if ml_target == 'None':
                        lines[idx] = "  ml_on_target: " + "False\n"
                    else:
                        lines[idx] = "  ml_on_target: " + "True\n"
                        
                if "  consider_max_only_on_feasible: " in line:
                    lines[idx] = "  consider_max_only_on_feasible: " + str(consider_only_true_max) + "\n"

                if "  epsilon_greedy: " in line:
                    lines[idx] = "  epsilon_greedy: " + str(epsilon_greedy) + "\n"

                if "  adaptive_method: " in line:

                    if adaptive_method_kernel == 'None':
                        lines[idx] = "  adaptive_method: " + "False\n"
                    else:
                        lines[idx] = "  adaptive_method: " + "True\n"

                if "      initial_beta: " in line and af == "ucb" and dataset == "oscarp":
                    if adaptive_method_kernel == "RBF":
                        lines[idx] = "      initial_beta: " + str(40.0) + "\n"
                    elif adaptive_method_kernel == "Matern":
                        lines[idx] = "      initial_beta: " + str(1.0) + "\n"

                if "    kernel: " in line and adaptive_method_kernel != 'None':
                    lines[idx] = "    kernel: " + adaptive_method_kernel + "\n"

                if "    ml_bounds: " in line:
                    lines[idx] = "    ml_bounds: " + "[0, %s]" %threshold + "\n"

                if "    eic_bounds: " in line:
                    lines[idx] = "    eic_bounds: " + "[0, %s]" %threshold + "\n"

                if "    ml_bounds_type: " in line:
                    lines[idx] = "    ml_bounds_type: " + ml_bounds + "\n"

                if "    ml_target_type: " in line:
                    lines[idx] = "    ml_target_type: " + ml_target + "\n"

            output_name = "input_files/%s/" %dataset + test_file.split("/")[1].split(".yaml")[0] + "_" + str(idx_setting) + ".yaml"
            generate_folder_structure("./input_files/%s/" %dataset)
            
            with open(output_name, 'w') as file:
                file.writelines(lines)

            idx_setting += 1

            if "is_DiscreteBO" in config:
                is_DBO = config["is_DiscreteBO"]
            else:
                is_DBO = False

            setting.append([ml_bounds, ml_target, consider_only_true_max, epsilon_greedy, adaptive_method_kernel, af, threshold, is_DBO, output_name])
    
    """
    # Ablation study: all bounds but only one feature at a time
    thresholds = {
        "oscarp": [150, 300, 450, 600],
        "query26": [195000, 205000, 215000, 225000, 235000, 245000],
        "stereomatch": [8000, 10000, 12000, 17000, 20000, 40000],
        "ligen": [2.0, 2.1, 2.2, 2.45, 2.75]
    }

    maximum = {
        "oscarp": [-0.174361111, -0.174361111, -0.174361111, -0.174361111],
        "query26": [-4079658.0, -4079658.0, -4079658.0, -4022118.0, -683682, -683682.0],
        "stereomatch": [-50776.0, -45525.0, -40196.0, -40196.0, -37412.0, -36791.0],
        "ligen": [-567.312555400384, -567.312555400384, -464.349441178703, -424.355184049556, -340.099251789615]
    }

    configurations_ablation = [
        {"ml_bounds": "indicator", "ml_target": "None", "consider_only_true_max": True, "epsilon_greedy": False, "adaptive_method_kernel": "None", "af": "ei"},                        # maliboo
        {"ml_bounds": "probability", "ml_target": "None", "consider_only_true_max": True, "epsilon_greedy": False, "adaptive_method_kernel": "None", "af": "ei"},
        {"ml_bounds": "indicator", "ml_target": "indicator", "consider_only_true_max": True, "epsilon_greedy": False, "adaptive_method_kernel": "None", "af": "ei"},
        {"ml_bounds": "indicator", "ml_target": "probability", "consider_only_true_max": True, "epsilon_greedy": False, "adaptive_method_kernel": "None", "af": "ei"},
        {"ml_bounds": "indicator", "ml_target": "product", "consider_only_true_max": True, "epsilon_greedy": False, "adaptive_method_kernel": "None", "af": "ei"},
        {"ml_bounds": "indicator", "ml_target": "sum", "consider_only_true_max": True, "epsilon_greedy": False, "adaptive_method_kernel": "None", "af": "ei"},
        {"ml_bounds": "indicator", "ml_target": "None", "consider_only_true_max": True, "epsilon_greedy": True, "adaptive_method_kernel": "None", "af": "ei"}, 
        {"ml_bounds": "None", "ml_target": "None", "consider_only_true_max": True, "epsilon_greedy": False, "adaptive_method_kernel": "RBF", "af": "ucb", "is_DiscreteBO": True},       # discreteBO
        {"ml_bounds": "indicator", "ml_target": "None", "consider_only_true_max": True, "epsilon_greedy": False, "adaptive_method_kernel": "RBF", "af": "ucb"},
        {"ml_bounds": "indicator", "ml_target": "None", "consider_only_true_max": True, "epsilon_greedy": False, "adaptive_method_kernel": "RBF", "af": "ei"},
        {"ml_bounds": "indicator", "ml_target": "None", "consider_only_true_max": True, "epsilon_greedy": False, "adaptive_method_kernel": "Matern", "af": "ucb"},
        {"ml_bounds": "indicator", "ml_target": "None", "consider_only_true_max": True, "epsilon_greedy": False, "adaptive_method_kernel": "Matern", "af": "ei"}
    ]

    for config in configurations_ablation:
        for threshold in thresholds[dataset]:

            ml_bounds = config["ml_bounds"]
            ml_target = config["ml_target"]
            consider_only_true_max = config["consider_only_true_max"]
            epsilon_greedy = config["epsilon_greedy"]
            adaptive_method_kernel = config["adaptive_method_kernel"]
            af = config["af"]

            # Modify yaml file
            with open(test_file, 'r') as file:
                lines = file.readlines()

            output_path = "./outputs/%s/bounds_%s/target_%s/only_true_max_%s/eps_greedy_%s/adaptive_%s/af_%s/threshold_%s" %(dataset, ml_bounds, ml_target, consider_only_true_max, epsilon_greedy, adaptive_method_kernel, af, threshold)
            generate_folder_structure(output_path)

            for idx in range(len(lines)):

                line = lines[idx]

                if "  output_path: " in line:
                    lines[idx] = "  output_path: " + output_path + "\n"

                if "  max: " in line:
                    lines[idx] = "  max: " + str(maximum[dataset][thresholds[dataset].index(threshold)]) + "\n"

                if "  acquisition_function: " in line:
                    lines[idx] = "  acquisition_function: " + af + "\n"

                if "  ml_on_bounds: " in line: 

                    if ml_bounds == 'None':
                        lines[idx] = "  ml_on_bounds: " + "False\n"
                    else:
                        lines[idx] = "  ml_on_bounds: " + "True\n"

                if "  ml_on_target: " in line: 

                    if ml_target == 'None':
                        lines[idx] = "  ml_on_target: " + "False\n"
                    else:
                        lines[idx] = "  ml_on_target: " + "True\n"
                        
                if "  consider_max_only_on_feasible: " in line:
                    lines[idx] = "  consider_max_only_on_feasible: " + str(consider_only_true_max) + "\n"

                if "  epsilon_greedy: " in line:
                    lines[idx] = "  epsilon_greedy: " + str(epsilon_greedy) + "\n"

                if "  adaptive_method: " in line:

                    if adaptive_method_kernel == 'None':
                        lines[idx] = "  adaptive_method: " + "False\n"
                    else:
                        lines[idx] = "  adaptive_method: " + "True\n"

                if "      initial_beta: " in line and af == "ucb" and dataset == "oscarp":
                    if adaptive_method_kernel == "RBF":
                        lines[idx] = "      initial_beta: " + str(40.0) + "\n"
                    elif adaptive_method_kernel == "Matern":
                        lines[idx] = "      initial_beta: " + str(1.0) + "\n"

                if "    kernel: " in line and adaptive_method_kernel != 'None':
                    lines[idx] = "    kernel: " + adaptive_method_kernel + "\n"

                if "    ml_bounds: " in line:
                    lines[idx] = "    ml_bounds: " + "[0, %s]" %threshold + "\n"

                if "    eic_bounds: " in line:
                    lines[idx] = "    eic_bounds: " + "[0, %s]" %threshold + "\n"

                if "    ml_bounds_type: " in line:
                    lines[idx] = "    ml_bounds_type: " + ml_bounds + "\n"

                if "    ml_target_type: " in line:
                    lines[idx] = "    ml_target_type: " + ml_target + "\n"

            output_name = "input_files/%s/" %dataset + test_file.split(".yaml")[0] + "_" + str(idx_setting) + ".yaml"
            generate_folder_structure("./input_files/%s/" %dataset)

            with open(output_name, 'w') as file:
                file.writelines(lines)

            idx_setting += 1

            if "is_DiscreteBO" in config:
                is_DBO = config["is_DiscreteBO"]
            else:
                is_DBO = False

            setting.append([ml_bounds, ml_target, consider_only_true_max, epsilon_greedy, adaptive_method_kernel, af, threshold, is_DBO, output_name])
    """
    """
    configurations_ablation = []
    count = 0

    configurations = []

    for ml_bounds in ['indicator', 'probability']:
        for ml_target in ['None', 'indicator', 'probability', 'sum', 'product']:
            for consider_only_true_max in [True]:
                for epsilon_greedy in [True, False]:
                    for adaptive_method_kernel in ['None']:

                        if adaptive_method_kernel != 'None':
                            for af in ['ei', 'ucb']:

                                configuration_1 = {"ml_bounds": ml_bounds, 
                                                "ml_target": ml_target, 
                                                "consider_only_true_max": consider_only_true_max, 
                                                "epsilon_greedy": epsilon_greedy, 
                                                "adaptive_method_kernel": adaptive_method_kernel, 
                                                "af": af}
                                
                                if configuration_1 not in configurations_ablation:
                                    configurations.append(configuration_1)
                                else:
                                    count += 1
                            
                        else:
                            configuration_1 = {"ml_bounds": ml_bounds, 
                                            "ml_target": ml_target, 
                                            "consider_only_true_max": consider_only_true_max, 
                                            "epsilon_greedy": epsilon_greedy, 
                                            "adaptive_method_kernel": adaptive_method_kernel, 
                                            "af": 'ei'}
                        
                            if configuration_1 not in configurations_ablation:
                                configurations.append(configuration_1)
                            else:
                                count += 1
    """
    """
    thresholds = {
        "oscarp": [300],
        "query26": [205000],
        "stereomatch": [17000],
        "ligen": [2.1]
    }

    maximum = {
        "oscarp": [-0.174361111],
        "query26": [-4079658.0],
        "stereomatch": [-40196.0],
        "ligen": [-567.312555400384]
    }
    """
    """
    thresholds = {"query26_monodimensional": [4.19e5, 4.74e5, 6.08e5, 8.63e5, 2e6], 
                  "kmeans_monodimensional": [2.41e5, 2.83e5, 3.78e5, 1.19e6, 7e6], 
                  "sparkDL_monodimensional": [1.5747e6, 1.6955e6, 1.85866e6, 2.02753e6, 2.7304e6], 
                  "query40_monodimensional": [6.5994e5, 7.3999e5, 9.345e5, 1.30749e6, 2.559e6], 
                  "query55_monodimensional": [3.2354e5, 3.6502e5, 4.63543e5, 6.5250e5, 1.28e6], 
                  "query52": [262887, 334000, 407000, 645000, 2.25e6]}

    maximum = {"query26_monodimensional": [-15672682, -12993988, -10858860, -10267476, -10114512], 
                  "kmeans_monodimensional": [-9558960, -9032384, -8611056, -8410212, -8410212], 
                  "sparkDL_monodimensional": [-46993350, -29403774, -7202776, -7202776, -4912216], 
                  "query40_monodimensional": [-23679864, -20388928, -16707492, -15485604, -14860080], 
                  "query55_monodimensional": [-11619540, -9954336, -8315334, -7715940, -7448152], 
                  "query52": [-682215, -682215, -682215, -682215, -682215]}

    for config in configurations:
        for threshold in thresholds[dataset]:

            ml_bounds = config["ml_bounds"]
            ml_target = config["ml_target"]
            consider_only_true_max = config["consider_only_true_max"]
            epsilon_greedy = config["epsilon_greedy"]
            adaptive_method_kernel = config["adaptive_method_kernel"]
            af = config["af"]

            # Modify yaml file
            with open(test_file, 'r') as file:
                lines = file.readlines()

            output_path = "./outputs/%s/bounds_%s/target_%s/only_true_max_%s/eps_greedy_%s/adaptive_%s/af_%s/threshold_%s" %(dataset, ml_bounds, ml_target, consider_only_true_max, epsilon_greedy, adaptive_method_kernel, af, threshold)
            generate_folder_structure(output_path)

            for idx in range(len(lines)):

                line = lines[idx]

                if "  output_path: " in line:
                    lines[idx] = "  output_path: " + output_path + "\n"

                if "  max: " in line:
                    lines[idx] = "  max: " + str(maximum[dataset][thresholds[dataset].index(threshold)]) + "\n"

                if "  acquisition_function: " in line:
                    lines[idx] = "  acquisition_function: " + af + "\n"

                if "  ml_on_bounds: " in line: 

                    if ml_bounds == 'None':
                        lines[idx] = "  ml_on_bounds: " + "False\n"
                    else:
                        lines[idx] = "  ml_on_bounds: " + "True\n"

                if "  ml_on_target: " in line: 

                    if ml_target == 'None':
                        lines[idx] = "  ml_on_target: " + "False\n"
                    else:
                        lines[idx] = "  ml_on_target: " + "True\n"
                        
                if "  consider_max_only_on_feasible: " in line:
                    lines[idx] = "  consider_max_only_on_feasible: " + str(consider_only_true_max) + "\n"

                if "  epsilon_greedy: " in line:
                    lines[idx] = "  epsilon_greedy: " + str(epsilon_greedy) + "\n"

                if "  adaptive_method: " in line:

                    if adaptive_method_kernel == 'None':
                        lines[idx] = "  adaptive_method: " + "False\n"
                    else:
                        lines[idx] = "  adaptive_method: " + "True\n"

                if "      initial_beta: " in line and af == "ucb" and dataset == "oscarp":
                    if adaptive_method_kernel == "RBF":
                        lines[idx] = "      initial_beta: " + str(40.0) + "\n"
                    elif adaptive_method_kernel == "Matern":
                        lines[idx] = "      initial_beta: " + str(1.0) + "\n"

                if "    kernel: " in line and adaptive_method_kernel != 'None':
                    lines[idx] = "    kernel: " + adaptive_method_kernel + "\n"

                if "    ml_bounds: " in line:
                    lines[idx] = "    ml_bounds: " + "[0, %s]" %threshold + "\n"

                if "    eic_bounds: " in line:
                    lines[idx] = "    eic_bounds: " + "[0, %s]" %threshold + "\n"

                if "    ml_bounds_type: " in line:
                    lines[idx] = "    ml_bounds_type: " + ml_bounds + "\n"

                if "    ml_target_type: " in line:
                    lines[idx] = "    ml_target_type: " + ml_target + "\n"

            output_name = "input_files/%s/" %dataset + test_file.split(".yaml")[0] + "_" + str(idx_setting) + ".yaml"
            generate_folder_structure("./input_files/%s/" %dataset)

            with open(output_name, 'w') as file:
                file.writelines(lines)

            idx_setting += 1

            if "is_DiscreteBO" in config:
                is_DBO = config["is_DiscreteBO"]
            else:
                is_DBO = False

            setting.append([ml_bounds, ml_target, consider_only_true_max, epsilon_greedy, adaptive_method_kernel, af, threshold, is_DBO, output_name])
    """
    print("You are testing %s different settings" %idx_setting)
    
    def process_batch(settings):

        results = []

        for setting in settings:

            result = setting[:-2]
            res_sim = test_config(setting[-1], setting[-2])
            for elem in res_sim:
                result.append(round(elem,3))

            results.append(result)

        return results

    
    def split_list(input_list, num_chunks):
        
        avg_chunk_size = len(input_list) // num_chunks
        remainder = len(input_list) % num_chunks

        chunks = []
        start = 0
        for i in range(num_chunks):
            end = start + avg_chunk_size + (1 if i < remainder else 0)
            chunks.append(input_list[start:end])
            start = end

        return chunks


    setting_parallel = split_list(setting, cores_number)

    with Pool(processes = cores_number) as pool:

        partial_gp = functools.partial(process_batch)
            
        batch_results_parallel = pool.map(partial_gp, setting_parallel)

    for cc in range(cores_number):
        data = data + batch_results_parallel[cc] 
    
    #data = process_batch(setting)

    with open("%s_best_config.csv" %dataset, 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(header)
        writer.writerows(data)