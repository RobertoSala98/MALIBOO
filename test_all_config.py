from main import main as test_config
import csv
from multiprocessing import Pool
import functools
import os

cores_number = 6
datasets = ["oscarp", "query26", "stereomatch", "ligen"]

configurations_1 = [
    {"ml_bounds": "indicator", "ml_target": "None", "consider_only_true_max": False, "epsilon_greedy": False, "adaptive_method_kernel": "None", "af": "ei"},          # maliboo
    {"ml_bounds": "probability", "ml_target": "None", "consider_only_true_max": False, "epsilon_greedy": False, "adaptive_method_kernel": "None", "af": "ei"},
    {"ml_bounds": "indicator", "ml_target": "indicator", "consider_only_true_max": False, "epsilon_greedy": False, "adaptive_method_kernel": "None", "af": "ei"},
    {"ml_bounds": "indicator", "ml_target": "probability", "consider_only_true_max": False, "epsilon_greedy": False, "adaptive_method_kernel": "None", "af": "ei"},
    {"ml_bounds": "indicator", "ml_target": "sum", "consider_only_true_max": False, "epsilon_greedy": False, "adaptive_method_kernel": "None", "af": "ei"},
    {"ml_bounds": "indicator", "ml_target": "product", "consider_only_true_max": False, "epsilon_greedy": False, "adaptive_method_kernel": "None", "af": "ei"},
    {"ml_bounds": "indicator", "ml_target": "None", "consider_only_true_max": True, "epsilon_greedy": False, "adaptive_method_kernel": "None", "af": "ei"},
    {"ml_bounds": "indicator", "ml_target": "None", "consider_only_true_max": False, "epsilon_greedy": True, "adaptive_method_kernel": "None", "af": "ei"},
    {"ml_bounds": "indicator", "ml_target": "None", "consider_only_true_max": False, "epsilon_greedy": False, "adaptive_method_kernel": "RBF", "af": "ucb"},       # DiscreteBO            
    {"ml_bounds": "indicator", "ml_target": "None", "consider_only_true_max": False, "epsilon_greedy": False, "adaptive_method_kernel": "Matern", "af": "ucb"},
    {"ml_bounds": "indicator", "ml_target": "None", "consider_only_true_max": False, "epsilon_greedy": False, "adaptive_method_kernel": "RBF", "af": "ei"},          
    {"ml_bounds": "indicator", "ml_target": "None", "consider_only_true_max": False, "epsilon_greedy": False, "adaptive_method_kernel": "Matern", "af": "ei"}
]

configurations_2 = [
    {"ml_bounds": "indicator", "ml_target": "None", "consider_only_true_max": False, "epsilon_greedy": False, "adaptive_method_kernel": "None", "af": "eic"},          # maliboo
    {"ml_bounds": "probability", "ml_target": "None", "consider_only_true_max": False, "epsilon_greedy": False, "adaptive_method_kernel": "None", "af": "eic"},
    {"ml_bounds": "indicator", "ml_target": "indicator", "consider_only_true_max": False, "epsilon_greedy": False, "adaptive_method_kernel": "None", "af": "eic"},
    {"ml_bounds": "indicator", "ml_target": "probability", "consider_only_true_max": False, "epsilon_greedy": False, "adaptive_method_kernel": "None", "af": "eic"},
    {"ml_bounds": "indicator", "ml_target": "sum", "consider_only_true_max": False, "epsilon_greedy": False, "adaptive_method_kernel": "None", "af": "eic"},
    {"ml_bounds": "indicator", "ml_target": "product", "consider_only_true_max": False, "epsilon_greedy": False, "adaptive_method_kernel": "None", "af": "eic"},
    {"ml_bounds": "indicator", "ml_target": "None", "consider_only_true_max": True, "epsilon_greedy": False, "adaptive_method_kernel": "None", "af": "eic"},
    {"ml_bounds": "indicator", "ml_target": "None", "consider_only_true_max": False, "epsilon_greedy": True, "adaptive_method_kernel": "None", "af": "eic"},
    {"ml_bounds": "indicator", "ml_target": "None", "consider_only_true_max": False, "epsilon_greedy": False, "adaptive_method_kernel": "RBF", "af": "ucb"},       # DiscreteBO            
    {"ml_bounds": "indicator", "ml_target": "None", "consider_only_true_max": False, "epsilon_greedy": False, "adaptive_method_kernel": "Matern", "af": "ucb"},
    {"ml_bounds": "indicator", "ml_target": "None", "consider_only_true_max": False, "epsilon_greedy": False, "adaptive_method_kernel": "RBF", "af": "eic"},          
    {"ml_bounds": "indicator", "ml_target": "None", "consider_only_true_max": False, "epsilon_greedy": False, "adaptive_method_kernel": "Matern", "af": "eic"}
]

"""
thresholds = {
    "oscarp": [60, 150, 300, 450, 600],
    "query26": [185000, 195000, 205000, 215000, 225000],
    "stereomatch": [9500, 11000, 12500, 14000, 15500, 17000, 18500, 20000, 25000, 30000, 35000, 40000, 45000, 50000, 55000, 60000, 65000, 70000, 75000],
    "ligen": [1.999, 2.05, 2.10, 2.15, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7]
}
"""
thresholds = {
    "oscarp": [300],            # 300
    "query26": [205000],        # 205000
    "stereomatch": [17000],     # 17000
    "ligen": [2.1]              # 2.1
}


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

    test_file = '%s.yaml' %dataset

    header = ['ml_bounds', 'ml_target', 'consider_only_true_max', 'epsilon_greedy', 'adaptive_method_kernel', 'af', 'threshold', 'error (%)', 'std_dev (%)', 'error_cleaned (%)', 'std_dev_cleaned (%)', 'feasible_values_found (%)', 'avg_time (sec)']
    data = []

    setting = []
    idx_setting = 0

    if dataset in ['oscarp', 'ligen']:
        configurations = configurations_1
    elif dataset in ['stereomatch', 'query26']:
        configurations = configurations_2

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

            setting.append([ml_bounds, ml_target, consider_only_true_max, epsilon_greedy, adaptive_method_kernel, af, threshold, output_name])


    print("You are testing %s different settings" %idx_setting)


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


    def process_batch(settings):

        results = []

        for setting in settings:

            result = setting[:-1]
            res_sim = test_config(setting[-1])
            for elem in res_sim:
                result.append(round(elem,3))

            results.append(result)

        return results

    
    with Pool(processes = cores_number) as pool:

        partial_gp = functools.partial(process_batch)
            
        batch_results_parallel = pool.map(partial_gp, setting_parallel)

    for cc in range(cores_number):
        data = data + batch_results_parallel[cc] 

    #process_batch(setting)


    with open("%s.csv" %dataset, 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(header)
        writer.writerows(data)  
