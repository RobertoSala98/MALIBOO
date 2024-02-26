from main import main as test_config
import csv
from multiprocessing import Pool
import functools

cores_number = 10

"""
thresholds = {
    "oscarp": [60, 150, 300, 450, 600],
    "query26": [185000, 195000, 205000, 215000, 225000],
    "stereomatch": [9500, 11000, 12500, 14000, 15500, 17000, 18500, 20000, 25000, 30000, 35000, 40000, 45000, 50000, 55000, 60000, 65000, 70000, 75000],
    "ligen": [1.999, 2.05, 2.10, 2.15, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7]
}
"""
thresholds = {
    "oscarp": [300],
    "query26": [205000],
    "stereomatch": [17000],
    "ligen": [2.1]
}

# ATTENTION: WHEN IMPLEMENTING AUTOMATIC TEST ON DiscreteBO, CONSIDER THAT OSCARP HAS DIFFENT Beta if using RBF or Matern

for dataset in ["oscarp", "query26", "stereomatch", "ligen"]:

    test_file = '%s.yaml' %dataset

    header = ['acquisition_function', 'ml_bounds', 'ml_target', 'consider_only_true_max', 'error (%)', 'std_dev (%)', 'error_cleaned (%)', 'std_dev_cleaned (%)', 'feasible_values_found (%)', 'avg_time (sec)']
    data = []

    setting = []

    idx_setting = 0

    if dataset in ["oscarp", "ligen"]:
        acq_functions = ['ei', 'DiscreteBO']
    else:
        acq_functions = ['eic', 'DiscreteBO']

    for acq_function in acq_functions:
        for ml_bounds in ['indicator', 'probability']:
            for ml_target in ['None', 'indicator', 'probability', 'sum', 'product']:
                for consider_only_true_max in [True, False]:
                    for threshold in thresholds[dataset]:

                        # Modify yaml file
                        with open(test_file, 'r') as file:
                            lines = file.readlines()

                        for idx in range(len(lines)):

                            line = lines[idx]

                            output_path = "./outputs/%s/config_%s_bounded" %(dataset,idx_setting)
                            if "  output_path: " in line:
                                lines[idx] = "  output_path: " + output_path + "\n"

                            if "  acquisition_function: " in line:
                                lines[idx] = "  acquisition_function: " + acq_function + "\n"

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

                            if "    ml_bounds: " in line:
                                lines[idx] = "    ml_bounds: " + "[0, %s]" %threshold + "\n"

                            if "    eic_bounds: " in line:
                                lines[idx] = "    eic_bounds: " + "[0, %s]" %threshold + "\n"

                            if "    ml_bounds_type: " in line:
                                lines[idx] = "    ml_bounds_type: " + ml_bounds + "\n"

                            if "    ml_target_type: " in line:
                                lines[idx] = "    ml_target_type: " + ml_target + "\n"

                        output_name = "input_files/%s/" %dataset + test_file.split(".yaml")[0] + "_" + str(idx_setting) + ".yaml"

                        with open(output_name, 'w') as file:
                            file.writelines(lines)

                        idx_setting += 1

                        setting.append([acq_function, ml_bounds, ml_target, consider_only_true_max, output_name])


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


    with open("%s_constrained_all.csv" %dataset, 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(header)
        writer.writerows(data)  
