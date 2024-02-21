from main import main as test_config
import csv
from multiprocessing import Pool
import functools

cores_number = 10

for dataset in ["oscarp", "query26", "stereomatch", "ligen"]:

    test_file = '%s.yaml' %dataset

    header = ['ml_bounds', 'ml_target', 'consider_only_true_max', 'error (%)', 'std_dev (%)', 'error_cleaned (%)', 'std_dev_cleaned (%)', 'feasible_values_found (%)', 'avg_time (sec)']
    data = []

    setting = []

    idx_setting = 0

    if dataset in ["oscarp", "ligen"]:
        algorithms = ['ei', 'DiscreteBO']
    else:
        algorithms = ['eic', 'DiscreteBO']

    for algorithm in algorithms:
        for ml_bounds in ['None']:
            for ml_target in ['None', 'indicator', 'probability', 'sum', 'product']:
                for consider_only_true_max in [False]:

                    # Modify yaml file
                    with open(test_file, 'r') as file:
                        lines = file.readlines()

                    for idx in range(len(lines)):

                        line = lines[idx]

                        output_path = "./outputs/%s/config_%s" %(dataset,idx_setting)
                        if "  output_path: " in line:
                            lines[idx] = "  output_path: " + output_path + "\n"

                        if "  acquisition_function: " in line:
                            lines[idx] = "  acquisition_function: " + algorithm + "\n"

                        if "  ml_on_target: " in line: 

                            if ml_target == 'None':
                                lines[idx] = "  ml_on_target: " + "False\n"
                            else:
                                lines[idx] = "  ml_on_target: " + "True\n"
                                
                        if "  consider_max_only_on_feasible: " in line:
                            lines[idx] = "  consider_max_only_on_feasible: " + str(consider_only_true_max) + "\n"

                        if "    ml_bounds_type: " in line:
                            lines[idx] = "    ml_bounds_type: " + ml_bounds + "\n"

                        if "    ml_target_type: " in line:
                            lines[idx] = "    ml_target_type: " + ml_target + "\n"

                    output_name = "input_files/%s/" %dataset + test_file.split(".yaml")[0] + "_" + str(idx_setting) + ".yaml"

                    with open(output_name, 'w') as file:
                        file.writelines(lines)

                    idx_setting += 1

                    setting.append([algorithm, ml_bounds, ml_target, consider_only_true_max, output_name])


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


    with open("%s_noBounds_all.csv" %dataset, 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(header)
        writer.writerows(data)  
