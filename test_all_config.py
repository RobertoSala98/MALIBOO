from main import main as test_config
import csv
from multiprocessing import Pool
import functools

cores_number = 5
test_file = 'query26.yaml'

header = ['ml_bounds', 'ml_target', 'consider_only_true_max', 'error (%)', 'error_cleaned (%)', 'feasible_values_found (%)']
data = []

setting = []

idx_setting = 0

for ml_bounds in ['indicator', 'probability']:
    for ml_target in ['None', 'indicator', 'probability', 'sum', 'product']:
        for consider_only_true_max in [True, False]:

            # Modify yaml file
            with open(test_file, 'r') as file:
                lines = file.readlines()

            for idx in range(len(lines)):

                line = lines[idx]

                output_path = "./outputs/query26/config_%s" %idx_setting
                if "  output_path: " in line:
                    lines[idx] = "  output_path: " + output_path + "\n"

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

            output_name = "input_files/query26/" + test_file.split(".yaml")[0] + "_" + str(idx_setting) + ".yaml"

            with open(output_name, 'w') as file:
                file.writelines(lines)

            idx_setting += 1

            setting.append([ml_bounds, ml_target, consider_only_true_max, output_name])


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
        result.append(res_sim[0])
        result.append(res_sim[1])
        result.append(res_sim[2])

        results.append(result)

    return results


with Pool(processes = cores_number) as pool:

    partial_gp = functools.partial(process_batch)
        
    batch_results_parallel = pool.map(partial_gp, setting_parallel)

for cc in range(cores_number):
    data = data + batch_results_parallel[cc] 


with open("query26.csv", 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f, delimiter=',')
    writer.writerow(header)
    writer.writerows(data)  