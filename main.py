import argparse
import yaml
import pandas as pd
from maliboo import BayesianOptimization as BO
from numpy.random import randint
from maliboo.util import evaluate_max, print_final_results
import numpy as np
from math import sqrt
import time
import csv


def parse_yaml_file(file_path):
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    return data


def load_function_definition(function_name, function_data):

    parameters = function_data['parameters']
    body = function_data['body']

    exec(f'def {function_name}({", ".join(parameters)}): {body}')

    return locals()[function_name]


def mean(lst):
    if len(lst) == 0:
        return 0
    return sum(lst) / len(lst)


def variance(lst):
    if len(lst) == 0:
        return 0.0 

    mean_val = sum(lst) / len(lst)
    
    if abs(mean_val) == np.inf:
        return np.inf

    squared_diff = [(x - mean_val) ** 2 for x in lst]
    variance_val = sum(squared_diff) / len(lst)

    return variance_val


def main(yaml_file_path, is_DBO=False, print_res=True):
    parsed_data = parse_yaml_file(yaml_file_path)

    n0 = parsed_data['general_setting']['num_initial_points']

    variables_names = parsed_data['optimizer']['pbounds']

    if n0 == 0 and parsed_data['general_setting']['initial_points'] != 'None':
        init_points = pd.DataFrame()
        for point in parsed_data['general_setting']['initial_points']:
            point_ = {}
            for id_var in range(len(variables_names)):
                point_[variables_names[id_var]] = point[id_var]
            init_points = init_points.append(point_, ignore_index=True)
    else:
        init_points = None

    n_iter = parsed_data['general_setting']['num_iterations']

    debug = parsed_data['general_setting']['debug']
    verbose = parsed_data['general_setting']['verbose']

    output_path = parsed_data['general_setting']['output_path']
    delete_previous_output = parsed_data['general_setting']['delete_previous_output']

    real_max = parsed_data['general_setting']['max']

    repetitions = parsed_data['general_setting']['repetitions']

    if parsed_data['general_setting']['random_seed']:
        seeds = randint(0,1e9,repetitions).tolist()
    else:
        seeds = parsed_data['general_setting']['seeds'][:repetitions]

    if parsed_data['optimizer']['f'] != 'None':
        f = load_function_definition('f', parsed_data['optimizer']['f'])
    else:
        f = None

    pbounds = {}
    for id_var in range(len(variables_names)): 

        lb, ub = parsed_data['optimizer']['bounds'][id_var]

        if lb == 'None':
            lb = None
        if ub == 'None':
            ub = None

        pbounds[variables_names[id_var]] = (lb, ub)

    if parsed_data['optimizer']['dataset'] != 'None':
        dataset = parsed_data['optimizer']['dataset']

    target_column = parsed_data['optimizer']['target_column']

    stopping_criteria = {}
    if parsed_data['stopping_criteria']['stopping_criteria']:
        
        stopping_criteria['hard_stop'] = parsed_data['stopping_criteria']['hard_stop']
        stopping_criteria['conjunction'] = parsed_data['stopping_criteria']['conjunction']

    acq = parsed_data['acquisition_info']['acquisition_function']
    ml_on_bounds = parsed_data['acquisition_info']['ml_on_bounds']
    ml_on_target = parsed_data['acquisition_info']['ml_on_target']
    epsilon_greedy = parsed_data['acquisition_info']['epsilon_greedy']
    adaptive_method = parsed_data['acquisition_info']['adaptive_method']
    memory_queue_len = parsed_data['acquisition_info']['memory_queue_len']
    relaxation = parsed_data['acquisition_info']['relaxation']

    acquisition_info = {}

    if adaptive_method:

        if acq == 'ucb':
            acquisition_info['initial_beta'] = parsed_data['acquisition_info']['adaptive_method_parameters']['ucb_af']['initial_beta']
            acquisition_info['beta_h'] = parsed_data['acquisition_info']['adaptive_method_parameters']['ucb_af']['beta_h']

        acquisition_info['DBO_kernel'] = parsed_data['acquisition_info']['adaptive_method_parameters']['kernel']

        if acquisition_info['DBO_kernel'] == 'RBF':
            acquisition_info['initial_l'] = parsed_data['acquisition_info']['adaptive_method_parameters']['RBF_parameters']['initial_l']
            acquisition_info['l_h'] = parsed_data['acquisition_info']['adaptive_method_parameters']['RBF_parameters']['l_h']
            acquisition_info['sigma_2'] = parsed_data['acquisition_info']['adaptive_method_parameters']['RBF_parameters']['sigma_2']

        elif acquisition_info['DBO_kernel'] == 'Matern':
            acquisition_info['initial_nu'] = parsed_data['acquisition_info']['adaptive_method_parameters']['Matern_parameters']['initial_nu']
            acquisition_info['nu_h'] = parsed_data['acquisition_info']['adaptive_method_parameters']['Matern_parameters']['nu_h']

    if acq == 'eic':

        lb, ub = parsed_data['acquisition_info']['eic']['eic_bounds']

        if lb == 'None':
            lb = None
        if ub == 'None':
            ub = None
        
        acquisition_info['eic_bounds'] = (lb, ub)

        if parsed_data['acquisition_info']['eic']['eic_P_function'] != 'None':
            eic_P_function = load_function_definition('eic_P_function', parsed_data['acquisition_info']['eic']['eic_P_function'])
            acquisition_info['eic_P_func'] = eic_P_function

        if parsed_data['acquisition_info']['eic']['eic_Q_function'] != 'None':
            eic_Q_function = load_function_definition('eic_Q_function', parsed_data['acquisition_info']['eic']['eic_Q_function'])
            acquisition_info['eic_Q_func'] = eic_Q_function

        if parsed_data['acquisition_info']['eic']['eic_ml_exp_B'] != 'None':
            acquisition_info['eic_ml_exp_B'] = parsed_data['acquisition_info']['eic']['eic_ml_exp_B']

        if ml_on_bounds:
            acquisition_info['eic_ml_var'] = parsed_data['acquisition_info']['eic']['eic_ml_var']

    consider_max_only_on_feasible = False

    if ml_on_bounds or is_DBO:
        acquisition_info['ml_target'] = parsed_data['acquisition_info']['ml_on_bounds_parameters']['ml_target']

        lb, ub = parsed_data['acquisition_info']['ml_on_bounds_parameters']['ml_bounds']

        if lb == 'None':
            lb = None
        if ub == 'None':
            ub = None
            
        acquisition_info['ml_bounds'] = (lb, ub)

    if ml_on_bounds:
        consider_max_only_on_feasible = parsed_data['acquisition_info']['consider_max_only_on_feasible']
        acquisition_info['ml_bounds_alpha'] = parsed_data['acquisition_info']['ml_on_bounds_parameters']['ml_bounds_alpha']

        acquisition_info['ml_bounds_type'] = parsed_data['acquisition_info']['ml_on_bounds_parameters']['ml_bounds_type']

        if parsed_data['stopping_criteria']['stopping_criteria']:

            lb, ub = parsed_data['acquisition_info']['ml_on_bounds_parameters']['ml_bounds_coeff']

            if lb == 'None':
                lb = None
            if ub == 'None':
                ub = None

            stopping_criteria['ml_bounds_coeff'] = (lb, ub)

    if ml_on_target:
        acquisition_info['ml_target_type'] = parsed_data['acquisition_info']['ml_on_target_parameters']['ml_target_type']
        acquisition_info['ml_target_alpha'] = parsed_data['acquisition_info']['ml_on_target_parameters']['ml_target_alpha']

        if parsed_data['acquisition_info']['ml_on_target_parameters']['ml_target_type'] in ['probability', 'indicator']:
            
            lb, ub = parsed_data['acquisition_info']['ml_on_target_parameters']['indicator_parameters']['coeff']

            if lb == 'None':
                lb = None
            if ub == 'None':
                ub = None

            acquisition_info['ml_target_coeff'] = (lb, ub)

        if parsed_data['acquisition_info']['ml_on_target_parameters']['ml_target_type'] == 'sum':
            acquisition_info['ml_target_gamma_iter0'] = parsed_data['acquisition_info']['ml_on_target_parameters']['sum_parameters']['ml_target_gamma_iter0']
            acquisition_info['ml_target_gamma_iterN'] = parsed_data['acquisition_info']['ml_on_target_parameters']['sum_parameters']['ml_target_gamma_iterN']
            acquisition_info['ml_target_gamma_max'] = parsed_data['acquisition_info']['ml_on_target_parameters']['sum_parameters']['ml_target_gamma_max']

    if epsilon_greedy:
        acquisition_info['eps_greedy_random_prob'] = parsed_data['acquisition_info']['epsilon_greedy_parameters']['prob_random_pick']

    if delete_previous_output:
        from os import system 
        system("rm -rf %s" %output_path)

    results = []
    durations = []

    for idx in range(len(seeds)):

        seed = seeds[idx]

        print("\nSimulation %s out of %s, Seed: %s\n" %(idx+1,len(seeds),seed))

        start_time = time.time()

        optimizer = BO(f=f, 
                    pbounds=pbounds,
                    dataset=dataset, 
                    target_column=target_column,
                    random_state=seed, 
                    output_path=output_path+"/%s" %idx, 
                    debug=debug,
                    verbose=verbose)
        
        if init_points != None:
            optimizer.add_initial_points(init_points)
        
        optimizer.maximize(init_points=n0, 
                           n_iter=n_iter, 
                           acq=acq,
                           ml_on_bounds=ml_on_bounds, 
                           ml_on_target=ml_on_target,
                           epsilon_greedy=epsilon_greedy,
                           adaptive_method=adaptive_method,
                           memory_queue_len=memory_queue_len, 
                           acq_info=acquisition_info,
                           stop_crit_info=stopping_criteria, 
                           relaxation=relaxation,
                           consider_max_only_on_feasible=consider_max_only_on_feasible)
        
        durations.append(time.time() - start_time)
        
        if ml_on_bounds or is_DBO:
            obtained_max = evaluate_max(dataset, output_path+"/%s" %idx + "/results.csv", target_column, {acquisition_info['ml_target']: acquisition_info['ml_bounds']}, print_res)
        else:
            obtained_max = optimizer.max['target']

        if print_res:
            #print("\nObtained min: " + str(round(-obtained_max,2)))
            #print("Real min: " + str(round(-real_max,2)))
            print("Error: " + str(round(100*(obtained_max - real_max)/real_max,2)) + " %%, Time: %s sec\n" %durations[-1])

        results.append(obtained_max)

    results_cleaned = [x for x in results if x != float('inf') and x != float('-inf')]

    if print_res:
        print("Average error: %s" %(round(100*(mean(results) - real_max)/real_max,2)) + "%\n")

    if is_DBO:
        print_final_results(output_path, real_max, n0, True, {acquisition_info['ml_target']: acquisition_info['ml_bounds']})
    else:
        print_final_results(output_path, real_max, n0)

    return 100*(mean(results) - real_max)/real_max, 100*sqrt(variance(results))/abs(real_max), 100*(mean(results_cleaned) - real_max)/(real_max), 100*sqrt(variance(results_cleaned))/abs(real_max), 100*len(results_cleaned)/repetitions, mean(durations)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parse YAML file.')
    parser.add_argument('-f', '--file', type=str, help='Path to the YAML file', required=True)
    args = parser.parse_args()

    avg_res, stddev_res, avg_clean_res, stddev_clean_res, feas, dur = main(args.file)
    dataset = args.file.split(".yaml")[0]

    header = ['error (%)', 'std_dev (%)', 'time (s)']
    data = [[round(avg_res,3), round(stddev_res,3), round(dur,3)]]

    with open("%s_single.csv" %dataset, 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(header)
        writer.writerows(data)  