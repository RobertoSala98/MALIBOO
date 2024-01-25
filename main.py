import argparse
import yaml
import pandas as pd
from maliboo import BayesianOptimization as BO
from numpy.random import randint
from maliboo.util import evaluate_max


def parse_yaml_file(file_path):
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    return data


def load_function_definition(function_name, function_data):

    parameters = function_data['parameters']
    body = function_data['body']

    exec(f'def {function_name}({", ".join(parameters)}): {body}')

    return locals()[function_name]


def main(yaml_file_path):
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

        lb, ub = parsed_data['stopping_criteria']['ml_bounds_coeff']

        if lb == 'None':
            lb = None
        if ub == 'None':
            ub = None

        stopping_criteria['ml_bounds_coeff'] = (lb, ub)

    acq = parsed_data['acquisition_info']['acquisition_function']
    ml_on_bounds = parsed_data['acquisition_info']['ml_on_bounds']
    ml_on_target = parsed_data['acquisition_info']['ml_on_target']
    memory_queue_len = parsed_data['acquisition_info']['memory_queue_len']
    relaxation = parsed_data['acquisition_info']['relaxation']

    acquisition_info = {}
    if acq == 'eic':

        lb, ub = parsed_data['acquisition_info']['eic']['eic_bounds']

        if lb == 'None':
            lb = None
        if ub == 'None':
            ub = None
        
        acquisition_info['eic_bounds'] = (lb, ub)

        if parsed_data['acquisition_info']['eic']['eic_P_function'] != 'None':
            eic_P_function = load_function_definition('eic_P_function', parsed_data['acquisition_info']['eic']['eic_P_function'])
            acquisition_info['eic_P_function'] = eic_P_function

        if parsed_data['acquisition_info']['eic']['eic_Q_function'] != 'None':
            eic_Q_function = load_function_definition('eic_Q_function', parsed_data['acquisition_info']['eic']['eic_Q_function'])
            acquisition_info['eic_Q_function'] = eic_Q_function

        if parsed_data['acquisition_info']['eic']['eic_ml_exp_B'] != 'None':
            acquisition_info['eic_ml_exp_B'] = parsed_data['acquisition_info']['eic']['eic_ml_exp_B']

        if ml_on_bounds:
            acquisition_info['eic_ml_var'] = parsed_data['acquisition_info']['eic']['eic_ml_var']

    if ml_on_bounds:
        acquisition_info['ml_target'] = parsed_data['acquisition_info']['ml_on_bounds_parameters']['ml_target']

        lb, ub = parsed_data['acquisition_info']['ml_on_bounds_parameters']['ml_bounds']

        if lb == 'None':
            lb = None
        if ub == 'None':
            ub = None
            
        acquisition_info['ml_bounds'] = (lb, ub)

    if ml_on_target:
        acquisition_info['ml_target_type'] = parsed_data['acquisition_info']['ml_on_target_parameters']['ml_target_type']

        if parsed_data['acquisition_info']['ml_on_target_parameters']['ml_target_type'] == 'sum':
            acquisition_info['ml_target_gamma_iter0'] = parsed_data['acquisition_info']['ml_on_target_parameters']['sum_parameters']['ml_target_gamma_iter0']
            acquisition_info['ml_target_gamma_iterN'] = parsed_data['acquisition_info']['ml_on_target_parameters']['sum_parameters']['ml_target_gamma_iterN']
            acquisition_info['ml_target_gamma_max'] = parsed_data['acquisition_info']['ml_on_target_parameters']['sum_parameters']['ml_target_gamma_max']

    if delete_previous_output:
        from os import system 
        system("rm -rf %s" %output_path)


    avg = 0.0

    for seed in seeds:

        print("\nSeed: %s\n" %seed)

        optimizer = BO(f=f, 
                    pbounds=pbounds,
                    dataset=dataset, 
                    target_column=target_column,
                    random_state=seed, 
                    output_path=output_path, 
                    debug=debug)
        
        if init_points != None:
            optimizer.add_initial_points(init_points)
        
        optimizer.maximize(init_points=n0, 
                           n_iter=n_iter, 
                           acq=acq,
                           ml_on_bounds=ml_on_bounds, 
                           ml_on_target=ml_on_target,
                           memory_queue_len=memory_queue_len, 
                           acq_info=acquisition_info,
                           stop_crit_info=stopping_criteria, 
                           relaxation=relaxation)
        
        obtained_max = evaluate_max(dataset, output_path + "/results.csv", target_column, {acquisition_info['ml_target']: acquisition_info['ml_bounds']})

        print("\nObtained min: " + str(round(-obtained_max,2)))
        print("Real min: " + str(round(-real_max,2)))
        print("Error: " + str(round(100*(obtained_max - real_max)/real_max,2)) + " %\n")

        avg += obtained_max/repetitions

    print("Average error: %s\n" %(round(100*(avg - real_max)/real_max,2)) + "%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parse YAML file.')
    parser.add_argument('-f', '--file', type=str, help='Path to the YAML file', required=True)
    args = parser.parse_args()

    main(args.file)