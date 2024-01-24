from maliboo import BayesianOptimization as BO
from numpy.random import randint
from maliboo.util import evaluate_max

def apply_BO(seed, n0, n_iter, debug, output_path, delete_previous_output):

    if delete_previous_output:
        import os; os.system("rm -rf %s" %output_path)

    optimizer = BO(f=None, 
                pbounds = {'#vm': [3, 26.00001], 
                           'ram': [0.2, 9.00001]},
                dataset='resources/query26_vm_ram.csv', 
                target_column='-cost',
                random_state=seed, 
                output_path=output_path, 
                debug=debug)
    
    def P(x):
        return x[:,0]

    stop_crit_info = {'hard_stop': False, 'conjunction': 'or', 'ml_bounds_coeff': (0.9, None)}
    
    acq_info = {'eic_ml_var': 'C', 'ml_target': 'time', 'eic_bounds': (0, 205000), 'eic_P_func': P, 'ml_bounds': (0, 205000)}
    optimizer.maximize(init_points=0, n_iter=n_iter, acq='eic',
                       ml_on_bounds=True, ml_on_target=True,
                        memory_queue_len=n_iter+n0, acq_info=acq_info,
                        stop_crit_info=stop_crit_info, relaxation=False)
    """
    acq_info = {'eic_ml_var': 'C', 'ml_target': 'time', 'ml_bounds': (0, 205000), 'alpha': 1, 'beta': 0.1}
    optimizer.maximize(init_points=n0, n_iter=n_iter, acq='MIVABO_ml', 
                    memory_queue_len=n_iter+n0, acq_info=acq_info) 
    """

    obtained_max = evaluate_max('resources/query26_vm_ram.csv', output_path + "/results.csv", '-cost', {'time': (0, 205000)})
    real_max = -4079658

    print("\nObtained min: " + str(round(-obtained_max,2)))
    print("Real min: " + str(round(-real_max,2)))
    print("Error: " + str(round(100*(obtained_max - real_max)/real_max,2)) + " %\n")

    return obtained_max



if __name__ == '__main__':

    n0 = 3
    n_iter = 30
    debug = False
    output_path = "./outputs_query26_eic_ml"
    delete_previous_output = True

    real_max = -4079658

    repetitions = 10
    avg = 0

    for seed in randint(0,1000,repetitions).tolist():
    #for seed in [ii for ii in range(repetitions)]:
        avg += apply_BO(seed, n0, n_iter, debug, output_path, delete_previous_output)/repetitions

    print("Average error: %s" %(round(100*(avg - real_max)/real_max,2)) + "%")

