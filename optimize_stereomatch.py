from maliboo import BayesianOptimization as BO

seed = 22121998
n0 = 3
n_iter = 60
debug = False
output_path = "./outputs_stereomatch_ei_ml"
delete_previous_output = True

if delete_previous_output:
    import os; os.system("rm -rf %s" %output_path)

optimizer = BO(f=None, 
               pbounds = {'confidence': [14, 64.00001], 
                          'hypo_step': [1, 3.00001],
                          'max_arm_length': [1, 16.00001], 
                          'num_threads': [1, 32.00001]},
               dataset='resources/stereomatch.csv', 
               target_column='-cost',
               random_state=seed, 
               output_path=output_path, 
               debug=debug)


stop_crit_info = {'hard_stop': False, 'conjunction': 'or', 'ml_bounds_coeff': (0.9, None)}
"""
acq_info = {'eic_ml_var': 'C', 'ml_target': 'time', 'eic_bounds': (0, 17000), 'ml_bounds': (0, 17000)}
optimizer.maximize(init_points=0, n_iter=n_iter, acq='eic_ml',
                    memory_queue_len=5, acq_info=acq_info,
                    stop_crit_info=stop_crit_info, relaxation=False)
"""
acq_info = {'eic_ml_var': 'C', 'ml_target': 'time', 'ml_bounds': (0, 17000), 'alpha': 1, 'beta': 0.1}
optimizer.maximize(init_points=n0, n_iter=n_iter, acq='MIVABO_ml', 
                   memory_queue_len=n_iter+n0, acq_info=acq_info) 

obtained_max = optimizer.max['target']
real_max = -36791

print("\nObtained min: " + str(round(-obtained_max,2)))
print("Real min: " + str(round(-real_max,2)))
print("Error: " + str(round(100*(obtained_max - real_max)/real_max,2)) + " %\n")