from maliboo import BayesianOptimization as BO

seed = 1
n0 = 3
n_iter = 60
debug = None
output_path = "./outputs_MIVABO"
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

#optimizer.maximize(init_points=n0, n_iter=n_iter, acq='ei', memory_queue_len=n_iter)
optimizer.maximize(init_points=n0, n_iter=n_iter, acq='MIVABO', memory_queue_len=n_iter)

obtained_max = optimizer.max['target']
real_max = -36791

print("\nObtained min: " + str(round(-obtained_max,2)))
print("Real min: " + str(round(-real_max,2)))
print("Error: " + str(round(100*(obtained_max - real_max)/real_max,2)) + " %\n")