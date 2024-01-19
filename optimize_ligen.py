from maliboo import BayesianOptimization as BO

seed = 1
n0 = 11
n_iter = 60
debug = False
output_path = "./outputs_ligen_MIVABO_ml"
delete_previous_output = True

if delete_previous_output:
    import os; os.system("rm -rf %s" %output_path)

optimizer = BO(f=None, 
               pbounds={'ALIGN_SPLIT': [8, 72.01], 
                        'OPTIMIZE_SPLIT': [8, 72.01],
                        'OPTIMIZE_REPS': [1, 5.01], 
                        'CUDA_THREADS': [32, 256.01],
                        'N_RESTART': [256, 1024.01], 
                        'CLIPPING': [10, 256.01],
                        'SIM_THRESH': [1, 4.01], 
                        'BUFFER_SIZE': [1, 50.01]},
               dataset='resources/ligen_synth_table.csv', 
               target_column='-RMSD^3*TIME',
               random_state=seed, 
               output_path=output_path, 
               debug=debug)

#optimizer.maximize(init_points=n0, n_iter=n_iter, acq='ucb', memory_queue_len=n_iter)
#optimizer.maximize(init_points=n0, n_iter=n_iter, acq='ei', memory_queue_len=n_iter)
#optimizer.maximize(init_points=n0, n_iter=n_iter, acq='poi', memory_queue_len=n_iter)
#optimizer.maximize(init_points=n0, n_iter=n_iter, acq='ei_ml', acq_info={'ml_target': 'RMSD_0.75', 'ml_bounds': (0.0, 2.1)}, memory_queue_len=n_iter)
#optimizer.maximize(init_points=n0, n_iter=n_iter, acq='ei_ml', acq_info={'ml_target': 'RMSD_0.75', 'ml_bounds': (0.0, 2.1)})
#optimizer.maximize(init_points=n0, n_iter=n_iter, acq='MIVABO', memory_queue_len=n0+n_iter)
optimizer.maximize(init_points=n0, n_iter=n_iter, acq='MIVABO_ml', memory_queue_len=n0+n_iter)

obtained_max = optimizer.max['target']
real_max = -567.312555400384

print("\nObtained min: " + str(round(-obtained_max,2)))
print("Real min: " + str(round(-real_max,2)))
print("Error: " + str(round(100*(obtained_max - real_max)/real_max,2)) + " %\n")