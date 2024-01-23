from maliboo import BayesianOptimization as BO

seed = 16
n0 = 6
n_iter = 9
debug = False
output_path = "./outputs_oscarp_combinatorial_ei_ml"
delete_previous_output = True

if delete_previous_output:
    import os; os.system("rm -rf %s" %output_path)

optimizer = BO(f=None, 
               pbounds = {'parallelism_ffmpeg-0': [2,4.00001], 
                          'parallelism_librosa': [2, 6.00001],
                          'parallelism_ffmpeg-1': [2, 4.00001], 
                          'parallelism_ffmpeg-2': [2, 8.00001],
                          'parallelism_deepspeech': [2, 4.00001]},
               dataset='resources/OSCAR-P_combinatorial.csv', 
               target_column='-cost',
               random_state=seed, 
               output_path=output_path, 
               debug=debug)

optimizer.maximize(init_points=n0, n_iter=n_iter, acq='ei_ml', acq_info={'ml_target': 'total_time', 'ml_bounds': (0.0, 300.00001)}, memory_queue_len=n_iter+n0)

obtained_max = optimizer.max['target']
real_max = -0.173553833333333

print("\nObtained min: " + str(round(-obtained_max,5)))
print("Real min: " + str(round(-real_max,5)))
print("Error: " + str(round(100*(obtained_max - real_max)/real_max,5)) + " %\n")