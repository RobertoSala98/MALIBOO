from maliboo import BayesianOptimization as BO

optimizer = BO(f=None, 
               pbounds={'ALIGN_SPLIT': [8, 72.01], 
                        'OPTIMIZE_SPLIT': [8, 72.01],
                        'OPTIMIZE_REPS': [1, 5.01], 
                        'CUDA_THREADS': [32, 256.01],
                        'N_RESTART': [256, 1024.01], 
                        'CLIPPING': [10, 256.01],
                        'SIM_THRESH': [1, 4.01], 
                        'BUFFER_SIZE': [1048576, 52428800.01]},
               dataset='resources/ligen_synth_table.csv', target_column='-RMSD^3*TIME')

optimizer.maximize(init_points=10, n_iter=100)
