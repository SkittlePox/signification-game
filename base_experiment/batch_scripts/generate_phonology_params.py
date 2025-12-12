# generate_params.py
import itertools

param_grid = {
    'listener_arch_indices': [0, 1, 2, 3, 4, 5, 6],
    'speaker_arch_indices': [0, 1, 2, 3, 4],
}

for ablate_condition in ("50", "25"):
    with open(f'sweep_params/sweep_phones_params_{ablate_condition}.txt', 'w') as f:
        num_params = 0
        for params in itertools.product(*param_grid.values()):
            depth, width = params
            f.write(f'LISTENER_ARCH="conv-ablate-{ablate_condition}-{depth}" SPEAKER_ARCH="splines-ablate-{ablate_condition}-{width}"\n')
            num_params += 1
        print("Number of parameter combinations for grid:", num_params)
    
    with open(f'sweep_params/sweep_phones_listener_pretrain_{ablate_condition}.txt', 'w') as f:
        num_params = 0
        for param in param_grid['listener_arch_indices']:
            f.write(f'LISTENER_ARCH="conv-ablate-{ablate_condition}-{param}"\n')
            num_params += 1
        print("Number of parameter combinations for listeners:", num_params)
