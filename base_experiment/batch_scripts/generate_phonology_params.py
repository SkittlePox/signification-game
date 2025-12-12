# generate_params.py
import itertools

param_grid = {
    'listener_arch_indices': [0, 1, 2, 3, 4, 5, 6],
    'speaker_arch_indices': [0, 1, 2, 3, 4],
}

### These are without any pretrained listeners
for ablate_condition in ("50", "25"):
    with open(f'sweep_params/sweep_phones_no_pretraining_params_{ablate_condition}.txt', 'w') as f:
        num_params = 0
        for params in itertools.product(*param_grid.values()):
            listener_arch_idx, speaker_arch_idx = params
            f.write(f'LISTENER_ARCH="conv-ablate-{ablate_condition}-{listener_arch_idx}" SPEAKER_ARCH="splines-ablate-{ablate_condition}-{speaker_arch_idx}" PRETRAINED_LISTENERS=""\n')
            num_params += 1
        print("Number of parameter combinations for grid:", num_params)
    
    with open(f'sweep_params/sweep_phones_listener_pretrain_{ablate_condition}.txt', 'w') as f:
        num_params = 0
        for param in param_grid['listener_arch_indices']:
            f.write(f'LISTENER_ARCH="conv-ablate-{ablate_condition}-{param}"\n')
            num_params += 1
        print("Number of parameter combinations for listeners:", num_params)


pretrained_listeners = {    # Indexed by the number of epochs trained
    "250 epochs": { # Indexed by ablate condition
        "25": [
            "agents-cifar10b-woven-glade-2275-2000e-a-0",
            "agents-cifar10b-woven-glade-2275-2000e-b-1",
            "agents-cifar10b-woven-glade-2275-2000e-5000dp-2",
            "agents-cifar10b-woven-glade-2275-2000e-5000dp-c6fb",
            "agents-cifar10b-woven-glade-2275-2000e-5000dp-c6fb",
            "agents-cifar10b-woven-glade-2275-2000e-5000dp-c6fb",
            "agents-cifar10b-woven-glade-2275-2000e-5000dp-c6fb",
        ],
        "50": [
            "agents-cifar10b-woven-glade-2275-2000e-c-0",
            "agents-cifar10b-woven-glade-2275-2000e-d-1",
            "agents-cifar10b-woven-glade-2275-2000e-5000dp-c6fb",
            "agents-cifar10b-woven-glade-2275-2000e-5000dp-c6fb",
            "agents-cifar10b-woven-glade-2275-2000e-5000dp-c6fb",
            "agents-cifar10b-woven-glade-2275-2000e-5000dp-c6fb",
            "agents-cifar10b-woven-glade-2275-2000e-5000dp-c6fb",
        ]
    },
    "1000 epochs": {
        "25": [
            "agents-cifar10b-woven-glade-2275-2000e-e-0",
            "agents-cifar10b-woven-glade-2275-2000e-f-1",
            "agents-cifar10b-woven-glade-2275-2000e-5000dp-c6fb",
            "agents-cifar10b-woven-glade-2275-2000e-5000dp-c6fb",
            "agents-cifar10b-woven-glade-2275-2000e-5000dp-c6fb",
            "agents-cifar10b-woven-glade-2275-2000e-5000dp-c6fb",
            "agents-cifar10b-woven-glade-2275-2000e-5000dp-c6fb",
        ],
        "50": [
            "agents-cifar10b-woven-glade-2275-2000e-g-0",
            "agents-cifar10b-woven-glade-2275-2000e-h-1",
            "agents-cifar10b-woven-glade-2275-2000e-5000dp-c6fb",
            "agents-cifar10b-woven-glade-2275-2000e-5000dp-c6fb",
            "agents-cifar10b-woven-glade-2275-2000e-5000dp-c6fb",
            "agents-cifar10b-woven-glade-2275-2000e-5000dp-c6fb",
            "agents-cifar10b-woven-glade-2275-2000e-5000dp-c6fb",
        ]
    }
}

### These are with pretrained listeners
for epoch_condition in ("250 epochs", "1000 epochs"):
    for ablate_condition in ("50", "25"):
        with open(f'sweep_params/sweep_phones_params_pretrained_{ablate_condition}_{epoch_condition.replace(" ","_")}.txt', 'w') as f:
            num_params = 0
            for params in itertools.product(*param_grid.values()):
                listener_arch_idx, speaker_arch_idx = params
                f.write(f'LISTENER_ARCH="conv-ablate-{ablate_condition}-{listener_arch_idx}" SPEAKER_ARCH="splines-ablate-{ablate_condition}-{speaker_arch_idx}" PRETRAINED_LISTENERS="{pretrained_listeners[epoch_condition][ablate_condition][listener_arch_idx]}"\n')
                num_params += 1
            print("Number of parameter combinations for grid:", num_params)
