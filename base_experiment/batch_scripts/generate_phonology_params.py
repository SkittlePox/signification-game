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
            "agents-cifar10b-helpful-glade-85-250e-5000dp-d001",
            "agents-cifar10b-wobbly-oath-86-250e-5000dp-5ff9",
            "agents-cifar10b-major-salad-87-250e-5000dp-6ac2",
            "agents-cifar10b-rare-snowflake-88-250e-5000dp-a0b6",
            "agents-cifar10b-usual-wind-89-250e-5000dp-70bc",
            "agents-cifar10b-visionary-firefly-90-250e-5000dp-6b65",
            "agents-cifar10b-robust-aardvark-91-250e-5000dp-94c0",
        ],
        "50": [
            "agents-cifar10b-hopeful-monkey-71-250e-5000dp-3b02",
            "agents-cifar10b-helpful-resonance-72-250e-5000dp-c454",
            "agents-cifar10b-driven-galaxy-73-250e-5000dp-40d2",
            "agents-cifar10b-usual-darkness-74-250e-5000dp-a58a",
            "agents-cifar10b-cool-glade-75-250e-5000dp-1c1e",
            "agents-cifar10b-fresh-snow-76-250e-5000dp-4607",
            "agents-cifar10b-azure-wave-77-250e-5000dp-2f7a",
        ]
    },
    "1000 epochs": {
        "25": [
            "agents-cifar10b-dashing-rain-79-1000e-5000dp-1479",
            "agents-cifar10b-stellar-violet-79-1000e-5000dp-0ce8",
            "agents-cifar10b-rosy-planet-82-1000e-5000dp-e4a8",
            "agents-cifar10b-skilled-mountain-81-1000e-5000dp-1c00",
            "agents-cifar10b-misunderstood-frog-78-1000e-5000dp-d4ae",
            "agents-cifar10b-balmy-serenity-83-1000e-5000dp-2822",
            "agents-cifar10b-expert-shape-84-1000e-5000dp-e344",
        ],
        "50": [
            "agents-cifar10b-stellar-tree-64-1000e-5000dp-0d05",
            "agents-cifar10b-absurd-bush-68-1000e-5000dp-2f0b",
            "agents-cifar10b-ruby-terrain-68-1000e-5000dp-4e61",
            "agents-cifar10b-devoted-lake-67-1000e-5000dp-fa85",
            "agents-cifar10b-rare-feather-65-1000e-5000dp-aade",
            "agents-cifar10b-fiery-dragon-70-1000e-5000dp-12bb",
            "agents-cifar10b-logical-disco-66-1000e-5000dp-7e6a",
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
