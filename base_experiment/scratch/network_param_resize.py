# Try to normalize some stuff
import flax
import jax
import jax.numpy as jnp
import sys
import pathlib

local_path = str(pathlib.Path().resolve().parent)
sys.path.append(local_path)

from quantized_agents import ActorCriticSpeakerRNNQuantized, SPEAKER_ARCH_RNN_QUANTIZATION_PARAMETERS

def rescale_params(params, target_norm, exclude_paths=None):
    """Rescale parameters, optionally excluding certain paths."""
    exclude_paths = exclude_paths or []
    
    # Flatten with paths
    flat_with_paths = flax.traverse_util.flatten_dict(params, sep='/')
    
    # Separate included and excluded params
    included = {k: v for k, v in flat_with_paths.items() 
                if not any(excl in k for excl in exclude_paths)}
    excluded = {k: v for k, v in flat_with_paths.items() 
                if any(excl in k for excl in exclude_paths)}
    
    # Calculate norm only from included params
    current_norm = jnp.sqrt(sum(jnp.sum(v**2) for v in included.values()))
    scale = target_norm / current_norm
    
    # Rescale only included params
    rescaled_included = {k: v * scale for k, v in included.items()}
    
    # Combine back
    all_params = {**rescaled_included, **excluded}
    return flax.traverse_util.unflatten_dict(all_params, sep='/')

def rescale_params_to(params, target_params, exclude_paths=None):
    """Rescale parameters, optionally excluding certain paths."""
    exclude_paths = exclude_paths or []
    
    # Flatten with paths
    flat_with_paths_A = flax.traverse_util.flatten_dict(params, sep='/')
    flat_with_paths_B = flax.traverse_util.flatten_dict(target_params, sep='/')
    
    # Separate included and excluded params
    included_A = {k: v for k, v in flat_with_paths_A.items() 
                if not any(excl in k for excl in exclude_paths)}
    excluded_A = {k: v for k, v in flat_with_paths_A.items() 
                if any(excl in k for excl in exclude_paths)}

    included_B = {k: v for k, v in flat_with_paths_B.items() 
                if not any(excl in k for excl in exclude_paths)}

    # Calculate target norm only from target params
    target_norm = jnp.sqrt(sum(jnp.sum(v**2) for v in included_B.values()))
    
    # Calculate norm only from included params
    current_norm = jnp.sqrt(sum(jnp.sum(v**2) for v in included_A.values()))
    scale = target_norm / current_norm
    
    # Rescale only included params
    rescaled_included = {k: v * scale for k, v in included_A.items()}
    
    # Combine back
    all_params = {**rescaled_included, **excluded_A}
    return flax.traverse_util.unflatten_dict(all_params, sep='/')

def l2_norm(params):
    return jnp.sqrt(sum(jnp.sum(p**2) for p in jax.tree_util.tree_leaves(params))) 

def show_layers(params):
    flat = flax.traverse_util.flatten_dict(params, sep='/')
    for path in sorted(flat.keys()):
        print(path, flat[path].shape)

config_small = {"SPEAKER_STDDEV": 1.0, "SPEAKER_STDDEV2": 0.4, "SPEAKER_SQUISH": 0.3}
config_large = {"SPEAKER_STDDEV": 1.0, "SPEAKER_STDDEV2": 0.4, "SPEAKER_SQUISH": 0.3}

config_small["SPEAKER_ARCH_RNN_QUANTIZATION_PARAMETERS"] = SPEAKER_ARCH_RNN_QUANTIZATION_PARAMETERS['splines-rnn-quantized-A03H-0']['SPEAKER_ARCH_RNN_QUANTIZATION_PARAMETERS']
config_large["SPEAKER_ARCH_RNN_QUANTIZATION_PARAMETERS"] = SPEAKER_ARCH_RNN_QUANTIZATION_PARAMETERS['splines-rnn-quantized-A03H-5']['SPEAKER_ARCH_RNN_QUANTIZATION_PARAMETERS']

small_speaker_network = ActorCriticSpeakerRNNQuantized(num_classes=11, num_splines=3, spline_action_dim=6, config=config_small)
large_speaker_network = ActorCriticSpeakerRNNQuantized(num_classes=11, num_splines=3, spline_action_dim=6, config=config_large)

rng = jax.random.PRNGKey(50)

rng, p_rng, d_rng, n_rng = jax.random.split(rng, 4)
init_x = jnp.zeros(
        (1,),
        dtype=jnp.int32
    )
params_small = small_speaker_network.init({'params': p_rng, 'dropout': d_rng, 'noise': n_rng}, init_x)
params_large = large_speaker_network.init({'params': p_rng, 'dropout': d_rng, 'noise': n_rng}, init_x)

small_norm = l2_norm(params_small)
large_norm = l2_norm(params_large)


# Usage
new_params_large = rescale_params(
    params_large, 
    small_norm,
    exclude_paths=['Dense_0', 'Dense_1', 'Dense_2', 'Dense_7', 'Dense_8', 'Dense_9', 'Embed_0', ]
)

new_params_large_s = rescale_params_to(
    params_large, 
    params_small,
    exclude_paths=['Dense_0', 'Dense_1', 'Dense_2', 'Dense_7', 'Dense_8', 'Dense_9', 'Embed_0', ]
)

print(small_norm)
print(large_norm)
print(l2_norm(new_params_large_s))
