"""
Based on PureJaxRL Implementation of PPO
"""
import os
from functools import partial
import jax
import jax.experimental
import jax.flatten_util
import jax.numpy as jnp
import optax
from typing import Sequence, NamedTuple, Any, Dict, Tuple
import wandb
import hydra
import json
import cloudpickle
from flax.training import train_state, orbax_utils
import orbax.checkpoint
import pickle
from torchvision.utils import make_grid
from torchvision.datasets import MNIST
from omegaconf import OmegaConf
from simplified_signification_game import SimplifiedSignificationGame, State
from agents import *
import pathlib
import icon_probe
import time
from utils import get_anneal_schedule, get_train_freezing, speaker_penalty_whitesum_fn, speaker_penalty_curve_fn, center_obs, save_agents, make_grid_jnp


class TrainState(train_state.TrainState):
    key: jax.Array

class Transition(NamedTuple):
    speaker_action: jnp.ndarray
    speaker_reward: jnp.ndarray
    speaker_value: jnp.ndarray
    speaker_log_prob: jnp.ndarray
    speaker_obs: jnp.ndarray
    speaker_alive: jnp.ndarray
    listener_action: jnp.ndarray
    listener_reward: jnp.ndarray
    listener_value: jnp.ndarray
    listener_log_prob: jnp.ndarray
    listener_obs: jnp.ndarray
    listener_alive: jnp.ndarray
    channel_map: jnp.ndarray
    listener_obs_source: jnp.ndarray


class HalfTransition(NamedTuple):
    action: jnp.ndarray
    reward: jnp.ndarray
    value: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    alive: jnp.ndarray


def define_env(config):
    dataset_name = config["ENV_DATASET"]
    if dataset_name == 'mnist':        
        from utils import to_jax

        mnist_dataset = MNIST('/tmp/mnist/', download=True)
        images, labels = to_jax(mnist_dataset, num_datapoints=config["ENV_NUM_DATAPOINTS"])  # This should also be in ENV_KWARGS
        images = images.astype('float32') / 255.0

        # images are shape (5000, 28, 28)  dtype('float32') (pixels are between 0.0 and 1.0)
        # labels are shape (5000,)  dtype('int32') (labels are from 0 to 9)
        
        env = SimplifiedSignificationGame(**config["ENV_KWARGS"], dataset=(images, labels))
        return env

    elif dataset_name == 'cifar10':
        download_path = '/tmp/cifar10/'
        os.makedirs(download_path, exist_ok=True)
        
        # Check if dataset already exists
        dataset_files = os.listdir(download_path)
        if len(dataset_files) > 0:
            print(f"Dataset already exists in {download_path}.")
        else:
            dataset_link = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
            
            import requests
            from tqdm import tqdm

            response = requests.get(dataset_link, stream=True)
            total_size = int(response.headers.get('content-length', 0))
            block_size = 1024
            progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True)
            with open(os.path.join(download_path, 'cifar-10.tar.gz'), 'wb') as f:
                for data in response.iter_content(block_size):
                    f.write(data)
                    progress_bar.update(len(data))
            progress_bar.close()
            if total_size != 0 and progress_bar.n != total_size:
                print("Failed to download the dataset.")
            else:
                print("Dataset downloaded successfully.")
                import tarfile
                tar = tarfile.open(os.path.join(download_path, 'cifar-10.tar.gz'), 'r:gz')
                tar.extractall(download_path)
                tar.close()

        with open(download_path+'cifar-10-batches-py/data_batch_1', 'rb') as f:   # Each batch is 10,000 images. Probably don't need that many
            data_batch_1 = pickle.load(f, encoding='bytes')

        raw_images = data_batch_1[b'data'].astype('float32') / 255.0

        red_channel = raw_images[:, :1024].reshape(10000, 32, 32)
        green_channel = raw_images[:, 1024:2048].reshape(10000, 32, 32)
        blue_channel = raw_images[:, 2048:].reshape(10000, 32, 32)

        # Convert to grayscale using the weighted formula: 0.299*R + 0.587*G + 0.114*B
        images = jnp.array((0.299 * red_channel + 0.587 * green_channel + 0.114 * blue_channel), dtype=jnp.float32)[:config["ENV_NUM_DATAPOINTS"]]
        labels = jnp.array(data_batch_1[b'labels'], dtype=jnp.int32)[:config["ENV_NUM_DATAPOINTS"]]
            
        env = SimplifiedSignificationGame(**config["ENV_KWARGS"], dataset=(images, labels))
        return env
        
    elif dataset_name in ('cifar100', 'cifar15', 'cifar20'):    # This will also work for cifar 15 and 20
        download_path = '/tmp/cifar100/'
        os.makedirs(download_path, exist_ok=True)
        
        # Check if dataset already exists
        dataset_files = os.listdir(download_path)
        if len(dataset_files) > 0:
            print(f"Dataset already exists in {download_path}.")
        else:
            dataset_link = 'https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz'
            
            import requests
            from tqdm import tqdm

            response = requests.get(dataset_link, stream=True)
            total_size = int(response.headers.get('content-length', 0))
            block_size = 1024
            progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True)
            with open(os.path.join(download_path, 'cifar-100.tar.gz'), 'wb') as f:
                for data in response.iter_content(block_size):
                    f.write(data)
                    progress_bar.update(len(data))
            progress_bar.close()
            if total_size != 0 and progress_bar.n != total_size:
                print("Failed to download the dataset.")
            else:
                print("Dataset downloaded successfully.")
                import tarfile
                tar = tarfile.open(os.path.join(download_path, 'cifar-100.tar.gz'), 'r:gz')
                tar.extractall(download_path)
                tar.close()

        # with open(download_path+'cifar-100-python/meta', 'rb') as f:   # metadata with keys [b'fine_label_names', b'coarse_label_names']
        #     meta_data = pickle.load(f, encoding='bytes')
        # fine_labels = {l: i for i, l in enumerate(meta_data[b'fine_label_names'])}

        acceptable_label_names = config["ENV_DATASET_CATEGORIES"]
        acceptable_labels = jnp.array(list(acceptable_label_names.values()))

        with open(download_path+'cifar-100-python/train', 'rb') as f:   # 50,000 images
            train_data = pickle.load(f, encoding='bytes')
        
        # with open(download_path+'cifar-100-python/test', 'rb') as f:
        #     test_data = pickle.load(f, encoding='bytes')

        raw_images = train_data[b'data'].astype('float32') / 255.0

        red_channel = raw_images[:, :1024].reshape(50000, 32, 32)
        green_channel = raw_images[:, 1024:2048].reshape(50000, 32, 32)
        blue_channel = raw_images[:, 2048:].reshape(50000, 32, 32)

        # Convert to grayscale using the weighted formula: 0.299*R + 0.587*G + 0.114*B
        all_images = jnp.array((0.299 * red_channel + 0.587 * green_channel + 0.114 * blue_channel), dtype=jnp.float32)
        all_labels = jnp.array(train_data[b'fine_labels'], dtype=jnp.int32)

        mask = jnp.isin(all_labels, acceptable_labels)
        
        filtered_images = all_images[mask]
        filtered_labels = all_labels[mask]

        new_labels = jnp.arange(len(acceptable_labels))
        mapped_indices = jnp.searchsorted(acceptable_labels, filtered_labels)
        # Remap using new_labels
        converted_labels = new_labels[mapped_indices]
            
        env = SimplifiedSignificationGame(**config["ENV_KWARGS"], dataset=(filtered_images[:config["ENV_NUM_DATAPOINTS"]], converted_labels[:config["ENV_NUM_DATAPOINTS"]]))
        return env

    elif dataset_name == "veg":
        from utils import load_images_to_array
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()

        download_path = '/tmp/veggies/'
        os.makedirs(download_path, exist_ok=True)

        # Check if dataset already exists
        dataset_files = os.listdir(download_path)
        if len(dataset_files) > 0:
            print(f"Dataset already exists in {download_path}.")
        else:
            dataset = 'misrakahmed/vegetable-image-dataset'
            api.dataset_download_files(dataset, path=download_path, unzip=True)
        
        images, labels = load_images_to_array(directory=download_path+"Vegetable Images/train/", categories=config["ENV_DATASET_CATEGORIES"], target_size=(config["ENV_KWARGS"]["image_dim"], config["ENV_KWARGS"]["image_dim"]), num_datapoints=config["ENV_NUM_DATAPOINTS"])
        images /= 255.0

        env = SimplifiedSignificationGame(**config["ENV_KWARGS"], dataset=(images, labels))
        return env

def initialize_listener(env, rng, config, i):
    if config["LISTENER_ARCH"] == 'conv':
        listener_network = ActorCriticListenerConv(action_dim=config["ENV_KWARGS"]["num_classes"], image_dim=config["ENV_KWARGS"]["image_dim"], config=config)
    elif config["LISTENER_ARCH"] == 'conv-boost':
        listener_network = ActorCriticListenerBoost(action_dim=config["ENV_KWARGS"]["num_classes"], image_dim=config["ENV_KWARGS"]["image_dim"], config=config)
    elif config["LISTENER_ARCH"] == 'conv-reduced':
        listener_network = ActorCriticListenerConvReduced(action_dim=config["ENV_KWARGS"]["num_classes"], image_dim=config["ENV_KWARGS"]["image_dim"], config=config)
    elif config["LISTENER_ARCH"] == 'conv-sigmoid':
        listener_network = ActorCriticListenerConvSigmoid(action_dim=config["ENV_KWARGS"]["num_classes"], image_dim=config["ENV_KWARGS"]["image_dim"], config=config)
    elif config["LISTENER_ARCH"] == 'conv-small':
        listener_network = ActorCriticListenerConvSmall(action_dim=config["ENV_KWARGS"]["num_classes"], image_dim=config["ENV_KWARGS"]["image_dim"], config=config)
    elif config["LISTENER_ARCH"] == 'dense':
        listener_network = ActorCriticListenerDense(action_dim=config["ENV_KWARGS"]["num_classes"], image_dim=config["ENV_KWARGS"]["image_dim"], config=config)
    elif config["LISTENER_ARCH"] == 'dense-batchnorm':
        listener_network = ActorCriticListenerDenseBatchnorm(action_dim=config["ENV_KWARGS"]["num_classes"], image_dim=config["ENV_KWARGS"]["image_dim"], config=config)
    
    rng, p_rng, d_rng, n_rng = jax.random.split(rng, 4)
    init_x = jnp.zeros(
            (config["ENV_KWARGS"]["image_dim"]**2,)
        )
    network_params = listener_network.init({'params': p_rng, 'dropout': d_rng, 'noise': n_rng}, init_x)
    
    lr_func = get_anneal_schedule(config["LISTENER_LR_SCHEDULE"], config["NUM_MINIBATCHES_LISTENER"])
    
    if config["LISTENER_OPTIMIZER"] == 'adam':
        tx = optax.chain(
            optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
            optax.adam(learning_rate=lr_func, b1=config["OPTIMIZER_LISTENER_B1"], b2=config["OPTIMIZER_LISTENER_B2"], eps=1e-5),
        )
    elif config["LISTENER_OPTIMIZER"] == 'adamw':
        tx = optax.chain(
            optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
            optax.adamw(learning_rate=lr_func, b1=config["OPTIMIZER_LISTENER_B1"], b2=config["OPTIMIZER_LISTENER_B2"], eps=1e-5),
        )
    elif config["LISTENER_OPTIMIZER"] == 'adamaxw':
        tx = optax.chain(
            optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
            optax.adamaxw(learning_rate=lr_func, b1=config["OPTIMIZER_LISTENER_B1"], b2=config["OPTIMIZER_LISTENER_B2"], eps=1e-5),
        )
    elif config["LISTENER_OPTIMIZER"] == 'adamax':
        tx = optax.chain(
            optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
            optax.adamax(learning_rate=lr_func, b1=config["OPTIMIZER_LISTENER_B1"], b2=config["OPTIMIZER_LISTENER_B2"], eps=1e-5),
        )

    train_state = TrainState.create(
        apply_fn=listener_network.apply,
        params=network_params,
        key=rng,
        tx=tx,
    )
    
    if config["PRETRAINED_LISTENERS"] != "":  # Just assuming that all listeners should be loaded
        # We modify the trainstate by putting what we want in it.

        local_path = str(pathlib.Path().resolve())
        model_path_str = "/base_experiment/models/" if config["DEBUGGER"] else "/models/"
        checkpoint_name = local_path+model_path_str+config["PRETRAINED_LISTENERS"]+f'/listener_{i}.agent'
        
        empty_checkpoint = {'model': train_state}
        orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        raw_restored = orbax_checkpointer.restore(checkpoint_name, item=empty_checkpoint)

        # new_tx = optax.chain(
        #     optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
        #     optax.adam(learning_rate=lr_func, b1=config["OPTIMIZER_LISTENER_B1"], b2=config["OPTIMIZER_LISTENER_B2"], eps=1e-5),
        # )

        # This LR function is for logging purposes only, and it could be wrong! need to change the train state below
        train_state = raw_restored['model']# .replace(tx=new_tx, key=rng)

        # I can reset the counts of the opt_states
        # train_state.opt_state[1][0] = train_state.opt_state[1][0]._replace(count=jnp.array(0))

        listener_network = None

    return listener_network, train_state, lr_func

def initialize_speaker(env, rng, config, i):
    # Passing num_classes = env_kwargs.num_classes + 1 so we can use the additional channel to sample from the general space of signals
    if config["SPEAKER_ARCH"] == 'full_image':
        speaker_network = ActorCriticSpeakerFullImage(latent_dim=config["SPEAKER_LATENT_DIM"], num_classes=config["ENV_KWARGS"]["num_classes"]+1, image_dim=config["ENV_KWARGS"]["image_dim"], config=config)
    elif config["SPEAKER_ARCH"] == 'full_image_setvariance':
        speaker_network = ActorCriticSpeakerFullImageSetVariance(latent_dim=config["SPEAKER_LATENT_DIM"], num_classes=config["ENV_KWARGS"]["num_classes"]+1, image_dim=config["ENV_KWARGS"]["image_dim"], config=config)
    elif config["SPEAKER_ARCH"] == 'gauss_splat':
        speaker_network = ActorCriticSpeakerGaussSplat(latent_dim=config["SPEAKER_LATENT_DIM"], num_classes=config["ENV_KWARGS"]["num_classes"]+1, action_dim=config["ENV_KWARGS"]["speaker_action_dim"], config=config)
    elif config["SPEAKER_ARCH"] == 'gauss_splatcovar':
        speaker_network = ActorCriticSpeakerGaussSplatCov(latent_dim=config["SPEAKER_LATENT_DIM"], num_classes=config["ENV_KWARGS"]["num_classes"]+1, action_dim=config["ENV_KWARGS"]["speaker_action_dim"], config=config)
    elif config["SPEAKER_ARCH"] == 'gauss_splatchol':
        speaker_network = ActorCriticSpeakerGaussSplatChol(latent_dim=config["SPEAKER_LATENT_DIM"], num_classes=config["ENV_KWARGS"]["num_classes"]+1, action_dim=config["ENV_KWARGS"]["speaker_action_dim"], config=config)
    elif config["SPEAKER_ARCH"] == 'splines':
        speaker_network = ActorCriticSpeakerSplines(latent_dim=config["SPEAKER_LATENT_DIM"], num_classes=config["ENV_KWARGS"]["num_classes"]+1, action_dim=config["ENV_KWARGS"]["speaker_action_dim"], config=config)
    elif config["SPEAKER_ARCH"] == 'splinesnoise':
        speaker_network = ActorCriticSpeakerSplinesNoise(latent_dim=config["SPEAKER_LATENT_DIM"], num_classes=config["ENV_KWARGS"]["num_classes"]+1, action_dim=config["ENV_KWARGS"]["speaker_action_dim"], noise_dim=config["SPEAKER_NOISE_LATENT_DIM"], noise_stddev=config["SPEAKER_NOISE_LATENT_STDDEV"], config=config)

    rng, p_rng, d_rng, n_rng = jax.random.split(rng, 4)
    init_x = jnp.zeros(
            (1,),
            dtype=jnp.int32
        )
    network_params = speaker_network.init({'params': p_rng, 'dropout': d_rng, 'noise': n_rng}, init_x)

    lr_func = get_anneal_schedule(config["SPEAKER_LR_SCHEDULE"], config["NUM_MINIBATCHES_SPEAKER"])
    tx = optax.chain(
        optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
        optax.adam(learning_rate=lr_func, b1=config["OPTIMIZER_SPEAKER_B1"], b2=config["OPTIMIZER_SPEAKER_B2"], eps=1e-5),
    )
    
    train_state = TrainState.create(
        apply_fn=speaker_network.apply,
        params=network_params,
        key=rng,
        tx=tx,
    )
    if config["PRETRAINED_SPEAKERS"] != "":  # Just assuming that all speakers should be loaded
        # We modify the trainstate by putting what we want in it.

        local_path = str(pathlib.Path().resolve())
        model_path_str = "/base_experiment/models/" if config["DEBUGGER"] else "/models/"
        checkpoint_name = local_path+model_path_str+config["PRETRAINED_SPEAKERS"]+f'/speaker_{i}.agent'
        
        empty_checkpoint = {'model': train_state}
        orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        raw_restored = orbax_checkpointer.restore(checkpoint_name, item=empty_checkpoint)

        # new_tx = optax.chain(
        #     optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
        #     optax.adam(learning_rate=lr_func, b1=config["OPTIMIZER_LISTENER_B1"], b2=config["OPTIMIZER_LISTENER_B2"], eps=1e-5),
        # )

        train_state = raw_restored['model']# .replace(tx=new_tx, key=rng)
        # This is failing but I don't know why. Probably something with cloudpickle but I'm not sure. Maybe need to reconstruct the trainstate by hand.

        listener_network = None
    

    return speaker_network, train_state, lr_func

def execute_individual_listener(__rng, _listener_apply_fn, _listener_params_i, _listener_obs_i):
    __rng, dropout_key, noise_key = jax.random.split(__rng, 3)
    _listener_obs_i = _listener_obs_i.ravel()
    policy, value = _listener_apply_fn(_listener_params_i, _listener_obs_i, rngs={'dropout': dropout_key, 'noise': noise_key})
    action = policy.sample(seed=__rng)
    log_prob = policy.log_prob(action)
    return action, log_prob, value

def execute_individual_speaker(__rng, _speaker_apply_fn, _speaker_params_i, _speaker_obs_i):
    __rng, dropout_key, noise_key = jax.random.split(__rng, 3)
    _speaker_obs_i = _speaker_obs_i.ravel()
    policy, value = _speaker_apply_fn(_speaker_params_i, _speaker_obs_i, rngs={'dropout': dropout_key, 'noise': noise_key})
    action = policy.sample(seed=__rng)
    log_prob = policy.log_prob(action)
    return jnp.clip(action, a_min=0.0, a_max=1.0), log_prob, value

def execute_tom_listener(__rng, _speaker_apply_fn, _speaker_params_i, _listener_apply_fn, _listener_params_i, _listener_obs_i, speaker_action_transform_fn, listener_n_samples=50, num_classes=10, speaker_action_dim=12, listener_pr_weight=1.5):  # obs is an image
    # P(r_i|s) p= P(s|r_i)P(r_i)P_R(r_i)
    # P_R(r_i) = 1 / int_S f_i(s)p(s) ds + epsilon
    #   or in logits l_i terms: 1 / int_S exp(l_i(s)) p(s) ds + epsilon
    # P(s|r_i) p= f_i(s) / sum f_j(s) for j != i

    __rng, listener_dropout_key, listener_noise_key = jax.random.split(__rng, 3)
    __rng, speaker_dropout_key, speaker_noise_key = jax.random.split(__rng, 3)
    __rng, pi_sample_key = jax.random.split(__rng)

    ###### Find the referent for which P(r|s) is the highest

    #### Calculate P(s|r_i) for each possible referent

    listener_assessments, values = _listener_apply_fn(_listener_params_i, _listener_obs_i, rngs={'dropout': listener_dropout_key, 'noise': listener_noise_key}) # This is a categorical distribution.
    
    def calc_psr(logits, index):
        numerator = logits[index]
        denominator = jax.nn.logsumexp(logits.at[index].set(-jnp.inf), axis=0)  # Set index to neg inf to remove it from denominator sum
        return numerator - denominator

    vmap_calc_psr = jax.vmap(calc_psr, in_axes=(None, 0))
    log_psrs = vmap_calc_psr(listener_assessments.logits[0], jnp.arange(num_classes))

    #### Calculate P_R(r_i) for each referent

    # Sample signal space n times    
    signal_distribution = _speaker_apply_fn(_speaker_params_i, jnp.array(jnp.arange(num_classes), dtype=jnp.int32), rngs={'dropout': speaker_dropout_key, 'noise': speaker_noise_key})[0]    # This is a distrax distribution. Index 1 is values. Using num_classes for fresh speaker samplings
    signal_param_samples = signal_distribution.sample(seed=__rng, sample_shape=(listener_n_samples)).reshape((-1, speaker_action_dim))    # This is shaped (n_samples*num_classes, speaker_action_dim). The reshape is to merge samples from all generators together
    signal_param_samples = jnp.clip(signal_param_samples, a_min=0.0, a_max=1.0)

    # Generate actual images
    signal_samples = speaker_action_transform_fn(signal_param_samples)    # This is shaped (n_samples, image_dim, image_dim)

    # Assess them using the listener
    listener_assessments = _listener_apply_fn(_listener_params_i, signal_samples, rngs={'dropout': listener_dropout_key, 'noise': listener_noise_key})[0] # This is a categorical distribution. Index 1 is values
    # This has nearly everything we need. At this point I could take the logits, the probs, or the logprobs and do the calculation
    
    # Sum exponentiated logits using exp(logsumexp) vertically?
    log_pRs = -(jax.nn.logsumexp(listener_assessments.logits, axis=0) - jnp.log(listener_n_samples))    # These should technically be multiplied by p(s) before summing, but assuming random uniform dist I'm just dividing by n_samples
    log_pRs *= listener_pr_weight        # NOTE: This will definitely need to be tuned. Between 0.1 and 2.0 I'm guessing. Maybe need a sweep later.

    #### Calculate P(r_i|s)
    log_prss = log_psrs + log_pRs - jnp.log(num_classes) # Assuming uniform random referent distribution means I can divide by num_classes. Using log rules
    log_prss -= jax.nn.logsumexp(log_prss)
    prss = jnp.exp(log_prss)

    pictogram_pi = distrax.Categorical(probs=prss)
    
    pictogram_action = pictogram_pi.sample(seed=pi_sample_key)
    log_prob = pictogram_pi.log_prob(pictogram_action)

    # What should be the value here?? Perhaps I should pass the observation directly through the listener agent again and extract that value? I'm not sure.
    return pictogram_action.reshape(-1,), log_prob.reshape(-1,), values.reshape(-1,)

def execute_tom_speaker(__rng, _speaker_apply_fn, _speaker_params_i, _listener_apply_fn, _listener_params_i, _speaker_obs_i, speaker_action_transform_fn, speaker_n_search=50, num_classes=10, speaker_action_dim=12):
    # P(s|r_i) p= f_i(s) / sum f_j(s) for j != i       here, f_i(s) represents unnormalized probabilites.
    # In terms of logits exp(l_i(s)) = f_i(s)
    # P(s|r_i) p= exp( l_i(s) - log sum exp(l_j(s)) for j != i )

    __rng, listener_dropout_key, listener_noise_key = jax.random.split(__rng, 3)
    __rng, speaker_dropout_key, speaker_noise_key = jax.random.split(__rng, 3)
    __rng, numer_key, denom_key = jax.random.split(__rng, 3)
    __rng, pi_sample_key = jax.random.split(__rng)

    ######### Search for the signal with the highest P(r|s)

    ###### Generate candidate stimuli

    # Sample lots of possible signals.
    search_signal_gut_policy, values = _speaker_apply_fn(_speaker_params_i, jnp.array([_speaker_obs_i], dtype=jnp.int32), rngs={'dropout': speaker_dropout_key, 'noise': speaker_noise_key})    # This is a distrax distribution. Index 1 is values. Using speaker index _speaker_obs_i, for generating pictograms of that class
    search_signal_param_samples = search_signal_gut_policy.sample(seed=numer_key, sample_shape=(speaker_action_dim, speaker_n_search))[0]    # This is shaped (speaker_n_search, 1, speaker_action_size)
    search_signal_param_samples = jnp.clip(search_signal_param_samples, a_min=0.0, a_max=1.0)
    
    # Generate actual images
    search_signal_samples = speaker_action_transform_fn(search_signal_param_samples)
    
    # Need to iterate over search space and find the highest numerator/denominator??
    listener_logits = _listener_apply_fn(_listener_params_i, search_signal_samples, rngs={'dropout': speaker_dropout_key, 'noise': speaker_noise_key})[0].logits # This is shaped (n_search, num_classes)

    # Isolate obs referent index
    # Numerators should be shape [n_search,]
    numerators = listener_logits[:, _speaker_obs_i]
    
    # Set the logits for obs to -jnp.inf so they don't show up in the denominator calculation
    denominators = jax.nn.logsumexp(listener_logits.at[:, _speaker_obs_i].set(-jnp.inf), axis=1)

    psr = jnp.exp(numerators-denominators)

    ###### Select signal with the highest P(s|r)
    maxindex = jnp.argmax(psr)

    maxaction = search_signal_param_samples[maxindex]
    
    log_prob = search_signal_gut_policy.log_prob(maxaction)

    return maxaction, log_prob.reshape(-1,), values[maxindex].reshape(-1,)

def get_speaker_examples(rng, speaker_apply_fn, speaker_params, speaker_action_transform, config):
    speaker_obs = jnp.tile(jnp.arange(config["ENV_KWARGS"]["num_classes"]), config["SPEAKER_EXAMPLE_NUM"])
    speaker_rngs = jax.random.split(rng, len(speaker_obs))
    center_listener_obs = config["ENV_KWARGS"]["center_listener_obs"]
    sp_action_dim = config["ENV_KWARGS"]["speaker_action_dim"]
    num_speakers = config["ENV_KWARGS"]["num_speakers"]
    
    def get_speaker_outputs(speaker_params_i):
        vmap_execute_speaker_test = jax.vmap(execute_individual_speaker, in_axes=(0, None, None, 0))
        speaker_actions = vmap_execute_speaker_test(speaker_rngs, speaker_apply_fn, speaker_params_i, speaker_obs)[0]   # Indices 1 and 2 are for logprobs and values. 0 
        return speaker_actions.reshape(-1, sp_action_dim)

    vmap_get_speaker_outputs = jax.vmap(get_speaker_outputs, in_axes=(0))
    speaker_actions = vmap_get_speaker_outputs(speaker_params).reshape((-1, sp_action_dim))
    speaker_images = speaker_action_transform(speaker_actions)
    
    if center_listener_obs:
        speaker_images = center_obs(speaker_images)
    return speaker_images

def get_tom_speaker_examples(rng, listener_apply_fn, listener_params, speaker_apply_fn, speaker_params, speaker_action_transform, config):
    env_kwargs = config["ENV_KWARGS"]
    speaker_obs = jnp.tile(jnp.arange(env_kwargs["num_classes"]), config["SPEAKER_EXAMPLE_NUM"])
    speaker_rngs = jax.random.split(rng, len(speaker_obs))
    center_listener_obs = env_kwargs["center_listener_obs"]
    sp_action_dim = env_kwargs["speaker_action_dim"]
    num_speakers = env_kwargs["num_speakers"]

    def get_speaker_outputs(speaker_params_i, listener_params_i):
        vmap_execute_speaker_tom_test = jax.vmap(execute_tom_speaker, in_axes=(0, None, None, None, None, 0, None, None, None, None))
        speaker_actions = vmap_execute_speaker_tom_test(speaker_rngs, speaker_apply_fn, speaker_params_i, listener_apply_fn, listener_params_i, speaker_obs, speaker_action_transform, config["SPEAKER_N_SEARCH"], env_kwargs["num_classes"], sp_action_dim)[0]   # Indices 1 and 2 are for logprobs and values. 0 
        return speaker_actions.reshape(-1, sp_action_dim)
        
    vmap_get_speaker_outputs = jax.vmap(get_speaker_outputs, in_axes=(0, 0))
    speaker_actions = vmap_get_speaker_outputs(speaker_params, listener_params).reshape((-1, sp_action_dim))
    speaker_images = speaker_action_transform(speaker_actions)
    
    if center_listener_obs:
        speaker_images = center_obs(speaker_images)
    return speaker_images

def calculate_gae_listeners(trans_batch, last_val, gamma, gae_lambda):
    def _get_advantages(gae_and_next_value, transition):
        gae, next_value = gae_and_next_value
        alive, value, reward = (
            transition.listener_alive,
            transition.listener_value,
            transition.listener_reward,
        )
        delta = reward + gamma * next_value * alive - value
        gae = delta + gamma * gae_lambda * alive * gae
        gae = gae * alive
        return (gae, value), gae

    _, advantages = jax.lax.scan(
        _get_advantages,
        (jnp.zeros_like(last_val), last_val),
        trans_batch,
        reverse=True,
        unroll=16,
    )
    return advantages, advantages + trans_batch.listener_value * trans_batch.listener_alive

def calculate_gae_speakers(trans_batch, last_val, gamma, gae_lambda):
    def _get_advantages(gae_and_next_value, transition):
        gae, next_value = gae_and_next_value
        alive, value, reward = (
            transition.speaker_alive,
            transition.speaker_value,
            transition.speaker_reward,
        )
        delta = reward + gamma * next_value * alive - value
        gae = delta + gamma * gae_lambda * alive * gae
        gae = gae * alive
        return (gae, value), gae

    _, advantages = jax.lax.scan(
        _get_advantages,
        (jnp.zeros_like(last_val), last_val),
        trans_batch,
        reverse=True,
        unroll=16,
    )
    return advantages, advantages + trans_batch.speaker_value * trans_batch.speaker_alive

def env_step(runner_state, speaker_apply_fn, listener_apply_fn, env, config):    # This function is passed to jax.lax.scan, which means it cannot have any pythonic control flow (e.g., no "if" statements, "while" loops, etc.)
    """This function literally is just for collecting rollouts, which involves applying the joint policy to the env and stepping forward."""
    batched_speaker_params, batched_listener_params, log_env_state, obs, rng = runner_state
    speaker_obs, listener_obs = obs

    env_kwargs = config["ENV_KWARGS"]

    speaker_obs = speaker_obs.ravel()
    listener_obs = listener_obs.reshape((listener_obs.shape[0]*listener_obs.shape[1], listener_obs.shape[2]*listener_obs.shape[3]))
    listener_obs_source = log_env_state.env_state.listener_obs_source.reshape(-1, 1)
    global_agent_inferential_mode = log_env_state.env_state.agent_inferential_mode[0]
    speaker_action_transform = env._env.speaker_action_transform

    ## set aside parameters for tom agents
    num_classes = env_kwargs["num_classes"]
    
    listener_n_samples = config["LISTENER_N_SAMPLES"]
    listener_pr_weight = config["LISTENER_PR_WEIGHT"]
    speaker_n_search = config["SPEAKER_N_SEARCH"]

    speaker_action_dim = env_kwargs["speaker_action_dim"]

    ##### COLLECT ACTIONS FROM AGENTS
    rng, l_rng, r_rng = jax.random.split(rng, 3)
    listener_rngs = jax.random.split(l_rng, len(listener_obs))
    speaker_rngs = jax.random.split(r_rng, len(speaker_obs))
    
    ## COLLECT LISTENER ACTIONS
    # Collect naive actions
    naive_listener_actions, naive_listener_log_probs, naive_listener_values = jax.vmap(execute_individual_listener, in_axes=(0, None, 0, 0))(listener_rngs, listener_apply_fn, batched_listener_params, listener_obs)
    # Collect tom actions
    def calc_tom_listener_actions():
        full_tom_listener_actions, full_tom_listener_log_probs, full_tom_listener_values = jax.vmap(execute_tom_listener, in_axes=(0, None, 0, None, 0, 0, None, None, None, None, None))(listener_rngs, speaker_apply_fn, batched_speaker_params, listener_apply_fn, batched_listener_params, listener_obs, speaker_action_transform, listener_n_samples, num_classes, speaker_action_dim, listener_pr_weight)
        tom_listener_actions = jax.lax.select(listener_obs_source == 1, full_tom_listener_actions, naive_listener_actions)
        tom_listener_log_probs = jax.lax.select(listener_obs_source == 1, full_tom_listener_log_probs, naive_listener_log_probs)
        tom_listener_values = jax.lax.select(listener_obs_source == 1, full_tom_listener_values, naive_listener_values)
        return tom_listener_actions, tom_listener_log_probs, tom_listener_values
    # Choose the right ones
    listener_actions, listener_log_probs, listener_values = jax.lax.cond(
        global_agent_inferential_mode == 0,
        lambda _: (naive_listener_actions, naive_listener_log_probs, naive_listener_values),
        lambda _: calc_tom_listener_actions(),
        None)
    listener_actions = listener_actions.reshape((1, -1))
    listener_log_probs = listener_log_probs.reshape((1, -1))
    listener_values = listener_values.reshape((1, -1))

    ## COLLECT SPEAKER ACTIONS
    # No need to save computation, since speakers don't change their speaking based on their listener, unlike listeners.
    speaker_actions, speaker_log_probs, speaker_values = jax.lax.cond(
        global_agent_inferential_mode == 0,
        lambda _: jax.vmap(execute_individual_speaker, in_axes=(0, None, 0, 0))(speaker_rngs, speaker_apply_fn, batched_speaker_params, speaker_obs),
        lambda _: jax.vmap(execute_tom_speaker, in_axes=(0, None, 0, None, 0, 0, None, None, None, None))(listener_rngs, speaker_apply_fn, batched_speaker_params, listener_apply_fn, batched_listener_params, speaker_obs, speaker_action_transform, speaker_n_search, num_classes, speaker_action_dim),
        None)
    speaker_actions = speaker_actions.reshape((1, -1, speaker_action_dim))
    speaker_log_probs = speaker_log_probs.reshape((1, -1))
    speaker_values = speaker_values.reshape((1, -1))

    ##### STEP ENV
    next_rng, rng = jax.random.split(rng)
    rng_step = jax.random.split(rng, 1)
    new_obs, env_state, rewards, alives, info = env.step(rng_step, log_env_state, (speaker_actions, listener_actions, listener_log_probs)) # Passing speaker actions, listener actions, AND listener log probs

    speaker_reward, listener_reward = rewards
    speaker_alive, listener_alive = alives

    speaker_obs = speaker_obs.reshape((1, -1))
    listener_obs = listener_obs.reshape((1, env_kwargs["num_listeners"], env_kwargs["image_dim"], env_kwargs["image_dim"]))

    channel_map = log_env_state.env_state.channel_map
    listener_obs_source = log_env_state.env_state.listener_obs_source

    transition = Transition(
        speaker_actions,
        speaker_reward,
        speaker_values,
        speaker_log_probs,
        speaker_obs,
        speaker_alive,
        listener_actions,
        listener_reward,
        listener_values,
        listener_log_probs,
        listener_obs,
        listener_alive,
        channel_map,
        listener_obs_source
    )

    runner_state = (batched_speaker_params, batched_listener_params, env_state, new_obs, next_rng)
    
    return runner_state, transition

def update_minibatch_listener(runner_state, listener_apply_fn, listener_optimizer_tx, clip_eps, l2_reg_coef_listener, vf_coef, ent_coef_listener):
    __rng, trans_batch_i, advantages_i, targets_i, listener_params_i, listener_opt_state_i = runner_state

    def _loss_fn(params, _obs, _actions, values, log_probs, advantages, targets, alive):
        # COLLECT ACTIONS AND LOG_PROBS FOR TRAJ ACTIONS
        dropout_key, noise_key = jax.random.split(__rng)
        _i_policy, _i_value = listener_apply_fn(params, _obs, rngs={'dropout': dropout_key, 'noise': noise_key})
        # _i_log_prob = _i_policy.log_prob(_actions) 
        # _i_log_prob = jnp.maximum(_i_log_prob, jnp.ones_like(_i_log_prob) * -1e8)
        _i_log_prob = jnp.clip(_i_policy.log_prob(_actions), -50.0, 1.0)
        log_probs = jnp.clip(log_probs, -50.0, 1.0)
        
        # CALCULATE VALUE LOSS
        values = jnp.clip(values, -1e3, 1e3)
        _i_value = jnp.clip(_i_value, -1e3, 1e3)
        targets = jnp.clip(targets, -1e3, 1e3)
        value_pred_clipped = values + (
                _i_value - values
        ).clip(-clip_eps, clip_eps)

        value_losses = jnp.square(values - targets) * alive
        value_losses_clipped = jnp.square(value_pred_clipped - targets) * alive
        value_loss = (0.5 * jnp.maximum(value_losses, value_losses_clipped).sum() / (alive.sum() + 1e-8))

        # CALCULATE ACTOR LOSS
        ratio = jnp.exp(_i_log_prob - log_probs)
        gae_for_i = jnp.clip((advantages - advantages.mean()) / (advantages.std() + 1e-8), -10, 10)
        loss_actor1 = ratio * gae_for_i * alive
        loss_actor2 = (
                jnp.clip(
                    ratio,
                    1.0 - clip_eps,
                    1.0 + clip_eps,
                )
                * gae_for_i * alive
        )
        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
        loss_actor = loss_actor.sum() / (alive.sum() + 1e-8)
        entropy = (_i_policy.entropy() * alive).sum() / (alive.sum() + 1e-8)

        # Calculate L2 regularization
        l2_mag = jnp.linalg.norm(jax.flatten_util.ravel_pytree(params)[0])
        l2_penalty = l2_reg_coef_listener * l2_mag

        total_loss = (
                loss_actor
                + vf_coef * value_loss
                - ent_coef_listener * entropy
                + l2_penalty
        )
        return total_loss, (value_loss, loss_actor, entropy)
 
    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True, allow_int=False)

    total_loss, grads = grad_fn(
        listener_params_i,    # params don't change across minibatches, they are for the same agent.
        trans_batch_i.obs,
        trans_batch_i.action, 
        trans_batch_i.value, 
        trans_batch_i.log_prob,
        advantages_i, 
        targets_i,
        trans_batch_i.alive
    )
    
    updates, new_opt_state = listener_optimizer_tx.update(grads, listener_opt_state_i)
    new_params = optax.apply_updates(listener_params_i, updates)

    # (new_params, new_opt_state)
    runner_state = (new_params, new_opt_state)

    return runner_state, total_loss

def update_minibatch_speaker(runner_state, speaker_apply_fn, speaker_optimizer_tx, clip_eps, l2_reg_coef_speaker, vf_coef, ent_coef_speaker):
    __rng, trans_batch_i, advantages_i, targets_i, speaker_params_i, speaker_opt_state_i = runner_state

    def _loss_fn(params, _obs, _actions, values, log_probs, advantages, targets, alive):
        # COLLECT ACTIONS AND LOG_PROBS FOR TRAJ ACTIONS
        dropout_key, noise_key = jax.random.split(__rng)
        _i_policy, _i_value = speaker_apply_fn(params, _obs, rngs={'dropout': dropout_key, 'noise': noise_key})
        # _i_log_prob = jnp.sum(_i_policy.log_prob(_actions), axis=1) # Sum log-probs for individual pixels to get log-probs of whole image
        _i_log_prob = jnp.clip(_i_policy.log_prob(_actions), -1e8, 1e8)
        log_probs = jnp.clip(log_probs, -1e8, 1e8)

        # CALCULATE VALUE LOSS
        values = jnp.clip(values, -1e3, 1e3)
        _i_value = jnp.clip(_i_value, -1e3, 1e3)
        targets = jnp.clip(targets, -1e3, 1e3)
        value_pred_clipped = values + (
                _i_value - values
        ).clip(-clip_eps, clip_eps)

        value_losses = jnp.square(values - targets) * alive
        value_losses_clipped = jnp.square(value_pred_clipped - targets) * alive
        value_loss = (0.5 * jnp.maximum(value_losses, value_losses_clipped).sum() / (alive.sum() + 1e-8))

        # CALCULATE ACTOR LOSS
        ratio = jnp.exp(_i_log_prob - log_probs)
        gae_for_i = jnp.clip((advantages - advantages.mean()) / (advantages.std() + 1e-8), -10, 10)
        loss_actor1 = ratio * gae_for_i * alive
        loss_actor2 = (
                jnp.clip(
                    ratio,
                    1.0 - clip_eps,
                    1.0 + clip_eps,
                )
                * gae_for_i * alive
        )
        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
        loss_actor = loss_actor.sum() / (alive.sum() + 1e-8)
        entropy = (_i_policy.entropy() * alive).sum() / (alive.sum() + 1e-8)

        # Calculate L2 regularization
        l2_mag = jnp.linalg.norm(jax.flatten_util.ravel_pytree(params)[0])
        l2_penalty = l2_reg_coef_speaker * l2_mag

        total_loss = (
                loss_actor
                + vf_coef * value_loss
                - ent_coef_speaker * entropy
                + l2_penalty
        )
        return total_loss, (value_loss, loss_actor, entropy)
    
    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True, allow_int=False)

    total_loss, grads = grad_fn(
        speaker_params_i,    # params don't change across minibatches, they are for the same agent.
        trans_batch_i.obs,
        trans_batch_i.action, 
        trans_batch_i.value,
        trans_batch_i.log_prob,
        advantages_i,
        targets_i,
        trans_batch_i.alive
    )
    
    updates, new_opt_state = speaker_optimizer_tx.update(grads, speaker_opt_state_i)
    new_params = optax.apply_updates(speaker_params_i, updates)

    # (new_params, new_opt_state)
    runner_state = (new_params, new_opt_state)

    return runner_state, total_loss

def wandb_callback(metrics):
    (speaker_loss_for_logging, listener_loss_for_logging, optimizer_params_stats_for_logging, agent_param_stats_for_logging, env_info_for_logging, trimmed_transition_batch, speaker_examples, update_step, speaker_example_logging_params, final_speaker_images, probe_logging_params, probe_logits) = metrics
    
    num_speakers = trimmed_transition_batch.speaker_alive.shape[-1]
    num_listeners = trimmed_transition_batch.listener_alive.shape[-1]

    metric_dict = {}

    channel_ratio, speaker_referent_span = env_info_for_logging
    metric_dict.update({"env/avg_channel_ratio": channel_ratio})
    metric_dict.update({"env/speaker_referent_span": speaker_referent_span})

    speaker_loss_total, (speaker_loss_value, speaker_loss_actor, speaker_entropy) = speaker_loss_for_logging
    metric_dict.update({f"loss/total loss/speaker {i}": speaker_loss_total[i] for i in range(len(speaker_loss_total))})
    metric_dict.update({f"loss/value loss/speaker {i}": speaker_loss_value[i] for i in range(len(speaker_loss_value))})
    metric_dict.update({f"loss/actor loss/speaker {i}": speaker_loss_actor[i] for i in range(len(speaker_loss_actor))})
    metric_dict.update({f"loss/entropy/speaker {i}": speaker_entropy[i] for i in range(len(speaker_entropy))})
    
    metric_dict.update({"loss averages/total loss speakers": jnp.mean(speaker_loss_total).item()})
    metric_dict.update({"loss averages/value loss speakers": jnp.mean(speaker_loss_value).item()})
    metric_dict.update({"loss averages/actor loss speakers": jnp.mean(speaker_loss_actor).item()})
    metric_dict.update({"loss averages/entropy speakers": jnp.mean(speaker_entropy).item()})

    listener_loss_total, (listener_loss_value, listener_loss_actor, listener_entropy) = listener_loss_for_logging
    metric_dict.update({f"loss/total loss/listener {i}": listener_loss_total[i] for i in range(len(listener_loss_total))})
    metric_dict.update({f"loss/value loss/listener {i}": listener_loss_value[i] for i in range(len(listener_loss_value))})
    metric_dict.update({f"loss/actor loss/listener {i}": listener_loss_actor[i] for i in range(len(listener_loss_actor))})
    metric_dict.update({f"loss/entropy/listener {i}": listener_entropy[i] for i in range(len(listener_entropy))})
    
    metric_dict.update({"loss averages/total loss listeners": jnp.mean(listener_loss_total).item()})
    metric_dict.update({"loss averages/value loss listeners": jnp.mean(listener_loss_value).item()})
    metric_dict.update({"loss averages/actor loss listeners": jnp.mean(listener_loss_actor).item()})
    metric_dict.update({"loss averages/entropy listeners": jnp.mean(listener_entropy).item()})

    speaker_nu, speaker_mu, listener_nu, listener_mu, speaker_current_lr, listener_current_lr = optimizer_params_stats_for_logging
    metric_dict.update({"optimizer/avg magnitude speaker nu": jnp.mean(speaker_nu).item()})
    metric_dict.update({"optimizer/avg magnitude speaker mu": jnp.mean(speaker_mu).item()})
    metric_dict.update({"optimizer/avg magnitude listener nu": jnp.mean(listener_nu).item()})
    metric_dict.update({"optimizer/avg magnitude listener mu": jnp.mean(listener_mu).item()})
    metric_dict.update({"optimizer/mean learning rate speaker": jnp.mean(speaker_current_lr).item()})
    metric_dict.update({"optimizer/mean learning rate listener": jnp.mean(listener_current_lr).item()})

    speaker_param_magnitude, listener_param_magnitude = agent_param_stats_for_logging
    metric_dict.update({f"param magnitude/speaker {i}": speaker_param_magnitude[i] for i in range(len(speaker_param_magnitude))})
    metric_dict.update({f"param magnitude/listener {i}": listener_param_magnitude[i] for i in range(len(listener_param_magnitude))})
    metric_dict.update({"param magnitude/all speakers": jnp.mean(speaker_param_magnitude).item()})
    metric_dict.update({"param magnitude/all listeners": jnp.mean(listener_param_magnitude).item()})

    ##### Transition batch logging
    image_from_speaker_channel = trimmed_transition_batch.listener_obs_source.astype('bool')
    image_from_env_channel = jnp.invert(image_from_speaker_channel)
    
    #### Reward logging
    mean_speaker_rewards = jnp.mean(trimmed_transition_batch.speaker_reward, axis=0)
    metric_dict.update({f"reward/mean reward/speaker {i}": mean_speaker_rewards[i].item() for i in range(len(mean_speaker_rewards))})
    metric_dict.update({"reward/mean reward/all speakers": jnp.mean(mean_speaker_rewards).item()})

    mean_listener_rewards = jnp.mean(trimmed_transition_batch.listener_reward, axis=0)
    metric_dict.update({f"reward/mean reward/listener {i}": mean_listener_rewards[i].item() for i in range(len(mean_listener_rewards))})
    metric_dict.update({"reward/mean reward/all listeners": jnp.mean(mean_listener_rewards).item()})

    mean_listener_rewards_for_speaker_images = jnp.sum(trimmed_transition_batch.listener_reward * image_from_speaker_channel, axis=0) / (jnp.sum(image_from_speaker_channel, axis=0) + 1e-8)
    metric_dict.update({f"reward/mean reward by image source/speaker images listener {i}": mean_listener_rewards_for_speaker_images[i].item() for i in range(len(mean_listener_rewards_for_speaker_images))})
    metric_dict.update({"reward/mean reward by image source/speaker images all listeners": jnp.mean(mean_listener_rewards_for_speaker_images).item()})

    mean_listener_rewards_for_env_images = jnp.sum(trimmed_transition_batch.listener_reward * image_from_env_channel, axis=0) / (jnp.sum(image_from_env_channel, axis=0) + 1e-8)
    metric_dict.update({f"reward/mean reward by image source/env images listener {i}": mean_listener_rewards_for_env_images[i].item() for i in range(len(mean_listener_rewards_for_env_images))})
    metric_dict.update({"reward/mean reward by image source/env images all listeners": jnp.mean(mean_listener_rewards_for_env_images).item()})
    
    #### Agent action log probs logging
    mean_speaker_log_probs = jnp.mean(trimmed_transition_batch.speaker_log_prob, axis=0)
    metric_dict.update({f"predictions/action log probs/speaker {i}": mean_speaker_log_probs[i].item() for i in range(len(mean_speaker_log_probs))})
    metric_dict.update({"predictions/action log probs/all speakers": jnp.mean(mean_speaker_log_probs).item()})

    mean_listener_log_probs = jnp.mean(trimmed_transition_batch.listener_log_prob, axis=0)
    metric_dict.update({f"predictions/action log probs/listener {i}": mean_listener_log_probs[i].item() for i in range(len(mean_listener_log_probs))})
    metric_dict.update({"predictions/action log probs/all listeners": jnp.mean(mean_listener_log_probs).item()})
    
    mean_listener_log_probs_for_speaker_images = jnp.sum(trimmed_transition_batch.listener_log_prob * image_from_speaker_channel, axis=0) / (jnp.sum(image_from_speaker_channel, axis=0) + 1e-8)
    metric_dict.update({f"predictions/action log probs/listener {i} for speaker images": mean_listener_log_probs_for_speaker_images[i].item() for i in range(len(mean_listener_log_probs_for_speaker_images))})
    metric_dict.update({"predictions/action log probs/all listeners for speaker images": jnp.mean(mean_listener_log_probs_for_speaker_images).item()})

    mean_listener_log_probs_for_env_images = jnp.sum(trimmed_transition_batch.listener_log_prob * image_from_env_channel, axis=0) / (jnp.sum(image_from_env_channel, axis=0) + 1e-8)
    metric_dict.update({f"predictions/action log probs/listener {i} for env images": mean_listener_log_probs_for_env_images[i].item() for i in range(len(mean_listener_log_probs_for_env_images))})
    metric_dict.update({"predictions/action log probs/all listeners for env images": jnp.mean(mean_listener_log_probs_for_env_images).item()})
    
    #### Speaker example logging
    speaker_example_debug_flag, speaker_example_logging_iter = speaker_example_logging_params
    if (update_step + 1 - speaker_example_debug_flag) % speaker_example_logging_iter == 0:
        gut_speaker_examples, tom_speaker_examples = speaker_examples

        gut_speaker_examples_image = make_grid_jnp(jnp.expand_dims(gut_speaker_examples, axis=1), nrow=num_speakers, pad_value=0.25)
        final_gut_speaker_example_images = wandb.Image(np.array(gut_speaker_examples_image), caption="speaker_examples")
        metric_dict.update({"env/speaker_examples": final_gut_speaker_example_images})

        tom_speaker_examples_image = make_grid_jnp(jnp.expand_dims(tom_speaker_examples, axis=1), nrow=num_speakers, pad_value=0.25)
        final_tom_speaker_example_images = wandb.Image(np.array(tom_speaker_examples_image), caption="tom_speaker_examples")
        metric_dict.update({"env/tom_speaker_examples": final_tom_speaker_example_images})

        listener_images = make_grid_jnp(jnp.expand_dims(trimmed_transition_batch.listener_obs[-1], axis=1), nrow=num_listeners, pad_value=0.25)
        final_listener_images = wandb.Image(np.array(listener_images), caption=f"classified as: {str(trimmed_transition_batch.listener_action[-1])}")
        metric_dict.update({"env/last_listener_obs": final_listener_images})

        speaker_images = make_grid_jnp(jnp.expand_dims(final_speaker_images, axis=1), nrow=num_listeners, pad_value=0.25)
        final_speaker_images = wandb.Image(np.array(speaker_images), caption=f"tried generating: {str(trimmed_transition_batch.speaker_obs[-2])}")
        metric_dict.update({"env/speaker_images": final_speaker_images})
        

    ##### Iconicity Probe Logging   # This strikes me as something that belongs in the main scan loop.
    probe_logging_iter, probe_num_examples = probe_logging_params
    if (update_step + 1) % probe_logging_iter == 0:
        p_labels = trimmed_transition_batch.speaker_obs[:probe_num_examples].reshape(probe_num_examples, -1)

        aggregate_probe_entropy, aggregate_probe_per_class_entropy = icon_probe.calculate_entropy(probe_logits, p_labels)
        
        metric_dict.update({f'probe/entropy/all speakers average': aggregate_probe_entropy})
        metric_dict.update({f'probe/entropy/all speakers class {i}': aggregate_probe_per_class_entropy[i] for i in range(len(aggregate_probe_per_class_entropy))})

        for i in range(num_speakers):
            probe_entropy, probe_per_class_entropy = icon_probe.calculate_entropy(probe_logits[:, i, :], p_labels[:, i])

            metric_dict.update({f'probe/entropy/speaker {i} average': probe_entropy})
            metric_dict.update({f'probe/entropy/speaker {i} class {j}': probe_per_class_entropy[j] for j in range(len(probe_per_class_entropy))})
    
    wandb.log(metric_dict)

def make_train(config):
    env = define_env(config)
    env = SimpSigGameLogWrapper(env)

    env_kwargs = config["ENV_KWARGS"]

    config["NUM_ACTORS"] = (env_kwargs["num_speakers"] + env_kwargs["num_listeners"]) * config["NUM_ENVS"]
    config["NUM_MINIBATCHES_LISTENER"] = config["NUM_STEPS"] // config["MINIBATCH_SIZE_LISTENER"]
    config["NUM_MINIBATCHES_SPEAKER"] = config["NUM_STEPS"] // config["MINIBATCH_SIZE_SPEAKER"]
    
    def train(rng):
        # MAKE AGENTS
        rng, rng_s, rng_l = jax.random.split(rng, 3)    # rng_s for speakers, rng_l for listeners
        listener_rngs = jax.random.split(rng_l, env_kwargs["num_listeners"] * config["NUM_ENVS"])   # Make an rng key for each listener
        speaker_rngs = jax.random.split(rng_s, env_kwargs["num_speakers"] * config["NUM_ENVS"])   # Make an rng key for each speaker
        
        listeners_stuff = [initialize_listener(env, x_rng, config, i) for i, x_rng in enumerate(listener_rngs)]
        _listener_networks, listener_train_states, listener_lr_funcs = zip(*listeners_stuff) # listener_lr_funcs is for logging only, it's not actually used directly by the optimizer
        
        speakers_stuff = [initialize_speaker(env, x_rng, config, i) for i, x_rng in enumerate(speaker_rngs)]
        _speaker_networks, speaker_train_states, speaker_lr_funcs = zip(*speakers_stuff) # speaker_lr_funcs is for logging only, it's not actually used directly by the optimizer

        # LOAD ICON PROBE
        local_path = str(pathlib.Path().resolve())
        model_path_str = "/base_experiment/models/" if config["DEBUGGER"] else "/models/"
        raw_restored = icon_probe.load_probe_model(local_path+model_path_str+config["PROBE_MODEL_NAME"], None, action_dim=env_kwargs['num_classes'], opt=config["PROBE_OPTIMIZER"], no_train=True)
        probe_train_state = raw_restored['model']

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, 1)
        obs, log_env_state = env.reset(reset_rng, jnp.zeros((len(reset_rng))))  # log_env_state is a single variable, but each variable it has is actually batched

        speaker_train_freezing_fn = get_train_freezing(config["SPEAKER_TRAIN_SCHEDULE"])
        listener_train_freezing_fn = get_train_freezing(config["LISTENER_TRAIN_SCHEDULE"])
        
        num_speaker_minibatches = config["NUM_MINIBATCHES_SPEAKER"]
        num_listener_minibatches = config["NUM_MINIBATCHES_LISTENER"]

        # DEFINE TRAINING VARIABLES/FUNCTIONS ###########
        speaker_apply_fn = speaker_train_states[0].apply_fn
        listener_apply_fn = listener_train_states[0].apply_fn

        batched_speaker_params = jax.tree.map(lambda *args: jnp.stack(args), *[ts.params for ts in speaker_train_states])
        batched_listener_params = jax.tree.map(lambda *args: jnp.stack(args), *[ts.params for ts in listener_train_states])
        
        speaker_optimizer_tx = speaker_train_states[0].tx
        listener_optimizer_tx = listener_train_states[0].tx

        batched_speaker_opt_states = jax.vmap(speaker_optimizer_tx.init)(batched_speaker_params)
        
        if config["PRETRAINED_LISTENERS"] == "":    # New agents, new optimizer state
            batched_listener_opt_states = jax.vmap(listener_optimizer_tx.init)(batched_listener_params)
        else:   # Loaded agents, old optimizer state
            if config["RESET_LISTENER_OPTIMIZER_COUNTS"]:   # Set counts to 0 so that we can use a new lr schedule
                batched_listener_opt_states = jax.tree.map(lambda *args: jnp.stack(args), *[(ts.opt_state[0], (ts.opt_state[1][0]._replace(count=jnp.array(0)),ts.opt_state[1][1]._replace(count=jnp.array(0)))) for ts in listener_train_states])
            else:       # Don't reset counts. This is not recommended unless you want to resume training and use the exact same scheduler as when the agents were spawned.
                batched_listener_opt_states = jax.tree.map(lambda *args: jnp.stack(args), *[ts.opt_state for ts in listener_train_states])
        
        speaker_action_transform = env._env.speaker_action_transform

        _calculate_gae_listeners = partial(calculate_gae_listeners, gamma=config["GAMMA"], gae_lambda=config["GAE_LAMBDA"])
        _calculate_gae_speakers = partial(calculate_gae_speakers, gamma=config["GAMMA"], gae_lambda=config["GAE_LAMBDA"])

        def update_speaker(_rng, trans_batch_for_speaker_i, advantages_for_speaker_i, targets_for_speaker_i, speaker_params_i, speaker_opt_state_i):
            carry_state = (_rng, speaker_params_i, speaker_opt_state_i)
            step_data = (trans_batch_for_speaker_i, advantages_for_speaker_i, targets_for_speaker_i)
            
            partial_update_speaker = partial(update_minibatch_speaker, speaker_apply_fn=speaker_apply_fn, speaker_optimizer_tx=speaker_optimizer_tx, clip_eps=config["CLIP_EPS"], l2_reg_coef_speaker=config["L2_REG_COEF_SPEAKER"], vf_coef=config["VF_COEF"], ent_coef_speaker=config["ENT_COEF_SPEAKER"])
            
            def scan_step(carry, step_input):
                _rng, speaker_params_i, speaker_opt_state_i = carry
                trans_batch, advantages, targets = step_input
                _this_rng, _next_rng = jax.random.split(_rng)

                speaker_update_args = (_this_rng, trans_batch, advantages, targets, speaker_params_i, speaker_opt_state_i)
                (updated_speaker_params, updated_speaker_opt_state), loss = partial_update_speaker(speaker_update_args)
                
                # Carry forward updated state
                return (_next_rng, updated_speaker_params, updated_speaker_opt_state), loss
            
            (final_rng, final_speaker_params, final_speaker_opt_state), total_loss = jax.lax.scan(scan_step, carry_state, step_data)
            return (final_speaker_params, final_speaker_opt_state), total_loss
        
        def update_listener(_rng, trans_batch_for_listener_i, advantages_for_listener_i, targets_for_listener_i, listener_params_i, listener_opt_state_i):
            carry_state = (_rng, listener_params_i, listener_opt_state_i)
            step_data = (trans_batch_for_listener_i, advantages_for_listener_i, targets_for_listener_i)
            
            partial_update_listener = partial(update_minibatch_listener, listener_apply_fn=listener_apply_fn, listener_optimizer_tx=listener_optimizer_tx, clip_eps=config["CLIP_EPS"], l2_reg_coef_listener=config["L2_REG_COEF_LISTENER"], vf_coef=config["VF_COEF"], ent_coef_listener=config["ENT_COEF_LISTENER"])
            
            def scan_step(carry, step_input):
                _rng, listener_params_i, listener_opt_state_i = carry
                trans_batch, advantages, targets = step_input
                _this_rng, _next_rng = jax.random.split(_rng)

                listener_update_args = (_this_rng, trans_batch, advantages, targets, listener_params_i, listener_opt_state_i)
                (updated_listener_params, updated_listener_opt_state), loss = partial_update_listener(listener_update_args)
                
                # Carry forward updated state
                return (_next_rng, updated_listener_params, updated_listener_opt_state), loss
            
            (final_rng, final_listener_params, final_listener_opt_state), total_loss = jax.lax.scan(scan_step, carry_state, step_data)
            return (final_listener_params, final_listener_opt_state), total_loss
        
        vmap_update_speaker = jax.vmap(update_speaker, in_axes=(0, 2, 2, 2, 0, 0))
        vmap_update_listener = jax.vmap(update_listener, in_axes=(0, 2, 2, 2, 0, 0))
        
        ##########################

        runner_state = (batched_speaker_params, batched_listener_params, batched_speaker_opt_states, batched_listener_opt_states, log_env_state, obs, _rng)
        
        # TRAIN LOOP
        def _update_step(runner_state, update_step, env, config):
            #### Unpack runner_state

            batched_speaker_params, batched_listener_params, batched_speaker_opt_states, batched_listener_opt_states, log_env_state, last_obs, rng = runner_state
            ## listener_train_state is a tuple of TrainStates of length num_envs * env.num_listeners

            rng_folded = jax.random.fold_in(key=rng, data=update_step)
            this_rng, next_rng = jax.random.split(rng_folded)
            last_obs, log_env_state = env.reset(this_rng.reshape((1, 2)), jnp.ones((1,)) * update_step)

            ######### COLLECT TRAJECTORIES
            runner_state_for_env_step = (batched_speaker_params, batched_listener_params, log_env_state, last_obs, this_rng)
            partial_env_step = partial(env_step, speaker_apply_fn=speaker_apply_fn, listener_apply_fn=listener_apply_fn, env=env, config=config)
            runner_state_from_env_step, transition_batch = jax.lax.scan(lambda rs, _: partial_env_step(rs), runner_state_for_env_step, jnp.arange(config['NUM_STEPS'] + 2))  # +2, then skim away the first step and last step
            _, _, new_log_env_state, new_obs, _ = runner_state_from_env_step
            ## transition_batch is an instance of Transition with batched sub-objects
            ## The shape of transition_batch is (num_steps, num_envs, ...) because it's the output of jax.lax.scan, which enumerates over steps

            ## Instead of executing the agents on the final observation to get their values, we are simply going to ignore the last observation from traj_batch.
            ## We'll need to get the final value in transition_batch and cut off the last index
            ## We want to cleave off the final step, so it should go from shape (A, B, C) to shape (A-1, B, C)
            ## We also need to shift rewards and alives for speakers over by 1 to the left. speaker gets a delayed reward.
            trimmed_transition_batch = Transition( # Assuming a single environment, so squeezing here. E.g. speaker_action will now be shape (num_steps, num_agents)
                speaker_action=jnp.squeeze(transition_batch.speaker_action[1:-1, ...]),
                speaker_reward=jnp.squeeze(transition_batch.speaker_reward[2:, ...]),
                speaker_value=jnp.squeeze(transition_batch.speaker_value[1:-1, ...]),
                speaker_log_prob=jnp.squeeze(transition_batch.speaker_log_prob[1:-1, ...]),
                speaker_obs=jnp.squeeze(transition_batch.speaker_obs[1:-1, ...]),
                speaker_alive=jnp.squeeze(transition_batch.speaker_alive[2:, ...]),
                listener_action=jnp.squeeze(transition_batch.listener_action[1:-1, ...]),
                listener_reward=jnp.squeeze(transition_batch.listener_reward[1:-1, ...]),
                listener_value=jnp.squeeze(transition_batch.listener_value[1:-1, ...]),
                listener_log_prob=jnp.squeeze(transition_batch.listener_log_prob[1:-1, ...]),
                listener_obs=jnp.squeeze(transition_batch.listener_obs[1:-1, ...]),
                listener_alive=jnp.squeeze(transition_batch.listener_alive[1:-1, ...]),
                channel_map=jnp.squeeze(transition_batch.channel_map[1:-1, ...]),
                listener_obs_source=jnp.squeeze(transition_batch.listener_obs_source[1:-1, ...])
            )

            ####### CALCULATE ADVANTAGE #############
            ## This is unnecessary for bandits, but ideally this code can be extended to pomdps

            #### At this point we can selectively train the speakers and listeners based on whether they are alive and whether train freezing is on
            train_speaker = speaker_train_freezing_fn(update_step)
            train_listener = listener_train_freezing_fn(update_step)

            ####
            # listener_advantages, listener_targets = _calculate_gae_listeners(trimmed_transition_batch, jnp.squeeze(transition_batch.listener_value[-1]))
            # speaker_advantages, speaker_targets = _calculate_gae_speakers(trimmed_transition_batch, jnp.squeeze(transition_batch.speaker_value[-1]))
            #### The below lines implement train freezing while the above commented-out lines do not.
            listener_advantages, listener_targets = jax.lax.cond(train_listener, lambda _: _calculate_gae_listeners(trimmed_transition_batch, jnp.squeeze(transition_batch.listener_value[-1])),
                                                                 lambda _: (jnp.zeros((config["NUM_STEPS"], env_kwargs["num_listeners"])),
                                                                            jnp.zeros((config["NUM_STEPS"], env_kwargs["num_listeners"]))), operand=None)
            
            speaker_advantages, speaker_targets = jax.lax.cond(train_speaker, lambda _: _calculate_gae_speakers(trimmed_transition_batch, jnp.squeeze(transition_batch.speaker_value[-1])),
                                                                 lambda _: (jnp.zeros((config["NUM_STEPS"], env_kwargs["num_speakers"])),
                                                                            jnp.zeros((config["NUM_STEPS"], env_kwargs["num_speakers"]))), operand=None)
            
            ##### UPDATE AGENTS #####################
            update_speaker_rng, update_listener_rng, this_rng = jax.random.split(this_rng, 3)
            update_listener_rngs = jax.random.split(update_listener_rng, env_kwargs["num_listeners"])
            update_speaker_rngs = jax.random.split(update_speaker_rng, env_kwargs["num_speakers"])
            
            ### Listeners: Reshape variables into minibatches
            listener_advantages = listener_advantages.reshape((num_listener_minibatches, -1) + listener_advantages.shape[1:])
            listener_targets = listener_targets.reshape((num_listener_minibatches, -1) + listener_targets.shape[1:])
            
            listener_transition_batch = HalfTransition(
                action=trimmed_transition_batch.listener_action.reshape((num_listener_minibatches, -1) + trimmed_transition_batch.listener_action.shape[1:]),
                reward=trimmed_transition_batch.listener_reward.reshape((num_listener_minibatches, -1) + trimmed_transition_batch.listener_reward.shape[1:]),
                value=trimmed_transition_batch.listener_value.reshape((num_listener_minibatches, -1) + trimmed_transition_batch.listener_value.shape[1:]),
                log_prob=trimmed_transition_batch.listener_log_prob.reshape((num_listener_minibatches, -1) + trimmed_transition_batch.listener_log_prob.shape[1:]),
                obs=trimmed_transition_batch.listener_obs.reshape((num_listener_minibatches, -1) + trimmed_transition_batch.listener_obs.shape[1:]),
                alive=trimmed_transition_batch.listener_alive.reshape((num_listener_minibatches, -1) + trimmed_transition_batch.listener_alive.shape[1:]),
                # channel_map=trimmed_transition_batch.channel_map.reshape((num_listener_minibatches, -1) + trimmed_transition_batch.channel_map.shape[1:])
            )

            ######## This code was for verifying that update_listener was dealing with minibatches properly. Other changes in the codebase are necessary to fully test it. Get rid of randomness and carry the old opt state and network params
            # update_listener_rngs = jnp.broadcast_to(update_listener_rng, (env_kwargs["num_listeners"], 2))
            # listener_advantages2 = jnp.tile(jnp.expand_dims(listener_advantages[0], axis=0), (2, 1, 1))
            # listener_targets2 = jnp.tile(jnp.expand_dims(listener_targets[0], axis=0), (2, 1, 1))
            # listener_transition_batch2 = ListenerTransition(
            #     action=jnp.tile(jnp.expand_dims(trimmed_transition_batch.listener_action[:16, ...], axis=0), (2, 1, 1)),
            #     reward=jnp.tile(jnp.expand_dims(trimmed_transition_batch.listener_reward[:16, ...], axis=0), (2, 1, 1)),
            #     value=jnp.tile(jnp.expand_dims(trimmed_transition_batch.listener_value[:16, ...], axis=0), (2, 1, 1)),
            #     log_prob=jnp.tile(jnp.expand_dims(trimmed_transition_batch.listener_log_prob[:16, ...], axis=0), (2, 1, 1)),
            #     obs=jnp.tile(jnp.expand_dims(trimmed_transition_batch.listener_obs[:16, ...], axis=0), (2, 1, 1, 1, 1)),
            #     alive=jnp.tile(jnp.expand_dims(trimmed_transition_batch.listener_alive[:16, ...], axis=0), (2, 1, 1)),
            #     # channel_map=jnp.tile(jnp.expand_dims(trimmed_transition_batch.channel_map[:16, ...], axis=0), (2, 1, 1, 1)),
            # )
            ############################################
            

            ### Speakers: Reshape variables into minibatches
            speaker_advantages = speaker_advantages.reshape((num_speaker_minibatches, -1) + speaker_advantages.shape[1:])
            speaker_targets = speaker_targets.reshape((num_speaker_minibatches, -1) + speaker_targets.shape[1:])
            
            speaker_transition_batch = HalfTransition(
                action=trimmed_transition_batch.speaker_action.reshape((num_speaker_minibatches, -1) + trimmed_transition_batch.speaker_action.shape[1:]),
                reward=trimmed_transition_batch.speaker_reward.reshape((num_speaker_minibatches, -1) + trimmed_transition_batch.speaker_reward.shape[1:]),
                value=trimmed_transition_batch.speaker_value.reshape((num_speaker_minibatches, -1) + trimmed_transition_batch.speaker_value.shape[1:]),
                log_prob=trimmed_transition_batch.speaker_log_prob.reshape((num_speaker_minibatches, -1) + trimmed_transition_batch.speaker_log_prob.shape[1:]),
                obs=trimmed_transition_batch.speaker_obs.reshape((num_speaker_minibatches, -1) + trimmed_transition_batch.speaker_obs.shape[1:]),
                alive=trimmed_transition_batch.speaker_alive.reshape((num_speaker_minibatches, -1) + trimmed_transition_batch.speaker_alive.shape[1:]),
            )
            ###############
            
            ##### Finally execute the compiled update functions
            ### The below implements train freezing
            final_listener_outputs = jax.lax.cond(train_listener, 
                                                  lambda _: vmap_update_listener(update_listener_rngs, listener_transition_batch, listener_advantages, listener_targets, batched_listener_params, batched_listener_opt_states),
                                                  lambda _: ((batched_listener_params, batched_listener_opt_states), (jnp.zeros((env_kwargs["num_listeners"], num_listener_minibatches)), (jnp.zeros((env_kwargs["num_listeners"], num_listener_minibatches)), jnp.zeros((env_kwargs["num_listeners"], num_listener_minibatches)), jnp.zeros((env_kwargs["num_listeners"], num_listener_minibatches))))), None)
            final_speaker_outputs = jax.lax.cond(train_speaker,
                                                 lambda _: vmap_update_speaker(update_speaker_rngs, speaker_transition_batch, speaker_advantages, speaker_targets, batched_speaker_params, batched_speaker_opt_states),
                                                 lambda _: ((batched_speaker_params, batched_speaker_opt_states), (jnp.zeros((env_kwargs["num_speakers"], num_speaker_minibatches)), (jnp.zeros((env_kwargs["num_speakers"], num_speaker_minibatches)), jnp.zeros((env_kwargs["num_speakers"], num_speaker_minibatches)), jnp.zeros((env_kwargs["num_speakers"], num_speaker_minibatches))))), None)
            ## Unpack the outputs
            (final_listener_params, final_listener_opt_states), (listener_loss_total, (listener_loss_value, listener_loss_actor, listener_entropy)) = final_listener_outputs
            (final_speaker_params, final_speaker_opt_states), (speaker_loss_total, (speaker_loss_value, speaker_loss_actor, speaker_entropy)) = final_speaker_outputs
            ########################################################

            ## Update the runner_state for the next scan loop
            runner_state = (final_speaker_params, final_listener_params, final_speaker_opt_states, final_listener_opt_states, new_log_env_state, new_obs, next_rng)
            
            ########################################################
            ####################### LOGGING ########################
            #######################  BELOW  ########################
            ########################################################

            speaker_loss_for_logging = jax.tree.map(lambda x: jnp.mean(x, axis=1), (speaker_loss_total, (speaker_loss_value, speaker_loss_actor, speaker_entropy)))
            listener_loss_for_logging = jax.tree.map(lambda x: jnp.mean(x, axis=1), (listener_loss_total, (listener_loss_value, listener_loss_actor, listener_entropy)))
            
            ## Collect speaker examples            
            gut_speaker_examples = jax.lax.cond((update_step + 1 - config["SPEAKER_EXAMPLE_DEBUG"]) % config["SPEAKER_EXAMPLE_LOGGING_ITER"] == 0, 
                                            lambda _: get_speaker_examples(next_rng, speaker_apply_fn, batched_speaker_params, speaker_action_transform, config), 
                                            lambda _: jnp.zeros((env_kwargs["num_speakers"]*config["SPEAKER_EXAMPLE_NUM"]*env_kwargs["num_classes"], env_kwargs["image_dim"], env_kwargs["image_dim"])), operand=None)
            tom_speaker_examples = jax.lax.cond(((update_step + 1 - config["SPEAKER_EXAMPLE_DEBUG"]) % config["SPEAKER_EXAMPLE_LOGGING_ITER"] == 0) & (env.agent_inferential_mode_fn(update_step) == 1), 
                                                lambda _: get_tom_speaker_examples(next_rng, listener_apply_fn, batched_listener_params, speaker_apply_fn, batched_speaker_params, speaker_action_transform, config), 
                                                lambda _: jnp.zeros((env_kwargs["num_speakers"]*config["SPEAKER_EXAMPLE_NUM"]*env_kwargs["num_classes"], env_kwargs["image_dim"], env_kwargs["image_dim"])), operand=None)
            speaker_examples = (gut_speaker_examples, tom_speaker_examples)
            ## Both sets of examples are shape (num_classes * num_speakers * speaker_example_num, image_dim, image_dim)
            
            ## Collect the last set of speaker-generated images for this epoch.
            final_speaker_images = speaker_action_transform(trimmed_transition_batch.speaker_action[-2].reshape((env_kwargs["num_speakers"]), -1))
            if env_kwargs["center_listener_obs"]:
                final_speaker_images = center_obs(final_speaker_images)
            ## speaker_images is shaped (num_speakers, image_dim, image_dim)

            ### Calculate optimizer param stats (L2-Norm) and learning rates (assuming they are the same for all agents of that type)
            speaker_nu = jax.vmap(lambda x: jnp.linalg.norm(jax.flatten_util.ravel_pytree(x)[0]), in_axes=(0))(final_speaker_opt_states[1][0].nu)
            speaker_mu = jax.vmap(lambda x: jnp.linalg.norm(jax.flatten_util.ravel_pytree(x)[0]), in_axes=(0))(final_speaker_opt_states[1][0].mu)
            listener_nu = jax.vmap(lambda x: jnp.linalg.norm(jax.flatten_util.ravel_pytree(x)[0]), in_axes=(0))(final_listener_opt_states[1][0].nu)
            listener_mu = jax.vmap(lambda x: jnp.linalg.norm(jax.flatten_util.ravel_pytree(x)[0]), in_axes=(0))(final_listener_opt_states[1][0].mu)
            ## Each is shaped (num_speakers,) or (num_listeners,)
            speaker_current_lr = speaker_lr_funcs[0](batched_speaker_opt_states[1][0].count)
            listener_current_lr = listener_lr_funcs[0](batched_listener_opt_states[1][0].count)
            ## These are just floats. They are also not guaranteed to be correct.
            optimizer_params_stats_for_logging = (speaker_nu, speaker_mu, listener_nu, listener_mu, speaker_current_lr, listener_current_lr)
            ###

            ## Calculate agent param stats (L2-Norm)
            speaker_param_magnitude = jax.vmap(lambda x: jnp.linalg.norm(jax.flatten_util.ravel_pytree(x)[0]), in_axes=(0))(final_speaker_params)
            listener_param_magnitude = jax.vmap(lambda x: jnp.linalg.norm(jax.flatten_util.ravel_pytree(x)[0]), in_axes=(0))(final_listener_params)
            agent_param_stats_for_logging = (speaker_param_magnitude, listener_param_magnitude)
            ## Each is shaped (num_speakers,) or (num_listeners,)

            ### Evaluate iconicity probe and action penalties
            def get_probe_logits():
                speaker_images_for_icon_probe = jnp.expand_dims(speaker_action_transform(trimmed_transition_batch.speaker_action[:config["PROBE_NUM_EXAMPLES"]].reshape((-1, env_kwargs["speaker_action_dim"]))), axis=3)
                # I need to calculate the whitesum penalty based on the speaker_images. I don't know what shape they are
                return probe_train_state.apply_fn({'params': probe_train_state.params}, speaker_images_for_icon_probe).reshape((-1, env_kwargs["num_speakers"], env_kwargs["num_classes"]))
            probe_logits = jax.lax.cond((update_step + 1) % config["PROBE_LOGGING_ITER"] == 0, lambda _: get_probe_logits(), lambda _: jnp.zeros((config["PROBE_NUM_EXAMPLES"], env_kwargs["num_speakers"], env_kwargs["num_classes"])), operand=None)
            ###

            ### Collect env channel info
            channel_ratio = log_env_state.env_state.requested_num_speaker_images[0] / env_kwargs["num_channels"]
            speaker_referent_span = log_env_state.env_state.requested_speaker_referent_span[0]
            env_info_for_logging = (channel_ratio, speaker_referent_span)
            ###

            ### Some other debug info
            speaker_example_logging_params = (config["SPEAKER_EXAMPLE_DEBUG"], config["SPEAKER_EXAMPLE_LOGGING_ITER"])
            probe_logging_params = (config["PROBE_LOGGING_ITER"], config["PROBE_NUM_EXAMPLES"])

            metrics_for_logging = (speaker_loss_for_logging, listener_loss_for_logging, optimizer_params_stats_for_logging, agent_param_stats_for_logging, env_info_for_logging, trimmed_transition_batch, speaker_examples, update_step, speaker_example_logging_params, final_speaker_images, probe_logging_params, probe_logits)# trimmed_transition_batch, log_env_state, speaker_current_lr, listener_current_lr, speaker_examples, tom_speaker_examples, speaker_images, update_step, probe_logits)

            jax.experimental.io_callback(wandb_callback, None, metrics_for_logging)
            
            return runner_state, update_step + 1

        ### For debugging speaker examples
        # speaker_exs = get_speaker_examples(runner_state, env, config)
        # speaker_example_images = make_grid(torch.tensor(np.array(speaker_exs.reshape((-1, 1, env_kwargs["image_dim"], env_kwargs["image_dim"])))), nrow=env_kwargs["num_classes"], pad_value=0.25)
        # speaker_exs2 = get_tom_speaker_examples(runner_state, env, config) # the output of this should be shape (200, 32, 32)
        # speaker_example_images2 = make_grid(torch.tensor(np.array(speaker_exs2.reshape((-1, 1, env_kwargs["image_dim"], env_kwargs["image_dim"])))), nrow=env_kwargs["num_classes"], pad_value=0.25)
        
        partial_update_fn = partial(_update_step, env=env, config=config)
        runner_state, _ = jax.lax.scan( # Perform the update step for a specified number of updates and update the runner state
            partial_update_fn, runner_state, jnp.arange(config['UPDATE_EPOCHS'])
        )

        return {"runner_state": runner_state, "train_states": (speaker_train_states, listener_train_states)}

    return train


@hydra.main(version_base=None, config_path="config", config_name="default")
def main(config):

    config = OmegaConf.to_container(
        config, resolve=True, throw_on_missing=True
    )
    # print(OmegaConf.to_yaml(config))
    # return
    run = wandb.init(
        entity=config["ENTITY"],
        project=config["PROJECT"],
        tags=["main"],
        config=config,
        mode=config["WANDB_MODE"],
        save_code=True,
        notes=config["WANDB_NOTES"]
    )

    rng = jax.random.PRNGKey(config["JAX_RANDOM_SEED"])
    # with jax.profiler.trace("/tmp/jax-trace", create_perfetto_link=True):
    train = make_train(config)
    result = train(rng)
    run.finish()

    if config["PICKLE_FINAL_AGENTS"]:
        config["WANDB_RUN_NAME"] = run.name
        
        speaker_train_states = result["train_states"][0]
        listener_train_states = result["train_states"][1]
        final_speaker_params, final_listener_params, final_speaker_opt_states, final_listener_opt_states, _, _, _ = result["runner_state"]
        
        unbatched_speaker_params = [jax.tree.map(lambda x: x[i], final_speaker_params) for i in range(len(speaker_train_states))]
        unbatched_listener_params = [jax.tree.map(lambda x: x[i], final_listener_params) for i in range(len(listener_train_states))]

        unbatched_speaker_opt_states = [jax.tree.map(lambda x: x[i], final_speaker_opt_states) for i in range(len(speaker_train_states))]
        unbatched_listener_opt_states = [jax.tree.map(lambda x: x[i], final_listener_opt_states) for i in range(len(listener_train_states))]

        updated_speaker_train_states = [
            ts.replace(params=p, opt_state=o)
            for ts, p, o in zip(speaker_train_states, unbatched_speaker_params, unbatched_speaker_opt_states)
        ]
        updated_listener_train_states = [
            ts.replace(params=p, opt_state=o)
            for ts, p, o in zip(listener_train_states, unbatched_listener_params, unbatched_listener_opt_states)
        ]
        save_agents(updated_listener_train_states, updated_speaker_train_states, config)

    print("Done")


if __name__ == "__main__":
    main()
    # with jax.profiler.trace("/tmp/jax-trace", create_perfetto_link=True):
    #     main()

    # interact -t 6:00:00 -q gpu -f quadrortx
