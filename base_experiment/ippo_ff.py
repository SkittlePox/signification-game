"""
Based on PureJaxRL Implementation of PPO
"""
from functools import partial
import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple, Any, Dict, Tuple
from flax.training import train_state
import jaxmarl
import wandb
import hydra
import torch
from torchvision.utils import make_grid
from torchvision.datasets import MNIST
from omegaconf import OmegaConf
from simplified_signification_game import SimplifiedSignificationGame, State
from agents import *


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


@jax.profiler.annotate_function
def define_env(config):
    if config["ENV_DATASET"] == 'mnist':        
        from utils import to_jax

        mnist_dataset = MNIST('/tmp/mnist/', download=True)
        images, labels = to_jax(mnist_dataset, num_datapoints=config["ENV_NUM_DATAPOINTS"])  # This should also be in ENV_KWARGS
        images = images.astype('float32') / 255.0
        
        env = SimplifiedSignificationGame(**config["ENV_KWARGS"], dataset=(images, labels))
        return env


@jax.profiler.annotate_function
def initialize_listener(env, rng, config):
    if config["ENV_LISTENER_ARCH"] == 'conv':
        listener_network = ActorCriticListenerConv(action_dim=config["ENV_KWARGS"]["num_classes"], image_dim=config["ENV_KWARGS"]["image_dim"], config=config)
    elif config["ENV_LISTENER_ARCH"] == 'convsmall':
        listener_network = ActorCriticListenerConvSmall(action_dim=config["ENV_KWARGS"]["num_classes"], image_dim=config["ENV_KWARGS"]["image_dim"], config=config)
    elif config["ENV_LISTENER_ARCH"] == 'dense':
        listener_network = ActorCriticListenerDense(action_dim=config["ENV_KWARGS"]["num_classes"], image_dim=config["ENV_KWARGS"]["image_dim"], config=config)

    rng, _rng = jax.random.split(rng)
    init_x = jnp.zeros(
            (1, config["NUM_ENVS"], config["ENV_KWARGS"]["image_dim"]**2)
        )
    network_params = listener_network.init({'params': _rng, 'dropout': _rng}, init_x)
    
    def linear_schedule(count):
        frac = 1.0 - ((count * config["ANNEAL_LR_LISTENER_MULTIPLIER"]) / (config["NUM_MINIBATCHES_LISTENER"] * config["UPDATE_EPOCHS"]))
        return config["LR_LISTENER"] * frac
    if config["ANNEAL_LR_LISTENER"]:
        tx = optax.chain(
            optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
            optax.adam(learning_rate=linear_schedule, eps=1e-5),
        )
        lr_func = linear_schedule
    else:
        tx = optax.chain(optax.clip_by_global_norm(config["MAX_GRAD_NORM"]), optax.adam(config["LR_LISTENER"], eps=1e-5))
        lr_func = lambda *args: config["LR_LISTENER"]
    
    train_state = TrainState.create(
        apply_fn=listener_network.apply,
        params=network_params,
        key=rng,
        tx=tx,
    )
    return listener_network, train_state, lr_func

@jax.profiler.annotate_function
def initialize_speaker(env, rng, config):
    if config["ENV_SPEAKER_ARCH"] == 'gauss':
        speaker_network = ActorCriticSpeaker(latent_dim=32, num_classes=config["ENV_KWARGS"]["num_classes"], image_dim=config["ENV_KWARGS"]["image_dim"], config=config)
    elif config["ENV_SPEAKER_ARCH"] == 'gausssmall':
        speaker_network = ActorCriticSpeakerGaussSmall(latent_dim=32, num_classes=config["ENV_KWARGS"]["num_classes"], image_dim=config["ENV_KWARGS"]["image_dim"], config=config)
    elif config["ENV_SPEAKER_ARCH"] == 'gausssmallnovar':
        speaker_network = ActorCriticSpeakerGaussSmallNovariance(latent_dim=32, num_classes=config["ENV_KWARGS"]["num_classes"], image_dim=config["ENV_KWARGS"]["image_dim"], config=config)
    elif config["ENV_SPEAKER_ARCH"] == 'gausssplat':
        speaker_network = ActorCriticSpeakerGaussSplat(latent_dim=32, num_classes=config["ENV_KWARGS"]["num_classes"], action_dim=config["ENV_KWARGS"]["speaker_action_dim"], config=config)
    elif config["ENV_SPEAKER_ARCH"] == 'gausssplatcovar':
        speaker_network = ActorCriticSpeakerGaussSplatCov(latent_dim=32, num_classes=config["ENV_KWARGS"]["num_classes"], action_dim=config["ENV_KWARGS"]["speaker_action_dim"], config=config)

    rng, _rng = jax.random.split(rng)
    init_y = jnp.zeros(
            (1, config["NUM_ENVS"], 1),
            dtype=jnp.int32
        )
    network_params = speaker_network.init({'params': _rng, 'dropout': _rng}, init_y)

    # For the learning rate
    def linear_schedule(count):
        frac = 1.0 - ((count * config["ANNEAL_LR_SPEAKER_MULTIPLIER"]) / (config["NUM_MINIBATCHES_SPEAKER"] * config["UPDATE_EPOCHS"]))   # I don't know exactly how this works.
        # jax.debug.print(str(count))
        return config["LR_SPEAKER"] * frac

    if config["ANNEAL_LR_SPEAKER"]:
        tx = optax.chain(
            optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
            optax.adam(learning_rate=linear_schedule, eps=1e-5),
        )
        lr_func = linear_schedule
    else:
        tx = optax.chain(optax.clip_by_global_norm(config["MAX_GRAD_NORM"]), optax.adam(config["LR_SPEAKER"], eps=1e-5))
        lr_func = lambda *args: config["LR_SPEAKER"]
    
    train_state = TrainState.create(
        apply_fn=speaker_network.apply,
        params=network_params,
        key=rng,
        tx=tx,
    )
    return speaker_network, train_state, lr_func

@jax.profiler.annotate_function
def execute_individual_listener(__rng, _listener_train_state_i, _listener_obs_i):
    __rng, dropout_key = jax.random.split(__rng)
    _listener_obs_i = _listener_obs_i.ravel()
    policy, value = _listener_train_state_i.apply_fn(_listener_train_state_i.params, _listener_obs_i, rngs={'dropout': dropout_key})
    action = policy.sample(seed=__rng)
    log_prob = policy.log_prob(action)
    return action, log_prob, value

@jax.profiler.annotate_function
def execute_individual_speaker(__rng, _speaker_train_state_i, _speaker_obs_i):
    __rng, dropout_key = jax.random.split(__rng)
    _speaker_obs_i = _speaker_obs_i.reshape((1, -1, 1))
    policy, value = _speaker_train_state_i.apply_fn(_speaker_train_state_i.params, _speaker_obs_i, rngs={'dropout': dropout_key})
    action = policy.sample(seed=__rng)
    log_prob = policy.log_prob(action)
    # log_prob = jnp.sum(policy.log_prob(action), axis=1) # Sum log-probs for individual pixels to get log-probs of whole image
    return jnp.clip(action, a_min=0.0, a_max=1.0), log_prob, value


def get_speaker_examples(runner_state, env, config):
    _, speaker_train_states, log_env_state, obs, rng = runner_state
    env_rngs = jax.random.split(rng, len(speaker_train_states))
    speaker_obs = jnp.arange(config["ENV_KWARGS"]["num_classes"])
    speaker_outputs = [execute_individual_speaker(env_rngs[i], speaker_train_states[i], speaker_obs) for i in range(config["ENV_KWARGS"]["num_speakers"])]
    speaker_action = jnp.array([o[0] for o in speaker_outputs]).reshape(config["NUM_ENVS"], -1, config["ENV_KWARGS"]["speaker_action_dim"])
    speaker_action = speaker_action.reshape((config["ENV_KWARGS"]["num_speakers"] * len(speaker_obs), -1))   # TODO: this could prob be changed
    speaker_images = env._env.speaker_action_transform(speaker_action).reshape((config["ENV_KWARGS"]["num_speakers"], config["ENV_KWARGS"]["num_classes"], config["ENV_KWARGS"]["image_dim"], config["ENV_KWARGS"]["image_dim"]))
    return speaker_images


@jax.profiler.annotate_function
def env_step(runner_state, env, config):
    """This function literally is just for collecting rollouts, which involves applying the joint policy to the env and stepping forward."""
    listener_train_states, speaker_train_states, log_env_state, obs, rng = runner_state
    speaker_obs, listener_obs = obs

    env_kwargs = config["ENV_KWARGS"]

    speaker_obs = speaker_obs.ravel()
    listener_obs = listener_obs.reshape((listener_obs.shape[0]*listener_obs.shape[1], listener_obs.shape[2]*listener_obs.shape[3]))

    ##### COLLECT ACTIONS FROM AGENTS
    rng, _rng = jax.random.split(rng)
    env_rngs = jax.random.split(_rng, len(listener_train_states))

    # COLLECT LISTENER ACTIONS
    listener_outputs = [execute_individual_listener(*args) for args in zip(env_rngs, listener_train_states, listener_obs)]
    listener_action = jnp.array([o[0] for o in listener_outputs], dtype=jnp.int32).reshape(config["NUM_ENVS"], -1)
    listener_log_prob = jnp.array([o[1] for o in listener_outputs]).reshape(config["NUM_ENVS"], -1)
    listener_value = jnp.array([o[2] for o in listener_outputs]).reshape(config["NUM_ENVS"], -1)

    # COLLECT SPEAKER ACTIONS
    speaker_outputs = [execute_individual_speaker(*args) for args in zip(env_rngs, speaker_train_states, speaker_obs)]
    speaker_action = jnp.array([o[0] for o in speaker_outputs]).reshape(config["NUM_ENVS"], -1, env_kwargs["speaker_action_dim"])
    speaker_log_prob = jnp.array([o[1] for o in speaker_outputs]).reshape(config["NUM_ENVS"], -1)
    speaker_value = jnp.array([o[2] for o in speaker_outputs]).reshape(config["NUM_ENVS"], -1)

    # TODO: I could make this a debug option
    # rng_step = jax.random.split(_rng, config["NUM_ENVS"])
    # SIMULATE SPEAKER ACTIONS. TEMPORARY, FOR DEBUGGING, UNTIL SPEAKERS WORK:
    # speaker_action = jnp.array([env.action_space(agent).sample(rng_step[0]) for i, agent in enumerate(env.agents) if agent.startswith("speaker")])
    # speaker_action = jnp.expand_dims(speaker_action, 0).repeat(config["NUM_ENVS"], axis=0)
    # speaker_value = jnp.zeros((config["NUM_ENVS"], env_kwargs["num_speakers"]), dtype=jnp.float32)
    # speaker_log_prob = jnp.zeros_like(speaker_action)    # This will eventually be replaced by real speaker logprobs which should actually be a single float per agent!
    ###############################################

    ##### STEP ENV
    rng_step = jax.random.split(_rng, config["NUM_ENVS"])
    new_obs, env_state, rewards, alives, info = env.step(rng_step, log_env_state, (speaker_action, listener_action))

    speaker_alive = jnp.array([alives[f"speaker_{v}"] for v in range(env_kwargs["num_speakers"])]).reshape(config["NUM_ENVS"], -1)
    listener_alive = jnp.array([alives[f"listener_{v}"] for v in range(env_kwargs["num_listeners"])]).reshape(config["NUM_ENVS"], -1)

    speaker_reward = jnp.array([rewards[f"speaker_{v}"] for v in range(env_kwargs["num_speakers"])]).reshape(config["NUM_ENVS"], -1)
    listener_reward = jnp.array([rewards[f"listener_{v}"] for v in range(env_kwargs["num_listeners"])]).reshape(config["NUM_ENVS"], -1)

    speaker_obs = speaker_obs.reshape((config["NUM_ENVS"], -1))
    listener_obs = listener_obs.reshape((config["NUM_ENVS"], env_kwargs["num_channels"], env_kwargs["image_dim"], env_kwargs["image_dim"]))

    transition = Transition(
        speaker_action,
        speaker_reward,
        speaker_value,
        speaker_log_prob,
        speaker_obs,
        speaker_alive,
        listener_action,
        listener_reward,
        listener_value,
        listener_log_prob,
        listener_obs,
        listener_alive
    )

    runner_state = (listener_train_states, speaker_train_states, env_state, new_obs, rng)
    
    return runner_state, transition


@jax.profiler.annotate_function
def update_minibatch_listener(j, trans_batch_i, advantages_i, targets_i, train_state, config):
    # j is for iterating through minibatches

    def _loss_fn(params, _obs, _actions, values, log_probs, advantages, targets, alive):
        # COLLECT ACTIONS AND LOG_PROBS FOR TRAJ ACTIONS
        dropout_key = jax.random.fold_in(key=train_state.key, data=j)
        _i_policy, _i_value = train_state.apply_fn(params, _obs, rngs={'dropout': dropout_key})
        _i_log_prob = _i_policy.log_prob(_actions)

        # CALCULATE VALUE LOSS
        value_pred_clipped = values + (
                _i_value - values
        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])

        value_losses = jnp.square(values - targets) * alive
        value_losses_clipped = jnp.square(value_pred_clipped - targets) * alive
        value_loss = (0.5 * jnp.maximum(value_losses, value_losses_clipped).sum() / (alive.sum() + 1e-8))

        # CALCULATE ACTOR LOSS
        ratio = jnp.exp(_i_log_prob - log_probs)
        gae_for_i = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        loss_actor1 = ratio * gae_for_i * alive
        loss_actor2 = (
                jnp.clip(
                    ratio,
                    1.0 - config["CLIP_EPS"],
                    1.0 + config["CLIP_EPS"],
                )
                * gae_for_i * alive
        )
        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
        loss_actor = loss_actor.sum() / (alive.sum() + 1e-8)
        entropy = (_i_policy.entropy() * alive).sum() / (alive.sum() + 1e-8)

        total_loss = (
                loss_actor
                + config["VF_COEF"] * value_loss
                - config["ENT_COEF_LISTENER"] * entropy
        )
        return total_loss, (value_loss, loss_actor, entropy)
 
    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True, allow_int=False)

    total_loss, grads = grad_fn(
        train_state.params,    # params don't change across minibatches, they are for the same agent.
        trans_batch_i.listener_obs[j],
        trans_batch_i.listener_action[j], 
        trans_batch_i.listener_value[j], 
        trans_batch_i.listener_log_prob[j],
        advantages_i[j], 
        targets_i[j],
        trans_batch_i.listener_alive[j]
    )
    
    train_state = train_state.apply_gradients(grads=grads)

    return train_state, total_loss

@jax.profiler.annotate_function
def update_minibatch_speaker(j, trans_batch_i, advantages_i, targets_i, train_state, config):
    # j is for iterating through minibatches

    def _loss_fn(params, _obs, _actions, values, log_probs, advantages, targets, alive):
        # COLLECT ACTIONS AND LOG_PROBS FOR TRAJ ACTIONS
        dropout_key = jax.random.fold_in(key=train_state.key, data=j)
        _i_policy, _i_value = train_state.apply_fn(params, _obs, rngs={'dropout': dropout_key})
        # _i_log_prob = jnp.sum(_i_policy.log_prob(_actions), axis=1) # Sum log-probs for individual pixels to get log-probs of whole image
        _i_log_prob = _i_policy.log_prob(_actions)

        # CALCULATE VALUE LOSS
        value_pred_clipped = values + (
                _i_value - values
        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])

        value_losses = jnp.square(values - targets) * alive
        value_losses_clipped = jnp.square(value_pred_clipped - targets) * alive
        value_loss = (0.5 * jnp.maximum(value_losses, value_losses_clipped).sum() / (alive.sum() + 1e-8))

        # CALCULATE ACTOR LOSS
        ratio = jnp.exp(_i_log_prob - log_probs)
        gae_for_i = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        loss_actor1 = ratio * gae_for_i * alive
        loss_actor2 = (
                jnp.clip(
                    ratio,
                    1.0 - config["CLIP_EPS"],
                    1.0 + config["CLIP_EPS"],
                )
                * gae_for_i * alive
        )
        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
        loss_actor = loss_actor.sum() / (alive.sum() + 1e-8)
        entropy = (_i_policy.entropy() * alive).sum() / (alive.sum() + 1e-8)

        total_loss = (
                loss_actor
                + config["VF_COEF"] * value_loss
                - config["ENT_COEF_SPEAKER"] * entropy
        )
        return total_loss, (value_loss, loss_actor, entropy)
 
    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True, allow_int=False)

    total_loss, grads = grad_fn(
        train_state.params,    # params don't change across minibatches, they are for the same agent.
        trans_batch_i.speaker_obs[j],
        trans_batch_i.speaker_action[j], 
        trans_batch_i.speaker_value[j], 
        trans_batch_i.speaker_log_prob[j],
        advantages_i[j], 
        targets_i[j],
        trans_batch_i.speaker_alive[j]
    )
    
    train_state = train_state.apply_gradients(grads=grads)

    return train_state, total_loss



@jax.profiler.annotate_function
def make_train(config):
    env = define_env(config)
    env = SimpSigGameLogWrapper(env)

    env_kwargs = config["ENV_KWARGS"]

    config["NUM_ACTORS"] = (env_kwargs["num_speakers"] + env_kwargs["num_listeners"]) * config["NUM_ENVS"]
    config["NUM_MINIBATCHES_LISTENER"] = config["NUM_STEPS"] // config["MINIBATCH_SIZE_LISTENER"]
    config["NUM_MINIBATCHES_SPEAKER"] = config["NUM_STEPS"] // config["MINIBATCH_SIZE_SPEAKER"]
    
    @jax.profiler.annotate_function
    def train(rng):
        # MAKE AGENTS
        rng, rng_s, rng_l = jax.random.split(rng, 3)    # rng_s for speakers, rng_l for listeners
        listener_rngs = jax.random.split(rng_l, config["ENV_KWARGS"]["num_listeners"] * config["NUM_ENVS"])   # Make an rng key for each listener
        speaker_rngs = jax.random.split(rng_s, config["ENV_KWARGS"]["num_speakers"] * config["NUM_ENVS"])   # Make an rng key for each speaker
        
        listeners_stuff = [initialize_listener(env, x_rng, config) for x_rng in listener_rngs]
        listener_networks, listener_train_states, listener_lr_funcs = zip(*listeners_stuff) # listener_lr_funcs is for logging only, it's not actually used directly by the optimizer
        
        speakers_stuff = [initialize_speaker(env, x_rng, config) for x_rng in speaker_rngs]
        speaker_networks, speaker_train_states, speaker_lr_funcs = zip(*speakers_stuff) # speaker_lr_funcs is for logging only, it's not actually used directly by the optimizer

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obs, log_env_state = env.reset(reset_rng, jnp.zeros((len(reset_rng))))  # log_env_state is a single variable, but each variable it has is actually batched
        
        # TRAIN LOOP
        def _update_step(runner_state, update_step, env, config):
            # runner_state is actually a tuple of runner_states, one per agent

            new_reset_rng = jax.random.split(runner_state[4], config["NUM_ENVS"]+1)
            last_obs, log_env_state = env.reset(new_reset_rng[:-1], jnp.ones((config["NUM_ENVS"])) * update_step)  # This should probably be a new rng each time, also there should be multiple envs!!! This function keeps using the same env.
            runner_state = (runner_state[0], runner_state[1], log_env_state, last_obs, new_reset_rng[-1])
            
            # COLLECT TRAJECTORIES
            runner_state, transition_batch = jax.lax.scan(lambda rs, _: env_step(rs, env, config), runner_state, None, config['NUM_STEPS'] + 1)
            # transition_batch is an instance of Transition with batched sub-objects
            # The shape of transition_batch is (num_steps, num_envs, ...) because it's the output of jax.lax.scan, which enumerates over steps

            # Instead of executing the agents on the final observation to get their values, we are simply going to ignore the last observation from traj_batch.
            # We'll need to get the final value in transition_batch and cut off the last index
            # We want to cleave off the final step, so it should go from shape (A, B, C) to shape (A-1, B, C)
            # trimmed_transition_batch = Transition(**{k: v[:-1, ...] for k, v in transition_batch._asdict().items()})
            trimmed_transition_batch = Transition(**{
                k: (v[1:, ...] if k in ('speaker_reward') else v[:-1, ...]) # We need to shift rewards for speakers over by 1 to the left. speaker gets a delayed reward.
                for k, v in transition_batch._asdict().items()
            })

            # CALCULATE ADVANTAGE #############
            listener_train_state, speaker_train_state, log_env_state, last_obs, rng = runner_state
            # listener_train_state is a tuple of TrainStates of length num_envs * env.num_listeners

            def _calculate_gae_listeners(trans_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    alive, value, reward = (
                        transition.listener_alive,
                        transition.listener_value,
                        transition.listener_reward,
                    )
                    delta = reward + config["GAMMA"] * next_value * alive - value
                    gae = delta + config["GAMMA"] * config["GAE_LAMBDA"] * alive * gae
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    trans_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + trans_batch.listener_value

            def _calculate_gae_speakers(trans_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    alive, value, reward = (
                        transition.speaker_alive,
                        transition.speaker_value,
                        transition.speaker_reward,
                    )
                    delta = reward + config["GAMMA"] * next_value * alive - value
                    gae = delta + config["GAMMA"] * config["GAE_LAMBDA"] * alive * gae
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    trans_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + trans_batch.speaker_value

            listener_advantages, listener_targets = _calculate_gae_listeners(trimmed_transition_batch, transition_batch.listener_value[-1])
            speaker_advantages, speaker_targets = _calculate_gae_speakers(trimmed_transition_batch, transition_batch.speaker_value[-1])
            
            ##########################

            def _update_a_listener(i, listener_train_states, listener_trans_batch, listener_advantages, listener_targets):                
                listener_train_state_i = listener_train_states[i]
                listener_advantages_i = listener_advantages.reshape((config["NUM_STEPS"], -1))[:, i].reshape((config["NUM_MINIBATCHES_LISTENER"], -1))
                listener_targets_i = listener_targets.reshape((config["NUM_STEPS"], -1))[:, i].reshape((config["NUM_MINIBATCHES_LISTENER"], -1))
                
                listener_trans_batch_i = Transition(
                    speaker_action=listener_trans_batch.speaker_action,
                    speaker_reward=listener_trans_batch.speaker_reward,
                    speaker_value=listener_trans_batch.speaker_value,
                    speaker_log_prob=listener_trans_batch.speaker_log_prob,
                    speaker_obs=listener_trans_batch.speaker_obs,
                    speaker_alive=listener_trans_batch.speaker_alive,
                    listener_action=jnp.float32(listener_trans_batch.listener_action.reshape((config["NUM_STEPS"], -1))[:, i].reshape((config["NUM_MINIBATCHES_LISTENER"], -1))),
                    listener_reward=listener_trans_batch.listener_reward.reshape((config["NUM_STEPS"], -1))[:, i].reshape((config["NUM_MINIBATCHES_LISTENER"], -1)),
                    listener_value=listener_trans_batch.listener_value.reshape((config["NUM_STEPS"], -1))[:, i].reshape((config["NUM_MINIBATCHES_LISTENER"], -1)),
                    listener_log_prob=listener_trans_batch.listener_log_prob.reshape((config["NUM_STEPS"], -1))[:, i].reshape((config["NUM_MINIBATCHES_LISTENER"], -1)),
                    listener_obs=listener_trans_batch.listener_obs.reshape((config["NUM_STEPS"], listener_trans_batch.listener_obs.shape[1]*listener_trans_batch.listener_obs.shape[2], -1))[:, i, :].reshape((config["NUM_MINIBATCHES_LISTENER"], -1, env_kwargs["image_dim"]**2)),
                    listener_alive=jnp.float32(listener_trans_batch.listener_alive.reshape((config["NUM_STEPS"], -1))[:, i].reshape((config["NUM_MINIBATCHES_LISTENER"], -1)))
                )

                # Iterate through batches
                new_listener_train_state_i, total_loss = jax.lax.scan(lambda train_state, i: update_minibatch_listener(i, listener_trans_batch_i, listener_advantages_i, listener_targets_i, train_state, config), listener_train_state_i, jnp.arange(config["NUM_MINIBATCHES_LISTENER"]))

                return new_listener_train_state_i, total_loss
            
            def _update_a_speaker(i, speaker_train_states, speaker_trans_batch, speaker_advantages, speaker_targets):                
                speaker_train_state_i = speaker_train_states[i]
                speaker_advantages_i = speaker_advantages.reshape((config["NUM_STEPS"], -1))[:, i].reshape((config["NUM_MINIBATCHES_SPEAKER"], -1))
                speaker_targets_i = speaker_targets.reshape((config["NUM_STEPS"], -1))[:, i].reshape((config["NUM_MINIBATCHES_SPEAKER"], -1))
                
                speaker_trans_batch_i = Transition(
                    speaker_action=speaker_trans_batch.speaker_action.reshape((config["NUM_STEPS"], env_kwargs["speaker_action_dim"], -1))[:, :, i].reshape((config["NUM_MINIBATCHES_SPEAKER"], -1, env_kwargs["speaker_action_dim"])),
                    speaker_reward=speaker_trans_batch.speaker_reward.reshape((config["NUM_STEPS"], -1))[:, i].reshape((config["NUM_MINIBATCHES_SPEAKER"], -1)),
                    speaker_value=speaker_trans_batch.speaker_value.reshape((config["NUM_STEPS"], -1))[:, i].reshape((config["NUM_MINIBATCHES_SPEAKER"], -1)),
                    speaker_log_prob=speaker_trans_batch.speaker_log_prob.reshape((config["NUM_STEPS"], -1))[:, i].reshape((config["NUM_MINIBATCHES_SPEAKER"], -1)),
                    speaker_obs=speaker_trans_batch.speaker_obs.reshape((config["NUM_STEPS"], -1))[:, i].reshape((config["NUM_MINIBATCHES_SPEAKER"], -1)),
                    speaker_alive=jnp.float32(speaker_trans_batch.speaker_alive.reshape((config["NUM_STEPS"], -1))[:, i].reshape((config["NUM_MINIBATCHES_SPEAKER"], -1))),
                    listener_action=speaker_trans_batch.listener_action,
                    listener_reward=speaker_trans_batch.listener_reward,
                    listener_value=speaker_trans_batch.listener_value,
                    listener_log_prob=speaker_trans_batch.listener_log_prob,
                    listener_obs=speaker_trans_batch.listener_obs,
                    listener_alive=speaker_trans_batch.listener_alive
                )

                # Iterate through batches
                new_speaker_train_state_i, total_loss = jax.lax.scan(lambda train_state, i: update_minibatch_speaker(i, speaker_trans_batch_i, speaker_advantages_i, speaker_targets_i, train_state, config), speaker_train_state_i, jnp.arange(config["NUM_MINIBATCHES_SPEAKER"]))

                return new_speaker_train_state_i, total_loss

            listener_map_outputs = tuple(map(lambda i: _update_a_listener(i, listener_train_state, trimmed_transition_batch, listener_advantages, listener_targets), range(len(listener_rngs))))
            listener_train_state = tuple([lmo[0] for lmo in listener_map_outputs])
            
            speaker_map_outputs = tuple(map(lambda i: _update_a_speaker(i, speaker_train_state, trimmed_transition_batch, speaker_advantages, speaker_targets), range(len(speaker_rngs))))
            speaker_train_state = tuple([lmo[0] for lmo in speaker_map_outputs])
            runner_state = (listener_train_state, speaker_train_state, log_env_state, last_obs, rng)

            # Below is just for logging

            listener_loss = tuple([lmo[1] for lmo in listener_map_outputs])
            speaker_loss = tuple([lmo[1] for lmo in speaker_map_outputs])

            speaker_current_lr = jnp.array([speaker_lr_funcs[i](speaker_train_state[i].opt_state[1][0].count) for i in range(len(speaker_train_state))])
            listener_current_lr = jnp.array([listener_lr_funcs[i](listener_train_state[i].opt_state[1][0].count) for i in range(len(listener_train_state))])
            speaker_examples = jax.lax.cond((update_step + 1) % config["SPEAKER_EXAMPLE_LOGGING_ITER"] == 0, lambda _: get_speaker_examples(runner_state, env, config), lambda _: jnp.zeros((env_kwargs["num_speakers"], env_kwargs["num_classes"], env_kwargs["image_dim"], env_kwargs["image_dim"])), operand=None)
            speaker_images = env._env.speaker_action_transform(trimmed_transition_batch.speaker_action[-2].reshape((len(speaker_train_state), -1))).reshape((len(speaker_train_state), -1, env_kwargs["image_dim"], env_kwargs["image_dim"]))   # TODO: This code is not robust to more than 1 env

            def wandb_callback(metrics):
                ll, sl, tb, les, speaker_lr, listener_lr, speaker_exs, speaker_imgs, u_step = metrics
                lr = tb.listener_reward
                sr = tb.speaker_reward
                logp = tb.listener_log_prob
                lv = tb.listener_value
                sv = tb.speaker_value

                listener_actions = tb.listener_action
                listener_obs = tb.listener_obs
                speaker_actions = tb.speaker_action
                speaker_obs = tb.speaker_obs

                metric_dict = {}

                metric_dict.update({"env/avg_num_speaker_images": jnp.mean(les.env_state.requested_num_speaker_images)})

                listener_images = listener_obs[-1, 0, :, :, :].reshape((-1, 1, env_kwargs["image_dim"], env_kwargs["image_dim"]))
                
                listener_images = make_grid(torch.tensor(listener_images), nrow=env_kwargs["num_listeners"])
                final_listener_images = wandb.Image(listener_images, caption=f"classified as: {str(listener_actions[-1, 0, :].ravel())}")

                speaker_images = make_grid(torch.tensor(speaker_imgs), nrow=env_kwargs["num_speakers"])
                final_speaker_images = wandb.Image(speaker_images, caption=f"tried generating: {str(speaker_obs[-2, 0, :].ravel())}")
                
                metric_dict.update({"env/speaker_images": final_speaker_images})
                metric_dict.update({"env/last_listener_obs": final_listener_images})
                metric_dict.update({f"env/speaker_labels/speaker {i}": les.env_state.speaker_labels[:, i].item() for i in range(les.env_state.speaker_labels.shape[-1])})

                # agent, total_loss, (value_loss, loss_actor, entropy)
                metric_dict.update({f"loss/total loss/listener {i}": jnp.mean(ll[i][0]).item() for i in range(len(ll))})
                metric_dict.update({f"loss/value loss/listener {i}": jnp.mean(ll[i][1][0]).item() for i in range(len(ll))})
                metric_dict.update({f"loss/actor loss/listener {i}": jnp.mean(ll[i][1][1]).item() for i in range(len(ll))})
                metric_dict.update({f"loss/entropy/listener {i}": jnp.mean(ll[i][1][2]).item() for i in range(len(ll))})
                # loss_dict["average_loss"] = jnp.mean(ll)
                # metric_dict.udpate({"loss/average loss for listeners"})

                metric_dict.update({f"loss/total loss/speaker {i}": jnp.mean(sl[i][0]).item() for i in range(len(sl))})
                metric_dict.update({f"loss/value loss/speaker {i}": jnp.mean(sl[i][1][0]).item() for i in range(len(sl))})
                metric_dict.update({f"loss/actor loss/speaker {i}": jnp.mean(sl[i][1][1]).item() for i in range(len(sl))})
                metric_dict.update({f"loss/entropy/speaker {i}": jnp.mean(sl[i][1][2]).item() for i in range(len(sl))})

                lr = lr.T
                # metric_dict.update({f"cumulative reward/listener {i}": jnp.sum(r[i]).item() for i in range(len(r))})
                metric_dict.update({f"reward/mean reward/listener {i}": jnp.mean(lr[i]).item() for i in range(len(lr))})
                metric_dict.update({"reward/mean reward/all listeners": jnp.mean(lr).item()})

                sr = sr.T
                metric_dict.update({f"reward/mean reward/speaker {i}": jnp.mean(sr[i]).item() for i in range(len(sr))})
                metric_dict.update({"reward/mean reward/all speakers": jnp.mean(sr).item()})
                
                random_expected_reward = (env_kwargs["num_classes"] - 1) * env_kwargs["reward_failure"] + env_kwargs["reward_success"]
                if random_expected_reward != 0:
                    metric_dict.update({f"reward/mean reward over random/listener {i}": jnp.mean(lr[i]).item()/random_expected_reward for i in range(len(lr))})
                    metric_dict.update({f"reward/mean reward over random/speaker {i}": jnp.mean(sr[i]).item()/random_expected_reward for i in range(len(sr))})
                    # Average reward over random - based on (num_classes-1)*fail_reward + success_reward
                
                optimal_expected_reward = env_kwargs["reward_success"]
                if optimal_expected_reward != 0:
                    metric_dict.update({f"reward/mean reward over optimal/listener {i}": jnp.mean(lr[i]).item()/optimal_expected_reward for i in range(len(lr))})
                    metric_dict.update({f"reward/mean reward over optimal/speaker {i}": jnp.mean(sr[i]).item()/optimal_expected_reward for i in range(len(sr))})
                    # Average reward over optimal - based on success_reward
                
                logp = logp.T
                lv = lv.T
                metric_dict.update({f"predictions/mean action log probs/listener {i}": jnp.mean(logp[i]).item() for i in range(len(logp))})
                metric_dict.update({f"predictions/mean state value estimate/listener {i}": jnp.mean(lv[i]).item() for i in range(len(lv))})

                metric_dict.update({"learning rate/average speaker": jnp.mean(speaker_lr).item()})
                metric_dict.update({"learning rate/average listener": jnp.mean(listener_lr).item()})
                
                if (u_step + 1) % config["SPEAKER_EXAMPLE_LOGGING_ITER"] == 0:
                    speaker_example_images = make_grid(torch.tensor(speaker_exs.reshape((-1, 1, env_kwargs["image_dim"], env_kwargs["image_dim"]))), nrow=env_kwargs["num_classes"])
                    final_speaker_example_images = wandb.Image(speaker_example_images, caption="speaker_examples")
                    metric_dict.update({"env/speaker_examples": final_speaker_example_images})
                
                wandb.log(metric_dict)
            jax.experimental.io_callback(wandb_callback, None, (listener_loss, speaker_loss, trimmed_transition_batch, log_env_state, speaker_current_lr, listener_current_lr, speaker_examples, speaker_images, update_step))
            
            return runner_state, update_step + 1

        rng, _rng = jax.random.split(rng)
        runner_state = (listener_train_states, speaker_train_states, log_env_state, obs, _rng)

        partial_update_fn = partial(_update_step, env=env, config=config)
        runner_state, traj_batch = jax.lax.scan( # Perform the update step for a specified number of updates and update the runner state
            partial_update_fn, runner_state, jnp.arange(config['UPDATE_EPOCHS'])
        )

        return {"runner_state": runner_state, "traj_batch": traj_batch}

    return train


@hydra.main(version_base=None, config_path="config", config_name="default")
def main(config):
    config = OmegaConf.to_container(
        config, resolve=True, throw_on_missing=True
    )
    wandb.init(
        entity=config["ENTITY"],
        project=config["PROJECT"],
        tags=["main"],
        config=config,
        mode=config["WANDB_MODE"],
        save_code=True
    )
    # with jax.profiler.trace("/tmp/jax-trace", create_perfetto_link=True):
    rng = jax.random.PRNGKey(51)
    # train_jit = jax.jit(make_train(config), device=jax.devices()[0]) # The environment may or may not be jittable.
    train = make_train(config)
    out = train(rng)
    print("Done")


@jax.profiler.annotate_function
def test_rollout_execution(config, rng):    # I don't think this funciton works anymore. I've changed a lot.
    env = define_env(config)
    env = SimpSigGameLogWrapper(env)

    config["NUM_ACTORS"] = (config["ENV_KWARGS"]["num_speakers"] + config["ENV_KWARGS"]["num_listeners"]) * config["NUM_ENVS"]
    config["NUM_MINIBATCHES"] = config["NUM_STEPS"] // config["MINIBATCH_SIZE_LISTENER"]

    # MAKE AGENTS
    rng, rng_s, rng_l = jax.random.split(rng, 3)    # rng_s for speakers, rng_l for listeners
    listener_rngs = jax.random.split(rng_l, config["ENV_KWARGS"]["num_listeners"] * config["NUM_ENVS"])   # Make an rng key for each listener
    speaker_rngs = jax.random.split(rng_s, config["ENV_KWARGS"]["num_speakers"] * config["NUM_ENVS"])   # Make an rng key for each speaker
    
    listeners_stuff = [initialize_listener(env, x_rng, config) for x_rng in listener_rngs]
    listener_networks, listener_train_states = zip(*listeners_stuff)
    
    speakers_stuff = [initialize_speaker(env, x_rng, config) for x_rng in speaker_rngs]
    speaker_networks, speaker_train_states = zip(*speakers_stuff)
    

    # INIT ENV
    rng, _rng = jax.random.split(rng)
    reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
    obs, log_env_state = env.reset(reset_rng, jnp.zeros((len(reset_rng))))  # log_env_state is a single variable, but each variable it has is actually batched

    rng, _rng = jax.random.split(rng)
    runner_state = (listener_train_states, speaker_train_states, log_env_state, obs, _rng)

    # runner_state, transition = env_step(runner_state, env, config)    # This was for testing a single env_step
    runner_state, traj_batch = jax.lax.scan(lambda rs, _: env_step(rs, env, config), runner_state, None, config['NUM_STEPS']) # This is if everything is working
    # traj_batch is a Transition with sub-objects of shape (num_steps, num_envs, ...). It represents a rollout.
    
    return {"runner_state": runner_state, "traj_batch": traj_batch}


@hydra.main(version_base=None, config_path="config", config_name="test")
def test(config):
    config = OmegaConf.to_container(
        config, resolve=True, throw_on_missing=True
    )
    wandb.init(
        entity=config["ENTITY"],
        project=config["PROJECT"],
        tags=["test"],
        config=config,
        mode=config["WANDB_MODE"],
        save_code=True
    )
    # with jax.profiler.trace("/tmp/jax-trace", create_perfetto_link=True):
    rng = jax.random.PRNGKey(50)
    out = test_rollout_execution(config, rng)
    print(out['runner_state'])


if __name__ == "__main__":
    main()
    # with jax.profiler.trace("/tmp/jax-trace", create_perfetto_link=True):
    #     test()
