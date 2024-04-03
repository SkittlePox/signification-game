"""
Based on PureJaxRL Implementation of PPO
"""
from functools import partial
import jax
import jax.numpy as jnp
import flax.linen as nn
import chex
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple, Any, Dict, Tuple
from flax.training.train_state import TrainState
import distrax
from jaxmarl.wrappers.baselines import LogWrapper, LogEnvState
import jaxmarl
import wandb
import functools
import hydra
import torch
from torchvision.utils import make_grid
from torchvision.datasets import MNIST
from omegaconf import OmegaConf
from simplified_signification_game import SimplifiedSignificationGame, State


class SimpSigGameLogWrapper(LogWrapper):
    @partial(jax.jit, static_argnums=(0,))
    def reset(self, keys: chex.PRNGKey, epochs: Sequence[int]) -> Tuple[chex.Array, State]:
        obs, env_state = jax.vmap(self._env.reset)(keys, epochs)
        state = jax.vmap(lambda e_state: LogEnvState(
            e_state,
            jnp.zeros((self._env.num_agents,)),
            jnp.zeros((self._env.num_agents,)),
            jnp.zeros((self._env.num_agents,)),
            jnp.zeros((self._env.num_agents,)),
        ))(env_state)
        return obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        keys: chex.PRNGKey,
        state: LogEnvState,
        action,
    ) -> Tuple[chex.Array, LogEnvState, float, bool, dict]:
        obs, env_state, reward, done, info = jax.vmap(self._env.step_env)(
            keys, state.env_state, action
        )
        ep_done = done["__all__"]
        new_episode_return = state.episode_returns + self._batchify_floats(reward).T
        new_episode_length = state.episode_lengths + 1
        old_ep_done = ep_done

        ep_done = jnp.repeat(ep_done[:, None], new_episode_return.shape[1], axis=1)   # This is a major hotfix. Right now ep_done is one per env, as opposed to one per agent per env. This line extrapolates the one per env to one per agent per env.
        state = LogEnvState(
            env_state=env_state,
            episode_returns=new_episode_return * (1 - ep_done),
            episode_lengths=new_episode_length * (1 - ep_done),
            returned_episode_returns=state.returned_episode_returns * (1 - ep_done)
            + new_episode_return * ep_done,
            returned_episode_lengths=state.returned_episode_lengths * (1 - ep_done)
            + new_episode_length * ep_done,
        )
        if self.replace_info:
            info = {}
        info["returned_episode_returns"] = state.returned_episode_returns
        info["returned_episode_lengths"] = state.returned_episode_lengths
        # info["returned_episode"] = jnp.full((self._env.num_agents,), old_ep_done) # This doesn't work for some reason
        return obs, state, reward, done, info


# There should be two kinds of ActorCritics, one for listeners and one for speakers. For now, this will be for listeners.
class ActorCriticListenerConv(nn.Module):
    action_dim: Sequence[int]
    image_dim: Sequence[int]
    config: Dict

    @nn.compact
    def __call__(self, x):
        x = x.reshape(-1, self.image_dim, self.image_dim, 1)  # Assuming x is flat, and image_dim is [height, width]

        # Convolutional layers
        x = nn.Conv(features=32, kernel_size=(3, 3), strides=(1, 1), padding='SAME')(x)
        x = nn.sigmoid(x)
        x = nn.Conv(features=64, kernel_size=(3, 3), strides=(1, 1), padding='SAME')(x)
        x = nn.sigmoid(x)
        x = x.reshape((x.shape[0], -1))  # Flatten
        
        # Embedding Layer
        embedding = nn.Dense(512, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        embedding = nn.sigmoid(embedding)
        embedding = nn.Dense(512, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(embedding)
        embedding = nn.sigmoid(embedding)

        # Actor Layer
        actor_mean = nn.Dense(512, kernel_init=orthogonal(2), bias_init=constant(0.0))(embedding)
        actor_mean = nn.sigmoid(actor_mean)
        actor_mean = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(actor_mean)

        # Action Logits
        # unavail_actions = 1 - avail_actions
        # action_logits = actor_mean - (unavail_actions * 1e10)
        pi = distrax.Categorical(logits=actor_mean)

        # Critic Layer
        critic = nn.Dense(512, kernel_init=orthogonal(2), bias_init=constant(0.0))(embedding)
        critic = nn.sigmoid(critic)
        critic = nn.Dense(512, kernel_init=orthogonal(2), bias_init=constant(0.0))(critic)
        critic = nn.sigmoid(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(critic)

        return pi, jnp.squeeze(critic, axis=-1)


class ActorCriticListenerDense(nn.Module):
    action_dim: Sequence[int]
    image_dim: Sequence[int]
    config: Dict

    @nn.compact
    def __call__(self, x):
        obs = x
        # Embedding Layer
        embedding = nn.Dense(512, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(obs)
        embedding = nn.sigmoid(embedding)

        # Actor Layer
        actor_mean = nn.Dense(512, kernel_init=orthogonal(2), bias_init=constant(0.0))(embedding)
        actor_mean = nn.sigmoid(actor_mean)
        actor_mean = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(actor_mean)

        # Action Logits
        # unavail_actions = 1 - avail_actions
        # action_logits = actor_mean - (unavail_actions * 1e10)
        pi = distrax.Categorical(logits=actor_mean)

        # Critic Layer
        critic = nn.Dense(512, kernel_init=orthogonal(2), bias_init=constant(0.0))(embedding)
        critic = nn.sigmoid(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(critic)

        return pi, jnp.squeeze(critic, axis=-1)
    

class ActorCriticSpeaker(nn.Module):
    latent_dim: int
    num_classes: int
    config: Dict

    @nn.compact
    def __call__(self, y):
        y = nn.Embed(self.num_classes, self.latent_dim)(y)
        # y = jnp.squeeze(y, axis=(0))
        # z = jnp.concatenate([z, y], axis=-1)
        z = nn.Dense(7 * 7 * 256)(y)
        z = nn.relu(z)
        z = z.reshape((-1, 7, 7, 256))
        z = nn.ConvTranspose(128, kernel_size=(4, 4), strides=(2, 2), padding='SAME')(z)
        z = nn.relu(z)
        z = nn.ConvTranspose(64, kernel_size=(4, 4), strides=(2, 2), padding='SAME')(z)
        z = nn.relu(z)
        z = nn.ConvTranspose(1, kernel_size=(3, 3), padding='SAME')(z)
        z = jnp.squeeze(z, axis=-1)  # Remove the channel dimension
        z = jnp.reshape(z, (-1, 28 * 28))  # Flatten the image

        # Actor Mean
        actor_mean = nn.Dense(28 * 28)(z)
        actor_mean = nn.sigmoid(actor_mean)  # Apply sigmoid to squash outputs between 0 and 1

        # Actor Standard Deviation
        actor_std = nn.Dense(28 * 28)(z)
        actor_std = nn.softplus(actor_std) + 1e-6  # Ensure positive standard deviation

        # Create a multivariate normal distribution with diagonal covariance matrix
        pi = distrax.MultivariateNormalDiag(loc=actor_mean, scale_diag=actor_std)

        # Critic
        critic = nn.Dense(512, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(z)
        critic = nn.sigmoid(critic)
        critic = nn.Dense(512, kernel_init=orthogonal(2), bias_init=constant(0.0))(critic)
        critic = nn.sigmoid(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(critic)

        return pi, jnp.squeeze(critic, axis=-1)
    

class ActorCriticSpeakerBetaDist(nn.Module):
    latent_dim: int
    num_classes: int
    config: Dict

    @nn.compact
    def __call__(self, y):
        y = nn.Embed(self.num_classes, self.latent_dim)(y)
        # y = jnp.squeeze(y, axis=(0))
        # z = jnp.concatenate([z, y], axis=-1)
        z = nn.Dense(7 * 7 * 256)(y)
        z = nn.relu(z)
        z = z.reshape((-1, 7, 7, 256))
        z = nn.ConvTranspose(128, kernel_size=(4, 4), strides=(2, 2), padding='SAME')(z)
        z = nn.relu(z)
        z = nn.ConvTranspose(64, kernel_size=(4, 4), strides=(2, 2), padding='SAME')(z)
        z = nn.relu(z)
        z = nn.ConvTranspose(1, kernel_size=(3, 3), padding='SAME')(z)
        z = jnp.squeeze(z, axis=-1)  # Remove the channel dimension
        z = jnp.reshape(z, (-1, 28 * 28))  # Flatten the image

        # Actor Alpha
        actor_alpha = nn.Dense(28 * 28)(z)
        actor_alpha = nn.softplus(actor_alpha) + 1e-6  # Ensure positive alpha

        # Actor Beta
        actor_beta = nn.Dense(28 * 28)(z)
        actor_beta = nn.softplus(actor_beta) + 1e-6  # Ensure positive beta

        # Create a beta distribution
        pi = distrax.Beta(alpha=actor_alpha, beta=actor_beta)

        # Critic
        critic = nn.Dense(512, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(z)
        critic = nn.sigmoid(critic)
        critic = nn.Dense(512, kernel_init=orthogonal(2), bias_init=constant(0.0))(critic)
        critic = nn.sigmoid(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(critic)

        return pi, jnp.squeeze(critic, axis=-1)


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
def initialize_listener(env, rng, config, learning_rate):
    if config["ENV_LISTENER_ARCH"] == 'conv':
        listener_network = ActorCriticListenerConv(action_dim=config["ENV_KWARGS"]["num_classes"], image_dim=config["ENV_KWARGS"]["image_dim"], config=config)
    elif config["ENV_LISTENER_ARCH"] == 'dense':
        listener_network = ActorCriticListenerDense(action_dim=config["ENV_KWARGS"]["num_classes"], image_dim=config["ENV_KWARGS"]["image_dim"], config=config)

    rng, _rng = jax.random.split(rng)
    init_x = jnp.zeros(
            (1, config["NUM_ENVS"], config["ENV_KWARGS"]["image_dim"]**2)
        )
    network_params = listener_network.init(_rng, init_x)    # I'm not sure how this works, I need to control the size of the inputs and the size of the outputs
    if config["ANNEAL_LR_LISTENER"]:
        tx = optax.chain(
            optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
            optax.adam(learning_rate=learning_rate, eps=1e-5),
        )
    else:
        tx = optax.chain(optax.clip_by_global_norm(config["MAX_GRAD_NORM"]), optax.adam(config["LR_LISTENER"], eps=1e-5))
    
    train_state = TrainState.create(
        apply_fn=listener_network.apply,
        params=network_params,
        tx=tx,
    )
    return listener_network, train_state

@jax.profiler.annotate_function
def initialize_speaker(env, rng, config, learning_rate):
    if config["ENV_SPEAKER_ARCH"] == 'gauss':
        speaker_network = ActorCriticSpeaker(latent_dim=32, num_classes=config["ENV_KWARGS"]["num_classes"], config=config)
    elif config["ENV_SPEAKER_ARCH"] == 'beta':
        speaker_network = ActorCriticSpeakerBetaDist(latent_dim=32, num_classes=config["ENV_KWARGS"]["num_classes"], config=config)

    rng, _rng = jax.random.split(rng)
    init_y = jnp.zeros(
            (1, config["NUM_ENVS"], 1),
            dtype=jnp.int32
        )
    z = jax.random.normal(rng, (1, config["NUM_ENVS"], 32))
    network_params = speaker_network.init(_rng, init_y)
    if config["ANNEAL_LR_SPEAKER"]:
        tx = optax.chain(
            optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
            optax.adam(learning_rate=learning_rate, eps=1e-5),
        )
    else:
        tx = optax.chain(optax.clip_by_global_norm(config["MAX_GRAD_NORM"]), optax.adam(config["LR_SPEAKER"], eps=1e-5))
    
    train_state = TrainState.create(
        apply_fn=speaker_network.apply,
        params=network_params,
        tx=tx,
    )
    return speaker_network, train_state


@jax.profiler.annotate_function
def define_env(config):
    if config["ENV_DATASET"] == 'mnist':        
        from utils import to_jax

        mnist_dataset = MNIST('/tmp/mnist/', download=True)
        images, labels = to_jax(mnist_dataset, num_datapoints=config["ENV_NUM_DATAPOINTS"])  # This should also be in ENV_KWARGS
        images = images.astype('float32') / 255.0
        
        # env = SimplifiedSignificationGame(num_speakers, num_listeners, num_channels, num_classes, channel_ratio_fn=ret_0, dataset=(images, labels), image_dim=28, **config["ENV_KWARGS"])
        env = SimplifiedSignificationGame(**config["ENV_KWARGS"], dataset=(images, labels))
        return env

@jax.profiler.annotate_function
def execute_individual_listener(__rng, _listener_train_state_i, _listener_obs_i):
    _listener_obs_i = _listener_obs_i.ravel()
    policy, value = _listener_train_state_i.apply_fn(_listener_train_state_i.params, _listener_obs_i)
    action = policy.sample(seed=__rng)
    log_prob = policy.log_prob(action)
    return action, log_prob, value

@jax.profiler.annotate_function
def execute_individual_speaker(__rng, _speaker_train_state_i, _speaker_obs_i):
    # z = jax.random.normal(__rng, (1, 1, 32))   # I should be splitting this rng again
    # TODO: This looks terrible, I'm sure there's a better way
    _speaker_obs_i = jnp.expand_dims(_speaker_obs_i, axis=(0))
    _speaker_obs_i = jnp.expand_dims(_speaker_obs_i, axis=(0))
    _speaker_obs_i = jnp.expand_dims(_speaker_obs_i, axis=(0))
    policy, value = _speaker_train_state_i.apply_fn(_speaker_train_state_i.params, _speaker_obs_i)
    action = policy.sample(seed=__rng)
    log_prob = policy.log_prob(action)
    return jnp.clip(action, a_min=0.0, a_max=1.0), log_prob, value  # TODO: Clipping may be a bad idea, might want to reparameterize to a beta distribution instead of a multivariate

# @jax.jit
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
    # TODO: This should be cleaned up, look at how the speaker outputs are sorted/aggregated
    listener_outputs = [execute_individual_listener(*args) for args in zip(env_rngs, listener_train_states, listener_obs)]
    l_a = jnp.array([jnp.array([*o]) for o in listener_outputs])
    listener_action = jnp.asarray(l_a[:, 0], jnp.int32)
    listener_log_prob = l_a[:, 1]
    listener_value = l_a[:, 2].reshape(config["NUM_ENVS"], -1)

    listener_action = listener_action.reshape(config["NUM_ENVS"], -1)
    listener_log_prob = listener_log_prob.reshape(config["NUM_ENVS"], -1)

    # COLLECT SPEAKER ACTIONS
    speaker_outputs = [execute_individual_speaker(*args) for args in zip(env_rngs, speaker_train_states, speaker_obs)]
    speaker_action = jnp.array([o[0] for o in speaker_outputs]).reshape(config["NUM_ENVS"], -1, env_kwargs["image_dim"], env_kwargs["image_dim"])
    speaker_log_prob = jnp.array([o[1] for o in speaker_outputs]).reshape(config["NUM_ENVS"], -1)
    speaker_value = jnp.array([o[2] for o in speaker_outputs]).reshape(config["NUM_ENVS"], -1)

    # TODO: This can probably be deleted soon
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
def test_rollout_execution(config, rng):
    env = define_env(config)
    env = SimpSigGameLogWrapper(env)

    config["NUM_ACTORS"] = (config["ENV_KWARGS"]["num_speakers"] + config["ENV_KWARGS"]["num_listeners"]) * config["NUM_ENVS"]
    config["NUM_MINIBATCHES"] = config["NUM_STEPS"] // config["MINIBATCH_SIZE"]
    
    # For the learning rate
    def linear_schedule(count):
        frac = 1.0 - (count // config["UPDATE_EPOCHS"])   # I don't know exactly how this works.
        # jax.debug.print(str(count))
        return config["LR_LISTENER"] * frac

    # MAKE AGENTS
    rng, rng_s, rng_l = jax.random.split(rng, 3)    # rng_s for speakers, rng_l for listeners
    listener_rngs = jax.random.split(rng_l, config["ENV_KWARGS"]["num_listeners"] * config["NUM_ENVS"])   # Make an rng key for each listener
    speaker_rngs = jax.random.split(rng_s, config["ENV_KWARGS"]["num_speakers"] * config["NUM_ENVS"])   # Make an rng key for each speaker
    
    listeners_stuff = [initialize_listener(env, x_rng, config, linear_schedule) for x_rng in listener_rngs]
    listener_networks, listener_train_states = zip(*listeners_stuff)
    
    speakers_stuff = [initialize_speaker(env, x_rng, config, linear_schedule) for x_rng in speaker_rngs]
    speaker_networks, speaker_train_states = zip(*speakers_stuff)
    

    # INIT ENV
    rng, _rng = jax.random.split(rng)
    reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
    obs, log_env_state = env.reset(reset_rng, jnp.zeros((len(reset_rng))))  # log_env_state is a single variable, but each variable it has is actually batched

    """
    init_transition = Transition( # This is no longer needed, but it may be helpful to know what types and shapes things are in the future.
        jnp.zeros((config["NUM_ENVS"], env.num_agents), dtype=bool),
        jnp.zeros((config["NUM_ENVS"], max(env.num_speakers, 1), env.image_dim, env.image_dim), dtype=jnp.float32),
        jnp.zeros((config["NUM_ENVS"], max(env.num_listeners, 1)), dtype=jnp.int32),
        jnp.zeros((config["NUM_ENVS"], env.num_agents), dtype=jnp.float32),     # rewards
        jnp.zeros((config["NUM_ENVS"], env.num_agents), dtype=jnp.float32),     # values
        jnp.zeros((config["NUM_ENVS"], max(env.num_speakers, 1), env.image_dim, env.image_dim), dtype=jnp.float32),
        jnp.zeros((config["NUM_ENVS"], max(env.num_listeners, 1)), dtype=jnp.float32),
        obs[0],
        obs[1]
    )
    """

    rng, _rng = jax.random.split(rng)
    runner_state = (listener_train_states, speaker_train_states, log_env_state, obs, _rng)

    # runner_state, transition = env_step(runner_state, env, config)    # This was for testing a single env_step
    runner_state, traj_batch = jax.lax.scan(lambda rs, _: env_step(rs, env, config), runner_state, None, config['NUM_STEPS']) # This is if everything is working
    # traj_batch is a Transition with sub-objects of shape (num_steps, num_envs, ...). It represents a rollout.
    
    return {"runner_state": runner_state, "traj_batch": traj_batch}

@jax.profiler.annotate_function
def update_minibatch_listener(j, trans_batch_i, advantages_i, targets_i, train_state, config):
    # j is for iterating through minibatches

    def _loss_fn(params, _obs, _actions, values, log_probs, advantages, targets):
        # COLLECT ACTIONS AND LOG_PROBS FOR TRAJ ACTIONS
        _i_policy, _i_value = train_state.apply_fn(params, _obs)
        _i_log_prob = _i_policy.log_prob(_actions)

        # CALCULATE VALUE LOSS
        value_pred_clipped = values + (
                _i_value - values
        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])

        value_losses = jnp.square(values - targets)
        value_losses_clipped = jnp.square(value_pred_clipped - targets)
        value_loss = (0.5 * jnp.maximum(value_losses, value_losses_clipped).mean())

        # CALCULATE ACTOR LOSS
        ratio = jnp.exp(_i_log_prob - log_probs)
        gae_for_i = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        loss_actor1 = ratio * gae_for_i
        loss_actor2 = (
                jnp.clip(
                    ratio,
                    1.0 - config["CLIP_EPS"],
                    1.0 + config["CLIP_EPS"],
                )
                * gae_for_i
        )
        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
        loss_actor = loss_actor.mean()
        entropy = _i_policy.entropy().mean()

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
        targets_i[j]
    )
    
    train_state = train_state.apply_gradients(grads=grads)

    return train_state, total_loss

@jax.profiler.annotate_function
def update_minibatch_speaker(j, trans_batch_i, advantages_i, targets_i, train_state, config):
    # j is for iterating through minibatches

    def _loss_fn(params, _obs, _actions, values, log_probs, advantages, targets):
        # COLLECT ACTIONS AND LOG_PROBS FOR TRAJ ACTIONS
        _i_policy, _i_value = train_state.apply_fn(params, _obs)
        _i_log_prob = _i_policy.log_prob(_actions)

        # CALCULATE VALUE LOSS
        value_pred_clipped = values + (
                _i_value - values
        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])

        value_losses = jnp.square(values - targets)
        value_losses_clipped = jnp.square(value_pred_clipped - targets)
        value_loss = (0.5 * jnp.maximum(value_losses, value_losses_clipped).mean())

        # CALCULATE ACTOR LOSS
        ratio = jnp.exp(_i_log_prob - log_probs)
        gae_for_i = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        loss_actor1 = ratio * gae_for_i
        loss_actor2 = (
                jnp.clip(
                    ratio,
                    1.0 - config["CLIP_EPS"],
                    1.0 + config["CLIP_EPS"],
                )
                * gae_for_i
        )
        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
        loss_actor = loss_actor.mean()
        entropy = _i_policy.entropy().mean()

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
        targets_i[j]
    )
    
    train_state = train_state.apply_gradients(grads=grads)

    return train_state, total_loss



@jax.profiler.annotate_function
def make_train(config):
    env = define_env(config)
    env = SimpSigGameLogWrapper(env)

    env_kwargs = config["ENV_KWARGS"]

    config["NUM_ACTORS"] = (env_kwargs["num_speakers"] + env_kwargs["num_listeners"]) * config["NUM_ENVS"]
    config["NUM_MINIBATCHES"] = config["NUM_STEPS"] // config["MINIBATCH_SIZE"]
    
    # For the learning rate
    def linear_schedule(count):
        frac = 1.0 - (count // config["UPDATE_EPOCHS"])   # I don't know exactly how this works.
        # jax.debug.print(str(count))
        return config["LR_LISTENER"] * frac

    @jax.profiler.annotate_function
    def train(rng):
        # MAKE AGENTS
        rng, rng_s, rng_l = jax.random.split(rng, 3)    # rng_s for speakers, rng_l for listeners
        listener_rngs = jax.random.split(rng_l, config["ENV_KWARGS"]["num_listeners"] * config["NUM_ENVS"])   # Make an rng key for each listener
        speaker_rngs = jax.random.split(rng_s, config["ENV_KWARGS"]["num_speakers"] * config["NUM_ENVS"])   # Make an rng key for each speaker
        
        listeners_stuff = [initialize_listener(env, x_rng, config, linear_schedule) for x_rng in listener_rngs]
        listener_networks, listener_train_states = zip(*listeners_stuff)
        
        speakers_stuff = [initialize_speaker(env, x_rng, config, linear_schedule) for x_rng in speaker_rngs]
        speaker_networks, speaker_train_states = zip(*speakers_stuff)

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obs, log_env_state = env.reset(reset_rng, jnp.zeros((len(reset_rng))))  # log_env_state is a single variable, but each variable it has is actually batched
        
        # TRAIN LOOP
        def _update_step(runner_state, update_step, env, config):
            # runner_state is actually a tuple of runner_states, one per agent

            last_obs, log_env_state = env.reset(reset_rng, jnp.ones((len(reset_rng))) * update_step)  # This should probably be a new rng each time, also there should be multiple envs!!! This function keeps using the same env.
            runner_state = (runner_state[0], runner_state[1], log_env_state, last_obs, runner_state[4])
            
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
                listener_advantages_i = listener_advantages.reshape((config["NUM_STEPS"], -1))[:, i].reshape((config["NUM_MINIBATCHES"], -1))
                listener_targets_i = listener_targets.reshape((config["NUM_STEPS"], -1))[:, i].reshape((config["NUM_MINIBATCHES"], -1))
                
                listener_trans_batch_i = Transition(
                    speaker_action=listener_trans_batch.speaker_action,
                    speaker_reward=listener_trans_batch.speaker_reward,
                    speaker_value=listener_trans_batch.speaker_value,
                    speaker_log_prob=listener_trans_batch.speaker_log_prob,
                    speaker_obs=listener_trans_batch.speaker_obs,
                    speaker_alive=listener_trans_batch.speaker_alive,
                    listener_action=jnp.float32(listener_trans_batch.listener_action.reshape((config["NUM_STEPS"], -1))[:, i].reshape((config["NUM_MINIBATCHES"], -1))),
                    listener_reward=listener_trans_batch.listener_reward.reshape((config["NUM_STEPS"], -1))[:, i].reshape((config["NUM_MINIBATCHES"], -1)),
                    listener_value=listener_trans_batch.listener_value.reshape((config["NUM_STEPS"], -1))[:, i].reshape((config["NUM_MINIBATCHES"], -1)),
                    listener_log_prob=listener_trans_batch.listener_log_prob.reshape((config["NUM_STEPS"], -1))[:, i].reshape((config["NUM_MINIBATCHES"], -1)),
                    listener_obs=listener_trans_batch.listener_obs.reshape((config["NUM_STEPS"], listener_trans_batch.listener_obs.shape[1]*listener_trans_batch.listener_obs.shape[2], -1))[:, i, :].reshape((config["NUM_MINIBATCHES"], -1, env_kwargs["image_dim"]**2)),
                    listener_alive=listener_trans_batch.listener_alive.reshape((config["NUM_STEPS"], -1))[:, i].reshape((config["NUM_MINIBATCHES"], -1))
                )

                # Iterate through batches
                new_listener_train_state_i, total_loss = jax.lax.scan(lambda train_state, i: update_minibatch_listener(i, listener_trans_batch_i, listener_advantages_i, listener_targets_i, train_state, config), listener_train_state_i, jnp.arange(config["NUM_MINIBATCHES"]))

                return new_listener_train_state_i, total_loss
            
            def _update_a_speaker(i, speaker_train_states, speaker_trans_batch, speaker_advantages, speaker_targets):                
                speaker_train_state_i = speaker_train_states[i]
                speaker_advantages_i = speaker_advantages.reshape((config["NUM_STEPS"], -1))[:, i].reshape((config["NUM_MINIBATCHES"], -1))
                speaker_targets_i = speaker_targets.reshape((config["NUM_STEPS"], -1))[:, i].reshape((config["NUM_MINIBATCHES"], -1))
                
                speaker_trans_batch_i = Transition(
                    speaker_action=speaker_trans_batch.speaker_action.reshape((config["NUM_STEPS"], env_kwargs["image_dim"] * env_kwargs["image_dim"], -1))[:, :, i].reshape((config["NUM_MINIBATCHES"], -1, env_kwargs["image_dim"]**2)),
                    speaker_reward=speaker_trans_batch.speaker_reward.reshape((config["NUM_STEPS"], -1))[:, i].reshape((config["NUM_MINIBATCHES"], -1)),
                    speaker_value=speaker_trans_batch.speaker_value.reshape((config["NUM_STEPS"], -1))[:, i].reshape((config["NUM_MINIBATCHES"], -1)),
                    speaker_log_prob=speaker_trans_batch.speaker_log_prob.reshape((config["NUM_STEPS"], -1))[:, i].reshape((config["NUM_MINIBATCHES"], -1)),
                    speaker_obs=speaker_trans_batch.speaker_obs.reshape((config["NUM_STEPS"], -1))[:, i].reshape((config["NUM_MINIBATCHES"], -1)),
                    speaker_alive=speaker_trans_batch.speaker_alive.reshape((config["NUM_STEPS"], -1))[:, i].reshape((config["NUM_MINIBATCHES"], -1)),
                    listener_action=speaker_trans_batch.listener_action,
                    listener_reward=speaker_trans_batch.listener_reward,
                    listener_value=speaker_trans_batch.listener_value,
                    listener_log_prob=speaker_trans_batch.listener_log_prob,
                    listener_obs=speaker_trans_batch.listener_obs,
                    listener_alive=speaker_trans_batch.listener_alive
                )

                # Iterate through batches
                new_speaker_train_state_i, total_loss = jax.lax.scan(lambda train_state, i: update_minibatch_speaker(i, speaker_trans_batch_i, speaker_advantages_i, speaker_targets_i, train_state, config), speaker_train_state_i, jnp.arange(config["NUM_MINIBATCHES"]))

                return new_speaker_train_state_i, total_loss

            # Commenting out for debugging. Only need the speakers working rn.
            listener_map_outputs = tuple(map(lambda i: _update_a_listener(i, listener_train_state, trimmed_transition_batch, listener_advantages, listener_targets), range(len(listener_rngs))))
            listener_train_state = tuple([lmo[0] for lmo in listener_map_outputs])
            listener_loss = tuple([lmo[1] for lmo in listener_map_outputs])

            speaker_map_outputs = tuple(map(lambda i: _update_a_speaker(i, speaker_train_state, trimmed_transition_batch, speaker_advantages, speaker_targets), range(len(speaker_rngs))))
            speaker_train_state = tuple([lmo[0] for lmo in speaker_map_outputs])
            speaker_loss = tuple([lmo[1] for lmo in speaker_map_outputs])

            # wandb logging
            def wandb_callback(metrics):
                ll, sl, tb, les = metrics
                lr = tb.listener_reward
                sr = tb.speaker_reward
                logp = tb.listener_log_prob
                lv = tb.listener_value
                sv = tb.speaker_value

                listener_actions = tb.listener_action
                listener_obs = tb.listener_obs
                speaker_actions = tb.speaker_action

                metric_dict = {}

                metric_dict.update({"env/avg_num_speaker_images": jnp.mean(les.env_state.requested_num_speaker_images)})
                # TODO: Lucas do more logging of env state info here.

                # # remove singleton values (squeeze from (MINIBATCH_SIZE,1,1) --> (MINIBATCH_SIZE))
                # speaker_values = speaker_values.squeeze()
                # speaker_actions = speaker_actions.squeeze()
                # listener_actions = listener_actions.squeeze()


                # image_idx = listener_actions.shape[0] - 1
                # image_log = {}
                # image_log.update({f"speaker_value ": speaker_values[image_idx]})
                # image_log.update({f"listener_action": listener_actions[image_idx]})
                # wandb.log(wandb.Image(speaker_actions[image_idx], mode="RGBA"))
                # wandb.log(image_log)

                speaker_images = speaker_actions[-1, 0, :, :, :].reshape((-1, 1, env_kwargs["image_dim"], env_kwargs["image_dim"]))
                speaker_images = wandb.Image(make_grid(torch.tensor(speaker_images), nrow=env_kwargs["num_speakers"]), caption="speaker_actions")
                listener_images = listener_obs[-1, 0, :, :, :].reshape((-1, 1, env_kwargs["image_dim"], env_kwargs["image_dim"]))
                listener_images = wandb.Image(make_grid(torch.tensor(listener_images), nrow=env_kwargs["num_listeners"]), caption="listener_observations")
                metric_dict.update({"env/last_speaker_actions": speaker_images})
                metric_dict.update({"env/last_listener_obs": listener_images})

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
                
                logp = logp.T
                lv = lv.T
                metric_dict.update({f"predictions/mean action log probs/listener {i}": jnp.mean(logp[i]).item() for i in range(len(logp))})
                metric_dict.update({f"predictions/mean state value estimate/listener {i}": jnp.mean(lv[i]).item() for i in range(len(lv))})

                # la = la.T
                # metric_dict.update({""})
                
                wandb.log(metric_dict)

            jax.experimental.io_callback(wandb_callback, None, (listener_loss, speaker_loss, trimmed_transition_batch, log_env_state))

            runner_state = (listener_train_state, speaker_train_state, log_env_state, last_obs, rng)
            return runner_state, update_step + 1

        rng, _rng = jax.random.split(rng)
        runner_state = (listener_train_states, speaker_train_states, log_env_state, obs, _rng)

        partial_update_fn = partial(_update_step, env=env, config=config)
        runner_state, _, traj_batch = jax.lax.scan( # Perform the update step for a specified number of updates and update the runner state
            partial_update_fn, runner_state, jnp.arange(config['UPDATE_EPOCHS']), config["UPDATE_EPOCHS"]
        )

        return {"runner_state": runner_state, "traj_batch": traj_batch}

    return train


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
    rng = jax.random.PRNGKey(50)
    # train_jit = jax.jit(make_train(config), device=jax.devices()[0]) # The environment may or may not be jittable.
    train = make_train(config)
    out = train(rng)
    print("Done")


if __name__ == "__main__":
    main()
    # with jax.profiler.trace("/tmp/jax-trace", create_perfetto_link=True):
    #     test()
