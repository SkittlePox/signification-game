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
import matplotlib.pyplot as plt
import hydra
from omegaconf import OmegaConf
from simplified_signification_game import SimplifiedSignificationGame, State



def batchify(x: dict, agent_list, num_actors):
    x = jnp.stack([x[a] for a in agent_list])
    return x.reshape((num_actors, -1))

def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_actors):
    x = x.reshape((num_actors, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}

class SimpSigGameLogWrapper(LogWrapper):
    @partial(jax.jit, static_argnums=(0,))
    def reset(self, keys: chex.PRNGKey) -> Tuple[chex.Array, State]:
        obs, env_state = jax.vmap(self._env.reset)(keys)
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
        obs, env_state, reward, done, info = jax.vmap(self._env.step)(
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
class ActorCriticListener(nn.Module):
    action_dim: Sequence[int]
    image_dim: Sequence[int]
    config: Dict

    @nn.compact
    def __call__(self, x):
        obs = x
        # Embedding Layer
        embedding = nn.Dense(512, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(obs)
        embedding = nn.relu(embedding)

        # Actor Layer
        actor_mean = nn.Dense(512, kernel_init=orthogonal(2), bias_init=constant(0.0))(embedding)
        actor_mean = nn.relu(actor_mean)
        actor_mean = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(actor_mean)

        # Action Logits
        # unavail_actions = 1 - avail_actions
        # action_logits = actor_mean - (unavail_actions * 1e10)
        pi = distrax.Categorical(logits=actor_mean)

        # Critic Layer
        critic = nn.Dense(512, kernel_init=orthogonal(2), bias_init=constant(0.0))(embedding)
        critic = nn.relu(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(critic)

        return pi, jnp.squeeze(critic, axis=-1)


class Transition(NamedTuple):
    done: jnp.ndarray
    speaker_action: jnp.ndarray
    listener_action: jnp.ndarray
    reward: jnp.ndarray
    value: jnp.ndarray
    speaker_log_prob: jnp.ndarray
    listener_log_prob: jnp.ndarray
    speaker_obs: jnp.ndarray
    listener_obs: jnp.ndarray

def initialize_listener(env, rng, config, learning_rate):
    listener_network = ActorCriticListener(action_dim=env.num_classes, image_dim=env.image_dim, config=config)

    rng, _rng = jax.random.split(rng)
    init_x = jnp.zeros(
            (1, config["NUM_ENVS"], env.image_dim**2)
        )
    network_params = listener_network.init(_rng, init_x)    # I'm not sure how this works, I need to control the size of the inputs and the size of the outputs
    if config["ANNEAL_LR"]:
        tx = optax.chain(
            optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
            optax.adam(learning_rate=learning_rate, eps=1e-5),
        )
    else:
        tx = optax.chain(optax.clip_by_global_norm(config["MAX_GRAD_NORM"]), optax.adam(config["LR"], eps=1e-5))
    
    train_state = TrainState.create(
        apply_fn=listener_network.apply,
        params=network_params,
        tx=tx,
    )
    return listener_network, train_state

def define_env(config):
    if config["ENV_DATASET"] == 'mnist':
        def ret_0(iteration):
            return 0.5
        
        # Define parameters for a signification game
        num_speakers = 5
        num_listeners = 10
        num_channels = 10
        num_classes = 10

        from torchvision.datasets import MNIST
        from utils import to_jax

        mnist_dataset = MNIST('/tmp/mnist/', download=True)
        images, labels = to_jax(mnist_dataset, num_datapoints=100)
        env = SimplifiedSignificationGame(num_speakers, num_listeners, num_channels, num_classes, channel_ratio_fn=ret_0, dataset=(images, labels), image_dim=28, **config["ENV_KWARGS"])
        
        return env

def execute_individual_listener(__rng, _listener_train_state_i, _listener_obs_i):
    _listener_obs_i = _listener_obs_i.ravel()
    policy, value = _listener_train_state_i.apply_fn(_listener_train_state_i.params, _listener_obs_i)
    action = policy.sample(seed=__rng)
    log_prob = policy.log_prob(action)
    return action, log_prob, value

def env_step(runner_state, env, config):
    """This function literally is just for collecting rollouts, which involves applying the joint policy to the env and stepping forward."""
    listener_train_states, log_env_state, obs, last_done, rng = runner_state
    speaker_obs, listener_obs = obs
    old_speaker_obs = speaker_obs
    old_listener_obs = listener_obs

    speaker_obs = speaker_obs.ravel()
    listener_obs = listener_obs.reshape((listener_obs.shape[0]*listener_obs.shape[1], listener_obs.shape[2]*listener_obs.shape[3]))

    ##### COLLECT ACTIONS FROM AGENTS
    rng, _rng = jax.random.split(rng)
    env_rngs = jax.random.split(_rng, len(listener_train_states))

    # COLLECT LISTENER ACTIONS
    listener_outputs = [execute_individual_listener(*args) for args in zip(env_rngs, listener_train_states, listener_obs)]
    l_a = jnp.array([jnp.array([*o]) for o in listener_outputs])
    listener_actions = jnp.asarray(l_a[:, 0], jnp.int32)
    listener_log_probs = l_a[:, 1]
    listener_values = l_a[:, 2]

    listener_actions = listener_actions.reshape(config["NUM_ENVS"], -1)
    listener_log_probs = listener_log_probs.reshape(config["NUM_ENVS"], -1)

    # SIMULATE SPEAKER ACTIONS. TEMPORARY, FOR DEBUGGING, UNTIL SPEAKERS WORK:
    rng_step = jax.random.split(_rng, config["NUM_ENVS"])
    speaker_actions = jnp.array([env.action_space(agent).sample(rng_step[0]) for i, agent in enumerate(env.agents) if agent.startswith("speaker")])
    speaker_actions = jnp.expand_dims(speaker_actions, 0).repeat(config["NUM_ENVS"], axis=0)
    speaker_values = jnp.zeros((config["NUM_ENVS"] * env.num_speakers), dtype=jnp.float32)
    speaker_log_probs = jnp.zeros_like(speaker_actions)    # This will eventually be replaced by real speaker logprobs

    values = jnp.concatenate((speaker_values, listener_values))
    v = values.reshape(config["NUM_ENVS"], -1)
    ###############################################

    ##### STEP ENV
    new_obs, env_state, rewards, dones, info = env.step(rng_step, log_env_state, (speaker_actions, listener_actions))

    # rewards is a dictionary but it needs to be a jnp array
    r = jnp.array([v for k,v in rewards.items() if k != "__all__"]) # Right now this doesn't ensure the correct ordering though
    d = jnp.array([v for k,v in dones.items() if k != "__all__"]) # Right now this doesn't ensure the correct ordering though
    # These appear to be in listener-speaker order. listeners first, speakers second
    # I can easily flip the order around:
    r = jnp.vstack([r[env.num_listeners:], r[:env.num_listeners]])
    d = jnp.vstack([d[env.num_listeners:], d[:env.num_listeners]])

    r = r.reshape(config["NUM_ENVS"], -1)
    d = d.reshape(config["NUM_ENVS"], -1)

    # How do I put values in here??? What is the value for this transition?
    
    transition = Transition(
        d,
        speaker_actions,
        listener_actions,
        r,
        v,
        speaker_log_probs,
        listener_log_probs,
        old_speaker_obs,
        old_listener_obs
    )

    runner_state = (listener_train_states, env_state, new_obs, d, _rng) # We should be returning the new_obs, the agents haven't seen it yet.
    # I'm not sure if d here is correct
    return runner_state, transition


def test_rollout_execution(config, rng):
    env = define_env(config)
    env = SimpSigGameLogWrapper(env)

    config["NUM_ACTORS"] = env.num_agents * config["NUM_ENVS"]
    config["NUM_UPDATES"] = (
            config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
            config["NUM_ACTORS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )
    
    # For the learning rate
    def linear_schedule(count):
        frac = 1.0 - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"])) / config["NUM_UPDATES"]
        return config["LR"] * frac

    # MAKE AGENTS
    rng, rng_s, rng_l = jax.random.split(rng, 3)    # rng_s for speakers, rng_l for listeners
    listener_rngs = jax.random.split(rng_l, len(env.listener_agents) * config["NUM_ENVS"])   # Make an rng key for each listener
    
    listeners_stuff = [initialize_listener(env, x_rng, config, linear_schedule) for x_rng in listener_rngs]
    listener_networks, listener_train_states = zip(*listeners_stuff)
    # TODO eventually: Add speaker networks and train states

    # INIT ENV
    rng, _rng = jax.random.split(rng)
    reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
    obs, log_env_state = env.reset(reset_rng)  # log_env_state is a single variable, but each variable it has is actually batched

    # init_transition = Transition( # This is no longer needed, but it may be helpful to know what types and shapes things are in the future.
    #     jnp.zeros((config["NUM_ENVS"], env.num_agents), dtype=bool),
    #     jnp.zeros((config["NUM_ENVS"], max(env.num_speakers, 1), env.image_dim, env.image_dim), dtype=jnp.float32),
    #     jnp.zeros((config["NUM_ENVS"], max(env.num_listeners, 1)), dtype=jnp.int32),
    #     jnp.zeros((config["NUM_ENVS"], env.num_agents), dtype=jnp.float32),     # rewards
    #     jnp.zeros((config["NUM_ENVS"], env.num_agents), dtype=jnp.float32),     # values
    #     jnp.zeros((config["NUM_ENVS"], max(env.num_speakers, 1), env.image_dim, env.image_dim), dtype=jnp.float32),
    #     jnp.zeros((config["NUM_ENVS"], max(env.num_listeners, 1)), dtype=jnp.float32),
    #     obs[0],
    #     obs[1]
    # )

    rng, _rng = jax.random.split(rng)
    runner_state = (listener_train_states, log_env_state, obs, jnp.zeros((config["NUM_ENVS"], env.num_agents), dtype=bool), _rng)

    # runner_state, transition = env_step(runner_state, env, config)    # This was for testing a single env_step
    runner_state, traj_batch = jax.lax.scan(lambda rs, _: env_step(rs, env, config), runner_state, None, config['NUM_STEPS'])
    # traj_batch is a Transition with sub-objects of shape (num_steps, num_envs, ...). It represents a rollout.
    
    return {"runner_state": runner_state, "traj_batch": traj_batch}


def update_minbatch(batch, train_state, config, rng):
    trans_batch, advantages, targets = batch

    def _loss_fn(train_state, traj_batch, gae, targets):
        # TODO: This needs to be re-worked. It should be written on a per-agent basis!

        # RERUN NETWORK
        # pi, value = network.apply(params,
        #                             (traj_batch.obs, traj_batch.done, traj_batch.avail_actions))
        # log_prob = pi.log_prob(traj_batch.action) # importantly this is a different action than the one from the policy

        # COLLECT LISTENER ACTIONS AND LOG_PROBS FOR TRAJ actions

        # An important difference here than above is that the obs and traj_actions should actually be batched, while the train state is not.
        def _get_individual_listener_logprobs_for_traj_action(_listener_train_state_i, _listener_obs_i, _traj_action_i):
            policy, value = _listener_train_state_i.apply_fn(_listener_train_state_i.params, _listener_obs_i)
            log_prob = policy.log_prob(_traj_action_i)
            return log_prob

        # I can't do a vmap call here because I need to manually iterate through train_state.

        # log_probs = jax.vmap(_get_individual_listener_logprobs_for_traj_action)(train_state, traj_batch.listener_obs, traj_batch.listener_action)

        # I need to iterate over batches too! 
        # The first two zeroes are for the minibatch items. The second indexing are for selecting the agent
        # We also have to reshape the observations
        lo = traj_batch.listener_obs[0][0][:, 0, ...].reshape((-1, traj_batch.listener_obs.shape[-1]*traj_batch.listener_obs.shape[-1]))
        la = traj_batch.listener_action[0][0][:, 0, ...]
        o = _get_individual_listener_logprobs_for_traj_action(train_state[0][0], lo, la)
        # log_prob_out = [_get]
        # policy, value = [ for _listener_train_state_i, _listener_obs_i in zip(train_state, traj_batch.obs)]
        
        listener_outputs = [execute_individual_listener(*args) for args in zip(env_rngs, listener_train_states, listener_obs)]
        l_a = jnp.array([jnp.array([*o]) for o in listener_outputs])
        listener_actions = jnp.asarray(l_a[:, 0], jnp.int32)
        listener_log_probs = l_a[:, 1]
        listener_values = l_a[:, 2]

        listener_actions = listener_actions.reshape(config["NUM_ENVS"], -1)
        listener_log_probs = listener_log_probs.reshape(config["NUM_ENVS"], -1)


        # CALCULATE VALUE LOSS
        value_pred_clipped = traj_batch.value + (
                value - traj_batch.value
        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
        value_losses = jnp.square(value - targets)
        value_losses_clipped = jnp.square(value_pred_clipped - targets)
        value_loss = (
                0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
        )

        # CALCULATE ACTOR LOSS
        ratio = jnp.exp(log_prob - traj_batch.log_prob)
        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
        loss_actor1 = ratio * gae
        loss_actor2 = (
                jnp.clip(
                    ratio,
                    1.0 - config["CLIP_EPS"],
                    1.0 + config["CLIP_EPS"],
                )
                * gae
        )
        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
        loss_actor = loss_actor.mean()
        entropy = pi.entropy().mean()

        total_loss = (
                loss_actor
                + config["VF_COEF"] * value_loss
                - config["ENT_COEF"] * entropy
        )
        return total_loss, (value_loss, loss_actor, entropy)

    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True, allow_int=True)
    total_loss, grads = grad_fn(
        train_state, trans_batch, advantages, targets
    )
    train_state = train_state.apply_gradients(grads=grads)
    return train_state, total_loss


def make_train(config):
    env = define_env(config)
    env = SimpSigGameLogWrapper(env)

    config["NUM_ACTORS"] = env.num_agents * config["NUM_ENVS"]
    config["NUM_UPDATES"] = (
            config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
            config["NUM_ACTORS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )
    
    # For the learning rate
    def linear_schedule(count):
        frac = 1.0 - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"])) / config["NUM_UPDATES"]
        return config["LR"] * frac

    def train(rng):
        # MAKE AGENTS
        rng, rng_s, rng_l = jax.random.split(rng, 3)    # rng_s for speakers, rng_l for listeners
        listener_rngs = jax.random.split(rng_l, len(env.listener_agents) * config["NUM_ENVS"])   # Make an rng key for each listener
        
        listeners_stuff = [initialize_listener(env, x_rng, config, linear_schedule) for x_rng in listener_rngs]
        listener_networks, listener_train_states = zip(*listeners_stuff)
        # TODO eventually: Add speaker networks and train states

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obs, log_env_state = env.reset(reset_rng)  # log_env_state is a single variable, but each variable it has is actually batched
        
        # TRAIN LOOP
        def _update_step(update_runner_state, env, config):
            runner_state, update_steps = update_runner_state
            
            # COLLECT TRAJECTORIES
            runner_state, transition_batch = jax.lax.scan(lambda rs, _: env_step(rs, env, config), runner_state, None, config['NUM_STEPS'])
            # transition_batch is an instance of Transition with batched sub-objects
            # The shape of transition_batch is (num_steps, num_envs, ...) because it's the output of jax.lax.scan, which enumerates over steps

            # Instead of executing the agents on the final observation to get their values, we are simply going to ignore the last observation from traj_batch.
            # We'll need to get the final value in transition_batch and cut off the last index
            # We want to cleave off the final step, so it should go from shape (A, B, C) to shape (A-1, B, C)
            trimmed_transition_batch = Transition(
                done=transition_batch.done[:-1, ...],
                speaker_action=transition_batch.speaker_action[:-1, ...],
                listener_action=transition_batch.listener_action[:-1, ...],
                reward=transition_batch.reward[:-1, ...],
                value=transition_batch.value[:-1, ...],
                speaker_log_prob=transition_batch.speaker_log_prob[:-1, ...],
                listener_log_prob=transition_batch.listener_log_prob[:-1, ...],
                speaker_obs=transition_batch.speaker_obs[:-1, ...],
                listener_obs=transition_batch.listener_obs[:-1, ...]
            )

            # CALCULATE ADVANTAGE
            listener_train_state, log_env_state, last_obs, last_done, rng = runner_state
            # listener_train_state is a tuple of TrainStates of length num_envs * env.num_listeners

            def _calculate_gae(trans_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.done,
                        transition.value,
                        transition.reward,
                    )
                    delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                    gae = delta + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    trans_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + trans_batch.value

            advantages, targets = _calculate_gae(trimmed_transition_batch, transition_batch.value[-1])

            ##### Just get the above working for now.

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                train_state, trans_batch, advantages, targets, rng = update_state
                rng, _rng = jax.random.split(rng)

                # I need to now isolate out the speaker and listener transitions, advantages, and targets
                speaker_train_state, listener_train_state = train_state

                listener_trans_batch = Transition(
                    done=trans_batch.done[..., env.num_speakers:],
                    speaker_action=trans_batch.speaker_action,
                    listener_action=trans_batch.listener_action,
                    reward=trans_batch.reward[..., env.num_speakers:],
                    value=trans_batch.value[..., env.num_speakers:],
                    speaker_log_prob=trans_batch.speaker_log_prob,
                    listener_log_prob=trans_batch.listener_log_prob,
                    speaker_obs=trans_batch.speaker_obs,
                    listener_obs=trans_batch.listener_obs
                )
                
                listener_advantages = advantages[..., env.num_speakers:]
                listener_targets = targets[..., env.num_speakers:]

                listener_train_state_minibatch = tuple(listener_train_state[i:i + env.num_listeners] for i in range(0, len(listener_train_state), env.num_listeners))

                listener_trans_minibatch = Transition(
                    done=listener_trans_batch.done.reshape((config["NUM_MINIBATCHES"], -1, *listener_trans_batch.done.shape[1:])),
                    speaker_action=listener_trans_batch.speaker_action.reshape((config["NUM_MINIBATCHES"], -1, *listener_trans_batch.speaker_action.shape[1:])),
                    listener_action=listener_trans_batch.listener_action.reshape((config["NUM_MINIBATCHES"], -1, *listener_trans_batch.listener_action.shape[1:])),
                    reward=listener_trans_batch.reward.reshape((config["NUM_MINIBATCHES"], -1, *listener_trans_batch.reward.shape[1:])),
                    value=listener_trans_batch.value.reshape((config["NUM_MINIBATCHES"], -1, *listener_trans_batch.value.shape[1:])),
                    speaker_log_prob=listener_trans_batch.speaker_log_prob,
                    listener_log_prob=listener_trans_batch.listener_log_prob.reshape((config["NUM_MINIBATCHES"], -1, *listener_trans_batch.listener_log_prob.shape[1:])),
                    speaker_obs=listener_trans_batch.speaker_obs,
                    listener_obs=listener_trans_batch.listener_obs.reshape((config["NUM_MINIBATCHES"], -1, *listener_trans_batch.listener_obs.shape[1:]))
                )

                listener_advantages_minibatch = listener_advantages.reshape((config["NUM_MINIBATCHES"], -1, config["NUM_ENVS"], env.num_listeners))
                listener_targets_minibatch = listener_targets.reshape((config["NUM_MINIBATCHES"], -1, config["NUM_ENVS"], env.num_listeners))

                # I'm going to reshape everything
                listener_minibatch = (listener_trans_minibatch, listener_advantages_minibatch, listener_targets_minibatch)

                # batch = (trans_batch, advantages.squeeze(), targets.squeeze())
                # permutation = jax.random.permutation(_rng, env.num_listeners)

                # shuffled_batch = jax.tree_util.tree_map(
                #     lambda x: jnp.take(x, permutation, axis=1), listener_batch
                # )

                # If we shuffle, then we may not be able to keep track of which agent to execute. So no shuffling for now.

                # minibatches = jax.tree_util.tree_map(
                #     lambda x: jnp.swapaxes(
                #         jnp.reshape(
                #             x,
                #             [x.shape[0], config["NUM_MINIBATCHES"], -1]
                #             + list(x.shape[2:]),
                #         ),
                #         1,
                #         0,
                #     ),
                #     listener_batch,
                # )

                # Let's also ignore minibatching for now
                # We are definitely going to need to minibatch actually.

                # I'm going to have to re-write _update_minbatch, since it executes the agents

                # train_state, total_loss = jax.lax.scan(
                #     lambda mb, ls: update_minbatch, train_state, minibatches
                # )
                train_state, total_loss = jax.lax.scan(lambda lb, _: update_minbatch(lb, listener_train_state_minibatch, config, _rng), listener_minibatch, None, config['NUM_STEPS'] - 1)
                update_state = (train_state, trans_batch, advantages, targets, rng)
                return update_state, total_loss

            train_state = (None, listener_train_states) # In the future this will be a tuple containing also speaker_train_state
            update_state = (train_state, trimmed_transition_batch, advantages, targets, rng)

            update_state, total_loss = _update_epoch(update_state, None)
            
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            train_state = update_state[0]
            metric = traj_batch.info
            rng = update_state[-1]

            def callback(metric):
                wandb.log(
                    {
                        "returns": metric["returned_episode_returns"][-1, :].mean(),
                        "env_step": metric["update_steps"]
                        * config["NUM_ENVS"]
                        * config["NUM_STEPS"],
                    }
                )
            metric["update_steps"] = update_steps
            # jax.experimental.io_callback(callback, None, metric)
            update_steps = update_steps + 1
            runner_state = (train_state, env_state, last_obs, last_done, rng)
            return (runner_state, update_steps), None

        rng, _rng = jax.random.split(rng)
        # Initialize the runner state with listener_train_states, env_state, obsv, a zero vector for whether the env is done (currently unused and probably the wrong shape), and _rng
        runner_state = (listener_train_states, log_env_state, obs, jnp.zeros((config["NUM_ENVS"], env.num_agents), dtype=bool), _rng)

        # runner_state, _ = jax.lax.scan( # Perform the update step for a specified number of updates and update the runner state
        #     _update_step, (runner_state, 0), None, config["NUM_UPDATES"]
        # )

        runner_state, traj_batch = _update_step((runner_state, 0), env, config)
        # runner_state = collect_rollouts(runner_state, env, config)
        return {"runner_state": runner_state, "traj_batch": traj_batch}

    return train


@hydra.main(version_base=None, config_path="config", config_name="test")
def test(config):
    config = OmegaConf.to_container(config) 

    rng = jax.random.PRNGKey(50)
    out = test_rollout_execution(config, rng)
    print(out['runner_state'])


@hydra.main(version_base=None, config_path="config", config_name="test")
def main(config):
    config = OmegaConf.to_container(config) 

    # Setting aside wandb for now.
    # wandb.init(
    #     entity=config["ENTITY"],
    #     project=config["PROJECT"],
    #     tags=["IPPO", "FF", config["ENV_NAME"]],
    #     config=config,
    #     mode=config["WANDB_MODE"],
    # )
    
    rng = jax.random.PRNGKey(50)
    # train_jit = jax.jit(make_train(config), device=jax.devices()[0]) # The environment may or may not be jittable.
    train = make_train(config)
    out = train(rng)


if __name__ == "__main__":
    main()
    '''results = out["metrics"]["returned_episode_returns"].mean(-1).reshape(-1)
    jnp.save('hanabi_results', results)
    plt.plot(results)
    plt.xlabel("Update Step")
    plt.ylabel("Return")
    plt.savefig(f'IPPO_{config["ENV_NAME"]}.png')'''
