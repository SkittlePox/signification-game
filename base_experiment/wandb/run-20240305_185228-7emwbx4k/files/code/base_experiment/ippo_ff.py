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
        num_speakers = 2
        num_listeners = 5
        num_channels = 5
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

# @jax.jit
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
    # config["MINIBATCH_SIZE"] = (
    #         config["NUM_ACTORS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    # )
    config["NUM_MINIBATCHES"] = config["NUM_STEPS"] // config["MINIBATCH_SIZE"]
    
    # For the learning rate
    def linear_schedule(count):
        frac = 1.0 - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"])) / config["NUM_UPDATES"]   # This calculation may be wrong
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
    runner_state = (listener_train_states, log_env_state, obs, jnp.zeros((config["NUM_ENVS"], env.num_agents), dtype=bool), _rng)

    # runner_state, transition = env_step(runner_state, env, config)    # This was for testing a single env_step
    runner_state, traj_batch = jax.lax.scan(lambda rs, _: env_step(rs, env, config), runner_state, None, config['NUM_STEPS'])
    # traj_batch is a Transition with sub-objects of shape (num_steps, num_envs, ...). It represents a rollout.
    
    return {"runner_state": runner_state, "traj_batch": traj_batch}

def update_minibatch(j, listener_trans_batch_i, listener_advantages_i, listener_targets_i, listener_train_state, config):
    # j is for iterating through batches

    def _loss_fn(params, listener_obs, listener_actions, values, log_probs, advantages, targets):
        # COLLECT LISTENER ACTIONS AND LOG_PROBS FOR TRAJ ACTIONS
        listener_i_policy, listener_i_value = listener_train_state.apply_fn(params, listener_obs)
        listener_i_log_prob = listener_i_policy.log_prob(listener_actions)

        # CALCULATE VALUE LOSS
        value_pred_clipped = values + (
                listener_i_value - values
        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
        value_losses = jnp.square(values - targets)
        value_losses_clipped = jnp.square(value_pred_clipped - targets)
        value_loss = (0.5 * jnp.maximum(value_losses, value_losses_clipped).mean())

        # CALCULATE ACTOR LOSS
        ratio = jnp.exp(listener_i_log_prob - log_probs)
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
        entropy = listener_i_policy.entropy().mean()

        total_loss = (
                loss_actor
                + config["VF_COEF"] * value_loss
                - config["ENT_COEF"] * entropy
        )
        return total_loss, (value_loss, loss_actor, entropy)

    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True, allow_int=False)

    total_loss, grads = grad_fn(
        listener_train_state.params,
        listener_trans_batch_i.listener_obs[j],
        listener_trans_batch_i.listener_action[j], 
        listener_trans_batch_i.value[j], 
        listener_trans_batch_i.listener_log_prob[j],
        listener_advantages_i[j], 
        listener_targets_i[j]
    )
    
    listener_train_state = listener_train_state.apply_gradients(grads=grads)

    return listener_train_state, total_loss


def make_train(config):
    env = define_env(config)
    env = SimpSigGameLogWrapper(env)

    config["NUM_ACTORS"] = env.num_agents * config["NUM_ENVS"]
    config["NUM_UPDATES"] = (
            config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    # config["MINIBATCH_SIZE"] = (
    #         config["NUM_ACTORS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    # )
    config["NUM_MINIBATCHES"] = config["NUM_STEPS"] // config["MINIBATCH_SIZE"]
    
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
        def _update_step(runner_state, update_step, env, config):
            # runner_state is actually a tuple of runner_states, one per agent
            
            # COLLECT TRAJECTORIES
            runner_state, transition_batch = jax.lax.scan(lambda rs, _: env_step(rs, env, config), runner_state, None, config['NUM_STEPS'] + 1)
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

            # UPDATE NETWORK (this is listener-specific)
            def _update_a_listener(i, listener_train_states, listener_trans_batch, listener_advantages, listener_targets):                
                listener_train_state_i = listener_train_states[i]
                listener_advantages_i = listener_advantages[:, i].reshape((config["NUM_MINIBATCHES"], -1))
                listener_targets_i = listener_targets[:, i].reshape((config["NUM_MINIBATCHES"], -1))
                
                listener_trans_batch_i = Transition(
                    done=listener_trans_batch.done[:, i].reshape((config["NUM_MINIBATCHES"], -1)),
                    speaker_action=listener_trans_batch.speaker_action,
                    listener_action=jnp.float32(listener_trans_batch.listener_action[:, i].reshape((config["NUM_MINIBATCHES"], -1))),
                    reward=listener_trans_batch.reward[:, i].reshape((config["NUM_MINIBATCHES"], -1)),
                    value=listener_trans_batch.value[:, i].reshape((config["NUM_MINIBATCHES"], -1)),
                    speaker_log_prob=listener_trans_batch.speaker_log_prob,
                    listener_log_prob=listener_trans_batch.listener_log_prob[:, i].reshape((config["NUM_MINIBATCHES"], -1)),
                    speaker_obs=jnp.float32(listener_trans_batch.speaker_obs.reshape((config["NUM_MINIBATCHES"], -1, *listener_trans_batch.speaker_obs.shape[1:]))),
                    listener_obs=listener_trans_batch.listener_obs[:, i, ...].reshape((config["NUM_MINIBATCHES"], -1, listener_trans_batch.listener_obs[:, i, ...].shape[1] * listener_trans_batch.listener_obs[:, i, ...].shape[2]))
                )

                # This line tests a single update
                # new_listener_train_state_i, total_loss = update_minibatch(0, listener_trans_batch_i, listener_advantages_i, listener_targets_i, listener_train_state_i, config)                
                
                # Iterate through batches
                new_listener_train_state_i, total_loss = jax.lax.scan(lambda train_state, i: update_minibatch(i, listener_trans_batch_i, listener_advantages_i, listener_targets_i, train_state, config), listener_train_state_i, jnp.arange(config["NUM_MINIBATCHES"]))

                # This code is from the source codebase and does not work. Keeping it here for future work.
                # metric = traj_batch.info
                # rng = update_state[-1]

                # def callback(metric):
                #     wandb.log(
                #         {
                #             "returns": metric["returned_episode_returns"][-1, :].mean(),
                #             "env_step": metric["update_steps"]
                #             * config["NUM_ENVS"]
                #             * config["NUM_STEPS"],
                #         }
                #     )
                # # metric["update_steps"] = update_steps
                # jax.experimental.io_callback(callback, None, metric)

                return new_listener_train_state_i, total_loss

                
            """
            For the below comments on the shapes of things (these numbers change based on the yaml):
            "NUM_ENVS": 8
            "NUM_STEPS": 36
            env.image_dim: 28
            env.num_listeners: 10
            env.num_speakers: 5
            """
            listener_trans_batch = Transition(
                done=trimmed_transition_batch.done[..., env.num_speakers:].reshape((config["NUM_STEPS"], -1)),
                speaker_action=trimmed_transition_batch.speaker_action, # This is shape (36, 8, 5, 28, 28) but I'm not going to bother reshaping
                listener_action=trimmed_transition_batch.listener_action.reshape((config["NUM_STEPS"], -1)),
                reward=trimmed_transition_batch.reward[..., env.num_speakers:].reshape((config["NUM_STEPS"], -1)),
                value=trimmed_transition_batch.value[..., env.num_speakers:].reshape((config["NUM_STEPS"], -1)),
                speaker_log_prob=trimmed_transition_batch.speaker_log_prob, # This is shape (36, 8, 5, 28, 28) but I'm not going to bother reshaping
                listener_log_prob=trimmed_transition_batch.listener_log_prob.reshape((config["NUM_STEPS"], -1)),
                speaker_obs=trimmed_transition_batch.speaker_obs,   # This is shape (36, 8, 5) but I'm not going to bother reshaping
                listener_obs=trimmed_transition_batch.listener_obs.reshape((config["NUM_STEPS"], -1, env.image_dim, env.image_dim)),
            )

            listener_advantages = advantages[..., env.num_speakers:].reshape((config["NUM_STEPS"], -1))
            listener_targets = targets[..., env.num_speakers:].reshape((config["NUM_STEPS"], -1))
            
            listener_map_outputs = tuple(map(lambda i: _update_a_listener(i, listener_train_state, listener_trans_batch, listener_advantages, listener_targets), range(len(listener_rngs))))
            listener_train_state = tuple([lmo[0] for lmo in listener_map_outputs])
            listener_loss = tuple([lmo[1] for lmo in listener_map_outputs])

            def callback(ll):
                # agent, total_loss, (value_loss, loss_actor, entropy)
                

                loss_dict = {f"total loss for listener {i}": jnp.mean(ll[i][0]).item() for i in range(len(ll))}
                loss_dict.update({f"value loss for listener {i}": jnp.mean(ll[i][1]).item() for i in range(len(ll))})
                loss_dict.update({f"actor loss for listener {i}": jnp.mean(ll[i][2]).item() for i in range(len(ll))})
                loss_dict.update({f"entropy for listener {i}": jnp.mean(ll[i][3]).item() for i in range(len(ll))})
                # loss_dict["average_loss"] = jnp.mean(ll)
                wandb.log(loss_dict)

            jax.experimental.io_callback(callback, None, listener_loss)

            runner_state = (listener_train_state, log_env_state, last_obs, last_done, rng)
            return runner_state, update_step + 1

        rng, _rng = jax.random.split(rng)
        runner_state = (listener_train_states, log_env_state, obs, jnp.zeros((config["NUM_ENVS"], env.num_agents), dtype=bool), _rng)

        partial_update_fn = partial(_update_step, env=env, config=config)
        runner_state, traj_batch = jax.lax.scan( # Perform the update step for a specified number of updates and update the runner state
            partial_update_fn, runner_state, jnp.arange(config['NUM_UPDATES']), config["NUM_UPDATES"]
        )

        return {"runner_state": runner_state, "traj_batch": traj_batch}

    return train


@hydra.main(version_base=None, config_path="config", config_name="test")
def test(config):
    wandb.init(
        # entity=config["ENTITY"],
        project=config["PROJECT"],
        tags=["test"],
        config=config,
        mode=config["WANDB_MODE"],
        save_code=True
    )
    with jax.profiler.trace("/tmp/jax-trace", create_perfetto_link=True):
        config = OmegaConf.to_container(config) 
        rng = jax.random.PRNGKey(50)
        out = test_rollout_execution(config, rng)
        print(out['runner_state'])


@hydra.main(version_base=None, config_path="config", config_name="test")
def main(config):
    # print(config)
    wandb.init(
        # entity=config["ENTITY"],
        project=config["PROJECT"],
        tags=["main"],
        # config=config,
        mode=config["WANDB_MODE"],
        save_code=True
    )
    with jax.profiler.trace("/tmp/jax-trace", create_perfetto_link=True):
        config = OmegaConf.to_container(config) 
        rng = jax.random.PRNGKey(50)
        # train_jit = jax.jit(make_train(config), device=jax.devices()[0]) # The environment may or may not be jittable.
        train = make_train(config)
        out = train(rng)
        print("Done")


if __name__ == "__main__":
    main()
    '''results = out["metrics"]["returned_episode_returns"].mean(-1).reshape(-1)
    jnp.save('hanabi_results', results)
    plt.plot(results)
    plt.xlabel("Update Step")
    plt.ylabel("Return")
    plt.savefig(f'IPPO_{config["ENV_NAME"]}.png')'''
