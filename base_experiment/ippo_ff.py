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
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    speaker_obs: jnp.ndarray
    listener_obs: jnp.ndarray
    info: jnp.ndarray
    speaker_avail_actions: jnp.ndarray
    listener_avail_actions: jnp.ndarray

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

def env_step(runner_state, env, config):
    """This function literally is just for collecting rollouts, which involves applying the joint policy to the env and stepping forward."""
    listener_train_states, env_state, obsv, last_done, rng = runner_state
    speaker_obsv, listener_obvs = obsv

    # COLLECT ACTIONS FROM AGENTS
    rng, _rng = jax.random.split(rng)


    # def get_listener_action(listener_train_state, obsv, listener_rng):
    #     obsv = obsv.ravel()
    #     policy, critic = listener_train_state.apply_fn(listener_train_state.params, obsv)

    # The question I have at this point is is env_state batched already? The answer is yes, because it is being passed via runner_state. However, runner_state can be batched!

    listener_rngs = jax.random.split(_rng, len(env.listener_agents))

    all_actions = []
    actions_agg = []
    log_probs_agg = []
    # Collect the actions for all the agents for the current env state for each environment
    for j in range(config["NUM_ENVS"]):
        actions = []
        log_probs = []
        listener_obsv_env = listener_obvs[j]
        for i in range(len(listener_train_states)):
            listener_train_state, obsv, listener_rng = listener_train_states[i], listener_obsv_env[i], listener_rngs[i]
            obsv = obsv.ravel()
            # jax.debug.print("obsv: {y}", y=obsv)
            policy, critic = listener_train_state.apply_fn(listener_train_state.params, obsv)
            action = policy.sample(seed=listener_rng)
            log_prob = policy.log_prob(action)
            actions.append(action)
            log_probs.append(log_prob)
        all_actions.append((actions, log_probs))
        actions_agg.append(actions)
        log_probs_agg.append(log_probs)

    # def get_actions_and_log_probs(listener_train_state, obsv, listener_rng):
    #     obsv = obsv.ravel()
    #     policy, critic = listener_train_state.apply_fn(listener_train_state.params, obsv)
    #     action = policy.sample(seed=listener_rng)
    #     log_prob = policy.log_prob(action)
    #     return action, log_prob

    # get_actions_and_log_probs_vmap = jax.vmap(get_actions_and_log_probs, in_axes=(0, 0, 0))
    # all_actions = [get_actions_and_log_probs_vmap(listener_train_states, listener_obsv_env, listener_rngs) for listener_obsv_env in listener_obsv]
    
    # env_act = unbatchify(action, env.agents, config["NUM_ENVS"], env.num_agents)

    # STEP ENV
    rng_step = jax.random.split(_rng, config["NUM_ENVS"])

    obsv, env_state, rewards, dones, info = env.step_env(rng_step[0], env_state[0], actions_agg[0])

    # step_env_vmap = jax.vmap(env.step_env, in_axes=(0, 0, 0))
    # obsv, env_state, rewards, dones, info = step_env_vmap(rng_step, env_state, actions_agg)


    # obsv, env_state, reward, done, info = jax.vmap(env.step, in_axes=(0, 0, 0))(    # TODO: Need to work on this.
    #     rng_step, env_state, env_act
    # )
    # info = jax.tree_map(lambda x: x.reshape((config["NUM_ACTORS"])), info)
    # done_batch = batchify(done, env.agents, config["NUM_ACTORS"]).squeeze()
    # transition = Transition(
    #     dones,
    #     action.squeeze(),
    #     value.squeeze(),
    #     batchify(reward, env.agents, config["NUM_ACTORS"]).squeeze(),
    #     log_prob.squeeze(),
    #     obs_batch,
    #     info,
    #     avail_actions
    # )
    runner_state = (listener_train_states, env_state, obsv, dones, rng)
    return runner_state


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
    listener_rngs = jax.random.split(rng_l, len(env.listener_agents) * config["NUM_ENVS"])   # Make an rng key for each listener, for each environment
    
    listeners_stuff = [initialize_listener(env, x_rng, config, linear_schedule) for x_rng in listener_rngs]
    listener_networks, listener_train_states = zip(*listeners_stuff)
    # TODO eventually: Add speaker networks and train states

    # INIT ENV
    rng, _rng = jax.random.split(rng)
    reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
    obsv, log_env_state = env.reset(reset_rng)    # These should now be length config["NUM_ENVS"]

    # TODO: These do not seem to be the right shape.
    jax.debug.print("obsv prior: {y}", y=obsv)
    jax.debug.print("env_state: {y}", y=log_env_state)

    # Let's run a single step of env_step for now

    

    def collect_rollouts(runner_state, env, config):
        runner_state, _ = jax.lax.scan(lambda rs, _: env_step(rs, env, config), runner_state, None, config['NUM_STEPS'])
        return runner_state
    

    rng, _rng = jax.random.split(rng)
    runner_state = (listener_train_states, log_env_state, obsv, jnp.zeros((config["NUM_ACTORS"]), dtype=bool), _rng) # Initialize the runner state with listener_train_states, env_state, obsv, a zero vector for initial actions, and _rng
    # We don't need to include the networks, the apply function is stored in the train states.

    runner_state = env_step(runner_state, env, config)

    # runner_state = collect_rollouts(runner_state, env, config)
    return {"runner_state": runner_state}   # Return the updated runner state


def make_train(config):
    env = define_env(config)
    env = LogWrapper(env)

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
        listener_rngs = jax.random.split(rng_l, len(env.listener_agents))   # Make an rng key for each listener
        listeners_stuff = jax.lax.map(lambda x: initialize_listener(env, x[0], x[1], config, learning_rate=linear_schedule), (env.listener_agents, listener_rngs))
        listener_networks, listener_train_states = zip(*listeners_stuff)
        # TODO: Add speaker networks and train states

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        env = define_env(config)

        obsv, env_state = jax.lax.map(lambda rng: env.reset(rng), reset_rng) # env.reset is currently not vmappable because it uses a dataloader, but if it were then this would be too

        # TRAIN LOOP
        def _update_step(update_runner_state, unused):
            # COLLECT TRAJECTORIES
            runner_state, update_steps = update_runner_state
            runner_state, traj_batch = jax.lax.scan(
                env_step, runner_state, None, config["NUM_STEPS"]
            )

            # CALCULATE ADVANTAGE
            listener_train_states, env_state, last_obs, last_done, rng = runner_state
            last_obs_batch = batchify(last_obs, env.agents, config["NUM_ACTORS"])
            avail_actions = jnp.ones((config["NUM_ACTORS"], env.action_space(env.agents[0]).n))
            ac_in = (last_obs_batch[np.newaxis, :], last_done[np.newaxis, :], avail_actions)
            _, last_val = network.apply(train_state.params, ac_in)
            last_val = last_val.squeeze()

            # I believe we don't need to modify this
            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.done,
                        transition.value,
                        transition.reward,
                    )
                    delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                    gae = (
                            delta
                            + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                    )
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val)

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, traj_batch, gae, targets):
                        # RERUN NETWORK
                        pi, value = network.apply(params,
                                                  (traj_batch.obs, traj_batch.done, traj_batch.avail_actions))
                        log_prob = pi.log_prob(traj_batch.action)

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

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(
                        train_state.params, traj_batch, advantages, targets
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

                train_state, traj_batch, advantages, targets, rng = update_state
                rng, _rng = jax.random.split(rng)

                batch = (traj_batch, advantages.squeeze(), targets.squeeze())
                permutation = jax.random.permutation(_rng, config["NUM_ACTORS"])

                shuffled_batch = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=1), batch
                )

                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.swapaxes(
                        jnp.reshape(
                            x,
                            [x.shape[0], config["NUM_MINIBATCHES"], -1]
                            + list(x.shape[2:]),
                        ),
                        1,
                        0,
                    ),
                    shuffled_batch,
                )

                train_state, total_loss = jax.lax.scan(
                    _update_minbatch, train_state, minibatches
                )
                update_state = (train_state, traj_batch, advantages, targets, rng)
                return update_state, total_loss

            update_state = (train_state, traj_batch, advantages, targets, rng)
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
        runner_state = (listener_train_states, env_state, obsv, jnp.zeros((config["NUM_ACTORS"]), dtype=bool), _rng) # Initialize the runner state with listener_train_states, env_state, obsv, a zero vector for initial actions, and _rng
        runner_state, _ = jax.lax.scan( # Perform the update step for a specified number of updates and update the runner state
            _update_step, (runner_state, 0), None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state}   # Return the updated runner state

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
    test()
    '''results = out["metrics"]["returned_episode_returns"].mean(-1).reshape(-1)
    jnp.save('hanabi_results', results)
    plt.plot(results)
    plt.xlabel("Update Step")
    plt.ylabel("Return")
    plt.savefig(f'IPPO_{config["ENV_NAME"]}.png')'''
