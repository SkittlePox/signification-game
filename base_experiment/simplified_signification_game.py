from functools import partial
from typing import Tuple, Dict, Callable
import numpy as np
import jax, chex
import jax.numpy as jnp
import jax_dataloader as jdl
from jax import lax
from torchvision.datasets import MNIST
from flax import struct
from jaxmarl.environments.multi_agent_env import MultiAgentEnv
from gymnax.environments.spaces import Discrete, Box


@struct.dataclass
class State:
    next_channel_map: chex.Array  # [num_speakers + num_channels] * [num_listeners] * num_channels
    # Each index represents a communication channel, which is a speaker index and a listener index.
    next_env_images: chex.Array  # [image_size] * num_channels
    # For each channel, an image is drawn from the dataset, in case it must be used in a channel.
    next_env_labels: chex.Array  # [num_classes] * num_channels
    next_speaker_labels: chex.Array  # [num_classes] * num_speakers

    channel_map: chex.Array  # [num_speakers + num_channels] * [num_listeners] * num_channels
    env_images: chex.Array  # [image_size] * num_channels
    env_labels: chex.Array  # [num_classes] * num_channels
    speaker_labels: chex.Array  # [num_classes] * num_speakers

    speaker_images: chex.Array  # [image_size] * num_speakers

    previous_channel_map: chex.Array  # [num_speakers + num_channels] * [num_listeners] * num_channels
    previous_env_images: chex.Array  # [image_size] * num_channels
    previous_env_labels: chex.Array  # [num_classes] * num_channels
    previous_speaker_labels: chex.Array  # [num_classes] * num_speakers

    previous_speaker_images: chex.Array  # [image_size] * num_speakers

    iteration: int


class SimplifiedSignificationGame(MultiAgentEnv):
    def __init__(self, num_speakers: int, num_listeners: int, num_channels: int, num_classes: int, channel_ratio_fn: Callable, dataloader: jdl.DataLoader, image_dim: int, **kwargs: dict) -> None:
        super().__init__(num_agents=num_speakers + num_listeners)
        self.num_speakers = num_speakers
        self.num_listeners = num_listeners
        self.num_channels = num_channels    # We expect num_listeners to be equal to num_channels
        self.num_classes = num_classes
        self.channel_ratio_fn = channel_ratio_fn    # This function returns the ratio of the communication channels from the environment vs from the speakers. With 0 being all from the environment and 1 being all from the speakers.
        self.dataloader = dataloader    # We expect the dataloader to have batch_size=num_channels
        self.image_dim = image_dim
        self.kwargs = kwargs
        # TODO: Move the above comments to an actual docstring

        self.speaker_agents = ["speaker_{}".format(i) for i in range(num_speakers)]
        self.listener_agents = ["listener_{}".format(i) for i in range(num_listeners)]
        self.agents = self.speaker_agents + self.listener_agents

        self.observation_spaces = {**{agent: Discrete(num_classes) for agent in self.speaker_agents}, **{agent: Box(low=0, high=255, shape=(28, 28), dtype=jnp.float32) for agent in self.listener_agents}}
        self.action_spaces = {**{agent: Box(low=0, high=255, shape=(28, 28), dtype=jnp.float32) for agent in self.speaker_agents}, **{agent: Discrete(num_classes) for agent in self.listener_agents}}

    @partial(jax.jit, static_argnums=[0])
    def get_obs(self, state: State) -> dict:
        """Returns the observation for each agent."""
        
        @partial(jax.vmap, in_axes=[0, None])
        def _speaker_observation(aidx: int, state: State) -> jnp.ndarray:
            # The speakers need to see the newly generated assigned classes
            return state.next_speaker_labels[aidx]

        @partial(jax.vmap, in_axes=[0, None])
        def _listener_observation(aidx: int, state: State) -> jnp.ndarray:
            # The listeners need to see the newly generated images (which were generated from last-state's next_speaker_labels, i.e. speaker_labels) according to the channel map
            return state.channel_map[:, aidx][0]
        
        speaker_obs = _speaker_observation(jnp.arange(self.num_speakers), state)
        listener_obs = _listener_observation(jnp.arange(self.num_listeners), state)
        return {**{agent: speaker_obs[i] for i, agent in enumerate(self.speaker_agents)}, **{agent: listener_obs[i] for i, agent in enumerate(self.listener_agents)}}
    
    # @partial(jax.jit, static_argnums=[0])
    def step_env(self, key: chex.PRNGKey, state: State, actions: dict):
        """Performs a step in the environment."""
        
        speaker_actions = jnp.array([actions[agent] for agent in self.speaker_agents])
        listener_actions = jnp.array([actions[agent] for agent in self.listener_agents])

        ######## First, evaluate the current state.

        @partial(jax.vmap, in_axes=[0, None])
        def _evaluate_channel_rewards(c: int, listener_actions: jnp.ndarray) -> jnp.ndarray:
            channel = state.channel_map[c]
            speaker_index = channel[0]
            listener_index = channel[1]

            # Determine the correct label based on the speaker index
            is_speaker = speaker_index < self.num_speakers
            label = jnp.where(is_speaker, state.speaker_labels[speaker_index], state.env_labels[speaker_index-self.num_speakers])

            # Check if the listener's action matches the correct label and convert boolean to integer
            listener_correct = (listener_actions[listener_index] == label).astype(int)

            # Return reward based on whether the listener was correct
            reward = 2 * listener_correct - 1

            return speaker_index, listener_index, reward

        speaker_indices, listener_indices, rewards = _evaluate_channel_rewards(jnp.arange(self.num_channels), listener_actions)

        # Generate a reward vector containing all the rewards for each speaker and listener
        speaker_rewards = jnp.zeros(self.num_speakers + self.num_channels)
        listener_rewards = jnp.zeros(self.num_listeners)

        ##### This may also not be jittable #####
        # Iterate over the speaker_indices and listener_indices and add the rewards to the correct indices
        for i in range(self.num_channels):
            speaker_rewards = speaker_rewards.at[speaker_indices[i]].add(rewards[i])
            listener_rewards = listener_rewards.at[listener_indices[i]].add(rewards[i])
        ##### This may also not be jittable #####

        rewards = {**{agent: speaker_rewards[i] for i, agent in enumerate(self.speaker_agents)}, **{agent: listener_rewards[i] for i, agent in enumerate(self.listener_agents)}}
        rewards["__all__"] = sum(rewards.values())
        dones = {agent: False for agent in self.agents}
        dones["__all__"] = False


        ######## Then, update the state.

        key, k1, k2, k3, k4 = jax.random.split(key, 5)
        
        ##### This is the only line in this function that is not jittable #####
        next_env_images, next_env_labels = next(iter(self.dataloader))  # We expect the dataloader to have batch_size=num_channels.
        ##### This is the only line in this function that is not jittable #####

        next_speaker_labels = jax.random.randint(key, (self.num_speakers,), 0, self.num_classes)
        
        # We can take the first num_channels * channel_ratio_fn(iteration) elements from the speakers, and the rest from the environment, and then shuffle them.
        requested_num_speaker_images = jnp.floor(self.num_channels * self.channel_ratio_fn(state.iteration)).astype(int)
        # requested_num_speaker_images should not be greater than self.num_speakers
        # assert requested_num_speaker_images <= self.num_speakers, f"requested_num_speaker_images ({requested_num_speaker_images}) cannot be greater than self.num_speakers ({self.num_speakers})"
        
        # Collect num_speakers speakers
        speaker_ids = jax.random.permutation(k2, self.num_speakers)
        speaker_ids = jnp.pad(speaker_ids, (0, self.num_channels-self.num_speakers))
        # Collect num_env environment channels
        env_ids = jax.random.permutation(k3, self.num_channels) + self.num_speakers

        # Make a mask of size self.num_channels that is 1 if the index is less than requested_num_speaker_images, and 0 otherwise
        mask = jnp.where(jnp.arange(self.num_channels) < requested_num_speaker_images, 1, 0)
        possible_speakers = jnp.where(mask, speaker_ids, env_ids)
        speakers = jax.random.permutation(k4, possible_speakers).reshape((-1, 1))

        # listeners = jax.random.permutation(k1, self.num_listeners).reshape((-1, 1))[:self.num_channels]
        listeners = jax.lax.slice(jax.random.permutation(k1, self.num_listeners).reshape((-1, 1)), [0, 0], [self.num_channels, 1])
        next_channel_map = jnp.hstack((speakers, listeners))

        state = State(
            next_channel_map=next_channel_map,
            next_env_images=next_env_images,
            next_env_labels=next_env_labels,
            next_speaker_labels=next_speaker_labels,

            channel_map=state.next_channel_map,
            env_images=state.next_env_images,
            env_labels=state.next_env_labels,
            speaker_labels=state.next_speaker_labels,

            speaker_images=speaker_actions,

            previous_channel_map=state.channel_map,
            previous_env_images=state.env_images,
            previous_env_labels=state.env_labels,
            previous_speaker_labels=state.speaker_labels,

            previous_speaker_images=state.speaker_images,

            iteration=state.iteration + 1
        )
        
        return lax.stop_gradient(self.get_obs(state)), lax.stop_gradient(state), rewards, dones, {}
    
    def reset(self, key: chex.PRNGKey) -> Tuple[Dict, State]:
        """Reset the environment"""
        key, k1, k2, k3, k4 = jax.random.split(key, 5)
        
        ##### This is the only line in this function that is not jittable #####
        next_env_images, next_env_labels = next(iter(self.dataloader))  # We expect the dataloader to have batch_size=num_channels.
        ##### This is the only line in this function that is not jittable #####

        next_speaker_labels = jax.random.randint(key, (self.num_speakers,), 0, self.num_classes)
        
        # We can take the first num_channels * channel_ratio_fn(iteration) elements from the speakers, and the rest from the environment, and then shuffle them.
        requested_num_speaker_images = jnp.floor(self.num_channels * self.channel_ratio_fn(0)).astype(int)
        # requested_num_speaker_images should not be greater than self.num_speakers
        # assert requested_num_speaker_images <= self.num_speakers, f"requested_num_speaker_images ({requested_num_speaker_images}) cannot be greater than self.num_speakers ({self.num_speakers})"
        
        # Collect num_speakers speakers
        speaker_ids = jax.random.permutation(k2, self.num_speakers)
        speaker_ids = jnp.pad(speaker_ids, (0, self.num_channels-self.num_speakers))
        # Collect num_env environment channels
        env_ids = jax.random.permutation(k3, self.num_channels) + self.num_speakers

        # Make a mask of size self.num_channels that is 1 if the index is less than requested_num_speaker_images, and 0 otherwise
        mask = jnp.where(jnp.arange(self.num_channels) < requested_num_speaker_images, 1, 0)
        possible_speakers = jnp.where(mask, speaker_ids, env_ids)
        speakers = jax.random.permutation(k4, possible_speakers).reshape((-1, 1))

        # listeners = jax.random.permutation(k1, self.num_listeners).reshape((-1, 1))[:self.num_channels]
        listeners = jax.lax.slice(jax.random.permutation(k1, self.num_listeners).reshape((-1, 1)), [0, 0], [self.num_channels, 1])
        next_channel_map = jnp.hstack((speakers, listeners))

        state = State(
            next_channel_map=next_channel_map,
            next_env_images=next_env_images,
            next_env_labels=next_env_labels,
            next_speaker_labels=next_speaker_labels,

            channel_map=jnp.zeros_like(next_channel_map),
            env_images=jnp.zeros_like(next_env_images),
            env_labels=jnp.zeros_like(next_env_labels),
            speaker_labels=jnp.zeros_like(next_speaker_labels),

            speaker_images=jnp.zeros((max(self.num_speakers, 1), self.image_dim, self.image_dim), dtype=jnp.float32),  # This max is to avoid an error when num_speakers is 0 from the get_obs function

            previous_channel_map=jnp.zeros_like(next_channel_map),
            previous_env_images=jnp.zeros_like(next_env_images),
            previous_env_labels=jnp.zeros_like(next_env_labels),
            previous_speaker_labels=jnp.zeros_like(next_speaker_labels),

            previous_speaker_images=jnp.zeros((max(self.num_speakers, 1), self.image_dim, self.image_dim), dtype=jnp.float32),  # This max is to avoid an error when num_speakers is 0 from the get_obs function

            iteration=0
        )

        return self.get_obs(state), state
    
    def observation_space(self, agent: str):
        """Observation space for a given agent."""
        return self.observation_spaces[agent]

    def action_space(self, agent: str):
        """Action space for a given agent."""
        return self.action_spaces[agent]

    @property
    def agent_classes(self) -> dict:
        """Returns a dictionary with agent classes, used in environments with hetrogenous agents."""
        return {"speakers": self.speaker_agents, "listeners": self.listener_agents}


def mnist_signification_game():
    """Runs a simplified signification game on MNIST."""
    
    class FlattenAndCast(object):
        def __call__(self, pic):
            return np.ravel(np.array(pic, dtype=jnp.float32))
    
    class JustCast(object):
        def __call__(self, pic):
            return np.array(pic, dtype=jnp.float32)
        
    def ret_0(iteration):
        return 0.0
    
    # Define parameters for a signification game
    num_speakers = 10
    num_listeners = 10
    num_channels = 10
    num_classes = 10

    mnist_dataset = MNIST('/tmp/mnist/', download=True, transform=JustCast())
    # If we use FlattenAndCast(), the images are flattened to be 1D arrays, otherwise they are 2D arrays
    # The shape of the images will be (60000, 28, 28) if we use JustCast(), otherwise it is (60000, 784)

    dataloader = jdl.DataLoader(mnist_dataset, 'pytorch', batch_size=num_listeners, shuffle=True)
    # The batch size is the number of listeners, so calling next(training_generator) will return a tuple of length 2,
    # where the first element is a batch of num_listeners images and the second element is a batch of num_listener labels.
    
    # Run this to verify the dataloader works:
    # batch = next(iter(dataloader))
    # print(batch)

    key = jax.random.PRNGKey(0)
    key, key_reset, key_act, key_step = jax.random.split(key, 4)
    
    env = SimplifiedSignificationGame(num_speakers, num_listeners, num_channels, num_classes, channel_ratio_fn=ret_0, dataloader=dataloader, image_dim=28)
    obs, state = env.reset(key_reset)
    
    print(list(obs.keys()))
    print(obs)
    print(state)

    action_keys = jax.random.split(key_act, len(env.agents))
    actions = {agent: env.action_space(agent).sample(action_keys[i]) for i, agent in enumerate(env.agents)}

    for agent, agent_action in actions.items():
        if agent.startswith("speaker"):
            continue
        print(f"Action for {agent}: {agent_action}")

    key, key_step = jax.random.split(key_step, 2)
    obs, state, reward, done, infos = env.step(key_step, state, actions)
    
    key, key_step = jax.random.split(key_step, 2)
    obs, state, reward, done, infos = env.step(key_step, state, actions)

    key, key_step = jax.random.split(key_step, 2)
    obs, state, reward, done, infos = env.step(key_step, state, actions)

    print(state.previous_channel_map)
    print(state.previous_speaker_labels)

    for agent, agent_reward in reward.items():
        print(f"Reward for {agent}: {agent_reward}")

    # for channel in state.previous_channel_map:
    #     print(f"Channel {channel}: speaker spoke {state.previous_speaker_labels[channel[0]]} and listener heard {actions[f'listener_{channel[1]}']}")


if __name__ == '__main__':
    mnist_signification_game()
    # You can run this file with python -W 'ignore' improved_signification_game.py to ignore the warnings
