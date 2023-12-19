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
        
        speaker_actions = jnp.array([actions[f"speaker_{i}"] for i in range(self.num_speakers)])
        listener_actions = jnp.array([actions[f"listener_{i}"] for i in range(self.num_listeners)])
        # The first num_speakers actions are images, the last num_listeners actions are classes

        # ACTIONS: The listeners report their classification from the last images, the speakers report the image they generated from the last classes
        # CURRENT STATE: The last images and classes

        # NEXT STATE:
        # New: The channel_class_assignment is updated with the labels from the dataloader, next(dataloader)[1]
        # New: The env_images are updated with the images from the dataloader, next(dataloader)[0]
        # New: listener_image_from_env_or_speaker_mask is updated with a random array of booleans, jax.random.randint(key, (self.num_listeners,), 0, 2).astype(bool)
        # New: listener_channel_assignment is updated based on the value of listener_image_from_env_or_speaker_mask
        # New: speaker_images are updated with the actions from the speakers
        # Old: old_env_images, old_listener_image_from_env_or_speaker_mask, and old_listener_channel_assignment are updated with the values from the current state
        
        key1, key2, key3 = jax.random.split(key, 3)
        new_env_images, new_channel_class_assignment = next(iter(self.dataloader))    # This is not jittable :(

        # When indices of new_listener_image_from_env_or_speaker_mask are true we use the env image, when false we use the speaker image
        if self.kwargs.get("env_images_only", False):    # This is static, so it should be jittable
            new_listener_image_from_env_or_speaker_mask = jnp.full((self.num_listeners,), True)
        else:
            new_listener_image_from_env_or_speaker_mask = jax.random.randint(key1, (self.num_listeners,), 0, 2).astype(bool)
        
        # The listener channel assignment is either the environment channel assignment or the speaker channel assignment, depending on the value of new_listener_image_from_env_or_speaker_mask
        _listener_channel_assignment_env = jax.random.randint(key2, (self.num_listeners,), 0, self.num_listeners)
        _listener_channel_assignment_speaker = jax.random.randint(key3, (self.num_listeners,), 0, self.num_speakers)
        new_listener_channel_assignment = jnp.where(new_listener_image_from_env_or_speaker_mask, _listener_channel_assignment_env, _listener_channel_assignment_speaker)

        state = State(
            new_channel_class_assignment=new_channel_class_assignment,
            new_env_images=new_env_images,
            new_listener_image_from_env_or_speaker_mask=new_listener_image_from_env_or_speaker_mask,
            new_listener_channel_assignment=new_listener_channel_assignment,
            speaker_images=speaker_actions if self.num_speakers > 0 else jnp.zeros((1, 28, 28), dtype=jnp.float32),  # This max is to avoid an error when num_speakers is 0 from the get_obs function. This is static, so it should be jittable
            old_env_images=state.new_env_images,
            old_channel_class_assignment=state.new_channel_class_assignment,
            old_listener_image_from_env_or_speaker_mask=state.new_listener_image_from_env_or_speaker_mask,
            old_listener_channel_assignment=state.new_listener_channel_assignment,
            iteration=state.iteration + 1
        )

        dones = {agent: False for agent in self.agents}
        dones["__all__"] = False

        # The listener was attempting to classify the channel, we have the channel labels but they are in a different order.
        # old_listener_channel_assignment tells us which channel the listener was attempting to classify, and old_channel_class_assignment tells us the class of that channel.
        listener_rewards = jnp.where(listener_actions == state.old_channel_class_assignment[state.old_listener_channel_assignment], 1, -1)
        # Also give the same rewards to the speakers who spoke the images that were classified correctly, as per state.old_listener_image_from_env_or_speaker_mask and state.old_listener_channel_assignment
        # print(listener_rewards)
        speaker_rewards = jnp.where(jnp.logical_and(~state.old_listener_image_from_env_or_speaker_mask, listener_rewards != 0), listener_rewards, 0)
        # I need to rearrange the speaker awards according to the channel assignment, so that the speaker rewards are in the same order as the speaker actions
        # Rearrange the speaker rewards so they are in the same order as the speaker actions
        speaker_rewards = jnp.take(speaker_rewards, state.old_listener_channel_assignment)  # For some reason this (sorta) works, but I don't know why
        speaker_rewards = jnp.take(speaker_rewards, state.old_listener_channel_assignment)

        rewards = {**{"speaker_{}".format(i): speaker_rewards[i] for i in range(self.num_speakers)}, **{"listener_{}".format(i): listener_rewards[i] for i in range(self.num_listeners)}}
        rewards["__all__"] = sum(rewards.values())

        return lax.stop_gradient(self.get_obs(state)), lax.stop_gradient(state), rewards, dones, {}
    
    def reset(self, key: chex.PRNGKey) -> Tuple[Dict, State]:
        """Reset the environment"""
        key, k1, k2, k3, k4 = jax.random.split(key, 5)
        
        ##### This is the only line in this function that is not jittable #####
        next_env_images, next_env_labels = next(iter(self.dataloader))  # We expect the dataloader to have batch_size=num_channels.
        ##### This is the only line in this function that is not jittable #####

        next_speaker_labels = jax.random.randint(key, (self.num_speakers,), 0, self.num_classes)

        listeners = jax.random.permutation(k1, self.num_listeners).reshape((-1, 1))[:self.num_channels]        
        # We can take the first num_channels * channel_ratio_fn(iteration) elements from the speakers, and the rest from the environment, and then shuffle them.
        # First of all decide on an actual number, the closest integer to num_channels * channel_ratio_fn(iteration)
        num_speakers = jnp.floor(self.num_channels * self.channel_ratio_fn(0)).astype(int)
        num_env = self.num_channels - num_speakers
        
        # Collect num_speakers speakers
        speaker_ids = jax.random.permutation(k2, self.num_speakers)[:num_speakers].reshape((-1, 1))
        # Collect num_env environment channels
        env_ids = jax.random.permutation(k3, self.num_channels)[:num_env].reshape((-1, 1))
        # Concatenate the two arrays and shuffle them
        possible_speakers = jnp.vstack((speaker_ids, env_ids))
        speakers = jax.random.permutation(k4, possible_speakers)

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
        return 0
    
    # Define parameters for a signification game
    num_speakers = 5
    num_listeners = 5
    num_channels = 5
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

    return

    action_keys = jax.random.split(key_act, len(env.agents))
    actions = {agent: env.action_space(agent).sample(action_keys[i]) for i, agent in enumerate(env.agents)}

    for agent, agent_action in actions.items():
        if agent.startswith("speaker"):
            continue
        print(f"Action for {agent}: {agent_action}")

    obs, state, reward, done, infos = env.step(key_step, state, actions)    # Running this twice, since the first state cannot yield reward.
    obs, state, reward, done, infos = env.step(key_step, state, actions)

    # print(obs, state, reward, done, infos)
    # print("True classes:", state.old_channel_class_assignment)
    print(state.old_listener_image_from_env_or_speaker_mask)
    print("Channel assigment:", state.old_listener_channel_assignment)
    # print(reward)

    for i in range(num_listeners):
        print(f"listener_{i} attended to {'env channel' if state.old_listener_image_from_env_or_speaker_mask[i] else 'speaker'} {state.old_listener_channel_assignment[i]} with true class {state.old_channel_class_assignment[state.old_listener_channel_assignment[i]]}")

    for agent, agent_reward in reward.items():
        print(f"Reward for {agent}: {agent_reward}")


if __name__ == '__main__':
    mnist_signification_game()
    # You can run this file with python -W 'ignore' improved_signification_game.py to ignore the warnings
