from functools import partial
from typing import Tuple, Dict
import numpy as np
import jax, chex
import jax.numpy as jnp
import jax_dataloader as jdl
from jax import lax
from torchvision.datasets import MNIST
from flax import struct
from jaxmarl.environments.multi_agent_env import MultiAgentEnv
from gymnax.environments.spaces import Discrete, Box


# The chex.Arrays could alternatively be jnp.ndarrays, should test for performance
@struct.dataclass
class State:
    # Newly generated, to be evaluated next state after speakers generate images
    new_channel_class_assignment: chex.Array  # [num_classes] * num_speakers
    new_env_images: chex.Array  # [image_size] * num_listeners
    new_listener_image_from_env_or_speaker_mask: chex.Array  # [bool] * num_listeners
    new_listener_channel_assignment: chex.Array  # [max(num_speakers, num_listers)] * num_listeners

    # Newly generated as a function of previous state, to be evaluated this state
    speaker_images: chex.Array  # [image_size] * num_speakers

    # From previous state, to be evaluated this state
    old_env_images: chex.Array  # [image_size] * num_listeners
    old_channel_class_assignment: chex.Array  # [num_classes] * num_speakers
    old_listener_image_from_env_or_speaker_mask: chex.Array  # [bool] * num_listeners
    old_listener_channel_assignment: chex.Array  # [max(num_speakers, num_listers)] * num_listeners

    iteration: int

# The current state changes based on whether it's the speaker or listeners turn, but we could change that
# so that at timestep t, the speaker generates an image, and the listener guesses the class of the image at timestep t+1. The listener will still guess the image at time-step t, but it will be based on the image generated at time-step t-1.

class SimplifiedSignificationGame(MultiAgentEnv):
    def __init__(self, num_speakers: int, num_listeners: int, num_classes: int, dataloader: jdl.DataLoader) -> None:
        super().__init__(num_agents=num_speakers + num_listeners)
        self.num_speakers = num_speakers
        self.num_listeners = num_listeners
        self.num_classes = num_classes
        self.dataloader = dataloader

        self.agents = ["speaker_{}".format(i) for i in range(num_speakers)] + ["listener_{}".format(i) for i in range(num_listeners)]

        # TODO: Add agent classes and fix this
        self.speaker_agents = None
        self.listener_agents = None

        self.observation_spaces = {**{"speaker_{}".format(i): Discrete(num_classes) for i in range(num_speakers)}, **{"listener_{}".format(i): Box(low=0, high=255, shape=(28, 28), dtype=jnp.float32) for i in range(num_listeners)}}
        self.action_spaces = {**{"speaker_{}".format(i): Box(low=0, high=255, shape=(28, 28), dtype=jnp.float32) for i in range(num_speakers)}, **{"listener_{}".format(i): Discrete(num_classes) for i in range(num_listeners)}}

    @partial(jax.jit, static_argnums=[0])
    def get_obs(self, state: State) -> dict:
        """Returns the observation for each agent."""

        @partial(jax.vmap, in_axes=[0, None])
        def _observation(aidx: int, state: State) -> jnp.ndarray:
            # The speakers see the classes, the listeners see the images
            return jnp.where(aidx < self.num_speakers, 
                             state.new_channel_class_assignment[aidx],
                             jnp.where(state.new_listener_image_from_env_or_speaker_mask[aidx], 
                                       state.new_env_images[state.new_listener_channel_assignment[aidx]], 
                                       state.speaker_images[state.new_listener_channel_assignment[aidx]]))

        obs = _observation(jnp.arange(self.num_speakers+self.num_listeners), state)
        return {a: obs[i] for i, a in enumerate(self.agents)}
    
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
        new_env_images, new_speaker_class_assignment = next(iter(self.dataloader))    # This is not jittable :(

        # new_listener_image_from_env_or_speaker_mask = jax.random.randint(key1, (self.num_listeners,), 0, 2).astype(bool) # When true we use the env image, when false we use the speaker image
        new_listener_image_from_env_or_speaker_mask = jnp.full((self.num_listeners,), True)

        _listener_channel_assignment_env = jax.random.randint(key2, (self.num_listeners,), 0, self.num_listeners)
        _listener_channel_assignment_speaker = jax.random.randint(key3, (self.num_listeners,), 0, self.num_speakers)
        new_listener_channel_assignment = jnp.where(new_listener_image_from_env_or_speaker_mask, _listener_channel_assignment_env, _listener_channel_assignment_speaker)

        state = State(
            new_channel_class_assignment=new_speaker_class_assignment,
            new_env_images=new_env_images,
            new_listener_image_from_env_or_speaker_mask=new_listener_image_from_env_or_speaker_mask,
            new_listener_channel_assignment=new_listener_channel_assignment,
            speaker_images=speaker_actions,
            old_env_images=state.new_env_images,
            old_channel_class_assignment=state.new_channel_class_assignment,
            old_listener_image_from_env_or_speaker_mask=state.new_listener_image_from_env_or_speaker_mask,
            old_listener_channel_assignment=state.new_listener_channel_assignment,
            iteration=state.iteration + 1
        )

        dones = {agent: False for agent in self.agents}
        dones["__all__"] = False

        # TODO: Calculate reward based on the listeners actions and the old_channel_class_assignment
        rewards = {agent: 0 for agent in self.agents}
        rewards["__all__"] = 0

        print(type(state))

        return lax.stop_gradient(self.get_obs(state)), lax.stop_gradient(state), rewards, dones, {}
    
    def reset(self, key: chex.PRNGKey) -> Tuple[Dict, State]:
        """Reset the environment"""

        new_env_images, new_speaker_class_assignment = next(iter(self.dataloader))

        key1, key2, key3 = jax.random.split(key, 3)
        
        # new_listener_image_from_env_or_speaker_mask = jax.random.randint(key1, (self.num_listeners,), 0, 2).astype(bool) # When true we use the env image, when false we use the speaker image
        new_listener_image_from_env_or_speaker_mask = jnp.full((self.num_listeners,), True)

        _listener_channel_assignment_env = jax.random.randint(key2, (self.num_listeners,), 0, self.num_listeners)
        _listener_channel_assignment_speaker = jax.random.randint(key3, (self.num_listeners,), 0, self.num_speakers)
        new_listener_channel_assignment = jnp.where(new_listener_image_from_env_or_speaker_mask, _listener_channel_assignment_env, _listener_channel_assignment_speaker)

        state = State(
            new_channel_class_assignment=new_speaker_class_assignment,
            new_env_images=new_env_images,
            new_listener_image_from_env_or_speaker_mask=new_listener_image_from_env_or_speaker_mask,
            new_listener_channel_assignment=new_listener_channel_assignment,
            speaker_images=jnp.zeros((max(self.num_speakers, 1), 28, 28), dtype=jnp.float32),  # This max is to avoid an error when num_speakers is 0 from the get_obs function
            old_env_images=jnp.zeros((self.num_listeners, 28, 28), dtype=jnp.float32),
            old_channel_class_assignment=jnp.zeros((self.num_listeners,), dtype=jnp.int32),
            old_listener_image_from_env_or_speaker_mask=jnp.full((self.num_listeners,), True),
            old_listener_channel_assignment=jnp.zeros((self.num_listeners,), dtype=jnp.int32),
            iteration=0
        )
        # print(state.speaker_images.shape)
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
    
    # Define parameters for a signification game
    num_speakers = 2
    num_listeners = 5
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
    
    env = SimplifiedSignificationGame(num_speakers, num_listeners, num_classes, dataloader)
    obs, state = env.reset(key_reset)
    
    print(list(obs.keys()))
    print(obs)
    print(state)

    actions = {agent: env.action_space(agent).sample(key_act) for i, agent in enumerate(env.agents)}
    
    print(actions)

    obs, state, reward, done, infos = env.step(key_step, state, actions)

    print(obs, state, reward, done, infos)



if __name__ == '__main__':
    mnist_signification_game()
