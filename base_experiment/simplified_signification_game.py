from functools import partial
import numpy as np
import jax, chex
import jax.numpy as jnp
import jax_dataloader as jdl
from torchvision.datasets import MNIST
from flax import struct
from jaxmarl.environments.multi_agent_env import MultiAgentEnv


# The chex.Arrays could alternatively be jnp.ndarrays, should test for performance
@struct.dataclass
class State:
    # Newly generated, to be evaluated next state
    new_speaker_class_assignment: chex.Array  # [num_classes] * num_speakers
    new_env_images: chex.Array  # [image_size] * num_listeners
    new_listener_image_from_env_or_speaker_mask: chex.Array  # [bool] * num_listeners
    new_listener_channel_assignment: chex.Array  # [max(num_speakers, num_listers)] * num_listeners

    # Newly generated as a function of previous state, to be evaluated this state
    speaker_images: chex.Array  # [image_size] * num_speakers

    # From previous state
    old_env_images: chex.Array  # [image_size] * num_listeners
    old_listener_image_from_env_or_speaker_mask: chex.Array  # [bool] * num_listeners
    old_listener_channel_assignment: chex.Array  # [max(num_speakers, num_listers)] * num_listeners

# The current state changes based on whether it's the speaker or listeners turn, but we could change that
# so that at timestep t, the speaker generates an image, and the listener guesses the class of the image at timestep t+1. The listener will still guess the image at time-step t, but it will be based on the image generated at time-step t-1.

class SimplifiedSignificationGame(MultiAgentEnv):
    def __init__(self, num_speakers: int, num_listeners: int, num_classes: int, dataloader: jdl.DataLoader) -> None:
        super().__init__(num_agents=num_speakers + num_listeners)
        self.num_speakers = num_speakers
        self.num_listeners = num_listeners
        self.num_classes = num_classes
        self.dataloader = dataloader

    @partial(jax.jit, static_argnums=[0])
    def get_obs(self, state: State) -> dict:
        """Returns the observation for each agent."""

        @partial(jax.vmap, in_axes=[0, None])
        def _observation(aidx: int, state: State) -> jnp.ndarray:
            # Agents 0 through num_speakers-1 are speakers and see speaker_class_assignment[i]
            # Agents num_speakers through num_speakers + num_listeners - 1 are listeners and see listener_image_assignment[i]
            if aidx < self.num_speakers:
                return state.speaker_class_assignment[aidx]
            else:
                return state.listener_image_assignment[aidx - self.num_speakers]
        
        # Here is an alternate version using jax.lax.cond which may or may not be faster:
        # def _observation(aidx: int, state: State) -> jnp.ndarray:
        #     return jax.lax.cond(aidx < self.num_speakers,
        #                         lambda _: state.speaker_class_assignment[aidx],
        #                         lambda _: state.listener_image_assignment[aidx - self.num_speakers],
        #                         operand=None)

        obs = _observation(self.agent_range, state)
        return {a: obs[i] for i, a in enumerate(self.agents)}
    
    @partial(jax.jit, static_argnums=[0])
    def step_env(self, key: chex.PRNGKey, state: State, actions: dict):
        """Performs a step in the environment."""
        
        actions = jnp.array([actions[i] for i in self.agents])
        # The first num_speakers actions are images, the last num_listeners actions are classes

        # ACTIONS: The listeners report their classification from the last images, the speakers report the image they generated from the last classes
        # CURRENT STATE: The last images and classes

        # NEXT STATE:
        # New: The speaker_class_assignment is updated with the labels from the dataloader, next(dataloader)[1]
        # New: The env_images are updated with the images from the dataloader, next(dataloader)[0]
        # New: listener_image_from_env_or_speaker_mask is updated with a random array of booleans, jax.random.randint(key, (self.num_listeners,), 0, 2).astype(bool)
        # New: listener_channel_assignment is updated based on the value of listener_image_from_env_or_speaker_mask
        # New: speaker_images are updated with the actions from the speakers
        # Old: old_env_images, old_listener_image_from_env_or_speaker_mask, and old_listener_channel_assignment are updated with the values from the current state

        new_env_images, new_speaker_class_assignment = next(self.dataloader)
        new_listener_image_from_env_or_speaker_mask = jax.random.randint(key, (self.num_listeners,), 0, 2).astype(bool) # When true we use the env image, when false we use the speaker image

        _listener_channel_assignment_env = jax.random.randint(key, (self.num_listeners,), 0, self.num_listeners)
        _listener_channel_assignment_speaker = jax.random.randint(key, (self.num_listeners,), 0, self.num_speakers)
        new_listener_channel_assignment = jnp.where(new_listener_image_from_env_or_speaker_mask, _listener_channel_assignment_env, _listener_channel_assignment_speaker)

        state = State(
            new_speaker_class_assignment=new_speaker_class_assignment,
            new_env_images=new_env_images,
            new_listener_image_from_env_or_speaker_mask=new_listener_image_from_env_or_speaker_mask,
            new_listener_channel_assignment=new_listener_channel_assignment,
            speaker_images=actions[:self.num_speakers],
            old_env_images=state.new_env_images,
            old_listener_image_from_env_or_speaker_mask=state.new_listener_image_from_env_or_speaker_mask,
            old_listener_channel_assignment=state.new_listener_channel_assignment
        )


def mnist_signification_game():
    """Runs a simplified signification game on MNIST."""
    
    class FlattenAndCast(object):
        def __call__(self, pic):
            return np.ravel(np.array(pic, dtype=jnp.float32))
    
    class JustCast(object):
        def __call__(self, pic):
            return np.array(pic, dtype=jnp.float32)
    
    # Define parameters for a signification game
    num_speakers = 0
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

    env = SimplifiedSignificationGame(num_speakers, num_listeners, num_classes, dataloader)


if __name__ == '__main__':
    mnist_signification_game()
