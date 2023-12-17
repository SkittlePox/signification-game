from functools import partial
import numpy as np
import jax, chex
import jax.numpy as jnp
from jax.tree_util import tree_map
from torch.utils import data
from torchvision.datasets import MNIST
from flax import struct
from jaxmarl.environments.multi_agent_env import MultiAgentEnv

# Likely need to make a dataclass for the state, this is the underlying state of the environment. Later there will be a get_obs method in the environment that will return the observation of the agent.
@struct.dataclass
class State:
    speaker_class_assignment: chex.Array  # [num_classes] * num_speakers
    listener_image_assignment: chex.Array  # [image_size] * num_listeners
    speaker_to_listener_map: chex.Array  # [num_speakers] * num_listeners
    previous_speaker_to_listener_map: chex.Array  # [num_speakers] * num_listeners

# The current state changes based on whether it's the speaker or listeners turn, but we could change that
# so that at timestep t, the speaker generates an image, and the listener guesses the class of the image at timestep t+1. The listener will still guess the image at time-step t, but it will be based on the image generated at time-step t-1.

class SimplifiedSignificationGame(MultiAgentEnv):
    def __init__(self, num_speakers: int, num_listeners: int, num_classes: int, dataset) -> None:
        super().__init__(num_agents=num_speakers + num_listeners)
        self.num_speakers = num_speakers
        self.num_listeners = num_listeners
        self.num_classes = num_classes
        self.dataset = dataset  # This is a tfds dataset, which is a generator that returns a dictionary with keys 'image' and 'label'. It's batched so that each batch has num_listeners images and labels.

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
        
        # get the actions as array
        actions = jnp.array([actions[i] for i in self.agents])
        # The first num_speakers actions are images, the last num_listeners actions are classes
        
        # The listener should classify the images in listener_image_assignment and return an integer. The reward for both agents is a function of this.
        
        # At the next state, speaker_class_assignment is a random array of integers between 0 and num_classes-1
        # At the next state, listener_image_assignment is the actions of the speakers re-arranged to match speaker_to_listener_map
        # The speaker_to_listener_map is an array of integers between 0 and num_speakers-1, where speaker_to_listener_map[i] is the index of the speaker whose image the listener at index i sees.
        
        # Arrange the actions of the speakers to match the speaker_to_listener_map
        speaker_actions = actions[:self.num_speakers]
        listener_image_assignment = jnp.take(speaker_actions, state.speaker_to_listener_map)

        state = State(
            speaker_class_assignment=jax.random.randint(key, (self.num_speakers,), 0, self.num_classes),
            listener_image_assignment=listener_image_assignment,
            speaker_to_listener_map=jax.random.randint(key, (self.num_listeners,), 0, self.num_speakers),
            previous_speaker_to_listener_map=state.speaker_to_listener_map
        )

def numpy_collate(batch):
    return tree_map(np.asarray, data.default_collate(batch))

class NumpyLoader(data.DataLoader):
        def __init__(self, dataset, batch_size=1,
                        shuffle=False, sampler=None,
                        batch_sampler=None, num_workers=0,
                        pin_memory=False, drop_last=False,
                        timeout=0, worker_init_fn=None):
            super(self.__class__, self).__init__(dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                sampler=sampler,
                batch_sampler=batch_sampler,
                num_workers=num_workers,
                collate_fn=numpy_collate,
                pin_memory=pin_memory,
                drop_last=drop_last,
                timeout=timeout,
                worker_init_fn=worker_init_fn)

class FlattenAndCast(object):
    def __call__(self, pic):
        return np.ravel(np.array(pic, dtype=jnp.float32))

def mnist_signification_game():
    """Runs a signification game on MNIST."""
    
    # Define parameters for a signification game
    num_speakers = 0    # Speakers don't do anything at the moment
    num_listeners = 5
    num_classes = 10

    # Retrieve mnist dataset from torchvision
    mnist_dataset = MNIST('/tmp/mnist/', download=True)
    # training_generator = NumpyLoader(mnist_dataset, batch_size=num_listeners, num_workers=0)

    # train_images = np.array(mnist_dataset.train_data).reshape(len(mnist_dataset.train_data), -1)
    # train_labels = one_hot(np.array(mnist_dataset.train_labels), n_targets)

    # env = SimplifiedSignificationGame(num_speakers, num_listeners, num_classes, dataset=ds)


if __name__ == '__main__':
    mnist_signification_game()
