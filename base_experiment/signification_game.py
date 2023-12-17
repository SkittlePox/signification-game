from functools import partial

import jax, chex
import jax.numpy as jnp
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

# I think this class should be very simple to implement.
class SignificationGame(MultiAgentEnv):
    def __init__(self, num_speakers: int, num_listeners: int, num_classes: int) -> None:
        super().__init__(num_agents=num_speakers + num_listeners)
        self.num_speakers = num_speakers
        self.num_listeners = num_listeners
        self.num_classes = num_classes

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

        state = State(
            speaker_class_assignment=jax.random.randint(key, (self.num_speakers,), 0, self.num_classes),
            listener_image_assignment=,
            speaker_to_listener_map=jax.random.randint(key, (self.num_listeners,), 0, self.num_speakers),
            previous_speaker_to_listener_map=state.speaker_to_listener_map
        )

