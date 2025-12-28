from functools import partial
from typing import Tuple, Dict, Callable, Union
import numpy as np
import jax, chex
import jax.numpy as jnp
from jax import lax
from torchvision.datasets import MNIST
from flax import struct
from jaxmarl.environments.multi_agent_env import MultiAgentEnv
from gymnax.environments.spaces import Discrete, Box
from utils import (
    get_channel_ratio_fn,
    get_speaker_referent_span_fn,
    get_reward_parity_fn,
    get_speaker_action_transform,
    get_agent_inferential_mode_fn,
    speaker_penalty_whitesum_fn,
    speaker_penalty_curve_fn,
    speaker_penalty_right_angle_fn,
    speaker_penalty_right_angle_or_straight_fn,
    speaker_penalty_similar_curve_fn,
    speaker_penalty_too_close_to_border_fn,
    speaker_penalty_spline_continuity_fn,
    speaker_penalty_zipfian_size_fn,
    create_unitary_channel_map,
)
import math

from utils import to_jax, center_obs, shift_obs


# The first num_speakers channel indices refer to speaker generated images,
#  while the remaining num_channel channel indices refer to env generated images

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
    listener_obs_source: chex.Array # [1, 0] * num_listeners (1 if from a speaker, 0 if from environment)

    speaker_images: chex.Array  # [image_size] * num_speakers

    previous_channel_map: chex.Array  # [num_speakers + num_channels] * [num_listeners] * num_channels
    previous_env_images: chex.Array  # [image_size] * num_channels
    previous_env_labels: chex.Array  # [num_classes] * num_channels
    previous_speaker_labels: chex.Array  # [num_classes] * num_speakers

    previous_speaker_images: chex.Array  # [image_size] * num_speakers
    previous_speaker_actions: chex.Array # [num_speakers] * speaker_action_dim

    iteration: int
    epoch: int
    requested_num_speaker_images: int
    requested_speaker_referent_span: int
    agent_inferential_mode: float


class SimplifiedSignificationGame(MultiAgentEnv):
    def __init__(
        self,
        num_speakers: int,
        num_listeners: int,
        num_channels: int,
        num_classes: int,
        channel_ratio_fn: Union[Callable, str],
        speaker_referent_span_fn: Union[Callable, str],
        reward_parity_fn: Union[Callable, str],
        agent_inferential_mode_fn: Union[Callable, str],
        speaker_action_transform: Union[Callable, str],
        speaker_action_dim: int,
        dataset: tuple,
        image_dim: int,
        speaker_reward_success: float = 1.0,
        speaker_reward_failure: float = -0.1,
        listener_reward_success: float = 1.0,
        listener_reward_failure: float = -0.1,
        log_prob_rewards: bool = False,
        speaker_whitesum_penalty_coef: float = 0.0,
        speaker_curve_penalty_coef: float = 0.0,
        speaker_right_angle_penalty_coef: float = 0.0,
        speaker_right_angle_or_straight_penalty_coef: float = 0.0,
        speaker_similar_curve_penalty_coef: float = 0.0,
        speaker_too_close_to_penalty_coef: float = 0.3,
        speaker_spline_continuity_penalty_coef: float = 0.0,
        speaker_zipfian_size_penalty_coef: float = 0.0,
        gaussian_noise_stddev: float = 0.0,
        speaker_assignment_method: str = "random",
        mandate_unitary_channel_map: bool = False,
        center_and_reshuffle_listener_obs: bool = False,
        **kwargs: dict,
    ) -> None:
        super().__init__(num_agents=num_speakers + num_listeners)
        self.num_speakers = num_speakers
        self.num_listeners = num_listeners
        self.num_channels = num_channels    # We expect num_listeners to be equal to num_channels
        self.num_classes = num_classes
        self.speaker_action_dim = speaker_action_dim
        self.channel_ratio_fn = get_channel_ratio_fn(channel_ratio_fn, kwargs) if isinstance(channel_ratio_fn, str) else lambda _: channel_ratio_fn if isinstance(channel_ratio_fn, int) else channel_ratio_fn
        self.speaker_referent_span_fn = get_speaker_referent_span_fn(speaker_referent_span_fn, kwargs) if isinstance(speaker_referent_span_fn, str) else lambda _: speaker_referent_span_fn if isinstance(speaker_referent_span_fn, int) else speaker_referent_span_fn
        self.reward_parity_fn = get_reward_parity_fn(reward_parity_fn, kwargs) if isinstance(reward_parity_fn, str) else reward_parity_fn
        self.agent_inferential_mode_fn = get_agent_inferential_mode_fn(str(agent_inferential_mode_fn), kwargs)
        self.speaker_action_transform = get_speaker_action_transform(speaker_action_transform, image_dim) if isinstance(speaker_action_transform, str) else speaker_action_transform
        self.speaker_whitesum_penalty_coef = speaker_whitesum_penalty_coef
        self.speaker_curve_penalty_coef = speaker_curve_penalty_coef
        self.speaker_right_angle_penalty_coef = speaker_right_angle_penalty_coef
        self.speaker_right_angle_or_straight_penalty_coef = speaker_right_angle_or_straight_penalty_coef
        self.speaker_similar_curve_penalty_coef = speaker_similar_curve_penalty_coef
        self.speaker_too_close_to_penalty_coef = speaker_too_close_to_penalty_coef
        self.speaker_spline_continuity_penalty_coef = speaker_spline_continuity_penalty_coef
        self.speaker_zipfian_size_penalty_coef = speaker_zipfian_size_penalty_coef
        self.stored_env_images = dataset[0]
        self.stored_env_labels = dataset[1]
        self.image_dim = image_dim
        self.speaker_reward_success = speaker_reward_success
        self.speaker_reward_failure = speaker_reward_failure
        self.listener_reward_success = listener_reward_success
        self.listener_reward_failure = listener_reward_failure
        self.log_prob_rewards = log_prob_rewards
        self.gaussian_noise_stddev = gaussian_noise_stddev
        self.speaker_assignment_method = speaker_assignment_method
        self.center_and_reshuffle_listener_obs = center_and_reshuffle_listener_obs
        self.mandate_unitary_channel_map = mandate_unitary_channel_map
        self.kwargs = kwargs

        self.speaker_agents = ["speaker_{}".format(i) for i in range(num_speakers)]
        self.listener_agents = ["listener_{}".format(i) for i in range(num_listeners)]
        self.agents = self.speaker_agents + self.listener_agents

        self.observation_spaces = {**{agent: Discrete(num_classes) for agent in self.speaker_agents}, **{agent: Box(low=0, high=255, shape=(image_dim, image_dim), dtype=jnp.float32) for agent in self.listener_agents}}
        self.action_spaces = {**{agent: Box(low=0, high=255, shape=(image_dim, image_dim), dtype=jnp.float32) for agent in self.speaker_agents}, **{agent: Discrete(num_classes) for agent in self.listener_agents}}  # TODO migrate: This may need to change, unsure. Sampling randomly from this may fail.

    @partial(jax.jit, static_argnums=(0,))
    def load_images(self, key: chex.PRNGKey, num_imgs: int = -1):
        """Returns a random set of images and labels."""
        num_imgs = self.num_channels if num_imgs == -1 else num_imgs

        key, subkey = jax.random.split(key)
        indices = jax.random.randint(subkey, shape=(num_imgs,), minval=0, maxval=len(self.stored_env_images))
        
        images = self.stored_env_images[indices]
        labels = self.stored_env_labels[indices]
        return images, labels
    
    
    @partial(jax.jit, static_argnums=[0, 3])
    def get_obs(self, key: chex.PRNGKey, state: State, as_dict: bool = False):
        """Returns the observation for each agent."""
        
        @partial(jax.vmap, in_axes=[0, None])
        def _speaker_observation(aidx: int, state: State) -> jnp.ndarray:
            # The speakers need to see the newly generated assigned classes
            return state.next_speaker_labels[aidx]

        @partial(jax.vmap, in_axes=[0, 0, None])
        def _listener_observation(obs_key: chex.PRNGKey, aidx: int, state: State) -> jnp.ndarray:
            # The listeners need to see the newly generated images (which were generated from last-state's next_speaker_labels, i.e. speaker_labels) according to the channel map
            ch = state.channel_map
            speaker_index = ch[ch[:, 1].argsort()][:, 0][aidx]
            
            image = lax.cond(speaker_index < self.num_speakers,
                             lambda _: state.speaker_images[speaker_index],
                             lambda _: state.env_images[speaker_index-self.num_speakers],
                             operand=None)

            noise = jax.random.normal(obs_key, image.shape) * self.gaussian_noise_stddev
            image += noise
            
            return jnp.nan_to_num(jnp.clip(image, 0.0, 1.0))
        
        if as_dict:
            observations = {}
            if self.num_speakers != 0:
                speaker_obs = _speaker_observation(jnp.arange(self.num_speakers), state)
                observations = {agent: speaker_obs[i] for i, agent in enumerate(self.speaker_agents)}

            listener_obs_keys = jax.random.split(key, self.num_listeners)
            listener_obs = _listener_observation(listener_obs_keys, jnp.arange(self.num_listeners), state)
            observations.update({agent: listener_obs[i] for i, agent in enumerate(self.listener_agents)})
            return observations
        else:
            if self.num_speakers != 0:
                speaker_obs = _speaker_observation(jnp.arange(self.num_speakers), state)
            else:
                speaker_obs = None
            listener_obs_keys = jax.random.split(key, self.num_listeners)
            listener_obs = _listener_observation(listener_obs_keys, jnp.arange(self.num_listeners), state)
            return speaker_obs, listener_obs

    @partial(jax.jit, static_argnums=[0, 4])
    def step_env(self, key: chex.PRNGKey, state: State, actions, as_dict: bool = False):
        """Performs a step in the environment."""
        
        if as_dict or isinstance(actions, dict):
            speaker_actions = jnp.array([actions[agent] for agent in self.speaker_agents])
            listener_actions = jnp.array([actions[agent] for agent in self.listener_agents])
            listener_log_prob = jnp.ones_like(listener_actions)
        else:
            speaker_actions = actions[0]
            listener_actions = actions[1]
            listener_log_prob = actions[2]

        ######## First, evaluate the current state.

        @partial(jax.vmap, in_axes=[0, None, None, None])
        def _evaluate_channel_rewards(c: int, listener_actions: jnp.ndarray, listener_log_prob: jnp.ndarray, symmetric_rewards: jnp.ndarray) -> jnp.ndarray:
            channel = state.channel_map[c]
            speaker_index = channel[0]
            listener_index = channel[1]

            # Determine the correct label based on the speaker index
            is_speaker = speaker_index < self.num_speakers
            label = jnp.where(is_speaker, state.speaker_labels[speaker_index], state.env_labels[speaker_index-self.num_speakers])

            # Check if the listener's action matches the correct label and convert boolean to integer
            listener_correct = (listener_actions[listener_index] == label).astype(jnp.int32)
            listener_confidence = jnp.exp(listener_log_prob[listener_index])

            # Return reward based on whether the listener was correct. These are indexed by channel.
            speaker_channel_reward = jnp.where(listener_correct, self.speaker_reward_success, self.speaker_reward_failure)
            speaker_channel_reward *= (listener_confidence**2) ** self.log_prob_rewards   # Multiply by logprobs only if self.log_prob_rewards == True

            listener_channel_reward_symmetric = jnp.where(listener_correct, self.listener_reward_success, self.listener_reward_failure)

            # Calculating listener rewards is different in the asymmetric setting. The easiest way to do this is to re-calculate listener_correct
            # The right way to do it would be to zero-out the listener rewards in the calculated listener_channel_reward, but that would require some special calculation of indices.
            # Taking the quick are dirty route.

            label2 = jnp.where(is_speaker, -1, label)   # Forcing a wrong label. labels are never negative.
            listener_correct2 = (listener_actions[listener_index] == label2).astype(jnp.int32)
            listener_channel_reward_asymmetric = jnp.where(listener_correct2, self.listener_reward_success, self.listener_reward_failure)

            listener_channel_reward = jnp.where(symmetric_rewards, listener_channel_reward_symmetric, listener_channel_reward_asymmetric)

            return speaker_index, listener_index, speaker_channel_reward, listener_channel_reward

        speaker_indices, listener_indices, speaker_channel_rewards, listener_channel_rewards = _evaluate_channel_rewards(jnp.arange(self.num_channels), listener_actions, listener_log_prob, self.reward_parity_fn(state.epoch))

        # Generate a reward vector containing all the rewards for each speaker and listener
        speaker_rewards = jnp.zeros(self.num_speakers + self.num_channels)
        listener_rewards = jnp.zeros(self.num_listeners)
        initial_rewards_tuple = (speaker_rewards, listener_rewards, speaker_indices, listener_indices, speaker_channel_rewards, listener_channel_rewards)
        
        def update_rewards(loop_idx, rewards_tuple):
            speaker_rewards, listener_rewards, speaker_indices, listener_indices, speaker_channel_rewards, listener_channel_rewards = rewards_tuple
            # Update speaker and listener rewards
            speaker_reward = speaker_channel_rewards[loop_idx]
            listener_reward = listener_channel_rewards[loop_idx]
            new_speaker_rewards = speaker_rewards.at[speaker_indices[loop_idx]].add(speaker_reward)
            new_listener_rewards = listener_rewards.at[listener_indices[loop_idx]].add(listener_reward)
            return new_speaker_rewards, new_listener_rewards, speaker_indices, listener_indices, speaker_channel_rewards, listener_channel_rewards

        speaker_rewards_near_final, listener_rewards_final, _, _, _, _ = jax.lax.fori_loop(0, self.num_channels, update_rewards, initial_rewards_tuple)
        speaker_rewards_near_final = jax.lax.select(state.iteration == 0, jnp.zeros(self.num_speakers + self.num_channels), speaker_rewards_near_final)

        # Calculate speaker penalties and multiply them by their associated weights
        speaker_whitesum_penalty = speaker_penalty_whitesum_fn(state.speaker_images)
        speaker_curve_penalty = speaker_penalty_curve_fn(state.previous_speaker_actions)
        speaker_right_angle_penalty = speaker_penalty_right_angle_fn(state.previous_speaker_actions)
        speaker_right_angle_or_straight_penalty = speaker_penalty_right_angle_or_straight_fn(state.previous_speaker_actions)
        speaker_similar_curve_penalty = speaker_penalty_similar_curve_fn(state.previous_speaker_actions)
        speaker_too_close_to_border_penalty = speaker_penalty_too_close_to_border_fn(state.previous_speaker_actions)
        # speaker_spline_continuity_penalty = speaker_penalty_spline_continuity_fn(state.previous_speaker_actions)
        # speaker_zipfian_size_penalty = speaker_penalty_zipfian_size_fn(state.previous_speaker_actions)
        speaker_penalties = (
            speaker_whitesum_penalty * self.speaker_whitesum_penalty_coef
            + speaker_curve_penalty * self.speaker_curve_penalty_coef
            + speaker_right_angle_penalty * self.speaker_right_angle_penalty_coef
            + speaker_right_angle_or_straight_penalty * self.speaker_right_angle_or_straight_penalty_coef
            + speaker_similar_curve_penalty * self.speaker_similar_curve_penalty_coef
            + speaker_too_close_to_border_penalty * self.speaker_too_close_to_penalty_coef
            # + speaker_spline_continuity_penalty * self.speaker_spline_continuity_penalty_coef
            # + speaker_zipfian_size_penalty * self.speaker_zipfian_size_penalty_coef
        )

        # Apply the penalties
        speaker_penalties = jnp.pad(speaker_penalties, (0, self.num_channels), constant_values=0.0)
        speaker_rewards_final = jnp.where(speaker_rewards_near_final > 0,
                                          speaker_rewards_near_final + speaker_penalties,
                                          speaker_rewards_near_final)

        # rewards = {**{agent: speaker_rewards_final[i] for i, agent in enumerate(self.speaker_agents)}, **{agent: listener_rewards_final[i] for i, agent in enumerate(self.listener_agents)}}
        # rewards["__all__"] = sum(rewards.values())

        speaker_rewards = speaker_rewards_final[:self.num_speakers]
        listener_rewards = listener_rewards_final

        speaker_alives = jnp.isin(jnp.arange(self.num_speakers), state.channel_map[:, 0]).astype(jnp.int32)
        listener_alives = jnp.isin(jnp.arange(self.num_listeners), state.channel_map[:, 1]).astype(jnp.int32)

        # alives = {**{agent: speaker_alives[i] for i, agent in enumerate(self.speaker_agents)}, **{agent: listener_alives[i] for i, agent in enumerate(self.listener_agents)}}
        # alives = {**{agent: 1 if i in state.channel_map[:, 0] else 0 for i, agent in enumerate(self.speaker_agents)}, **{agent: 1 if i in state.channel_map[:, 1] else 0 for i, agent in enumerate(self.listener_agents)}}
        # alives["__all__"] = 0 # It's important that this is False at all times. Because the MARL library thinks this variable is actually "dones", and __all__==True would signify end of episode

        ######## Then, update the state.
        key, k1, k2, k3, k4, k5, k6, obs_key = jax.random.split(key, 8)
        
        next_env_images, next_env_labels = self.load_images(k5)

        next_speaker_labels = jax.random.randint(key, (self.num_speakers,), 0, self.speaker_referent_span_fn(state.epoch))
        
        # We can take the first num_channels * channel_ratio_fn(iteration) elements from the speakers, and the rest from the environment, and then shuffle them.
        requested_num_speaker_images = jax.lax.min(jnp.floor(self.num_channels * self.channel_ratio_fn(state.epoch)).astype(jnp.int32), self.num_channels)
        # requested_num_speaker_images should not be greater than self.num_speakers
        # assert requested_num_speaker_images <= self.num_speakers, f"requested_num_speaker_images ({requested_num_speaker_images}) cannot be greater than self.num_speakers ({self.num_speakers})"
        
        # Collect num_speakers speakers
        if self.speaker_assignment_method == "random":
            speaker_ids = jax.random.permutation(k2, self.num_speakers)
        elif self.speaker_assignment_method == "arange":
            speaker_ids = jnp.arange(self.num_speakers) # NOTE: I'm not sure this will work with more than one env in its current state.
        # NOTE: It appears that this does work, but I have not tested it rigorously
        
        speaker_ids = jnp.pad(speaker_ids, (0, max(self.num_channels-self.num_speakers, 0)), mode='wrap')
        speaker_ids = speaker_ids[:self.num_channels]

        # Collect num_env environment channels
        env_ids = jax.random.permutation(k3, self.num_channels) + self.num_speakers

        # Make a mask of size self.num_channels that is 1 if the index is less than requested_num_speaker_images, and 0 otherwise
        mask = jnp.where(jnp.arange(self.num_channels) < requested_num_speaker_images, 1, 0)
        possible_speakers = jnp.where(mask, speaker_ids, env_ids)
        speakers = jax.random.permutation(k4, possible_speakers).reshape((-1, 1))

        listeners = jax.lax.slice(jax.random.permutation(k1, self.num_listeners).reshape((-1, 1)), [0, 0], [self.num_channels, 1])
        next_channel_map = jnp.hstack((speakers, listeners))

        speaker_images_for_new_state = self.speaker_action_transform(speaker_actions)
        if self.center_and_reshuffle_listener_obs:
            # Get rid of background, then center
            speaker_images_for_new_state -= 0.3
            speaker_images_for_new_state *= -1
            speaker_images_for_new_state = center_obs(speaker_images_for_new_state)
            
            # Now randomly translate the images and add the background back
            speaker_images_for_new_state = shift_obs(speaker_images_for_new_state, jax.random.split(k6, len(speaker_images_for_new_state)))
            speaker_images_for_new_state *= -1
            speaker_images_for_new_state = jnp.clip(speaker_images_for_new_state + 0.3, 0.0, 1.0)
            
        # Calculate listener_obs_source based on state.next_channel_map. It should be the size of the number of speakers and be 0 if from env, 1 if from speaker. Based on channel ratio fn.
        listener_obs_values = jnp.where(state.next_channel_map[:, 0] < self.num_speakers, 1, 0)
        listener_obs_indices = state.next_channel_map[:, 1]
        listener_obs_source = jnp.zeros((self.num_listeners)).at[listener_obs_indices].set(listener_obs_values).squeeze()

        state = State(
            next_channel_map=next_channel_map,
            next_env_images=next_env_images,
            next_env_labels=next_env_labels,
            next_speaker_labels=next_speaker_labels,

            channel_map=state.next_channel_map,
            env_images=state.next_env_images,
            env_labels=state.next_env_labels,
            speaker_labels=state.next_speaker_labels,
            listener_obs_source=listener_obs_source,

            speaker_images=speaker_images_for_new_state,

            previous_channel_map=state.channel_map,
            previous_env_images=state.env_images,
            previous_env_labels=state.env_labels,
            previous_speaker_labels=state.speaker_labels,

            previous_speaker_images=state.speaker_images,
            previous_speaker_actions=speaker_actions,

            iteration=state.iteration + 1,
            epoch=state.epoch,
            requested_num_speaker_images=requested_num_speaker_images,   # For next state
            requested_speaker_referent_span=self.speaker_referent_span_fn(state.epoch),
            agent_inferential_mode=self.agent_inferential_mode_fn(state.epoch)
        )
        
        return lax.stop_gradient(self.get_obs(obs_key, state, as_dict)), lax.stop_gradient(state), (lax.stop_gradient(speaker_rewards), lax.stop_gradient(listener_rewards)), (lax.stop_gradient(speaker_alives), lax.stop_gradient(listener_alives)), {}
    
    @partial(jax.jit, static_argnums=[0, 3])
    def reset(self, key: chex.PRNGKey, epoch: int = 0, as_dict: bool = False) -> Tuple[Dict, State]:
        """Reset the environment"""
        key, k1, k2, k3, k4, k5, k6, obs_key = jax.random.split(key, 8)
        
        next_env_images, next_env_labels = self.load_images(k5)

        next_speaker_labels = jax.random.randint(key, (self.num_speakers,), 0, self.speaker_referent_span_fn(epoch))
        
        # We can take the first num_channels * channel_ratio_fn(iteration) elements from the speakers, and the rest from the environment, and then shuffle them.
        requested_num_speaker_images = jnp.floor(self.num_channels * self.channel_ratio_fn(epoch)).astype(jnp.int32)
        # requested_num_speaker_images should not be greater than self.num_speakers
        # assert requested_num_speaker_images <= self.num_speakers, f"requested_num_speaker_images ({requested_num_speaker_images}) cannot be greater than self.num_speakers ({self.num_speakers})"
        
        # Collect num_speakers speakers
        if self.speaker_assignment_method == "random":
            speaker_ids = jax.random.permutation(k2, self.num_speakers)
        elif self.speaker_assignment_method == "arange":
            speaker_ids = jnp.arange(self.num_speakers)
        
        speaker_ids = jnp.pad(speaker_ids, (0, max(self.num_channels-self.num_speakers, 0)), mode='wrap')
        speaker_ids = speaker_ids[:self.num_channels]
        
        # Collect num_env environment channels
        env_ids = jax.random.permutation(k3, self.num_channels) + self.num_speakers

        # Make a mask of size self.num_channels that is 1 if the index is less than requested_num_speaker_images, and 0 otherwise
        mask = jnp.where(jnp.arange(self.num_channels) < requested_num_speaker_images, 1, 0)
        possible_speakers = jnp.where(mask, speaker_ids, env_ids)
        speakers = jax.random.permutation(k4, possible_speakers).reshape((-1, 1))

        listeners = jax.lax.slice(jax.random.permutation(k1, self.num_listeners).reshape((-1, 1)), [0, 0], [self.num_channels, 1])
        next_channel_map = jnp.hstack((speakers, listeners))

        # Calculate listener_obs_source based on next_channel_map. It should be the size of the number of speakers and be 0 if from env, 1 if from speaker. Based on channel ratio fn.
        listener_obs_values = jnp.where(next_channel_map[:, 0] < self.num_speakers, 1, 0)
        listener_obs_indices = next_channel_map[:, 1]
        listener_obs_source = jnp.zeros((self.num_listeners)).at[listener_obs_indices].set(listener_obs_values).squeeze()
        
        state = State(
            next_channel_map=next_channel_map,
            next_env_images=next_env_images,
            next_env_labels=next_env_labels,
            next_speaker_labels=next_speaker_labels,

            channel_map=next_channel_map,
            env_images=jnp.zeros_like(next_env_images),
            env_labels=jnp.zeros_like(next_env_labels),
            speaker_labels=jnp.zeros_like(next_speaker_labels),
            listener_obs_source=listener_obs_source,

            speaker_images=jnp.zeros((max(self.num_speakers, 1), self.image_dim, self.image_dim), dtype=jnp.float32),  # This max is to avoid an error when num_speakers is 0 from the get_obs function

            previous_channel_map=jnp.zeros_like(next_channel_map),
            previous_env_images=jnp.zeros_like(next_env_images),
            previous_env_labels=jnp.zeros_like(next_env_labels),
            previous_speaker_labels=jnp.zeros_like(next_speaker_labels),

            previous_speaker_images=jnp.zeros((max(self.num_speakers, 1), self.image_dim, self.image_dim), dtype=jnp.float32),  # This max is to avoid an error when num_speakers is 0 from the get_obs function
            previous_speaker_actions=jnp.zeros((self.num_speakers, self.speaker_action_dim)),

            iteration=0,
            epoch=epoch,
            requested_num_speaker_images=requested_num_speaker_images,   # For next state
            requested_speaker_referent_span=self.speaker_referent_span_fn(epoch),
            agent_inferential_mode=self.agent_inferential_mode_fn(epoch)
        )

        return lax.stop_gradient(self.get_obs(obs_key, state, as_dict)), lax.stop_gradient(state)
    
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
    
    def render_mnist(self, state: State, actions: dict = None) -> None:
        """DEPRECATED Renders the environment (mnist)."""

        import matplotlib as mpl
        from matplotlib.backends.backend_agg import (
            FigureCanvasAgg as FigureCanvas,
        )
        from matplotlib.figure import Figure
        from PIL import Image

        fig = Figure((8, 4))
        canvas = FigureCanvas(fig)
        mpl.rcParams["xtick.labelbottom"] = False
        mpl.rcParams["ytick.labelleft"] = False

        num_channels = state.channel_map.shape[0]

        @partial(jax.vmap, in_axes=[0, None])
        def get_image_label(channel_index: int, state: State):
            channel = state.previous_channel_map[channel_index]
            speaker_index = channel[0]
            listener_index = channel[1]

            
            image = lax.cond(speaker_index < self.num_speakers,
                             lambda _: state.speaker_images[speaker_index],
                             lambda _: state.previous_env_images[speaker_index-self.num_speakers],
                             operand=None)
                             
            label = lax.cond(speaker_index < self.num_speakers,
                             lambda _: state.previous_speaker_labels[speaker_index],
                             lambda _: state.previous_env_labels[speaker_index-self.num_speakers],
                             operand=None)

            label_is_from_speaker = speaker_index < self.num_speakers
                             
            return image, label, label_is_from_speaker
        
        images, labels, labels_are_from_speaker = get_image_label(jnp.arange(num_channels), state)
 
        fig.set_dpi(300)  
        fig.suptitle("Signification Game Round\nTop Label: Source, Destination (Listener, Speaker, or Environment) \n Bottom Label: Previous Channel Mapping", fontsize=10)
        for i in range(num_channels):
            if i < num_channels // 2:
                row = 0
            else:
                row = 1
    
            ax = fig.add_subplot(2, num_channels // 2, (i % (num_channels // 2)) + 1 + (row * (num_channels // 2)))
            ax.imshow(
                images[i],
                cmap="Greys",
                vmin=0,
                vmax=255,
                aspect="equal",
                interpolation="none"
            )
            ax.set_aspect("equal")
            ax.margins(0.05)  
            ax.annotate(
                ('S' if labels_are_from_speaker[i] else 'E') + str(labels[i]) + ('' if actions is None else ' L' + str(actions[f"listener_{i}"])),
                fontsize=12,
                color="black",
                xy=(0.5, 0), 
                xycoords="axes fraction",
                xytext=(0, -15), 
                textcoords="offset points",
                ha='center'  
            )
            ax.annotate(
                str(state.previous_channel_map[i]),
                fontsize=10,
                color="black",
                xy=(0.5, 0),  
                xycoords="axes fraction",
                xytext=(0, -30),  
                textcoords="offset points",
                ha='center'  
            )
            ax.set_xticks([])
            ax.set_yticks([])
        
        canvas.draw()
        image = Image.frombytes(
            "RGB",
            fig.canvas.get_width_height(),
            fig.canvas.tostring_rgb(),
        )
        return image


def test_mnist_signification_game():
    """Runs a simplified signification game on MNIST."""
    def ret_0(iteration):
        return 0.5
    
    # Define parameters for a signification game
    num_speakers = 2
    num_listeners = 4
    num_channels = 4
    num_classes = 10

    mnist_dataset = MNIST('/tmp/mnist/', download=True)
    images, labels = to_jax(mnist_dataset, num_datapoints=100)

    key = jax.random.PRNGKey(7)
    key, key_reset, key_act, key_step = jax.random.split(key, 4)
    
    env = SimplifiedSignificationGame(num_speakers, num_listeners, num_channels, num_classes, channel_ratio_fn="all_speakers", dataset=(images, labels), image_dim=28, speaker_action_transform='image', speaker_action_penalty='whitesum')
    obs, state = env.reset(key_reset, epoch=10, as_dict=True)
    
    print(list(obs.keys()))
    print(obs)
    print(state)

    action_keys = jax.random.split(key_act, len(env.agents))
    actions = {agent: env.action_space(agent).sample(action_keys[i]) for i, agent in enumerate(env.agents)}

    for agent, agent_action in actions.items():
        if agent.startswith("speaker"):
            continue
        print(f"Action for {agent}: {agent_action}")

    key, key_step = jax.random.split(key, 2)
    obs, state, reward, done, infos = env.step(key_step, state, actions)

    key, key_step = jax.random.split(key, 2)
    obs, state, reward, done, infos = env.step(key_step, state, actions)

    key, key_step = jax.random.split(key, 2)
    obs, state, reward, done, infos = env.step(key_step, state, actions)

    print("channel_map:")
    print(state.previous_channel_map)
    print("speaker_labels:")
    print(state.previous_speaker_labels)
    print("env_labels:")
    print(state.previous_env_labels)

    for agent, agent_reward in reward.items():
        print(f"Reward for {agent}: {agent_reward}")

    img = env.render_mnist(state, actions)

    # Show image
    img.show()

    # for channel in state.previous_channel_map:
    #     print(f"Channel {channel}: speaker spoke {state.previous_speaker_labels[channel[0]]} and listener heard {actions[f'listener_{channel[1]}']}")


if __name__ == '__main__':
    test_mnist_signification_game()
    # You can run this file with python -W 'ignore' improved_signification_game.py to ignore the warnings
