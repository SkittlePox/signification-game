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
import math

from utils import to_jax


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
    epoch: int
    requested_num_speaker_images: int


class SimplifiedSignificationGame(MultiAgentEnv):
    def __init__(self, num_speakers: int, num_listeners: int, num_channels: int, num_classes: int, channel_ratio_fn: Union[Callable, str], speaker_action_transform: Union[Callable, str], dataset: tuple, image_dim: int, reward_success: float = 1.0, reward_failure: float = -0.1, **kwargs: dict) -> None:
        super().__init__(num_agents=num_speakers + num_listeners)
        self.num_speakers = num_speakers
        self.num_listeners = num_listeners
        self.num_channels = num_channels    # We expect num_listeners to be equal to num_channels
        self.num_classes = num_classes
        self.stored_env_images = dataset[0]
        self.stored_env_labels = dataset[1]
        self.image_dim = image_dim
        self.reward_success = reward_success
        self.reward_failure = reward_failure
        self.kwargs = kwargs
        # TODO: Move the above comments to an actual docstring

        if isinstance(channel_ratio_fn, str):
            def ret_0(iteration):
                return 0.0
            
            def ret_1(iteration):
                return 1.0
            
            def s_curve(x):
                return 1.0 / (1.0 + jnp.exp(-1 * (jnp.array(x, float) - 200))) + 0.01
            
            def linear(x):
                return x / 400.0
            
            def fifth(x):
                return 1.0 / 5.0
            
            def half(x):
                return 0.5
            
            def fifth_at_500(x):
                return jax.lax.cond(x < 500, lambda _: 0.0, lambda _: 0.2, None)
            
            def tenth_at_500(x):
                return jax.lax.cond(x < 500, lambda _: 0.0, lambda _: 0.1, None)
            
            def fifth_at_1k(x):
                return jax.lax.cond(x < 1000, lambda _: 0.0, lambda _: 0.2, None)

            if channel_ratio_fn in ("all_env", "ret_0", "ret0"):
                self.channel_ratio_fn = ret_0
            elif channel_ratio_fn in ("all_speakers", "all_speaker", "ret_1", "ret1"):
                self.channel_ratio_fn = ret_1
            elif channel_ratio_fn == "sigmoid1":
                self.channel_ratio_fn = s_curve
            elif channel_ratio_fn == "linear":
                self.channel_ratio_fn = linear
            elif channel_ratio_fn == "fifth":
                self.channel_ratio_fn = fifth
            elif channel_ratio_fn == "half":
                self.channel_ratio_fn = half
            elif channel_ratio_fn == "fifth_at_500":
                self.channel_ratio_fn = fifth_at_500
            elif channel_ratio_fn == "tenth_at_500":
                self.channel_ratio_fn = tenth_at_500
            elif channel_ratio_fn == "fifth_at_1k":
                self.channel_ratio_fn = fifth_at_1k
        else:
            self.channel_ratio_fn = channel_ratio_fn    # This function returns the ratio of the communication channels from the environment vs from the speakers. With 0 being all from the environment and 1 being all from the speakers.

        if isinstance(speaker_action_transform, str):
            @jax.vmap
            def identity(actions: jnp.array):
                return actions
            
            @jax.vmap
            def image(actions: jnp.array):
                return actions.reshape(-1, image_dim, image_dim)
            
            @jax.vmap
            def gauss_splat(actions: jnp.array):
                def paint_normalized_gaussians_on_array(array_shape, gaussians_params):
                    """
                    Paint multiple 2D Gaussians with normalized parameters on a 2D array, using a NumPy array for parameters.
                    
                    Parameters:
                    - array_shape: tuple of int, shape of the 2D array (height, width).
                    - gaussians_params: NumPy array with each row representing the parameters for a Gaussian 
                    (normalized mean x, normalized mean y, normalized variance x, normalized variance y, amplitude).
                    
                    Returns:
                    - 2D numpy array with the Gaussians painted on it.
                    """
                    y, x = jnp.indices(array_shape)  # Create a grid of x and y coordinates
                    array = jnp.zeros(array_shape)

                    @jax.vmap
                    def compute_gaussian(params):
                        x_mu_norm, y_mu_norm, sigma_x2_norm, sigma_y2_norm, amplitude = params
                        
                        # Convert normalized mean to actual coordinates
                        x_mu = x_mu_norm * array_shape[1]
                        y_mu = y_mu_norm * array_shape[0]

                        # Convert normalized covariance to actual variances
                        sigma_x2 = sigma_x2_norm * array_shape[1]**2
                        sigma_y2 = sigma_y2_norm * array_shape[0]**2

                        # Compute the 2D Gaussian formula
                        gaussian = amplitude * jnp.exp(-(((x - x_mu)**2 / (2 * sigma_x2)) + ((y - y_mu)**2 / (2 * sigma_y2))))
                        return gaussian

                    array += compute_gaussian(gaussians_params)
                    array = jnp.sum(array, axis=0)

                    return jnp.clip(array, a_min=0.0, a_max=1.0)

                gaussians_params = actions.reshape(-1, 5)
                array_shape = (image_dim, image_dim)

                gaussian_array = paint_normalized_gaussians_on_array(array_shape, gaussians_params)
                return gaussian_array
            
            @jax.vmap
            def gauss_splat_covar(actions: jnp.array):
                def paint_normalized_gaussians_on_array(array_shape, gaussians_params):
                    """
                    Paint multiple 2D Gaussians with full covariance matrices on a 2D array.
                    
                    Parameters:
                    - array_shape: tuple of int, shape of the 2D array (height, width).
                    - gaussians_params: JAX array with each row representing the parameters for a Gaussian 
                    (normalized mean x, normalized mean y, normalized variance x, normalized variance y, 
                    normalized covariance xy, amplitude).
                    
                    Returns:
                    - 2D JAX array with the Gaussians painted on it.
                    """
                    y, x = jnp.indices(array_shape)  # Create a grid of x and y coordinates
                    array = jnp.zeros(array_shape)

                    @jax.vmap
                    def compute_gaussian(params):
                        x_mu_norm, y_mu_norm, amplitude, sigma_x2_norm, sigma_y2_norm, sigma_xy_norm = params
                        
                        # Convert normalized mean to actual coordinates
                        x_mu = x_mu_norm * array_shape[1]
                        y_mu = y_mu_norm * array_shape[0]

                        # Convert normalized covariance to actual values
                        sigma_x2 = sigma_x2_norm * array_shape[1]**2 * 0.01
                        sigma_y2 = sigma_y2_norm * array_shape[0]**2 * 0.01
                        sigma_xy = (2 * sigma_xy_norm - 1) * array_shape[1] * array_shape[0] * 0.002

                        # Construct the covariance matrix and its inverse
                        cov_matrix = jnp.array([[sigma_x2, sigma_xy], [sigma_xy, sigma_y2]])
                        inv_cov_matrix = jnp.linalg.inv(cov_matrix)

                        # Compute the Gaussian function
                        X = jnp.vstack((x.ravel() - x_mu, y.ravel() - y_mu))
                        gaussian = amplitude * jnp.exp(-0.5 * jnp.sum(X.T @ inv_cov_matrix * X.T, axis=1)).reshape(array_shape)
                        
                        return gaussian

                    gaussians = compute_gaussian(gaussians_params)
                    array += jnp.sum(gaussians, axis=0)  # Sum contributions from all Gaussians

                    return jnp.clip(array, a_min=0.0, a_max=1.0)

                # Assuming 'actions' includes the additional covariance parameters
                gaussians_params = actions.reshape(-1, 6)  # Reshape based on the new parameter structure
                image_dim = 28  # As per your indication, assuming a fixed image dimension
                array_shape = (image_dim, image_dim)

                gaussian_array = paint_normalized_gaussians_on_array(array_shape, gaussians_params)
                return gaussian_array

            @jax.vmap
            def gauss_splat_chol(actions: jnp.array):
                def paint_normalized_gaussians_on_array(array_shape, gaussians_params):
                    """
                    Paint multiple 2D Gaussians on a 2D array, parameters defined via Cholesky decomposition.
                    
                    Parameters:
                    - array_shape: tuple of int, shape of the 2D array (height, width).
                    - gaussians_params: JAX array with each row representing the parameters for a Gaussian 
                    (normalized mean x, normalized mean y, amplitude, L_{11}, L_{21}, L_{22}).
                    
                    Returns:
                    - 2D JAX array with the Gaussians painted on it.
                    """
                    y, x = jnp.indices(array_shape)  # Create a grid of x and y coordinates
                    array = jnp.zeros(array_shape)

                    @jax.vmap
                    def compute_gaussian(params):
                        x_mu_norm, y_mu_norm, amplitude, L_11, L_21, L_22 = params

                        L_11 *= 0.1
                        L_22 *= 0.1
                        L_21 = ((2 * L_21) - 1) * 0.01
                        
                        # Convert normalized mean to actual coordinates
                        x_mu = x_mu_norm * array_shape[1]
                        y_mu = y_mu_norm * array_shape[0]

                        # Construct the covariance matrix from Cholesky decomposition
                        L = jnp.array([[L_11, 0], [L_21, L_22]])
                        cov_matrix = L @ L.T

                        # Convert normalized covariance matrix to actual values
                        cov_matrix = cov_matrix * jnp.array([[array_shape[1]**2, array_shape[1]*array_shape[0]], 
                                                            [array_shape[1]*array_shape[0], array_shape[0]**2]])
                        inv_cov_matrix = jnp.linalg.inv(cov_matrix)

                        # Compute the Gaussian function
                        X = jnp.vstack((x.ravel() - x_mu, y.ravel() - y_mu))
                        gaussian = amplitude * jnp.exp(-0.5 * jnp.sum(X.T @ inv_cov_matrix * X.T, axis=1)).reshape(array_shape)
                        
                        return gaussian

                    gaussians = compute_gaussian(gaussians_params)
                    array += jnp.sum(gaussians, axis=0)  # Sum contributions from all Gaussians

                    return jnp.clip(array, a_min=0.0, a_max=1.0)

                # Assuming 'actions' includes the Cholesky decomposition parameters
                gaussians_params = actions.reshape(-1, 6)  # Reshape based on the new parameter structure
                image_dim = 28  # Fixed image dimension
                array_shape = (image_dim, image_dim)

                gaussian_array = paint_normalized_gaussians_on_array(array_shape, gaussians_params)
                return gaussian_array

            if speaker_action_transform == "identity":
                self.speaker_action_transform = identity
            elif speaker_action_transform == "image":
                self.speaker_action_transform = image
            elif speaker_action_transform == "gausssplat":
                self.speaker_action_transform = gauss_splat
            elif speaker_action_transform == "gausssplatcovar":
                self.speaker_action_transform = gauss_splat_covar
            elif speaker_action_transform == "gausssplatchol":
                self.speaker_action_transform = gauss_splat_chol

        self.speaker_agents = ["speaker_{}".format(i) for i in range(num_speakers)]
        self.listener_agents = ["listener_{}".format(i) for i in range(num_listeners)]
        self.agents = self.speaker_agents + self.listener_agents

        self.observation_spaces = {**{agent: Discrete(num_classes) for agent in self.speaker_agents}, **{agent: Box(low=0, high=255, shape=(28, 28), dtype=jnp.float32) for agent in self.listener_agents}}
        self.action_spaces = {**{agent: Box(low=0, high=255, shape=(28, 28), dtype=jnp.float32) for agent in self.speaker_agents}, **{agent: Discrete(num_classes) for agent in self.listener_agents}}  # TODO migrate: This may need to change, unsure. Sampling randomly from this may fail.

    @partial(jax.jit, static_argnums=(0,))
    def load_images(self, key: chex.PRNGKey, num_imgs: int = -1):
        """Returns a random set of images and labels."""
        num_imgs = self.num_channels if num_imgs == -1 else num_imgs

        key, subkey = jax.random.split(key)
        indices = jax.random.randint(subkey, shape=(num_imgs,), minval=0, maxval=len(self.stored_env_images))
        
        images = self.stored_env_images[indices]
        labels = self.stored_env_labels[indices]
        return images, labels
    
    
    @partial(jax.jit, static_argnums=[0, 2])
    def get_obs(self, state: State, as_dict: bool = False):
        """Returns the observation for each agent."""
        
        @partial(jax.vmap, in_axes=[0, None])
        def _speaker_observation(aidx: int, state: State) -> jnp.ndarray:
            # The speakers need to see the newly generated assigned classes
            return state.next_speaker_labels[aidx]

        @partial(jax.vmap, in_axes=[0, None])
        def _listener_observation(aidx: int, state: State) -> jnp.ndarray:
            # The listeners need to see the newly generated images (which were generated from last-state's next_speaker_labels, i.e. speaker_labels) according to the channel map
            ch = state.channel_map
            speaker_index = ch[ch[:, 1].argsort()][:, 0][aidx]
            
            image = lax.cond(speaker_index < self.num_speakers,
                             lambda _: state.speaker_images[speaker_index],
                             lambda _: state.env_images[speaker_index-self.num_speakers],
                             operand=None)
            
            return image
        
        if as_dict:
            observations = {}
            if self.num_speakers != 0:
                speaker_obs = _speaker_observation(jnp.arange(self.num_speakers), state)
                observations = {agent: speaker_obs[i] for i, agent in enumerate(self.speaker_agents)}

            listener_obs = _listener_observation(jnp.arange(self.num_listeners), state)
            observations.update({agent: listener_obs[i] for i, agent in enumerate(self.listener_agents)})
            return observations
        else:
            if self.num_speakers != 0:
                speaker_obs = _speaker_observation(jnp.arange(self.num_speakers), state)
            else:
                speaker_obs = None
            listener_obs = _listener_observation(jnp.arange(self.num_listeners), state)
            return speaker_obs, listener_obs

    @partial(jax.jit, static_argnums=[0, 4])
    def step_env(self, key: chex.PRNGKey, state: State, actions, as_dict: bool = False):
        """Performs a step in the environment."""
        
        if isinstance(actions, dict):
            speaker_actions = jnp.array([actions[agent] for agent in self.speaker_agents])
            listener_actions = jnp.array([actions[agent] for agent in self.listener_agents])
        else:
            speaker_actions = actions[0]
            listener_actions = actions[1]

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
            listener_correct = (listener_actions[listener_index] == label).astype(jnp.int32)

            # Return reward based on whether the listener was correct
            reward = jnp.where(listener_correct, self.reward_success, self.reward_failure)

            return speaker_index, listener_index, reward

        speaker_indices, listener_indices, rewards = _evaluate_channel_rewards(jnp.arange(self.num_channels), listener_actions)

        # Generate a reward vector containing all the rewards for each speaker and listener
        speaker_rewards = jnp.zeros(self.num_speakers + self.num_channels)
        listener_rewards = jnp.zeros(self.num_listeners)
        initial_rewards_tuple = (speaker_rewards, listener_rewards, speaker_indices, listener_indices, rewards)
        
        def update_rewards(loop_idx, rewards_tuple):
            speaker_rewards, listener_rewards, speaker_indices, listener_indices, rewards = rewards_tuple
            # Update speaker and listener rewards
            reward = rewards[loop_idx]
            new_speaker_rewards = speaker_rewards.at[speaker_indices[loop_idx]].add(reward)
            new_listener_rewards = listener_rewards.at[listener_indices[loop_idx]].add(reward)
            return new_speaker_rewards, new_listener_rewards, speaker_indices, listener_indices, rewards

        speaker_rewards_final, listener_rewards_final, _, _, _ = jax.lax.fori_loop(0, self.num_channels, update_rewards, initial_rewards_tuple)
        speaker_rewards_final = jax.lax.select(state.iteration == 0, jnp.zeros(self.num_speakers + self.num_channels), speaker_rewards_final)
        # listener_rewards_final = jax.lax.select(state.iteration == 0, jnp.zeros(self.num_listeners), listener_rewards_final)

        rewards = {**{agent: speaker_rewards_final[i] for i, agent in enumerate(self.speaker_agents)}, **{agent: listener_rewards_final[i] for i, agent in enumerate(self.listener_agents)}}
        rewards["__all__"] = sum(rewards.values())


        speaker_alives = jnp.isin(jnp.arange(self.num_speakers), state.channel_map[:, 0]).astype(jnp.int32)
        listener_alives = jnp.isin(jnp.arange(self.num_listeners), state.channel_map[:, 1]).astype(jnp.int32)

        alives = {**{agent: speaker_alives[i] for i, agent in enumerate(self.speaker_agents)}, **{agent: listener_alives[i] for i, agent in enumerate(self.listener_agents)}}
        # alives = {**{agent: 1 if i in state.channel_map[:, 0] else 0 for i, agent in enumerate(self.speaker_agents)}, **{agent: 1 if i in state.channel_map[:, 1] else 0 for i, agent in enumerate(self.listener_agents)}}
        alives["__all__"] = 0 # It's important that this is False. Because the MARL library thinks this variable is actually "dones", and __all__ True would signify end of episode

        ######## Then, update the state.
        key, k1, k2, k3, k4, k5 = jax.random.split(key, 6)
        
        next_env_images, next_env_labels = self.load_images(k5)

        next_speaker_labels = jax.random.randint(key, (self.num_speakers,), 0, self.num_classes)
        
        # We can take the first num_channels * channel_ratio_fn(iteration) elements from the speakers, and the rest from the environment, and then shuffle them.
        requested_num_speaker_images = jnp.floor(self.num_channels * self.channel_ratio_fn(state.epoch)).astype(jnp.int32)
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

            speaker_images=self.speaker_action_transform(speaker_actions),

            previous_channel_map=state.channel_map,
            previous_env_images=state.env_images,
            previous_env_labels=state.env_labels,
            previous_speaker_labels=state.speaker_labels,

            previous_speaker_images=state.speaker_images,

            iteration=state.iteration + 1,
            epoch=state.epoch,
            requested_num_speaker_images=requested_num_speaker_images   # For next state
        )
        
        return lax.stop_gradient(self.get_obs(state, as_dict)), lax.stop_gradient(state), rewards, alives, {}
    
    @partial(jax.jit, static_argnums=[0, 3])
    def reset(self, key: chex.PRNGKey, epoch: int = 0, as_dict: bool = False) -> Tuple[Dict, State]:
        """Reset the environment"""
        key, k1, k2, k3, k4, k5 = jax.random.split(key, 6)
        
        next_env_images, next_env_labels = self.load_images(k5)

        next_speaker_labels = jax.random.randint(key, (self.num_speakers,), 0, self.num_classes)
        
        # We can take the first num_channels * channel_ratio_fn(iteration) elements from the speakers, and the rest from the environment, and then shuffle them.
        requested_num_speaker_images = jnp.floor(self.num_channels * self.channel_ratio_fn(epoch)).astype(jnp.int32)
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

            channel_map=next_channel_map,   # If we don't do this we get strange rewards
            env_images=jnp.zeros_like(next_env_images),
            env_labels=jnp.zeros_like(next_env_labels),
            speaker_labels=jnp.zeros_like(next_speaker_labels),

            speaker_images=jnp.zeros((max(self.num_speakers, 1), self.image_dim, self.image_dim), dtype=jnp.float32),  # This max is to avoid an error when num_speakers is 0 from the get_obs function

            previous_channel_map=jnp.zeros_like(next_channel_map),
            previous_env_images=jnp.zeros_like(next_env_images),
            previous_env_labels=jnp.zeros_like(next_env_labels),
            previous_speaker_labels=jnp.zeros_like(next_speaker_labels),

            previous_speaker_images=jnp.zeros((max(self.num_speakers, 1), self.image_dim, self.image_dim), dtype=jnp.float32),  # This max is to avoid an error when num_speakers is 0 from the get_obs function

            iteration=0,
            epoch=epoch,
            requested_num_speaker_images=requested_num_speaker_images   # For next state
        )

        return self.get_obs(state, as_dict), state
    
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
        """Renders the environment (mnist)."""

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
    num_speakers = 10
    num_listeners = 10
    num_channels = 10
    num_classes = 10

    mnist_dataset = MNIST('/tmp/mnist/', download=True)
    images, labels = to_jax(mnist_dataset, num_datapoints=100)

    key = jax.random.PRNGKey(0)
    key, key_reset, key_act, key_step = jax.random.split(key, 4)
    
    env = SimplifiedSignificationGame(num_speakers, num_listeners, num_channels, num_classes, channel_ratio_fn=ret_0, dataset=(images, labels), image_dim=28)
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
