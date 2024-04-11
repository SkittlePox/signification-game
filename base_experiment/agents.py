from functools import partial
import chex
import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
from flax.linen.initializers import constant, orthogonal
import distrax
from jaxmarl.wrappers.baselines import LogWrapper, LogEnvState
from typing import Sequence, Tuple, Dict
from simplified_signification_game import SimplifiedSignificationGame, State

class SimpSigGameLogWrapper(LogWrapper):
    @partial(jax.jit, static_argnums=(0,))
    def reset(self, keys: chex.PRNGKey, epochs: Sequence[int]) -> Tuple[chex.Array, State]:
        obs, env_state = jax.vmap(self._env.reset)(keys, epochs)
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
        obs, env_state, reward, done, info = jax.vmap(self._env.step_env)(
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
class ActorCriticListenerConv(nn.Module):
    action_dim: Sequence[int]
    image_dim: Sequence[int]
    config: Dict

    @nn.compact
    def __call__(self, x):
        x = x.reshape(-1, self.image_dim, self.image_dim, 1)  # Assuming x is flat, and image_dim is [height, width]

        # Convolutional layers
        x = nn.Conv(features=32, kernel_size=(3, 3), strides=(1, 1), padding='SAME')(x)
        x = nn.sigmoid(x)
        x = nn.Conv(features=64, kernel_size=(3, 3), strides=(1, 1), padding='SAME')(x)
        x = nn.sigmoid(x)
        x = x.reshape((x.shape[0], -1))  # Flatten
        
        # Embedding Layer
        embedding = nn.Dense(512, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        embedding = nn.sigmoid(embedding)
        embedding = nn.Dense(512, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(embedding)
        embedding = nn.sigmoid(embedding)

        # Actor Layer
        actor_mean = nn.Dense(512, kernel_init=orthogonal(2), bias_init=constant(0.0))(embedding)
        actor_mean = nn.sigmoid(actor_mean)
        actor_mean = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(actor_mean)

        # Action Logits
        # unavail_actions = 1 - avail_actions
        # action_logits = actor_mean - (unavail_actions * 1e10)
        pi = distrax.Categorical(logits=actor_mean)

        # Critic Layer
        critic = nn.Dense(512, kernel_init=orthogonal(2), bias_init=constant(0.0))(embedding)
        critic = nn.sigmoid(critic)
        critic = nn.Dense(512, kernel_init=orthogonal(2), bias_init=constant(0.0))(critic)
        critic = nn.sigmoid(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(critic)

        return pi, jnp.squeeze(critic, axis=-1)
    

class ActorCriticListenerConvSmall(nn.Module):
    action_dim: Sequence[int]
    image_dim: Sequence[int]
    config: Dict

    @nn.compact
    def __call__(self, x):
        x = x.reshape(-1, self.image_dim, self.image_dim, 1)  # Assuming x is flat, and image_dim is [height, width]

        # Convolutional layers
        x = nn.Conv(features=32, kernel_size=(3, 3), strides=(1, 1), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.Conv(features=64, kernel_size=(3, 3), strides=(1, 1), padding='SAME')(x)
        x = nn.relu(x)
        x = x.reshape((x.shape[0], -1))  # Flatten
        
        # Embedding Layer
        embedding = nn.Dense(128)(x)
        embedding = nn.relu(embedding)
        embedding = nn.Dropout(rate=self.config["LISTENER_DROPOUT"], deterministic=False)(embedding)
        embedding = nn.Dense(128)(embedding)
        embedding = nn.relu(embedding)
        embedding = nn.Dropout(rate=self.config["LISTENER_DROPOUT"], deterministic=False)(embedding)
        embedding = nn.Dense(128)(embedding)
        embedding = nn.relu(embedding)

        # Actor Layer
        actor_mean = nn.Dense(128)(embedding)
        actor_mean = nn.relu(actor_mean)
        actor_mean = nn.Dense(self.action_dim)(actor_mean)
        actor_mean = nn.softmax(actor_mean)
        pi = distrax.Categorical(probs=actor_mean)

        # Critic Layer
        critic = nn.Dense(128)(embedding)
        critic = nn.relu(critic)
        # critic = nn.Dropout(rate=self.config["LISTENER_DROPOUT"], deterministic=False)(critic)
        critic = nn.Dense(128)(critic)
        critic = nn.relu(critic)
        critic = nn.Dense(1)(critic)

        return pi, jnp.squeeze(critic, axis=-1)


class ActorCriticListenerDense(nn.Module):
    action_dim: Sequence[int]
    image_dim: Sequence[int]
    config: Dict

    @nn.compact
    def __call__(self, x):
        obs = x
        # Embedding Layer
        embedding = nn.Dense(512)(obs)
        embedding = nn.sigmoid(embedding)
        # embedding = nn.Dense(512)(embedding)
        # embedding = nn.sigmoid(embedding)
        embedding = nn.Dense(512)(embedding)
        embedding = nn.sigmoid(embedding)

        # Actor Layer
        actor_mean = nn.Dense(32)(embedding)
        actor_mean = nn.sigmoid(actor_mean)
        actor_mean = nn.Dense(self.action_dim)(actor_mean)

        # Action Logits
        # unavail_actions = 1 - avail_actions
        # action_logits = actor_mean - (unavail_actions * 1e10)
        pi = distrax.Categorical(logits=actor_mean)

        # Critic Layer
        critic = nn.Dense(512)(embedding)
        critic = nn.sigmoid(critic)
        critic = nn.Dense(1)(critic)

        return pi, jnp.squeeze(critic, axis=-1)
    

class ActorCriticSpeaker(nn.Module):
    latent_dim: int
    num_classes: int
    image_dim: Sequence[int]
    config: Dict

    @nn.compact
    def __call__(self, y):
        y = nn.Embed(self.num_classes, self.latent_dim)(y)
        # y = jnp.squeeze(y, axis=(0))
        # z = jnp.concatenate([z, y], axis=-1)
        z = nn.Dense(7 * 7 * 256)(y)
        z = nn.relu(z)
        z = z.reshape((-1, 7, 7, 256))
        z = nn.ConvTranspose(128, kernel_size=(4, 4), strides=(2, 2), padding='SAME')(z)
        z = nn.relu(z)
        z = nn.ConvTranspose(64, kernel_size=(4, 4), strides=(2, 2), padding='SAME')(z)
        z = nn.relu(z)
        z = nn.ConvTranspose(1, kernel_size=(3, 3), padding='SAME')(z)
        z = jnp.squeeze(z, axis=-1)  # Remove the channel dimension
        z = jnp.reshape(z, (-1, self.image_dim ** 2))  # Flatten the image

        # Actor Mean
        actor_mean = nn.Dense(self.image_dim ** 2)(z)
        actor_mean = nn.sigmoid(actor_mean)  # Apply sigmoid to squash outputs between 0 and 1

        # Actor Standard Deviation
        actor_std = nn.Dense(self.image_dim ** 2, kernel_init=orthogonal(np.sqrt(2)))(z)
        actor_std = nn.softplus(actor_std) * 0.25 + 1e-6  # Ensure positive standard deviation

        # Create a multivariate normal distribution with diagonal covariance matrix
        pi = distrax.MultivariateNormalDiag(loc=actor_mean, scale_diag=actor_std)

        # Critic
        critic = nn.Dense(512, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(z)
        critic = nn.sigmoid(critic)
        critic = nn.Dense(512, kernel_init=orthogonal(2), bias_init=constant(0.0))(critic)
        critic = nn.sigmoid(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(critic)

        return pi, jnp.squeeze(critic, axis=-1)
    

class ActorCriticSpeakerGaussSmall(nn.Module):
    latent_dim: int
    num_classes: int
    image_dim: Sequence[int]
    config: Dict

    @nn.compact
    def __call__(self, obs):
        y = nn.Embed(self.num_classes, self.latent_dim)(obs)
        z = nn.Dense(32, kernel_init=nn.initializers.he_normal())(y)
        z = nn.relu(z)
        z = nn.Dense(256, kernel_init=nn.initializers.he_normal())(z)
        z = nn.relu(z)
        z = nn.Dense(256, kernel_init=nn.initializers.he_normal())(z)
        z = nn.relu(z)

        # Actor Mean
        actor_mean = nn.Dense(self.image_dim ** 2, kernel_init=nn.initializers.he_normal())(z)
        actor_mean = nn.sigmoid(actor_mean)  # Apply sigmoid to squash outputs between 0 and 1

        # Actor Standard Deviation
        actor_std = nn.Dense(self.image_dim ** 2)(z)
        actor_std = nn.softplus(actor_std) * 0.25 + 1e-6  # Ensure positive standard deviation

        # Create a multivariate normal distribution with diagonal covariance matrix
        pi = distrax.MultivariateNormalDiag(loc=actor_mean, scale_diag=actor_std)

        # Critic
        critic = nn.Dense(512)(z)
        critic = nn.sigmoid(critic)
        critic = nn.Dense(1)(critic)

        return pi, jnp.squeeze(critic, axis=-1)

class ActorCriticSpeakerGaussSmallNovariance(nn.Module):
    latent_dim: int
    num_classes: int
    image_dim: Sequence[int]
    config: Dict

    @nn.compact
    def __call__(self, obs):
        y = nn.Embed(self.num_classes, self.latent_dim)(obs)
        z = nn.Dense(32, kernel_init=nn.initializers.he_normal())(y)
        z = nn.relu(z)
        z = nn.Dropout(rate=self.config["SPEAKER_DROPOUT"], deterministic=False)(z)
        z = nn.Dense(256, kernel_init=nn.initializers.he_normal())(z)
        z = nn.relu(z)
        z = nn.Dropout(rate=self.config["SPEAKER_DROPOUT"], deterministic=False)(z)
        z = nn.Dense(256, kernel_init=nn.initializers.he_normal())(z)
        z = nn.relu(z)

        # Actor Mean
        actor_mean = nn.Dense(self.image_dim ** 2, kernel_init=nn.initializers.he_normal())(z)
        actor_mean = nn.sigmoid(actor_mean)  # Apply sigmoid to squash outputs between 0 and 1

        # Create a multivariate normal distribution with diagonal covariance matrix
        pi = distrax.MultivariateNormalDiag(loc=actor_mean, scale_diag=jnp.ones_like(actor_mean)*0.05)  # TODO: This scale diag is not being calculated right. Take a look at how it's done in ActorCriticSpeakerGaussSplat

        # Critic
        critic = nn.Dense(512)(z)
        critic = nn.sigmoid(critic)
        critic = nn.Dropout(rate=self.config["SPEAKER_DROPOUT"], deterministic=False)(critic)
        critic = nn.Dense(512)(critic)
        critic = nn.sigmoid(critic)
        critic = nn.Dropout(rate=self.config["SPEAKER_DROPOUT"], deterministic=False)(critic)
        critic = nn.Dense(512)(critic)
        critic = nn.sigmoid(critic)
        critic = nn.Dense(1)(critic)

        return pi, jnp.squeeze(critic, axis=-1)


class ActorCriticSpeakerGaussSplat(nn.Module):
    latent_dim: int
    num_classes: int
    action_dim: Sequence[int]
    config: Dict

    @nn.compact
    def __call__(self, obs):
        y = nn.Embed(self.num_classes, self.latent_dim)(obs)
        z = nn.Dense(32, kernel_init=nn.initializers.he_uniform())(y)
        z = nn.relu(z)
        z = nn.Dropout(rate=self.config["SPEAKER_DROPOUT"], deterministic=False)(z)
        z = nn.Dense(128, kernel_init=nn.initializers.he_uniform())(z)
        z = nn.relu(z)
        z = nn.Dropout(rate=self.config["SPEAKER_DROPOUT"], deterministic=False)(z)
        z = nn.Dense(128, kernel_init=nn.initializers.he_uniform())(z)
        z = nn.relu(z)

        # Actor Mean
        actor_mean = nn.Dense(self.action_dim, kernel_init=nn.initializers.normal(0.2))(z)
        actor_mean = nn.sigmoid(actor_mean)  # Apply sigmoid to squash outputs between 0 and 1

        # x_mu_norm, y_mu_norm, sigma_x2_norm, sigma_y2_norm, amplitude. e.g.
        #[0.5, 0.5, 0.01, 0.01, 1.0],
        #[0.7, 0.7, 0.005, 0.005, 0.5],
        #[0.3, 0.4, 0.015, 0.01, 0.75]
        scale_factor = jnp.tile(jnp.array([1.0, 1.0, 0.02, 0.02, 1.0], dtype=jnp.float32), actor_mean.shape[-1]//5)
        actor_mean *= scale_factor

        # Create a multivariate normal distribution with diagonal covariance matrix
        pi = distrax.MultivariateNormalDiag(loc=actor_mean, scale_diag=jnp.ones_like(actor_mean)*0.001)

        # Critic
        critic = nn.Dense(128)(z)
        critic = nn.sigmoid(critic)
        critic = nn.Dropout(rate=self.config["SPEAKER_DROPOUT"], deterministic=False)(critic)
        critic = nn.Dense(32)(critic)
        critic = nn.sigmoid(critic)
        critic = nn.Dropout(rate=self.config["SPEAKER_DROPOUT"], deterministic=False)(critic)
        critic = nn.Dense(32)(critic)
        critic = nn.sigmoid(critic)
        critic = nn.Dense(1)(critic)

        return pi, jnp.squeeze(critic, axis=-1)


class ActorCriticSpeakerGaussSplatCov(nn.Module):
    latent_dim: int
    num_classes: int
    action_dim: Sequence[int]
    config: Dict

    @nn.compact
    def __call__(self, obs):
        y = nn.Embed(self.num_classes, self.latent_dim)(obs)
        z = nn.Dense(32, kernel_init=nn.initializers.he_uniform())(y)
        z = nn.relu(z)
        z = nn.Dense(128, kernel_init=nn.initializers.he_uniform())(z)
        z = nn.relu(z)
        z = nn.Dense(128, kernel_init=nn.initializers.he_uniform())(z)
        z = nn.relu(z)

        # Actor Mean
        actor_mean = nn.Dense(self.action_dim, kernel_init=nn.initializers.normal(1.0/3.0))(z)
        actor_mean = nn.sigmoid(actor_mean)  # Apply sigmoid to squash outputs between 0 and 1
        
        # Create a multivariate normal distribution with diagonal covariance matrix
        pi = distrax.MultivariateNormalDiag(loc=actor_mean, scale_diag=jnp.ones_like(actor_mean)*0.01)

        # Critic
        critic = nn.Dense(128)(actor_mean)
        critic = nn.sigmoid(critic)
        # critic = nn.Dropout(rate=self.config["SPEAKER_DROPOUT"], deterministic=False)(critic)
        critic = nn.Dense(128)(critic)
        critic = nn.sigmoid(critic)
        # critic = nn.Dropout(rate=self.config["SPEAKER_DROPOUT"], deterministic=False)(critic)
        critic = nn.Dense(32)(critic)
        critic = nn.sigmoid(critic)
        critic = nn.Dense(1)(critic)

        return pi, jnp.squeeze(critic, axis=-1)


class ActorCriticSpeakerGaussSplatChol(nn.Module):
    latent_dim: int
    num_classes: int
    action_dim: Sequence[int]
    config: Dict

    @nn.compact
    def __call__(self, obs):
        y = nn.Embed(self.num_classes, self.latent_dim)(obs)
        z = nn.Dense(32, kernel_init=nn.initializers.he_uniform())(y)
        z = nn.relu(z)
        z = nn.Dense(128, kernel_init=nn.initializers.he_uniform())(z)
        z = nn.relu(z)
        z = nn.Dense(128, kernel_init=nn.initializers.he_uniform())(z)
        z = nn.relu(z)

        # Actor Mean
        actor_mean = nn.Dense(self.action_dim, kernel_init=nn.initializers.normal(0.33))(z) # Experiment with this!
        actor_mean = nn.sigmoid(actor_mean)  # Apply sigmoid to squash outputs between 0 and 1

        # scale_diag = nn.Dense(self.action_dim, kernel_init=nn.initializers.normal(0.33))(z)
        # scale_diag = nn.sigmoid(scale_diag) * 0.01
        
        # Create a multivariate normal distribution with diagonal covariance matrix
        pi = distrax.MultivariateNormalDiag(loc=actor_mean, scale_diag=jnp.ones_like(actor_mean)*0.01)

        # Critic
        critic = nn.Dense(128)(actor_mean)
        critic = nn.sigmoid(critic)
        # critic = nn.Dropout(rate=self.config["SPEAKER_DROPOUT"], deterministic=False)(critic)
        critic = nn.Dense(128)(critic)
        critic = nn.sigmoid(critic)
        # critic = nn.Dropout(rate=self.config["SPEAKER_DROPOUT"], deterministic=False)(critic)
        critic = nn.Dense(32)(critic)
        critic = nn.sigmoid(critic)
        critic = nn.Dense(1)(critic)

        return pi, jnp.squeeze(critic, axis=-1)
