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
from ablated_agents import ActorCriticListenerConvAblationReady, LISTENER_ARCH_ABLATION_PARAMETERS, ActorCriticSpeakerSplinesAblationReady, SPEAKER_ARCH_ABLATION_PARAMETERS

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
        ep_done = jnp.concatenate(done, axis=1)
        _reward = jnp.concatenate(reward, axis=1)
        new_episode_return = state.episode_returns + _reward    # (1, 18)
        new_episode_length = state.episode_lengths + 1
        old_ep_done = ep_done

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
        return obs, state, reward, done, info   # done is a (speaker, listener) tuple of jnp arrays of shape (1, num_agents)
    

class CustomBatchNorm(nn.Module):
    epsilon: float = 1e-5

    @nn.compact
    def __call__(self, x):
        # Calculate mean and variance
        mean = jnp.mean(x, axis=0, keepdims=True)
        variance = jnp.var(x, axis=0, keepdims=True)

        # Normalize
        x_normalized = (x - mean) / jnp.sqrt(variance + self.epsilon)

        # Learnable scale and shift parameters
        gamma = self.param('gamma', nn.initializers.ones, (1, x.shape[-1]))
        beta = self.param('beta', nn.initializers.zeros, (1, x.shape[-1]))

        return gamma * x_normalized + beta
        

class ActorCriticListenerConv(nn.Module):
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
        critic = nn.Dense(128)(critic)
        critic = nn.relu(critic)
        critic = nn.Dense(1)(critic)

        return pi, jnp.squeeze(critic, axis=-1)
    

class ActorCriticListenerBoost(nn.Module):
    action_dim: Sequence[int]
    image_dim: Sequence[int]
    config: Dict

    @nn.compact
    def __call__(self, x):
        x = x.reshape(-1, self.image_dim, self.image_dim, 1)  # Assuming x is flat, and image_dim is [height, width]

        # Convolutional layers
        x = nn.Conv(features=16, kernel_size=(3, 3), strides=(1, 1), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.Conv(features=32, kernel_size=(3, 3), strides=(1, 1), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.Conv(features=64, kernel_size=(3, 3), strides=(1, 1), padding='SAME')(x)
        x = nn.relu(x)
        x = x.reshape((x.shape[0], -1))  # Flatten
        
        # Embedding Layer
        embedding = nn.Dense(256)(x)
        embedding = nn.relu(embedding)
        # embedding = nn.Dropout(rate=self.config["LISTENER_DROPOUT"], deterministic=False)(embedding)
        embedding = nn.Dense(256)(embedding)
        embedding = nn.relu(embedding)
        # embedding = nn.Dropout(rate=self.config["LISTENER_DROPOUT"], deterministic=False)(embedding)
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
    
class ActorCriticListenerConvReduced(nn.Module):
    action_dim: Sequence[int]
    image_dim: Sequence[int]
    config: Dict

    @nn.compact
    def __call__(self, x):
        x = x.reshape(-1, self.image_dim, self.image_dim, 1)  # Assuming x is flat, and image_dim is [height, width]

        # Convolutional layers
        x = nn.Conv(features=16, kernel_size=(3, 3), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.Conv(features=32, kernel_size=(3, 3), padding='SAME')(x)
        x = nn.relu(x)
        x = x.reshape((x.shape[0], -1))  # Flatten
        
        # Embedding Layer
        embedding = nn.Dense(128)(x)
        embedding = nn.relu(embedding)
        # embedding = nn.Dropout(rate=self.config["LISTENER_DROPOUT"], deterministic=False)(embedding)
        # embedding = nn.Dense(128)(embedding)
        # embedding = nn.relu(embedding)
        # embedding = nn.Dropout(rate=self.config["LISTENER_DROPOUT"], deterministic=False)(embedding)
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


class ActorCriticListenerConvSmall(nn.Module):
    action_dim: Sequence[int]
    image_dim: Sequence[int]
    config: Dict

    @nn.compact
    def __call__(self, x):
        x = x.reshape(-1, self.image_dim, self.image_dim, 1)  # Assuming x is flat, and image_dim is [height, width]

        # Convolutional layers
        x = nn.Conv(features=16, kernel_size=(3, 3), strides=(1, 1), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.Conv(features=32, kernel_size=(3, 3), strides=(1, 1), padding='SAME')(x)
        x = nn.relu(x)
        x = x.reshape((x.shape[0], -1))  # Flatten
        
        # Embedding Layer
        embedding = nn.Dense(128)(x)
        embedding = nn.relu(embedding)
        embedding = nn.Dropout(rate=self.config["LISTENER_DROPOUT"], deterministic=False)(embedding)
        
        # Actor Layer
        actor_mean = nn.Dense(128)(embedding)
        actor_mean = nn.relu(actor_mean)
        actor_mean = nn.Dense(self.action_dim)(actor_mean)
        actor_mean = nn.softmax(actor_mean)
        pi = distrax.Categorical(probs=actor_mean)

        # Critic Layer
        critic = nn.Dense(128)(embedding)
        critic = nn.relu(critic)
        critic = nn.Dense(1)(critic)

        return pi, jnp.squeeze(critic, axis=-1)

    

class ActorCriticListenerConvSigmoid(nn.Module):
    action_dim: Sequence[int]
    image_dim: Sequence[int]
    config: Dict

    @nn.compact
    def __call__(self, x):
        x = x.reshape(-1, self.image_dim, self.image_dim, 1)  # Assuming x is flat, and image_dim is [height, width]

        # Convolutional layers
        x = nn.Conv(features=32, kernel_size=(3, 3), strides=(1, 1), padding='SAME')(x)
        x = nn.sigmoid(x)
        x = nn.Conv(features=32, kernel_size=(3, 3), strides=(1, 1), padding='SAME')(x)
        x = nn.sigmoid(x)
        x = x.reshape((x.shape[0], -1))  # Flatten
        
        # Embedding Layer
        embedding = nn.Dense(128)(x)
        embedding = nn.sigmoid(embedding)
        embedding = nn.Dropout(rate=self.config["LISTENER_DROPOUT"], deterministic=False)(embedding)
        embedding = nn.Dense(128)(embedding)
        embedding = nn.sigmoid(embedding)
        embedding = nn.Dropout(rate=self.config["LISTENER_DROPOUT"], deterministic=False)(embedding)
        embedding = nn.Dense(128)(embedding)
        embedding = nn.sigmoid(embedding)

        # Actor Layer
        actor_mean = nn.Dense(128)(embedding)
        actor_mean = nn.sigmoid(actor_mean)
        actor_mean = nn.Dense(self.action_dim)(actor_mean)
        actor_mean = nn.softmax(actor_mean)
        pi = distrax.Categorical(probs=actor_mean)

        # Critic Layer
        critic = nn.Dense(128)(embedding)
        critic = nn.sigmoid(critic)
        # critic = nn.Dropout(rate=self.config["LISTENER_DROPOUT"], deterministic=False)(critic)
        critic = nn.Dense(128)(critic)
        critic = nn.sigmoid(critic)
        critic = nn.Dense(1)(critic)

        return pi, jnp.squeeze(critic, axis=-1)


class ActorCriticListenerDense(nn.Module):
    action_dim: Sequence[int]
    image_dim: Sequence[int]
    config: Dict

    @nn.compact
    def __call__(self, x):
        x = x.reshape(-1, self.image_dim, self.image_dim, 1)  # Assuming x is flat, and image_dim is [height, width]

        # Convolutional layers
        x = nn.Conv(features=32, kernel_size=(3, 3), strides=(1, 1), padding='SAME')(x)
        x = nn.sigmoid(x)
        x = x.reshape((x.shape[0], -1))

        # Embedding Layer
        embedding = nn.Dense(128)(x)
        embedding = nn.relu(embedding)
        embedding = nn.Dense(128)(embedding)
        embedding = nn.relu(embedding)
        embedding = nn.Dense(128)(embedding)
        embedding = nn.relu(embedding)
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
        critic = nn.Dense(128)(critic)
        critic = nn.relu(critic)
        critic = nn.Dense(1)(critic)

        return pi, jnp.squeeze(critic, axis=-1)

class ActorCriticListenerDenseBatchnorm(nn.Module):
    action_dim: Sequence[int]
    image_dim: Sequence[int]
    config: Dict

    @nn.compact
    def __call__(self, x):
        obs = x
        # Embedding Layer
        embedding = nn.Dense(512, kernel_init=nn.initializers.he_normal())(obs)
        embedding = CustomBatchNorm()(embedding)
        embedding = nn.relu(embedding)
        embedding = nn.Dense(256, kernel_init=nn.initializers.he_normal())(embedding)
        embedding = CustomBatchNorm()(embedding)
        embedding = nn.relu(embedding)
        embedding = nn.Dense(256, kernel_init=nn.initializers.he_normal())(embedding)
        embedding = CustomBatchNorm()(embedding)
        embedding = nn.relu(embedding)

        # Actor Layer
        actor_mean = nn.Dense(128)(embedding)
        actor_mean = CustomBatchNorm()(actor_mean)
        actor_mean = nn.relu(actor_mean)
        actor_mean = nn.Dense(self.action_dim)(actor_mean)

        # Action Logits
        pi = distrax.Categorical(logits=actor_mean)

        # Critic Layer
        critic = nn.Dense(512)(embedding)
        critic = CustomBatchNorm()(critic)
        critic = nn.relu(critic)
        critic = nn.Dense(256)(critic)
        critic = CustomBatchNorm()(critic)
        critic = nn.relu(critic)
        critic = nn.Dense(1)(critic)

        return pi, jnp.squeeze(critic, axis=-1)
    

class ActorCriticSpeakerFullImage(nn.Module):
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

class ActorCriticSpeakerFullImageSetVariance(nn.Module):
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
        actor_mean = nn.Dense(self.action_dim, kernel_init=nn.initializers.normal(self.config["SPEAKER_STD_DEV"]))(z)  # TODO: Eventually I can sweep over these parameters
        actor_mean = nn.sigmoid(actor_mean)  # Apply sigmoid to squash outputs between 0 and 1

        scale_diag = nn.Dense(self.action_dim, kernel_init=nn.initializers.normal(self.config["SPEAKER_STD_DEV2"]))(z)
        scale_diag = nn.sigmoid(scale_diag) * 0.2 + 1e-8
        
        # Create a multivariate normal distribution with diagonal covariance matrix
        pi = distrax.MultivariateNormalDiag(loc=actor_mean, scale_diag=scale_diag)

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
    

class ActorCriticSpeakerSplines(nn.Module):
    latent_dim: int
    num_classes: int
    action_dim: Sequence[int]
    config: Dict

    @nn.compact
    def __call__(self, obs):
        y = nn.Embed(self.num_classes, self.latent_dim)(obs)
        z = nn.Dense(128, kernel_init=nn.initializers.he_uniform())(y)
        z = nn.relu(z)
        z = nn.Dense(128, kernel_init=nn.initializers.he_uniform())(z)
        z = nn.relu(z)
        z = nn.Dense(128, kernel_init=nn.initializers.he_uniform())(z)
        z = nn.relu(z)

        # Actor Mean
        actor_mean = nn.Dense(self.action_dim, kernel_init=nn.initializers.normal(self.config["SPEAKER_STDDEV"]))(z)  # TODO: Eventually I can sweep over these parameters
        actor_mean = nn.sigmoid(actor_mean)  # Apply sigmoid to squash outputs between 0 and 1

        scale_diag = nn.Dense(self.action_dim, kernel_init=nn.initializers.normal(self.config["SPEAKER_STDDEV2"]))(z)
        scale_diag = nn.sigmoid(scale_diag) * self.config["SPEAKER_SQUISH"] + 1e-8
        
        # Create a multivariate normal distribution with diagonal covariance matrix
        pi = distrax.MultivariateNormalDiag(loc=actor_mean, scale_diag=scale_diag)

        # Critic
        critic = nn.Dense(128)(z)
        critic = nn.sigmoid(critic)
        # critic = nn.Dropout(rate=self.config["SPEAKER_DROPOUT"], deterministic=False)(critic)
        critic = nn.Dense(128)(critic)
        critic = nn.sigmoid(critic)
        # critic = nn.Dropout(rate=self.config["SPEAKER_DROPOUT"], deterministic=False)(critic)
        critic = nn.Dense(32)(critic)
        critic = nn.sigmoid(critic)
        critic = nn.Dense(1)(critic)

        return pi, jnp.squeeze(critic, axis=-1) # Return policy function and value



class ActorCriticSpeakerSplinesNoise(nn.Module):
    latent_dim: int
    num_classes: int
    action_dim: Sequence[int]
    noise_dim: int
    noise_stddev: float
    config: Dict

    @nn.compact
    def __call__(self, obs):
        y = nn.Embed(self.num_classes, self.latent_dim)(obs)
        noise = self.noise_stddev * jax.random.normal(self.make_rng('noise'), shape=(obs.shape[0], self.noise_dim,))
        y = jnp.concatenate([y, noise], axis=-1)

        z = nn.Dense(32, kernel_init=nn.initializers.he_uniform())(y)
        z = nn.sigmoid(z)
        z = nn.Dense(128, kernel_init=nn.initializers.he_uniform())(z)
        z = nn.sigmoid(z)
        z = nn.Dense(128, kernel_init=nn.initializers.he_uniform())(z)
        z = nn.sigmoid(z)

        # Actor Mean
        actor_mean = nn.Dense(self.action_dim, kernel_init=nn.initializers.normal(self.config["SPEAKER_STDDEV"]))(z)  # TODO: Eventually I can sweep over these parameters
        actor_mean = nn.sigmoid(actor_mean)  # Apply sigmoid to squash outputs between 0 and 1

        scale_diag = nn.Dense(self.action_dim, kernel_init=nn.initializers.normal(self.config["SPEAKER_STDDEV2"]))(z)   # TODO: I should stick to a consistent schema
        scale_diag = nn.sigmoid(scale_diag) * self.config["SPEAKER_SQUISH"] + 1e-8
        
        # Create a multivariate normal distribution with diagonal covariance matrix
        pi = distrax.MultivariateNormalDiag(loc=actor_mean, scale_diag=scale_diag)

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

class ActorCriticListenerBottleneck(nn.Module):
    """
    This actor-critic model has a bottleneck with the embedding layer 
    which leads to more compact representations and may make ToM reasoning
    better.
    """
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

        # Bottleneck Layer
        bottleneck = nn.Dense(32)(x)
        bottleneck = nn.relu(bottleneck)
        
        # Embedding Layer
        embedding = nn.Dense(128)(bottleneck)
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
        critic = nn.Dense(128)(critic)
        critic = nn.relu(critic)
        critic = nn.Dense(1)(critic)

        return pi, jnp.squeeze(critic, axis=-1)

class ActorCriticListenerLargeKernel(nn.Module):
    """
    This actor-critic model includes a larger kernel than other
    models, perhaps enabling listeners to be receptive to more 
    sensitive to local changes. 
    """
    action_dim: Sequence[int]
    image_dim: Sequence[int]
    config: Dict

    @nn.compact
    def __call__(self, x):
        x = x.reshape(-1, self.image_dim, self.image_dim, 1)  # Assuming x is flat, and image_dim is [height, width]

        # Convolutional layers
        x = nn.Conv(features=32, kernel_size=(5, 5), strides=(1, 1), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.Conv(features=64, kernel_size=(5, 5), strides=(1, 1), padding='SAME')(x)
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
        critic = nn.Dense(128)(critic)
        critic = nn.relu(critic)
        critic = nn.Dense(1)(critic)

        return pi, jnp.squeeze(critic, axis=-1)

class ActorCriticListenerStrided(nn.Module):
    """
    This actor-critic model includes a larger stride than other
    models, perhaps enabling listeners to be receptive to global
    fetaures.
    """
    action_dim: Sequence[int]
    image_dim: Sequence[int]
    config: Dict

    @nn.compact
    def __call__(self, x):
        x = x.reshape(-1, self.image_dim, self.image_dim, 1)  # Assuming x is flat, and image_dim is [height, width]

        # Convolutional layers
        x = nn.Conv(features=32, kernel_size=(3, 3), strides=(3, 3), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.Conv(features=64, kernel_size=(3, 3), strides=(3, 3), padding='SAME')(x)
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
        critic = nn.Dense(128)(critic)
        critic = nn.relu(critic)
        critic = nn.Dense(1)(critic)

        return pi, jnp.squeeze(critic, axis=-1)

class ActorCriticListenerStrongEmbedding(nn.Module):
    """
    This model has a strong embedding layer that could lead 
    to more complex representations of the observed splines. The 
    actor and critic layers themselves are barebones to allow the 
    internal representation to do the heavy-lifting.
    """
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
        embedding = nn.Dense(64)(x)
        embedding = nn.LayerNorm()(embedding)
        embedding = nn.relu(embedding)
        embedding = nn.Dropout(rate=self.config["LISTENER_DROPOUT"], deterministic=False)(embedding)

        embedding = nn.Dense(128)(embedding)
        embedding = nn.LayerNorm()(embedding)
        embedding = nn.relu(embedding)
        embedding = nn.Dropout(rate=self.config["LISTENER_DROPOUT"], deterministic=False)(embedding)

        embedding = nn.Dense(256)(embedding)
        embedding = nn.LayerNorm()(embedding)
        embedding = nn.relu(embedding)
        embedding = nn.Dropout(rate=self.config["LISTENER_DROPOUT"], deterministic=False)(embedding)

        embedding = nn.Dense(256)(embedding)
        embedding = nn.LayerNorm()(embedding)
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
        critic = nn.Dense(128)(critic)
        critic = nn.relu(critic)
        critic = nn.Dense(1)(critic)

        return pi, jnp.squeeze(critic, axis=-1)

class ActorCriticListenerStrongConvWeakEmbed(nn.Module):
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
        x = nn.Conv(features=64, kernel_size=(3, 3), strides=(1, 1), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = x.reshape((x.shape[0], -1))  # Flatten
        
        # Embedding Layer
        embedding = nn.Dense(128)(x)
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
        critic = nn.Dense(128)(critic)
        critic = nn.relu(critic)
        critic = nn.Dense(1)(critic)

        return pi, jnp.squeeze(critic, axis=-1)

class ActorCriticListenerHighFeatures(nn.Module):
    action_dim: Sequence[int]
    image_dim: Sequence[int]
    config: Dict

    @nn.compact
    def __call__(self, x):
        x = x.reshape(-1, self.image_dim, self.image_dim, 1)  # Assuming x is flat, and image_dim is [height, width]

        # Convolutional layers
        x = nn.Conv(features=64, kernel_size=(3, 3), strides=(1, 1), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.Conv(features=128, kernel_size=(3, 3), strides=(1, 1), padding='SAME')(x)
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
        critic = nn.Dense(128)(critic)
        critic = nn.relu(critic)
        critic = nn.Dense(1)(critic)

        return pi, jnp.squeeze(critic, axis=-1)

class ActorCriticListenerLowFeatures(nn.Module):
    action_dim: Sequence[int]
    image_dim: Sequence[int]
    config: Dict

    @nn.compact
    def __call__(self, x):
        x = x.reshape(-1, self.image_dim, self.image_dim, 1)  # Assuming x is flat, and image_dim is [height, width]

        # Convolutional layers
        x = nn.Conv(features=32, kernel_size=(3, 3), strides=(1, 1), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.Conv(features=32, kernel_size=(3, 3), strides=(1, 1), padding='SAME')(x)
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
        critic = nn.Dense(128)(critic)
        critic = nn.relu(critic)
        critic = nn.Dense(1)(critic)

        return pi, jnp.squeeze(critic, axis=-1)

class ActorCriticListenerHighEmbedding(nn.Module):
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
        embedding = nn.Dense(256)(x)
        embedding = nn.relu(embedding)
        embedding = nn.Dropout(rate=self.config["LISTENER_DROPOUT"], deterministic=False)(embedding)
        embedding = nn.Dense(256)(embedding)
        embedding = nn.relu(embedding)
        embedding = nn.Dropout(rate=self.config["LISTENER_DROPOUT"], deterministic=False)(embedding)
        embedding = nn.Dense(256)(embedding)
        embedding = nn.relu(embedding)

        # Actor Layer
        actor_mean = nn.Dense(256)(embedding)
        actor_mean = nn.relu(actor_mean)
        actor_mean = nn.Dense(self.action_dim)(actor_mean)
        actor_mean = nn.softmax(actor_mean)
        pi = distrax.Categorical(probs=actor_mean)

        # Critic Layer
        critic = nn.Dense(256)(embedding)
        critic = nn.relu(critic)
        critic = nn.Dense(256)(critic)
        critic = nn.relu(critic)
        critic = nn.Dense(1)(critic)

        return pi, jnp.squeeze(critic, axis=-1)

class ActorCriticListenerLowEmbedding(nn.Module):
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
        embedding = nn.Dense(64)(x)
        embedding = nn.relu(embedding)
        embedding = nn.Dropout(rate=self.config["LISTENER_DROPOUT"], deterministic=False)(embedding)
        embedding = nn.Dense(64)(embedding)
        embedding = nn.relu(embedding)
        embedding = nn.Dropout(rate=self.config["LISTENER_DROPOUT"], deterministic=False)(embedding)
        embedding = nn.Dense(64)(embedding)
        embedding = nn.relu(embedding)

        # Actor Layer
        actor_mean = nn.Dense(64)(embedding)
        actor_mean = nn.relu(actor_mean)
        actor_mean = nn.Dense(self.action_dim)(actor_mean)
        actor_mean = nn.softmax(actor_mean)
        pi = distrax.Categorical(probs=actor_mean)

        # Critic Layer
        critic = nn.Dense(64)(embedding)
        critic = nn.relu(critic)
        critic = nn.Dense(64)(critic)
        critic = nn.relu(critic)
        critic = nn.Dense(1)(critic)

        return pi, jnp.squeeze(critic, axis=-1)

class ActorCriticListenerLessConv(nn.Module):
    action_dim: Sequence[int]
    image_dim: Sequence[int]
    config: Dict

    @nn.compact
    def __call__(self, x):
        x = x.reshape(-1, self.image_dim, self.image_dim, 1)  # Assuming x is flat, and image_dim is [height, width]

        # Convolutional layers
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
        critic = nn.Dense(128)(critic)
        critic = nn.relu(critic)
        critic = nn.Dense(1)(critic)

        return pi, jnp.squeeze(critic, axis=-1)

class ActorCriticListenerLessEmbed(nn.Module):
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

        # Actor Layer
        actor_mean = nn.Dense(128)(embedding)
        actor_mean = nn.relu(actor_mean)
        actor_mean = nn.Dense(self.action_dim)(actor_mean)
        actor_mean = nn.softmax(actor_mean)
        pi = distrax.Categorical(probs=actor_mean)

        # Critic Layer
        critic = nn.Dense(128)(embedding)
        critic = nn.relu(critic)
        critic = nn.Dense(128)(critic)
        critic = nn.relu(critic)
        critic = nn.Dense(1)(critic)

        return pi, jnp.squeeze(critic, axis=-1)
    
def examine_speaker():
    speaker_network = ActorCriticSpeakerSplines(latent_dim=64, num_classes=10, action_dim=12, config={"SPEAKER_STDDEV": 0.6, "SPEAKER_STDDEV2": 0.2})
    
    rng = jax.random.PRNGKey(52)
    rng, _rng = jax.random.split(rng)

    init_y = jnp.zeros(
            (1, 1, 1),
            dtype=jnp.int32
        )
    network_params = speaker_network.init({'params': _rng, 'dropout': _rng}, init_y)

    obs = jnp.array([0])
    pi, critic = speaker_network.apply(network_params, obs)
    a = pi.sample(seed=rng)
    print(a.tolist())
    


if __name__ == "__main__":
    examine_speaker()
