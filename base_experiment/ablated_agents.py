import jax.numpy as jnp
import flax.linen as nn
import distrax
from typing import Sequence, Dict


# This is a copy of ActorCriticListenerConv from agents.py, placed here as a reference
class __ActorCriticListenerConv(nn.Module):
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


class ActorCriticListenerConvAblationReady(nn.Module):
    action_dim: Sequence[int]
    image_dim: Sequence[int]
    config: Dict

    @nn.compact
    def __call__(self, x):
        # Get architecture from config with defaults
        arch = self.config.get("LISTENER_ARCH_ABLATION_PARAMS", {})
        conv_features = arch.get("conv_features", [32, 64])
        embedding_dims = arch.get("embedding_dims", [128, 128, 128])
        actor_hidden = arch.get("actor_hidden", 128)
        critic_hidden = arch.get("critic_hidden", [128, 128])
        
        x = x.reshape(-1, self.image_dim, self.image_dim, 1)

        # Conv layers
        for features in conv_features:
            x = nn.Conv(features=features, kernel_size=(3, 3), strides=(1, 1), padding='SAME')(x)
            x = nn.relu(x)
        
        x = x.reshape((x.shape[0], -1))
        
        # Embedding
        embedding = x
        for i, dim in enumerate(embedding_dims):
            embedding = nn.Dense(dim)(embedding)
            embedding = nn.relu(embedding)
            if i < len(embedding_dims) - 1:
                embedding = nn.Dropout(rate=self.config["LISTENER_DROPOUT"], deterministic=False)(embedding)

        # Actor
        actor_mean = nn.Dense(actor_hidden)(embedding)
        actor_mean = nn.relu(actor_mean)
        actor_mean = nn.Dense(self.action_dim)(actor_mean)
        actor_mean = nn.softmax(actor_mean)
        pi = distrax.Categorical(probs=actor_mean)

        # Critic
        critic = embedding
        for dim in critic_hidden:
            critic = nn.Dense(dim)(critic)
            critic = nn.relu(critic)
        critic = nn.Dense(1)(critic)

        return pi, jnp.squeeze(critic, axis=-1)


LISTENER_ARCH_ABLATION_PARAMETERS = {
    "conv-ablate-0": {
        "LISTENER_ARCH_ABLATION_PARAMS": {
            "conv_features": [32, 64],
            "embedding_dims": [128, 128, 128],
            "actor_hidden": 128,
            "critic_hidden": [128, 128]
        }
    },
    "conv-ablate-50-1": {
        "LISTENER_ARCH_ABLATION_PARAMS": {
            "conv_features": [16, 64],
            "embedding_dims": [128, 128, 128],
            "actor_hidden": 128,
            "critic_hidden": [128, 128]
        }
    },
    "conv-ablate-50-2": {
        "LISTENER_ARCH_ABLATION_PARAMS": {
            "conv_features": [32, 32],
            "embedding_dims": [128, 128, 128],
            "actor_hidden": 128,
            "critic_hidden": [128, 128]
        }
    },
    "conv-ablate-50-3": {
        "LISTENER_ARCH_ABLATION_PARAMS": {
            "conv_features": [32, 64],
            "embedding_dims": [64, 128, 128],
            "actor_hidden": 128,
            "critic_hidden": [128, 128]
        }
    },
    "conv-ablate-50-4": {
        "LISTENER_ARCH_ABLATION_PARAMS": {
            "conv_features": [32, 64],
            "embedding_dims": [128, 64, 128],
            "actor_hidden": 128,
            "critic_hidden": [128, 128]
        }
    },
    "conv-ablate-50-5": {
        "LISTENER_ARCH_ABLATION_PARAMS": {
            "conv_features": [32, 64],
            "embedding_dims": [128, 128, 64],
            "actor_hidden": 128,
            "critic_hidden": [128, 128]
        }
    },
    "conv-ablate-50-6": {
        "LISTENER_ARCH_ABLATION_PARAMS": {
            "conv_features": [32, 64],
            "embedding_dims": [128, 128, 128],
            "actor_hidden": 64,
            "critic_hidden": [128, 128]
        }
    },
    "conv-ablate-25-1": {
        "LISTENER_ARCH_ABLATION_PARAMS": {
            "conv_features": [8, 64],
            "embedding_dims": [128, 128, 128],
            "actor_hidden": 128,
            "critic_hidden": [128, 128]
        }
    },
    "conv-ablate-25-2": {
        "LISTENER_ARCH_ABLATION_PARAMS": {
            "conv_features": [32, 16],
            "embedding_dims": [128, 128, 128],
            "actor_hidden": 128,
            "critic_hidden": [128, 128]
        }
    },
    "conv-ablate-25-3": {
        "LISTENER_ARCH_ABLATION_PARAMS": {
            "conv_features": [32, 64],
            "embedding_dims": [32, 128, 128],
            "actor_hidden": 128,
            "critic_hidden": [128, 128]
        }
    },
    "conv-ablate-25-4": {
        "LISTENER_ARCH_ABLATION_PARAMS": {
            "conv_features": [32, 64],
            "embedding_dims": [128, 32, 128],
            "actor_hidden": 128,
            "critic_hidden": [128, 128]
        }
    },
    "conv-ablate-25-5": {
        "LISTENER_ARCH_ABLATION_PARAMS": {
            "conv_features": [32, 64],
            "embedding_dims": [128, 128, 32],
            "actor_hidden": 128,
            "critic_hidden": [128, 128]
        }
    },
    "conv-ablate-25-6": {
        "LISTENER_ARCH_ABLATION_PARAMS": {
            "conv_features": [32, 64],
            "embedding_dims": [128, 128, 128],
            "actor_hidden": 32,
            "critic_hidden": [128, 128]
        }
    },
}

# This is a copy of ActorCriticSpeakerSplines from agents.py, placed here as a reference
class __ActorCriticSpeakerSplines(nn.Module):
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

class ActorCriticSpeakerSplinesAblationReady(nn.Module):
    num_classes: int
    action_dim: Sequence[int]
    config: Dict

    @nn.compact
    def __call__(self, obs):
        # Get architecture from config with defaults
        arch = self.config.get("SPEAKER_ARCH_ABLATION_PARAMS", {})
        embedding_latent_dim = arch.get("embedding_latent_dim", 128)
        embedding_dims = arch.get("embedding_dims", [128, 128, 128])

        y = nn.Embed(self.num_classes, embedding_latent_dim)(obs)
        z = y
        for i, dim in enumerate(embedding_dims):
            z = nn.Dense(dim, kernel_init=nn.initializers.he_uniform())(z)
            z = nn.relu(z)

        # Actor Mean
        actor_mean = nn.Dense(self.action_dim, kernel_init=nn.initializers.normal(self.config["SPEAKER_STDDEV"]))(z)
        actor_mean = nn.sigmoid(actor_mean)  # Apply sigmoid to squash outputs between 0 and 1

        scale_diag = nn.Dense(self.action_dim, kernel_init=nn.initializers.normal(self.config["SPEAKER_STDDEV2"]))(z)
        scale_diag = nn.sigmoid(scale_diag) * self.config["SPEAKER_SQUISH"] + 1e-8
        
        # Create a multivariate normal distribution with diagonal covariance matrix
        pi = distrax.MultivariateNormalDiag(loc=actor_mean, scale_diag=scale_diag)

        # Critic
        critic = nn.Dense(128)(z)
        critic = nn.sigmoid(critic)
        critic = nn.Dense(128)(critic)
        critic = nn.sigmoid(critic)
        critic = nn.Dense(32)(critic)
        critic = nn.sigmoid(critic)
        critic = nn.Dense(1)(critic)

        return pi, jnp.squeeze(critic, axis=-1) # Return policy function and value

SPEAKER_ARCH_ABLATION_PARAMETERS = {
    "splines-ablate-0": {
        "SPEAKER_ARCH_ABLATION_PARAMETERS": {
            "embedding_latent_dim": 128,
            "embedding_dims": [128, 128, 128]
        }
    },
    "splines-ablate-50-1": {
        "SPEAKER_ARCH_ABLATION_PARAMETERS": {
            "embedding_latent_dim": 64,
            "embedding_dims": [128, 128, 128]
        }
    },
    "splines-ablate-50-2": {
        "SPEAKER_ARCH_ABLATION_PARAMETERS": {
            "embedding_latent_dim": 128,
            "embedding_dims": [64, 128, 128]
        }
    },
    "splines-ablate-50-3": {
        "SPEAKER_ARCH_ABLATION_PARAMETERS": {
            "embedding_latent_dim": 128,
            "embedding_dims": [128, 64, 128]
        }
    },
    "splines-ablate-50-4": {
        "SPEAKER_ARCH_ABLATION_PARAMETERS": {
            "embedding_latent_dim": 128,
            "embedding_dims": [128, 128, 64]
        }
    },
    "splines-ablate-25-1": {
        "SPEAKER_ARCH_ABLATION_PARAMETERS": {
            "embedding_latent_dim": 32,
            "embedding_dims": [128, 128, 128]
        }
    },
    "splines-ablate-25-2": {
        "SPEAKER_ARCH_ABLATION_PARAMETERS": {
            "embedding_latent_dim": 128,
            "embedding_dims": [32, 128, 128]
        }
    },
    "splines-ablate-25-3": {
        "SPEAKER_ARCH_ABLATION_PARAMETERS": {
            "embedding_latent_dim": 128,
            "embedding_dims": [128, 32, 128]
        }
    },
    "splines-ablate-25-4": {
        "SPEAKER_ARCH_ABLATION_PARAMETERS": {
            "embedding_latent_dim": 128,
            "embedding_dims": [128, 128, 32]
        }
    },
}
