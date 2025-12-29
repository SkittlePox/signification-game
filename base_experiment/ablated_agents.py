import jax.numpy as jnp
import flax.linen as nn
import jax
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
        conv_kernels = arch.get("conv_kernels", [])
        conv_strides = arch.get("conv_strides", [])
        conv_padding = arch.get("conv_padding", 'SAME')
        embedding_dims = arch.get("embedding_dims", [128, 128, 128])
        actor_hidden = arch.get("actor_hidden", 128)
        critic_hidden = arch.get("critic_hidden", [128, 128])
        
        if conv_kernels == []:  # Default kernel size is 3
            conv_kernels = [3] * len(conv_features)
        
        if conv_strides == []:  # Default stride is 1
            conv_strides = [1] * len(conv_features)
        
        x = x.reshape(-1, self.image_dim, self.image_dim, 1)

        # Conv layers
        for features, kernel, stride in zip(conv_features, conv_kernels, conv_strides):
            x = nn.Conv(features=features, kernel_size=(kernel, kernel), strides=(stride, stride), padding=conv_padding)(x)
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
    "conv-ablate-50-0": {
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
    "conv-ablate-25-0": {
        "LISTENER_ARCH_ABLATION_PARAMS": {
            "conv_features": [32, 64],
            "embedding_dims": [128, 128, 128],
            "actor_hidden": 128,
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
    "conv-ablate-micro-A-0": {
        "LISTENER_ARCH_ABLATION_PARAMS": {
            "conv_features": [2, 2, 2],
            "conv_kernels": [3, 5, 3],
            "conv_strides": [1, 5, 3],
            "embedding_dims": [64, 64, 64],
            "actor_hidden": 32,
            "critic_hidden": [64, 64]
        }
    },
    "conv-ablate-micro-A-1": {
        "LISTENER_ARCH_ABLATION_PARAMS": {
            "conv_features": [4, 4, 4],
            "conv_kernels": [3, 5, 3],
            "conv_strides": [1, 5, 3],
            "embedding_dims": [64, 64, 64],
            "actor_hidden": 32,
            "critic_hidden": [64, 64]
        }
    },
    "conv-ablate-micro-A-2": {
        "LISTENER_ARCH_ABLATION_PARAMS": {
            "conv_features": [8, 8, 8],
            "conv_kernels": [3, 5, 3],
            "conv_strides": [1, 5, 3],
            "embedding_dims": [64, 64, 64],
            "actor_hidden": 32,
            "critic_hidden": [64, 64]
        }
    },
    "conv-ablate-micro-A-3": {
        "LISTENER_ARCH_ABLATION_PARAMS": {
            "conv_features": [16, 16, 16],
            "conv_kernels": [3, 5, 3],
            "conv_strides": [1, 5, 3],
            "embedding_dims": [64, 64, 64],
            "actor_hidden": 32,
            "critic_hidden": [64, 64]
        }
    },
    "conv-ablate-micro-B-0": {
        "LISTENER_ARCH_ABLATION_PARAMS": {
            "conv_features": [2, 2, 2],
            "conv_kernels": [5, 4, 3],
            "conv_strides": [1, 4, 2],
            "embedding_dims": [64, 64, 64],
            "actor_hidden": 32,
            "critic_hidden": [64, 64]
        }
    },
    "conv-ablate-micro-B-1": {
        "LISTENER_ARCH_ABLATION_PARAMS": {
            "conv_features": [4, 4, 4],
            "conv_kernels": [5, 4, 3],
            "conv_strides": [1, 4, 2],
            "embedding_dims": [64, 64, 64],
            "actor_hidden": 32,
            "critic_hidden": [64, 64]
        }
    },
    "conv-ablate-micro-B-2": {
        "LISTENER_ARCH_ABLATION_PARAMS": {
            "conv_features": [8, 8, 8],
            "conv_kernels": [5, 4, 3],
            "conv_strides": [1, 4, 2],
            "embedding_dims": [64, 64, 64],
            "actor_hidden": 32,
            "critic_hidden": [64, 64]
        }
    },
    "conv-ablate-micro-B-3": {
        "LISTENER_ARCH_ABLATION_PARAMS": {
            "conv_features": [16, 16, 16],
            "conv_kernels": [5, 4, 3],
            "conv_strides": [1, 4, 2],
            "embedding_dims": [64, 64, 64],
            "actor_hidden": 32,
            "critic_hidden": [64, 64]
        }
    },
    "conv-ablate-micro-C-0": {
        "LISTENER_ARCH_ABLATION_PARAMS": {
            "conv_features": [16, 16, 8],
            "conv_kernels": [3, 3, 5],
            "conv_strides": [1, 3, 5],
            "embedding_dims": [8],
            "actor_hidden": 8,
            "critic_hidden": [8, 8]
        }
    },
    "conv-ablate-micro-C-1": {
        "LISTENER_ARCH_ABLATION_PARAMS": {
            "conv_features": [32, 16, 16],
            "conv_kernels": [3, 3, 5],
            "conv_strides": [1, 3, 5],
            "embedding_dims": [8],
            "actor_hidden": 8,
            "critic_hidden": [8, 8]
        }
    },
}

class ActorCriticListenerConvSkipPoolReady(nn.Module):
    action_dim: Sequence[int]
    image_dim: Sequence[int]
    config: Dict

    @nn.compact
    def __call__(self, x):
        # Get architecture from config with defaults
        arch = self.config.get("LISTENER_ARCH_SKIPPOOL_PARAMETERS", {})
        conv_features = arch.get("conv_features", [24, 16, 8])
        conv_kernels = arch.get("conv_kernels", [3, 7, 11])
        conv_strides = arch.get("conv_strides", [])
        embedding_dims = arch.get("embedding_dims", [128, 128, 128])
        actor_hidden = arch.get("actor_hidden", 128)
        critic_hidden = arch.get("critic_hidden", [128, 128])
        
        if conv_kernels == []:  # Default kernel size is 3
            conv_kernels = [3] * len(conv_features)
        
        if conv_strides == []:  # Default stride is 1
            conv_strides = [1] * len(conv_features)
        
        x = x.reshape(-1, self.image_dim, self.image_dim, 1)

        # Fine scale
        x1 = nn.Conv(features=conv_features[0], kernel_size=(conv_kernels[0], conv_kernels[0]), strides=(conv_strides[0], conv_strides[0]))(x)
        x1 = nn.relu(x1)
        x1 = jnp.mean(x1, axis=(1, 2))
        
        # Medium scale - larger kernel
        x2 = nn.Conv(features=conv_features[1], kernel_size=(conv_kernels[1], conv_kernels[1]), strides=(conv_strides[1], conv_strides[1]))(x)
        x2 = nn.relu(x2)
        x2 = jnp.mean(x2, axis=(1, 2))
        
        # Coarse scale - even larger kernel
        x3 = nn.Conv(features=conv_features[2], kernel_size=(conv_kernels[2], conv_kernels[2]), strides=(conv_strides[2], conv_strides[2]))(x)
        x3 = nn.relu(x3)
        x3 = jnp.mean(x3, axis=(1, 2))
        
        # Concatenate multi-scale features
        x = jnp.concatenate([x1, x2, x3], axis=-1)     
        
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

LISTENER_ARCH_SKIPPOOL_PARAMETERS = {
    "conv-skippool-0": {
        "LISTENER_ARCH_SKIPPOOL_PARAMETERS": {
            "conv_features": [24, 16, 8],
            "conv_kernels": [3, 7, 11],
            "embedding_dims": [8],
            "actor_hidden": 8,
            "critic_hidden": [8, 8]
        }
    },
    "conv-skippool-1": {
        "LISTENER_ARCH_SKIPPOOL_PARAMETERS": {
            "conv_features": [8, 8, 8],
            "conv_kernels": [3, 7, 11],
            "embedding_dims": [8],
            "actor_hidden": 8,
            "critic_hidden": [8, 8]
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
    "splines-ablate-50-0": {
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
    "splines-ablate-25-0": {
        "SPEAKER_ARCH_ABLATION_PARAMETERS": {
            "embedding_latent_dim": 128,
            "embedding_dims": [128, 128, 128]
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
