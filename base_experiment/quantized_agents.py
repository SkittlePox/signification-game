from functools import partial
import chex
import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
from flax.linen.initializers import constant, orthogonal
import distrax
from typing import Sequence, Tuple, Dict

class VectorQuantizer(nn.Module):
    """Vector Quantization layer."""
    
    @nn.compact
    def __call__(self, inputs, num_embeddings: int = 64, embedding_dim: int = 64, 
                 commitment_cost: float = 0.25):
        # inputs shape: (batch, height, width, channels)
        
        embedding = self.param(
            'embedding',
            nn.initializers.variance_scaling(
                scale=1.0, mode='fan_in', distribution='uniform'
            ),
            (num_embeddings, embedding_dim)
        )
        
        # Flatten spatial dimensions
        input_shape = inputs.shape
        flat_inputs = jnp.reshape(inputs, (-1, embedding_dim))
        
        # Calculate distances to codebook entries
        distances = (
            jnp.sum(flat_inputs**2, axis=1, keepdims=True)
            - 2 * jnp.matmul(flat_inputs, embedding.T)
            + jnp.sum(embedding**2, axis=1, keepdims=True).T
        )
        
        # Find nearest codebook entry
        encoding_indices = jnp.argmin(distances, axis=1)
        encodings = jax.nn.one_hot(encoding_indices, num_embeddings)
        
        # Quantize
        quantized = jnp.matmul(encodings, embedding)
        quantized = jnp.reshape(quantized, input_shape)
        
        # Straight-through estimator for gradient flow
        quantized = inputs + jax.lax.stop_gradient(quantized - inputs)
        
        # Calculate VQ loss (only used during training)
        e_latent_loss = jnp.mean((jax.lax.stop_gradient(quantized) - inputs) ** 2)
        q_latent_loss = jnp.mean((quantized - jax.lax.stop_gradient(inputs)) ** 2)
        vq_loss = q_latent_loss + commitment_cost * e_latent_loss

        # if not deterministic:
        #     e_latent_loss = jnp.mean((jax.lax.stop_gradient(quantized) - inputs) ** 2)
        #     q_latent_loss = jnp.mean((quantized - jax.lax.stop_gradient(inputs)) ** 2)
        #     vq_loss = q_latent_loss + commitment_cost * e_latent_loss
        # else:
        #     vq_loss = 0.0
        
        return quantized, vq_loss, encoding_indices

class ActorCriticListenerConvAblationQuantizeReady(nn.Module):
    action_dim: Sequence[int]
    image_dim: Sequence[int]
    config: Dict

    @nn.compact
    def __call__(self, x):
        # Get architecture from config with defaults
        arch = self.config.get("LISTENER_ARCH_ABLATION_QUANTIZATION_PARAMETERS", {})
        conv_features = arch.get("conv_features", [32, 64])
        embedding_dims = arch.get("embedding_dims", [128, 128, 128])
        actor_hidden = arch.get("actor_hidden", 128)
        critic_hidden = arch.get("critic_hidden", [128, 128])
        conv_strides = arch.get("conv_strides", [1, 1])
        
        # VQ parameters from config
        use_vq = arch.get("use_vq", False)
        vq_num_embeddings = arch.get("vq_num_embeddings", 512)
        vq_embedding_dim = arch.get("vq_embedding_dim", 64)
        vq_commitment_cost = arch.get("vq_commitment_cost", 0.25)
        vq_after_layer = arch.get("vq_after_layer", -1)  # -1 means after all conv layers, 0 means after first, etc.
        
        x = x.reshape(-1, self.image_dim, self.image_dim, 1)

        # Conv layers with VQ insertion
        vq_loss = 0.0
        encoding_indices = None
        for i, features in enumerate(conv_features):
            x = nn.Conv(features=features, kernel_size=(3, 3), strides=(conv_strides[i], conv_strides[i]), padding='SAME')(x)
            x = nn.relu(x)
            
            # Insert VQ after specified layer
            if use_vq and i == vq_after_layer:
                # Project to VQ embedding dimension if needed
                if x.shape[-1] != vq_embedding_dim:
                    x = nn.Dense(features=vq_embedding_dim)(x)
                
                x, vq_loss, encoding_indices = VectorQuantizer()(
                    x,
                    num_embeddings=vq_num_embeddings,
                    embedding_dim=vq_embedding_dim,
                    commitment_cost=vq_commitment_cost,
                )
        
        # VQ after all conv layers (if vq_after_layer == -1 or >= len(conv_features))
        if use_vq and (vq_after_layer == -1 or vq_after_layer >= len(conv_features)):
            # Project to VQ embedding dimension if needed
            if x.shape[-1] != vq_embedding_dim:
                x = nn.Dense(features=vq_embedding_dim)(x)
            
            x, vq_loss, encoding_indices = VectorQuantizer()(
                x,
                num_embeddings=vq_num_embeddings,
                embedding_dim=vq_embedding_dim,
                commitment_cost=vq_commitment_cost,
            )
        
        x = x.reshape((x.shape[0], -1))
        
        # Embedding
        embedding = x
        for i, dim in enumerate(embedding_dims):
            embedding = nn.Dense(dim)(embedding)
            embedding = nn.relu(embedding)

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

        return pi, jnp.squeeze(critic, axis=-1), vq_loss, encoding_indices


LISTENER_ARCH_ABLATION_QUANTIZATION_PARAMETERS = {
    "conv-quantize-test2": {
        "LISTENER_ARCH_ABLATION_QUANTIZATION_PARAMETERS": {
            "conv_features": [32, 64],
            "embedding_dims": [128, 128, 128],
            "actor_hidden": 128,
            "critic_hidden": [128, 128],
            "use_vq": True,
            "vq_after_layer": 1,  # After first conv layer (64 features)
            "vq_num_embeddings": 4,
            "vq_embedding_dim": 64,
            "conv_strides": [1, 1]
        }
    },
    "conv-quantize-stride": {
        "LISTENER_ARCH_ABLATION_QUANTIZATION_PARAMETERS": {
            "conv_features": [32, 32, 32],
            "embedding_dims": [128, 128, 128],
            "actor_hidden": 128,
            "critic_hidden": [128, 128],
            "use_vq": True,
            "vq_after_layer": 2,  # After first conv layer (64 features)
            "vq_num_embeddings": 4,
            "vq_embedding_dim": 64,
            "conv_strides": [1, 1, 1]
        }
    },
    # VQ after first conv layer (index 0)
    "conv-quantize-a": {
        "LISTENER_ARCH_ABLATION_QUANTIZATION_PARAMETERS": {
            "conv_features": [32, 64],
            "use_vq": True,
            "vq_after_layer": 0,  # After first conv layer (32 features)
            "vq_num_embeddings": 256,
            "vq_embedding_dim": 32,  # Match first conv output
        }
    },

    # VQ after second conv layer (index 1)
    "conv-quantize-b": {
        "LISTENER_ARCH_ABLATION_QUANTIZATION_PARAMETERS": {
            "conv_features": [32, 64],
            "use_vq": True,
            "vq_after_layer": 1,  # After second conv layer (64 features)
            "vq_num_embeddings": 256,
            "vq_embedding_dim": 64,
        }
    },

    # VQ after all conv layers (default)
    "conv-quantize-c": {
        "LISTENER_ARCH_ABLATION_QUANTIZATION_PARAMETERS": {
            "conv_features": [32, 64],
            "use_vq": True,
            "vq_after_layer": -1,  # After all conv layers
            "vq_num_embeddings": 256,
            "vq_embedding_dim": 64,
        }
    },

    # With 3 conv layers, VQ in the middle
    "conv-quantize-d": {
        "LISTENER_ARCH_ABLATION_QUANTIZATION_PARAMETERS": {
            "conv_features": [32, 64, 128],
            "use_vq": True,
            "vq_after_layer": 1,  # After middle layer
            "vq_num_embeddings": 512,
            "vq_embedding_dim": 64,
        }
}
}


class ActorCriticListenerConvQuantization(nn.Module):
    action_dim: Sequence[int]
    image_dim: Sequence[int]
    config: Dict

    @nn.compact
    def __call__(self, x):
        # Get architecture from config with defaults
        arch = self.config.get("LISTENER_ARCH_ABLATION_QUANTIZATION_PARAMETERS", {})
        conv_features = arch.get("conv_features", [32, 64])
        embedding_dims = arch.get("embedding_dims", [128, 128, 128])
        actor_hidden = arch.get("actor_hidden", 128)
        critic_hidden = arch.get("critic_hidden", [128, 128])
        conv_strides = arch.get("conv_strides", [1, 1])
        
        # VQ parameters from config
        use_vq = arch.get("use_vq", False)
        vq_num_embeddings = arch.get("vq_num_embeddings", 512)
        vq_embedding_dim = arch.get("vq_embedding_dim", 64)
        vq_commitment_cost = arch.get("vq_commitment_cost", 0.25)
        vq_after_layer = arch.get("vq_after_layer", -1)  # -1 means after all conv layers, 0 means after first, etc.
        
        x = x.reshape(-1, self.image_dim, self.image_dim, 1)

        # Conv layers with VQ insertion
        vq_loss = 0.0
        encoding_indices = None
        for i, features in enumerate(conv_features):
            x = nn.Conv(features=features, kernel_size=(3, 3), strides=(conv_strides[i], conv_strides[i]), padding='SAME')(x)
            x = nn.relu(x)
            
            # Insert VQ after specified layer
            if use_vq and i == vq_after_layer:
                # Project to VQ embedding dimension if needed
                if x.shape[-1] != vq_embedding_dim:
                    x = nn.Dense(features=vq_embedding_dim)(x)
                
                x, vq_loss, encoding_indices = VectorQuantizer()(
                    x,
                    num_embeddings=vq_num_embeddings,
                    embedding_dim=vq_embedding_dim,
                    commitment_cost=vq_commitment_cost,
                )
        
        # VQ after all conv layers (if vq_after_layer == -1 or >= len(conv_features))
        if use_vq and (vq_after_layer == -1 or vq_after_layer >= len(conv_features)):
            # Project to VQ embedding dimension if needed
            if x.shape[-1] != vq_embedding_dim:
                x = nn.Dense(features=vq_embedding_dim)(x)
            
            x, vq_loss, encoding_indices = VectorQuantizer()(
                x,
                num_embeddings=vq_num_embeddings,
                embedding_dim=vq_embedding_dim,
                commitment_cost=vq_commitment_cost,
            )
        
        x = x.reshape((x.shape[0], -1))
        
        # Embedding
        embedding = x
        for i, dim in enumerate(embedding_dims):
            embedding = nn.Dense(dim)(embedding)
            embedding = nn.relu(embedding)

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

        return pi, jnp.squeeze(critic, axis=-1), vq_loss, encoding_indices


class ActorCriticSpeakerDenseQuantized(nn.Module):
    action_dim: Sequence[int]
    image_dim: Sequence[int]
    config: Dict

    @nn.compact
    def __call__(self, x):
        # Get architecture from config with defaults
        arch = self.config.get("SPEAKER_ARCH_QUANTIZATION_PARAMETERS", {})
        embedding_dims = arch.get("embedding_dims", [128, 128, 128])
        actor_dims = arch.get("actor_dims", [128])
        critic_dims = arch.get("critic_dims", [128, 128])

        # VQ parameters from config
        vq_num_embeddings = arch.get("vq_num_embeddings", 512)
        vq_embedding_dim = arch.get("vq_embedding_dim", 64)
        vq_commitment_cost = arch.get("vq_commitment_cost", 0.25)
        
        x = x.reshape((-1, self.image_dim**2))
        
        # Embedding
        embedding = x
        for dim in embedding_dims:
            embedding = nn.Dense(dim)(embedding)
            embedding = nn.relu(embedding)

        # Quantization
        # Project to VQ embedding dimension if needed
        if x.shape[-1] != vq_embedding_dim:
            x = nn.Dense(features=vq_embedding_dim)(x)
        
        x, vq_loss, encoding_indices = VectorQuantizer()(
            x,
            num_embeddings=vq_num_embeddings,
            embedding_dim=vq_embedding_dim,
            commitment_cost=vq_commitment_cost,
        )

        # Actor
        actor_mean = embedding
        for dim in actor_dims:
            actor_mean = nn.Dense(dim)(actor_mean)
            actor_mean = nn.relu(actor_mean)

        # squeeze to action space
        actor_mean = nn.Dense(self.action_dim)(actor_mean)
        actor_mean = nn.softmax(actor_mean)
        pi = distrax.Categorical(probs=actor_mean)

        # Critic
        critic = embedding
        for dim in critic_dims:
            critic = nn.Dense(dim)(critic)
            critic = nn.relu(critic)
        critic = nn.Dense(1)(critic)

        return pi, jnp.squeeze(critic, axis=-1), vq_loss, encoding_indices

SPEAKER_ARCH_QUANTIZATION_PARAMETERS = {
    "speaker-quantize-0": {
        "SPEAKER_ARCH_QUANTIZATION_PARAMETERS": {
            "embedding_dims": [64, 64, 64],
            "actor_dims": [64],
            "critic_dims": [64, 64],
            "vq_num_embeddings": 4,
            "vq_embedding_dim": 64,
            "vq_commitment_cost": 0.25
        }
    },
}
