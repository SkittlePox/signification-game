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
    num_embeddings: int
    embedding_dim: int
    commitment_cost: float
    straight_through_estimator: bool = True
    
    @nn.compact
    def __call__(self, inputs):
        # inputs shape: (batch, height, width, channels)
        
        embedding = self.param(
            'embedding',
            nn.initializers.variance_scaling(
                scale=1.0, mode='fan_in', distribution='uniform'
            ),
            (self.num_embeddings, self.embedding_dim)
        )
        
        # Flatten spatial dimensions
        input_shape = inputs.shape
        flat_inputs = jnp.reshape(inputs, (-1, self.embedding_dim))
        
        # Calculate distances to codebook entries
        distances = (
            jnp.sum(flat_inputs**2, axis=1, keepdims=True)
            - 2 * jnp.matmul(flat_inputs, embedding.T)
            + jnp.sum(embedding**2, axis=1, keepdims=True).T
        )
        
        # Find nearest codebook entry
        encoding_indices = jnp.argmin(distances, axis=1)
        encodings = jax.nn.one_hot(encoding_indices, self.num_embeddings)
        
        # Quantize
        quantized = jnp.matmul(encodings, embedding)
        quantized = jnp.reshape(quantized, input_shape)

        if self.straight_through_estimator:
            # Straight-through estimator for gradient flow
            quantized = inputs + jax.lax.stop_gradient(quantized - inputs)
            # Encoder gets gradients as if no quantization happened
        else:
            # Without STE
            quantized = jax.lax.stop_gradient(quantized)
            # Encoder only gets gradients from commitment loss
        
        # Calculate VQ loss (only used during training)
        e_latent_loss = jnp.mean((jax.lax.stop_gradient(quantized) - inputs) ** 2)
        q_latent_loss = jnp.mean((quantized - jax.lax.stop_gradient(inputs)) ** 2)
        vq_loss = q_latent_loss + self.commitment_cost * e_latent_loss

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
                
                x, vq_loss, encoding_indices = VectorQuantizer(
                    num_embeddings=vq_num_embeddings,
                    embedding_dim=vq_embedding_dim,
                    commitment_cost=vq_commitment_cost,
                )(x)
        
        # VQ after all conv layers (if vq_after_layer == -1 or >= len(conv_features))
        if use_vq and (vq_after_layer == -1 or vq_after_layer >= len(conv_features)):
            # Project to VQ embedding dimension if needed
            if x.shape[-1] != vq_embedding_dim:
                x = nn.Dense(features=vq_embedding_dim)(x)
            
            x, vq_loss, encoding_indices = VectorQuantizer(
                num_embeddings=vq_num_embeddings,
                embedding_dim=vq_embedding_dim,
                commitment_cost=vq_commitment_cost,
            )(x)
        
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
                
                x, vq_loss, encoding_indices = VectorQuantizer(
                    num_embeddings=vq_num_embeddings,
                    embedding_dim=vq_embedding_dim,
                    commitment_cost=vq_commitment_cost,
                )(x)
        
        # VQ after all conv layers (if vq_after_layer == -1 or >= len(conv_features))
        if use_vq and (vq_after_layer == -1 or vq_after_layer >= len(conv_features)):
            # Project to VQ embedding dimension if needed
            if x.shape[-1] != vq_embedding_dim:
                x = nn.Dense(features=vq_embedding_dim)(x)
            
            x, vq_loss, encoding_indices = VectorQuantizer(
                num_embeddings=vq_num_embeddings,
                embedding_dim=vq_embedding_dim,
                commitment_cost=vq_commitment_cost,
            )(x)
        
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
    num_classes: int
    action_dim: Sequence[int]
    config: Dict

    @nn.compact
    def __call__(self, obs):
        # Get architecture from config with defaults
        arch = self.config.get("SPEAKER_ARCH_ABLATION_PARAMS", {})
        embedding_latent_dim = arch.get("embedding_latent_dim", 128)
        embedding_dims = arch.get("embedding_dims", [128, 128, 128])
        critic_dims = arch.get("critic_dims", [128, 128, 32])

        # VQ parameters from config
        vq_num_embeddings = arch.get("vq_num_embeddings", 512)
        vq_embedding_dim = arch.get("vq_embedding_dim", 64)
        vq_commitment_cost = arch.get("vq_commitment_cost", 0.25)

        y = nn.Embed(self.num_classes, embedding_latent_dim)(obs)
        z = y
        for i, dim in enumerate(embedding_dims):
            z = nn.Dense(dim, kernel_init=nn.initializers.he_uniform())(z)
            z = nn.relu(z)

        # Quantization
        # Project to VQ embedding dimension if needed
        if z.shape[-1] != vq_embedding_dim:
            z = nn.Dense(features=vq_embedding_dim)(z)
        
        z, vq_loss, encoding_indices = VectorQuantizer(
            num_embeddings=vq_num_embeddings,
            embedding_dim=vq_embedding_dim,
            commitment_cost=vq_commitment_cost,)(z)

        # Actor Mean
        actor_mean = nn.Dense(self.action_dim, kernel_init=nn.initializers.normal(self.config["SPEAKER_STDDEV"]))(z)
        actor_mean = nn.sigmoid(actor_mean)  # Apply sigmoid to squash outputs between 0 and 1

        scale_diag = nn.Dense(self.action_dim, kernel_init=nn.initializers.normal(self.config["SPEAKER_STDDEV2"]))(z)
        scale_diag = nn.sigmoid(scale_diag) * self.config["SPEAKER_SQUISH"] + 1e-8
        
        # Create a multivariate normal distribution with diagonal covariance matrix
        pi = distrax.MultivariateNormalDiag(loc=actor_mean, scale_diag=scale_diag)

        # Critic
        critic = z
        for i, dim in enumerate(critic_dims):
            critic = nn.Dense(dim, kernel_init=nn.initializers.he_uniform())(critic)
            critic = nn.sigmoid(critic)

        critic = nn.Dense(1)(critic)

        return pi, jnp.squeeze(critic, axis=-1), vq_loss, encoding_indices


SPEAKER_ARCH_QUANTIZATION_PARAMETERS = {
    "splines-quantized-A9-0": {
        "SPEAKER_ARCH_ABLATION_PARAMETERS": {
            "embedding_latent_dim": 8,
            "embedding_dims": [4, 4],
            "critic_dims": [8, 8],
            "vq_num_embeddings": 16,
            "vq_embedding_dim": 16,
            "vq_commitment_cost": 0.25
        }
    },
    "splines-quantized-A9-1": {
        "SPEAKER_ARCH_ABLATION_PARAMETERS": {
            "embedding_latent_dim": 8,
            "embedding_dims": [4, 4],
            "critic_dims": [8, 8],
            "vq_num_embeddings": 8,
            "vq_embedding_dim": 16,
            "vq_commitment_cost": 0.25
        }
    },
    "splines-quantized-A9-2": {
        "SPEAKER_ARCH_ABLATION_PARAMETERS": {
            "embedding_latent_dim": 8,
            "embedding_dims": [4, 4],
            "critic_dims": [8, 8],
            "vq_num_embeddings": 4,
            "vq_embedding_dim": 16,
            "vq_commitment_cost": 0.25
        }
    },
    "splines-quantized-A9-3": {
        "SPEAKER_ARCH_ABLATION_PARAMETERS": {
            "embedding_latent_dim": 8,
            "embedding_dims": [4, 4],
            "critic_dims": [8, 8],
            "vq_num_embeddings": 64,
            "vq_embedding_dim": 8,
            "vq_commitment_cost": 0.25
        }
    },
    "splines-quantized-A9-4": {
        "SPEAKER_ARCH_ABLATION_PARAMETERS": {
            "embedding_latent_dim": 8,
            "embedding_dims": [4, 4],
            "critic_dims": [8, 8],
            "vq_num_embeddings": 128,
            "vq_embedding_dim": 8,
            "vq_commitment_cost": 0.25
        }
    },
}


class ActorCriticSpeakerDensePerSplineQuantized(nn.Module):
    num_classes: int
    num_splines: int
    spline_action_dim: int
    config: Dict

    @nn.compact
    def __call__(self, obs):
        # Get architecture from config with defaults
        arch = self.config.get("SPEAKER_ARCH_PERSPLINE_QUANTIZATION_PARAMETERS", {})
        embedding_latent_dim = arch.get("embedding_latent_dim", 128)
        embedding_dims = arch.get("embedding_dims", [128, 128, 128])
        critic_dims = arch.get("critic_dims", [128, 128, 32])

        # VQ parameters from config
        vq_num_embeddings = arch.get("vq_num_embeddings", 512)
        vq_embedding_dim = arch.get("vq_embedding_dim", 64)
        vq_commitment_cost = arch.get("vq_commitment_cost", 0.25)
        use_vq = arch.get("use_vq", False)

        y = nn.Embed(self.num_classes, embedding_latent_dim)(obs)
        z = y
        for i, dim in enumerate(embedding_dims):
            z = nn.Dense(dim, kernel_init=nn.initializers.he_uniform())(z)
            z = nn.relu(z)

        # Create the vq layer and dense adapter if needed
        vq_layer = VectorQuantizer(num_embeddings=vq_num_embeddings, embedding_dim=vq_embedding_dim, commitment_cost=vq_commitment_cost,)
        vq_adapter = nn.Dense(features=vq_embedding_dim)

        # Get action parameters spline by spline and quantize
        def decode_spline(_x):
            actor_mean_i = nn.Dense(self.spline_action_dim, kernel_init=nn.initializers.normal(self.config["SPEAKER_STDDEV"]))(_x)
            actor_mean_i = nn.sigmoid(actor_mean_i)  # Apply sigmoid to squash outputs between 0 and 1

            scale_diag_i = nn.Dense(self.spline_action_dim, kernel_init=nn.initializers.normal(self.config["SPEAKER_STDDEV2"]))(_x)
            scale_diag_i = nn.sigmoid(scale_diag_i) * self.config["SPEAKER_SQUISH"] + 1e-8

            return (actor_mean_i, scale_diag_i)

        actor_means = []
        actor_scale_diags = []

        # NOTE: I'm pretty sure this is basically useless since it's decoding the same value for each spline. This was a practice agent
        for _ in range(self.num_splines):
            actor_mean_i, scale_diag_i = decode_spline(z)
            actor_means.append(actor_mean_i)
            actor_scale_diags.append(scale_diag_i)
        
        actor_mean = jnp.concatenate(actor_means, axis=1)
        actor_scale_diag = jnp.concatenate(actor_scale_diags, axis=1)
        
        # Create a multivariate normal distribution with diagonal covariance matrix
        pi = distrax.MultivariateNormalDiag(loc=actor_mean, scale_diag=actor_scale_diag)

        # Critic
        critic = z
        for i, dim in enumerate(critic_dims):
            critic = nn.Dense(dim, kernel_init=nn.initializers.he_uniform())(critic)
            critic = nn.sigmoid(critic)

        critic = nn.Dense(1)(critic)

        return pi, jnp.squeeze(critic, axis=-1)

SPEAKER_ARCH_PERSPLINE_QUANTIZATION_PARAMETERS = {
    "splines-perspline-quantized-A9-0": {
        "SPEAKER_ARCH_PERSPLINE_QUANTIZATION_PARAMETERS": {
            "embedding_latent_dim": 8,
            "embedding_dims": [4, 4],
            "critic_dims": [8, 8],
            "vq_num_embeddings": 16,
            "vq_embedding_dim": 16,
            "vq_commitment_cost": 0.25,
            "use_vq": False
        }
    },
}


class ActorCriticSpeakerRNNQuantized(nn.Module):
    num_classes: int
    num_splines: int
    spline_action_dim: int
    config: Dict

    @nn.compact
    def __call__(self, obs):
        # Get architecture from config with defaults
        arch = self.config.get("SPEAKER_ARCH_RNN_QUANTIZATION_PARAMETERS", {})
        embedding_latent_dim = arch.get("embedding_latent_dim", 128)
        embedding_dims = arch.get("embedding_dims", [128, 128, 128])
        critic_dims = arch.get("critic_dims", [128, 128, 32])
        rnn_hidden_dim = arch.get("rnn_hidden_dim", 16)
        use_pos_encs = arch.get("use_pos_encs", False)
        use_pos_embs = arch.get("use_pos_embs", False)
        pos_emb_latent_dim = arch.get("pos_emb_latent_dim", 8)  # make sure that this + the last embedding_dim sums to the rnn_hidden_dim!

        # VQ parameters from config
        vq_num_embeddings = arch.get("vq_num_embeddings", 64)
        vq_embedding_dim = arch.get("vq_embedding_dim", 64)
        vq_commitment_cost = arch.get("vq_commitment_cost", 0.25)
        use_vq = arch.get("use_vq", False)
        straight_through_estimator = arch.get("straight_through_estimator", True)

        # Other parameters from config
        use_tanh = self.config.get("SPEAKER_USE_TANH", False)
        use_tanh_last = self.config.get("SPEAKER_USE_TANH_LAST", False)
        sigmoid_temp = self.config.get("SPEAKER_SIGMOID_TEMP", 1.0)

        y = nn.Embed(self.num_classes, embedding_latent_dim)(obs)
        z = y
        for i, dim in enumerate(embedding_dims):
            z = nn.Dense(dim, kernel_init=nn.initializers.he_uniform())(z)
            if use_tanh:
                z = nn.tanh(z)
            else:
                z = nn.relu(z)

        # Create the SimpleCell
        cell = nn.SimpleCell(features=rnn_hidden_dim)
        actor_mean_dense = nn.Dense(self.spline_action_dim, kernel_init=nn.initializers.normal(self.config["SPEAKER_STDDEV"]))
        actor_scale_diag_dense = nn.Dense(self.spline_action_dim, kernel_init=nn.initializers.normal(self.config["SPEAKER_STDDEV2"]))

        # Create the vq layer and dense adapter if needed
        vq_layer = VectorQuantizer(num_embeddings=vq_num_embeddings, embedding_dim=vq_embedding_dim, commitment_cost=vq_commitment_cost, straight_through_estimator=straight_through_estimator)
        vq_adapter = nn.Dense(features=vq_embedding_dim)

        # Create a positional encoding dense adapter if needed
        pos_enc_adapter = nn.Dense(features=rnn_hidden_dim)

        # Create embeddings for positional encodings if needed
        pos_embeddings = nn.Embed(6, pos_emb_latent_dim)    # NOTE: Assuming we'll never use more than 6 splines for single sign

        # Get action parameters spline by spline
        def decode_spline(_x):
            actor_mean_i = actor_mean_dense(_x)
            scale_diag_i = actor_scale_diag_dense(_x)

            if use_tanh_last:
                actor_mean_i = (nn.tanh(actor_mean_i) + 1) / 2  # Apply sigmoid to squash outputs between 0 and 1
                scale_diag_i = (nn.tanh(scale_diag_i / sigmoid_temp) + 1) / 2 * self.config["SPEAKER_SQUISH"] + 1e-8
            else:
                actor_mean_i = nn.sigmoid(actor_mean_i / sigmoid_temp)  # Apply sigmoid to squash outputs between 0 and 1
                scale_diag_i = nn.sigmoid(scale_diag_i / sigmoid_temp) * self.config["SPEAKER_SQUISH"] + 1e-8

            return (actor_mean_i, scale_diag_i)

        initial_carry = nn.Dense(rnn_hidden_dim, name='carry_init')(z)

        carry = initial_carry
        actor_means = []
        actor_scale_diags = []
        total_vq_loss = 0.0
        encoding_indices = []

        for i in range(self.num_splines):
            ### Decide whether to mess with the carry

            # Option A: Use position encoding as input
            if use_pos_encs:
                position = jnp.full((z.shape[0], 1), i / self.num_splines)
                inputs = jnp.concatenate([z, position], axis=-1)
                inputs = pos_enc_adapter(inputs)
            
            # Option B: Use positional *embeddings*
            elif use_pos_embs:
                position = pos_embeddings(jnp.ones((z.shape[0]), dtype=jnp.int32) * i)
                inputs = jnp.concatenate([z, position], axis=-1)
            
            # Option C: just use z
            else:
                inputs = z
            
            carry, _ = cell(carry, inputs)

            ### Optional Quantization
            if use_vq:
                # Project to VQ embedding dimension if needed
                if carry.shape[-1] != vq_embedding_dim:
                    carry = vq_adapter(carry)
                
                carry, vq_i_loss, encoding_i_indices = vq_layer(carry)
                total_vq_loss += vq_i_loss
                encoding_indices.append(encoding_i_indices)

            actor_mean_i, scale_diag_i = decode_spline(carry)
            actor_means.append(actor_mean_i)
            actor_scale_diags.append(scale_diag_i)
        
        actor_mean = jnp.concatenate(actor_means, axis=1)
        actor_scale_diag = jnp.concatenate(actor_scale_diags, axis=1)
        
        # Create a multivariate normal distribution with diagonal covariance matrix
        pi = distrax.MultivariateNormalDiag(loc=actor_mean, scale_diag=actor_scale_diag)

        # Critic
        critic = z
        for i, dim in enumerate(critic_dims):
            critic = nn.Dense(dim, kernel_init=nn.initializers.he_uniform())(critic)
            critic = nn.tanh(critic)    # Note: I changed this from sigmoid on 1/17

        critic = nn.Dense(1)(critic)

        return pi, jnp.squeeze(critic, axis=-1), total_vq_loss, encoding_indices

SPEAKER_ARCH_RNN_QUANTIZATION_PARAMETERS = {
    "splines-rnn-quantized-A9-0": {
        "SPEAKER_ARCH_RNN_QUANTIZATION_PARAMETERS": {
            "embedding_latent_dim": 8,
            "embedding_dims": [4, 4],
            "critic_dims": [8, 8],
            "rnn_hidden_dim": 16,
            "vq_num_embeddings": 16,
            "vq_embedding_dim": 16,
            "vq_commitment_cost": 0.25,
            "use_vq": False
        }
    },
    "splines-rnn-quantized-A9-1": {
        "SPEAKER_ARCH_RNN_QUANTIZATION_PARAMETERS": {
            "embedding_latent_dim": 8,
            "embedding_dims": [4, 4],
            "critic_dims": [8, 8],
            "rnn_hidden_dim": 8,
            "vq_num_embeddings": 16,
            "vq_embedding_dim": 16,
            "vq_commitment_cost": 0.25,
            "use_vq": False
        }
    },
    "splines-rnn-quantized-A9-2": {
        "SPEAKER_ARCH_RNN_QUANTIZATION_PARAMETERS": {
            "embedding_latent_dim": 8,
            "embedding_dims": [4, 4],
            "critic_dims": [8, 8],
            "rnn_hidden_dim": 4,
            "vq_num_embeddings": 16,
            "vq_embedding_dim": 16,
            "vq_commitment_cost": 0.25,
            "use_vq": False
        }
    },
    "splines-rnn-quantized-A9-3": {
        "SPEAKER_ARCH_RNN_QUANTIZATION_PARAMETERS": {
            "embedding_latent_dim": 8,
            "embedding_dims": [4, 4],
            "critic_dims": [8, 8],
            "rnn_hidden_dim": 2,
            "vq_num_embeddings": 16,
            "vq_embedding_dim": 16,
            "vq_commitment_cost": 0.25,
            "use_vq": False
        }
    },
    "splines-rnn-quantized-B-0": {
        "SPEAKER_ARCH_RNN_QUANTIZATION_PARAMETERS": {
            "embedding_latent_dim": 12,
            "embedding_dims": [2, 2],
            "critic_dims": [8, 8],
            "rnn_hidden_dim": 2,
            "vq_num_embeddings": 16,
            "vq_embedding_dim": 16,
            "vq_commitment_cost": 0.25,
            "use_vq": False
        }
    },
    "splines-rnn-quantized-B-1": {
        "SPEAKER_ARCH_RNN_QUANTIZATION_PARAMETERS": {
            "embedding_latent_dim": 6,
            "embedding_dims": [4, 2],
            "critic_dims": [8, 8],
            "rnn_hidden_dim": 2,
            "vq_num_embeddings": 16,
            "vq_embedding_dim": 16,
            "vq_commitment_cost": 0.25,
            "use_vq": False
        }
    },
    "splines-rnn-quantized-B-2": {
        "SPEAKER_ARCH_RNN_QUANTIZATION_PARAMETERS": {
            "embedding_latent_dim": 4,
            "embedding_dims": [2, 2],
            "critic_dims": [8, 8],
            "rnn_hidden_dim": 2,
            "vq_num_embeddings": 16,
            "vq_embedding_dim": 16,
            "vq_commitment_cost": 0.25,
            "use_vq": False
        }
    },
    "splines-rnn-quantized-C-0": {
        "SPEAKER_ARCH_RNN_QUANTIZATION_PARAMETERS": {
            "embedding_latent_dim": 4,
            "embedding_dims": [2, 2],
            "critic_dims": [8, 8],
            "rnn_hidden_dim": 2,
            "vq_num_embeddings": 32,
            "vq_embedding_dim": 2,
            "vq_commitment_cost": 0.25,
            "use_vq": True
        }
    },
    "splines-rnn-quantized-C-1": {
        "SPEAKER_ARCH_RNN_QUANTIZATION_PARAMETERS": {
            "embedding_latent_dim": 4,
            "embedding_dims": [2, 2],
            "critic_dims": [8, 8],
            "rnn_hidden_dim": 2,
            "vq_num_embeddings": 8,
            "vq_embedding_dim": 2,
            "vq_commitment_cost": 0.25,
            "use_vq": True
        }
    },
    "splines-rnn-quantized-C-2": {
        "SPEAKER_ARCH_RNN_QUANTIZATION_PARAMETERS": {
            "embedding_latent_dim": 4,
            "embedding_dims": [2, 2],
            "critic_dims": [8, 8],
            "rnn_hidden_dim": 2,
            "vq_num_embeddings": 2,
            "vq_embedding_dim": 2,
            "vq_commitment_cost": 0.25,
            "use_vq": True
        }
    },
    "splines-rnn-quantized-C-3": {
        "SPEAKER_ARCH_RNN_QUANTIZATION_PARAMETERS": {
            "embedding_latent_dim": 4,
            "embedding_dims": [2, 2],
            "critic_dims": [8, 8],
            "rnn_hidden_dim": 2,
            "vq_num_embeddings": 32,
            "vq_embedding_dim": 2,
            "vq_commitment_cost": 0.5,
            "use_vq": True
        }
    },
    "splines-rnn-quantized-C-4": {
        "SPEAKER_ARCH_RNN_QUANTIZATION_PARAMETERS": {
            "embedding_latent_dim": 4,
            "embedding_dims": [2, 2],
            "critic_dims": [8, 8],
            "rnn_hidden_dim": 2,
            "vq_num_embeddings": 8,
            "vq_embedding_dim": 2,
            "vq_commitment_cost": 0.5,
            "use_vq": True
        }
    },
    "splines-rnn-quantized-C-5": {
        "SPEAKER_ARCH_RNN_QUANTIZATION_PARAMETERS": {
            "embedding_latent_dim": 4,
            "embedding_dims": [2, 2],
            "critic_dims": [8, 8],
            "rnn_hidden_dim": 2,
            "vq_num_embeddings": 2,
            "vq_embedding_dim": 2,
            "vq_commitment_cost": 0.5,
            "use_vq": True
        }
    },
    "splines-rnn-quantized-C-6": {
        "SPEAKER_ARCH_RNN_QUANTIZATION_PARAMETERS": {
            "embedding_latent_dim": 4,
            "embedding_dims": [2, 2],
            "critic_dims": [8, 8],
            "rnn_hidden_dim": 2,
            "vq_num_embeddings": 2,
            "vq_embedding_dim": 2,
            "vq_commitment_cost": 0.8,
            "use_vq": True
        }
    },
    "splines-rnn-quantized-C-7": {
        "SPEAKER_ARCH_RNN_QUANTIZATION_PARAMETERS": {
            "embedding_latent_dim": 4,
            "embedding_dims": [2, 2],
            "critic_dims": [8, 8],
            "rnn_hidden_dim": 2,
            "vq_num_embeddings": 2,
            "vq_embedding_dim": 2,
            "vq_commitment_cost": 0.95,
            "use_vq": True
        }
    },
    "splines-rnn-quantized-A0-0": {     # Base is micro-A-0
        "SPEAKER_ARCH_RNN_QUANTIZATION_PARAMETERS": {
            "embedding_latent_dim": 32,
            "embedding_dims": [32, 16, 8],
            "critic_dims": [16, 16],
            "rnn_hidden_dim": 8,
            "use_vq": False
        }
    },
    "splines-rnn-quantized-A0-1": {     # Base is micro-A-3
        "SPEAKER_ARCH_RNN_QUANTIZATION_PARAMETERS": {
            "embedding_latent_dim": 32,
            "embedding_dims": [32, 16, 16],
            "critic_dims": [16, 16],
            "rnn_hidden_dim": 16,
            "use_vq": False
        }
    },
    "splines-rnn-quantized-A0-2": {     # Base is micro-A-3
        "SPEAKER_ARCH_RNN_QUANTIZATION_PARAMETERS": {
            "embedding_latent_dim": 32,
            "embedding_dims": [32, 32, 32],
            "critic_dims": [16, 16],
            "rnn_hidden_dim": 32,
            "use_vq": False
        }
    },
    "splines-rnn-quantized-A0-3": {     # Base is micro-A-3
        "SPEAKER_ARCH_RNN_QUANTIZATION_PARAMETERS": {
            "embedding_latent_dim": 32,
            "embedding_dims": [32, 32, 64],
            "critic_dims": [16, 16],
            "rnn_hidden_dim": 64,
            "use_vq": False
        }
    },
    "splines-rnn-quantized-A3Q-0": {     # Base is micro-A-3, quantized
        "SPEAKER_ARCH_RNN_QUANTIZATION_PARAMETERS": {
            "embedding_latent_dim": 32,
            "embedding_dims": [32, 16, 16],
            "critic_dims": [16, 16],
            "rnn_hidden_dim": 16,
            "vq_num_embeddings": 32,
            "vq_embedding_dim": 16,
            "vq_commitment_cost": 0.25,
            "use_vq": True
        }
    },
    "splines-rnn-quantized-A3Q-1": {     # Base is micro-A-3, quantized
        "SPEAKER_ARCH_RNN_QUANTIZATION_PARAMETERS": {
            "embedding_latent_dim": 32,
            "embedding_dims": [32, 16, 16],
            "critic_dims": [16, 16],
            "rnn_hidden_dim": 16,
            "vq_num_embeddings": 16,
            "vq_embedding_dim": 16,
            "vq_commitment_cost": 0.25,
            "use_vq": True
        }
    },
    "splines-rnn-quantized-A3Q-2": {     # Base is micro-A-3, quantized
        "SPEAKER_ARCH_RNN_QUANTIZATION_PARAMETERS": {
            "embedding_latent_dim": 32,
            "embedding_dims": [32, 16, 16],
            "critic_dims": [16, 16],
            "rnn_hidden_dim": 16,
            "vq_num_embeddings": 8,
            "vq_embedding_dim": 16,
            "vq_commitment_cost": 0.25,
            "use_vq": True
        }
    },
    "splines-rnn-quantized-A3Q-3": {     # Base is micro-A-3, quantized
        "SPEAKER_ARCH_RNN_QUANTIZATION_PARAMETERS": {
            "embedding_latent_dim": 32,
            "embedding_dims": [32, 16, 16],
            "critic_dims": [16, 16],
            "rnn_hidden_dim": 16,
            "vq_num_embeddings": 4,
            "vq_embedding_dim": 16,
            "vq_commitment_cost": 0.25,
            "use_vq": True
        }
    },
    "splines-rnn-quantized-A3Q-4": {     # Base is micro-A-3, quantized. has different commitment cost
        "SPEAKER_ARCH_RNN_QUANTIZATION_PARAMETERS": {
            "embedding_latent_dim": 32,
            "embedding_dims": [32, 16, 16],
            "critic_dims": [16, 16],
            "rnn_hidden_dim": 16,
            "vq_num_embeddings": 32,
            "vq_embedding_dim": 16,
            "vq_commitment_cost": 0.1,
            "use_vq": True
        }
    },
    "splines-rnn-quantized-A3Q-5": {     # Base is micro-A-3, quantized. has different commitment cost
        "SPEAKER_ARCH_RNN_QUANTIZATION_PARAMETERS": {
            "embedding_latent_dim": 32,
            "embedding_dims": [32, 16, 16],
            "critic_dims": [16, 16],
            "rnn_hidden_dim": 16,
            "vq_num_embeddings": 32,
            "vq_embedding_dim": 16,
            "vq_commitment_cost": 0.5,
            "use_vq": True
        }
    },
    "splines-rnn-quantized-A3Q-6": {     # Base is micro-A-3, quantized. has different commitment cost
        "SPEAKER_ARCH_RNN_QUANTIZATION_PARAMETERS": {
            "embedding_latent_dim": 32,
            "embedding_dims": [32, 16, 16],
            "critic_dims": [16, 16],
            "rnn_hidden_dim": 16,
            "vq_num_embeddings": 64,
            "vq_embedding_dim": 16,
            "vq_commitment_cost": 0.25,
            "use_vq": True
        }
    },
    "splines-rnn-quantized-A3Q-7": {     # Base is micro-A-3, quantized. has different commitment cost
        "SPEAKER_ARCH_RNN_QUANTIZATION_PARAMETERS": {
            "embedding_latent_dim": 32,
            "embedding_dims": [32, 16, 16],
            "critic_dims": [16, 16],
            "rnn_hidden_dim": 16,
            "vq_num_embeddings": 64,
            "vq_embedding_dim": 16,
            "vq_commitment_cost": 0.05,
            "use_vq": True
        }
    },
    "splines-rnn-quantized-A3Q-8": {     # Base is micro-A-3, quantized. has different commitment cost
        "SPEAKER_ARCH_RNN_QUANTIZATION_PARAMETERS": {
            "embedding_latent_dim": 32,
            "embedding_dims": [32, 16, 16],
            "critic_dims": [16, 16],
            "rnn_hidden_dim": 16,
            "vq_num_embeddings": 64,
            "vq_embedding_dim": 16,
            "vq_commitment_cost": 0.01,
            "use_vq": True
        }
    },
    "splines-rnn-quantized-A3Q-9": {     # Base is micro-A-3, quantized. has different commitment cost
        "SPEAKER_ARCH_RNN_QUANTIZATION_PARAMETERS": {
            "embedding_latent_dim": 32,
            "embedding_dims": [32, 16, 16],
            "critic_dims": [16, 16],
            "rnn_hidden_dim": 16,
            "vq_num_embeddings": 64,
            "vq_embedding_dim": 16,
            "vq_commitment_cost": 1.0,
            "use_vq": True
        }
    },
    "splines-rnn-quantized-A3Q-10": {     # Base is micro-A-3, quantized. has different commitment cost
        "SPEAKER_ARCH_RNN_QUANTIZATION_PARAMETERS": {
            "embedding_latent_dim": 32,
            "embedding_dims": [32, 16, 16],
            "critic_dims": [16, 16],
            "rnn_hidden_dim": 16,
            "vq_num_embeddings": 64,
            "vq_embedding_dim": 16,
            "vq_commitment_cost": 3.0,
            "use_vq": True
        }
    },
    "splines-rnn-quantized-A0P-0": {     # Base is micro-A-3, with positional encodings
        "SPEAKER_ARCH_RNN_QUANTIZATION_PARAMETERS": {
            "embedding_latent_dim": 32,
            "embedding_dims": [32, 16, 8],
            "critic_dims": [16, 16],
            "rnn_hidden_dim": 8,
            "use_vq": False,
            "use_pos_encs": True
        }
    },
    "splines-rnn-quantized-A0P-1": {     # Base is micro-A-3, with positional encodings
        "SPEAKER_ARCH_RNN_QUANTIZATION_PARAMETERS": {
            "embedding_latent_dim": 32,
            "embedding_dims": [32, 16, 16],
            "critic_dims": [16, 16],
            "rnn_hidden_dim": 16,
            "use_vq": False,
            "use_pos_encs": True
        }
    },
    "splines-rnn-quantized-A3Q-NoSTE-0": {     # Base is micro-A-3, quantized
        "SPEAKER_ARCH_RNN_QUANTIZATION_PARAMETERS": {
            "embedding_latent_dim": 32,
            "embedding_dims": [32, 16, 16],
            "critic_dims": [16, 16],
            "rnn_hidden_dim": 16,
            "vq_num_embeddings": 32,
            "vq_embedding_dim": 16,
            "vq_commitment_cost": 0.25,
            "use_vq": True,
            "straight_through_estimator": False
        }
    },
    "splines-rnn-quantized-A0PE-1": {     # Base is micro-A-3, with positional *embeddings*
        "SPEAKER_ARCH_RNN_QUANTIZATION_PARAMETERS": {
            "embedding_latent_dim": 32,
            "embedding_dims": [32, 16, 8],
            "critic_dims": [16, 16],
            "rnn_hidden_dim": 16,
            "use_vq": False,
            "use_pos_embs": True,
            "pos_emb_latent_dim": 8
        }
    },
    "splines-rnn-quantized-A0PE-2": {     # Base is micro-A-3, with positional *embeddings*
        "SPEAKER_ARCH_RNN_QUANTIZATION_PARAMETERS": {
            "embedding_latent_dim": 32,
            "embedding_dims": [32, 16, 16],
            "critic_dims": [16, 16],
            "rnn_hidden_dim": 32,
            "use_vq": False,
            "use_pos_embs": True,
            "pos_emb_latent_dim": 16
        }
    },
    "splines-rnn-quantized-A0PE-3": {     # Base is micro-A-3, with positional *embeddings*
        "SPEAKER_ARCH_RNN_QUANTIZATION_PARAMETERS": {
            "embedding_latent_dim": 32,
            "embedding_dims": [32, 32, 32],
            "critic_dims": [16, 16],
            "rnn_hidden_dim": 64,
            "use_vq": False,
            "use_pos_embs": True,
            "pos_emb_latent_dim": 32
        }
    },
    "splines-rnn-quantized-A0PE-4": {     # Base is micro-A-3, with positional *embeddings*
        "SPEAKER_ARCH_RNN_QUANTIZATION_PARAMETERS": {
            "embedding_latent_dim": 32,
            "embedding_dims": [32, 32, 64],
            "critic_dims": [16, 16],
            "rnn_hidden_dim": 128,
            "use_vq": False,
            "use_pos_embs": True,
            "pos_emb_latent_dim": 64
        }
    },
    "splines-rnn-quantized-A0PEL-0": {     # Base is micro-A-3, with positional *embeddings*
        "SPEAKER_ARCH_RNN_QUANTIZATION_PARAMETERS": {
            "embedding_latent_dim": 32,
            "embedding_dims": [32, 32, 16],
            "critic_dims": [16, 16],
            "rnn_hidden_dim": 32,
            "use_vq": False,
            "use_pos_embs": True,
            "pos_emb_latent_dim": 16
        }
    },
    "splines-rnn-quantized-A0PEL-1": {     # Base is micro-A-3, with positional *embeddings*
        "SPEAKER_ARCH_RNN_QUANTIZATION_PARAMETERS": {
            "embedding_latent_dim": 32,
            "embedding_dims": [32, 32, 24],
            "critic_dims": [16, 16],
            "rnn_hidden_dim": 32,
            "use_vq": False,
            "use_pos_embs": True,
            "pos_emb_latent_dim": 8
        }
    },
    "splines-rnn-quantized-A0PEL-2": {     # Base is micro-A-3, with positional *embeddings*
        "SPEAKER_ARCH_RNN_QUANTIZATION_PARAMETERS": {
            "embedding_latent_dim": 32,
            "embedding_dims": [32, 32, 28],
            "critic_dims": [16, 16],
            "rnn_hidden_dim": 32,
            "use_vq": False,
            "use_pos_embs": True,
            "pos_emb_latent_dim": 4
        }
    },
    "splines-rnn-quantized-A0PEL-3": {     # Base is micro-A-3, with positional *embeddings*
        "SPEAKER_ARCH_RNN_QUANTIZATION_PARAMETERS": {
            "embedding_latent_dim": 32,
            "embedding_dims": [32, 32, 8],
            "critic_dims": [16, 16],
            "rnn_hidden_dim": 32,
            "use_vq": False,
            "use_pos_embs": True,
            "pos_emb_latent_dim": 24
        }
    },
    "splines-rnn-quantized-A0PEL-4": {     # Base is micro-A-3, with positional *embeddings*
        "SPEAKER_ARCH_RNN_QUANTIZATION_PARAMETERS": {
            "embedding_latent_dim": 32,
            "embedding_dims": [32, 32, 4],
            "critic_dims": [16, 16],
            "rnn_hidden_dim": 32,
            "use_vq": False,
            "use_pos_embs": True,
            "pos_emb_latent_dim": 28
        }
    },
}
