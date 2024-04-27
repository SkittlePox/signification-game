import jax
import jax.numpy as jnp
import jax.lax as lax

#### This file is for helping me understand some jax functions ####

num_speakers = 5
num_channels = 10
num_listeners = 15
requested_num_speaker_images = 3

key = jax.random.PRNGKey(0)

speaker_ids = jax.random.permutation(key, num_speakers)
speaker_ids = jnp.pad(speaker_ids, (0, num_channels-num_speakers))

mask = jnp.where(jnp.arange(num_channels) < requested_num_speaker_images, 1, 0)

env_ids = jax.random.permutation(key, num_channels) + num_speakers

# join the two arrays based on the mask
possible_speakers = jnp.where(mask, speaker_ids, env_ids)

speakers = jax.random.permutation(key, possible_speakers).reshape((-1, 1))
listeners = jax.random.permutation(key, num_listeners).reshape((-1, 1))[:num_channels]
listeners2 = jax.lax.slice(jax.random.permutation(key, num_listeners).reshape((-1, 1)), [0, 0], [num_channels, 1])

print(speakers)
print(listeners)
print(listeners2)

def cumsum_step(carry, x):
    # Here, `carry` holds the accumulated sum so far, and `x` is the current item.
    new_carry = carry + x
    return new_carry, new_carry  # Pass the updated sum and also return it as the result

# Initial carry value is 0 because we start with no sum
carry_init = 0

# Example input array
xs = jnp.array([1, 2, 3, 4])

# Run scan
_, cumsum = lax.scan(cumsum_step, carry_init, xs)

print(cumsum)
