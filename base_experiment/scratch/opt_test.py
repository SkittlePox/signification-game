import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState
import optax

# Create TrainState with a shared optimizer
def create_train_state():
    params = {'w': jnp.array([1.0]), 'b': jnp.array([0.0])}
    return params

# Initialize shared optimizer
tx = optax.adam(learning_rate=0.1)  # Single optimizer shared across agents

# Create agent-specific TrainStates
train_params = (create_train_state(), create_train_state())
train_opt_states = tuple(tx.init(p) for p in train_params)

# Mock gradients
grads = (
    {'w': jnp.array([0.1]), 'b': jnp.array([0.2])},
    {'w': jnp.array([0.3]), 'b': jnp.array([0.4])}
)

# Step 1: Batch Params, Opt State, and Gradients
batched_params = jax.tree.map(lambda *args: jnp.stack(args), *train_params)
batched_opt_states = jax.vmap(tx.init)(batched_params)
batched_grads = jax.tree.map(lambda *args: jnp.stack(args), *grads)

# Step 2: Apply Gradients Using Shared Optimizer
def apply_gradients_fn(params, opt_state, grads):
    updates, new_opt_state = tx.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_opt_state

# Apply updates with vmap
new_params, new_opt_states = jax.vmap(
    apply_gradients_fn,
    in_axes=(0, 0, 0)
)(batched_params, batched_opt_states, batched_grads)

# Step 3: Unbatch and Reconstruct TrainStates
unbatched_params = jax.tree.map(lambda x: list(x), new_params)
unbatched_opt_states = jax.tree.map(lambda x: list(x), new_opt_states)

print(unbatched_params)
print(unbatched_opt_states)

# Reconstruct TrainState objects
updated_train_states = tuple(
    TrainState.create(
        apply_fn=None,
        params=p,
        tx=tx,  # Shared optimizer
        opt_state=o
    )
    for p, o in zip(unbatched_params, unbatched_opt_states)
)

# Verify Results
for i, ts in enumerate(updated_train_states):
    print(f"TrainState {i}: params={ts.params}, opt_state={ts.opt_state}")
