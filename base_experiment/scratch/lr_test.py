import jax.numpy as jnp

def parse_schedule(description):
    segments = description.split()
    changes = []
    i = 0
    while i < len(segments):
        if segments[i] in ['jump', 'anneal']:
            # Determine initial_lr based on whether it's the start or a subsequent segment
            if i == 1:
                initial_lr = float(segments[i-1])
            else:
                initial_lr = float(changes[-1][2])  # Use the final_lr of the last segment
            change_type = segments[i]
            final_lr = float(segments[i+2])
            at_step = int(segments[i+4])
            changes.append((initial_lr, change_type, final_lr, at_step))
            i += 4  # skip to the next relevant segment
        i += 1

    def schedule(step):
        lr = jnp.array(changes[0][0])  # Default to the initial LR in the first segment

        for index, (start_lr, change_type, end_lr, change_step) in enumerate(changes):
            condition = step >= change_step

            if change_type == 'jump':
                lr = jnp.where(condition, end_lr, lr)
            elif change_type == 'anneal':
                # Calculate duration based on the next change step or extend to the current step if last segment
                next_change_step = changes[index + 1][3] if index + 1 < len(changes) else step
                duration = next_change_step - change_step
                fraction = (step - change_step) / duration
                lr = jnp.where(condition, jnp.interp(fraction, jnp.array([0, 1]), jnp.array([start_lr, end_lr])), lr)

        return lr

    return schedule

# Example Usage
# lr_schedule = parse_schedule("1e-3 jump to 1e-2 at 100 anneal to 1e-4 at 300")
# # To get values, we can use JAX's jit to compile this function
from jax import jit
# compiled_schedule = jit(lr_schedule)

# # Testing the compiled schedule function
# print(compiled_schedule(50))   # Outputs 0.001
# print(compiled_schedule(150))  # Outputs 0.01
# print(compiled_schedule(350))  # Outputs a value linearly annealed between 0.01 and 0.0001


# Example Usage
lr_schedule_description_1 = "1e-3 jump to 1e-2 at 100 anneal to 1e-4 at 300"
lr_schedule_description_2 = "1e-3 anneal to 1e-2 at 200 jump to 1e-5 at 400"
lr_schedule_description_3 = "1e-3 anneal to 1e-2 at 100 jump to 1e-1 at 200 anneal to 1e-6 at 300"

# Compiling the schedules
compiled_schedule_1 = jit(parse_schedule(lr_schedule_description_1))
compiled_schedule_2 = jit(parse_schedule(lr_schedule_description_2))
compiled_schedule_3 = jit(parse_schedule(lr_schedule_description_3))

# Testing various points in the schedule
test_cases = [
    (compiled_schedule_1, 50, 0.001),   # Before any change
    (compiled_schedule_1, 150, 0.01),   # Post jump, pre-anneal
    (compiled_schedule_1, 350, 0.0001), # Post anneal

    (compiled_schedule_2, 100, 0.001),  # During anneal
    (compiled_schedule_2, 300, 0.01),   # Post anneal, pre jump
    (compiled_schedule_2, 450, 0.00001),# Post jump

    (compiled_schedule_3, 50, 0.001),   # Before any change
    (compiled_schedule_3, 150, 0.01),   # During first anneal
    (compiled_schedule_3, 250, 0.1),    # Post jump, before second anneal
    (compiled_schedule_3, 350, 0.000001)# Post second anneal
]

# Running the test cases
for schedule, step, expected in test_cases:
    result = schedule(step)
    assert jnp.isclose(result, expected, rtol=1e-3, atol=1e-3), f"Test failed at step {step}: expected {expected}, got {result}"
    print(f"Test passed at step {step}: expected {expected}, got {result}")
