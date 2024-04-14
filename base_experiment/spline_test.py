import jax.numpy as jnp
from jax import vmap, jit


canvas_shape = (28, 28)

@vmap
def paint_multiple_splines(all_spline_params: jnp.array):
    """Paint multiple splines on a single canvas."""

    @vmap
    def paint_spline_on_canvas(spline_params: jnp.array):
        """Paint a single spline on the canvas with specified thickness using advanced indexing."""

        def bezier_spline(t, P0, P1, P2):
            """Compute points on a quadratic Bézier spline for a given t."""
            t = t[:, None]  # Shape (N, 1) to broadcast with P0, P1, P2 of shape (2,)
            P = (1 - t)**2 * P0 + 2 * (1 - t) * t * P1 + t**2 * P2
            return P  # Returns shape (N, 2), a list of points on the spline

        brush_size = 0
        # canvas_shape = (28, 28)

        spline_params *= canvas_shape[0]
        
        # P0, P1, P2 = spline_params.reshape((3, 2)) 
        P0, P1, P2 = spline_params[0:2], spline_params[2:4], spline_params[4:6]
        t_values = jnp.linspace(0, 1, num=50)
        spline_points = bezier_spline(t_values, P0, P1, P2)
        x_points, y_points = jnp.round(spline_points).astype(int).T

        # Generate brush offsets
        brush_offsets = jnp.array([(dx, dy) for dx in range(-brush_size, brush_size + 1) 
                                            for dy in range(-brush_size, brush_size + 1)])
        x_offsets, y_offsets = brush_offsets.T

        # Calculate all indices to update for each point (broadcasting magic)
        all_x_indices = x_points[:, None] + x_offsets
        all_y_indices = y_points[:, None] + y_offsets

        # Flatten indices and filter out-of-bound ones
        all_x_indices = jnp.clip(all_x_indices.flatten(), 0, canvas_shape[0])
        all_y_indices = jnp.clip(all_y_indices.flatten(), 0, canvas_shape[1])

        # Update the canvas
        canvas = jnp.zeros(canvas_shape)
        canvas = canvas.at[all_x_indices, all_y_indices].add(1)
        return canvas

    # Vmap over splines and sum contributions
    all_spline_params = jnp.clip(all_spline_params, 0.0, 1.0)
    canvas = jnp.clip(paint_spline_on_canvas(all_spline_params.reshape(-1, 6)).sum(axis=0), 0.0, 1.0)
    return canvas

# Example usage
all_spline_params = jnp.array([[[[10, 10], [50, 20], [90, 10]],
                               [[10, 90], [50, 80], [90, 90]],
                               [[50, 50], [60, 60], [70, 40]],
                               [[50, 50], [60, 60], [70, 40]]],
                               [[[20, 10], [50, 20], [90, 10]],
                               [[10, 90], [50, 80], [90, 90]],
                               [[50, 50], [60, 60], [70, 40]],
                               [[50, 50], [60, 60], [70, 40]]]]) * (1.0 / 100.0)  # Shape (2, 4, 3, 2)
all_spline_params = all_spline_params.reshape(2, 24)

splines = jnp.array([[0.37210482358932495, 0.5101042985916138, 0.535472571849823, 0.3512147068977356, 0.20872655510902405, 0.27592694759368896, 0.5394357442855835, 0.6043564081192017, 0.4065988063812256, 0.5282517075538635, 0.6306172609329224, 0.3307557702064514]])

canvas = paint_multiple_splines(splines) # shape (num images, num splines * 6)

# Visualization with matplotlib
import matplotlib.pyplot as plt
fig, axs = plt.subplots(1, 2, figsize=(12, 6))
axs[0].imshow(canvas[0], cmap='viridis', origin='lower')
axs[0].set_title('Optimized Painted Splines on Canvas 1')
axs[0].set_xlabel('X')
axs[0].set_ylabel('Y')

axs[1].imshow(canvas[1], cmap='viridis', origin='lower')
axs[1].set_title('Optimized Painted Splines on Canvas 2')
axs[1].set_xlabel('X')
axs[1].set_ylabel('Y')

plt.tight_layout()
plt.show()
