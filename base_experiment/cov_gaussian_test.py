import jax.numpy as jnp
from jax import jit, vmap

def add_single_gaussian(x, y, gaussian_params):
    """
    Add a single Gaussian defined by `gaussian_params` to a 28x28 grid.
    """
    x_mu_norm, y_mu_norm, amplitude, sigma_x2_norm, sigma_y2_norm, sigma_xy_norm = gaussian_params
    
    # Convert normalized mean to actual coordinates in a 28x28 array
    x_mu = x_mu_norm * 28
    y_mu = y_mu_norm * 28

    # Convert normalized variances and covariance to actual values
    sigma_x2 = sigma_x2_norm * 28**2
    sigma_y2 = sigma_y2_norm * 28**2
    sigma_xy = sigma_xy_norm * 28 * 28

    # Construct the covariance matrix and its inverse
    cov_matrix = jnp.array([[sigma_x2, sigma_xy], [sigma_xy, sigma_y2]])
    inv_cov_matrix = jnp.linalg.inv(cov_matrix)

    # Compute the Gaussian function
    X = jnp.vstack((x.ravel() - x_mu, y.ravel() - y_mu))
    gaussian = amplitude * jnp.exp(-0.5 * jnp.sum(X.T @ inv_cov_matrix * X.T, axis=1)).reshape((28, 28))
    
    return gaussian

@jit
def paint_gaussians_with_covariance_vmap(gaussians_params):
    array_shape = (28, 28)
    y, x = jnp.indices(array_shape)
    # Use vmap to apply `add_single_gaussian` across all Gaussian parameter sets
    gaussians = vmap(add_single_gaussian, in_axes=(None, None, 0))(x, y, gaussians_params)
    # Sum the contributions from all Gaussians
    return jnp.sum(gaussians, axis=0)

# Example usage with JAX array
gaussians_params = jnp.array([
    [0.5, 0.5, 1.0, 0.01, 0.01, 0],
    [0.7, 0.7, 0.5, 0.005, 0.005, 0.002],
    [0.3, 0.4, 0.75, 0.015, 0.01, -0.01]
])

gaussian_array = paint_gaussians_with_covariance_vmap(gaussians_params)

# Plotting
import matplotlib.pyplot as plt
plt.imshow(gaussian_array, origin='lower', cmap='viridis', extent=(0, 28, 0, 28))
plt.colorbar()
plt.title('Multiple 2D Gaussian Distributions with Covariance (28x28, JAX, vmap)')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
