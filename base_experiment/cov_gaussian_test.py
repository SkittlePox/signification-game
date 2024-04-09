import jax.numpy as jnp
from jax import jit, vmap
import jax

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



@jax.vmap
def gauss_splat_covar(actions: jnp.array):
    def paint_normalized_gaussians_on_array(array_shape, gaussians_params):
        """
        Paint multiple 2D Gaussians with full covariance matrices on a 2D array.
        
        Parameters:
        - array_shape: tuple of int, shape of the 2D array (height, width).
        - gaussians_params: JAX array with each row representing the parameters for a Gaussian 
        (normalized mean x, normalized mean y, normalized variance x, normalized variance y, 
        normalized covariance xy, amplitude).
        
        Returns:
        - 2D JAX array with the Gaussians painted on it.
        """
        y, x = jnp.indices(array_shape)  # Create a grid of x and y coordinates
        array = jnp.zeros(array_shape)

        @jax.vmap
        def compute_gaussian(params):
            x_mu_norm, y_mu_norm, amplitude, sigma_x2_norm, sigma_y2_norm, sigma_xy_norm = params
            
            # Convert normalized mean to actual coordinates
            x_mu = x_mu_norm * array_shape[1]
            y_mu = y_mu_norm * array_shape[0]

            # Convert normalized covariance to actual values
            sigma_x2 = sigma_x2_norm * array_shape[1]**2
            sigma_y2 = sigma_y2_norm * array_shape[0]**2
            sigma_xy = (2 * sigma_xy_norm - 1) * array_shape[1] * array_shape[0]

            # Construct the covariance matrix and its inverse
            cov_matrix = jnp.array([[sigma_x2, sigma_xy], [sigma_xy, sigma_y2]])
            inv_cov_matrix = jnp.linalg.inv(cov_matrix)

            # Compute the Gaussian function
            X = jnp.vstack((x.ravel() - x_mu, y.ravel() - y_mu))
            gaussian = amplitude * jnp.exp(-0.5 * jnp.sum(X.T @ inv_cov_matrix * X.T, axis=1)).reshape(array_shape)
            
            return gaussian

        gaussians = compute_gaussian(gaussians_params)
        array += jnp.sum(gaussians, axis=0)  # Sum contributions from all Gaussians

        return jnp.clip(array, a_min=0.0, a_max=1.0)

    # Assuming 'actions' includes the additional covariance parameters
    gaussians_params = actions.reshape(-1, 6)  # Reshape based on the new parameter structure
    image_dim = 28  # As per your indication, assuming a fixed image dimension
    array_shape = (image_dim, image_dim)

    gaussian_array = paint_normalized_gaussians_on_array(array_shape, gaussians_params)
    return gaussian_array




# Example usage with JAX array
gaussians_params = jnp.array([
    [0.5, 0.5, 1.0, 0.01, 0.01, 0],
    [0.7, 0.7, 0.5, 0.005, 0.005, 0.002],
    [0.3, 0.4, 0.75, 0.015, 0.01, -0.01]
])

gaussian_array = gauss_splat_covar(gaussians_params)

# Plotting
import matplotlib.pyplot as plt
plt.imshow(gaussian_array, origin='lower', cmap='viridis', extent=(0, 28, 0, 28))
plt.colorbar()
plt.title('Multiple 2D Gaussian Distributions with Covariance (28x28, JAX, vmap)')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()


