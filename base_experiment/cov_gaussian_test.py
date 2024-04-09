import jax.numpy as jnp
import matplotlib.pyplot as plt

def paint_gaussians_with_covariance_on_array(array_shape, gaussians_params):
    """
    Paint multiple 2D Gaussians with full covariance matrices on a 2D array, 
    using a NumPy array for parameters.
    
    Parameters:
    - array_shape: tuple of int, shape of the 2D array (height, width).
    - gaussians_params: NumPy array with each row representing the parameters for a Gaussian 
      (normalized mean x, normalized mean y, normalized variance x, normalized variance y, 
      normalized covariance xy, amplitude).
    
    Returns:
    - 2D numpy array with the Gaussians painted on it.
    """
    y, x = jnp.indices(array_shape)  # Create a grid of x and y coordinates
    array = jnp.zeros(array_shape)

    for params in gaussians_params:
        x_mu_norm, y_mu_norm, sigma_x2_norm, sigma_y2_norm, sigma_xy_norm, amplitude = params
        
        # Convert normalized mean to actual coordinates
        x_mu = x_mu_norm * array_shape[1]
        y_mu = y_mu_norm * array_shape[0]

        # Convert normalized variances and covariance to actual values
        sigma_x2 = sigma_x2_norm * array_shape[1]**2
        sigma_y2 = sigma_y2_norm * array_shape[0]**2
        sigma_xy = sigma_xy_norm * array_shape[1] * array_shape[0]

        # Construct the covariance matrix
        cov_matrix = jnp.array([[sigma_x2, sigma_xy], [sigma_xy, sigma_y2]])
        inv_cov_matrix = jnp.linalg.inv(cov_matrix)

        # Create the meshgrid arrays for x and y
        X = jnp.vstack((x.ravel() - x_mu, y.ravel() - y_mu))
        
        # Compute the Gaussian function
        gaussian = amplitude * jnp.exp(-0.5 * jnp.sum(X.T @ inv_cov_matrix * X.T, axis=1)).reshape(array_shape)
        array += gaussian

    return array

# Example usage
array_shape = (100, 100)
# Each row: normalized mean x, normalized mean y, normalized variance x, normalized variance y,
# normalized covariance xy, amplitude
gaussians_params = jnp.array([
    [0.5, 0.5, 0.01, 0.01, 0, 1.0],  # Assuming no covariance (0) for simplicity
    [0.7, 0.7, 0.005, 0.005, 0.002, 0.5],
    [0.3, 0.4, 0.015, 0.01, -0.003, 0.75]
])

gaussian_array = paint_gaussians_with_covariance_on_array(array_shape, gaussians_params)

# Visualize the combined Gaussians
plt.imshow(gaussian_array, origin='lower', cmap='viridis', extent=(0, array_shape[1], 0, array_shape[0]))
plt.colorbar()
plt.title('Multiple 2D Gaussian Distributions with Covariance')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
