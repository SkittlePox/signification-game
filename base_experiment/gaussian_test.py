import jax.numpy as jnp
import matplotlib.pyplot as plt

def paint_normalized_gaussians_on_array(array_shape, gaussians_params):
    """
    Paint multiple 2D Gaussians with normalized parameters on a 2D array, using a NumPy array for parameters.
    
    Parameters:
    - array_shape: tuple of int, shape of the 2D array (height, width).
    - gaussians_params: NumPy array with each row representing the parameters for a Gaussian 
      (normalized mean x, normalized mean y, normalized variance x, normalized variance y, amplitude).
    
    Returns:
    - 2D numpy array with the Gaussians painted on it.
    """
    y, x = jnp.indices(array_shape)  # Create a grid of x and y coordinates
    array = jnp.zeros(array_shape)

    for params in gaussians_params:
        x_mu_norm, y_mu_norm, sigma_x2_norm, sigma_y2_norm, amplitude = params
        
        # Convert normalized mean to actual coordinates
        x_mu = x_mu_norm * array_shape[1]
        y_mu = y_mu_norm * array_shape[0]

        # Convert normalized covariance to actual variances
        sigma_x2 = sigma_x2_norm * array_shape[1]**2
        sigma_y2 = sigma_y2_norm * array_shape[0]**2

        # Compute the 2D Gaussian formula and add it to the array
        gaussian = amplitude * jnp.exp(-(((x - x_mu)**2 / (2 * sigma_x2)) + ((y - y_mu)**2 / (2 * sigma_y2))))
        array += gaussian

    return jnp.clip(array, a_min=0.0, a_max=1.0)

# Example usage
array_shape = (100, 100)
# Each row: normalized mean x, normalized mean y, normalized variance x, normalized variance y, amplitude
gaussians_params = jnp.array([
    [0.5, 0.5, 0.01, 0.01, 1.0],
    [0.7, 0.7, 0.005, 0.005, 0.5],
    [0.3, 0.4, 0.015, 0.01, 0.75]
])

gaussian_array = paint_normalized_gaussians_on_array(array_shape, gaussians_params)

# Visualize the combined Gaussians
plt.imshow(gaussian_array, origin='lower', cmap='viridis', extent=(0, array_shape[1], 0, array_shape[0]))
plt.colorbar()
plt.title('Multiple 2D Gaussian Distributions (Normalized Params)')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
