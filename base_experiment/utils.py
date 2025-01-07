import os
import random
from PIL import Image
from torchvision import datasets
import jax
import jax.numpy as jnp
import numpy as np
import math
import yaml
import uuid
import pathlib
import cloudpickle
from flax.training import train_state, orbax_utils
import orbax.checkpoint
from omegaconf import OmegaConf

def to_jax(dataset, num_datapoints=100):
    images = []
    labels = []
    for i in range(num_datapoints):
        img, label = dataset[i]
        images.append(jnp.array(img, dtype=jnp.float32))
        labels.append(jnp.array(label, dtype=jnp.int32))
    return jnp.array(images), jnp.array(labels)


def load_images_to_array(directory, categories=14, num_datapoints=100, target_size=(224, 224), seed=50):
    """
    Load all images from the specified directory and resize them to target_size.
    Returns a JAX array with shape (B, N, N, 3).
    
    Args:
    directory (str): The path to the directory containing images.
    target_size (tuple): Desired (width, height) of images.
    
    Returns:
    jnp.ndarray: JAX array of images with shape (B, N, N, 3).
    """

    if isinstance(categories, int):
        num_categories = categories
        categories = os.listdir(directory)[:num_categories]
    elif isinstance(categories, list):
        num_categories = len(categories)

    num_images_per_cat = math.ceil(num_datapoints/num_categories)
    images = []
    labels = []

    for i, cat in enumerate(categories):
        image_files = [f for f in os.listdir(directory+cat) if f.endswith(('.png', '.jpg', '.jpeg'))][:num_images_per_cat]
        # TODO: Think about selecting out specific categories at this point if categories is a one-hot vector
        for filename in image_files:
            img_path = os.path.join(directory+cat, filename)
            with Image.open(img_path) as img:
                img = img.convert('L').resize(target_size)
                img_array = np.array(img)
                images.append(img_array)
                # label = np.zeros((num_categories))    # This would be for one-hot labels
                # label[i] = 1
                labels.append(i)
    # Stack images into a single numpy array and convert to JAX array
    images_array = jnp.array(np.stack(images, axis=0), dtype=jnp.float32)
    labels_array = jnp.array(np.stack(labels, axis=0), dtype=jnp.int32)

    key = jax.random.PRNGKey(seed)
    indices = jax.random.permutation(key, len(images))
    images_array = images_array[indices][:num_datapoints]
    labels_array = labels_array[indices][:num_datapoints]

    return images_array, labels_array


######## Used in ippo_ff.py ##########

def get_train_freezing(phrase):
    if phrase.startswith("on then"):
        param_str = phrase.split(" then ")[1]
        crf_params = param_str.split(" at ")
        if crf_params[0] == "off":
            return lambda x: jax.lax.cond(x < eval(crf_params[1]), lambda _: 1.0, lambda _: 0.0, operand=None)
        if crf_params[0] == "even":
            return lambda x: jax.lax.cond(x < eval(crf_params[1]), lambda _: 1.0, lambda x: (x % 2).astype(float), operand=x)
        if crf_params[0] == "odd":
            return lambda x: jax.lax.cond(x < eval(crf_params[1]), lambda _: 1.0, lambda x: ((x + 1) % 2).astype(float), operand=x)
    elif phrase.startswith("off then"):
        param_str = phrase.split(" then ")[1]
        crf_params = param_str.split(" at ")
        if crf_params[0] == "on":
            return lambda x: jax.lax.cond(x < eval(crf_params[1]), lambda _: 0.0, lambda _: 1.0, operand=None)
        if crf_params[0] == "even":
            return lambda x: jax.lax.cond(x < eval(crf_params[1]), lambda _: 0.0, lambda x: (x % 2).astype(float), operand=x)
        if crf_params[0] == "odd":
            return lambda x: jax.lax.cond(x < eval(crf_params[1]), lambda _: 0.0, lambda x: ((x + 1) % 2).astype(float), operand=x)
    elif phrase == "off":
        return lambda x: 0.0
    elif phrase == "even":
        return lambda x: (x % 2).astype(float)
    elif phrase == "odd":
        return lambda x: ((x + 1) % 2).astype(float)
    else:
        return lambda x: 1.0
    

def get_anneal_schedule(description, num_minibatches=1):
    description = str(description)
    segments = description.split()
    changes = []
    i = 0
    starter_lr = float(segments[0])
    while i < len(segments):
        if segments[i] in ['jump', 'anneal']:
            # Determine initial_lr based on whether it's the start or a subsequent segment
            if i == 1:
                initial_lr = float(segments[i-1])
                start_step = 0
            else:
                initial_lr = float(changes[-1][3])  # Use the final_lr of the last segment
                start_step = int(segments[i-1]) * num_minibatches
            change_type = segments[i]
            final_lr = float(segments[i+2])
            at_step = int(segments[i+4]) * num_minibatches
            changes.append((start_step, initial_lr, change_type, final_lr, at_step))
            i += 4  # skip to the next relevant segment
        i += 1

    def schedule(step):
        lr = starter_lr  # Default to the initial LR in the first segment

        for index, (start_step, start_lr, change_type, end_lr, change_step) in enumerate(changes):
            if change_type == 'jump':
                lr = jnp.where(step >= start_step, start_lr, lr)
                if index + 1 == len(changes):
                    lr = jnp.where(step >= change_step, end_lr, lr)
            elif change_type == 'anneal':
                duration = change_step - start_step
                fraction = (step - start_step) / duration
                lr = jnp.where(step >= start_step, jnp.interp(fraction, jnp.array([0, 1]), jnp.array([start_lr, end_lr])), lr)

        return lr

    return schedule


####### Saving agents

def save_model(train_state, model_name):
    checkpoint = {'model': train_state}
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    save_args = orbax_utils.save_args_from_target(checkpoint)
    # local_path = str(pathlib.Path().resolve())
    orbax_checkpointer.save(model_name, checkpoint, save_args=save_args)


def save_agents(listener_train_states, speaker_train_states, config):
    local_path = str(pathlib.Path().resolve())
    model_path_str = "/base_experiment/models/" if config["DEBUGGER"] else "/models/"
    model_uuid = str(uuid.uuid4())[:4]

    agent_logdir = local_path+model_path_str+'agents-'+f'{config["ENV_DATASET"]}-{config["WANDB_RUN_NAME"]}-{config["UPDATE_EPOCHS"]}e-{config["ENV_NUM_DATAPOINTS"]}dp-'+model_uuid+'/'
    os.makedirs(agent_logdir, exist_ok=True)

    for i, lts in enumerate(listener_train_states):
        save_model(lts, f'{agent_logdir}listener_{i}.agent')
        # with open(f'{agent_logdir}listener_{i}.pkl', 'wb') as f:
        #     cloudpickle.dump(lts, f)

    for i, sts in enumerate(speaker_train_states):
        save_model(sts, f'{agent_logdir}speaker_{i}.agent')
        # with open(f'{agent_logdir}speaker_{i}.pkl', 'wb') as f:
        #     cloudpickle.dump(sts, f)
    
    with open(f'{agent_logdir}config.yaml', 'w') as f:
        OmegaConf.save(config=config, f=f)


##################################################################

############ Used in simplified_signification_game.py ############

def get_channel_ratio_fn(phrase, params):
    # TODO: Eventually I will parameterize this better.
    def s_curve(x):
        return 1.0 / (1.0 + jnp.exp(-1 * (0.01*jnp.array(x, float) - 5))) + 1e-2
    
    def linear(x):
        return x / 400.0

    def get_sigmoid(sigmoid_offset, sigmoid_stretch, sigmoid_height, **kwargs):
        def sig_ch_fn(x):
            return sigmoid_height / (1.0 + jnp.exp(-1 * jnp.array(sigmoid_stretch, float) * (jnp.array(x, float) - jnp.array(sigmoid_offset, float)))) + 1e-2
        return sig_ch_fn

    if phrase in ("all_env", "ret_0", "ret0"):
        return lambda _: 0.0
    elif phrase in ("all_speakers", "all_speaker", "ret_1", "ret1"):
        return lambda _: 1.0
    elif phrase in ("half", "0.5"):
        return lambda _: 0.5
    elif phrase == "sigmoid1":
        return s_curve
    elif phrase == "linear":
        return linear
    elif phrase == "sigmoid-custom":
        return get_sigmoid(**params)
    elif phrase.startswith("sigmoid-custom-cutoff"):
        fn = get_sigmoid(**params)
        return lambda x: jax.lax.cond(x < eval(phrase.split("-")[-1]), fn, lambda _: 0.0, operand=x)
    else:
        if " at " in phrase:
            crf_params = phrase.split(" at ")
            return lambda x: jax.lax.cond(x < eval(crf_params[1]), lambda _: 0.0, lambda _: eval(crf_params[0]), operand=None)
        else:
            return lambda x: float(phrase)
        
def get_speaker_referent_span_fn(phrase, params):
    # This function returns a function over epochs that controls which referents we ask speakers to generate
    # The function outputs an integer, and the environment will sample referents between 0 and that integer

    # TODO: Flesh out this function

    return lambda x: int(phrase)

def get_reward_parity_fn(phrase, params):
    if phrase in ("coop", "cooperative", "symmetric"):
        return lambda _: 1.0
    elif phrase in ("manip", "adversarial", "asymmetric"):
        return lambda _: 0.0
    elif " at " in phrase:
        crf_params = phrase.split(" at ")
        return lambda x: jax.lax.cond(x < eval(crf_params[1]), lambda _: 0.0, lambda _: eval(crf_params[0]), operand=None)  # Assumed starting with manipulation
    
def get_agent_inferential_mode_fn(phrase, params):
    # This function returns a function over epochs that determines whether agents will use ToM or not (gut)
    # The function outputs a boolean, 0 for gut and 1 for ToM
    if phrase in ("gut", "reflexive"):
        return lambda _: 0
    elif phrase in ("tom", "ToM", "theory-of-mind"):
        return lambda _: 1
    elif " at " in phrase:
        crf_params = phrase.split(" at ")
        return lambda x: jax.lax.cond(x < eval(crf_params[1]), lambda _: 0, lambda _: 1, operand=None)  # Assumed starting with gut

@jax.jit
def create_unitary_channel_map(list1, list2, key):  # NOTE: This simply doesn't work. It's not easy to do this in a jittable fashion.
    """
    Creates a 2D array of channels where:
    - The first column is from list1.
    - The second column is from list2, shuffled to ensure no row has duplicates.

    Args:
        list1: JAX array of indices for the first column.
        list2: JAX array of indices for the second column.
        key: JAX random key for randomness.

    Returns:
        A JAX array with shape (n, 2) meeting the conditions.
    """
    # assert list1.shape == list2.shape, "Input lists must have the same size."
    n = list1.shape[0]

    # Shuffle list2 randomly
    shuffled_list2 = jax.random.permutation(key, list2)

    def resolve_conflicts(i, current):
        """
        Swap conflicting elements in shuffled_list2 with the next available index.
        """
        current_list2 = current["list2"]
        conflict = list1[i] == current_list2[i]

        def swap_indices(c):
            # Find the first valid index to swap
            available_indices = jnp.where(current_list2 != list1[i], size=n)[0]
            swap_idx = available_indices[0]  # Take the first valid swap
            swapped = current_list2.at[i].set(current_list2[swap_idx])
            swapped = swapped.at[swap_idx].set(current_list2[i])
            return {"list2": swapped}

        # Swap if conflict exists
        return jax.lax.cond(conflict, swap_indices, lambda x: x, current)

    # Iteratively resolve conflicts using lax.fori_loop
    result = jax.lax.fori_loop(0, n, resolve_conflicts, {"list2": shuffled_list2})

    # Stack the columns
    return jnp.stack([list1, result["list2"]], axis=1)

@jax.vmap
def speaker_penalty_whitesum_fn(images: jnp.array):
    # More white in the image corresponds to a higher penalty. Bounded between 0 and 1.
    return jnp.sum(images) / (images.shape[0] * images.shape[1])

@jax.vmap
def speaker_penalty_curve_fn(speaker_actions: jnp.array):
    @jax.vmap
    def bezier_curvature(spline_params: jnp.array):
        """
        Calculate a scalar value representing the "curvedness" of the Bézier spline.
        
        Args:
        P0, P1, P2 : numpy arrays of shape (2,)
            Control points of the Bézier spline, normalized between 0 and 1.
        
        Returns:
        curvature : float
            A scalar value representing the curvedness of the spline.
        """
        P0, P1, P2 = spline_params[0:2], spline_params[2:4], spline_params[4:6]
        # Vector from P0 to P2
        P0_P2 = P2 - P0
        
        # Vector from P0 to P1
        P0_P1 = P1 - P0
        
        # Project P0_P1 onto P0_P2 to find the closest point on the line P0P2 to P1
        proj_length = jnp.dot(P0_P1, P0_P2) / jnp.maximum(jnp.dot(P0_P2, P0_P2), 1e-4)
        closest_point = P0 + proj_length * P0_P2
        
        # Distance from P1 to the closest point on the line
        distance_to_line = jnp.linalg.norm(P1 - closest_point)
        
        # Normalize the curvature by the length of the line segment P0P2
        line_length = jnp.linalg.norm(P0_P2)
        curvature = distance_to_line / jnp.maximum(line_length, 1e-4)
        
        return jnp.nan_to_num(curvature)

    curve_penalty_per_spline = bezier_curvature(speaker_actions.reshape(-1, speaker_actions.shape[-1]))

    return jnp.average(curve_penalty_per_spline, axis=0)

@jax.vmap
def center_obs(image: jnp.array):
    image_shape = image.shape 
    center = jnp.array(image_shape) / 2  # Center of the image

    # Compute the center of mass
    coords = jnp.meshgrid(jnp.arange(image_shape[0]), jnp.arange(image_shape[1]), indexing='ij')
    coords = jnp.array(coords)
    center_of_mass = jnp.sum(coords * image[None, :, :], axis=(1, 2)) / jnp.sum(image)

    # Compute the translation
    translation = center - center_of_mass
    
    # Shift the image based on the computed translation
    recentered_image = jax.image.scale_and_translate(image, image_shape, (0, 1), jnp.array([1.0, 1.0]), translation, method="linear")
    
    return recentered_image

        
def get_speaker_action_transform(fn_name, image_dim):
    @jax.vmap
    def identity(actions: jnp.array):
        return actions
    
    @jax.vmap
    def image(actions: jnp.array):
        return actions.reshape(image_dim, image_dim)
    
    @jax.vmap
    def gauss_splat(actions: jnp.array):
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

            @jax.vmap
            def compute_gaussian(params):
                x_mu_norm, y_mu_norm, sigma_x2_norm, sigma_y2_norm, amplitude = params
                
                # Convert normalized mean to actual coordinates
                x_mu = x_mu_norm * array_shape[1]
                y_mu = y_mu_norm * array_shape[0]

                # Convert normalized covariance to actual variances
                sigma_x2 = sigma_x2_norm * array_shape[1]**2
                sigma_y2 = sigma_y2_norm * array_shape[0]**2

                # Compute the 2D Gaussian formula
                gaussian = amplitude * jnp.exp(-(((x - x_mu)**2 / (2 * sigma_x2 + 1e-8)) + ((y - y_mu)**2 / (2 * sigma_y2 + 1e-8))))
                return gaussian

            array += compute_gaussian(gaussians_params)
            array = jnp.sum(array, axis=0)

            return jnp.nan_to_num(jnp.clip(array, a_min=0.0, a_max=1.0))

        gaussians_params = actions.reshape(-1, 5)
        array_shape = (image_dim, image_dim)

        gaussian_array = paint_normalized_gaussians_on_array(array_shape, gaussians_params)
        return gaussian_array
    
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
                sigma_x2 = sigma_x2_norm * array_shape[1]**2 * 0.01 + 1e-8
                sigma_y2 = sigma_y2_norm * array_shape[0]**2 * 0.01 + 1e-8
                sigma_xy = (2 * sigma_xy_norm - 1) * array_shape[1] * array_shape[0] * 0.002 + 1e-8

                # Construct the covariance matrix and its inverse
                cov_matrix = jnp.array([[sigma_x2, sigma_xy], [sigma_xy, sigma_y2]])
                inv_cov_matrix = jnp.linalg.inv(cov_matrix)

                # Compute the Gaussian function
                X = jnp.vstack((x.ravel() - x_mu, y.ravel() - y_mu))
                gaussian = amplitude * jnp.exp(-0.5 * jnp.sum(X.T @ inv_cov_matrix * X.T, axis=1)).reshape(array_shape)
                
                return gaussian

            gaussians = compute_gaussian(gaussians_params)
            array += jnp.sum(gaussians, axis=0)  # Sum contributions from all Gaussians

            return jnp.nan_to_num(jnp.clip(array, a_min=0.0, a_max=1.0))

        # Assuming 'actions' includes the additional covariance parameters
        gaussians_params = actions.reshape(-1, 6)  # Reshape based on the new parameter structure
        image_dim = 28  # As per your indication, assuming a fixed image dimension
        array_shape = (image_dim, image_dim)

        gaussian_array = paint_normalized_gaussians_on_array(array_shape, gaussians_params)
        return gaussian_array

    @jax.vmap
    def gauss_splat_chol(actions: jnp.array):
        def paint_normalized_gaussians_on_array(array_shape, gaussians_params):
            """
            Paint multiple 2D Gaussians on a 2D array, parameters defined via Cholesky decomposition.
            
            Parameters:
            - array_shape: tuple of int, shape of the 2D array (height, width).
            - gaussians_params: JAX array with each row representing the parameters for a Gaussian 
            (normalized mean x, normalized mean y, amplitude, L_{11}, L_{21}, L_{22}).
            
            Returns:
            - 2D JAX array with the Gaussians painted on it.
            """
            y, x = jnp.indices(array_shape)  # Create a grid of x and y coordinates
            array = jnp.zeros(array_shape)

            @jax.vmap
            def compute_gaussian(params):
                x_mu_norm, y_mu_norm, amplitude, L_11, L_21, L_22 = params

                amplitude = (amplitude * 0.5) + 1
                L_11 *= 0.15
                L_22 *= 0.15
                L_21 = ((2 * L_21) - 1) * 0.005
                
                # Convert normalized mean to actual coordinates
                x_mu = x_mu_norm * array_shape[1]
                y_mu = y_mu_norm * array_shape[0]

                # Construct the covariance matrix from Cholesky decomposition
                L = jnp.array([[L_11, 0], [L_21, L_22]])
                cov_matrix = L @ L.T

                # Convert normalized covariance matrix to actual values
                cov_matrix = cov_matrix * jnp.array([[array_shape[1]**2, array_shape[1]*array_shape[0]], 
                                                    [array_shape[1]*array_shape[0], array_shape[0]**2]])
                inv_cov_matrix = jnp.linalg.inv(cov_matrix)

                # Compute the Gaussian function
                X = jnp.vstack((x.ravel() - x_mu, y.ravel() - y_mu))
                gaussian = amplitude * jnp.exp(-0.5 * jnp.sum(X.T @ inv_cov_matrix * X.T, axis=1)).reshape(array_shape)
                
                return gaussian

            gaussians = compute_gaussian(gaussians_params)
            array += jnp.sum(gaussians, axis=0)  # Sum contributions from all Gaussians

            return jnp.nan_to_num(jnp.clip(array, a_min=0.0, a_max=1.0))

        # Assuming 'actions' includes the Cholesky decomposition parameters
        gaussians_params = actions.reshape(-1, 6)  # Reshape based on the new parameter structure
        image_dim = 28  # Fixed image dimension
        array_shape = (image_dim, image_dim)

        gaussian_array = paint_normalized_gaussians_on_array(array_shape, gaussians_params)
        return gaussian_array

    # TODO: Make a new function, paint_multiple_splines_and_center
    @jax.vmap
    def paint_multiple_splines(all_spline_params: jnp.array):
        """Paint multiple splines on a single canvas. Requires speaker_action_dim be a multiple of 6."""

        @jax.vmap
        def paint_spline_on_canvas(spline_params: jnp.array):
            """Paint a single spline on the canvas with specified thickness using advanced indexing."""

            def bezier_spline(t, P0, P1, P2):
                """Compute points on a quadratic Bézier spline for a given t."""
                t = t[:, None]  # Shape (N, 1) to broadcast with P0, P1, P2 of shape (2,)
                P = (1 - t)**2 * P0 + 2 * (1 - t) * t * P1 + t**2 * P2
                return P  # Returns shape (N, 2), a list of points on the spline

            brush_size = 1

            spline_params *= image_dim
            
            # P0, P1, P2 = spline_params.reshape((3, 2)) 
            P0, P1, P2 = spline_params[0:2], spline_params[2:4], spline_params[4:6]
            t_values = jnp.linspace(0, 1, num=50)
            spline_points = bezier_spline(t_values, P0, P1, P2)
            x_points, y_points = jnp.round(spline_points).astype(int).T

            # Generate brush offsets
            brush_offsets = jnp.array([(dx, dy) for dx in range(-brush_size, brush_size)    # brush_size + 1
                                                for dy in range(-brush_size, brush_size)])  # brush_size + 1
            x_offsets, y_offsets = brush_offsets.T

            # Calculate all indices to update for each point (broadcasting magic)
            all_x_indices = x_points[:, None] + x_offsets
            all_y_indices = y_points[:, None] + y_offsets

            # Flatten indices and filter out-of-bound ones
            all_x_indices = jnp.clip(all_x_indices.flatten(), 0, image_dim)
            all_y_indices = jnp.clip(all_y_indices.flatten(), 0, image_dim)

            # Update the canvas
            canvas = jnp.zeros((image_dim, image_dim))
            canvas = canvas.at[all_x_indices, all_y_indices].add(1)
            return canvas

        # Vmap over splines and sum contributions
        all_spline_params = jnp.clip(all_spline_params, 0.0, 1.0)
        canvas = jnp.clip(paint_spline_on_canvas(all_spline_params.reshape(-1, 6)).sum(axis=0), 0.0, 1.0)
        return canvas
    
    @jax.vmap
    def paint_multiple_splines_with_intensity(all_spline_params: jnp.array):
        """Paint multiple splines on a single canvas. Requires speaker_action_dim be a multiple of 7."""

        @jax.vmap
        def paint_spline_on_canvas(spline_params: jnp.array):
            """Paint a single spline on the canvas with specified thickness using advanced indexing."""

            def bezier_spline(t, P0, P1, P2):
                """Compute points on a quadratic Bézier spline for a given t."""
                t = t[:, None]  # Shape (N, 1) to broadcast with P0, P1, P2 of shape (2,)
                P = (1 - t)**2 * P0 + 2 * (1 - t) * t * P1 + t**2 * P2
                return P  # Returns shape (N, 2), a list of points on the spline

            brush_size = 1

            spline_params *= image_dim
            
            # P0, P1, P2 = spline_params.reshape((3, 2)) 
            P0, P1, P2, W = spline_params[0:2], spline_params[2:4], spline_params[4:6], spline_params[6]
            W *= -0.003  # This is the weight param. -0.005 is too dark. -0.002 may be too light.
            t_values = jnp.linspace(0, 1, num=50)
            spline_points = bezier_spline(t_values, P0, P1, P2)
            x_points, y_points = jnp.round(spline_points).astype(int).T

            # Generate brush offsets
            brush_offsets = jnp.array([(dx, dy) for dx in range(-brush_size, brush_size)    # brush_size + 1
                                                for dy in range(-brush_size, brush_size)])  # brush_size + 1
            x_offsets, y_offsets = brush_offsets.T

            # Calculate all indices to update for each point (broadcasting magic)
            all_x_indices = x_points[:, None] + x_offsets
            all_y_indices = y_points[:, None] + y_offsets

            # Flatten indices and filter out-of-bound ones
            all_x_indices = jnp.clip(all_x_indices.flatten(), 0, image_dim)
            all_y_indices = jnp.clip(all_y_indices.flatten(), 0, image_dim)

            # Update the canvas
            canvas = jnp.ones((image_dim, image_dim)) * 0.2 # This is the background color!
            canvas = canvas.at[all_x_indices, all_y_indices].add(W)
            return canvas

        # Vmap over splines and sum contributions
        all_spline_params = jnp.clip(all_spline_params, 0.0, 1.0)
        canvas = jnp.clip(paint_spline_on_canvas(all_spline_params.reshape(-1, 7)).sum(axis=0), 0.0, 1.0)
        return canvas

    if fn_name == "identity":
        return identity
    elif fn_name == "image":
        return image
    elif fn_name == "gauss_splat":
        return gauss_splat
    elif fn_name == "gauss_splatcovar":
        return gauss_splat_covar
    elif fn_name == "gauss_splatchol":
        return gauss_splat_chol
    elif fn_name == "splines":
        return paint_multiple_splines
    elif fn_name == "splines_weight":
        return paint_multiple_splines_with_intensity


##################################################################

def make_grid_jnp(images, nrow=8, padding=1, pad_value=0.0):
    """
    Create a grid of images using JAX.
    
    Args:
        images: jnp.array of shape (N, C, H, W) - batch of images.
        nrow: Number of images per row.
        padding: Padding between images.
        pad_value: Value to use for padding.
    
    Returns:
        jnp.array: Grid image of shape (C, H_grid, W_grid)
    """
    N, C, H, W = images.shape  # Extract dimensions
    
    # Calculate number of rows in the grid
    nrows = (N + nrow - 1) // nrow  # Ensure all images fit in the grid

    # Pad the batch to fit exactly into the grid
    pad_images = jnp.pad(
        images,
        ((0, nrow * nrows - N), (0, 0), (padding, padding), (padding, padding)),
        mode='constant',
        constant_values=pad_value
    )
    
    # Reshape into a grid
    pad_images = pad_images.reshape(nrows, nrow, C, H + 2 * padding, W + 2 * padding)
    
    # Rearrange axes to stack rows and columns
    pad_images = pad_images.transpose(0, 3, 1, 4, 2)  # (nrows, H+pad, ncols, W+pad, C)
    grid = pad_images.reshape(nrows * (H + 2 * padding), nrow * (W + 2 * padding), C)
    
    return grid.transpose(2, 0, 1)  # Return in (C, H_grid, W_grid)

if __name__ == "__main__":
    # Step 1: Download MNIST Dataset
    # mnist = datasets.MNIST(root='/tmp/mnist/', download=True)

    # # Step 2: Convert to Jax arrays
    # images, labels = to_jax(mnist, num_datapoints=100)
    # print(images.shape)

    image = np.zeros((1, 28, 28))

    image[0][5][5] = 100

    image = jnp.array(image)

    new_image = center_obs(image)
    print(new_image)
