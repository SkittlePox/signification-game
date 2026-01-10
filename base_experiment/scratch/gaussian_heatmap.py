import jax
import jax.numpy as jnp
from jax.scipy.stats import norm
import numpy as np
from PIL import Image

def create_gaussian_heatmap(means, variances, width=256, x_min=0.0, x_max=1.0):
    """
    Create a heatmap of Gaussian PDFs using JAX.
    
    Args:
        means: shape (210,) - means of each Gaussian
        variances: shape (210,) - variances of each Gaussian
        width: number of pixels for x-axis (default 256)
        x_min, x_max: range of x values (default 0 to 1)
    
    Returns:
        heatmap: shape (210, 256) - normalized PDF values for visualization
    """
    # Convert to JAX arrays if needed
    means = jnp.asarray(means)
    variances = jnp.asarray(variances)
    
    # Create x grid
    x = jnp.linspace(x_min, x_max, width)
    
    # Reshape for broadcasting: x is (1, 256), means is (210, 1), etc.
    x_expanded = x[None, :]  # (1, 256)
    means_expanded = means[:, None]  # (210, 1)
    stds_expanded = jnp.sqrt(variances)[:, None]  # (210, 1)
    
    # Compute all PDFs at once using broadcasting
    pdfs = norm.pdf(x_expanded, loc=means_expanded, scale=stds_expanded)
    
    # Normalize each row to [0, 1] for better visualization
    row_min = pdfs.min(axis=1, keepdims=True)
    row_max = pdfs.max(axis=1, keepdims=True)
    normalized = (pdfs - row_min) / (row_max - row_min + 1e-8)
    
    return normalized

# Create a vmapped version that works on batches
create_gaussian_heatmap_batch = jax.vmap(
    create_gaussian_heatmap,
    in_axes=(0, 0, None, None, None),  # vmap over means and variances, keep other args fixed
    out_axes=0  # stack outputs along axis 0
)

def save_heatmap_as_image(heatmap, filename='gaussian_heatmap.png'):
    """Save the heatmap as a PNG image."""
    # Convert JAX array to NumPy first
    heatmap_np = np.array(heatmap)
    
    # Convert to uint8 (0-255 range)
    img_array = (heatmap_np * 255).astype(np.uint8)
    
    # Create PIL image in grayscale
    img = Image.fromarray(img_array, mode='L')
    img.save(filename)
    return img

# Test script
if __name__ == "__main__":
    print("Testing Gaussian heatmap generation...")
    
    # Test 1: Create diverse Gaussians with patterns
    key = jax.random.PRNGKey(42)
    
    # Create 210 Gaussians with some structure
    means = jnp.zeros(210)
    variances = jnp.zeros(210)
    
    # Group 1: Low mean, low variance (indices 0-70)
    key, subkey = jax.random.split(key)
    means = means.at[:70].set(jax.random.uniform(subkey, shape=(70,), minval=0.1, maxval=0.3))
    variances = variances.at[:70].set(jax.random.uniform(subkey, shape=(70,), minval=0.001, maxval=0.01))
    
    # Group 2: Middle mean, medium variance (indices 70-140)
    key, subkey = jax.random.split(key)
    means = means.at[70:140].set(jax.random.uniform(subkey, shape=(70,), minval=0.4, maxval=0.6))
    variances = variances.at[70:140].set(jax.random.uniform(subkey, shape=(70,), minval=0.01, maxval=0.03))
    
    # Group 3: High mean, mixed variance (indices 140-210)
    key, subkey = jax.random.split(key)
    means = means.at[140:210].set(jax.random.uniform(subkey, shape=(70,), minval=0.7, maxval=0.9))
    variances = variances.at[140:210].set(jax.random.uniform(subkey, shape=(70,), minval=0.005, maxval=0.04))
    
    # Generate heatmap
    heatmap = create_gaussian_heatmap(means, variances)
    
    # Basic assertions
    assert heatmap.shape == (210, 256), f"Expected shape (210, 256), got {heatmap.shape}"
    assert jnp.all(heatmap >= 0) and jnp.all(heatmap <= 1), "Values should be normalized to [0,1]"
    print(f"✓ Shape correct: {heatmap.shape}")
    print(f"✓ Values normalized: min={heatmap.min():.4f}, max={heatmap.max():.4f}")
    
    # Test 2: Edge cases
    print("\nTesting edge cases...")
    
    # Very narrow Gaussians (small variance)
    means_narrow = jnp.array([0.25, 0.5, 0.75])
    vars_narrow = jnp.array([0.0001, 0.0001, 0.0001])
    heatmap_narrow = create_gaussian_heatmap(means_narrow, vars_narrow, width=256)
    assert heatmap_narrow.shape == (3, 256), "Shape mismatch for narrow Gaussians"
    print(f"✓ Narrow Gaussians: shape {heatmap_narrow.shape}")
    
    # Very wide Gaussians (large variance)
    means_wide = jnp.array([0.25, 0.5, 0.75])
    vars_wide = jnp.array([0.1, 0.1, 0.1])
    heatmap_wide = create_gaussian_heatmap(means_wide, vars_wide, width=256)
    assert heatmap_wide.shape == (3, 256), "Shape mismatch for wide Gaussians"
    print(f"✓ Wide Gaussians: shape {heatmap_wide.shape}")
    
    # Test 3: Performance check
    import time
    
    # Generate large batch
    key, subkey = jax.random.split(key)
    means_perf = jax.random.uniform(subkey, shape=(210,), minval=0.1, maxval=0.9)
    vars_perf = jax.random.uniform(subkey, shape=(210,), minval=0.001, maxval=0.05)
    
    # Time the generation (including JIT compilation)
    start = time.time()
    heatmap_perf = create_gaussian_heatmap(means_perf, vars_perf)
    first_time = time.time() - start
    
    # Second run (should be faster due to JIT)
    start = time.time()
    heatmap_perf = create_gaussian_heatmap(means_perf, vars_perf)
    second_time = time.time() - start
    
    print(f"\nPerformance:")
    print(f"✓ First run (with JIT compilation): {first_time:.4f} seconds")
    print(f"✓ Second run (JIT cached): {second_time:.4f} seconds")
    print(f"✓ Speedup: {first_time/second_time:.1f}x")
    
    # Test 4: Verify PDF properties
    print("\nVerifying PDF properties...")
    
    # Check that peak is roughly at the mean
    test_mean = 0.5
    test_var = 0.01
    single_gaussian = create_gaussian_heatmap(
        jnp.array([test_mean]), 
        jnp.array([test_var]),
        width=256
    )
    peak_idx = jnp.argmax(single_gaussian[0])
    peak_x = peak_idx / 256.0  # Convert to [0, 1] range
    print(f"✓ Peak location: {peak_x:.3f} (expected near {test_mean})")
    assert abs(peak_x - test_mean) < 0.05, "Peak should be near the mean"
    
    # Test 5: VMAP BATCH PROCESSING
    print("\n=== Testing vmap batch processing ===")
    
    # Create multiple batches of Gaussian parameters
    batch_size = 5
    n_gaussians = 210
    
    key, subkey = jax.random.split(key)
    # Shape: (batch_size, n_gaussians)
    batch_means = jax.random.uniform(subkey, shape=(batch_size, n_gaussians), minval=0.1, maxval=0.9)
    
    key, subkey = jax.random.split(key)
    batch_variances = jax.random.uniform(subkey, shape=(batch_size, n_gaussians), minval=0.001, maxval=0.05)
    
    print(f"Input batch shapes:")
    print(f"  Means: {batch_means.shape}")
    print(f"  Variances: {batch_variances.shape}")
    
    # Process all batches at once using vmap
    start = time.time()
    batch_heatmaps = create_gaussian_heatmap_batch(batch_means, batch_variances, 256, 0.0, 1.0)
    vmap_time = time.time() - start
    
    print(f"✓ Vmapped output shape: {batch_heatmaps.shape}")
    assert batch_heatmaps.shape == (batch_size, n_gaussians, 256), \
        f"Expected shape ({batch_size}, {n_gaussians}, 256), got {batch_heatmaps.shape}"
    
    # Verify each batch is properly normalized
    for i in range(batch_size):
        batch_min = batch_heatmaps[i].min()
        batch_max = batch_heatmaps[i].max()
        assert batch_min >= 0 and batch_max <= 1, \
            f"Batch {i} not normalized: min={batch_min}, max={batch_max}"
    print(f"✓ All batches properly normalized")
    
    # Compare with sequential processing
    start = time.time()
    sequential_results = []
    for i in range(batch_size):
        result = create_gaussian_heatmap(batch_means[i], batch_variances[i])
        sequential_results.append(result)
    sequential_results = jnp.stack(sequential_results)
    sequential_time = time.time() - start
    
    # Check that results match
    assert jnp.allclose(batch_heatmaps, sequential_results, atol=1e-6), \
        "Vmapped results don't match sequential processing"
    print(f"✓ Vmapped results match sequential processing")
    
    print(f"\nPerformance comparison:")
    print(f"  Vmapped time: {vmap_time:.4f} seconds")
    print(f"  Sequential time: {sequential_time:.4f} seconds")
    print(f"  Speedup: {sequential_time/vmap_time:.2f}x")
    
    # Save one batch as an example
    save_heatmap_as_image(batch_heatmaps[0], 'batch_example.png')
    print(f"\n✓ Saved first batch to 'batch_example.png'")
    
    # Test 6: Different batch configurations
    print("\n=== Testing different batch configurations ===")
    
    # Small batch, many Gaussians
    small_batch_means = jax.random.uniform(key, shape=(2, 500), minval=0.1, maxval=0.9)
    small_batch_vars = jax.random.uniform(key, shape=(2, 500), minval=0.001, maxval=0.05)
    small_batch_result = create_gaussian_heatmap_batch(small_batch_means, small_batch_vars, 256, 0.0, 1.0)
    assert small_batch_result.shape == (2, 500, 256)
    print(f"✓ Small batch (2x500): {small_batch_result.shape}")
    
    # Large batch, few Gaussians
    large_batch_means = jax.random.uniform(key, shape=(20, 50), minval=0.1, maxval=0.9)
    large_batch_vars = jax.random.uniform(key, shape=(20, 50), minval=0.001, maxval=0.05)
    large_batch_result = create_gaussian_heatmap_batch(large_batch_means, large_batch_vars, 256, 0.0, 1.0)
    assert large_batch_result.shape == (20, 50, 256)
    print(f"✓ Large batch (20x50): {large_batch_result.shape}")
    
    print("\nAll tests passed! ✓")
