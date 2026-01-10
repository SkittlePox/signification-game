import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

def variance_weighted_wasserstein_v1(mu1, sigma1, mu2, sigma2, alpha=1.0):
    """Original version - has the problem you identified"""
    w2_squared = (mu1 - mu2)**2 + (sigma1 - sigma2)**2
    combined_variance = jnp.maximum(sigma1**2 + sigma2**2, 1e-10)
    weighted_distance = w2_squared / (combined_variance**alpha)
    return jnp.sqrt(weighted_distance)

def variance_weighted_wasserstein_v2(mu1, sigma1, mu2, sigma2, alpha=1.0):
    """Alternative: Weight by inverse of minimum variance"""
    w2_squared = (mu1 - mu2)**2 + (sigma1 - sigma2)**2
    min_variance = jnp.maximum(jnp.minimum(sigma1**2, sigma2**2), 1e-10)
    weighted_distance = w2_squared / (min_variance**alpha)
    return jnp.sqrt(weighted_distance)

def variance_weighted_wasserstein_v3(mu1, sigma1, mu2, sigma2, alpha=1.0):
    """Alternative: Multiplicative penalty for low variance"""
    w2 = jnp.sqrt((mu1 - mu2)**2 + (sigma1 - sigma2)**2)
    avg_variance = (sigma1**2 + sigma2**2) / 2
    # Penalty factor that increases distance for small variances
    penalty = (1 / (avg_variance + 0.01))**alpha
    return w2 * penalty

def variance_weighted_wasserstein_v4(mu1, sigma1, mu2, sigma2, beta=1.0):
    """Alternative: Additive penalty based on precision"""
    w2 = jnp.sqrt((mu1 - mu2)**2 + (sigma1 - sigma2)**2)
    # Add a term that increases with precision (1/variance)
    precision_term = beta * jnp.sqrt(1/(sigma1**2 + 0.01) + 1/(sigma2**2 + 0.01))
    return w2 + w2 * precision_term

def test_variance_scaling():
    """Test how distances behave when scaling variances"""
    
    print("="*60)
    print("TESTING VARIANCE SCALING BEHAVIOR")
    print("="*60)
    print("\nCase 1: Same means, different variances")
    print("-"*40)
    
    # Test different pairs with same mean difference but scaled variances
    test_cases = [
        (0.5, 0.1, 0.5, 0.2),   # Small variances
        (0.5, 0.2, 0.5, 0.4),   # Medium variances (2x scale)
        (0.5, 0.4, 0.5, 0.8),   # Large variances (4x scale)
    ]
    
    print("\nVersion 1 (Original - divides by combined variance):")
    for mu1, sig1, mu2, sig2 in test_cases:
        dist = variance_weighted_wasserstein_v1(mu1, sig1, mu2, sig2, alpha=1.0)
        print(f"  σ₁={sig1:.1f}, σ₂={sig2:.1f}: distance = {dist:.4f}")
    
    print("\nVersion 2 (Divides by minimum variance):")
    for mu1, sig1, mu2, sig2 in test_cases:
        dist = variance_weighted_wasserstein_v2(mu1, sig1, mu2, sig2, alpha=1.0)
        print(f"  σ₁={sig1:.1f}, σ₂={sig2:.1f}: distance = {dist:.4f}")
    
    print("\nVersion 3 (Multiplicative precision penalty):")
    for mu1, sig1, mu2, sig2 in test_cases:
        dist = variance_weighted_wasserstein_v3(mu1, sig1, mu2, sig2, alpha=0.5)
        print(f"  σ₁={sig1:.1f}, σ₂={sig2:.1f}: distance = {dist:.4f}")
    
    print("\nVersion 4 (Additive precision term):")
    for mu1, sig1, mu2, sig2 in test_cases:
        dist = variance_weighted_wasserstein_v4(mu1, sig1, mu2, sig2, beta=0.1)
        print(f"  σ₁={sig1:.1f}, σ₂={sig2:.1f}: distance = {dist:.4f}")
    
    print("\n" + "="*60)
    print("\nCase 2: Different means, same variances (should favor narrow)")
    print("-"*40)
    
    test_cases_2 = [
        (0.4, 0.1, 0.6, 0.1),   # Narrow distributions, Δμ=0.2
        (0.4, 0.3, 0.6, 0.3),   # Wide distributions, Δμ=0.2
    ]
    
    print("\nVersion 1 (Original):")
    for mu1, sig1, mu2, sig2 in test_cases_2:
        dist = variance_weighted_wasserstein_v1(mu1, sig1, mu2, sig2, alpha=1.0)
        print(f"  σ={sig1:.1f}, Δμ=0.2: distance = {dist:.4f}")
    
    print("\nVersion 2 (Min variance):")
    for mu1, sig1, mu2, sig2 in test_cases_2:
        dist = variance_weighted_wasserstein_v2(mu1, sig1, mu2, sig2, alpha=1.0)
        print(f"  σ={sig1:.1f}, Δμ=0.2: distance = {dist:.4f}")
    
    print("\nVersion 3 (Multiplicative):")
    for mu1, sig1, mu2, sig2 in test_cases_2:
        dist = variance_weighted_wasserstein_v3(mu1, sig1, mu2, sig2, alpha=0.5)
        print(f"  σ={sig1:.1f}, Δμ=0.2: distance = {dist:.4f}")
    
    print("\nVersion 4 (Additive):")
    for mu1, sig1, mu2, sig2 in test_cases_2:
        dist = variance_weighted_wasserstein_v4(mu1, sig1, mu2, sig2, beta=0.1)
        print(f"  σ={sig1:.1f}, Δμ=0.2: distance = {dist:.4f}")

def create_comparison_visualization():
    """Visualize how different versions behave"""
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    # Row 1: Same means, varying variance difference
    sigma1 = 0.2
    sigma2_range = np.linspace(0.2, 1.0, 50)
    
    for idx, (func, name, param, param_name) in enumerate([
        (variance_weighted_wasserstein_v1, "V1: Original", 1.0, "α"),
        (variance_weighted_wasserstein_v2, "V2: Min Variance", 1.0, "α"),
        (variance_weighted_wasserstein_v3, "V3: Multiplicative", 0.5, "α"),
        (variance_weighted_wasserstein_v4, "V4: Additive", 0.1, "β"),
    ]):
        ax = axes[0, idx]
        
        # Test with different base variance scales
        for scale in [0.5, 1.0, 2.0]:
            distances = [func(0.5, sigma1*scale, 0.5, s2*scale, param) 
                        for s2 in sigma2_range]
            ax.plot(sigma2_range*scale, distances, label=f'scale={scale}')
        
        ax.set_xlabel('σ₂')
        ax.set_ylabel('Distance')
        ax.set_title(f'{name} ({param_name}={param})\nμ₁=μ₂=0.5, varying σ₂')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Row 2: Different means, same variances
    mean_sep_range = np.linspace(0, 0.5, 50)
    
    for idx, (func, name, param, param_name) in enumerate([
        (variance_weighted_wasserstein_v1, "V1: Original", 1.0, "α"),
        (variance_weighted_wasserstein_v2, "V2: Min Variance", 1.0, "α"),
        (variance_weighted_wasserstein_v3, "V3: Multiplicative", 0.5, "α"),
        (variance_weighted_wasserstein_v4, "V4: Additive", 0.1, "β"),
    ]):
        ax = axes[1, idx]
        
        # Test with different variance levels
        for sigma in [0.05, 0.2, 0.4]:
            distances = [func(0.3, sigma, 0.3 + sep, sigma, param) 
                        for sep in mean_sep_range]
            ax.plot(mean_sep_range, distances, label=f'σ={sigma:.2f}')
        
        ax.set_xlabel('Mean Separation (Δμ)')
        ax.set_ylabel('Distance')
        ax.set_title(f'{name}\nσ₁=σ₂, varying Δμ')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Comparison of Variance-Weighted Distance Formulations', fontsize=14, y=1.02)
    plt.tight_layout()
    
    # Save the figure
    plt.savefig('wasserstein_variance_comparison.png', dpi=150, bbox_inches='tight')
    print("\nVisualization saved as 'wasserstein_variance_comparison.png'")

if __name__ == "__main__":
    test_variance_scaling()
    create_comparison_visualization()
