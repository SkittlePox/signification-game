import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

def variance_weighted_wasserstein_v1(mu1, sigma1, mu2, sigma2, alpha=0.5):
    """
    Wasserstein distance weighted by inverse of combined variance.
    alpha controls the strength of variance weighting.
    """
    # Standard 2-Wasserstein distance for 1D Gaussians
    w2_squared = (mu1 - mu2)**2 + (sigma1 - sigma2)**2
    
    # Combined variance (or you could use average variance)
    combined_variance = sigma1**2 + sigma2**2
    
    # Avoid division by zero for identical zero-variance distributions
    combined_variance = jnp.maximum(combined_variance, 1e-10)
    
    # Weight by inverse variance (smaller variance = higher weight = more similar)
    weighted_distance = w2_squared / (combined_variance**alpha)
    
    return jnp.sqrt(weighted_distance)

def variance_weighted_wasserstein_v2(mu1, sigma1, mu2, sigma2, alpha=1.0):
    """Alternative: Weight by inverse of minimum variance"""
    w2_squared = (mu1 - mu2)**2 + (sigma1 - sigma2)**2
    min_variance = jnp.maximum(jnp.minimum(sigma1**2, sigma2**2), 1e-10)
    weighted_distance = w2_squared / (min_variance**alpha)
    return jnp.sqrt(weighted_distance)

def variance_weighted_wasserstein_v3(mu1, sigma1, mu2, sigma2, alpha=0.5):
    """
    Wasserstein distance weighted by inverse of combined variance.
    alpha controls the strength of variance weighting.
    """
    # Standard 2-Wasserstein distance for 1D Gaussians
    w2_squared = (mu1 - mu2)**2 + (sigma1 - sigma2)**2
    
    # Combined variance (or you could use average variance)
    combined_variance = sigma1**2 + sigma2**2
    
    # Avoid division by zero for identical zero-variance distributions
    combined_variance = jnp.maximum(combined_variance, 1e-10)
    
    # Weight by inverse variance (smaller variance = higher weight = more similar)
    weighted_distance = w2_squared / (combined_variance**alpha)

    weighted_distance = weighted_distance * (sigma1 + sigma2) / 2
    
    return jnp.sqrt(weighted_distance)

def variance_weighted_wasserstein_v4(mu1, sigma1, mu2, sigma2, alpha=2.0):
    """
    Wasserstein distance weighted by inverse of combined variance.
    alpha controls the strength of variance weighting.
    """
    # Standard 2-Wasserstein distance for 1D Gaussians
    w2_squared = (mu1 - mu2)**2 + (sigma1 + sigma2)**alpha
    
    # Combined variance (or you could use average variance)
    combined_variance = sigma1**2 + sigma2**2
    
    # Avoid division by zero for identical zero-variance distributions
    combined_variance = jnp.maximum(combined_variance, 1e-10)
    
    # Weight by inverse variance (smaller variance = higher weight = more similar)
    weighted_distance = w2_squared / (combined_variance**alpha)

    weighted_distance = weighted_distance * (sigma1 + sigma2) / 2
    
    return w2_squared

variance_weighted_wasserstein = variance_weighted_wasserstein_v4

def run_tests():
    """Test that variance-weighted Wasserstein distance has the desired properties."""
    
    print("="*60)
    print("TESTING VARIANCE-WEIGHTED WASSERSTEIN DISTANCE")
    print("="*60)
    
    # Test different alpha values
    alphas = [0.5, 1.0, 1.5]
    
    for alpha in alphas:
        print(f"\n{'='*60}")
        print(f"TESTING WITH ALPHA = {alpha}")
        print(f"{'='*60}")
        
        # Test 1: Identical distributions should have distance 0
        print("\n1. IDENTICAL DISTRIBUTIONS TEST")
        print("-" * 40)
        test_cases = [
            (0.5, 0.1, 0.5, 0.1, "Narrow identical"),
            (0.5, 0.5, 0.5, 0.5, "Wide identical"),
            (0.2, 0.3, 0.2, 0.3, "Random identical"),
        ]
        
        for mu1, sig1, mu2, sig2, desc in test_cases:
            dist = variance_weighted_wasserstein(mu1, sig1, mu2, sig2, alpha)
            print(f"{desc}: μ₁={mu1:.1f}, σ₁={sig1:.1f}, μ₂={mu2:.1f}, σ₂={sig2:.1f}")
            print(f"  Distance = {dist:.6f} (should be 0)")
            if jnp.abs(dist) >= 1e-10:
                print(f"  ❌ FAILED: {desc} should have distance 0")
        
        # Test 2: Narrow distributions should be more different when separated
        print("\n2. VARIANCE SENSITIVITY TEST")
        print("-" * 40)
        print("Testing: Narrow distributions separated by Δμ should be MORE different")
        print("         than wide distributions separated by same Δμ")
        
        separations = [0.1, 0.2, 0.5]
        for sep in separations:
            # Narrow gaussians separated
            narrow_dist = variance_weighted_wasserstein(0.3, 0.05, 0.3 + sep, 0.05, alpha)
            # Wide gaussians separated  
            wide_dist = variance_weighted_wasserstein(0.3, 0.3, 0.3 + sep, 0.3, alpha)
            
            print(f"\nΔμ = {sep:.1f}:")
            print(f"  Narrow (σ=0.05): distance = {narrow_dist:.4f}")
            print(f"  Wide   (σ=0.30): distance = {wide_dist:.4f}")
            print(f"  Ratio (narrow/wide) = {narrow_dist/wide_dist:.2f}x")
            
            if narrow_dist <= wide_dist:
                print(f"  ❌ FAILED: Narrow distributions should be more different when separated")
        
        # Test 3: Distance increases with mean separation
        print("\n3. MEAN SEPARATION TEST")
        print("-" * 40)
        print("Testing: Distance should increase with mean separation")
        
        sigma_fixed = 0.2
        mean_seps = [0.0, 0.1, 0.2, 0.3, 0.5]
        distances = []
        
        for sep in mean_seps:
            dist = variance_weighted_wasserstein(0.3, sigma_fixed, 0.3 + sep, sigma_fixed, alpha)
            distances.append(dist)
            print(f"Δμ = {sep:.1f}: distance = {dist:.4f}")
        
        # Check monotonic increase
        for i in range(1, len(distances)):
            if distances[i] <= distances[i-1]:
                print(f"  ❌ FAILED: Distance should increase with mean separation at index {i}")
        
        # Test 4: Distance increases with variance difference
        print("\n4. VARIANCE DIFFERENCE TEST")
        print("-" * 40)
        print("Testing: Distance should increase with variance difference")
        
        mu_fixed = 0.5
        sigma1_fixed = 0.2
        sigma2_values = [0.2, 0.25, 0.3, 0.35, 0.4]
        distances = []
        
        for sig2 in sigma2_values:
            dist = variance_weighted_wasserstein(mu_fixed, sigma1_fixed, mu_fixed, sig2, alpha)
            distances.append(dist)
            print(f"σ₁ = {sigma1_fixed:.2f}, σ₂ = {sig2:.2f}: distance = {dist:.4f}")
        
        # Check monotonic increase
        for i in range(1, len(distances)):
            if distances[i] <= distances[i-1]:
                print(f"  ❌ FAILED: Distance should increase with variance difference at index {i}")
        
        # Test 5: Symmetry
        print("\n5. SYMMETRY TEST")
        print("-" * 40)
        print("Testing: d(A,B) should equal d(B,A)")
        
        test_pairs = [
            (0.3, 0.1, 0.7, 0.2),
            (0.1, 0.3, 0.9, 0.1),
            (0.5, 0.5, 0.2, 0.05),
        ]
        
        for mu1, sig1, mu2, sig2 in test_pairs:
            dist_ab = variance_weighted_wasserstein(mu1, sig1, mu2, sig2, alpha)
            dist_ba = variance_weighted_wasserstein(mu2, sig2, mu1, sig1, alpha)
            print(f"d(μ={mu1:.1f},σ={sig1:.2f} → μ={mu2:.1f},σ={sig2:.2f}) = {dist_ab:.4f}")
            print(f"d(μ={mu2:.1f},σ={sig2:.2f} → μ={mu1:.1f},σ={sig1:.2f}) = {dist_ba:.4f}")
            if jnp.abs(dist_ab - dist_ba) >= 1e-10:
                print(f"  ❌ FAILED: Distance should be symmetric")
    
    print("\n" + "="*60)
    print("ALL TESTS COMPLETED!")
    print("="*60)
    
    # Visualization for alpha = 1.0
    create_visualization(alpha=1.2)

def create_visualization(alpha=1.0):
    """Create visualizations showing the distance behavior."""
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. Heatmap: Distance as function of two means (fixed small variance)
    ax = axes[0, 0]
    mus = np.linspace(0, 1, 50)
    M1, M2 = np.meshgrid(mus, mus)
    sigma_small = 0.1
    D = np.zeros_like(M1)
    for i in range(len(mus)):
        for j in range(len(mus)):
            D[i, j] = variance_weighted_wasserstein(M1[i, j], sigma_small, M2[i, j], sigma_small, alpha)
    
    im = ax.contourf(M1, M2, D, levels=20, cmap='viridis')
    ax.set_xlabel('μ₁')
    ax.set_ylabel('μ₂')
    ax.set_title(f'Distance vs Means (σ₁=σ₂={sigma_small})')
    plt.colorbar(im, ax=ax)
    
    # 2. Heatmap: Distance as function of two means (fixed large variance)
    ax = axes[0, 1]
    sigma_large = 0.4
    D = np.zeros_like(M1)
    for i in range(len(mus)):
        for j in range(len(mus)):
            D[i, j] = variance_weighted_wasserstein(M1[i, j], sigma_large, M2[i, j], sigma_large, alpha)
    
    im = ax.contourf(M1, M2, D, levels=20, cmap='viridis')
    ax.set_xlabel('μ₁')
    ax.set_ylabel('μ₂')
    ax.set_title(f'Distance vs Means (σ₁=σ₂={sigma_large})')
    plt.colorbar(im, ax=ax)
    
    # 3. Comparison of different alpha values
    ax = axes[0, 2]
    separations = np.linspace(0, 1, 50)
    alphas = [0.5, 1.0, 1.5, 2.0]
    sigma = 0.2
    
    for a in alphas:
        distances = [variance_weighted_wasserstein(0.3, sigma, 0.3 + sep, sigma, a) 
                    for sep in separations]
        ax.plot(separations, distances, label=f'α={a:.1f}')
    
    ax.set_xlabel('Mean Separation (Δμ)')
    ax.set_ylabel('Variance-Weighted Wasserstein')
    ax.set_title('Effect of α Parameter')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Line plot: Distance vs mean separation for different variances
    ax = axes[1, 0]
    separations = np.linspace(0, 1, 50)
    sigmas = [0.05, 0.1, 0.2, 0.3, 0.4]
    
    for sigma in sigmas:
        distances = [variance_weighted_wasserstein(0.3, sigma, 0.3 + sep, sigma, alpha) 
                    for sep in separations]
        ax.plot(separations, distances, label=f'σ={sigma:.2f}')
    
    ax.set_xlabel('Mean Separation (Δμ)')
    ax.set_ylabel('Variance-Weighted Wasserstein')
    ax.set_title(f'Distance vs Mean Separation (α={alpha})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 5. Heatmap: Distance as function of two variances (fixed means)
    ax = axes[1, 1]
    sigmas = np.linspace(0.01, 1, 50)
    S1, S2 = np.meshgrid(sigmas, sigmas)
    mu_sep = 0.3
    D = np.zeros_like(S1)
    for i in range(len(sigmas)):
        for j in range(len(sigmas)):
            D[i, j] = variance_weighted_wasserstein(0.6, S1[i, j], 0.6, S2[i, j], alpha)
    
    im = ax.contourf(S1, S2, D, levels=20, cmap='viridis')
    ax.set_xlabel('σ₁')
    ax.set_ylabel('σ₂')
    ax.set_title(f'Distance vs Std Devs (μ₁=0.6, μ₂=0.6, α={alpha})')
    plt.colorbar(im, ax=ax)
    
    # 6. Direct comparison: narrow vs wide distributions
    ax = axes[1, 2]
    separations = np.linspace(0, 1, 50)
    
    # Narrow distributions
    narrow_dists = [variance_weighted_wasserstein(0.3, 0.05, 0.3 + sep, 0.05, alpha) 
                   for sep in separations]
    # Wide distributions
    wide_dists = [variance_weighted_wasserstein(0.3, 0.3, 0.3 + sep, 0.3, alpha) 
                 for sep in separations]
    
    ax.plot(separations, narrow_dists, 'b-', label='Narrow (σ=0.05)', linewidth=2)
    ax.plot(separations, wide_dists, 'r-', label='Wide (σ=0.30)', linewidth=2)
    
    # Plot ratio
    ax2 = ax.twinx()
    ratio = np.array(narrow_dists) / (np.array(wide_dists) + 1e-10)
    ax2.plot(separations, ratio, 'g--', label='Ratio (narrow/wide)', alpha=0.7)
    ax2.set_ylabel('Ratio', color='g')
    ax2.tick_params(axis='y', labelcolor='g')
    
    ax.set_xlabel('Mean Separation (Δμ)')
    ax.set_ylabel('Distance')
    ax.set_title(f'Narrow vs Wide Comparison (α={alpha})')
    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'Variance-Weighted Wasserstein Distance (α={alpha})', fontsize=14, y=1.02)
    plt.tight_layout()
    
    # Save the figure
    plt.savefig('variance_weighted_wasserstein_analysis.png', dpi=150, bbox_inches='tight')
    print("\nVisualization saved as 'variance_weighted_wasserstein_analysis.png'")

if __name__ == "__main__":
    run_tests()
