import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

def bhattacharyya_distance(mu1, sigma1, mu2, sigma2):
    """
    Bhattacharyya distance between two 1D Gaussians.
    Lower values = more similar.
    """
    # Average variance
    sigma_avg = (sigma1**2 + sigma2**2) / 2
    
    # Bhattacharyya coefficient
    term1 = 0.25 * (mu1 - mu2)**2 / sigma_avg
    term2 = 0.5 * jnp.log(sigma_avg / (sigma1 * sigma2))
    
    return term1 + term2

def run_tests():
    """Test that Bhattacharyya distance has the desired properties."""
    
    print("="*60)
    print("TESTING BHATTACHARYYA DISTANCE")
    print("="*60)
    
    # Test 1: Identical distributions should have distance 0
    print("\n1. IDENTICAL DISTRIBUTIONS TEST")
    print("-" * 40)
    test_cases = [
        (0.5, 0.1, 0.5, 0.1, "Narrow identical"),
        (0.5, 0.5, 0.5, 0.5, "Wide identical"),
        (0.2, 0.3, 0.2, 0.3, "Random identical"),
    ]
    
    for mu1, sig1, mu2, sig2, desc in test_cases:
        dist = bhattacharyya_distance(mu1, sig1, mu2, sig2)
        print(f"{desc}: μ₁={mu1:.1f}, σ₁={sig1:.1f}, μ₂={mu2:.1f}, σ₂={sig2:.1f}")
        print(f"  Distance = {dist:.6f} (should be 0)")
        assert jnp.abs(dist) < 1e-10, f"Failed: {desc} should have distance 0"
    
    # Test 2: Narrow identical distributions should be more similar than wide ones
    # when separated by the same amount
    print("\n2. VARIANCE SENSITIVITY TEST")
    print("-" * 40)
    print("Testing: Narrow distributions separated by Δμ should be MORE different")
    print("         than wide distributions separated by same Δμ")
    
    separations = [0.1, 0.2, 0.5]
    for sep in separations:
        # Narrow gaussians separated
        narrow_dist = bhattacharyya_distance(0.3, 0.05, 0.3 + sep, 0.05)
        # Wide gaussians separated  
        wide_dist = bhattacharyya_distance(0.3, 0.3, 0.3 + sep, 0.3)
        
        print(f"\nΔμ = {sep:.1f}:")
        print(f"  Narrow (σ=0.05): distance = {narrow_dist:.4f}")
        print(f"  Wide   (σ=0.30): distance = {wide_dist:.4f}")
        print(f"  Ratio (narrow/wide) = {narrow_dist/wide_dist:.2f}x")
        
        assert narrow_dist > wide_dist, \
            f"Failed: Narrow distributions should be more different when separated"
    
    # Test 3: Distance increases with mean separation
    print("\n3. MEAN SEPARATION TEST")
    print("-" * 40)
    print("Testing: Distance should increase with mean separation")
    
    sigma_fixed = 0.2
    mean_seps = [0.0, 0.1, 0.2, 0.3, 0.5]
    distances = []
    
    for sep in mean_seps:
        dist = bhattacharyya_distance(0.3, sigma_fixed, 0.3 + sep, sigma_fixed)
        distances.append(dist)
        print(f"Δμ = {sep:.1f}: distance = {dist:.4f}")
    
    # Check monotonic increase
    for i in range(1, len(distances)):
        assert distances[i] > distances[i-1], \
            f"Failed: Distance should increase with mean separation"
    
    # Test 4: Distance increases with variance difference
    print("\n4. VARIANCE DIFFERENCE TEST")
    print("-" * 40)
    print("Testing: Distance should increase with variance difference")
    
    mu_fixed = 0.5
    sigma1_fixed = 0.2
    sigma2_values = [0.2, 0.25, 0.3, 0.35, 0.4]
    distances = []
    
    for sig2 in sigma2_values:
        dist = bhattacharyya_distance(mu_fixed, sigma1_fixed, mu_fixed, sig2)
        distances.append(dist)
        print(f"σ₁ = {sigma1_fixed:.2f}, σ₂ = {sig2:.2f}: distance = {dist:.4f}")
    
    # Check monotonic increase
    for i in range(1, len(distances)):
        assert distances[i] > distances[i-1], \
            f"Failed: Distance should increase with variance difference"
    
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
        dist_ab = bhattacharyya_distance(mu1, sig1, mu2, sig2)
        dist_ba = bhattacharyya_distance(mu2, sig2, mu1, sig1)
        print(f"d(μ={mu1:.1f},σ={sig1:.2f} → μ={mu2:.1f},σ={sig2:.2f}) = {dist_ab:.4f}")
        print(f"d(μ={mu2:.1f},σ={sig2:.2f} → μ={mu1:.1f},σ={sig1:.2f}) = {dist_ba:.4f}")
        assert jnp.abs(dist_ab - dist_ba) < 1e-10, "Failed: Distance should be symmetric"
    
    print("\n" + "="*60)
    print("ALL TESTS PASSED! ✓")
    print("="*60)
    
    # Visualization
    create_visualization()

def create_visualization():
    """Create visualizations showing the distance behavior."""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Heatmap: Distance as function of two means (fixed small variance)
    ax = axes[0, 0]
    mus = np.linspace(0, 1, 50)
    M1, M2 = np.meshgrid(mus, mus)
    sigma_small = 0.1
    D = np.zeros_like(M1)
    for i in range(len(mus)):
        for j in range(len(mus)):
            D[i, j] = bhattacharyya_distance(M1[i, j], sigma_small, M2[i, j], sigma_small)
    
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
            D[i, j] = bhattacharyya_distance(M1[i, j], sigma_large, M2[i, j], sigma_large)
    
    im = ax.contourf(M1, M2, D, levels=20, cmap='viridis')
    ax.set_xlabel('μ₁')
    ax.set_ylabel('μ₂')
    ax.set_title(f'Distance vs Means (σ₁=σ₂={sigma_large})')
    plt.colorbar(im, ax=ax)
    
    # 3. Line plot: Distance vs mean separation for different variances
    ax = axes[1, 0]
    separations = np.linspace(0, 1, 50)
    sigmas = [0.05, 0.1, 0.2, 0.3, 0.4]
    
    for sigma in sigmas:
        distances = [bhattacharyya_distance(0.3, sigma, 0.3 + sep, sigma) 
                    for sep in separations]
        ax.plot(separations, distances, label=f'σ={sigma:.2f}')
    
    ax.set_xlabel('Mean Separation (Δμ)')
    ax.set_ylabel('Bhattacharyya Distance')
    ax.set_title('Distance vs Mean Separation for Different Variances')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Heatmap: Distance as function of two variances (fixed means)
    ax = axes[1, 1]
    sigmas = np.linspace(0.01, 1, 50)
    S1, S2 = np.meshgrid(sigmas, sigmas)
    mu_sep = 0.3
    D = np.zeros_like(S1)
    for i in range(len(sigmas)):
        for j in range(len(sigmas)):
            D[i, j] = bhattacharyya_distance(0.3, S1[i, j], 0.6, S2[i, j])
    
    im = ax.contourf(S1, S2, D, levels=20, cmap='viridis')
    ax.set_xlabel('σ₁')
    ax.set_ylabel('σ₂')
    ax.set_title(f'Distance vs Std Devs (μ₁=0.3, μ₂=0.6)')
    plt.colorbar(im, ax=ax)
    
    plt.suptitle('Bhattacharyya Distance Behavior', fontsize=14, y=1.02)
    plt.tight_layout()
    
    # Save the figure
    plt.savefig('bhattacharyya_distance_analysis.png', dpi=150, bbox_inches='tight')
    print("\nVisualization saved as 'bhattacharyya_distance_analysis.png'")

if __name__ == "__main__":
    run_tests()
