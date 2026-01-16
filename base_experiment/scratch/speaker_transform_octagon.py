import jax.numpy as jnp

def constrain_to_octagon(points):
    """
    Constrain points to regular octagon centered at (0.5, 0.5)
    inscribed in unit square [0, 1] x [0, 1]
    points: shape (..., 2) 
    """
    center = 0.5
    
    # Shift to origin
    shifted = points - center
    x, y = shifted[..., 0], shifted[..., 1]
    
    # For a regular octagon inscribed in a square of side length 1 (radius 0.5)
    # The octagon has 8 sides, each defined by a halfplane
    # For a regular octagon, the key parameter is:
    # If the square has side 1, the octagon's diagonal corners are at ±0.5
    # The cut-off distance from corner is 0.5 * (2 - sqrt(2)) ≈ 0.2929
    
    # The halfplanes are:
    # 1. x ≤ 0.5
    # 2. y ≤ 0.5  
    # 3. x ≥ -0.5
    # 4. y ≥ -0.5
    # 5-8. The diagonal cuts: |x| + |y| ≤ 0.5 * sqrt(2)
    
    # Actually, for a regular octagon inscribed in a square:
    # The constraint is: max(|x|, |y|, (|x|+|y|)/sqrt(2)) ≤ 0.5
    
    abs_x = jnp.abs(x)
    abs_y = jnp.abs(y)
    
    # Three distance metrics:
    # 1. Horizontal/vertical distance (for square sides)
    dist_box = jnp.maximum(abs_x, abs_y)
    
    # 2. Diagonal distance (for octagon's angled sides)
    # For octagon inscribed in square: diagonal constraint
    dist_diag = (abs_x + abs_y) / jnp.sqrt(2)
    
    # Combined constraint
    dist = jnp.maximum(dist_box, dist_diag)
    
    # Scale down if outside octagon (radius is 0.5 for unit square)
    scale = jnp.where(dist > 0, jnp.minimum(1.0, 0.5 / dist), 1.0)
    
    # Apply scale
    constrained = shifted * scale[..., None]
    
    # Shift back
    return constrained + center


# Test points
test_points = jnp.array([
    [0.5, 0.5],    # Center - should stay
    [1.0, 0.5],    # Right edge - should map to octagon edge
    [0.9, 0.9],    # Corner - should map to octagon corner
    [0.7, 0.7],    # Diagonal - might need constraint
    [0.6, 0.5],    # Inside - should stay
])

constrained = constrain_to_octagon(test_points)

import matplotlib.pyplot as plt

# Visualize
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

# Draw octagon
theta = jnp.linspace(0, 2*jnp.pi, 9)
octagon_x = 0.5 + 0.5 * jnp.cos(theta + jnp.pi/8)
octagon_y = 0.5 + 0.5 * jnp.sin(theta + jnp.pi/8)

# But for a proper regular octagon in a square:
# The vertices alternate between edge midpoints and cut corners
oct_points = jnp.array([
    [0.5 + 0.5, 0.5 + 0.207],  # Right side, shifted up
    [0.5 + 0.207, 0.5 + 0.5],  # Top side, shifted right
    [0.5 - 0.207, 0.5 + 0.5],  # Top side, shifted left
    [0.5 - 0.5, 0.5 + 0.207],  # Left side, shifted up
    [0.5 - 0.5, 0.5 - 0.207],  # Left side, shifted down
    [0.5 - 0.207, 0.5 - 0.5],  # Bottom side, shifted left
    [0.5 + 0.207, 0.5 - 0.5],  # Bottom side, shifted right
    [0.5 + 0.5, 0.5 - 0.207],  # Right side, shifted down
    [0.5 + 0.5, 0.5 + 0.207],  # Close the shape
])

ax1.plot(oct_points[:, 0], oct_points[:, 1], 'b-', label='Octagon')
ax1.scatter(test_points[:, 0], test_points[:, 1], c='red', label='Original')
ax1.set_title('Original Points')

ax2.plot(oct_points[:, 0], oct_points[:, 1], 'b-', label='Octagon')
ax2.scatter(constrained[:, 0], constrained[:, 1], c='green', label='Constrained')
ax2.set_title('Constrained Points')

for ax in [ax1, ax2]:
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.1)
    ax.set_aspect('equal')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.show()
