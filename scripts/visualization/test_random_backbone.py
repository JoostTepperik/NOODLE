"""
Diagnostic script for random backbone triangle generation.

Tests:
1. Local geometry (N-CA-C triangles) - should be PERFECT
2. Peptide bonds (C-N between residues) - should be BROKEN
3. Rotation distribution - should be uniform
4. Visualizations

Run with: python test_random_backbone.py
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path

# Import the random backbone generator
from random_backbone_cloud import (
    generate_random_backbone_cloud,
    check_local_geometry,
    check_peptide_bonds,
    get_ideal_backbone_frame
)


# ============================================================================
# VALIDATION TESTS
# ============================================================================

def test_local_geometry(N, CA, C, tolerance=1e-4):
    """
    Test that N-CA-C triangles have perfect geometry.
    
    This should ALWAYS pass - it's the whole point of the frame approach.
    """
    print("\n" + "="*60)
    print("TEST 1: LOCAL GEOMETRY (N-CA-C triangles)")
    print("="*60)
    
    geom = check_local_geometry(N, CA, C)
    
    # Expected values
    expected_n_ca = 1.458
    expected_ca_c = 1.526
    expected_angle = 111.0
    
    # Check bonds
    n_ca_mean = np.mean(geom['n_ca_bonds'])
    n_ca_std = np.std(geom['n_ca_bonds'])
    ca_c_mean = np.mean(geom['ca_c_bonds'])
    ca_c_std = np.std(geom['ca_c_bonds'])
    
    print(f"\nBond lengths:")
    print(f"  N-CA: {n_ca_mean:.6f} ± {n_ca_std:.6f} Å")
    print(f"  Expected: {expected_n_ca} Å")
    print(f"  Error: {abs(n_ca_mean - expected_n_ca):.2e} Å")
    
    print(f"\n  CA-C: {ca_c_mean:.6f} ± {ca_c_std:.6f} Å")
    print(f"  Expected: {expected_ca_c} Å")
    print(f"  Error: {abs(ca_c_mean - expected_ca_c):.2e} Å")
    
    # Check angles
    angle_mean = np.mean(geom['n_ca_c_angles'])
    angle_std = np.std(geom['n_ca_c_angles'])
    
    print(f"\nN-CA-C angle:")
    print(f"  {angle_mean:.6f} ± {angle_std:.6f}°")
    print(f"  Expected: {expected_angle}°")
    print(f"  Error: {abs(angle_mean - expected_angle):.2e}°")
    
    # Pass/fail
    n_ca_pass = abs(n_ca_mean - expected_n_ca) < tolerance
    ca_c_pass = abs(ca_c_mean - expected_ca_c) < tolerance
    angle_pass = abs(angle_mean - expected_angle) < tolerance
    
    all_pass = n_ca_pass and ca_c_pass and angle_pass
    
    print(f"\n{'✓ PASS' if all_pass else '✗ FAIL'}: Local geometry is {'perfect' if all_pass else 'BROKEN'}")
    
    return all_pass


def test_peptide_bonds(N, CA, C):
    """
    Test that peptide bonds are broken.
    
    This should ALWAYS show broken bonds - that's the starting point.
    """
    print("\n" + "="*60)
    print("TEST 2: PEPTIDE BONDS (C-N between residues)")
    print("="*60)
    
    peptide = check_peptide_bonds(N, CA, C)
    
    if len(peptide) == 0:
        print("\nOnly 1 residue - no peptide bonds to check")
        return True
    
    expected = 1.329  # Ideal peptide bond length
    mean_dist = np.mean(peptide)
    std_dist = np.std(peptide)
    min_dist = np.min(peptide)
    max_dist = np.max(peptide)
    
    print(f"\nPeptide bond C-N distances:")
    print(f"  Mean: {mean_dist:.2f} Å")
    print(f"  Std:  {std_dist:.2f} Å")
    print(f"  Range: {min_dist:.2f} - {max_dist:.2f} Å")
    print(f"  Expected: {expected} Å (if connected)")
    
    # Count how many are close to ideal
    close_to_ideal = np.sum(np.abs(np.array(peptide) - expected) < 0.5)
    far_from_ideal = len(peptide) - close_to_ideal
    
    print(f"\n  Close to ideal (<0.5Å error): {close_to_ideal}/{len(peptide)}")
    print(f"  Broken (>0.5Å error): {far_from_ideal}/{len(peptide)}")
    
    # Should be mostly broken
    mostly_broken = far_from_ideal > len(peptide) * 0.8
    
    print(f"\n{'✓ PASS' if mostly_broken else '⚠ WARNING'}: Peptide bonds are {'broken (expected)' if mostly_broken else 'mostly intact (unexpected)'}")
    
    return mostly_broken


def test_rotation_uniformity(N, CA, C, n_bins=10):
    """
    Test that rotations are uniformly distributed.
    
    Check if the orientations cover SO(3) uniformly.
    """
    print("\n" + "="*60)
    print("TEST 3: ROTATION UNIFORMITY")
    print("="*60)
    
    n_residues = len(N)
    
    # Get orientation vectors (CA to C direction)
    orientations = C - CA
    orientations = orientations / np.linalg.norm(orientations, axis=1, keepdims=True)
    
    # Convert to spherical coordinates
    # x = sin(theta) * cos(phi)
    # y = sin(theta) * sin(phi)
    # z = cos(theta)
    
    theta = np.arccos(np.clip(orientations[:, 2], -1, 1))  # Polar angle
    phi = np.arctan2(orientations[:, 1], orientations[:, 0])  # Azimuthal angle
    
    # For uniform distribution on sphere:
    # theta should be distributed as sin(theta) (not uniform!)
    # phi should be uniform
    
    print(f"\nGenerated {n_residues} random orientations")
    print(f"\nSpherical coordinates:")
    print(f"  Theta (polar): {np.degrees(theta.min()):.1f}° - {np.degrees(theta.max()):.1f}°")
    print(f"  Phi (azimuthal): {np.degrees(phi.min()):.1f}° - {np.degrees(phi.max()):.1f}°")
    
    # Simple uniformity test: check if phi is roughly uniform
    phi_hist, _ = np.histogram(phi, bins=n_bins)
    expected_count = n_residues / n_bins
    max_deviation = np.max(np.abs(phi_hist - expected_count))
    
    print(f"\nPhi uniformity check:")
    print(f"  Expected count per bin: {expected_count:.1f}")
    print(f"  Max deviation: {max_deviation:.1f}")
    print(f"  Relative deviation: {max_deviation/expected_count:.2%}")
    
    # With small sample, allow 50% deviation
    uniform = max_deviation / expected_count < 0.5
    
    print(f"\n{'✓ PASS' if uniform else '⚠ WARNING'}: Rotations appear {'uniform' if uniform else 'possibly biased'}")
    
    return uniform, theta, phi


# ============================================================================
# VISUALIZATIONS
# ============================================================================

def plot_random_cloud_3d(N, CA, C, save_path='random_cloud_3d.png'):
    """
    3D visualization of the random cloud.
    Shows each triangle with peptide bonds.
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    n_residues = len(N)
    
    # Plot each triangle
    for i in range(n_residues):
        # Triangle edges
        triangle = np.array([N[i], CA[i], C[i], N[i]])
        ax.plot(triangle[:, 0], triangle[:, 1], triangle[:, 2], 
                'b-', linewidth=2, alpha=0.6)

        # Peptide bonds (broken)
        if i < n_residues - 1:
            ax.plot([C[i, 0], N[i+1, 0]], 
                   [C[i, 1], N[i+1, 1]], 
                   [C[i, 2], N[i+1, 2]], 
                   'r--', linewidth=1, alpha=0.3)
    
    ax.set_xlabel('X (Å)')
    ax.set_ylabel('Y (Å)')
    ax.set_zlabel('Z (Å)')
    ax.set_title(f'Random Backbone Cloud ({n_residues} residues)\nBlue=N, Black=CA, Red=C, Dashed=Broken peptide bonds')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved: {save_path}")
    plt.close()


def plot_geometry_distributions(N, CA, C, save_path='geometry_distributions.png'):
    """
    Plot distributions of bond lengths and angles.
    """
    geom = check_local_geometry(N, CA, C)
    peptide = check_peptide_bonds(N, CA, C)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # N-CA bonds
    ax = axes[0, 0]
    n_ca_range = np.ptp(geom['n_ca_bonds'])  # Peak to peak
    if n_ca_range > 1e-6:
        ax.hist(geom['n_ca_bonds'], bins=20, alpha=0.7, edgecolor='black')
    else:
        # All identical - show single bar
        ax.bar([1.458], [len(geom['n_ca_bonds'])], width=0.001, alpha=0.7, edgecolor='black')
    ax.axvline(1.458, color='red', linestyle='--', linewidth=2, label='Expected: 1.458Å')
    ax.set_xlabel('Bond length (Å)')
    ax.set_ylabel('Count')
    ax.set_title(f'N-CA Bond Lengths (σ={np.std(geom["n_ca_bonds"]):.2e}Å)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # CA-C bonds
    ax = axes[0, 1]
    ca_c_range = np.ptp(geom['ca_c_bonds'])
    if ca_c_range > 1e-6:
        ax.hist(geom['ca_c_bonds'], bins=20, alpha=0.7, edgecolor='black', color='orange')
    else:
        ax.bar([1.526], [len(geom['ca_c_bonds'])], width=0.001, alpha=0.7, edgecolor='black', color='orange')
    ax.axvline(1.526, color='red', linestyle='--', linewidth=2, label='Expected: 1.526Å')
    ax.set_xlabel('Bond length (Å)')
    ax.set_ylabel('Count')
    ax.set_title(f'CA-C Bond Lengths (σ={np.std(geom["ca_c_bonds"]):.2e}Å)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # N-CA-C angles
    ax = axes[1, 0]
    angle_range = np.ptp(geom['n_ca_c_angles'])
    if angle_range > 1e-6:
        ax.hist(geom['n_ca_c_angles'], bins=20, alpha=0.7, edgecolor='black', color='green')
    else:
        ax.bar([111.0], [len(geom['n_ca_c_angles'])], width=0.01, alpha=0.7, edgecolor='black', color='green')
    ax.axvline(111.0, color='red', linestyle='--', linewidth=2, label='Expected: 111.0°')
    ax.set_xlabel('Angle (degrees)')
    ax.set_ylabel('Count')
    ax.set_title(f'N-CA-C Bond Angles (σ={np.std(geom["n_ca_c_angles"]):.2e}°)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Peptide bonds (should be broken)
    ax = axes[1, 1]
    if len(peptide) > 0:
        ax.hist(peptide, bins=20, alpha=0.7, edgecolor='black', color='red')
        ax.axvline(1.329, color='blue', linestyle='--', linewidth=2, label='Ideal: 1.329Å')
        ax.set_xlabel('Distance (Å)')
        ax.set_ylabel('Count')
        ax.set_title('Peptide Bond C-N Distances (BROKEN in random cloud)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'Only 1 residue\n(no peptide bonds)', 
               ha='center', va='center', transform=ax.transAxes, fontsize=14)
        ax.set_title('Peptide Bond Distances')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()


def plot_rotation_distribution(theta, phi, save_path='rotation_distribution.png'):
    """
    Plot distribution of rotation orientations.
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # Theta distribution (should follow sin(theta))
    ax = axes[0]
    ax.hist(np.degrees(theta), bins=30, alpha=0.7, edgecolor='black', density=True)
    
    # Expected distribution: proportional to sin(theta)
    theta_range = np.linspace(0, np.pi, 100)
    expected = np.sin(theta_range) / 2  # Normalized
    ax.plot(np.degrees(theta_range), expected, 'r-', linewidth=2, 
           label='Expected: ∝ sin(θ)')
    
    ax.set_xlabel('Theta (degrees)')
    ax.set_ylabel('Density')
    ax.set_title('Polar Angle Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Phi distribution (should be uniform)
    ax = axes[1]
    ax.hist(np.degrees(phi), bins=30, alpha=0.7, edgecolor='black', color='orange')
    ax.axhline(len(phi)/30, color='red', linestyle='--', linewidth=2, label='Uniform')
    ax.set_xlabel('Phi (degrees)')
    ax.set_ylabel('Count')
    ax.set_title('Azimuthal Angle Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2D distribution on sphere
    ax = axes[2]
    ax.scatter(np.degrees(phi), np.degrees(theta), alpha=0.5, s=20)
    ax.set_xlabel('Phi (degrees)')
    ax.set_ylabel('Theta (degrees)')
    ax.set_title('Orientation on Sphere')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-180, 180)
    ax.set_ylim(0, 180)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()


def plot_individual_triangles(N, CA, C, n_show=5, save_path='individual_triangles.png'):
    """
    Show individual N-CA-C triangles to verify geometry.
    """
    n_residues = len(N)
    n_show = min(n_show, n_residues)
    
    fig, axes = plt.subplots(1, n_show, figsize=(4*n_show, 4))
    if n_show == 1:
        axes = [axes]
    
    ideal = get_ideal_backbone_frame()
    
    for i, ax in enumerate(axes):
        # Actual triangle
        triangle = np.array([N[i], CA[i], C[i], N[i]])
        ax.plot(triangle[:, 0], triangle[:, 1], 'b-', linewidth=2, label='Actual')
        ax.scatter([N[i, 0]], [N[i, 1]], c='blue', s=100, marker='o')
        ax.scatter([CA[i, 0]], [CA[i, 1]], c='black', s=100, marker='s')
        ax.scatter([C[i, 0]], [C[i, 1]], c='red', s=100, marker='^')
        
        # Add labels
        ax.text(N[i, 0], N[i, 1], ' N', fontsize=10)
        ax.text(CA[i, 0], CA[i, 1], ' CA', fontsize=10)
        ax.text(C[i, 0], C[i, 1], ' C', fontsize=10)
        
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title(f'Residue {i}')
        ax.set_xlabel('X (Å)')
        ax.set_ylabel('Y (Å)')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()


# ============================================================================
# MAIN TEST SUITE
# ============================================================================

def run_full_diagnostics(n_residues=20, position_scale=10.0, seed=42):
    """
    Complete diagnostic suite for random backbone generation.
    """
    print("\n" + "="*70)
    print("RANDOM BACKBONE GENERATION - DIAGNOSTIC SUITE")
    print("="*70)
    print(f"\nParameters:")
    print(f"  Residues: {n_residues}")
    print(f"  Position scale: {position_scale} Å")
    print(f"  Seed: {seed}")
    
    # Generate random cloud
    print(f"\nGenerating random backbone cloud...")
    N, CA, C, frames = generate_random_backbone_cloud(
        n_residues=n_residues,
        position_scale=position_scale,
        seed=seed
    )
    
    # Run tests
    test1_pass = test_local_geometry(N, CA, C)
    test2_pass = test_peptide_bonds(N, CA, C)
    test3_pass, theta, phi = test_rotation_uniformity(N, CA, C)
    
    # Generate visualizations
    print("\n" + "="*60)
    print("GENERATING VISUALIZATIONS")
    print("="*60)
    
    plot_random_cloud_3d(N, CA, C)
    plot_geometry_distributions(N, CA, C)
    plot_rotation_distribution(theta, phi)
    plot_individual_triangles(N, CA, C, n_show=min(5, n_residues))
    
    # Save PDB
    from nerf_reconstruction import ProteinBackboneReconstructor
    reconstructor = ProteinBackboneReconstructor()
    
    O = np.zeros_like(N)
    for i in range(n_residues):
        v = N[i+1] - C[i] if i < n_residues - 1 else N[i] - C[i]
        v = v / np.linalg.norm(v)
        O[i] = C[i] - v * 1.229
    
    sequence = "A" * n_residues
    reconstructor.save_pdb("random_cloud_diagnostic.pdb", sequence, N, CA, C, O)
    print("\n✓ Saved: random_cloud_diagnostic.pdb")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    all_pass = test1_pass and test2_pass
    
    print(f"\nTest Results:")
    print(f"  {'✓' if test1_pass else '✗'} Local geometry (N-CA-C triangles): {'PASS' if test1_pass else 'FAIL'}")
    print(f"  {'✓' if test2_pass else '✗'} Peptide bonds (broken): {'PASS' if test2_pass else 'FAIL'}")
    print(f"  {'✓' if test3_pass else '⚠'} Rotation uniformity: {'PASS' if test3_pass else 'WARNING'}")
    
    print(f"\nGenerated files:")
    print(f"  - random_cloud_3d.png")
    print(f"  - geometry_distributions.png")
    print(f"  - rotation_distribution.png")
    print(f"  - individual_triangles.png")
    print(f"  - random_cloud_diagnostic.pdb")
    
    print(f"\n{'='*70}")
    print(f"{'✓ ALL TESTS PASSED' if all_pass else '✗ SOME TESTS FAILED'}")
    print(f"{'='*70}\n")
    
    return all_pass


# ============================================================================
# RUN DIAGNOSTICS
# ============================================================================

if __name__ == "__main__":
    import sys
    
    # Allow command line arguments
    n_residues = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    position_scale = float(sys.argv[2]) if len(sys.argv) > 2 else 10.0
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 42
    
    success = run_full_diagnostics(
        n_residues=n_residues,
        position_scale=position_scale,
        seed=seed
    )
    
    sys.exit(0 if success else 1)