"""
Validate and visualize structure alignment.

Check if Kabsch alignment is working correctly and show RMSD breakdown.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def validate_kabsch_alignment(N1, CA1, C1, N2, CA2, C2, name1="Structure 1", name2="Structure 2"):
    """
    Validate Kabsch alignment and show detailed comparison.
    
    Returns alignment quality metrics and creates diagnostic plots.
    """
    print(f"\n{'='*60}")
    print(f"ALIGNMENT VALIDATION: {name1} vs {name2}")
    print(f"{'='*60}")
    
    # Check if structures are same length
    if len(CA1) != len(CA2):
        print(f"✗ Length mismatch: {len(CA1)} vs {len(CA2)} residues")
        return None
    
    n_res = len(CA1)
    print(f"Residues: {n_res}")
    
    # 1. Pre-alignment RMSD
    rmsd_before = np.sqrt(np.mean(np.sum((CA1 - CA2)**2, axis=1)))
    print(f"\nBefore alignment:")
    print(f"  RMSD: {rmsd_before:.3f} Å")
    print(f"  CA1 centroid: {np.mean(CA1, axis=0)}")
    print(f"  CA2 centroid: {np.mean(CA2, axis=0)}")
    
    # 2. Perform Kabsch alignment
    centroid1 = np.mean(CA1, axis=0)
    centroid2 = np.mean(CA2, axis=0)
    
    CA1_centered = CA1 - centroid1
    CA2_centered = CA2 - centroid2
    
    # Cross-covariance matrix
    H = CA1_centered.T @ CA2_centered
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    
    # Ensure proper rotation (not reflection)
    if np.linalg.det(R) < 0:
        print("  ! Fixing reflection")
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    
    print(f"\nRotation matrix determinant: {np.linalg.det(R):.6f}")
    print(f"  (Should be +1.0 for proper rotation)")
    
    # Apply alignment
    N1_aligned = (R @ (N1 - centroid1).T).T + centroid2
    CA1_aligned = (R @ (CA1 - centroid1).T).T + centroid2
    C1_aligned = (R @ (C1 - centroid1).T).T + centroid2
    
    # 3. Post-alignment RMSD
    rmsd_after = np.sqrt(np.mean(np.sum((CA1_aligned - CA2)**2, axis=1)))
    print(f"\nAfter alignment:")
    print(f"  RMSD: {rmsd_after:.3f} Å")
    print(f"  Improvement: {rmsd_before - rmsd_after:.3f} Å")
    
    # 4. Per-residue RMSD
    per_res_rmsd = np.sqrt(np.sum((CA1_aligned - CA2)**2, axis=1))
    print(f"\nPer-residue RMSD:")
    print(f"  Mean: {np.mean(per_res_rmsd):.3f} Å")
    print(f"  Std:  {np.std(per_res_rmsd):.3f} Å")
    print(f"  Max:  {np.max(per_res_rmsd):.3f} Å (residue {np.argmax(per_res_rmsd)+1})")
    print(f"  Min:  {np.min(per_res_rmsd):.3f} Å (residue {np.argmin(per_res_rmsd)+1})")
    
    # 5. Check if alignment makes sense
    # Aligned structures should have similar centroids
    centroid_aligned = np.mean(CA1_aligned, axis=0)
    centroid_diff = np.linalg.norm(centroid_aligned - centroid2)
    print(f"\nCentroid alignment:")
    print(f"  Distance after alignment: {centroid_diff:.6f} Å")
    print(f"  {'✓ Good' if centroid_diff < 1e-6 else '✗ BAD - centroids not aligned!'}")
    
    # 6. Visualize
    plot_alignment_comparison(
        CA1_aligned, CA2, per_res_rmsd,
        name1, name2, rmsd_after
    )
    
    return {
        'rmsd_before': rmsd_before,
        'rmsd_after': rmsd_after,
        'per_residue_rmsd': per_res_rmsd,
        'aligned_structures': (N1_aligned, CA1_aligned, C1_aligned)
    }


def plot_alignment_comparison(CA1_aligned, CA2, per_res_rmsd, name1, name2, total_rmsd):
    """Plot alignment visualization."""
    fig = plt.figure(figsize=(16, 5))
    
    # 1. 3D overlay
    ax = fig.add_subplot(131, projection='3d')
    ax.plot(CA1_aligned[:, 0], CA1_aligned[:, 1], CA1_aligned[:, 2], 
           'b-o', label=name1, alpha=0.7, markersize=4)
    ax.plot(CA2[:, 0], CA2[:, 1], CA2[:, 2], 
           'r-o', label=name2, alpha=0.7, markersize=4)
    
    # Draw lines between corresponding residues
    for i in range(len(CA1_aligned)):
        ax.plot([CA1_aligned[i, 0], CA2[i, 0]],
               [CA1_aligned[i, 1], CA2[i, 1]],
               [CA1_aligned[i, 2], CA2[i, 2]],
               'gray', alpha=0.3, linewidth=0.5)
    
    ax.set_xlabel('X (Å)')
    ax.set_ylabel('Y (Å)')
    ax.set_zlabel('Z (Å)')
    ax.set_title(f'Aligned Structures\nRMSD = {total_rmsd:.2f} Å')
    ax.legend()
    
    # 2. Per-residue RMSD
    ax = fig.add_subplot(132)
    residues = np.arange(1, len(per_res_rmsd) + 1)
    ax.bar(residues, per_res_rmsd, edgecolor='black')
    ax.axhline(np.mean(per_res_rmsd), color='red', linestyle='--', 
              label=f'Mean: {np.mean(per_res_rmsd):.2f} Å')
    ax.set_xlabel('Residue')
    ax.set_ylabel('RMSD (Å)')
    ax.set_title('Per-Residue RMSD')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. RMSD distribution
    ax = fig.add_subplot(133)
    ax.hist(per_res_rmsd, bins=20, edgecolor='black')
    ax.axvline(np.mean(per_res_rmsd), color='red', linestyle='--',
              linewidth=2, label=f'Mean: {np.mean(per_res_rmsd):.2f} Å')
    ax.set_xlabel('RMSD (Å)')
    ax.set_ylabel('Count')
    ax.set_title('RMSD Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('alignment_validation.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved: alignment_validation.png")
    plt.close()


def test_kabsch_with_known_transformation():
    """
    Test Kabsch algorithm with known rotation and translation.
    
    If this fails, the algorithm is broken!
    """
    print("\n" + "="*60)
    print("TESTING KABSCH ALGORITHM")
    print("="*60)
    
    # Create a simple structure
    np.random.seed(42)
    n_res = 10
    CA_original = np.random.randn(n_res, 3) * 5
    
    # Apply known transformation
    angle = np.pi / 4  # 45 degrees
    R_true = np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle), np.cos(angle), 0],
        [0, 0, 1]
    ])
    t_true = np.array([10, 5, -3])
    
    CA_transformed = (R_true @ CA_original.T).T + t_true
    
    print(f"\nApplied transformation:")
    print(f"  Rotation: 45° around Z-axis")
    print(f"  Translation: {t_true}")
    
    # Use Kabsch to recover transformation
    result = validate_kabsch_alignment(
        CA_original, CA_original, CA_original,
        CA_transformed, CA_transformed, CA_transformed,
        "Original", "Transformed"
    )
    
    if result:
        rmsd = result['rmsd_after']
        print(f"\nRecovered RMSD: {rmsd:.6f} Å")
        if rmsd < 1e-6:
            print("✓ Kabsch algorithm is WORKING CORRECTLY")
        else:
            print("✗ Kabsch algorithm has PROBLEMS")
        
        return rmsd < 1e-6
    
    return False


if __name__ == "__main__":
    # Test the algorithm first
    if test_kabsch_with_known_transformation():
        print("\n" + "="*60)
        print("Kabsch implementation verified!")
        print("High RMSD means:")
        print("  1. Predicted structure is genuinely different")
        print("  2. Model needs better training")
        print("  3. CDR3 loops are inherently difficult")
        print("="*60)
    else:
        print("\n✗ Kabsch implementation has bugs - fix before trusting RMSD!")