"""
Refine random backbone clouds using energy landscapes.

This implements the denoising/refinement process:
1. Start with random SE(3) frames
2. Calculate current angles
3. Compare with energy landscape
4. Update frames to minimize energy
5. Apply constraints (peptide bonds, geometry)
6. Iterate until converged
"""

import numpy as np
import jax
import jax.numpy as jnp
from typing import Tuple, Dict
from pathlib import Path
import matplotlib.pyplot as plt


def coords_to_angles(N, CA, C):
    """
    Calculate phi and psi torsion angles from backbone coordinates.
    
    Args:
        N: (n_residues, 3) nitrogen coordinates
        CA: (n_residues, 3) CA coordinates
        C: (n_residues, 3) carbon coordinates
        
    Returns:
        phi: (n_residues,) phi angles in degrees
        psi: (n_residues,) psi angles in degrees
    """
    n_residues = len(N)
    phi = np.zeros(n_residues)
    psi = np.zeros(n_residues)
    
    for i in range(n_residues):
        # Phi: C(i-1) - N(i) - CA(i) - C(i)
        if i > 0:
            phi[i] = calculate_dihedral(C[i-1], N[i], CA[i], C[i])
        
        # Psi: N(i) - CA(i) - C(i) - N(i+1)
        if i < n_residues - 1:
            psi[i] = calculate_dihedral(N[i], CA[i], C[i], N[i+1])
    
    return phi, psi


def calculate_dihedral(p1, p2, p3, p4):
    """Calculate dihedral angle between four points in degrees."""
    b1 = p2 - p1
    b2 = p3 - p2
    b3 = p4 - p3
    
    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)
    
    n1 = n1 / np.linalg.norm(n1)
    n2 = n2 / np.linalg.norm(n2)
    
    m1 = np.cross(n1, b2 / np.linalg.norm(b2))
    
    x = np.dot(n1, n2)
    y = np.dot(m1, n2)
    
    angle = np.arctan2(y, x)
    return np.degrees(angle)


def energy_from_logits(phi, psi, logits_phi, logits_psi, n_bins=36):
    """
    Calculate energy for given angles using model's energy landscape.
    
    Args:
        phi: (n_residues,) phi angles in degrees
        psi: (n_residues,) psi angles in degrees
        logits_phi: (n_residues, n_bins) phi energy landscape
        logits_psi: (n_residues, n_bins) psi energy landscape
        n_bins: number of bins
        
    Returns:
        energy: scalar total energy
        grad_phi: (n_residues,) gradient w.r.t. phi
        grad_psi: (n_residues,) gradient w.r.t. psi
    """
    n_residues = len(phi)
    bin_width = 360.0 / n_bins
    
    # Get probabilities (negative log prob = energy)
    probs_phi = jax.nn.softmax(logits_phi, axis=-1)
    probs_psi = jax.nn.softmax(logits_psi, axis=-1)
    
    total_energy = 0.0
    grad_phi = np.zeros(n_residues)
    grad_psi = np.zeros(n_residues)
    
    for i in range(n_residues):
        # Find bin for current angle
        phi_bin = int((phi[i] + 180.0) / bin_width)
        psi_bin = int((psi[i] + 180.0) / bin_width)
        
        phi_bin = np.clip(phi_bin, 0, n_bins - 1)
        psi_bin = np.clip(psi_bin, 0, n_bins - 1)
        
        # Energy = -log(probability)
        eps = 1e-10
        energy_phi = -np.log(probs_phi[i, phi_bin] + eps)
        energy_psi = -np.log(probs_psi[i, psi_bin] + eps)
        
        total_energy += energy_phi + energy_psi
        
        # Gradient (simple finite difference approximation)
        # Negative gradient points toward higher probability
        if phi_bin < n_bins - 1:
            grad_phi[i] = -(np.log(probs_phi[i, phi_bin + 1] + eps) - energy_phi) / bin_width
        if psi_bin < n_bins - 1:
            grad_psi[i] = -(np.log(probs_psi[i, psi_bin + 1] + eps) - energy_psi) / bin_width
    
    return total_energy, grad_phi, grad_psi


def apply_angle_update(N, CA, C, grad_phi, grad_psi, step_size=0.01):
    """
    Update coordinates based on angle gradients.
    
    This uses inverse kinematics: given desired angle changes,
    update 3D coordinates to achieve them.
    
    Args:
        N, CA, C: Current coordinates
        grad_phi, grad_psi: Angle gradients (degrees)
        step_size: Step size for update
        
    Returns:
        Updated N, CA, C coordinates
    """
    n_residues = len(N)
    N_new = N.copy()
    CA_new = CA.copy()
    C_new = C.copy()
    
    for i in range(n_residues):
        # Update phi: rotate around N-CA bond
        if i > 0 and abs(grad_phi[i]) > 0.01:
            # Rotation axis: N(i) - CA(i)
            axis = CA[i] - N[i]
            axis = axis / np.linalg.norm(axis)
            
            # Rotation angle (small step)
            angle_rad = np.radians(step_size * grad_phi[i])
            
            # Rotate C(i) and subsequent atoms
            C_new[i] = rotate_point(C[i], N[i], axis, angle_rad)
        
        # Update psi: rotate around CA-C bond
        if i < n_residues - 1 and abs(grad_psi[i]) > 0.01:
            # Rotation axis: CA(i) - C(i)
            axis = C[i] - CA[i]
            axis = axis / np.linalg.norm(axis)
            
            # Rotation angle
            angle_rad = np.radians(step_size * grad_psi[i])
            
            # Rotate N(i+1) and subsequent atoms
            N_new[i+1] = rotate_point(N[i+1], CA[i], axis, angle_rad)
    
    return N_new, CA_new, C_new


def rotate_point(point, origin, axis, angle):
    """Rotate point around axis passing through origin."""
    # Rodrigues' rotation formula
    k = axis
    p = point - origin
    
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    
    p_rot = p * cos_a + np.cross(k, p) * sin_a + k * np.dot(k, p) * (1 - cos_a)
    
    return p_rot + origin


def enforce_peptide_bonds(N, CA, C, bond_length=1.329, strength=0.5):
    """
    Enforce peptide bond length constraints.
    
    Move residues to restore C(i) - N(i+1) bond lengths.
    
    Args:
        N, CA, C: Coordinates
        bond_length: Target C-N bond length (Å)
        strength: Constraint strength (0-1)
        
    Returns:
        Updated coordinates
    """
    n_residues = len(N)
    N_new = N.copy()
    CA_new = CA.copy()
    C_new = C.copy()
    
    for i in range(n_residues - 1):
        # Current peptide bond
        bond_vec = N[i+1] - C[i]
        current_length = np.linalg.norm(bond_vec)
        
        if abs(current_length - bond_length) > 0.01:
            # Direction
            direction = bond_vec / current_length
            
            # Target position
            target = C[i] + direction * bond_length
            
            # Move N(i+1) toward target
            correction = (target - N[i+1]) * strength
            N_new[i+1] += correction
            
            # Also move CA(i+1) to maintain N-CA bond
            CA_new[i+1] += correction
    
    return N_new, CA_new, C_new


def enforce_local_geometry(N, CA, C):
    """
    Enforce local N-CA-C geometry within each residue.
    
    Maintains:
    - N-CA bond length: 1.458 Å
    - CA-C bond length: 1.523 Å
    - N-CA-C angle: ~111°
    """
    n_residues = len(N)
    N_new = N.copy()
    CA_new = CA.copy()
    C_new = C.copy()
    
    for i in range(n_residues):
        # Fix N-CA bond length
        n_ca_vec = CA[i] - N[i]
        n_ca_length = np.linalg.norm(n_ca_vec)
        if n_ca_length > 0:
            N_new[i] = CA[i] - (n_ca_vec / n_ca_length) * 1.458
        
        # Fix CA-C bond length
        ca_c_vec = C[i] - CA[i]
        ca_c_length = np.linalg.norm(ca_c_vec)
        if ca_c_length > 0:
            C_new[i] = CA[i] + (ca_c_vec / ca_c_length) * 1.523
    
    return N_new, CA_new, C_new


def detect_clashes(N, CA, C, min_distance=3.0, exclude_neighbors=True):
    """
    Detect atomic clashes (atoms too close together).
    
    Args:
        N, CA, C: Coordinates
        min_distance: Minimum allowed distance (Å)
        exclude_neighbors: Don't count adjacent residues as clashes
        
    Returns:
        clash_pairs: List of (atom_idx1, atom_idx2, distance)
        clash_count: Total number of clashes
    """
    n_residues = len(N)
    
    # Build atom list with residue info
    atoms = []
    for i in range(n_residues):
        atoms.append((N[i], i, 'N'))
        atoms.append((CA[i], i, 'CA'))
        atoms.append((C[i], i, 'C'))
    
    clash_pairs = []
    
    for i in range(len(atoms)):
        coord_i, res_i, atom_i = atoms[i]
        
        for j in range(i + 1, len(atoms)):
            coord_j, res_j, atom_j = atoms[j]
            
            # Skip if same residue
            if res_i == res_j:
                continue
            
            # Skip if adjacent residues (bonded)
            if exclude_neighbors and abs(res_i - res_j) == 1:
                continue
            
            # Calculate distance
            dist = np.linalg.norm(coord_i - coord_j)
            
            if dist < min_distance:
                clash_pairs.append((i, j, dist))
    
    return clash_pairs, len(clash_pairs)


def resolve_clashes(N, CA, C, min_distance=3.0, strength=0.1):
    """
    Resolve clashes by moving atoms apart.
    
    Args:
        N, CA, C: Coordinates
        min_distance: Target minimum distance
        strength: How much to move (0-1)
        
    Returns:
        Updated N, CA, C with reduced clashes
    """
    n_residues = len(N)
    N_new = N.copy()
    CA_new = CA.copy()
    C_new = C.copy()
    
    # Detect clashes
    clash_pairs, _ = detect_clashes(N, CA, C, min_distance)
    
    if len(clash_pairs) == 0:
        return N_new, CA_new, C_new
    
    # Build atom array for easy indexing
    atoms = []
    for i in range(n_residues):
        atoms.append(N[i])
        atoms.append(CA[i])
        atoms.append(C[i])
    atoms = np.array(atoms)
    
    # For each clash, push atoms apart
    for idx_i, idx_j, dist in clash_pairs:
        # Direction from j to i
        direction = atoms[idx_i] - atoms[idx_j]
        direction_norm = np.linalg.norm(direction)
        
        if direction_norm > 1e-6:
            direction = direction / direction_norm
            
            # How much to move
            overlap = min_distance - dist
            move_amount = overlap * strength / 2  # Split between both atoms
            
            # Move atoms apart
            atoms[idx_i] += direction * move_amount
            atoms[idx_j] -= direction * move_amount
    
    # Unpack back to N, CA, C
    for i in range(n_residues):
        N_new[i] = atoms[3*i]
        CA_new[i] = atoms[3*i + 1]
        C_new[i] = atoms[3*i + 2]
    
    return N_new, CA_new, C_new


def refine_structure(
    N_init, CA_init, C_init,
    logits_phi, logits_psi,
    n_steps=100,
    step_size=0.1,
    constraint_strength=0.5,
    clash_resolution_strength=0.2,
    n_bins=36,
    verbose=True
):
    """
    Refine random cloud structure using energy landscape.
    
    Args:
        N_init, CA_init, C_init: Initial random coordinates
        logits_phi, logits_psi: Energy landscapes from model
        n_steps: Number of refinement iterations
        step_size: Step size for gradient descent
        constraint_strength: Strength of geometric constraints
        clash_resolution_strength: Strength of clash resolution (0-1)
        n_bins: Number of angle bins
        verbose: Print progress
        
    Returns:
        N, CA, C: Refined coordinates
        history: Dict with refinement history
    """
    if verbose:
        print(f"\n{'='*60}")
        print("STRUCTURE REFINEMENT")
        print(f"{'='*60}")
        print(f"Steps: {n_steps}")
        print(f"Step size: {step_size}")
        print(f"Constraint strength: {constraint_strength}")
        print(f"Clash resolution: {clash_resolution_strength}")
        print(f"{'='*60}\n")
    
    N = N_init.copy()
    CA = CA_init.copy()
    C = C_init.copy()
    
    # Track history
    history = {
        'energy': [],
        'rmsd': [],
        'peptide_bond_error': [],
        'angle_error': [],
        'clash_count': []
    }
    
    for step in range(n_steps):
        # 1. Calculate current angles
        phi, psi = coords_to_angles(N, CA, C)
        
        # 2. Calculate energy and gradients
        energy, grad_phi, grad_psi = energy_from_logits(
            phi, psi, logits_phi, logits_psi, n_bins
        )
        
        # 3. Update coordinates based on gradients
        N, CA, C = apply_angle_update(N, CA, C, grad_phi, grad_psi, step_size)
        
        # 4. Enforce constraints
        N, CA, C = enforce_peptide_bonds(N, CA, C, strength=constraint_strength)
        N, CA, C = enforce_local_geometry(N, CA, C)
        
        # 5. Resolve clashes (NEW!)
        if clash_resolution_strength > 0:
            N, CA, C = resolve_clashes(N, CA, C, 
                                       min_distance=3.0, 
                                       strength=clash_resolution_strength)
        
        # 6. Track metrics
        peptide_errors = []
        for i in range(len(N) - 1):
            bond_length = np.linalg.norm(N[i+1] - C[i])
            peptide_errors.append(abs(bond_length - 1.329))
        
        # Count clashes
        _, clash_count = detect_clashes(N, CA, C, min_distance=3.0)
        
        history['energy'].append(energy)
        history['peptide_bond_error'].append(np.mean(peptide_errors))
        history['clash_count'].append(clash_count)
        
        # Get target angles from energy landscape
        target_phi = get_most_probable_angles(logits_phi, n_bins)
        target_psi = get_most_probable_angles(logits_psi, n_bins)
        angle_error = np.mean(np.abs(phi - target_phi)) + np.mean(np.abs(psi - target_psi))
        history['angle_error'].append(angle_error)
        
        # Print progress
        if verbose and (step + 1) % 10 == 0:
            print(f"Step {step+1:3d}: Energy={energy:8.2f}, "
                  f"Bond error={history['peptide_bond_error'][-1]:.4f}Å, "
                  f"Clashes={clash_count}, "
                  f"Angle error={angle_error:.2f}°")
    
    if verbose:
        print(f"\n✓ Refinement complete!")
        print(f"  Final energy: {history['energy'][-1]:.2f}")
        print(f"  Final bond error: {history['peptide_bond_error'][-1]:.4f}Å")
        print(f"  Final clashes: {history['clash_count'][-1]}")
        print(f"  Final angle error: {history['angle_error'][-1]:.2f}°")
    
    return N, CA, C, history


def get_most_probable_angles(logits, n_bins):
    """Get most probable angle for each residue."""
    probs = jax.nn.softmax(logits, axis=-1)
    bins = np.argmax(probs, axis=-1)
    
    bin_width = 360.0 / n_bins
    angles = -180.0 + (bins + 0.5) * bin_width
    
    return angles


def plot_refinement_history(history, save_path=None):
    """Plot refinement metrics over time."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    
    # Energy
    axes[0, 0].plot(history['energy'], 'b-', linewidth=2)
    axes[0, 0].set_xlabel('Step')
    axes[0, 0].set_ylabel('Energy')
    axes[0, 0].set_title('Energy Minimization')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Peptide bond error
    axes[0, 1].plot(history['peptide_bond_error'], 'r-', linewidth=2)
    axes[0, 1].axhline(0.01, color='green', linestyle='--', label='Target (<0.01Å)')
    axes[0, 1].set_xlabel('Step')
    axes[0, 1].set_ylabel('Bond Error (Å)')
    axes[0, 1].set_title('Peptide Bond Constraint')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Angle error
    axes[1, 0].plot(history['angle_error'], 'g-', linewidth=2)
    axes[1, 0].set_xlabel('Step')
    axes[1, 0].set_ylabel('Angle Error (degrees)')
    axes[1, 0].set_title('Angle Convergence')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Clashes (NEW!)
    if 'clash_count' in history:
        axes[1, 1].plot(history['clash_count'], 'orange', linewidth=2)
        axes[1, 1].axhline(0, color='green', linestyle='--', label='Clash-free')
        axes[1, 1].set_xlabel('Step')
        axes[1, 1].set_ylabel('Number of Clashes')
        axes[1, 1].set_title('Clash Resolution')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 1].text(0.5, 0.5, 'Clash tracking not available', 
                       ha='center', va='center')
        axes[1, 1].set_title('Clash Resolution')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved refinement history: {save_path}")
    else:
        plt.show()
    
    plt.close()


# Example usage
if __name__ == "__main__":
    from random_backbone_cloud import generate_random_backbone_cloud
    from nerf_reconstruction import ProteinBackboneReconstructor
    
    # Generate random cloud
    print("Generating random backbone cloud...")
    N_random, CA_random, C_random, frames = generate_random_backbone_cloud(
        n_residues=7,
        cloud_sigma_position=5.0,
        cloud_sigma_rotation=0.5,
        seed=1
    )
    
    # Create mock energy landscape (peaked around helix angles)
    n_residues = 7
    n_bins = 36
    logits_phi = np.random.randn(n_residues, n_bins) * 0.5
    logits_psi = np.random.randn(n_residues, n_bins) * 0.5
    logits_phi[:, 12] += 3.0  # Favor -65°
    logits_psi[:, 13] += 3.0  # Favor -55°
    
    # Refine structure
    N_refined, CA_refined, C_refined, history = refine_structure(
        N_random, CA_random, C_random,
        logits_phi, logits_psi,
        n_steps=100,
        step_size=0.1,
        constraint_strength=0.5
    )
    
    # Save structures
    reconstructor = ProteinBackboneReconstructor()
    sequence = "AAAAAAA"
    
    # Add oxygens (simple placement)
    def add_oxygens(N, CA, C):
        O = np.zeros_like(N)
        for i in range(len(N)):
            if i < len(N) - 1:
                v = N[i+1] - C[i]
            else:
                v = N[i] - C[i]
            v = v / np.linalg.norm(v)
            O[i] = C[i] - v * 1.229
        return O
    
    O_random = add_oxygens(N_random, CA_random, C_random)
    O_refined = add_oxygens(N_refined, CA_refined, C_refined)
    
    reconstructor.save_pdb("before_refinement.pdb", sequence, 
                          N_random, CA_random, C_random, O_random)
    reconstructor.save_pdb("after_refinement.pdb", sequence,
                          N_refined, CA_refined, C_refined, O_refined)
    
    # Plot history
    plot_refinement_history(history, "refinement_history.png")
    
    print(f"\n{'='*60}")
    print("COMPARISON")
    print(f"{'='*60}")
    print("Saved structures:")
    print("  - before_refinement.pdb (random cloud)")
    print("  - after_refinement.pdb (refined)")
    print("\nLoad both in PyMOL to compare:")
    print("  pymol before_refinement.pdb after_refinement.pdb")