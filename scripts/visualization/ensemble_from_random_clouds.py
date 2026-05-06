"""
Ensemble refinement from random backbone clouds.

Pipeline:
1. Generate random backbone cloud (broken peptide bonds, diverse angles)
2. Extract phi/psi angles from random cloud
3. Refine angles using energy landscape gradients
   - Each converges to LOCAL minimum (not global)
   - Creates diverse ensemble of realistic structures
4. Rebuild with NeRF (restores peptide bonds)

The random cloud provides initial diversity.
Energy refinement finds realistic conformations.
Local minima = ensemble of valid structures.
"""

import sys

import json
sys.path.append('/home/jtepperik/thesis/energy_model/scripts')

import jax
import jax.numpy as jnp
import numpy as np
from pathlib import Path
import orbax.checkpoint as ocp
from models.full_model import TorsionPredictor
import matplotlib.pyplot as plt
from flax.training import checkpoints, train_state
from random_backbone_cloud import generate_random_backbone_cloud
from nerf_reconstruction import ProteinBackboneReconstructor


# ============================================================================
# MODEL LOADING
# ============================================================================

def load_model(config_path=None):
    checkpoint_dir = '/home/jtepperik/thesis/energy_model/scripts/training/outputs/energy_loss_c3'
    config_path = config_path or Path(checkpoint_dir) / 'config.json'
    with open(config_path) as f:
        config = json.load(f)

    model = TorsionPredictor(
        max_context   = config['max_context'],    # 21, not 3
        embed_dim     = config['embed_dim'],
        hidden_dim    = config['hidden_dim'],
        n_layers      = config['n_layers'],
        dropout_rate  = config['dropout_rate'],
        n_bins        = config['n_bins'],         # 36 for joint model
    )

    state = checkpoints.restore_checkpoint(
        ckpt_dir = Path(checkpoint_dir) / 'checkpoints',
        target   = None,
        prefix   = 'best_',
    )
    params = state['params']
    return model, params

# ============================================================================
# STRUCTURE ALIGNMENT (KABSCH ALGORITHM)
# ============================================================================

def align_structure_to_reference(N, CA, C, N_ref, CA_ref, C_ref):
    """
    Align structure to reference using Kabsch algorithm.
    
    Aligns based on CA atoms (most stable).
    
    Args:
        N, CA, C: Structure to align
        N_ref, CA_ref, C_ref: Reference structure
        
    Returns:
        N_aligned, CA_aligned, C_aligned: Aligned coordinates
    """
    # Center both structures on their CA centroids
    centroid = np.mean(CA, axis=0)
    centroid_ref = np.mean(CA_ref, axis=0)
    
    CA_centered = CA - centroid
    CA_ref_centered = CA_ref - centroid_ref
    
    # Kabsch algorithm: find optimal rotation
    H = CA_centered.T @ CA_ref_centered  # Cross-covariance matrix
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    
    # Ensure proper rotation (not reflection)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    
    # Apply rotation and translation to all atoms
    N_aligned = (R @ (N - centroid).T).T + centroid_ref
    CA_aligned = (R @ (CA - centroid).T).T + centroid_ref
    C_aligned = (R @ (C - centroid).T).T + centroid_ref
    
    return N_aligned, CA_aligned, C_aligned


def calculate_rmsd(CA1, CA2):
    """Calculate RMSD between two sets of CA atoms."""
    diff = CA1 - CA2
    return np.sqrt(np.mean(np.sum(diff**2, axis=1)))


# ============================================================================
# ENERGY CALCULATION
# ============================================================================

AA_TO_IDX = {
    'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, 'I': 7, 'K': 8,
    'L': 9, 'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14, 'S': 15, 'T': 16,
    'V': 17, 'W': 18, 'Y': 19, 'PAD': 20
}

N_BINS = 72
BIN_WIDTH = 360.0 / N_BINS


def get_all_energies_and_gradients_batched(model, params, sequence, phi, psi):
    """
    Get energies and gradients for ALL residues in one batch call.
    Much faster than calling model for each residue separately.
    """
    encoded = np.array([AA_TO_IDX[aa] for aa in sequence.upper()])
    seq_len = len(encoded)
    
    # Build windows for ALL residues at once
    batch_windows = []
    valid_residues = []  # Which residues to compute (exclude first/last)
    
    for position in range(1, seq_len - 1):  # Skip first and last
        window = []
        for pos in range(position - 3, position + 4):
            window.append(encoded[pos] if 0 <= pos < seq_len else 20)
        batch_windows.append(window)
        valid_residues.append(position)
    
    if len(batch_windows) == 0:
        return [], [], []
    
    # Single batched call to model
    batch_residues = jnp.array(batch_windows)  # (n_residues-2, 7)
    batch_mask = jnp.ones((len(batch_windows), 7), dtype=bool)
    
    logits_phi, logits_psi = model.apply(
        {'params': params}, batch_residues, batch_mask,
        training=False, rngs={'dropout': jax.random.PRNGKey(0)}
    )
    
    # Process all residues at once
    probs_phi = np.array(jax.nn.softmax(logits_phi, axis=-1))  # (n_residues-2, 72)
    probs_psi = np.array(jax.nn.softmax(logits_psi, axis=-1))
    
    # Interpolated energy function
    def interpolated_energy(angle, probs):
        idx = (angle + 180.0) / BIN_WIDTH
        idx_low = int(np.floor(idx)) % N_BINS
        idx_high = int(np.ceil(idx)) % N_BINS
        frac = idx - np.floor(idx)
        
        e_low = -np.log(probs[idx_low] + 1e-10)
        e_high = -np.log(probs[idx_high] + 1e-10)
        return (1 - frac) * e_low + frac * e_high
    
    energies = []
    grad_phis = []
    grad_psis = []
    
    delta = 2.5  # Half bin width
    
    for i, res_idx in enumerate(valid_residues):
        # Current energy
        e_phi = interpolated_energy(phi[res_idx], probs_phi[i])
        e_psi = interpolated_energy(psi[res_idx], probs_psi[i])
        
        # Gradients (central difference)
        e_phi_plus = interpolated_energy(phi[res_idx] + delta, probs_phi[i])
        e_phi_minus = interpolated_energy(phi[res_idx] - delta, probs_phi[i])
        e_psi_plus = interpolated_energy(psi[res_idx] + delta, probs_psi[i])
        e_psi_minus = interpolated_energy(psi[res_idx] - delta, probs_psi[i])
        
        grad_phi = (e_phi_plus - e_phi_minus) / (2 * delta)
        grad_psi = (e_psi_plus - e_psi_minus) / (2 * delta)
        
        energies.append(e_phi + e_psi)
        grad_phis.append((res_idx, grad_phi))
        grad_psis.append((res_idx, grad_psi))
    
    return energies, grad_phis, grad_psis


def coords_to_angles(N, CA, C):
    """Calculate phi, psi from coordinates."""
    n = len(N)
    phi = np.zeros(n)
    psi = np.zeros(n)
    
    def dihedral(p1, p2, p3, p4):
        """Calculate dihedral angle between 4 points (IUPAC convention)."""
        b0 = p1 - p2        # reversed direction — this is what fixes the sign
        b1 = p3 - p2
        b2 = p4 - p3

        b1_hat = b1 / (np.linalg.norm(b1) + 1e-10)
        v = b0 - np.dot(b0, b1_hat) * b1_hat   # b0 projected perp to b1
        w = b2 - np.dot(b2, b1_hat) * b1_hat   # b2 projected perp to b1

        x = np.dot(v, w)
        y = np.dot(np.cross(b1_hat, v), w)
        return np.degrees(np.arctan2(y, x))
    
    for i in range(n):
        if i > 0:
            phi[i] = dihedral(C[i-1], N[i], CA[i], C[i])
        if i < n - 1:
            psi[i] = dihedral(N[i], CA[i], C[i], N[i+1])
    
    return phi, psi


# ============================================================================
# REFINEMENT FROM RANDOM CLOUD
# ============================================================================

def refine_from_random_cloud(
    N_init, CA_init, C_init,
    sequence,
    model,
    params,
    n_steps=200,
    learning_rate=1.0,  # Increased from 1.0
    early_stop_threshold=0.01,
    verbose=True
):
    """
    Refine structure from random cloud.
    
    Optimizations:
    - Batched model calls (much faster)
    - Higher learning rate (faster convergence)
    - Early stopping (stop when converged)
    """
    if verbose:
        print(f"\n{'='*60}")
        print("REFINING FROM RANDOM CLOUD")
        print(f"{'='*60}")
        print(f"  Max steps: {n_steps}")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Early stop threshold: {early_stop_threshold}")
    
    # Extract initial angles from random cloud
    phi = coords_to_angles(N_init, CA_init, C_init)[0]
    psi = coords_to_angles(N_init, CA_init, C_init)[1]
    
    if verbose:
        print(f"\nInitial angles from random cloud:")
        print(f"  Phi range: [{phi.min():.1f}°, {phi.max():.1f}°]")
        print(f"  Psi range: [{psi.min():.1f}°, {psi.max():.1f}°]")
    
    energies = []
    converged_step = None
    
    for step in range(n_steps):
        # Batched energy and gradient computation (MUCH faster)
        residue_energies, grad_phis, grad_psis = get_all_energies_and_gradients_batched(
            model, params, sequence, phi, psi
        )
        
        total_energy = sum(residue_energies)
        energies.append(total_energy)
        
        # Update all angles at once
        for res_idx, grad_phi in grad_phis:
            phi[res_idx] -= learning_rate * grad_phi
            phi[res_idx] = np.clip(phi[res_idx], -180, 180)
        
        for res_idx, grad_psi in grad_psis:
            psi[res_idx] -= learning_rate * grad_psi
            psi[res_idx] = np.clip(psi[res_idx], -180, 180)
        
        if verbose and (step % 5 == 0 or step == n_steps - 1):  # Print every 5 steps
            avg_phi = np.mean(phi[1:-1])
            avg_psi = np.mean(psi[1:-1])
            print(f"  Step {step:3d}: E={total_energy:7.3f}, φ_avg={avg_phi:6.1f}°, ψ_avg={avg_psi:6.1f}°")
        
        # Early stopping: if energy change is small
        if step > 5:
            energy_change = abs(energies[-1] - energies[-6]) / 5  # Avg change over last 5 steps
            if energy_change < early_stop_threshold:
                converged_step = step
                if verbose:
                    print(f"\n  ✓ Converged at step {step} (energy change < {early_stop_threshold})")
                break
    
    if verbose:
        print(f"\n✓ Refinement complete")
        if converged_step:
            print(f"  Converged: step {converged_step}/{n_steps}")
        print(f"  Initial energy: {energies[0]:.3f}")
        print(f"  Final energy: {energies[-1]:.3f}")
        print(f"  Improvement: {energies[0] - energies[-1]:+.3f}")
        print(f"\nFinal angles:")
        print(f"  Phi range: [{phi.min():.1f}°, {phi.max():.1f}°]")
        print(f"  Psi range: [{psi.min():.1f}°, {psi.max():.1f}°]")
    
    # Rebuild structure using NeRF (restores peptide bonds!)
    reconstructor = ProteinBackboneReconstructor()
    N, CA, C, O = reconstructor.build_backbone(
        sequence=sequence,
        phi_angles=phi,
        psi_angles=psi
    )
    
    return N, CA, C, O, phi, psi, energies


# ============================================================================
# ENSEMBLE GENERATION
# ============================================================================

def generate_ensemble_from_random_clouds(
    sequence,
    model,
    params,
    n_structures=10,
    n_steps=200,
    learning_rate=1.0,
    position_scale=10.0,
    output_dir="ensemble"
):
    """
    Generate ensemble by refining multiple random clouds.
    
    Each random cloud converges to a LOCAL minimum → diverse ensemble.
    
    Args:
        sequence: Amino acid sequence
        model, params: Trained model
        n_structures: Number of structures to generate
        n_steps: Refinement steps per structure
        learning_rate: Gradient descent step size
        position_scale: Random cloud spread
        output_dir: Where to save structures
        
    Returns:
        List of (N, CA, C, O, phi, psi, final_energy) tuples
    """
    print(f"\n{'='*70}")
    print("ENSEMBLE GENERATION FROM RANDOM CLOUDS")
    print(f"{'='*70}")
    print(f"Sequence: {sequence}")
    print(f"Ensemble size: {n_structures}")
    print(f"Refinement steps: {n_steps}")
    print(f"{'='*70}\n")
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    reconstructor = ProteinBackboneReconstructor()
    ensemble = []
    reference_structure = None  # Will be set to first structure
    
    for i in range(n_structures):
        print(f"\n{'='*70}")
        print(f"STRUCTURE {i+1}/{n_structures}")
        print(f"{'='*70}")
        
        # Generate random cloud
        print("\nGenerating random backbone cloud...")
        N_init, CA_init, C_init, frames = generate_random_backbone_cloud(
            n_residues=len(sequence),
            position_scale=position_scale,
            seed=42 + i  # Different seed for each structure
        )
        
        # Save random cloud
        O_init = np.zeros_like(N_init)
        for j in range(len(sequence)):
            v = N_init[j+1] - C_init[j] if j < len(sequence) - 1 else N_init[j] - C_init[j]
            v /= np.linalg.norm(v)
            O_init[j] = C_init[j] - v * 1.229
        
        reconstructor.save_pdb(
            str(output_path / f"structure_{i:02d}_random.pdb"),
            sequence, N_init, CA_init, C_init, O_init
        )
        
        # Refine
        N, CA, C, O, phi, psi, energies = refine_from_random_cloud(
            N_init, CA_init, C_init,
            sequence,
            model,
            params,
            n_steps=n_steps,
            learning_rate=learning_rate,
            verbose=True
        )
        
        # Align to first structure (reference frame)
        if i == 0:
            # First structure becomes reference
            reference_structure = (N.copy(), CA.copy(), C.copy())
            N_aligned, CA_aligned, C_aligned = N, CA, C
            rmsd = 0.0
            print(f"\n  ✓ Structure {i+1} (REFERENCE)")
        else:
            # Align to reference
            N_ref, CA_ref, C_ref = reference_structure
            N_aligned, CA_aligned, C_aligned = align_structure_to_reference(
                N, CA, C, N_ref, CA_ref, C_ref
            )
            rmsd = calculate_rmsd(CA_aligned, CA_ref)
            print(f"\n  ✓ Structure {i+1} aligned to reference (RMSD: {rmsd:.3f} Å)")
        
        # Recompute oxygen for aligned structure
        O_aligned = np.zeros_like(N_aligned)
        for j in range(len(sequence)):
            v = N_aligned[j+1] - C_aligned[j] if j < len(sequence) - 1 else N_aligned[j] - C_aligned[j]
            v /= np.linalg.norm(v)
            O_aligned[j] = C_aligned[j] - v * 1.229
        
        # Save aligned structure
        reconstructor.save_pdb(
            str(output_path / f"structure_{i:02d}_refined.pdb"),
            sequence, N_aligned, CA_aligned, C_aligned, O_aligned
        )
        
        ensemble.append((N_aligned, CA_aligned, C_aligned, O_aligned, phi, psi, energies[-1], rmsd))
        
        print(f"  Final energy: {energies[-1]:.3f}")
    
    # Summary
    print(f"\n{'='*70}")
    print("ENSEMBLE SUMMARY")
    print(f"{'='*70}\n")
    
    final_energies = [e for *_, e, rmsd in ensemble]
    rmsds = [rmsd for *_, rmsd in ensemble]
    
    print(f"Final energies:")
    print(f"  Mean: {np.mean(final_energies):.3f}")
    print(f"  Std:  {np.std(final_energies):.3f}")
    print(f"  Range: [{np.min(final_energies):.3f}, {np.max(final_energies):.3f}]")
    
    print(f"\nRMSD from reference:")
    print(f"  Mean: {np.mean(rmsds[1:]):.3f} Å")  # Skip reference (rmsd=0)
    print(f"  Std:  {np.std(rmsds[1:]):.3f} Å")
    print(f"  Range: [{np.min(rmsds[1:]):.3f}, {np.max(rmsds[1:]):.3f}] Å")
    
    # Plot energy distribution
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.hist(final_energies, bins=min(10, n_structures), edgecolor='black')
    plt.xlabel('Final Energy')
    plt.ylabel('Count')
    plt.title('Energy Distribution Across Ensemble')
    plt.grid(True, alpha=0.3)
    
    # Plot RMSD distribution
    plt.subplot(1, 3, 2)
    plt.hist(rmsds[1:], bins=min(10, n_structures-1), edgecolor='black', color='orange')
    plt.xlabel('RMSD from Reference (Å)')
    plt.ylabel('Count')
    plt.title('Structural Diversity (RMSD)')
    plt.grid(True, alpha=0.3)
    
    # Plot angle distribution (Ramachandran)
    plt.subplot(1, 3, 3)
    all_phi = []
    all_psi = []
    for *_, phi, psi, _, _ in ensemble:
        all_phi.extend(phi[1:-1])  # Exclude first/last
        all_psi.extend(psi[1:-1])
    
    plt.scatter(all_phi, all_psi, alpha=0.5, s=20)
    plt.axvline(-60, color='red', linestyle='--', alpha=0.3, label='α-helix φ')
    plt.axhline(-45, color='red', linestyle='--', alpha=0.3, label='α-helix ψ')
    plt.xlabel('Phi (degrees)')
    plt.ylabel('Psi (degrees)')
    plt.title('Ramachandran Plot (All Residues)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / 'ensemble_analysis.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved: {output_path}/ensemble_analysis.png")
    
    # Save all structures in a single multi-model PDB for comparison
    print(f"\nCreating multi-model PDB for easy comparison...")
    with open(output_path / 'ensemble_all_aligned.pdb', 'w') as f:
        for model_num, (N, CA, C, O, _, _, _, _) in enumerate(ensemble, 1):
            f.write(f"MODEL     {model_num:4d}\n")
            
            atom_num = 1
            for res_num, aa in enumerate(sequence, 1):
                # N
                f.write(f"ATOM  {atom_num:5d}  N   {aa:3s} A{res_num:4d}    "
                       f"{N[res_num-1,0]:8.3f}{N[res_num-1,1]:8.3f}{N[res_num-1,2]:8.3f}"
                       f"  1.00  0.00           N  \n")
                atom_num += 1
                # CA
                f.write(f"ATOM  {atom_num:5d}  CA  {aa:3s} A{res_num:4d}    "
                       f"{CA[res_num-1,0]:8.3f}{CA[res_num-1,1]:8.3f}{CA[res_num-1,2]:8.3f}"
                       f"  1.00  0.00           C  \n")
                atom_num += 1
                # C
                f.write(f"ATOM  {atom_num:5d}  C   {aa:3s} A{res_num:4d}    "
                       f"{C[res_num-1,0]:8.3f}{C[res_num-1,1]:8.3f}{C[res_num-1,2]:8.3f}"
                       f"  1.00  0.00           C  \n")
                atom_num += 1
                # O
                f.write(f"ATOM  {atom_num:5d}  O   {aa:3s} A{res_num:4d}    "
                       f"{O[res_num-1,0]:8.3f}{O[res_num-1,1]:8.3f}{O[res_num-1,2]:8.3f}"
                       f"  1.00  0.00           O  \n")
                atom_num += 1
            
            f.write("ENDMDL\n")
    
    print(f"✓ Saved: {output_path}/ensemble_all_aligned.pdb")
    print(f"  Load in PyMOL with: load {output_path}/ensemble_all_aligned.pdb")
    
    print(f"\n{'='*70}")
    print(f"DONE - {n_structures} structures in {output_dir}/")
    print(f"{'='*70}\n")
    
    return ensemble


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    # Load model
    model, params = load_model()
    
    # Generate ensemble (fewer steps needed with optimizations)
    sequence = "CRVNHVTLSQPKIVKW"
    ensemble = generate_ensemble_from_random_clouds(
        sequence=sequence,
        model=model,
        params=params,
        n_structures=5,
        n_steps=200,  # Reduced from 50 - faster convergence
        learning_rate=3.0,  # Increased from 1.0
        position_scale=10.0,
        output_dir="ensemble_alanine"
    )