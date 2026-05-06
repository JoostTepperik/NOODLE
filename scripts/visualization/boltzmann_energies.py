"""
Convert binned predictions to Boltzmann energies.

From probability distribution p(bin) to energy landscape E(bin):
    E(bin) = -kT * ln(p(bin))
    
Or relative to most probable state:
    ΔE(bin) = -kT * ln(p(bin) / p_max)
"""

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# Physical constants
R_GAS = 8.314  # J/(mol·K) - gas constant
T_ROOM = 298.15  # K - room temperature (25°C)
kT_ROOM = R_GAS * T_ROOM / 1000  # kJ/mol at room temp ≈ 2.48 kJ/mol


def logits_to_energies(
    logits: np.ndarray,
    temperature: float = T_ROOM,
    reference: str = 'max'
) -> np.ndarray:
    """
    Convert logits to Boltzmann energies.
    
    Args:
        logits: (n_bins,) or (..., n_bins) logits from model
        temperature: Temperature in Kelvin (default 298.15 K = 25°C)
        reference: 'max' (most probable = 0) or 'absolute' (use raw probabilities)
        
    Returns:
        energies: Boltzmann energies in kJ/mol
    """
    # Get probabilities
    probs = jax.nn.softmax(logits, axis=-1)
    
    # Calculate kT in kJ/mol
    kT = R_GAS * temperature / 1000  # kJ/mol
    
    # Avoid log(0) by adding small epsilon
    eps = 1e-10
    probs = np.maximum(probs, eps)
    
    if reference == 'max':
        # Energy relative to most probable state
        p_max = np.max(probs, axis=-1, keepdims=True)
        energies = -kT * np.log(probs / p_max)
    else:
        # Absolute free energy (includes partition function)
        energies = -kT * np.log(probs)
    
    return energies


def get_angle_energy_landscape(
    logits_phi: np.ndarray,
    logits_psi: np.ndarray,
    n_bins: int = 36,
    temperature: float = T_ROOM
) -> tuple:
    """
    Get energy landscape for phi/psi angles.
    
    Args:
        logits_phi: (seq_len, n_bins) logits for phi
        logits_psi: (seq_len, n_bins) logits for psi
        n_bins: number of bins (default 36)
        temperature: temperature in K
        
    Returns:
        phi_angles, psi_angles, phi_energies, psi_energies
    """
    # Convert logits to energies
    phi_energies = logits_to_energies(logits_phi, temperature, reference='max')
    psi_energies = logits_to_energies(logits_psi, temperature, reference='max')
    
    # Get angle bins
    bin_width = 360.0 / n_bins
    angles = np.array([-180.0 + (i + 0.5) * bin_width for i in range(n_bins)])
    
    return angles, angles, phi_energies, psi_energies


def plot_energy_landscape(
    logits_phi: np.ndarray,
    logits_psi: np.ndarray,
    residue_idx: int = 0,
    n_bins: int = 36,
    temperature: float = T_ROOM,
    save_path: str = None
):
    """
    Plot energy landscape for a single residue.
    
    Args:
        logits_phi: (seq_len, n_bins) phi logits
        logits_psi: (seq_len, n_bins) psi logits
        residue_idx: which residue to plot (default 0)
        n_bins: number of bins
        temperature: temperature in K
        save_path: optional path to save figure
    """
    angles, _, phi_energies, psi_energies = get_angle_energy_landscape(
        logits_phi, logits_psi, n_bins, temperature
    )
    
    # Get energies for this residue
    E_phi = phi_energies[residue_idx]
    E_psi = psi_energies[residue_idx]
    
    # Get most probable angles
    phi_probs = jax.nn.softmax(logits_phi[residue_idx])
    psi_probs = jax.nn.softmax(logits_psi[residue_idx])
    
    phi_predicted = angles[np.argmax(phi_probs)]
    psi_predicted = angles[np.argmax(psi_probs)]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Phi energy landscape
    ax = axes[0]
    ax.plot(angles, E_phi, 'b-', linewidth=2)
    ax.axvline(phi_predicted, color='red', linestyle='--', label=f'Predicted: {phi_predicted:.1f}°')
    ax.fill_between(angles, E_phi, alpha=0.3)
    ax.set_xlabel('Phi (φ) [degrees]', fontsize=12)
    ax.set_ylabel('Energy [kJ/mol]', fontsize=12)
    ax.set_title(f'Phi Energy Landscape (Residue {residue_idx})', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_xlim(-180, 180)
    
    # Psi energy landscape
    ax = axes[1]
    ax.plot(angles, E_psi, 'g-', linewidth=2)
    ax.axvline(psi_predicted, color='red', linestyle='--', label=f'Predicted: {psi_predicted:.1f}°')
    ax.fill_between(angles, E_psi, alpha=0.3)
    ax.set_xlabel('Psi (ψ) [degrees]', fontsize=12)
    ax.set_ylabel('Energy [kJ/mol]', fontsize=12)
    ax.set_title(f'Psi Energy Landscape (Residue {residue_idx})', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_xlim(-180, 180)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved energy landscape to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_2d_energy_landscape(
    logits_phi: np.ndarray,
    logits_psi: np.ndarray,
    residue_idx: int = 0,
    n_bins: int = 36,
    temperature: float = T_ROOM,
    save_path: str = None
):
    """
    Plot 2D Ramachandran-style energy landscape.
    
    Args:
        logits_phi: (seq_len, n_bins) phi logits
        logits_psi: (seq_len, n_bins) psi logits
        residue_idx: which residue to plot
        n_bins: number of bins
        temperature: temperature in K
        save_path: optional path to save
    """
    # Get probabilities
    phi_probs = jax.nn.softmax(logits_phi[residue_idx])
    psi_probs = jax.nn.softmax(logits_psi[residue_idx])
    
    # Create 2D probability distribution (assuming independence)
    probs_2d = np.outer(psi_probs, phi_probs)
    
    # Convert to energy
    kT = R_GAS * temperature / 1000
    eps = 1e-10
    probs_2d = np.maximum(probs_2d, eps)
    energies_2d = -kT * np.log(probs_2d / np.max(probs_2d))
    
    # Get angles
    bin_width = 360.0 / n_bins
    angles = np.array([-180.0 + (i + 0.5) * bin_width for i in range(n_bins)])
    
    # Get predicted angles
    phi_predicted = angles[np.argmax(phi_probs)]
    psi_predicted = angles[np.argmax(psi_probs)]
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 9))
    
    # Energy contours
    levels = np.arange(0, 15, 1)  # Energy levels in kJ/mol
    contour = ax.contourf(angles, angles, energies_2d, levels=levels, cmap='RdYlBu_r')
    ax.contour(angles, angles, energies_2d, levels=levels, colors='black', alpha=0.2, linewidths=0.5)
    
    # Mark predicted point
    ax.plot(phi_predicted, psi_predicted, 'r*', markersize=20, 
            label=f'Predicted: ({phi_predicted:.1f}°, {psi_predicted:.1f}°)',
            markeredgecolor='black', markeredgewidth=1.5)
    
    # Mark common regions
    # Alpha helix region
    ax.add_patch(plt.Rectangle((-90, -75), 60, 60, 
                               fill=False, edgecolor='blue', linewidth=2, 
                               linestyle='--', label='α-helix region'))
    
    # Beta sheet region
    ax.add_patch(plt.Rectangle((-150, 90), 60, 90, 
                               fill=False, edgecolor='green', linewidth=2,
                               linestyle='--', label='β-sheet region'))
    
    ax.set_xlabel('Phi (φ) [degrees]', fontsize=12)
    ax.set_ylabel('Psi (ψ) [degrees]', fontsize=12)
    ax.set_title(f'2D Energy Landscape (Residue {residue_idx})', fontsize=14, fontweight='bold')
    ax.set_xlim(-180, 180)
    ax.set_ylim(-180, 180)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left')
    
    # Colorbar
    cbar = plt.colorbar(contour, ax=ax)
    cbar.set_label('Energy [kJ/mol]', fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved 2D energy landscape to {save_path}")
    else:
        plt.show()
    
    plt.close()


def analyze_energy_distribution(
    logits_phi: np.ndarray,
    logits_psi: np.ndarray,
    residue_idx: int = 0,
    n_bins: int = 36,
    temperature: float = T_ROOM
):
    """
    Analyze energy distribution for a residue.
    
    Prints statistics about the energy landscape.
    """
    angles, _, phi_energies, psi_energies = get_angle_energy_landscape(
        logits_phi, logits_psi, n_bins, temperature
    )
    
    E_phi = phi_energies[residue_idx]
    E_psi = psi_energies[residue_idx]
    
    # Get probabilities
    phi_probs = jax.nn.softmax(logits_phi[residue_idx])
    psi_probs = jax.nn.softmax(logits_psi[residue_idx])
    
    print(f"\n{'='*60}")
    print(f"ENERGY ANALYSIS - Residue {residue_idx}")
    print(f"{'='*60}")
    print(f"Temperature: {temperature:.2f} K ({temperature-273.15:.1f}°C)")
    print(f"kT = {R_GAS * temperature / 1000:.3f} kJ/mol")
    
    print(f"\nPHI ANGLE:")
    print(f"  Minimum energy: {E_phi.min():.3f} kJ/mol at {angles[np.argmin(E_phi)]:.1f}°")
    print(f"  Maximum energy: {E_phi.max():.3f} kJ/mol at {angles[np.argmax(E_phi)]:.1f}°")
    print(f"  Energy range: {E_phi.max() - E_phi.min():.3f} kJ/mol")
    print(f"  Most probable: {angles[np.argmax(phi_probs)]:.1f}° (p={phi_probs.max():.4f})")
    print(f"  Entropy: {-np.sum(phi_probs * np.log(phi_probs + 1e-10)):.3f}")
    
    print(f"\nPSI ANGLE:")
    print(f"  Minimum energy: {E_psi.min():.3f} kJ/mol at {angles[np.argmin(E_psi)]:.1f}°")
    print(f"  Maximum energy: {E_psi.max():.3f} kJ/mol at {angles[np.argmax(E_psi)]:.1f}°")
    print(f"  Energy range: {E_psi.max() - E_psi.min():.3f} kJ/mol")
    print(f"  Most probable: {angles[np.argmax(psi_probs)]:.1f}° (p={psi_probs.max():.4f})")
    print(f"  Entropy: {-np.sum(psi_probs * np.log(psi_probs + 1e-10)):.3f}")
    
    # Model confidence
    print(f"\nMODEL CONFIDENCE:")
    phi_confidence = phi_probs.max()
    psi_confidence = psi_probs.max()
    
    if phi_confidence > 0.5:
        print(f"  Phi: HIGH confidence (p={phi_confidence:.3f})")
    elif phi_confidence > 0.3:
        print(f"  Phi: MEDIUM confidence (p={phi_confidence:.3f})")
    else:
        print(f"  Phi: LOW confidence (p={phi_confidence:.3f})")
        
    if psi_confidence > 0.5:
        print(f"  Psi: HIGH confidence (p={psi_confidence:.3f})")
    elif psi_confidence > 0.3:
        print(f"  Psi: MEDIUM confidence (p={psi_confidence:.3f})")
    else:
        print(f"  Psi: LOW confidence (p={psi_confidence:.3f})")
    
    print(f"{'='*60}\n")


# Example usage
if __name__ == "__main__":
    # Simulate some logits (would come from your model)
    n_bins = 36
    seq_len = 5
    
    # Create example logits (peaked around alpha helix angles)
    # Phi ~ -60° (bin ~12), Psi ~ -45° (bin ~13)
    logits_phi = np.random.randn(seq_len, n_bins) * 0.5
    logits_psi = np.random.randn(seq_len, n_bins) * 0.5
    
    # Make bin 12 and 13 more probable
    logits_phi[:, 12] += 3.0
    logits_psi[:, 13] += 3.0
    
    # Analyze
    analyze_energy_distribution(logits_phi, logits_psi, residue_idx=0)
    
    # Plot 1D energy landscapes
    plot_energy_landscape(logits_phi, logits_psi, residue_idx=0, 
                         save_path="energy_landscape_1d.png")
    
    # Plot 2D Ramachandran-style energy landscape
    plot_2d_energy_landscape(logits_phi, logits_psi, residue_idx=0,
                            save_path="energy_landscape_2d.png")