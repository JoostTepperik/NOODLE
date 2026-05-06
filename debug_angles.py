# Debug script: scripts/debug_angles.py
import h5py
import numpy as np
import matplotlib.pyplot as plt

with h5py.File('data/training_diverse_2k/training_data.h5', 'r') as f:
    phi = f['phi_continuous'][:]
    psi = f['psi_continuous'][:]
    
    print("Phi angles:")
    print(f"  Range: [{phi.min():.1f}, {phi.max():.1f}]")
    print(f"  Mean: {phi.mean():.1f}°")
    print(f"  Std: {phi.std():.1f}°")
    print(f"  NaN count: {np.isnan(phi).sum()}")
    
    print("\nPsi angles:")
    print(f"  Range: [{psi.min():.1f}, {psi.max():.1f}]")
    print(f"  Mean: {psi.mean():.1f}°")
    print(f"  Std: {psi.std():.1f}°")
    print(f"  NaN count: {np.isnan(psi).sum()}")
    
    # Plot distributions
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.hist(phi, bins=72, range=(-180, 180), alpha=0.7)
    ax1.set_xlabel('φ (degrees)')
    ax1.set_ylabel('Count')
    ax1.set_title('Phi Distribution')
    ax1.axvline(-60, color='r', linestyle='--', alpha=0.5, label='α-helix')
    ax1.legend()
    
    ax2.hist(psi, bins=72, range=(-180, 180), alpha=0.7)
    ax2.set_xlabel('ψ (degrees)')
    ax2.set_ylabel('Count')
    ax2.set_title('Psi Distribution')
    ax2.axvline(-45, color='r', linestyle='--', alpha=0.5, label='α-helix')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('angle_distributions.png', dpi=150)
    print("\nSaved angle_distributions.png")
    
    # Check Ramachandran
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.hexbin(phi, psi, gridsize=50, cmap='Blues', mincnt=1)
    ax.set_xlabel('φ (degrees)')
    ax.set_ylabel('ψ (degrees)')
    ax.set_title('Ramachandran Plot')
    ax.set_xlim(-180, 180)
    ax.set_ylim(-180, 180)
    ax.grid(True, alpha=0.3)
    plt.savefig('ramachandran_data.png', dpi=150)
    print("Saved ramachandran_data.png")