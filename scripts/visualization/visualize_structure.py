"""
Visualize and analyze predicted protein structures.

This script helps you:
1. Calculate basic statistics (bond lengths, angles)
2. Create Ramachandran plots
3. Validate the reconstruction quality
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def parse_pdb(pdb_file: str):
    """
    Parse PDB file and extract coordinates.
    
    Returns:
        Dict with atom coordinates: {'N': array, 'CA': array, 'C': array, 'O': array}
    """
    coords = {'N': [], 'CA': [], 'C': [], 'O': []}
    
    with open(pdb_file, 'r') as f:
        for line in f:
            if line.startswith('ATOM'):
                atom_name = line[12:16].strip()
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                
                if atom_name in coords:
                    coords[atom_name].append([x, y, z])
    
    # Convert to numpy arrays
    for atom in coords:
        coords[atom] = np.array(coords[atom])
    
    return coords


def calculate_bond_length(coords, atom1: str, atom2: str):
    """Calculate bond lengths between two atom types."""
    c1 = coords[atom1]
    c2 = coords[atom2]
    
    # Handle arrays of different lengths
    min_len = min(len(c1), len(c2))
    c1 = c1[:min_len]
    c2 = c2[:min_len]
    
    distances = np.linalg.norm(c1 - c2, axis=1)
    return distances


def calculate_bond_angle(p1, p2, p3):
    """
    Calculate angle between three points (p2 is the vertex).
    Returns angle in degrees.
    """
    v1 = p1 - p2
    v2 = p3 - p2
    
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Avoid numerical errors
    
    angle = np.arccos(cos_angle)
    return np.degrees(angle)


def calculate_dihedral_angle(p1, p2, p3, p4):
    """
    Calculate dihedral angle between four points.
    Returns angle in degrees (-180 to 180).
    """
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


def extract_torsion_angles(coords):
    """
    Extract phi and psi angles from backbone coordinates.
    
    Returns:
        phi_angles, psi_angles (in degrees)
    """
    N = coords['N']
    CA = coords['CA']
    C = coords['C']
    
    n_residues = len(CA)
    phi = np.zeros(n_residues)
    psi = np.zeros(n_residues)
    
    for i in range(n_residues):
        # Phi: C(i-1) - N(i) - CA(i) - C(i)
        if i > 0:
            phi[i] = calculate_dihedral_angle(
                C[i-1], N[i], CA[i], C[i]
            )
        
        # Psi: N(i) - CA(i) - C(i) - N(i+1)
        if i < n_residues - 1:
            psi[i] = calculate_dihedral_angle(
                N[i], CA[i], C[i], N[i+1]
            )
    
    return phi, psi


def analyze_structure(pdb_file: str):
    """Comprehensive structure analysis."""
    print(f"\n{'='*60}")
    print(f"ANALYZING: {pdb_file}")
    print(f"{'='*60}\n")
    
    # Parse PDB
    coords = parse_pdb(pdb_file)
    n_residues = len(coords['CA'])
    
    print(f"Residues: {n_residues}")
    print(f"Total atoms: {sum(len(coords[a]) for a in coords)}\n")
    
    # Bond length analysis
    print("BOND LENGTHS:")
    print("-" * 40)
    
    bond_pairs = [('N', 'CA'), ('CA', 'C'), ('C', 'O')]
    expected_lengths = {'N-CA': 1.458, 'CA-C': 1.523, 'C-O': 1.229}
    
    for atom1, atom2 in bond_pairs:
        if len(coords[atom1]) > 0 and len(coords[atom2]) > 0:
            lengths = calculate_bond_length(coords, atom1, atom2)
            bond_name = f"{atom1}-{atom2}"
            expected = expected_lengths.get(bond_name, 0)
            
            print(f"{bond_name:8s}: {lengths.mean():.3f} ± {lengths.std():.3f} Å", end='')
            if expected > 0:
                error = abs(lengths.mean() - expected)
                print(f"  (expected: {expected:.3f} Å, error: {error:.3f} Å)")
                if error > 0.05:
                    print(f"          ⚠️  Large error! Check reconstruction")
            else:
                print()
    
    # Bond angle analysis
    print("\nBOND ANGLES:")
    print("-" * 40)
    
    angles_list = []
    for i in range(1, n_residues):
        # N-CA-C angle
        angle = calculate_bond_angle(
            coords['N'][i], coords['CA'][i], coords['C'][i]
        )
        angles_list.append(('N-CA-C', angle, 110.99))
        
        # CA-C-N angle (with next residue)
        if i < n_residues - 1:
            angle = calculate_bond_angle(
                coords['CA'][i], coords['C'][i], coords['N'][i+1]
            )
            angles_list.append(('CA-C-N', angle, 116.64))
    
    # Print statistics
    for angle_name in ['N-CA-C', 'CA-C-N']:
        angles = [a[1] for a in angles_list if a[0] == angle_name]
        if angles:
            expected = angles_list[0][2] if angle_name == 'N-CA-C' else 116.64
            angles = np.array(angles)
            print(f"{angle_name:8s}: {angles.mean():.2f}° ± {angles.std():.2f}°", end='')
            error = abs(angles.mean() - expected)
            print(f"  (expected: {expected:.2f}°, error: {error:.2f}°)")
            if error > 5:
                print(f"          ⚠️  Large error! Check reconstruction")
    
    # Torsion angles
    print("\nTORSION ANGLES (PHI, PSI):")
    print("-" * 40)
    
    phi, psi = extract_torsion_angles(coords)
    
    # Exclude first/last residues (undefined angles)
    phi_valid = phi[1:]
    psi_valid = psi[:-1]
    
    print(f"Phi: {phi_valid.mean():.1f}° ± {phi_valid.std():.1f}°")
    print(f"     Range: [{phi_valid.min():.1f}°, {phi_valid.max():.1f}°]")
    print(f"Psi: {psi_valid.mean():.1f}° ± {psi_valid.std():.1f}°")
    print(f"     Range: [{psi_valid.min():.1f}°, {psi_valid.max():.1f}°]")
    
    # Classify secondary structure
    print("\nSECONDARY STRUCTURE ESTIMATE:")
    print("-" * 40)
    
    helix_count = np.sum((phi_valid > -90) & (phi_valid < -30) & 
                         (psi_valid > -75) & (psi_valid < -15))
    beta_count = np.sum((phi_valid > -150) & (phi_valid < -90) & 
                        (psi_valid > 90) & (psi_valid < 180))
    
    total = len(phi_valid)
    print(f"Alpha helix: {helix_count}/{total} residues ({100*helix_count/total:.1f}%)")
    print(f"Beta strand: {beta_count}/{total} residues ({100*beta_count/total:.1f}%)")
    print(f"Other:       {total - helix_count - beta_count}/{total} residues")
    
    return coords, phi, psi


def plot_ramachandran(phi, psi, title="Ramachandran Plot", save_file=None):
    """Create Ramachandran plot of phi/psi angles."""
    # Remove first/last residues (undefined)
    phi_valid = phi[1:]
    psi_valid = psi[:-1]
    
    plt.figure(figsize=(8, 8))
    
    # Plot typical regions
    plt.axhspan(-75, -15, -90, -30, alpha=0.1, color='blue', label='α-helix region')
    plt.axhspan(90, 180, -150, -90, alpha=0.1, color='red', label='β-strand region')
    
    # Plot data points
    plt.scatter(phi_valid, psi_valid, c='black', s=50, alpha=0.6, edgecolors='white')
    
    plt.xlabel('Phi (φ) [degrees]', fontsize=12)
    plt.ylabel('Psi (ψ) [degrees]', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    
    plt.xlim(-180, 180)
    plt.ylim(-180, 180)
    plt.axhline(0, color='gray', linestyle='--', alpha=0.3)
    plt.axvline(0, color='gray', linestyle='--', alpha=0.3)
    
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    if save_file:
        plt.savefig(save_file, dpi=150, bbox_inches='tight')
        print(f"Saved Ramachandran plot to {save_file}")
    else:
        plt.show()
    
    plt.close()


def visualize_backbone_trace(coords, save_file=None):
    """Create 3D visualization of backbone trace."""
    CA = coords['CA']
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot CA trace
    ax.plot(CA[:, 0], CA[:, 1], CA[:, 2], 'b-', linewidth=2, label='Backbone trace')
    ax.scatter(CA[:, 0], CA[:, 1], CA[:, 2], c='red', s=50, alpha=0.6, label='CA atoms')
    
    # Mark N and C termini
    ax.scatter(*CA[0], c='green', s=200, marker='o', label='N-terminus')
    ax.scatter(*CA[-1], c='purple', s=200, marker='s', label='C-terminus')
    
    ax.set_xlabel('X (Å)')
    ax.set_ylabel('Y (Å)')
    ax.set_zlabel('Z (Å)')
    ax.set_title('Backbone Trace (CA atoms)', fontweight='bold')
    ax.legend()
    
    if save_file:
        plt.savefig(save_file, dpi=150, bbox_inches='tight')
        print(f"Saved backbone trace to {save_file}")
    else:
        plt.show()
    
    plt.close()


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) > 1:
        pdb_file = sys.argv[1]
    else:
        pdb_file = "test_helix.pdb"
    
    if not Path(pdb_file).exists():
        print(f"Error: File not found: {pdb_file}")
        print(f"Usage: python visualize_structure.py <pdb_file>")
        sys.exit(1)
    
    # Analyze structure
    coords, phi, psi = analyze_structure(pdb_file)
    
    # Create plots
    base_name = Path(pdb_file).stem
    plot_ramachandran(phi, psi, 
                     title=f"Ramachandran Plot: {base_name}",
                     save_file=f"{base_name}_ramachandran.png")
    
    visualize_backbone_trace(coords, 
                           save_file=f"{base_name}_trace.png")
    
    print(f"\n{'='*60}")
    print("✓ Analysis complete!")
    print(f"  - Ramachandran plot: {base_name}_ramachandran.png")
    print(f"  - Backbone trace: {base_name}_trace.png")
    print(f"{'='*60}\n")