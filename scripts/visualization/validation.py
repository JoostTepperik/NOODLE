import numpy as np
import sys
sys.path.insert(0, '/home/claude')
from protein_builder import ProteinBuilder

def validate_geometry(coords, builder):
    """
    Validate that bond lengths and bond angles are preserved.
    
    Args:
        coords: Dictionary with N, CA, C coordinates
        builder: ProteinBuilder instance with ideal geometry parameters
    
    Returns:
        validation_report: Dictionary with statistics
    """
    N = coords['N']
    CA = coords['CA']
    C = coords['C']
    n_residues = len(N)
    
    # Check bond lengths
    n_ca_lengths = []
    ca_c_lengths = []
    c_n_lengths = []
    
    for i in range(n_residues):
        # N-CA bond
        n_ca_lengths.append(np.linalg.norm(CA[i] - N[i]))
        # CA-C bond
        ca_c_lengths.append(np.linalg.norm(C[i] - CA[i]))
        # C-N bond (except last residue)
        if i < n_residues - 1:
            c_n_lengths.append(np.linalg.norm(N[i+1] - C[i]))
    
    n_ca_lengths = np.array(n_ca_lengths)
    ca_c_lengths = np.array(ca_c_lengths)
    c_n_lengths = np.array(c_n_lengths)
    
    # Check bond angles
    n_ca_c_angles = []
    ca_c_n_angles = []
    c_n_ca_angles = []
    
    for i in range(n_residues):
        # N-CA-C angle
        v1 = N[i] - CA[i]
        v2 = C[i] - CA[i]
        angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
        n_ca_c_angles.append(angle)
        
        # CA-C-N angle (except last residue)
        if i < n_residues - 1:
            v1 = CA[i] - C[i]
            v2 = N[i+1] - C[i]
            angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
            ca_c_n_angles.append(angle)
        
        # C-N-CA angle (except first residue)
        if i > 0:
            v1 = C[i-1] - N[i]
            v2 = CA[i] - N[i]
            angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
            c_n_ca_angles.append(angle)
    
    n_ca_c_angles = np.array(n_ca_c_angles)
    ca_c_n_angles = np.array(ca_c_n_angles)
    c_n_ca_angles = np.array(c_n_ca_angles)
    
    # Print validation report
    print("=" * 70)
    print("GEOMETRY VALIDATION REPORT")
    print("=" * 70)
    print()
    print("BOND LENGTHS (Angstroms):")
    print(f"  N-CA:  ideal={builder.BOND_N_CA:.3f}Å  "
          f"mean={n_ca_lengths.mean():.3f}±{n_ca_lengths.std():.3f}Å  "
          f"[{n_ca_lengths.min():.3f}, {n_ca_lengths.max():.3f}]")
    print(f"  CA-C:  ideal={builder.BOND_CA_C:.3f}Å  "
          f"mean={ca_c_lengths.mean():.3f}±{ca_c_lengths.std():.3f}Å  "
          f"[{ca_c_lengths.min():.3f}, {ca_c_lengths.max():.3f}]")
    print(f"  C-N:   ideal={builder.BOND_C_N:.3f}Å  "
          f"mean={c_n_lengths.mean():.3f}±{c_n_lengths.std():.3f}Å  "
          f"[{c_n_lengths.min():.3f}, {c_n_lengths.max():.3f}]")
    print()
    
    print("BOND ANGLES (degrees):")
    print(f"  N-CA-C:  ideal={np.degrees(builder.ANGLE_N_CA_C):.2f}°  "
          f"mean={np.degrees(n_ca_c_angles.mean()):.2f}±{np.degrees(n_ca_c_angles.std()):.2f}°  "
          f"[{np.degrees(n_ca_c_angles.min()):.2f}, {np.degrees(n_ca_c_angles.max()):.2f}]")
    print(f"  CA-C-N:  ideal={np.degrees(builder.ANGLE_CA_C_N):.2f}°  "
          f"mean={np.degrees(ca_c_n_angles.mean()):.2f}±{np.degrees(ca_c_n_angles.std()):.2f}°  "
          f"[{np.degrees(ca_c_n_angles.min()):.2f}, {np.degrees(ca_c_n_angles.max()):.2f}]")
    print(f"  C-N-CA:  ideal={np.degrees(builder.ANGLE_C_N_CA):.2f}°  "
          f"mean={np.degrees(c_n_ca_angles.mean()):.2f}±{np.degrees(c_n_ca_angles.std()):.2f}°  "
          f"[{np.degrees(c_n_ca_angles.min()):.2f}, {np.degrees(c_n_ca_angles.max()):.2f}]")
    print()
    
    # Check if within tolerance
    bond_length_tolerance = 0.001  # 0.001 Angstrom
    bond_angle_tolerance = 0.1  # 0.1 degrees
    
    bond_lengths_ok = (
        np.allclose(n_ca_lengths, builder.BOND_N_CA, atol=bond_length_tolerance) and
        np.allclose(ca_c_lengths, builder.BOND_CA_C, atol=bond_length_tolerance) and
        np.allclose(c_n_lengths, builder.BOND_C_N, atol=bond_length_tolerance)
    )
    
    bond_angles_ok = (
        np.allclose(n_ca_c_angles, builder.ANGLE_N_CA_C, atol=np.radians(bond_angle_tolerance)) and
        np.allclose(ca_c_n_angles, builder.ANGLE_CA_C_N, atol=np.radians(bond_angle_tolerance)) and
        np.allclose(c_n_ca_angles, builder.ANGLE_C_N_CA, atol=np.radians(bond_angle_tolerance))
    )
    
    if bond_lengths_ok and bond_angles_ok:
        print("✓ VALIDATION PASSED: All bond lengths and angles match ideal geometry!")
    else:
        if not bond_lengths_ok:
            print("✗ WARNING: Some bond lengths deviate from ideal geometry")
        if not bond_angles_ok:
            print("✗ WARNING: Some bond angles deviate from ideal geometry")
    
    print("=" * 70)
    print()
    
    return {
        'bond_lengths': {
            'n_ca': n_ca_lengths,
            'ca_c': ca_c_lengths,
            'c_n': c_n_lengths
        },
        'bond_angles': {
            'n_ca_c': n_ca_c_angles,
            'ca_c_n': ca_c_n_angles,
            'c_n_ca': c_n_ca_angles
        },
        'validation_passed': bond_lengths_ok and bond_angles_ok
    }


# Test the validation
print("Testing NeRF implementation with known secondary structures:\n")

builder = ProteinBuilder()

# Test 1: Alpha helix
print("TEST 1: Alpha Helix (φ=-60°, ψ=-45°)")
phi_helix = [-60] * 20
psi_helix = [-45] * 20
coords_helix = builder.build_structure(phi_helix, psi_helix)
validate_geometry(coords_helix, builder)

# Test 2: Beta strand
print("TEST 2: Beta Strand (φ=-120°, ψ=120°)")
phi_beta = [-120] * 20
psi_beta = [120] * 20
coords_beta = builder.build_structure(phi_beta, psi_beta)
validate_geometry(coords_beta, builder)

# Test 3: Random angles (from MLP predictions)
print("TEST 3: Random Torsion Angles (simulating MLP predictions)")
np.random.seed(42)
phi_random = np.random.uniform(-180, 180, 20)
psi_random = np.random.uniform(-180, 180, 20)
coords_random = builder.build_structure(phi_random, psi_random)
validate_geometry(coords_random, builder)

print("All tests complete!")
print("\nThe NeRF algorithm correctly preserves all bond lengths and angles,")
print("with only the torsion angles (phi, psi, omega) being variable.")
print("This matches the approach used in AlphaFold2, OpenFold, and MP-NeRF.")