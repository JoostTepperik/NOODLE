"""
Random backbone initialization using SE(3) frames.

Faithful to diffusion model approach (RFdiffusion, AlphaFold2):
- Each residue = rigid body with (N, CA, C) triangle
- Sample random SE(3) transformations
- Apply to ideal local geometry
- Creates random cloud with broken peptide bonds

References:
- OpenFold: openfold/utils/rigid_utils.py
- RFdiffusion: diffusion/tools/angle.py
"""

import numpy as np
from typing import Tuple

def get_ideal_backbone_frame():
    """
    Get ideal N-CA-C triangle in local coordinate frame.
    
    Frame definition:
    - CA at origin
    - C along positive x-axis
    - N in xy-plane with positive y
    
    Returns:
        dict with 'N', 'CA', 'C' positions
    """
    # Bond lengths (Angstroms)
    N_CA_LENGTH = 1.458
    CA_C_LENGTH = 1.526
    
    # N-CA-C bond angle (degrees → radians)
    NCA_C_ANGLE = np.radians(111.0)
    
    # CA at origin
    CA = np.array([0.0, 0.0, 0.0])
    
    # C along x-axis
    C = np.array([CA_C_LENGTH, 0.0, 0.0])
    
    # N in xy-plane
    # Angle is measured from CA-C to CA-N
    # So N is at angle (180 - 111) = 69° from negative x-axis
    angle_from_neg_x = np.pi - NCA_C_ANGLE
    N = np.array([
        -N_CA_LENGTH * np.cos(angle_from_neg_x),
        N_CA_LENGTH * np.sin(angle_from_neg_x),
        0.0
    ])
    
    return {'N': N, 'CA': CA, 'C': C}


IDEAL_FRAME = get_ideal_backbone_frame()


def random_quaternion_uniform(rng=None):
    """
    Sample uniformly from SO(3) via quaternions.
    
    Uses method from "Uniform Random Rotations" (Shoemake, 1992)
    
    Args:
        rng: numpy RandomState (if None, uses np.random)
        
    Returns:
        q: (4,) unit quaternion [w, x, y, z]
    """
    if rng is None:
        rng = np.random
    
    # Sample 3 uniform random numbers
    u1, u2, u3 = rng.uniform(0, 1, 3)
    
    # Shoemake's method for uniform quaternion
    q = np.array([
        np.sqrt(1 - u1) * np.sin(2 * np.pi * u2),
        np.sqrt(1 - u1) * np.cos(2 * np.pi * u2),
        np.sqrt(u1) * np.sin(2 * np.pi * u3),
        np.sqrt(u1) * np.cos(2 * np.pi * u3)
    ])
    
    # Convention: [w, x, y, z] with w as scalar part
    return np.array([q[3], q[0], q[1], q[2]])


def quaternion_to_rotation(q):
    """
    Convert unit quaternion to 3x3 rotation matrix.
    
    Args:
        q: (4,) quaternion [w, x, y, z]
        
    Returns:
        R: (3, 3) rotation matrix
    """
    w, x, y, z = q
    
    # Normalize (safety)
    norm = np.sqrt(w*w + x*x + y*y + z*z)
    w, x, y, z = w/norm, x/norm, y/norm, z/norm
    
    # Rotation matrix from quaternion
    R = np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - w*z),     2*(x*z + w*y)],
        [    2*(x*y + w*z), 1 - 2*(x*x + z*z),     2*(y*z - w*x)],
        [    2*(x*z - w*y),     2*(y*z + w*x), 1 - 2*(x*x + y*y)]
    ])
    
    return R


def make_se3_frame(rotation, translation):
    """
    Create 4x4 SE(3) transformation matrix.
    
    Args:
        rotation: (3, 3) rotation matrix
        translation: (3,) translation vector
        
    Returns:
        T: (4, 4) homogeneous transformation matrix
    """
    T = np.eye(4)
    T[:3, :3] = rotation
    T[:3, 3] = translation
    return T


def apply_se3_transform(transform, points):
    """
    Apply SE(3) transformation to points.
    
    Args:
        transform: (4, 4) SE(3) matrix OR (3, 3) rotation + (3,) translation
        points: (n, 3) or (3,) points
        
    Returns:
        transformed: same shape as points
    """
    if isinstance(transform, tuple):
        rotation, translation = transform
    else:
        rotation = transform[:3, :3]
        translation = transform[:3, 3]
    
    if points.ndim == 1:
        return rotation @ points + translation
    else:
        return (rotation @ points.T).T + translation


def generate_random_backbone_cloud(
    n_residues: int,
    position_scale: float = 10.0,
    seed: int = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, list]:
    """
    Generate random backbone cloud using SE(3) frames.
    
    Args:
        n_residues: number of residues
        position_scale: scale for random positions (Angstroms)
        seed: random seed for reproducibility
        
    Returns:
        N: (n_residues, 3) nitrogen positions
        CA: (n_residues, 3) alpha carbon positions  
        C: (n_residues, 3) carbon positions
        frames: list of (rotation, translation) tuples for each residue
    """
    rng = np.random.RandomState(seed)
    
    # Ideal local geometry
    n_local = IDEAL_FRAME['N']
    ca_local = IDEAL_FRAME['CA']
    c_local = IDEAL_FRAME['C']
    
    # Storage
    N = np.zeros((n_residues, 3))
    CA = np.zeros((n_residues, 3))
    C = np.zeros((n_residues, 3))
    frames = []
    
    for i in range(n_residues):
        # Sample random rotation (uniform on SO(3))
        q = random_quaternion_uniform(rng)
        rotation = quaternion_to_rotation(q)
        
        # Sample random translation
        # Using Gaussian ~ N(0, position_scale^2)
        translation = rng.randn(3) * position_scale
        
        # Apply SE(3) transform to local triangle
        N[i] = rotation @ n_local + translation
        CA[i] = rotation @ ca_local + translation
        C[i] = rotation @ c_local + translation
        
        frames.append((rotation, translation))
    
    return N, CA, C, frames

def check_local_geometry(N, CA, C):
    """
    Verify local N-CA-C geometry is preserved.
    
    Returns dict of bond lengths and angles for each residue.
    """
    n_residues = len(N)
    
    results = {
        'n_ca_bonds': [],
        'ca_c_bonds': [],
        'n_ca_c_angles': []
    }
    
    for i in range(n_residues):
        # Bond lengths
        n_ca = np.linalg.norm(CA[i] - N[i])
        ca_c = np.linalg.norm(C[i] - CA[i])
        
        # Angle
        v1 = N[i] - CA[i]
        v2 = C[i] - CA[i]
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        angle = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))
        
        results['n_ca_bonds'].append(n_ca)
        results['ca_c_bonds'].append(ca_c)
        results['n_ca_c_angles'].append(angle)
    
    return results


def check_peptide_bonds(N, CA, C):
    """
    Check C(i)-N(i+1) peptide bond lengths.
    
  
    """
    n_residues = len(N)
    peptide_bonds = []
    
    for i in range(n_residues - 1):
        bond_length = np.linalg.norm(N[i+1] - C[i])
        peptide_bonds.append(bond_length)
    
    return peptide_bonds


if __name__ == "__main__":
    print("Generating random backbone cloud...")
    print("="*60)
    
    # Generate cloud
    N, CA, C, frames = generate_random_backbone_cloud(
        n_residues=7,
        position_scale=10.0,
        seed=1
    )
    
    print(f"\n✓ Generated {len(N)} residues")
    print(f"  N shape: {N.shape}")
    print(f"  CA shape: {CA.shape}")
    print(f"  C shape: {C.shape}")
    
    # Check local geometry (should be PERFECT)
    print("\nLocal geometry (N-CA-C triangles):")
    geom = check_local_geometry(N, CA, C)
    
    print(f"  N-CA bonds: {np.mean(geom['n_ca_bonds']):.4f} ± {np.std(geom['n_ca_bonds']):.4f} Å")
    print(f"  CA-C bonds: {np.mean(geom['ca_c_bonds']):.4f} ± {np.std(geom['ca_c_bonds']):.4f} Å")
    print(f"  N-CA-C angles: {np.mean(geom['n_ca_c_angles']):.2f} ± {np.std(geom['n_ca_c_angles']):.2f}°")
    print(f"  Expected: N-CA=1.458Å, CA-C=1.526Å, angle=111.0°")
    
    # Check peptide bonds (should be BROKEN)
    print("\nPeptide bonds (C-N between residues):")
    peptide = check_peptide_bonds(N, CA, C)
    print(f"  C-N distances: {np.mean(peptide):.2f} ± {np.std(peptide):.2f} Å")
    print(f"  Expected: ~1.33Å (but these are BROKEN in random cloud)")
    print(f"  Range: {np.min(peptide):.2f} - {np.max(peptide):.2f} Å")
    
    # Save as PDB
    from nerf_reconstruction import ProteinBackboneReconstructor
    reconstructor = ProteinBackboneReconstructor()
    
    # Add oxygen
    O = np.zeros_like(N)
    for i in range(len(N)):
        if i < len(N) - 1:
            v = N[i+1] - C[i]
        else:
            v = N[i] - C[i]
        v = v / np.linalg.norm(v)
        O[i] = C[i] - v * 1.229
    
    reconstructor.save_pdb(
        "random_cloud.pdb",
        "AAAAAAA",
        N, CA, C, O
    )
    
    print(f"\n✓ Saved: random_cloud.pdb")
    print("  Load in PyMOL to see the random cloud")
    print("  Each triangle (N-CA-C) has perfect geometry")
    print("  But peptide bonds between triangles are broken")