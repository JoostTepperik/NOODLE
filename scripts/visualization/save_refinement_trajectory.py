"""
Generate refinement trajectory for movie/animation.

Shows the evolution from random cloud to refined structure.
Saves structures at regular intervals during gradient descent.
"""

import sys
sys.path.append('/home/jtepperik/thesis/energy_model/scripts')

import jax
import jax.numpy as jnp
import numpy as np
from pathlib import Path
import json

from models.full_model import TorsionPredictor
import orbax.checkpoint as ocp

from random_backbone_cloud import generate_random_backbone_cloud
from nerf_reconstruction import ProteinBackboneReconstructor
from ensemble_from_random_clouds import (
    load_model,
    coords_to_angles,
    get_all_energies_and_gradients_batched,
    align_structure_to_reference,
    calculate_rmsd
)


def measure_peptide_bond_lengths(C, N):
    """
    Measure actual C(i)-N(i+1) distances from coordinates.
    
    Args:
        C: C atom coordinates (n_res, 3)
        N: N atom coordinates (n_res, 3)
        
    Returns:
        bond_lengths: Array of C(i)-N(i+1) distances (n_res-1,)
    """
    bond_lengths = []
    for i in range(len(C) - 1):
        distance = np.linalg.norm(N[i+1] - C[i])
        bond_lengths.append(distance)
    return np.array(bond_lengths)


def refine_with_trajectory(
    sequence,
    model,
    params,
    n_steps=50,
    save_every=5,
    learning_rate=3.0,
    native_CA=None  # For alignment if provided
):
    """
    Refine structure and save trajectory with gradual bond healing.
    
    Key feature: Peptide bonds gradually shorten from broken → 1.329Å
    This creates smooth animation instead of sudden snap.
    
    Args:
        sequence: Amino acid sequence
        model, params: Trained model
        n_steps: Total gradient descent steps
        save_every: Save structure every N steps
        learning_rate: Gradient descent learning rate
        native_CA: Native structure for alignment (optional)
        
    Returns:
        trajectory: List of (step, N, CA, C, O, phi, psi, energy, bond_length)
        final_structure: Final refined structure
    """
    reconstructor = ProteinBackboneReconstructor()
    n_res = len(sequence)
    
    print(f"\nGenerating refinement trajectory for: {sequence}")
    print(f"  Steps: {n_steps}, saving every {save_every} steps")
    
    # Step 1: Generate random backbone cloud
    N_random, CA_random, C_random, O_random = generate_random_backbone_cloud(sequence, seed=42)
    
    # Measure initial broken bond lengths
    initial_bond_lengths = measure_peptide_bond_lengths(C_random, N_random)
    mean_initial = np.mean(initial_bond_lengths)
    print(f"  Initial peptide bonds: {mean_initial:.2f} ± {np.std(initial_bond_lengths):.2f} Å (broken)")
    print(f"  Target: 1.329 Å (standard)")
    
    # Step 2: Extract initial angles from random cloud
    phi, psi = coords_to_angles(N_random, CA_random, C_random)
    
    print(f"  Initial random angles:")
    print(f"    Phi range: [{np.min(phi[1:-1]):.1f}, {np.max(phi[1:-1]):.1f}]°")
    print(f"    Psi range: [{np.min(psi[:-1]):.1f}, {np.max(psi[:-1]):.1f}]°")
    
    # Trajectory storage
    trajectory = []
    
    # Save initial state (random cloud - KEEP broken bonds!)
    energies_init, _, _ = get_all_energies_and_gradients_batched(model, params, sequence, phi, psi)
    energy = sum(energies_init)
    
    trajectory.append((0, N_random.copy(), CA_random.copy(), C_random.copy(), O_random.copy(), 
                      phi.copy(), psi.copy(), energy, mean_initial))
    print(f"  Step 0: E = {energy:.2f}, bond = {mean_initial:.2f} Å")
    
    # Step 3: Gradient descent with gradual bond healing
    for step in range(1, n_steps + 1):
        # Get energies and gradients
        energies, grad_phis, grad_psis = get_all_energies_and_gradients_batched(
            model, params, sequence, phi, psi
        )
        
        # Convert gradient lists to arrays
        grad_phi_array = np.zeros(n_res)
        grad_psi_array = np.zeros(n_res)
        
        for res_idx, grad in grad_phis:
            grad_phi_array[res_idx] = grad
        for res_idx, grad in grad_psis:
            grad_psi_array[res_idx] = grad
        
        # Update angles
        phi[1:-1] -= learning_rate * grad_phi_array[1:-1]
        psi[:-1] -= learning_rate * grad_psi_array[:-1]
        
        # Clip to valid range
        phi = np.clip(phi, -180, 180)
        psi = np.clip(psi, -180, 180)
        
        # Calculate energy
        energy = sum(energies)
        
        # Save at intervals
        if step % save_every == 0 or step == n_steps:
            # Calculate target peptide bond length (linear interpolation)
            progress = step / n_steps
            target_bond_length = mean_initial * (1 - progress) + 1.329 * progress
            
            # Rebuild with NeRF using current bond length
            N_rebuilt, CA_rebuilt, C_rebuilt, O_rebuilt = reconstructor.build_backbone(
                sequence, phi, psi, peptide_bond_length=target_bond_length
            )
            
            trajectory.append((step, N_rebuilt, CA_rebuilt, C_rebuilt, O_rebuilt,
                             phi.copy(), psi.copy(), energy, target_bond_length))
            
            print(f"  Step {step}: E = {energy:.2f}, bond = {target_bond_length:.2f} Å")
    
    print(f"\n  Saved {len(trajectory)} frames")
    print(f"  Energy: {trajectory[0][7]:.2f} → {trajectory[-1][7]:.2f}")
    print(f"  Bond length: {trajectory[0][8]:.2f} → {trajectory[-1][8]:.2f} Å")
    
    # Final structure
    final_structure = trajectory[-1][1:5]  # N, CA, C, O
    
    return trajectory, final_structure


def save_trajectory_pdb(trajectory, sequence, name, output_dir=".", native_CA=None):
    """
    Save trajectory as multi-model PDB for movie.
    
    Args:
        trajectory: List of (step, N, CA, C, O, phi, psi, energy)
        sequence: Amino acid sequence
        name: Structure name
        output_dir: Where to save
        native_CA: Optional native structure for alignment
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    def one_to_three(aa):
        conversion = {
            'A': 'ALA', 'C': 'CYS', 'D': 'ASP', 'E': 'GLU',
            'F': 'PHE', 'G': 'GLY', 'H': 'HIS', 'I': 'ILE',
            'K': 'LYS', 'L': 'LEU', 'M': 'MET', 'N': 'ASN',
            'P': 'PRO', 'Q': 'GLN', 'R': 'ARG', 'S': 'SER',
            'T': 'THR', 'V': 'VAL', 'W': 'TRP', 'Y': 'TYR'
        }
        return conversion.get(aa, 'UNK')
    
    pdb_file = output_path / f"trajectory_{name}.pdb"
    
    # If native provided, align all frames to it
    if native_CA is not None:
        print(f"\n  Aligning {len(trajectory)} frames to native...")
    
    print(f"\n  Writing trajectory PDB...")
    
    with open(pdb_file, 'w') as f:
        f.write(f"REMARK   Refinement trajectory: {name}\n")
        f.write(f"REMARK   Sequence: {sequence} ({len(sequence)} residues)\n")
        f.write(f"REMARK   Frames: {len(trajectory)}\n")
        f.write(f"REMARK   Energy: {trajectory[0][7]:.2f} → {trajectory[-1][7]:.2f}\n")
        f.write(f"REMARK   Peptide bonds: {trajectory[0][8]:.2f} → {trajectory[-1][8]:.2f} Å\n")
        f.write(f"REMARK   \n")
        f.write(f"REMARK   Frame 1: Random starting configuration\n")
        f.write(f"REMARK   Frames 2-{len(trajectory)}: Gradual refinement + bond healing\n\n")
        
        for i, (step, N, CA, C, O, phi, psi, energy, bond_len) in enumerate(trajectory, 1):
            # Align to native if provided
            if native_CA is not None:
                N_aligned, CA_aligned, C_aligned = align_structure_to_reference(
                    N, CA, C, native_CA, native_CA, native_CA
                )
                
                # Also align O atoms
                centroid_CA = np.mean(CA, axis=0)
                centroid_native = np.mean(native_CA, axis=0)
                CA_centered = CA - centroid_CA
                native_centered = native_CA - centroid_native
                H = CA_centered.T @ native_centered
                U, S, Vt = np.linalg.svd(H)
                R = Vt.T @ U.T
                if np.linalg.det(R) < 0:
                    Vt[-1, :] *= -1
                    R = Vt.T @ U.T
                O_aligned = (R @ (O - centroid_CA).T).T + centroid_native
                
                rmsd = calculate_rmsd(CA_aligned, native_CA)
            else:
                N_aligned, CA_aligned, C_aligned, O_aligned = N, CA, C, O
                rmsd = 0.0
            
            f.write(f"MODEL     {i:4d}\n")
            f.write(f"REMARK   Step {step}, Energy: {energy:.2f}, Bond: {bond_len:.2f} Å")
            if native_CA is not None:
                f.write(f", RMSD: {rmsd:.2f} Å")
            f.write("\n")
            
            atom_num = 1
            for j, aa in enumerate(sequence):
                resname = one_to_three(aa)
                resnum = j + 1
                
                for atom_name, coord in [('N', N_aligned[j]), ('CA', CA_aligned[j]), 
                                         ('C', C_aligned[j]), ('O', O_aligned[j])]:
                    f.write(f"ATOM  {atom_num:5d}  {atom_name:3s} {resname:3s} A{resnum:4d}    "
                           f"{coord[0]:8.3f}{coord[1]:8.3f}{coord[2]:8.3f}"
                           f"  1.00  0.00           {atom_name[0]:1s}  \n")
                    atom_num += 1
            f.write("ENDMDL\n")
    
    # Create PyMOL script
    pymol_script = output_path / f"play_trajectory_{name}.pml"
    with open(pymol_script, 'w') as f:
        f.write(f"# PyMOL script to play refinement trajectory\n")
        f.write(f"# Load with: pymol {pymol_script.name}\n\n")
        
        f.write(f"load {pdb_file.name}, trajectory\n\n")
        
        f.write("# Settings\n")
        f.write("bg_color white\n")
        f.write("hide everything\n")
        f.write("show cartoon\n")
        f.write("color marine, trajectory\n")
        f.write("set cartoon_fancy_helices, 1\n")
        f.write("set cartoon_smooth_loops, 1\n\n")
        
        if native_CA is not None:
            f.write("# Native structure (if available)\n")
            f.write("# load native.pdb, native\n")
            f.write("# color red, native\n")
            f.write("# set cartoon_transparency, 0.5, native\n\n")
        
        f.write("# Animation\n")
        f.write("zoom trajectory\n")
        f.write("mset 1 x{}\n".format(len(trajectory)))
        f.write("frame 1\n\n")
        
        f.write("# Play movie\n")
        f.write("# mplay\n\n")
        
        f.write("print ''\n")
        f.write(f"print 'Loaded refinement trajectory: {name}'\n")
        f.write(f"print 'Frames: {len(trajectory)}'\n")
        f.write(f"print 'Energy: {trajectory[0][7]:.2f} → {trajectory[-1][7]:.2f}'\n")
        f.write(f"print 'Peptide bonds: {trajectory[0][8]:.2f} → {trajectory[-1][8]:.2f} Å'\n")
        f.write("print ''\n")
        f.write("print 'Frame 1: Random starting conformation (broken bonds)'\n")
        f.write("print 'Frames 2-N: Gradual refinement + bond healing'\n")
        f.write("print ''\n")
        f.write("print 'Commands:'\n")
        f.write("print '  mplay  - Play movie'\n")
        f.write("print '  mstop  - Stop movie'\n")
        f.write("print '  frame N - Go to frame N'\n")
    
    print(f"  ✓ Saved trajectory: {pdb_file.name}")
    print(f"  ✓ Saved script: {pymol_script.name}")
    print(f"\n  View in PyMOL:")
    print(f"    cd {output_path.absolute()}")
    print(f"    pymol {pymol_script.name}")
    print(f"    mplay  # Play movie")
    
    return pdb_file, pymol_script


if __name__ == "__main__":
    # Example: Generate trajectory for a CDR3 loop
    
    # Load model
    print("Loading model...")
    model, params = load_model()
    
    # Example sequence (or load from CDR3 dataset)
    sequence = "ASSLAPGTSYGKLT"  # 14 residues
    
    # Generate trajectory
    trajectory, final_structure = refine_with_trajectory(
        sequence,
        model,
        params,
        n_steps=50,
        save_every=5,  # Save every 5 steps = 11 frames
        learning_rate=3.0
    )
    
    # Save as movie
    save_trajectory_pdb(
        trajectory,
        sequence,
        name="example_cdr3",
        output_dir="trajectory_movies"
    )
    
    print("\n✓ Done! Load in PyMOL to see refinement animation.")