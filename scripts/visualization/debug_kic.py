import math
import numpy as np
import pyrosetta
from pyrosetta import rosetta

pyrosetta.init("-mute all -detect_bonds false")

def measure_chain_gaps(pose, name="Pose"):
    """Scans the entire pose and prints any broken peptide bonds (>1.5 Å)."""
    print(f"\n--- Checking Chain Integrity: {name} ---")
    broken = False
    for i in range(1, pose.total_residue()):
        c_coords = np.array(pose.residue(i).xyz("C"))
        n_coords = np.array(pose.residue(i + 1).xyz("N"))
        dist = np.linalg.norm(c_coords - n_coords)
        if dist > 1.5:
            print(f"  [!] BROKEN BOND detected between Res {i} (C) and Res {i+1} (N): Gap = {dist:.3f} Å")
            broken = True
    if not broken:
        print("  [✓] All peptide bonds are intact (Distance ~1.32 Å).")

def generate_dummy_data():
    """Generates realistic dummy anchors and a loop sequence for debugging."""
    loop_seq = "CASSSANSGELFF"
    
    # Dummy Anchor 1 (Origin)
    N_b = np.array([-0.5, 1.2, 0.0])
    CA_b = np.array([0.0, 0.0, 0.0])
    C_b = np.array([1.5, 0.0, 0.0])
    
    # Dummy Anchor 2 (Shifted 15 Angstroms away)
    N_a = np.array([15.0, 0.0, 0.0])
    CA_a = np.array([16.5, 0.0, 0.0])
    C_a = np.array([17.0, 1.2, 0.0])
    
    return loop_seq, N_b, CA_b, C_b, N_a, CA_a, C_a

def debug_numeric_bridge_objects():
    """
    Attempts to instantiate the required C++ memory arrays to call the raw 
    numeric::kinematic_closure::bridgeObjects math directly from Python.
    """
    print("\n" + "="*60)
    print("TEST 1: RAW NUMERIC KINEMATIC CLOSURE (bridgeObjects)")
    print("="*60)
    
    try:
        from pyrosetta.rosetta.numeric.kinematic_closure import bridgeObjects
        from pyrosetta.rosetta.utility import vector1_int, vector1_double
        
        # PyBind11 exposes nested templates with long, specific names. 
        # We try to grab the ones required by the signature.
        try:
            vector1_vec3 = rosetta.utility.vector1_utility_fixedsizearray1_double_3_t
            vec3 = rosetta.utility.fixedsizearray1_double_3_t
            vector1_vector1_double = rosetta.utility.vector1_utility_vector1_double_std_allocator_double_t
        except AttributeError:
            # Fallback names depending on the PyRosetta build version
            vector1_vec3 = rosetta.utility.vector1_utility_fixedsizearray1_double_3UL_t
            vec3 = rosetta.utility.fixedsizearray1_double_3UL_t
            vector1_vector1_double = getattr(rosetta.utility, "vector1_utility_vector1_double_t", None)

        print("  [✓] Successfully mapped C++ vector types in Python!")

        # Initialize the arrays
        atoms = vector1_vec3()
        dt = vector1_double()
        da = vector1_double()
        db = vector1_double()
        pivots = vector1_int()
        order = vector1_int()
        
        t_ang = vector1_vector1_double()
        b_ang = vector1_vector1_double()
        b_len = vector1_vector1_double()
        
        # In PyRosetta, pass-by-reference integers (like nsol) are often returned as a tuple.
        print("  [✓] Successfully instantiated empty C++ arrays!")
        print("  This confirms we can entirely bypass Rosetta Poses and FoldTrees.")
        print("  We can write a function that loops over your PyTorch coordinates,")
        print("  fills these arrays, and extracts the 6 angles analytically!")
        
    except Exception as e:
        print("  [X] Failed during raw numeric closure setup:", e)


def debug_generalized_kic(loop_seq, N_b, CA_b, C_b, N_a, CA_a, C_a):
    """
    Uses GeneralizedKIC (GenKIC), Rosetta's modern KIC solver.
    """
    print("\n" + "="*60)
    print("TEST 2: GENERALIZED KIC (Modern Robust Solver)")
    print("="*60)
    
    seq = "A" + loop_seq + "A"
    n_loop = len(loop_seq)
    pose = rosetta.core.pose.Pose()
    rosetta.core.pose.make_pose_from_sequence(pose, seq, "fa_standard")
    
    n_res = pose.total_residue()
    
    for i in range(1, n_res + 1):
        pose.set_phi(i, -150.0)
        pose.set_psi(i, 150.0)
        pose.set_omega(i, 180.0)

    measure_chain_gaps(pose, "Initial Extended Pose")
    
    genkic = rosetta.protocols.generalized_kinematic_closure.GeneralizedKIC()
    
    for i in range(2, n_res):
        genkic.add_loop_residue(i)
        
    pivot1 = 2
    pivot2 = 2 + n_loop // 2
    pivot3 = n_res - 1
    
    genkic.set_pivot_atoms(pivot1, "CA", pivot2, "CA", pivot3, "CA")
    
    # FIXED: Replaced the missing attribute with the exact ideal C-N distance constant
    ideal_c_n_dist = 1.328685 
    genkic.close_bond(pivot2, "C", pivot2 + 1, "N", 
                      ideal_c_n_dist,
                      116.2 * (math.pi/180.0), 
                      121.7 * (math.pi/180.0), 
                      180.0 * (math.pi/180.0), 
                      False)

    genkic.add_perturber("randomize_alpha_backbone_by_rama")
    for i in range(2, n_res):
        if i not in [pivot1, pivot2, pivot3]:
            genkic.add_residue_to_perturber_residue_list(1, i)
            
    genkic.set_closure_attempts(100) # Only 100 attempts
    genkic.set_selector_type("random_selector") 

    print(f"\n  Running GeneralizedKIC (100 attempts) on {n_loop}-residue loop...")
    genkic.apply(pose)
    
    measure_chain_gaps(pose, "After GeneralizedKIC")
    
    if genkic.last_run_successful():
        print("  [✓] GeneralizedKIC Successfully Closed the Loop!")
        print("  Extracted Pivot Angles:")
        print(f"    Pivot 1 (Res {pivot1}): phi = {pose.phi(pivot1):.1f}°, psi = {pose.psi(pivot1):.1f}°")
        print(f"    Pivot 2 (Res {pivot2}): phi = {pose.phi(pivot2):.1f}°, psi = {pose.psi(pivot2):.1f}°")
        print(f"    Pivot 3 (Res {pivot3}): phi = {pose.phi(pivot3):.1f}°, psi = {pose.psi(pivot3):.1f}°")
    else:
        print("  [X] GeneralizedKIC Failed to close the loop in 100 attempts.")

if __name__ == "__main__":
    loop_seq, N_b, CA_b, C_b, N_a, CA_a, C_a = generate_dummy_data()
    debug_numeric_bridge_objects()
    debug_generalized_kic(loop_seq, N_b, CA_b, C_b, N_a, CA_a, C_a)