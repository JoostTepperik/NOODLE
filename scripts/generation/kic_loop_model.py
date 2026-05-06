"""
kic_loop_modeling.py

Optimized for High-Throughput CPU Cluster generation (Snellius).
Features: Multiprocessing, Pre-computed Arrays, Pure Numpy IPC,
Anchor-Aware Intra Clashes, Linear Translation Morphing.
"""

import math
import multiprocessing
import numpy as np
import torch
from typing import Optional, Tuple, List

from loop_modeling_nerf import (
    cache_energy_distributions, compute_energy,
    compute_O_atoms, VDW_RADII, _sample_from_joint
)
from utils import _to_router

def suppress_alpha_helices(probs_joint: list, penalty: float = 0.01) -> list:
    """
    Dramatically reduces the probability of sampling alpha-helical angles
    to force elongated (beta-sheet-like) CDR3 loops.
    """
    biased_probs = []
    for p in probs_joint:
        p_mod = p.copy()
        
        # Alpha-helix Ramachandran region:
        # phi roughly -100 to -40 (bins 8 to 14)
        # psi roughly -70 to -10  (bins 11 to 17)
        for phi_bin in range(8, 15):
            for psi_bin in range(11, 18):
                p_mod[phi_bin, psi_bin] *= penalty
                
        # Re-normalize so it sums to 1.0
        p_mod /= p_mod.sum()
        biased_probs.append(p_mod)
        
    return biased_probs

def _worker_init():
    """Initializes PyRosetta and unique random seeds inside each CPU worker."""
    import pyrosetta
    pyrosetta.init("-mute all -detect_bonds false", silent=True)
    
    # Force this specific CPU core to get a completely unique random seed
    import os
    import random
    # Generate a truly random seed from the OS entropy pool
    seed = int.from_bytes(os.urandom(4), byteorder='little')
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


def _worker_chunk(args):
    """The tight execution loop running on a single CPU core."""
    (chunk_size, loop_seq, N_b, CA_b, C_b, N_a, CA_a, C_a,
     probs_joint_biased, probs_joint_raw, filter_order,
     max_init_energy, max_init_intra, max_init_clash,
     k_clash, clash_weight, framework_grid) = args

    import pyrosetta
    from pyrosetta import rosetta

    n_loop = len(loop_seq)
    n_before = len(N_b)
    n_after = len(N_a)
    n_full = n_before + n_loop + n_after
    
    # 1. Setup Pose & GenKIC
    seq = "A" + loop_seq + "A"
    pose = rosetta.core.pose.Pose()
    rosetta.core.pose.make_pose_from_sequence(pose, seq, "fa_standard")
    n_res = pose.total_residue()
    
    import random
    # Pick 3 unique random residues in the loop to act as GenKIC pivots
    p1_idx, p2_idx, p3_idx = sorted(random.sample(range(n_loop), 3))
    
    pivot1 = 2 + p1_idx
    pivot2 = 2 + p2_idx
    pivot3 = 2 + p3_idx
    pivot_indices = [p1_idx, p2_idx, p3_idx]
    
    # The fold tree cut MUST happen at pivot2
    ft = rosetta.core.kinematics.FoldTree()
    ft.add_edge(1, n_res, 1)  
    ft.add_edge(1, pivot2, -1)
    ft.add_edge(n_res, pivot2 + 1, -1)
    pose.fold_tree(ft)
    # ------------------------------------
    
    stub1 = rosetta.core.kinematics.Stub(
        rosetta.numeric.xyzVector_double_t(*CA_b[-1]),
        rosetta.numeric.xyzVector_double_t(*N_b[-1]),
        rosetta.numeric.xyzVector_double_t(*C_b[-1])
    )
    stub1 = rosetta.core.kinematics.Stub(
        rosetta.numeric.xyzVector_double_t(*CA_b[-1]),
        rosetta.numeric.xyzVector_double_t(*N_b[-1]),
        rosetta.numeric.xyzVector_double_t(*C_b[-1])
    )
    stub2 = rosetta.core.kinematics.Stub(
        rosetta.numeric.xyzVector_double_t(*CA_a[0]),
        rosetta.numeric.xyzVector_double_t(*N_a[0]),
        rosetta.numeric.xyzVector_double_t(*C_a[0])
    )
    jump = rosetta.core.kinematics.Jump()
    jump.from_stubs(stub1, stub2)
    pose.set_jump(1, jump)
    
    for i in range(1, n_res + 1):
        pose.set_phi(i, -150.0)
        pose.set_psi(i, 150.0)
        pose.set_omega(i, 180.0)

    pivot1, pivot3 = 2, n_res - 1
    p1_idx, p2_idx, p3_idx = 0, n_loop // 2 - 1, n_loop - 1
    pivot_indices = [p1_idx, p2_idx, p3_idx]
    
    genkic = rosetta.protocols.generalized_kinematic_closure.GeneralizedKIC()
    for i in range(2, n_res): genkic.add_loop_residue(i)
    genkic.set_pivot_atoms(pivot1, "CA", pivot2, "CA", pivot3, "CA")
    genkic.close_bond(
        pivot2, "C", pivot2 + 1, "N",       
        pivot2, "CA", pivot2 + 1, "CA",     
        1.328685, 116.2 * (math.pi / 180.0), 121.7 * (math.pi / 180.0), math.pi, False, False                               
    )
    genkic.set_closure_attempts(1) 
    genkic.set_selector_type("random_selector")

    # 2. Pre-compute constants for fast math
    pyt_anc1 = np.stack([N_b[-1], CA_b[-1], C_b[-1]])
    mean_pyt1 = pyt_anc1.mean(axis=0)
    
    # Radii for the entire sequence (flanks + loop)
    radii_full = np.tile([VDW_RADII['N'], VDW_RADII['CA'], VDW_RADII['C']], n_full).astype(np.float32)
    vdw_sum = (radii_full[:, None] + radii_full[None, :]) * 0.8
    loop_radii = radii_full[n_before * 3 : (n_before + n_loop) * 3]
    
    res_indices = np.arange(n_full * 3) // 3
    valid_pairs = np.abs(res_indices[:, None] - res_indices[None, :]) > 1
    
    # Identify loop atoms so we don't penalize native flank-vs-flank clashes
    is_loop = np.zeros(n_full * 3, dtype=bool)
    is_loop[n_before * 3 : (n_before + n_loop) * 3] = True
    loop_in_pair = is_loop[:, None] | is_loop[None, :]
    intra_mask = valid_pairs & np.triu(np.ones((n_full * 3, n_full * 3), dtype=bool), k=1) & loop_in_pair
    
    filters = [f.strip() for f in filter_order.split(',')]
    weights = np.linspace(1, n_loop, n_loop)[:, np.newaxis] / (n_loop + 1)
    
    local_pool = []
    stats = {'kic_failed': 0, 'intra_clash_failed': 0, 'energy_failed': 0, 'fw_clash_failed': 0, 'accepted': 0}

    # 3. Tight Execution Loop
    for _ in range(chunk_size):
        phis_rad, psis_rad = zip(*[_sample_from_joint(probs_joint_biased[i]) for i in range(n_loop)])
        
        anchor_psis = np.linspace(-math.pi, math.pi, 10)
        np.random.shuffle(anchor_psis)
        
        for anchor_psi in anchor_psis:
            work_pose = pose.clone()
            work_pose.set_psi(1, math.degrees(anchor_psi))
            
            for i in range(n_loop):
                res_idx = 2 + i
                if res_idx not in (pivot1, pivot2, pivot3):
                    work_pose.set_phi(res_idx, math.degrees(phis_rad[i]))
                    work_pose.set_psi(res_idx, math.degrees(psis_rad[i]))
                    
            try:
                genkic.apply(work_pose)
            except Exception:
                stats['kic_failed'] += 1; continue
                
            if not genkic.last_run_successful():
                stats['kic_failed'] += 1; continue

            # Morph Alignment
            ros_anc1 = np.array([work_pose.residue(1).xyz("N"), work_pose.residue(1).xyz("CA"), work_pose.residue(1).xyz("C")])
            mean_ros1 = ros_anc1.mean(axis=0)
            U1, _, Vt1 = np.linalg.svd((ros_anc1 - mean_ros1).T @ (pyt_anc1 - mean_pyt1))
            R1 = Vt1.T @ np.diag([1.0, 1.0, np.linalg.det(Vt1.T @ U1.T)]) @ U1.T
            
            N_coords = np.array([work_pose.residue(i).xyz("N") for i in range(2, 2 + n_loop)])
            CA_coords = np.array([work_pose.residue(i).xyz("CA") for i in range(2, 2 + n_loop)])
            C_coords = np.array([work_pose.residue(i).xyz("C") for i in range(2, 2 + n_loop)])
            
            N_aligned  = (N_coords - mean_ros1) @ R1.T + mean_pyt1
            CA_aligned = (CA_coords - mean_ros1) @ R1.T + mean_pyt1
            C_aligned  = (C_coords - mean_ros1) @ R1.T + mean_pyt1
            
            ros_anc2_N = np.array(work_pose.residue(n_res).xyz("N"))
            aligned_anc2_N = (ros_anc2_N - mean_ros1) @ R1.T + mean_pyt1
            delta = N_a[0] - aligned_anc2_N
            
            N_np  = N_aligned + weights * delta
            CA_np = CA_aligned + weights * delta
            C_np  = C_aligned + weights * delta
            O_np = compute_O_atoms(N_np, CA_np, C_np)
            
            # Stack the full backbone (before + loop + after)
            N_full  = np.vstack([N_b, N_np, N_a])
            CA_full = np.vstack([CA_b, CA_np, CA_a])
            C_full  = np.vstack([C_b, C_np, C_a])
            
            full_atoms = np.empty((n_full * 3, 3), dtype=np.float32)
            full_atoms[0::3] = N_full
            full_atoms[1::3] = CA_full
            full_atoms[2::3] = C_full
            
            loop_atoms = full_atoms[n_before * 3 : (n_before + n_loop) * 3]

            # Dynamic Filtering
            phi_out = [work_pose.phi(i) for i in range(2, 2 + n_loop)]
            psi_out = [work_pose.psi(i) for i in range(1, 2 + n_loop)] 
            
            intra_clash_score = 0.0; fw_clash = 0.0; pivot_energy = 0.0
            computed_intra = False; computed_energy = False; computed_fw = False
            failed = False
            
            for f_name in filters:
                if f_name == 'energy' and not computed_energy:
                    for p_idx in pivot_indices:
                        p_phi_t = torch.tensor([[math.radians(phi_out[p_idx])]], dtype=torch.float32)
                        p_psi_t = torch.tensor([[0.0, math.radians(psi_out[p_idx+1])]], dtype=torch.float32)
                        e_p = compute_energy(p_phi_t, p_psi_t, [probs_joint_raw[p_idx]])
                        pivot_energy += float(e_p.item())
                    computed_energy = True
                    if max_init_energy is not None and pivot_energy > max_init_energy:
                        stats['energy_failed'] += 1
                        failed = True; break

                elif f_name == 'intra' and not computed_intra:
                    dist_mat = np.linalg.norm(full_atoms[:, None, :] - full_atoms[None, :, :], axis=-1)
                    overlaps = np.clip(vdw_sum - dist_mat, 0, None)[intra_mask]
                    intra_clash_score = float(k_clash * np.sum(overlaps))
                    computed_intra = True
                    if max_init_intra is not None and intra_clash_score > max_init_intra:
                        stats['intra_clash_failed'] += 1
                        failed = True; break

                elif f_name == 'fw' and not computed_fw:
                    if framework_grid is not None:
                        fw_clash = framework_grid.query_score_np(loop_atoms, loop_radii, k_clash=k_clash)
                    computed_fw = True
                    if max_init_clash is not None and fw_clash > max_init_clash:
                        stats['fw_clash_failed'] += 1
                        failed = True; break

            if failed: continue

            # Compute missing metrics for combined score
            if not computed_intra:
                dist_mat = np.linalg.norm(full_atoms[:, None, :] - full_atoms[None, :, :], axis=-1)
                overlaps = np.clip(vdw_sum - dist_mat, 0, None)[intra_mask]
                intra_clash_score = float(k_clash * np.sum(overlaps))
            if not computed_fw and framework_grid is not None:
                fw_clash = framework_grid.query_score_np(loop_atoms, loop_radii, k_clash=k_clash)

            phi_t = torch.tensor(np.deg2rad(phi_out), dtype=torch.float32).unsqueeze(0)
            psi_t = torch.tensor(np.deg2rad(psi_out), dtype=torch.float32).unsqueeze(0)
            total_energy = float(compute_energy(phi_t, psi_t, probs_joint_raw).item())

            combined_score = total_energy + clash_weight * (fw_clash + intra_clash_score)
            
            structure = (
                N_full,
                CA_full,
                C_full,
                O_np,  # Master process will attach flanks
                np.array(phi_out),
                np.array(psi_out[1:]), 
                total_energy,    
                0.0     
            )
            local_pool.append((combined_score, structure))
            stats['accepted'] += 1
            break 
            
    return local_pool, stats


def apply_temperature(probs: np.ndarray, temperature: float) -> np.ndarray:
    if temperature == 1.0: return probs
    p = np.clip(probs, 1e-10, None)
    p_biased = p ** (1.0 / temperature)
    return p_biased / p_biased.sum()


def refine_loop_3d_frames(
    full_sequence:    str,
    loop_start:       int,
    loop_end:         int,
    N_flank_before:   np.ndarray,
    CA_flank_before:  np.ndarray,
    C_flank_before:   np.ndarray,
    O_flank_before:   np.ndarray,
    N_flank_after:    np.ndarray,
    CA_flank_after:   np.ndarray,
    C_flank_after:    np.ndarray,
    O_flank_after:    np.ndarray,
    model_or_router,
    params=None,
    n_structures:     int   = 10,
    seed:             int   = None,
    framework_coords: Optional[np.ndarray] = None,
    framework_radii:  Optional[np.ndarray] = None,
    k_clash:          float = 1.0,
    clash_weight:     float = 5.0,
    max_init_clash:   float = 2.0,
    max_init_intra:   float = 2.0,
    max_init_energy:  float = 30.0,
    framework_grid    = None,
    n_samples:        int   = 1000000, 
    temperature:      float = 0.8,
    filter_order:     str   = 'energy,intra,fw',
    n_cpus:           int   = 24
) -> Tuple[list, list, list]:
    
    router = _to_router(model_or_router, params)
    loop_seq = full_sequence[loop_start:loop_end]
    n_loop = len(loop_seq)
    
       # Inside refine_loop_3d_frames:
    probs_joint_raw = cache_energy_distributions(router, loop_seq)
    
    # 1. Kill the alpha helices to promote extended structures!
    probs_joint_extended = suppress_alpha_helices(probs_joint_raw, penalty=0.25)
    
    # 2. Apply temperature
    probs_joint_biased = [apply_temperature(pj, temperature) for pj in probs_joint_extended]

    print(f"\n  Distributed KIC Refinement: {full_sequence}")
    print(f"    Loop: {loop_seq} ({n_loop} res) | Cores: {n_cpus}")
    print(f"    Targeting {n_samples} total attempts...")

    # Calculate chunks
    chunks = []
    attempts_per_chunk = min(1000, math.ceil(n_samples / (n_cpus * 4))) 

    remaining = n_samples

    while remaining > 0:
        c_size = min(remaining, attempts_per_chunk)
        chunks.append((
            c_size, loop_seq, N_flank_before, CA_flank_before, C_flank_before,
            N_flank_after, CA_flank_after, C_flank_after,
            probs_joint_biased, probs_joint_raw, filter_order,
            max_init_energy, max_init_intra, max_init_clash,
            k_clash, clash_weight, framework_grid
        ))
        remaining -= c_size

    # Launch multiprocessing pool
    global_pool = []
    global_stats = {'kic_failed': 0, 'intra_clash_failed': 0, 'energy_failed': 0, 'fw_clash_failed': 0, 'accepted': 0}
    
    with multiprocessing.Pool(processes=n_cpus, initializer=_worker_init) as pool:
        for local_pool, stats in pool.imap_unordered(_worker_chunk, chunks):
            
            # Reconstruct the structures with O flanks
            for score, struct in local_pool:
                complete_struct = (
                    struct[0], struct[1], struct[2], 
                    np.vstack([O_flank_before, struct[3], O_flank_after]), 
                    struct[4], struct[5], struct[6], struct[7]
                )
                global_pool.append((score, complete_struct))
                
            for k in stats: global_stats[k] += stats[k]

    print(f"    --- Global Sampling Stats ---")
    print(f"    Attempts:    {n_samples}")
    print(f"    KIC Failed:  {global_stats['kic_failed']}")
    for f in [x.strip() for x in filter_order.split(',')]:
        if f == 'energy': print(f"    Pivot Energy:{global_stats['energy_failed']} rejected (> {max_init_energy})")
        if f == 'intra':  print(f"    Intra Clash: {global_stats['intra_clash_failed']} rejected (> {max_init_intra})")
        if f == 'fw':     print(f"    FW Clash:    {global_stats['fw_clash_failed']} rejected (> {max_init_clash})")
    print(f"    Accepted:    {global_stats['accepted']} valid structures pooled")
    print(f"    -----------------------------")
    
    global_pool.sort(key=lambda x: x[0])
    ensemble = [item[1] for item in global_pool]
    
    return ensemble[:n_structures], probs_joint_raw, []