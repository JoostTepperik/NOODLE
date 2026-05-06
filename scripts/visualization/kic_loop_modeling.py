"""
kic_loop_modeling.py

Pure KIC-based loop modeling — analytical closure with learned energy sampling.
Uses GeneralizedKIC (GenKIC) to analytically close the loop. 
Includes Coordinate Morphing (Linear Translation) for perfect closure without squishing, 
temperature-biased sampling, and dynamic cascaded filtering.
"""

import math
import numpy as np
import torch
from typing import Optional, Tuple, List

import pyrosetta
pyrosetta.init("-mute all -detect_bonds false")
from pyrosetta import rosetta

from loop_modeling_nerf import (
    cache_energy_distributions, compute_energy,
    compute_O_atoms, VDW_RADII, _sample_from_joint
)
from utils import _to_router


def _setup_genkic_pose(
    loop_seq: str, 
    N_b: np.ndarray, CA_b: np.ndarray, C_b: np.ndarray,
    N_a: np.ndarray, CA_a: np.ndarray, C_a: np.ndarray
) -> rosetta.core.pose.Pose:
    seq = "A" + loop_seq + "A"
    pose = rosetta.core.pose.Pose()
    rosetta.core.pose.make_pose_from_sequence(pose, seq, "fa_standard")
    n_res = pose.total_residue()
    n_loop = len(loop_seq)
    
    ft = rosetta.core.kinematics.FoldTree()
    ft.add_edge(1, n_res, 1)  
    pivot2 = 1 + n_loop // 2
    ft.add_edge(1, pivot2, -1)
    ft.add_edge(n_res, pivot2 + 1, -1)
    pose.fold_tree(ft)
    
    stub1 = rosetta.core.kinematics.Stub(
        rosetta.numeric.xyzVector_double_t(*CA_b),
        rosetta.numeric.xyzVector_double_t(*N_b),
        rosetta.numeric.xyzVector_double_t(*C_b)
    )
    stub2 = rosetta.core.kinematics.Stub(
        rosetta.numeric.xyzVector_double_t(*CA_a),
        rosetta.numeric.xyzVector_double_t(*N_a),
        rosetta.numeric.xyzVector_double_t(*C_a)
    )
    
    jump = rosetta.core.kinematics.Jump()
    jump.from_stubs(stub1, stub2)
    pose.set_jump(1, jump)
    
    for i in range(1, n_res + 1):
        pose.set_phi(i, -150.0)
        pose.set_psi(i, 150.0)
        pose.set_omega(i, 180.0)
        
    return pose


def apply_temperature(probs: np.ndarray, temperature: float) -> np.ndarray:
    if temperature == 1.0:
        return probs
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
    n_steps:          int   = 1000,
    lr_energy:        float = 0.05,
    lr_closure:       float = 0.20,
    closure_weight:   float = 50.0,
    n_structures:     int   = 10,
    seed:             int   = None,
    eta_min:          float = 1e-4,
    n_frames:         int   = 0,
    framework_coords: Optional[np.ndarray] = None,
    framework_radii:  Optional[np.ndarray] = None,
    k_clash:          float = 100.0,
    clash_weight:     float = 1.0,
    clash_cutoff:     float = 8.0,
    clash_start_frac: float = 0.25,
    n_pulses:         int   = 3,
    clash_floor_frac: float = 0.02,
    clash_cap:        float = 1.5,
    clash_buffer:     float = 0.5,
    grid_resolution:  float = 0.5,
    max_init_clash:   float = 2.0,
    max_init_closure: float = None,
    max_init_intra:   float = 2.0,
    max_init_energy:  float = 30.0,
    max_init_attempts: int  = 500,
    framework_grid    = None,
    n_samples:         int   = 1000, 
    n_refine_steps:    int   = 50,
    temperature:       float = 0.8,
    filter_order:      str   = 'energy,intra,fw',
) -> Tuple[list, list, list]:
    
    router = _to_router(model_or_router, params)
    loop_seq = full_sequence[loop_start:loop_end]
    n_loop = len(loop_seq)
    
    probs_joint_raw = cache_energy_distributions(router, loop_seq)
    probs_joint_biased = [apply_temperature(pj, temperature) for pj in probs_joint_raw]

    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        
    print(f"\n  Pure GenKIC Refinement: {full_sequence}")
    print(f"    Loop: {loop_seq} ({n_loop} res) | Temp: {temperature} | Max Pivot E: {max_init_energy}")

    pose = _setup_genkic_pose(
        loop_seq, 
        N_flank_before[-1], CA_flank_before[-1], C_flank_before[-1],
        N_flank_after[0], CA_flank_after[0], C_flank_after[0]
    )
    
    n_res = pose.total_residue()
    pivot1, pivot2, pivot3 = 2, 1 + n_loop // 2, n_res - 1
    
    # Map Rosetta 1-indexed (with flanks) to PyTorch 0-indexed loop positions
    p1_idx = 0
    p2_idx = n_loop // 2 - 1
    p3_idx = n_loop - 1
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

    pool = []
    max_pool_size = max(500, n_samples // 2)
    
    stats = {
        'kic_failed': 0,
        'intra_clash_failed': 0,
        'energy_failed': 0,
        'fw_clash_failed': 0,
        'accepted': 0
    }
    
    print(f"    Sampling max {n_samples} attempts (Pool target: {max_pool_size})...")
    print(f"    Filter order: {filter_order}")
    
    pyt_anc1 = np.stack([N_flank_before[-1], CA_flank_before[-1], C_flank_before[-1]])
    mean_pyt1 = pyt_anc1.mean(axis=0)
    
    # Pre-compute intra-clash arrays to save time inside the loop
    radii = np.tile([VDW_RADII['N'], VDW_RADII['CA'], VDW_RADII['C']], n_loop).astype(np.float32)
    vdw_sum = (radii[:, None] + radii[None, :]) * 0.8
    res_indices = np.arange(n_loop * 3) // 3
    valid_pairs = np.abs(res_indices[:, None] - res_indices[None, :]) > 1
    intra_mask = valid_pairs & np.triu(np.ones((n_loop * 3, n_loop * 3), dtype=bool), k=1)
    
    filters = [f.strip() for f in filter_order.split(',')]

    for attempt in range(n_samples):
        if len(pool) >= max_pool_size:
            break
            
        phis_rad, psis_rad = zip(*[_sample_from_joint(probs_joint_biased[i]) for i in range(n_loop)])
        
        for anchor_psi in np.linspace(-math.pi, math.pi, 10):
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
                stats['kic_failed'] += 1
                continue
                
            if not genkic.last_run_successful():
                stats['kic_failed'] += 1
                continue

            # --- 1. MORPH ALIGNMENT (Translational Shift) ---
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
            delta = N_flank_after[0] - aligned_anc2_N
            weights = np.linspace(1, n_loop, n_loop)[:, np.newaxis] / (n_loop + 1)
            
            N_np  = N_aligned + weights * delta
            CA_np = CA_aligned + weights * delta
            C_np  = C_aligned + weights * delta
            O_np = compute_O_atoms(N_np, CA_np, C_np)
            
            loop_atoms = np.empty((n_loop * 3, 3), dtype=np.float32)
            loop_atoms[0::3] = N_np
            loop_atoms[1::3] = CA_np
            loop_atoms[2::3] = C_np

            # --- DYNAMIC FILTERING ---
            phi_out = [work_pose.phi(i) for i in range(2, 2 + n_loop)]
            psi_out = [work_pose.psi(i) for i in range(1, 2 + n_loop)] 
            
            intra_clash_score = 0.0
            pivot_energy = 0.0
            fw_clash = 0.0
            
            computed_intra = False
            computed_energy = False
            computed_fw = False
            failed = False
            
            for f_name in filters:
                # 1. Pivot Energy Filter
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

                # 2. Intra Clash Filter
                elif f_name == 'intra' and not computed_intra:
                    dist_mat = np.linalg.norm(loop_atoms[:, None, :] - loop_atoms[None, :, :], axis=-1)
                    overlaps = np.clip(vdw_sum - dist_mat, 0, None)[intra_mask]
                    intra_clash_score = float(k_clash * np.sum(overlaps))
                    computed_intra = True
                    if max_init_intra is not None and intra_clash_score > max_init_intra:
                        stats['intra_clash_failed'] += 1
                        failed = True; break

                # 3. Framework Clash Filter
                elif f_name == 'fw' and not computed_fw:
                    if framework_grid is not None:
                        fw_clash = framework_grid.query_score_np(loop_atoms, radii, k_clash=k_clash)
                    computed_fw = True
                    if max_init_clash is not None and fw_clash > max_init_clash:
                        stats['fw_clash_failed'] += 1
                        failed = True; break

            if failed:
                continue

            # Compute any remaining metrics needed for the final combined_score sorting
            if not computed_intra:
                dist_mat = np.linalg.norm(loop_atoms[:, None, :] - loop_atoms[None, :, :], axis=-1)
                overlaps = np.clip(vdw_sum - dist_mat, 0, None)[intra_mask]
                intra_clash_score = float(k_clash * np.sum(overlaps))
            
            if not computed_fw and framework_grid is not None:
                fw_clash = framework_grid.query_score_np(loop_atoms, radii, k_clash=k_clash)

            # Calculate total loop energy for the final output
            phi_t = torch.tensor(np.deg2rad(phi_out), dtype=torch.float32).unsqueeze(0)
            psi_t = torch.tensor(np.deg2rad(psi_out), dtype=torch.float32).unsqueeze(0)
            total_energy = float(compute_energy(phi_t, psi_t, probs_joint_raw).item())

            closure_dist = 0.0
            combined_score = total_energy + clash_weight * (fw_clash + intra_clash_score)
            
            structure = (
                np.vstack([N_flank_before, N_np, N_flank_after]),
                np.vstack([CA_flank_before, CA_np, CA_flank_after]),
                np.vstack([C_flank_before, C_np, C_flank_after]),
                np.vstack([O_flank_before, O_np, O_flank_after]),
                np.array(phi_out),
                np.array(psi_out[1:]), 
                total_energy,    
                closure_dist     
            )
            pool.append((combined_score, structure))
            stats['accepted'] += 1
            break 

    print(f"    --- Sampling Stats ---")
    print(f"    Attempts:    {attempt + 1}")
    print(f"    KIC Failed:  {stats['kic_failed']}")
    for f in filters:
        if f == 'energy': print(f"    Pivot Energy:{stats['energy_failed']} rejected (> {max_init_energy})")
        if f == 'intra':  print(f"    Intra Clash: {stats['intra_clash_failed']} rejected (> {max_init_intra})")
        if f == 'fw':     print(f"    FW Clash:    {stats['fw_clash_failed']} rejected (> {max_init_clash})")
    print(f"    Accepted:    {stats['accepted']} added to pool")
    print(f"    ----------------------")
    
    pool.sort(key=lambda x: x[0])
    ensemble = [item[1] for item in pool]
    
    return ensemble[:n_structures], probs_joint_raw, []