"""
test_on_cdr3.py

High-Throughput Hybrid Pipeline:
Stage 1: GenKIC (Lax filters + Alpha Helix Suppressor for extended Beta topologies)
Stage 2: Annealed Langevin Dynamics (Diffusing out of clashes while preserving KIC closure)
"""

import argparse
import json
from pathlib import Path
import numpy as np
import torch

from utils import (
    ModelRouter, load_cdr3_native, load_model,
    compute_loop_rmsds, save_pdbs, VDW_RADII
)

from kic_loop_model import refine_loop_3d_frames
from loop_modeling_nerf import (
    ideal_structure_pdb, aligned_loop_rmsd,
    cache_energy_distributions, compute_energy,
    build_framework_grid, sample_langevin_torsions, _pack_ensemble,
    _build_boundary_atoms
)

def _dihedral(p1, p2, p3, p4):
    """Calculate dihedral angle between 4 points in degrees."""
    b0 = p1 - p2;  b1 = p3 - p2;  b2 = p4 - p3
    b1h = b1 / (np.linalg.norm(b1) + 1e-10)
    v   = b0 - np.dot(b0, b1h) * b1h
    w   = b2 - np.dot(b2, b1h) * b1h
    return np.degrees(np.arctan2(np.dot(np.cross(b1h, v), w), np.dot(v, w)))

def coords_to_angles(N, CA, C):
    n   = len(N)
    phi = np.full(n, np.nan); psi = np.full(n, np.nan)
    for i in range(n):
        if i > 0:   phi[i] = _dihedral(C[i-1], N[i], CA[i], C[i])
        if i < n-1: psi[i] = _dihedral(N[i], CA[i], C[i], N[i+1])
    return phi, psi

def suppress_alpha_helices(probs_joint: list, penalty: float = 0.01) -> list:
    biased_probs = []
    for p in probs_joint:
        p_mod = p.copy()
        for phi_bin in range(8, 15):
            for psi_bin in range(11, 18):
                p_mod[phi_bin, psi_bin] *= penalty
        p_mod /= p_mod.sum()
        biased_probs.append(p_mod)
    return biased_probs


def test_on_cdr3_dataset(
    dataset_dir:      str,
    output_dir:       str,
    router:           ModelRouter,
    n_structures:     int   = 10,
    max_loops:        int   = None,
    complex_pdb_dir:  str   = "/home/jtepperik/thesis/data/reference_final",
    use_clash:        bool  = False,
    k_clash:          float = 100.0,
    clash_weight:     float = 1.0,
    clash_cutoff:     float = 8.0,
    clash_buffer:     float = 0.5,
    grid_resolution:  float = 0.5,
    max_init_clash:   float = 50.0,  # Lax for KIC
    max_init_intra:   float = 50.0,  # Lax for KIC
    max_init_energy:  float = 50.0,  # Lax for KIC
    n_samples:        int   = 1000000,
    temperature:      float = 0.8,
    filter_order:     str   = 'energy,intra,fw',
    n_cpus:           int   = 24
) -> list:

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    metadata_file = Path(dataset_dir) / 'cdr3_dataset.json'
    with open(metadata_file) as f:
        dataset = json.load(f)
    if max_loops: dataset = dataset[:max_loops]

    print(f"\n{'='*60}")
    print(f"HYBRID PIPELINE: GenKIC -> Langevin Dynamics")
    print(f"Filters: KIC_clash={max_init_clash} AlphaSuppressor=ON")
    print(f"{'='*60}")

    results = []

    for i, meta in enumerate(dataset, 1):
        pdb_id     = meta['pdb_id']
        chain      = meta['chain']
        full_seq   = meta['full_sequence']
        cdr3_seq   = meta['cdr3_sequence']
        loop_start = meta['loop_start']
        loop_end   = meta['loop_end']
        name       = f"{pdb_id}_{chain}"
        loop_out   = out / name
        loop_out.mkdir(parents=True, exist_ok=True)

        print(f"\n{i}/{len(dataset)}  {pdb_id} chain {chain}")

        pdb_file = Path(meta['pdb_file'])
        if not pdb_file.is_absolute(): pdb_file = Path(dataset_dir) / pdb_file.name
        
        seq, N_nat, CA_nat, C_nat, O_nat = load_cdr3_native(pdb_file)
        start = max(0, loop_start - 1); end = min(len(seq), loop_end + 1)
        ph, ps = coords_to_angles(N_nat[start:end], CA_nat[start:end], C_nat[start:end])
        
        offset = loop_start - start; n_loop = loop_end - loop_start
        CA_loop_nat = CA_nat[loop_start:loop_end]

        probs_joint = cache_energy_distributions(router, cdr3_seq)
        probs_joint = suppress_alpha_helices(probs_joint, penalty=0.01)

        fw_coords = fw_radii = fw_grid = loop_radii = None
        if use_clash:
            from utils import extract_framework_atoms
            complex_pdb = Path(complex_pdb_dir) / f"{pdb_id}.pdb"
            if complex_pdb.exists():
                fw_coords, fw_radii = extract_framework_atoms(
                    str(complex_pdb), tcr_chain=chain, full_sequence=full_seq,
                    loop_start=loop_start, loop_end=loop_end,
                    n_flank_before=meta['n_flank_before'], n_flank_after=meta['n_flank_after']
                )
                fw_grid = build_framework_grid(fw_coords, fw_radii, resolution=grid_resolution, buffer=clash_buffer)
                loop_radii = np.tile([VDW_RADII['N'], VDW_RADII['CA'], VDW_RADII['C']], n_loop).astype(np.float32)

        # STAGE 1: KIC 
        print("  Stage 1: GenKIC (Finding closed Beta-sheet topologies)...")
        kic_ensemble, _, _ = refine_loop_3d_frames(
            full_seq, loop_start, loop_end,
            N_nat[:loop_start],  CA_nat[:loop_start], C_nat[:loop_start],  O_nat[:loop_start],
            N_nat[loop_end:],    CA_nat[loop_end:],   C_nat[loop_end:],    O_nat[loop_end:],
            router, n_structures=n_structures * 3, 
            k_clash=k_clash, clash_weight=clash_weight,
            max_init_clash=max_init_clash, max_init_intra=max_init_intra, max_init_energy=max_init_energy,
            framework_grid=fw_grid, n_samples=n_samples, temperature=temperature, filter_order=filter_order, n_cpus=n_cpus
        )
        
        if not kic_ensemble:
            print("  No KIC structures found!"); continue

        # STAGE 2: LANGEVIN DYNAMICS
        print(f"  Stage 2: Langevin Dynamics on {len(kic_ensemble)} structures...")
        
        # THE FIX: Extract EXACT mathematical angles from the KIC 3D coordinates!
        phi_list, psi_list = [], []
        for s in kic_ensemble:
            # Get full sequence coordinates (anchor + loop + closure)
            N_seg  = s[0][loop_start-1 : loop_end+1]
            CA_seg = s[1][loop_start-1 : loop_end+1]
            C_seg  = s[2][loop_start-1 : loop_end+1]
            
            ph_kic, ps_kic = coords_to_angles(N_seg, CA_seg, C_seg)
            # ph_kic: [NaN, phi1, phi2, ..., phiN, NaN]
            # ps_kic: [psiAnc, psi1, psi2, ..., psiN, NaN]
            phi_list.append(ph_kic[1 : 1 + n_loop])
            psi_list.append(ps_kic[0 : 1 + n_loop])

        phi_init = torch.tensor(np.deg2rad(np.array(phi_list)), dtype=torch.float32)
        psi_init = torch.tensor(np.deg2rad(np.array(psi_list)), dtype=torch.float32)

        # Boundary atoms
        anc_N, anc_CA, anc_C = N_nat[loop_start-1], CA_nat[loop_start-1], C_nat[loop_start-1]
        N_clos = N_nat[loop_end]
        CA_clos = CA_nat[loop_end] if loop_end < len(CA_nat) else None
        C_clos  = C_nat[loop_end] if loop_end < len(C_nat) else None

        bnd_coords, bnd_radii = _build_boundary_atoms(
            anc_N, anc_CA, anc_C, N_clos, CA_closure=CA_clos, C_closure=C_clos
        )

        phi_opt, psi_opt = sample_langevin_torsions(
            phi_init=phi_init, psi_init=psi_init,
            anchor_N=anc_N, anchor_CA=anc_CA, anchor_C=anc_C, N_closure=N_clos,
            probs_joint=probs_joint,
            n_steps=1000, 
            base_lr=0.02, 
            base_noise=0.01, 
            framework_grid=fw_grid,
            loop_radii=loop_radii,
            boundary_coords=bnd_coords,
            boundary_radii=bnd_radii
        )

        N_clos_t = torch.tensor(N_nat[loop_end], dtype=torch.float32)
        ensemble = _pack_ensemble(
            phi_opt, psi_opt, probs_joint, n_loop,
            N_nat[:loop_start], CA_nat[:loop_start], C_nat[:loop_start], O_nat[:loop_start],
            N_nat[loop_end:], CA_nat[loop_end:], C_nat[loop_end:], O_nat[loop_end:],
            N_nat[loop_start-1], CA_nat[loop_start-1], C_nat[loop_start-1], N_clos_t
        )

        ensemble.sort(key=lambda x: x[6] + clash_weight * x[7])
        final_ensemble = ensemble[:n_structures]

        rmsds_anc = compute_loop_rmsds(final_ensemble, CA_loop_nat, loop_start, loop_end)
        rmsds_aln = np.array([aligned_loop_rmsd(CA, CA_nat, loop_start, loop_end) for _, CA, *_ in final_ensemble])
        best_anc = int(np.argmin(rmsds_anc)); best_aln = int(np.argmin(rmsds_aln))

        print(f"  Anchored RMSD: best={rmsds_anc[best_anc]:.3f}Å  mean={rmsds_anc.mean():.3f}Å")
        print(f"  Aligned RMSD:  best={rmsds_aln[best_aln]:.3f}Å  mean={rmsds_aln.mean():.3f}Å")

        save_pdbs(final_ensemble, list(range(len(final_ensemble))), full_seq, loop_start, loop_end, CA_loop_nat, name, str(loop_out))
        results.append({'pdb_id': pdb_id, 'best_rmsd': float(rmsds_anc[best_anc])})

    return results

def _parse():
    p = argparse.ArgumentParser(description='CDR3 Prediction (GenKIC + Langevin)')
    p.add_argument('--dataset', default='/home/jtepperik/thesis/energy_model/scripts/data_processing/cdr3_dataset')
    p.add_argument('--output', default='cdr3_test_results_hybrid')
    p.add_argument('--checkpoint', default='/home/jtepperik/thesis/energy_model/scripts/training/outputs/energy_loss_c3')
    
    p.add_argument('--n-cpus', type=int, default=1)
    p.add_argument('--n-samples', type=int, default=5000)
    p.add_argument('--n-structures', type=int, default=10)
    p.add_argument('--max-loops', type=int, default=1)

    p.add_argument('--clash', action='store_true')
    p.add_argument('--complex-dir', default='/home/jtepperik/thesis/data/reference_final')
    
    p.add_argument('--max-init-clash', type=float, default=75.0)
    p.add_argument('--max-init-intra', type=float, default=75.0)
    p.add_argument('--max-init-energy', type=float, default=20)
    p.add_argument('--temperature', type=float, default=1)
    p.add_argument('--filter-order', type=str, default='energy,intra,fw')
    
    return p.parse_args()


if __name__ == '__main__':
    args = _parse()
    router = load_model(args.checkpoint)

    test_on_cdr3_dataset(
        dataset_dir=args.dataset, output_dir=args.output, router=router,
        n_structures=args.n_structures, max_loops=args.max_loops,
        complex_pdb_dir=args.complex_dir, use_clash=args.clash,
        max_init_clash=args.max_init_clash, max_init_intra=args.max_init_intra,
        max_init_energy=args.max_init_energy, n_samples=args.n_samples,
        temperature=args.temperature, filter_order=args.filter_order, n_cpus=args.n_cpus
    )