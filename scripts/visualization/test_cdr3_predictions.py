"""
test_on_cdr3.py

CDR3 loop structure prediction pipeline.

Usage
─────
  python test_on_cdr3.py --dataset /path/to/cdr3_dataset --output results/ \\
      --checkpoint /path/to/checkpoint \\
      --n-structures 20 --n-steps 500 \\
      --select best_n --n-select 5 --rank-by rmsd \\
      --clash --max-init-clash 2.0 --max-init-intra 2.0

Output structure
────────────────
  <output>/<pdb_id>_<chain>/
    ideal_<name>.pdb
    pdbs/
      structure_*.pdb        (selected structures)
      ensemble_<name>.pdb    (all structures, ranked by closure)
      summary.txt
    trajectory/
      trajectory_<name>.pdb
      trajectory_<name>.pml
    energy_plots/
      heatmap_<name>_native.png        ← native structure (when --plot-energy)
      landscape_<name>_native.png
      ramachandran_<name>_native.png
      heatmap_<name>_struct*.png       ← selected structures
      landscape_<name>_struct*.png
      ramachandran_<name>_struct*.png
  <output>/results.json
  <output>/summary.png
"""

import argparse
import json
from pathlib import Path

import numpy as np

from utils import (
    ModelRouter, OutputConfig,
    load_cdr3_native, load_model,
    compute_loop_rmsds,
    save_pdbs, save_trajectory,
    plot_energy, plot_native_energy, plot_summary,
    _compute_shared_vmax,
)

# Default imports from NeRF module — may be overridden by --method kic
from loop_modeling_nerf import (
    refine_loop_3d_frames as _refine_nerf, random_ensemble,
    ideal_energy, ideal_structure_pdb,
    ensemble_diversity, aligned_loop_rmsd,
    cache_energy_distributions, compute_energy,
    compute_native_clash_score,
    build_framework_grid,
    N_BINS,
)

# Will be set to either _refine_nerf or _refine_kic based on --method



# ─────────────────────────────────────────────────────────────────────────────
# Dihedral helpers
# ─────────────────────────────────────────────────────────────────────────────

def _dihedral(p1, p2, p3, p4):
    b0 = p1 - p2;  b1 = p3 - p2;  b2 = p4 - p3
    b1h = b1 / (np.linalg.norm(b1) + 1e-10)
    v   = b0 - np.dot(b0, b1h) * b1h
    w   = b2 - np.dot(b2, b1h) * b1h
    return np.degrees(np.arctan2(np.dot(np.cross(b1h, v), w), np.dot(v, w)))


def coords_to_angles(N, CA, C):
    n   = len(N)
    phi = np.full(n, np.nan)
    psi = np.full(n, np.nan)
    for i in range(n):
        if i > 0:   phi[i] = _dihedral(C[i-1], N[i], CA[i], C[i])
        if i < n-1: psi[i] = _dihedral(N[i], CA[i], C[i], N[i+1])
    return phi, psi


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def test_on_cdr3_dataset(
    dataset_dir:      str,
    output_dir:       str,
    router:           ModelRouter,
    output_config:    OutputConfig,
    n_structures:     int   = 10,
    max_loops:        int   = None,
    n_steps:          int   = 1000,
    lr_energy:        float = 0.05,
    lr_closure:       float = 0.20,
    closure_weight:   float = 50.0,
    eta_min:          float = 1e-4,
    n_frames:         int   = 50,
    # Clash detection
    complex_pdb_dir:  str   = "/home/jtepperik/thesis/data/reference_final",
    use_clash:        bool  = False,
    k_clash:          float = 1.0,
    clash_weight:     float = 5.0,
    clash_cutoff:     float = 8.0,
    clash_start_frac: float = 0.25,
    # New: pulsed ramping + grid
    n_pulses:         int   = 3,
    clash_floor_frac: float = 0.02,
    clash_cap:        float = 1.5,
    clash_buffer:     float = 0.5,
    grid_resolution:  float = 0.5,
    max_init_clash:    float = 2.0,
    max_init_closure:  float = None,
    max_init_intra:    float = 2.0,
    max_init_energy:   float = 30.0,
    max_init_attempts: int   = 500,
    # KIC-specific (passed through, ignored by NeRF)
    n_samples:         int   = 1000,
    n_refine_steps:    int   = 50,
    temperature:       float = 0.8,
    filter_order:      str   = 'energy,intra,fw',
) -> list:

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    metadata_file = Path(dataset_dir) / 'cdr3_dataset.json'
    if not metadata_file.exists():
        raise FileNotFoundError(f"{metadata_file} not found")

    with open(metadata_file) as f:
        dataset = json.load(f)
    if max_loops:
        dataset = dataset[:max_loops]

    print(f"\n{'='*60}")
    print(f"CDR3 STRUCTURE PREDICTION  (joint {N_BINS}×{N_BINS} model)")
    print(f"Loops: {len(dataset)}  Structures/loop: {n_structures}")
    if use_clash:
        print(f"Clash: ENABLED  k={k_clash}  weight={clash_weight}  "
              f"cutoff={clash_cutoff}Å  start={clash_start_frac:.0%}")
        print(f"  Pulsed: {n_pulses} cycles  floor={clash_floor_frac:.0%}  "
              f"cap={clash_cap}Å  grid={grid_resolution}Å")
    print(f"Filters: max_clash={max_init_clash}  max_intra={max_init_intra}  "
          f"max_energy={max_init_energy}")
    print(f"         max_attempts={max_init_attempts}  temp={temperature}  order={filter_order}")
    print(f"Output: {out.absolute()}\n{'='*60}")

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

        print(f"\n{'='*60}")
        print(f"{i}/{len(dataset)}  {pdb_id} chain {chain}")
        print(f"  Full: {full_seq}")
        print(f"  Loop: {cdr3_seq}  ({loop_start+1}–{loop_end})")

        # Load native structure
        pdb_file = Path(meta['pdb_file'])
        if not pdb_file.is_absolute():
            pdb_file = Path(dataset_dir) / pdb_file.name
        if not pdb_file.exists():
            print(f"  PDB not found: {pdb_file} — skipping");  continue

        seq, N_nat, CA_nat, C_nat, O_nat = load_cdr3_native(pdb_file)
        if seq != full_seq:
            print(f"  Sequence mismatch — skipping");  continue

        # Native angles
        start  = max(0, loop_start - 1)
        end    = min(len(seq), loop_end + 1)
        ph, ps = coords_to_angles(
            N_nat[start:end], CA_nat[start:end], C_nat[start:end]
        )
        offset      = loop_start - start
        n_loop      = loop_end - loop_start
        phi_n       = ph[offset : offset + n_loop]
        psi_n       = ps[offset : offset + n_loop]
        CA_loop_nat = CA_nat[loop_start:loop_end]

        # Cache distributions (used for both native energy and optimization)
        probs_joint = cache_energy_distributions(router, cdr3_seq)

        # Native torsion energy
        import torch
        phi_t = torch.tensor(
            np.deg2rad(np.nan_to_num(phi_n)), dtype=torch.float32
        ).unsqueeze(0)
        psi_t = torch.tensor(
            np.deg2rad(np.concatenate([[0.0], np.nan_to_num(psi_n), [0.0]])),
            dtype=torch.float32,
        ).unsqueeze(0)
        e_native = float(compute_energy(phi_t, psi_t, probs_joint)[0].item())

        print(f"  Native E={e_native:.2f}  "
              f"phi_mean={np.nanmean(phi_n):.1f}°  "
              f"psi_mean={np.nanmean(psi_n):.1f}°")

        # Ideal structure
        ideal_structure_pdb(
            probs_joint, cdr3_seq,
            N_nat[loop_start-1], CA_nat[loop_start-1], C_nat[loop_start-1],
            str(loop_out / f"ideal_{name}.pdb"),
        )

        # Framework extraction for clash detection
        fw_coords = fw_radii = fw_grid = None
        if use_clash:
            from utils import extract_framework_atoms
            complex_pdb = Path(complex_pdb_dir) / f"{pdb_id}.pdb"
            if complex_pdb.exists():
                try:
                    fw_coords, fw_radii = extract_framework_atoms(
                        str(complex_pdb),
                        tcr_chain      = chain,
                        full_sequence  = full_seq,
                        loop_start     = loop_start,
                        loop_end       = loop_end,
                        n_flank_before = meta['n_flank_before'],
                        n_flank_after  = meta['n_flank_after'],
                    )
                    # Build precomputed grid
                    fw_grid = build_framework_grid(
                        fw_coords, fw_radii,
                        resolution=grid_resolution,
                        buffer=clash_buffer,
                    )
                except Exception as exc:
                    print(f"  Warning: framework extraction failed ({exc})"
                          f" — no clash detection")
            else:
                print(f"  Warning: {complex_pdb} not found — no clash detection")

        # Native clash score
        native_clash = None
        if use_clash and fw_coords is not None:
            native_clash = compute_native_clash_score(
                N_nat[loop_start:loop_end],
                CA_nat[loop_start:loop_end],
                C_nat[loop_start:loop_end],
                framework_coords = fw_coords,
                framework_radii  = fw_radii,
                framework_grid   = fw_grid,
                k_clash          = k_clash,
                cutoff           = clash_cutoff,
            )
            print(f"  Native clash (overlaps only): intra={native_clash['intra']:.3f} ({native_clash['n_intra']} pairs)  "
                  f"fw={native_clash['framework']:.3f} ({native_clash['n_framework']} pairs)  "
                  f"total={native_clash['total']:.3f}")
        elif not use_clash:
            # Intra-loop clash only, no framework
            native_clash = compute_native_clash_score(
                N_nat[loop_start:loop_end],
                CA_nat[loop_start:loop_end],
                C_nat[loop_start:loop_end],
                k_clash = k_clash,
            )
            print(f"  Native intra-loop clash: {native_clash['intra']:.3f}")

        # Native energy plots — deferred until after ensemble (shared vmax)
        # (moved below)

        # Optimised ensemble
        ensemble, probs_joint, trajectory = refine_loop_3d_frames(
            full_seq, loop_start, loop_end,
            N_nat[:loop_start],  CA_nat[:loop_start],
            C_nat[:loop_start],  O_nat[:loop_start],
            N_nat[loop_end:],    CA_nat[loop_end:],
            C_nat[loop_end:],    O_nat[loop_end:],
            router,
            n_steps          = n_steps,
            lr_energy        = lr_energy,
            lr_closure       = lr_closure,
            closure_weight   = closure_weight,
            n_structures     = n_structures,
            eta_min          = eta_min,
            n_frames         = n_frames if output_config.save_trajectory else 0,
            framework_coords = fw_coords,
            framework_radii  = fw_radii,
            k_clash          = k_clash,
            clash_weight     = clash_weight,
            clash_cutoff     = clash_cutoff,
            clash_start_frac = clash_start_frac,
            # New parameters
            n_pulses          = n_pulses,
            clash_floor_frac  = clash_floor_frac,
            clash_cap         = clash_cap,
            clash_buffer      = clash_buffer,
            grid_resolution   = grid_resolution,
            max_init_clash    = max_init_clash,
            max_init_closure  = max_init_closure,
            max_init_intra    = max_init_intra,
            max_init_attempts = max_init_attempts,
            # Pass pre-built grid to avoid duplicate construction
            framework_grid    = fw_grid,
            # KIC-specific (ignored by NeRF method)

        )
        if not ensemble:
            print("  No structures returned");  continue

        # Metrics
        rmsds_anc = compute_loop_rmsds(ensemble, CA_loop_nat, loop_start, loop_end)
        rmsds_aln = np.array([
            aligned_loop_rmsd(CA, CA_nat, loop_start, loop_end)
            for _, CA, *_ in ensemble
        ])
        best_anc = int(np.argmin(rmsds_anc))
        best_aln = int(np.argmin(rmsds_aln))

        print(f"\n  Anchored: best={rmsds_anc[best_anc]:.3f}Å  "
              f"mean={rmsds_anc.mean():.3f}Å")
        print(f"  Aligned:  best={rmsds_aln[best_aln]:.3f}Å  "
              f"mean={rmsds_aln.mean():.3f}Å")

        _, mean_div, overall_div = ensemble_diversity(ensemble, loop_start, loop_end)
        e_ideal = ideal_energy(probs_joint)
        e_best  = ensemble[best_anc][-2]
        print(f"  Energy:   native={e_native:.2f}  best={e_best:.2f}  "
              f"ideal={e_ideal:.2f}  gap={e_best-e_ideal:.2f}")
        print(f"  Diversity: {overall_div:.3f}Å mean pairwise RMSD")

        # Random baselines (optional)
        rmsds_uni = rmsds_mod = np.array([])
        if output_config.run_baselines:
            ens_uni, _ = random_ensemble(
                full_seq, loop_start, loop_end,
                N_nat[:loop_start], CA_nat[:loop_start],
                C_nat[:loop_start], O_nat[:loop_start],
                N_nat[loop_end:],   CA_nat[loop_end:],
                C_nat[loop_end:],   O_nat[loop_end:],
                router, n_structures=n_structures, mode='uniform',
            )
            ens_mod, _ = random_ensemble(
                full_seq, loop_start, loop_end,
                N_nat[:loop_start], CA_nat[:loop_start],
                C_nat[:loop_start], O_nat[:loop_start],
                N_nat[loop_end:],   CA_nat[loop_end:],
                C_nat[loop_end:],   O_nat[loop_end:],
                router, n_structures=n_structures, mode='model_sample',
            )
            rmsds_uni = compute_loop_rmsds(ens_uni, CA_loop_nat, loop_start, loop_end)
            rmsds_mod = compute_loop_rmsds(ens_mod, CA_loop_nat, loop_start, loop_end)

            print(f"\n  {'Method':<28} {'Best RMSD':>10}  {'Mean RMSD':>10}")
            print(f"  {'─'*52}")
            print(f"  {'Optimised (anchored)':<28} "
                  f"{rmsds_anc[best_anc]:>9.3f}Å  {rmsds_anc.mean():>9.3f}Å")
            print(f"  {'Optimised (aligned)':<28} "
                  f"{rmsds_aln[best_aln]:>9.3f}Å  {rmsds_aln.mean():>9.3f}Å")
            print(f"  {'Uniform + closure':<28} "
                  f"{rmsds_uni.min():>9.3f}Å  {rmsds_uni.mean():>9.3f}Å")
            print(f"  {'Model sample + closure':<28} "
                  f"{rmsds_mod.min():>9.3f}Å  {rmsds_mod.mean():>9.3f}Å")
        else:
            print(f"\n  Anchored: best={rmsds_anc[best_anc]:.3f}Å  "
                  f"mean={rmsds_anc.mean():.3f}Å")
            print(f"  Aligned:  best={rmsds_aln[best_aln]:.3f}Å  "
                  f"mean={rmsds_aln.mean():.3f}Å")

        # Output selection
        output_config.output_dir = str(loop_out)
        selected_idx = output_config.resolve_indices(
            ensemble, CA_loop_nat, loop_start, loop_end,
        )
        print(f"\n  Output for structures: {[idx+1 for idx in selected_idx]}")

        top_idx = int(np.argmin(
            compute_loop_rmsds(ensemble, CA_loop_nat, loop_start, loop_end)
        ))

        if output_config.save_pdbs:
            save_pdbs(
                ensemble, selected_idx, full_seq,
                loop_start, loop_end, CA_loop_nat, name, str(loop_out),
            )

        if output_config.save_trajectory:
            save_trajectory(
                trajectory, top_idx, full_seq,
                loop_start, loop_end,
                N_nat[:loop_start],  CA_nat[:loop_start],
                C_nat[:loop_start],  O_nat[:loop_start],
                N_nat[loop_end:],    CA_nat[loop_end:],
                C_nat[loop_end:],    O_nat[loop_end:],
                CA_loop_nat, name, str(loop_out),
            )

        if output_config.plot_energy:
            # Compute shared colour scales across native + generated
            rama_range, heatmap_range = _compute_shared_vmax(
                probs_joint,
                phi_native=phi_n, psi_native=psi_n,
                ensemble=ensemble, selected_idx=selected_idx,
            )

            # Native energy plots (deferred to here for shared vmax)
            plot_native_energy(
                phi_n, psi_n, probs_joint, cdr3_seq,
                name, str(loop_out),
                native_clash=native_clash,
                rama_range=rama_range, heatmap_range=heatmap_range,
            )

            # Generated structure energy plots
            plot_energy(
                ensemble, selected_idx, cdr3_seq,
                probs_joint, name, str(loop_out),
                rama_range=rama_range, heatmap_range=heatmap_range,
            )

        results.append({
            'pdb_id':                pdb_id,
            'chain':                 chain,
            'sequence':              cdr3_seq,
            'loop_length':           n_loop,
            'best_rmsd':             float(rmsds_anc[best_anc]),
            'mean_rmsd':             float(rmsds_anc.mean()),
            'all_rmsds':             rmsds_anc.tolist(),
            'best_rmsd_aln':         float(rmsds_aln[best_aln]),
            'mean_rmsd_aln':         float(rmsds_aln.mean()),
            'all_rmsds_aln':         rmsds_aln.tolist(),
            'e_native':              e_native,
            'best_energy':           float(e_best),
            'ideal_energy':          float(e_ideal),
            'energy_gap':            float(e_best - e_ideal),
            'mean_diversity':        float(overall_div),
            'clash_enabled':         use_clash and fw_coords is not None,
            'native_clash_intra':    native_clash['intra']     if native_clash else None,
            'native_clash_fw':       native_clash['framework'] if native_clash else None,
            'native_clash_total':    native_clash['total']     if native_clash else None,
            'baseline_uniform_best': float(rmsds_uni.min())  if rmsds_uni.size else None,
            'baseline_uniform_mean': float(rmsds_uni.mean()) if rmsds_uni.size else None,
            'baseline_model_best':   float(rmsds_mod.min())  if rmsds_mod.size else None,
            'baseline_model_mean':   float(rmsds_mod.mean()) if rmsds_mod.size else None,
            'per_struct_div':        mean_div.tolist(),
            'best_closure':          float(ensemble[best_anc][-1]),
            'selected_indices':      selected_idx,
        })

    # Summary across all loops
    if results:
        ba = [r['best_rmsd']     for r in results]
        al = [r['best_rmsd_aln'] for r in results]
        print(f"\n{'='*60}\nSUMMARY ({len(results)} loops)")
        print(f"\n  {'Metric':<28} {'Anchored':>10}  {'Aligned':>10}")
        print(f"  {'─'*50}")
        print(f"  {'Mean best RMSD':<28} {np.mean(ba):>9.3f}Å  {np.mean(al):>9.3f}Å")
        print(f"  {'Median best RMSD':<28} {np.median(ba):>9.3f}Å  {np.median(al):>9.3f}Å")
        for t in [1, 2, 3]:
            print(f"  {'< '+str(t)+'Å':<28} "
                  f"{sum(r<t for r in ba):>9}/{len(ba)}  "
                  f"{sum(r<t for r in al):>9}/{len(al)}")

        with open(out / 'results.json', 'w') as f:
            json.dump(results, f, indent=2)

        if output_config.plot_summary:
            if output_config.run_baselines:
                plot_summary(results, str(out))
            else:
                print("    Summary plot skipped (baselines not run)")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse():
    p = argparse.ArgumentParser(description='CDR3 loop structure prediction')

    # Data
    p.add_argument('--dataset',
                   default='/home/jtepperik/thesis/energy_model/scripts/data_processing/cdr3_dataset')
    p.add_argument('--output',     default='cdr3_test_results7')
    p.add_argument('--checkpoint',
                   default='/home/jtepperik/thesis/energy_model/scripts/training/outputs/energy_loss_c3')

    # Method selection
    p.add_argument('--method', default='nerf', choices=['nerf', 'kic'],
                   help='Loop modeling method: nerf (gradient descent) or kic (kinematic closure)')
    p.add_argument('--n-samples',  type=int, default=1000,
                   help='[KIC] Number of KIC closure attempts')
    p.add_argument('--n-refine-steps', type=int, default=50,
                   help='[KIC] Gradient refinement steps after closure')

    # Optimisation
    p.add_argument('--n-structures',   type=int,   default=1)
    p.add_argument('--max-loops',      type=int,   default=1)
    p.add_argument('--n-steps',        type=int,   default=250)
    p.add_argument('--lr-energy',      type=float, default=0.45)
    p.add_argument('--lr-closure',     type=float, default=0.05)
    p.add_argument('--closure-weight', type=float, default=15.0)
    p.add_argument('--eta-min',        type=float, default=1e-3)
    p.add_argument('--n-frames',       type=int,   default=50)

    # Output selection
    p.add_argument('--select',   default='best_n',
                   choices=['best_n', 'indices', 'all'])
    p.add_argument('--n-select', type=int, default=5)
    p.add_argument('--indices',  type=int, nargs='+', default=None)
    p.add_argument('--rank-by',  default='rmsd',
                   choices=['rmsd', 'closure', 'energy'])

    # Output toggles (all on by default — pass --no-X to disable)
    p.add_argument('--no-pdbs',       action='store_true')
    p.add_argument('--no-trajectory', action='store_true')
    p.add_argument('--no-energy',     action='store_true')
    p.add_argument('--no-summary',    action='store_true')
    p.add_argument('--no-baselines',  action='store_false')

    # Clash detection
    p.add_argument('--clash',            action='store_true',
                   help='Enable clash detection against full TCR-pMHC complex')
    p.add_argument('--complex-dir',
                   default='/home/jtepperik/thesis/data/reference_final')
    p.add_argument('--k-clash',          type=float, default=1.0)
    p.add_argument('--clash-weight',     type=float, default=5.0)
    p.add_argument('--clash-cutoff',     type=float, default=8.0)
    p.add_argument('--clash-start-frac', type=float, default=0.1)

    # New: pulsed ramping + grid + init filtering
    p.add_argument('--n-pulses',          type=int,   default=3,
                   help='Number of Rosetta-style clash ramp cycles')
    p.add_argument('--clash-floor-frac',  type=float, default=0.02,
                   help='Minimum clash weight as fraction of full (per pulse)')
    p.add_argument('--clash-cap',         type=float, default=1.5,
                   help='Max overlap (Å) for linear-capped potential')
    p.add_argument('--clash-buffer',      type=float, default=0.5,
                   help='Buffer zone width (Å) for pre-contact gradient signal')
    p.add_argument('--grid-resolution',   type=float, default=0.5,
                   help='Framework grid voxel size (Å)')
    
    p.add_argument('--max-init-clash',    type=float, default=5.0,
                   help='Max fw clash score to accept an initialisation')
    p.add_argument('--max-init-closure',  type=float, default=None,
                   help='Max closure distance (Å) to accept an initialisation')
    p.add_argument('--max-init-intra',    type=float, default=1.0,
                   help='Max intra-loop clash score to accept an initialisation')
    p.add_argument('--max-init-energy',   type=float, default=20.0,
                   help='Max sum energy for the 3 pivot residues')
                   
    p.add_argument('--max-init-attempts', type=int,   default=500,
                   help='Max sampling attempts for filtered init')
    p.add_argument('--temperature',       type=float, default=0.8,
                   help='[KIC] Temperature for biasing sampling towards low energy')
    p.add_argument('--filter-order',      type=str, default='fw,energy,intra',
                   help='Comma-separated order of KIC filters (energy, intra, fw)')

    return p.parse_args()


if __name__ == '__main__':
    args = _parse()

    router = load_model(args.checkpoint)

    # Switch loop modeling method
    global refine_loop_3d_frames
    if args.method == 'kic':
        from kic_loop_modeling import refine_loop_3d_frames as _refine_kic
        refine_loop_3d_frames = _refine_kic
        print(f"Method: KIC (analytical closure, {args.n_samples} samples)")
    else:
        refine_loop_3d_frames = _refine_nerf
        print(f"Method: NeRF (gradient descent, {args.n_steps} steps)")

    cfg = OutputConfig(
        selection_mode  = args.select,
        n_select        = args.n_select,
        indices         = args.indices or [],
        rank_by         = args.rank_by,
        save_pdbs       = not args.no_pdbs,
        save_trajectory = not args.no_trajectory,
        plot_energy     = not args.no_energy,
        plot_summary    = not args.no_summary,
        run_baselines   = not args.no_baselines,
        output_dir      = args.output,
    )

    test_on_cdr3_dataset(
        dataset_dir       = args.dataset,
        output_dir        = args.output,
        router            = router,
        output_config     = cfg,
        n_structures      = args.n_structures,
        max_loops         = args.max_loops,
        n_steps           = args.n_steps,
        lr_energy         = args.lr_energy,
        lr_closure        = args.lr_closure,
        closure_weight    = args.closure_weight,
        eta_min           = args.eta_min,
        n_frames          = args.n_frames,
        complex_pdb_dir   = args.complex_dir,
        use_clash         = args.clash,            
        k_clash           = args.k_clash,
        clash_weight      = args.clash_weight,
        clash_cutoff      = args.clash_cutoff,
        clash_start_frac  = args.clash_start_frac,
        # New parameters
        n_pulses          = args.n_pulses,
        clash_floor_frac  = args.clash_floor_frac,
        clash_cap         = args.clash_cap,
        clash_buffer      = args.clash_buffer,
        grid_resolution   = args.grid_resolution,
        max_init_clash    = args.max_init_clash,
        max_init_closure  = args.max_init_closure,
        max_init_intra    = args.max_init_intra,
        max_init_energy   = args.max_init_energy,
        max_init_attempts = args.max_init_attempts,
        # KIC-specific (ignored by NeRF method)
        n_samples         = args.n_samples,
        n_refine_steps    = args.n_refine_steps,
        temperature       = args.temperature,
        filter_order      = args.filter_order,
    )