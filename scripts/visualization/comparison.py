"""
Test CDR3 loop modeling -- four methods compared:
  1. NeRF          : refine_loop_3d_frames  (torsion-only, exact geometry)
  2. SE(3)         : refine_loop_se3_fixed  (bond-axis rotations + bond lengths)
  3. XYZ           : refine_loop_xyz_unconstrained (raw positions, all soft)
  4. Baselines     : random_ensemble uniform + model_sample (NeRF geometry, no energy guidance)
"""

import sys
sys.path.append('/home/jtepperik/thesis/energy_model/scripts')

import numpy as np
from pathlib import Path
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from ensemble_from_random_clouds import load_model, coords_to_angles
from se3_loop_modeling import refine_loop_se3_fixed
from se3_loop_modeling_free import refine_loop_xyz_unconstrained
from loop_modeling_nerf import (
    refine_loop_3d_frames,
    random_ensemble,
    ideal_energy, ensemble_diversity,
    kabsch, aligned_loop_rmsd,
    ideal_structure_pdb,
    cache_energy_distributions,
)

ONE_TO_THREE = {
    'A': 'ALA', 'C': 'CYS', 'D': 'ASP', 'E': 'GLU',
    'F': 'PHE', 'G': 'GLY', 'H': 'HIS', 'I': 'ILE',
    'K': 'LYS', 'L': 'LEU', 'M': 'MET', 'N': 'ASN',
    'P': 'PRO', 'Q': 'GLN', 'R': 'ARG', 'S': 'SER',
    'T': 'THR', 'V': 'VAL', 'W': 'TRP', 'Y': 'TYR',
}
THREE_TO_ONE = {v: k for k, v in ONE_TO_THREE.items()}


# ============================================================================
# PDB I/O
# ============================================================================

def load_cdr3_native(pdb_file):
    N_list, CA_list, C_list, O_list, sequence = [], [], [], [], []
    with open(pdb_file) as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            atom = line[12:16].strip()
            res  = line[17:20].strip()
            x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
            if atom == "CA": sequence.append(THREE_TO_ONE.get(res, 'X'))
            if   atom == "N":  N_list .append([x, y, z])
            elif atom == "CA": CA_list.append([x, y, z])
            elif atom == "C":  C_list .append([x, y, z])
            elif atom == "O":  O_list .append([x, y, z])
    return (
        ''.join(sequence),
        np.array(N_list), np.array(CA_list),
        np.array(C_list), np.array(O_list),
    )


def _write_pdb_atoms(f, sequence, N, CA, C, O, atom_num=1):
    for i, aa in enumerate(sequence):
        res = ONE_TO_THREE.get(aa, 'UNK')
        for name, coord in [('N', N[i]), ('CA', CA[i]), ('C', C[i])]:
            f.write(f"ATOM  {atom_num:5d}  {name:<3s} {res} A{i+1:4d}    "
                    f"{coord[0]:8.3f}{coord[1]:8.3f}{coord[2]:8.3f}"
                    f"  1.00  0.00           {name[0]}  \n")
            atom_num += 1
        if np.linalg.norm(O[i]) > 1e-6:
            f.write(f"ATOM  {atom_num:5d}  O   {res} A{i+1:4d}    "
                    f"{O[i,0]:8.3f}{O[i,1]:8.3f}{O[i,2]:8.3f}"
                    f"  1.00  0.00           O  \n")
            atom_num += 1
    return atom_num


# ============================================================================
# RMSD
# ============================================================================

def loop_rmsds(ensemble, CA_native_loop, loop_start, loop_end):
    rmsds = [
        float(np.sqrt(np.mean(np.sum(
            (CA[loop_start:loop_end] - CA_native_loop) ** 2, axis=1
        ))))
        for _, CA, *_ in ensemble
    ]
    return rmsds, int(np.argmin(rmsds))


def aligned_loop_rmsds(ensemble, CA_native_full, loop_start, loop_end):
    rmsds = [
        aligned_loop_rmsd(CA_full, CA_native_full, loop_start, loop_end)
        for _, CA_full, *_ in ensemble
    ]
    return rmsds, int(np.argmin(rmsds))


# ============================================================================
# SAVING
# ============================================================================

def save_ensemble(ensemble, full_sequence, loop_start, loop_end,
                  CA_native_full, name, output_dir):
    out = Path(output_dir) / f"ensemble_{name}"
    out.mkdir(parents=True, exist_ok=True)

    CA_native_loop = CA_native_full[loop_start:loop_end]
    rmsds_anc, _ = loop_rmsds(ensemble, CA_native_loop, loop_start, loop_end)
    rmsds_aln    = [aligned_loop_rmsd(CA, CA_native_full, loop_start, loop_end)
                    for _, CA, *_ in ensemble]
    order = np.argsort(rmsds_anc)

    for i, (N, CA, C, O, *rest) in enumerate(ensemble):
        fname = (f"structure_{i+1:02d}_"
                 f"anc{rmsds_anc[i]:.2f}_aln{rmsds_aln[i]:.2f}A.pdb")
        with open(out / fname, 'w') as f:
            f.write(f"REMARK anchored_rmsd={rmsds_anc[i]:.3f}A  "
                    f"aligned_rmsd={rmsds_aln[i]:.3f}A  "
                    f"closure={rest[-1]:.4f}A  E={rest[-2]:.3f}\n")
            _write_pdb_atoms(f, full_sequence, N, CA, C, O)
            f.write("END\n")

    with open(out / f"ensemble_{name}.pdb", 'w') as f:
        for rank, idx in enumerate(order, 1):
            N, CA, C, O, *rest = ensemble[idx]
            f.write(f"MODEL {rank:4d}\n")
            f.write(f"REMARK anchored={rmsds_anc[idx]:.3f}A  "
                    f"aligned={rmsds_aln[idx]:.3f}A  "
                    f"closure={rest[-1]:.4f}A\n")
            _write_pdb_atoms(f, full_sequence, N, CA, C, O)
            f.write("ENDMDL\n")

    with open(out / "summary.txt", 'w') as f:
        f.write(f"{'Rank':>4}  {'Struct':>10}  {'Anchored':>10}  "
                f"{'Aligned':>10}  {'Closure':>10}  {'Energy':>10}\n")
        f.write("-" * 58 + "\n")
        for rank, idx in enumerate(order, 1):
            rest = ensemble[idx][4:]
            f.write(f"{rank:>4}  structure_{idx+1:02d}  "
                    f"{rmsds_anc[idx]:>9.3f}A  {rmsds_aln[idx]:>9.3f}A  "
                    f"{rest[-1]:>9.4f}A  {rest[-2]:>10.2f}\n")

    print(f"    Saved {len(ensemble)} structures -> {out.name}/  "
          f"(best anc={rmsds_anc[order[0]]:.3f}A  best aln={min(rmsds_aln):.3f}A)")


def save_pymol_comparison(ensemble, best_idx, full_sequence, loop_start, loop_end,
                           N_native, CA_native, C_native, O_native, name, output_dir):
    out = Path(output_dir)
    rmsds, _ = loop_rmsds(ensemble, CA_native[loop_start:loop_end], loop_start, loop_end)
    N_p, CA_p, C_p, O_p = ensemble[best_idx][:4]

    per_res  = np.sqrt(np.sum(
        (CA_p[loop_start:loop_end] - CA_native[loop_start:loop_end]) ** 2, axis=1
    ))
    bfactors = np.zeros(len(full_sequence))
    bfactors[loop_start:loop_end] = per_res * 10

    pred_pdb = out / f"predicted_{name}.pdb"
    with open(pred_pdb, 'w') as f:
        f.write(f"REMARK Predicted {name}  best_rmsd={rmsds[best_idx]:.3f}A\n")
        atom_num = 1
        for i, aa in enumerate(full_sequence):
            res = ONE_TO_THREE.get(aa, 'UNK')
            bf  = bfactors[i]
            for aname, coord in [('N', N_p[i]), ('CA', CA_p[i]), ('C', C_p[i])]:
                f.write(f"ATOM  {atom_num:5d}  {aname:<3s} {res} A{i+1:4d}    "
                        f"{coord[0]:8.3f}{coord[1]:8.3f}{coord[2]:8.3f}"
                        f"  1.00{bf:6.2f}           {aname[0]}  \n")
                atom_num += 1
            if np.linalg.norm(O_p[i]) > 1e-6:
                f.write(f"ATOM  {atom_num:5d}  O   {res} A{i+1:4d}    "
                        f"{O_p[i,0]:8.3f}{O_p[i,1]:8.3f}{O_p[i,2]:8.3f}"
                        f"  1.00{bf:6.2f}           O  \n")
                atom_num += 1
        f.write("END\n")

    native_pdb = out / f"native_{name}.pdb"
    with open(native_pdb, 'w') as f:
        f.write(f"REMARK Native {name}\n")
        _write_pdb_atoms(f, full_sequence, N_native, CA_native, C_native, O_native)
        f.write("END\n")

    worst = loop_start + int(np.argmax(per_res)) + 1
    with open(out / f"view_{name}.pml", 'w') as f:
        f.write(f"load {pred_pdb.name}, pred\nload {native_pdb.name}, native\n")
        f.write("bg_color white\nset cartoon_smooth_loops, 1\n\n")
        f.write("hide everything\nshow cartoon, pred\nshow cartoon, native\n")
        f.write("color grey80, pred\ncolor grey60, native\n")
        f.write(f"spectrum b, blue_white_red, pred and resi {loop_start+1}-{loop_end}\n")
        f.write(f"color red, native and resi {loop_start+1}-{loop_end}\n")
        f.write("set cartoon_transparency, 0.4, native\n")
        f.write(f"distance d, pred and resi {worst} and name CA, "
                f"native and resi {worst} and name CA\n")
        f.write("zoom all\n")

    print(f"    PyMOL: {pred_pdb.name}, {native_pdb.name}, view_{name}.pml")


# ============================================================================
# VISUALIZATION
# ============================================================================

def visualize_loop(CA_pred, CA_native, per_res_rmsd, name, sequence, rmsd, output_dir):
    fig = plt.figure(figsize=(15, 5))

    ax = fig.add_subplot(131, projection='3d')
    ax.plot(*CA_pred.T,   'b-o', label='Predicted', alpha=0.7, markersize=5)
    ax.plot(*CA_native.T, 'r-o', label='Native',    alpha=0.7, markersize=5)
    for i in range(len(CA_pred)):
        ax.plot([CA_pred[i,0], CA_native[i,0]],
                [CA_pred[i,1], CA_native[i,1]],
                [CA_pred[i,2], CA_native[i,2]], 'gray', alpha=0.3, linewidth=1)
    ax.set_title(f'{name}\nRMSD={rmsd:.2f}A'); ax.legend()

    ax = fig.add_subplot(132)
    res  = np.arange(1, len(per_res_rmsd) + 1)
    cols = ['green' if r < 1 else 'orange' if r < 2 else 'red' for r in per_res_rmsd]
    ax.bar(res, per_res_rmsd, color=cols, edgecolor='black')
    ax.axhline(1.0, color='green',  linestyle='--', alpha=0.5)
    ax.axhline(2.0, color='orange', linestyle='--', alpha=0.5)
    ax.set_xlabel('Residue'); ax.set_ylabel('RMSD (A)')
    ax.set_title(f'Per-residue RMSD\n{sequence}')

    ax = fig.add_subplot(133)
    ax.plot(res, per_res_rmsd, 'bo-')
    ax.fill_between(res, per_res_rmsd, alpha=0.3)
    ax.axhline(np.mean(per_res_rmsd), color='red', linestyle='--',
               label=f'Mean {np.mean(per_res_rmsd):.2f}A')
    ax.set_xlabel('Residue'); ax.set_ylabel('RMSD (A)')
    ax.set_title('RMSD along sequence'); ax.legend()

    plt.tight_layout()
    path = Path(output_dir) / f"alignment_{name}.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Plot: {path.name}")


def plot_summary(results, output_dir):
    best_nerf = [r['best_rmsd_nerf'] for r in results]
    best_se3  = [r['best_rmsd_se3']  for r in results]
    best_xyz  = [r['best_rmsd_xyz']  for r in results]
    best_uni  = [r['baseline_uniform_best'] for r in results]
    best_msmp = [r['baseline_model_best']   for r in results]

    fig, axes = plt.subplots(1, 3, figsize=(21, 5))

    all_vals = best_nerf + best_se3 + best_xyz + best_uni + best_msmp
    bins = np.linspace(0, max(all_vals) * 1.05, 18)
    for vals, label, color in [
        (best_nerf, f'NeRF torsion    (mean {np.mean(best_nerf):.2f}A)', 'mediumseagreen'),
        (best_se3,  f'SE(3) bond-axis (mean {np.mean(best_se3):.2f}A)',  'steelblue'),
        (best_xyz,  f'XYZ unconstr.   (mean {np.mean(best_xyz):.2f}A)',  'darkorange'),
        (best_uni,  f'Random uniform  (mean {np.mean(best_uni):.2f}A)',  'lightcoral'),
        (best_msmp, f'Model sample    (mean {np.mean(best_msmp):.2f}A)', 'tomato'),
    ]:
        axes[0].hist(vals, bins=bins, alpha=0.5, edgecolor='black', label=label, color=color)
    axes[0].set_xlabel('Best RMSD (A)'); axes[0].set_title('Distribution')
    axes[0].legend(fontsize=7)

    # Scatter: NeRF (x-axis) vs all others
    for vals, label, color in [
        (best_se3,  'SE(3)',        'steelblue'),
        (best_xyz,  'XYZ',         'darkorange'),
        (best_uni,  'uniform',     'lightcoral'),
        (best_msmp, 'model sample','tomato'),
    ]:
        axes[1].scatter(best_nerf, vals, alpha=0.8, s=55, color=color,
                        edgecolors='white', linewidths=0.4, label=label)
    lim = max(max(best_nerf), max(best_se3), max(best_xyz),
              max(best_uni), max(best_msmp)) * 1.05
    axes[1].plot([0, lim], [0, lim], 'k--', linewidth=0.8, alpha=0.4, label='y = x')
    axes[1].set_xlabel('NeRF torsion RMSD (A)')
    axes[1].set_ylabel('Comparison RMSD (A)')
    axes[1].set_title('NeRF vs all methods\n(below diagonal = NeRF wins)')
    axes[1].legend(fontsize=7)

    for vals, label, color, ls in [
        (best_nerf, 'NeRF torsion',  'mediumseagreen', '-'),
        (best_se3,  'SE(3)',         'steelblue',       '--'),
        (best_xyz,  'XYZ unconstr.', 'darkorange',      '-.'),
        (best_uni,  'Random uniform','lightcoral',      (0,(3,1,1,1))),
        (best_msmp, 'Model sample',  'tomato',          ':'),
    ]:
        s = sorted(vals)
        c = np.arange(1, len(s)+1) / len(s) * 100
        axes[2].plot(s, c, color=color, linestyle=ls, linewidth=2, label=label)
    for t, c in [(1,'red'), (2,'orange'), (3,'green')]:
        axes[2].axvline(t, color=c, linestyle='--', alpha=0.3, linewidth=0.8)
    axes[2].set_xlabel('RMSD (A)'); axes[2].set_ylabel('Success (%)')
    axes[2].set_title('Cumulative success rate'); axes[2].legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'summary.png', dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================================
# MAIN TEST LOOP
# ============================================================================

def test_on_cdr3_dataset(
    dataset_dir      = "cdr3_dataset",
    output_dir       = "cdr3_test_results",
    model            = None,
    params           = None,
    n_structures     = 10,
    max_loops        = None,
    n_steps          = 1000,
    # NeRF knobs
    lr_energy        = 0.05,
    lr_closure       = 0.20,
    closure_weight   = 50.0,
    # SE(3) knobs
    lr_torsion       = 0.05,
    lr_bond          = 2.0,
    bond_weight      = 10.0,
    position_scale   = 10.0,
    n_frames         = 20,
    # XYZ knobs
    xyz_lr           = 0.05,
    xyz_bond_weight  = 100.0,
    xyz_angle_weight = 50.0,
):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if model is None or params is None:
        model, params = load_model()

    metadata_file = Path(dataset_dir) / "cdr3_dataset.json"
    if not metadata_file.exists():
        raise FileNotFoundError(f"{metadata_file} not found")

    with open(metadata_file) as f:
        dataset = json.load(f)
    if max_loops:
        dataset = dataset[:max_loops]

    print(f"\n{'='*60}")
    print(f"CDR3 STRUCTURE PREDICTION  --  method comparison")
    print(f"  1. NeRF torsion    (exact geometry, energy-guided phi/psi)")
    print(f"  2. SE(3) bond-axis (exact intra, soft inter, bond-axis DOFs)")
    print(f"  3. XYZ unconstr.   (all geometry soft, raw position DOFs)")
    print(f"  4. Random baselines (NeRF geometry, no energy guidance)")
    print(f"{'='*60}")
    print(f"Loops: {len(dataset)}  Structures/method: {n_structures}")
    print(f"Output: {out.absolute()}\n")

    results = []

    for i, meta in enumerate(dataset, 1):
        print(f"\n{'='*60}")
        print(f"{i}/{len(dataset)}  {meta['pdb_id']} chain {meta['chain']}")

        full_seq   = meta['full_sequence']
        cdr3_seq   = meta['cdr3_sequence']
        loop_start = meta['loop_start']
        loop_end   = meta['loop_end']
        print(f"  Full: {full_seq}   Loop: {cdr3_seq} ({loop_start+1}-{loop_end})")

        pdb_file = Path(meta['pdb_file'])
        if not pdb_file.is_absolute():
            pdb_file = Path(dataset_dir) / pdb_file.name
        if not pdb_file.exists():
            print(f"  PDB not found: {pdb_file}"); continue

        seq, N_nat, CA_nat, C_nat, O_nat = load_cdr3_native(pdb_file)
        if seq != full_seq:
            print(f"  Sequence mismatch -- skipping"); continue

        phi_n, psi_n = coords_to_angles(
            N_nat[loop_start:loop_end],
            CA_nat[loop_start:loop_end],
            C_nat[loop_start:loop_end],
        )
        ca_ca = np.linalg.norm(
            CA_nat[loop_start+1:loop_end] - CA_nat[loop_start:loop_end-1], axis=1
        )
        print(f"  Native: phi={np.mean(phi_n[1:-1]):.1f}  psi={np.mean(psi_n[1:-1]):.1f}  "
              f"CA-CA={np.mean(ca_ca):.2f}+/-{np.std(ca_ca):.2f}A")

        name = f"{meta['pdb_id']}_{meta['chain']}"
        (Path(out) / name).mkdir(parents=True, exist_ok=True)
        probs_phi, probs_psi = cache_energy_distributions(model, params, cdr3_seq)
        CA_loop_nat = CA_nat[loop_start:loop_end]

        # shared flank slices (used by all methods)
        kw_flanks = dict(
            N_flank_before  = N_nat[:loop_start],
            CA_flank_before = CA_nat[:loop_start],
            C_flank_before  = C_nat[:loop_start],
            O_flank_before  = O_nat[:loop_start],
            N_flank_after   = N_nat[loop_end:],
            CA_flank_after  = CA_nat[loop_end:],
            C_flank_after   = C_nat[loop_end:],
            O_flank_after   = O_nat[loop_end:],
        )

        # ideal-energy structure (NeRF argmax, no flanks)
        ideal_pdb_path = Path(out) / name / f"ideal_{name}.pdb"
        ideal_structure_pdb(
            probs_phi, probs_psi, cdr3_seq,
            N_nat[loop_start - 1], CA_nat[loop_start - 1], C_nat[loop_start - 1],
            str(ideal_pdb_path),
        )

        # ── 1. NeRF torsion ───────────────────────────────────────────────────
        print(f"\n  -- NeRF torsion --")
        ens_nerf = refine_loop_3d_frames(
            full_seq, loop_start, loop_end,
            **kw_flanks,
            model=model, params=params,
            n_steps=n_steps,
            lr_energy=lr_energy,
            lr_closure=lr_closure,
            closure_weight=closure_weight,
            n_structures=n_structures,
        )
        rmsds_nerf, best_nerf_idx = loop_rmsds(ens_nerf, CA_loop_nat, loop_start, loop_end)
        rmsds_aln,  best_aln_idx  = aligned_loop_rmsds(ens_nerf, CA_nat, loop_start, loop_end)
        print(f"  NeRF anchored  -- best={rmsds_nerf[best_nerf_idx]:.3f}A  "
              f"mean={np.mean(rmsds_nerf):.3f}A  "
              f"closure={ens_nerf[best_nerf_idx][-1]:.4f}A")
        print(f"  NeRF aligned   -- best={rmsds_aln[best_aln_idx]:.3f}A  "
              f"mean={np.mean(rmsds_aln):.3f}A")

        # ── 2. SE(3) bond-axis ────────────────────────────────────────────────
        print(f"\n  -- SE(3) bond-axis --")
        traj_path = str(Path(out) / name / f"trajectory_{name}.pdb") if n_frames > 0 else None
        ens_se3 = refine_loop_se3_fixed(
            full_seq, loop_start, loop_end,
            **kw_flanks,
            model=model, params=params,
            n_steps=n_steps,
            lr_torsion=lr_torsion,
            lr_bond=lr_bond,
            bond_weight=bond_weight,
            closure_weight=closure_weight,
            n_structures=n_structures,
            position_scale=position_scale,
            n_frames=n_frames,
            trajectory_path=traj_path,
            CA_native_loop=CA_loop_nat,
        )
        rmsds_se3, best_se3_idx = loop_rmsds(ens_se3, CA_loop_nat, loop_start, loop_end)
        print(f"  SE(3) anchored -- best={rmsds_se3[best_se3_idx]:.3f}A  "
              f"mean={np.mean(rmsds_se3):.3f}A  "
              f"closure={ens_se3[best_se3_idx][-1]:.4f}A")

        # ── 3. XYZ unconstrained ──────────────────────────────────────────────
        print(f"\n  -- XYZ unconstrained --")
        ens_xyz = refine_loop_xyz_unconstrained(
            full_seq, loop_start, loop_end,
            **kw_flanks,
            model=model, params=params,
            n_steps=n_steps,
            lr=xyz_lr,
            bond_weight=xyz_bond_weight,
            angle_weight=xyz_angle_weight,
            closure_weight=closure_weight,
            n_structures=n_structures,
            position_scale=position_scale,
        )
        rmsds_xyz, best_xyz_idx = loop_rmsds(ens_xyz, CA_loop_nat, loop_start, loop_end)
        print(f"  XYZ unconstr.  -- best={rmsds_xyz[best_xyz_idx]:.3f}A  "
              f"mean={np.mean(rmsds_xyz):.3f}A  "
              f"closure={ens_xyz[best_xyz_idx][-1]:.4f}A")

        # ── 4. Random baselines ───────────────────────────────────────────────
        print(f"\n  -- Random baselines --")
        ens_uniform, _, _ = random_ensemble(
            full_seq, loop_start, loop_end, **kw_flanks,
            model=model, params=params,
            n_structures=n_structures, mode='uniform',
        )
        ens_model, _, _ = random_ensemble(
            full_seq, loop_start, loop_end, **kw_flanks,
            model=model, params=params,
            n_structures=n_structures, mode='model_sample',
        )
        rmsds_uni,   _ = loop_rmsds(ens_uniform, CA_loop_nat, loop_start, loop_end)
        rmsds_msamp, _ = loop_rmsds(ens_model,   CA_loop_nat, loop_start, loop_end)

        # ── Summary table ─────────────────────────────────────────────────────
        print(f"\n  {'Method':<30} {'Best RMSD':>10}  {'Mean RMSD':>10}")
        print(f"  {'-'*54}")
        print(f"  {'NeRF torsion':<30} {rmsds_nerf[best_nerf_idx]:>9.3f}A  {np.mean(rmsds_nerf):>9.3f}A")
        print(f"  {'NeRF aligned (Kabsch)':<30} {rmsds_aln[best_aln_idx]:>9.3f}A  {np.mean(rmsds_aln):>9.3f}A")
        print(f"  {'SE(3) bond-axis':<30} {rmsds_se3[best_se3_idx]:>9.3f}A  {np.mean(rmsds_se3):>9.3f}A")
        print(f"  {'XYZ unconstrained':<30} {rmsds_xyz[best_xyz_idx]:>9.3f}A  {np.mean(rmsds_xyz):>9.3f}A")
        print(f"  {'Random uniform':<30} {min(rmsds_uni):>9.3f}A  {np.mean(rmsds_uni):>9.3f}A")
        print(f"  {'Random model sample':<30} {min(rmsds_msamp):>9.3f}A  {np.mean(rmsds_msamp):>9.3f}A")

        # ── Diversity ─────────────────────────────────────────────────────────
        _, _, div_nerf = ensemble_diversity(ens_nerf, loop_start, loop_end)
        _, _, div_se3  = ensemble_diversity(ens_se3,  loop_start, loop_end)
        _, _, div_xyz  = ensemble_diversity(ens_xyz,  loop_start, loop_end)
        print(f"\n  Diversity -- NeRF={div_nerf:.3f}A  SE(3)={div_se3:.3f}A  XYZ={div_xyz:.3f}A")

        # ── Energy ────────────────────────────────────────────────────────────
        e_ideal     = ideal_energy(probs_phi, probs_psi)
        e_best_nerf = ens_nerf[best_nerf_idx][-2]
        e_best_se3  = ens_se3[best_se3_idx][-2]
        e_best_xyz  = ens_xyz[best_xyz_idx][-2]
        print(f"  Energy ideal={e_ideal:.2f}  "
              f"NeRF={e_best_nerf:.2f} (+{e_best_nerf-e_ideal:.2f})  "
              f"SE(3)={e_best_se3:.2f} (+{e_best_se3-e_ideal:.2f})  "
              f"XYZ={e_best_xyz:.2f} (+{e_best_xyz-e_ideal:.2f})")

        # ── Save best SE(3) as primary output ─────────────────────────────────
        save_ensemble(ens_se3, full_seq, loop_start, loop_end, CA_nat, name, out)
        save_pymol_comparison(ens_se3, best_se3_idx, full_seq, loop_start, loop_end,
                              N_nat, CA_nat, C_nat, O_nat, name, out)
        CA_best = ens_se3[best_se3_idx][1][loop_start:loop_end]
        per_res = np.sqrt(np.sum((CA_best - CA_loop_nat) ** 2, axis=1))
        visualize_loop(CA_best, CA_loop_nat, per_res, name, cdr3_seq,
                       rmsds_se3[best_se3_idx], out)

        results.append({
            'pdb_id':            meta['pdb_id'],
            'chain':             meta['chain'],
            'sequence':          cdr3_seq,
            'loop_length':       len(cdr3_seq),
            # NeRF
            'best_rmsd_nerf':    float(rmsds_nerf[best_nerf_idx]),
            'mean_rmsd_nerf':    float(np.mean(rmsds_nerf)),
            'all_rmsds_nerf':    [float(r) for r in rmsds_nerf],
            'best_rmsd_aln':     float(rmsds_aln[best_aln_idx]),
            'mean_rmsd_aln':     float(np.mean(rmsds_aln)),
            # SE(3)
            'best_rmsd_se3':     float(rmsds_se3[best_se3_idx]),
            'mean_rmsd_se3':     float(np.mean(rmsds_se3)),
            'all_rmsds_se3':     [float(r) for r in rmsds_se3],
            # XYZ
            'best_rmsd_xyz':     float(rmsds_xyz[best_xyz_idx]),
            'mean_rmsd_xyz':     float(np.mean(rmsds_xyz)),
            'all_rmsds_xyz':     [float(r) for r in rmsds_xyz],
            # energy
            'ideal_energy':      float(e_ideal),
            'best_energy_nerf':  float(e_best_nerf),
            'best_energy_se3':   float(e_best_se3),
            'best_energy_xyz':   float(e_best_xyz),
            'energy_gap_nerf':   float(e_best_nerf - e_ideal),
            'energy_gap_se3':    float(e_best_se3  - e_ideal),
            'energy_gap_xyz':    float(e_best_xyz  - e_ideal),
            # diversity
            'diversity_nerf':    float(div_nerf),
            'diversity_se3':     float(div_se3),
            'diversity_xyz':     float(div_xyz),
            # baselines
            'baseline_uniform_best': float(min(rmsds_uni)),
            'baseline_uniform_mean': float(np.mean(rmsds_uni)),
            'baseline_model_best':   float(min(rmsds_msamp)),
            'baseline_model_mean':   float(np.mean(rmsds_msamp)),
            # closure
            'best_closure_nerf': float(ens_nerf[best_nerf_idx][-1]),
            'best_closure_se3':  float(ens_se3[best_se3_idx][-1]),
            'best_closure_xyz':  float(ens_xyz[best_xyz_idx][-1]),
        })

    if results:
        n = len(results)
        print(f"\n{'='*60}\nSUMMARY ({n} loops)")
        print(f"\n  {'Metric':<28} {'NeRF':>9}  {'SE(3)':>9}  {'XYZ':>9}  {'Aligned':>9}")
        print(f"  {'-'*68}")
        for label, k_nerf, k_se3, k_xyz, k_aln in [
            ('Mean best RMSD',   'mean_rmsd_nerf', 'mean_rmsd_se3', 'mean_rmsd_xyz', 'mean_rmsd_aln'),
        ]:
            print(f"  {label:<28} "
                  f"{np.mean([r[k_nerf] for r in results]):>8.3f}A  "
                  f"{np.mean([r[k_se3]  for r in results]):>8.3f}A  "
                  f"{np.mean([r[k_xyz]  for r in results]):>8.3f}A  "
                  f"{np.mean([r[k_aln]  for r in results]):>8.3f}A")
        print(f"  {'Median best RMSD':<28} "
              f"{np.median([r['best_rmsd_nerf'] for r in results]):>8.3f}A  "
              f"{np.median([r['best_rmsd_se3']  for r in results]):>8.3f}A  "
              f"{np.median([r['best_rmsd_xyz']  for r in results]):>8.3f}A  "
              f"{np.median([r['best_rmsd_aln']  for r in results]):>8.3f}A")
        for thresh in [1, 2, 3]:
            nerf_k = sum(r['best_rmsd_nerf'] < thresh for r in results)
            se3_k  = sum(r['best_rmsd_se3']  < thresh for r in results)
            xyz_k  = sum(r['best_rmsd_xyz']  < thresh for r in results)
            aln_k  = sum(r['best_rmsd_aln']  < thresh for r in results)
            print(f"  {'< '+str(thresh)+'A':<28} "
                  f"{nerf_k:>8}/{n}  {se3_k:>8}/{n}  "
                  f"{xyz_k:>8}/{n}  {aln_k:>8}/{n}")

        with open(out / 'results.json', 'w') as f:
            json.dump(results, f, indent=2)
        plot_summary(results, out)

    return results


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--dataset",           default="/home/jtepperik/thesis/energy_model/scripts/data_processing/cdr3_dataset")
    p.add_argument("--output",            default="cdr3_test_results")
    p.add_argument("--n-structures",      type=int,   default=50)
    p.add_argument("--max-loops",         type=int,   default=5)
    p.add_argument("--n-steps",           type=int,   default=1000)
    # NeRF
    p.add_argument("--lr-energy",         type=float, default=0.05)
    p.add_argument("--lr-closure",        type=float, default=0.20)
    p.add_argument("--closure-weight",    type=float, default=25.0)
    # SE(3)
    p.add_argument("--lr-torsion",        type=float, default=0.025)
    p.add_argument("--lr-bond",           type=float, default=3.0)
    p.add_argument("--bond-weight",       type=float, default=5.0)
    p.add_argument("--position-scale",    type=float, default=5.0)
    p.add_argument("--n-frames",          type=int,   default=100)
    # XYZ
    p.add_argument("--xyz-lr",            type=float, default=0.05)
    p.add_argument("--xyz-bond-weight",   type=float, default=100.0)
    p.add_argument("--xyz-angle-weight",  type=float, default=50.0)
    args = p.parse_args()

    test_on_cdr3_dataset(
        dataset_dir      = args.dataset,
        output_dir       = args.output,
        n_structures     = args.n_structures,
        max_loops        = args.max_loops,
        n_steps          = args.n_steps,
        lr_energy        = args.lr_energy,
        lr_closure       = args.lr_closure,
        closure_weight   = args.closure_weight,
        lr_torsion       = args.lr_torsion,
        lr_bond          = args.lr_bond,
        bond_weight      = args.bond_weight,
        position_scale   = args.position_scale,
        n_frames         = args.n_frames,
        xyz_lr           = args.xyz_lr,
        xyz_bond_weight  = args.xyz_bond_weight,
        xyz_angle_weight = args.xyz_angle_weight,
    )