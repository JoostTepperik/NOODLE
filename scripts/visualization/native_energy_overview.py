"""
native_energy_overview.py  (fully self-contained)
──────────────────────────────────────────────────
Per-residue energy breakdown for native CDR3 loop structures.

No imports from the thesis codebase — everything is inlined here so the
bin convention is explicit and consistent throughout.

Bin convention (matches training data):
  72 bins spanning [-180°, 180°)
  bin k  →  centre = -180 + (k + 0.5) * 5°
  bin 0  →  -177.5°,  bin 35 → -2.5°,  bin 36 → +2.5°,  bin 71 → +177.5°

Usage:
    python native_energy_overview.py \
        --dataset  /path/to/cdr3_dataset \
        --output   native_energy_overview \
        --max-loops 5
"""

import sys
sys.path.append('/home/jtepperik/thesis/energy_model/scripts')

import argparse
import json
import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp

from models.full_model import TorsionPredictor   # adjust if path differs

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

N_BINS      = 72
BIN_WIDTH   = 360.0 / N_BINS                            # 5°
BIN_CENTRES = np.array([-180.0 + (k + 0.5) * BIN_WIDTH for k in range(N_BINS)])

MAX_CONTEXT = 7
CONTEXT_RAD = MAX_CONTEXT // 2    # 3

AA_TO_IDX = {
    'A':0,'R':1,'N':2,'D':3,'C':4,'Q':5,'E':6,'G':7,'H':8,'I':9,
    'L':10,'K':11,'M':12,'F':13,'P':14,'S':15,'T':16,'W':17,'Y':18,'V':19,
}
PAD_IDX = 20

THREE_TO_ONE = {
    'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C',
    'GLN':'Q','GLU':'E','GLY':'G','HIS':'H','ILE':'I',
    'LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P',
    'SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V',
}


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_model():
    model = TorsionPredictor(
        max_context=MAX_CONTEXT, embed_dim=64, hidden_dim=768,
        n_layers=3, dropout_rate=0.1,
        prediction_type='binned', n_bins=N_BINS,
    )
    ckpt_path = '/home/jtepperik/thesis/energy_model/scripts/training/outputs/feedforward_binned_19448143/checkpoints/best_10'
    checkpointer = ocp.StandardCheckpointer()
    restored = checkpointer.restore(ckpt_path)
    params = restored['params']
    stds = [float(np.std(l)) for l in jax.tree_util.tree_leaves(params)]
    print(f"✓ Model loaded  (weight std: {min(stds):.4f}–{max(stds):.4f})")
    return model, params


# ─────────────────────────────────────────────────────────────────────────────
# Angle prediction
# ─────────────────────────────────────────────────────────────────────────────

def predict_distributions(model, params, sequence: str):
    """
    Sliding-window prediction for each residue in `sequence`.

    Returns:
        probs_phi, probs_psi — lists of (72,) np.ndarray in [-180, 180) bin order
    """
    encoded  = np.array([AA_TO_IDX.get(aa, PAD_IDX) for aa in sequence.upper()])
    seq_len  = len(encoded)
    probs_phi, probs_psi = [], []

    for i in range(seq_len):
        window = []
        for pos in range(i - CONTEXT_RAD, i + CONTEXT_RAD + 1):
            window.append(int(encoded[pos]) if 0 <= pos < seq_len else PAD_IDX)

        batch_res  = jnp.array(window)[None, :]
        batch_mask = jnp.ones((1, MAX_CONTEXT), dtype=bool)

        logits_phi, logits_psi = model.apply(
            {'params': params}, batch_res, batch_mask,
            training=False, rngs={'dropout': jax.random.PRNGKey(0)},
        )

        # logits: (1, 72) — bins in [-180, 180) matching training convention
        probs_phi.append(np.array(jax.nn.softmax(logits_phi[0])))
        probs_psi.append(np.array(jax.nn.softmax(logits_psi[0])))

    return probs_phi, probs_psi


# ─────────────────────────────────────────────────────────────────────────────
# PDB loader
# ─────────────────────────────────────────────────────────────────────────────

def load_backbone(pdb_file):
    N_l, CA_l, C_l, seq = [], [], [], []
    with open(pdb_file) as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            if line[16] not in (' ', 'A'):   # skip alt conformations
                continue
            atom = line[12:16].strip()
            res  = line[17:20].strip()
            x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
            if atom == "CA": seq.append(THREE_TO_ONE.get(res, 'X'))
            if   atom == "N":  N_l .append([x, y, z])
            elif atom == "CA": CA_l.append([x, y, z])
            elif atom == "C":  C_l .append([x, y, z])
    return ''.join(seq), np.array(N_l), np.array(CA_l), np.array(C_l)


# ─────────────────────────────────────────────────────────────────────────────
# Dihedral / coords → angles
# ─────────────────────────────────────────────────────────────────────────────

def _dihedral(p1, p2, p3, p4):
    """IUPAC dihedral angle in degrees, range (-180, 180]."""
    b0 = p1 - p2
    b1 = p3 - p2
    b2 = p4 - p3
    b1_hat = b1 / (np.linalg.norm(b1) + 1e-10)
    v = b0 - np.dot(b0, b1_hat) * b1_hat
    w = b2 - np.dot(b2, b1_hat) * b1_hat
    return np.degrees(np.arctan2(np.dot(np.cross(b1_hat, v), w), np.dot(v, w)))


def coords_to_angles(N, CA, C):
    """phi, psi in degrees; phi[0] and psi[-1] are 0 (undefined at termini)."""
    n   = len(N)
    phi = np.zeros(n)
    psi = np.zeros(n)
    for i in range(n):
        if i > 0:
            phi[i] = _dihedral(C[i-1], N[i], CA[i], C[i])
        if i < n - 1:
            psi[i] = _dihedral(N[i], CA[i], C[i], N[i+1])
    return phi, psi


# ─────────────────────────────────────────────────────────────────────────────
# Energy helpers
# ─────────────────────────────────────────────────────────────────────────────

def _interp_prob(angle_deg: float, probs: np.ndarray) -> float:
    """
    Linearly interpolate probability for angle_deg.
    probs is indexed by BIN_CENTRES in [-180, 180) order.
    """
    a      = ((angle_deg + 180.0) % 360.0) - 180.0   # normalise to [-180,180)
    bin_f  = (a + 180.0) / BIN_WIDTH                  # continuous bin index
    idx_lo = int(bin_f) % N_BINS
    idx_hi = (idx_lo + 1) % N_BINS
    w      = bin_f - int(bin_f)
    return float((1.0 - w) * probs[idx_lo] + w * probs[idx_hi])


def per_residue_energy(phi_deg, psi_deg, probs_phi, probs_psi):
    rows = []
    for i in range(len(phi_deg)):
        e_phi = -math.log(_interp_prob(phi_deg[i], probs_phi[i]) + 1e-10) \
                if i < len(probs_phi) else float('nan')
        e_psi = -math.log(_interp_prob(psi_deg[i], probs_psi[i]) + 1e-10) \
                if i < len(probs_psi) else float('nan')
        e_tot = sum(v for v in [e_phi, e_psi] if not math.isnan(v))
        rows.append({
            'idx': i, 'phi': float(phi_deg[i]), 'psi': float(psi_deg[i]),
            'e_phi': e_phi, 'e_psi': e_psi, 'e_total': e_tot,
        })
    return rows


def ideal_energy(probs_phi, probs_psi):
    e = 0.0
    for pp, ps in zip(probs_phi, probs_psi):
        e -= math.log(float(np.max(pp)) + 1e-10)
        e -= math.log(float(np.max(ps)) + 1e-10)
    return e


# ─────────────────────────────────────────────────────────────────────────────
# Console table
# ─────────────────────────────────────────────────────────────────────────────

def print_energy_table(rows, loop_seq, name, e_ideal):
    print(f"\n  {'─'*68}")
    print(f"  Energy overview: {name}  (loop: {loop_seq})")
    print(f"  {'─'*68}")
    print(f"  {'#':>3}  {'AA':>3}  {'φ (°)':>8}  {'ψ (°)':>8}  "
          f"{'E_φ':>8}  {'E_ψ':>8}  {'E_tot':>8}")
    print(f"  {'─'*68}")
    e_sum = 0.0
    for r in rows:
        aa = loop_seq[r['idx']] if r['idx'] < len(loop_seq) else '?'
        print(f"  {r['idx']+1:>3}  {aa:>3}  {r['phi']:>8.1f}  {r['psi']:>8.1f}  "
              f"{r['e_phi']:>8.3f}  {r['e_psi']:>8.3f}  {r['e_total']:>8.3f}")
        e_sum += r['e_total']
    print(f"  {'─'*68}")
    print(f"  {'Total':>38}  {'':>8}  {'':>8}  {e_sum:>8.3f}")
    print(f"  {'Ideal (argmax)':>38}  {'':>8}  {'':>8}  {e_ideal:>8.3f}")
    print(f"  {'Gap':>38}  {'':>8}  {'':>8}  {e_sum - e_ideal:>8.3f}")
    print(f"  {'─'*68}")


# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_energy_heatmap(rows, loop_seq, name, e_ideal, out_path):
    n   = len(rows)
    mat = np.array([[r['e_phi'] for r in rows],
                    [r['e_psi'] for r in rows]], dtype=float)
    cmap = LinearSegmentedColormap.from_list(
        'energy', ['#1a6faf', '#74c476', '#fed976', '#e31a1c'], N=256)

    fig, axes = plt.subplots(2, 1, figsize=(max(8, n * 0.7 + 2), 5),
                             gridspec_kw={'height_ratios': [3, 1]})
    fig.suptitle(f"Per-residue energy: {name}  |  loop: {loop_seq}",
                 fontsize=11, fontweight='bold', y=1.01)

    ax   = axes[0]
    vmax = np.nanpercentile(mat, 95)
    im   = ax.imshow(mat, aspect='auto', cmap=cmap, vmin=0, vmax=vmax,
                     interpolation='nearest')
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['φ  (phi)', 'ψ  (psi)'], fontsize=9)
    ax.set_xticks(range(n))
    ax.set_xticklabels(
        [f"{loop_seq[i]}\n{i+1}" if i < len(loop_seq) else str(i+1)
         for i in range(n)], fontsize=8)
    ax.set_xlabel('Residue', fontsize=9)
    for ri in range(2):
        for ci in range(n):
            val = mat[ri, ci]
            if not math.isnan(val):
                tc = 'white' if val > vmax * 0.6 else 'black'
                ax.text(ci, ri, f"{val:.2f}", ha='center', va='center',
                        fontsize=7, color=tc, fontweight='bold')
    plt.colorbar(im, ax=ax, label='−log p', shrink=0.8, pad=0.02)

    ax2       = axes[1]
    e_phi_arr = [r['e_phi']   for r in rows]
    e_psi_arr = [r['e_psi']   for r in rows]
    totals    = [r['e_total'] for r in rows]
    x = np.arange(n)
    ax2.bar(x, e_phi_arr, color='#4292c6', label='φ', width=0.4)
    ax2.bar(x, e_psi_arr, bottom=e_phi_arr, color='#ef6548', label='ψ', width=0.4)
    ax2.axhline(sum(totals) / n, color='black', linestyle='--', linewidth=0.8,
                label=f'mean {sum(totals)/n:.2f}')
    ax2.set_xticks(x)
    ax2.set_xticklabels([str(i+1) for i in range(n)], fontsize=8)
    ax2.set_ylabel('E_total', fontsize=8)
    ax2.legend(fontsize=7, ncol=3, loc='upper right')

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"    Heatmap → {Path(out_path).name}")


def plot_per_residue_distributions(rows, loop_seq, probs_phi, probs_psi,
                                   name, out_path):
    n         = len(rows)
    cols      = min(n, 5)
    rows_grid = math.ceil(n / cols)

    fig = plt.figure(figsize=(cols * 4.0, rows_grid * 3.8))
    fig.suptitle(f"Distribution + native angle: {name}  |  {loop_seq}",
                 fontsize=11, fontweight='bold')

    for idx, r in enumerate(rows):
        aa = loop_seq[r['idx']] if r['idx'] < len(loop_seq) else '?'

        # ── phi ──────────────────────────────────────────────────────────────
        ax = fig.add_subplot(rows_grid * 2, cols, idx + 1)
        if r['idx'] < len(probs_phi):
            p = np.array(probs_phi[r['idx']])
            ax.fill_between(BIN_CENTRES, p, alpha=0.35, color='#4292c6')
            ax.plot(BIN_CENTRES, p, color='#4292c6', linewidth=1.0)
            ax.axvline(r['phi'], color='#e31a1c', linewidth=1.8,
                       label=f"φ={r['phi']:.0f}°")
        ax.set_xlim(-180, 180)
        ax.set_xticks([-180, -90, 0, 90, 180])
        ax.tick_params(labelsize=6)
        ax.set_title(f"Res {r['idx']+1} ({aa})  E_φ={r['e_phi']:.2f}",
                     fontsize=7.5, pad=2)
        ax.set_ylabel('p(φ)', fontsize=6)
        if idx == 0:
            ax.legend(fontsize=6, loc='upper left')

        # ── psi ──────────────────────────────────────────────────────────────
        ax2 = fig.add_subplot(rows_grid * 2, cols, idx + 1 + cols * rows_grid)
        if r['idx'] < len(probs_psi):
            p = np.array(probs_psi[r['idx']])
            ax2.fill_between(BIN_CENTRES, p, alpha=0.35, color='#ef6548')
            ax2.plot(BIN_CENTRES, p, color='#ef6548', linewidth=1.0)
            ax2.axvline(r['psi'], color='#2ca25f', linewidth=1.8,
                        label=f"ψ={r['psi']:.0f}°")
        ax2.set_xlim(-180, 180)
        ax2.set_xticks([-180, -90, 0, 90, 180])
        ax2.tick_params(labelsize=6)
        ax2.set_title(f"Res {r['idx']+1} ({aa})  E_ψ={r['e_psi']:.2f}",
                      fontsize=7.5, pad=2)
        ax2.set_ylabel('p(ψ)', fontsize=6)
        ax2.set_xlabel('angle (°)', fontsize=6)
        if idx == 0:
            ax2.legend(fontsize=6, loc='upper left')

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"    Distributions → {Path(out_path).name}")


def plot_ramachandran_energy(rows, loop_seq, name, out_path):
    phi_vals  = [r['phi']     for r in rows]
    psi_vals  = [r['psi']     for r in rows]
    e_vals    = [r['e_total'] for r in rows]
    aa_labels = [f"{loop_seq[r['idx']]}{r['idx']+1}" if r['idx'] < len(loop_seq)
                 else str(r['idx']+1) for r in rows]

    cmap = LinearSegmentedColormap.from_list(
        'energy', ['#1a9850', '#ffffbf', '#d73027'], N=256)

    fig, ax = plt.subplots(figsize=(6, 5.5))
    sc = ax.scatter(phi_vals, psi_vals, c=e_vals, cmap=cmap, s=80,
                    edgecolors='black', linewidths=0.5, zorder=3)
    for phi, psi, lbl in zip(phi_vals, psi_vals, aa_labels):
        ax.annotate(lbl, (phi, psi), textcoords='offset points',
                    xytext=(4, 4), fontsize=7)
    plt.colorbar(sc, ax=ax, label='E_total (−log p)', shrink=0.85)
    ax.axhline(0, color='grey', lw=0.5, ls='--', alpha=0.4)
    ax.axvline(0, color='grey', lw=0.5, ls='--', alpha=0.4)
    ax.set_xlim(-180, 180); ax.set_ylim(-180, 180)
    ax.set_xlabel('φ (degrees)', fontsize=10)
    ax.set_ylabel('ψ (degrees)', fontsize=10)
    ax.set_title(f"Ramachandran (coloured by energy)\n{name}  |  {loop_seq}",
                 fontsize=10)
    ax.set_xticks(range(-180, 181, 60))
    ax.set_yticks(range(-180, 181, 60))
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"    Ramachandran → {Path(out_path).name}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run(dataset_dir, output_dir, model, params, max_loops):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    with open(Path(dataset_dir) / "cdr3_dataset.json") as f:
        dataset = json.load(f)
    if max_loops:
        dataset = dataset[:max_loops]

    all_results = []

    for i, meta in enumerate(dataset, 1):
        name       = f"{meta['pdb_id']}_{meta['chain']}"
        full_seq   = meta['full_sequence']
        cdr3_seq   = meta['cdr3_sequence']
        loop_start = meta['loop_start']
        loop_end   = meta['loop_end']

        print(f"\n{'='*60}")
        print(f"{i}/{len(dataset)}  {name}  loop={cdr3_seq}")

        pdb_file = Path(meta['pdb_file'])
        if not pdb_file.is_absolute():
            pdb_file = Path(dataset_dir) / pdb_file.name
        if not pdb_file.exists():
            print(f"  PDB not found: {pdb_file}"); continue

        seq, N_nat, CA_nat, C_nat = load_backbone(pdb_file)
        if seq != full_seq:
            print(f"  Sequence mismatch — skipping"); continue

        print(f"  Predicting distributions for loop: {cdr3_seq} ...")
        probs_phi, probs_psi = predict_distributions(model, params, cdr3_seq)

        # Include one flanking residue on each side so the terminal loop
        # residues get real phi/psi values rather than 0
        start  = max(0, loop_start - 1)
        end    = min(len(seq), loop_end + 1)
        phi_full, psi_full = coords_to_angles(
            N_nat[start:end], CA_nat[start:end], C_nat[start:end],
        )
        offset = loop_start - start
        n_loop = loop_end - loop_start
        phi_n  = phi_full[offset : offset + n_loop]
        psi_n  = psi_full[offset : offset + n_loop]

        rows    = per_residue_energy(phi_n, psi_n, probs_phi, probs_psi)
        e_ideal = ideal_energy(probs_phi, probs_psi)
        e_total = sum(r['e_total'] for r in rows)

        print_energy_table(rows, cdr3_seq, name, e_ideal)

        loop_out = out / name
        loop_out.mkdir(exist_ok=True)

        plot_energy_heatmap(rows, cdr3_seq, name, e_ideal,
                            loop_out / f"heatmap_{name}.png")
        plot_per_residue_distributions(rows, cdr3_seq, probs_phi, probs_psi,
                                       name, loop_out / f"distributions_{name}.png")
        plot_ramachandran_energy(rows, cdr3_seq, name,
                                 loop_out / f"ramachandran_{name}.png")

        all_results.append({
            'name':     name,
            'sequence': cdr3_seq,
            'e_total':  e_total,
            'e_ideal':  e_ideal,
            'e_gap':    e_total - e_ideal,
            'residues': rows,
        })

    # ── Cross-loop summary ────────────────────────────────────────────────────
    if len(all_results) > 1:
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
        fig.suptitle("Cross-loop native energy summary", fontsize=12, fontweight='bold')

        names  = [r['name']    for r in all_results]
        totals = [r['e_total'] for r in all_results]
        ideals = [r['e_ideal'] for r in all_results]
        gaps   = [r['e_gap']   for r in all_results]
        x = np.arange(len(names)); w = 0.35

        axes[0].bar(x - w/2, totals, w, label='Native E', color='#4292c6', alpha=0.85)
        axes[0].bar(x + w/2, ideals, w, label='Ideal E',  color='#74c476', alpha=0.85)
        axes[0].set_xticks(x)
        axes[0].set_xticklabels([n.replace('_', '\n') for n in names], fontsize=7)
        axes[0].set_ylabel('Total energy (−log p)')
        axes[0].set_title('Native vs ideal energy per loop')
        axes[0].legend()

        axes[1].bar(x, gaps,
                    color=['#e31a1c' if g > 0 else '#1a9850' for g in gaps],
                    alpha=0.85)
        axes[1].axhline(0, color='black', linewidth=0.8)
        axes[1].set_xticks(x)
        axes[1].set_xticklabels([n.replace('_', '\n') for n in names], fontsize=7)
        axes[1].set_ylabel('Gap (native − ideal)')
        axes[1].set_title('Energy gap per loop\n(red = above ideal, green = below)')

        plt.tight_layout()
        fig.savefig(out / 'summary_energy.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"\n  Summary → summary_energy.png")

    with open(out / 'native_energy.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  JSON → native_energy.json")
    return all_results


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset",   default="/home/jtepperik/thesis/energy_model/scripts/visualization/cdr3_test_results/ensemble_7na5_b_E")
    p.add_argument("--output",    default="native_energy_overview")
    p.add_argument("--max-loops", type=int, default=None)
    args = p.parse_args()

    model, params = load_model()
    run(args.dataset, args.output, model, params, args.max_loops)