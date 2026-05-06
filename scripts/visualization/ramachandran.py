"""
plot_ramachandran_predictions.py

Visualise the model's learned joint (φ, ψ) distribution for every amino acid
as a grid of Ramachandran heatmaps — one subplot per residue type.

The model is queried with each amino acid placed in a neutral context
(all-alanine window by default, or no context / padding).  This shows what
the model has learned about each residue's backbone preferences independent
of neighbour identity.

Usage
─────
  python plot_ramachandran_predictions.py \\
      --checkpoint /path/to/checkpoint \\
      --output ramachandran_all_aa.png \\
      --context neutral        # 'neutral' (all-Ala) | 'none' (all-PAD)
      --style probability      # 'probability' | 'energy' (-log P)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

from utils import load_model, ModelRouter, _to_router


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

AA_ORDER = list('ACDEFGHIKLMNPQRSTVWY')   # 20 standard, alphabetical

AA_FULL_NAME = {
    'A':'Ala','C':'Cys','D':'Asp','E':'Glu','F':'Phe',
    'G':'Gly','H':'His','I':'Ile','K':'Lys','L':'Leu',
    'M':'Met','N':'Asn','P':'Pro','Q':'Gln','R':'Arg',
    'S':'Ser','T':'Thr','V':'Val','W':'Trp','Y':'Tyr',
}

AA_TO_IDX = {
            'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4,
            'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9,
            'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14,
            'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19
}
PAD_IDX     = 20
ALA_IDX     = AA_TO_IDX['A']
N_BINS      = 36
BIN_CENTRES = np.array([-180.0 + (k + 0.5) * 10.0 for k in range(N_BINS)])


# ─────────────────────────────────────────────────────────────────────────────
# Prediction
# ─────────────────────────────────────────────────────────────────────────────

def predict_aa(
    aa:       str,
    router:   ModelRouter,
    context:  str = 'neutral',
    max_context: int = 21,
) -> np.ndarray:
    """
    Query the model for amino acid `aa` and return the (N_BINS, N_BINS)
    joint probability table.

    Args:
        aa:          one-letter amino acid code
        router:      ModelRouter (uses specialist model if available)
        context:     'neutral' — surround with alanine
                     'none'    — all padding tokens
        max_context: model context window size

    Returns:
        probs: (N_BINS, N_BINS) joint probability array (sums to 1)
    """
    model, params = router.get(aa.upper())
    center = max_context // 2
    aa_idx = AA_TO_IDX.get(aa.upper(), PAD_IDX)

    if context == 'neutral':
        window = [ALA_IDX] * max_context
        window[center] = aa_idx
    else:   # 'none'
        window = [PAD_IDX] * max_context
        window[center] = aa_idx

    batch_res  = jnp.array(window)[None, :]
    batch_mask = jnp.ones((1, max_context), dtype=bool)

    logits = model.apply(
        {'params': params},
        batch_res, batch_mask,
        training=False,
        rngs={'dropout': jax.random.PRNGKey(0)},
    )
    p = np.array(jax.nn.softmax(logits[0]))
    return p.reshape(N_BINS, N_BINS)


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_all_amino_acids(
    router:      ModelRouter,
    output_path: str = 'ramachandran_all_aa.png',
    context:     str = 'neutral',
    style:       str = 'probability',
    max_context: int = 21,
    figsize:     tuple = (20, 16),
    dpi:         int  = 150,
):
    """
    Plot predicted (φ, ψ) distributions for all 20 amino acids in a 4×5 grid.

    Args:
        router:      ModelRouter
        output_path: output PNG path
        context:     'neutral' (all-Ala neighbours) or 'none' (padding)
        style:       'probability' — raw P(φ,ψ)
                     'energy'      — −log P(φ,ψ) (low = preferred)
        max_context: must match trained model
        figsize:     figure size in inches
        dpi:         output resolution
    """
    n_rows, n_cols = 4, 5

    # Collect all distributions first so we can set a shared colour scale
    all_probs = {}
    print("Querying model for all 20 amino acids...")
    for aa in AA_ORDER:
        all_probs[aa] = predict_aa(aa, router, context, max_context)
        print(f"  {aa} ({AA_FULL_NAME[aa]}) — max P = {all_probs[aa].max():.4f}")

    # Colour maps matching the classic Ramachandran plot style:
    # probability: white → light green → dark green
    # energy:      dark green → light green → white → yellow → red
    if style == 'probability':
        cmap = LinearSegmentedColormap.from_list(
            'rama_prob',
            ['#ffffff', '#c8e6c9', '#66bb6a', '#2e7d32'],
            N=256,
        )
        # Shared vmax = 95th percentile across all AAs to avoid outlier domination
        all_vals = np.concatenate([p.ravel() for p in all_probs.values()])
        vmin, vmax = 0.0, float(np.percentile(all_vals, 99))
        cbar_label = 'P(φ, ψ)'
    else:
        cmap = LinearSegmentedColormap.from_list(
            'rama_energy',
            ['#1b5e20', '#66bb6a', '#c8e6c9', '#ffffff', '#fff9c4', '#e53935'],
            N=256,
        )
        all_nll = np.concatenate([
            (-np.log(p + 1e-10)).ravel() for p in all_probs.values()
        ])
        vmin = 0.0
        vmax = float(np.percentile(all_nll, 97))
        cbar_label = '−log P(φ, ψ)  (energy)'

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=figsize,
        sharex=True, sharey=True,
    )
    fig.suptitle(
        f"Predicted Ramachandran distributions — {style}\n"
        f"Context: {context}  |  {N_BINS}×{N_BINS} bins (10° each)",
        fontsize=14, fontweight='bold', y=1.01,
    )

    for idx, aa in enumerate(AA_ORDER):
        ax   = axes[idx // n_cols, idx % n_cols]
        p    = all_probs[aa]
        data = p if style == 'probability' else -np.log(p + 1e-10)

        # imshow: rows = φ (outer axis in our joint table), cols = ψ
        # We want φ on x-axis and ψ on y-axis as in canonical Ramachandran plots,
        # so transpose and flip y so -180 is at bottom
        im = ax.imshow(
            data.T,                       # (psi, phi) after transpose
            origin='lower',               # ψ=-180 at bottom
            extent=[-180, 180, -180, 180],
            aspect='equal',
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            interpolation='bilinear',
        )

        # Reference lines at 0°
        ax.axhline(0, color='grey', lw=0.4, ls='--', alpha=0.5)
        ax.axvline(0, color='grey', lw=0.4, ls='--', alpha=0.5)

        # Annotate canonical secondary structure regions
        _annotate_ss_regions(ax)

        ax.set_title(
            f"{aa}  {AA_FULL_NAME[aa]}",
            fontsize=9, fontweight='bold', pad=3,
        )
        ax.set_xlim(-180, 180);  ax.set_ylim(-180, 180)
        ax.set_xticks([-135, -45, 45, 135])
        ax.set_yticks([-135, -45, 45, 135])
        ax.tick_params(labelsize=6)

        # Axis labels only on edges
        if idx // n_cols == n_rows - 1:
            ax.set_xlabel('φ (°)', fontsize=8)
        if idx % n_cols == 0:
            ax.set_ylabel('ψ (°)', fontsize=8)

    # Shared colourbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    sm = plt.cm.ScalarMappable(cmap=cmap,
                                norm=mcolors.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    fig.colorbar(sm, cax=cbar_ax, label=cbar_label)

    plt.tight_layout(rect=[0, 0, 0.91, 1.0])
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"\n✓  Saved → {output_path}")


def _annotate_ss_regions(ax):
    """
    Add subtle marker dots at canonical secondary structure positions.
    Matches the reference diagram annotation style.
    """
    regions = {
        'α':  (-57,  -47, '#1b5e20'),   # right-handed α-helix
        'β':  (-119, 113, '#1b5e20'),   # antiparallel β-sheet
        'Lα': ( 57,   47, '#4a148c'),   # left-handed α-helix
    }
    for label, (phi, psi, color) in regions.items():
        ax.plot(phi, psi, marker='s', markersize=3,
                color=color, alpha=0.6, zorder=5)


# ─────────────────────────────────────────────────────────────────────────────
# Comparison plot: side-by-side context modes or model comparison
# ─────────────────────────────────────────────────────────────────────────────

def plot_single_aa(
    aa:          str,
    router:      ModelRouter,
    output_path: str  = None,
    context:     str  = 'neutral',
    max_context: int  = 21,
    dpi:         int  = 150,
) -> np.ndarray:
    """
    Plot a single large Ramachandran heatmap for one amino acid.
    Also returns the (N_BINS, N_BINS) probability array.
    """
    p = predict_aa(aa, router, context, max_context)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    for ax, (data, title, cmap_name) in zip(axes, [
        (p,                  'Probability P(φ,ψ)',
         LinearSegmentedColormap.from_list('p', ['#fff','#c8e6c9','#2e7d32'], N=256)),
        (-np.log(p + 1e-10), 'Energy −log P(φ,ψ)',
         LinearSegmentedColormap.from_list('e', ['#1b5e20','#c8e6c9','#fff','#ffeb3b','#e53935'], N=256)),
    ]):
        vmax = float(np.percentile(data, 99))
        im = ax.imshow(
            data.T, origin='lower',
            extent=[-180, 180, -180, 180],
            aspect='equal', cmap=cmap_name,
            vmin=0, vmax=vmax, interpolation='bilinear',
        )
        ax.axhline(0, color='grey', lw=0.5, ls='--', alpha=0.5)
        ax.axvline(0, color='grey', lw=0.5, ls='--', alpha=0.5)
        _annotate_ss_regions(ax)
        ax.set_xlabel('φ (°)');  ax.set_ylabel('ψ (°)')
        ax.set_title(title, fontsize=10)
        ax.set_xticks(range(-180, 181, 60))
        ax.set_yticks(range(-180, 181, 60))
        plt.colorbar(im, ax=ax, shrink=0.85)

    fig.suptitle(
        f"{aa}  {AA_FULL_NAME.get(aa.upper(), '')}  |  context={context}",
        fontsize=12, fontweight='bold',
    )
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        print(f"✓  Saved → {output_path}")
    else:
        plt.show()

    return p


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Visualise model Ramachandran predictions for all amino acids'
    )
    parser.add_argument('--checkpoint', required=True,
                        help='Checkpoint directory containing config.json')
    parser.add_argument('--output',  default='ramachandran_all_aa.png')
    parser.add_argument('--context', default='neutral',
                        choices=['neutral', 'none'],
                        help='neutral=all-Ala context  none=all-PAD')
    parser.add_argument('--style',   default='probability',
                        choices=['probability', 'energy'])
    parser.add_argument('--single',  default=None,
                        help='Plot a single amino acid (e.g. --single G)')
    parser.add_argument('--dpi',     type=int, default=150)
    args = parser.parse_args()

    router = load_model(args.checkpoint)

    if args.single:
        out = args.output.replace('.png', f'_{args.single.upper()}.png')
        plot_single_aa(
            args.single.upper(), router,
            output_path=out,
            context=args.context,
        )
    else:
        plot_all_amino_acids(
            router,
            output_path=args.output,
            context=args.context,
            style=args.style,
            dpi=args.dpi,
        )