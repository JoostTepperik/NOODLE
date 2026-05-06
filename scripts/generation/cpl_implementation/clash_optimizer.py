"""
clash_optimizer.py

Standalone clash minimization for CDR3 loops via torsion-space gradient descent.

Goal
----
Answer the question: can gradient-based torsion optimization find low-clash
CDR3 loop conformations without any closure constraint or neural energy guidance?

Method
------
Multi-start Adam optimization in torsion space (phi, psi for each loop residue).
Clash energy is a differentiable softplus potential computed against the
precomputed framework atom set.  Gradient flows through the NeRF coordinate
chain via PyTorch autograd.

SOTA optimization choices:
  - Adam with cosine annealing LR schedule
  - Multiple random restarts (--n-restarts) with best-of-N selection
  - Ramachandran-biased initialization (more restarts land in valid regions)
  - Warm restarts: perturb best solution found so far, re-optimize
  - Gradient clipping to prevent exploding gradients through deep NeRF chain
  - Early stopping when clash reaches near-zero

No KIC closure is applied — this is intentional.  The output PDBs will not
be geometrically closed.  The purpose is purely to characterize the clash
energy landscape in torsion space.

Output
------
  <output_dir>/
    <pdb_id>_<chain>/
      best_clash.pdb         — lowest-clash conformation found
      all_restarts.csv       — per-restart final clash scores
      optimization_log.json  — full per-loop results
    summary.png              — clash score distributions across loops
    results.json             — per-loop summary stats

Usage
-----
  python clash_optimizer.py \\
    --dataset /path/to/cdr3_dataset \\
    --output clash_results \\
    --n-restarts 20 \\
    --n-steps 500 \\
    --clash
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from scipy.spatial import KDTree

from nerf import build_loop, BOND_LENGTHS, BOND_ANGLES_RAD
from utils import (
    VDW_RADII,
    load_cdr3_native,
    extract_framework_atoms,
    write_pdb_atoms,
    ONE_TO_THREE,
)


# ─────────────────────────────────────────────────────────────────────────────
# Ramachandran-biased initialization
# ─────────────────────────────────────────────────────────────────────────────

# Ideal peptide C-N bond length in Ångströms (used for closure penalty)
_IDEAL_CN_BOND = 1.32868

_RAMA_REGIONS = [
    (-60.0,  -45.0, 20.0, 0.35),   # alpha-helix
    (-120.0, 130.0, 20.0, 0.40),   # beta-sheet
    (60.0,   45.0,  20.0, 0.05),   # left-handed helix
    (0.0,    0.0,  180.0, 0.20),   # flat / other
]
_RAMA_WEIGHTS = np.array([r[3] for r in _RAMA_REGIONS])
_RAMA_WEIGHTS /= _RAMA_WEIGHTS.sum()


def _rama_init(n_loop: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """Sample phi/psi from Ramachandran-weighted Gaussian regions."""
    phi = np.empty(n_loop)
    psi = np.empty(n_loop)
    regions = rng.choice(len(_RAMA_REGIONS), size=n_loop, p=_RAMA_WEIGHTS)
    for i, ri in enumerate(regions):
        mu_phi, mu_psi, std, _ = _RAMA_REGIONS[ri]
        phi[i] = rng.normal(np.radians(mu_phi), np.radians(std))
        psi[i] = rng.normal(np.radians(mu_psi), np.radians(std))
    return phi, psi


# ─────────────────────────────────────────────────────────────────────────────
# Differentiable NeRF backbone (single structure, PyTorch)
# ─────────────────────────────────────────────────────────────────────────────

def _nerf_place(
    a: torch.Tensor, b: torch.Tensor, c: torch.Tensor,
    bl: torch.Tensor, ba: torch.Tensor, tor: torch.Tensor,
) -> torch.Tensor:
    """Place one atom via NeRF formula. All inputs are (3,) except tor scalar."""
    bc   = c - b
    bc_n = bc / (torch.norm(bc) + 1e-8)
    nabc = torch.linalg.cross(b - a, bc)
    nabc = nabc / (torch.norm(nabc) + 1e-8)
    m    = torch.linalg.cross(nabc, bc_n)
    d    = torch.stack([
        -torch.cos(ba),
         torch.sin(ba) * torch.cos(tor),
         torch.sin(ba) * torch.sin(tor),
    ]) * bl
    return c + torch.stack([bc_n, m, nabc], dim=1) @ d


def build_loop_torch(
    phi_v:   torch.Tensor,   # (n,) free variable
    psi_v:   torch.Tensor,   # (n,) free variable
    prev_N:  np.ndarray,
    prev_CA: np.ndarray,
    prev_C:  np.ndarray,
    psi_prev: float,
) -> torch.Tensor:            # (n*3, 3)  N/CA/C interleaved
    """Build loop backbone as differentiable torch tensor."""
    n = phi_v.shape[0]

    BL_CN  = torch.tensor(BOND_LENGTHS['C_N'],      dtype=torch.float32)
    BL_NCA = torch.tensor(BOND_LENGTHS['N_CA'],     dtype=torch.float32)
    BL_CAC = torch.tensor(BOND_LENGTHS['CA_C'],     dtype=torch.float32)
    BA_CCN = torch.tensor(BOND_ANGLES_RAD['CA_C_N'],dtype=torch.float32)
    BA_CNC = torch.tensor(BOND_ANGLES_RAD['C_N_CA'],dtype=torch.float32)
    BA_NCC = torch.tensor(BOND_ANGLES_RAD['N_CA_C'],dtype=torch.float32)
    OMEGA  = torch.tensor(math.pi, dtype=torch.float32)
    PSI_P  = torch.tensor(psi_prev, dtype=torch.float32)

    a3 = torch.tensor(prev_N,  dtype=torch.float32)
    a2 = torch.tensor(prev_CA, dtype=torch.float32)
    a1 = torch.tensor(prev_C,  dtype=torch.float32)

    atoms = torch.zeros(n * 3, 3)
    for i in range(n):
        Ni  = _nerf_place(a3, a2, a1, BL_CN,  BA_CCN,
                          psi_v[i-1] if i > 0 else PSI_P)
        CAi = _nerf_place(a2, a1, Ni,  BL_NCA, BA_CNC, OMEGA)
        Ci  = _nerf_place(a1, Ni, CAi, BL_CAC, BA_NCC, phi_v[i])
        atoms[i*3]   = Ni
        atoms[i*3+1] = CAi
        atoms[i*3+2] = Ci
        a3, a2, a1 = Ni, CAi, Ci

    return atoms


# ─────────────────────────────────────────────────────────────────────────────
# Clash energy
# ─────────────────────────────────────────────────────────────────────────────

def softplus_clash(
    atoms:    torch.Tensor,   # (n_loop_atoms, 3)
    loop_r:   torch.Tensor,   # (n_loop_atoms,)
    fw_t:     torch.Tensor,   # (n_fw, 3)
    fw_r:     torch.Tensor,   # (n_fw,)
    softness: float = 0.8,
    k:        float = 1.0,
) -> torch.Tensor:
    """
    Differentiable softplus clash energy.

        E = sum_{pairs} k * log(1 + exp(d_min - dist))

    where d_min = softness * (r_loop + r_fw).
    Smooth approximation to max(0, d_min - dist).
    """
    diff    = atoms[:, None, :] - fw_t[None, :, :]    # (nL, nFW, 3)
    dists   = torch.norm(diff, dim=2) + 1e-8           # (nL, nFW)
    d_min   = softness * (loop_r[:, None] + fw_r[None, :])
    overlap = d_min - dists
    return k * torch.log1p(torch.exp(torch.clamp(overlap, -20, 20))).sum()


def count_hard_clashes(
    N: np.ndarray, CA: np.ndarray, C: np.ndarray,
    fw_coords: np.ndarray, fw_radii: np.ndarray,
    softness: float = 0.8,
) -> int:
    """Count atom pairs with actual VdW overlap (for reporting)."""
    n = len(N)
    atoms = np.empty((n*3, 3), dtype=np.float32)
    atoms[0::3] = N; atoms[1::3] = CA; atoms[2::3] = C
    loop_r = np.array([VDW_RADII['N'], VDW_RADII['CA'], VDW_RADII['C']] * n,
                      dtype=np.float32)
    diff   = atoms[:, None, :] - fw_coords[None, :, :]
    dists  = np.linalg.norm(diff, axis=2) + 1e-8
    d_min  = softness * (loop_r[:, None] + fw_radii[None, :])
    return int((dists < d_min).sum())


# ─────────────────────────────────────────────────────────────────────────────
# Single-run optimizer
# ─────────────────────────────────────────────────────────────────────────────

def optimize_clash_single(
    phi_init:    np.ndarray,
    psi_init:    np.ndarray,
    prev_N:      np.ndarray,
    prev_CA:     np.ndarray,
    prev_C:      np.ndarray,
    psi_prev:    float,
    fw_t:        torch.Tensor,
    fw_r:        torch.Tensor,
    loop_r:      torch.Tensor,
    target_N_t:  torch.Tensor = None,
    k_close:     float = 10.0,
    n_steps:     int   = 500,
    lr:          float = 0.05,
    lr_min:      float = 1e-4,
    grad_clip:   float = 1.0,
    softness:    float = 0.8,
    k:           float = 1.0,
    early_stop:  float = 0.01,
) -> Tuple[np.ndarray, np.ndarray, float, list]:
    """
    Single Adam + cosine LR optimization run.

    Returns:
        phi_best, psi_best, best_clash, loss_history
    """
    phi_v = torch.tensor(phi_init, dtype=torch.float32, requires_grad=True)
    psi_v = torch.tensor(psi_init, dtype=torch.float32, requires_grad=True)

    opt = torch.optim.Adam([phi_v, psi_v], lr=lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=n_steps, eta_min=lr_min
    )

    best_loss = float('inf')
    best_phi  = phi_init.copy()
    best_psi  = psi_init.copy()
    history   = []

    for step in range(n_steps):
        opt.zero_grad()

        atoms = build_loop_torch(phi_v, psi_v, prev_N, prev_CA, prev_C, psi_prev)
        loss  = softplus_clash(atoms, loop_r, fw_t, fw_r, softness=softness, k=k)

        if target_N_t is not None:
            closure_loss = k_close * (torch.norm(atoms[-1] - target_N_t) - _IDEAL_CN_BOND) ** 2
            loss = loss + closure_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_([phi_v, psi_v], grad_clip)
        opt.step()
        sch.step()

        val = loss.item()
        history.append(val)
        if val < best_loss:
            best_loss = val
            best_phi  = phi_v.detach().numpy().copy()
            best_psi  = psi_v.detach().numpy().copy()

        if best_loss < early_stop:
            break

    return best_phi, best_psi, best_loss, history


# ─────────────────────────────────────────────────────────────────────────────
# Multi-start optimizer with warm restarts
# ─────────────────────────────────────────────────────────────────────────────

def optimize_clash_multistart(
    n_loop:    int,
    prev_N:    np.ndarray,
    prev_CA:   np.ndarray,
    prev_C:    np.ndarray,
    psi_prev:  float,
    fw_coords: np.ndarray,
    fw_radii:  np.ndarray,
    target_N:  np.ndarray = None,
    k_close:   float = 10.0,
    n_restarts:     int   = 20,
    n_warm_restarts: int  = 5,
    n_steps:        int   = 500,
    n_steps_warm:   int   = 200,
    lr:             float = 0.05,
    grad_clip:      float = 1.0,
    softness:       float = 0.8,
    k:              float = 1.0,
    cutoff:         float = 12.0,
    seed:           int   = 42,
    verbose:        bool  = True,
) -> Tuple[np.ndarray, np.ndarray, float, list, dict]:
    """
    SOTA multi-start optimization:

    1. N random restarts from Ramachandran-biased initialization
    2. W warm restarts from the best solution found (perturbed + re-optimized)
    3. Return globally best (phi, psi) and all restart scores

    Args:
        n_restarts:      Cold starts from random initialization.
        n_warm_restarts: Warm starts perturbing the best cold-start solution.
        n_steps:         Steps per cold restart.
        n_steps_warm:    Steps per warm restart (shorter, already near minimum).
        cutoff:          Framework atom cutoff in Angstroms.

    Returns:
        phi_best, psi_best, best_clash_energy, restart_scores, histories
    """
    rng = np.random.default_rng(seed)

    target_N_t = torch.tensor(target_N, dtype=torch.float32) if target_N is not None else None

    # Prefilter framework atoms using initial centroid estimate
    # Use anchor position as proxy — saves building loop before we have torsions
    anchor_pos = prev_C
    tree       = KDTree(fw_coords)
    # Use large initial cutoff since we don't know loop extent yet
    nearby     = tree.query_ball_point(anchor_pos, r=cutoff + n_loop *0.5)
    if len(nearby) == 0:
        nearby = list(range(len(fw_coords)))  # fallback: use all

    fw_near = fw_coords[nearby].astype(np.float32)
    fr_near = fw_radii[nearby].astype(np.float32)
    fw_t    = torch.tensor(fw_near, dtype=torch.float32)
    fw_r    = torch.tensor(fr_near, dtype=torch.float32)
    loop_r  = torch.tensor(
        [VDW_RADII['N'], VDW_RADII['CA'], VDW_RADII['C']] * n_loop,
        dtype=torch.float32,
    )

    if verbose:
        print(f"      Framework atoms nearby: {len(nearby)}/{len(fw_coords)}")

    best_phi   = None
    best_psi   = None
    best_loss  = float('inf')
    all_scores = []
    histories  = {}

    # ── Cold restarts ─────────────────────────────────────────────────────
    for i in range(n_restarts):
        phi0, psi0 = _rama_init(n_loop, rng)
        phi_opt, psi_opt, final_loss, history = optimize_clash_single(
            phi0, psi0,
            prev_N, prev_CA, prev_C, psi_prev,
            fw_t, fw_r, loop_r,
            target_N_t=target_N_t, k_close=k_close,
            n_steps=n_steps, lr=lr, grad_clip=grad_clip,
            softness=softness, k=k,
        )
        all_scores.append(('cold', i, float(final_loss)))
        histories[f'cold_{i}'] = history

        if verbose:
            print(f"        cold restart {i+1:2d}/{n_restarts}  "
                  f"clash={final_loss:.3f}  "
                  f"{'*best*' if final_loss < best_loss else ''}")

        if final_loss < best_loss:
            best_loss = final_loss
            best_phi  = phi_opt.copy()
            best_psi  = psi_opt.copy()

        if best_loss < 1e-4:
            if verbose:
                print("        Found zero-clash solution! Stopping cold restarts.")
            break

    # ── Warm restarts from best cold solution ─────────────────────────────
    if best_phi is not None and n_warm_restarts > 0 and best_loss >= 1e-4:
        if verbose:
            print(f"      Warm restarts from best cold (E={best_loss:.3f})...")

        # Perturbation scale: small enough to stay near minimum, large enough
        # to escape local basin
        perturb_scale = 0.15  # ~8.5 degrees

        for i in range(n_warm_restarts):
            phi0 = best_phi + rng.normal(0, perturb_scale, n_loop)
            psi0 = best_psi + rng.normal(0, perturb_scale, n_loop)
            phi_opt, psi_opt, final_loss, history = optimize_clash_single(
                phi0, psi0,
                prev_N, prev_CA, prev_C, psi_prev,
                fw_t, fw_r, loop_r,
                target_N_t=target_N_t, k_close=k_close,
                n_steps=n_steps_warm, lr=lr*0.5, grad_clip=grad_clip,
                softness=softness, k=k,
            )
            all_scores.append(('warm', i, float(final_loss)))
            histories[f'warm_{i}'] = history

            if verbose:
                print(f"        warm restart  {i+1:2d}/{n_warm_restarts}  "
                      f"clash={final_loss:.3f}  "
                      f"{'*best*' if final_loss < best_loss else ''}")

            if final_loss < best_loss:
                best_loss = final_loss
                best_phi  = phi_opt.copy()
                best_psi  = psi_opt.copy()

    return best_phi, best_psi, best_loss, all_scores, histories


# ─────────────────────────────────────────────────────────────────────────────
# Output helpers
# ─────────────────────────────────────────────────────────────────────────────

def save_loop_pdb(
    path:     Path,
    sequence: str,
    N:        np.ndarray,
    CA:       np.ndarray,
    C:        np.ndarray,
    O:        np.ndarray,
    remarks:  list,
):
    with open(path, 'w') as f:
        for r in remarks:
            f.write(f"REMARK {r}\n")
        write_pdb_atoms(f, sequence, N, CA, C, O)
        f.write("END\n")


def compute_O_atoms(N: np.ndarray, CA: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Approximate carbonyl O positions."""
    n = len(CA)
    O = np.zeros((n, 3))
    for i in range(n):
        v_ca = CA[i] - C[i]; v_ca /= np.linalg.norm(v_ca) + 1e-8
        if i < n - 1:
            v_n = N[i+1] - C[i]; v_n /= np.linalg.norm(v_n) + 1e-8
            bis = v_ca + v_n; bn = np.linalg.norm(bis)
            O[i] = C[i] - 1.229 * (bis / bn if bn > 1e-8 else v_ca)
        else:
            O[i] = C[i] - 1.229 * v_ca
    return O


def coords_to_angles(N, CA, C):
    """Compute phi/psi in degrees."""
    n   = len(N)
    phi = np.full(n, np.nan)
    psi = np.full(n, np.nan)
    for i in range(n):
        if i > 0:
            b0 = C[i-1]-N[i]; b1 = CA[i]-N[i];  b2 = C[i]-CA[i]
            b1h = b1/(np.linalg.norm(b1)+1e-10)
            v = b0-np.dot(b0,b1h)*b1h; w = b2-np.dot(b2,b1h)*b1h
            phi[i] = np.degrees(np.arctan2(np.dot(np.cross(b1h,v),w),np.dot(v,w)))
        if i < n-1:
            b0 = CA[i]-N[i]; b1 = C[i]-CA[i]; b2 = N[i+1]-C[i]
            b1h = b1/(np.linalg.norm(b1)+1e-10)
            v = b0-np.dot(b0,b1h)*b1h; w = b2-np.dot(b2,b1h)*b1h
            psi[i] = np.degrees(np.arctan2(np.dot(np.cross(b1h,v),w),np.dot(v,w)))
    return phi, psi


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def _parse():
    p = argparse.ArgumentParser(
        description='CDR3 loop clash minimization via torsion gradient descent'
    )
    p.add_argument('--dataset',     required=True,
                   help='Path to cdr3_dataset directory')
    p.add_argument('--output',      default='clash_optimizer_results',
                   help='Output directory')
    p.add_argument('--complex-dir',
                   default='/home/jtepperik/thesis/data/reference_final',
                   help='Directory of full TCR-pMHC PDBs for framework extraction')
    p.add_argument('--max-loops',   type=int, default=None)

    # Optimization
    p.add_argument('--n-restarts',      type=int,   default=5,
                   help='Number of cold random restarts')
    p.add_argument('--n-warm-restarts', type=int,   default=2,
                   help='Number of warm restarts from best cold solution')
    p.add_argument('--n-steps',         type=int,   default=200,
                   help='Gradient steps per cold restart')
    p.add_argument('--n-steps-warm',    type=int,   default=100,
                   help='Gradient steps per warm restart')
    p.add_argument('--lr',              type=float, default=0.05,
                   help='Adam learning rate')
    p.add_argument('--grad-clip',       type=float, default=1.0,
                   help='Gradient clipping norm')
    p.add_argument('--softness',        type=float, default=0.8,
                   help='VdW softness factor')
    p.add_argument('--k-clash',         type=float, default=1.0,
                   help='Softplus force constant')
    p.add_argument('--k-close',         type=float, default=10.0,
                   help='C-terminal closure penalty strength')
    p.add_argument('--cutoff',          type=float, default=12.0,
                   help='Framework atom cutoff (A)')
    p.add_argument('--seed',            type=int,   default=42)
    return p.parse_args()


def main():
    args = _parse()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    metadata_file = Path(args.dataset) / 'cdr3_dataset.json'
    with open(metadata_file) as f:
        dataset = json.load(f)
    if args.max_loops:
        dataset = dataset[:args.max_loops]

    print(f"\n{'='*60}")
    print(f"CDR3 CLASH OPTIMIZER")
    print(f"Loops: {len(dataset)}  Cold restarts: {args.n_restarts}  "
          f"Warm: {args.n_warm_restarts}  Steps: {args.n_steps}")
    print(f"LR: {args.lr}  Grad clip: {args.grad_clip}  "
          f"k: {args.k_clash}  Softness: {args.softness}")
    print(f"Output: {out.absolute()}\n{'='*60}")

    all_results = []

    for idx, meta in enumerate(dataset, 1):
        pdb_id     = meta['pdb_id']
        chain      = meta['chain']
        full_seq   = meta['full_sequence']
        cdr3_seq   = meta['cdr3_sequence']
        loop_start = meta['loop_start']
        loop_end   = meta['loop_end']
        n_loop     = loop_end - loop_start
        name       = f"{pdb_id}_{chain}"
        loop_out   = out / name
        loop_out.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"{idx}/{len(dataset)}  {pdb_id} chain {chain}  "
              f"loop={cdr3_seq} ({n_loop} res)")

        # Load native structure
        pdb_file = Path(meta['pdb_file'])
        if not pdb_file.is_absolute():
            pdb_file = Path(args.dataset) / pdb_file.name
        if not pdb_file.exists():
            print(f"  PDB not found: {pdb_file} — skipping"); continue

        seq, N_nat, CA_nat, C_nat, O_nat = load_cdr3_native(str(pdb_file))
        if seq != full_seq:
            print(f"  Sequence mismatch — skipping"); continue

        # Anchor atoms
        prev_N  = N_nat[loop_start - 1]
        prev_CA = CA_nat[loop_start - 1]
        prev_C  = C_nat[loop_start - 1]
        target_N = N_nat[loop_end]

        # psi_prev
        _anc = max(0, loop_start - 2)
        ph, ps = coords_to_angles(
            N_nat[_anc: loop_start + 1],
            CA_nat[_anc: loop_start + 1],
            C_nat[_anc: loop_start + 1],
        )
        _idx = loop_start - 1 - _anc
        psi_prev = float(np.deg2rad(ps[_idx])) if not np.isnan(ps[_idx]) else 0.0

        # Native clash baseline
        complex_pdb = Path(args.complex_dir) / f"{pdb_id}.pdb"
        if not complex_pdb.exists():
            print(f"  Complex PDB not found: {complex_pdb} — skipping"); continue

        try:
            fw_coords, fw_radii = extract_framework_atoms(
                str(complex_pdb), tcr_chain=chain,
                full_sequence=full_seq,
                loop_start=loop_start, loop_end=loop_end,
                n_flank_before=meta['n_flank_before'],
                n_flank_after=meta['n_flank_after'],
                anchor_C_coord=C_nat[loop_start - 1],
                target_N_coord=N_nat[loop_end],
            )
        except Exception as e:
            print(f"  Framework extraction failed: {e} — skipping"); continue

        native_clashes = count_hard_clashes(
            N_nat[loop_start:loop_end],
            CA_nat[loop_start:loop_end],
            C_nat[loop_start:loop_end],
            fw_coords, fw_radii,
        )
        print(f"  Native hard clashes: {native_clashes}")

        # Check junction bonds
        bond_n = float(np.linalg.norm(C_nat[loop_start-1] - N_nat[loop_start]))
        bond_c = float(np.linalg.norm(C_nat[loop_end-1]   - N_nat[loop_end]))
        print(f"  Junction bonds: N-term={bond_n:.3f}A  C-term={bond_c:.3f}A")

        # Run multi-start optimizer
        t0 = time.time()
        phi_best, psi_best, best_clash, restart_scores, histories = optimize_clash_multistart(
            n_loop    = n_loop,
            prev_N    = prev_N,
            prev_CA   = prev_CA,
            prev_C    = prev_C,
            psi_prev  = psi_prev,
            fw_coords = fw_coords,
            fw_radii  = fw_radii,
            target_N  = target_N,
            k_close   = args.k_close,
            n_restarts      = args.n_restarts,
            n_warm_restarts = args.n_warm_restarts,
            n_steps         = args.n_steps,
            n_steps_warm    = args.n_steps_warm,
            lr              = args.lr,
            grad_clip       = args.grad_clip,
            softness        = args.softness,
            k               = args.k_clash,
            cutoff          = args.cutoff,
            seed            = args.seed + idx,
            verbose         = True,
        )
        elapsed = time.time() - t0

        # Plot intermediate clash scores from optimization histories
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(8, 5))
            for run_name, history in histories.items():
                linestyle = '-' if 'cold' in run_name else '--'
                alpha = 0.8 if 'cold' in run_name else 1.0
                ax.plot(history, label=run_name, linestyle=linestyle, alpha=alpha)
            ax.set_yscale('log')
            ax.set_xlabel('Gradient Step')
            ax.set_ylabel('Clash Energy')
            ax.set_title(f'Optimization History for {pdb_id}_{chain}')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            fig.savefig(loop_out / 'optimization_history.png', dpi=150, bbox_inches='tight')
            plt.close(fig)
        except Exception as e:
            print(f"  (Optimization history plot failed: {e})")

        # Build best structure
        N_b, CA_b, C_b, O_b = build_loop(
            prev_N, prev_CA, prev_C, psi_prev, phi_best, psi_best
        )
        O_b = compute_O_atoms(N_b, CA_b, C_b)

        hard_clashes_best = count_hard_clashes(
            N_b, CA_b, C_b, fw_coords, fw_radii
        )

        # Native RMSD (anchored, no alignment)
        CA_loop_nat = CA_nat[loop_start:loop_end]
        rmsd = float(np.sqrt(np.mean(np.sum((CA_b - CA_loop_nat)**2, axis=1))))

        print(f"\n  Best clash energy:  {best_clash:.3f}")
        print(f"  Hard clashes:       {hard_clashes_best}  "
              f"(native={native_clashes})")
        print(f"  RMSD to native:     {rmsd:.3f}A")
        print(f"  Time:               {elapsed:.1f}s")

        # Save best PDB
        save_loop_pdb(
            loop_out / 'best_clash.pdb',
            cdr3_seq, N_b, CA_b, C_b, O_b,
            remarks=[
                f"Clash-optimized loop: {pdb_id} chain {chain}",
                f"Sequence: {cdr3_seq}",
                f"Clash energy: {best_clash:.4f}",
                f"Hard clashes: {hard_clashes_best}  (native={native_clashes})",
                f"RMSD to native: {rmsd:.3f}A",
                f"Restarts: {args.n_restarts} cold + {args.n_warm_restarts} warm",
                f"Steps per restart: {args.n_steps}",
                f"NOTE: Loop is NOT KIC-closed (no closure constraint applied)",
            ],
        )

        # Save native loop PDB for comparison
        save_loop_pdb(
            loop_out / 'native.pdb',
            cdr3_seq,
            N_nat[loop_start:loop_end],
            CA_nat[loop_start:loop_end],
            C_nat[loop_start:loop_end],
            O_nat[loop_start:loop_end],
            remarks=[
                f"Native loop: {pdb_id} chain {chain}",
                f"Hard clashes: {native_clashes}",
            ],
        )

        # Save restart scores CSV
        import csv
        with open(loop_out / 'restart_scores.csv', 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['type', 'index', 'clash_energy'])
            for row in restart_scores:
                w.writerow(row)

        all_results.append({
            'pdb_id':             pdb_id,
            'chain':              chain,
            'sequence':           cdr3_seq,
            'loop_length':        n_loop,
            'native_hard_clashes': native_clashes,
            'best_clash_energy':  float(best_clash),
            'best_hard_clashes':  hard_clashes_best,
            'rmsd_to_native':     rmsd,
            'time_s':             elapsed,
            'n_restarts':         args.n_restarts,
        })

    # ── Summary ───────────────────────────────────────────────────────────
    with open(out / 'results.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    if all_results:
        clash_e    = [r['best_clash_energy']   for r in all_results]
        hard       = [r['best_hard_clashes']   for r in all_results]
        nat_hard   = [r['native_hard_clashes'] for r in all_results]
        rmsds      = [r['rmsd_to_native']      for r in all_results]

        print(f"\n{'='*60}\nSUMMARY ({len(all_results)} loops)")
        print(f"  {'Metric':<30} {'Mean':>8}  {'Min':>8}  {'Max':>8}")
        print(f"  {'─'*56}")
        print(f"  {'Best clash energy':<30} "
              f"{np.mean(clash_e):>8.2f}  "
              f"{min(clash_e):>8.2f}  "
              f"{max(clash_e):>8.2f}")
        print(f"  {'Best hard clashes':<30} "
              f"{np.mean(hard):>8.1f}  "
              f"{min(hard):>8d}  "
              f"{max(hard):>8d}")
        print(f"  {'Native hard clashes':<30} "
              f"{np.mean(nat_hard):>8.1f}  "
              f"{min(nat_hard):>8d}  "
              f"{max(nat_hard):>8d}")
        print(f"  {'RMSD to native (A)':<30} "
              f"{np.mean(rmsds):>8.2f}  "
              f"{min(rmsds):>8.2f}  "
              f"{max(rmsds):>8.2f}")
        print(f"\n  Loops with 0 hard clashes: "
              f"{sum(h==0 for h in hard)}/{len(hard)}")
        print(f"  Results: {out / 'results.json'}")

        # Simple matplotlib summary
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, 3, figsize=(14, 4))
            fig.suptitle('CDR3 Clash Optimization Summary', fontweight='bold')

            axes[0].bar(range(len(hard)), nat_hard, alpha=0.5,
                        label='Native', color='red')
            axes[0].bar(range(len(hard)), hard, alpha=0.8,
                        label='Optimized', color='steelblue')
            axes[0].set_xlabel('Loop index')
            axes[0].set_ylabel('Hard clashes')
            axes[0].set_title('Hard clashes: native vs optimized')
            axes[0].legend()

            axes[1].scatter(nat_hard, hard, s=60, color='steelblue',
                            edgecolors='black', linewidths=0.5)
            lim = max(max(nat_hard)+1, max(hard)+1, 1)
            axes[1].plot([0, lim], [0, lim], 'r--', alpha=0.5, label='y=x')
            axes[1].set_xlabel('Native hard clashes')
            axes[1].set_ylabel('Optimized hard clashes')
            axes[1].set_title('Optimized vs native\n(below diagonal = improvement)')
            axes[1].legend()

            lengths = [r['loop_length'] for r in all_results]
            axes[2].scatter(lengths, clash_e, s=60, color='steelblue',
                            edgecolors='black', linewidths=0.5)
            axes[2].set_xlabel('Loop length (residues)')
            axes[2].set_ylabel('Best clash energy')
            axes[2].set_title('Clash energy vs loop length')

            plt.tight_layout()
            fig.savefig(out / 'summary.png', dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"  Summary plot: {out / 'summary.png'}")
        except Exception as e:
            print(f"  (Summary plot failed: {e})")


if __name__ == '__main__':
    main()