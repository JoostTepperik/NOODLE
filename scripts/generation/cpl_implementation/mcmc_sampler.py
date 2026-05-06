"""
mcmc_sampler.py

MCMC-based synthetic CDR3 loop conformation generator.

Strategy
--------
  Shear moves + KIC closure + Metropolis-Hastings acceptance.

  A shear move on residue i perturbs psi[i-1] by +delta and phi[i] by -delta
  simultaneously. The equal-and-opposite rotation approximately preserves
  the Cartesian position of CA[i] and all downstream atoms, localising the
  structural change to the i-1/i peptide plane. This keeps clash changes
  predictable and acceptance rates tractable.

  After each shear proposal, KIC analytically re-closes the loop onto the
  C-terminal anchor -- every proposed structure is geometrically closed.

  When multiple KIC solutions exist for a single proposal (up to 16),
  the lowest-energy solution is chosen as the proposal.

  Metropolis-Hastings acceptance uses:
      E_total = E_neural + w_clash * E_clash

  where E_neural is -log P from a per-residue Ramachandran joint table
  and E_clash uses score_clashes() from utils.py (softplus potential).

Basin-hopping escape
--------------------
  After `patience` consecutive rejections, a random contiguous sub-segment
  of residues is resampled from the neural model distribution and KIC
  re-closes. This allows the chain to jump between energy basins.

Step-size adaptation
--------------------
  Delta is adapted during burn-in using Nesterov dual averaging,
  targeting `target_accept` (default 0.40).

References
----------
  Shear moves:    Canutescu & Dunbrack (2003), Protein Sci. 12, 963-972.
  KIC closure:    Coutsias et al. (2004), J. Comput. Chem. 25, 510-528.
  Metropolis:     Metropolis et al. (1953), J. Chem. Phys. 21, 1087.
  Basin-hopping:  Wales & Doye (1997), J. Phys. Chem. A 101, 5111.
  Dual averaging: Nesterov (2009); Hoffman & Gelman (2014) NUTS.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from kic import kic_close_given_torsions
from nerf import build_loop, BOND_LENGTHS, BOND_ANGLES_RAD
from utils import VDW_RADII
# score_clashes from loop_modeler returns a raw count, not used here


# -----------------------------------------------------------------------------
# Neural energy helpers
# These are self-contained here since loop_modeling_nerf.py is out of scope.
# -----------------------------------------------------------------------------

N_BINS    = 36
BIN_WIDTH = 360.0 / N_BINS   # 10 degrees per bin


def _bin_angle(angle_deg: float) -> int:
    """Map angle in degrees to bin index in [-180, 180)."""
    return int((angle_deg + 180.0) % 360.0 / BIN_WIDTH) % N_BINS


def _interp_prob(phi_deg: float, psi_deg: float, prob_table: np.ndarray) -> float:
    """
    Bilinear interpolation into a (N_BINS, N_BINS) joint probability table.
    Handles periodic boundary by wrapping indices.
    """
    n  = prob_table.shape[0]
    bw = 360.0 / n

    pf = (phi_deg + 180.0) % 360.0 / bw
    sf = (psi_deg + 180.0) % 360.0 / bw

    pl = int(pf) % n;  ph = (pl + 1) % n
    sl = int(sf) % n;  sh = (sl + 1) % n
    pw = pf - int(pf);  sw = sf - int(sf)

    return (
        (1-pw)*(1-sw)*prob_table[pl, sl] +
        (1-pw)*   sw *prob_table[pl, sh] +
           pw *(1-sw)*prob_table[ph, sl] +
           pw *   sw *prob_table[ph, sh]
    )


def neural_energy(
    phi_rad: np.ndarray,   # (n_loop,)
    psi_rad: np.ndarray,   # (n_loop,)
    probs_joint: list,     # list of (N_BINS, N_BINS) arrays
) -> float:
    """E_neural = sum_i  -log P(phi[i], psi[i] | context_i)."""
    phi_deg = np.degrees(phi_rad)
    psi_deg = np.degrees(psi_rad)
    e = 0.0
    for i in range(len(phi_rad)):
        if i < len(probs_joint):
            p = _interp_prob(phi_deg[i], psi_deg[i], probs_joint[i])
            e -= math.log(max(float(p), 1e-10))
    return e


def ideal_energy(probs_joint: list) -> float:
    """
    Lower bound on neural energy: sum_i -log max P_joint[i].
    This is the energy achieved if every residue sits at its
    Ramachandran probability maximum.  Use as a zero reference.
    """
    e = 0.0
    for pj in probs_joint:
        e -= math.log(max(float(np.array(pj).max()), 1e-10))
    return e


def normalized_energy(
    phi_rad: np.ndarray,
    psi_rad: np.ndarray,
    probs_joint: list,
    e_ideal: float,
    n_loop_atoms: int,
    e_clash: float,
    w_clash: float = 1.0,
) -> Tuple[float, float, float]:
    """
    Compute normalized energies with interpretable scales.

    E_neural_norm = E_neural - E_ideal
        >= 0, units of nats above the per-loop minimum.
        Typical range: 0 (perfect) to ~5*n_loop (very bad).

    E_clash_norm = E_clash / n_loop_atoms
        Per-atom clash energy, independent of loop length.
        Typical range: 0 (no clashes) to ~50 (severe clashing).

    E_total = E_neural_norm + w_clash * E_clash_norm

    Returns (E_neural_norm, E_clash_norm, E_total).
    """
    e_n = neural_energy(phi_rad, psi_rad, probs_joint) - e_ideal
    e_c = e_clash / max(n_loop_atoms, 1)
    return e_n, e_c, e_n + w_clash * e_c


def sample_from_joint(prob_table: np.ndarray) -> Tuple[float, float]:
    """
    Draw a (phi_rad, psi_rad) pair from a joint probability table.
    Returns angles in radians.
    """
    p = np.array(prob_table, dtype=np.float64).ravel()
    p = np.clip(p, 0, None)
    p /= p.sum()
    k = np.random.choice(N_BINS * N_BINS, p=p)
    phi_deg = -180.0 + (k // N_BINS + 0.5) * BIN_WIDTH
    psi_deg = -180.0 + (k %  N_BINS + 0.5) * BIN_WIDTH
    return np.deg2rad(phi_deg), np.deg2rad(psi_deg)


# -----------------------------------------------------------------------------
# State container
# -----------------------------------------------------------------------------

@dataclass
class LoopSample:
    phi_rad:  np.ndarray   # (n_loop,)
    psi_rad:  np.ndarray   # (n_loop,)
    N:        np.ndarray   # (n_loop, 3)
    CA:       np.ndarray   # (n_loop, 3)
    C:        np.ndarray   # (n_loop, 3)
    O:        np.ndarray   # (n_loop, 3)
    e_neural: float        # raw -log P
    e_clash:  float        # raw softplus clash
    # Normalized versions set after construction (requires e_ideal, n_atoms)
    e_neural_norm: float = 0.0   # e_neural - e_ideal
    e_clash_norm:  float = 0.0   # e_clash / n_loop_atoms

    def copy(self) -> 'LoopSample':
        return LoopSample(
            self.phi_rad.copy(), self.psi_rad.copy(),
            self.N.copy(), self.CA.copy(), self.C.copy(), self.O.copy(),
            self.e_neural, self.e_clash,
            self.e_neural_norm, self.e_clash_norm,
        )


# -----------------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------------

def _wrap(a: float) -> float:
    """Wrap angle to (-pi, pi]."""
    return float(np.arctan2(np.sin(a), np.cos(a)))


def _softplus_clash(
    N: np.ndarray,
    CA: np.ndarray,
    C: np.ndarray,
    fw_coords: np.ndarray,
    fw_radii: np.ndarray,
    softness: float = 0.8,
    k: float = 0.5,
) -> float:
    """
    Softplus clash energy between loop backbone (N, CA, C) and framework atoms.

        E = sum_{all pairs} k * log(1 + exp(d_min - dist))

    where d_min = softness * (r_loop + r_fw).

    k=0.5 keeps clash energy on a similar scale to E_neural (~3-5 per residue),
    so neither term dominates the Metropolis criterion.  At k=0.5 a single
    hard overlap (dist << d_min) contributes ~0.5 * overlap_angstroms energy.
    Uses only N, CA, C (no O) to match intra-loop clash convention.
    """
    n = len(N)
    loop_names  = ['N', 'CA', 'C'] * n
    loop_coords = np.empty((n * 3, 3), dtype=np.float32)
    loop_coords[0::3] = N
    loop_coords[1::3] = CA
    loop_coords[2::3] = C
    loop_radii = np.array(
        [VDW_RADII['N'], VDW_RADII['CA'], VDW_RADII['C']] * n,
        dtype=np.float32,
    )

    # Pairwise distances: (n_loop_atoms, n_fw)
    diff  = loop_coords[:, None, :] - fw_coords[None, :, :]   # (nL, nFW, 3)
    dists = np.linalg.norm(diff, axis=2) + 1e-8                # (nL, nFW)
    d_min = softness * (loop_radii[:, None] + fw_radii[None, :])

    # Softplus: log(1 + exp(d_min - dist)) — positive when overlapping
    overlap = d_min - dists
    energy  = np.sum(k * np.log1p(np.exp(np.clip(overlap, -20, 20))))
    return float(energy)


def _eval_sample(
    phi_sol: np.ndarray,
    psi_sol: np.ndarray,
    prev_N: np.ndarray,
    prev_CA: np.ndarray,
    prev_C: np.ndarray,
    psi_prev: float,
    probs_joint: list,
    e_ideal: float,
    fw_coords: Optional[np.ndarray],
    fw_radii: Optional[np.ndarray],
) -> LoopSample:
    """Build backbone and compute raw + normalized energy terms."""
    N, CA, C, O = build_loop(prev_N, prev_CA, prev_C, psi_prev, phi_sol, psi_sol)
    e_n = neural_energy(phi_sol, psi_sol, probs_joint)
    e_clash = (
        _softplus_clash(N, CA, C, fw_coords, fw_radii)
        if fw_coords is not None else 0.0
    )
    n_atoms = len(N) * 3
    e_n_norm = e_n - e_ideal
    e_c_norm = e_clash / max(n_atoms, 1)
    return LoopSample(phi_sol, psi_sol, N, CA, C, O,
                      e_n, e_clash, e_n_norm, e_c_norm)


def _kic_best(
    phi: np.ndarray,
    psi: np.ndarray,
    prev_N: np.ndarray,
    prev_CA: np.ndarray,
    prev_C: np.ndarray,
    psi_prev: float,
    target_N: np.ndarray,
    probs_joint: list,
    e_ideal: float,
    fw_coords: Optional[np.ndarray],
    fw_radii: Optional[np.ndarray],
    w_clash: float,
    n_grid: int,
    tol: float,
) -> Optional[LoopSample]:
    """
    KIC-close (phi, psi), evaluate all solutions, return the one with
    lowest normalized weighted total energy. Returns None if no solution found.
    """
    solutions = kic_close_given_torsions(
        phi, psi, prev_N, prev_CA, prev_C, psi_prev, target_N,
        n_grid=n_grid, tol=tol,
    )
    if not solutions:
        return None

    best: Optional[LoopSample] = None
    best_total = float('inf')

    for phi_sol, psi_sol in solutions:
        s = _eval_sample(
            phi_sol, psi_sol,
            prev_N, prev_CA, prev_C, psi_prev,
            probs_joint, e_ideal, fw_coords, fw_radii,
        )
        total = s.e_neural_norm + w_clash * s.e_clash_norm
        if total < best_total:
            best_total = total
            best = s

    return best


# -----------------------------------------------------------------------------
# Sampler
# -----------------------------------------------------------------------------

# ─────────────────────────────────────────────────────────────────────────────
# Aggressive clash pre-optimizer
# ─────────────────────────────────────────────────────────────────────────────

def clash_minimize(
    phi_init:  np.ndarray,
    psi_init:  np.ndarray,
    prev_N:    np.ndarray,
    prev_CA:   np.ndarray,
    prev_C:    np.ndarray,
    psi_prev:  float,
    fw_coords: Optional[np.ndarray],
    fw_radii:  Optional[np.ndarray],
    n_steps:   int   = 100,
    lr:        float = 0.15,
    softness:  float = 0.8,
    k:         float = 0.5,
    cutoff:    float = 8.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Aggressive gradient descent minimizing ONLY clash energy.

    Ignores KIC closure and neural energy — sole goal is getting the loop
    out of clashing regions fast.  KIC re-closes afterward.

    Speed optimisations vs naive approach:
    - Framework atoms prefiltered to those within `cutoff` A of the initial
      loop centroid, reducing ~4000 -> ~100-300 atoms per step.
    - Fewer steps (100) with larger lr (0.15) — clash landscape is smooth
      enough to converge quickly.
    - Sequential NeRF loop kept in torch for autograd but on CPU (fast for
      n_loop <= 20).

    Returns phi_opt, psi_opt (not guaranteed closed; call KIC afterward).
    """
    import torch
    from scipy.spatial import KDTree

    if fw_coords is None:
        return phi_init.copy(), psi_init.copy()

    # ── Prefilter framework atoms near the initial loop ───────────────────
    # Build initial loop to find its centroid
    N0, CA0, C0, _ = build_loop(prev_N, prev_CA, prev_C, psi_prev,
                                 phi_init, psi_init)
    loop_centroid = np.vstack([N0, CA0, C0]).mean(axis=0)
    loop_extent   = np.linalg.norm(
        np.vstack([N0, CA0, C0]) - loop_centroid, axis=1
    ).max()

    # Keep only framework atoms within cutoff + loop_extent
    tree    = KDTree(fw_coords)
    nearby  = tree.query_ball_point(loop_centroid,
                                    r=cutoff + loop_extent + 2.0)
    if len(nearby) == 0:
        return phi_init.copy(), psi_init.copy()

    fw_near = fw_coords[nearby].astype(np.float32)
    fr_near = fw_radii[nearby].astype(np.float32)

    n    = len(phi_init)
    fw_t = torch.tensor(fw_near, dtype=torch.float32)
    fw_r = torch.tensor(fr_near, dtype=torch.float32)

    aN  = torch.tensor(prev_N,  dtype=torch.float32)
    aCA = torch.tensor(prev_CA, dtype=torch.float32)
    aC  = torch.tensor(prev_C,  dtype=torch.float32)

    loop_r = torch.tensor(
        [VDW_RADII['N'], VDW_RADII['CA'], VDW_RADII['C']] * n,
        dtype=torch.float32,
    )

    phi_v = torch.tensor(phi_init, dtype=torch.float32, requires_grad=True)
    psi_v = torch.tensor(psi_init, dtype=torch.float32, requires_grad=True)
    opt   = torch.optim.Adam([phi_v, psi_v], lr=lr)

    BL_CN  = torch.tensor(BOND_LENGTHS['C_N'],  dtype=torch.float32)
    BL_NCA = torch.tensor(BOND_LENGTHS['N_CA'], dtype=torch.float32)
    BL_CAC = torch.tensor(BOND_LENGTHS['CA_C'], dtype=torch.float32)
    BA_CCN = torch.tensor(BOND_ANGLES_RAD['CA_C_N'], dtype=torch.float32)
    BA_CNC = torch.tensor(BOND_ANGLES_RAD['C_N_CA'], dtype=torch.float32)
    BA_NCC = torch.tensor(BOND_ANGLES_RAD['N_CA_C'], dtype=torch.float32)
    OMEGA  = torch.tensor(math.pi, dtype=torch.float32)

    def _place(a, b, c, bl, ba, tor):
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

    best_loss = float('inf')
    best_phi  = phi_init.copy()
    best_psi  = psi_init.copy()

    for _ in range(n_steps):
        opt.zero_grad()

        N_list, CA_list, C_list = [], [], []
        a3, a2, a1 = aN, aCA, aC
        for i in range(n):
            Ni  = _place(a3, a2, a1, BL_CN,  BA_CCN, psi_v[i])
            CAi = _place(a2, a1, Ni,  BL_NCA, BA_CNC, OMEGA)
            Ci  = _place(a1, Ni, CAi, BL_CAC, BA_NCC, phi_v[i])
            N_list.append(Ni); CA_list.append(CAi); C_list.append(Ci)
            a3, a2, a1 = Ni, CAi, Ci

        atoms = torch.zeros(n * 3, 3)
        atoms[0::3] = torch.stack(N_list)
        atoms[1::3] = torch.stack(CA_list)
        atoms[2::3] = torch.stack(C_list)

        diff    = atoms[:, None, :] - fw_t[None, :, :]
        dists   = torch.norm(diff, dim=2) + 1e-8
        d_min   = softness * (loop_r[:, None] + fw_r[None, :])
        overlap = d_min - dists
        loss    = k * torch.log1p(
            torch.exp(torch.clamp(overlap, -20, 20))
        ).sum()

        loss.backward()
        opt.step()

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_phi  = phi_v.detach().numpy().copy()
            best_psi  = psi_v.detach().numpy().copy()

        # Early exit if clash is negligible
        if best_loss < 0.1:
            break

    return best_phi, best_psi


class MCMCSampler:
    """
    Shear-move + KIC + Metropolis-Hastings MCMC sampler for CDR3 loops.

    Args
    ----
    probs_joint:    Per-residue joint (phi, psi) probability tables,
                    each (N_BINS, N_BINS).  From cache_energy_distributions().
    prev_N/CA/C:    (3,) last N-terminal anchor atom coordinates.
    psi_prev:       psi of the N-terminal anchor residue (radians).
    target_N:       (3,) first C-terminal anchor N coordinate.
    fw_coords:      (N_fw, 3) framework atom positions, or None.
    fw_radii:       (N_fw,) framework VdW radii (required if fw_coords given).
    fw_radii:       VdW radii for framework atoms (required if fw_coords given).
    w_clash:        Weight on clash energy in Metropolis criterion.
    temperature:    Metropolis temperature (higher = more permissive).
    step_size:      Initial shear move magnitude in radians (~8.5 deg).
    target_accept:  Target acceptance rate for step-size adaptation.
    patience:       Consecutive rejections before basin-hop escape.
    escape_seg_len: Residues to resample in escape move.
    n_grid:         KIC outer grid resolution (3600 = 0.1 deg).
    kic_tol:        KIC closure tolerance in Angstroms.
    seed:           RNG seed.
    """

    def __init__(
        self,
        probs_joint:    list,
        prev_N:         np.ndarray,
        prev_CA:        np.ndarray,
        prev_C:         np.ndarray,
        psi_prev:       float,
        target_N:       np.ndarray,
        fw_coords:      Optional[np.ndarray] = None,
        fw_radii:       Optional[np.ndarray] = None,

        w_clash:        float = 1.0,
        temperature:    float = 1.0,
        step_size:      float = 0.35,   # ~20 deg — aggressive start
        target_accept:  float = 0.40,
        patience:       int   = 10,
        escape_seg_len: int   = -1,   # -1 = half the loop length
        n_grid:         int   = 3600,
        kic_tol:        float = 0.05,
        seed:           Optional[int] = None,
    ):
        self.probs_joint    = probs_joint
        self.n_loop         = len(probs_joint)
        self.prev_N         = prev_N
        self.prev_CA        = prev_CA
        self.prev_C         = prev_C
        self.psi_prev       = psi_prev
        self.target_N       = target_N
        self.fw_coords      = fw_coords
        self.fw_radii       = fw_radii
        self.e_ideal        = ideal_energy(probs_joint)

        self.w_clash        = w_clash
        self.temperature    = temperature
        self.step_size      = step_size
        self.target_accept  = target_accept
        self.patience       = patience
        self.escape_seg_len = escape_seg_len
        self.n_grid         = n_grid
        self.kic_tol        = kic_tol
        self.rng            = np.random.default_rng(seed)

        # Shear candidates: exclude residue 0 (psi_prev fixed) and
        # last two residues (KIC pivot residues ψ[L-2], φ[L-1], ψ[L-1])
        self._shear_cands = list(range(1, self.n_loop - 2))

        # Stats
        self.n_proposed = self.n_accepted = self.n_no_sol = self.n_escapes = 0

        # Nesterov dual averaging
        self._log_step = math.log(step_size)
        self._H_bar    = 0.0
        self._mu       = math.log(10 * step_size)
        self._gamma, self._t0, self._kappa, self._m = 0.05, 10, 0.75, 0

    # ── convenience ──────────────────────────────────────────────────────────

    def _total(self, s: LoopSample) -> float:
        """Weighted total using normalized energies (both ~0-50 range)."""
        return s.e_neural_norm + self.w_clash * s.e_clash_norm

    def _kic(self, phi: np.ndarray, psi: np.ndarray) -> Optional[LoopSample]:
        return _kic_best(
            phi, psi,
            self.prev_N, self.prev_CA, self.prev_C, self.psi_prev, self.target_N,
            self.probs_joint, self.e_ideal, self.fw_coords, self.fw_radii,
            self.w_clash, self.n_grid, self.kic_tol,
        )

    # ── initialisation ────────────────────────────────────────────────────────

    def _init_state(self, max_attempts: int = 500) -> LoopSample:
        """
        Find a closed initial conformation without requiring native torsions.

        The core challenge: for long loops (>12 res), sampling all n_loop
        torsions independently gives extremely low KIC closure probability
        because the chain end drifts far from target_N.

        Strategy — directed sampling with variable-length free segment:
        ----------------------------------------------------------------
        Fix the first (n_loop - n_free) torsions from the neural model,
        then exhaustively try random completions for the last n_free
        residues.  Since KIC controls the final 3 pivots analytically,
        only n_free = 3-5 additional residues need to land near target_N.
        This reduces the random search space from n_loop dimensions to
        n_free dimensions, increasing closure probability by orders of
        magnitude for long loops.

        n_free scales with loop length: longer loops need more free
        residues because the fixed prefix drifts more.

        Multiple independent restarts diversify the fixed prefix, ensuring
        the init sample is not always the same conformation.
        """
        best: Optional[LoopSample] = None

        # Number of freely-randomised tail residues (excluding KIC pivots)
        # For short loops (<=8): fix all, just try many random full samples
        # For medium (9-14):     free last 3 residues before pivots
        # For long (15+):        free last 5 residues before pivots
        if self.n_loop <= 8:
            n_free = self.n_loop - 2   # all non-pivot residues are free
        elif self.n_loop <= 14:
            n_free = min(3, self.n_loop - 2)
        else:
            n_free = min(5, self.n_loop - 2)

        n_fixed = max(0, self.n_loop - 2 - n_free)  # residues 0..n_fixed-1 fixed

        # Attempts budget: try many fixed prefixes, each with inner tries
        n_outer = max(10, max_attempts // 20)   # different fixed prefixes
        n_inner = max(20, max_attempts // n_outer)

        attempt_total = 0
        for outer in range(n_outer):
            if best is not None:
                break

            # Sample fixed prefix from neural model
            if n_fixed > 0:
                fixed_phis, fixed_psis = zip(*[
                    sample_from_joint(self.probs_joint[i])
                    for i in range(n_fixed)
                ])
                fixed_phis = np.array(fixed_phis)
                fixed_psis = np.array(fixed_psis)
            else:
                fixed_phis = np.array([])
                fixed_psis = np.array([])

            for inner in range(n_inner):
                attempt_total += 1

                # Free tail: alternate between model sampling and uniform
                # Model sampling gives Ramachandran-valid angles
                # Uniform covers geometrically closeable regions the model misses
                if inner % 3 == 0:
                    tail_ph, tail_ps = zip(*[
                        sample_from_joint(self.probs_joint[n_fixed + j])
                        for j in range(n_free)
                    ]) if n_free > 0 else ([], [])
                    tail_phis = np.array(tail_ph) if n_free > 0 else np.array([])
                    tail_psis = np.array(tail_ps) if n_free > 0 else np.array([])
                else:
                    tail_phis = self.rng.uniform(-np.pi, np.pi, n_free)
                    tail_psis = self.rng.uniform(-np.pi, np.pi, n_free)

                # Combine: fixed prefix + free tail + 2 KIC pivot slots
                # (KIC pivot slots are filled by kic_close_given_torsions)
                phi_prop = np.concatenate([fixed_phis, tail_phis,
                                           np.zeros(2)])  # pivot placeholders
                psi_prop = np.concatenate([fixed_psis, tail_psis,
                                           np.zeros(2)])  # pivot placeholders

                c = self._kic(phi_prop, psi_prop)
                if c is not None:
                    if best is None or self._total(c) < self._total(best):
                        best = c
                    break   # found one for this prefix, move on

        if best is None:
            raise RuntimeError(
                f"No closed initial conformation after {attempt_total} attempts "
                f"(loop_length={self.n_loop}, n_free={n_free}, n_fixed={n_fixed}). "
                f"Try increasing --max-init-attempts."
            )

        print(f"      Init (directed, attempt {attempt_total}, "
              f"n_fixed={n_fixed} n_free={n_free}): "
              f"E={self._total(best):.2f}  "
              f"(n_norm={best.e_neural_norm:.1f}  "
              f"cl_norm={best.e_clash_norm:.2f}  "
              f"e_ideal={self.e_ideal:.1f})")
        return best

    # ── proposals ────────────────────────────────────────────────────────────

    def _shear(self, cur: LoopSample) -> Optional[LoopSample]:
        """
        Shear move: psi[i-1] += +delta, phi[i] += -delta, then KIC re-close.
        """
        if not self._shear_cands:
            return self._escape(cur, seg_len=1)

        i     = int(self.rng.choice(self._shear_cands))
        delta = self.rng.normal(0.0, self.step_size)

        phi = cur.phi_rad.copy()
        psi = cur.psi_rad.copy()
        psi[i-1] = _wrap(psi[i-1] + delta)
        phi[i]   = _wrap(phi[i]   - delta)

        proposal = self._kic(phi, psi)
        if proposal is None:
            self.n_no_sol += 1
        return proposal

    def _escape(
        self, cur: LoopSample, seg_len: Optional[int] = None
    ) -> Optional[LoopSample]:
        """
        Basin-hop escape move.  Alternates between three strategies:

        1. Full resample  — all non-pivot residues resampled from the neural
                            model.  Most disruptive, best for escaping deep
                            clash basins.
        2. Half resample  — a contiguous segment of ~half the loop resampled.
                            Good balance between disruption and locality.
        3. Uniform full   — all non-pivot residues resampled uniformly.
                            Used when model sampling repeatedly fails to find
                            closeable conformations.

        Strategy is chosen randomly with weights 50/40/10.
        Avoids the last two residues (KIC pivot residues).
        """
        n_free = max(1, self.n_loop - 2)   # residues 0..n_loop-3 are free

        strategy = self.rng.choice(['full', 'half', 'uniform'],
                                    p=[0.50, 0.40, 0.10])

        phi = cur.phi_rad.copy()
        psi = cur.psi_rad.copy()

        if strategy == 'full' or strategy == 'uniform':
            for j in range(n_free):
                if strategy == 'uniform':
                    phi[j] = self.rng.uniform(-np.pi, np.pi)
                    psi[j] = self.rng.uniform(-np.pi, np.pi)
                else:
                    phi[j], psi[j] = sample_from_joint(self.probs_joint[j])

        else:  # half
            if seg_len is None:
                seg_len = (self.escape_seg_len if self.escape_seg_len > 0
                           else max(1, n_free // 2))
            seg_len   = min(seg_len, n_free)
            max_start = max(0, n_free - seg_len)
            start     = int(self.rng.integers(0, max_start + 1))
            end       = min(start + seg_len, n_free)
            for j in range(start, end):
                phi[j], psi[j] = sample_from_joint(self.probs_joint[j])

        return self._kic(phi, psi)

    # ── step-size adaptation (Nesterov dual averaging) ────────────────────────

    def _adapt(self, accepted: bool) -> None:
        self._m += 1
        m = self._m
        w = 1.0 / (m + self._t0)
        self._H_bar = (1 - w) * self._H_bar + w * (self.target_accept - float(accepted))
        log_eps = self._mu - (math.sqrt(m) / self._gamma) * self._H_bar
        self._log_step = (
            m ** (-self._kappa) * log_eps
            + (1 - m ** (-self._kappa)) * self._log_step
        )
        self.step_size = math.exp(log_eps)

    # ── main loop ─────────────────────────────────────────────────────────────

    def run(
        self,
        n_samples:       int,
        burnin:          int  = 200,
        thin:            int  = 5,
        adapt_burnin:    bool = True,
        max_init_tries:  int  = 500,
        verbose:         bool = True,
        report_interval: int  = 100,
    ) -> List[LoopSample]:
        """
        Run the MCMC chain and collect n_samples after burn-in.

        Args:
            n_samples:       Samples to collect (post burn-in).
            burnin:          Steps to discard.
            thin:            Keep every thin-th step.
            adapt_burnin:    Adapt step size during burn-in only.
            max_init_tries:  Attempts for initial closed conformation.
            verbose:         Print progress.
            report_interval: Steps between progress reports.

        Returns:
            List of LoopSample objects.
        """
        t0  = time.time()
        raw = self._init_state(max_attempts=max_init_tries)

        # Aggressively minimize clash before starting MCMC.
        # This gets the loop out of clashing regions fast using gradient
        # descent, without caring about KIC closure or neural energy.
        # KIC re-closes afterward.
        if self.fw_coords is not None:
            print(f"      Clash pre-minimization (100 steps, cutoff=8A)...")
            phi_opt, psi_opt = clash_minimize(
                raw.phi_rad, raw.psi_rad,
                self.prev_N, self.prev_CA, self.prev_C, self.psi_prev,
                self.fw_coords, self.fw_radii,
            )
            cur = self._kic(phi_opt, psi_opt)
            if cur is None:
                # KIC couldn't close after optimization — use original
                print(f"      KIC failed after pre-min, using raw init")
                cur = raw
            else:
                print(f"      Post-min: E={self._total(cur):.2f}  "
                      f"(n_norm={cur.e_neural_norm:.1f} "
                      f"cl_norm={cur.e_clash_norm:.2f})")
        else:
            cur = raw
        samples: List[LoopSample] = []
        streak  = 0
        total   = burnin + n_samples * thin

        if verbose:
            print(f"\n      MCMC: {total} steps "
                  f"({burnin} burn-in + {n_samples}x{thin} thinned)  "
                  f"init E={self._total(cur):.2f}  "
                  f"step={self.step_size:.4f} rad")

        for step in range(total):
            self.n_proposed += 1

            # Basin-hop if stuck, otherwise shear
            if streak >= self.patience:
                proposal = self._escape(cur)
                streak = 0
                self.n_escapes += 1
            else:
                proposal = self._shear(cur)

            # Metropolis acceptance
            accepted = False
            if proposal is not None:
                dE = self._total(proposal) - self._total(cur)
                if dE <= 0.0 or math.log(self.rng.uniform()) < -dE / self.temperature:
                    cur      = proposal
                    accepted = True
                    streak   = 0
                else:
                    streak += 1
            else:
                # KIC found no solution — counts as a full rejection toward escape
                streak += 1

            if accepted:
                self.n_accepted += 1
            # Adapt throughout — freeze only in final 10% to stabilise
            if step < int(0.90 * total):
                self._adapt(accepted)
            if step >= burnin and (step - burnin) % thin == 0:
                samples.append(cur.copy())

            if verbose and (step + 1) % report_interval == 0:
                ar = self.n_accepted / max(self.n_proposed, 1)
                kic_rate = self.n_no_sol / max(self.n_proposed, 1)
                print(f"        step {step+1:5d}/{total}  "
                      f"collected={len(samples)}/{n_samples}  "
                      f"accept={ar:.2f}  delta={self.step_size:.4f}  "
                      f"E={self._total(cur):.2f} "
                      f"(n_norm={cur.e_neural_norm:.1f} "
                      f"cl_norm={cur.e_clash_norm:.2f})  "
                      f"esc={self.n_escapes}  "
                      f"no_kic={self.n_no_sol}({kic_rate:.0%})")

        if verbose:
            ar = self.n_accepted / max(self.n_proposed, 1)
            print(f"\n      Done: {len(samples)} samples  {time.time()-t0:.1f}s  "
                  f"accept={ar:.3f}  "
                  f"no_kic={self.n_no_sol}/{self.n_proposed}  "
                  f"escapes={self.n_escapes}")
            if samples:
                ev = [self._total(s) for s in samples]
                print(f"      E: min={min(ev):.2f}  "
                      f"mean={np.mean(ev):.2f}  "
                      f"max={max(ev):.2f}")

        return samples

    def to_ensemble(
        self,
        samples:         List[LoopSample],
        sequence:        str,
        N_flank_before:  np.ndarray,
        CA_flank_before: np.ndarray,
        C_flank_before:  np.ndarray,
        O_flank_before:  np.ndarray,
        N_flank_after:   np.ndarray,
        CA_flank_after:  np.ndarray,
        C_flank_after:   np.ndarray,
        O_flank_after:   np.ndarray,
    ) -> list:
        """
        Convert LoopSample list to the extended ensemble format:

            (N, CA, C, O, phi_deg, psi_deg, e_neural, e_clash, closure_dist)

        where N/CA/C/O include the flank residues, phi_deg/psi_deg are
        loop-only torsions in degrees, and closure_dist = 0.0 (KIC exact).
        """
        from nerf import compute_O_atoms  # local import to avoid circular
        ensemble = []
        for s in samples:
            O_loop = compute_O_atoms(s.N, s.CA, s.C)
            ensemble.append((
                np.vstack([N_flank_before,  s.N,  N_flank_after]),
                np.vstack([CA_flank_before, s.CA, CA_flank_after]),
                np.vstack([C_flank_before,  s.C,  C_flank_after]),
                np.vstack([O_flank_before,  O_loop, O_flank_after]),
                np.degrees(s.phi_rad),
                np.degrees(s.psi_rad),
                s.e_neural,
                s.e_clash,
                0.0,   # KIC guarantees exact closure
            ))
        return ensemble


# -----------------------------------------------------------------------------
# Convenience entry point
# -----------------------------------------------------------------------------

def generate_synthetic_dataset(
    probs_joint:     list,
    sequence:        str,
    prev_N:          np.ndarray,
    prev_CA:         np.ndarray,
    prev_C:          np.ndarray,
    psi_prev:        float,
    target_N:        np.ndarray,
    N_flank_before:  np.ndarray,
    CA_flank_before: np.ndarray,
    C_flank_before:  np.ndarray,
    O_flank_before:  np.ndarray,
    N_flank_after:   np.ndarray,
    CA_flank_after:  np.ndarray,
    C_flank_after:   np.ndarray,
    O_flank_after:   np.ndarray,
    fw_coords:       Optional[np.ndarray] = None,
    fw_radii:        Optional[np.ndarray] = None,

    n_samples:       int   = 500,
    burnin:          int   = 200,
    thin:            int   = 5,
    w_clash:         float = 1.0,
    temperature:     float = 1.0,
    step_size:       float = 0.15,
    patience:        int   = 30,
    escape_seg_len:  int   = 4,
    seed:            Optional[int] = None,
    verbose:         bool  = True,
) -> list:
    """
    Initialise -> run MCMC -> return ensemble.

    Uses only kic.py, nerf.py, utils.py from the base codebase.

    Returns list of:
        (N, CA, C, O, phi_deg, psi_deg, e_neural, e_clash, closure_dist)
    """
    sampler = MCMCSampler(
        probs_joint=probs_joint,
        prev_N=prev_N, prev_CA=prev_CA, prev_C=prev_C,
        psi_prev=psi_prev, target_N=target_N,
        fw_coords=fw_coords, fw_radii=fw_radii,
        w_clash=w_clash, temperature=temperature, step_size=step_size,
        patience=patience, escape_seg_len=escape_seg_len, seed=seed,
    )
    samples = sampler.run(
        n_samples=n_samples, burnin=burnin, thin=thin, verbose=verbose,
    )
    return sampler.to_ensemble(
        samples, sequence,
        N_flank_before, CA_flank_before, C_flank_before, O_flank_before,
        N_flank_after,  CA_flank_after,  C_flank_after,  O_flank_after,
    )


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def _parse():
    import argparse
    p = argparse.ArgumentParser(
        description='MCMC synthetic CDR3 loop dataset generation'
    )

    # Data
    p.add_argument('--dataset',
                   required=True,
                   help='Path to cdr3_dataset directory (contains cdr3_dataset.json)')
    p.add_argument('--output',
                   default='mcmc_results',
                   help='Output directory')
    p.add_argument('--checkpoint',
                   required=True,
                   help='Path to energy model checkpoint')
    p.add_argument('--complex-dir',
                   default='/home/jtepperik/thesis/data/reference_final',
                   help='Directory of full TCR-pMHC complex PDBs for clash detection')

    # Loop selection
    p.add_argument('--max-loops',     type=int,   default=None,
                   help='Max number of loops to process (default: all)')

    # Sampling
    p.add_argument('--n-samples',     type=int,   default=500,
                   help='Number of MCMC samples per loop after burn-in')
    p.add_argument('--burnin',        type=int,   default=200,
                   help='Burn-in steps to discard')
    p.add_argument('--thin',          type=int,   default=5,
                   help='Thinning interval (keep every nth step)')
    p.add_argument('--temperature',   type=float, default=1.0,
                   help='Metropolis temperature')
    p.add_argument('--step-size',     type=float, default=0.15,
                   help='Initial shear move magnitude (radians)')
    p.add_argument('--target-accept', type=float, default=0.40,
                   help='Target acceptance rate for step-size adaptation')
    p.add_argument('--patience',      type=int,   default=30,
                   help='Consecutive rejections before basin-hop escape')
    p.add_argument('--escape-seg-len',type=int,   default=4,
                   help='Sub-segment length for basin-hop escape')
    p.add_argument('--seed',          type=int,   default=None,
                   help='Global RNG seed')

    # Clash
    p.add_argument('--clash',         action='store_true',
                   help='Enable framework clash detection')
    p.add_argument('--w-clash',       type=float, default=1.0,
                   help='Weight on clash energy in Metropolis criterion')

    # Output
    p.add_argument('--save-pdbs',     action='store_true',
                   help='Write PDB files for each sampled structure')
    p.add_argument('--report-interval', type=int, default=100,
                   help='Steps between progress reports')

    return p.parse_args()




def main():
    import json
    from pathlib import Path
    from utils import (
        load_model,
        load_cdr3_native,
        extract_framework_atoms,
        compute_loop_rmsds,
        save_pdbs,
        cache_energy_distributions,
        coords_to_angles,
    )

    args = _parse()

    router = load_model(args.checkpoint)

    # ── Load dataset ──────────────────────────────────────────────────────
    metadata_file = Path(args.dataset) / 'cdr3_dataset.json'
    if not metadata_file.exists():
        raise FileNotFoundError(f"{metadata_file} not found")

    with open(metadata_file) as f:
        dataset = json.load(f)
    if args.max_loops:
        dataset = dataset[:args.max_loops]

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"MCMC CDR3 SYNTHETIC DATASET GENERATION")
    print(f"Loops: {len(dataset)}  Samples/loop: {args.n_samples}")
    print(f"Burn-in: {args.burnin}  Thin: {args.thin}  Temp: {args.temperature}")
    print(f"Step size: {args.step_size} rad  Target accept: {args.target_accept}")
    print(f"Patience: {args.patience}  Escape seg: {args.escape_seg_len}")
    if args.clash:
        print(f"Clash: ENABLED  w={args.w_clash}")
    print(f"Output: {out.absolute()}\n{'='*60}")

    all_results = []

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
        print(f"  Loop: {cdr3_seq} ({loop_end - loop_start} res)")

        # ── Load native structure ─────────────────────────────────────────
        pdb_file = Path(meta['pdb_file'])
        if not pdb_file.is_absolute():
            pdb_file = Path(args.dataset) / pdb_file.name
        if not pdb_file.exists():
            print(f"  PDB not found: {pdb_file} — skipping")
            continue

        seq, N_nat, CA_nat, C_nat, O_nat = load_cdr3_native(str(pdb_file))
        if seq != full_seq:
            print(f"  Sequence mismatch — skipping")
            continue

        # Anchor coordinates
        prev_N   = N_nat[loop_start - 1]
        prev_CA  = CA_nat[loop_start - 1]
        prev_C   = C_nat[loop_start - 1]
        target_N = N_nat[loop_end]

        # Diagnostic: check peptide bond integrity at both loop junctions.
        # C_nat[loop_start-1] -> N_nat[loop_start]: anchor C to first loop N (~1.33A)
        # C_nat[loop_end-1]   -> N_nat[loop_end]:   last loop C to target N (~1.33A)
        bond_n_term = float(np.linalg.norm(C_nat[loop_start-1] - N_nat[loop_start]))
        bond_c_term = float(np.linalg.norm(C_nat[loop_end-1]   - N_nat[loop_end]))
        print(f"  Loop junctions: N-term bond={bond_n_term:.3f}A  "
              f"C-term bond={bond_c_term:.3f}A  "
              f"(ideal ~1.33A)  seq_len={len(seq)}")
        if bond_n_term > 2.0 or bond_c_term > 2.0:
            print(f"  WARNING: peptide bond > 2A suggests index error. ")
            print(f"  Full seq in PDB: {seq}")
            print(f"  JSON full_seq:   {full_seq}")

        # Compute psi_prev and native loop torsions from the native structure.
        # We need a window: [anchor-1, anchor, loop_0 .. loop_end, cterm_flank]
        _anc_start = max(0, loop_start - 2)
        _win_end   = min(len(seq), loop_end + 2)
        ph_full, ps_full = coords_to_angles(
            N_nat[_anc_start: _win_end],
            CA_nat[_anc_start: _win_end],
            C_nat[_anc_start: _win_end],
        )
        # psi_prev: psi of residue loop_start-1 in the window
        _psi_idx = loop_start - 1 - _anc_start
        psi_prev = float(np.deg2rad(ps_full[_psi_idx])) if not np.isnan(ps_full[_psi_idx]) else 0.0

        n_loop = loop_end - loop_start

        # ── Cache energy distributions ────────────────────────────────────
        print(f"  Caching energy distributions for {cdr3_seq}...")
        probs_joint = cache_energy_distributions(router, cdr3_seq)

        # ── Framework clash ───────────────────────────────────────────────
        fw_coords = fw_radii = None
        if args.clash:
            complex_pdb = Path(args.complex_dir) / f"{pdb_id}.pdb"
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
                except Exception as exc:
                    print(f"  Warning: framework extraction failed ({exc})")
            else:
                print(f"  Warning: {complex_pdb} not found — no clash detection")

        # ── Run MCMC ──────────────────────────────────────────────────────
        rng_seed = (args.seed + i) if args.seed is not None else None

        try:
            ensemble = generate_synthetic_dataset(
                probs_joint    = probs_joint,
                sequence       = cdr3_seq,
                prev_N         = prev_N,
                prev_CA        = prev_CA,
                prev_C         = prev_C,
                psi_prev       = psi_prev,
                target_N       = target_N,
                N_flank_before = N_nat[:loop_start],
                CA_flank_before= CA_nat[:loop_start],
                C_flank_before = C_nat[:loop_start],
                O_flank_before = O_nat[:loop_start],
                N_flank_after  = N_nat[loop_end:],
                CA_flank_after = CA_nat[loop_end:],
                C_flank_after  = C_nat[loop_end:],
                O_flank_after  = O_nat[loop_end:],
                fw_coords      = fw_coords,
                fw_radii       = fw_radii,
                n_samples      = args.n_samples,
                burnin         = args.burnin,
                thin           = args.thin,
                w_clash        = args.w_clash,
                temperature    = args.temperature,
                step_size      = args.step_size,
                patience       = args.patience,
                escape_seg_len = args.escape_seg_len,
                seed           = rng_seed,
                verbose        = True,
            )
        except RuntimeError as e:
            print(f"  MCMC failed: {e} — skipping")
            continue

        if not ensemble:
            print("  No samples collected — skipping")
            continue

        # ── Metrics ───────────────────────────────────────────────────────
        CA_loop_nat   = CA_nat[loop_start:loop_end]
        rmsds         = compute_loop_rmsds(ensemble, CA_loop_nat, loop_start, loop_end)
        e_neural_vals = [e[6] for e in ensemble]
        e_clash_vals  = [e[7] for e in ensemble]

        print(f"\n  Samples:  {len(ensemble)}")
        print(f"  RMSD:     best={rmsds.min():.3f}A  mean={rmsds.mean():.3f}A")
        print(f"  E_neural: min={min(e_neural_vals):.2f}  mean={np.mean(e_neural_vals):.2f}")
        print(f"  E_clash:  min={min(e_clash_vals):.2f}  mean={np.mean(e_clash_vals):.2f}")

        # ── Save PDBs ─────────────────────────────────────────────────────
        if args.save_pdbs:
            # save_pdbs expects (N, CA, C, O, energy, closure) — use e_neural, 0.0
            ens_compat = [(e[0], e[1], e[2], e[3], e[6], e[8]) for e in ensemble]
            selected   = list(range(min(10, len(ensemble))))
            save_pdbs(
                ens_compat, selected, full_seq,
                loop_start, loop_end, CA_loop_nat, name, str(loop_out),
            )

        # ── Save torsion CSV for flow matching training ───────────────────
        n_loop   = loop_end - loop_start
        rows     = [np.concatenate([e[4], e[5]]) for e in ensemble]
        csv_path = loop_out / f"{name}_torsions.csv"
        header   = (
            ','.join([f'phi_{j}' for j in range(n_loop)]) + ',' +
            ','.join([f'psi_{j}' for j in range(n_loop)])
        )
        np.savetxt(str(csv_path), np.array(rows), delimiter=',',
                   header=header, comments='')
        print(f"  Torsions saved: {csv_path}")

        all_results.append({
            'pdb_id':        pdb_id,
            'chain':         chain,
            'sequence':      cdr3_seq,
            'loop_length':   n_loop,
            'n_samples':     len(ensemble),
            'best_rmsd':     float(rmsds.min()),
            'mean_rmsd':     float(rmsds.mean()),
            'mean_e_neural': float(np.mean(e_neural_vals)),
            'mean_e_clash':  float(np.mean(e_clash_vals)),
        })

    # ── Summary ───────────────────────────────────────────────────────────
    if all_results:
        with open(out / 'mcmc_results.json', 'w') as f:
            json.dump(all_results, f, indent=2)

        rmsds_best    = [r['best_rmsd'] for r in all_results]
        total_samples = sum(r['n_samples'] for r in all_results)
        print(f"\n{'='*60}\nSUMMARY ({len(all_results)} loops)")
        print(f"  Mean best RMSD:   {np.mean(rmsds_best):.3f}A")
        print(f"  Median best RMSD: {np.median(rmsds_best):.3f}A")
        for t in [1, 2, 3]:
            print(f"  < {t}A: {sum(r < t for r in rmsds_best)}/{len(rmsds_best)}")
        print(f"  Total samples: {total_samples}")
        print(f"  Results: {out / 'mcmc_results.json'}")


if __name__ == '__main__':
    main()