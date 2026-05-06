"""
loop_modeling_nerf.py

NeRF-based loop modeling — joint (φ, ψ) energy model.

Responsibilities
────────────────
  Pure computation — geometry, energy, and optimization only.
  All I/O, plotting, and model loading live in utils.py.

Parameterisation
────────────────
  Free variables : phi[i], psi[i]  (one pair per loop residue, radians)
  Fixed geometry : ideal backbone bond lengths and angles
  Soft constraint: C-terminal closure  C_loop[-1] → N_flank_after[0]

Energy terms
────────────
  E_torsion : Σ_i  −log P(phi[i], psi[i] | context_i)   — always active
  E_closure : progressive harmonic restraint to C-terminal N atom
  E_clash   : soft repulsion between loop and framework atoms (optional)
  E_intra   : soft repulsion between non-bonded loop backbone atoms (optional)

Schedulers
──────────
  LR         : CosineAnnealingLR from peak to eta_min over n_steps
  Closure    : weight ramps linearly from 0 → closure_weight over n_steps
               so the optimizer first explores torsion energy freely
  Clash      : pulsed ramping (Rosetta-style) — multiple cycles of
               low→high clash weight, allowing escape from clash-induced
               local minima between pulses.  Starts at clash_start_frac.

Clash potential
───────────────
  Linear-capped soft-core: E = k * min(overlap, cap)
  Bounded gradient prevents clash from dominating at high overlaps.
  Framework clash uses a precomputed 3D penalty grid (trilinear
  interpolation, O(1) per atom, fully differentiable) instead of
  per-step KD-tree queries.

Initialisation
──────────────
  Rejection sampling filters model-sampled (φ,ψ) initialisations
  against the framework grid, keeping only low-clash conformations.

Bin convention
──────────────
  36 bins spanning [-180°, 180°) — 10° per bin.
  bin k centre = -180 + (k + 0.5) * 10°
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import torch

from utils import ModelRouter, _to_router, ONE_TO_THREE, VDW_RADII


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

N_BINS      = 36
BIN_WIDTH   = 360.0 / N_BINS   # 10°
BIN_CENTRES = np.array([-180.0 + (k + 0.5) * BIN_WIDTH for k in range(N_BINS)])

AA_TO_IDX = {
    'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4,
    'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9,
    'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14,
    'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19,
}
PAD_IDX     = 20
MAX_CONTEXT = 3
CONTEXT_RAD = MAX_CONTEXT // 2   # 1 residue each side

# Ideal backbone geometry (Å / radians)
BL_CN  = 1.329;  BL_NCA = 1.458;  BL_CAC = 1.525
BA_CCN = np.deg2rad(116.2)
BA_CNC = np.deg2rad(121.7)
BA_NCC = np.deg2rad(111.2)
OMEGA  = np.pi

_BL_CN  = torch.tensor(BL_CN,  dtype=torch.float32)
_BL_NCA = torch.tensor(BL_NCA, dtype=torch.float32)
_BL_CAC = torch.tensor(BL_CAC, dtype=torch.float32)
_BA_CCN = torch.tensor(BA_CCN, dtype=torch.float32)
_BA_CNC = torch.tensor(BA_CNC, dtype=torch.float32)
_BA_NCC = torch.tensor(BA_NCC, dtype=torch.float32)
_OMEGA  = torch.tensor(OMEGA,  dtype=torch.float32)

# Default soft-core clash parameters
DEFAULT_CLASH_CAP    = 1.5   # Å — maximum overlap contribution per pair
DEFAULT_CLASH_BUFFER = 0.5   # Å — buffer zone width beyond d_min


def _buffered_clash_potential(
    dist:      torch.Tensor,   # (...)  pairwise distances
    d_min:     torch.Tensor,   # (...)  contact thresholds (softness * sum of radii)
    k_clash:   float,
    cap:       float = DEFAULT_CLASH_CAP,
    buffer:    float = DEFAULT_CLASH_BUFFER,
    k_buffer:  float = None,   # auto-computed if None
) -> torch.Tensor:
    """
    Buffered linear-capped clash potential.

    Three zones:
      dist > d_min + buffer : E = 0  (no interaction)
      d_min < dist ≤ d_min + buffer : E = k_buf * t²  (smooth quadratic onset)
      dist ≤ d_min : E = k_buf + k_clash * min(d_min - dist, cap)  (linear-capped clash)

    where t = (d_min + buffer - dist) / buffer ∈ [0, 1].

    The quadratic buffer provides gradient signal BEFORE atoms overlap,
    gently pushing near-contact pairs apart.  The potential and its first
    derivative are continuous at all boundaries.

    k_buffer defaults to k_clash * buffer / 2, which makes the gradient
    at the d_min boundary match the linear clash slope (C1 continuity).
    """
    if buffer <= 0:
        # No buffer — fall back to original capped potential
        overlap = torch.clamp(d_min - dist, min=0.0)
        return k_clash * torch.minimum(overlap, torch.tensor(cap))

    if k_buffer is None:
        # C1-continuous join: quadratic gradient at d_min = k_clash
        # d/dt [k_buf * t²] at t=1 = 2*k_buf/buffer = k_clash
        # => k_buf = k_clash * buffer / 2
        k_buffer = k_clash * buffer / 2.0

    d_edge = d_min + buffer   # outer edge of buffer zone

    # Buffer zone: d_min < dist <= d_min + buffer
    t_buffer = torch.clamp((d_edge - dist) / buffer, min=0.0, max=1.0)
    e_buffer = k_buffer * t_buffer ** 2

    # Clash zone: dist <= d_min
    overlap = torch.clamp(d_min - dist, min=0.0)
    capped  = torch.minimum(overlap, torch.tensor(cap))
    e_clash = k_buffer + k_clash * capped

    # Select: clash zone where overlapping, buffer zone otherwise
    is_clashing = dist < d_min
    return torch.where(is_clashing, e_clash, e_buffer)


def _count_real_clashes(
    dist:    torch.Tensor,
    d_min:   torch.Tensor,
) -> int:
    """Count atom pairs with actual VdW overlap (dist < d_min). For diagnostics."""
    return int((dist < d_min).sum().item())


# ─────────────────────────────────────────────────────────────────────────────
# Core NeRF operations
# ─────────────────────────────────────────────────────────────────────────────

@torch.jit.script
def place_atom_b(
    a: torch.Tensor, b: torch.Tensor, c: torch.Tensor,
    bond_length: torch.Tensor, bond_angle: torch.Tensor,
    torsion: torch.Tensor,
) -> torch.Tensor:
    """
    Batched NeRF atom placement.
    a, b, c: (B, 3);  bond_length/bond_angle: scalar;  torsion: (B,)
    Returns: (B, 3)
    """
    bc    = c - b
    bc_n  = bc / (torch.norm(bc, dim=-1, keepdim=True) + 1e-8)
    n_abc = torch.linalg.cross(b - a, bc)
    n_abc = n_abc / (torch.norm(n_abc, dim=-1, keepdim=True) + 1e-8)
    col2  = torch.linalg.cross(n_abc, bc_n)
    M     = torch.stack([bc_n, col2, n_abc], dim=-1)
    d_local = torch.stack([
        -torch.cos(bond_angle).expand(torsion.shape[0]),
         torch.sin(bond_angle) * torch.cos(torsion),
        -torch.sin(bond_angle) * torch.sin(torsion),
    ], dim=-1) * bond_length
    return c + torch.bmm(M, d_local.unsqueeze(-1)).squeeze(-1)


def build_backbone(
    phi:       torch.Tensor,   # (B, n_loop)
    psi:       torch.Tensor,   # (B, n_loop+1)
    anchor_N:  np.ndarray,
    anchor_CA: np.ndarray,
    anchor_C:  np.ndarray,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Batched NeRF backbone. Returns N, CA, C each (B, n_loop, 3)."""
    B, n = phi.shape
    assert psi.shape == (B, n + 1)

    a3 = torch.tensor(anchor_N,  dtype=torch.float32).unsqueeze(0).expand(B, -1)
    a2 = torch.tensor(anchor_CA, dtype=torch.float32).unsqueeze(0).expand(B, -1)
    a1 = torch.tensor(anchor_C,  dtype=torch.float32).unsqueeze(0).expand(B, -1)

    N_list, CA_list, C_list = [], [], []
    for i in range(n):
        N_i  = place_atom_b(a3, a2, a1, _BL_CN,  _BA_CCN, psi[:, i])
        CA_i = place_atom_b(a2, a1, N_i, _BL_NCA, _BA_CNC, _OMEGA.expand(B))
        C_i  = place_atom_b(a1, N_i, CA_i, _BL_CAC, _BA_NCC, phi[:, i])
        N_list.append(N_i);  CA_list.append(CA_i);  C_list.append(C_i)
        a3, a2, a1 = N_i, CA_i, C_i

    return (torch.stack(N_list, dim=1),
            torch.stack(CA_list, dim=1),
            torch.stack(C_list, dim=1))


def place_N_after(
    N_last: torch.Tensor, CA_last: torch.Tensor,
    C_last: torch.Tensor, psi_last: torch.Tensor,
) -> torch.Tensor:
    """Place virtual N_after for C-terminal closure. Returns (B, 3)."""
    return place_atom_b(N_last, CA_last, C_last, _BL_CN, _BA_CCN, psi_last)


def compute_O_atoms(N: np.ndarray, CA: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Approximate carbonyl O positions from backbone N, CA, C."""
    n = len(CA)
    O = np.zeros((n, 3))
    for i in range(n):
        v_ca = CA[i] - C[i];  v_ca /= (np.linalg.norm(v_ca) + 1e-8)
        if i < n - 1:
            v_n  = N[i+1] - C[i];  v_n /= (np.linalg.norm(v_n) + 1e-8)
            bis  = v_ca + v_n;  bn = np.linalg.norm(bis)
            O[i] = C[i] - 1.229 * (bis / bn if bn > 1e-8 else v_ca)
        else:
            O[i] = C[i] - 1.229 * v_ca
    return O


def kabsch(P: np.ndarray, Q: np.ndarray):
    P = P - P.mean(axis=0);  Q = Q - Q.mean(axis=0)
    U, _, Vt = np.linalg.svd(P.T @ Q)
    d = np.linalg.det(Vt.T @ U.T)
    return Vt.T @ np.diag([1.0, 1.0, d]) @ U.T, P, Q


def aligned_loop_rmsd(
    CA_pred_full: np.ndarray, CA_native_full: np.ndarray,
    loop_start: int, loop_end: int,
) -> float:
    R, P_c, Q_c = kabsch(CA_pred_full, CA_native_full)
    diff = P_c[loop_start:loop_end] @ R.T - Q_c[loop_start:loop_end]
    return float(np.sqrt(np.mean(np.sum(diff ** 2, axis=1))))


# ─────────────────────────────────────────────────────────────────────────────
# Energy distributions
# ─────────────────────────────────────────────────────────────────────────────

def cache_energy_distributions(
    model_or_router,
    sequence: str,
    params=None,
) -> list:
    """
    Predict per-residue joint (φ, ψ) probability tables for `sequence`.
    Returns list of len(sequence) arrays, each (N_BINS, N_BINS).
    """
    router  = _to_router(model_or_router, params)
    encoded = np.array([AA_TO_IDX.get(aa, PAD_IDX) for aa in sequence.upper()])
    n       = len(encoded)
    probs   = []

    print(f"      Caching distributions for {n} residues...")
    for i in range(n):
        aa             = sequence[i].upper()
        model, mparams = router.get(aa)
        window = [int(encoded[pos]) if 0 <= pos < n else PAD_IDX
                  for pos in range(i - CONTEXT_RAD, i + CONTEXT_RAD + 1)]
        logits = model.apply(
            {'params': mparams},
            jnp.array(window)[None, :],
            jnp.ones((1, MAX_CONTEXT), dtype=bool),
            training=False,
            rngs={'dropout': jax.random.PRNGKey(0)},
        )
        probs.append(np.array(jax.nn.softmax(logits[0])).reshape(N_BINS, N_BINS))

    print(f"      Cached {n} joint ({N_BINS}×{N_BINS}) distributions")
    return probs


def _interp_prob_2d(
    phi_deg: torch.Tensor,
    psi_deg: torch.Tensor,
    probs_joint: np.ndarray,
) -> torch.Tensor:
    """Differentiable bilinear interpolation into a joint (φ,ψ) table."""
    pt  = torch.tensor(probs_joint, dtype=torch.float32)
    n   = pt.shape[0]
    bw  = 360.0 / n
    pf  = torch.fmod(phi_deg + 180.0, 360.0) / bw
    sf  = torch.fmod(psi_deg + 180.0, 360.0) / bw
    pl  = torch.floor(pf).long() % n;  ph = (pl + 1) % n
    sl  = torch.floor(sf).long() % n;  sh = (sl + 1) % n
    pw  = pf - torch.floor(pf);        sw = sf - torch.floor(sf)
    return (
        (1-pw)*(1-sw)*pt[pl,sl] + (1-pw)*sw*pt[pl,sh] +
           pw *(1-sw)*pt[ph,sl] +    pw *sw*pt[ph,sh]
    )


def compute_energy(
    phi_rad:     torch.Tensor,   # (B, n_loop)
    psi_rad:     torch.Tensor,   # (B, n_loop+1)
    probs_joint: list,
) -> torch.Tensor:               # (B,)
    """Joint energy: E = Σ_i  -log P(phi[i], psi[i] | context_i)."""
    B, n   = phi_rad.shape
    energy = torch.zeros(B, dtype=torch.float32)
    phi_d  = torch.rad2deg(phi_rad)
    psi_d  = torch.rad2deg(psi_rad)
    for i in range(n):
        if i < len(probs_joint):
            p = _interp_prob_2d(phi_d[:, i], psi_d[:, i+1], probs_joint[i])
            energy = energy - torch.log(p + 1e-10)
    return energy


def ideal_energy(probs_joint: list) -> float:
    """Lower-bound energy: Σ_i -log max P_joint[i]."""
    return sum(-math.log(float(np.array(p).max()) + 1e-10) for p in probs_joint)


# ─────────────────────────────────────────────────────────────────────────────
# Precomputed framework penalty grid  [NEW]
# ─────────────────────────────────────────────────────────────────────────────

class FrameworkGrid:
    """
    Precomputed 3D voxel grid encoding the maximum VdW penalty from
    framework atoms at each spatial position.

    The grid is built once from fixed framework atom positions and radii.
    During optimization, clash energy for any loop atom is computed via
    differentiable trilinear interpolation — O(1) per atom, no KD-tree.

    Grid values represent the "penalty potential" at each voxel:
      V(x) = max over nearby fw atoms of: max(0, softness*(r_fw + r_probe) - dist)

    where r_probe is the largest backbone VdW radius (CA = 1.87Å).
    During optimization, the actual loop atom radius refines this via:
      E_clash = k * min(V(x_loop) * r_loop/r_probe, cap)
    """

    def __init__(
        self,
        framework_coords: np.ndarray,   # (N_fw, 3)
        framework_radii:  np.ndarray,   # (N_fw,)
        resolution:       float = 0.5,  # Å per voxel
        padding:          float = 6.0,  # Å beyond framework extent
        softness:         float = 0.8,
        probe_radius:     float = 1.87, # CA radius — largest backbone
        buffer:           float = DEFAULT_CLASH_BUFFER,
    ):
        self.resolution = resolution
        self.softness   = softness
        self.probe_radius = probe_radius
        self.buffer     = buffer

        # Grid bounds
        self.origin = framework_coords.min(axis=0) - padding
        hi          = framework_coords.max(axis=0) + padding
        self.shape  = np.ceil((hi - self.origin) / resolution).astype(int)

        # Build grid: for each voxel, store the maximum "proximity" to any fw atom.
        # proximity = d_min - dist  (positive = overlapping, negative = approaching)
        # We store values down to -buffer (the buffer zone edge).
        # Values below -buffer are left at the sentinel -inf (no interaction).
        print(f"      Building framework grid: {self.shape} voxels "
              f"({np.prod(self.shape):,} total) @ {resolution}Å resolution "
              f"(buffer={buffer}Å)...")

        SENTINEL = -999.0
        grid = np.full(self.shape, SENTINEL, dtype=np.float32)

        # Extend reach to include buffer zone
        max_reach = (max(framework_radii) + probe_radius) / softness + buffer + 1.0
        reach_vox = int(np.ceil(max_reach / resolution)) + 1

        for atom_idx in range(len(framework_coords)):
            coord = framework_coords[atom_idx]
            r_fw  = framework_radii[atom_idx]
            d_min = softness * (r_fw + probe_radius)

            # Voxel index of this atom
            center_idx = ((coord - self.origin) / resolution).astype(int)

            # Iterate over nearby voxels
            lo_v = np.maximum(center_idx - reach_vox, 0)
            hi_v = np.minimum(center_idx + reach_vox + 1, self.shape)

            # Generate coordinate arrays for the sub-block
            xs = np.arange(lo_v[0], hi_v[0])
            ys = np.arange(lo_v[1], hi_v[1])
            zs = np.arange(lo_v[2], hi_v[2])

            if len(xs) == 0 or len(ys) == 0 or len(zs) == 0:
                continue

            # World coordinates of voxel centres
            wx = self.origin[0] + (xs + 0.5) * resolution
            wy = self.origin[1] + (ys + 0.5) * resolution
            wz = self.origin[2] + (zs + 0.5) * resolution

            # Distances from this atom to all voxels in sub-block
            dx = wx - coord[0]
            dy = wy - coord[1]
            dz = wz - coord[2]
            dist = np.sqrt(
                dx[:, None, None]**2 +
                dy[None, :, None]**2 +
                dz[None, None, :]**2
            ) + 1e-8

            # Proximity: positive = overlapping, negative = approaching
            proximity = d_min - dist

            # Only store values within buffer zone or overlapping
            # (proximity >= -buffer)
            mask = proximity >= -buffer
            sub = grid[lo_v[0]:hi_v[0], lo_v[1]:hi_v[1], lo_v[2]:hi_v[2]]
            # Take element-wise max with existing grid values
            # (sentinel values get overwritten by any real proximity)
            update = np.where(mask, proximity, SENTINEL)
            np.maximum(sub, update, out=sub)

        # Replace remaining sentinels with a value below -buffer
        # so they produce zero energy
        grid[grid <= SENTINEL + 1] = -buffer - 1.0

        self.grid_tensor = torch.tensor(grid, dtype=torch.float32)
        self.origin_t    = torch.tensor(self.origin, dtype=torch.float32)

        n_active = int((grid > -buffer).sum())
        n_clash  = int((grid > 0).sum())
        print(f"      Grid built: {n_clash:,} clash voxels, "
              f"{n_active:,} active (clash+buffer) "
              f"({100*n_active/grid.size:.1f}% fill)")

    def query_energy(
        self,
        loop_atoms:  torch.Tensor,   # (B, n_atoms, 3)
        loop_radii:  torch.Tensor,   # (n_atoms,)
        k_clash:     float = 100.0,
        cap:         float = DEFAULT_CLASH_CAP,
    ) -> torch.Tensor:               # (B,)
        """
        Differentiable framework clash energy via trilinear interpolation.

        The grid stores signed proximity values (positive = overlapping,
        negative = approaching within buffer zone).  The buffered potential
        is applied to the interpolated proximity after radius scaling.

        Returns per-structure clash energy (B,).
        """
        B, n_atoms, _ = loop_atoms.shape

        # Convert to fractional grid coordinates
        frac = (loop_atoms - self.origin_t.unsqueeze(0).unsqueeze(0)) / self.resolution

        # Trilinear interpolation
        gx = torch.clamp(frac[..., 0], 0, self.shape[0] - 1.001)
        gy = torch.clamp(frac[..., 1], 0, self.shape[1] - 1.001)
        gz = torch.clamp(frac[..., 2], 0, self.shape[2] - 1.001)

        ix = gx.long();  fx = gx - ix.float()
        iy = gy.long();  fy = gy - iy.float()
        iz = gz.long();  fz = gz - iz.float()

        ix1 = torch.clamp(ix + 1, max=self.shape[0] - 1)
        iy1 = torch.clamp(iy + 1, max=self.shape[1] - 1)
        iz1 = torch.clamp(iz + 1, max=self.shape[2] - 1)

        g = self.grid_tensor
        c000 = g[ix,  iy,  iz ]
        c001 = g[ix,  iy,  iz1]
        c010 = g[ix,  iy1, iz ]
        c011 = g[ix,  iy1, iz1]
        c100 = g[ix1, iy,  iz ]
        c101 = g[ix1, iy,  iz1]
        c110 = g[ix1, iy1, iz ]
        c111 = g[ix1, iy1, iz1]

        # Trilinear interpolation → signed proximity
        proximity = (
            c000 * (1-fx)*(1-fy)*(1-fz) +
            c001 * (1-fx)*(1-fy)*fz      +
            c010 * (1-fx)*fy    *(1-fz)  +
            c011 * (1-fx)*fy    *fz      +
            c100 * fx    *(1-fy)*(1-fz)  +
            c101 * fx    *(1-fy)*fz      +
            c110 * fx    *fy    *(1-fz)  +
            c111 * fx    *fy    *fz
        )  # (B, n_atoms)

        # Scale by actual atom radius / probe radius
        radius_scale = loop_radii.unsqueeze(0) / self.probe_radius
        scaled_proximity = proximity * radius_scale

        # Convert signed proximity to distance-like value for _buffered_clash_potential
        # proximity > 0 means overlap, proximity < 0 means approaching
        # _buffered_clash_potential expects (dist, d_min) where d_min - dist = proximity
        # We use a virtual d_min=0 and dist=-proximity, so overlap = 0 - (-prox) = prox
        virtual_dist = -scaled_proximity
        virtual_d_min = torch.zeros_like(virtual_dist)
        energy = _buffered_clash_potential(
            virtual_dist, virtual_d_min, k_clash,
            cap=cap, buffer=self.buffer,
        )

        return energy.sum(dim=-1)  # (B,)

    def query_score_np(
        self,
        atoms: np.ndarray,        # (n_atoms, 3)
        radii: np.ndarray,        # (n_atoms,)
        k_clash: float = 100.0,
        cap:     float = DEFAULT_CLASH_CAP,
    ) -> float:
        """Non-differentiable numpy query for diagnostic / init filtering."""
        buffer = self.buffer
        k_buffer = k_clash * buffer / 2.0 if buffer > 0 else 0.0

        frac = (atoms - self.origin) / self.resolution
        gx = np.clip(frac[:, 0], 0, self.shape[0] - 1.001)
        gy = np.clip(frac[:, 1], 0, self.shape[1] - 1.001)
        gz = np.clip(frac[:, 2], 0, self.shape[2] - 1.001)

        ix = gx.astype(int);  fx = gx - ix
        iy = gy.astype(int);  fy = gy - iy
        iz = gz.astype(int);  fz = gz - iz

        ix1 = np.minimum(ix + 1, self.shape[0] - 1)
        iy1 = np.minimum(iy + 1, self.shape[1] - 1)
        iz1 = np.minimum(iz + 1, self.shape[2] - 1)

        g = self.grid_tensor.numpy()
        proximity = (
            g[ix,  iy,  iz ] * (1-fx)*(1-fy)*(1-fz) +
            g[ix,  iy,  iz1] * (1-fx)*(1-fy)*fz      +
            g[ix,  iy1, iz ] * (1-fx)*fy    *(1-fz)  +
            g[ix,  iy1, iz1] * (1-fx)*fy    *fz      +
            g[ix1, iy,  iz ] * fx    *(1-fy)*(1-fz)  +
            g[ix1, iy,  iz1] * fx    *(1-fy)*fz      +
            g[ix1, iy1, iz ] * fx    *fy    *(1-fz)  +
            g[ix1, iy1, iz1] * fx    *fy    *fz
        )
        radius_scale = radii / self.probe_radius
        scaled = proximity * radius_scale

        # Apply buffered potential in numpy
        energy = np.zeros_like(scaled)
        if buffer > 0:
            # Buffer zone: -buffer < scaled <= 0
            in_buffer = (scaled > -buffer) & (scaled <= 0)
            t = (buffer + scaled) / buffer  # 0 at edge, 1 at contact
            energy[in_buffer] = k_buffer * t[in_buffer] ** 2
        # Clash zone: scaled > 0
        clashing = scaled > 0
        capped = np.minimum(scaled[clashing], cap)
        energy[clashing] = k_buffer + k_clash * capped

        return float(energy.sum())


def build_framework_grid(
    framework_coords: np.ndarray,
    framework_radii:  np.ndarray,
    resolution:       float = 0.5,
    padding:          float = 6.0,
    softness:         float = 0.8,
    buffer:           float = DEFAULT_CLASH_BUFFER,
) -> FrameworkGrid:
    """Convenience constructor for FrameworkGrid."""
    return FrameworkGrid(
        framework_coords, framework_radii,
        resolution=resolution, padding=padding, softness=softness,
        buffer=buffer,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Clash energy — intra-loop + boundary anchors
# ─────────────────────────────────────────────────────────────────────────────

def _build_boundary_atoms(
    anchor_N:  np.ndarray,    # (3,) — last flank-before residue
    anchor_CA: np.ndarray,
    anchor_C:  np.ndarray,
    N_closure: np.ndarray,    # (3,) — first flank-after residue N
    CA_closure: Optional[np.ndarray] = None,  # first flank-after CA (if available)
    C_closure:  Optional[np.ndarray] = None,  # first flank-after C  (if available)
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Collect fixed boundary atoms that the loop can clash with but are
    NOT part of the loop itself or the framework grid.

    Returns:
        boundary_coords: (n_boundary, 3) float32
        boundary_radii:  (n_boundary,) float32
    """
    coords = [anchor_N, anchor_CA, anchor_C, N_closure]
    radii  = [VDW_RADII['N'], VDW_RADII['CA'], VDW_RADII['C'], VDW_RADII['N']]
    if CA_closure is not None:
        coords.append(CA_closure)
        radii.append(VDW_RADII['CA'])
    if C_closure is not None:
        coords.append(C_closure)
        radii.append(VDW_RADII['C'])
    return (np.array(coords, dtype=np.float32),
            np.array(radii, dtype=np.float32))


def _build_boundary_exclusion_pairs(
    n_loop:     int,
    n_boundary: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build valid (non-bonded) pair indices between loop atoms and boundary atoms.

    Boundary atom layout:
      0=N_anc, 1=CA_anc, 2=C_anc  (last flank-before residue)
      3=N_clos [, 4=CA_clos, 5=C_clos]  (first flank-after residue)

    Loop atom layout:
      k = res * 3 + type   where type: 0=N, 1=CA, 2=C

    Excluded pairs (1-2 and 1-3 across the peptide bonds):
      C_anc(2) ↔ N_loop[0](0):   1-2 peptide bond
      CA_anc(1) ↔ N_loop[0](0):  1-3  (CA_anc — C_anc — N_loop0)
      C_anc(2) ↔ CA_loop[0](1):  1-3  (C_anc — N_loop0 — CA_loop0)

      C_loop[-1](n_loop*3 - 1) ↔ N_clos(3): 1-2 peptide bond
      CA_loop[-1](n_loop*3 - 2) ↔ N_clos(3): 1-3
      C_loop[-1](n_loop*3 - 1) ↔ CA_clos(4): 1-3  (if CA_clos present)

    Returns idx_loop, idx_boundary — valid pair indices.
    """
    n_loop_atoms = n_loop * 3
    pairs_loop = []
    pairs_bnd  = []

    # Pre-compute excluded pairs
    excluded = set()

    # N-terminal boundary: C_anc(2) bonds to N_loop[0](loop atom 0)
    excluded.add((0, 2))   # loop N[0] ↔ boundary C_anc
    excluded.add((0, 1))   # loop N[0] ↔ boundary CA_anc  (1-3)
    excluded.add((1, 2))   # loop CA[0] ↔ boundary C_anc  (1-3)

    # C-terminal boundary: C_loop[-1] bonds to N_clos(boundary 3)
    last_C  = n_loop_atoms - 1   # C of last loop residue
    last_CA = n_loop_atoms - 2   # CA of last loop residue
    excluded.add((last_C, 3))    # loop C[-1] ↔ boundary N_clos
    excluded.add((last_CA, 3))   # loop CA[-1] ↔ boundary N_clos (1-3)
    if n_boundary > 4:
        excluded.add((last_C, 4))  # loop C[-1] ↔ boundary CA_clos (1-3)

    for loop_idx in range(n_loop_atoms):
        for bnd_idx in range(n_boundary):
            if (loop_idx, bnd_idx) not in excluded:
                pairs_loop.append(loop_idx)
                pairs_bnd.append(bnd_idx)

    return (torch.tensor(pairs_loop, dtype=torch.long),
            torch.tensor(pairs_bnd,  dtype=torch.long))


def compute_boundary_clash_energy(
    loop_N:          torch.Tensor,    # (B, n_loop, 3)
    loop_CA:         torch.Tensor,
    loop_C:          torch.Tensor,
    boundary_coords: torch.Tensor,    # (n_boundary, 3)
    boundary_radii:  torch.Tensor,    # (n_boundary,)
    loop_radii:      torch.Tensor,    # (n_loop*3,)
    idx_loop:        torch.Tensor,    # pair indices into loop atoms
    idx_bnd:         torch.Tensor,    # pair indices into boundary atoms
    k_clash:         float = 100.0,
    softness:        float = 0.8,
    cap:             float = DEFAULT_CLASH_CAP,
    buffer:          float = DEFAULT_CLASH_BUFFER,
) -> torch.Tensor:                    # (B,)
    """
    Differentiable clash energy between loop backbone atoms and fixed
    boundary atoms (anchor residue + closure residue).

    Uses buffered linear-capped potential.
    """
    B = loop_N.shape[0]
    # Stack loop atoms: (B, n_loop*3, 3)
    loop_atoms = torch.stack([loop_N, loop_CA, loop_C], dim=2).reshape(B, -1, 3)

    d_min = softness * (loop_radii[idx_loop] + boundary_radii[idx_bnd])

    # Distances: loop atoms are batched, boundary is fixed
    diff = loop_atoms[:, idx_loop, :] - boundary_coords[idx_bnd].unsqueeze(0)
    dist = torch.norm(diff, dim=-1) + 1e-8  # (B, n_pairs)

    energy = _buffered_clash_potential(
        dist, d_min.unsqueeze(0), k_clash, cap=cap, buffer=buffer,
    )
    return energy.sum(dim=-1)  # (B,)


def _build_intraloop_exclusion_mask(n_loop: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pre-compute valid (non-bonded) atom pair indices for intra-loop clash.

    Backbone atoms are ordered N, CA, C per residue so atom k has:
      residue = k // 3,  type = k % 3  (0=N, 1=CA, 2=C)

    Excluded: 1-2 bonds and 1-3 pairs within and across peptide bonds.
    Returns idx_i, idx_j — upper-triangle valid pair indices.
    """
    n_atoms = n_loop * 3
    exclude = torch.zeros(n_atoms, n_atoms, dtype=torch.bool)

    for k in range(n_atoms):
        rk, tk = k // 3, k % 3
        for l in range(k + 1, n_atoms):
            rl, tl = l // 3, l % 3
            if rk == rl:
                exclude[k, l] = True
            elif rl == rk + 1:
                if tk == 2 and tl == 0:   # C–N  1-2
                    exclude[k, l] = True
                elif tk == 1 and tl == 0: # CA–N 1-3
                    exclude[k, l] = True
                elif tk == 2 and tl == 1: # C–CA 1-3
                    exclude[k, l] = True

    valid = torch.triu(~exclude, diagonal=1)
    return valid.nonzero(as_tuple=True)


def compute_intraloop_clash_energy(
    loop_N:   torch.Tensor,   # (B, n_loop, 3)
    loop_CA:  torch.Tensor,
    loop_C:   torch.Tensor,
    idx_i:    torch.Tensor,
    idx_j:    torch.Tensor,
    radii:    torch.Tensor,   # (n_loop*3,)
    k_clash:  float = 100.0,
    softness: float = 0.8,
    cap:      float = DEFAULT_CLASH_CAP,
    buffer:   float = DEFAULT_CLASH_BUFFER,
) -> torch.Tensor:            # (B,)
    """
    Soft buffered repulsion between non-bonded loop backbone atom pairs.

    Uses buffered linear-capped potential: provides gradient signal in a
    buffer zone before contact, then linear-capped penalty on overlap.
    """
    atoms = torch.stack([loop_N, loop_CA, loop_C], dim=2).reshape(
        loop_N.shape[0], -1, 3
    )
    d_min = softness * (radii[idx_i] + radii[idx_j])
    diff  = atoms[:, idx_i, :] - atoms[:, idx_j, :]
    dist  = torch.norm(diff, dim=-1) + 1e-8
    energy = _buffered_clash_potential(
        dist, d_min.unsqueeze(0), k_clash, cap=cap, buffer=buffer,
    )
    return energy.sum(dim=-1)


def compute_framework_clash_energy(
    loop_N:           torch.Tensor,
    loop_CA:          torch.Tensor,
    loop_C:           torch.Tensor,
    loop_radii:       np.ndarray,
    framework_coords: np.ndarray,
    framework_radii:  np.ndarray,
    kdtree,
    k_clash:          float = 100.0,
    softness:         float = 0.8,
    cutoff:           float = 8.0,
) -> torch.Tensor:            # (B,)
    """
    LEGACY: Soft harmonic repulsion between loop and fixed framework.

    Kept for backward compatibility. Prefer FrameworkGrid.query_energy()
    for new code — it's faster and fully differentiable.
    """
    B = loop_N.shape[0]
    loop_atoms = torch.stack([loop_N, loop_CA, loop_C], dim=2).reshape(B, -1, 3)
    fw_tensor  = torch.tensor(framework_coords, dtype=torch.float32)
    fw_r_t     = torch.tensor(framework_radii,  dtype=torch.float32)
    loop_r_t   = torch.tensor(loop_radii,       dtype=torch.float32)
    energy = torch.zeros(B, dtype=torch.float32)

    for b in range(B):
        loop_np        = loop_atoms[b].detach().numpy()
        neighbour_sets = kdtree.query_ball_point(loop_np, r=cutoff)
        all_idx = [ns for ns in neighbour_sets if len(ns) > 0]
        if not all_idx:
            continue
        nearby_idx = np.unique(np.concatenate(all_idx))
        fw_nearby  = fw_tensor[nearby_idx]
        fw_r_near  = fw_r_t[nearby_idx]
        diff  = loop_atoms[b].unsqueeze(1) - fw_nearby.unsqueeze(0)
        dist  = torch.norm(diff, dim=-1) + 1e-8
        d_min = softness * (loop_r_t.unsqueeze(1) + fw_r_near.unsqueeze(0))
        energy[b] = ((k_clash / 2.0) * torch.clamp(d_min - dist, min=0.0) ** 2).sum()

    return energy


# ─────────────────────────────────────────────────────────────────────────────
# Native clash score — numpy only, no gradient needed
# ─────────────────────────────────────────────────────────────────────────────

def compute_native_clash_score(
    N_loop:           np.ndarray,   # (n_loop, 3)
    CA_loop:          np.ndarray,
    C_loop:           np.ndarray,
    framework_coords: Optional[np.ndarray] = None,
    framework_radii:  Optional[np.ndarray] = None,
    kdtree                                 = None,
    framework_grid:   Optional[FrameworkGrid] = None,
    k_clash:          float = 100.0,
    softness:         float = 0.8,
    cutoff:           float = 8.0,
) -> Dict[str, float]:
    """
    Compute clash scores for the native loop conformation.

    No gradients — pure numpy computation for diagnostic purposes.
    Reports overlap-only energy (no buffer zone contributions) so the
    scores reflect actual physical clashes.

    Returns a dict with keys:
      'intra', 'framework', 'total' — overlap-only energies
      'n_intra', 'n_framework'      — number of overlapping atom pairs
    """
    n_loop = len(N_loop)

    # ── Intra-loop ────────────────────────────────────────────────────────
    atoms = np.empty((n_loop * 3, 3), dtype=np.float32)
    for i in range(n_loop):
        atoms[i*3]   = N_loop[i]
        atoms[i*3+1] = CA_loop[i]
        atoms[i*3+2] = C_loop[i]

    radii_arr = np.tile(
        [VDW_RADII['N'], VDW_RADII['CA'], VDW_RADII['C']], n_loop
    ).astype(np.float32)

    n_atoms = n_loop * 3
    e_intra   = 0.0
    n_intra   = 0
    for k in range(n_atoms):
        rk, tk = k // 3, k % 3
        for l in range(k + 1, n_atoms):
            rl, tl = l // 3, l % 3
            excluded = False
            if rk == rl:
                excluded = True
            elif rl == rk + 1:
                if (tk == 2 and tl == 0) or (tk == 1 and tl == 0) or (tk == 2 and tl == 1):
                    excluded = True
            if excluded:
                continue
            d_min = softness * (radii_arr[k] + radii_arr[l])
            dist  = float(np.linalg.norm(atoms[k] - atoms[l])) + 1e-8
            overlap = d_min - dist
            if overlap > 0:
                capped = min(overlap, DEFAULT_CLASH_CAP)
                e_intra += k_clash * capped
                n_intra += 1

    # ── Framework (overlap-only, no buffer) ───────────────────────────────
    e_fw  = 0.0
    n_fw  = 0
    if framework_grid is not None:
        # Use grid but compute overlap-only (buffer=0)
        frac = (atoms - framework_grid.origin) / framework_grid.resolution
        gx = np.clip(frac[:, 0], 0, framework_grid.shape[0] - 1.001)
        gy = np.clip(frac[:, 1], 0, framework_grid.shape[1] - 1.001)
        gz = np.clip(frac[:, 2], 0, framework_grid.shape[2] - 1.001)

        ix = gx.astype(int);  fx = gx - ix
        iy = gy.astype(int);  fy = gy - iy
        iz = gz.astype(int);  fz = gz - iz

        ix1 = np.minimum(ix + 1, framework_grid.shape[0] - 1)
        iy1 = np.minimum(iy + 1, framework_grid.shape[1] - 1)
        iz1 = np.minimum(iz + 1, framework_grid.shape[2] - 1)

        g = framework_grid.grid_tensor.numpy()
        proximity = (
            g[ix,  iy,  iz ] * (1-fx)*(1-fy)*(1-fz) +
            g[ix,  iy,  iz1] * (1-fx)*(1-fy)*fz      +
            g[ix,  iy1, iz ] * (1-fx)*fy    *(1-fz)  +
            g[ix,  iy1, iz1] * (1-fx)*fy    *fz      +
            g[ix1, iy,  iz ] * fx    *(1-fy)*(1-fz)  +
            g[ix1, iy,  iz1] * fx    *(1-fy)*fz      +
            g[ix1, iy1, iz ] * fx    *fy    *(1-fz)  +
            g[ix1, iy1, iz1] * fx    *fy    *fz
        )
        # Scale by actual radius
        radius_scale = radii_arr / framework_grid.probe_radius
        scaled = proximity * radius_scale

        # Overlap-only: only positive proximity (actual VdW overlap)
        overlapping = scaled > 0
        n_fw = int(overlapping.sum())
        if n_fw > 0:
            capped = np.minimum(scaled[overlapping], DEFAULT_CLASH_CAP)
            e_fw = float(k_clash * capped.sum())

    elif framework_coords is not None and framework_radii is not None and kdtree is not None:
        fw_r = framework_radii.astype(np.float32)
        for k in range(n_atoms):
            nearby = kdtree.query_ball_point(atoms[k], r=cutoff)
            if not nearby:
                continue
            for j in nearby:
                d_min = softness * (radii_arr[k] + fw_r[j])
                dist  = float(np.linalg.norm(atoms[k] - framework_coords[j])) + 1e-8
                overlap = d_min - dist
                if overlap > 0:
                    capped = min(overlap, DEFAULT_CLASH_CAP)
                    e_fw += k_clash * capped
                    n_fw += 1

    return {
        'intra':       float(e_intra),
        'framework':   float(e_fw),
        'total':       float(e_intra + e_fw),
        'n_intra':     n_intra,
        'n_framework': n_fw,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Pulsed clash schedule  [NEW]
# ─────────────────────────────────────────────────────────────────────────────

def pulsed_clash_weight(
    step:             int,
    n_steps:          int,
    clash_weight:     float,
    clash_start_frac: float = 0.25,
    n_pulses:         int   = 3,
    floor_frac:       float = 0.02,
) -> float:
    """
    Rosetta-style pulsed repulsive ramping.

    Before clash_start_frac * n_steps: returns 0 (clash inactive).
    After that: divides the remaining steps into `n_pulses` cycles.
    Each cycle ramps from floor_frac * clash_weight to clash_weight.

    The pulsing allows the optimizer to escape clash-induced local minima
    at the start of each cycle while still converging to a low-clash
    solution by the end of the final cycle.

    Args:
        step:             current optimisation step
        n_steps:          total number of steps
        clash_weight:     target maximum clash weight
        clash_start_frac: fraction of steps before clash kicks in
        n_pulses:         number of ramp cycles (Rosetta uses 3-5)
        floor_frac:       minimum weight as fraction of clash_weight

    Returns:
        effective clash weight for this step
    """
    clash_start_step = int(clash_start_frac * n_steps)
    if step < clash_start_step:
        return 0.0

    active_steps = n_steps - clash_start_step
    if active_steps <= 0:
        return clash_weight

    # Position within active phase [0, 1)
    t = (step - clash_start_step) / active_steps

    # Which pulse and where within it
    pulse_frac = t * n_pulses
    within_pulse = pulse_frac - int(pulse_frac)  # [0, 1) within current pulse

    # For the last pulse, make sure we end at full weight
    if int(pulse_frac) >= n_pulses:
        return clash_weight

    # Ramp within pulse: floor → full
    ramp = floor_frac + (1.0 - floor_frac) * within_pulse
    return clash_weight * ramp


# ─────────────────────────────────────────────────────────────────────────────
# Sampling helpers
# ─────────────────────────────────────────────────────────────────────────────

def _argmax_joint(pj: np.ndarray) -> Tuple[float, float]:
    k = int(np.argmax(pj))
    return (-180.0 + (k // N_BINS + 0.5) * BIN_WIDTH,
            -180.0 + (k %  N_BINS + 0.5) * BIN_WIDTH)


def _sample_from_joint(pj: np.ndarray) -> Tuple[float, float]:
    """Draw a correlated (phi_rad, psi_rad) pair from a joint distribution."""
    p = np.array(pj, dtype=np.float64).ravel()
    p = np.clip(p, 0, None);  p /= p.sum()
    k = np.random.choice(N_BINS * N_BINS, p=p)
    return (np.deg2rad(-180.0 + (k // N_BINS + 0.5) * BIN_WIDTH),
            np.deg2rad(-180.0 + (k %  N_BINS + 0.5) * BIN_WIDTH))


def ideal_structure_pdb(
    probs_joint: list, loop_seq: str,
    anchor_N: np.ndarray, anchor_CA: np.ndarray, anchor_C: np.ndarray,
    path: str,
):
    """Build and save a PDB of the argmax-energy loop conformation."""
    from utils import write_pdb_atoms
    n_loop = len(loop_seq)
    phi_d, psi_d = zip(*[_argmax_joint(probs_joint[i]) for i in range(n_loop)])
    phi_t = torch.tensor(np.deg2rad(phi_d), dtype=torch.float32).unsqueeze(0)
    psi_t = torch.tensor(
        np.deg2rad(np.concatenate([[-57.0], list(psi_d[:-1]), [-57.0]])),
        dtype=torch.float32,
    ).unsqueeze(0)
    with torch.no_grad():
        N_t, CA_t, C_t = build_backbone(phi_t, psi_t, anchor_N, anchor_CA, anchor_C)
    N_np = N_t[0].numpy();  CA_np = CA_t[0].numpy();  C_np = C_t[0].numpy()
    O_np = compute_O_atoms(N_np, CA_np, C_np)
    with open(path, 'w') as f:
        f.write(f"REMARK Ideal-energy structure: {loop_seq}\n")
        write_pdb_atoms(f, loop_seq, N_np, CA_np, C_np, O_np)
        f.write("END\n")
    print(f"  ✓  Ideal structure → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Rejection-sampled initialisation  [NEW]
# ─────────────────────────────────────────────────────────────────────────────

def _clash_filtered_init(
    probs_joint:    list,
    n_loop:         int,
    anchor_N:       np.ndarray,
    anchor_CA:      np.ndarray,
    anchor_C:       np.ndarray,
    N_closure:      np.ndarray,
    n_structures:   int,
    framework_grid: Optional[FrameworkGrid] = None,
    loop_radii_np:  Optional[np.ndarray]    = None,
    k_clash:        float = 100.0,
    max_clash:      float = 50.0,
    max_closure:    float = None,    # Å — max closure distance to accept (None=disabled)
    max_intra:      float = None,    # max intra-loop clash energy to accept (None=disabled)
    max_attempts:   int   = 500,
    clash_buffer:   float = DEFAULT_CLASH_BUFFER,  # use same buffer as optimizer
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sample (φ,ψ) from the joint model distribution and filter by
    multiple criteria, keeping only initialisations that pass all
    active filters.

    Filters (each independently enabled by setting a threshold):
      - Framework clash (max_clash): grid-based fw clash score
      - Closure distance (max_closure): Å distance from loop C-term to N_closure
      - Intra-loop clash (max_intra): intra-loop backbone clash energy

    If not enough samples pass, fills with best-of-k ranked by a
    weighted composite score.

    Returns phi_init (n_structures, n_loop), psi_init (n_structures, n_loop+1).
    """
    any_filter = (
        (framework_grid is not None and loop_radii_np is not None)
        or max_closure is not None
        or max_intra is not None
    )

    if not any_filter:
        # No filtering — standard model sampling
        phi_rows, psi_rows = [], []
        for _ in range(n_structures):
            phis, psis = zip(*[_sample_from_joint(probs_joint[i]) for i in range(n_loop)])
            phi_rows.append(torch.tensor(phis, dtype=torch.float32))
            psi_rows.append(torch.cat([
                torch.FloatTensor(1).uniform_(-np.pi, np.pi),
                torch.tensor(psis[:-1], dtype=torch.float32),
                torch.FloatTensor(1).uniform_(-np.pi, np.pi),
            ]))
        return torch.stack(phi_rows), torch.stack(psi_rows)

    N_clos_t = torch.tensor(N_closure, dtype=torch.float32)

    # Pre-build intra-loop exclusion mask if filtering on intra clash
    intra_idx_i = intra_idx_j = intra_radii = None
    if max_intra is not None:
        intra_idx_i, intra_idx_j = _build_intraloop_exclusion_mask(n_loop)
        intra_radii = torch.tensor(
            [VDW_RADII['N'], VDW_RADII['CA'], VDW_RADII['C']] * n_loop,
            dtype=torch.float32,
        )

    accepted_phi   = []
    accepted_psi   = []
    rejected_scores = []   # composite score for best-of-k fallback
    rejected_phi    = []
    rejected_psi    = []

    filter_names = []
    if framework_grid is not None and loop_radii_np is not None:
        filter_names.append(f"fw<{max_clash}")
    if max_closure is not None:
        filter_names.append(f"cl<{max_closure}Å")
    if max_intra is not None:
        filter_names.append(f"intra<{max_intra}")

    n_rejected_by = {'fw': 0, 'closure': 0, 'intra': 0}

    for attempt in range(max_attempts):
        if len(accepted_phi) >= n_structures:
            break

        phis, psis = zip(*[_sample_from_joint(probs_joint[i]) for i in range(n_loop)])
        phi_t = torch.tensor(phis, dtype=torch.float32).unsqueeze(0)
        psi_t = torch.cat([
            torch.FloatTensor(1).uniform_(-np.pi, np.pi),
            torch.tensor(psis[:-1], dtype=torch.float32),
            torch.FloatTensor(1).uniform_(-np.pi, np.pi),
        ]).unsqueeze(0)

        with torch.no_grad():
            N_t, CA_t, C_t = build_backbone(phi_t, psi_t, anchor_N, anchor_CA, anchor_C)

        passed    = True
        score_fw  = 0.0
        score_cl  = 0.0
        score_int = 0.0

        # Build atoms array (needed by fw and intra filters)
        atoms_np = np.empty((n_loop * 3, 3), dtype=np.float32)
        for i in range(n_loop):
            atoms_np[i*3]   = N_t[0, i].numpy()
            atoms_np[i*3+1] = CA_t[0, i].numpy()
            atoms_np[i*3+2] = C_t[0, i].numpy()

        # ── Closure distance (always computed for composite score) ─────
        N_virt = place_N_after(N_t[:, -1], CA_t[:, -1], C_t[:, -1], psi_t[:, -1])
        score_cl = float(torch.norm(N_virt - N_clos_t).item())
        if max_closure is not None and score_cl > max_closure:
            passed = False
            n_rejected_by['closure'] += 1

        # ── Framework clash ───────────────────────────────────────────
        if framework_grid is not None and loop_radii_np is not None:
            score_fw = framework_grid.query_score_np(atoms_np, loop_radii_np, k_clash=k_clash)
            if max_clash is not None and score_fw > max_clash:
                passed = False
                n_rejected_by['fw'] += 1

        # ── Intra-loop clash ──────────────────────────────────────────
        if max_intra is not None:
            score_int = float(compute_intraloop_clash_energy(
                N_t, CA_t, C_t, intra_idx_i, intra_idx_j,
                intra_radii, k_clash=k_clash, buffer=clash_buffer,
            ).item())
            if score_int > max_intra:
                passed = False
                n_rejected_by['intra'] += 1

        if passed:
            accepted_phi.append(phi_t.squeeze(0))
            accepted_psi.append(psi_t.squeeze(0))
        else:
            # Composite rejection score: weighted sum for ranking fallback
            composite = score_fw + 10.0 * score_cl + score_int
            rejected_scores.append(composite)
            rejected_phi.append(phi_t.squeeze(0))
            rejected_psi.append(psi_t.squeeze(0))

    n_accepted = len(accepted_phi)
    n_needed   = n_structures - n_accepted

    if n_needed > 0 and rejected_phi:
        order = np.argsort(rejected_scores)[:n_needed]
        for idx in order:
            accepted_phi.append(rejected_phi[idx])
            accepted_psi.append(rejected_psi[idx])

    while len(accepted_phi) < n_structures:
        phis, psis = zip(*[_sample_from_joint(probs_joint[i]) for i in range(n_loop)])
        accepted_phi.append(torch.tensor(phis, dtype=torch.float32))
        accepted_psi.append(torch.cat([
            torch.FloatTensor(1).uniform_(-np.pi, np.pi),
            torch.tensor(psis[:-1], dtype=torch.float32),
            torch.FloatTensor(1).uniform_(-np.pi, np.pi),
        ]))

    n_total = min(attempt + 1, max_attempts) if max_attempts > 0 else 0
    accept_rate = n_accepted / max(n_total, 1) * 100
    rej_str = "  ".join(f"{k}={v}" for k, v in n_rejected_by.items() if v > 0)
    print(f"      Filtered init [{', '.join(filter_names)}]: "
          f"{n_accepted}/{n_total} accepted ({accept_rate:.0f}%), "
          f"{n_needed} filled from best rejected")
    if rej_str:
        print(f"      Rejected by: {rej_str}")

    return torch.stack(accepted_phi[:n_structures]), torch.stack(accepted_psi[:n_structures])


# ─────────────────────────────────────────────────────────────────────────────
# Ensemble analysis
# ─────────────────────────────────────────────────────────────────────────────

def ensemble_diversity(
    ensemble: list, loop_start: int, loop_end: int,
) -> Tuple[np.ndarray, np.ndarray, float]:
    CA = np.stack([s[1][loop_start:loop_end] for s in ensemble])
    B  = len(CA)
    pw = np.zeros((B, B))
    for i in range(B):
        for j in range(i+1, B):
            d = float(np.sqrt(np.mean(np.sum((CA[i]-CA[j])**2, axis=1))))
            pw[i,j] = pw[j,i] = d
    mean_div = pw.sum(axis=1) / (B - 1)
    return pw, mean_div, float(pw.sum() / (B * (B-1)))


# ─────────────────────────────────────────────────────────────────────────────
# Optimization
# ─────────────────────────────────────────────────────────────────────────────

def optimize_torsions(
    phi_init:         torch.Tensor,
    psi_init:         torch.Tensor,
    anchor_N:         np.ndarray,
    anchor_CA:        np.ndarray,
    anchor_C:         np.ndarray,
    N_closure:        np.ndarray,
    probs_joint:      list,
    n_steps:          int   = 1000,
    lr_energy:        float = 0.05,
    lr_closure:       float = 0.20,
    closure_weight:   float = 50.0,
    eta_min:          float = 1e-4,
    n_frames:         int   = 0,
    # Grid-based framework clash
    framework_grid:   Optional[FrameworkGrid] = None,
    # Legacy KD-tree framework clash (backward compat)
    framework_coords: Optional[np.ndarray] = None,
    framework_radii:  Optional[np.ndarray] = None,
    loop_radii:       Optional[np.ndarray] = None,
    k_clash:          float = 100.0,
    clash_weight:     float = 1.0,
    clash_cutoff:     float = 8.0,
    clash_start_frac: float = 0.25,
    # Pulsed ramping
    n_pulses:         int   = 3,
    clash_floor_frac: float = 0.02,
    clash_cap:        float = DEFAULT_CLASH_CAP,
    clash_buffer:     float = DEFAULT_CLASH_BUFFER,
    # Boundary clash (anchor + closure atoms)
    boundary_coords:  Optional[np.ndarray] = None,   # (n_boundary, 3)
    boundary_radii:   Optional[np.ndarray] = None,    # (n_boundary,)
) -> Tuple[torch.Tensor, torch.Tensor, list]:
    """
    Batched Adam optimization of loop torsion angles.

    Schedulers
    ──────────
      LR      : CosineAnnealingLR  (peak → eta_min over n_steps)
      Closure : linear ramp  0 → closure_weight  over n_steps
      Clash   : pulsed ramp  (Rosetta-style) starting at clash_start_frac
                n_pulses cycles of floor→full weight

    Returns phi_best, psi_best (checkpointed by closure), trajectory frames.
    """
    from scipy.spatial import KDTree

    B, n = phi_init.shape

    phi      = phi_init.clone().requires_grad_(True)
    psi_body = psi_init[:, 1:n].clone().requires_grad_(True)
    psi_anc  = psi_init[:, 0:1].clone().requires_grad_(True)
    psi_clos = psi_init[:, n:n+1].clone().requires_grad_(True)

    optimizer = torch.optim.Adam([
        {'params': [phi, psi_body],     'lr': lr_energy},
        {'params': [psi_anc, psi_clos], 'lr': lr_closure},
    ])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_steps, eta_min=eta_min,
    )
    N_close_t = torch.tensor(N_closure, dtype=torch.float32).unsqueeze(0)

    # Determine clash mode
    use_grid   = framework_grid is not None
    use_legacy = (not use_grid
                  and framework_coords is not None
                  and framework_radii is not None
                  and loop_radii is not None)
    kdtree = KDTree(framework_coords) if use_legacy else None

    # Loop radii tensor for grid queries
    loop_radii_t = None
    if use_grid and loop_radii is not None:
        loop_radii_t = torch.tensor(loop_radii, dtype=torch.float32)

    intra_idx_i, intra_idx_j = _build_intraloop_exclusion_mask(n)
    intra_radii = torch.tensor(
        [VDW_RADII['N'], VDW_RADII['CA'], VDW_RADII['C']] * n,
        dtype=torch.float32,
    )

    # Boundary clash: anchor + closure atoms
    use_boundary = boundary_coords is not None and boundary_radii is not None
    bnd_coords_t = bnd_radii_t = bnd_idx_loop = bnd_idx_bnd = None
    if use_boundary:
        bnd_coords_t = torch.tensor(boundary_coords, dtype=torch.float32)
        bnd_radii_t  = torch.tensor(boundary_radii,  dtype=torch.float32)
        bnd_idx_loop, bnd_idx_bnd = _build_boundary_exclusion_pairs(
            n, len(boundary_radii),
        )

    clash_mode_str = "grid (precomputed)" if use_grid else (
        "legacy KD-tree" if use_legacy else "intra-loop only"
    )
    print(f"      B={B}  n_steps={n_steps}  "
          f"lr_energy={lr_energy}  lr_closure={lr_closure}  "
          f"closure_weight={closure_weight}")
    print(f"      Clash: {clash_mode_str}  k={k_clash}  weight={clash_weight}  "
          f"start={clash_start_frac:.0%}  pulses={n_pulses}  cap={clash_cap}Å")
    print(f"      Schedulers: LR=cosine  closure=linear  "
          f"clash=pulsed({n_pulses} cycles)")

    best_cl       = torch.full((B,), float('inf'))
    best_phi_ckpt = phi_init.clone()
    best_psi_ckpt = psi_init.clone()

    frame_steps: set = set()
    if n_frames > 0:
        dense = min(50, n_steps, n_frames)
        frame_steps = set(range(dense))
        remaining   = n_frames - dense
        if remaining > 0 and n_steps > dense:
            interval = max(1, (n_steps - dense) // remaining)
            frame_steps |= set(range(dense, n_steps, interval))
        frame_steps.add(n_steps - 1)

    trajectory = []

    for step in range(n_steps):
        optimizer.zero_grad()

        psi      = torch.cat([psi_anc, psi_body, psi_clos], dim=1)
        N, CA, C = build_backbone(phi, psi, anchor_N, anchor_CA, anchor_C)
        energy   = compute_energy(phi, psi, probs_joint).mean()

        progress  = step / max(n_steps - 1, 1)
        N_virt    = place_N_after(N[:, -1], CA[:, -1], C[:, -1], psi_clos[:, 0])
        closure   = torch.sum((N_virt - N_close_t) ** 2, dim=-1).mean()
        w_closure = closure_weight * progress
        loss      = energy + w_closure * closure

        e_intra = torch.tensor(0.0)
        e_fw    = torch.tensor(0.0)

        # Compute clash weight via pulsed schedule
        w_clash = pulsed_clash_weight(
            step, n_steps, clash_weight,
            clash_start_frac=clash_start_frac,
            n_pulses=n_pulses,
            floor_frac=clash_floor_frac,
        )

        # Intra-loop clash
        if len(intra_idx_i) > 0:
            e_intra = compute_intraloop_clash_energy(
                N, CA, C, intra_idx_i, intra_idx_j,
                intra_radii, k_clash=k_clash, cap=clash_cap,
                buffer=clash_buffer,
            ).mean()

        # Framework clash
        if use_grid and loop_radii_t is not None:
            loop_atoms = torch.stack([N, CA, C], dim=2).reshape(B, -1, 3)
            e_fw = framework_grid.query_energy(
                loop_atoms, loop_radii_t,
                k_clash=k_clash, cap=clash_cap,
            ).mean()
        elif use_legacy:
            e_fw = compute_framework_clash_energy(
                N, CA, C, loop_radii, framework_coords,
                framework_radii, kdtree, k_clash=k_clash, cutoff=clash_cutoff,
            ).mean()

        # Boundary clash: loop ↔ anchor/closure atoms
        e_bnd = torch.tensor(0.0)
        if use_boundary:
            e_bnd = compute_boundary_clash_energy(
                N, CA, C, bnd_coords_t, bnd_radii_t, intra_radii,
                bnd_idx_loop, bnd_idx_bnd,
                k_clash=k_clash, cap=clash_cap, buffer=clash_buffer,
            ).mean()

        if w_clash > 0:
            loss = loss + w_clash * (e_intra + e_fw + e_bnd)

        with torch.no_grad():
            cl_now   = torch.norm(N_virt - N_close_t, dim=-1)
            psi_full = torch.cat([psi_anc, psi_body, psi_clos], dim=1)
            improved = cl_now < best_cl
            best_cl[improved]       = cl_now[improved]
            best_phi_ckpt[improved] = phi.detach()[improved]
            best_psi_ckpt[improved] = psi_full.detach()[improved]

        if step in frame_steps:
            with torch.no_grad():
                trajectory.append((
                    step,
                    N.detach().clone(),
                    CA.detach().clone(),
                    C.detach().clone(),
                    compute_energy(phi, psi, probs_joint).detach().clone(),
                    cl_now.detach().clone(),
                ))

        loss.backward()
        optimizer.step()
        scheduler.step()

        if step % 50 == 0 or step == n_steps - 1:
            with torch.no_grad():
                cl = torch.norm(
                    place_N_after(N[:,-1], CA[:,-1], C[:,-1], psi_clos[:,0])
                    - N_close_t, dim=-1)

                # Diagnostic: count actual VdW overlaps (not just buffer proximity)
                n_clashes_intra = 0
                n_clashes_fw    = 0
                n_clashes_bnd   = 0
                if len(intra_idx_i) > 0:
                    atoms_d = torch.stack([N, CA, C], dim=2).reshape(B, -1, 3)
                    d_min_i = 0.8 * (intra_radii[intra_idx_i] + intra_radii[intra_idx_j])
                    dist_i  = torch.norm(atoms_d[:, intra_idx_i] - atoms_d[:, intra_idx_j], dim=-1)
                    n_clashes_intra = int((dist_i < d_min_i.unsqueeze(0)).sum().item()) // B
                if use_boundary:
                    atoms_d = torch.stack([N, CA, C], dim=2).reshape(B, -1, 3)
                    d_min_b = 0.8 * (intra_radii[bnd_idx_loop] + bnd_radii_t[bnd_idx_bnd])
                    dist_b  = torch.norm(atoms_d[:, bnd_idx_loop] - bnd_coords_t[bnd_idx_bnd].unsqueeze(0), dim=-1)
                    n_clashes_bnd = int((dist_b < d_min_b.unsqueeze(0)).sum().item()) // B

            clash_val = (f"  E_intra={e_intra.item():.3f}"
                         f"  E_fw={e_fw.item():.3f}"
                         f"  E_bnd={e_bnd.item():.3f}"
                         f"  w_clash={w_clash:.3f}"
                         f"  overlaps={n_clashes_intra}i+{n_clashes_bnd}b")
            print(f"        step {step:4d}: E={energy.item():.3f}  "
                  f"w_cl={w_closure:.1f}  "
                  f"cl_mean={cl.mean().item():.4f}Å  "
                  f"cl_best={cl.min().item():.4f}Å  "
                  f"lr={scheduler.get_last_lr()[0]:.5f}{clash_val}")

    never = best_cl == float('inf')
    if never.any():
        psi_f = torch.cat([psi_anc, psi_body, psi_clos], dim=1).detach()
        best_phi_ckpt[never] = phi.detach()[never]
        best_psi_ckpt[never] = psi_f[never]

    return best_phi_ckpt, best_psi_ckpt, trajectory


def _optimize_closure_only(
    phi_batch: torch.Tensor, psi_batch: torch.Tensor,
    anchor_N: np.ndarray, anchor_CA: np.ndarray, anchor_C: np.ndarray,
    N_closure: np.ndarray,
    n_steps: int = 500, lr: float = 0.20, eta_min: float = 1e-4,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Optimize only boundary psi angles for baseline closure refinement."""
    B, n     = phi_batch.shape
    N_clos_t = torch.tensor(N_closure, dtype=torch.float32).unsqueeze(0)
    phi_f    = phi_batch.detach()
    body_f   = psi_batch[:, 1:n].detach()
    psi_anc  = psi_batch[:, 0:1].clone().requires_grad_(True)
    psi_clos = psi_batch[:, n:n+1].clone().requires_grad_(True)

    opt = torch.optim.Adam([psi_anc, psi_clos], lr=lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=n_steps, eta_min=eta_min,
    )
    best_cl   = torch.full((B,), float('inf'))
    best_anc  = psi_anc.detach().clone()
    best_clos = psi_clos.detach().clone()

    for _ in range(n_steps):
        opt.zero_grad()
        psi      = torch.cat([psi_anc, body_f, psi_clos], dim=1)
        N, CA, C = build_backbone(phi_f, psi, anchor_N, anchor_CA, anchor_C)
        N_v      = place_N_after(N[:,-1], CA[:,-1], C[:,-1], psi_clos[:,0])
        torch.sum((N_v - N_clos_t)**2, dim=-1).mean().backward()
        opt.step();  sch.step()
        with torch.no_grad():
            cl = torch.norm(N_v - N_clos_t, dim=-1)
            improved = cl < best_cl
            best_cl[improved]   = cl[improved]
            best_anc[improved]  = psi_anc.detach()[improved]
            best_clos[improved] = psi_clos.detach()[improved]

    return phi_f, torch.cat([best_anc, body_f, best_clos], dim=1)


# ─────────────────────────────────────────────────────────────────────────────
# Ensemble packing
# ─────────────────────────────────────────────────────────────────────────────

def _pack_ensemble(
    phi_batch, psi_batch, probs_joint, n_loop,
    N_fb, CA_fb, C_fb, O_fb,
    N_fa, CA_fa, C_fa, O_fa,
    anc_N, anc_CA, anc_C, N_clos_t,
) -> list:
    with torch.no_grad():
        N_t, CA_t, C_t = build_backbone(phi_batch, psi_batch, anc_N, anc_CA, anc_C)
        N_v    = place_N_after(N_t[:,-1], CA_t[:,-1], C_t[:,-1], psi_batch[:,-1])
        cl_all = torch.norm(N_v - N_clos_t, dim=-1)
        e_all  = compute_energy(phi_batch, psi_batch, probs_joint)

    ensemble = []
    for idx in range(phi_batch.shape[0]):
        N_np  = N_t[idx].numpy();  CA_np = CA_t[idx].numpy()
        C_np  = C_t[idx].numpy();  O_np  = compute_O_atoms(N_np, CA_np, C_np)

        # Normalize angles to (-180°, 180°] to prevent out-of-range plot positions
        phi_np = np.degrees(np.arctan2(
            np.sin(phi_batch[idx].numpy()),
            np.cos(phi_batch[idx].numpy()),
        ))
        psi_np = np.degrees(np.arctan2(
            np.sin(psi_batch[idx, 1:n_loop+1].numpy()),
            np.cos(psi_batch[idx, 1:n_loop+1].numpy()),
        ))

        ensemble.append((
            np.vstack([N_fb,  N_np,  N_fa]),
            np.vstack([CA_fb, CA_np, CA_fa]),
            np.vstack([C_fb,  C_np,  C_fa]),
            np.vstack([O_fb,  O_np,  O_fa]),
            phi_np,
            psi_np,
            float(e_all[idx].item()),
            float(cl_all[idx].item()),
        ))
    return ensemble


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def random_ensemble(
    full_sequence:   str,
    loop_start:      int,
    loop_end:        int,
    N_flank_before:  np.ndarray,
    CA_flank_before: np.ndarray,
    C_flank_before:  np.ndarray,
    O_flank_before:  np.ndarray,
    N_flank_after:   np.ndarray,
    CA_flank_after:  np.ndarray,
    C_flank_after:   np.ndarray,
    O_flank_after:   np.ndarray,
    model_or_router,
    params=None,
    n_structures:  int   = 10,
    mode:          str   = 'uniform',
    closure_steps: int   = 500,
    closure_lr:    float = 0.20,
    seed:          int   = None,
) -> Tuple[list, list]:
    """
    Build a random baseline ensemble.
    mode='uniform'      — sample φ/ψ uniformly
    mode='model_sample' — sample jointly from learned distribution
    Returns (ensemble, probs_joint).
    """
    assert mode in ('uniform', 'model_sample')
    router   = _to_router(model_or_router, params)
    loop_seq = full_sequence[loop_start:loop_end]
    n_loop   = len(loop_seq)
    anc_N    = N_flank_before[-1].copy()
    anc_CA   = CA_flank_before[-1].copy()
    anc_C    = C_flank_before[-1].copy()
    N_clos   = N_flank_after[0].copy()
    N_clos_t = torch.tensor(N_clos, dtype=torch.float32)

    probs_joint = cache_energy_distributions(router, loop_seq)

    phi_rows, psi_rows = [], []
    for i in range(n_structures):
        if seed is not None:
            torch.manual_seed(seed + i);  np.random.seed(seed + i)
        if mode == 'uniform':
            phi_rows.append(torch.FloatTensor(n_loop).uniform_(-np.pi, np.pi))
            psi_rows.append(torch.FloatTensor(n_loop+1).uniform_(-np.pi, np.pi))
        else:
            phis, psis = zip(*[_sample_from_joint(probs_joint[j])
                                for j in range(n_loop)])
            phi_rows.append(torch.tensor(phis, dtype=torch.float32))
            psi_rows.append(torch.cat([
                torch.FloatTensor(1).uniform_(-np.pi, np.pi),
                torch.tensor(psis[:-1], dtype=torch.float32),
                torch.FloatTensor(1).uniform_(-np.pi, np.pi),
            ]))

    phi_b = torch.stack(phi_rows);  psi_b = torch.stack(psi_rows)
    phi_b, psi_b = _optimize_closure_only(
        phi_b, psi_b, anc_N, anc_CA, anc_C, N_clos,
        n_steps=closure_steps, lr=closure_lr,
    )
    ensemble = _pack_ensemble(
        phi_b, psi_b, probs_joint, n_loop,
        N_flank_before, CA_flank_before, C_flank_before, O_flank_before,
        N_flank_after,  CA_flank_after,  C_flank_after,  O_flank_after,
        anc_N, anc_CA, anc_C, N_clos_t,
    )
    print(f"    Baseline ({mode}+closure): {n_structures} structures")
    return ensemble, probs_joint


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
    # New parameters
    n_pulses:         int   = 3,
    clash_floor_frac: float = 0.02,
    clash_cap:        float = DEFAULT_CLASH_CAP,
    clash_buffer:     float = DEFAULT_CLASH_BUFFER,
    grid_resolution:  float = 0.5,
    max_init_clash:   float = 50.0,
    max_init_closure: float = None,    # Å — max closure distance at init (None=disabled)
    max_init_intra:   float = None,    # max intra-loop clash at init (None=disabled)
    max_init_attempts: int  = 500,
    # Pre-built grid (avoids duplicate build when caller already has one)
    framework_grid:   Optional[FrameworkGrid] = None,
) -> Tuple[list, list, list]:
    """
    Main entry point: energy-guided NeRF loop ensemble generation.

    Changes from previous version:
      - Framework clash uses precomputed 3D grid (O(1) per atom, differentiable)
      - Pulsed Rosetta-style clash ramping (n_pulses cycles of floor→full)
      - Linear-capped soft-core potential (bounded gradients)
      - Rejection-sampled initialisation (clash-filtered)
      - Boundary clash: loop ↔ anchor/closure residue atoms (fixes blind spot)
      - Accepts pre-built FrameworkGrid to avoid duplicate construction

    Returns:
        ensemble    : list of (N,CA,C,O, phi_deg,psi_deg, energy, closure)
        probs_joint : list of (N_BINS,N_BINS) per residue
        trajectory  : batched frame list (empty when n_frames=0)
    """
    router   = _to_router(model_or_router, params)
    loop_seq = full_sequence[loop_start:loop_end]
    n_loop   = len(loop_seq)
    anc_N    = N_flank_before[-1].copy()
    anc_CA   = CA_flank_before[-1].copy()
    anc_C    = C_flank_before[-1].copy()
    N_clos   = N_flank_after[0].copy()
    N_clos_t = torch.tensor(N_clos, dtype=torch.float32)

    print(f"\n  NeRF refinement: {full_sequence}")
    print(f"    Loop: {loop_seq} ({n_loop} res)  "
          f"anchor–closure: {np.linalg.norm(C_flank_before[-1]-N_flank_after[0]):.2f}Å")

    probs_joint = cache_energy_distributions(router, loop_seq)

    if seed is not None:
        torch.manual_seed(seed);  np.random.seed(seed)

    # Build framework grid (reuse if caller already built one)
    fw_grid = framework_grid
    loop_radii: Optional[np.ndarray] = None

    if fw_grid is None and framework_coords is not None and framework_radii is not None:
        fw_grid = build_framework_grid(
            framework_coords, framework_radii,
            resolution=grid_resolution, buffer=clash_buffer,
        )

    if framework_coords is not None and framework_radii is not None:
        loop_radii = np.tile(
            [VDW_RADII['N'], VDW_RADII['CA'], VDW_RADII['C']], n_loop
        ).astype(np.float32)

    # Build boundary atoms (anchor residue + closure residue)
    CA_clos = CA_flank_after[0].copy() if len(CA_flank_after) > 0 else None
    C_clos  = C_flank_after[0].copy()  if len(C_flank_after) > 0  else None
    bnd_coords, bnd_radii = _build_boundary_atoms(
        anc_N, anc_CA, anc_C, N_clos,
        CA_closure=CA_clos, C_closure=C_clos,
    )
    print(f"    Boundary clash: {len(bnd_radii)} atoms "
          f"(anchor N/CA/C + closure N"
          f"{'/CA' if CA_clos is not None else ''}"
          f"{'/C' if C_clos is not None else ''})")

    # Initialise from model distribution with clash filtering
    print(f"    Initialising {n_structures} structures from model distribution "
          f"(clash filter: {'ON' if fw_grid else 'OFF'})...")

    phi_init, psi_init = _clash_filtered_init(
        probs_joint, n_loop, anc_N, anc_CA, anc_C,
        N_closure      = N_clos,
        n_structures   = n_structures,
        framework_grid = fw_grid,
        loop_radii_np  = loop_radii,
        k_clash        = k_clash,
        max_clash      = max_init_clash,
        max_closure    = max_init_closure,
        max_intra      = max_init_intra,
        max_attempts   = max_init_attempts,
        clash_buffer   = clash_buffer,
    )

    phi_opt, psi_opt, trajectory = optimize_torsions(
        phi_init, psi_init,
        anc_N, anc_CA, anc_C, N_clos, probs_joint,
        n_steps=n_steps, lr_energy=lr_energy, lr_closure=lr_closure,
        closure_weight=closure_weight, eta_min=eta_min, n_frames=n_frames,
        framework_grid   = fw_grid,
        loop_radii       = loop_radii,
        k_clash          = k_clash,
        clash_weight     = clash_weight,
        clash_cutoff     = clash_cutoff,
        clash_start_frac = clash_start_frac,
        n_pulses         = n_pulses,
        clash_floor_frac = clash_floor_frac,
        clash_cap        = clash_cap,
        clash_buffer     = clash_buffer,
        boundary_coords  = bnd_coords,
        boundary_radii   = bnd_radii,
    )

    ensemble = _pack_ensemble(
        phi_opt, psi_opt, probs_joint, n_loop,
        N_flank_before, CA_flank_before, C_flank_before, O_flank_before,
        N_flank_after,  CA_flank_after,  C_flank_after,  O_flank_after,
        anc_N, anc_CA, anc_C, N_clos_t,
    )
    cl = [e[-1] for e in ensemble]
    print(f"    Closure: mean={np.mean(cl):.4f}Å  best={min(cl):.4f}Å")
    return ensemble, probs_joint, trajectory