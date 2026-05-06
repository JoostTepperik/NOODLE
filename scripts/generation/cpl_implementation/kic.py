"""
Analytical Kinematic Closure (KIC) for protein loop backbone.

Identical algorithm to the original, but the outer t1 grid scan is
fully vectorised — no Python loop over grid points.  Everything else
(Brent refinement, analytic inner/t3 solve) is unchanged.

See the original module docstring for algorithm details.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

from nerf import BOND_ANGLES_RAD as _A, BOND_LENGTHS as _L, build_loop, nerf

# ─────────────────────────────────────────────────────────────────────────────
# Pre-computed geometry constant
# ─────────────────────────────────────────────────────────────────────────────

_D_CA_N_SQ: float = (
    _L['CA_C'] ** 2
    + _L['C_N']  ** 2
    - 2.0 * _L['CA_C'] * _L['C_N'] * np.cos(_A['CA_C_N'])
)

_T1_DEDUP_RAD: float = 1e-6


# ─────────────────────────────────────────────────────────────────────────────
# Vectorised NeRF over a grid of dihedrals
# ─────────────────────────────────────────────────────────────────────────────

def _nerf_vec(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    bond_length: float,
    bond_angle: float,
    dihedrals: np.ndarray,          # (N,)
) -> np.ndarray:                    # (N, 3)
    """
    Vectorised NeRF placement over a (N,) array of dihedral angles.

    Equivalent to calling nerf(a, b, c, bond_length, bond_angle, d)
    for each d in dihedrals, but implemented as a single numpy broadcast
    with no Python loop.

    Matches nerf.py exactly:
        d = c + R @ [-bl*cos(ba), bl*sin(ba)*cos(d), bl*sin(ba)*sin(d)]
    where R = [bc_hat | n_hat | m_hat].
    """
    bc     = c - b
    bc_hat = bc / np.linalg.norm(bc)

    n = a - b
    n = n - np.dot(n, bc_hat) * bc_hat
    n_len = np.linalg.norm(n)
    if n_len < 1e-8:
        tmp   = (np.array([0., 0., 1.]) if abs(bc_hat[2]) < 0.9
                 else np.array([1., 0., 0.]))
        n     = tmp - np.dot(tmp, bc_hat) * bc_hat
        n_len = np.linalg.norm(n)
    n_hat = n / n_len
    m_hat = np.cross(bc_hat, n_hat)

    # Pre-compute scalar geometry
    r      = bond_length * np.sin(bond_angle)   # circle radius
    center = c - bond_length * np.cos(bond_angle) * bc_hat  # (3,)

    # dihedrals: (N,)  →  placed atoms: (N, 3)
    cos_d = np.cos(dihedrals)   # (N,)
    sin_d = np.sin(dihedrals)   # (N,)
    return center + r * (cos_d[:, None] * n_hat + sin_d[:, None] * m_hat)


# ─────────────────────────────────────────────────────────────────────────────
# Vectorised bridge: N[k+1] and Cα[k+1] over a grid of t1 values
# ─────────────────────────────────────────────────────────────────────────────

def _bridge_atoms_vec(
    t1_grid: np.ndarray,            # (N,)
    N_k: np.ndarray,
    CA_k: np.ndarray,
    C_k: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:  # (N, 3), (N, 3)
    """
    Fully vectorised placement of N[k+1] and Cα[k+1] over all t1 values.

    N[k+1] placed via _nerf_vec — no loop.

    Cα[k+1] uses ω = π, so sin(ω) = 0 and the placement reduces to:
        Cα[k+1] = centre(t1) - r · n_hat(t1)
    where centre and n_hat are functions of N_k1(t1).  We compute these
    entirely with broadcasting — no Python loop over grid points.

    Frame for Cα[k+1]:  reference atoms are (CA_k, C_k, N_k1).
        bc_hat = (N_k1 - C_k) / |N_k1 - C_k|      — varies with t1
        n_raw  = (C_k - CA_k) − dot(C_k-CA_k, bc_hat)·bc_hat
        n_hat  = n_raw / |n_raw|
        centre = N_k1 − L_NCA·cos(A_CNC)·bc_hat
        Cα     = centre − L_NCA·sin(A_CNC)·n_hat   (cos(π)=−1, sin(π)=0)
    """
    # ── N[k+1] over full grid ────────────────────────────────────────────
    N_k1_all = _nerf_vec(N_k, CA_k, C_k, _L['C_N'], _A['CA_C_N'], t1_grid)
    # (N, 3)

    # ── Cα[k+1]: vectorised frame construction ───────────────────────────
    bl  = _L['N_CA']
    ba  = _A['C_N_CA']
    cos_ba = np.cos(ba)
    sin_ba = np.sin(ba)

    # bc = N_k1 - C_k,  shape (N, 3)
    bc     = N_k1_all - C_k                          # (N, 3)
    bc_len = np.linalg.norm(bc, axis=1, keepdims=True)  # (N, 1)
    bc_hat = bc / bc_len                             # (N, 3)

    # nerf(CA_k, C_k, N_k1, ...) uses a=CA_k, b=C_k, c=N_k1
    # n = a - b = CA_k - C_k  (constant across all t1)
    ca_to_c = CA_k - C_k                             # (3,)
    dots    = (bc_hat * ca_to_c).sum(axis=1, keepdims=True)  # (N, 1)
    n_raw   = ca_to_c - dots * bc_hat                # (N, 3)
    n_len   = np.linalg.norm(n_raw, axis=1, keepdims=True)   # (N, 1)

    # Handle near-collinear edge case (rare; fall back to arbitrary perp)
    bad = (n_len < 1e-8).ravel()
    if bad.any():
        tmp = (np.array([0., 0., 1.])
               if abs(bc_hat[bad][0, 2]) < 0.9
               else np.array([1., 0., 0.]))
        n_raw[bad] = tmp - (bc_hat[bad] * tmp).sum(axis=1, keepdims=True) * bc_hat[bad]
        n_len[bad] = np.linalg.norm(n_raw[bad], axis=1, keepdims=True)

    n_hat = n_raw / n_len                            # (N, 3)

    # centre = N_k1 - bl*cos(ba)*bc_hat
    centre   = N_k1_all - bl * cos_ba * bc_hat      # (N, 3)
    # CA_k1 = centre + bl*sin(ba)*cos(π)*n_hat = centre - bl*sin(ba)*n_hat
    CA_k1_all = centre - bl * sin_ba * n_hat        # (N, 3)

    return N_k1_all, CA_k1_all


# ─────────────────────────────────────────────────────────────────────────────
# Vectorised outer residual f(t1) over the full grid
# ─────────────────────────────────────────────────────────────────────────────

def _outer_f_grid(
    t1_grid: np.ndarray,
    N_k: np.ndarray,
    CA_k: np.ndarray,
    C_k: np.ndarray,
    target_N: np.ndarray,
) -> np.ndarray:
    """
    f(t1) = |Cα[k+1](t1) − target_N|² − d²  evaluated over the full grid.

    Returns (N,) array of residuals — sign changes locate roots.
    """
    _, CA_k1_all = _bridge_atoms_vec(t1_grid, N_k, CA_k, C_k)
    diff = CA_k1_all - target_N                  # (N, 3)
    return (diff * diff).sum(axis=1) - _D_CA_N_SQ  # (N,)


# ─────────────────────────────────────────────────────────────────────────────
# Scalar helpers for Brent + inner/t3 solve (unchanged from original)
# ─────────────────────────────────────────────────────────────────────────────

def _bridge_atoms(t1, N_k, CA_k, C_k):
    N_k1  = nerf(N_k,  CA_k, C_k,   _L['C_N'],  _A['CA_C_N'], t1)
    CA_k1 = nerf(CA_k, C_k,  N_k1,  _L['N_CA'], _A['C_N_CA'], np.pi)
    return N_k1, CA_k1


def _outer_f(t1, N_k, CA_k, C_k, target_N):
    _, CA_k1 = _bridge_atoms(t1, N_k, CA_k, C_k)
    diff = CA_k1 - target_N
    return float(np.dot(diff, diff)) - _D_CA_N_SQ


def _nerf_circle(a, b, c, bond_length, bond_angle):
    bc     = c - b
    bc_hat = bc / np.linalg.norm(bc)
    n = a - b
    n = n - np.dot(n, bc_hat) * bc_hat
    n_len = np.linalg.norm(n)
    if n_len < 1e-8:
        tmp   = (np.array([0., 0., 1.]) if abs(bc_hat[2]) < 0.9
                 else np.array([1., 0., 0.]))
        n     = tmp - np.dot(tmp, bc_hat) * bc_hat
        n_len = np.linalg.norm(n)
    n_hat = n / n_len
    m_hat = np.cross(bc_hat, n_hat)
    radius = bond_length * np.sin(bond_angle)
    D = c - bond_length * np.cos(bond_angle) * bc_hat
    E = radius * n_hat
    F = radius * m_hat
    return D, E, F, n_hat, m_hat, radius


def _solve_inner(t1, N_k, CA_k, C_k, target_N):
    N_k1, CA_k1 = _bridge_atoms(t1, N_k, CA_k, C_k)
    D2, E2, F2, _, _, r_C = _nerf_circle(
        C_k, N_k1, CA_k1, _L['CA_C'], _A['N_CA_C'])
    delta = target_N - D2
    P = float(np.dot(delta, E2))
    Q = float(np.dot(delta, F2))
    R = 0.5 * (float(np.dot(delta, delta)) + r_C ** 2 - _L['C_N'] ** 2)
    rho_sq = P * P + Q * Q
    if rho_sq < 1e-20:
        return []
    rho = float(np.sqrt(rho_sq))
    arg = float(np.clip(R / rho, -1.0, 1.0))
    if abs(R / rho) > 1.0 + 1e-6:
        return []
    phi_ang     = float(np.arctan2(Q, P))
    delta_angle = float(np.arccos(arg))
    return [(phi_ang + sign * delta_angle,) for sign in (-1.0, +1.0)]


def _solve_t3(t1, t2, N_k, CA_k, C_k, target_N):
    N_k1, CA_k1 = _bridge_atoms(t1, N_k, CA_k, C_k)
    C_k1 = nerf(C_k, N_k1, CA_k1, _L['CA_C'], _A['N_CA_C'], t2)
    D3, E3, F3, _, _, r3 = _nerf_circle(
        N_k1, CA_k1, C_k1, _L['C_N'], _A['CA_C_N'])
    vec  = target_N - D3
    r3_sq = r3 ** 2 + 1e-30
    t3   = float(np.arctan2(
        np.dot(vec, F3) / r3_sq,
        np.dot(vec, E3) / r3_sq,
    ))
    placed = D3 + np.cos(t3) * E3 + np.sin(t3) * F3
    err    = float(np.linalg.norm(placed - target_N))
    return t3, err


# ─────────────────────────────────────────────────────────────────────────────
# Main KIC tripeptide closure  (vectorised outer scan)
# ─────────────────────────────────────────────────────────────────────────────

def kic_tripeptide_close(
    N_k:      np.ndarray,
    CA_k:     np.ndarray,
    C_k:      np.ndarray,
    target_N: np.ndarray,
    n_grid:   int   = 3600,
    tol:      float = 0.05,
) -> list[tuple[float, float, float, float]]:
    """
    Find all KIC solutions (t1, t2, t3) for a two-residue bridge.

    Identical interface to the original, but the outer grid scan is
    fully vectorised — f(t1) is evaluated for all grid points in a
    single numpy call before any Python branching.
    """
    t1_grid = np.linspace(-np.pi, np.pi, n_grid, endpoint=False)

    # ── 1. Vectorised outer scan ──────────────────────────────────────────
    f_vals = _outer_f_grid(t1_grid, N_k, CA_k, C_k, target_N)  # (N,) — no loop

    solutions: list      = []
    seen_t1: list[float] = []

    # Locate sign changes
    sign_changes = np.where(f_vals[:-1] * f_vals[1:] <= 0)[0]

    for i in sign_changes:
        try:
            t1_sol = brentq(
                lambda t: _outer_f(t, N_k, CA_k, C_k, target_N),
                t1_grid[i], t1_grid[i + 1],
                xtol=1e-9, rtol=1e-9, maxiter=200,
            )
        except ValueError:
            continue

        if any(abs(t1_sol - ts) < _T1_DEDUP_RAD for ts in seen_t1):
            continue
        seen_t1.append(t1_sol)

        for (t2_sol,) in _solve_inner(t1_sol, N_k, CA_k, C_k, target_N):
            t3_sol, err = _solve_t3(t1_sol, t2_sol, N_k, CA_k, C_k, target_N)
            if err > tol:
                continue
            solutions.append((t1_sol, t2_sol, t3_sol, err))

    return solutions


# ─────────────────────────────────────────────────────────────────────────────
# Full-loop KIC  (unchanged interface)
# ─────────────────────────────────────────────────────────────────────────────

def kic_close_given_torsions(
    phi:      np.ndarray,
    psi:      np.ndarray,
    prev_N:   np.ndarray,
    prev_CA:  np.ndarray,
    prev_C:   np.ndarray,
    psi_prev: float,
    target_N: np.ndarray,
    n_grid:   int   = 3600,
    tol:      float = 0.05,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    KIC closure for a full loop — identical interface to the original.
    See original module docstring for full parameter documentation.
    """
    n_loop = len(phi)
    if n_loop < 2:
        raise ValueError(
            f"kic_close_given_torsions requires n_loop ≥ 2; got {n_loop}."
        )

    pivot_k = n_loop - 2

    psi_nterm = np.empty(pivot_k + 1)
    psi_nterm[:pivot_k] = psi[:pivot_k]
    psi_nterm[pivot_k]  = psi[pivot_k]

    N_frag, CA_frag, C_frag, _ = build_loop(
        prev_N, prev_CA, prev_C, psi_prev,
        phi[: pivot_k + 1], psi_nterm,
    )
    N_k  = N_frag[pivot_k]
    CA_k = CA_frag[pivot_k]
    C_k  = C_frag[pivot_k]

    raw = kic_tripeptide_close(N_k, CA_k, C_k, target_N, n_grid, tol)

    results: list = []
    for t1, t2, t3, _err in raw:
        phi_sol = phi.copy()
        psi_sol = psi.copy()
        psi_sol[pivot_k]     = t1
        phi_sol[pivot_k + 1] = t2
        psi_sol[pivot_k + 1] = t3
        results.append((phi_sol, psi_sol))

    return results