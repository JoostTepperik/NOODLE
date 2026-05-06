"""
tripeptide_loop_closure.py

Kinematic closure for protein backbone loops using NeRF geometry.
Solves for 6 pivot torsion angles (φ,ψ at 3 pivots) that close the
loop exactly, using scipy.optimize.fsolve with multiple random starts
to find up to 16 solutions.

Works directly in NeRF's coordinate/torsion convention.
No PyRosetta dependency.
"""

from __future__ import annotations
import numpy as np
from scipy.optimize import fsolve
from typing import List, Tuple, Optional


# ─── Ideal geometry (must match NeRF exactly) ────────────────────────────────

BL_CN  = 1.329
BL_NCA = 1.458
BL_CAC = 1.525
BA_NCC = 1.9408061282176945   # ~111.2°
BA_CNC = 2.1240656996770992   # ~121.7°
BA_CCN = 2.028072590817411    # ~116.2°
OMEGA  = np.pi


# ─── NeRF atom placement (numpy, matches loop_modeling_nerf exactly) ─────────

def _place_atom(a, b, c, bl, ba, tor):
    """Place atom D given A,B,C, bond C-D=bl, angle B-C-D=ba, dihedral A-B-C-D=tor."""
    bc   = c - b
    bc_n = bc / (np.linalg.norm(bc) + 1e-15)
    n    = np.cross(b - a, bc)
    n    = n / (np.linalg.norm(n) + 1e-15)
    m    = np.cross(n, bc_n)
    d_local = np.array([
        -np.cos(ba),
         np.sin(ba) * np.cos(tor),
        -np.sin(ba) * np.sin(tor),
    ]) * bl
    return c + np.column_stack([bc_n, m, n]) @ d_local


def _build_chain(phi, psi_nerf, anc_N, anc_CA, anc_C):
    """
    Build backbone from NeRF-convention torsion angles.
    phi:      (n,) radians
    psi_nerf: (n+1,) radians — psi_nerf[i] places N_i from previous C
    Returns N, CA, C each (n, 3).
    """
    n = len(phi)
    N = np.zeros((n, 3)); CA = np.zeros((n, 3)); C = np.zeros((n, 3))
    a3, a2, a1 = anc_N.copy(), anc_CA.copy(), anc_C.copy()
    for i in range(n):
        Ni  = _place_atom(a3, a2, a1, BL_CN,  BA_CCN, psi_nerf[i])
        CAi = _place_atom(a2, a1, Ni,  BL_NCA, BA_CNC, OMEGA)
        Ci  = _place_atom(a1, Ni, CAi,  BL_CAC, BA_NCC, phi[i])
        N[i] = Ni; CA[i] = CAi; C[i] = Ci
        a3, a2, a1 = Ni, CAi, Ci
    return N, CA, C


def _inverse_nerf_torsion(a, b, c, d, bl, ba):
    """Given atoms A,B,C and target D, compute the torsion angle for _place_atom."""
    bc   = c - b
    bc_n = bc / (np.linalg.norm(bc) + 1e-15)
    n_   = np.cross(b - a, bc)
    nn   = np.linalg.norm(n_)
    if nn < 1e-10:
        return 0.0
    n_   = n_ / nn
    m_   = np.cross(n_, bc_n)
    R    = np.column_stack([bc_n, m_, n_])
    d_vec = (d - c) / (bl + 1e-15)
    d_local = R.T @ d_vec  # use R^T instead of solve (R is orthonormal)
    sin_ba = np.sin(ba)
    if abs(sin_ba) < 1e-12:
        return 0.0
    cos_tor =  d_local[1] / sin_ba
    sin_tor = -d_local[2] / sin_ba
    return np.arctan2(sin_tor, cos_tor)


# ─── TLC solver ──────────────────────────────────────────────────────────────

def solve_tlc(
    anchor_N:      np.ndarray,
    anchor_CA:     np.ndarray,
    anchor_C:      np.ndarray,
    closure_N:     np.ndarray,
    closure_CA:    np.ndarray,
    closure_C:     np.ndarray,
    phi:           np.ndarray,    # (n_loop,) all phi, radians
    psi_nerf:      np.ndarray,    # (n_loop+1,) NeRF psi convention
    pivot_indices: List[int] = None,
    n_starts:      int = 64,
    tol:           float = 0.01,  # Å — convergence tolerance
) -> List[Tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Solve loop closure for 3 pivot residues.

    Finds all sets of pivot (φ,ψ) that close the chain from anchor to
    closure target, with non-pivot torsions held fixed.

    Returns list of (phi, psi_nerf, closure_dist, N, CA, C) tuples.
    """
    n = len(phi)
    if pivot_indices is None:
        pivot_indices = [0, n // 2, n - 1]
    p1, p2, p3 = pivot_indices

    # Compute the closure torsion: what psi_nerf[n] would need to be
    # to place the virtual N at closure_N from the last C
    # First build the chain to get the last atoms
    N_init, CA_init, C_init = _build_chain(phi, psi_nerf, anchor_N, anchor_CA, anchor_C)
    closure_psi = _inverse_nerf_torsion(
        N_init[-1], CA_init[-1], C_init[-1], closure_N, BL_CN, BA_CCN,
    )

    # Extended psi: n+2 entries (original n+1 plus the closure psi)
    psi_ext = np.append(psi_nerf, closure_psi)

    # The 6 unknowns: φ_{p1}, ψ_{p1}, φ_{p2}, ψ_{p2}, φ_{p3}, ψ_{p3}
    # In NeRF convention:
    #   φ_{pi} = phi[pi]
    #   ψ_{pi} = psi_ext[pi+1]  (places N_{pi+1} from C_{pi})
    # For the LAST pivot p3:
    #   ψ_{p3} = psi_ext[p3+1]
    #   If p3 = n-1, then psi_ext[n] is the closure psi

    def residual(x):
        phi_t = phi.copy()
        psi_t = psi_ext.copy()

        phi_t[p1] = x[0]; psi_t[p1+1] = x[1]
        phi_t[p2] = x[2]; psi_t[p2+1] = x[3]
        phi_t[p3] = x[4]; psi_t[p3+1] = x[5]

        N_t, CA_t, C_t = _build_chain(phi_t, psi_t[:n+1], anchor_N, anchor_CA, anchor_C)

        # Virtual N after loop: placed from last residue using psi_t[n]
        # (which is the closure torsion, possibly modified by the solver)
        N_virt = _place_atom(N_t[-1], CA_t[-1], C_t[-1],
                              BL_CN, BA_CCN, psi_t[n])
        # Virtual CA after closure N
        CA_virt = _place_atom(CA_t[-1], C_t[-1], N_virt,
                               BL_NCA, BA_CNC, OMEGA)

        return np.concatenate([N_virt - closure_N, CA_virt - closure_CA])

    # Initial guess
    x0_base = np.array([
        phi[p1], psi_ext[p1+1],
        phi[p2], psi_ext[p2+1],
        phi[p3], psi_ext[p3+1],
    ])

    solutions = []
    seen = []

    for trial in range(n_starts):
        if trial == 0:
            x0 = x0_base.copy()
        else:
            x0 = x0_base + np.random.uniform(-np.pi, np.pi, 6)

        try:
            sol, info, ier, _ = fsolve(residual, x0, full_output=True)
            if ier != 1:
                continue
            res_norm = np.linalg.norm(info['fvec'])
            if res_norm > tol:
                continue

            # Deduplicate
            is_dup = False
            for prev in seen:
                diff = np.abs(np.remainder(sol - prev + np.pi, 2*np.pi) - np.pi)
                if np.max(diff) < 0.02:
                    is_dup = True
                    break
            if is_dup:
                continue
            seen.append(sol.copy())

            # Build final structure
            phi_f = phi.copy()
            psi_f = psi_ext.copy()
            phi_f[p1] = sol[0]; psi_f[p1+1] = sol[1]
            phi_f[p2] = sol[2]; psi_f[p2+1] = sol[3]
            phi_f[p3] = sol[4]; psi_f[p3+1] = sol[5]

            N_f, CA_f, C_f = _build_chain(phi_f, psi_f[:n+1],
                                           anchor_N, anchor_CA, anchor_C)
            N_virt = _place_atom(N_f[-1], CA_f[-1], C_f[-1],
                                  BL_CN, BA_CCN, psi_f[n])
            cl = float(np.linalg.norm(N_virt - closure_N))

            solutions.append((phi_f, psi_f[:n+1], cl, N_f, CA_f, C_f))
        except Exception:
            continue

    return solutions


# ─── Test ────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 60)
    print("TLC Solver Test")
    print("=" * 60)

    n_test = 7
    np.random.seed(42)
    phi_t = np.random.uniform(-np.pi, np.pi, n_test)
    psi_t = np.random.uniform(-np.pi, np.pi, n_test + 1)

    anc_N  = np.array([0.0, 1.2, 0.0])
    anc_CA = np.array([1.458, 0.0, 0.0])
    anc_C  = np.array([2.5, 1.0, 0.3])

    # Build to get closure atoms
    N_ch, CA_ch, C_ch = _build_chain(phi_t, psi_t, anc_N, anc_CA, anc_C)

    # Place closure atoms after the loop
    clos_N  = _place_atom(N_ch[-1], CA_ch[-1], C_ch[-1], BL_CN,  BA_CCN, 0.5)
    clos_CA = _place_atom(CA_ch[-1], C_ch[-1], clos_N,   BL_NCA, BA_CNC, OMEGA)
    clos_C  = _place_atom(C_ch[-1], clos_N, clos_CA,     BL_CAC, BA_NCC, 0.3)

    print(f"  n_loop: {n_test}, pivots: [0, {n_test//2}, {n_test-1}]")
    print(f"  Anchor C → Closure N: {np.linalg.norm(C_ch[-1] - clos_N):.3f}Å")

    # Perturb pivot torsions so the chain is NOT closed, then solve
    phi_open = phi_t.copy()
    psi_open = psi_t.copy()
    phi_open[0] += 0.5
    phi_open[n_test//2] -= 0.3
    phi_open[n_test-1] += 0.7

    N_open, CA_open, C_open = _build_chain(phi_open, psi_open, anc_N, anc_CA, anc_C)
    N_virt_open = _place_atom(N_open[-1], CA_open[-1], C_open[-1], BL_CN, BA_CCN, 0.5)
    print(f"  Open chain closure gap: {np.linalg.norm(N_virt_open - clos_N):.3f}Å")

    print(f"\n  Solving TLC...")
    solutions = solve_tlc(
        anc_N, anc_CA, anc_C,
        clos_N, clos_CA, clos_C,
        phi_open, psi_open,
        pivot_indices=[0, n_test//2, n_test-1],
        n_starts=128,
    )

    print(f"  Solutions found: {len(solutions)}")
    for i, (phi_s, psi_s, cl, N_s, CA_s, C_s) in enumerate(solutions):
        # Check if solution matches original torsions
        phi_diff = np.max(np.abs(np.remainder(phi_s - phi_t + np.pi, 2*np.pi) - np.pi))
        print(f"    Sol {i+1}: closure={cl:.6f}Å  "
              f"max_phi_diff_to_original={np.degrees(phi_diff):.1f}°")

    print("\n" + "=" * 60)