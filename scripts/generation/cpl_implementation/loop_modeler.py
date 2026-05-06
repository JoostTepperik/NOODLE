"""
CDR3 loop modeller: NeRF backbone building + closure + clash filtering.

Two closure strategies are provided:

KIC (recommended)
-----------------
:func:`kic_build_all_solutions` takes **pre-specified** φ/ψ torsion angles
and analytically finds **all** closed backbone conformations (up to 16) via
Kinematic Closure (KIC).  Only the three C-terminal pivot torsions are
adjusted; all other angles are held exactly at the supplied values.

CCD (legacy)
------------
:func:`sample_loops` randomly samples φ/ψ from Ramachandran regions and
iteratively closes each conformation with Cyclic Coordinate Descent (CCD).
CCD modifies *all* torsion angles and finds only *one* solution per trial.
It does **not** implement analytical KIC.

Ensemble element format (both strategies)::

    (N, CA, C, O, energy, closure_dist)

where ``N, CA, C, O`` are (n_loop, 3) float arrays, ``energy`` is a float
clash-count score (lower = better), and ``closure_dist`` is the closure
RMSD in Å (virtual N and Cα vs. anchor targets).

References
----------
* Coutsias et al. (2004) "A Kinematic View of Loop Closure."
  J. Comput. Chem. 25, 510–528.
* Canutescu & Dunbrack (2003) "Cyclic coordinate descent: a robotics
  algorithm for protein loop closure." Protein Sci. 12, 963–972.
"""
from __future__ import annotations

import numpy as np

from nerf import (
    BOND_ANGLES_RAD,
    BOND_LENGTHS,
    build_loop,
    nerf,
)
from utils import VDW_RADII, _BACKBONE_ATOMS

# ─────────────────────────────────────────────────────────────────────────────
# Ramachandran sampling
# ─────────────────────────────────────────────────────────────────────────────

# (mean_phi_deg, mean_psi_deg, std_deg, weight)
_RAMA_REGIONS = [
    (-60.0,  -45.0, 15.0, 0.35),   # α-helix
    (-120.0, 130.0, 15.0, 0.40),   # β-sheet
    (60.0,   45.0,  15.0, 0.05),   # left-handed helix
    (0.0,    0.0,   180.0, 0.20),  # uniform / other
]
_RAMA_WEIGHTS  = np.array([r[3] for r in _RAMA_REGIONS])
_RAMA_WEIGHTS /= _RAMA_WEIGHTS.sum()


def _sample_torsions(n_res: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """
    Sample φ/ψ angles for *n_res* residues from Ramachandran-weighted
    Gaussian regions.

    Returns:
        phi, psi — each (n_res,) float64 array in radians.
    """
    phi = np.empty(n_res)
    psi = np.empty(n_res)
    region_idx = rng.choice(len(_RAMA_REGIONS), size=n_res, p=_RAMA_WEIGHTS)
    for i, ri in enumerate(region_idx):
        mu_phi, mu_psi, std, _ = _RAMA_REGIONS[ri]
        phi[i] = rng.normal(np.radians(mu_phi), np.radians(std))
        psi[i] = rng.normal(np.radians(mu_psi), np.radians(std))
    return phi, psi


# ─────────────────────────────────────────────────────────────────────────────
# CCD closure
# ─────────────────────────────────────────────────────────────────────────────

def _ccd_optimal_rotation(
    axis_from:   np.ndarray,
    axis_to:     np.ndarray,
    mobile_pts:  np.ndarray,
    target_pts:  np.ndarray,
) -> float:
    """
    Analytically compute the optimal rotation δθ around the axis
    *axis_from* → *axis_to* that minimises::

        Σ_k |R(δθ) · (mobile_pts[k] – axis_from) + axis_from – target_pts[k]|²

    Derivation: for the perpendicular components c_k⊥ and t_k⊥ of
    (mobile - p) and (target - p) relative to the rotation axis u::

        δθ = atan2( Σ u·(c_k⊥ × t_k⊥),  Σ c_k⊥·t_k⊥ )

    Args:
        axis_from, axis_to: (3,) endpoints that define the rotation axis.
        mobile_pts: (k, 3) current positions of atoms that will move.
        target_pts: (k, 3) desired positions.

    Returns:
        δθ in radians.
    """
    u = axis_to - axis_from
    u_norm = np.linalg.norm(u)
    if u_norm < 1e-8:
        return 0.0
    u /= u_norm
    p = axis_from

    A_sum = 0.0
    B_sum = 0.0
    for c, t in zip(mobile_pts, target_pts):
        c_p = c - p
        t_p = t - p
        c_perp = c_p - np.dot(c_p, u) * u
        t_perp = t_p - np.dot(t_p, u) * u
        # scalar triple product: u · (c_perp × t_perp) = (u × c_perp) · t_perp
        A_sum += float(np.dot(u, np.cross(c_perp, t_perp)))
        B_sum += float(np.dot(c_perp, t_perp))

    return float(np.arctan2(A_sum, B_sum))


def _virtual_anchors(
    N: np.ndarray, CA: np.ndarray, C: np.ndarray, psi_last: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Place virtual N and Cα of the first C-terminal anchor residue from the
    last loop C′ atom.

    The virtual Cα uses ω = π (trans peptide).
    """
    L = BOND_LENGTHS
    A = BOND_ANGLES_RAD
    virt_N  = nerf(N[-1], CA[-1], C[-1], L['C_N'],  A['CA_C_N'], psi_last)
    virt_CA = nerf(CA[-1], C[-1], virt_N, L['N_CA'], A['C_N_CA'], np.pi)
    return virt_N, virt_CA


def ccd_closure(
    prev_N:    np.ndarray,
    prev_CA:   np.ndarray,
    prev_C:    np.ndarray,
    psi_prev:  float,
    phi:       np.ndarray,
    psi:       np.ndarray,
    target_N:  np.ndarray,
    target_CA: np.ndarray,
    omega:     np.ndarray | None = None,
    n_iter:    int   = 200,
    tol:       float = 0.05,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Cyclic Coordinate Descent loop closure.

    Iteratively adjusts *phi* and *psi* (φ/ψ for every loop residue) so that
    the virtual N and Cα placed after the last loop residue match
    *target_N* and *target_CA* (the first C-terminal anchor atoms).

    Each sweep visits torsions from C-terminus to N-terminus; for each one
    the analytically optimal rotation is applied and the backbone is rebuilt
    immediately so subsequent steps see up-to-date positions.

    Args:
        prev_N, prev_CA, prev_C:
            (3,) last N-terminal anchor residue coordinates.
        psi_prev: ψ of the anchor residue (fixed; not modified).
        phi:      (n_loop,) φ angles in radians — modified in place.
        psi:      (n_loop,) ψ angles in radians — modified in place.
        target_N:  (3,) target N  of first C-terminal anchor residue.
        target_CA: (3,) target Cα of first C-terminal anchor residue.
        omega:    (n_loop,) ω angles in radians (default: all π).
        n_iter:   Maximum number of outer CCD sweeps.
        tol:      Convergence threshold in Å (RMSD of virtual anchors).

    Returns:
        phi, psi — optimised torsion arrays (copies).
        closure  — final RMSD in Å of virtual anchors vs targets.
    """
    n_loop = len(phi)
    phi    = phi.copy()
    psi    = psi.copy()
    if omega is None:
        omega = np.full(n_loop, np.pi)

    targets = np.array([target_N, target_CA])  # (2, 3)

    for _sweep in range(n_iter):
        N, CA, C, _ = build_loop(prev_N, prev_CA, prev_C, psi_prev, phi, psi, omega)
        virt_N, virt_CA = _virtual_anchors(N, CA, C, psi[-1])
        mobile = np.array([virt_N, virt_CA])

        closure = float(np.sqrt(np.mean(np.sum((mobile - targets) ** 2, axis=1))))
        if closure < tol:
            break

        # Reverse sweep: C-terminus → N-terminus
        for i in range(n_loop - 1, -1, -1):
            # ── optimise ψ[i]: axis Cα[i] → C′[i] ───────────────────────
            delta = _ccd_optimal_rotation(CA[i], C[i], mobile, targets)
            psi[i] = float(np.arctan2(
                np.sin(psi[i] + delta), np.cos(psi[i] + delta)
            ))

            N, CA, C, _ = build_loop(prev_N, prev_CA, prev_C, psi_prev, phi, psi, omega)
            virt_N, virt_CA = _virtual_anchors(N, CA, C, psi[-1])
            mobile = np.array([virt_N, virt_CA])

            # ── optimise φ[i]: axis N[i] → Cα[i] ────────────────────────
            delta = _ccd_optimal_rotation(N[i], CA[i], mobile, targets)
            phi[i] = float(np.arctan2(
                np.sin(phi[i] + delta), np.cos(phi[i] + delta)
            ))

            N, CA, C, _ = build_loop(prev_N, prev_CA, prev_C, psi_prev, phi, psi, omega)
            virt_N, virt_CA = _virtual_anchors(N, CA, C, psi[-1])
            mobile = np.array([virt_N, virt_CA])

    closure = float(np.sqrt(np.mean(np.sum((mobile - targets) ** 2, axis=1))))
    return phi, psi, closure


# ─────────────────────────────────────────────────────────────────────────────
# Clash detection
# ─────────────────────────────────────────────────────────────────────────────

# VdW radii for the four backbone atoms placed by NeRF (same order as stacking)
_LOOP_ATOM_NAMES = list(_BACKBONE_ATOMS)   # ('N', 'CA', 'C', 'O')
_LOOP_ATOM_RADII = np.array([VDW_RADII[a] for a in _LOOP_ATOM_NAMES], dtype=np.float32)


def _has_clash(
    N:              np.ndarray,
    CA:             np.ndarray,
    C:              np.ndarray,
    O:              np.ndarray,
    fw_coords:      np.ndarray,
    fw_radii:       np.ndarray,
    overlap_tol:    float = 0.6,
) -> bool:
    """
    Return ``True`` if any loop backbone atom clashes with the framework.

    A clash is defined as::

        distance(loop_atom, fw_atom) < r_loop + r_fw − overlap_tol

    Args:
        N, CA, C, O:   (n_loop, 3) loop backbone coordinates.
        fw_coords:     (N_fw, 3)   float32 framework atom positions.
        fw_radii:      (N_fw,)     float32 framework vdW radii.
        overlap_tol:   Allowed overlap tolerance in Å (default 0.6).

    Returns:
        bool — True if at least one clash is detected.
    """
    # Stack all loop atoms: (4*n_loop, 3)
    loop_coords  = np.vstack([N, CA, C, O]).astype(np.float32)
    n_loop_atoms = len(N)
    # vstack layout: all N rows first, then CA, C, O — so repeat each radius n times
    loop_radii   = np.repeat(_LOOP_ATOM_RADII, n_loop_atoms)

    # Pairwise differences: (n_loop_atoms*4, N_fw, 3)
    diff      = loop_coords[:, None, :] - fw_coords[None, :, :]
    dists     = np.linalg.norm(diff, axis=2)            # (n_loop_atoms*4, N_fw)
    thresholds = (loop_radii[:, None] + fw_radii[None, :]
                  - overlap_tol)                         # (n_loop_atoms*4, N_fw)
    return bool(np.any(dists < thresholds))


def _intra_loop_clash(
    N:           np.ndarray,
    CA:          np.ndarray,
    C:           np.ndarray,
    O:           np.ndarray,
    overlap_tol: float = 0.6,
    min_sep:     int   = 3,
) -> bool:
    """
    Return ``True`` if any intra-loop atom pair closer than *min_sep* residues
    clashes.

    Pairs within *min_sep* sequence positions are skipped (bonded neighbours).

    Args:
        N, CA, C, O:   (n_loop, 3) loop backbone coordinates.
        overlap_tol:   Allowed overlap tolerance in Å (default 0.6).
        min_sep:       Minimum residue separation to check (default 3).

    Returns:
        bool — True if any intra-loop clash is detected.
    """
    # Build flat list: for residue i, atoms are [N, CA, C, O] at indices 4i..4i+3
    n_res   = len(N)
    coords  = np.vstack([N, CA, C, O])                      # (4*n_res, 3)
    # vstack layout: all N first, then CA, C, O — use repeat not tile
    radii   = np.repeat(_LOOP_ATOM_RADII, n_res)            # (4*n_res,)

    # With vstack([N, CA, C, O]) the layout is:
    #   rows 0..n-1     → N atoms for residues 0..n-1
    #   rows n..2n-1    → CA atoms for residues 0..n-1
    #   rows 2n..3n-1   → C  atoms for residues 0..n-1
    #   rows 3n..4n-1   → O  atoms for residues 0..n-1
    # The residue for flat index ai is ai % n_res.
    # All four atoms of the same residue i map to the same residue index i,
    # so bonded/same-residue pairs are correctly excluded by the min_sep filter.
    n_atoms = 4 * n_res
    for ai in range(n_atoms):
        res_i = ai % n_res
        for aj in range(ai + 1, n_atoms):
            res_j = aj % n_res
            if abs(res_i - res_j) < min_sep:
                continue
            dist = float(np.linalg.norm(coords[ai] - coords[aj]))
            if dist < radii[ai] + radii[aj] - overlap_tol:
                return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Energy scoring (simple clash count)
# ─────────────────────────────────────────────────────────────────────────────

def score_clashes(
    N:           np.ndarray,
    CA:          np.ndarray,
    C:           np.ndarray,
    O:           np.ndarray,
    fw_coords:   np.ndarray,
    fw_radii:    np.ndarray,
    overlap_tol: float = 0.6,
) -> float:
    """
    Count the number of loop–framework atom clashes (lower is better).

    Args:
        N, CA, C, O:   (n_loop, 3) loop backbone coordinates.
        fw_coords:     (N_fw, 3)   framework atom positions.
        fw_radii:      (N_fw,)     framework vdW radii.
        overlap_tol:   Overlap tolerance in Å (default 0.6).

    Returns:
        float — number of clashing atom pairs.
    """
    loop_coords  = np.vstack([N, CA, C, O]).astype(np.float32)
    n_loop_atoms = len(N)
    # vstack layout: all N first, then CA, C, O — use repeat not tile
    loop_radii   = np.repeat(_LOOP_ATOM_RADII, n_loop_atoms)

    diff       = loop_coords[:, None, :] - fw_coords[None, :, :]
    dists      = np.linalg.norm(diff, axis=2)
    thresholds = loop_radii[:, None] + fw_radii[None, :] - overlap_tol
    return float(np.sum(dists < thresholds))


# ─────────────────────────────────────────────────────────────────────────────
# Main sampling function
# ─────────────────────────────────────────────────────────────────────────────

def sample_loops(
    sequence:       str,
    prev_N:         np.ndarray,
    prev_CA:        np.ndarray,
    prev_C:         np.ndarray,
    psi_prev:       float,
    target_N:       np.ndarray,
    target_CA:      np.ndarray,
    target_C:       np.ndarray,
    fw_coords:      np.ndarray | None = None,
    fw_radii:       np.ndarray | None = None,
    n_samples:      int   = 200,
    closure_tol:    float = 0.5,
    overlap_tol:    float = 0.6,
    ccd_iter:       int   = 200,
    ccd_tol:        float = 0.05,
    check_intra:    bool  = True,
    rng_seed:       int | None = None,
    verbose:        bool  = False,
) -> list:
    """
    Sample CDR3 loop conformations via NeRF backbone building + CCD closure.

    For each trial:

    1. Randomly draw φ/ψ from Ramachandran-weighted Gaussian regions.
    2. Build backbone with :func:`nerf.build_loop`.
    3. Run :func:`ccd_closure` to close onto (*target_N*, *target_CA*).
    4. Reject if closure RMSD > *closure_tol*.
    5. Optionally reject intra-loop backbone clashes.
    6. Optionally reject loop–framework backbone clashes.
    7. Score with :func:`score_clashes` (framework clash count).

    The returned ensemble is directly compatible with ``utils.save_pdbs``
    and ``utils.compute_loop_rmsds``:

        ensemble[i] = (N, CA, C, O, energy, closure_dist)

    where ``N, CA, C, O`` are (n_loop, 3) float64 arrays,
    ``energy`` is the clash-count float, and ``closure_dist`` is in Å.

    Args:
        sequence:    One-letter amino acid string for the loop residues.
        prev_N, prev_CA, prev_C:
            (3,) coordinates of the last N-terminal anchor residue.
        psi_prev:
            ψ torsion of the anchor residue (radians); fixed during CCD.
        target_N, target_CA, target_C:
            (3,) coordinates of the first C-terminal anchor residue.
            ``target_N`` / ``target_CA`` are used as CCD closure targets;
            ``target_C`` is stored for reference only.
        fw_coords:   (N_fw, 3) float32 framework atom positions, or None to
                     skip framework clash checking.
        fw_radii:    (N_fw,)   float32 framework vdW radii (required when
                     *fw_coords* is provided).
        n_samples:   Number of accepted structures to collect.
        closure_tol: Maximum closure RMSD (Å) to accept a structure (0.5 Å).
        overlap_tol: VdW overlap tolerance for clash detection in Å (0.6 Å).
        ccd_iter:    Maximum CCD sweeps per structure (default 200).
        ccd_tol:     CCD inner convergence threshold in Å (default 0.05).
        check_intra: If True, reject structures with intra-loop clashes.
        rng_seed:    Optional integer seed for reproducibility.
        verbose:     Print per-sample progress.

    Returns:
        List of ``(N, CA, C, O, energy, closure_dist)`` tuples; length ≤
        *n_samples*.
    """
    n_loop = len(sequence)
    rng    = np.random.default_rng(rng_seed)

    use_fw  = fw_coords is not None and fw_radii is not None
    ensemble: list = []

    n_tried = n_clashes_fw = n_clashes_intra = n_open = 0

    while len(ensemble) < n_samples:
        n_tried += 1

        # ── 1. Sample torsions ────────────────────────────────────────────
        phi, psi = _sample_torsions(n_loop, rng)

        # ── 2. CCD closure ────────────────────────────────────────────────
        phi, psi, closure = ccd_closure(
            prev_N, prev_CA, prev_C, psi_prev,
            phi, psi,
            target_N, target_CA,
            n_iter=ccd_iter, tol=ccd_tol,
        )

        if closure > closure_tol:
            n_open += 1
            continue

        # ── 3. Rebuild final backbone ─────────────────────────────────────
        N, CA, C, O = build_loop(prev_N, prev_CA, prev_C, psi_prev, phi, psi)

        # ── 4. Intra-loop clash filter ────────────────────────────────────
        if check_intra and _intra_loop_clash(N, CA, C, O, overlap_tol):
            n_clashes_intra += 1
            continue

        # ── 5. Framework clash filter ─────────────────────────────────────
        if use_fw and _has_clash(N, CA, C, O, fw_coords, fw_radii, overlap_tol):
            n_clashes_fw += 1
            continue

        # ── 6. Score ──────────────────────────────────────────────────────
        energy = (score_clashes(N, CA, C, O, fw_coords, fw_radii, overlap_tol)
                  if use_fw else 0.0)

        ensemble.append((N, CA, C, O, energy, closure))

        if verbose:
            print(f"    [{len(ensemble):4d}/{n_samples}]  "
                  f"closure={closure:.3f} Å  energy={energy:.0f}  "
                  f"(tried {n_tried}, open {n_open}, "
                  f"fw_clash {n_clashes_fw}, intra_clash {n_clashes_intra})")

    print(f"    Sampling complete: {len(ensemble)} structures from "
          f"{n_tried} trials  "
          f"(open={n_open}, fw_clash={n_clashes_fw}, "
          f"intra_clash={n_clashes_intra})")

    return ensemble


# ─────────────────────────────────────────────────────────────────────────────
# KIC: build all solutions from given torsion angles
# ─────────────────────────────────────────────────────────────────────────────

def kic_build_all_solutions(
    sequence:    str,
    phi:         np.ndarray,
    psi:         np.ndarray,
    prev_N:      np.ndarray,
    prev_CA:     np.ndarray,
    prev_C:      np.ndarray,
    psi_prev:    float,
    target_N:    np.ndarray,
    target_CA:   np.ndarray,
    target_C:    np.ndarray,
    fw_coords:   np.ndarray | None = None,
    fw_radii:    np.ndarray | None = None,
    closure_tol: float = 0.05,
    overlap_tol: float = 0.6,
    n_grid:      int   = 3600,
    check_intra: bool  = True,
    verbose:     bool  = False,
) -> list:
    """
    Analytically close a loop with **given** torsion angles and return all
    valid KIC solutions as a backbone ensemble.

    This is the correct entry point when the user already has φ/ψ torsion
    angles (e.g. generated by a generative model or taken from a database).
    Unlike :func:`sample_loops` (which uses iterative CCD and modifies *all*
    torsion angles), this function:

    * Holds φ[0 … L−2] and ψ[0 … L−3] **exactly** at the supplied values.
    * Finds all (ψ[L−2], φ[L−1], ψ[L−1]) pivot angles that place the
      virtual N of the C-terminal anchor exactly on *target_N*, using the
      analytical Kinematic Closure algorithm implemented in :mod:`kic`.
    * Returns up to 16 closed conformations — **all** solutions that exist
      for this combination of fixed torsions and anchor positions.

    Pivot torsions
    --------------
    The three pivot torsions are chosen at the C-terminal end of the loop:

        t1 = ψ[L−2]   rotates Cα[L−2]–C[L−2], places   N[L−1]
        t2 = φ[L−1]   rotates  N[L−1]–Cα[L−1], places   C[L−1]
        t3 = ψ[L−1]   rotates Cα[L−1]–C[L−1], places virtual N[L]

    Residues 0 … L−3 and the φ of residue L−2 are entirely unchanged.

    Ensemble element format
    -----------------------
    Each element of the returned list is a tuple::

        (N, CA, C, O, energy, closure_dist)

    where ``N, CA, C, O`` are (n_loop, 3) float64 arrays compatible with
    ``utils.write_pdb_atoms`` and ``utils.save_pdbs``.

    Args:
        sequence:    One-letter amino acid string for the loop residues.
        phi, psi:    (n_loop,) input torsion arrays in radians.
        prev_N, prev_CA, prev_C:
            (3,) last N-terminal anchor atom coordinates.
        psi_prev:    ψ of the N-terminal anchor residue (radians).
        target_N, target_CA, target_C:
            (3,) first C-terminal anchor residue atom coordinates.
            *target_N* is the KIC closure target; *target_CA* is used to
            compute the reported closure RMSD; *target_C* is not used.
        fw_coords:   (N_fw, 3) float32 framework atom positions, or None to
                     skip framework clash checking.
        fw_radii:    (N_fw,)   float32 framework vdW radii (required when
                     *fw_coords* is provided).
        closure_tol: Maximum closure RMSD in Å for an accepted solution
                     (default 0.05 — strict, because KIC is analytical).
        overlap_tol: VdW overlap tolerance for clash detection in Å (0.6 Å).
        n_grid:      Number of grid points for the t1 scan (default 3600,
                     which gives ≈ 0.1° resolution — sufficient for ≤16 roots).
        check_intra: If True, reject structures with intra-loop backbone
                     clashes.
        verbose:     Print per-solution details.

    Returns:
        List of ``(N, CA, C, O, energy, closure_dist)`` tuples.  The list
        may be empty if no closed solution exists for the given angles and
        anchor geometry.  When solutions exist, the list has at most 16
        elements.
    """
    from kic import kic_close_given_torsions
    from nerf import measure_closure

    use_fw   = fw_coords is not None and fw_radii is not None
    ensemble: list = []

    # ── Analytical KIC: find all closed torsion arrays ────────────────────────
    kic_solutions = kic_close_given_torsions(
        phi, psi,
        prev_N, prev_CA, prev_C, psi_prev,
        target_N, n_grid=n_grid, tol=closure_tol,
    )

    if verbose:
        print(f"    KIC: {len(kic_solutions)} raw solution(s) for "
              f"{len(phi)}-residue loop")

    n_intra = 0
    n_fw    = 0

    for phi_sol, psi_sol in kic_solutions:
        # Build full backbone using the KIC-solved torsion angles
        N, CA, C, O = build_loop(prev_N, prev_CA, prev_C, psi_prev,
                                 phi_sol, psi_sol)

        # Measure closure quality (RMSD of virtual N and Cα vs targets)
        cl           = measure_closure(N, CA, C, psi_sol[-1], target_N, target_CA)
        closure_dist = cl['rmsd']

        if closure_dist > closure_tol:
            # Numerical safety check — should not normally trigger
            continue

        # Intra-loop clash filter
        if check_intra and _intra_loop_clash(N, CA, C, O, overlap_tol):
            n_intra += 1
            continue

        # Framework clash filter
        if use_fw and _has_clash(N, CA, C, O, fw_coords, fw_radii, overlap_tol):
            n_fw += 1
            continue

        energy = (score_clashes(N, CA, C, O, fw_coords, fw_radii, overlap_tol)
                  if use_fw else 0.0)

        ensemble.append((N, CA, C, O, energy, closure_dist))

        if verbose:
            n_loop = len(phi_sol)
            print(f"    [sol {len(ensemble):2d}]  "
                  f"closure={closure_dist:.4f} Å  energy={energy:.0f}  "
                  f"ψ[{n_loop-2}]={np.degrees(psi_sol[n_loop-2]):6.1f}°  "
                  f"φ[{n_loop-1}]={np.degrees(phi_sol[n_loop-1]):6.1f}°  "
                  f"ψ[{n_loop-1}]={np.degrees(psi_sol[n_loop-1]):6.1f}°")

    print(f"    KIC accepted {len(ensemble)}/{len(kic_solutions)} raw solutions"
          f"  (intra_clash={n_intra}, fw_clash={n_fw})")

    return ensemble
