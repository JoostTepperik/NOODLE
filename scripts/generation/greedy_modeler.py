"""
Greedy residue-by-residue backbone placement.

Places one residue at a time from the N-terminal anchor, sampling many
(φ, ψ) candidates at each position and immediately rejecting any that
introduce a steric clash with the framework or with already-placed loop
residues.  Surviving clash-free candidates are optionally scored by an
external energy callable.  A **beam search** keeps the top *beam_width*
partial structures at each step.

No loop closure is attempted — the chain grows forward from the anchor
and the resulting open-chain structures are returned directly.

Algorithm
---------
For each residue position *i* = 0 … len(sequence)−1:

1. Sample *n_candidates* (φ, ψ) pairs from Ramachandran-weighted
   Gaussian regions for **each** current beam state.
2. Place residue *i* using ideal NeRF geometry from the last anchor /
   last placed residue.
3. Hard-reject candidates whose new atoms clash with the framework
   (*fw_coords* / *fw_radii*) or with already-placed residues ≥
   *min_sep* positions away.
4. Score surviving candidates with the optional *energy_fn*.
5. Carry the top *beam_width* candidates forward as the new beam.

Placement is vectorised: for a given beam state *N_i* and *CA_i* are
the same for all n_candidates (they depend only on the previous psi,
which is fixed), while *C_i* and *O_i* are computed in a single numpy
broadcast over all candidates.

Output format
-------------
Compatible with ``utils.save_pdbs`` and ``utils.compute_loop_rmsds``::

    ensemble[i] = (N, CA, C, O, energy, closure_dist)

where ``N, CA, C, O`` are (len(sequence), 3) float64 arrays,
``energy`` is the accumulated score over all placed residues, and
``closure_dist`` is always **0.0** (closure not attempted).
"""
from __future__ import annotations

import numpy as np
from typing import Callable, List, Optional, Tuple

from nerf import BOND_ANGLES_RAD, BOND_LENGTHS, nerf as _nerf
from loop_modeler import _LOOP_ATOM_RADII
from utils import VDW_RADII

# ─────────────────────────────────────────────────────────────────────────────
# Type aliases
# ─────────────────────────────────────────────────────────────────────────────

EnergyFn = Callable[
    [np.ndarray, np.ndarray, np.ndarray, np.ndarray], float
]

# Ensemble element: (N, CA, C, O, energy, closure_dist)
EnsembleEntry = Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float
]

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


def _sample_torsion_batch(
    n: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample *n* (φ, ψ) pairs from Ramachandran-weighted Gaussian regions."""
    region_idx = rng.choice(len(_RAMA_REGIONS), size=n, p=_RAMA_WEIGHTS)
    phi = np.empty(n)
    psi = np.empty(n)
    for k, ri in enumerate(region_idx):
        mu_phi, mu_psi, std, _ = _RAMA_REGIONS[ri]
        phi[k] = rng.normal(np.radians(mu_phi), np.radians(std))
        psi[k] = rng.normal(np.radians(mu_psi), np.radians(std))
    return phi, psi


# ─────────────────────────────────────────────────────────────────────────────
# Vectorised residue placement
# ─────────────────────────────────────────────────────────────────────────────

def _place_one_residue(
    ref_N:   np.ndarray,
    ref_CA:  np.ndarray,
    ref_C:   np.ndarray,
    ref_psi: float,
    phi_i:   float,
    psi_i:   float,
    omega_i: float = np.pi,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Place one residue's backbone atoms (N, Cα, C′, O) via NeRF.

    Mirrors the interior step of :func:`nerf.build_loop`: the reference
    atoms are the previous residue's N, Cα, C′ (or the anchor for the
    first loop residue) and *ref_psi* is their ψ angle.

    Args:
        ref_N, ref_CA, ref_C:  (3,) reference atoms.
        ref_psi:               ψ of the reference residue (places new N).
        phi_i, psi_i:          φ and ψ for the new residue (radians).
        omega_i:               ω for the new residue (default π, trans).

    Returns:
        N_i, CA_i, C_i, O_i — each (3,) float64.
    """
    L = BOND_LENGTHS
    A = BOND_ANGLES_RAD
    N_i  = _nerf(ref_N,  ref_CA, ref_C,  L['C_N'],  A['CA_C_N'], ref_psi)
    CA_i = _nerf(ref_CA, ref_C,  N_i,    L['N_CA'], A['C_N_CA'], omega_i)
    C_i  = _nerf(ref_C,  N_i,   CA_i,   L['CA_C'], A['N_CA_C'], phi_i)
    O_i  = _nerf(N_i,    CA_i,  C_i,    L['C_O'],  A['CA_C_O'], psi_i + np.pi)
    return N_i, CA_i, C_i, O_i


def _batch_place_residues(
    ref_N:   np.ndarray,
    ref_CA:  np.ndarray,
    ref_C:   np.ndarray,
    ref_psi: float,
    phi_arr: np.ndarray,
    psi_arr: np.ndarray,
    omega:   float = np.pi,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Vectorised placement of *n* residues sharing the same reference atoms.

    For a fixed beam state the reference atoms (ref_N, ref_CA, ref_C) and
    ref_psi are the same for all *n* candidates, so N_i and CA_i are
    identical across the batch.  Only C_i (depends on φ) and O_i (depends
    on φ and ψ) vary.

    Args:
        ref_N, ref_CA, ref_C:  (3,) reference atoms (shared).
        ref_psi:               ψ of reference residue (radians, shared).
        phi_arr, psi_arr:      (n,) arrays of candidate torsion angles.
        omega:                 ω for all new residues (default π).

    Returns:
        N_arr, CA_arr, C_arr, O_arr — each (n, 3) float64.
        N_arr and CA_arr are constant across the batch (broadcast slices).
    """
    n = len(phi_arr)
    L = BOND_LENGTHS
    A = BOND_ANGLES_RAD

    # ── N_i and CA_i: same for every candidate ────────────────────────────
    N_i  = _nerf(ref_N,  ref_CA, ref_C, L['C_N'],  A['CA_C_N'], ref_psi)
    CA_i = _nerf(ref_CA, ref_C,  N_i,   L['N_CA'], A['C_N_CA'], omega)

    # ── C_i: varies with phi.  ────────────────────────────────────────────
    # nerf(ref_C, N_i, CA_i, ..., phi[k])  — a=ref_C, b=N_i, c=CA_i
    # Local frame: bc = c - b = CA_i - N_i  (shared across candidates).
    bc    = CA_i - N_i
    bc_hat = bc / np.linalg.norm(bc)
    n_vec  = ref_C - N_i                   # a - b
    n_vec  = n_vec - np.dot(n_vec, bc_hat) * bc_hat
    n_norm = np.linalg.norm(n_vec)
    if n_norm < 1e-8:
        # a, b, c are collinear — pick an arbitrary axis perpendicular to bc_hat.
        # Use z-axis unless bc_hat is nearly parallel to it (|z-component| ≥ 0.9),
        # in which case use x-axis to ensure the cross product has magnitude.
        tmp   = (np.array([0., 0., 1.])
                 if abs(bc_hat[2]) < 0.9 else np.array([1., 0., 0.]))
        n_vec = tmp - np.dot(tmp, bc_hat) * bc_hat
    n_hat  = n_vec / np.linalg.norm(n_vec)
    m_hat  = np.cross(bc_hat, n_hat)

    bl_C   = L['CA_C']
    ba_C   = A['N_CA_C']
    offset = -bl_C * np.cos(ba_C)          # scalar — same for all k
    amp_C  =  bl_C * np.sin(ba_C)          # scalar

    # C_arr[k] = CA_i + offset*bc_hat + amp*(cos(phi[k])*n_hat + sin(phi[k])*m_hat)
    C_arr = (CA_i[None, :]
             + offset * bc_hat[None, :]
             + amp_C * (np.cos(phi_arr)[:, None] * n_hat[None, :]
                        + np.sin(phi_arr)[:, None] * m_hat[None, :]))  # (n, 3)

    # ── O_i: varies with both phi (through C_i) and psi.  ────────────────
    # nerf(N_i, CA_i, C_arr[k], L['C_O'], A['CA_C_O'], psi[k]+pi)
    bc_O     = C_arr - CA_i[None, :]                                # (n, 3)
    bc_O_norm = np.linalg.norm(bc_O, axis=1, keepdims=True)         # (n, 1)
    bc_O_hat  = bc_O / np.maximum(bc_O_norm, 1e-8)                  # (n, 3)

    n_O   = N_i - CA_i                                               # (3,)  shared
    dot_n = (bc_O_hat * n_O[None, :]).sum(axis=1, keepdims=True)    # (n, 1)
    n_perp = n_O[None, :] - dot_n * bc_O_hat                        # (n, 3)
    n_perp_norm = np.linalg.norm(n_perp, axis=1, keepdims=True)     # (n, 1)
    n_hat_O = n_perp / np.maximum(n_perp_norm, 1e-8)                # (n, 3)
    m_hat_O = np.cross(bc_O_hat, n_hat_O)                           # (n, 3)

    bl_O  = L['C_O']
    ba_O  = A['CA_C_O']
    dihs  = psi_arr + np.pi                                          # (n,)
    off_O = -bl_O * np.cos(ba_O)
    amp_O =  bl_O * np.sin(ba_O)

    O_arr = (C_arr
             + off_O * bc_O_hat
             + amp_O * (np.cos(dihs)[:, None] * n_hat_O
                        + np.sin(dihs)[:, None] * m_hat_O))         # (n, 3)

    N_arr  = np.broadcast_to(N_i,  (n, 3)).copy()
    CA_arr = np.broadcast_to(CA_i, (n, 3)).copy()
    return N_arr, CA_arr, C_arr, O_arr


# ─────────────────────────────────────────────────────────────────────────────
# Per-residue clash detection
# ─────────────────────────────────────────────────────────────────────────────

def _batch_fw_clash_mask(
    N_arr:       np.ndarray,
    CA_arr:      np.ndarray,
    C_arr:       np.ndarray,
    O_arr:       np.ndarray,
    fw_coords:   np.ndarray,
    fw_radii:    np.ndarray,
    overlap_tol: float = 0.6,
) -> np.ndarray:
    """
    Return a boolean array of shape (n,) — True where candidate *k* clashes
    with the framework.

    Since N and CA are identical across candidates, they are checked once
    (if either clashes the whole batch is flagged).  C and O are checked
    per candidate.

    Args:
        N_arr, CA_arr, C_arr, O_arr:  each (n, 3).
        fw_coords:   (N_fw, 3) float32 framework atoms.
        fw_radii:    (N_fw,)   float32 framework vdW radii.
        overlap_tol: VdW overlap tolerance in Å.

    Returns:
        (n,) bool — True where a clash exists.
    """
    n = len(N_arr)
    clash = np.zeros(n, dtype=bool)

    # ── N_i: same for all candidates ─────────────────────────────────────
    d_N = np.linalg.norm(N_arr[0][None, :] - fw_coords, axis=1)  # (N_fw,)
    if np.any(d_N < VDW_RADII['N'] + fw_radii - overlap_tol):
        return np.ones(n, dtype=bool)

    # ── CA_i: same for all candidates ────────────────────────────────────
    d_CA = np.linalg.norm(CA_arr[0][None, :] - fw_coords, axis=1)  # (N_fw,)
    if np.any(d_CA < VDW_RADII['CA'] + fw_radii - overlap_tol):
        return np.ones(n, dtype=bool)

    # ── C_i: varies with phi ──────────────────────────────────────────────
    diff_C = C_arr[:, None, :] - fw_coords[None, :, :]          # (n, N_fw, 3)
    dists_C = np.linalg.norm(diff_C, axis=2)                    # (n, N_fw)
    thresh_C = VDW_RADII['C'] + fw_radii[None, :] - overlap_tol # (1, N_fw)
    clash |= np.any(dists_C < thresh_C, axis=1)

    # ── O_i: varies with psi ──────────────────────────────────────────────
    diff_O  = O_arr[:, None, :] - fw_coords[None, :, :]         # (n, N_fw, 3)
    dists_O = np.linalg.norm(diff_O, axis=2)                    # (n, N_fw)
    thresh_O = VDW_RADII['O'] + fw_radii[None, :] - overlap_tol # (1, N_fw)
    clash |= np.any(dists_O < thresh_O, axis=1)

    return clash


def _residue_intra_clash(
    N_placed:    np.ndarray,
    CA_placed:   np.ndarray,
    C_placed:    np.ndarray,
    O_placed:    np.ndarray,
    N_i:         np.ndarray,
    CA_i:        np.ndarray,
    C_i:         np.ndarray,
    O_i:         np.ndarray,
    overlap_tol: float = 0.6,
    min_sep:     int   = 3,
) -> bool:
    """
    Return True if the new residue clashes with any already-placed residue
    that is at least *min_sep* sequence positions away.

    *N_placed* etc. are (n_placed, 3) arrays of residues placed so far;
    the new residue will be at index n_placed.

    Args:
        N_placed, CA_placed, C_placed, O_placed:  (n_placed, 3) placed atoms.
        N_i, CA_i, C_i, O_i:  (3,) new residue atoms.
        overlap_tol:  VdW overlap tolerance in Å.
        min_sep:      Minimum residue separation to check (default 3).

    Returns:
        bool — True if any intra-loop clash is detected.
    """
    n_placed = len(N_placed)
    max_j    = n_placed - min_sep   # last index (inclusive) to check
    if max_j < 0:
        return False

    new_atoms = np.array([N_i, CA_i, C_i, O_i], dtype=np.float32)  # (4, 3)
    thresh = _LOOP_ATOM_RADII[:, None] + _LOOP_ATOM_RADII[None, :] - overlap_tol  # (4, 4)

    for j in range(max_j + 1):
        old_atoms = np.array(
            [N_placed[j], CA_placed[j], C_placed[j], O_placed[j]],
            dtype=np.float32,
        )
        diff  = new_atoms[:, None, :] - old_atoms[None, :, :]  # (4, 4, 3)
        dists = np.linalg.norm(diff, axis=2)                    # (4, 4)
        if bool(np.any(dists < thresh)):
            return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Main greedy placement function
# ─────────────────────────────────────────────────────────────────────────────

def greedy_place_residues(
    sequence:      str,
    prev_N:        np.ndarray,
    prev_CA:       np.ndarray,
    prev_C:        np.ndarray,
    psi_prev:      float,
    fw_coords:     Optional[np.ndarray] = None,
    fw_radii:      Optional[np.ndarray] = None,
    energy_fn:     Optional[EnergyFn]   = None,
    energy_weight: float = 1.0,
    n_candidates:  int   = 200,
    beam_width:    int   = 10,
    overlap_tol:   float = 0.6,
    check_intra:   bool  = True,
    min_sep:       int   = 3,
    rng_seed:      Optional[int] = None,
    verbose:       bool  = False,
) -> List[EnsembleEntry]:
    """
    Greedy residue-by-residue backbone placement with beam search.

    Residues are placed one at a time from the N-terminal anchor toward
    the C-terminus.  At each position *n_candidates* random (φ, ψ) pairs
    are drawn from Ramachandran-weighted Gaussian regions for every current
    beam state.  Candidates that introduce any steric clash are immediately
    discarded.  The remaining clash-free candidates are scored and the top
    *beam_width* are carried forward.

    No loop closure is attempted.  The output structures are open chains
    that extend *len(sequence)* residues from the anchor.

    Args:
        sequence:      One-letter amino acid string for the residues to
                       place (e.g. the loop or peptide internal residues).
        prev_N, prev_CA, prev_C:
            (3,) backbone atoms of the last N-terminal anchor residue.
        psi_prev:      ψ torsion of the anchor residue (radians); controls
                       where the first loop N atom is placed.
        fw_coords:     (N_fw, 3) float32 framework atom positions, or None
                       to skip framework clash checking.
        fw_radii:      (N_fw,)   float32 framework vdW radii (required when
                       *fw_coords* is provided).
        energy_fn:     Optional callable ``(N, CA, C, O) -> float`` that
                       scores the partial backbone grown so far.  Lower
                       values are preferred.  When None, all clash-free
                       candidates are equally ranked (random beam ordering).
        energy_weight: Multiplier applied to *energy_fn* output.
        n_candidates:  Random (φ, ψ) candidates sampled per beam state per
                       residue step (default 200).
        beam_width:    Maximum number of partial structures retained after
                       each residue step (default 10).
        overlap_tol:   VdW overlap tolerance in Å for hard-reject clash
                       filtering (default 0.6).
        check_intra:   If True, also reject candidates that clash with
                       already-placed loop residues ≥ *min_sep* away.
        min_sep:       Minimum residue separation for intra-loop clash
                       checking (default 3; bonded neighbours are skipped).
        rng_seed:      Optional integer seed for reproducibility.
        verbose:       Print per-residue progress.

    Returns:
        List of ``(N, CA, C, O, energy, closure_dist)`` tuples.

        * ``N, CA, C, O`` — (len(sequence), 3) float64 arrays.
        * ``energy``       — accumulated *energy_fn* score across all
                             placed residues; 0.0 when *energy_fn* is None.
        * ``closure_dist`` — always 0.0 (no closure attempted).

        The list contains at most *beam_width* entries.  It may be empty
        if the beam collapses because all candidates clash at some position.
    """
    n_loop = len(sequence)
    if n_loop == 0:
        if verbose:
            print("    Greedy placement: 0 structure(s) from 0 candidates")
        return []

    use_fw = fw_coords is not None and fw_radii is not None
    rng    = np.random.default_rng(rng_seed)

    # ── Beam state representation ─────────────────────────────────────────
    # Each state is a tuple:
    #   (ref_N, ref_CA, ref_C, ref_psi,
    #    N_list, CA_list, C_list, O_list,   ← Python lists of (3,) arrays
    #    energy)
    beam: list = [(
        prev_N.copy(), prev_CA.copy(), prev_C.copy(), float(psi_prev),
        [], [], [], [],   # no residues placed yet
        0.0,              # accumulated energy
    )]

    n_tried_total = 0
    n_clash_fw    = 0
    n_clash_intra = 0

    for i in range(n_loop):
        if not beam:
            break

        n_sampled_step  = len(beam) * n_candidates   # actual samples this step
        next_candidates: list = []   # (energy, new_state_tuple)

        for b_idx, state in enumerate(beam):
            ref_N, ref_CA, ref_C, ref_psi, N_list, CA_list, C_list, O_list, base_energy = state

            # ── Sample and place n_candidates for this beam entry ─────────
            phi_arr, psi_arr = _sample_torsion_batch(n_candidates, rng)
            n_tried_total += n_candidates

            N_arr, CA_arr, C_arr, O_arr = _batch_place_residues(
                ref_N, ref_CA, ref_C, ref_psi, phi_arr, psi_arr,
            )

            # ── Framework clash filter (vectorised) ───────────────────────
            if use_fw:
                fw_clash = _batch_fw_clash_mask(
                    N_arr, CA_arr, C_arr, O_arr,
                    fw_coords, fw_radii, overlap_tol,
                )
                n_clash_fw += int(fw_clash.sum())
                valid_mask = ~fw_clash
            else:
                valid_mask = np.ones(n_candidates, dtype=bool)

            valid_indices = np.where(valid_mask)[0]

            # ── Intra-loop + energy scoring (per surviving candidate) ──────
            N_placed  = np.array(N_list,  dtype=np.float64) if N_list  else np.empty((0, 3))
            CA_placed = np.array(CA_list, dtype=np.float64) if CA_list else np.empty((0, 3))
            C_placed  = np.array(C_list,  dtype=np.float64) if C_list  else np.empty((0, 3))
            O_placed  = np.array(O_list,  dtype=np.float64) if O_list  else np.empty((0, 3))

            for k in valid_indices:
                N_k  = N_arr[k]
                CA_k = CA_arr[k]
                C_k  = C_arr[k]
                O_k  = O_arr[k]

                # Intra-loop clash
                if check_intra and _residue_intra_clash(
                    N_placed, CA_placed, C_placed, O_placed,
                    N_k, CA_k, C_k, O_k,
                    overlap_tol, min_sep,
                ):
                    n_clash_intra += 1
                    continue

                # Energy score for this partial backbone
                step_energy = 0.0
                if energy_fn is not None:
                    has_placed = len(N_placed) > 0
                    full_N  = np.vstack([N_placed,  N_k [None, :]]) if has_placed else N_k [None, :]
                    full_CA = np.vstack([CA_placed, CA_k[None, :]]) if has_placed else CA_k[None, :]
                    full_C  = np.vstack([C_placed,  C_k [None, :]]) if has_placed else C_k [None, :]
                    full_O  = np.vstack([O_placed,  O_k [None, :]]) if has_placed else O_k [None, :]
                    step_energy = energy_weight * float(
                        energy_fn(full_N, full_CA, full_C, full_O)
                    )

                total_energy = base_energy + step_energy
                psi_k        = float(psi_arr[k])

                new_state = (
                    N_k.copy(), CA_k.copy(), C_k.copy(), psi_k,
                    N_list  + [N_k.copy()],
                    CA_list + [CA_k.copy()],
                    C_list  + [C_k.copy()],
                    O_list  + [O_k.copy()],
                    total_energy,
                )
                next_candidates.append((total_energy, new_state))

        if not next_candidates:
            if verbose:
                print(
                    f"    Residue {i+1}/{n_loop}: no valid candidates — "
                    f"beam collapsed "
                    f"(fw_clash={n_clash_fw}, intra_clash={n_clash_intra})"
                )
            beam = []
            break

        # ── Prune beam to top beam_width by energy ────────────────────────
        next_candidates.sort(key=lambda x: x[0])
        beam = [state for _, state in next_candidates[:beam_width]]

        if verbose:
            best_e  = next_candidates[0][0]
            n_valid = len(next_candidates)
            print(
                f"    Residue {i+1:2d}/{n_loop}: "
                f"{n_valid:4d} valid / {n_sampled_step:4d} sampled  "
                f"(fw_clash={n_clash_fw}, intra={n_clash_intra})  "
                f"best_energy={best_e:.3f}"
            )

    # ── Assemble output ensemble ──────────────────────────────────────────
    ensemble: List[EnsembleEntry] = []
    for state in beam:
        _, _, _, _, N_list, CA_list, C_list, O_list, energy = state
        if len(N_list) != n_loop:
            continue  # incomplete (should not happen if beam is non-empty)
        ensemble.append((
            np.array(N_list,  dtype=np.float64),
            np.array(CA_list, dtype=np.float64),
            np.array(C_list,  dtype=np.float64),
            np.array(O_list,  dtype=np.float64),
            energy,
            0.0,
        ))

    print(
        f"    Greedy placement: {len(ensemble)} structure(s) from "
        f"{n_tried_total:,} candidates  "
        f"(fw_clash={n_clash_fw}, intra_clash={n_clash_intra})"
    )
    return ensemble