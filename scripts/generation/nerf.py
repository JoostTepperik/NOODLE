"""
Natural Extension Reference Frame (NeRF) backbone builder.

Converts protein backbone torsion angles (φ, ψ, ω) to Cartesian coordinates
using ideal bond lengths and bond angles (Engh & Huber 1991).

The main entry point for loop modelling is :func:`build_loop`, which places
*n_loop* residues starting from the last N-terminal anchor residue.

Reference:
    Parsons et al. (2005) Practical conversion from torsion space to
    Cartesian space for in silico protein synthesis.
    J. Comput. Chem. 26, 1063–1068.
"""
from __future__ import annotations

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Ideal backbone geometry (Engh & Huber 1991)
# ─────────────────────────────────────────────────────────────────────────────

BOND_LENGTHS: dict[str, float] = {
    'N_CA': 1.458,   # N–Cα
    'CA_C': 1.525,   # Cα–C′
    'C_N':  1.329,   # C′–N  (peptide bond)
    'C_O':  1.231,   # C′=O  (carbonyl)
}

BOND_ANGLES_RAD: dict[str, float] = {
    'C_N_CA':  np.radians(121.7),   # C′–N–Cα   (angle at N)
    'N_CA_C':  np.radians(111.2),   # N–Cα–C′   (angle at Cα)
    'CA_C_N':  np.radians(116.2),   # Cα–C′–N   (angle at C′)
    'CA_C_O':  np.radians(120.8),   # Cα–C′=O   (angle at C′)
}


# ─────────────────────────────────────────────────────────────────────────────
# Core NeRF primitive
# ─────────────────────────────────────────────────────────────────────────────

def nerf(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    bond_length: float,
    bond_angle: float,
    dihedral: float,
) -> np.ndarray:
    """
    Place atom *d* given three reference atoms using the NeRF formula.

    *d* is bonded to *c* such that::

        |c – d|                = bond_length  (Å)
        ∠(b, c, d)             = bond_angle   (radians)
        dihedral(a, b, c, d)   = dihedral     (radians)

    Args:
        a, b, c:     (3,) reference atom Cartesian positions.
        bond_length: Distance c → d in Å.
        bond_angle:  Bond angle at c in radians (∠ b–c–d).
        dihedral:    Torsion angle a–b–c–d in radians.

    Returns:
        d: (3,) Cartesian coordinates of the placed atom.
    """
    bc     = c - b
    bc_hat = bc / np.linalg.norm(bc)

    # Component of (a – b) perpendicular to the b→c axis.
    n = a - b
    n = n - np.dot(n, bc_hat) * bc_hat
    if np.linalg.norm(n) < 1e-8:           # a, b, c collinear — pick arbitrary ⊥
        tmp = (np.array([0., 0., 1.])
               if abs(bc_hat[2]) < 0.9
               else np.array([1., 0., 0.]))
        n = tmp - np.dot(tmp, bc_hat) * bc_hat
    n_hat = n / np.linalg.norm(n)
    m_hat = np.cross(bc_hat, n_hat)

    # d in the local frame defined by (bc_hat, n_hat, m_hat).
    d_local = np.array([
        -bond_length * np.cos(bond_angle),
         bond_length * np.sin(bond_angle) * np.cos(dihedral),
         bond_length * np.sin(bond_angle) * np.sin(dihedral),
    ])
    return c + np.column_stack([bc_hat, n_hat, m_hat]) @ d_local


# ─────────────────────────────────────────────────────────────────────────────
# Torsion measurement helper
# ─────────────────────────────────────────────────────────────────────────────

def get_torsion(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    d: np.ndarray,
) -> float:
    """
    Measure the torsion angle a–b–c–d in radians ∈ (−π, π].
    """
    b1 = b - a
    b2 = c - b
    b3 = d - c
    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)
    n1n, n2n = np.linalg.norm(n1), np.linalg.norm(n2)
    if n1n < 1e-8 or n2n < 1e-8:
        return 0.0
    n1 /= n1n
    n2 /= n2n
    m = np.cross(n1, b2 / np.linalg.norm(b2))
    return float(np.arctan2(np.dot(m, n2), np.dot(n1, n2)))


# ─────────────────────────────────────────────────────────────────────────────
# Loop backbone builder
# ─────────────────────────────────────────────────────────────────────────────

def build_loop(
    prev_N:   np.ndarray,
    prev_CA:  np.ndarray,
    prev_C:   np.ndarray,
    psi_prev: float,
    phi:      np.ndarray,
    psi:      np.ndarray,
    omega:    np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build loop backbone coordinates (N, Cα, C′, O) via NeRF.

    The first loop residue's N atom is placed from the last N-terminal anchor
    residue (*prev_N*, *prev_CA*, *prev_C*) using *psi_prev*.  All subsequent
    atoms are placed in sequence using ideal geometry and the supplied
    *phi* / *psi* / *omega* arrays.

    Carbonyl O placement uses dihedral N–Cα–C′–O = psi + π, which positions
    O trans to the next N in the peptide plane (standard approximation).

    Closure check
    ~~~~~~~~~~~~~
    After building, the "virtual" N of the first C-terminal anchor residue
    can be computed as::

        virt_N = nerf(N[-1], CA[-1], C[-1],
                      BOND_LENGTHS['C_N'], BOND_ANGLES_RAD['CA_C_N'],
                      psi[-1])

    If ``|virt_N – target_N|`` is small the loop closes onto the C-anchor.
    Use :func:`loop_modeler.ccd_closure` to drive this distance to zero.

    Args:
        prev_N, prev_CA, prev_C:
            (3,) coordinates of the last N-terminal anchor residue.
        psi_prev:
            Psi torsion of the anchor residue (radians).  Determines where
            the first loop N is placed.
        phi:   (n_loop,) φ angles in radians for loop residues 0 … n−1.
        psi:   (n_loop,) ψ angles in radians for loop residues 0 … n−1.
        omega: (n_loop,) ω angles in radians.  Defaults to all π (trans).

    Returns:
        N, CA, C, O — each (n_loop, 3) float64 arrays.
    """
    n_loop = len(phi)
    if omega is None:
        omega = np.full(n_loop, np.pi)

    L = BOND_LENGTHS
    A = BOND_ANGLES_RAD

    N_arr  = np.empty((n_loop, 3))
    CA_arr = np.empty((n_loop, 3))
    C_arr  = np.empty((n_loop, 3))
    O_arr  = np.empty((n_loop, 3))

    # ── Residue 0: seed from last anchor atom ─────────────────────────────
    # psi_prev = N_prev–CA_prev–C_prev–N[0]
    N_arr[0]  = nerf(prev_N, prev_CA, prev_C,
                     L['C_N'], A['CA_C_N'], psi_prev)
    # omega[0] = CA_prev–C_prev–N[0]–CA[0]
    CA_arr[0] = nerf(prev_CA, prev_C, N_arr[0],
                     L['N_CA'], A['C_N_CA'], omega[0])
    # phi[0] = C_prev–N[0]–CA[0]–C[0]
    C_arr[0]  = nerf(prev_C, N_arr[0], CA_arr[0],
                     L['CA_C'], A['N_CA_C'], phi[0])
    O_arr[0]  = nerf(N_arr[0], CA_arr[0], C_arr[0],
                     L['C_O'], A['CA_C_O'], psi[0] + np.pi)

    # ── Residues 1 … n_loop−1 ────────────────────────────────────────────
    for i in range(1, n_loop):
        # psi[i-1] = N[i-1]–CA[i-1]–C[i-1]–N[i]
        N_arr[i]  = nerf(N_arr[i-1], CA_arr[i-1], C_arr[i-1],
                         L['C_N'], A['CA_C_N'], psi[i-1])
        # omega[i] = CA[i-1]–C[i-1]–N[i]–CA[i]
        CA_arr[i] = nerf(CA_arr[i-1], C_arr[i-1], N_arr[i],
                         L['N_CA'], A['C_N_CA'], omega[i])
        # phi[i] = C[i-1]–N[i]–CA[i]–C[i]
        C_arr[i]  = nerf(C_arr[i-1], N_arr[i], CA_arr[i],
                         L['CA_C'], A['N_CA_C'], phi[i])
        O_arr[i]  = nerf(N_arr[i], CA_arr[i], C_arr[i],
                         L['C_O'], A['CA_C_O'], psi[i] + np.pi)

    return N_arr, CA_arr, C_arr, O_arr


# ─────────────────────────────────────────────────────────────────────────────
# Closure measurement
# ─────────────────────────────────────────────────────────────────────────────

def measure_closure(
    N:         np.ndarray,
    CA:        np.ndarray,
    C:         np.ndarray,
    psi_last:  float,
    target_N:  np.ndarray,
    target_CA: np.ndarray,
) -> dict:
    """
    Measure loop closure quality after :func:`build_loop`.

    Virtually places the N and Cα of the first C-terminal anchor residue
    from the loop's last C′ atom and compares against the known target
    positions.

    Args:
        N, CA, C:   (n_loop, 3) arrays returned by :func:`build_loop`.
        psi_last:   psi[-1] used when building (places the virtual N).
        target_N:   (3,) known N position of first C-terminal anchor residue.
        target_CA:  (3,) known Cα position of first C-terminal anchor residue.

    Returns:
        dict with keys:

        * ``'bond_length'``: distance C′(last) → target_N in Å (ideal ≈ 1.329).
        * ``'virt_n_err'``:  |virtual_N − target_N| in Å.
        * ``'virt_ca_err'``: |virtual_Cα − target_Cα| in Å.
        * ``'rmsd'``:        RMSD of (virtual_N, virtual_Cα) vs targets in Å.
    """
    L = BOND_LENGTHS
    A = BOND_ANGLES_RAD

    virt_N  = nerf(N[-1], CA[-1], C[-1],
                   L['C_N'], A['CA_C_N'], psi_last)
    virt_CA = nerf(CA[-1], C[-1], virt_N,
                   L['N_CA'], A['C_N_CA'], np.pi)

    bond_len  = float(np.linalg.norm(C[-1] - target_N))
    n_err     = float(np.linalg.norm(virt_N  - target_N))
    ca_err    = float(np.linalg.norm(virt_CA - target_CA))
    rmsd      = float(np.sqrt(0.5 * (n_err**2 + ca_err**2)))

    return {
        'bond_length': bond_len,
        'virt_n_err':  n_err,
        'virt_ca_err': ca_err,
        'rmsd':        rmsd,
    }