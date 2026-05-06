"""
SE(3) frame-based loop modeling.

Pipeline
────────
  1. Random init   : independent SE(3) frames — broken inter-triangle geometry
  2. Sequential fix: one forward pass snapping each frame to exact geometry
                     relative to the previous frame, preserving torsion character
  3. Optimize      : rotations around bond axes + peptide bond length shortening

After step 2, intra-triangle geometry is exact. Inter-triangle peptide bonds
remain broken (lengths from random cloud, directions fixed by exact angles).

Optimization free variables
───────────────────────────
  psi[i]   — rotation around CA[i]-C[i] axis
              rotates ALL downstream frames i+1..n as a rigid body
              keeps all bond lengths and angles intact by construction

  phi[i]   — rotation around N[i]-CA[i] axis
              rotates all downstream frames from i onwards
              keeps all bond lengths and angles intact by construction

  d_CN[i]  — peptide bond length C[i]→N[i+1]  (initialized from random cloud)
              pure translation of ALL downstream frames i+1..n along the
              current C[i]→N[i+1] unit direction
              CA-C-N and C-N-CA angles unchanged (direction unchanged)
              only the scalar length changes

  EXACT throughout (by construction, no constraint needed):
    N-CA   = 1.458 Å
    CA-C   = 1.523 Å
    N-CA-C = 110.99°
    CA-C-N = 116.64°
    C-N-CA = 121.38°
    omega  = 180.0°

  Soft (only peptide bond length, initialized broken, optimized toward ideal):
    C-N  →  1.329 Å   (closure loss drives this at the terminal junction)
"""

import torch
import numpy as np
from typing import Tuple, List

# ─────────────────────────────────────────────────────────────────────────────
# Ideal backbone geometry
# ─────────────────────────────────────────────────────────────────────────────

BL_NCA   = 1.458
BL_CAC   = 1.523
BL_CN    = 1.329

BA_NCA_C = np.deg2rad(110.99)   # N-CA-C
BA_CA_CN = np.deg2rad(116.64)   # CA-C-N  (angle at C)
BA_CN_CA = np.deg2rad(121.38)   # C-N-CA  (angle at N)
OMEGA    = np.pi                # trans peptide


# ─────────────────────────────────────────────────────────────────────────────
# Ideal rigid triangle in local frame
#   CA at origin, C along +x, N in xy-plane
# ─────────────────────────────────────────────────────────────────────────────

def _build_local_triangle() -> np.ndarray:
    """Returns (3, 3): rows = [N, CA, C] in local frame."""
    CA = np.zeros(3)
    C  = np.array([BL_CAC, 0.0, 0.0])
    a  = np.pi - BA_NCA_C
    N  = np.array([-BL_NCA * np.cos(a), BL_NCA * np.sin(a), 0.0])
    return np.stack([N, CA, C], axis=0)

_LOCAL_TRIANGLE_NP = _build_local_triangle()   # (3, 3) numpy constant


# ─────────────────────────────────────────────────────────────────────────────
# SE(3) frame: rotation matrix + translation
# ─────────────────────────────────────────────────────────────────────────────

def frame_atoms(R: np.ndarray, t: np.ndarray):
    """
    Given rotation R (3,3) and translation t (3,), return world-frame
    N, CA, C positions by applying SE(3) to the local triangle.
    """
    world = (_LOCAL_TRIANGLE_NP @ R.T) + t   # (3, 3)
    return world[0], world[1], world[2]       # N, CA, C  each (3,)


def frame_from_atoms(N: np.ndarray, CA: np.ndarray, C: np.ndarray):
    """
    Recover (R, t) from observed N, CA, C positions.
    Frame: x = CA→C, y = Gram-Schmidt(CA→N vs x), z = x×y, t = CA.
    """
    x  = C - CA;         x  /= np.linalg.norm(x)  + 1e-8
    cn = N - CA
    y  = cn - np.dot(cn, x) * x;  y /= np.linalg.norm(y) + 1e-8
    z  = np.cross(x, y);           z /= np.linalg.norm(z) + 1e-8
    R  = np.column_stack([x, y, z])   # (3,3): columns are world axes in local coords
    return R, CA.copy()


# ─────────────────────────────────────────────────────────────────────────────
# Place one atom via NeRF-style placement (numpy, used only in sequential fix)
# ─────────────────────────────────────────────────────────────────────────────

def _place_atom_np(a, b, c, bond_length, bond_angle, torsion):
    """
    Place atom d given three reference atoms a, b, c and ideal geometry.
    Pure numpy, used once during the sequential fix — not in optimization.
    """
    bc   = c - b
    bc_n = bc / (np.linalg.norm(bc) + 1e-8)

    n_abc = np.cross(b - a, bc)
    n_abc = n_abc / (np.linalg.norm(n_abc) + 1e-8)

    col2 = np.cross(n_abc, bc_n)
    M    = np.column_stack([bc_n, col2, n_abc])   # (3, 3)

    d_local = np.array([
        -np.cos(bond_angle),
         np.sin(bond_angle) * np.cos(torsion),
        -np.sin(bond_angle) * np.sin(torsion),
    ]) * bond_length

    return c + M @ d_local


# ─────────────────────────────────────────────────────────────────────────────
# Extract psi / phi from a frame relative to the previous frame
# ─────────────────────────────────────────────────────────────────────────────

def _extract_psi(N_prev, CA_prev, C_prev, N_cur):
    """psi[i] = dihedral(N[i], CA[i], C[i], N[i+1])"""
    return _dihedral_np(N_prev, CA_prev, C_prev, N_cur)


def _extract_phi(C_prev, N_cur, CA_cur, C_cur):
    """phi[i] = dihedral(C[i-1], N[i], CA[i], C[i])"""
    return _dihedral_np(C_prev, N_cur, CA_cur, C_cur)


def _dihedral_np(p0, p1, p2, p3):
    b1 = p0 - p1;  b2 = p2 - p1;  b3 = p3 - p2
    n1 = np.cross(b1, b2);  n1 /= np.linalg.norm(n1) + 1e-8
    n2 = np.cross(b2, b3);  n2 /= np.linalg.norm(n2) + 1e-8
    b2n = b2 / (np.linalg.norm(b2) + 1e-8)
    m1  = np.cross(n1, b2n)
    return np.arctan2(np.dot(m1, n2), np.dot(n1, n2))


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Sequential geometry fix
#
# For each residue i (in order 0..n-1):
#   - Previous frame gives exact C[i-1], N[i-1], CA[i-1]
#   - Extract approximate psi from random frame i relative to prev frame
#   - Use that psi to place N[i] exactly (correct bond length + angles + omega)
#   - Extract approximate phi from random frame's C[i] relative to new N[i]
#   - Use that phi to place CA[i] and C[i] exactly
#   - Build new SE(3) frame from the snapped N[i], CA[i], C[i]
#
# Result: all inter-triangle geometry exact, torsions ≈ random init values
# ─────────────────────────────────────────────────────────────────────────────

def sequential_geometry_fix(
    frames_R:  List[np.ndarray],   # list of (3,3) rotation matrices, length n
    frames_t:  List[np.ndarray],   # list of (3,) translations, length n
    anchor_N:  np.ndarray,         # (3,) N  of last flank_before residue
    anchor_CA: np.ndarray,         # (3,) CA of last flank_before residue
    anchor_C:  np.ndarray,         # (3,) C  of last flank_before residue
) -> Tuple[List[np.ndarray], List[np.ndarray], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    One forward pass: snap each frame to exact geometry relative to prev,
    preserving the torsion character of the random cloud.

    The inter-triangle angles (CA-C-N, C-N-CA, omega) and intra-triangle
    geometry are all made exact. The peptide bond LENGTHS are initialized
    from the random cloud's implied distances — they remain broken and are
    the only free scalar per junction going into optimization.

    How: for each residue i, the random frame's N position implies a psi angle.
    We snap N[i] to the exact bond direction (correct angles + omega) but keep
    the scalar distance as d_CN[i] = ||N_rand - prev_C||. This gives each
    junction a random broken bond length with an exact bond direction.

    Returns:
        fixed_R    : list of (3,3) rotation matrices
        fixed_t    : list of (3,) translations
        N_arr      : (n, 3) fixed N positions
        CA_arr     : (n, 3) fixed CA positions
        C_arr      : (n, 3) fixed C positions
        d_CN_arr   : (n,)   peptide bond lengths (broken, from random cloud)
    """
    n = len(frames_R)

    fixed_R, fixed_t = [], []
    N_arr  = np.zeros((n, 3))
    CA_arr = np.zeros((n, 3))
    C_arr  = np.zeros((n, 3))
    d_CN_arr = np.zeros(n, dtype=np.float32)

    prev_N  = anchor_N.copy()
    prev_CA = anchor_CA.copy()
    prev_C  = anchor_C.copy()

    for i in range(n):
        R_rand, t_rand = frames_R[i], frames_t[i]
        N_rand, CA_rand, C_rand = frame_atoms(R_rand, t_rand)

        psi_approx = _extract_psi(prev_N, prev_CA, prev_C, N_rand)
        N_exact = _place_atom_np(prev_N, prev_CA, prev_C, BL_CN, BA_CA_CN, psi_approx)
        CA_fixed = _place_atom_np(prev_CA, prev_C, N_exact, BL_NCA, BA_CN_CA, OMEGA)
        phi_approx = _extract_phi(prev_C, N_exact, CA_fixed, C_rand)
        C_fixed = _place_atom_np(prev_C, N_exact, CA_fixed, BL_CAC, BA_NCA_C, phi_approx)

        cn_dir = N_exact - prev_C
        cn_dir = cn_dir / (np.linalg.norm(cn_dir) + 1e-8)
        d_rand = float(np.linalg.norm(N_rand - prev_C))
        delta  = cn_dir * (d_rand - BL_CN)

        N_fixed  = N_exact  + delta
        CA_fixed = CA_fixed + delta
        C_fixed  = C_fixed  + delta

        R_fixed, t_fixed = frame_from_atoms(N_fixed, CA_fixed, C_fixed)
        fixed_R.append(R_fixed)
        fixed_t.append(t_fixed)
        N_arr[i]    = N_fixed
        CA_arr[i]   = CA_fixed
        C_arr[i]    = C_fixed
        d_CN_arr[i] = d_rand

        prev_N  = N_fixed
        prev_CA = CA_fixed
        prev_C  = C_fixed

    return fixed_R, fixed_t, N_arr, CA_arr, C_arr, d_CN_arr


# ─────────────────────────────────────────────────────────────────────────────
# Rodrigues rotation around an arbitrary axis (batched, differentiable)
# ─────────────────────────────────────────────────────────────────────────────

def _rotate_points_around_axis(
    points: torch.Tensor,   # (B, m, 3)  points to rotate
    origin: torch.Tensor,   # (B, 3)     point on axis
    axis:   torch.Tensor,   # (B, 3)     unit axis direction
    angle:  torch.Tensor,   # (B,)       rotation angle in radians
) -> torch.Tensor:          # (B, m, 3)
    """
    Rotate all points around the axis passing through origin by angle.
    Uses Rodrigues' rotation formula. Differentiable w.r.t. angle.
    """
    axis = axis / (axis.norm(dim=-1, keepdim=True) + 1e-8)   # ensure unit
    p    = points - origin.unsqueeze(1)                        # (B, m, 3) relative to origin
    a    = axis.unsqueeze(1)                                   # (B, 1, 3)
    c    = torch.cos(angle).unsqueeze(-1).unsqueeze(-1)        # (B, 1, 1)
    s    = torch.sin(angle).unsqueeze(-1).unsqueeze(-1)

    # Rodrigues: p*cos + (k×p)*sin + k*(k·p)*(1-cos)
    cross   = torch.linalg.cross(a.expand_as(p), p)           # (B, m, 3)
    dot     = (a * p).sum(-1, keepdim=True)                    # (B, m, 1)
    rotated = p * c + cross * s + a * dot * (1 - c)

    return rotated + origin.unsqueeze(1)


# ─────────────────────────────────────────────────────────────────────────────
# Build chain from initial positions + cumulative bond-axis rotations + lengths
# ─────────────────────────────────────────────────────────────────────────────

def build_chain_from_frames(
    N_init:   torch.Tensor,   # (B, n, 3)  initial N  positions (after fix)
    CA_init:  torch.Tensor,   # (B, n, 3)  initial CA positions
    C_init:   torch.Tensor,   # (B, n, 3)  initial C  positions
    psi:      torch.Tensor,   # (B, n)     rotation around CA[i]-C[i] axis
    phi:      torch.Tensor,   # (B, n)     rotation around N[i]-CA[i] axis
    d_CN:     torch.Tensor,   # (B, n-1)   inner peptide bond lengths C[i]→N[i+1]
    d_anc:    torch.Tensor,   # (B,)       anchor bond length anchor_C→N[0]
    anchor_C: torch.Tensor,   # (B, 3)     fixed anchor C
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build final atom positions by applying cumulative bond-axis rotations
    and bond length changes to the fixed-geometry initial frames.

    d_anc controls anchor_C→N[0] distance — translates the entire chain
    along the anchor_C→N[0] unit direction.

    d_CN[i] controls C[i]→N[i+1] — translates residues i+1..n-1 along
    the current C[i]→N[i+1] unit direction.

    Returns N, CA, C each (B, n, 3).
    """
    B, n = psi.shape

    N  = N_init.clone()
    CA = CA_init.clone()
    C  = C_init.clone()

    # ── Anchor bond: translate entire chain along anchor_C→N[0] ──────────────
    anc_vec  = N[:, 0] - anchor_C                                  # (B, 3)
    anc_unit = anc_vec / (anc_vec.norm(dim=-1, keepdim=True) + 1e-8)
    anc_curr = anc_vec.norm(dim=-1)                                # (B,)
    anc_delta = (d_anc - anc_curr).unsqueeze(-1) * anc_unit        # (B, 3)
    N  = N  + anc_delta.unsqueeze(1)
    CA = CA + anc_delta.unsqueeze(1)
    C  = C  + anc_delta.unsqueeze(1)

    for i in range(n):
        # ── psi[i]: rotate around CA[i]-C[i] axis ────────────────────────────
        if i < n - 1:
            psi_axis   = C[:, i] - CA[:, i]
            downstream = torch.arange(i + 1, n)
            pts        = torch.stack([N[:, j] for j in downstream] +
                                     [CA[:, j] for j in downstream] +
                                     [C[:, j] for j in downstream], dim=1)
            rotated = _rotate_points_around_axis(pts, CA[:, i], psi_axis, psi[:, i])
            m = len(downstream)
            for k, j in enumerate(downstream):
                N[:, j]  = rotated[:, k]
                CA[:, j] = rotated[:, k + m]
                C[:, j]  = rotated[:, k + 2 * m]

        # ── phi[i]: rotate around N[i]-CA[i] axis ────────────────────────────
        phi_axis = CA[:, i] - N[:, i]
        all_pts  = [C[:, i]] + [N[:, j]  for j in range(i+1, n)] + \
                               [CA[:, j] for j in range(i+1, n)] + \
                               [C[:, j]  for j in range(i+1, n)]
        if all_pts:
            pts     = torch.stack(all_pts, dim=1)
            rotated = _rotate_points_around_axis(pts, N[:, i], phi_axis, phi[:, i])
            C[:, i] = rotated[:, 0]
            m = n - i - 1
            for k, j in enumerate(range(i+1, n)):
                N[:, j]  = rotated[:, 1 + k]
                CA[:, j] = rotated[:, 1 + m + k]
                C[:, j]  = rotated[:, 1 + 2*m + k]

        # ── d_CN[i]: translate residues i+1..n along C[i]→N[i+1] ─────────────
        if i < n - 1:
            cn_vec  = N[:, i + 1] - C[:, i]
            cn_unit = cn_vec / (cn_vec.norm(dim=-1, keepdim=True) + 1e-8)
            cn_curr = cn_vec.norm(dim=-1)
            delta   = (d_CN[:, i] - cn_curr).unsqueeze(-1) * cn_unit
            for j in range(i + 1, n):
                N[:, j]  = N[:, j]  + delta
                CA[:, j] = CA[:, j] + delta
                C[:, j]  = C[:, j]  + delta

    return N, CA, C


# ─────────────────────────────────────────────────────────────────────────────
# Energy scoring
# ─────────────────────────────────────────────────────────────────────────────

def _interp_prob(angle_deg, probs):
    n_bins     = len(probs)
    angle_norm = torch.fmod(angle_deg + 360.0, 360.0)
    bin_idx    = (angle_norm / 360.0) * n_bins
    idx_lo     = torch.floor(bin_idx).long() % n_bins
    idx_hi     = (idx_lo + 1) % n_bins
    w          = bin_idx - torch.floor(bin_idx)
    pt         = torch.tensor(probs, dtype=torch.float32, device=angle_deg.device)
    return (1.0 - w) * pt[idx_lo] + w * pt[idx_hi]


def compute_energy(phi_rad, psi_rad, probs_phi, probs_psi):
    """Returns (B,)."""
    energy  = torch.zeros(phi_rad.shape[0], device=phi_rad.device)
    phi_deg = torch.rad2deg(phi_rad)
    psi_deg = torch.rad2deg(psi_rad)
    for i in range(phi_rad.shape[1]):
        if i < len(probs_phi):
            energy -= torch.log(_interp_prob(phi_deg[:, i], probs_phi[i]) + 1e-10)
        if i < len(probs_psi):
            energy -= torch.log(_interp_prob(psi_deg[:, i], probs_psi[i]) + 1e-10)
    return energy


# ─────────────────────────────────────────────────────────────────────────────
# Dihedral from positions (differentiable)
# ─────────────────────────────────────────────────────────────────────────────

def _dihedral_torch(p0, p1, p2, p3):
    """All inputs (B, 3). Returns (B,) radians."""
    b1 = p1 - p0;  b2 = p2 - p1;  b3 = p3 - p2
    n1 = torch.linalg.cross(b1, b2)
    n2 = torch.linalg.cross(b2, b3)
    n1 = n1 / (n1.norm(dim=-1, keepdim=True) + 1e-8)
    n2 = n2 / (n2.norm(dim=-1, keepdim=True) + 1e-8)
    b2n = b2 / (b2.norm(dim=-1, keepdim=True) + 1e-8)
    m1  = torch.linalg.cross(n1, b2n)
    return torch.atan2((m1 * n2).sum(-1), (n1 * n2).sum(-1))


def extract_phi_psi_from_positions(
    N:      torch.Tensor,   # (B, n, 3)
    CA:     torch.Tensor,
    C:      torch.Tensor,
    anc_C:  torch.Tensor,   # (B, 3)
    clos_N: torch.Tensor,   # (B, 3)
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract phi/psi from atom positions. Returns (B, n) each."""
    B, n, _ = N.shape
    phi_list, psi_list = [], []
    for i in range(n):
        c_prev = anc_C       if i == 0     else C[:, i - 1]
        n_next = clos_N      if i == n - 1 else N[:, i + 1]
        phi_list.append(_dihedral_torch(c_prev,   N[:, i], CA[:, i],  C[:, i]))
        psi_list.append(_dihedral_torch(N[:, i], CA[:, i],  C[:, i], n_next))
    return torch.stack(phi_list, 1), torch.stack(psi_list, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Optimization
# ─────────────────────────────────────────────────────────────────────────────

def optimize_se3_bondaxis(
    N_init:    torch.Tensor,
    CA_init:   torch.Tensor,
    C_init:    torch.Tensor,
    psi_init:  torch.Tensor,   # (B, n)
    phi_init:  torch.Tensor,   # (B, n)
    d_CN_init: torch.Tensor,   # (B, n-1)
    d_anc_init:torch.Tensor,   # (B,)
    anchor_C:  np.ndarray,
    N_closure: np.ndarray,
    probs_phi: list,
    probs_psi: list,
    n_steps:        int   = 1000,
    lr_torsion:     float = 0.05,
    lr_bond:        float = 2.0,   # high LR: bonds start 20-90Å from target
    bond_weight:    float = 10.0,  # separate from closure_weight
    closure_weight: float = 50.0,
    n_frames:       int   = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list]:
    """
    Optimize psi/phi rotations + all bond lengths.

    d_anc: scalar per structure controlling anchor_C→N[0] distance.
           Translates the entire chain rigidly — keeps all geometry intact.
    d_CN:  per-junction inner bond lengths C[i]→N[i+1].

    Returns (psi, phi, d_CN, d_anc, trajectory).
    trajectory is a list of (step, N, CA, C) snapshots for structure 0,
    or empty list if n_frames=0.
    """
    B, n = psi_init.shape

    psi   = psi_init.clone().requires_grad_(True)
    phi   = phi_init.clone().requires_grad_(True)
    d_CN  = d_CN_init.clone().requires_grad_(True)
    d_anc = d_anc_init.clone().requires_grad_(True)

    anc_C_t  = torch.tensor(anchor_C,  dtype=torch.float32).unsqueeze(0).expand(B, -1)
    clos_N_t = torch.tensor(N_closure, dtype=torch.float32).unsqueeze(0).expand(B, -1)

    optimizer = torch.optim.Adam([
        {'params': [psi, phi], 'lr': lr_torsion},
        {'params': [d_CN, d_anc], 'lr': lr_bond},
    ])

    best_cl        = torch.full((B,), float('inf'))   # tracks energy + closure_weight * cl_dist
    best_psi_ckpt  = psi_init.clone()
    best_phi_ckpt  = phi_init.clone()
    best_dcn_ckpt  = d_CN_init.clone()
    best_danc_ckpt = d_anc_init.clone()

    trajectory = []
    frame_steps = set()
    if n_frames > 0:
        frame_steps = set(np.linspace(0, n_steps - 1, n_frames, dtype=int).tolist())

    print(f"      Bond-axis optimisation  B={B}  n_loop={n}  n_steps={n_steps}")
    print(f"      Initial d_anc: mean={d_anc_init.mean():.3f}Å  (target {BL_CN}Å)")
    print(f"      Initial d_CN:  mean={d_CN_init.mean():.3f}Å ± {d_CN_init.std():.3f}Å  (target {BL_CN}Å)")

    for step in range(n_steps):
        optimizer.zero_grad()

        N, CA, C = build_chain_from_frames(
            N_init, CA_init, C_init, psi, phi, d_CN, d_anc, anc_C_t)

        phi_pos, psi_pos = extract_phi_psi_from_positions(
            N, CA, C, anc_C_t, clos_N_t)
        energy = compute_energy(phi_pos, psi_pos, probs_phi, probs_psi).mean()

        # All bond lengths toward BL_CN — L1 loss so large initial distances
        # don't dwarf the gradient; constant weight (not annealed) so bonds
        # converge throughout the run.
        anc_bond    = (N[:, 0] - anc_C_t).norm(dim=-1)                    # (B,) == d_anc
        l_anc_bond  = (anc_bond  - BL_CN).abs().mean()
        l_inner     = torch.tensor(0.0)
        if n > 1:
            inner_bonds = (N[:, 1:] - C[:, :-1]).norm(dim=-1)             # (B, n-1)
            l_inner     = (inner_bonds - BL_CN).abs().mean()

        # Closure (annealed)
        cl_dist   = (C[:, -1] - clos_N_t).norm(dim=-1)                    # (B,)
        progress  = step / max(n_steps - 1, 1)
        l_closure = ((cl_dist - BL_CN) ** 2).mean()

        loss = (energy
                + bond_weight * (l_anc_bond + l_inner)
                + closure_weight * progress * l_closure)
        loss.backward()
        optimizer.step()

        # Clamp bond lengths to physically valid range after each step
        with torch.no_grad():
            d_CN.clamp_(min=0.5)
            d_anc.clamp_(min=0.5)

        # Checkpoint: save when energy + closure jointly improve
        # (closure-only checkpoint can save states with bad geometry)
        with torch.no_grad():
            score = energy.detach() + closure_weight * cl_dist  # (B,) scalar per structure
            improved = score < best_cl
            best_cl[improved]         = score[improved]
            best_psi_ckpt[improved]   = psi.detach()[improved]
            best_phi_ckpt[improved]   = phi.detach()[improved]
            best_dcn_ckpt[improved]   = d_CN.detach()[improved]
            best_danc_ckpt[improved]  = d_anc.detach()[improved]

        # Trajectory snapshot (structure 0 only)
        if step in frame_steps:
            with torch.no_grad():
                trajectory.append((step,
                                   N[0].clone(),
                                   CA[0].clone(),
                                   C[0].clone()))

        if step % 200 == 0 or step == n_steps - 1:
            with torch.no_grad():
                inner_bl = (N[:, 1:] - C[:, :-1]).norm(dim=-1).mean().item() if n > 1 else 0.0
            print(f"        step {step:4d}:  "
                  f"E={energy.item():.2f}  "
                  f"anc={anc_bond.mean().item():.3f}Å  "
                  f"inner={inner_bl:.3f}Å  "
                  f"closure={cl_dist.mean().item():.4f}Å")

    return best_psi_ckpt, best_phi_ckpt, best_dcn_ckpt, best_danc_ckpt, trajectory


# ─────────────────────────────────────────────────────────────────────────────
# Geometry validation
# ─────────────────────────────────────────────────────────────────────────────

def validate_geometry(N, CA, C, label=""):
    """
    Print bond lengths and angles for a single structure.
    N, CA, C: (n, 3) numpy arrays.
    """
    n = len(CA)
    nca  = np.linalg.norm(CA - N, axis=-1)
    cac  = np.linalg.norm(C - CA, axis=-1)

    # N-CA-C angles
    ba_nca_c = []
    for i in range(n):
        v1 = N[i] - CA[i];  v2 = C[i] - CA[i]
        cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
        ba_nca_c.append(np.degrees(np.arccos(np.clip(cos_a, -1, 1))))

    print(f"  Geometry check {label}:")
    print(f"    N-CA:    mean={nca.mean():.6f}Å  std={nca.std():.2e}  (target {BL_NCA})")
    print(f"    CA-C:    mean={cac.mean():.6f}Å  std={cac.std():.2e}  (target {BL_CAC})")
    print(f"    N-CA-C:  mean={np.mean(ba_nca_c):.4f}°  std={np.std(ba_nca_c):.2e}"
          f"  (target {np.degrees(BA_NCA_C):.2f}°)")

    if n > 1:
        cn   = np.linalg.norm(N[1:] - C[:-1], axis=-1)
        print(f"    C-N:     mean={cn.mean():.6f}Å  std={cn.std():.2e}  (target {BL_CN})")

        cacn, cnca, omegas = [], [], []
        for i in range(n - 1):
            v1 = CA[i] - C[i];  v2 = N[i+1] - C[i]
            cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
            cacn.append(np.degrees(np.arccos(np.clip(cos_a, -1, 1))))

            v1 = C[i] - N[i+1];  v2 = CA[i+1] - N[i+1]
            cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
            cnca.append(np.degrees(np.arccos(np.clip(cos_a, -1, 1))))

            om = _dihedral_np(CA[i], C[i], N[i+1], CA[i+1])
            omegas.append(np.degrees(om))

        # ±180° are identical trans conformations — use |omega| to avoid
        # spurious mean=0 when values alternate between +180 and -180
        omegas_abs = np.abs(omegas)
        print(f"    CA-C-N:  mean={np.mean(cacn):.4f}°  std={np.std(cacn):.2e}"
              f"  (target {np.degrees(BA_CA_CN):.2f}°)")
        print(f"    C-N-CA:  mean={np.mean(cnca):.4f}°  std={np.std(cnca):.2e}"
              f"  (target {np.degrees(BA_CN_CA):.2f}°)")
        print(f"    omega:   mean={np.mean(omegas_abs):.4f}°  std={np.std(omegas_abs):.2e}"
              f"  (target 180.00°)  [reported as |omega|]")


# ─────────────────────────────────────────────────────────────────────────────
# O atom computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_O_atoms(N, CA, C):
    n, O = len(CA), np.zeros((len(CA), 3))
    for i in range(n):
        v_ca = CA[i] - C[i];  v_ca /= np.linalg.norm(v_ca) + 1e-8
        if i < n - 1:
            v_n  = N[i+1] - C[i];  v_n /= np.linalg.norm(v_n) + 1e-8
            bis  = v_ca + v_n;      bn = np.linalg.norm(bis)
            O[i] = C[i] + 1.229 * (bis / bn if bn > 1e-8 else v_ca)
        else:
            O[i] = C[i] - 1.229 * v_ca
    return O


# ─────────────────────────────────────────────────────────────────────────────
# Trajectory PDB writer
# ─────────────────────────────────────────────────────────────────────────────

ONE_TO_THREE = {
    'A':'ALA','C':'CYS','D':'ASP','E':'GLU','F':'PHE','G':'GLY',
    'H':'HIS','I':'ILE','K':'LYS','L':'LEU','M':'MET','N':'ASN',
    'P':'PRO','Q':'GLN','R':'ARG','S':'SER','T':'THR','V':'VAL',
    'W':'TRP','Y':'TYR',
}

def save_trajectory_pdb(
    trajectory,              # list of (step, N_loop, CA_loop, C_loop) tensors
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
    CA_native_loop:   np.ndarray,   # (n_loop, 3) for RMSD annotation
    output_path:      str,
):
    """
    Write a multi-MODEL PDB file showing the optimization trajectory.
    Each MODEL corresponds to one trajectory snapshot.
    B-factor column contains per-residue CA deviation from native (×10).
    """
    from pathlib import Path
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        for model_idx, (step, N_t, CA_t, C_t) in enumerate(trajectory, 1):
            N_loop  = N_t.cpu().numpy()  if hasattr(N_t,  'numpy') else N_t
            CA_loop = CA_t.cpu().numpy() if hasattr(CA_t, 'numpy') else CA_t
            C_loop  = C_t.cpu().numpy()  if hasattr(C_t,  'numpy') else C_t
            O_loop  = compute_O_atoms(N_loop, CA_loop, C_loop)

            N_full  = np.vstack([N_flank_before,  N_loop,  N_flank_after])
            CA_full = np.vstack([CA_flank_before, CA_loop, CA_flank_after])
            C_full  = np.vstack([C_flank_before,  C_loop,  C_flank_after])
            O_full  = np.vstack([O_flank_before,  O_loop,  O_flank_after])

            # Per-residue RMSD from native for B-factor colouring
            rmsd_loop = np.sqrt(np.sum((CA_loop - CA_native_loop)**2, axis=1))
            rmsd_full = np.zeros(len(full_sequence))
            rmsd_full[loop_start:loop_end] = rmsd_loop * 10.0   # scale for visibility

            loop_rmsd = float(np.sqrt(np.mean(rmsd_loop**2)))

            f.write(f"MODEL {model_idx:6d}\n")
            f.write(f"REMARK step={step}  loop_rmsd={loop_rmsd:.3f}A\n")

            atom_num = 1
            for i, aa in enumerate(full_sequence):
                res = ONE_TO_THREE.get(aa, 'UNK')
                bf  = rmsd_full[i]
                for aname, coord in [('N', N_full[i]), ('CA', CA_full[i]), ('C', C_full[i])]:
                    f.write(f"ATOM  {atom_num:5d}  {aname:<3s} {res} A{i+1:4d}    "
                            f"{coord[0]:8.3f}{coord[1]:8.3f}{coord[2]:8.3f}"
                            f"  1.00{bf:6.2f}           {aname[0]}  \n")
                    atom_num += 1
                if np.linalg.norm(O_full[i]) > 1e-6:
                    f.write(f"ATOM  {atom_num:5d}  O   {res} A{i+1:4d}    "
                            f"{O_full[i,0]:8.3f}{O_full[i,1]:8.3f}{O_full[i,2]:8.3f}"
                            f"  1.00{bf:6.2f}           O  \n")
                    atom_num += 1
            f.write("ENDMDL\n")

    print(f"    Trajectory: {len(trajectory)} frames → {output_path}")

    # Write companion PyMOL script
    from pathlib import Path as _Path
    pdb_name   = _Path(output_path).stem          # e.g. trajectory_7pbc_b_E
    pml_path   = str(_Path(output_path).with_suffix('.pml'))
    n_models   = len(trajectory)
    resi_start = loop_start + 1                   # 1-indexed
    resi_end   = loop_end                         # loop_end is exclusive, so last resi = loop_end

    with open(pml_path, 'w') as pml:
        pml.write(f"# Optimisation trajectory: {pdb_name}\n")
        pml.write(f"# {n_models} states  |  loop resi {resi_start}-{resi_end}\n")
        pml.write(f"# B-factor = per-residue RMSD vs native x 10\n")
        pml.write(f"load {_Path(output_path).name}, traj\n")
        pml.write(f"bg_color white\n")
        pml.write(f"set cartoon_smooth_loops, 1\n")
        pml.write(f"set cartoon_fancy_helices, 1\n")
        pml.write(f"hide everything, traj\n")
        pml.write(f"show cartoon, traj\n")
        pml.write(f"color grey80, traj\n")
        pml.write(f"spectrum b, blue_white_red, traj and resi {resi_start}-{resi_end}, minimum=0, maximum=30\n")
        pml.write(f"mset 1 -{n_models}\n")
        pml.write(f"set movie_fps, 4\n")
        pml.write(f"zoom traj\n")
        pml.write(f"mplay\n")

    print(f"    PyMOL script  → {pml_path}")

def refine_loop_se3_fixed(
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
    model,
    params,
    n_steps:          int   = 1000,
    lr_torsion:       float = 0.05,
    lr_bond:          float = 2.0,
    bond_weight:      float = 10.0,
    closure_weight:   float = 50.0,
    n_structures:     int   = 10,
    position_scale:   float = 10.0,
    seed:             int   = None,
    n_frames:         int   = 0,     # trajectory snapshots (0 = off)
    trajectory_path:  str   = None,  # where to write trajectory PDB
    CA_native_loop:   np.ndarray = None,  # required if n_frames > 0
):
    """
    SE(3) loop modeling: random cloud → sequential fix → bond-axis optimization.

    Free variables:
      psi[i], phi[i]  — torsion angles via bond-axis rotations
      d_CN[i]         — inner peptide bond lengths C[i]→N[i+1]
      d_anc           — anchor bond length anchor_C→N[0]

    All angles, intra-triangle geometry, and omega are exact throughout.
    """
    from loop_modeling_nerf import cache_energy_distributions
    from random_backbone_cloud import generate_random_backbone_cloud

    loop_seq = full_sequence[loop_start:loop_end]
    n_loop   = len(loop_seq)

    print(f"\n  SE(3)-fixed loop refinement:")
    print(f"    Sequence  : {full_sequence}  loop={loop_seq} ({n_loop} res)")
    print(f"    Anchor C → closure N: "
          f"{np.linalg.norm(C_flank_before[-1] - N_flank_after[0]):.2f} Å")

    anc_N  = N_flank_before[-1].copy()
    anc_CA = CA_flank_before[-1].copy()
    anc_C  = C_flank_before[-1].copy()
    N_clos = N_flank_after[0].copy()

    probs_phi, probs_psi = cache_energy_distributions(model, params, loop_seq)

    all_N_init    = []
    all_CA_init   = []
    all_C_init    = []
    all_psi_init  = []
    all_phi_init  = []
    all_dcn_init  = []
    all_danc_init = []

    for idx in range(n_structures):
        s = None if seed is None else seed + idx
        N_c, CA_c, C_c, _ = generate_random_backbone_cloud(
            n_residues=n_loop, position_scale=position_scale, seed=s)

        frames_R, frames_t = [], []
        for i in range(n_loop):
            R_i, t_i = frame_from_atoms(N_c[i], CA_c[i], C_c[i])
            frames_R.append(R_i)
            frames_t.append(t_i)

        _, _, N_fix, CA_fix, C_fix, d_CN_fix = sequential_geometry_fix(
            frames_R, frames_t, anc_N, anc_CA, anc_C)

        if idx == 0:
            print(f"\n    After sequential fix (structure 1):")
            validate_geometry(N_fix, CA_fix, C_fix, label="fixed")

        # Extract torsion angles
        psi_row = np.zeros(n_loop, dtype=np.float32)
        phi_row = np.zeros(n_loop, dtype=np.float32)
        prev_N_f = anc_N;  prev_CA_f = anc_CA;  prev_C_f = anc_C
        for i in range(n_loop):
            psi_row[i] = _dihedral_np(prev_N_f, prev_CA_f, prev_C_f, N_fix[i])
            phi_row[i] = _dihedral_np(prev_C_f, N_fix[i], CA_fix[i], C_fix[i])
            prev_N_f = N_fix[i];  prev_CA_f = CA_fix[i];  prev_C_f = C_fix[i]

        # Anchor bond length: distance from anchor_C to N_fix[0]
        d_anc_val = float(np.linalg.norm(N_fix[0] - anc_C))

        # Inner bond lengths: d_CN_fix already has these from the fix
        # d_CN_fix has length n, but we only need n-1 inner bonds
        # (index 0 is the anchor bond, indices 1..n-1 are inner bonds)
        d_cn_inner = d_CN_fix[1:] if n_loop > 1 else np.array([], dtype=np.float32)

        all_N_init.append(torch.tensor(N_fix,      dtype=torch.float32))
        all_CA_init.append(torch.tensor(CA_fix,    dtype=torch.float32))
        all_C_init.append(torch.tensor(C_fix,      dtype=torch.float32))
        all_psi_init.append(torch.tensor(psi_row,  dtype=torch.float32))
        all_phi_init.append(torch.tensor(phi_row,  dtype=torch.float32))
        all_dcn_init.append(torch.tensor(d_cn_inner, dtype=torch.float32))
        all_danc_init.append(torch.tensor([d_anc_val], dtype=torch.float32))

    N_batch    = torch.stack(all_N_init)     # (B, n, 3)
    CA_batch   = torch.stack(all_CA_init)
    C_batch    = torch.stack(all_C_init)
    psi_batch  = torch.stack(all_psi_init)   # (B, n)
    phi_batch  = torch.stack(all_phi_init)
    dcn_batch  = torch.stack(all_dcn_init)   # (B, n-1)
    danc_batch = torch.cat(all_danc_init)    # (B,)

    psi_opt, phi_opt, dcn_opt, danc_opt, trajectory = optimize_se3_bondaxis(
        N_batch, CA_batch, C_batch,
        psi_batch, phi_batch, dcn_batch, danc_batch,
        anc_C, N_clos,
        probs_phi, probs_psi,
        n_steps=n_steps,
        lr_torsion=lr_torsion,
        lr_bond=lr_bond,
        bond_weight=bond_weight,
        closure_weight=closure_weight,
        n_frames=n_frames,
    )

    print(f"    [debug] n_frames={n_frames}  len(trajectory)={len(trajectory)}  trajectory_path={trajectory_path}")
    if n_frames > 0 and trajectory and trajectory_path is not None:
        print(f"    Saving trajectory ({len(trajectory)} frames) → {trajectory_path}")
        _ca_native = CA_native_loop if CA_native_loop is not None else \
                     np.zeros((n_loop, 3))
        try:
            save_trajectory_pdb(
                trajectory,
                full_sequence, loop_start, loop_end,
                N_flank_before, CA_flank_before, C_flank_before, O_flank_before,
                N_flank_after,  CA_flank_after,  C_flank_after,  O_flank_after,
                _ca_native, trajectory_path,
            )
        except Exception as e:
            import traceback
            print(f"    ERROR saving trajectory: {e}")
            traceback.print_exc()

    # Build final structures
    anc_C_t  = torch.tensor(anc_C,  dtype=torch.float32).unsqueeze(0).expand(n_structures, -1)
    clos_N_t = torch.tensor(N_clos, dtype=torch.float32).unsqueeze(0).expand(n_structures, -1)

    with torch.no_grad():
        N_t, CA_t, C_t = build_chain_from_frames(
            N_batch, CA_batch, C_batch,
            psi_opt, phi_opt, dcn_opt, danc_opt, anc_C_t)
        phi_pos, psi_pos = extract_phi_psi_from_positions(
            N_t, CA_t, C_t, anc_C_t, clos_N_t)
        cl_all = (C_t[:, -1] - clos_N_t).norm(dim=-1)
        e_all  = compute_energy(phi_pos, psi_pos, probs_phi, probs_psi)

    print(f"\n    Results ({n_structures} structures):")
    for idx in range(n_structures):
        print(f"      [{idx+1:2d}] closure={cl_all[idx].item():.4f}Å  "
              f"E={e_all[idx].item():.2f}  "
              f"d_anc={danc_opt[idx].item():.3f}Å  "
              f"d_CN mean={dcn_opt[idx].mean().item():.3f}Å" if n_loop > 1 else
              f"      [{idx+1:2d}] closure={cl_all[idx].item():.4f}Å  "
              f"E={e_all[idx].item():.2f}  "
              f"d_anc={danc_opt[idx].item():.3f}Å")

    print(f"\n    After optimization (structure 1):")
    validate_geometry(N_t[0].numpy(), CA_t[0].numpy(), C_t[0].numpy(),
                      label="optimized")

    ensemble = []
    for idx in range(n_structures):
        N_np  = N_t[idx].numpy()
        CA_np = CA_t[idx].numpy()
        C_np  = C_t[idx].numpy()
        O_np  = compute_O_atoms(N_np, CA_np, C_np)

        ensemble.append((
            np.vstack([N_flank_before,  N_np,  N_flank_after]),
            np.vstack([CA_flank_before, CA_np, CA_flank_after]),
            np.vstack([C_flank_before,  C_np,  C_flank_after]),
            np.vstack([O_flank_before,  O_np,  O_flank_after]),
            torch.rad2deg(phi_pos[idx]).numpy(),
            torch.rad2deg(psi_pos[idx]).numpy(),
            float(e_all[idx].item()),
            float(cl_all[idx].item()),
        ))

    return ensemble