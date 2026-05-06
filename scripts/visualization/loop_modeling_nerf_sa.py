"""
NeRF-based loop modeling.

Parameterisation
────────────────
  Free variables : phi[i], psi[i]  (one pair per loop residue, in RADIANS)
  Everything else is fixed by ideal backbone geometry:
    - N-CA  = 1.458 Å
    - CA-C  = 1.525 Å
    - C-N   = 1.329 Å  (peptide)
    - N-CA-C angle  = 111.2°
    - CA-C-N angle  = 116.2°
    - C-N-CA angle  = 121.7°
    - omega = 180°   (trans peptide)

  Because each atom is placed from the previous three atoms + one torsion
  via the NeRF recurrence, bond lengths and bond angles CANNOT deviate from
  ideal — there is no gradient path to change them.  The only soft constraint
  needed is C-terminal closure (C_loop[-1] → N_flank_after[0] ≈ 1.329 Å),
  because the N-terminal end is anchored exactly by construction.

Optimization
────────────
  Simulated annealing over torsion angles.
  Each step proposes a random wrapped-Gaussian perturbation to one or all
  angles, evaluates the new energy + closure penalty, and accepts/rejects
  via the Metropolis criterion.

  Temperature schedule: exponential decay T_start → T_end over n_steps.
  Closure weight is annealed identically to the old gradient version so
  early steps explore freely and late steps enforce closure.

Energy
──────
  E = Σ -log p(phi[i] | seq) - log p(psi[i] | seq)
  Angles are converted to degrees for the cached distributions.
"""

import torch
import numpy as np
from typing import Tuple
import jax
import jax.numpy as jnp

from openfold.utils.rigid_utils import Rigid, Rotation


# ─────────────────────────────────────────────────────────────────────────────
# Ideal backbone geometry  (all in Å or radians)
# ─────────────────────────────────────────────────────────────────────────────

BL_CN   = 1.329   # C  → N  peptide bond
BL_NCA  = 1.458   # N  → CA
BL_CAC  = 1.525   # CA → C

BA_CCN  = np.deg2rad(116.2)   # CA-C-N  (angle at C, looking toward next N)
BA_CNC  = np.deg2rad(121.7)   # C-N-CA  (angle at N)
BA_NCC  = np.deg2rad(111.2)   # N-CA-C  (tau, angle at CA)

OMEGA   = np.pi               # trans peptide (180°)


# ─────────────────────────────────────────────────────────────────────────────
# Core NeRF operation
# ─────────────────────────────────────────────────────────────────────────────

_BL_CN  = torch.tensor(BL_CN,  dtype=torch.float32)
_BL_NCA = torch.tensor(BL_NCA, dtype=torch.float32)
_BL_CAC = torch.tensor(BL_CAC, dtype=torch.float32)
_BA_CCN = torch.tensor(BA_CCN, dtype=torch.float32)
_BA_CNC = torch.tensor(BA_CNC, dtype=torch.float32)
_BA_NCC = torch.tensor(BA_NCC, dtype=torch.float32)
_OMEGA  = torch.tensor(OMEGA,  dtype=torch.float32)


@torch.jit.script
def place_atom_b(
    a:          torch.Tensor,
    b:          torch.Tensor,
    c:          torch.Tensor,
    bond_length: torch.Tensor,
    bond_angle:  torch.Tensor,
    torsion:     torch.Tensor,
) -> torch.Tensor:
    """Batched NeRF placement. Places atom d for B structures simultaneously."""
    bc   = c - b
    bc_n = bc / (torch.norm(bc, dim=-1, keepdim=True) + 1e-8)

    n_abc = torch.linalg.cross(b - a, bc)
    n_abc = n_abc / (torch.norm(n_abc, dim=-1, keepdim=True) + 1e-8)

    col2 = torch.linalg.cross(n_abc, bc_n)
    M    = torch.stack([bc_n, col2, n_abc], dim=-1)

    d_local = torch.stack([
        -torch.cos(bond_angle).expand(torsion.shape[0]),
         torch.sin(bond_angle) * torch.cos(torsion),
        -torch.sin(bond_angle) * torch.sin(torsion),
    ], dim=-1) * bond_length

    return c + torch.bmm(M, d_local.unsqueeze(-1)).squeeze(-1)


def build_backbone(
    phi:       torch.Tensor,   # (B, n_loop) radians
    psi:       torch.Tensor,   # (B, n_loop+1) radians
    anchor_N:  np.ndarray,
    anchor_CA: np.ndarray,
    anchor_C:  np.ndarray,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Batched NeRF backbone reconstruction.
    Returns N, CA, C each of shape (B, n_loop, 3).
    """
    B, n = phi.shape
    assert psi.shape == (B, n + 1), f"psi must be ({B}, {n+1}), got {tuple(psi.shape)}"

    a3 = torch.tensor(anchor_N,  dtype=torch.float32).unsqueeze(0).expand(B, -1)
    a2 = torch.tensor(anchor_CA, dtype=torch.float32).unsqueeze(0).expand(B, -1)
    a1 = torch.tensor(anchor_C,  dtype=torch.float32).unsqueeze(0).expand(B, -1)

    N_list, CA_list, C_list = [], [], []

    for i in range(n):
        N_i  = place_atom_b(a3, a2, a1, _BL_CN,  _BA_CCN, psi[:, i])
        CA_i = place_atom_b(a2, a1, N_i, _BL_NCA, _BA_CNC, _OMEGA.expand(B))
        C_i  = place_atom_b(a1, N_i, CA_i, _BL_CAC, _BA_NCC, phi[:, i])
        N_list.append(N_i); CA_list.append(CA_i); C_list.append(C_i)
        a3, a2, a1 = N_i, CA_i, C_i

    return (torch.stack(N_list,  dim=1),
            torch.stack(CA_list, dim=1),
            torch.stack(C_list,  dim=1))


def place_N_after(
    N_last:   torch.Tensor,
    CA_last:  torch.Tensor,
    C_last:   torch.Tensor,
    psi_last: torch.Tensor,
) -> torch.Tensor:
    """Place virtual N_after for all B structures (C-terminal closure)."""
    return place_atom_b(N_last, CA_last, C_last, _BL_CN, _BA_CCN, psi_last)


# ─────────────────────────────────────────────────────────────────────────────
# Energy
# ─────────────────────────────────────────────────────────────────────────────

def cache_energy_distributions(model, params, sequence):
    print("      Caching energy distributions...")
    try:
        from complete_prediction import predict_angles
        logits_phi, logits_psi = predict_angles(model, params, sequence, n_bins=72)
        probs_phi = [jax.nn.softmax(lp) for lp in logits_phi]
        probs_psi = [jax.nn.softmax(lp) for lp in logits_psi]
    except (ImportError, AttributeError):
        print("        Warning: using uniform placeholder")
        n_bins    = 360
        probs_phi = [np.ones(n_bins) / n_bins for _ in range(len(sequence) - 1)]
        probs_psi = [np.ones(n_bins) / n_bins for _ in range(len(sequence) - 1)]
    print(f"      Cached {len(probs_phi)} phi + {len(probs_psi)} psi distributions")
    return probs_phi, probs_psi


def _interp_prob(angle_deg: torch.Tensor, probs: np.ndarray) -> torch.Tensor:
    n_bins     = len(probs)
    angle_norm = torch.fmod(angle_deg + 360.0, 360.0)
    bin_idx    = (angle_norm / 360.0) * n_bins
    idx_lo     = torch.floor(bin_idx).long() % n_bins
    idx_hi     = (idx_lo + 1) % n_bins
    w          = bin_idx - torch.floor(bin_idx)
    pt         = torch.tensor(probs, dtype=torch.float32)
    return (1.0 - w) * pt[idx_lo] + w * pt[idx_hi]


def compute_energy(phi_rad: torch.Tensor, psi_rad: torch.Tensor,
                   probs_phi: list, probs_psi: list) -> torch.Tensor:
    """
    Batched energy: E = Σ -log p(phi[i]) - log p(psi[i]).
    phi_rad : (B, n_loop), psi_rad : (B, n_loop+1)
    Returns : (B,)
    """
    B, n   = phi_rad.shape
    energy = torch.zeros(B, dtype=torch.float32)
    phi_deg = torch.rad2deg(phi_rad)
    psi_deg = torch.rad2deg(psi_rad)

    for i in range(n):
        if i < len(probs_phi):
            energy = energy - torch.log(_interp_prob(phi_deg[:, i], probs_phi[i]) + 1e-10)
        if i < len(probs_psi):
            energy = energy - torch.log(_interp_prob(psi_deg[:, i + 1], probs_psi[i]) + 1e-10)

    return energy


# ─────────────────────────────────────────────────────────────────────────────
# O atom computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_O_atoms(N: np.ndarray, CA: np.ndarray, C: np.ndarray) -> np.ndarray:
    n, O = len(CA), np.zeros((len(CA), 3))
    for i in range(n):
        v_ca = CA[i] - C[i];  v_ca /= (np.linalg.norm(v_ca) + 1e-8)
        if i < n - 1:
            v_n = N[i+1] - C[i];  v_n /= (np.linalg.norm(v_n) + 1e-8)
            bis = v_ca + v_n;  bn = np.linalg.norm(bis)
            # O is opposite to the bisector (trans to N[i+1] across the carbonyl)
            O[i] = C[i] - 1.229 * (bis / bn if bn > 1e-8 else v_ca)
        else:
            O[i] = C[i] - 1.229 * v_ca
    return O


# ─────────────────────────────────────────────────────────────────────────────
# Kabsch alignment
# ─────────────────────────────────────────────────────────────────────────────

def kabsch(P: np.ndarray, Q: np.ndarray):
    P = P - P.mean(axis=0)
    Q = Q - Q.mean(axis=0)
    U, _, Vt = np.linalg.svd(P.T @ Q)
    d = np.linalg.det(Vt.T @ U.T)
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return R, P, Q


def aligned_loop_rmsd(CA_pred_full: np.ndarray,
                      CA_native_full: np.ndarray,
                      loop_start: int, loop_end: int) -> float:
    R, P_c, Q_c = kabsch(CA_pred_full, CA_native_full)
    diff = P_c[loop_start:loop_end] @ R.T - Q_c[loop_start:loop_end]
    return float(np.sqrt(np.mean(np.sum(diff ** 2, axis=1))))


# ─────────────────────────────────────────────────────────────────────────────
# Ensemble analysis
# ─────────────────────────────────────────────────────────────────────────────

def ideal_energy(probs_phi: list, probs_psi: list) -> float:
    energy = 0.0
    for i in range(len(probs_phi)):
        energy -= np.log(np.array(probs_phi[i]).max() + 1e-10)
        energy -= np.log(np.array(probs_psi[i]).max() + 1e-10)
    return float(energy)


def ideal_structure_pdb(
    probs_phi, probs_psi, loop_seq,
    anchor_N, anchor_CA, anchor_C, path,
):
    n_loop  = len(loop_seq)
    N_BINS  = len(probs_phi[0])
    DEG_PER_BIN = 360.0 / N_BINS

    def _argmax_deg(probs):
        k = int(np.argmax(probs))
        deg = k * DEG_PER_BIN
        return deg - 360.0 if deg >= 180.0 else deg

    phi_deg      = np.array([_argmax_deg(probs_phi[i]) for i in range(n_loop)], dtype=np.float32)
    psi_body_deg = np.array([_argmax_deg(probs_psi[i]) for i in range(n_loop - 1)], dtype=np.float32)

    phi_t = torch.tensor(np.deg2rad(phi_deg), dtype=torch.float32).unsqueeze(0)
    psi_t = torch.tensor(
        np.deg2rad(np.concatenate([[-57.0], psi_body_deg, [-57.0]])),
        dtype=torch.float32,
    ).unsqueeze(0)

    with torch.no_grad():
        N_t, CA_t, C_t = build_backbone(phi_t, psi_t, anchor_N, anchor_CA, anchor_C)

    N_np, CA_np, C_np = N_t[0].numpy(), CA_t[0].numpy(), C_t[0].numpy()
    O_np = compute_O_atoms(N_np, CA_np, C_np)

    one_to_three = {
        'A':'ALA','C':'CYS','D':'ASP','E':'GLU','F':'PHE','G':'GLY',
        'H':'HIS','I':'ILE','K':'LYS','L':'LEU','M':'MET','N':'ASN',
        'P':'PRO','Q':'GLN','R':'ARG','S':'SER','T':'THR','V':'VAL',
        'W':'TRP','Y':'TYR',
    }
    with open(path, "w") as f:
        f.write(f"REMARK  Ideal-energy structure for loop: {loop_seq}\n")
        f.write(f"REMARK  Angles = argmax of predicted phi/psi distributions\n")
        f.write(f"REMARK  No flanks — loop residues only\n")
        atom_num = 1
        for i in range(n_loop):
            resname3 = one_to_three.get(loop_seq[i], "UNK")
            for atom_name, coords in [("N", N_np), ("CA", CA_np), ("C", C_np), ("O", O_np)]:
                x, y, z = coords[i]
                f.write(
                    f"ATOM  {atom_num:5d}  {atom_name:<3s} {resname3:3s} A{i+1:4d}    "
                    f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           {atom_name[0]:1s}  \n"
                )
                atom_num += 1
        f.write("END\n")

    print(f"  ✓  Ideal structure → {path}")
    return phi_deg, np.concatenate([[-57.0], psi_body_deg, [-57.0]])


def ensemble_diversity(ensemble, loop_start: int, loop_end: int):
    CA_loops = np.stack([s[1][loop_start:loop_end] for s in ensemble])
    B = len(CA_loops)
    pairwise = np.zeros((B, B))
    for i in range(B):
        for j in range(i + 1, B):
            diff = CA_loops[i] - CA_loops[j]
            rmsd = float(np.sqrt(np.mean(np.sum(diff ** 2, axis=1))))
            pairwise[i, j] = rmsd
            pairwise[j, i] = rmsd
    mean_div = pairwise.sum(axis=1) / (B - 1)
    overall  = float(pairwise.sum() / (B * (B - 1)))
    return pairwise, mean_div, float(overall)


# ─────────────────────────────────────────────────────────────────────────────
# Simulated Annealing optimisation
# ─────────────────────────────────────────────────────────────────────────────

def _score_batch(
    phi:     torch.Tensor,   # (B, n)
    psi:     torch.Tensor,   # (B, n+1)
    anchor_N, anchor_CA, anchor_C,
    N_close_t: torch.Tensor,
    probs_phi, probs_psi,
    closure_weight: float,
) -> torch.Tensor:
    """Evaluate energy + closure penalty for all B structures. Returns (B,)."""
    with torch.no_grad():
        N, CA, C  = build_backbone(phi, psi, anchor_N, anchor_CA, anchor_C)
        N_virtual = place_N_after(N[:, -1], CA[:, -1], C[:, -1], psi[:, -1])
        cl_dist   = torch.norm(N_virtual - N_close_t, dim=-1)          # (B,)
        energy    = compute_energy(phi, psi, probs_phi, probs_psi)      # (B,)
    return energy + closure_weight * cl_dist                            # (B,)


def optimize_torsions_sa(
    phi_init:       torch.Tensor,   # (B, n_loop)
    psi_init:       torch.Tensor,   # (B, n_loop+1)
    anchor_N:       np.ndarray,
    anchor_CA:      np.ndarray,
    anchor_C:       np.ndarray,
    N_closure:      np.ndarray,
    probs_phi:      list,
    probs_psi:      list,
    n_steps:        int   = 5000,
    T_start:        float = 5.0,    # initial temperature (energy units = -log p)
    T_end:          float = 0.05,   # final temperature
    step_size:      float = 0.3,    # std of wrapped-Gaussian perturbation (rad)
    closure_weight: float = 50.0,   # weight on closure distance
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Batched simulated annealing over torsion angles.

    All B structures run independently in parallel (vectorised).
    Each step:
      1. Perturb ALL angles simultaneously with wrapped-Gaussian noise.
      2. Evaluate energy + closure penalty for proposed and current state.
      3. Accept per structure via Metropolis: always if dE < 0, else with
         probability exp(-dE / T).
      4. Track best-ever state per structure.

    Temperature: exponential decay T_start → T_end.
    Closure weight: linearly annealed 0 → closure_weight over n_steps,
                    matching the old gradient version's schedule.

    No gradients are computed — pure numpy/torch without autograd.
    """
    B, n = phi_init.shape

    N_close_t = torch.tensor(N_closure, dtype=torch.float32).unsqueeze(0)  # (1, 3)

    # Current state
    phi_cur = phi_init.clone()
    psi_cur = psi_init.clone()

    # Scores with zero closure weight at start
    score_cur = _score_batch(
        phi_cur, psi_cur,
        anchor_N, anchor_CA, anchor_C,
        N_close_t, probs_phi, probs_psi,
        closure_weight=0.0,
    )

    # Best-ever checkpoint
    best_score = score_cur.clone()
    best_phi   = phi_cur.clone()
    best_psi   = psi_cur.clone()

    # Temperature schedule: exponential decay
    log_T_start = np.log(T_start)
    log_T_end   = np.log(T_end)

    print(f"      SA  B={B}  n={n}  n_steps={n_steps}")
    print(f"      T: {T_start} → {T_end}  step_size={step_size}rad  cl_weight={closure_weight}")

    for step in range(n_steps):
        progress = step / max(n_steps - 1, 1)
        T        = np.exp(log_T_start + progress * (log_T_end - log_T_start))
        cl_w     = closure_weight * progress   # anneal closure weight

        # Propose: perturb all angles with wrapped Gaussian, all structures at once
        phi_prop = phi_cur + torch.randn_like(phi_cur) * step_size
        psi_prop = psi_cur + torch.randn_like(psi_cur) * step_size
        # Wrap to [-pi, pi]
        phi_prop = (phi_prop + np.pi) % (2 * np.pi) - np.pi
        psi_prop = (psi_prop + np.pi) % (2 * np.pi) - np.pi

        score_prop = _score_batch(
            phi_prop, psi_prop,
            anchor_N, anchor_CA, anchor_C,
            N_close_t, probs_phi, probs_psi,
            closure_weight=cl_w,
        )

        # Metropolis acceptance per structure
        dE      = score_prop - score_cur                           # (B,)
        log_acc = (-dE / T).clamp(max=0.0)                        # (B,)
        accept  = torch.log(torch.rand(B)) < log_acc              # (B,) bool

        phi_cur = torch.where(accept.unsqueeze(1), phi_prop, phi_cur)
        psi_cur = torch.where(accept.unsqueeze(1), psi_prop, psi_cur)
        score_cur = torch.where(accept, score_prop, score_cur)

        # Update best checkpoint (use full closure weight for checkpoint scoring)
        full_score = _score_batch(
            phi_cur, psi_cur,
            anchor_N, anchor_CA, anchor_C,
            N_close_t, probs_phi, probs_psi,
            closure_weight=closure_weight,
        )
        improved = full_score < best_score
        best_score = torch.where(improved, full_score, best_score)
        best_phi   = torch.where(improved.unsqueeze(1), phi_cur, best_phi)
        best_psi   = torch.where(improved.unsqueeze(1), psi_cur, best_psi)

        if step % (n_steps // 10) == 0 or step == n_steps - 1:
            with torch.no_grad():
                N, CA, C  = build_backbone(phi_cur, psi_cur, anchor_N, anchor_CA, anchor_C)
                N_virt    = place_N_after(N[:, -1], CA[:, -1], C[:, -1], psi_cur[:, -1])
                cl        = torch.norm(N_virt - N_close_t, dim=-1)
                energy    = compute_energy(phi_cur, psi_cur, probs_phi, probs_psi)
            acc_rate = accept.float().mean().item()
            print(f"        step {step:5d}  T={T:.4f}  "
                  f"E={energy.mean().item():.2f}  "
                  f"closure mean={cl.mean().item():.4f}A  "
                  f"best={cl.min().item():.4f}A  "
                  f"acc={acc_rate:.2f}")

    return best_phi, best_psi


# kept for closure-only baseline refinement (gradient is fine here,
# it is only optimising 2 junction angles with a simple L2 target)
def _optimize_closure_only_batched(
    phi_batch: torch.Tensor,
    psi_batch: torch.Tensor,
    anchor_N:  np.ndarray,
    anchor_CA: np.ndarray,
    anchor_C:  np.ndarray,
    N_closure: np.ndarray,
    n_steps:   int   = 500,
    lr:        float = 0.20,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Batched closure-only gradient descent.
    Body angles frozen; only psi_anc and psi_clos updated.
    """
    B, n      = phi_batch.shape
    N_clos_t  = torch.tensor(N_closure, dtype=torch.float32).unsqueeze(0)

    phi_fixed  = phi_batch.detach()
    psi_body_f = psi_batch[:, 1:n].detach()
    psi_anc    = psi_batch[:, 0:1].clone().requires_grad_(True)
    psi_clos   = psi_batch[:, n:n+1].clone().requires_grad_(True)

    optimizer = torch.optim.Adam([psi_anc, psi_clos], lr=lr)

    best_cl        = torch.full((B,), float("inf"))
    best_psi_anc   = psi_anc.detach().clone()
    best_psi_clos  = psi_clos.detach().clone()

    for _ in range(n_steps):
        optimizer.zero_grad()
        psi_full = torch.cat([psi_anc, psi_body_f, psi_clos], dim=1)
        N, CA, C = build_backbone(phi_fixed, psi_full, anchor_N, anchor_CA, anchor_C)
        N_virt   = place_N_after(N[:, -1], CA[:, -1], C[:, -1], psi_clos[:, 0])
        loss     = torch.sum((N_virt - N_clos_t) ** 2, dim=-1).mean()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            cl = torch.norm(N_virt - N_clos_t, dim=-1)
            improved = cl < best_cl
            best_cl[improved]       = cl[improved]
            best_psi_anc[improved]  = psi_anc.detach()[improved]
            best_psi_clos[improved] = psi_clos.detach()[improved]

    psi_out = torch.cat([best_psi_anc, psi_body_f, best_psi_clos], dim=1)
    return phi_fixed, psi_out


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def random_ensemble(
    full_sequence, loop_start, loop_end,
    N_flank_before, CA_flank_before, C_flank_before, O_flank_before,
    N_flank_after,  CA_flank_after,  C_flank_after,  O_flank_after,
    model, params,
    n_structures:    int   = 10,
    mode:            str   = "uniform",
    refine_closure:  bool  = True,
    closure_steps:   int   = 500,
    closure_lr:      float = 0.20,
    seed:            int   = None,
):
    """
    Random baseline ensemble (uniform or model_sample angles),
    optionally refined with closure-only gradient descent on junction angles.
    """
    assert mode in ("uniform", "model_sample")

    loop_seq = full_sequence[loop_start:loop_end]
    n_loop   = len(loop_seq)

    anc_N  = N_flank_before[-1].copy()
    anc_CA = CA_flank_before[-1].copy()
    anc_C  = C_flank_before[-1].copy()
    N_clos = N_flank_after[0].copy()
    N_clos_t = torch.tensor(N_clos, dtype=torch.float32)

    probs_phi, probs_psi = cache_energy_distributions(model, params, loop_seq)

    N_BINS      = 72
    DEG_PER_BIN = 360.0 / N_BINS

    def _sample_model(probs):
        p = np.array(probs, dtype=np.float64)
        p = np.clip(p, 0, None); p /= p.sum(); p[-1] += 1.0 - p.sum()
        return np.deg2rad(np.random.choice(N_BINS, p=p) * DEG_PER_BIN - 180.0)

    phi_rows, psi_rows = [], []
    for idx in range(n_structures):
        s = None if seed is None else seed + idx
        if s is not None:
            torch.manual_seed(s); np.random.seed(s)

        if mode == "uniform":
            phi_rows.append(torch.FloatTensor(n_loop).uniform_(-np.pi, np.pi))
            psi_rows.append(torch.FloatTensor(n_loop + 1).uniform_(-np.pi, np.pi))
        else:
            phi_rows.append(torch.tensor(
                [_sample_model(probs_phi[i]) for i in range(n_loop)], dtype=torch.float32))
            psi_rows.append(torch.cat([
                torch.FloatTensor(1).uniform_(-np.pi, np.pi),
                torch.tensor([_sample_model(probs_psi[i]) for i in range(n_loop - 1)],
                             dtype=torch.float32),
                torch.FloatTensor(1).uniform_(-np.pi, np.pi),
            ]))

    phi_batch = torch.stack(phi_rows)
    psi_batch = torch.stack(psi_rows)

    if refine_closure:
        phi_batch, psi_batch = _optimize_closure_only_batched(
            phi_batch, psi_batch, anc_N, anc_CA, anc_C, N_clos,
            n_steps=closure_steps, lr=closure_lr,
        )

    ensemble = []
    with torch.no_grad():
        N_t, CA_t, C_t = build_backbone(phi_batch, psi_batch, anc_N, anc_CA, anc_C)
        N_virt_all = place_N_after(N_t[:, -1], CA_t[:, -1], C_t[:, -1], psi_batch[:, -1])
        cl_all     = torch.norm(N_virt_all - N_clos_t, dim=-1)
        energy_all = compute_energy(phi_batch, psi_batch, probs_phi, probs_psi)

    for idx in range(n_structures):
        N_np  = N_t[idx].numpy()
        CA_np = CA_t[idx].numpy()
        C_np  = C_t[idx].numpy()
        O_np  = compute_O_atoms(N_np, CA_np, C_np)
        ensemble.append((
            np.vstack([N_flank_before, N_np, N_flank_after]),
            np.vstack([CA_flank_before, CA_np, CA_flank_after]),
            np.vstack([C_flank_before, C_np, C_flank_after]),
            np.vstack([O_flank_before, O_np, O_flank_after]),
            torch.rad2deg(phi_batch[idx]).numpy(),
            torch.rad2deg(psi_batch[idx, 1:n_loop+1]).numpy(),
            float(energy_all[idx].item()),
            float(cl_all[idx].item()),
        ))

    label = f"{mode}+closure" if refine_closure else mode
    print(f"    Random baseline ({label}): {n_structures} structures built")
    return ensemble, probs_phi, probs_psi


def refine_loop_3d_frames(
    full_sequence, loop_start, loop_end,
    N_flank_before, CA_flank_before, C_flank_before, O_flank_before,
    N_flank_after,  CA_flank_after,  C_flank_after,  O_flank_after,
    model, params,
    n_steps:        int   = 5000,
    T_start:        float = 5.0,
    T_end:          float = 0.15,
    step_size:      float = 0.1,
    closure_weight: float = 50.0,
    n_structures:   int   = 10,
    seed:           int   = None,
    # legacy params kept for call-site compatibility
    lr_energy:      float = 0.05,
    lr_closure:     float = 0.20,
    learning_rate:  float = 0.01,
    cloud_sigma:    float = 5.0,
    bond_weight:    float = 100.0,
):
    """
    NeRF loop modeling via simulated annealing.

    n_structures independent SA runs starting from uniform random torsion
    angles.  Each run uses the same annealing schedule; diversity comes from
    the different random initialisations and stochastic acceptance decisions.
    """
    loop_seq = full_sequence[loop_start:loop_end]
    n_loop   = len(loop_seq)

    print(f"\n  NeRF-SA loop refinement:")
    print(f"    Sequence : {full_sequence}  loop={loop_seq} ({n_loop} res)")
    print(f"    Anchor C → closure N dist: "
          f"{np.linalg.norm(C_flank_before[-1] - N_flank_after[0]):.2f} Å")

    anc_N  = N_flank_before[-1].copy()
    anc_CA = CA_flank_before[-1].copy()
    anc_C  = C_flank_before[-1].copy()
    N_clos = N_flank_after[0].copy()

    probs_phi, probs_psi = cache_energy_distributions(model, params, loop_seq)

    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    phi_init = torch.FloatTensor(n_structures, n_loop).uniform_(-np.pi, np.pi)
    psi_init = torch.FloatTensor(n_structures, n_loop + 1).uniform_(-np.pi, np.pi)

    phi_opt, psi_opt = optimize_torsions_sa(
        phi_init, psi_init,
        anc_N, anc_CA, anc_C, N_clos,
        probs_phi, probs_psi,
        n_steps=n_steps,
        T_start=T_start,
        T_end=T_end,
        step_size=step_size,
        closure_weight=closure_weight,
    )

    with torch.no_grad():
        N_t, CA_t, C_t = build_backbone(phi_opt, psi_opt, anc_N, anc_CA, anc_C)
        N_virtual = place_N_after(N_t[:, -1], CA_t[:, -1], C_t[:, -1], psi_opt[:, -1])

    N_clos_t = torch.tensor(N_clos, dtype=torch.float32)
    cl_dists = torch.norm(N_virtual - N_clos_t, dim=-1)

    print(f"\n    Closure distances (all {n_structures} structures):")
    for idx in range(n_structures):
        print(f"      [{idx+1:2d}] closure={cl_dists[idx].item():.4f}A")

    ensemble = []
    for idx in range(n_structures):
        N_np  = N_t[idx].numpy()
        CA_np = CA_t[idx].numpy()
        C_np  = C_t[idx].numpy()
        O_np  = compute_O_atoms(N_np, CA_np, C_np)

        energy  = compute_energy(
            phi_opt[idx:idx+1], psi_opt[idx:idx+1], probs_phi, probs_psi
        )[0].item()

        ensemble.append((
            np.vstack([N_flank_before, N_np, N_flank_after]),
            np.vstack([CA_flank_before, CA_np, CA_flank_after]),
            np.vstack([C_flank_before, C_np, C_flank_after]),
            np.vstack([O_flank_before, O_np, O_flank_after]),
            torch.rad2deg(phi_opt[idx]).numpy(),
            torch.rad2deg(psi_opt[idx, 1:n_loop+1]).numpy(),
            energy,
            float(cl_dists[idx].item()),
        ))

    return ensemble