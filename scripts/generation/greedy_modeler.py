"""
Greedy residue-by-residue backbone placement.

Places one residue at a time from the N-terminal anchor, sampling many
(phi, psi) candidates at each position and immediately rejecting those
that introduce steric clashes.  A beam search keeps the top *beam_width*
partial structures at each step.

Energy scoring uses the trained TorsionPredictor.  The forward pass is
JIT-compiled and run ONCE per loop position (not once per candidate) --
scoring all candidates is then pure numpy bin-indexing, O(n_candidates).

No loop closure is attempted.

Conventions
-----------
* AA_TO_IDX  -- must match training dataloader exactly (see constant below).
* Bin layout -- 36 bins, 10 deg each, centres at -180 + (k+0.5)*10 deg.
* NeRF sign  -- _nerf_place expects NEGATED dihedrals vs IUPAC.
               All torsions stored here are IUPAC; negation at call sites.
"""
from __future__ import annotations

import sys
import argparse
from pathlib import Path
from typing import List, Optional

import numpy as np
import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nerf import BOND_ANGLES_RAD, BOND_LENGTHS, nerf as _nerf_place
from loop_modeler import _LOOP_ATOM_RADII
from utils import VDW_RADII

_MODEL_SCRIPTS = "/home/jtepperik/thesis/energy_model/scripts"
if _MODEL_SCRIPTS not in sys.path:
    sys.path.insert(0, _MODEL_SCRIPTS)
from models.full_model import TorsionPredictor

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

AA_TO_IDX: dict[str, int] = {
    'A':  0, 'C':  1, 'D':  2, 'E':  3, 'F':  4,
    'G':  5, 'H':  6, 'I':  7, 'K':  8, 'L':  9,
    'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14,
    'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19,
}
PAD_IDX: int = 20

N_BINS: int       = 36
BIN_WIDTH: float  = 360.0 / N_BINS          # 10 deg
BIN_CENTRES       = np.array([-180.0 + (k + 0.5) * BIN_WIDTH for k in range(N_BINS)])

MAX_CONTEXT: int  = 3
CONTEXT_HALF: int = MAX_CONTEXT // 2        # 1

CKPT_PATH: str = (
    "/home/jtepperik/thesis/energy_model/scripts/training/outputs"
    "/energy_loss_c3/checkpoints/best_8"
)

# -----------------------------------------------------------------------------
# Model loading
# -----------------------------------------------------------------------------

def load_model(ckpt_path: str = CKPT_PATH) -> tuple:
    """Load TorsionPredictor checkpoint.  Returns (model, params)."""
    model = TorsionPredictor(
        max_context=MAX_CONTEXT,
        embed_dim=64,
        hidden_dim=768,
        n_layers=3,
        dropout_rate=0.1,
    )
    checkpointer = ocp.StandardCheckpointer()
    restored = checkpointer.restore(ckpt_path)
    params = restored["params"]
    stds = [float(np.std(leaf)) for leaf in jax.tree_util.tree_leaves(params)]
    print(f"  Model loaded (weight std: {min(stds):.4f} - {max(stds):.4f})")
    return model, params


# -----------------------------------------------------------------------------
# Energy tables  (pre-computed once per sequence, before placement starts)
# -----------------------------------------------------------------------------

def _build_context_window(sequence: str, pos: int) -> np.ndarray:
    """(MAX_CONTEXT,) int32 context window centred on pos, padded at edges."""
    encoded = np.array(
        [AA_TO_IDX.get(aa, PAD_IDX) for aa in sequence.upper()],
        dtype=np.int32,
    )
    n = len(encoded)
    window = np.full(MAX_CONTEXT, PAD_IDX, dtype=np.int32)
    for k in range(MAX_CONTEXT):
        src = pos - CONTEXT_HALF + k
        if 0 <= src < n:
            window[k] = encoded[src]
    return window


def build_log_p_tables(
    model,
    params,
    sequence: str,
) -> list[np.ndarray]:
    """
    Run one JIT-compiled forward pass per loop position.

    Returns a list of (N_BINS, N_BINS) float32 arrays -- log P(phi_bin, psi_bin)
    for each residue.  Scoring candidates afterwards is pure numpy indexing.
    """
    @jax.jit
    def _forward(residues, mask):
        return model.apply(
            {"params": params},
            residues,
            mask,
            training=False,
            rngs={"dropout": jax.random.PRNGKey(0)},
        )

    n_loop = len(sequence)
    print(f"  Pre-computing energy tables for {n_loop} positions ...", flush=True)
    tables: list[np.ndarray] = []
    for i in range(n_loop):
        window     = _build_context_window(sequence, i)
        batch_res  = jnp.array(window[None, :], dtype=jnp.int32)
        batch_mask = jnp.ones((1, MAX_CONTEXT), dtype=jnp.bool_)
        logits     = _forward(batch_res, batch_mask)    # (1, N_BINS, N_BINS)
        log_p      = np.array(
            jax.nn.log_softmax(logits[0].ravel())
        ).reshape(N_BINS, N_BINS)
        tables.append(log_p)
    print("  Energy tables ready.", flush=True)
    return tables


def score_candidates(
    log_p_table: np.ndarray,
    phi_deg:     np.ndarray,
    psi_deg:     np.ndarray,
) -> np.ndarray:
    """
    Vectorised: return -log P(phi_k, psi_k) for every candidate k.
    Shape (n,).  Pure numpy -- no JAX overhead.
    """
    phi_bins = (((phi_deg + 180.0) % 360.0) / BIN_WIDTH).astype(int) % N_BINS
    psi_bins = (((psi_deg + 180.0) % 360.0) / BIN_WIDTH).astype(int) % N_BINS
    return -log_p_table[phi_bins, psi_bins]


# -----------------------------------------------------------------------------
# Ramachandran sampling
# -----------------------------------------------------------------------------

_RAMA_REGIONS = [
    (-63.0,  -43.0, 12.0, 0.35),
    (-118.0, 130.0, 12.0, 0.40),
    ( 57.0,   47.0, 12.0, 0.05),
]
_RAMA_WEIGHTS = np.array([r[3] for r in _RAMA_REGIONS], dtype=np.float64)
_RAMA_WEIGHTS /= _RAMA_WEIGHTS.sum()
_UNIFORM_WEIGHT = 0.20


def _sample_torsion_batch(n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Sample n (phi, psi) pairs in radians: biased Ramachandran + uniform."""
    phi = np.empty(n)
    psi = np.empty(n)
    n_uniform = int(round(_UNIFORM_WEIGHT * n))
    n_biased  = n - n_uniform
    phi[:n_uniform] = rng.uniform(-np.pi, np.pi, n_uniform)
    psi[:n_uniform] = rng.uniform(-np.pi, np.pi, n_uniform)
    if n_biased > 0:
        region_idx = rng.choice(len(_RAMA_REGIONS), size=n_biased, p=_RAMA_WEIGHTS)
        for k, ri in enumerate(region_idx):
            mu_phi, mu_psi, std, _ = _RAMA_REGIONS[ri]
            phi[n_uniform + k] = rng.normal(np.radians(mu_phi), np.radians(std))
            psi[n_uniform + k] = rng.normal(np.radians(mu_psi), np.radians(std))
    idx = rng.permutation(n)
    return phi[idx], psi[idx]


# -----------------------------------------------------------------------------
# Residue placement (vectorised NeRF)
# -----------------------------------------------------------------------------

def _batch_place_residues(
    ref_N:   np.ndarray,
    ref_CA:  np.ndarray,
    ref_C:   np.ndarray,
    ref_psi: float,
    phi_arr: np.ndarray,
    psi_arr: np.ndarray,
    omega:   float = np.pi,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Place n candidate residues sharing the same anchor atoms.
    N and CA are identical for all candidates; only C and O vary with phi/psi.
    Returns (N_arr, CA_arr, C_arr, O_arr) each (n, 3).
    NeRF sign convention applied analytically: cos(-x)=cos(x), sin(-x)=-sin(x).
    """
    n = len(phi_arr)
    L = BOND_LENGTHS
    A = BOND_ANGLES_RAD

    N_i  = _nerf_place(ref_N,  ref_CA, ref_C,  L["C_N"],  A["CA_C_N"], -ref_psi)
    CA_i = _nerf_place(ref_CA, ref_C,  N_i,    L["N_CA"], A["C_N_CA"], -omega)

    bc     = CA_i - N_i
    bc_hat = bc / np.linalg.norm(bc)
    n_vec  = ref_C - N_i
    n_vec -= np.dot(n_vec, bc_hat) * bc_hat
    if np.linalg.norm(n_vec) < 1e-8:
        tmp   = np.array([0.0, 0.0, 1.0]) if abs(bc_hat[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
        n_vec = tmp - np.dot(tmp, bc_hat) * bc_hat
    n_hat = n_vec / np.linalg.norm(n_vec)
    m_hat = np.cross(bc_hat, n_hat)

    off_C = -L["CA_C"] * np.cos(A["N_CA_C"])
    amp_C =  L["CA_C"] * np.sin(A["N_CA_C"])
    C_arr = (
        CA_i[None, :]
        + off_C * bc_hat[None, :]
        + amp_C * ( np.cos(phi_arr)[:, None] * n_hat[None, :]
                  - np.sin(phi_arr)[:, None] * m_hat[None, :])
    )

    bc_O     = C_arr - CA_i[None, :]
    bc_O_hat = bc_O / np.maximum(np.linalg.norm(bc_O, axis=1, keepdims=True), 1e-8)
    n_O      = N_i[None, :] - CA_i[None, :]
    dot_n    = (bc_O_hat * n_O).sum(axis=1, keepdims=True)
    n_perp   = n_O - dot_n * bc_O_hat
    n_hat_O  = n_perp / np.maximum(np.linalg.norm(n_perp, axis=1, keepdims=True), 1e-8)
    m_hat_O  = np.cross(bc_O_hat, n_hat_O)

    dihs  = psi_arr + np.pi
    off_O = -L["C_O"] * np.cos(A["CA_C_O"])
    amp_O =  L["C_O"] * np.sin(A["CA_C_O"])
    O_arr = (
        C_arr
        + off_O * bc_O_hat
        + amp_O * ( np.cos(dihs)[:, None] * n_hat_O
                  - np.sin(dihs)[:, None] * m_hat_O)
    )

    N_arr  = np.broadcast_to(N_i,  (n, 3)).copy()
    CA_arr = np.broadcast_to(CA_i, (n, 3)).copy()
    return N_arr, CA_arr, C_arr, O_arr


# -----------------------------------------------------------------------------
# Clash detection
# -----------------------------------------------------------------------------

def _batch_fw_clash_mask(
    N_arr: np.ndarray, CA_arr: np.ndarray, C_arr: np.ndarray, O_arr: np.ndarray,
    fw_coords: np.ndarray, fw_radii: np.ndarray, overlap_tol: float = 0.6,
) -> np.ndarray:
    n = len(C_arr)
    if np.any(np.linalg.norm(N_arr[0][None, :] - fw_coords, axis=1) < VDW_RADII["N"] + fw_radii - overlap_tol):
        return np.ones(n, dtype=bool)
    if np.any(np.linalg.norm(CA_arr[0][None, :] - fw_coords, axis=1) < VDW_RADII["CA"] + fw_radii - overlap_tol):
        return np.ones(n, dtype=bool)
    clash = np.zeros(n, dtype=bool)
    clash |= np.any(np.linalg.norm(C_arr[:, None, :] - fw_coords[None, :, :], axis=2) < VDW_RADII["C"] + fw_radii[None, :] - overlap_tol, axis=1)
    clash |= np.any(np.linalg.norm(O_arr[:, None, :] - fw_coords[None, :, :], axis=2) < VDW_RADII["O"] + fw_radii[None, :] - overlap_tol, axis=1)
    return clash


def _intra_clash(
    N_placed: np.ndarray, CA_placed: np.ndarray,
    C_placed: np.ndarray, O_placed: np.ndarray,
    N_i: np.ndarray, CA_i: np.ndarray, C_i: np.ndarray, O_i: np.ndarray,
    overlap_tol: float = 0.6, min_sep: int = 3,
) -> bool:
    max_j = len(N_placed) - min_sep
    if max_j < 0:
        return False
    new_atoms = np.array([N_i, CA_i, C_i, O_i], dtype=np.float32)
    thresh = _LOOP_ATOM_RADII[:, None] + _LOOP_ATOM_RADII[None, :] - overlap_tol
    for j in range(max_j + 1):
        old_atoms = np.array([N_placed[j], CA_placed[j], C_placed[j], O_placed[j]], dtype=np.float32)
        if bool(np.any(np.linalg.norm(new_atoms[:, None, :] - old_atoms[None, :, :], axis=2) < thresh)):
            return True
    return False


# -----------------------------------------------------------------------------
# Beam state
# -----------------------------------------------------------------------------

class _BeamState:
    __slots__ = (
        "ref_N", "ref_CA", "ref_C", "ref_psi",
        "N_list", "CA_list", "C_list", "O_list",
        "phi_list", "psi_list", "energy",
    )

    def __init__(self, ref_N, ref_CA, ref_C, ref_psi):
        self.ref_N   = ref_N.copy()
        self.ref_CA  = ref_CA.copy()
        self.ref_C   = ref_C.copy()
        self.ref_psi = float(ref_psi)
        self.N_list:   list = []
        self.CA_list:  list = []
        self.C_list:   list = []
        self.O_list:   list = []
        self.phi_list: list = []
        self.psi_list: list = []
        self.energy:   float = 0.0

    def extended(self, N_k, CA_k, C_k, O_k, phi_k, psi_k, step_e):
        s = _BeamState.__new__(_BeamState)
        s.ref_N    = N_k.copy();  s.ref_CA = CA_k.copy()
        s.ref_C    = C_k.copy();  s.ref_psi = float(psi_k)
        s.N_list   = self.N_list   + [N_k.copy()]
        s.CA_list  = self.CA_list  + [CA_k.copy()]
        s.C_list   = self.C_list   + [C_k.copy()]
        s.O_list   = self.O_list   + [O_k.copy()]
        s.phi_list = self.phi_list + [float(phi_k)]
        s.psi_list = self.psi_list + [float(psi_k)]
        s.energy   = self.energy + step_e
        return s

    @property
    def N_arr(self)   -> np.ndarray: return np.array(self.N_list,  np.float64)
    @property
    def CA_arr(self)  -> np.ndarray: return np.array(self.CA_list, np.float64)
    @property
    def C_arr(self)   -> np.ndarray: return np.array(self.C_list,  np.float64)
    @property
    def O_arr(self)   -> np.ndarray: return np.array(self.O_list,  np.float64)
    @property
    def phi_deg(self) -> np.ndarray: return np.degrees(self.phi_list)
    @property
    def psi_deg(self) -> np.ndarray: return np.degrees(self.psi_list)


# -----------------------------------------------------------------------------
# Output
# -----------------------------------------------------------------------------

class EnsembleEntry:
    """8-tuple compatible output: (N, CA, C, O, phi_deg, psi_deg, energy, closure)."""
    __slots__ = ("N", "CA", "C", "O", "phi_deg", "psi_deg", "energy", "closure")

    def __init__(self, N, CA, C, O, phi_deg, psi_deg, energy):
        self.N = N; self.CA = CA; self.C = C; self.O = O
        self.phi_deg = phi_deg; self.psi_deg = psi_deg
        self.energy = energy; self.closure = 0.0

    def as_tuple(self):
        return (self.N, self.CA, self.C, self.O,
                self.phi_deg, self.psi_deg, self.energy, self.closure)


# -----------------------------------------------------------------------------
# Main placement
# -----------------------------------------------------------------------------

def greedy_place_residues(
    sequence:      str,
    prev_N:        np.ndarray,
    prev_CA:       np.ndarray,
    prev_C:        np.ndarray,
    psi_prev:      float,
    fw_coords:     Optional[np.ndarray] = None,
    fw_radii:      Optional[np.ndarray] = None,
    log_p_tables:  Optional[list]       = None,
    n_candidates:  int   = 200,
    beam_width:    int   = 10,
    overlap_tol:   float = 0.6,
    check_intra:   bool  = True,
    min_sep:       int   = 3,
    rng_seed:      Optional[int] = None,
    verbose:       bool  = False,
) -> List[EnsembleEntry]:
    """
    Place len(sequence) residues greedily from the N-terminal anchor.

    log_p_tables: output of build_log_p_tables() -- list of (N_BINS, N_BINS)
                  log-prob arrays, one per loop residue.  Pass None to skip
                  energy scoring (clash filter only).
    """
    n_loop = len(sequence)
    if n_loop == 0:
        return []

    use_fw     = fw_coords is not None and fw_radii is not None
    use_energy = log_p_tables is not None
    rng        = np.random.default_rng(rng_seed)

    beam: list[_BeamState] = [_BeamState(prev_N, prev_CA, prev_C, psi_prev)]
    n_tried = n_clash_fw = n_clash_intra = 0

    for i in range(n_loop):
        if not beam:
            break

        next_candidates: list[tuple[float, _BeamState]] = []

        for state in beam:
            phi_arr, psi_arr = _sample_torsion_batch(n_candidates, rng)
            n_tried += n_candidates

            N_arr, CA_arr, C_arr, O_arr = _batch_place_residues(
                state.ref_N, state.ref_CA, state.ref_C, state.ref_psi,
                phi_arr, psi_arr,
            )

            # Framework clash -- vectorised.
            if use_fw:
                fw_clash = _batch_fw_clash_mask(
                    N_arr, CA_arr, C_arr, O_arr, fw_coords, fw_radii, overlap_tol,
                )
                n_clash_fw += int(fw_clash.sum())
                valid_idx = np.where(~fw_clash)[0]
            else:
                valid_idx = np.arange(n_candidates)

            if len(valid_idx) == 0:
                continue

            # Energy scoring -- one numpy index call for all surviving candidates.
            if use_energy:
                step_energies = score_candidates(
                    log_p_tables[i],
                    np.degrees(phi_arr[valid_idx]),
                    np.degrees(psi_arr[valid_idx]),
                )
            else:
                step_energies = np.zeros(len(valid_idx))

            # Intra-loop clash -- still per-candidate (geometry-dependent).
            N_pl  = state.N_arr  if state.N_list  else np.empty((0, 3))
            CA_pl = state.CA_arr if state.CA_list else np.empty((0, 3))
            C_pl  = state.C_arr  if state.C_list  else np.empty((0, 3))
            O_pl  = state.O_arr  if state.O_list  else np.empty((0, 3))

            for j, k in enumerate(valid_idx):
                if check_intra and _intra_clash(
                    N_pl, CA_pl, C_pl, O_pl,
                    N_arr[k], CA_arr[k], C_arr[k], O_arr[k],
                    overlap_tol, min_sep,
                ):
                    n_clash_intra += 1
                    continue

                new_state = state.extended(
                    N_arr[k], CA_arr[k], C_arr[k], O_arr[k],
                    phi_arr[k], psi_arr[k], float(step_energies[j]),
                )
                next_candidates.append((new_state.energy, new_state))

        if not next_candidates:
            if verbose:
                print(f"    Residue {i+1}/{n_loop}: beam collapsed")
            beam = []
            break

        next_candidates.sort(key=lambda x: x[0])
        beam = [s for _, s in next_candidates[:beam_width]]

        if verbose:
            print(
                f"    Residue {i+1:2d}/{n_loop}: "
                f"{len(next_candidates):4d} valid / "
                f"{len(beam) * n_candidates:4d} sampled  "
                f"best_E={next_candidates[0][0]:.3f}"
            )

    ensemble: list[EnsembleEntry] = []
    for state in beam:
        if len(state.N_list) != n_loop:
            continue
        ensemble.append(EnsembleEntry(
            state.N_arr, state.CA_arr, state.C_arr, state.O_arr,
            state.phi_deg, state.psi_deg, state.energy,
        ))

    print(
        f"    Greedy: {len(ensemble)} structures from {n_tried:,} candidates "
        f"(fw_clash={n_clash_fw:,}  intra_clash={n_clash_intra:,})"
    )
    return ensemble



# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Greedy CDR3 loop placement")
    p.add_argument("--dataset",      required=True,
                   help="Directory containing cdr3_dataset.json")
    p.add_argument("--output",       default="greedy_results")
    p.add_argument("--max-loops",    type=int, default=None)
    p.add_argument("--complex-dir",  default=None,
                   help="Directory of full TCR-pMHC complex PDBs for framework clash")
    p.add_argument("--n-candidates", type=int, default=200)
    p.add_argument("--beam-width",   type=int, default=10)
    p.add_argument("--seed",         type=int, default=42)
    p.add_argument("--no-energy",    action="store_true")
    p.add_argument("--verbose",      action="store_true")
    return p.parse_args()


def _dihedral_4pts(p1, p2, p3, p4) -> float:
    """Dihedral angle in radians defined by four points."""
    b0 = p1 - p2; b1 = p3 - p2; b2 = p4 - p3
    b1h = b1 / (np.linalg.norm(b1) + 1e-10)
    v   = b0 - np.dot(b0, b1h) * b1h
    w   = b2 - np.dot(b2, b1h) * b1h
    return float(np.arctan2(np.dot(np.cross(b1h, v), w), np.dot(v, w)))


def main() -> None:
    import json
    args = _parse_args()

    # ── Imports from sibling modules ─────────────────────────────────────────
    from utils import load_cdr3_native, extract_framework_atoms, save_pdbs, compute_loop_rmsds
    from loop_modeling_nerf import compute_O_atoms

    # ── Model ────────────────────────────────────────────────────────────────
    model = params = None
    if not args.no_energy:
        print("Loading neural energy model ...")
        model, params = load_model()

    # ── Dataset ───────────────────────────────────────────────────────────────
    metadata_file = Path(args.dataset) / "cdr3_dataset.json"
    with open(metadata_file) as f:
        dataset = json.load(f)
    if args.max_loops:
        dataset = dataset[:args.max_loops]

    out_root = Path(args.output)
    out_root.mkdir(parents=True, exist_ok=True)

    results = []

    for idx, meta in enumerate(dataset, 1):
        pdb_id     = meta["pdb_id"]
        chain      = meta["chain"]
        full_seq   = meta["full_sequence"]
        cdr3_seq   = meta["cdr3_sequence"]
        loop_start = meta["loop_start"]
        loop_end   = meta["loop_end"]
        name       = f"{pdb_id}_{chain}"

        print(f"\n{'='*60}")
        print(f"  {idx}/{len(dataset)}  {pdb_id} chain {chain}  CDR3: {cdr3_seq}")

        # Native structure for anchors + RMSD
        pdb_file = Path(meta["pdb_file"])
        if not pdb_file.is_absolute():
            pdb_file = Path(args.dataset) / pdb_file.name
        seq_nat, N_nat, CA_nat, C_nat, O_nat = load_cdr3_native(str(pdb_file))

        # The loop PDB contains only n_flank_before + n_loop + n_flank_after
        # residues. Indices into this array are LOCAL — unrelated to
        # loop_start/loop_end in the JSON (those index into full_sequence).
        n_flank_before = meta.get('n_flank_before', 2)
        n_flank_after  = meta.get('n_flank_after',  2)
        n_loop_res     = loop_end - loop_start

        anc_idx = n_flank_before - 1           # last flank-before residue
        loop_s  = n_flank_before               # first loop residue in PDB array
        loop_e  = n_flank_before + n_loop_res  # one past last loop residue

        anc_N   = N_nat[anc_idx]
        anc_CA  = CA_nat[anc_idx]
        anc_C   = C_nat[anc_idx]
        psi_anc = _dihedral_4pts(
            N_nat[anc_idx], CA_nat[anc_idx],
            C_nat[anc_idx], N_nat[loop_s],
        )
        CA_native_loop = CA_nat[loop_s:loop_e]

        # Framework atoms (optional)
        fw_coords = fw_radii = None
        if args.complex_dir is not None:
            complex_pdb = Path(args.complex_dir) / f"{pdb_id}.pdb"
            if complex_pdb.exists():
                fw_coords, fw_radii = extract_framework_atoms(
                    str(complex_pdb),
                    tcr_chain      = chain,
                    full_sequence  = full_seq,
                    loop_start     = loop_start,
                    loop_end       = loop_end,
                    n_flank_before = meta.get("n_flank_before", 1),
                    n_flank_after  = meta.get("n_flank_after", 1),
                )
            else:
                print(f"    WARNING: complex PDB not found: {complex_pdb}")

        # Energy tables
        log_p_tables = None
        if model is not None:
            log_p_tables = build_log_p_tables(model, params, cdr3_seq)

        # Greedy placement
        ensemble = greedy_place_residues(
            sequence     = cdr3_seq,
            prev_N       = anc_N,
            prev_CA      = anc_CA,
            prev_C       = anc_C,
            psi_prev     = psi_anc,
            fw_coords    = fw_coords,
            fw_radii     = fw_radii,
            log_p_tables = log_p_tables,
            n_candidates = args.n_candidates,
            beam_width   = args.beam_width,
            rng_seed     = args.seed,
            verbose      = args.verbose,
        )

        if not ensemble:
            print(f"    No structures generated.")
            results.append({"pdb_id": pdb_id, "chain": chain, "best_rmsd": None})
            continue

        # Convert EnsembleEntry list to the (N,CA,C,O,phi,psi,energy,closure)
        # 8-tuple format that save_pdbs / compute_loop_rmsds expect.
        # Prepend/append flanks so the full chain is written.
        loop_out = out_root / name
        loop_out.mkdir(exist_ok=True)

        # Use local PDB indices (loop_s, loop_e) throughout.
        # N_nat here is the short loop-PDB array, not the full TCR chain.
        packed = []
        for e in ensemble:
            O_loop = compute_O_atoms(e.N, e.CA, e.C)
            N_full  = np.vstack([N_nat[:loop_s],  e.N,    N_nat[loop_e:]])
            CA_full = np.vstack([CA_nat[:loop_s], e.CA,   CA_nat[loop_e:]])
            C_full  = np.vstack([C_nat[:loop_s],  e.C,    C_nat[loop_e:]])
            O_full  = np.vstack([O_nat[:loop_s],  O_loop, O_nat[loop_e:]])
            packed.append((N_full, CA_full, C_full, O_full,
                           e.phi_deg, e.psi_deg, e.energy, e.closure))

        rmsds = compute_loop_rmsds(packed, CA_native_loop, loop_s, loop_e)
        best_idx_list = [int(np.argmin(rmsds))]

        # full_seq is the full TCR sequence — save_pdbs needs full-chain
        # coords but we only have the loop PDB fragment here, so write
        # using the local fragment sequence and local indices.
        local_seq = seq_nat  # flank + loop + flank from the loop PDB
        save_pdbs(
            ensemble       = packed,
            selected_idx   = best_idx_list,
            full_sequence  = local_seq,
            loop_start     = loop_s,
            loop_end       = loop_e,
            CA_native_loop = CA_native_loop,
            name           = name,
            output_dir     = str(loop_out),
        )

        best_rmsd = float(rmsds.min())
        results.append({"pdb_id": pdb_id, "chain": chain, "best_rmsd": best_rmsd})
        print(f"    RMSD: best={best_rmsd:.3f}A  mean={rmsds.mean():.3f}A")

    # Summary
    valid = [r for r in results if r["best_rmsd"] is not None]
    if valid:
        rmsds_all = [r["best_rmsd"] for r in valid]
        print(f"\n{'='*60}")
        print(f"  Processed {len(valid)}/{len(results)} loops")
        print(f"  RMSD: mean={np.mean(rmsds_all):.3f}A  "
              f"best={min(rmsds_all):.3f}A  worst={max(rmsds_all):.3f}A")


if __name__ == "__main__":
    main()