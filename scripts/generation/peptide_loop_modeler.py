"""
pMHC peptide backbone modeller — main entry point.

Changes in this version
~~~~~~~~~~~~~~~~~~~~~~~
1. **IMGT / insertion-code robustness** — residue keys are (resnum, icode)
   tuples so IMGT-numbered TCR files and standard RCSB files both work.

2. **Anchor-torsion fix** — ``_random_torsions`` and all optimisation loops
   operate on ``n_internal = n_pep - 2`` torsions only (anchors excluded).

3. **NN energy function** — ``build_nn_energy_fn`` wraps a loaded
   ``ModelRouter`` into an ``EnergyFn`` callable.  For each internal residue
   it runs the JAX forward pass, interpolates ``−log P(φ, ψ)`` from the
   predicted joint distribution, and returns the sum over residues.

4. **Framework extraction adapter** — ``extract_framework_atoms_for_peptide``
   wraps ``utils.extract_framework_atoms`` with the right arguments for the
   pMHC case: the whole peptide chain is excluded, and the framework is
   everything else in the complex.

5. **Chain auto-detection** — ``detect_peptide_chain`` picks the shortest
   chain (almost always the peptide in a pMHC file).

Usage (CLI)
~~~~~~~~~~~
    # With NN energy model:
    python pmhc_peptide_modeler_main.py \\
        --pdb 7rm4_b.pdb --chain C --method both \\
        --checkpoint /path/to/checkpoint_dir \\
        --n_restarts 100 --verbose

    # Clash-only (no NN scorer):
    python pmhc_peptide_modeler_main.py \\
        --pdb 7rm4_b.pdb --method gd --n_restarts 50
"""
from __future__ import annotations

import argparse
import math
import sys
import warnings
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

# ── project imports ───────────────────────────────────────────────────────────
from nerf import build_loop
from loop_modeler import (
    ccd_closure,
    score_clashes,
    _intra_loop_clash,
    _has_clash,
)
from utils import (
    ONE_TO_THREE,
    THREE_TO_ONE,
    VDW_RADII,
    extract_framework_atoms,
    write_pdb_atoms,
    load_model,
    ModelRouter,
)

# ─────────────────────────────────────────────────────────────────────────────
# Type aliases
# ─────────────────────────────────────────────────────────────────────────────

EnergyFn      = Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], float]
EnsembleEntry = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]


# ─────────────────────────────────────────────────────────────────────────────
# 0.  Amino-acid index table
#     Must match the training dataloader ordering exactly.
#     VERIFY: compare against your dataloader's AA_TO_IDX if scores look wrong.
# ─────────────────────────────────────────────────────────────────────────────

AA_TO_IDX: Dict[str, int] = {
    'A':  0, 'C':  1, 'D':  2, 'E':  3, 'F':  4,
    'G':  5, 'H':  6, 'I':  7, 'K':  8, 'L':  9,
    'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14,
    'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19,
}

# Bin centres for the 36-bin (10° each) torsion distribution.
# Training convention: bin k centred at −180 + (k + 0.5) × 10°.
_N_BINS      = 36
_BIN_WIDTH   = 360.0 / _N_BINS                        # 10°
_BIN_CENTRES = np.array(
    [-180.0 + (k + 0.5) * _BIN_WIDTH for k in range(_N_BINS)]
)


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Neural-network energy function
# ─────────────────────────────────────────────────────────────────────────────

def _angle_to_bin(angle_deg: float) -> Tuple[int, float]:
    """Map angle (degrees) to (bin_index, fractional_offset) for bilinear interp."""
    f    = ((angle_deg + 180.0) % 360.0) / _BIN_WIDTH
    idx  = int(f) % _N_BINS
    frac = f - int(f)
    return idx, frac


def _interp_2d(phi_deg: float, psi_deg: float, joint_prob: np.ndarray) -> float:
    """
    Bilinear interpolation on the (N_BINS × N_BINS) joint probability table.

    Returns a probability clamped to [1e-10, 1] to avoid log(0).
    """
    pi, pw = _angle_to_bin(phi_deg)
    si, sw = _angle_to_bin(psi_deg)
    ph = (pi + 1) % _N_BINS
    sh = (si + 1) % _N_BINS
    val = (
        (1 - pw) * (1 - sw) * joint_prob[pi, si] +
        (1 - pw) *       sw  * joint_prob[pi, sh] +
              pw  * (1 - sw) * joint_prob[ph, si] +
              pw  *       sw  * joint_prob[ph, sh]
    )
    return float(np.clip(val, 1e-10, 1.0))


def build_nn_energy_fn(
    router:    ModelRouter,
    sequence:  str,
    prev_N:    np.ndarray,
    prev_CA:   np.ndarray,
    prev_C:    np.ndarray,
    target_N:  np.ndarray,
    target_CA: np.ndarray,
    target_C:  np.ndarray,
    use_jit:   bool = True,
) -> EnergyFn:
    """
    Build an EnergyFn backed by the loaded TorsionPredictor.

    Energy = Σ_i  −log P(φ_i, ψ_i | sequence)
    summed over internal residues only (indices 1 … n−2).

    The forward pass is called once per energy evaluation with the full
    internal sequence as context.

    Args:
        router:   ModelRouter from utils.load_model().
        sequence: Full peptide sequence including anchor residues.
        use_jit:  Apply jax.jit to the forward pass.

    Notes on the forward pass signature
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Assumed interface (VERIFY against your TorsionPredictor):

        logits = model.apply({'params': params}, aa_idx, training=False)
        # aa_idx : (n_internal,) int32 JAX array
        # logits : (n_internal, N_BINS, N_BINS) float32

    If your model returns log-probabilities instead of raw logits, set
    ``model_outputs_logprobs=True`` (add this kwarg or edit the softmax
    block below).
    """
    import jax
    import jax.numpy as jnp

    internal_seq = sequence[1:-1]
    n_internal   = len(internal_seq)

    # ── Build the full sequence index array (including anchors) ─────────────
    # The model uses a sliding context window of size max_context centred on
    # each target residue.  We index into the *full* peptide (n_pep residues)
    # so that edge residues get proper left/right context from the anchors.
    PAD_IDX   = 20   # model convention: index 20 = PAD token
    max_ctx   = router.general_model.max_context   # e.g. 3
    full_seq  = sequence                            # n_pep residues including anchors

    full_idx: List[int] = []
    warned_unknown: set = set()
    for aa in full_seq:
        idx = AA_TO_IDX.get(aa)
        if idx is None:
            if aa not in warned_unknown:
                warnings.warn(
                    f"build_nn_energy_fn: unknown AA '{aa}' mapped to 0 (ALA). "
                    "Check AA_TO_IDX ordering.",
                    stacklevel=2,
                )
                warned_unknown.add(aa)
            idx = 0
        full_idx.append(idx)

    # Pad the sequence on both sides so every residue can be centred in a
    # window of width max_context:  pad = max_context // 2
    half = max_ctx // 2
    padded_idx = [PAD_IDX] * half + full_idx + [PAD_IDX] * half
    # padded_idx[i : i+max_ctx] is the context window centred on full_seq[i]

    # Pre-build batched input for all n_internal internal residues at once.
    # Internal residues in full_seq are at indices 1 … n_pep-2.
    # In padded_idx they are at positions  1+half … n_pep-2+half.
    residues_batch = []   # (n_internal, max_ctx) int
    mask_batch     = []   # (n_internal, max_ctx) bool  — True = real token
    for j in range(n_internal):
        full_pos = j + 1          # position in full_seq (0-indexed)
        pad_pos  = full_pos       # position in padded_idx (half already added)
        window   = padded_idx[pad_pos : pad_pos + max_ctx]
        mask_row = [tok != PAD_IDX for tok in window]
        residues_batch.append(window)
        mask_batch.append(mask_row)

    # Shape: (n_internal, max_ctx)
    residues_jnp = jnp.array(residues_batch, dtype=jnp.int8)
    mask_jnp     = jnp.array(mask_batch,     dtype=bool)

    model  = router.general_model
    params = router.general_params

    # Single batched forward pass for all internal residues.
    # Returns logits: (n_internal, n_bins * n_bins)
    def _forward(res, msk):
        return model.apply({'params': params}, res, msk, training=False)

    if use_jit:
        _forward = jax.jit(_forward)

    def energy_fn(
        N:  np.ndarray,
        CA: np.ndarray,
        C:  np.ndarray,
        O:  np.ndarray,
    ) -> float:
        from nerf import get_torsion

        # Stack anchor atoms so we can measure torsions at the boundaries.
        # N_full shape: (n_pep, 3)  —  index 0 = N-anchor, n_pep-1 = C-anchor
        N_full  = np.vstack([prev_N[None],  N,  target_N[None]])
        CA_full = np.vstack([prev_CA[None], CA, target_CA[None]])
        C_full  = np.vstack([prev_C[None],  C,  target_C[None]])

        # Measure φ/ψ for internal residues (full_seq indices 1…n_pep-2).
        phi_deg = np.empty(n_internal)
        psi_deg = np.empty(n_internal)
        for j in range(n_internal):
            i = j + 1
            phi_deg[j] = math.degrees(
                get_torsion(C_full[i-1], N_full[i], CA_full[i], C_full[i])
            )
            psi_deg[j] = math.degrees(
                get_torsion(N_full[i], CA_full[i], C_full[i], N_full[i+1])
            )

        # Batched forward pass → (n_internal, n_bins*n_bins) logits
        logits_flat = np.array(_forward(residues_jnp, mask_jnp))

        # Numerically stable softmax → probability tables (n_internal, n_bins, n_bins)
        logits_flat = logits_flat - logits_flat.max(axis=1, keepdims=True)
        exp_l       = np.exp(logits_flat)
        probs       = (exp_l / exp_l.sum(axis=1, keepdims=True)
                       ).reshape(n_internal, _N_BINS, _N_BINS)

        # Sum −log P(φ_i, ψ_i) over internal residues
        total = 0.0
        for j in range(n_internal):
            p      = _interp_2d(phi_deg[j], psi_deg[j], probs[j])
            total += -math.log(p)
        return total

    return energy_fn


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Framework extraction adapter
# ─────────────────────────────────────────────────────────────────────────────

def extract_framework_atoms_for_peptide(
    pdb_file:      str | Path,
    peptide_chain: str,
    peptide_seq:   str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract framework atoms from a pMHC PDB, excluding the full peptide chain.

    Adapts utils.extract_framework_atoms (designed for CDR3 loops with flank
    offsets) to the pMHC case by passing loop_start=0, loop_end=len(seq),
    n_flank_before=0, n_flank_after=0 — so every peptide residue is excluded
    and the framework is all remaining chains (MHC α/β, TCR α/β, etc.).

    Args:
        pdb_file:      Path to the pMHC PDB.
        peptide_chain: Chain ID of the peptide (e.g. 'C').
        peptide_seq:   Full one-letter peptide sequence.

    Returns:
        coords : (N_fw, 3) float32
        radii  : (N_fw,)   float32
    """
    n = len(peptide_seq)
    return extract_framework_atoms(
        complex_pdb    = str(pdb_file),
        tcr_chain      = peptide_chain,
        full_sequence  = peptide_seq,
        loop_start     = 0,
        loop_end       = n,
        n_flank_before = 0,
        n_flank_after  = 0,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 3.  IMGT / RCSB–compatible PDB parsing
# ─────────────────────────────────────────────────────────────────────────────

def _parse_reskey(line: str) -> Tuple[int, str]:
    """
    Return (resnum, icode) from a PDB ATOM line.

    IMGT files: resnum=112, icode='A'/'B'/...  (insertion codes in col 26).
    RCSB files: icode=' ' → stripped to ''.
    Tuples sort correctly: (112,'') < (112,'A') < (112,'B').
    """
    resnum = int(line[22:26])
    icode  = line[26].strip()
    return (resnum, icode)


def extract_peptide_sequence(
    pdb_file: str | Path,
    chain:    str,
) -> Tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract backbone atoms for *chain*, handling IMGT insertion codes.

    Returns:
        sequence, N, CA, C, O  (all (n_res, 3) float64)
    """
    residues: Dict[Tuple[int, str], dict] = {}
    with open(pdb_file) as fh:
        for line in fh:
            if not line.startswith('ATOM') or len(line) < 54:
                continue
            if line[21] != chain or line[16] not in (' ', 'A'):
                continue
            atom    = line[12:16].strip()
            resname = line[17:20].strip()
            reskey  = _parse_reskey(line)
            xyz     = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
            residues.setdefault(reskey, {'resname': resname})
            if atom in ('N', 'CA', 'C', 'O'):
                residues[reskey][atom] = np.array(xyz, dtype=np.float64)

    if not residues:
        raise ValueError(
            f"extract_peptide_sequence: chain '{chain}' not found in {pdb_file}"
        )

    seq_chars, N_l, CA_l, C_l, O_l = [], [], [], [], []
    for rk in sorted(residues):
        rd    = residues[rk]
        label = f"{rk[0]}{rk[1]}"
        seq_chars.append(THREE_TO_ONE.get(rd['resname'], 'X'))
        for atom in ('N', 'CA', 'C', 'O'):
            if atom not in rd:
                warnings.warn(
                    f"Residue {label} chain '{chain}' missing '{atom}'; "
                    "using zero coordinates.", stacklevel=2
                )
        N_l .append(rd.get('N',  np.zeros(3)))
        CA_l.append(rd.get('CA', np.zeros(3)))
        C_l .append(rd.get('C',  np.zeros(3)))
        O_l .append(rd.get('O',  np.zeros(3)))

    return (
        ''.join(seq_chars),
        np.array(N_l,  dtype=np.float64),
        np.array(CA_l, dtype=np.float64),
        np.array(C_l,  dtype=np.float64),
        np.array(O_l,  dtype=np.float64),
    )


def detect_peptide_chain(pdb_file: str | Path) -> str:
    """Return the chain ID of the shortest chain (almost always the peptide)."""
    chain_res: Dict[str, set] = {}
    with open(pdb_file) as fh:
        for line in fh:
            if not line.startswith('ATOM') or len(line) < 27:
                continue
            chain_res.setdefault(line[21], set()).add(_parse_reskey(line))
    if not chain_res:
        raise ValueError(f"detect_peptide_chain: no ATOM records in {pdb_file}")
    lengths    = {ch: len(rks) for ch, rks in chain_res.items()}
    min_len    = min(lengths.values())
    candidates = sorted(ch for ch, l in lengths.items() if l == min_len)
    print(f"[detect_peptide_chain] chain lengths: "
          + ", ".join(f"{c}:{l}" for c, l in sorted(lengths.items())))
    print(f"[detect_peptide_chain] selected '{candidates[0]}' ({min_len} residues)")
    return candidates[0]


def extract_pmhc_anchors(
    pdb_file:      str | Path,
    peptide_chain: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float,
           np.ndarray, np.ndarray, np.ndarray]:
    """N- and C-terminal anchor geometry from the crystal structure."""
    from nerf import get_torsion
    seq, N, CA, C, O = extract_peptide_sequence(pdb_file, peptide_chain)
    if len(seq) < 2:
        raise ValueError(f"Peptide has only {len(seq)} residue(s); need ≥ 2.")
    psi_prev = get_torsion(N[0], CA[0], C[0], N[1])
    return N[0], CA[0], C[0], psi_prev, N[-1], CA[-1], C[-1]


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Combined scoring
# ─────────────────────────────────────────────────────────────────────────────

def _softplus_clash_score(
    N:           np.ndarray,
    CA:          np.ndarray,
    C:           np.ndarray,
    O:           np.ndarray,
    fw_coords:   np.ndarray,
    fw_radii:    np.ndarray,
    softness:    float = 0.8,
    k:           float = 5.0,
) -> float:
    """
    Continuous softplus clash penalty between loop and framework atoms.

    For each loop–framework atom pair with distance r and sum-of-radii d_min:

        penalty_ij = k · log(1 + exp(d_min − r))

    This is always positive and grows linearly for large overlaps, giving a
    smooth gradient signal even for near-miss overlaps where the hard
    count (score_clashes) returns 0.  This is the same formulation used in
    the PyTorch optimisation pipeline.

    Args:
        N, CA, C, O:  (n_pep, 3) loop backbone atoms.
        fw_coords:    (N_fw, 3) float32 framework atom positions.
        fw_radii:     (N_fw,)   float32 framework vdW radii.
        softness:     Scaling factor applied to vdW radii (default 0.8,
                      matching the RCD convention used elsewhere).
        k:            Sharpness of the softplus (default 5.0).

    Returns:
        float — total softplus penalty (0 when no atoms are close).
    """
    # Stack all loop backbone atoms: shape (4*n_pep, 3)
    loop_atoms  = np.concatenate([N, CA, C, O], axis=0).astype(np.float32)

    # vdW radii for N, CA, C, O repeated across residues
    n_pep = len(N)
    loop_radii = np.tile(
        np.array([VDW_RADII['N'], VDW_RADII['CA'],
                  VDW_RADII['C'], VDW_RADII['O']], dtype=np.float32),
        n_pep,
    )                                                   # (4*n_pep,)

    total = 0.0
    fw_coords_f = fw_coords.astype(np.float32)

    for i, (la, lr) in enumerate(zip(loop_atoms, loop_radii)):
        # Vectorised distance to all framework atoms
        diff = fw_coords_f - la                        # (N_fw, 3)
        dist = np.sqrt((diff * diff).sum(axis=1))      # (N_fw,)

        # Threshold: sum of scaled radii
        d_min = softness * (lr + fw_radii)             # (N_fw,)

        # Softplus: k * log(1 + exp(d_min - dist))
        # Use numerically stable form: max(x,0) + log(1 + exp(-|x|))
        x        = d_min - dist
        penalty  = k * (np.maximum(x, 0.0) + np.log1p(np.exp(-np.abs(x))))
        total   += float(penalty.sum())

    return total


def _softplus_intra_clash(
    N:        np.ndarray,
    CA:       np.ndarray,
    C:        np.ndarray,
    O:        np.ndarray,
    softness: float = 0,
    k:        float = 5.0,
    min_seq_sep: int = 3,
) -> float:
    """
    Continuous softplus intra-loop clash penalty.

    Only residue pairs separated by ≥ min_seq_sep positions are checked
    (i.e. bonded neighbours are excluded).

    Returns:
        float — total softplus penalty (0 for clash-free conformations).
    """
    n_pep      = len(N)
    all_atoms  = np.concatenate([N, CA, C, O], axis=0).astype(np.float32)
    # residue index for each atom: N_0…N_{n-1}, CA_0…CA_{n-1}, …
    res_idx    = np.tile(np.arange(n_pep), 4)
    atom_radii = np.concatenate([
        np.full(n_pep, VDW_RADII['N'],  dtype=np.float32),
        np.full(n_pep, VDW_RADII['CA'], dtype=np.float32),
        np.full(n_pep, VDW_RADII['C'],  dtype=np.float32),
        np.full(n_pep, VDW_RADII['O'],  dtype=np.float32),
    ])

    n_atoms = len(all_atoms)
    total   = 0.0
    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            if abs(int(res_idx[i]) - int(res_idx[j])) < min_seq_sep:
                continue
            diff  = all_atoms[i] - all_atoms[j]
            dist  = float(np.sqrt((diff * diff).sum()))
            d_min = softness * (atom_radii[i] + atom_radii[j])
            x     = d_min - dist
            total += k * (max(x, 0.0) + math.log1p(math.exp(-abs(x))))
    return total


def score_peptide(
    N: np.ndarray, CA: np.ndarray, C: np.ndarray, O: np.ndarray,
    fw_coords:      Optional[np.ndarray] = None,
    fw_radii:       Optional[np.ndarray] = None,
    overlap_tol:    float = 0.6,    # kept for _has_clash / hard filter
    energy_fn:      Optional[EnergyFn] = None,
    energy_weight:  float = 1.0,
    clash_weight:   float = 1.0,
    softness:       float = 0.8,
    clash_k:        float = 5.0,
    score_intra:    bool  = True,
) -> float:
    """
    Combined energy: softplus clash penalty + optional NN energy.

    Uses a continuous softplus clash penalty instead of the hard integer
    count from score_clashes.  This provides a nonzero, smooth objective
    for the coordinate-descent / SA optimisers even when no atoms are
    within the hard overlap threshold.

    The hard _has_clash / _intra_loop_clash filters are still used as
    *acceptance filters* in the outer restart loops; this function is only
    the per-step optimisation objective.
    """
    clash = 0.0
    if fw_coords is not None and fw_radii is not None:
        clash += _softplus_clash_score(
            N, CA, C, O, fw_coords, fw_radii, softness, clash_k
        )
    if score_intra:
        clash += _softplus_intra_clash(N, CA, C, O, softness, clash_k)

    nn_e = float(energy_fn(N, CA, C, O)) if energy_fn is not None else 0.0
    return clash_weight * clash + energy_weight * nn_e


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Ramachandran sampling — internal residues only
# ─────────────────────────────────────────────────────────────────────────────

_RAMA_REGIONS = [
    (-60.0,  -45.0, 15.0, 0.35),   # α-helix
    (-120.0, 130.0, 15.0, 0.40),   # β-strand
    ( 60.0,   45.0, 15.0, 0.05),   # left-handed α
    (  0.0,    0.0, 180.0, 0.20),  # broad / PPII
]
_RAMA_W  = np.array([r[3] for r in _RAMA_REGIONS])
_RAMA_W /= _RAMA_W.sum()


def _random_torsions(n_internal: int, rng: np.random.Generator):
    """Sample φ/ψ for n_internal residues (anchor residues excluded)."""
    phi = np.empty(n_internal)
    psi = np.empty(n_internal)
    ri  = rng.choice(len(_RAMA_REGIONS), size=n_internal, p=_RAMA_W)
    for i, r in enumerate(ri):
        mu_phi, mu_psi, std, _ = _RAMA_REGIONS[r]
        phi[i] = rng.normal(np.radians(mu_phi), np.radians(std))
        psi[i] = rng.normal(np.radians(mu_psi), np.radians(std))
    return phi, psi


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Gradient-free coordinate descent
# ─────────────────────────────────────────────────────────────────────────────

def gradient_descent_peptide(
    sequence:      str,
    prev_N:        np.ndarray,
    prev_CA:       np.ndarray,
    prev_C:        np.ndarray,
    psi_prev:      float,
    target_N:      np.ndarray,
    target_CA:     np.ndarray,
    target_C:      np.ndarray,
    fw_coords:     Optional[np.ndarray] = None,
    fw_radii:      Optional[np.ndarray] = None,
    energy_fn:     Optional[EnergyFn] = None,
    energy_weight: float = 1.0,
    clash_weight:  float = 1.0,
    overlap_tol:   float = 0.6,
    softness:      float = 0.8,
    clash_k:       float = 5.0,
    closure_tol:   float = 0.5,
    ccd_iter:      int   = 500,
    ccd_tol:       float = 0.05,
    n_restarts:    int   = 50,
    n_inner:       int   = 20,
    step_size:     float = 0.1,
    step_decay:    float = 0.95,
    check_intra:   bool  = True,
    rng_seed:      Optional[int] = None,
    verbose:       bool  = False,
) -> List[EnsembleEntry]:
    n_pep      = len(sequence)
    n_internal = n_pep - 2
    if n_internal < 1:
        raise ValueError(f"Peptide needs ≥ 3 residues (got {n_pep}).")

    rng    = np.random.default_rng(rng_seed)
    use_fw = fw_coords is not None and fw_radii is not None

    def _energy(phi, psi):
        N_, CA_, C_, O_ = build_loop(prev_N, prev_CA, prev_C, psi_prev, phi, psi)
        return score_peptide(N_, CA_, C_, O_,
                             fw_coords, fw_radii, overlap_tol,
                             energy_fn, energy_weight, clash_weight,
                             softness=softness, clash_k=clash_k)

    ensemble: List[EnsembleEntry] = []

    for restart in range(n_restarts):
        phi, psi = _random_torsions(n_internal, rng)
        phi, psi, closure = ccd_closure(
            prev_N, prev_CA, prev_C, psi_prev,
            phi, psi, target_N, target_CA,
            n_iter=ccd_iter, tol=ccd_tol,
        )
        if closure > closure_tol:
            if verbose:
                print(f"  [GD {restart+1:3d}]  open (closure={closure:.3f} Å)")
            continue

        step = step_size
        for _pass in range(n_inner):
            improved = False
            for i in range(n_internal):
                for angle_arr in (phi, psi):
                    cur_val = float(angle_arr[i])
                    cur_E   = _energy(phi, psi)
                    best_val, best_E = cur_val, cur_E
                    for delta in (+step, -step):
                        angle_arr[i] = cur_val + delta
                        e = _energy(phi, psi)
                        if e < best_E:
                            best_E, best_val = e, cur_val + delta
                    angle_arr[i] = best_val
                    if best_val != cur_val:
                        improved = True
            step *= step_decay
            if not improved:
                break

        phi, psi, closure = ccd_closure(
            prev_N, prev_CA, prev_C, psi_prev,
            phi, psi, target_N, target_CA,
            n_iter=ccd_iter, tol=ccd_tol,
        )
        if closure > closure_tol:
            continue

        N, CA, C, O = build_loop(prev_N, prev_CA, prev_C, psi_prev, phi, psi)
        if check_intra and _intra_loop_clash(N, CA, C, O, overlap_tol):
            continue
        if use_fw and _has_clash(N, CA, C, O, fw_coords, fw_radii, overlap_tol):
            continue

        energy = score_peptide(N, CA, C, O,
                               fw_coords, fw_radii, overlap_tol,
                               energy_fn, energy_weight, clash_weight)
        ensemble.append((N, CA, C, O, energy, closure))
        if verbose:
            print(f"  [GD {restart+1:3d}]  closure={closure:.3f} Å  energy={energy:.4f}")

    if verbose:
        print(f"  GD done: {len(ensemble)}/{n_restarts} accepted")
    return ensemble


# ─────────────────────────────────────────────────────────────────────────────
# 7.  Simulated annealing
# ─────────────────────────────────────────────────────────────────────────────

def simulated_annealing_peptide(
    sequence:      str,
    prev_N:        np.ndarray,
    prev_CA:       np.ndarray,
    prev_C:        np.ndarray,
    psi_prev:      float,
    target_N:      np.ndarray,
    target_CA:     np.ndarray,
    target_C:      np.ndarray,
    fw_coords:     Optional[np.ndarray] = None,
    fw_radii:      Optional[np.ndarray] = None,
    energy_fn:     Optional[EnergyFn] = None,
    energy_weight: float = 1.0,
    clash_weight:  float = 1.0,
    overlap_tol:   float = 0.6,
    softness:      float = 0.8,
    clash_k:       float = 5.0,
    closure_tol:   float = 0.5,
    ccd_iter:      int   = 200,
    ccd_tol:       float = 0.05,
    n_restarts:    int   = 20,
    n_steps:       int   = 2000,
    T_init:        float = 5.0,
    T_final:       float = 0.1,
    step_size:     float = 0.3,
    check_intra:   bool  = True,
    rng_seed:      Optional[int] = None,
    verbose:       bool  = False,
) -> List[EnsembleEntry]:
    n_pep      = len(sequence)
    n_internal = n_pep - 2
    if n_internal < 1:
        raise ValueError(f"Peptide needs ≥ 3 residues (got {n_pep}).")

    rng    = np.random.default_rng(rng_seed)
    use_fw = fw_coords is not None and fw_radii is not None

    def _score(phi, psi):
        N_, CA_, C_, O_ = build_loop(prev_N, prev_CA, prev_C, psi_prev, phi, psi)
        return score_peptide(N_, CA_, C_, O_,
                             fw_coords, fw_radii, overlap_tol,
                             energy_fn, energy_weight, clash_weight,
                             softness=softness, clash_k=clash_k)

    ensemble: List[EnsembleEntry] = []
    T_range = T_init - T_final

    for restart in range(n_restarts):
        phi, psi = _random_torsions(n_internal, rng)
        phi, psi, closure = ccd_closure(
            prev_N, prev_CA, prev_C, psi_prev,
            phi, psi, target_N, target_CA,
            n_iter=ccd_iter, tol=ccd_tol,
        )
        if closure > closure_tol:
            if verbose:
                print(f"  [SA {restart+1:3d}]  initial open ({closure:.3f} Å)")
            continue

        cur_E    = _score(phi, psi)
        best_phi = phi.copy(); best_psi = psi.copy()
        best_E   = cur_E;      best_cl  = closure
        n_accept = 0

        for step in range(n_steps):
            T = T_init - T_range * (step / max(n_steps - 1, 1))
            prop_phi = phi.copy()
            prop_psi = psi.copy()
            res_idx  = rng.integers(0, n_internal)
            delta    = rng.normal(0.0, step_size)
            if rng.integers(0, 2) == 0:
                prop_phi[res_idx] += delta
            else:
                prop_psi[res_idx] += delta

            prop_phi, prop_psi, prop_cl = ccd_closure(
                prev_N, prev_CA, prev_C, psi_prev,
                prop_phi, prop_psi, target_N, target_CA,
                n_iter=ccd_iter, tol=ccd_tol,
            )
            if prop_cl > closure_tol:
                continue

            prop_E  = _score(prop_phi, prop_psi)
            delta_E = prop_E - cur_E
            if delta_E <= 0.0 or rng.random() < math.exp(-delta_E / max(T, 1e-12)):
                phi, psi, closure = prop_phi, prop_psi, prop_cl
                cur_E = prop_E; n_accept += 1
                if cur_E < best_E:
                    best_phi = phi.copy(); best_psi = psi.copy()
                    best_E = cur_E;        best_cl  = closure

        N, CA, C, O = build_loop(prev_N, prev_CA, prev_C, psi_prev, best_phi, best_psi)
        if check_intra and _intra_loop_clash(N, CA, C, O, overlap_tol):
            if verbose:
                print(f"  [SA {restart+1:3d}]  intra-clash in best")
            continue
        if use_fw and _has_clash(N, CA, C, O, fw_coords, fw_radii, overlap_tol):
            if verbose:
                print(f"  [SA {restart+1:3d}]  fw-clash in best")
            continue

        ensemble.append((N, CA, C, O, best_E, best_cl))
        if verbose:
            print(f"  [SA {restart+1:3d}]  "
                  f"E={best_E:.4f}  closure={best_cl:.3f} Å  "
                  f"accept={n_accept/max(n_steps,1):.2%}")

    if verbose:
        print(f"  SA done: {len(ensemble)}/{n_restarts} accepted")
    return ensemble


# ─────────────────────────────────────────────────────────────────────────────
# 8.  Output
# ─────────────────────────────────────────────────────────────────────────────

def save_peptide_pdbs(
    ensemble:   List[EnsembleEntry],
    sequence:   str,
    name:       str,
    output_dir: str | Path,
    native_CA:  Optional[np.ndarray] = None,
    native_N:   Optional[np.ndarray] = None,
    native_C:   Optional[np.ndarray] = None,
    native_O:   Optional[np.ndarray] = None,
    verbose:    bool = True,
) -> None:
    """
    Write PDB files for the ensemble.

    Ensemble atoms (N, CA, C, O) are (n_internal, 3) from build_loop.
    We reconstruct the full (n_pep, 3) arrays by prepending/appending
    the native anchor atoms before calling write_pdb_atoms.
    native_N/CA/C/O must be the full (n_pep, 3) native backbone arrays.
    """
    out = Path(output_dir) / 'pdbs'
    out.mkdir(parents=True, exist_ok=True)
    rank_order = np.argsort([float(e[4]) for e in ensemble])

    has_anchors = (native_N is not None and native_CA is not None
                   and native_C is not None and native_O is not None)

    def _full(arr, nat):
        """Prepend/append native anchor rows to an internal (n_internal,3) array."""
        if not has_anchors:
            return arr
        if len(arr) == len(nat):
            return arr   # already full length
        return np.vstack([nat[:1], arr, nat[-1:]])

    def _rmsd(CA_pred):
        if native_CA is None:
            return float('nan')
        native_internal = native_CA[1:-1] if len(native_CA) > len(CA_pred) else native_CA
        return float(np.sqrt(np.mean(np.sum((CA_pred - native_internal) ** 2, axis=1))))

    for rank, idx in enumerate(rank_order, 1):
        N, CA, C, O, energy, closure = ensemble[idx]
        N_w  = _full(N,  native_N)
        CA_w = _full(CA, native_CA)
        C_w  = _full(C,  native_C)
        O_w  = _full(O,  native_O)
        rv = _rmsd(CA)
        rs = f"{rv:.2f}" if not math.isnan(rv) else 'na'
        fname = f"peptide_{rank:03d}_E{energy:.4f}_cl{closure:.3f}_rmsd{rs}.pdb"
        with open(out / fname, 'w') as fh:
            fh.write(f"REMARK rank={rank} energy={energy:.4f} "
                     f"closure={closure:.4f}A rmsd={rs}\n")
            write_pdb_atoms(fh, sequence, N_w, CA_w, C_w, O_w)
            fh.write("END\n")

    with open(out / f"ensemble_{name}.pdb", 'w') as fh:
        for rank, idx in enumerate(rank_order, 1):
            N, CA, C, O, energy, closure = ensemble[idx]
            N_w  = _full(N,  native_N)
            CA_w = _full(CA, native_CA)
            C_w  = _full(C,  native_C)
            O_w  = _full(O,  native_O)
            rv = _rmsd(CA); rs = f"{rv:.3f}" if not math.isnan(rv) else 'na'
            fh.write(f"MODEL {rank:4d}\n")
            fh.write(f"REMARK rank={rank} energy={energy:.4f} "
                     f"closure={closure:.4f}A rmsd={rs}\n")
            write_pdb_atoms(fh, sequence, N_w, CA_w, C_w, O_w)
            fh.write("ENDMDL\n")

    header = f"{'Rank':>4}  {'Energy':>10}  {'Closure(A)':>10}  {'RMSD(A)':>8}"
    with open(out / 'summary.txt', 'w') as fh:
        fh.write(header + '\n' + '─' * len(header) + '\n')
        for rank, idx in enumerate(rank_order, 1):
            _, CA, _, _, energy, closure = ensemble[idx]
            rv = _rmsd(CA)
            rs = f"{rv:8.3f}" if not math.isnan(rv) else f"{'na':>8}"
            fh.write(f"{rank:>4}  {energy:>10.4f}  {closure:>10.4f}  {rs}\n")

    if verbose:
        print(f"  Wrote {len(ensemble)} structures → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# 9.  CLI
# ─────────────────────────────────────────────────────────────────────────────

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="pMHC peptide backbone modeller (GD / SA, NN energy optional)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    io = p.add_argument_group("I/O")
    io.add_argument('--pdb',        required=True)
    io.add_argument('--chain',      default=None,
                    help="Peptide chain ID. Auto-detected if omitted.")
    io.add_argument('--output_dir', default='results')
    io.add_argument('--name',       default=None,
                    help="Ensemble name. Defaults to PDB stem.")

    nn = p.add_argument_group("Neural network energy")
    nn.add_argument('--checkpoint', default=None,
                    help="Path to checkpoint dir for TorsionPredictor. "
                         "If omitted, no NN energy is used (clash-only).")
    nn.add_argument('--config', default=None,
                    help="Path to config.json (defaults to <checkpoint>/config.json).")
    nn.add_argument('--no_jit', action='store_true',
                    help="Disable JAX JIT on the NN forward pass.")
    nn.add_argument('--energy_weight', type=float, default=1.0)
    nn.add_argument('--clash_weight',  type=float, default=1.0)
    nn.add_argument('--softness', type=float, default=0.8,
                    help="vdW radius scaling for softplus clash (default 0.8).")
    nn.add_argument('--clash_k',  type=float, default=5.0,
                    help="Softplus sharpness for clash penalty (default 5.0).")

    m = p.add_argument_group("Method")
    m.add_argument('--method', choices=['gd', 'sa', 'both'], default='gd')

    s = p.add_argument_group("Sampling")
    s.add_argument('--n_restarts',   type=int,   default=50)
    s.add_argument('--closure_tol',  type=float, default=0.5)
    s.add_argument('--overlap_tol',  type=float, default=0.6)
    s.add_argument('--ccd_iter',     type=int,   default=200)
    s.add_argument('--rng_seed',     type=int,   default=None)
    s.add_argument('--no_intra',     action='store_true')
    s.add_argument('--no_framework', action='store_true',
                   help="Skip framework clash scoring.")

    gd = p.add_argument_group("Gradient descent")
    gd.add_argument('--n_inner',    type=int,   default=20)
    gd.add_argument('--step_size',  type=float, default=0.1)
    gd.add_argument('--step_decay', type=float, default=0.95)

    sa = p.add_argument_group("Simulated annealing")
    sa.add_argument('--n_steps',  type=int,   default=2000)
    sa.add_argument('--T_init',   type=float, default=5.0)
    sa.add_argument('--T_final',  type=float, default=0.1)
    sa.add_argument('--sa_step',  type=float, default=0.3)

    p.add_argument('--verbose', action='store_true')
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)

    pdb_path = Path(args.pdb)
    if not pdb_path.exists():
        print(f"ERROR: PDB not found: {pdb_path}", file=sys.stderr)
        return 1

    name  = args.name or pdb_path.stem
    chain = args.chain or detect_peptide_chain(pdb_path)

    # ── Sequence and anchor geometry ──────────────────────────────────────
    print(f"[main] Extracting sequence from chain '{chain}'")
    seq, N_nat, CA_nat, C_nat, O_nat = extract_peptide_sequence(pdb_path, chain)
    print(f"[main] Sequence ({len(seq)} residues): {seq}")

    (prev_N, prev_CA, prev_C,
     psi_prev,
     target_N, target_CA, target_C) = extract_pmhc_anchors(pdb_path, chain)

    # ── Framework atoms ───────────────────────────────────────────────────
    fw_coords: Optional[np.ndarray] = None
    fw_radii:  Optional[np.ndarray] = None

    if not args.no_framework:
        try:
            fw_coords, fw_radii = extract_framework_atoms_for_peptide(
                pdb_path, chain, seq
            )
            print(f"[main] Framework: {len(fw_coords)} atoms")
        except Exception as exc:
            warnings.warn(
                f"[main] Framework extraction failed ({exc}); "
                "clash scoring disabled.", stacklevel=2
            )

    # ── NN energy function ────────────────────────────────────────────────
    energy_fn: Optional[EnergyFn] = None

    if args.checkpoint:
        print(f"[main] Loading NN model from {args.checkpoint}")
        try:
            router    = load_model(args.checkpoint, args.config)
            energy_fn = build_nn_energy_fn(
                router,
                sequence  = seq,
                prev_N    = prev_N,
                prev_CA   = prev_CA,
                prev_C    = prev_C,
                target_N  = target_N,
                target_CA = target_CA,
                target_C  = target_C,
                use_jit   = not args.no_jit,
            )
            print("[main] NN energy function ready")
        except Exception as exc:
            warnings.warn(
                f"[main] Could not load NN model ({exc}); "
                "energy_fn disabled.", stacklevel=2
            )
    else:
        print("[main] No --checkpoint provided; clash-only scoring.")

    # ── Shared kwargs ─────────────────────────────────────────────────────
    shared = dict(
        sequence      = seq,
        prev_N        = prev_N,  prev_CA   = prev_CA,  prev_C    = prev_C,
        psi_prev      = psi_prev,
        target_N      = target_N, target_CA = target_CA, target_C = target_C,
        fw_coords     = fw_coords,
        fw_radii      = fw_radii,
        energy_fn     = energy_fn,
        energy_weight = args.energy_weight,
        clash_weight  = args.clash_weight,
        softness      = args.softness,
        clash_k       = args.clash_k,
        overlap_tol   = args.overlap_tol,
        closure_tol   = args.closure_tol,
        ccd_iter      = args.ccd_iter,
        ccd_tol       = 0.05,
        n_restarts    = args.n_restarts,
        check_intra   = not args.no_intra,
        rng_seed      = args.rng_seed,
        verbose       = args.verbose,
    )

    ensemble: List[EnsembleEntry] = []

    if args.method in ('gd', 'both'):
        print(f"\n[main] Gradient descent ({args.n_restarts} restarts) …")
        gd_ens = gradient_descent_peptide(
            **shared,
            n_inner    = args.n_inner,
            step_size  = args.step_size,
            step_decay = args.step_decay,
        )
        print(f"[main] GD accepted: {len(gd_ens)}")
        ensemble.extend(gd_ens)

    if args.method in ('sa', 'both'):
        print(f"\n[main] Simulated annealing "
              f"({args.n_restarts} restarts × {args.n_steps} steps) …")
        sa_ens = simulated_annealing_peptide(
            **shared,
            n_steps   = args.n_steps,
            T_init    = args.T_init,
            T_final   = args.T_final,
            step_size = args.sa_step,
        )
        print(f"[main] SA accepted: {len(sa_ens)}")
        ensemble.extend(sa_ens)

    if not ensemble:
        print("[main] No structures accepted. Try --n_restarts, "
              "--closure_tol, or --no_intra.", file=sys.stderr)
        return 1

    print(f"\n[main] Saving {len(ensemble)} structures → {args.output_dir}")
    save_peptide_pdbs(
        ensemble   = ensemble,
        sequence   = seq,
        name       = name,
        output_dir = args.output_dir,
        native_CA  = CA_nat,
        native_N   = N_nat,
        native_C   = C_nat,
        native_O   = O_nat,
        verbose    = True,
    )
    print("[main] Done.")
    return 0


if __name__ == '__main__':
    sys.exit(main())