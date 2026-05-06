"""
utils.py

Shared utilities for loop modeling pipeline.

Sections
────────
  1. Model loading and routing   (ModelRouter, load_model, _to_router)
  2. Output configuration        (OutputConfig)
  3. PDB I/O                     (load_cdr3_native, write_pdb_atoms)
  4. Structure selection         (select_structures)
  5. Output writers              (save_pdbs, save_trajectory)
  6. Plots                       (plot_energy, plot_summary)
"""

from __future__ import annotations

import json
import math
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple
import sys
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Model loading and routing
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ModelRouter:
    """
    Routes each residue to the appropriate specialist JAX/Flax model.

    The general model handles all residues.  Optional specialist models for
    glycine ('G') and proline ('P') override it for those residue types.

    Usage
    ─────
      # General model only
      router = ModelRouter(general_model, general_params)

      # With glycine / proline specialists
      router = ModelRouter(general_model, general_params,
                           gly_model=gm, gly_params=gp,
                           pro_model=pm, pro_params=pp)

      # Query
      model, params = router.get('G')
    """
    general_model:  object
    general_params: object
    gly_model:      object = None
    gly_params:     object = None
    pro_model:      object = None
    pro_params:     object = None

    def get(self, aa: str):
        """Return (model, params) for amino acid one-letter code."""
        if aa == 'G' and self.gly_model is not None:
            return self.gly_model, self.gly_params
        if aa == 'P' and self.pro_model is not None:
            return self.pro_model, self.pro_params
        return self.general_model, self.general_params

    @classmethod
    def from_pair(cls, model, params) -> ModelRouter:
        """Wrap a plain (model, params) pair with no specialists."""
        return cls(general_model=model, general_params=params)


def _to_router(model_or_router, params=None) -> ModelRouter:
    """Accept either a ModelRouter or a plain (model, params) pair."""
    if isinstance(model_or_router, ModelRouter):
        return model_or_router
    return ModelRouter.from_pair(model_or_router, params)


def load_model(checkpoint_dir: str, config_path: str = None) -> ModelRouter:
    """
    Load a trained TorsionPredictor from a checkpoint directory.

    Reads hyperparameters from config.json written by train.py so the model
    architecture is always consistent with the saved weights.

    Args:
        checkpoint_dir: directory containing config.json and checkpoints/
        config_path:    optional explicit path to config.json

    Returns:
        ModelRouter wrapping the loaded model (general model only)
    """
    from flax.training import checkpoints
    # Import here so utils.py has no hard dependency on the model package
    # when used in contexts that only need routing / I/O helpers.
    sys.path.append('/home/jtepperik/thesis/energy_model/scripts')
    from models.full_model import TorsionPredictor

    checkpoint_dir = Path(checkpoint_dir)
    cfg_path = Path(config_path) if config_path else checkpoint_dir / 'config.json'

    with open(cfg_path) as f:
        config = json.load(f)

    model = TorsionPredictor(
        max_context  = config['max_context'],
        embed_dim    = config['embed_dim'],
        hidden_dim   = config['hidden_dim'],
        n_layers     = config['n_layers'],
        dropout_rate = config['dropout_rate'],
        n_bins       = config['n_bins'],
    )

    state = checkpoints.restore_checkpoint(
        ckpt_dir = checkpoint_dir / 'checkpoints',
        target   = None,
        prefix   = 'best_',
    )
    return ModelRouter.from_pair(model, state['params'])


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Output configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class OutputConfig:
    """
    Controls which structures receive output and what is generated.

    Selection
    ─────────
      selection_mode : 'best_n'   — top-N structures by rank_by criterion
                       'indices'  — explicit list of structure indices
                       'all'      — every structure in the ensemble

      n_select       : number of structures to select (best_n mode)
      indices        : explicit list e.g. [0, 4, 9]  (indices mode)
      rank_by        : 'rmsd'    — requires CA_native_loop (default)
                       'closure' — rank by closure distance
                       'energy'  — rank by torsion energy

    Outputs (all independent, all default True)
    ───────
      save_pdbs       : write individual PDB + ranked multi-model PDB
      save_trajectory : write trajectory PDB + PyMOL script for top-1 structure
      plot_energy     : per-residue energy landscape, heatmap, Ramachandran
      plot_summary    : ensemble-level RMSD / energy summary figure

    Trajectory note
    ───────────────
      A trajectory is only available for the single top-ranked structure
      (n_frames must be > 0 in the optimisation call).
    """
    # Selection
    selection_mode: str        = 'best_n'
    n_select:       int        = 5
    indices:        List[int]  = field(default_factory=list)
    rank_by:        str        = 'rmsd'

    # Outputs
    save_pdbs:       bool = True
    save_trajectory: bool = True
    plot_energy:     bool = True
    plot_summary:    bool = True
    run_baselines:   bool = True   # uniform + model_sample random ensembles

    # Root output directory
    output_dir: str = 'output'

    def resolve_indices(
        self,
        ensemble:        list,
        CA_native_loop:  Optional[np.ndarray],
        loop_start:      int,
        loop_end:        int,
    ) -> List[int]:
        """
        Return a sorted list of structure indices to generate output for.

        Ranking is performed over the full ensemble; the top n_select are
        returned.  When rank_by='rmsd' and no native is available, falls back
        to 'closure' with a warning.
        """
        n = len(ensemble)

        if self.selection_mode == 'indices':
            bad = [i for i in self.indices if i < 0 or i >= n]
            if bad:
                warnings.warn(f"OutputConfig: indices {bad} out of range — ignored")
            return sorted(i for i in self.indices if 0 <= i < n)

        if self.selection_mode == 'all':
            return list(range(n))

        # best_n — compute scores over full ensemble
        criterion = self.rank_by
        if criterion == 'rmsd' and CA_native_loop is None:
            warnings.warn("rank_by='rmsd' requested but CA_native_loop not provided — "
                          "falling back to 'closure'")
            criterion = 'closure'

        scores = _rank_scores(ensemble, CA_native_loop, loop_start, loop_end, criterion)
        order  = np.argsort(scores)
        return sorted(order[:self.n_select].tolist())


def _rank_scores(
    ensemble:       list,
    CA_native_loop: Optional[np.ndarray],
    loop_start:     int,
    loop_end:       int,
    criterion:      str,
) -> np.ndarray:
    """Return a score array (lower = better) for the full ensemble."""
    if criterion == 'rmsd':
        return np.array([
            float(np.sqrt(np.mean(np.sum(
                (CA[loop_start:loop_end] - CA_native_loop) ** 2, axis=1
            )))) for _, CA, *_ in ensemble
        ])
    if criterion == 'closure':
        return np.array([float(e[-1]) for e in ensemble])
    if criterion == 'energy':
        return np.array([float(e[-2]) for e in ensemble])
    raise ValueError(f"Unknown rank_by criterion: {criterion!r}")


# ─────────────────────────────────────────────────────────────────────────────
# 3.  PDB I/O and framework extraction
# ─────────────────────────────────────────────────────────────────────────────

# VdW radii (Å) for backbone + Cβ atoms.
# Softness factor 0.8 applied at clash computation time (RCD paper convention).
VDW_RADII = {
    'N':  1.65,
    'CA': 1.87,
    'C':  1.76,
    'O':  1.40,
    'CB': 1.87,
}
_BACKBONE_ATOMS = ('N', 'CA', 'C', 'O')

ONE_TO_THREE = {
    'A':'ALA','C':'CYS','D':'ASP','E':'GLU','F':'PHE','G':'GLY',
    'H':'HIS','I':'ILE','K':'LYS','L':'LEU','M':'MET','N':'ASN',
    'P':'PRO','Q':'GLN','R':'ARG','S':'SER','T':'THR','V':'VAL',
    'W':'TRP','Y':'TYR',
}
THREE_TO_ONE = {v: k for k, v in ONE_TO_THREE.items()}


def load_cdr3_native(pdb_file: str):
    """
    Parse a CDR3 PDB file.

    Returns:
        sequence:  one-letter string
        N, CA, C, O:  (n_res, 3) float arrays
    """
    N_list, CA_list, C_list, O_list, sequence = [], [], [], [], []
    with open(pdb_file) as f:
        for line in f:
            if not line.startswith('ATOM'):
                continue
            if line[16] not in (' ', 'A'):
                continue
            atom = line[12:16].strip()
            res  = line[17:20].strip()
            xyz  = float(line[30:38]), float(line[38:46]), float(line[46:54])
            if atom == 'CA':
                sequence.append(THREE_TO_ONE.get(res, 'X'))
            if   atom == 'N':  N_list .append(xyz)
            elif atom == 'CA': CA_list.append(xyz)
            elif atom == 'C':  C_list .append(xyz)
            elif atom == 'O':  O_list .append(xyz)
    return (
        ''.join(sequence),
        np.array(N_list),  np.array(CA_list),
        np.array(C_list),  np.array(O_list),
    )


def write_pdb_atoms(f, sequence: str, N, CA, C, O, atom_num: int = 1) -> int:
    """Write ATOM records for one structure; returns the next atom number."""
    for i, aa in enumerate(sequence):
        res = ONE_TO_THREE.get(aa, 'UNK')
        for name, coord in [('N', N[i]), ('CA', CA[i]), ('C', C[i])]:
            f.write(f"ATOM  {atom_num:5d}  {name:<3s} {res} A{i+1:4d}    "
                    f"{coord[0]:8.3f}{coord[1]:8.3f}{coord[2]:8.3f}"
                    f"  1.00  0.00           {name[0]}  \n")
            atom_num += 1
        if np.linalg.norm(O[i]) > 1e-6:
            f.write(f"ATOM  {atom_num:5d}  O   {res} A{i+1:4d}    "
                    f"{O[i,0]:8.3f}{O[i,1]:8.3f}{O[i,2]:8.3f}"
                    f"  1.00  0.00           O  \n")
            atom_num += 1
    return atom_num


# ─────────────────────────────────────────────────────────────────────────────
# Replacement for extract_framework_atoms in utils.py
# ─────────────────────────────────────────────────────────────────────────────
# Drop-in replacement.  Adds a coordinate-based fallback for when the sequence
# string match fails (e.g. IMGT insertion codes cause residue ordering issues).
#
# Fallback strategy:
#   1. Parse ALL CA atoms from tcr_chain with their coordinates and residue keys.
#   2. Find the CA atom in the complex PDB closest to the known N-terminal anchor
#      (loop_start - 1) and C-terminal anchor (loop_end) coordinates from the
#      CDR3 PDB.  These atoms are NOT excluded so we look just outside the loop.
#   3. Exclude all residues between (and including) those two anchor positions
#      in the chain sequence order.
#
# This is robust to IMGT insertion codes because it uses spatial proximity
# rather than sequence string matching.

def extract_framework_atoms(
    complex_pdb:    str,
    tcr_chain:      str,
    full_sequence:  str,
    loop_start:     int,
    loop_end:       int,
    n_flank_before: int,
    n_flank_after:  int,
    # Optional: pass anchor coordinates for coordinate-based fallback
    anchor_C_coord:   np.ndarray | None = None,  # C of last N-flank residue
    target_N_coord:   np.ndarray | None = None,  # N of first C-flank residue
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract fixed framework atom coordinates and vdW radii from a full
    TCR-pMHC complex PDB for use in clash detection.

    Excludes loop residues + flank anchors from tcr_chain.
    All other chains are always included.

    When sequence string matching fails (e.g. IMGT insertion codes), falls
    back to coordinate-based residue identification using anchor_C_coord and
    target_N_coord to locate the loop boundaries in the complex PDB.

    Args:
        complex_pdb:      Path to full complex PDB.
        tcr_chain:        Chain containing the CDR3 loop.
        full_sequence:    Full sequence string (loop + flanks).
        loop_start:       Loop start index into full_sequence (0-based).
        loop_end:         Loop end index into full_sequence (0-based).
        n_flank_before:   N-terminal flank residue count.
        n_flank_after:    C-terminal flank residue count.
        anchor_C_coord:   (3,) C coordinate of residue loop_start-1.
                          Used for coordinate fallback when seq match fails.
        target_N_coord:   (3,) N coordinate of residue loop_end.
                          Used for coordinate fallback when seq match fails.

    Returns:
        coords: (N_atoms, 3) float32
        radii:  (N_atoms,)  float32
    """
    import warnings

    # ── Try sequence-string match first ──────────────────────────────────
    tcr_resnums = _parse_chain_resnums(complex_pdb, tcr_chain, full_sequence)
    exclude_resnums: set = set()

    if tcr_resnums is not None:
        # Sequence match succeeded — standard exclusion by index
        excl_idx = (set(range(loop_start, loop_end))
                    | set(range(0, n_flank_before))
                    | set(range(loop_end, loop_end + n_flank_after)))
        exclude_resnums = {tcr_resnums[i] for i in excl_idx
                           if i < len(tcr_resnums)}

    elif anchor_C_coord is not None and target_N_coord is not None:
        # ── Coordinate-based fallback ─────────────────────────────────────
        # Parse all CA atoms from tcr_chain to get (resnum, icode, coord)
        chain_cas: list = []   # [(resnum_str, coord)]
        seen: set = set()
        with open(complex_pdb) as f:
            for line in f:
                if not line.startswith('ATOM'):
                    continue
                if line[21] != tcr_chain:
                    continue
                if line[12:16].strip() != 'CA':
                    continue
                resnum = line[22:26].strip()
                icode  = line[26].strip()
                key    = (resnum, icode)
                if key in seen:
                    continue
                seen.add(key)
                coord = np.array([float(line[30:38]),
                                   float(line[38:46]),
                                   float(line[46:54])])
                chain_cas.append((resnum, icode, coord))

        if len(chain_cas) == 0:
            warnings.warn(
                f"extract_framework_atoms: chain {tcr_chain} has no CA atoms "
                f"in {complex_pdb} — no exclusion applied"
            )
        else:
            coords_arr = np.array([c[2] for c in chain_cas])

            # Find residue in complex PDB closest to anchor C (loop_start-1)
            # The C atom of the anchor is ~1.5A from its own CA, so closest
            # CA to anchor_C_coord is the anchor residue itself
            dists_n = np.linalg.norm(coords_arr - anchor_C_coord, axis=1)
            anchor_seq_idx = int(np.argmin(dists_n))

            # Find residue closest to target N (loop_end residue)
            dists_c = np.linalg.norm(coords_arr - target_N_coord, axis=1)
            target_seq_idx = int(np.argmin(dists_c))

            # Exclude everything from (anchor - n_flank_before) to
            # (target + n_flank_after) inclusive, in sequence order
            excl_start = max(0, anchor_seq_idx - n_flank_before + 1)
            excl_end   = min(len(chain_cas), target_seq_idx + n_flank_after + 1)

            for i in range(excl_start, excl_end):
                rn, ic, _ = chain_cas[i]
                # Store as integer resnum for matching with PDB parser below
                try:
                    exclude_resnums.add(int(rn))
                except ValueError:
                    pass   # non-numeric resnum — skip

            warnings.warn(
                f"extract_framework_atoms: sequence match failed for chain "
                f"{tcr_chain} in {complex_pdb}. Used coordinate fallback — "
                f"excluding {len(exclude_resnums)} residues "
                f"(seq positions {excl_start}-{excl_end-1})."
            )
    else:
        warnings.warn(
            f"extract_framework_atoms: could not locate full_sequence in "
            f"chain {tcr_chain} of {complex_pdb} and no anchor coordinates "
            f"provided — no exclusion applied. "
            f"Pass anchor_C_coord and target_N_coord for coordinate fallback."
        )

    # ── Parse atoms, excluding identified residues ────────────────────────
    coords_list: list = []
    radii_list:  list = []

    current_key     = None
    current_chain   = None
    current_resnum  = None
    current_resname = None
    buf: dict = {}

    def _flush():
        if current_chain is None:
            return
        if current_chain == tcr_chain and current_resnum in exclude_resnums:
            return
        aa = THREE_TO_ONE.get(current_resname, 'X')
        for atom in _BACKBONE_ATOMS:
            if atom in buf:
                coords_list.append(buf[atom])
                radii_list.append(VDW_RADII[atom])
        if 'CB' in buf and aa != 'G':
            coords_list.append(buf['CB'])
            radii_list.append(VDW_RADII['CB'])

    with open(complex_pdb) as f:
        for line in f:
            if not line.startswith('ATOM'):
                continue
            if line[16] not in (' ', 'A'):
                continue
            chain   = line[21]
            resnum  = int(line[22:26].strip())
            resname = line[17:20].strip()
            atom    = line[12:16].strip()
            if atom not in (*_BACKBONE_ATOMS, 'CB'):
                continue
            key = (chain, resnum)
            if key != current_key:
                _flush()
                current_key     = key
                current_chain   = chain
                current_resnum  = resnum
                current_resname = resname
                buf = {}
            buf[atom] = [float(line[30:38]),
                         float(line[38:46]),
                         float(line[46:54])]
    _flush()

    if not coords_list:
        raise ValueError(f"No framework atoms extracted from {complex_pdb}")

    coords = np.array(coords_list, dtype=np.float32)
    radii  = np.array(radii_list,  dtype=np.float32)
    print(f"    Framework: {len(coords):,} atoms  "
          f"({len(exclude_resnums)} residues excluded)  ({complex_pdb})")
    return coords, radii

def _parse_chain_resnums(
    pdb_file: str, chain: str, full_sequence: str,
) -> Optional[List[int]]:
    """
    Find `full_sequence` as a substring of `chain` and return the matching
    residue numbers from the PDB.  Returns None on failure.
    """
    resnums: list  = []
    seq_chars: list = []
    seen: set = set()

    with open(pdb_file) as f:
        for line in f:
            if not line.startswith('ATOM'):
                continue
            if line[21] != chain or line[12:16].strip() != 'CA':
                continue
            rn = int(line[22:26].strip())
            if rn in seen:
                continue
            seen.add(rn)
            resnums.append(rn)
            seq_chars.append(THREE_TO_ONE.get(line[17:20].strip(), 'X'))

    chain_seq = ''.join(seq_chars)
    idx = chain_seq.find(full_sequence)
    if idx == -1:
        # Fuzzy fallback — allow up to 10 % mismatch
        best_score, best_idx = 0, -1
        n = len(full_sequence)
        for i in range(len(chain_seq) - n + 1):
            score = sum(a == b for a, b in zip(chain_seq[i:i+n], full_sequence))
            if score > best_score:
                best_score, best_idx = score, i
        if best_idx >= 0 and best_score / len(full_sequence) >= 0.90:
            idx = best_idx
        else:
            return None

    return resnums[idx: idx + len(full_sequence)]


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Structure selection helpers
# ─────────────────────────────────────────────────────────────────────────────

def compute_loop_rmsds(
    ensemble:       list,
    CA_native_loop: np.ndarray,
    loop_start:     int,
    loop_end:       int,
) -> np.ndarray:
    """Anchored (no alignment) per-structure RMSD to native loop Cα."""
    return np.array([
        float(np.sqrt(np.mean(np.sum(
            (CA[loop_start:loop_end] - CA_native_loop) ** 2, axis=1
        )))) for _, CA, *_ in ensemble
    ])


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Output writers
# ─────────────────────────────────────────────────────────────────────────────

def save_pdbs(
    ensemble:       list,
    selected_idx:   List[int],
    full_sequence:  str,
    loop_start:     int,
    loop_end:       int,
    CA_native_loop: Optional[np.ndarray],
    name:           str,
    output_dir:     str,
):
    """
    Write individual PDB files for selected structures and a ranked
    multi-model PDB covering the full ensemble.

    Files written
    ─────────────
      <output_dir>/pdbs/structure_<N>_<metrics>.pdb   — one per selected structure
      <output_dir>/pdbs/ensemble_<name>.pdb            — full ensemble, ranked
      <output_dir>/pdbs/summary.txt                    — ranking table
    """
    out = Path(output_dir) / 'pdbs'
    out.mkdir(parents=True, exist_ok=True)

    # Compute metrics for all structures
    rmsds    = (compute_loop_rmsds(ensemble, CA_native_loop, loop_start, loop_end)
                if CA_native_loop is not None
                else np.full(len(ensemble), float('nan')))
    closures = np.array([float(e[-1]) for e in ensemble])
    energies = np.array([float(e[-2]) for e in ensemble])

    # Rank by closure for the multi-model PDB (always available)
    rank_order = np.argsort(closures)

    # Individual PDBs for selected structures
    for idx in selected_idx:
        N, CA, C, O = ensemble[idx][:4]
        rmsd_str = f"{rmsds[idx]:.2f}" if not math.isnan(rmsds[idx]) else 'na'
        fname    = (f"structure_{idx+1:02d}_"
                    f"rmsd{rmsd_str}_cl{closures[idx]:.3f}_E{energies[idx]:.2f}.pdb")
        with open(out / fname, 'w') as f:
            f.write(f"REMARK structure {idx+1}  rmsd={rmsd_str}A  "
                    f"closure={closures[idx]:.4f}A  energy={energies[idx]:.3f}\n")
            write_pdb_atoms(f, full_sequence, N, CA, C, O)
            f.write("END\n")

    # Multi-model PDB (all structures, ranked by closure)
    with open(out / f"ensemble_{name}.pdb", 'w') as f:
        for rank, idx in enumerate(rank_order, 1):
            N, CA, C, O = ensemble[idx][:4]
            rmsd_str = f"{rmsds[idx]:.3f}" if not math.isnan(rmsds[idx]) else 'na'
            f.write(f"MODEL {rank:4d}\n")
            f.write(f"REMARK rank={rank}  rmsd={rmsd_str}A  "
                    f"closure={closures[idx]:.4f}A  energy={energies[idx]:.3f}\n")
            write_pdb_atoms(f, full_sequence, N, CA, C, O)
            f.write("ENDMDL\n")

    # Summary table
    with open(out / 'summary.txt', 'w') as f:
        header = f"{'Rank':>4}  {'Struct':>8}  {'RMSD(A)':>9}  {'Closure(A)':>10}  {'Energy':>8}  {'Selected':>8}"
        f.write(header + '\n')
        f.write('─' * len(header) + '\n')
        sel_set = set(selected_idx)
        for rank, idx in enumerate(rank_order, 1):
            rmsd_str = f"{rmsds[idx]:9.3f}" if not math.isnan(rmsds[idx]) else f"{'na':>9}"
            f.write(f"{rank:>4}  {idx+1:>8}  {rmsd_str}  "
                    f"{closures[idx]:>10.4f}  {energies[idx]:>8.2f}  "
                    f"{'✓' if idx in sel_set else '':>8}\n")

    print(f"    PDbs: {len(selected_idx)} individual + ensemble → {out}")


def save_trajectory(
    trajectory:      list,
    top_idx:         int,
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
    CA_native_loop:  Optional[np.ndarray],
    name:            str,
    output_dir:      str,
):
    """
    Write trajectory PDB + companion PyMOL script for the top-ranked structure.

    The trajectory is stored as a multi-MODEL PDB.  B-factors encode per-residue
    Cα deviation from native (×10) when a native structure is available.
    """
    from loop_modeling_nerf import compute_O_atoms   # avoid circular at module level

    if not trajectory:
        print("    Trajectory: no frames recorded (n_frames=0)")
        return

    out      = Path(output_dir) / 'trajectory'
    out.mkdir(parents=True, exist_ok=True)
    pdb_path = out / f"trajectory_{name}.pdb"

    _ca_native = (CA_native_loop if CA_native_loop is not None
                  else np.zeros((loop_end - loop_start, 3)))

    with open(pdb_path, 'w') as f:
        for model_idx, frame in enumerate(trajectory, 1):
            step    = frame[0]
            N_b     = frame[1][top_idx]
            CA_b    = frame[2][top_idx]
            C_b     = frame[3][top_idx]
            energy  = float(frame[4][top_idx]) if len(frame) > 4 else None
            closure = float(frame[5][top_idx]) if len(frame) > 5 else None

            N_np  = N_b.cpu().numpy()  if hasattr(N_b,  'numpy') else N_b
            CA_np = CA_b.cpu().numpy() if hasattr(CA_b, 'numpy') else CA_b
            C_np  = C_b.cpu().numpy()  if hasattr(C_b,  'numpy') else C_b
            O_np  = compute_O_atoms(N_np, CA_np, C_np)

            N_full  = np.vstack([N_flank_before,  N_np,  N_flank_after])
            CA_full = np.vstack([CA_flank_before, CA_np, CA_flank_after])
            C_full  = np.vstack([C_flank_before,  C_np,  C_flank_after])
            O_full  = np.vstack([O_flank_before,  O_np,  O_flank_after])

            rmsd_loop = np.sqrt(np.sum((CA_np - _ca_native) ** 2, axis=1))
            bfactor   = np.zeros(len(full_sequence))
            bfactor[loop_start:loop_end] = rmsd_loop * 10.0
            loop_rmsd = float(np.sqrt(np.mean(rmsd_loop ** 2)))

            f.write(f"MODEL {model_idx:6d}\n")
            remark = f"REMARK step={step}  loop_rmsd={loop_rmsd:.3f}A"
            if energy  is not None: remark += f"  energy={energy:.2f}"
            if closure is not None: remark += f"  closure={closure:.4f}A"
            f.write(remark + '\n')

            atom_num = 1
            for i, aa in enumerate(full_sequence):
                res = ONE_TO_THREE.get(aa, 'UNK')
                bf  = bfactor[i]
                for aname, coord in [('N', N_full[i]), ('CA', CA_full[i]),
                                     ('C', C_full[i])]:
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

    # PyMOL script — black bg, licorice, pseudoatom labels
    n_models  = len(trajectory)
    pml_path  = out / f"trajectory_{name}.pml"
    with open(pml_path, 'w') as pml:
        pml.write(f"# Trajectory: {name}  ({n_models} frames)\n")
        pml.write(f"# Loop resi {loop_start+1}-{loop_end}\n\n")
        pml.write(f"load {pdb_path.name}, traj\n")
        pml.write(f"bg_color black\n")
        pml.write(f"set stick_radius, 0.15\n")
        pml.write(f"set stick_ball, on\n")
        pml.write(f"set stick_ball_ratio, 1.5\n")
        pml.write(f"hide everything, traj\n")
        pml.write(f"show sticks, traj\n")
        pml.write(f"color grey60, traj\n")
        pml.write(f"spectrum b, blue_white_red, traj and resi "
                  f"{loop_start+1}-{loop_end}, minimum=0, maximum=30\n\n")

        # Create pseudoatom labels for each state — positioned in top-left corner
        # We use a single pseudoatom object and set its position per-state
        # to the top-left of the current view using scene_to_screen mapping
        pml.write(f"# Per-frame labels via pseudoatoms (top-left corner)\n")
        pml.write(f"set label_relative_mode, 1\n")  # labels in screen-relative coords
        for state_idx, frame in enumerate(trajectory, 1):
            step    = frame[0]
            energy  = float(frame[4][top_idx]) if len(frame) > 4 else None
            closure = float(frame[5][top_idx]) if len(frame) > 5 else None
            label   = f"step {step}"
            if energy  is not None: label += f" | E={energy:.2f}"
            if closure is not None: label += f" | cl={closure:.4f}A"
            pml.write(f'pseudoatom lbl, pos=[0,0,0], state={state_idx}, '
                      f'label="{label}"\n')

        pml.write(f"\nhide everything, lbl\n")
        pml.write(f"show labels, lbl\n")
        pml.write(f"set label_color, white, lbl\n")
        pml.write(f"set label_size, -0.8\n")  # negative = Angstrom-sized, scales with zoom
        pml.write(f"set label_position, [-25, 20, 0]\n")  # offset to top-left
        pml.write(f"set label_font_id, 7\n")

        pml.write(f"\nmset 1 -{n_models}\nset movie_fps, 4\nzoom traj\nmplay\n")

    print(f"    Trajectory: {n_models} frames → {pdb_path.name}")
    print(f"    PyMOL:      {pml_path.name}")


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Plots
# ─────────────────────────────────────────────────────────────────────────────

# Bin centres for marginal distribution plots (36 bins, 10° each)
_BIN_CENTRES = np.array([-180.0 + (k + 0.5) * 10.0 for k in range(36)])


def _marginals(probs_joint: np.ndarray):
    p = np.array(probs_joint, dtype=np.float64)
    return p.sum(axis=1), p.sum(axis=0)


def _interp_1d(angle_deg: float, probs: np.ndarray) -> float:
    n  = len(probs)
    bw = 360.0 / n
    f  = ((angle_deg + 180.0) % 360.0) / bw
    lo = int(f) % n;  hi = (lo + 1) % n
    w  = f - int(f)
    return float((1 - w) * probs[lo] + w * probs[hi])


def _interp_2d(phi_deg: float, psi_deg: float, pj: np.ndarray) -> float:
    n  = pj.shape[0]
    bw = 360.0 / n
    pf = ((phi_deg + 180.0) % 360.0) / bw
    sf = ((psi_deg + 180.0) % 360.0) / bw
    pl = int(pf) % n;  ph = (pl + 1) % n
    sl = int(sf) % n;  sh = (sl + 1) % n
    pw = pf - int(pf);  sw = sf - int(sf)
    return float(
        (1-pw)*(1-sw)*pj[pl,sl] + (1-pw)*sw*pj[pl,sh] +
           pw *(1-sw)*pj[ph,sl] +    pw *sw*pj[ph,sh]
    )


def _per_residue_energy(phi_arr, psi_arr, probs_joint):
    """Compute per-residue energy dict list from angles (degrees) and joint tables."""
    rows = []
    for i, (phi, psi) in enumerate(zip(phi_arr, psi_arr)):
        e_joint = e_phi = e_psi = float('nan')
        if i < len(probs_joint) and not (math.isnan(phi) or math.isnan(psi)):
            pj = probs_joint[i]
            e_joint = -math.log(max(abs(_interp_2d(phi, psi, pj)), 1e-10))
            p_phi, p_psi = _marginals(pj)
            e_phi = -math.log(max(_interp_1d(phi, p_phi), 1e-10))
            e_psi = -math.log(max(_interp_1d(psi, p_psi), 1e-10))
        rows.append(dict(idx=i, phi=phi, psi=psi,
                         e_joint=e_joint, e_phi=e_phi, e_psi=e_psi,
                         e_total=e_joint))
    return rows


def _compute_shared_vmax(
    probs_joint:  list,
    phi_native:   np.ndarray = None,
    psi_native:   np.ndarray = None,
    ensemble:     list = None,
    selected_idx: List[int] = None,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """
    Compute shared vmin/vmax values for Ramachandran and heatmap plots,
    ensuring native and generated structures use the same colour scale.

    Returns ((vmin_rama, vmax_rama), (vmin_heatmap, vmax_heatmap)):
      Ramachandran : vmin=4, vmax=argmax of joint energy
      Heatmap      : vmin=2, vmax=argmax of marginal energy / 2
    """
    all_joint   = []
    all_marginal = []

    def _collect(phi_arr, psi_arr):
        rows = _per_residue_energy(phi_arr, psi_arr, probs_joint)
        for r in rows:
            if not math.isnan(r['e_joint']):
                all_joint.append(r['e_joint'])
            if not math.isnan(r['e_phi']):
                all_marginal.append(r['e_phi'])
            if not math.isnan(r['e_psi']):
                all_marginal.append(r['e_psi'])

    # Native angles
    if phi_native is not None and psi_native is not None:
        _collect(phi_native, psi_native)

    # Selected ensemble structures
    if ensemble is not None and selected_idx is not None:
        for idx in selected_idx:
            _collect(ensemble[idx][4], ensemble[idx][5])

    # Compute ranges
    max_joint    = max(all_joint)    if all_joint    else 8.0
    max_marginal = max(all_marginal) if all_marginal else 4.0

    rama_range    = (4.0, max_joint)
    heatmap_range = (2.0, max_marginal / 2.0)

    return rama_range, heatmap_range


def plot_energy(
    ensemble:      list,
    selected_idx:  List[int],
    loop_sequence: str,
    probs_joint:   list,
    name:          str,
    output_dir:    str,
    rama_range:    Tuple[float, float] = None,
    heatmap_range: Tuple[float, float] = None,
):
    """
    For each selected structure write three figures:
      - energy heatmap + stacked bar
      - per-residue φ/ψ landscape with angle markers
      - Ramachandran scatter coloured by joint energy

    rama_range / heatmap_range: (vmin, vmax) colour scale bounds.
    When provided (e.g. from _compute_shared_vmax), ensures native and
    generated plots use the same scale.
    """
    out = Path(output_dir) / 'energy_plots'
    out.mkdir(parents=True, exist_ok=True)

    # Compute ranges if not supplied
    if rama_range is None or heatmap_range is None:
        all_joint = []
        all_marginal = []
        for idx in selected_idx:
            for r in _per_residue_energy(ensemble[idx][4], ensemble[idx][5],
                                         probs_joint):
                if not math.isnan(r['e_joint']):
                    all_joint.append(r['e_joint'])
                if not math.isnan(r['e_phi']):
                    all_marginal.append(r['e_phi'])
                if not math.isnan(r['e_psi']):
                    all_marginal.append(r['e_psi'])
        if rama_range is None:
            rama_range = (4.0, max(all_joint) if all_joint else 8.0)
        if heatmap_range is None:
            heatmap_range = (2.0, max(all_marginal) / 2.0 if all_marginal else 4.0)

    for idx in selected_idx:
        phi_deg = ensemble[idx][4]
        psi_deg = ensemble[idx][5]
        tag     = f"{name}_struct{idx+1:02d}"
        rows    = _per_residue_energy(phi_deg, psi_deg, probs_joint)

        _plot_heatmap(rows, loop_sequence, tag, out,
                      vmin=heatmap_range[0], vmax=heatmap_range[1])
        _plot_landscapes(rows, loop_sequence, probs_joint, tag, out)
        _plot_ramachandran(rows, loop_sequence, tag, out,
                           vmin=rama_range[0], vmax=rama_range[1])


def plot_native_energy(
    phi_deg:       np.ndarray,
    psi_deg:       np.ndarray,
    probs_joint:   list,
    loop_sequence: str,
    name:          str,
    output_dir:    str,
    native_clash:  Optional[dict] = None,
    rama_range:    Tuple[float, float] = None,
    heatmap_range: Tuple[float, float] = None,
):
    """
    Write energy plots for the native loop conformation.

    Produces the same three figures as plot_energy but labelled 'native'.
    If native_clash is provided (dict with keys 'intra', 'framework', 'total'),
    the scores are annotated in the plot titles.
    """
    out = Path(output_dir) / 'energy_plots'
    out.mkdir(parents=True, exist_ok=True)

    tag  = f"{name}_native"
    rows = _per_residue_energy(phi_deg, psi_deg, probs_joint)

    # Annotate clash scores in the tag if available
    clash_str = ""
    if native_clash is not None:
        clash_str = (f"  intra={native_clash['intra']:.2f}"
                     f"  fw={native_clash['framework']:.2f}"
                     f"  total={native_clash['total']:.2f}")
        print(f"    Native clash: intra={native_clash['intra']:.3f}  "
              f"framework={native_clash['framework']:.3f}  "
              f"total={native_clash['total']:.3f}")

    display_tag = tag + clash_str

    # Compute ranges from native alone if not provided
    if rama_range is None or heatmap_range is None:
        all_joint = [r['e_joint'] for r in rows if not math.isnan(r['e_joint'])]
        all_marginal = []
        for r in rows:
            if not math.isnan(r['e_phi']):
                all_marginal.append(r['e_phi'])
            if not math.isnan(r['e_psi']):
                all_marginal.append(r['e_psi'])
        if rama_range is None:
            rama_range = (4.0, max(all_joint) if all_joint else 8.0)
        if heatmap_range is None:
            heatmap_range = (2.0, max(all_marginal) / 2.0 if all_marginal else 4.0)

    _plot_heatmap(rows, loop_sequence, display_tag, out,
                  vmin=heatmap_range[0], vmax=heatmap_range[1])
    _plot_landscapes(rows, loop_sequence, probs_joint, display_tag, out)
    _plot_ramachandran(rows, loop_sequence, display_tag, out,
                       vmin=rama_range[0], vmax=rama_range[1])
    print(f"    Native energy plots → {out}")


def _plot_heatmap(rows, loop_seq, tag, out, vmin=None, vmax=None):
    n    = len(rows)
    mat  = np.array([[r['e_phi'] for r in rows],
                     [r['e_psi'] for r in rows]], dtype=float)
    cmap = LinearSegmentedColormap.from_list(
        'energy', ['#1a6faf', '#74c476', '#fed976', '#e31a1c'], N=256)

    fig, axes = plt.subplots(2, 1, figsize=(max(8, n * 0.7 + 2), 5),
                             gridspec_kw={'height_ratios': [3, 1]})
    fig.suptitle(f"Per-residue energy: {tag}  |  {loop_seq}",
                 fontsize=11, fontweight='bold', y=1.01)

    ax   = axes[0]
    if vmin is None:
        vmin = 2.0
    if vmax is None:
        vmax = max(vmin + 0.1, float(np.nanmax(mat)) / 2.0)
    im   = ax.imshow(mat, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax,
                     interpolation='nearest')
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['φ marginal', 'ψ marginal'], fontsize=9)
    ax.set_xticks(range(n))
    ax.set_xticklabels(
        [f"{loop_seq[i]}\n{i+1}" if i < len(loop_seq) else str(i+1)
         for i in range(n)], fontsize=8)
    ax.set_xlabel('Residue', fontsize=9)
    for ri in range(2):
        for ci in range(n):
            val = mat[ri, ci]
            if not math.isnan(val):
                tc = 'white' if val > vmax * 0.6 else 'black'
                ax.text(ci, ri, f"{val:.2f}", ha='center', va='center',
                        fontsize=7, color=tc, fontweight='bold')
    plt.colorbar(im, ax=ax, label='−log p (marginal)', shrink=0.8, pad=0.02)

    ax2       = axes[1]
    e_phi_arr = [r['e_phi']   for r in rows]
    e_psi_arr = [r['e_psi']   for r in rows]
    e_jnt_arr = [r['e_joint'] for r in rows]
    x = np.arange(n)
    ax2.bar(x, e_phi_arr, color='#4292c6', label='φ marginal',
            width=0.35, align='edge')
    ax2.bar(x + 0.35, e_psi_arr, color='#ef6548', label='ψ marginal',
            width=0.35, align='edge')
    ax2.plot(x, e_jnt_arr, 'k--o', markersize=4, linewidth=1,
             label='joint E')
    ax2.axhline(np.nanmean(e_jnt_arr), color='grey', linestyle=':',
                linewidth=0.8, label=f"mean {np.nanmean(e_jnt_arr):.2f}")
    ax2.set_xticks(x)
    ax2.set_xticklabels([str(i+1) for i in range(n)], fontsize=8)
    ax2.set_ylabel('Energy (−log p)', fontsize=8)
    ax2.legend(fontsize=7, ncol=4, loc='upper right')

    plt.tight_layout()
    fig.savefig(out / f"heatmap_{tag}.png", dpi=150, bbox_inches='tight')
    plt.close(fig)


def _plot_landscapes(rows, loop_seq, probs_joint, tag, out):
    n          = len(rows)
    cols       = min(n, 5)
    n_rows_plt = math.ceil(n / cols)

    fig = plt.figure(figsize=(cols * 4.0, n_rows_plt * 3.8))
    fig.suptitle(f"Energy landscape: {tag}  |  {loop_seq}",
                 fontsize=11, fontweight='bold')

    for idx, r in enumerate(rows):
        aa = loop_seq[r['idx']] if r['idx'] < len(loop_seq) else '?'
        pj = probs_joint[r['idx']] if r['idx'] < len(probs_joint) else None
        p_phi, p_psi = _marginals(pj) if pj is not None else (None, None)

        # φ
        ax = fig.add_subplot(n_rows_plt * 2, cols, idx + 1)
        if p_phi is not None:
            nll = -np.log(p_phi + 1e-10)
            ax.fill_between(_BIN_CENTRES, nll, alpha=0.35, color='#4292c6')
            ax.plot(_BIN_CENTRES, nll, color='#4292c6', linewidth=1.0)
            ax.axvline(r['phi'], color='#e31a1c', linewidth=1.8,
                       label=f"φ={r['phi']:.0f}°")
            ymax = float(np.nanmax(nll))
            ax.plot(r['phi'], ymax * 0.92, 'r*', markersize=12, zorder=5,
                    clip_on=False)
            ax.text(r['phi'], ymax * 0.75, f"E={r['e_phi']:.2f}",
                    fontsize=6, color='#e31a1c', ha='center')
        ax.set_xlim(-180, 180)
        ax.set_xticks([-180, -90, 0, 90, 180])
        ax.tick_params(labelsize=6)
        ax.set_title(f"Res {r['idx']+1} ({aa})  "
                     f"E_φ={r['e_phi']:.2f}  E_joint={r['e_joint']:.2f}",
                     fontsize=7.5, pad=2)
        ax.set_ylabel('−log p(φ) marg.', fontsize=6)
        if idx == 0:
            ax.legend(fontsize=6, loc='upper left')

        # ψ
        ax2 = fig.add_subplot(n_rows_plt * 2, cols, idx + 1 + cols * n_rows_plt)
        if p_psi is not None:
            nll = -np.log(p_psi + 1e-10)
            ax2.fill_between(_BIN_CENTRES, nll, alpha=0.35, color='#ef6548')
            ax2.plot(_BIN_CENTRES, nll, color='#ef6548', linewidth=1.0)
            ax2.axvline(r['psi'], color='#2ca25f', linewidth=1.8,
                        label=f"ψ={r['psi']:.0f}°")
            ymax = float(np.nanmax(nll))
            ax2.plot(r['psi'], ymax * 0.92, 'g*', markersize=12, zorder=5,
                     clip_on=False)
            ax2.text(r['psi'], ymax * 0.75, f"E={r['e_psi']:.2f}",
                     fontsize=6, color='#2ca25f', ha='center')
        ax2.set_xlim(-180, 180)
        ax2.set_xticks([-180, -90, 0, 90, 180])
        ax2.tick_params(labelsize=6)
        ax2.set_title(f"Res {r['idx']+1} ({aa})  E_ψ={r['e_psi']:.2f}",
                      fontsize=7.5, pad=2)
        ax2.set_ylabel('−log p(ψ) marg.', fontsize=6)
        ax2.set_xlabel('angle (°)', fontsize=6)
        if idx == 0:
            ax2.legend(fontsize=6, loc='upper left')

    plt.tight_layout()
    fig.savefig(out / f"landscape_{tag}.png", dpi=150, bbox_inches='tight')
    plt.close(fig)


def _plot_ramachandran(rows, loop_seq, tag, out, vmin=None, vmax=None):
    from collections import Counter

    valid = [(r['phi'], r['psi'], r['e_joint'],
              f"{loop_seq[r['idx']] if r['idx'] < len(loop_seq) else '?'}"
              f"{r['idx']+1}",
              loop_seq[r['idx']] if r['idx'] < len(loop_seq) else '?')
             for r in rows if not math.isnan(r['e_joint'])]
    if not valid:
        return

    phi_v, psi_v, e_v, lbl_v, aa_v = zip(*valid)

    if vmin is None:
        vmin = 4.0
    if vmax is None:
        vmax = max(vmin + 0.1, max(e_v))

    # Deterministic jitter — separates overlapping points without
    # introducing random variation between runs
    jitter_phi = np.array([(hash(lbl) % 100 - 50) * 0.05 for lbl in lbl_v])
    jitter_psi = np.array([(hash(lbl+'_') % 100 - 50) * 0.05 for lbl in lbl_v])
    phi_plot   = np.array(phi_v) + jitter_phi
    psi_plot   = np.array(psi_v) + jitter_psi

    cmap = LinearSegmentedColormap.from_list(
        'energy', ['#1a9850', '#ffffbf', '#d73027'], N=256)

    fig, ax = plt.subplots(figsize=(6, 5.5))
    sc = ax.scatter(phi_plot, psi_plot, c=e_v, cmap=cmap, s=80,
                    edgecolors='black', linewidths=0.5, zorder=3, alpha=0.85,
                    vmin=vmin, vmax=vmax)

    for phi, psi, lbl, aa in zip(phi_plot, psi_plot, lbl_v, aa_v):
        color = 'blue' if aa == 'G' else 'purple' if aa == 'P' else 'black'
        weight = 'bold' if aa in ('G', 'P') else 'normal'
        ax.annotate(lbl, (phi, psi), textcoords='offset points',
                    xytext=(4, 4), fontsize=7, color=color, fontweight=weight)

    # Annotate bins with multiple occupants
    bin_counts = Counter(
        (round(p / 10) * 10, round(s / 10) * 10)
        for p, s in zip(phi_v, psi_v)
    )
    for (pb, sb), count in bin_counts.items():
        if count > 1:
            ax.text(pb, sb + 9, f'×{count}', fontsize=7,
                    color='grey', ha='center', style='italic')

    plt.colorbar(sc, ax=ax, label='Joint E  (−log P(φ,ψ))', shrink=0.85)
    ax.axhline(0, color='grey', lw=0.5, ls='--', alpha=0.4)
    ax.axvline(0, color='grey', lw=0.5, ls='--', alpha=0.4)
    ax.set_xlim(-180, 180);  ax.set_ylim(-180, 180)
    ax.set_xlabel('φ (°)', fontsize=10);  ax.set_ylabel('ψ (°)', fontsize=10)
    ax.set_title(f"Ramachandran (joint energy)\n{tag}  |  {loop_seq}",
                 fontsize=10)
    ax.set_xticks(range(-180, 181, 60));  ax.set_yticks(range(-180, 181, 60))
    plt.tight_layout()
    fig.savefig(out / f"ramachandran_{tag}.png", dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_summary(
    results:    list,
    output_dir: str,
):
    """
    Ensemble-level summary figure:
      - RMSD distribution histogram
      - Optimised vs baseline scatter
      - Cumulative success rate
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    best_anc  = [r['best_rmsd']             for r in results]
    best_aln  = [r['best_rmsd_aln']         for r in results]
    best_uni  = [r['baseline_uniform_best'] for r in results]
    best_msmp = [r['baseline_model_best']   for r in results]

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    all_vals = best_anc + best_aln + best_uni + best_msmp
    bins = np.linspace(0, max(all_vals) * 1.05, 18)
    for vals, label, color in [
        (best_anc,  f'Optimised anchored (mean {np.mean(best_anc):.2f}Å)',  'steelblue'),
        (best_aln,  f'Optimised aligned  (mean {np.mean(best_aln):.2f}Å)',  'royalblue'),
        (best_uni,  f'Random uniform     (mean {np.mean(best_uni):.2f}Å)',  'lightcoral'),
        (best_msmp, f'Model sample       (mean {np.mean(best_msmp):.2f}Å)', 'tomato'),
    ]:
        axes[0].hist(vals, bins=bins, alpha=0.5, edgecolor='black',
                     label=label, color=color)
    axes[0].set_xlabel('Best RMSD (Å)');  axes[0].set_title('RMSD Distribution')
    axes[0].legend(fontsize=7)

    for vals, label, color in [
        (best_uni,  'vs uniform',      'lightcoral'),
        (best_msmp, 'vs model sample', 'tomato'),
        (best_aln,  'vs aligned',      'royalblue'),
    ]:
        axes[1].scatter(best_anc, vals, alpha=0.8, s=55, color=color,
                        edgecolors='white', linewidths=0.4, label=label)
    lim = max(max(best_anc), max(best_uni), max(best_msmp), max(best_aln)) * 1.05
    axes[1].plot([0, lim], [0, lim], 'k--', linewidth=0.8, alpha=0.4, label='y=x')
    axes[1].set_xlabel('Optimised anchored RMSD (Å)')
    axes[1].set_ylabel('Comparison RMSD (Å)')
    axes[1].set_title('Optimised vs baselines\n(below diagonal = optimised wins)')
    axes[1].legend(fontsize=7)

    for vals, label, color, ls in [
        (best_anc,  'Optimised anchored', 'steelblue',  '-'),
        (best_aln,  'Optimised aligned',  'royalblue',  '--'),
        (best_uni,  'Random uniform',     'lightcoral', '-.'),
        (best_msmp, 'Model sample',       'tomato',     ':'),
    ]:
        s = sorted(vals)
        c = np.arange(1, len(s)+1) / len(s) * 100
        axes[2].plot(s, c, color=color, linestyle=ls, linewidth=2, label=label)
    for t, c in [(1, 'red'), (2, 'orange'), (3, 'green')]:
        axes[2].axvline(t, color=c, linestyle='--', alpha=0.3, linewidth=0.8)
    axes[2].set_xlabel('RMSD (Å)');  axes[2].set_ylabel('Success (%)')
    axes[2].set_title('Cumulative success rate');  axes[2].legend(fontsize=7)

    plt.tight_layout()
    fig.savefig(out / 'summary.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"    Summary plot → {out / 'summary.png'}")

    # ─────────────────────────────────────────────────────────────────────────────
# 7.  Energy model inference helpers
#     (previously in loop_modeling_nerf.py / test_on_cdr3.py)
# ─────────────────────────────────────────────────────────────────────────────

# Bin layout: 36 bins spanning [-180, 180), 10 degrees each
_N_BINS      = 36
_BIN_WIDTH_E = 360.0 / _N_BINS

_AA_TO_IDX = {
    'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4,
    'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9,
    'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14,
    'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19,
}
_PAD_IDX     = 20
_MAX_CONTEXT = 3
_CONTEXT_RAD = _MAX_CONTEXT // 2


def cache_energy_distributions(model_or_router, sequence: str) -> list:
    """
    Predict per-residue joint (phi, psi) probability tables for `sequence`.

    Returns a list of len(sequence) arrays, each (N_BINS, N_BINS), where
    N_BINS = 36 (10 deg/bin, [-180, 180)).

    Args:
        model_or_router: ModelRouter or plain (model, params) pair.
        sequence:        One-letter amino acid string for the loop.

    Returns:
        List of (36, 36) float32 numpy arrays (softmax probabilities).
    """
    import jax
    import jax.numpy as jnp

    router  = _to_router(model_or_router)
    encoded = np.array([_AA_TO_IDX.get(aa, _PAD_IDX) for aa in sequence.upper()])
    n       = len(encoded)
    probs   = []

    print(f"      Caching distributions for {n} residues...")
    for i in range(n):
        aa             = sequence[i].upper()
        model, mparams = router.get(aa)
        window = [int(encoded[pos]) if 0 <= pos < n else _PAD_IDX
                  for pos in range(i - _CONTEXT_RAD, i + _CONTEXT_RAD + 1)]
        logits = model.apply(
            {'params': mparams},
            jnp.array(window)[None, :],
            jnp.ones((1, _MAX_CONTEXT), dtype=bool),
            training=False,
            rngs={'dropout': jax.random.PRNGKey(0)},
        )
        probs.append(
            np.array(jax.nn.softmax(logits[0])).reshape(_N_BINS, _N_BINS)
        )

    print(f"      Cached {n} joint ({_N_BINS}x{_N_BINS}) distributions")
    return probs


def coords_to_angles(
    N:  np.ndarray,   # (n_res, 3)
    CA: np.ndarray,
    C:  np.ndarray,
) -> tuple:
    """
    Compute phi/psi torsion angles in degrees from backbone coordinates.
    Returns (phi, psi) each (n_res,) with NaN at termini where undefined.
    """
    n   = len(N)
    phi = np.full(n, np.nan)
    psi = np.full(n, np.nan)
    for i in range(n):
        if i > 0:
            b0  = C[i-1] - N[i];   b1 = CA[i] - N[i];  b2 = C[i] - CA[i]
            b1h = b1 / (np.linalg.norm(b1) + 1e-10)
            v   = b0 - np.dot(b0, b1h) * b1h
            w   = b2 - np.dot(b2, b1h) * b1h
            phi[i] = np.degrees(np.arctan2(
                np.dot(np.cross(b1h, v), w), np.dot(v, w)
            ))
        if i < n - 1:
            b0  = CA[i] - N[i];    b1 = C[i] - CA[i];  b2 = N[i+1] - C[i]
            b1h = b1 / (np.linalg.norm(b1) + 1e-10)
            v   = b0 - np.dot(b0, b1h) * b1h
            w   = b2 - np.dot(b2, b1h) * b1h
            psi[i] = np.degrees(np.arctan2(
                np.dot(np.cross(b1h, v), w), np.dot(v, w)
            ))
    return phi, psi