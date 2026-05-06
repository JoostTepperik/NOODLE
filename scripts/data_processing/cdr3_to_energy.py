"""
cdr3_cif_energy_pipeline.py  (fully self-contained)
─────────────────────────────────────────────────────
Extract CDR3 loops from a flat directory of STCRDab mmCIF files and plot
per-residue energy from the MLP torsion angle predictor.

Input:
    <cif_dir>/<pdbid>.cif   (flat directory, e.g. 1a1m.cif)

CDR3 extraction — two-tier:
    1. ANARCI / ImmunoPDB  (primary, when available)
       Applies IMGT numbering to each candidate TCR chain, then extracts
       residues 105–117 deterministically.
       Requires: pdb_selchain, pdb_reres (pdb-tools) and ImmunoPDB.py.
       Set IMMUNOPDB_PATH below to match your installation.
    2. Sequence heuristic  (automatic fallback)
       Finds Cys104 … Phe/Trp118 via the conserved FR4 G-x-G motif.

Handles:
  - TCR-only, TCR-pMHC, pMHC-only structures
  - Multiple CDR3s per file (alpha + beta chains)
  - NCS duplicates (deduplicates identical sequences within a structure)
  - Old-style CIFs with semicolon-delimited multiline entity descriptions

Bin convention (matches training data):
  72 bins spanning [-180°, 180°),  bin k → -180 + (k+0.5)*5°

Usage:
    # Alpha CDR3 from full database (ANARCI if available):
    python cdr3_cif_energy_pipeline.py

    # Beta CDR3:
    python cdr3_cif_energy_pipeline.py --chain_type beta

    # Both chains, quick test:
    python cdr3_cif_energy_pipeline.py --chain_type both --max_structures 10

    # Force sequence heuristic only:
    python cdr3_cif_energy_pipeline.py --no_anarci

    # Custom directory:
    python cdr3_cif_energy_pipeline.py \\
        --cif_dir /path/to/cifs --output ./my_results
"""

import sys
sys.path.append('/home/jtepperik/thesis/energy_model/scripts')

import argparse
import json
import math
import re
import shutil
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp

from models.full_model import TorsionPredictor


# ─────────────────────────────────────────────────────────────────────────────
# Constants  (single source of truth — never duplicated)
# ─────────────────────────────────────────────────────────────────────────────

N_BINS      = 72
BIN_WIDTH   = 360.0 / N_BINS                            # 5°
BIN_CENTRES = np.array([-180.0 + (k + 0.5) * BIN_WIDTH for k in range(N_BINS)])

MAX_CONTEXT = 7
CONTEXT_RAD = MAX_CONTEXT // 2    # 3

AA_TO_IDX = {
    'A':0,'R':1,'N':2,'D':3,'C':4,'Q':5,'E':6,'G':7,'H':8,'I':9,
    'L':10,'K':11,'M':12,'F':13,'P':14,'S':15,'T':16,'W':17,'Y':18,'V':19,
}
PAD_IDX = 20

THREE_TO_ONE = {
    'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C',
    'GLN':'Q','GLU':'E','GLY':'G','HIS':'H','ILE':'I',
    'LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P',
    'SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V',
    # common non-standard
    'MSE':'M','HSD':'H','HSE':'H','HSP':'H','HIE':'H','HID':'H','HIP':'H',
    'CSE':'C','SEC':'C',
}

# IMGT CDR3 heuristic parameters
CDR3_MIN_LEN  = 5
CDR3_MAX_LEN  = 25
# Note: chain length is NOT used as a filter — STCRDab CIFs may contain
# full V+C domains or split domains. The FR4 G-x-G motif is the reliable anchor.

CKPT_PATH = (
    '/home/jtepperik/thesis/energy_model/scripts/training/outputs/'
    'feedforward_binned_19448143/checkpoints/best_10'
)

IMMUNOPDB_PATH = (
    Path.home() / 'home' / 'jtepperik' / 'swifttcr' / 'tools' / 'ANARCI_master' / 'Example_scripts_and_sequences' / 'ImmunoPDB.py'
)


# IMGT CDR3 body: 105–117 inclusive (excludes anchoring Cys104 and FR4 Phe/Trp118)
IMGT_CDR3_START = 105
IMGT_CDR3_END   = 117
DIHEDRAL_FLANK  = 3   # residues either side of loop for accurate terminal phi/psi


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────

def load_model():
    model = TorsionPredictor(
        max_context=MAX_CONTEXT, embed_dim=64, hidden_dim=768,
        n_layers=3, dropout_rate=0.1,
        prediction_type='binned', n_bins=N_BINS,
    )
    checkpointer = ocp.StandardCheckpointer()
    restored = checkpointer.restore(CKPT_PATH)
    params = restored['params']
    stds = [float(np.std(l)) for l in jax.tree_util.tree_leaves(params)]
    print(f"✓ Model loaded  (weight std range: {min(stds):.4f}–{max(stds):.4f})")
    return model, params


def predict_distributions(model, params, sequence: str):
    """
    Sliding-window prediction for each residue in `sequence`.

    Returns:
        probs_phi, probs_psi — lists of (72,) np.ndarray in [-180, 180) bin order
    """
    encoded = np.array([AA_TO_IDX.get(aa, PAD_IDX) for aa in sequence.upper()])
    seq_len = len(encoded)
    probs_phi, probs_psi = [], []

    for i in range(seq_len):
        window = []
        for pos in range(i - CONTEXT_RAD, i + CONTEXT_RAD + 1):
            window.append(int(encoded[pos]) if 0 <= pos < seq_len else PAD_IDX)

        batch_res  = jnp.array(window)[None, :]
        batch_mask = jnp.ones((1, MAX_CONTEXT), dtype=bool)

        logits_phi, logits_psi = model.apply(
            {'params': params}, batch_res, batch_mask,
            training=False, rngs={'dropout': jax.random.PRNGKey(0)},
        )

        probs_phi.append(np.array(jax.nn.softmax(logits_phi[0])))
        probs_psi.append(np.array(jax.nn.softmax(logits_psi[0])))

    return probs_phi, probs_psi


# ─────────────────────────────────────────────────────────────────────────────
# mmCIF parser
# ─────────────────────────────────────────────────────────────────────────────

def _tokenise(row: str):
    """Tokenise a single mmCIF data row (handles single- and double-quoted strings).
    Note: semicolon-delimited multiline strings are handled upstream before
    rows reach this function."""
    tokens, i = [], 0
    while i < len(row):
        if row[i] in (' ', '\t'):
            i += 1
        elif row[i] == "'":
            j = row.find("'", i + 1)
            tokens.append(row[i+1:j] if j != -1 else row[i+1:])
            i = j + 1 if j != -1 else len(row)
        elif row[i] == '"':
            j = row.find('"', i + 1)
            tokens.append(row[i+1:j] if j != -1 else row[i+1:])
            i = j + 1 if j != -1 else len(row)
        else:
            j = i
            while j < len(row) and row[j] not in (' ', '\t'):
                j += 1
            tokens.append(row[i:j])
            i = j
    return tokens


def _collapse_semicolon_strings(lines: list) -> list:
    """
    Pre-process CIF lines to collapse semicolon-delimited multiline strings
    (;...\\n...\\n;) into a single quoted token on one line.
    This is needed for older PDB CIFs where _entity.pdbx_description spans
    multiple lines between semicolons.
    """
    out = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith(';'):
            # Start of a semicolon-quoted string — collect until closing ;
            parts = [line[1:].rstrip()]
            i += 1
            while i < len(lines):
                if lines[i].startswith(';'):
                    i += 1
                    break
                parts.append(lines[i].rstrip())
                i += 1
            combined = ' '.join(p.strip() for p in parts if p.strip())
            out.append(f"'{combined}'\n")
        else:
            out.append(line)
            i += 1
    return out


def parse_mmcif(cif_path: str) -> dict:
    """
    Parse an mmCIF file.

    Returns:
        dict: chain_id -> list of residue dicts (sorted by author residue number)
        Each residue dict:
            { 'resname': str,
              'auth_resnum': int,
              'ins_code': str,
              'atoms': { atom_name: np.ndarray([x, y, z]) } }
    """
    with open(cif_path) as fh:
        raw = fh.readlines()
    lines = _collapse_semicolon_strings(raw)

    col_headers = []
    atom_rows   = []
    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        if stripped == 'loop_':
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            if j < len(lines) and lines[j].strip().startswith('_atom_site.'):
                col_headers = []
                i = j
                while i < len(lines) and lines[i].strip().startswith('_atom_site.'):
                    col_headers.append(lines[i].strip().split('.')[1].rstrip())
                    i += 1
                # read data rows
                while i < len(lines):
                    row = lines[i].strip()
                    if not row or row.startswith('loop_') or \
                       row.startswith('_') or row.startswith('#'):
                        break
                    atom_rows.append(row)
                    i += 1
                continue
        i += 1

    if not col_headers:
        # Fallback to space-delimited ATOM lines (older CIFs / PDB-converted)
        return _parse_atom_lines(lines)

    def col(name):
        try:    return col_headers.index(name)
        except ValueError: return None

    c_group      = col('group_PDB')
    c_label_atom = col('label_atom_id')
    c_auth_atom  = col('auth_atom_id')
    c_label_comp = col('label_comp_id')
    c_auth_comp  = col('auth_comp_id')
    c_label_chain= col('label_asym_id')
    c_auth_chain = col('auth_asym_id')
    c_auth_seq   = col('auth_seq_id')
    c_label_seq  = col('label_seq_id')
    c_ins        = col('pdbx_PDB_ins_code')
    c_x          = col('Cartn_x')
    c_y          = col('Cartn_y')
    c_z          = col('Cartn_z')
    c_alt        = col('label_alt_id')

    atom_col  = c_auth_atom  if c_auth_atom  is not None else c_label_atom
    comp_col  = c_auth_comp  if c_auth_comp  is not None else c_label_comp
    chain_col = c_auth_chain if c_auth_chain is not None else c_label_chain
    seq_col   = c_auth_seq   if c_auth_seq   is not None else c_label_seq

    chains = defaultdict(dict)

    for row in atom_rows:
        toks = _tokenise(row)
        needed = max(filter(None, [c_x, c_y, c_z, atom_col, comp_col,
                                    chain_col, seq_col])) + 1
        if len(toks) < needed:
            continue
        if c_group is not None and toks[c_group] != 'ATOM':
            continue
        if c_alt is not None and toks[c_alt] not in ('.', '?', '', 'A'):
            continue

        resname = toks[comp_col]
        if resname not in THREE_TO_ONE:
            continue

        chain_id  = toks[chain_col]
        atom_name = toks[atom_col]
        ins_code  = ''
        if c_ins is not None and toks[c_ins] not in ('.', '?'):
            ins_code = toks[c_ins]

        try:
            auth_resnum = int(toks[seq_col])
            x = float(toks[c_x])
            y = float(toks[c_y])
            z = float(toks[c_z])
        except (ValueError, IndexError):
            continue

        key = (auth_resnum, ins_code)
        if key not in chains[chain_id]:
            chains[chain_id][key] = {
                'resname': resname,
                'auth_resnum': auth_resnum,
                'ins_code': ins_code,
                'atoms': {}
            }
        chains[chain_id][key]['atoms'][atom_name] = np.array([x, y, z])

    result = {}
    for chain_id, res_dict in chains.items():
        sorted_res = sorted(res_dict.values(),
                            key=lambda r: (r['auth_resnum'], r['ins_code']))
        if sorted_res:
            result[chain_id] = sorted_res
    return result


def _parse_atom_lines(lines):
    """
    Fallback parser for CIFs that look like PDB ATOM records or
    space-separated ATOM lines without a proper loop_ header block.
    Handles the format shown in the database:
      ATOM   2574 N N   . LEU B 2 40  ? -6.980 59.946 67.805 ...
    """
    chains = defaultdict(dict)
    for line in lines:
        if not line.startswith('ATOM'):
            continue
        parts = line.split()
        if len(parts) < 11:
            continue
        try:
            # Heuristic column detection: find first float-like coordinate triple
            coord_start = None
            for ci in range(4, len(parts) - 2):
                try:
                    float(parts[ci]); float(parts[ci+1]); float(parts[ci+2])
                    if '.' in parts[ci]:
                        coord_start = ci
                        break
                except ValueError:
                    continue
            if coord_start is None:
                continue

            x = float(parts[coord_start])
            y = float(parts[coord_start + 1])
            z = float(parts[coord_start + 2])

            # atom_name: second token after 'ATOM serial'
            atom_name = parts[2]
            # resname: look for 3-letter AA near the coordinate columns
            resname = None
            for ci in range(2, coord_start):
                if parts[ci] in THREE_TO_ONE:
                    resname = parts[ci]
            if resname is None:
                continue

            # chain: single letter before resname
            chain_id = None
            for ci in range(2, coord_start):
                if parts[ci] == resname and ci > 0 and len(parts[ci-1]) == 1 \
                        and parts[ci-1].isalpha():
                    chain_id = parts[ci-1]
                    break
            if chain_id is None:
                continue

            # resnum: integer after chain
            auth_resnum = None
            for ci in range(2, coord_start):
                if parts[ci] == resname:
                    # look for int after chain letter
                    for cj in range(ci, coord_start):
                        try:
                            auth_resnum = int(parts[cj])
                            break
                        except ValueError:
                            continue
                    break
            if auth_resnum is None:
                continue

        except (ValueError, IndexError):
            continue

        key = (auth_resnum, '')
        if key not in chains[chain_id]:
            chains[chain_id][key] = {
                'resname': resname,
                'auth_resnum': auth_resnum,
                'ins_code': '',
                'atoms': {}
            }
        chains[chain_id][key]['atoms'][atom_name] = np.array([x, y, z])

    result = {}
    for chain_id, res_dict in chains.items():
        sorted_res = sorted(res_dict.values(),
                            key=lambda r: (r['auth_resnum'], r['ins_code']))
        if sorted_res:
            result[chain_id] = sorted_res
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Chain type detection  (reads _entity / _struct_asym from CIF header)
# ─────────────────────────────────────────────────────────────────────────────

def parse_chain_types(cif_path: str) -> dict:
    """
    Read chain entity descriptions from the CIF header.

    Returns:
        dict: auth_chain_id -> chain_type str, one of:
              'alpha', 'beta', 'mhc', 'peptide', 'b2m', 'unknown'

    Method:
        1. Parse _entity.id + _entity.pdbx_description (key-value pairs)
        2. Parse _struct_asym loop: asym_id (label chain) + entity_id
        3. Parse _atom_site to map label chain -> auth chain
        Then classify each auth chain by its entity description.
    """
    with open(cif_path) as fh:
        raw = fh.readlines()
    lines = _collapse_semicolon_strings(raw)

    # ── 1. Entity descriptions ────────────────────────────────────────────────
    # Handles both loop_ style and key-value style (multiple sequential blocks).
    entity_desc = {}   # entity_id (str) -> description (str)

    def _best_desc(d1, d2):
        """Return whichever description is more informative (longer)."""
        return d1 if len(d1) >= len(d2) else d2

    i = 0
    while i < len(lines):
        s = lines[i].strip()

        # ── Loop style ──
        if s == 'loop_':
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            if j < len(lines) and lines[j].strip().startswith('_entity.'):
                hdrs = []
                k = j
                while k < len(lines) and lines[k].strip().startswith('_entity.'):
                    hdrs.append(lines[k].strip().split('.')[1].rstrip())
                    k += 1
                # Accept either pdbx_description or details
                ci_id   = hdrs.index('id') if 'id' in hdrs else None
                ci_desc = None
                for col_name in ('pdbx_description', 'details'):
                    if col_name in hdrs:
                        ci_desc = hdrs.index(col_name)
                        break
                if ci_id is not None and ci_desc is not None:
                    while k < len(lines):
                        row = lines[k].strip()
                        if not row or row.startswith('_') or row.startswith('loop_') \
                                or row.startswith('#'):
                            break
                        toks = _tokenise(row)
                        if len(toks) > max(ci_id, ci_desc):
                            eid  = toks[ci_id]
                            desc = toks[ci_desc].strip("'\"")
                            entity_desc[eid] = _best_desc(
                                entity_desc.get(eid, ''), desc)
                        k += 1
            i += 1
            continue

        # ── Key-value style: scan sequential _entity.* blocks ──
        # Each entity appears as a block of key-value pairs separated by blank lines
        # or just runs continuously. We accumulate id/desc pairs as we scan.
        if s.startswith('_entity.id'):
            parts = s.split(None, 1)
            eid = parts[1].strip() if len(parts) > 1 else lines[i+1].strip()
            # Search nearby lines for description
            for di in range(i + 1, min(i + 15, len(lines))):
                ds = lines[di].strip()
                if ds.startswith('_entity.') and not ds.startswith('_entity.id'):
                    if 'pdbx_description' in ds or 'details' in ds:
                        dp = ds.split(None, 1)
                        if len(dp) > 1:
                            desc = dp[1].strip().strip("'\"")
                        elif di + 1 < len(lines):
                            desc = lines[di+1].strip().strip("'\"")
                        else:
                            desc = ''
                        if desc and desc not in ('.', '?'):
                            entity_desc[eid] = _best_desc(
                                entity_desc.get(eid, ''), desc)
                elif ds.startswith('_entity.id') or ds == 'loop_':
                    break
        i += 1

    # ── 2. struct_asym: label_asym_id -> entity_id ────────────────────────────
    label_to_entity = {}   # label chain -> entity_id
    i = 0
    while i < len(lines):
        s = lines[i].strip()
        if s == 'loop_':
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            if j < len(lines) and lines[j].strip().startswith('_struct_asym.'):
                hdrs = []
                k = j
                while k < len(lines) and lines[k].strip().startswith('_struct_asym.'):
                    hdrs.append(lines[k].strip().split('.')[1].rstrip())
                    k += 1
                if 'id' in hdrs and 'entity_id' in hdrs:
                    ci_id  = hdrs.index('id')
                    ci_ent = hdrs.index('entity_id')
                    while k < len(lines):
                        row = lines[k].strip()
                        if not row or row.startswith('_') or row.startswith('loop_') \
                                or row.startswith('#'):
                            break
                        toks = _tokenise(row)
                        if len(toks) > max(ci_id, ci_ent):
                            label_to_entity[toks[ci_id]] = toks[ci_ent]
                        k += 1
        i += 1

    # ── 3. atom_site: label_asym_id -> auth_asym_id ───────────────────────────
    label_to_auth = {}
    i = 0
    while i < len(lines):
        s = lines[i].strip()
        if s == 'loop_':
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            if j < len(lines) and lines[j].strip().startswith('_atom_site.'):
                hdrs = []
                k = j
                while k < len(lines) and lines[k].strip().startswith('_atom_site.'):
                    hdrs.append(lines[k].strip().split('.')[1].rstrip())
                    k += 1
                cl = None; ca = None
                try: cl = hdrs.index('label_asym_id')
                except ValueError: pass
                try: ca = hdrs.index('auth_asym_id')
                except ValueError: pass
                if cl is not None and ca is not None:
                    while k < len(lines):
                        row = lines[k].strip()
                        if not row or row.startswith('loop_') or \
                           row.startswith('_') or row.startswith('#'):
                            break
                        toks = _tokenise(row)
                        if len(toks) > max(cl, ca):
                            label_to_auth[toks[cl]] = toks[ca]
                        k += 1
                break   # only need first atom_site block
        i += 1

    # ── 4. Classify ───────────────────────────────────────────────────────────
    def _classify_desc(desc: str) -> str:
        d = desc.lower()
        if any(x in d for x in ('alpha chain', 'tcr alpha', 'tcra', 't-cell receptor alpha',
                                  'trav', 'traj')):
            return 'alpha'
        if any(x in d for x in ('beta chain', 'tcr beta', 'tcrb', 't-cell receptor beta',
                                  'trbv', 'trbj')):
            return 'beta'
        if any(x in d for x in ('beta-2-microglobulin', 'beta 2 microglobulin', 'b2m', 'b2-m')):
            return 'b2m'
        if any(x in d for x in ('mhc', 'hla', 'h-2', 'h2-', 'major histocompatibility',
                                  'class i', 'class ii', 'class 1', 'class 2')):
            return 'mhc'
        if any(x in d for x in ('peptide', 'antigen', 'epitope')):
            return 'peptide'
        return 'unknown'

    # Build auth_chain -> type map
    auth_chain_type = {}
    for label_chain, entity_id in label_to_entity.items():
        auth_chain = label_to_auth.get(label_chain, label_chain)
        desc = entity_desc.get(entity_id, '')
        auth_chain_type[auth_chain] = _classify_desc(desc)

    return auth_chain_type


# ─────────────────────────────────────────────────────────────────────────────
# CDR3 identification  (IMGT heuristic, no external library)
# ─────────────────────────────────────────────────────────────────────────────

def _chain_seq(residues):
    return ''.join(THREE_TO_ONE.get(r['resname'], 'X') for r in residues)


# ─────────────────────────────────────────────────────────────────────────────
# ANARCI / ImmunoPDB IMGT renumbering
# ─────────────────────────────────────────────────────────────────────────────

def anarci_available() -> bool:
    return (IMMUNOPDB_PATH.exists()
            and shutil.which('pdb_selchain') is not None
            and shutil.which('pdb_reres') is not None)


def _residues_to_pdb_string(residues: list, chain_id: str = 'A') -> str:
    """
    Serialise a list of residue dicts to a minimal PDB ATOM string.
    Uses a fixed chain_id so ImmunoPDB sees a clean single-chain input.
    """
    ONE_TO_THREE = {v: k for k, v in THREE_TO_ONE.items() if len(k) == 3}
    lines = []
    atom_serial = 1
    for res in residues:
        resname = res['resname']
        resnum  = res['auth_resnum']
        icode   = res.get('ins_code', '') or ' '
        for atom_name, xyz in res['atoms'].items():
            # PDB fixed-width format
            an = atom_name.ljust(4) if len(atom_name) < 4 else atom_name[:4]
            lines.append(
                f"ATOM  {atom_serial:5d} {an:<4s} {resname:3s} {chain_id}"
                f"{resnum:4d}{icode:1s}   "
                f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}"
                f"  1.00  0.00           {atom_name[0]:>2s}\n"
            )
            atom_serial += 1
    lines.append("END\n")
    return ''.join(lines)


def _parse_imgt_pdb(pdb_text: str, chain_id: str = 'A') -> dict:
    """
    Read a PDB string produced by ImmunoPDB and return a mapping:
        (orig_resnum, ins_code) -> imgt_resnum

    ImmunoPDB shifts all residue numbers by +500 before renumbering, so the
    IMGT numbers in the output are the true IMGT positions.
    """
    imgt_map = {}
    for line in pdb_text.splitlines():
        if not line.startswith('ATOM') or len(line) < 27:
            continue
        if line[21] != chain_id:
            continue
        try:
            imgt_resnum = int(line[22:26])
        except ValueError:
            continue
        # ImmunoPDB inserts the original (shifted) number in the B-factor or
        # occupancy column — we don't need it; we just record each IMGT number
        # once per unique residue position.
        icode = line[26].strip()
        imgt_map[(imgt_resnum, icode)] = imgt_resnum
    return imgt_map


def imgt_number_chain(residues: list, tmp_dir: Path) -> dict | None:
    """
    Apply IMGT numbering to a single TCR chain via ImmunoPDB.py.

    Workflow (mirrors apply_anarci_numbering.py):
        1. Write residues as PDB to a temp file using chain 'A'
        2. pdb_reres -1 to reset numbering from 1 (avoids clash issues)
        3. ImmunoPDB -s imgt --receptor tr
        4. Parse the output to get IMGT residue numbers

    Returns a list of residue dicts with an added 'imgt_resnum' key,
    or None if ANARCI fails.

    The returned list is re-sorted by imgt_resnum so slicing 105–117 is trivial.
    """
    tmp_in  = tmp_dir / 'chain_in.pdb'
    tmp_res = tmp_dir / 'chain_reset.pdb'
    tmp_out = tmp_dir / 'chain_imgt.pdb'

    pdb_str = _residues_to_pdb_string(residues, chain_id='A')
    tmp_in.write_text(pdb_str)

    # Reset residue numbering to avoid large numbers confusing ANARCI
    r = subprocess.run(f"pdb_reres -1 {tmp_in} > {tmp_res}",
                       shell=True, capture_output=True, text=True)
    if r.returncode != 0 or tmp_res.stat().st_size == 0:
        return None

    # Run ImmunoPDB
    r = subprocess.run(
        ['python', str(IMMUNOPDB_PATH),
         '-i', str(tmp_res), '-o', str(tmp_out),
         '-s', 'imgt', '--receptor', 'tr'],
        capture_output=True, text=True)
    if r.returncode != 0 or not tmp_out.exists() or tmp_out.stat().st_size == 0:
        return None

    imgt_text = tmp_out.read_text()

    # Build a mapping from sequential position in the reset PDB (1-based)
    # to IMGT number, by reading both files in parallel by atom order.
    # Simpler: read the IMGT output and collect (imgt_resnum, icode) in order,
    # then zip with the input residue list.
    imgt_resnums = []
    seen = set()
    for line in imgt_text.splitlines():
        if not line.startswith('ATOM') or len(line) < 27:
            continue
        if line[21] != 'A':
            continue
        try:
            rn = int(line[22:26])
        except ValueError:
            continue
        ic = line[26].strip()
        key = (rn, ic)
        if key not in seen:
            seen.add(key)
            imgt_resnums.append(rn)

    if len(imgt_resnums) != len(residues):
        # ANARCI may drop or insert residues — can't safely align
        return None

    # Annotate each residue with its IMGT number
    annotated = []
    for res, imgt_rn in zip(residues, imgt_resnums):
        r2 = dict(res)   # shallow copy — atoms array is shared (read-only after parse)
        r2['imgt_resnum'] = imgt_rn
        annotated.append(r2)

    return annotated


def find_cdr3(residues):
    """
    Identify CDR3 using a two-anchor IMGT heuristic on a single chain.

    Anchor 1 (FR3 end):  conserved Cys at IMGT ~104
        → last Cys in the first 90% of the chain

    Anchor 2 (FR4 start): conserved Phe/Trp at IMGT ~118, which is
        IMMEDIATELY followed by the FR4 motif  G - x - G  (IMGT 119-121).
        This GxG check is the critical discriminator: it eliminates all
        non-CDR3 Phe/Trp residues (e.g. those in constant-domain loops or
        framework beta-strands), which do NOT have a downstream GxG.

    CDR3 body = Cys+1 .. Phe/Trp (exclusive), length CDR3_MIN_LEN..CDR3_MAX_LEN.

    Chain length is NOT used as a filter — STCRDab CIFs may contain full
    V+C domains or just V-domains on the same chain, so the length range
    is unreliable. The GxG anchor is the reliable constraint.

    Returns (cdr3_residues, loop_start_idx, loop_end_idx, loop_seq)
      where indices are into `residues`, or None if not found.
    """
    seq = _chain_seq(residues)
    n   = len(seq)

    if n < 60:   # absolute minimum for any V-domain fragment
        return None

    # ── Step 1: find all FR4 anchors: F/W followed by G.G within 3 positions ──
    # Pattern: seq[fw] in F/W
    #          seq[fw+1] == G   (IMGT 119, almost always Gly)
    #          seq[fw+3] == G   (IMGT 121, almost always Gly)
    #          seq[fw+2] can be anything (IMGT 120, T/Q/K/R...)
    # We also accept fw+2 == G (all-Gly FR4 occurs occasionally).
    fr4_positions = []
    for fw in range(n - 3):
        if seq[fw] in ('F', 'W') and seq[fw + 1] == 'G' and seq[fw + 3] == 'G':
            fr4_positions.append(fw)

    if not fr4_positions:
        return None

    # ── Step 2: for each FR4 anchor, search backwards for the FR3 Cys ──
    for fw_idx in fr4_positions:
        loop_end = fw_idx   # CDR3 body is exclusive of the F/W

        # Search for Cys upstream, must give a CDR3 of valid length
        search_start = max(0, fw_idx - CDR3_MAX_LEN - 1)
        search_end   = max(0, fw_idx - CDR3_MIN_LEN)

        # Take the LAST Cys in the valid upstream window
        cys_candidates = [i for i in range(search_start, search_end + 1)
                          if seq[i] == 'C']
        if not cys_candidates:
            continue

        cys_idx    = cys_candidates[-1]   # last (most C-terminal) Cys
        loop_start = cys_idx + 1
        loop_len   = loop_end - loop_start

        if not CDR3_MIN_LEN <= loop_len <= CDR3_MAX_LEN:
            continue

        cdr3_res = residues[loop_start:loop_end]
        loop_seq = _chain_seq(cdr3_res)

        # Sanity: ≥2 distinct amino acids (catches poly-X artefacts)
        if len(set(loop_seq)) < 2:
            continue

        # Sanity: CDR3 body should not itself contain the FR4 motif
        # (would indicate we grabbed too much)
        if 'FG' in loop_seq or 'WG' in loop_seq:
            continue

        return cdr3_res, loop_start, loop_end, loop_seq

    return None


def infer_tcr_chain_type(loop_seq: str) -> str:
    """
    Infer whether a CDR3 is from an alpha or beta chain purely from its
    body sequence (i.e. after the anchoring Cys has been stripped).

    Heuristic based on IMGT CDR3 statistics:
      Beta:  body almost always starts ASS (CASS... full), or ASSR/ASST/etc.
             Second position S + third position S -> ~85% beta
      Alpha: body starts A but second position is NOT S, OR starts AS but
             third position is not S -> strongly alpha
             Also: non-A first position -> ambiguous/alpha

    Returns: 'alpha', 'beta', or 'ambiguous'
    """
    if len(loop_seq) < 3:
        return 'ambiguous'

    p1, p2, p3 = loop_seq[0], loop_seq[1], loop_seq[2]

    # Strong beta indicators
    if p1 == 'A' and p2 == 'S' and p3 == 'S':
        return 'beta'

    # Strong alpha indicators
    if p1 == 'A' and p2 != 'S':
        return 'alpha'
    if p1 == 'A' and p2 == 'S' and p3 not in ('S',):
        return 'alpha'

    # Fallback
    return 'ambiguous'


def extract_cdr3_by_imgt(annotated_residues: list,
                          start: int = IMGT_CDR3_START,
                          end: int   = IMGT_CDR3_END):
    """
    Extract CDR3 from a chain that has been IMGT-numbered (has 'imgt_resnum' key).
    Selects residues with imgt_resnum in [start, end] inclusive, plus
    DIHEDRAL_FLANK flanking residues on each side for accurate phi/psi.

    Returns (loop_res, window_res, loop_offset_in_window, loop_seq, imgt_nums)
         or (None, None, None, None, None) if the window is empty.
    """
    loop_idx = [i for i, r in enumerate(annotated_residues)
                if start <= r['imgt_resnum'] <= end]
    if not loop_idx:
        return None, None, None, None, None

    lo, hi   = loop_idx[0], loop_idx[-1]
    n        = len(annotated_residues)
    win_s    = max(0, lo - DIHEDRAL_FLANK)
    win_e    = min(n, hi + DIHEDRAL_FLANK + 1)

    loop_res   = annotated_residues[lo : hi + 1]
    window_res = annotated_residues[win_s : win_e]
    offset     = lo - win_s
    loop_seq   = _chain_seq(loop_res)
    imgt_nums  = [r['imgt_resnum'] for r in loop_res]

    return loop_res, window_res, offset, loop_seq, imgt_nums
    """
    Classify a structure and extract CDR3 loops.

    Chain type resolution priority:
        1. Entity metadata if not 'unknown'
        2. CDR3 sequence prefix heuristic (ASS->beta, A[^S]->alpha)
        3. 'ambiguous' — extracted but flagged

    Deduplication: identical CDR3 sequences within one structure (NCS copies,
    crystal dimers) are deduplicated — only the first chain is kept.
    """
    cdr3_hits      = []
    peptide_chains = []
    mhc_chains     = []
    seen_seqs      = set()

    for chain_id, residues in chains.items():
        n           = len(residues)
        entity_type = chain_types.get(chain_id, 'unknown')

        if entity_type in ('mhc', 'b2m'):
            mhc_chains.append(chain_id)
            continue
        if entity_type == 'peptide':
            peptide_chains.append(chain_id)
            continue

        result = find_cdr3(residues)

        if result is None:
            if 7 <= n <= 25:
                peptide_chains.append(chain_id)
            elif n > 150:
                mhc_chains.append(chain_id)
            continue

        _, loop_start, loop_end, loop_seq = result

        # Deduplicate NCS copies
        if loop_seq in seen_seqs:
            continue
        seen_seqs.add(loop_seq)

        # Resolve chain type: entity metadata -> sequence heuristic
        if entity_type == 'alpha':
            tcr_chain_type = 'alpha'
        elif entity_type == 'beta':
            tcr_chain_type = 'beta'
        else:
            tcr_chain_type = infer_tcr_chain_type(loop_seq)

        # Apply filter
        if only == 'alpha' and tcr_chain_type not in ('alpha', 'ambiguous'):
            continue
        if only == 'beta'  and tcr_chain_type not in ('beta',  'ambiguous'):
            continue

        cdr3_hits.append((chain_id, residues, result, tcr_chain_type))

    has_tcr     = bool(cdr3_hits)
    has_peptide = bool(peptide_chains)
    has_mhc     = bool(mhc_chains)

    if has_tcr and has_mhc:
        stype = 'TCR-pMHC'
    elif has_tcr:
        stype = 'TCR'
    elif has_mhc and has_peptide:
        stype = 'pMHC'
    elif has_mhc:
        stype = 'MHC'
    else:
        stype = 'unknown'

    return stype, cdr3_hits


# ─────────────────────────────────────────────────────────────────────────────
# Dihedral angles  (IUPAC / BioPython Gram-Schmidt convention)
# ─────────────────────────────────────────────────────────────────────────────

def _dihedral(p1, p2, p3, p4):
    """Dihedral angle in degrees, range (−180, 180]."""
    b0 = p1 - p2      # note: p1-p2 (Gram-Schmidt convention)
    b1 = p3 - p2
    b2 = p4 - p3
    b1_hat = b1 / (np.linalg.norm(b1) + 1e-10)
    v = b0 - np.dot(b0, b1_hat) * b1_hat
    w = b2 - np.dot(b2, b1_hat) * b1_hat
    return np.degrees(np.arctan2(
        np.dot(np.cross(b1_hat, v), w),
        np.dot(v, w)
    ))


def extract_dihedrals(residues):
    """
    Compute phi/psi for each residue in a list.
    Returns (phi_array, psi_array), boundary values are NaN.
    """
    n   = len(residues)
    phi = np.full(n, np.nan)
    psi = np.full(n, np.nan)

    def get(res, atom):
        return res['atoms'].get(atom)

    for i in range(n):
        # phi: C[i-1], N[i], CA[i], C[i]
        if i > 0:
            C_prev = get(residues[i-1], 'C')
            N_i    = get(residues[i],   'N')
            CA_i   = get(residues[i],   'CA')
            C_i    = get(residues[i],   'C')
            if all(a is not None for a in [C_prev, N_i, CA_i, C_i]):
                phi[i] = _dihedral(C_prev, N_i, CA_i, C_i)

        # psi: N[i], CA[i], C[i], N[i+1]
        if i < n - 1:
            N_i    = get(residues[i],   'N')
            CA_i   = get(residues[i],   'CA')
            C_i    = get(residues[i],   'C')
            N_next = get(residues[i+1], 'N')
            if all(a is not None for a in [N_i, CA_i, C_i, N_next]):
                psi[i] = _dihedral(N_i, CA_i, C_i, N_next)

    return phi, psi


def dihedrals_with_flanks(chain_residues, loop_start, loop_end):
    """
    Compute phi/psi for the CDR3 loop including one flanking residue
    on each side so the terminal loop residues get proper values.

    Returns (phi_loop, psi_loop) arrays of length (loop_end - loop_start).
    """
    flank_start = max(0, loop_start - 1)
    flank_end   = min(len(chain_residues), loop_end + 1)
    window_res  = chain_residues[flank_start:flank_end]

    phi_w, psi_w = extract_dihedrals(window_res)

    offset   = loop_start - flank_start
    n_loop   = loop_end   - loop_start
    return phi_w[offset:offset + n_loop], psi_w[offset:offset + n_loop]


def _dihedrals_from_window(window_res, offset, loop_len):
    """
    Compute phi/psi for a pre-sliced window returned by extract_cdr3_by_imgt.
    `offset` is the index of loop_res[0] within window_res.
    Returns (phi_loop, psi_loop) of length loop_len.
    """
    phi_w, psi_w = extract_dihedrals(window_res)
    return phi_w[offset:offset + loop_len], psi_w[offset:offset + loop_len]


# ─────────────────────────────────────────────────────────────────────────────
# Energy helpers  (identical to native_energy_overview.py)
# ─────────────────────────────────────────────────────────────────────────────

def _interp_prob(angle_deg: float, probs: np.ndarray) -> float:
    """Linear interpolation of probability at angle_deg from BIN_CENTRES grid."""
    a     = ((angle_deg + 180.0) % 360.0) - 180.0
    bin_f = (a + 180.0) / BIN_WIDTH
    lo    = int(bin_f) % N_BINS
    hi    = (lo + 1) % N_BINS
    w     = bin_f - int(bin_f)
    return float((1.0 - w) * probs[lo] + w * probs[hi])


def per_residue_energy(phi_arr, psi_arr, probs_phi, probs_psi):
    """
    Compute per-residue energy dicts.
    Returns list of dicts with keys:
        idx, phi, psi, e_phi, e_psi, e_total
    """
    rows = []
    n    = len(phi_arr)
    for i in range(n):
        e_phi = (-math.log(_interp_prob(phi_arr[i], probs_phi[i]) + 1e-10)
                 if i < len(probs_phi) and not math.isnan(phi_arr[i])
                 else float('nan'))
        e_psi = (-math.log(_interp_prob(psi_arr[i], probs_psi[i]) + 1e-10)
                 if i < len(probs_psi) and not math.isnan(psi_arr[i])
                 else float('nan'))
        e_tot = sum(v for v in [e_phi, e_psi] if not math.isnan(v))
        rows.append({
            'idx':     i,
            'phi':     float(phi_arr[i]),
            'psi':     float(psi_arr[i]),
            'e_phi':   e_phi,
            'e_psi':   e_psi,
            'e_total': e_tot,
        })
    return rows


def ideal_energy(probs_phi, probs_psi):
    """Lower-bound energy if every residue sat at its argmax bin."""
    e = 0.0
    for pp, ps in zip(probs_phi, probs_psi):
        e -= math.log(float(np.max(pp)) + 1e-10)
        e -= math.log(float(np.max(ps)) + 1e-10)
    return e


# ─────────────────────────────────────────────────────────────────────────────
# Console table
# ─────────────────────────────────────────────────────────────────────────────

def print_energy_table(rows, loop_seq, name, e_ideal, imgt_nums=None):
    W   = 74
    lbl = 'IMGT' if imgt_nums else 'pos'
    fmt = lambda v: f"{v:8.3f}" if not math.isnan(v) else "     NaN"
    print(f"\n  {'─'*W}")
    print(f"  {name}  |  loop: {loop_seq}")
    print(f"  {'─'*W}")
    print(f"  {'#':>3}  {lbl:>5}  {'AA':>3}  {'φ':>8}  {'ψ':>8}  "
          f"{'E_φ':>8}  {'E_ψ':>8}  {'E_tot':>8}")
    print(f"  {'─'*W}")
    e_sum = 0.0
    for r in rows:
        aa  = loop_seq[r['idx']] if r['idx'] < len(loop_seq) else '?'
        pos = str(imgt_nums[r['idx']] if imgt_nums and r['idx'] < len(imgt_nums)
                  else r['idx'] + 1)
        print(f"  {r['idx']+1:>3}  {pos:>5}  {aa:>3}  "
              f"{r['phi']:8.1f}  {r['psi']:8.1f}  "
              f"{fmt(r['e_phi'])}  {fmt(r['e_psi'])}  {r['e_total']:>8.3f}")
        e_sum += r['e_total']
    print(f"  {'─'*W}")
    print(f"  {'Total':>60}  {e_sum:>8.3f}")
    print(f"  {'Ideal (argmax)':>60}  {e_ideal:>8.3f}")
    print(f"  {'Gap':>60}  {e_sum - e_ideal:>8.3f}")
    print(f"  {'─'*W}")


# ─────────────────────────────────────────────────────────────────────────────
# Plots  (style from native_energy_overview.py)
# ─────────────────────────────────────────────────────────────────────────────

_CMAP_E = LinearSegmentedColormap.from_list(
    'energy', ['#1a6faf', '#74c476', '#fed976', '#e31a1c'], N=256)


def _safe_name(s):
    return re.sub(r'[^\w\-]', '_', s)


def plot_heatmap(rows, loop_seq, name, e_ideal, imgt_nums, out_path):
    n   = len(rows)
    mat = np.array([[r['e_phi'] if not math.isnan(r['e_phi']) else 0 for r in rows],
                    [r['e_psi'] if not math.isnan(r['e_psi']) else 0 for r in rows]],
                   dtype=float)

    fig, axes = plt.subplots(2, 1, figsize=(max(8, n * 0.7 + 2), 5),
                             gridspec_kw={'height_ratios': [3, 1]})
    fig.suptitle(f"Per-residue energy: {name}  |  {loop_seq}",
                 fontsize=11, fontweight='bold', y=1.01)

    ax   = axes[0]
    vmax = max(np.nanpercentile(mat, 95), 0.1)
    im   = ax.imshow(mat, aspect='auto', cmap=_CMAP_E, vmin=0, vmax=vmax,
                     interpolation='nearest')
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['φ  (phi)', 'ψ  (psi)'], fontsize=9)
    ax.set_xticks(range(n))
    ax.set_xticklabels(
        [f"{loop_seq[i] if i < len(loop_seq) else '?'}\n"
         f"{imgt_nums[i] if imgt_nums and i < len(imgt_nums) else i+1}"
         for i in range(n)], fontsize=8)
    ax.set_xlabel('Residue  (AA / IMGT)' if imgt_nums else 'Residue', fontsize=9)
    for ri in range(2):
        for ci in range(n):
            v  = mat[ri, ci]
            tc = 'white' if v > vmax * 0.6 else 'black'
            ax.text(ci, ri, f"{v:.2f}", ha='center', va='center',
                    fontsize=7, color=tc, fontweight='bold')
    plt.colorbar(im, ax=ax, label='−log p', shrink=0.8, pad=0.02)

    ax2      = axes[1]
    e_phi_a  = [r['e_phi']   if not math.isnan(r['e_phi']) else 0 for r in rows]
    e_psi_a  = [r['e_psi']   if not math.isnan(r['e_psi']) else 0 for r in rows]
    totals   = [r['e_total'] for r in rows]
    x        = np.arange(n)
    ax2.bar(x, e_phi_a, color='#4292c6', label='φ', width=0.4)
    ax2.bar(x, e_psi_a, bottom=e_phi_a, color='#ef6548', label='ψ', width=0.4)
    mean_e = sum(totals) / max(n, 1)
    ax2.axhline(mean_e, color='black', linestyle='--', linewidth=0.8,
                label=f'mean {mean_e:.2f}')
    ax2.set_xticks(x)
    ax2.set_xticklabels([str(i+1) for i in range(n)], fontsize=8)
    ax2.set_ylabel('E_total', fontsize=8)
    ax2.legend(fontsize=7, ncol=3, loc='upper right')

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"    ↳ heatmap  → {Path(out_path).name}")


def plot_distributions(rows, loop_seq, probs_phi, probs_psi, name, imgt_nums, out_path):
    n         = len(rows)
    cols      = min(n, 5)
    n_rows    = math.ceil(n / cols)

    fig = plt.figure(figsize=(cols * 4.0, n_rows * 3.8))
    fig.suptitle(f"Torsion distributions + native angle: {name}  |  {loop_seq}",
                 fontsize=11, fontweight='bold')

    for idx, r in enumerate(rows):
        aa  = loop_seq[r['idx']] if r['idx'] < len(loop_seq) else '?'
        pos = str(imgt_nums[r['idx']] if imgt_nums and r['idx'] < len(imgt_nums)
                  else r['idx'] + 1)

        # phi
        ax = fig.add_subplot(n_rows * 2, cols, idx + 1)
        if r['idx'] < len(probs_phi):
            p = np.array(probs_phi[r['idx']])
            ax.fill_between(BIN_CENTRES, p, alpha=0.35, color='#4292c6')
            ax.plot(BIN_CENTRES, p, color='#4292c6', linewidth=1.0)
            if not math.isnan(r['phi']):
                ax.axvline(r['phi'], color='#e31a1c', linewidth=1.8,
                           label=f"φ={r['phi']:.0f}°")
        ax.set_xlim(-180, 180)
        ax.set_xticks([-180, -90, 0, 90, 180])
        ax.tick_params(labelsize=6)
        ax.set_title(f"IMGT {pos} ({aa})  E_φ={r['e_phi']:.2f}" if imgt_nums
                     else f"Res {pos} ({aa})  E_φ={r['e_phi']:.2f}",
                     fontsize=7.5, pad=2)
        ax.set_ylabel('p(φ)', fontsize=6)
        if idx == 0:
            ax.legend(fontsize=6, loc='upper left')

        # psi
        ax2 = fig.add_subplot(n_rows * 2, cols, idx + 1 + cols * n_rows)
        if r['idx'] < len(probs_psi):
            p = np.array(probs_psi[r['idx']])
            ax2.fill_between(BIN_CENTRES, p, alpha=0.35, color='#ef6548')
            ax2.plot(BIN_CENTRES, p, color='#ef6548', linewidth=1.0)
            if not math.isnan(r['psi']):
                ax2.axvline(r['psi'], color='#2ca25f', linewidth=1.8,
                            label=f"ψ={r['psi']:.0f}°")
        ax2.set_xlim(-180, 180)
        ax2.set_xticks([-180, -90, 0, 90, 180])
        ax2.tick_params(labelsize=6)
        ax2.set_title(f"IMGT {pos} ({aa})  E_ψ={r['e_psi']:.2f}" if imgt_nums
                      else f"Res {pos} ({aa})  E_ψ={r['e_psi']:.2f}",
                      fontsize=7.5, pad=2)
        ax2.set_ylabel('p(ψ)', fontsize=6)
        ax2.set_xlabel('angle (°)', fontsize=6)
        if idx == 0:
            ax2.legend(fontsize=6, loc='upper left')

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"    ↳ distributions → {Path(out_path).name}")


def plot_ramachandran(rows, loop_seq, name, imgt_nums, out_path):
    valid  = [r for r in rows if not math.isnan(r['phi'])]
    phi_v  = [r['phi']     for r in valid]
    psi_v  = [r['psi']     for r in valid]
    e_v    = [r['e_total'] for r in valid]
    labels = [
        f"{loop_seq[r['idx']] if r['idx'] < len(loop_seq) else '?'}"
        f"{imgt_nums[r['idx']] if imgt_nums and r['idx'] < len(imgt_nums) else r['idx']+1}"
        for r in valid
    ]

    cmap_r = LinearSegmentedColormap.from_list(
        'energy2', ['#1a9850', '#ffffbf', '#d73027'], N=256)

    fig, ax = plt.subplots(figsize=(6, 5.5))
    sc = ax.scatter(phi_v, psi_v, c=e_v, cmap=cmap_r, s=80,
                    edgecolors='black', linewidths=0.5, zorder=3)
    for phi, psi, lbl in zip(phi_v, psi_v, labels):
        ax.annotate(lbl, (phi, psi), textcoords='offset points',
                    xytext=(4, 4), fontsize=7)
    plt.colorbar(sc, ax=ax, label='E_total (−log p)', shrink=0.85)
    ax.axhline(0, color='grey', lw=0.5, ls='--', alpha=0.4)
    ax.axvline(0, color='grey', lw=0.5, ls='--', alpha=0.4)
    ax.set_xlim(-180, 180); ax.set_ylim(-180, 180)
    ax.set_xlabel('φ (degrees)', fontsize=10)
    ax.set_ylabel('ψ (degrees)', fontsize=10)
    ax.set_title(f"Ramachandran (by energy)\n{name}  |  {loop_seq}", fontsize=10)
    ax.set_xticks(range(-180, 181, 60))
    ax.set_yticks(range(-180, 181, 60))
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"    ↳ Ramachandran → {Path(out_path).name}")


def plot_cross_loop_summary(all_results, out_path):
    """Identical style to native_energy_overview.py summary plot."""
    if len(all_results) < 2:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Cross-loop native energy summary  (CIF database)",
                 fontsize=12, fontweight='bold')

    names  = [r['name']    for r in all_results]
    totals = [r['e_total'] for r in all_results]
    ideals = [r['e_ideal'] for r in all_results]
    gaps   = [r['e_gap']   for r in all_results]
    types  = [r['struct_type'] for r in all_results]

    type_colors = {'TCR': '#4292c6', 'TCR-pMHC': '#ef6548', 'pMHC': '#74c476',
                   'unknown': '#bdbdbd'}
    bar_colors  = [type_colors.get(t, '#bdbdbd') for t in types]

    x = np.arange(len(names)); w = 0.35

    axes[0].bar(x - w/2, totals, w, color=bar_colors, alpha=0.85, label='Native E')
    axes[0].bar(x + w/2, ideals, w, color='#74c476',  alpha=0.85, label='Ideal E')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([n.replace('_', '\n') for n in names],
                             fontsize=max(4, 8 - len(names) // 10),
                             rotation=45, ha='right')
    axes[0].set_ylabel('Total energy (−log p)')
    axes[0].set_title('Native vs ideal energy per loop')
    # custom legend for structure types
    from matplotlib.patches import Patch
    legend_els = [Patch(facecolor=c, label=t) for t, c in type_colors.items()
                  if t in types]
    legend_els.append(Patch(facecolor='#74c476', label='Ideal E'))
    axes[0].legend(handles=legend_els, fontsize=7)

    gap_colors = ['#e31a1c' if g > 0 else '#1a9850' for g in gaps]
    axes[1].bar(x, gaps, color=gap_colors, alpha=0.85)
    axes[1].axhline(0, color='black', linewidth=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([n.replace('_', '\n') for n in names],
                             fontsize=max(4, 8 - len(names) // 10),
                             rotation=45, ha='right')
    axes[1].set_ylabel('Gap (native − ideal)')
    axes[1].set_title('Energy gap per loop\n(red = above ideal, green = below)')

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  ↳ Cross-loop summary → {Path(out_path).name}")


def plot_structure_type_comparison(all_results, out_path):
    """Box plot of mean loop energy grouped by structure type."""
    from collections import defaultdict
    type_energies = defaultdict(list)
    for r in all_results:
        type_energies[r['struct_type']].append(r['e_total'])

    if len(type_energies) < 2:
        return  # nothing to compare

    types = sorted(type_energies.keys())
    data  = [type_energies[t] for t in types]
    colors= ['#4292c6', '#ef6548', '#74c476', '#bdbdbd'][:len(types)]

    fig, ax = plt.subplots(figsize=(5, 4))
    bp = ax.boxplot(data, tick_labels=types, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color); patch.set_alpha(0.7)
    ax.set_ylabel('Total loop energy  (−log p)', fontsize=10)
    ax.set_title('CDR3 Energy by Structure Type', fontsize=11)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  ↳ Structure-type comparison → {Path(out_path).name}")


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def process_directory(cif_dir, model, params, output_dir, max_structures,
                      chain_type='alpha', use_anarci=True):
    cif_dir    = Path(cif_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cif_files = sorted(
        list(cif_dir.glob('*.cif')) + list(cif_dir.glob('*.mmcif'))
    )
    if max_structures:
        cif_files = cif_files[:max_structures]

    if not cif_files:
        print(f"No .cif files found in {cif_dir}")
        return []

    all_results = []
    skipped     = []

    with tempfile.TemporaryDirectory(prefix='cdr3_anarci_') as _tmp:
        tmp_dir = Path(_tmp)

        for fi, cif_path in enumerate(cif_files, 1):
            pdb_id = cif_path.stem.upper()
            print(f"\n{'─'*60}")
            print(f"[{fi}/{len(cif_files)}]  {cif_path.name}")

            # ── Parse ────────────────────────────────────────────────────────
            try:
                chains = parse_mmcif(str(cif_path))
            except Exception as exc:
                print(f"  ✗ Parse error: {exc}")
                skipped.append((cif_path.name, f'parse error: {exc}'))
                continue

            if not chains:
                print("  ✗ No ATOM records")
                skipped.append((cif_path.name, 'no ATOM records'))
                continue

            # ── Entity metadata ───────────────────────────────────────────────
            try:
                chain_types = parse_chain_types(str(cif_path))
            except Exception:
                chain_types = {}

            if chain_types:
                type_summary = {cid: chain_types.get(cid, '?') for cid in sorted(chains)}
                print(f"  Chain types: {type_summary}")
            else:
                print(f"  Chain types: (entity metadata not found — using sequence heuristic)")

            # ── Classify + find CDR3 candidates (heuristic) ──────────────────
            stype, cdr3_hits = classify_and_extract(chains, chain_types, only=chain_type)
            n_chains = len(chains)
            print(f"  Type: {stype}  |  chains: {sorted(chains.keys())} ({n_chains})")

            if not cdr3_hits:
                print(f"  ↷ No CDR3 detected — skipped")
                skipped.append((cif_path.name, f'no CDR3 (type={stype})'))
                continue

            # ── Per-CDR3 processing ───────────────────────────────────────────
            for chain_id, chain_residues, cdr3_result, tcr_chain_type in cdr3_hits:
                _, heuristic_start, heuristic_end, heuristic_seq = cdr3_result
                name = f"{pdb_id}_{chain_id}"

                # ── Attempt ANARCI IMGT numbering ─────────────────────────────
                loop_seq  = None
                imgt_nums = None
                method    = 'heuristic'
                phi_n = psi_n = None

                if use_anarci:
                    chain_tmp = tmp_dir / f"{pdb_id}_{chain_id}"
                    chain_tmp.mkdir(exist_ok=True)
                    annotated = imgt_number_chain(chain_residues, chain_tmp)

                    if annotated is not None:
                        loop_res, window_res, offset, loop_seq, imgt_nums = \
                            extract_cdr3_by_imgt(annotated)
                        if loop_res is not None:
                            method = f"IMGT {IMGT_CDR3_START}–{IMGT_CDR3_END}"
                            phi_n, psi_n = _dihedrals_from_window(
                                window_res, offset, len(loop_res))
                        else:
                            print(f"  ⚠ Chain {chain_id}: ANARCI numbered but no "
                                  f"residues at IMGT {IMGT_CDR3_START}–{IMGT_CDR3_END} "
                                  f"— falling back to heuristic")
                    else:
                        print(f"  ⚠ Chain {chain_id}: ANARCI failed — "
                              f"falling back to heuristic")

                # ── Heuristic fallback ─────────────────────────────────────────
                if loop_seq is None:
                    loop_seq  = heuristic_seq
                    imgt_nums = None
                    method    = 'heuristic'
                    phi_n, psi_n = dihedrals_with_flanks(
                        chain_residues, heuristic_start, heuristic_end)

                print(f"  ✓ Chain {chain_id} ({tcr_chain_type}): "
                      f"CDR3 = {loop_seq}  ({len(loop_seq)} res)  [{method}]")
                if imgt_nums:
                    print(f"    IMGT residues: {imgt_nums}")

                # ── Model + energy ─────────────────────────────────────────────
                probs_phi, probs_psi = predict_distributions(model, params, loop_seq)
                rows    = per_residue_energy(phi_n, psi_n, probs_phi, probs_psi)
                e_ideal = ideal_energy(probs_phi, probs_psi)
                e_total = sum(r['e_total'] for r in rows)

                # Annotate rows with IMGT position label
                for i, r in enumerate(rows):
                    r['imgt_resnum'] = (imgt_nums[i] if imgt_nums and i < len(imgt_nums)
                                        else IMGT_CDR3_START + i)

                print_energy_table(rows, loop_seq, name, e_ideal, imgt_nums)

                # ── Plots ──────────────────────────────────────────────────────
                loop_out = output_dir / _safe_name(name)
                loop_out.mkdir(exist_ok=True)

                plot_heatmap(rows, loop_seq, name, e_ideal, imgt_nums,
                             loop_out / f"heatmap_{_safe_name(name)}.png")
                plot_distributions(rows, loop_seq, probs_phi, probs_psi, name, imgt_nums,
                                   loop_out / f"distributions_{_safe_name(name)}.png")
                plot_ramachandran(rows, loop_seq, name, imgt_nums,
                                  loop_out / f"ramachandran_{_safe_name(name)}.png")

                all_results.append({
                    'name':           name,
                    'pdb_id':         pdb_id,
                    'chain_id':       chain_id,
                    'tcr_chain_type': tcr_chain_type,
                    'struct_type':    stype,
                    'method':         method,
                    'sequence':       loop_seq,
                    'length':         len(loop_seq),
                    'imgt_nums':      imgt_nums,
                    'e_total':        e_total,
                    'e_ideal':        e_ideal,
                    'e_gap':          e_total - e_ideal,
                    'residues':       rows,
                })

    # ── Cross-loop summary ────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Processed {len(all_results)} CDR3 loops  "
          f"({len(skipped)} files skipped)")

    if all_results:
        n_anarci    = sum(1 for r in all_results if r['method'] != 'heuristic')
        n_heuristic = len(all_results) - n_anarci
        print(f"  ANARCI IMGT: {n_anarci}  |  heuristic: {n_heuristic}")

        plot_cross_loop_summary(all_results, output_dir / 'summary_energy.png')
        plot_structure_type_comparison(all_results,
                                       output_dir / 'summary_by_type.png')

        json_out = output_dir / 'cdr3_energy.json'
        with open(json_out, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"  ↳ JSON → {json_out.name}")

        tsv_out = output_dir / 'cdr3_energy_per_residue.tsv'
        with open(tsv_out, 'w') as f:
            f.write('pdb_id\tchain\ttcr_chain_type\tstruct_type\tmethod\t'
                    'sequence\tposition\timgt_resnum\taa\t'
                    'phi\tpsi\te_phi\te_psi\te_total\n')
            for loop in all_results:
                for r in loop['residues']:
                    aa = loop['sequence'][r['idx']] if r['idx'] < len(loop['sequence']) else '?'
                    f.write(
                        f"{loop['pdb_id']}\t{loop['chain_id']}\t"
                        f"{loop['tcr_chain_type']}\t{loop['struct_type']}\t"
                        f"{loop['method']}\t{loop['sequence']}\t"
                        f"{r['idx']+1}\t{r.get('imgt_resnum','?')}\t{aa}\t"
                        f"{r['phi']:.2f}\t{r['psi']:.2f}\t"
                        f"{r['e_phi']:.4f}\t{r['e_psi']:.4f}\t{r['e_total']:.4f}\n"
                    )
        print(f"  ↳ TSV  → {tsv_out.name}")

    if skipped:
        print(f"\nSkipped files:")
        for fname, reason in skipped:
            print(f"  ↷ {fname}: {reason}")

    return all_results


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='CDR3 loop energy analysis from mmCIF database')
    parser.add_argument(
        '--cif_dir',
        default='/home/jtepperik/thesis/data/tcr_phmc_pdbs/downloaded_trc3d_database_as_cif',
        help='Directory of .cif/.mmcif files  [default: STCRDab database path]')
    parser.add_argument(
        '--output', default='./cif_cdr3_energy',
        help='Output directory  [default: ./cif_cdr3_energy]')
    parser.add_argument(
        '--chain_type', default='alpha', choices=['alpha', 'beta', 'both'],
        help="TCR chain to extract CDR3 from: "
             "'alpha' (default), 'beta', or 'both'")
    parser.add_argument(
        '--no_anarci', action='store_true',
        help='Skip ANARCI IMGT numbering and use sequence heuristic only')
    parser.add_argument(
        '--max_structures', type=int, default=None,
        help='Limit number of CIF files processed (for testing)')
    args = parser.parse_args()

    use_anarci = not args.no_anarci and anarci_available()
    if not args.no_anarci and not use_anarci:
        print(f"⚠  ANARCI not available:")
        print(f"   ImmunoPDB.py : {IMMUNOPDB_PATH}  "
              f"({'found' if IMMUNOPDB_PATH.exists() else 'NOT FOUND'})")
        print(f"   pdb_selchain : {shutil.which('pdb_selchain') or 'NOT FOUND'}")
        print(f"   pdb_reres    : {shutil.which('pdb_reres') or 'NOT FOUND'}")
        print(f"   → Falling back to sequence heuristic.\n")
    elif use_anarci:
        print(f"✓ ANARCI available — will apply IMGT numbering "
              f"(fallback: sequence heuristic)")

    print("Loading model...")
    model, params = load_model()

    process_directory(
        cif_dir        = args.cif_dir,
        model          = model,
        params         = params,
        output_dir     = args.output,
        max_structures = args.max_structures,
        chain_type     = args.chain_type,
        use_anarci     = use_anarci,
    )