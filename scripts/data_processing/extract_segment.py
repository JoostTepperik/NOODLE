"""
Extract an arbitrary backbone segment (helix, strand, loop) from a PDB file
and save it in the same cdr3_dataset.json format used by test_cdr3_predictions.py.

The "segment" (helix/strand) plays the role of the CDR3 loop — it is what the
optimizer will reconstruct.  Flanking residues on each side are kept as anchors.

Usage
─────
Edit the SEGMENTS list at the bottom and run:

    python extract_segment.py

Output
──────
  segment_dataset/
    <pdb_id>_<chain>_<start>-<end>.pdb
    segment_dataset.json          ← drop-in replacement for cdr3_dataset.json
"""

import numpy as np
from pathlib import Path
import json


# ─────────────────────────────────────────────────────────────────────────────
# PDB parsing (same as cdr3 script)
# ─────────────────────────────────────────────────────────────────────────────

def three_to_one(resname):
    conv = {
        'ALA':'A','CYS':'C','ASP':'D','GLU':'E','PHE':'F','GLY':'G',
        'HIS':'H','ILE':'I','LYS':'K','LEU':'L','MET':'M','ASN':'N',
        'PRO':'P','GLN':'Q','ARG':'R','SER':'S','THR':'T','VAL':'V',
        'TRP':'W','TYR':'Y',
    }
    return conv.get(resname, 'X')

def one_to_three(aa):
    conv = {
        'A':'ALA','C':'CYS','D':'ASP','E':'GLU','F':'PHE','G':'GLY',
        'H':'HIS','I':'ILE','K':'LYS','L':'LEU','M':'MET','N':'ASN',
        'P':'PRO','Q':'GLN','R':'ARG','S':'SER','T':'THR','V':'VAL',
        'W':'TRP','Y':'TYR',
    }
    return conv.get(aa, 'UNK')


def parse_pdb_chain(pdb_file, chain_id):
    """
    Parse backbone atoms for one chain.
    Returns (sequence, residues) where residues is a list of dicts:
        {'resnum': int, 'resname': str, 'atoms': {'N': array, 'CA': array, ...}}
    Insertion codes are ignored — residues are indexed by resnum only.
    """
    residues   = []
    current    = None
    seen_resnums = set()

    with open(pdb_file) as f:
        for line in f:
            if not line.startswith('ATOM'):
                continue
            chain     = line[21]
            if chain != chain_id:
                continue
            resnum    = int(line[22:26])
            resname   = line[17:20].strip()
            atom_name = line[12:16].strip()
            x, y, z   = float(line[30:38]), float(line[38:46]), float(line[46:54])

            if resnum not in seen_resnums:
                if current is not None:
                    residues.append(current)
                current = {'resnum': resnum, 'resname': resname, 'atoms': {}}
                seen_resnums.add(resnum)

            if atom_name in ('N', 'CA', 'C', 'O'):
                current['atoms'][atom_name] = np.array([x, y, z])

    if current is not None:
        residues.append(current)

    sequence = ''.join(three_to_one(r['resname']) for r in residues)
    return sequence, residues


# ─────────────────────────────────────────────────────────────────────────────
# Extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_segment(
    pdb_file,
    chain_id,
    seg_start_resnum,   # first PDB residue number of the segment (inclusive)
    seg_end_resnum,     # last  PDB residue number of the segment (inclusive)
    n_flank_before = 3,
    n_flank_after  = 3,
):
    """
    Extract a segment + flanks from the chain.

    seg_start_resnum / seg_end_resnum are PDB residue numbers (as printed in
    the ATOM records), NOT 0-based indices.

    Returns a dict ready for save_segment_pdb, or None on failure.
    """
    sequence, residues = parse_pdb_chain(pdb_file, chain_id)

    # Build resnum → index map
    resnum_to_idx = {r['resnum']: i for i, r in enumerate(residues)}

    if seg_start_resnum not in resnum_to_idx:
        print(f"  ✗ Residue {seg_start_resnum} not found in chain {chain_id}")
        return None
    if seg_end_resnum not in resnum_to_idx:
        print(f"  ✗ Residue {seg_end_resnum} not found in chain {chain_id}")
        return None

    seg_start_idx = resnum_to_idx[seg_start_resnum]
    seg_end_idx   = resnum_to_idx[seg_end_resnum]

    # Flank indices
    flank_before_start = seg_start_idx - n_flank_before
    flank_after_end    = seg_end_idx   + n_flank_after + 1

    if flank_before_start < 0:
        print(f"  ✗ Not enough residues before segment "
              f"(need {n_flank_before}, have {seg_start_idx})")
        return None
    if flank_after_end > len(residues):
        print(f"  ✗ Not enough residues after segment")
        return None

    extracted = residues[flank_before_start:flank_after_end]

    # Check completeness
    for r in extracted:
        missing = [a for a in ('N','CA','C','O') if a not in r['atoms']]
        if missing:
            print(f"  ✗ Residue {r['resnum']} missing atoms: {missing}")
            return None

    full_sequence = ''.join(three_to_one(r['resname']) for r in extracted)
    loop_start    = n_flank_before
    loop_end      = n_flank_before + (seg_end_idx - seg_start_idx + 1)
    seg_sequence  = full_sequence[loop_start:loop_end]

    print(f"  Full sequence : {full_sequence}")
    print(f"  Segment       : {seg_sequence}  (pos {loop_start}–{loop_end}, "
          f"{loop_end - loop_start} residues)")
    print(f"  Flanks        : {n_flank_before} before, {n_flank_after} after")

    return {
        'full_sequence': full_sequence,
        'cdr3_sequence': seg_sequence,   # keep key name for compatibility
        'loop_start':    loop_start,
        'loop_end':      loop_end,
        'loop_length':   loop_end - loop_start,
        'total_length':  len(full_sequence),
        'n_flank_before': n_flank_before,
        'n_flank_after':  n_flank_after,
        'residues':       extracted,
        'seg_start_resnum': seg_start_resnum,
        'seg_end_resnum':   seg_end_resnum,
    }


def save_segment_pdb(seg_data, pdb_id, chain, output_dir):
    """Write extracted segment to PDB and return metadata dict."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    full_seq   = seg_data['full_sequence']
    loop_start = seg_data['loop_start']
    loop_end   = seg_data['loop_end']
    residues   = seg_data['residues']

    label    = f"{seg_data['seg_start_resnum']}-{seg_data['seg_end_resnum']}"
    pdb_path = out / f"{pdb_id}_{chain}_{label}.pdb"

    with open(pdb_path, 'w') as f:
        f.write(f"REMARK  PDB: {pdb_id}  Chain: {chain}\n")
        f.write(f"REMARK  Segment residues (PDB numbering): {label}\n")
        f.write(f"REMARK  Segment sequence: {seg_data['cdr3_sequence']}\n")
        f.write(f"REMARK  Full sequence (with flanks): {full_seq}\n")
        f.write(f"REMARK  Loop indices (0-based, half-open): "
                f"{loop_start}:{loop_end}\n")

        atom_num = 1
        for i, res in enumerate(residues):
            resname3 = one_to_three(full_seq[i])
            resnum   = i + 1
            bfactor  = 1.00 if loop_start <= i < loop_end else 0.00

            for atom_name in ('N', 'CA', 'C', 'O'):
                if atom_name not in res['atoms']:
                    continue
                coord = res['atoms'][atom_name]
                f.write(
                    f"ATOM  {atom_num:5d}  {atom_name:<3s} {resname3:3s} A"
                    f"{resnum:4d}    "
                    f"{coord[0]:8.3f}{coord[1]:8.3f}{coord[2]:8.3f}"
                    f"  1.00{bfactor:6.2f}           {atom_name[0]:1s}  \n"
                )
                atom_num += 1

        f.write("END\n")

    return {
        'pdb_id':        pdb_id,
        'chain':         chain,
        'full_sequence': full_seq,
        'cdr3_sequence': seg_data['cdr3_sequence'],
        'loop_start':    loop_start,
        'loop_end':      loop_end,
        'n_flank_before': seg_data['n_flank_before'],
        'n_flank_after':  seg_data['n_flank_after'],
        'loop_length':   seg_data['loop_length'],
        'total_length':  seg_data['total_length'],
        'pdb_file':      str(pdb_path.absolute()),
        'segment_type':  'helix',           # informational
        'seg_resnum':    f"{seg_data['seg_start_resnum']}-{seg_data['seg_end_resnum']}",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def build_segment_dataset(segments, pdb_directory, output_dir="segment_dataset"):
    """
    segments: list of dicts with keys:
        pdb_id, chain, seg_start, seg_end,
        n_flank_before (optional, default 4), n_flank_after (optional, default 4)
    """
    pdb_dir = Path(pdb_directory)
    all_meta = []

    print("\n" + "="*60)
    print("SEGMENT EXTRACTION")
    print("="*60)

    for s in segments:
        pdb_id  = s['pdb_id']
        chain   = s['chain']
        start   = s['seg_start']
        end     = s['seg_end']
        n_bef   = s.get('n_flank_before', 4)
        n_aft   = s.get('n_flank_after',  4)

        print(f"\n{pdb_id} chain {chain}  residues {start}–{end}:")

        pdb_file = pdb_dir / f"{pdb_id}.pdb"
        if not pdb_file.exists():
            print(f"  ✗ PDB not found: {pdb_file}")
            continue

        seg_data = extract_segment(
            pdb_file, chain, start, end, n_bef, n_aft
        )
        if seg_data is None:
            continue

        meta = save_segment_pdb(seg_data, pdb_id, chain, output_dir)
        all_meta.append(meta)
        print(f"  ✓ Saved → {Path(meta['pdb_file']).name}")

    json_path = Path(output_dir) / "cdr3_dataset.json"
    with open(json_path, 'w') as f:
        json.dump(all_meta, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Extracted {len(all_meta)} segment(s)")
    print(f"Saved to {output_dir}/cdr3_dataset.json")
    print(f"\nRun predictions with:")
    print(f"  python test_cdr3_predictions.py --dataset {output_dir}")
    return all_meta


if __name__ == "__main__":
    # ── Define segments to extract ────────────────────────────────────────────
    # seg_start / seg_end are PDB residue numbers (from the ATOM records),
    # NOT 0-based indices.  Check in PyMOL with: select resi 64-86 and chain E
    #
    # n_flank_before / n_flank_after: how many residues on each side to keep
    # as anchors. 4 is usually enough for a stable anchor.

    SEGMENTS = [
        {
            'pdb_id':         '7pbc_b',
            'chain':          'A',
            'seg_start':      64,    # first helix residue (PDB numbering)
            'seg_end':        86,    # last  helix residue (PDB numbering)
            'n_flank_before': 3,
            'n_flank_after':  3,
        },
    ]

    PDB_DIR = "/home/jtepperik/thesis/data/reference_final"

    build_segment_dataset(SEGMENTS, PDB_DIR, output_dir="segment_dataset")