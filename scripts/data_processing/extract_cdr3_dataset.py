"""
Simple CDR3 extraction from pre-curated list.

Uses existing PDB files and known CDR3 sequences.
No pattern matching or downloading needed!

Supports IMGT insertion codes (e.g. 112A, 112B).
"""

import numpy as np
from pathlib import Path
import json


def parse_cdr3_list(cdr3_list_text):
    """
    Parse the pre-curated CDR3 list.
    
    Format: pdb,chain,sequence
    Example: 7l1d_b,E,YFCASSGLAGGPVSGANVLTFGA
    
    Returns list of (pdb_id, chain, sequence) tuples
    """
    lines = cdr3_list_text.strip().split('\n')
    
    cdr3_entries = []
    for line in lines:
        if not line.strip() or line.startswith('#'):
            continue
        
        parts = line.strip().split(',')
        if len(parts) != 3:
            continue
        
        pdb_id, chain, sequence = parts
        cdr3_entries.append((pdb_id, chain, sequence))
    
    return cdr3_entries


def find_cdr3_in_sequence(sequence):
    """
    Find CDR3 boundaries in sequence that includes flanks.
    
    CDR3 pattern: C...C[FWY] or C...[FWY]
    Flanking sequences typically: YFC/YLC at start, *F[FG][PGE] at end
    
    Returns: (loop_start, loop_end) or None
    """
    first_c = sequence.find('C')
    if first_c == -1:
        return None
    
    last_aromatic = -1
    for i in range(len(sequence) - 1, first_c, -1):
        if sequence[i] in 'FWY':
            last_aromatic = i
            break
    
    if last_aromatic == -1:
        return None
    
    second_c = sequence.find('C', first_c + 1)
    if second_c != -1 and second_c < last_aromatic:
        return first_c, last_aromatic + 1
    else:
        return first_c, last_aromatic + 1


def three_to_one(resname):
    """Convert 3-letter to 1-letter amino acid code."""
    conversion = {
        'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E',
        'PHE': 'F', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
        'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N',
        'PRO': 'P', 'GLN': 'Q', 'ARG': 'R', 'SER': 'S',
        'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
    }
    return conversion.get(resname, 'X')


def one_to_three(aa):
    """Convert 1-letter to 3-letter amino acid code."""
    conversion = {
        'A': 'ALA', 'C': 'CYS', 'D': 'ASP', 'E': 'GLU',
        'F': 'PHE', 'G': 'GLY', 'H': 'HIS', 'I': 'ILE',
        'K': 'LYS', 'L': 'LEU', 'M': 'MET', 'N': 'ASN',
        'P': 'PRO', 'Q': 'GLN', 'R': 'ARG', 'S': 'SER',
        'T': 'THR', 'V': 'VAL', 'W': 'TRP', 'Y': 'TYR'
    }
    return conversion.get(aa, 'UNK')


def parse_residue_id(line: str) -> tuple[int, str]:
    """
    Parse residue sequence number and insertion code from a PDB ATOM line.

    PDB format:
      cols 23-26 (0-indexed 22:26): residue sequence number, right-justified
      col  27    (0-indexed 26:27): insertion code (blank if none)

    Returns (resseq: int, icode: str) where icode is '' for no insertion code.
    Raises ValueError if the residue number field is not a valid integer.
    """
    resseq = int(line[22:26])          # always an integer
    icode  = line[26:27].strip()       # 'A', 'B', … or ''
    return resseq, icode


def parse_pdb_chain(pdb_file: str | Path, chain_id: str):
    """
    Parse specific chain from PDB file.

    Handles IMGT insertion codes (e.g. 112, 112A, 112B …).

    Returns: (sequence, residues)
    where residues = [{'resnum': int, 'icode': str,
                        'resname': str,
                        'atoms': {'N': array, 'CA': array, …}}, …]
    """
    residues: list[dict] = []
    current_res: dict | None = None

    with open(pdb_file, 'r') as fh:
        for line in fh:
            if not line.startswith('ATOM'):
                continue

            chain = line[21:22].strip()
            if chain != chain_id:
                continue

            resname   = line[17:20].strip()
            atom_name = line[12:16].strip()

            try:
                resseq, icode = parse_residue_id(line)
            except ValueError:
                # Malformed residue number — skip
                continue

            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])

            # New residue when (resseq, icode) changes
            res_key = (resseq, icode)
            if current_res is None or current_res['res_key'] != res_key:
                if current_res is not None:
                    residues.append(current_res)
                current_res = {
                    'res_key': res_key,
                    'resnum':  resseq,
                    'icode':   icode,
                    'resname': resname,
                    'atoms':   {}
                }

            if atom_name in ('N', 'CA', 'C', 'O'):
                current_res['atoms'][atom_name] = np.array([x, y, z])

    if current_res is not None:
        residues.append(current_res)

    sequence = ''.join(three_to_one(r['resname']) for r in residues)
    return sequence, residues


def extract_cdr3_from_pdb(pdb_file: str | Path, chain: str, expected_sequence: str):
    """
    Extract CDR3 loop + flanks from PDB file.
    
    Args:
        pdb_file:          Path to PDB file
        chain:             Chain ID
        expected_sequence: Known sequence with flanks (from curated list)
        
    Returns:
        Dictionary with CDR3 info, or None if extraction fails
    """
    pdb_sequence, residues = parse_pdb_chain(pdb_file, chain)

    print(f"    PDB sequence: {pdb_sequence}")
    print(f"    Expected:     {expected_sequence}")

    idx = pdb_sequence.find(expected_sequence)

    if idx == -1:
        print(f"    ✗ Expected sequence not found in PDB!")
        if len(pdb_sequence) >= len(expected_sequence):
            best_match = 0
            best_idx   = -1
            for i in range(len(pdb_sequence) - len(expected_sequence) + 1):
                matches = sum(
                    1 for a, b in zip(
                        pdb_sequence[i:i + len(expected_sequence)],
                        expected_sequence
                    ) if a == b
                )
                if matches > best_match:
                    best_match = matches
                    best_idx   = i

            if best_match / len(expected_sequence) > 0.9:
                print(f"    ~ Approximate match at position {best_idx} "
                      f"({best_match}/{len(expected_sequence)} residues)")
                idx = best_idx
            else:
                return None
        else:
            return None

    extracted_residues = residues[idx : idx + len(expected_sequence)]

    for res in extracted_residues:
        if not all(atom in res['atoms'] for atom in ('N', 'CA', 'C', 'O')):
            print(f"    ✗ Missing backbone atoms in residue "
                  f"{res['resnum']}{res['icode']}")
            return None

    result = find_cdr3_in_sequence(expected_sequence)
    if result is None:
        print(f"    ✗ Could not identify CDR3 boundaries")
        return None

    loop_start, loop_end = result
    cdr3_sequence  = expected_sequence[loop_start:loop_end]
    n_flank_before = loop_start
    n_flank_after  = len(expected_sequence) - loop_end

    print(f"    ✓ Found CDR3: {cdr3_sequence}")
    print(f"      Full: {expected_sequence}")
    print(f"      Flanks: {n_flank_before} before, {n_flank_after} after")

    return {
        'full_sequence':   expected_sequence,
        'cdr3_sequence':   cdr3_sequence,
        'loop_start':      loop_start,
        'loop_end':        loop_end,
        'n_flank_before':  n_flank_before,
        'n_flank_after':   n_flank_after,
        'loop_length':     len(cdr3_sequence),
        'total_length':    len(expected_sequence),
        'residues':        extracted_residues,
    }


def save_cdr3_pdb(cdr3_data: dict, pdb_id: str, chain: str, output_dir: str | Path):
    """Save CDR3 structure to PDB file.

    Residue numbers in the output PDB preserve the original IMGT numbering,
    including insertion codes (e.g. 112A).
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    full_sequence  = cdr3_data['full_sequence']
    cdr3_sequence  = cdr3_data['cdr3_sequence']
    loop_start     = cdr3_data['loop_start']
    loop_end       = cdr3_data['loop_end']
    residues       = cdr3_data['residues']

    filename = f"{pdb_id}_{chain}_{cdr3_sequence[:10]}"
    pdb_file = output_path / f"{filename}.pdb"

    with open(pdb_file, 'w') as f:
        f.write(f"REMARK   PDB: {pdb_id}, Chain: {chain}\n")
        f.write(f"REMARK   CDR3 sequence: {cdr3_sequence}\n")
        f.write(f"REMARK   Full sequence (with flanks): {full_sequence}\n")
        f.write(f"REMARK   Loop residues: {loop_start+1}-{loop_end}\n")
        f.write(f"REMARK   Flanking: {cdr3_data['n_flank_before']} before, "
                f"{cdr3_data['n_flank_after']} after\n")

        atom_num = 1
        for i, res in enumerate(residues):
            resname_3 = one_to_three(full_sequence[i])
            resnum    = res['resnum']
            icode     = res['icode']          # '' or e.g. 'A'

            is_loop = loop_start <= i < loop_end
            bfactor = 1.00 if is_loop else 0.00

            for atom_name in ('N', 'CA', 'C', 'O'):
                if atom_name not in res['atoms']:
                    continue
                coord = res['atoms'][atom_name]
                # PDB cols: resnum right-justified in 4 chars (22:26),
                # icode in col 27 (26:27).
                f.write(
                    f"ATOM  {atom_num:5d}  {atom_name:3s} {resname_3:3s} "
                    f"A{resnum:4d}{icode:<1s}   "
                    f"{coord[0]:8.3f}{coord[1]:8.3f}{coord[2]:8.3f}"
                    f"  1.00{bfactor:6.2f}           {atom_name[0]:1s}  \n"
                )
                atom_num += 1

        f.write("END\n")

    return {
        'pdb_id':        pdb_id,
        'chain':         chain,
        'full_sequence': full_sequence,
        'cdr3_sequence': cdr3_sequence,
        'loop_start':    loop_start,
        'loop_end':      loop_end,
        'n_flank_before': cdr3_data['n_flank_before'],
        'n_flank_after':  cdr3_data['n_flank_after'],
        'loop_length':   len(cdr3_sequence),
        'total_length':  len(full_sequence),
        'pdb_file':      str(pdb_file.absolute()),
    }


def build_cdr3_dataset_from_list(
    cdr3_list_text: str,
    pdb_directory: str | Path = "/home/jtepperik/thesis/data/reference_final",
    output_dir:    str | Path = "cdr3_dataset",
):
    """
    Build CDR3 dataset from pre-curated list.
    
    Args:
        cdr3_list_text: String with CDR3 list (pdb,chain,sequence format)
        pdb_directory:  Where PDB files are stored
        output_dir:     Where to save extracted CDR3 structures
        
    Returns:
        List of metadata dictionaries
    """
    print("\n" + "="*70)
    print("CDR3 EXTRACTION FROM CURATED LIST")
    print("="*70)

    entries = parse_cdr3_list(cdr3_list_text)
    print(f"\nFound {len(entries)} CDR3 entries in list")

    pdb_dir   = Path(pdb_directory)
    all_cdr3s: list[dict] = []

    for pdb_id, chain, sequence in entries:
        print(f"\n{pdb_id} chain {chain}:")

        pdb_file = pdb_dir / f"{pdb_id}.pdb"
        if not pdb_file.exists():
            print(f"  ✗ PDB file not found: {pdb_file}")
            continue

        cdr3_data = extract_cdr3_from_pdb(pdb_file, chain, sequence)
        if cdr3_data is None:
            print(f"  ✗ Extraction failed")
            continue

        metadata = save_cdr3_pdb(cdr3_data, pdb_id, chain, output_dir)
        all_cdr3s.append(metadata)

    if not all_cdr3s:
        print("\n✗ No CDR3 loops extracted!")
        return []

    output_path  = Path(output_dir)
    summary_file = output_path / "cdr3_dataset.json"

    with open(summary_file, 'w') as f:
        json.dump(all_cdr3s, f, indent=2)

    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"Total CDR3 loops: {len(all_cdr3s)}")

    loop_lengths = [m['loop_length'] for m in all_cdr3s]
    print(f"\nCDR3 loop length distribution:")
    for length in sorted(set(loop_lengths)):
        count = loop_lengths.count(length)
        print(f"  {length:2d} residues: {count:3d} loops")

    total_lengths = [m['total_length'] for m in all_cdr3s]
    print(f"\nTotal length (with flanks):")
    print(f"  Min: {min(total_lengths)}, Max: {max(total_lengths)}, "
          f"Mean: {np.mean(total_lengths):.1f}")

    print(f"\nSaved to: {output_dir}/")
    print(f"  - {len(all_cdr3s)} PDB files")
    print(f"  - cdr3_dataset.json")

    return all_cdr3s


if __name__ == "__main__":
    CDR3_LIST = """
7l1d_b,E,YFCASSGLAGGPVSGANVLTFGA 
7na5_b,E,YFCASSQEPGGYAEQFFGP
7pbc_b,E,YFCASSFTDTQYFGP
7pbe_b,E,YLCASSSANSGELFFGE
7pdw_b,E,YFCASSFTDTQYFGP
7phr_b,E,YLCASSWGAPYEQYFGP
7qpj_b,E,YFCASSFATEAFFGQ
7rk7_b,E,YFCAISPTEEGGLIFPGNTIYFGE
7rm4_b,E,YLCASSLDPGDTGELFFGE
7rrg_b,E,YLCASSLVAETYEQYFGP
8d5q_b,E,LYCTCSAGRGGYAEQFFGP
8dnt_b,E,YLCASSLDLGADEQFFGP
8i5c_b,E,YFCASGDTGGYEQYFGP
8i5d_b,E,YLCASSLEGTVEETLYFGS
8shi_b,E,YFCASSYSEGEDEAFFGQ
8wte_b,E,YFCASSQDRGDSAETLYFGS
8wul_b,E,YFCASSQDRGDSAHTLYFGS
"""

    dataset = build_cdr3_dataset_from_list(
        CDR3_LIST,
        pdb_directory="/home/jtepperik/thesis/data/reference_final",
        output_dir="cdr3_dataset",
    )

    print(f"\n✓ Ready for testing!")
    print(f"\nNext steps:")
    print(f"  python test_cdr3_predictions.py")