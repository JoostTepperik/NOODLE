"""
Automatic CDR3 extraction using IMGT patterns.

Simpler approach that actually works!
"""

import requests
import numpy as np
from pathlib import Path
import json
import gzip
import re


# Known high-quality TCR structures
TCR_PDB_IDS = [
    '1BD2', '1FYT', '2BNR', '2IAM', '3QIU', '5EU6', '6JXR',
    '1KGC', '2BNU', '3HG1', '3MBE', '4JFH', '5BRZ', '6EQA',
    '1AO7', '1LP9', '2NX5', '3PL6', '4P5T', '5TEZ'
]


def download_pdb(pdb_id, output_dir="tcr_pdbs"):
    """Download PDB file."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    pdb_file = output_path / f"{pdb_id}.pdb"
    if pdb_file.exists():
        return pdb_file
    
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb.gz"
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        content = gzip.decompress(response.content).decode('utf-8')
        
        with open(pdb_file, 'w') as f:
            f.write(content)
        
        return pdb_file
    except Exception as e:
        print(f"  ✗ Failed to download {pdb_id}: {e}")
        return None


def three_to_one(resname):
    """Convert 3-letter to 1-letter."""
    conversion = {
        'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E',
        'PHE': 'F', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
        'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N',
        'PRO': 'P', 'GLN': 'Q', 'ARG': 'R', 'SER': 'S',
        'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
    }
    return conversion.get(resname, 'X')


def parse_pdb_file(pdb_file):
    """Parse PDB and extract chains."""
    chains = {}
    
    with open(pdb_file) as f:
        for line in f:
            if line.startswith("ATOM"):
                chain_id = line[21]
                resname = line[17:20].strip()
                resnum = int(line[22:26])
                atom_name = line[12:16].strip()
                
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                
                if chain_id not in chains:
                    chains[chain_id] = {}
                
                if resnum not in chains[chain_id]:
                    chains[chain_id][resnum] = {
                        'resname': resname,
                        'resnum': resnum,
                        'atoms': {}
                    }
                
                chains[chain_id][resnum]['atoms'][atom_name] = np.array([x, y, z])
    
    # Convert to sorted lists
    for chain_id in chains:
        residues = sorted(chains[chain_id].values(), key=lambda r: r['resnum'])
        chains[chain_id] = residues
    
    return chains


def find_cdr3_auto(sequence):
    """
    Automatically find CDR3 using multiple patterns.
    
    Patterns to try:
    1. C...C[FWY] (basic)
    2. C...[FWY]G.G (IMGT)
    3. Just C...aromatic (very permissive)
    """
    candidates = []
    
    # Method 1: Simple C...C[FWY]
    for i in range(len(sequence)):
        if sequence[i] == 'C':
            # Look for another C followed by aromatic
            for j in range(i + 6, min(i + 25, len(sequence))):  # Relaxed: 6-25 residues
                if sequence[j] == 'C' and j+1 < len(sequence) and sequence[j+1] in 'FWYH':
                    cdr3_seq = sequence[i:j+2]
                    candidates.append(('CC_aromatic', i, j+2, cdr3_seq))
    
    # Method 2: IMGT pattern C...[FWY]G.G
    for match in re.finditer(r'C.{4,22}[FWY]G.G', sequence):  # Relaxed length
        start = match.start()
        end = match.end() - 3  # Before GxG
        cdr3_seq = sequence[start:end]
        candidates.append(('imgt', start, end, cdr3_seq))
    
    # Method 3: Very permissive - just C followed by aromatic later
    for i in range(len(sequence)):
        if sequence[i] == 'C':
            for j in range(i + 6, min(i + 25, len(sequence))):
                if sequence[j] in 'FWYH':
                    cdr3_seq = sequence[i:j+1]
                    # Must have some diversity
                    if len(set(cdr3_seq)) >= 4:
                        candidates.append(('C_aromatic', i, j+1, cdr3_seq))
                        break  # Take first aromatic
    
    if not candidates:
        return None
    
    # Score candidates
    scored = []
    for method, start, end, seq in candidates:
        length = end - start
        diversity = len(set(seq))
        
        # Score: prefer length 10-15, high diversity
        score = 0
        
        # Length preference
        if 10 <= length <= 15:
            score += 20
        elif 8 <= length <= 18:
            score += 10
        elif 7 <= length <= 20:
            score += 5
        
        # Diversity (more unique amino acids = better)
        score += diversity * 2
        
        # Prefer IMGT and CC_aromatic methods
        if method == 'imgt':
            score += 15
        elif method == 'CC_aromatic':
            score += 10
        
        # Penalize repetitive sequences
        if 'GGGG' in seq or 'SSSS' in seq or 'AAAA' in seq:
            score -= 20
        
        # Penalize all same type
        if len(set(seq)) < 4:
            score -= 10
        
        scored.append((score, method, start, end, seq))
    
    # Return best
    scored.sort(reverse=True)
    
    # Only return if score is reasonable
    if scored[0][0] < 15:  # Minimum score threshold
        return None
    
    _, method, start, end, seq = scored[0]
    
    return (start, end, seq, method)


def extract_all_cdr3s(pdb_file, pdb_id, verbose=True):
    """Extract all CDR3 loops from a PDB file."""
    chains = parse_pdb_file(pdb_file)
    
    if verbose:
        print(f"  Chains in PDB: {list(chains.keys())}")
    
    cdr3_loops = []
    
    for chain_id, residues in chains.items():
        sequence = ''.join([three_to_one(r['resname']) for r in residues])
        
        if verbose:
            print(f"    Chain {chain_id}: {len(sequence)} residues")
        
        # Relaxed filter: any chain with 50-200 residues
        # (TCR variable domains are ~100-120, but can vary)
        if len(sequence) < 50 or len(sequence) > 200:
            if verbose:
                print(f"      Skipped (length out of range)")
            continue
        
        if verbose:
            print(f"      Sequence: {sequence[:80]}...")
        
        # Find CDR3
        result = find_cdr3_auto(sequence)
        
        if result:
            start, end, cdr3_seq, method = result
            
            if verbose:
                print(f"      ✓ Found CDR3: {cdr3_seq} (pos {start}-{end}, {method})")
            
            # Extract backbone atoms
            cdr3_residues = residues[start:end]
            
            # Check backbone completeness
            complete = True
            for res in cdr3_residues:
                if not all(atom in res['atoms'] for atom in ['N', 'CA', 'C', 'O']):
                    complete = False
                    break
            
            if not complete:
                if verbose:
                    print(f"      ✗ Incomplete backbone atoms")
                continue
            
            cdr3_loops.append({
                'pdb_id': pdb_id,
                'chain': chain_id,
                'sequence': cdr3_seq,
                'length': len(cdr3_seq),
                'start': start,
                'end': end,
                'residues': cdr3_residues,
                'method': method
            })
        else:
            if verbose:
                print(f"      ✗ No CDR3 pattern found")
    
    return cdr3_loops


def save_cdr3_pdb(cdr3, output_dir="cdr3_dataset"):
    """Save CDR3 loop as PDB file."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    pdb_id = cdr3['pdb_id']
    chain = cdr3['chain']
    sequence = cdr3['sequence']
    residues = cdr3['residues']
    
    filename = f"{pdb_id}_{chain}_{sequence[:10]}"
    pdb_file = output_path / f"{filename}.pdb"
    
    with open(pdb_file, 'w') as f:
        atom_num = 1
        for i, res in enumerate(residues):
            resname_3 = one_to_three(sequence[i])
            resnum = i + 1
            
            for atom_name in ['N', 'CA', 'C', 'O']:
                if atom_name in res['atoms']:
                    coord = res['atoms'][atom_name]
                    f.write(f"ATOM  {atom_num:5d}  {atom_name:3s} {resname_3:3s} A{resnum:4d}    "
                           f"{coord[0]:8.3f}{coord[1]:8.3f}{coord[2]:8.3f}"
                           f"  1.00  0.00           {atom_name[0]:1s}  \n")
                    atom_num += 1
        
        f.write("END\n")
    
    return {
        'pdb_id': pdb_id,
        'chain': chain,
        'sequence': sequence,
        'length': len(sequence),
        'pdb_file': str(pdb_file.absolute()),  # Save absolute path
        'method': cdr3['method']
    }


def one_to_three(aa):
    """Convert 1-letter to 3-letter."""
    conversion = {
        'A': 'ALA', 'C': 'CYS', 'D': 'ASP', 'E': 'GLU',
        'F': 'PHE', 'G': 'GLY', 'H': 'HIS', 'I': 'ILE',
        'K': 'LYS', 'L': 'LEU', 'M': 'MET', 'N': 'ASN',
        'P': 'PRO', 'Q': 'GLN', 'R': 'ARG', 'S': 'SER',
        'T': 'THR', 'V': 'VAL', 'W': 'TRP', 'Y': 'TYR'
    }
    return conversion.get(aa, 'UNK')


def build_cdr3_dataset(output_dir="cdr3_dataset", verbose=False):
    """Build CDR3 dataset automatically."""
    print("\n" + "="*70)
    print("AUTOMATIC CDR3 EXTRACTION")
    print("="*70)
    
    all_cdr3s = []
    
    for i, pdb_id in enumerate(TCR_PDB_IDS, 1):
        print(f"\n[{i}/{len(TCR_PDB_IDS)}] Processing {pdb_id}...")
        
        # Download
        pdb_file = download_pdb(pdb_id)
        if pdb_file is None:
            continue
        
        # Extract CDR3s (verbose for first few to see what's happening)
        show_details = (i <= 3) or verbose
        cdr3_loops = extract_all_cdr3s(pdb_file, pdb_id, verbose=show_details)
        
        if cdr3_loops:
            print(f"  ✓ Found {len(cdr3_loops)} CDR3 loop(s)")
            for cdr3 in cdr3_loops:
                print(f"    Chain {cdr3['chain']}: {cdr3['sequence']} ({cdr3['length']} res, {cdr3['method']})")
                
                meta = save_cdr3_pdb(cdr3, output_dir)
                all_cdr3s.append(meta)
        else:
            print(f"  ✗ No CDR3 loops found")
    
    # Save summary
    if len(all_cdr3s) == 0:
        print("\n✗ No CDR3 loops extracted!")
        print("\nThis might happen if:")
        print("  - PDB files don't contain TCR variable domains")
        print("  - Chains are too short/long")
        print("  - CDR3 regions have unusual sequences")
        return []
    
    output_path = Path(output_dir)
    summary_file = output_path / "cdr3_dataset.json"
    
    with open(summary_file, 'w') as f:
        json.dump(all_cdr3s, f, indent=2)
    
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"Total CDR3 loops: {len(all_cdr3s)}")
    
    lengths = [m['length'] for m in all_cdr3s]
    print(f"\nLength distribution:")
    for length in sorted(set(lengths)):
        count = lengths.count(length)
        print(f"  {length:2d} residues: {count:3d} loops")
    
    print(f"\nSaved to: {output_dir}/")
    print(f"  - {len(all_cdr3s)} PDB files")
    print(f"  - cdr3_dataset.json")
    
    return all_cdr3s


if __name__ == "__main__":
    dataset = build_cdr3_dataset(verbose=False)  # Set to True for full debugging