#!/usr/bin/env python3
"""
Convert AlphaFold3 CIF outputs to PDB and rename chains to TCR-pMHC convention.

Expected input layout:
    <input_dir>/<structure_id>/seed-*_sample-*/model.cif

Chain mapping applied (only when needed):
    A -> D  (TCR alpha)
    B -> E  (TCR beta)
    M -> A  (MHC alpha)
    N -> B  (MHC beta)
    P -> C  (peptide)

If the structure already contains chains D, E, A, B, C and lacks M/N/P, renaming is skipped.

Usage:
    python process_af3_structures.py [--input-dir DIR] [--output-dir DIR]
"""

import sys
import argparse
from pathlib import Path
from Bio import PDB

# Chain mapping from AF3 default to TCR-pMHC convention
CHAIN_MAPPING = {
    'A': 'D',  # TCR alpha
    'B': 'E',  # TCR beta
    'M': 'A',  # MHC alpha
    'N': 'B',  # MHC beta
    'P': 'C',  # peptide
}

TARGET_CHAINS = {'D', 'E', 'A', 'B', 'C'}
SOURCE_CHAINS = set(CHAIN_MAPPING.keys())


def rename_chains(structure, chain_mapping):
    """Rename chains in-place according to chain_mapping."""
    for model in structure:
        for chain in list(model.get_chains()):
            if chain.id in chain_mapping:
                chain.id = chain_mapping[chain.id]


def process_structure(cif_file, output_dir, chain_mapping):
    """
    Convert a single CIF file to PDB, renaming chains if necessary.

    Output filename: <structure_id>_<sample_dir>_renumbered.pdb
    e.g. 7l1d_tcrpmhc_rs0_sample-0_renumbered.pdb
    """
    structure_id = cif_file.parent.parent.name   # e.g. 7l1d_tcrpmhc_rs0
    sample_dir = cif_file.parent.name             # e.g. seed-1_sample-0
    sample_suffix = sample_dir.split('_')[-1]     # e.g. sample-0

    print(f"Processing: {structure_id}/{sample_dir}")

    parser = PDB.MMCIFParser(QUIET=True)
    structure = parser.get_structure('structure', cif_file)

    existing_chains = {chain.id for chain in list(structure.get_models())[0].get_chains()}

    if TARGET_CHAINS.issubset(existing_chains) and not existing_chains & SOURCE_CHAINS:
        print(f"  Already correctly named ({sorted(existing_chains)}), skipping rename")
    else:
        print(f"  Renaming chains from {sorted(existing_chains)}")
        rename_chains(structure, chain_mapping)

    output_name = f"{structure_id}_{sample_suffix}_renumbered.pdb"
    output_file = output_dir / output_name

    io = PDB.PDBIO()
    io.set_structure(structure)
    io.save(str(output_file))

    print(f"  Saved: {output_file.name}")
    return output_file


def main():
    parser = argparse.ArgumentParser(description='Convert AF3 CIF outputs to PDB with TCR-pMHC chain naming.')
    parser.add_argument('--input-dir', default='af3_outputs_cif',
                        help='Directory containing AF3 output subdirectories (default: af3_outputs_cif)')
    parser.add_argument('--output-dir', default='processed_structures',
                        help='Directory for output PDB files (default: processed_structures)')
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    cif_files = sorted(input_dir.glob('*/seed-*_sample-*/model.cif'))
    if not cif_files:
        print(f"ERROR: No CIF files found under {input_dir}/*/seed-*_sample-*/model.cif")
        return 1

    print(f"Found {len(cif_files)} CIF files in {input_dir}")

    success = failed = 0
    for cif_file in cif_files:
        try:
            process_structure(cif_file, output_dir, CHAIN_MAPPING)
            success += 1
        except Exception as e:
            print(f"  FAILED: {cif_file} — {e}")
            failed += 1

    print(f"\nDone — success: {success}, failed: {failed}")
    return 0 if failed == 0 else 1


if __name__ == '__main__':
    sys.exit(main())