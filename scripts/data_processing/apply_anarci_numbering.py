#!/usr/bin/env python3
"""
Apply ANARCI/ImmunoPDB IMGT numbering to TCR chains in TCR-pMHC PDB structures.

Assumes the swifttcr conda environment is active and that ImmunoPDB.py is available.
Set IMMUNOPDB_PATH to the location of ImmunoPDB.py on your system.

Usage:
    python apply_anarci_numbering.py [--input-dir DIR] [--output-dir DIR]

Chain layout expected in input PDB:
    D, E  — TCR alpha/beta
    A, B, C — pMHC
"""

import sys
import os
import shutil
import subprocess
import argparse
from pathlib import Path

# Path to ImmunoPDB.py — update this to match your installation
IMMUNOPDB_PATH = Path.home() / "AlphaFold3" / "tools" / "ANARCI" / "Example_scripts_and_sequences" / "ImmunoPDB.py"


def run_command(command):
    """Run a shell command, raise on failure."""
    try:
        result = subprocess.run(command, shell=True, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True, check=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error running: {command}\n{e.stderr}")
        raise


def renumber_tcr_chains(input_pdb, output_pdb):
    """
    Renumber TCR chains (D, E) using IMGT numbering via ImmunoPDB,
    then merge with pMHC chains (A, B, C).
    """
    print(f"\nProcessing: {input_pdb}")

    temp_dir = Path("temp_anarci")
    temp_dir.mkdir(exist_ok=True)

    base_name = Path(input_pdb).stem

    # Extract and shift TCR chains
    tcr_file = temp_dir / f"{base_name}_tcr.pdb"
    run_command(f"pdb_selchain -D,E {input_pdb} > {tcr_file}")

    tcr_shifted = temp_dir / f"{base_name}_tcr_shifted.pdb"
    run_command(f"pdb_reres -500 {tcr_file} > {tcr_shifted}")

    # Extract pMHC chains
    pmhc_file = temp_dir / f"{base_name}_pmhc.pdb"
    run_command(f"pdb_selchain -A,B,C {input_pdb} > {pmhc_file}")

    # Apply IMGT numbering via ImmunoPDB
    renumb_tcr = temp_dir / f"{base_name}_renumb_tcr.pdb"
    try:
        subprocess.run(
            ["python", str(IMMUNOPDB_PATH), "-i", str(tcr_shifted),
             "-o", str(renumb_tcr), "-s", "imgt", "--receptor", "tr"],
            check=True, capture_output=True, text=True
        )
    except subprocess.CalledProcessError as e:
        print(f"  ANARCI failed:\n{e.stderr}")
        return False

    # Merge renumbered TCR with pMHC
    run_command(f"cat {renumb_tcr} {pmhc_file} | grep '^ATOM' > {output_pdb}")

    shutil.rmtree(temp_dir)
    print(f"  Saved: {output_pdb}")
    return True


def main():
    parser = argparse.ArgumentParser(description='Apply ANARCI IMGT numbering to TCR-pMHC structures.')
    parser.add_argument('--input-dir', default='processed_structures',
                        help='Directory containing input PDB files (default: processed_structures)')
    parser.add_argument('--output-dir', default='processed_structures_anarci',
                        help='Directory for renumbered output PDB files (default: processed_structures_anarci)')
    args = parser.parse_args()

    conda_env = os.environ.get('CONDA_DEFAULT_ENV', 'unknown')
    print(f"Active conda environment: {conda_env}")
    if conda_env != 'swifttcr':
        print("WARNING: swifttcr environment not active. Run: conda activate swifttcr")

    if not IMMUNOPDB_PATH.exists():
        print(f"ERROR: ImmunoPDB.py not found at: {IMMUNOPDB_PATH}")
        print("Update IMMUNOPDB_PATH at the top of this script.")
        return 1

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    pdb_files = sorted(input_dir.glob('*.pdb'))
    if not pdb_files:
        print(f"No PDB files found in {input_dir}")
        return 1

    print(f"Input:  {input_dir}  ({len(pdb_files)} files)")
    print(f"Output: {output_dir}")

    success_count = 0
    failed_count = 0

    for pdb_file in pdb_files:
        output_name = pdb_file.name.replace('_renumbered', '_anarci')
        output_file = output_dir / output_name
        try:
            if renumber_tcr_chains(str(pdb_file), str(output_file)):
                success_count += 1
            else:
                failed_count += 1
        except Exception as e:
            print(f"  Failed: {e}")
            failed_count += 1

    print(f"\nDone — success: {success_count}, failed: {failed_count}")
    return 0 if failed_count == 0 else 1


if __name__ == '__main__':
    sys.exit(main())