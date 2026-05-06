"""
Process PDB structures and extract 7-mer torsion angle features
"""

import argparse
from pathlib import Path
from tqdm import tqdm

# Import classes
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_processing.extractors import SevenMerTorsionExtractor
from data_processing.dataset_creator import DatasetCreator
from data_processing.downloader import RSCBDownloader, get_diverse_from_rcsb


def main():
    parser = argparse.ArgumentParser(description='Extract 7-mer torsion features from PDB structures')
    parser.add_argument('--pdb_list', help='Path to file with PDB IDs (one per line)')
    parser.add_argument('--n_structures', type=int, default=1000, help='Number of structures to process')
    parser.add_argument('--use_diverse', action='store_true', help='Automatically fetch diverse structures')
    parser.add_argument('--output_dir', default='data/medium_7mer', help='Output directory')
    parser.add_argument('--max_workers', type=int, default=5, help='Parallel download workers')
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("7-mer Torsion Angle Feature Extraction")
    print("="*60)
    
    # Get PDB IDs
    if args.use_diverse:
        print("\nFetching diverse structures from RCSB...")
        pdb_ids = get_diverse_from_rcsb(target_count=args.n_structures)
    elif args.pdb_list:
        print(f"\nLoading PDB IDs from {args.pdb_list}...")
        with open(args.pdb_list) as f:
            pdb_ids = [line.strip().lower() for line in f if line.strip()]
        pdb_ids = pdb_ids[:args.n_structures]
    else:
        raise ValueError("Must specify either --pdb_list or --use_diverse")
    
    print(f"Processing {len(pdb_ids)} structures")
    
    # Download structures
    print("\nDownloading structures...")
    downloader = RSCBDownloader(output_dir / 'pdb_files')
    downloaded_files = downloader.download_structures(pdb_ids, max_workers=args.max_workers)
    
    print(f"\nSuccessfully downloaded: {len(downloaded_files)} structures")
    
    if len(downloaded_files) == 0:
        print("ERROR: No structures downloaded!")
        return
    
    # Extract 7-mer features
    print("\nExtracting 7-mer features...")
    extractor = SevenMerTorsionExtractor()
    
    all_features = []
    failed = []
    
    for pdb_file in tqdm(downloaded_files, desc='Extracting'):
        try:
            features = extractor.extract_from_structure(pdb_file)
            all_features.extend(features)
        except Exception as e:
            print(f"  ✗ Failed to extract from {pdb_file.name}: {e}")
            failed.append(pdb_file)
            continue
    
    print(f"\nExtraction complete:")
    print(f"  Successful: {len(downloaded_files) - len(failed)} structures")
    print(f"  Failed: {len(failed)} structures")
    print(f"  Total 7-mers: {len(all_features):,}")
    
    if len(all_features) == 0:
        print("ERROR: No features extracted!")
        return
    
    # Create HDF5 dataset
    print("\nCreating HDF5 dataset...")
    creator = DatasetCreator(output_dir)
    hdf5_file = creator.create_dataset(all_features, output_filename='septamer_dataset.h5')
    
    print("\n" + "="*60)
    print("Processing Complete!")
    print("="*60)
    print(f"\nOutput files:")
    print(f"  PDB files: {output_dir / 'pdb_files'}")
    print(f"  HDF5 dataset: {hdf5_file}")
    print(f"\nNext step:")
    print(f"  python scripts/create_training_data.py --input {hdf5_file} --output_dir data/training_7mer")


if __name__ == '__main__':
    main()