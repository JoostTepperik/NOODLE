#!/usr/bin/env python3
"""
Test script to verify data processing works
"""

from pathlib import Path
from data_processing import (
    PDBRedoDownloader,
    StructureFilter,
    TorsionExtractor,
    TorsionDataset
)

def test_pipeline():
    """Test the pipeline with a single structure"""
    
    # Create output directory
    output_dir = Path('data/test')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Download a single test structure
    print("Step 1: Downloading test structure (1ubq)...")
    downloader = PDBRedoDownloader(output_dir / 'pdb_files')
    
    test_pdb = '1ubq'  # Ubiquitin - small, well-studied protein
    pdb_file = downloader.download_structure(test_pdb)
    
    if pdb_file is None:
        print("ERROR: Download failed!")
        print("Trying alternative download method...")
        
        # Fallback: Download from regular PDB
        import requests
        pdb_file = output_dir / 'pdb_files' / f'{test_pdb}.pdb'
        pdb_file.parent.mkdir(parents=True, exist_ok=True)
        
        url = f'https://files.rcsb.org/download/{test_pdb.upper()}.pdb'
        response = requests.get(url)
        
        if response.status_code == 200:
            with open(pdb_file, 'w') as f:
                f.write(response.text)
            print(f"Downloaded from RCSB PDB: {pdb_file}")
        else:
            print("ERROR: Both download methods failed!")
            return
    
    print(f"✓ Downloaded: {pdb_file}")
    
    # Step 2: Check quality
    print("\nStep 2: Checking structure quality...")
    filter_obj = StructureFilter()
    
    try:
        metadata = filter_obj.parse_pdb(pdb_file)
        print(f"  Resolution: {metadata['resolution']} Å")
        print(f"  Method: {metadata['method']}")
        print(f"  R-free: {metadata['r_free']}")
        
        filters, passed = filter_obj.check_quality(metadata)
        print(f"  Quality filters: {filters}")
        print(f"  Passed: {passed}")
        
        if passed:
            issues = filter_obj.check_completeness(metadata['structure'])
            print(f"  Completeness issues: {len(issues)}")
    except Exception as e:
        print(f"ERROR in quality check: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 3: Extract torsion angles
    print("\nStep 3: Extracting torsion angles...")
    extractor = TorsionExtractor()
    
    try:
        triplets = extractor.extract_from_structure(pdb_file)
        print(f"✓ Extracted {len(triplets)} triplets")
        
        if len(triplets) > 0:
            print("\nExample triplet:")
            example = triplets[0]
            for key, value in example.items():
                print(f"  {key}: {value}")
    except Exception as e:
        print(f"ERROR in extraction: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 4: Save to HDF5
    print("\nStep 4: Saving to HDF5...")
    dataset = TorsionDataset(output_dir / 'test_dataset.h5')
    
    try:
        dataset.save_triplets(triplets)
        print("✓ Saved successfully!")
        
        # Test loading
        print("\nStep 5: Testing data loading...")
        loaded_data = dataset.load_subset(indices=slice(0, 5))
        print(f"✓ Loaded {len(loaded_data['phi'])} samples")
        print(f"  First phi angle: {loaded_data['phi'][0]:.2f}°")
        print(f"  First psi angle: {loaded_data['psi'][0]:.2f}°")
        
    except Exception as e:
        print(f"ERROR in saving/loading: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n" + "="*60)
    print("✓ ALL TESTS PASSED!")
    print("="*60)

if __name__ == '__main__':
    test_pipeline()