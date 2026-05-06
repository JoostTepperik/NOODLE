"""
diagnose_loading.py

Find out what's taking so long
"""

import h5py
import numpy as np
import time

h5_path = '/scratch-shared/jtepperik/data/training_variable_context/training_data.h5'

print("="*70)
print("DIAGNOSTIC: HDF5 Loading Performance")
print("="*70)

with h5py.File(h5_path, 'r') as f:
    print(f"\n1. File opened successfully")
    print(f"   Datasets: {list(f.keys())}")
    
    # Check indices
    print(f"\n2. Loading train_indices...")
    start = time.time()
    indices = f['train_indices'][:]
    elapsed = time.time() - start
    print(f"   Time: {elapsed:.2f}s")
    print(f"   Shape: {indices.shape}")
    print(f"   First 10: {indices[:10]}")
    print(f"   Sorted: {np.all(indices[:-1] <= indices[1:])}")
    
    # Check data shape
    print(f"\n3. Checking dataset shapes...")
    print(f"   residues: {f['residues'].shape}")
    print(f"   masks: {f['masks'].shape}")
    print(f"   phi: {f['phi'].shape}")
    print(f"   psi: {f['psi'].shape}")
    
    # Test small load
    print(f"\n4. Testing small sequential load (first 1000 samples)...")
    test_indices = np.arange(1000)
    start = time.time()
    test_data = f['residues'][test_indices]
    elapsed = time.time() - start
    print(f"   Time: {elapsed:.2f}s")
    print(f"   Speed: {len(test_indices)/elapsed:.0f} samples/sec")
    
    # Test full load with SORTED indices
    print(f"\n5. Testing FULL load with sorted indices...")
    print(f"   This is what's taking forever in your job!")
    sorted_indices = np.sort(indices)
    
    print(f"   Loading residues...")
    start = time.time()
    residues = f['residues'][sorted_indices]
    elapsed = time.time() - start
    print(f"   Time: {elapsed:.2f}s ({len(sorted_indices)/elapsed:.0f} samples/sec)")
    
    print(f"   Loading masks...")
    start = time.time()
    masks = f['masks'][sorted_indices]
    elapsed = time.time() - start
    print(f"   Time: {elapsed:.2f}s")
    
    print(f"   Loading phi...")
    start = time.time()
    phi = f['phi'][sorted_indices]
    elapsed = time.time() - start
    print(f"   Time: {elapsed:.2f}s")
    
    print(f"   Loading psi...")
    start = time.time()
    psi = f['psi'][sorted_indices]
    elapsed = time.time() - start
    print(f"   Time: {elapsed:.2f}s")
    
    print(f"\n   Total data loaded: {residues.nbytes / 1e9:.2f} GB")

print("\n" + "="*70)
print("DIAGNOSTIC COMPLETE")
print("="*70)