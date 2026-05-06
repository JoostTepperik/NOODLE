"""
One-time script: pre-sort the indices in the HDF5 file
"""

import h5py
import numpy as np

h5_path = 'data/training_variable_context/training_data.h5'

print("Sorting indices in-place...")

with h5py.File(h5_path, 'r+') as f:
    for split in ['train_indices', 'val_indices', 'test_indices']:
        print(f"  Sorting {split}...")
        indices = f[split][:]
        
        # Check if already sorted
        if np.all(indices[:-1] <= indices[1:]):
            print(f"    Already sorted!")
        else:
            # Sort and save
            sorted_indices = np.sort(indices)
            del f[split]
            f.create_dataset(split, data=sorted_indices, compression='gzip')
            print(f"    Sorted and saved!")

print("Done!")