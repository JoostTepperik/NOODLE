"""
data/dataloader.py

FAST version - loads contiguous blocks, no fancy indexing
"""

import h5py
import jax.numpy as jnp
import numpy as np


class VariableContextDataLoader:
    """
    Fast dataloader using contiguous block loading
    """
    
    def __init__(self, h5_path, split='train', batch_size=512, shuffle=True, max_context=None):
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        print(f"Loading {split} data to RAM...", flush=True)
        
        with h5py.File(h5_path, 'r') as f:
            # Get split indices
            if split == 'train':
                indices = f['train_indices'][:]
            elif split == 'val':
                indices = f['val_indices'][:]
            elif split == 'test':
                indices = f['test_indices'][:]
            else:
                raise ValueError(f"Unknown split: {split}")
            
            n_indices = len(indices)
            print(f"  Found {n_indices:,} indices", flush=True)
            
            # Use min/max to safely handle unsorted indices
            start_idx = int(indices.min())
            end_idx = int(indices.max()) + 1
            
            print(f"  Loading contiguous block [{start_idx}:{end_idx}]...", flush=True)
            
            block_residues = f['residues'][start_idx:end_idx]
            block_masks = f['masks'][start_idx:end_idx]
            block_phi = f['phi'][start_idx:end_idx]
            block_psi = f['psi'][start_idx:end_idx]
            
            print(f"  Loaded {end_idx - start_idx:,} samples in block", flush=True)
            
            if n_indices == (end_idx - start_idx):
                print(f"  Split is contiguous - using directly!", flush=True)
                self.residues = block_residues
                self.masks = block_masks
                self.phi = block_phi
                self.psi = block_psi
            else:
                print(f"  Selecting {n_indices:,} from block...", flush=True)
                relative_indices = indices - start_idx
                self.residues = block_residues[relative_indices]
                self.masks = block_masks[relative_indices]
                self.phi = block_phi[relative_indices]
                self.psi = block_psi[relative_indices]
            
            dataset_max_context = int(f.attrs['max_context'])
        
        # Use requested max_context if provided, otherwise use dataset's
        if max_context is not None:
            if max_context > dataset_max_context:
                raise ValueError(
                    f"Requested max_context={max_context} exceeds "
                    f"dataset max_context={dataset_max_context}"
                )
            self.max_context = max_context
        else:
            self.max_context = dataset_max_context

        self.n_samples = len(self.residues)
        self.n_batches = (self.n_samples + batch_size - 1) // batch_size
        
        size_mb = self.residues.nbytes / 1e6
        print(f"  ✓ Loaded {self.n_samples:,} samples ({size_mb:.0f} MB) "
              f"[context: {self.max_context}/{dataset_max_context}]", flush=True)
    
    def __iter__(self):
        if self.shuffle:
            perm = np.random.permutation(self.n_samples)
        else:
            perm = np.arange(self.n_samples)
        
        for i in range(0, self.n_samples, self.batch_size):
            idx = perm[i:i + self.batch_size]
            
            yield {
                'residues': jnp.array(self.residues[idx, :self.max_context]),
                'masks': jnp.array(self.masks[idx, :self.max_context]),
                'phi': jnp.array(self.phi[idx]) * np.pi / 180,
                'psi': jnp.array(self.psi[idx]) * np.pi / 180,
            }
    
    def __len__(self):
        return self.n_batches


def create_dataloaders(h5_path, batch_size=512, max_context=None):
    """Create train, val, test dataloaders"""
    
    print("\n" + "="*70, flush=True)
    print("CREATING DATALOADERS", flush=True)
    print("="*70 + "\n", flush=True)
    
    train_loader = VariableContextDataLoader(
        h5_path, split='train', batch_size=batch_size, shuffle=True,
        max_context=max_context
    )
    
    val_loader = VariableContextDataLoader(
        h5_path, split='val', batch_size=batch_size, shuffle=False,
        max_context=max_context
    )
    
    test_loader = VariableContextDataLoader(
        h5_path, split='test', batch_size=batch_size, shuffle=False,
        max_context=max_context
    )
    
    print("\n" + "="*70, flush=True)
    print("DATALOADERS READY", flush=True)
    print("="*70 + "\n", flush=True)
    
    return train_loader, val_loader, test_loader