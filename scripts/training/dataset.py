"""
PyTorch Dataset for 7-mer torsion angle prediction
"""

import h5py
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


class TorsionAngleDataset(Dataset):
    """
    Dataset for 7-mer torsion angle prediction
    
    Loads 7 residue types + 6 distances + angles
    """
    
    def __init__(self, hdf5_file, split='train', normalize_distances=True):
        """
        Args:
            hdf5_file: Path to HDF5 file with 7-mer data
            split: 'train', 'val', or 'test'
            normalize_distances: Whether to normalize CA-CA distances
        """
        self.hdf5_file = hdf5_file
        self.split = split
        self.normalize_distances = normalize_distances
        
        # Load data
        print(f"Loading {split} set...")
        with h5py.File(hdf5_file, 'r') as f:
            # Get split mask
            mask = f[f'{split}_mask'][:]
            
            # Load 7 residue types
            self.res_i_minus_3 = torch.from_numpy(f['res_i_minus_3'][:][mask]).long()
            self.res_i_minus_2 = torch.from_numpy(f['res_i_minus_2'][:][mask]).long()
            self.res_i_minus_1 = torch.from_numpy(f['res_i_minus_1'][:][mask]).long()
            self.res_i = torch.from_numpy(f['res_i'][:][mask]).long()
            self.res_i_plus_1 = torch.from_numpy(f['res_i_plus_1'][:][mask]).long()
            self.res_i_plus_2 = torch.from_numpy(f['res_i_plus_2'][:][mask]).long()
            self.res_i_plus_3 = torch.from_numpy(f['res_i_plus_3'][:][mask]).long()
            
            # Load 6 CA-CA distances
            self.ca_dist_i_minus_3 = torch.from_numpy(f['ca_dist_i_minus_3'][:][mask]).float()
            self.ca_dist_i_minus_2 = torch.from_numpy(f['ca_dist_i_minus_2'][:][mask]).float()
            self.ca_dist_i_minus_1 = torch.from_numpy(f['ca_dist_i_minus_1'][:][mask]).float()
            self.ca_dist_i = torch.from_numpy(f['ca_dist_i'][:][mask]).float()
            self.ca_dist_i_plus_1 = torch.from_numpy(f['ca_dist_i_plus_1'][:][mask]).float()
            self.ca_dist_i_plus_2 = torch.from_numpy(f['ca_dist_i_plus_2'][:][mask]).float()
            
            # Load angles
            # Option 1: Load bins (for classification models)
            if 'phi_bins' in f:
                self.phi_bins = torch.from_numpy(f['phi_bins'][:][mask]).long()
                self.psi_bins = torch.from_numpy(f['psi_bins'][:][mask]).long()
            
            # Option 2: Load continuous angles (for von Mises models)
            if 'phi_continuous' in f:
                phi_degrees = f['phi_continuous'][:][mask]
                psi_degrees = f['psi_continuous'][:][mask]
                
                # Convert to radians
                self.phi_radians = torch.from_numpy(phi_degrees * np.pi / 180.0).float()
                self.psi_radians = torch.from_numpy(psi_degrees * np.pi / 180.0).float()
            
            # Load metadata
            if 'n_phi_bins' in f.attrs:
                self.n_phi_bins = f.attrs['n_phi_bins']
                self.n_psi_bins = f.attrs['n_psi_bins']
            
            # Load normalization statistics if available
            if 'distance_mean' in f.attrs and normalize_distances:
                self.dist_mean = f.attrs['distance_mean']
                self.dist_std = f.attrs['distance_std']
            else:
                # Default values
                self.dist_mean = 3.8
                self.dist_std = 0.3
        
        # Normalize distances
        if normalize_distances:
            self.ca_dist_i_minus_3 = (self.ca_dist_i_minus_3 - self.dist_mean) / self.dist_std
            self.ca_dist_i_minus_2 = (self.ca_dist_i_minus_2 - self.dist_mean) / self.dist_std
            self.ca_dist_i_minus_1 = (self.ca_dist_i_minus_1 - self.dist_mean) / self.dist_std
            self.ca_dist_i = (self.ca_dist_i - self.dist_mean) / self.dist_std
            self.ca_dist_i_plus_1 = (self.ca_dist_i_plus_1 - self.dist_mean) / self.dist_std
            self.ca_dist_i_plus_2 = (self.ca_dist_i_plus_2 - self.dist_mean) / self.dist_std
        
        print(f"Loaded {split} set: {len(self)} samples")
    
    def __len__(self):
        return len(self.phi_radians)
    
    def __getitem__(self, idx):
        """
        Get a single 7-mer sample
        
        Returns:
            features: Dict with 7 residue types + 6 distances
            targets: Dict with angles (radians and/or bins)
        """
        features = {
            # 7 residue types
            'res_i_minus_3': self.res_i_minus_3[idx],
            'res_i_minus_2': self.res_i_minus_2[idx],
            'res_i_minus_1': self.res_i_minus_1[idx],
            'res_i': self.res_i[idx],
            'res_i_plus_1': self.res_i_plus_1[idx],
            'res_i_plus_2': self.res_i_plus_2[idx],
            'res_i_plus_3': self.res_i_plus_3[idx],
            
            # 6 CA-CA distances
            'ca_dist_i_minus_3': self.ca_dist_i_minus_3[idx],
            'ca_dist_i_minus_2': self.ca_dist_i_minus_2[idx],
            'ca_dist_i_minus_1': self.ca_dist_i_minus_1[idx],
            'ca_dist_i': self.ca_dist_i[idx],
            'ca_dist_i_plus_1': self.ca_dist_i_plus_1[idx],
            'ca_dist_i_plus_2': self.ca_dist_i_plus_2[idx],
        }
        
        targets = {}
        
        # Add bins if available
        if hasattr(self, 'phi_bins'):
            targets['phi_bin'] = self.phi_bins[idx]
            targets['psi_bin'] = self.psi_bins[idx]
        
        # Add radians if available
        if hasattr(self, 'phi_radians'):
            targets['phi_rad'] = self.phi_radians[idx]
            targets['psi_rad'] = self.psi_radians[idx]
        
        return features, targets


def create_dataloaders(hdf5_file, batch_size=1024, num_workers=4, normalize_distances=True):
    """
    Create train/val/test dataloaders for 7-mer dataset
    
    Args:
        hdf5_file: Path to training_data.h5
        batch_size: Batch size
        num_workers: Number of data loading workers
        normalize_distances: Whether to normalize distances
    
    Returns:
        train_loader, val_loader, test_loader
    """
    # Create datasets
    train_dataset = TorsionAngleDataset(hdf5_file, split='train', 
                                            normalize_distances=normalize_distances)
    val_dataset = TorsionAngleDataset(hdf5_file, split='val',
                                          normalize_distances=normalize_distances)
    test_dataset = TorsionAngleDataset(hdf5_file, split='test',
                                           normalize_distances=normalize_distances)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


# Test/debug function
if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python dataset.py <path_to_training_data.h5>")
        sys.exit(1)
    
    hdf5_file = sys.argv[1]
    
    print("="*60)
    print("Testing 7-mer Dataset")
    print("="*60)
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        hdf5_file,
        batch_size=32,
        num_workers=0  # Use 0 for debugging
    )
    
    # Test a batch
    print("\nTesting train loader...")
    for features, targets in train_loader:
        print("\nFeatures:")
        print(f"  res_i_minus_3: {features['res_i_minus_3'].shape} (dtype: {features['res_i_minus_3'].dtype})")
        print(f"  res_i_minus_2: {features['res_i_minus_2'].shape}")
        print(f"  res_i_minus_1: {features['res_i_minus_1'].shape}")
        print(f"  res_i: {features['res_i'].shape}")
        print(f"  res_i_plus_1: {features['res_i_plus_1'].shape}")
        print(f"  res_i_plus_2: {features['res_i_plus_2'].shape}")
        print(f"  res_i_plus_3: {features['res_i_plus_3'].shape}")
        print(f"  ca_dist_i_minus_3: {features['ca_dist_i_minus_3'].shape} (dtype: {features['ca_dist_i_minus_3'].dtype})")
        print(f"  ca_dist_i_minus_2: {features['ca_dist_i_minus_2'].shape}")
        print(f"  ca_dist_i_minus_1: {features['ca_dist_i_minus_1'].shape}")
        print(f"  ca_dist_i: {features['ca_dist_i'].shape}")
        print(f"  ca_dist_i_plus_1: {features['ca_dist_i_plus_1'].shape}")
        print(f"  ca_dist_i_plus_2: {features['ca_dist_i_plus_2'].shape}")
        
        print("\nTargets:")
        if 'phi_rad' in targets:
            print(f"  phi_rad: {targets['phi_rad'].shape} (dtype: {targets['phi_rad'].dtype})")
            print(f"  psi_rad: {targets['psi_rad'].shape}")
            print(f"  phi_rad range: [{targets['phi_rad'].min():.3f}, {targets['phi_rad'].max():.3f}]")
            print(f"  psi_rad range: [{targets['psi_rad'].min():.3f}, {targets['psi_rad'].max():.3f}]")
        
        if 'phi_bin' in targets:
            print(f"  phi_bin: {targets['phi_bin'].shape} (dtype: {targets['phi_bin'].dtype})")
            print(f"  psi_bin: {targets['psi_bin'].shape}")
            print(f"  phi_bin range: [{targets['phi_bin'].min()}, {targets['phi_bin'].max()}]")
            print(f"  psi_bin range: [{targets['psi_bin'].min()}, {targets['psi_bin'].max()}]")
        
        print("\nExample values (first 3 samples):")
        for i in range(min(3, features['res_i'].size(0))):
            print(f"\nSample {i}:")
            print(f"  Residues: {features['res_i_minus_3'][i].item()}, {features['res_i_minus_2'][i].item()}, "
                  f"{features['res_i_minus_1'][i].item()}, {features['res_i'][i].item()}, "
                  f"{features['res_i_plus_1'][i].item()}, {features['res_i_plus_2'][i].item()}, "
                  f"{features['res_i_plus_3'][i].item()}")
            print(f"  Distances: {features['ca_dist_i_minus_3'][i].item():.3f}, "
                  f"{features['ca_dist_i_minus_2'][i].item():.3f}, {features['ca_dist_i_minus_1'][i].item():.3f}, "
                  f"{features['ca_dist_i'][i].item():.3f}, {features['ca_dist_i_plus_1'][i].item():.3f}, "
                  f"{features['ca_dist_i_plus_2'][i].item():.3f}")
            if 'phi_rad' in targets:
                print(f"  φ: {targets['phi_rad'][i].item():.3f} rad ({targets['phi_rad'][i].item()*180/3.14159:.1f}°)")
                print(f"  ψ: {targets['psi_rad'][i].item():.3f} rad ({targets['psi_rad'][i].item()*180/3.14159:.1f}°)")
        
        break  # Only test first batch
    
    print("\n" + "="*60)
    print("✓ Dataset test passed!")
    print("="*60)