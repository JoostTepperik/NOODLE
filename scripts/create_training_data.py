#!/usr/bin/env python3
"""
Create training-ready dataset from extracted 7-mer features
- Discretize angles into bins
- Create train/val/test splits
- Compute normalization statistics
"""

import argparse
import h5py
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split


def discretize_angles(angles, n_bins=36):
    """
    Discretize angles into bins
    
    Args:
        angles: Array of angles in degrees [-180, 180]
        n_bins: Number of bins
    
    Returns:
        Bin indices [0, n_bins-1]
    """
    # Create bin edges from -180 to 180
    bin_edges = np.linspace(-180, 180, n_bins + 1)
    
    # Digitize (returns bin indices 1 to n_bins)
    bins = np.digitize(angles, bin_edges) - 1
    
    # Clip to valid range [0, n_bins-1]
    bins = np.clip(bins, 0, n_bins - 1)
    
    return bins


def create_splits(pdb_ids, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, random_seed=42):
    """
    Create train/val/test splits by structure (not by sample)
    
    Args:
        pdb_ids: Array of PDB IDs for each sample
        train_ratio, val_ratio, test_ratio: Split ratios
        random_seed: Random seed for reproducibility
    
    Returns:
        train_mask, val_mask, test_mask: Boolean arrays
    """
    # Decode if bytes
    if isinstance(pdb_ids[0], bytes) or isinstance(pdb_ids[0], np.bytes_):
        pdb_ids_str = np.array([pid.decode('utf-8') for pid in pdb_ids])
    else:
        pdb_ids_str = np.array([str(pid) for pid in pdb_ids])
    
    # Get unique structures
    unique_pdb_ids = np.unique(pdb_ids_str)
    n_structures = len(unique_pdb_ids)
    
    print(f"Creating splits from {n_structures} unique structures...")
    
    # Split structures
    train_structures, temp_structures = train_test_split(
        unique_pdb_ids,
        test_size=(1 - train_ratio),
        random_state=random_seed
    )
    
    val_structures, test_structures = train_test_split(
        temp_structures,
        test_size=test_ratio / (val_ratio + test_ratio),
        random_state=random_seed
    )
    
    print(f"  Train: {len(train_structures)} structures")
    print(f"  Val: {len(val_structures)} structures")
    print(f"  Test: {len(test_structures)} structures")
    
    # Create masks
    train_mask = np.array([pid in train_structures for pid in pdb_ids_str])
    val_mask = np.array([pid in val_structures for pid in pdb_ids_str])
    test_mask = np.array([pid in test_structures for pid in pdb_ids_str])
    
    # Verify
    if train_mask.sum() == 0:
        raise ValueError("Train mask is empty! Check split logic.")
    if val_mask.sum() == 0:
        raise ValueError("Val mask is empty! Check split logic.")
    if test_mask.sum() == 0:
        raise ValueError("Test mask is empty! Check split logic.")
    
    print(f"  Triplet distribution:")
    print(f"    Train: {train_mask.sum():,} ({100*train_mask.sum()/len(pdb_ids):.1f}%)")
    print(f"    Val: {val_mask.sum():,} ({100*val_mask.sum()/len(pdb_ids):.1f}%)")
    print(f"    Test: {test_mask.sum():,} ({100*test_mask.sum()/len(pdb_ids):.1f}%)")
    
    return train_mask, val_mask, test_mask


def compute_statistics(data, mask):
    """
    Compute normalization statistics on training set
    
    Args:
        data: Dict with 'ca_dist_*' arrays
        mask: Boolean mask for training set
    
    Returns:
        Dict with statistics
    """
    if mask.sum() == 0:
        print("⚠ WARNING: Empty mask! Using all data...")
        mask = np.ones(len(data['phi']), dtype=bool)
    
    # For 7-mer: collect all 6 distances
    all_distances = np.concatenate([
        data['ca_dist_i_minus_3'][mask],
        data['ca_dist_i_minus_2'][mask],
        data['ca_dist_i_minus_1'][mask],
        data['ca_dist_i'][mask],
        data['ca_dist_i_plus_1'][mask],
        data['ca_dist_i_plus_2'][mask],
    ])
    
    if len(all_distances) == 0:
        print("⚠ WARNING: No distance data!")
        return {
            'distance_mean': 3.8,
            'distance_std': 0.3,
            'phi_mean': -60.0,
            'phi_std': 50.0,
            'psi_mean': 0.0,
            'psi_std': 80.0,
        }
    
    stats = {
        'distance_mean': float(np.mean(all_distances)),
        'distance_std': float(np.std(all_distances)),
        'phi_mean': float(np.mean(data['phi'][mask])),
        'phi_std': float(np.std(data['phi'][mask])),
        'psi_mean': float(np.mean(data['psi'][mask])),
        'psi_std': float(np.std(data['psi'][mask])),
    }
    
    print(f"  Distance: {stats['distance_mean']:.2f} ± {stats['distance_std']:.2f} Å")
    print(f"  φ: {stats['phi_mean']:.1f}° ± {stats['phi_std']:.1f}°")
    print(f"  ψ: {stats['psi_mean']:.1f}° ± {stats['psi_std']:.1f}°")
    
    return stats


def main():
    parser = argparse.ArgumentParser(description='Create training dataset from 7-mer features')
    parser.add_argument('--input', required=True, help='Input HDF5 file (septamer_dataset.h5)')
    parser.add_argument('--output_dir', required=True, help='Output directory')
    parser.add_argument('--n_bins', type=int, default=36, help='Number of angle bins')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='Train split ratio')
    parser.add_argument('--val_ratio', type=float, default=0.1, help='Validation split ratio')
    parser.add_argument('--test_ratio', type=float, default=0.1, help='Test split ratio')
    parser.add_argument('--random_seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("Processing 7-mer Dataset for Training")
    print("="*60)
    
    # Load raw data
    print(f"\n1. Loading raw data from {args.input}...")
    with h5py.File(args.input, 'r') as f:
        # Check if it's actually 7-mer data
        context_size = f.attrs.get('context_size', 3)
        if context_size != 7:
            print(f"⚠ WARNING: Expected 7-mer data but got {context_size}-mer!")
            print("Proceeding anyway, but verify your input file...")
        
        data = {
            'pdb_ids': f['pdb_ids'][:],
            
            # 7 residue types
            'res_i_minus_3': f['res_i_minus_3'][:],
            'res_i_minus_2': f['res_i_minus_2'][:],
            'res_i_minus_1': f['res_i_minus_1'][:],
            'res_i': f['res_i'][:],
            'res_i_plus_1': f['res_i_plus_1'][:],
            'res_i_plus_2': f['res_i_plus_2'][:],
            'res_i_plus_3': f['res_i_plus_3'][:],
            
            # 6 distances
            'ca_dist_i_minus_3': f['ca_dist_i_minus_3'][:],
            'ca_dist_i_minus_2': f['ca_dist_i_minus_2'][:],
            'ca_dist_i_minus_1': f['ca_dist_i_minus_1'][:],
            'ca_dist_i': f['ca_dist_i'][:],
            'ca_dist_i_plus_1': f['ca_dist_i_plus_1'][:],
            'ca_dist_i_plus_2': f['ca_dist_i_plus_2'][:],
            
            # Angles
            'phi': f['phi'][:],
            'psi': f['psi'][:],
            'omega': f['omega'][:] if 'omega' in f else np.full(len(f['phi'][:]), 180.0),
        }
    
    n_samples = len(data['phi'])
    print(f"  Loaded {n_samples:,} 7-mers")
    
    # Discretize angles
    print(f"\n2. Discretizing angles ({args.n_bins} bins)...")
    phi_bins = discretize_angles(data['phi'], args.n_bins)
    psi_bins = discretize_angles(data['psi'], args.n_bins)
    
    print(f"  φ bins: [{phi_bins.min()}, {phi_bins.max()}]")
    print(f"  ψ bins: [{psi_bins.min()}, {psi_bins.max()}]")
    
    # Create splits
    print(f"\n3. Creating train/val/test splits...")
    train_mask, val_mask, test_mask = create_splits(
        data['pdb_ids'],
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_seed=args.random_seed
    )
    
    # Compute statistics on training set
    print(f"\n4. Computing statistics (on train set)...")
    stats = compute_statistics(data, train_mask)
    
    # Save processed dataset
    print(f"\n5. Saving processed data...")
    output_file = output_dir / 'training_data.h5'
    
    with h5py.File(output_file, 'w') as f:
        # Save all features
        print("  Saving 7 residue types...")
        f.create_dataset('res_i_minus_3', data=data['res_i_minus_3'], compression='gzip')
        f.create_dataset('res_i_minus_2', data=data['res_i_minus_2'], compression='gzip')
        f.create_dataset('res_i_minus_1', data=data['res_i_minus_1'], compression='gzip')
        f.create_dataset('res_i', data=data['res_i'], compression='gzip')
        f.create_dataset('res_i_plus_1', data=data['res_i_plus_1'], compression='gzip')
        f.create_dataset('res_i_plus_2', data=data['res_i_plus_2'], compression='gzip')
        f.create_dataset('res_i_plus_3', data=data['res_i_plus_3'], compression='gzip')
        
        print("  Saving 6 distances...")
        f.create_dataset('ca_dist_i_minus_3', data=data['ca_dist_i_minus_3'], compression='gzip')
        f.create_dataset('ca_dist_i_minus_2', data=data['ca_dist_i_minus_2'], compression='gzip')
        f.create_dataset('ca_dist_i_minus_1', data=data['ca_dist_i_minus_1'], compression='gzip')
        f.create_dataset('ca_dist_i', data=data['ca_dist_i'], compression='gzip')
        f.create_dataset('ca_dist_i_plus_1', data=data['ca_dist_i_plus_1'], compression='gzip')
        f.create_dataset('ca_dist_i_plus_2', data=data['ca_dist_i_plus_2'], compression='gzip')
        
        print("  Saving angles...")
        # Continuous angles (for von Mises)
        f.create_dataset('phi_continuous', data=data['phi'], compression='gzip')
        f.create_dataset('psi_continuous', data=data['psi'], compression='gzip')
        f.create_dataset('omega_continuous', data=data['omega'], compression='gzip')
        
        # Discretized bins (for classification)
        f.create_dataset('phi_bins', data=phi_bins, compression='gzip')
        f.create_dataset('psi_bins', data=psi_bins, compression='gzip')
        
        print("  Saving splits...")
        f.create_dataset('train_mask', data=train_mask, compression='gzip')
        f.create_dataset('val_mask', data=val_mask, compression='gzip')
        f.create_dataset('test_mask', data=test_mask, compression='gzip')
        
        print("  Saving metadata...")
        f.create_dataset('pdb_ids', data=data['pdb_ids'], compression='gzip')
        
        # Metadata attributes
        f.attrs['n_samples'] = n_samples
        f.attrs['context_size'] = 7
        f.attrs['n_residues'] = 7
        f.attrs['n_distances'] = 6
        f.attrs['n_phi_bins'] = args.n_bins
        f.attrs['n_psi_bins'] = args.n_bins
        
        # Statistics
        for key, value in stats.items():
            f.attrs[key] = value
    
    # Save config
    import json
    config = {
        'n_samples': n_samples,
        'context_size': 7,
        'n_residues': 7,
        'n_distances': 6,
        'n_phi_bins': args.n_bins,
        'n_psi_bins': args.n_bins,
        'train_samples': int(train_mask.sum()),
        'val_samples': int(val_mask.sum()),
        'test_samples': int(test_mask.sum()),
        **stats,
    }
    
    with open(output_dir / 'data_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\n✓ Dataset processing complete!")
    print(f"  Output: {output_file}")
    print(f"  Config: {output_dir / 'data_config.json'}")
    print(f"\nNext step:")
    print(f"  python training_jax/train_jax.py --data {output_file}")


if __name__ == '__main__':
    main()