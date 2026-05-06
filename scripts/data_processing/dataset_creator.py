"""
Create HDF5 dataset from extracted 7-mer features
"""

import h5py
import numpy as np
from pathlib import Path
from tqdm import tqdm


class DatasetCreator:
    """Create HDF5 dataset from 7-mer torsion angle features"""
    
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def create_dataset(self, all_features, output_filename='septamer_dataset.h5'):
        """
        Create HDF5 dataset from list of 7-mer feature dictionaries
        
        Args:
            all_features: List of dicts from SevenMerTorsionExtractor
            output_filename: Name of output HDF5 file
        
        Returns:
            Path to created HDF5 file
        """
        if len(all_features) == 0:
            raise ValueError("No features provided!")
        
        output_file = self.output_dir / output_filename
        
        print(f"\nCreating HDF5 dataset with {len(all_features):,} 7-mers...")
        print(f"Output: {output_file}")
        
        # Convert list of dicts to dict of arrays
        print("Converting to arrays...")
        data = self._convert_to_arrays(all_features)
        
        # Save to HDF5
        print("Writing to HDF5...")
        self._save_hdf5(data, output_file)
        
        print(f"✓ Dataset saved: {output_file}")
        print(f"  Total 7-mers: {len(data['phi']):,}")
        print(f"  Unique PDB IDs: {len(np.unique(data['pdb_ids']))}")
        
        return output_file
    
    def _convert_to_arrays(self, all_features):
        """Convert list of feature dicts to dict of numpy arrays"""
        
        n_samples = len(all_features)
        
        # Initialize arrays
        data = {
            'pdb_ids': [],
            'chain_ids': [],
            'residue_indices': np.zeros(n_samples, dtype=np.int32),
            
            # 7 residue types
            'res_i_minus_3': np.zeros(n_samples, dtype=np.int8),
            'res_i_minus_2': np.zeros(n_samples, dtype=np.int8),
            'res_i_minus_1': np.zeros(n_samples, dtype=np.int8),
            'res_i': np.zeros(n_samples, dtype=np.int8),
            'res_i_plus_1': np.zeros(n_samples, dtype=np.int8),
            'res_i_plus_2': np.zeros(n_samples, dtype=np.int8),
            'res_i_plus_3': np.zeros(n_samples, dtype=np.int8),
            
            # Torsion angles (center residue)
            'phi': np.zeros(n_samples, dtype=np.float32),
            'psi': np.zeros(n_samples, dtype=np.float32),
            'omega': np.zeros(n_samples, dtype=np.float32),
            
            # 6 CA-CA distances
            'ca_dist_i_minus_3': np.zeros(n_samples, dtype=np.float32),
            'ca_dist_i_minus_2': np.zeros(n_samples, dtype=np.float32),
            'ca_dist_i_minus_1': np.zeros(n_samples, dtype=np.float32),
            'ca_dist_i': np.zeros(n_samples, dtype=np.float32),
            'ca_dist_i_plus_1': np.zeros(n_samples, dtype=np.float32),
            'ca_dist_i_plus_2': np.zeros(n_samples, dtype=np.float32),
        }
        
        # Fill arrays
        for i, feature in enumerate(tqdm(all_features, desc='Converting')):
            data['pdb_ids'].append(feature['pdb_id'])
            data['chain_ids'].append(feature.get('chain_id', 'A'))
            data['residue_indices'][i] = feature['residue_index']
            
            # 7 residue types
            data['res_i_minus_3'][i] = feature['res_i_minus_3']
            data['res_i_minus_2'][i] = feature['res_i_minus_2']
            data['res_i_minus_1'][i] = feature['res_i_minus_1']
            data['res_i'][i] = feature['res_i']
            data['res_i_plus_1'][i] = feature['res_i_plus_1']
            data['res_i_plus_2'][i] = feature['res_i_plus_2']
            data['res_i_plus_3'][i] = feature['res_i_plus_3']
            
            # Torsion angles
            data['phi'][i] = feature['phi']
            data['psi'][i] = feature['psi']
            data['omega'][i] = feature.get('omega', 180.0)  # Default if missing
            
            # 6 distances
            data['ca_dist_i_minus_3'][i] = feature['ca_dist_i_minus_3']
            data['ca_dist_i_minus_2'][i] = feature['ca_dist_i_minus_2']
            data['ca_dist_i_minus_1'][i] = feature['ca_dist_i_minus_1']
            data['ca_dist_i'][i] = feature['ca_dist_i']
            data['ca_dist_i_plus_1'][i] = feature['ca_dist_i_plus_1']
            data['ca_dist_i_plus_2'][i] = feature['ca_dist_i_plus_2']
        
        # Convert string lists to numpy arrays
        data['pdb_ids'] = np.array(data['pdb_ids'], dtype='S10')
        data['chain_ids'] = np.array(data['chain_ids'], dtype='S10')
        
        return data
    
    def _save_hdf5(self, data, output_file):
        """Save data dict to HDF5 file"""
        
        with h5py.File(output_file, 'w') as f:
            # Save all arrays
            for key, arr in data.items():
                if key in ['pdb_ids', 'chain_ids']:
                    # String arrays
                    f.create_dataset(key, data=arr, compression='gzip')
                else:
                    # Numeric arrays
                    f.create_dataset(key, data=arr, compression='gzip', compression_opts=4)
            
            # Save metadata
            f.attrs['n_samples'] = len(data['phi'])
            f.attrs['n_unique_structures'] = len(np.unique(data['pdb_ids']))
            f.attrs['context_size'] = 7
            f.attrs['n_residues'] = 7
            f.attrs['n_distances'] = 6
            
            # Save statistics
            all_dists = np.concatenate([
                data['ca_dist_i_minus_3'],
                data['ca_dist_i_minus_2'],
                data['ca_dist_i_minus_1'],
                data['ca_dist_i'],
                data['ca_dist_i_plus_1'],
                data['ca_dist_i_plus_2'],
            ])
            
            f.attrs['distance_mean'] = float(np.mean(all_dists))
            f.attrs['distance_std'] = float(np.std(all_dists))
            f.attrs['phi_mean'] = float(np.mean(data['phi']))
            f.attrs['phi_std'] = float(np.std(data['phi']))
            f.attrs['psi_mean'] = float(np.mean(data['psi']))
            f.attrs['psi_std'] = float(np.std(data['psi']))
            
            print(f"\n  Statistics:")
            print(f"    Distance: {f.attrs['distance_mean']:.3f} ± {f.attrs['distance_std']:.3f} Å")
            print(f"    φ: {f.attrs['phi_mean']:.1f}° ± {f.attrs['phi_std']:.1f}°")
            print(f"    ψ: {f.attrs['psi_mean']:.1f}° ± {f.attrs['psi_std']:.1f}°")


def load_dataset(hdf5_file):
    """
    Load and inspect HDF5 dataset
    
    Args:
        hdf5_file: Path to HDF5 file
    
    Returns:
        Dict with dataset info
    """
    with h5py.File(hdf5_file, 'r') as f:
        info = {
            'n_samples': f.attrs['n_samples'],
            'n_structures': f.attrs['n_unique_structures'],
            'context_size': f.attrs.get('context_size', 3),
            'datasets': list(f.keys()),
            'attributes': dict(f.attrs),
        }
        
        # Get sample data
        info['sample'] = {
            'res_i_minus_3': f['res_i_minus_3'][0],
            'res_i_minus_2': f['res_i_minus_2'][0],
            'res_i_minus_1': f['res_i_minus_1'][0],
            'res_i': f['res_i'][0],
            'res_i_plus_1': f['res_i_plus_1'][0],
            'res_i_plus_2': f['res_i_plus_2'][0],
            'res_i_plus_3': f['res_i_plus_3'][0],
            'phi': f['phi'][0],
            'psi': f['psi'][0],
        }
    
    return info


# Test function
if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python dataset.py <path_to_septamer_dataset.h5>")
        sys.exit(1)
    
    hdf5_file = sys.argv[1]
    
    print("="*60)
    print("Inspecting 7-mer Dataset")
    print("="*60)
    
    info = load_dataset(hdf5_file)
    
    print(f"\nDataset Info:")
    print(f"  Samples: {info['n_samples']:,}")
    print(f"  Structures: {info['n_structures']}")
    print(f"  Context size: {info['context_size']}-mer")
    
    print(f"\nDatasets in HDF5:")
    for ds in info['datasets']:
        print(f"  - {ds}")
    
    print(f"\nStatistics:")
    for key, value in info['attributes'].items():
        if key not in ['n_samples', 'n_unique_structures', 'context_size', 'n_residues', 'n_distances']:
            print(f"  {key}: {value}")
    
    print(f"\nSample (first 7-mer):")
    sample = info['sample']
    print(f"  Residues: [{sample['res_i_minus_3']}, {sample['res_i_minus_2']}, {sample['res_i_minus_1']}, "
          f"{sample['res_i']}, {sample['res_i_plus_1']}, {sample['res_i_plus_2']}, {sample['res_i_plus_3']}]")
    print(f"  φ: {sample['phi']:.1f}°")
    print(f"  ψ: {sample['psi']:.1f}°")
    
    print("\n✓ Dataset inspection complete!")