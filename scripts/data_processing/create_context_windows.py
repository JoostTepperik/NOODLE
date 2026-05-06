"""
data/create_context_windows.py

Parallel with direct HDF5 reads per worker - no shared memory overhead
"""

import h5py
import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys
from multiprocessing import Pool


def process_structure_chunk_h5(args):
    """
    Each worker reads its own slice directly from HDF5.
    Safe for concurrent reads.
    """
    h5_path, boundaries_chunk, half_context = args

    residues_list = []
    masks_list = []
    phi_list = []
    psi_list = []

    if not boundaries_chunk:
        return residues_list, masks_list, phi_list, psi_list

    # Load only the data range this worker needs
    global_start = boundaries_chunk[0][0]
    global_end = boundaries_chunk[-1][1]

    with h5py.File(h5_path, 'r') as f:
        residue_types = f['residue_types'][global_start:global_end]
        phi = f['phi'][global_start:global_end]
        psi = f['psi'][global_start:global_end]

    for start, end in boundaries_chunk:
        # Translate to local indices
        local_start = start - global_start
        local_end = end - global_start

        for i in range(local_start, local_end):
            if np.isnan(phi[i]) or np.isnan(psi[i]):
                continue

            context_residues = []
            context_mask = []

            for offset in range(-half_context, half_context + 1):
                idx = i + offset
                if local_start <= idx < local_end:
                    context_residues.append(int(residue_types[idx]))
                    context_mask.append(True)
                else:
                    context_residues.append(20)
                    context_mask.append(False)

            residues_list.append(context_residues)
            masks_list.append(context_mask)
            phi_list.append(float(phi[i]))
            psi_list.append(float(psi[i]))

    return residues_list, masks_list, phi_list, psi_list


def create_variable_context_dataset(
    raw_h5_path,
    output_path,
    max_context=7,
    train_ratio=0.85,
    val_ratio=0.075,
    test_ratio=0.075,
    n_workers=16
):
    print(f"Creating variable context dataset (max_context={max_context})...")
    print(f"Using {n_workers} workers with direct HDF5 reads")
    print(f"Reading from: {raw_h5_path}")

    if not Path(raw_h5_path).exists():
        print(f"ERROR: Input file does not exist: {raw_h5_path}")
        sys.exit(1)

    half_context = max_context // 2
    context_size = 2 * half_context + 1

    # Load only pdb_ids to find boundaries (small - just ints)
    print("\nFinding structure boundaries...")
    with h5py.File(raw_h5_path, 'r') as f:
        pdb_ids = f['pdb_ids'][:]
        n_total = len(pdb_ids)

    print(f"Total residues: {n_total:,}")

    boundaries = [0]
    for i in tqdm(range(1, n_total), desc="Scanning", miniters=100000):
        if pdb_ids[i] != pdb_ids[i - 1]:
            boundaries.append(i)
    boundaries.append(n_total)
    del pdb_ids  # free RAM immediately

    n_structures = len(boundaries) - 1
    print(f"Found {n_structures:,} structures")

    # Split structures across workers
    structures_per_worker = max(1, n_structures // n_workers)
    worker_args = []
    for i in range(n_workers):
        start_struct = i * structures_per_worker
        end_struct = (
            min((i + 1) * structures_per_worker, n_structures)
            if i < n_workers - 1 else n_structures
        )
        boundaries_chunk = list(zip(
            boundaries[start_struct:end_struct],
            boundaries[start_struct + 1:end_struct + 1]
        ))
        worker_args.append((raw_h5_path, boundaries_chunk, half_context))

    # Process in parallel
    print(f"\nExtracting context windows in parallel ({n_workers} workers)...")
    with Pool(n_workers) as pool:
        results = list(tqdm(
            pool.imap(process_structure_chunk_h5, worker_args),
            total=len(worker_args),
            desc="Workers"
        ))

    # Count samples
    n_samples = sum(len(r[2]) for r in results)
    print(f"\nExtracted {n_samples:,} valid samples")

    if n_samples == 0:
        print("ERROR: No valid samples!")
        sys.exit(1)

    # Splits
    print("Creating splits...")
    np.random.seed(42)
    indices = np.random.permutation(n_samples)
    n_train = int(n_samples * train_ratio)
    n_val = int(n_samples * val_ratio)

    train_indices = np.sort(indices[:n_train])
    val_indices = np.sort(indices[n_train:n_train + n_val])
    test_indices = np.sort(indices[n_train + n_val:])

    print(f"  Train: {len(train_indices):,}")
    print(f"  Val:   {len(val_indices):,}")
    print(f"  Test:  {len(test_indices):,}")

    # Write to HDF5 in chunks
    print(f"\nWriting to {output_path}...")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    with h5py.File(output_path, 'w') as f_out:
        ds_residues = f_out.create_dataset(
            'residues', shape=(n_samples, context_size),
            dtype=np.int8, compression='gzip', compression_opts=4
        )
        ds_masks = f_out.create_dataset(
            'masks', shape=(n_samples, context_size),
            dtype=bool, compression='gzip', compression_opts=4
        )
        ds_phi = f_out.create_dataset(
            'phi', shape=(n_samples,),
            dtype=np.float32, compression='gzip', compression_opts=4
        )
        ds_psi = f_out.create_dataset(
            'psi', shape=(n_samples,),
            dtype=np.float32, compression='gzip', compression_opts=4
        )

        cursor = 0
        for res, masks, phis, psis in tqdm(results, desc="Writing chunks"):
            if not phis:
                continue
            n = len(phis)
            ds_residues[cursor:cursor + n] = np.array(res, dtype=np.int8)
            ds_masks[cursor:cursor + n] = np.array(masks, dtype=bool)
            ds_phi[cursor:cursor + n] = np.array(phis, dtype=np.float32)
            ds_psi[cursor:cursor + n] = np.array(psis, dtype=np.float32)
            cursor += n

        f_out.create_dataset('train_indices', data=train_indices, compression='gzip')
        f_out.create_dataset('val_indices', data=val_indices, compression='gzip')
        f_out.create_dataset('test_indices', data=test_indices, compression='gzip')

        f_out.attrs['n_samples'] = n_samples
        f_out.attrs['max_context'] = max_context
        f_out.attrs['n_train'] = len(train_indices)
        f_out.attrs['n_val'] = len(val_indices)
        f_out.attrs['n_test'] = len(test_indices)

    file_size = output_path.stat().st_size
    print(f"\n✓ Complete! Size: {file_size / 1e9:.2f} GB")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--max_context', type=int, default=7)
    parser.add_argument('--n_workers', type=int, default=16)
    args = parser.parse_args()

    create_variable_context_dataset(
        raw_h5_path=args.input,
        output_path=args.output,
        max_context=args.max_context,
        n_workers=args.n_workers
    )