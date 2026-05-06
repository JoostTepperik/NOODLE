#!/usr/bin/env python3
"""
Training script for JAX von Mises model
"""

import argparse
from pathlib import Path
import numpy as np
import h5py
from tqdm import tqdm
import pickle

import jax
import jax.numpy as jnp
from jax import random
import optax
from flax.training import train_state

from model_jax import VonMisesPredictor, train_step, eval_step


class TrainState(train_state.TrainState):
    """Extended train state with RNG key"""
    rng: jax.random.PRNGKey


def load_data(hdf5_file, split='train'):
    """Load 7-mer data from HDF5"""
    with h5py.File(hdf5_file, 'r') as f:
        mask = f[f'{split}_mask'][:]
        
        # Load continuous angles
        phi_degrees = f['phi_continuous'][:][mask]
        psi_degrees = f['psi_continuous'][:][mask]
        
        data = {
            # 7 residue types
            'res_i_minus_3': f['res_i_minus_3'][:][mask],
            'res_i_minus_2': f['res_i_minus_2'][:][mask],
            'res_i_minus_1': f['res_i_minus_1'][:][mask],
            'res_i': f['res_i'][:][mask],
            'res_i_plus_1': f['res_i_plus_1'][:][mask],
            'res_i_plus_2': f['res_i_plus_2'][:][mask],
            'res_i_plus_3': f['res_i_plus_3'][:][mask],
            
            # 6 CA-CA distances
            'ca_dist_i_minus_3': f['ca_dist_i_minus_3'][:][mask],
            'ca_dist_i_minus_2': f['ca_dist_i_minus_2'][:][mask],
            'ca_dist_i_minus_1': f['ca_dist_i_minus_1'][:][mask],
            'ca_dist_i': f['ca_dist_i'][:][mask],
            'ca_dist_i_plus_1': f['ca_dist_i_plus_1'][:][mask],
            'ca_dist_i_plus_2': f['ca_dist_i_plus_2'][:][mask],
            
            # Angles (convert to radians)
            'phi': phi_degrees * np.pi / 180.0,
            'psi': psi_degrees * np.pi / 180.0,
        }
    
    return data


def batch_generator(data, batch_size, rng, shuffle=True):
    """Generate batches for 7-mer"""
    n_samples = len(data['phi'])
    indices = np.arange(n_samples)
    
    if shuffle:
        rng, shuffle_rng = random.split(rng)
        indices = random.permutation(shuffle_rng, indices)
    
    for i in range(0, n_samples, batch_size):
        batch_indices = indices[i:i + batch_size]
        
        yield (
            data['res_i_minus_3'][batch_indices],
            data['res_i_minus_2'][batch_indices],
            data['res_i_minus_1'][batch_indices],
            data['res_i'][batch_indices],
            data['res_i_plus_1'][batch_indices],
            data['res_i_plus_2'][batch_indices],
            data['res_i_plus_3'][batch_indices],
            data['ca_dist_i_minus_3'][batch_indices],
            data['ca_dist_i_minus_2'][batch_indices],
            data['ca_dist_i_minus_1'][batch_indices],
            data['ca_dist_i'][batch_indices],
            data['ca_dist_i_plus_1'][batch_indices],
            data['ca_dist_i_plus_2'][batch_indices],
            data['phi'][batch_indices],
            data['psi'][batch_indices],
        ), rng

def create_train_state(rng, learning_rate, hidden_dim, n_layers):
    """Create initial training state"""
    model = VonMisesPredictor(hidden_dim=hidden_dim, n_layers=n_layers)
    
    # Initialize with dummy batch (7 residues + 6 distances for 7-mer)
    dummy_res = jnp.zeros(1, dtype=jnp.int32)
    dummy_dist = jnp.zeros(1, dtype=jnp.float32)
    
    params = model.init(
        rng,
        # 7 residue arguments
        dummy_res,  # res_i_minus_3
        dummy_res,  # res_i_minus_2
        dummy_res,  # res_i_minus_1
        dummy_res,  # res_i
        dummy_res,  # res_i_plus_1
        dummy_res,  # res_i_plus_2
        dummy_res,  # res_i_plus_3
        # 6 distance arguments
        dummy_dist,  # ca_dist_i_minus_3
        dummy_dist,  # ca_dist_i_minus_2
        dummy_dist,  # ca_dist_i_minus_1
        dummy_dist,  # ca_dist_i
        dummy_dist,  # ca_dist_i_plus_1
        dummy_dist,  # ca_dist_i_plus_2
        training=False
    )
    
    # Optimizer with learning rate schedule
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=learning_rate / 10,
        peak_value=learning_rate,
        warmup_steps=1000,
        decay_steps=50000,
        end_value=learning_rate / 100,
    )
    
    tx = optax.adam(schedule)
    
    state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
        rng=rng,
    )
    
    return state, model


def train_epoch(state, data, batch_size, rng):
    """Train one epoch"""
    losses = []
    
    n_samples = len(data['phi'])
    n_batches = (n_samples + batch_size - 1) // batch_size
    
    pbar = tqdm(total=n_batches, desc='Training')
    
    for batch, rng in batch_generator(data, batch_size, rng, shuffle=True):
        rng, dropout_rng = random.split(rng)
        state, loss = train_step(state, batch, dropout_rng)
        losses.append(float(loss))
        pbar.update(1)
    
    pbar.close()
    
    return state, np.mean(losses), rng


def evaluate(state, data, batch_size, rng):
    """Evaluate on dataset"""
    losses = []
    maes_phi = []
    maes_psi = []
    
    n_samples = len(data['phi'])
    n_batches = (n_samples + batch_size - 1) // batch_size
    
    pbar = tqdm(total=n_batches, desc='Evaluating')
    
    for batch, rng in batch_generator(data, batch_size, rng, shuffle=False):
        rng, dropout_rng = random.split(rng)
        loss, mae_phi, mae_psi = eval_step(state.params, state.apply_fn, batch, dropout_rng)
        
        losses.append(float(loss))
        maes_phi.append(float(mae_phi))
        maes_psi.append(float(mae_psi))
        pbar.update(1)
    
    pbar.close()
    
    return np.mean(losses), np.mean(maes_phi), np.mean(maes_psi), rng


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help='Path to training_data.h5')
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--n_layers', type=int, default=3)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--output_dir', default='outputs/jax_vonmises_v1')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    # Setup
    rng = random.PRNGKey(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("JAX Von Mises Training")
    print("="*60)
    
    # Load data
    print("\nLoading data...")
    train_data = load_data(args.data, 'train')
    val_data = load_data(args.data, 'val')
    test_data = load_data(args.data, 'test')
    
    print(f"  Train: {len(train_data['phi']):,} samples")
    print(f"  Val:   {len(val_data['phi']):,} samples")
    print(f"  Test:  {len(test_data['phi']):,} samples")
    
    # Create model
    print("\nCreating model...")
    rng, init_rng = random.split(rng)
    state, model = create_train_state(init_rng, args.lr, args.hidden_dim, args.n_layers)
    
    n_params = sum(x.size for x in jax.tree_util.tree_leaves(state.params))
    print(f"  Parameters: {n_params:,}")
    
    # Training
    print("\nTraining...")
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # Train
        rng, train_rng = random.split(rng)
        state, train_loss, train_rng = train_epoch(
            state, train_data, args.batch_size, train_rng
        )
        
        # Validate
        rng, val_rng = random.split(rng)
        val_loss, val_mae_phi, val_mae_psi, val_rng = evaluate(
            state, val_data, args.batch_size, val_rng
        )
        
        print(f"  Train Loss: {train_loss:.3f}")
        print(f"  Val Loss:   {val_loss:.3f}")
        print(f"  Val φ MAE:  {val_mae_phi:.1f}°")
        print(f"  Val ψ MAE:  {val_mae_psi:.1f}°")
        
        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            
            with open(output_dir / 'best_params.pkl', 'wb') as f:
                pickle.dump(state.params, f)
            
            print("  ✓ Saved best model")
    
    # Final test
    print("\n" + "="*60)
    print("Final Test Evaluation")
    print("="*60)
    
    rng, test_rng = random.split(rng)
    test_loss, test_mae_phi, test_mae_psi, _ = evaluate(
        state, test_data, args.batch_size, test_rng
    )
    
    print(f"\nTest Results:")
    print(f"  Loss:    {test_loss:.3f}")
    print(f"  φ MAE:   {test_mae_phi:.1f}°")
    print(f"  ψ MAE:   {test_mae_psi:.1f}°")
    
    print("\n✓ Training complete!")


if __name__ == '__main__':
    main()