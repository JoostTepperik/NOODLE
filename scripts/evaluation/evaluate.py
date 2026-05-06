"""
evaluation/evaluate.py

Comprehensive evaluation of trained model
"""

import jax
import jax.numpy as jnp
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from flax.training import checkpoints

import sys
sys.path.append(str(Path(__file__).parent.parent))

from data.dataloader import VariableContextDataLoader
from models.full_model import TorsionPredictor
from training.losses import compute_mae


def load_model(checkpoint_dir, max_context=7, hidden_dim=768):
    """Load trained model from checkpoint"""
    
    # Create model
    model = TorsionPredictor(
        max_context=max_context,
        embed_dim=64,
        hidden_dim=hidden_dim,
        n_layers=3,
        dropout_rate=0.1,
        use_refinement=False
    )
    
    # Load checkpoint
    state = checkpoints.restore_checkpoint(
        ckpt_dir=checkpoint_dir,
        target=None,
        prefix='best_'
    )
    
    return model, state


def evaluate_detailed(model, state, data_loader):
    """
    Detailed evaluation with per-residue analysis
    
    Returns:
        results: dict with detailed metrics
    """
    
    all_predictions = {
        'mu_phi': [],
        'mu_psi': [],
        'kappa_phi': [],
        'kappa_psi': [],
    }
    
    all_targets = {
        'phi': [],
        'psi': [],
    }
    
    all_residues = []
    all_masks = []
    
    print("Running evaluation...")
    for batch in tqdm(data_loader):
        # Predict
        mu_phi, kappa_phi, mu_psi, kappa_psi = model.apply(
            {'params': state['params']},
            batch['residues'],
            batch['masks'],
            training=False,
            rngs={'dropout': jax.random.PRNGKey(0)}
        )
        
        # Store
        all_predictions['mu_phi'].append(np.array(mu_phi))
        all_predictions['mu_psi'].append(np.array(mu_psi))
        all_predictions['kappa_phi'].append(np.array(kappa_phi))
        all_predictions['kappa_psi'].append(np.array(kappa_psi))
        
        all_targets['phi'].append(np.array(batch['phi']))
        all_targets['psi'].append(np.array(batch['psi']))
        
        all_residues.append(np.array(batch['residues']))
        all_masks.append(np.array(batch['masks']))
    
    # Concatenate
    predictions = {k: np.concatenate(v) for k, v in all_predictions.items()}
    targets = {k: np.concatenate(v) for k, v in all_targets.items()}
    residues = np.concatenate(all_residues)
    masks = np.concatenate(all_masks)
    
    # Convert to degrees
    predictions_deg = {
        'mu_phi': predictions['mu_phi'] * 180 / np.pi,
        'mu_psi': predictions['mu_psi'] * 180 / np.pi,
        'kappa_phi': predictions['kappa_phi'],
        'kappa_psi': predictions['kappa_psi'],
    }
    
    targets_deg = {
        'phi': targets['phi'] * 180 / np.pi,
        'psi': targets['psi'] * 180 / np.pi,
    }
    
    # Compute errors (circular distance)
    def circular_error(pred, true):
        error = np.abs(pred - true)
        error = np.minimum(error, 360 - error)
        return error
    
    error_phi = circular_error(predictions_deg['mu_phi'], targets_deg['phi'])
    error_psi = circular_error(predictions_deg['mu_psi'], targets_deg['psi'])
    
    # Overall metrics
    results = {
        'mae_phi': error_phi.mean(),
        'mae_psi': error_psi.mean(),
        'median_error_phi': np.median(error_phi),
        'median_error_psi': np.median(error_psi),
        'std_error_phi': error_phi.std(),
        'std_error_psi': error_psi.std(),
        'mean_kappa_phi': predictions['kappa_phi'].mean(),
        'mean_kappa_psi': predictions['kappa_psi'].mean(),
        'median_kappa_phi': np.median(predictions['kappa_phi']),
        'median_kappa_psi': np.median(predictions['kappa_psi']),
    }
    
    # Per-residue type analysis
    center_residues = residues[:, residues.shape[1] // 2]  # Center position
    
    results['per_residue'] = {}
    AA_NAMES = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
                'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
    
    for aa_idx, aa_name in enumerate(AA_NAMES):
        mask_aa = (center_residues == aa_idx)
        if mask_aa.sum() > 100:  # Only if enough samples
            results['per_residue'][aa_name] = {
                'mae_phi': error_phi[mask_aa].mean(),
                'mae_psi': error_psi[mask_aa].mean(),
                'n_samples': mask_aa.sum(),
            }
    
    # Context size analysis (edge vs center)
    n_valid = masks.sum(axis=1)  # Number of valid context positions
    
    results['by_context_size'] = {}
    for context_size in range(4, 8):  # 4-7 valid positions
        mask_context = (n_valid == context_size)
        if mask_context.sum() > 100:
            results['by_context_size'][context_size] = {
                'mae_phi': error_phi[mask_context].mean(),
                'mae_psi': error_psi[mask_context].mean(),
                'n_samples': mask_context.sum(),
            }
    
    # Store for plotting
    results['predictions'] = predictions_deg
    results['targets'] = targets_deg
    results['errors'] = {'phi': error_phi, 'psi': error_psi}
    
    return results


def plot_results(results, output_dir):
    """Create evaluation plots"""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Ramachandran plot: predicted vs true
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # True angles
    axes[0].hexbin(
        results['targets']['phi'],
        results['targets']['psi'],
        gridsize=50,
        cmap='Blues',
        mincnt=1
    )
    axes[0].set_xlabel('φ (degrees)')
    axes[0].set_ylabel('ψ (degrees)')
    axes[0].set_title('True Angles (Test Set)')
    axes[0].set_xlim(-180, 180)
    axes[0].set_ylim(-180, 180)
    
    # Predicted angles
    axes[1].hexbin(
        results['predictions']['mu_phi'],
        results['predictions']['mu_psi'],
        gridsize=50,
        cmap='Reds',
        mincnt=1
    )
    axes[1].set_xlabel('φ (degrees)')
    axes[1].set_ylabel('ψ (degrees)')
    axes[1].set_title('Predicted Angles')
    axes[1].set_xlim(-180, 180)
    axes[1].set_ylim(-180, 180)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'ramachandran.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Error distributions
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    axes[0].hist(results['errors']['phi'], bins=50, alpha=0.7, edgecolor='black')
    axes[0].axvline(results['mae_phi'], color='r', linestyle='--', 
                   label=f"MAE = {results['mae_phi']:.1f}°")
    axes[0].set_xlabel('φ Error (degrees)')
    axes[0].set_ylabel('Count')
    axes[0].set_title('φ Error Distribution')
    axes[0].legend()
    
    axes[1].hist(results['errors']['psi'], bins=50, alpha=0.7, edgecolor='black')
    axes[1].axvline(results['mae_psi'], color='r', linestyle='--',
                   label=f"MAE = {results['mae_psi']:.1f}°")
    axes[1].set_xlabel('ψ Error (degrees)')
    axes[1].set_ylabel('Count')
    axes[1].set_title('ψ Error Distribution')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'error_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Per-residue performance
    if 'per_residue' in results and results['per_residue']:
        residues = sorted(results['per_residue'].keys())
        mae_phi_per_res = [results['per_residue'][r]['mae_phi'] for r in residues]
        mae_psi_per_res = [results['per_residue'][r]['mae_psi'] for r in residues]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(residues))
        width = 0.35
        
        ax.bar(x - width/2, mae_phi_per_res, width, label='φ', alpha=0.8)
        ax.bar(x + width/2, mae_psi_per_res, width, label='ψ', alpha=0.8)
        
        ax.set_xlabel('Residue Type')
        ax.set_ylabel('MAE (degrees)')
        ax.set_title('Performance by Residue Type')
        ax.set_xticks(x)
        ax.set_xticklabels(residues)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'per_residue_performance.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 4. κ distributions
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    axes[0].hist(results['predictions']['kappa_phi'], bins=50, alpha=0.7, edgecolor='black')
    axes[0].axvline(results['mean_kappa_phi'], color='r', linestyle='--',
                   label=f"Mean = {results['mean_kappa_phi']:.1f}")
    axes[0].set_xlabel('κ_φ')
    axes[0].set_ylabel('Count')
    axes[0].set_title('φ Confidence Distribution')
    axes[0].legend()
    
    axes[1].hist(results['predictions']['kappa_psi'], bins=50, alpha=0.7, edgecolor='black')
    axes[1].axvline(results['mean_kappa_psi'], color='r', linestyle='--',
                   label=f"Mean = {results['mean_kappa_psi']:.1f}")
    axes[1].set_xlabel('κ_ψ')
    axes[1].set_ylabel('Count')
    axes[1].set_title('ψ Confidence Distribution')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'kappa_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Plots saved to {output_dir}")


def print_summary(results):
    """Print evaluation summary"""
    
    print("\n" + "="*70)
    print("EVALUATION SUMMARY")
    print("="*70)
    
    print(f"\nOverall Performance:")
    print(f"  φ MAE: {results['mae_phi']:.2f}° (median: {results['median_error_phi']:.2f}°)")
    print(f"  ψ MAE: {results['mae_psi']:.2f}° (median: {results['median_error_psi']:.2f}°)")
    print(f"  φ std: {results['std_error_phi']:.2f}°")
    print(f"  ψ std: {results['std_error_psi']:.2f}°")
    
    print(f"\nConfidence (κ) Statistics:")
    print(f"  κ_φ mean: {results['mean_kappa_phi']:.2f} (median: {results['median_kappa_phi']:.2f})")
    print(f"  κ_ψ mean: {results['mean_kappa_psi']:.2f} (median: {results['median_kappa_psi']:.2f})")
    
    if 'by_context_size' in results and results['by_context_size']:
        print(f"\nPerformance by Context Size:")
        for context_size in sorted(results['by_context_size'].keys()):
            metrics = results['by_context_size'][context_size]
            print(f"  {context_size} positions: φ={metrics['mae_phi']:.2f}°, "
                  f"ψ={metrics['mae_psi']:.2f}° (n={metrics['n_samples']:,})")
    
    if 'per_residue' in results and results['per_residue']:
        print(f"\nBest/Worst Performing Residues:")
        sorted_by_phi = sorted(
            results['per_residue'].items(),
            key=lambda x: x[1]['mae_phi']
        )
        print(f"  Best φ:  {sorted_by_phi[0][0]} ({sorted_by_phi[0][1]['mae_phi']:.1f}°)")
        print(f"  Worst φ: {sorted_by_phi[-1][0]} ({sorted_by_phi[-1][1]['mae_phi']:.1f}°)")
        
        sorted_by_psi = sorted(
            results['per_residue'].items(),
            key=lambda x: x[1]['mae_psi']
        )
        print(f"  Best ψ:  {sorted_by_psi[0][0]} ({sorted_by_psi[0][1]['mae_psi']:.1f}°)")
        print(f"  Worst ψ: {sorted_by_psi[-1][0]} ({sorted_by_psi[-1][1]['mae_psi']:.1f}°)")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate trained model')
    parser.add_argument('--checkpoint_dir', required=True,
                       help='Directory containing checkpoints')
    parser.add_argument('--data', required=True,
                       help='Path to training HDF5 file')
    parser.add_argument('--output_dir', default='outputs/evaluation',
                       help='Output directory for plots')
    parser.add_argument('--split', default='test', choices=['train', 'val', 'test'],
                       help='Which split to evaluate')
    parser.add_argument('--batch_size', type=int, default=512,
                       help='Batch size')
    args = parser.parse_args()
    
    # Load model
    model, state = load_model(args.checkpoint_dir)
    
    # Create dataloader
    data_loader = VariableContextDataLoader(
        args.data,
        split=args.split,
        batch_size=args.batch_size,
        shuffle=False
    )
    
    # Evaluate
    results = evaluate_detailed(model, state, data_loader)
    
    # Print summary
    print_summary(results)
    
    # Create plots
    plot_results(results, args.output_dir)
    
    print(f"\n✓ Evaluation complete!")