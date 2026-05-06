"""
Evaluate energy function quality for flow matching guidance

Tests:
1. Energy landscape shape (minima near true angles?)
2. Gradient direction (points toward true angles?)
3. Confidence calibration (κ values reasonable?)
"""

import h5py
import numpy as np
import jax
import jax.numpy as jnp
import pickle
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

# Add parent directory to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.model_jax import VonMisesPredictor

def load_model_and_data(model_path, data_path):
    """Load trained model and test data"""
    
    print("Loading model...")
    with open(model_path, 'rb') as f:
        params = pickle.load(f)
    
    model = VonMisesPredictor(hidden_dim=512, n_layers=4)
    
    print("Loading test data...")
    with h5py.File(data_path, 'r') as f:
        test_mask = f['test_mask'][:]
        
        data = {
            'test_indices': np.where(test_mask)[0],
            'res_i_minus_3': f['res_i_minus_3'][:],
            'res_i_minus_2': f['res_i_minus_2'][:],
            'res_i_minus_1': f['res_i_minus_1'][:],
            'res_i': f['res_i'][:],
            'res_i_plus_1': f['res_i_plus_1'][:],
            'res_i_plus_2': f['res_i_plus_2'][:],
            'res_i_plus_3': f['res_i_plus_3'][:],
            'ca_dist_i_minus_3': f['ca_dist_i_minus_3'][:],
            'ca_dist_i_minus_2': f['ca_dist_i_minus_2'][:],
            'ca_dist_i_minus_1': f['ca_dist_i_minus_1'][:],
            'ca_dist_i': f['ca_dist_i'][:],
            'ca_dist_i_plus_1': f['ca_dist_i_plus_1'][:],
            'ca_dist_i_plus_2': f['ca_dist_i_plus_2'][:],
            'phi': f['phi_continuous'][:] * np.pi / 180,  # Convert to radians
            'psi': f['psi_continuous'][:] * np.pi / 180,
        }
    
    print(f"  Test set: {len(data['test_indices']):,} samples")
    
    return params, model, data


def evaluate_single_sample(params, model, sample_data, angle_type='phi'):
    """
    Evaluate energy function for a single sample
    
    Args:
        params: Model parameters
        model: VonMisesPredictor7mer
        sample_data: Dict with residue types, distances, true angles
        angle_type: 'phi' or 'psi'
    
    Returns:
        Dict with energy metrics
    """
    
    # Prepare inputs
    res_inputs = [jnp.array([r]) for r in sample_data['residues']]
    ca_inputs = [jnp.array([d]) for d in sample_data['distances']]
    
    # Predict
    mu_phi, kappa_phi, mu_psi, kappa_psi = model.apply(
        params, *res_inputs, *ca_inputs,
        training=False, rngs={'dropout': jax.random.PRNGKey(0)}
    )
    
    mu_phi = float(mu_phi[0])
    kappa_phi = float(kappa_phi[0])
    mu_psi = float(mu_psi[0])
    kappa_psi = float(kappa_psi[0])
    
    # Select angle
    if angle_type == 'phi':
        true_angle = sample_data['true_phi']
        mu = mu_phi
        kappa = kappa_phi
    else:
        true_angle = sample_data['true_psi']
        mu = mu_psi
        kappa = kappa_psi
    
    # 1. Energy at true angle
    E_true = -kappa * np.cos(true_angle - mu)
    
    # 2. Minimum energy (at mu)
    E_min = -kappa
    
    # 3. Energy gap (how much worse is true vs optimum?)
    energy_gap = E_true - E_min
    
    # 4. Distance to minimum (circular)
    dist_to_min = abs(true_angle - mu)
    dist_to_min = min(dist_to_min, 2*np.pi - dist_to_min)  # Wrap around
    
    # 5. Test gradient directions at multiple points
    # Sample points from noise (0) to true angle
    test_points = np.linspace(0, true_angle, 6)[1:-1]  # Exclude endpoints
    
    gradient_qualities = []
    for angle_current in test_points:
        # Gradient at current point
        grad = kappa * np.sin(angle_current - mu)
        
        # Direction to true angle
        diff = true_angle - angle_current
        # Handle circular wrapping
        if abs(diff) > np.pi:
            diff = diff - np.sign(diff) * 2 * np.pi
        direction_to_true = np.sign(diff)
        
        # Gradient direction (negative grad means decrease angle)
        grad_direction = -np.sign(grad)
        
        # Check if gradient points toward true
        if direction_to_true == 0:
            quality = 1.0  # Already at true
        elif grad_direction == direction_to_true:
            quality = 1.0  # Correct direction
        elif grad_direction == 0:
            quality = 0.5  # No gradient (neutral)
        else:
            quality = 0.0  # Wrong direction
        
        gradient_qualities.append(quality)
    
    return {
        'energy_gap': energy_gap,
        'distance_to_min': dist_to_min * 180 / np.pi,  # Convert to degrees
        'gradient_quality': np.mean(gradient_qualities),
        'kappa': kappa,
        'true_angle': true_angle * 180 / np.pi,
        'pred_angle': mu * 180 / np.pi,
        'prediction_error': dist_to_min * 180 / np.pi,  # Same as distance to min
    }


def evaluate_energy_function(params, model, data, n_samples=1000, angle_type='phi'):
    """
    Evaluate energy function on multiple test samples
    """
    
    print(f"\nEvaluating {angle_type} energy function on {n_samples} samples...")
    
    results = {
        'energy_gaps': [],
        'distances_to_min': [],
        'gradient_qualities': [],
        'kappas': [],
        'prediction_errors': [],
    }
    
    test_indices = data['test_indices']
    n_eval = min(n_samples, len(test_indices))
    
    for i in tqdm(range(n_eval), desc=f"Evaluating {angle_type}"):
        idx = test_indices[i]
        
        # Prepare sample data
        sample_data = {
            'residues': [
                int(data['res_i_minus_3'][idx]),
                int(data['res_i_minus_2'][idx]),
                int(data['res_i_minus_1'][idx]),
                int(data['res_i'][idx]),
                int(data['res_i_plus_1'][idx]),
                int(data['res_i_plus_2'][idx]),
                int(data['res_i_plus_3'][idx]),
            ],
            'distances': [
                float(data['ca_dist_i_minus_3'][idx]),
                float(data['ca_dist_i_minus_2'][idx]),
                float(data['ca_dist_i_minus_1'][idx]),
                float(data['ca_dist_i'][idx]),
                float(data['ca_dist_i_plus_1'][idx]),
                float(data['ca_dist_i_plus_2'][idx]),
            ],
            'true_phi': data['phi'][idx],
            'true_psi': data['psi'][idx],
        }
        
        # Evaluate
        metrics = evaluate_single_sample(params, model, sample_data, angle_type)
        
        results['energy_gaps'].append(metrics['energy_gap'])
        results['distances_to_min'].append(metrics['distance_to_min'])
        results['gradient_qualities'].append(metrics['gradient_quality'])
        results['kappas'].append(metrics['kappa'])
        results['prediction_errors'].append(metrics['prediction_error'])
    
    return results


def print_summary(results_phi, results_psi):
    """Print comprehensive summary of energy function quality"""
    
    print("\n" + "="*70)
    print("ENERGY FUNCTION EVALUATION SUMMARY")
    print("="*70)
    
    for angle_name, results in [('φ (phi)', results_phi), ('ψ (psi)', results_psi)]:
        print(f"\n{'='*70}")
        print(f"{angle_name.upper()}")
        print(f"{'='*70}")
        
        # Convert to numpy arrays
        distances = np.array(results['distances_to_min'])
        gradients = np.array(results['gradient_qualities'])
        kappas = np.array(results['kappas'])
        energy_gaps = np.array(results['energy_gaps'])
        
        print(f"\n1. ENERGY LANDSCAPE QUALITY")
        print(f"   Average distance to minimum: {distances.mean():.1f}°")
        print(f"   Median distance: {np.median(distances):.1f}°")
        print(f"   Std dev: {distances.std():.1f}°")
        print(f"   ")
        print(f"   Distribution:")
        print(f"     < 30°: {(distances < 30).mean()*100:>5.1f}% (excellent)")
        print(f"     < 60°: {(distances < 60).mean()*100:>5.1f}% (good)")
        print(f"     < 90°: {(distances < 90).mean()*100:>5.1f}% (acceptable)")
        print(f"     ≥ 90°: {(distances >= 90).mean()*100:>5.1f}% (poor)")
        
        print(f"\n2. ENERGY GAP AT TRUE ANGLE")
        print(f"   Average energy gap: {energy_gaps.mean():.3f}")
        print(f"   Median energy gap: {np.median(energy_gaps):.3f}")
        print(f"   (Lower = better; 0 = true angle is at energy minimum)")
        print(f"   ")
        print(f"   Interpretation:")
        if energy_gaps.mean() < 1.0:
            print(f"     ✓ True angles are close to energy minima")
        elif energy_gaps.mean() < 3.0:
            print(f"     ~ True angles have moderate energy penalty")
        else:
            print(f"     ✗ True angles far from minima (high energy)")
        
        print(f"\n3. GRADIENT DIRECTION QUALITY")
        print(f"   Mean gradient quality: {gradients.mean():.3f}")
        print(f"   (1.0 = always points toward true, 0.0 = always wrong)")
        print(f"   ")
        print(f"   Distribution:")
        print(f"     Perfect (1.0):      {(gradients == 1.0).mean()*100:>5.1f}%")
        print(f"     Good (> 0.75):      {(gradients > 0.75).mean()*100:>5.1f}%")
        print(f"     Acceptable (> 0.5): {(gradients > 0.5).mean()*100:>5.1f}%")
        print(f"     Poor (≤ 0.5):       {(gradients <= 0.5).mean()*100:>5.1f}%")
        
        print(f"\n4. CONFIDENCE (κ) DISTRIBUTION")
        print(f"   Mean κ: {kappas.mean():.1f}")
        print(f"   Median κ: {np.median(kappas):.1f}")
        print(f"   Std dev: {kappas.std():.1f}")
        print(f"   Range: [{kappas.min():.1f}, {kappas.max():.1f}]")
        print(f"   ")
        print(f"   Distribution:")
        print(f"     High (> 10):   {(kappas > 10).mean()*100:>5.1f}% (strong guidance)")
        print(f"     Medium (5-10): {((kappas >= 5) & (kappas <= 10)).mean()*100:>5.1f}% (moderate)")
        print(f"     Low (< 5):     {(kappas < 5).mean()*100:>5.1f}% (weak guidance)")
    
    # Overall assessment
    print("\n" + "="*70)
    print("ASSESSMENT FOR FLOW MATCHING")
    print("="*70)
    
    # Criteria
    phi_dist_ok = np.mean(results_phi['distances_to_min']) < 60
    phi_grad_ok = np.mean(results_phi['gradient_qualities']) > 0.65
    phi_kappa_ok = np.median(results_phi['kappas']) > 5
    
    psi_dist_ok = np.mean(results_psi['distances_to_min']) < 90  # More lenient for psi
    psi_grad_ok = np.mean(results_psi['gradient_qualities']) > 0.60
    psi_kappa_ok = np.median(results_psi['kappas']) > 3
    
    print(f"\nφ angle guidance:")
    if phi_dist_ok and phi_grad_ok and phi_kappa_ok:
        print("  ✓ EXCELLENT - Should provide strong guidance")
    elif phi_dist_ok and phi_grad_ok:
        print("  ✓ GOOD - Should provide useful guidance")
    elif phi_dist_ok or phi_grad_ok:
        print("  ~ MODERATE - May provide weak guidance")
    else:
        print("  ✗ POOR - Unlikely to help much")
    
    print(f"\nψ angle guidance:")
    if psi_dist_ok and psi_grad_ok and psi_kappa_ok:
        print("  ✓ EXCELLENT - Should provide strong guidance")
    elif psi_dist_ok and psi_grad_ok:
        print("  ✓ GOOD - Should provide useful guidance")
    elif psi_dist_ok or psi_grad_ok:
        print("  ~ MODERATE - May provide weak guidance")
    else:
        print("  ✗ POOR - Unlikely to help much")
    
    print(f"\nOverall recommendation:")
    if (phi_dist_ok and phi_grad_ok) and (psi_dist_ok and psi_grad_ok):
        print("  ✓ Energy function is suitable for flow matching guidance!")
        print("  → Proceed with integration")
    elif (phi_dist_ok or phi_grad_ok) and (psi_dist_ok or psi_grad_ok):
        print("  ~ Energy function may provide weak but useful guidance")
        print("  → Worth testing in flow matching")
    else:
        print("  ✗ Energy function may not provide useful guidance")
        print("  → Consider model improvements or alternative approaches")


def plot_energy_distributions(results_phi, results_psi, output_dir):
    """Create visualization plots"""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    for row, (angle_name, results) in enumerate([('φ', results_phi), ('ψ', results_psi)]):
        distances = np.array(results['distances_to_min'])
        gradients = np.array(results['gradient_qualities'])
        kappas = np.array(results['kappas'])
        
        # Distance to minimum
        axes[row, 0].hist(distances, bins=50, edgecolor='black', alpha=0.7)
        axes[row, 0].axvline(30, color='g', linestyle='--', label='Good (<30°)')
        axes[row, 0].axvline(60, color='orange', linestyle='--', label='Acceptable (<60°)')
        axes[row, 0].axvline(90, color='r', linestyle='--', label='Poor (≥90°)')
        axes[row, 0].set_xlabel('Distance to Energy Minimum (°)')
        axes[row, 0].set_ylabel('Count')
        axes[row, 0].set_title(f'{angle_name}: Distance to Minimum')
        axes[row, 0].legend()
        
        # Gradient quality
        axes[row, 1].hist(gradients, bins=20, edgecolor='black', alpha=0.7)
        axes[row, 1].axvline(0.75, color='g', linestyle='--', label='Good (>0.75)')
        axes[row, 1].axvline(0.5, color='orange', linestyle='--', label='Acceptable (>0.5)')
        axes[row, 1].set_xlabel('Gradient Quality Score')
        axes[row, 1].set_ylabel('Count')
        axes[row, 1].set_title(f'{angle_name}: Gradient Direction Quality')
        axes[row, 1].legend()
        
        # Kappa distribution
        axes[row, 2].hist(kappas, bins=50, edgecolor='black', alpha=0.7)
        axes[row, 2].axvline(5, color='orange', linestyle='--', label='Weak (<5)')
        axes[row, 2].axvline(10, color='g', linestyle='--', label='Strong (>10)')
        axes[row, 2].set_xlabel('κ (Confidence)')
        axes[row, 2].set_ylabel('Count')
        axes[row, 2].set_title(f'{angle_name}: Confidence Distribution')
        axes[row, 2].legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'energy_function_evaluation.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved plot: {output_dir / 'energy_function_evaluation.png'}")
    plt.close()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate energy function quality')
    parser.add_argument('--model', default='outputs/jax_7mer_v1/best_params.pkl',
                        help='Path to model parameters')
    parser.add_argument('--data', default='data/training_7mer/training_data.h5',
                        help='Path to training data')
    parser.add_argument('--n_samples', type=int, default=1000,
                        help='Number of test samples to evaluate')
    parser.add_argument('--output_dir', default='outputs/energy_evaluation',
                        help='Output directory for plots')
    args = parser.parse_args()
    
    # Load model and data
    params, model, data = load_model_and_data(args.model, args.data)
    
    # Evaluate φ
    results_phi = evaluate_energy_function(
        params, model, data, 
        n_samples=args.n_samples, 
        angle_type='phi'
    )
    
    # Evaluate ψ
    results_psi = evaluate_energy_function(
        params, model, data, 
        n_samples=args.n_samples, 
        angle_type='psi'
    )
    
    # Print summary
    print_summary(results_phi, results_psi)
    
    # Create plots
    plot_energy_distributions(results_phi, results_psi, args.output_dir)
    
    print("\n" + "="*70)
    print("✓ Evaluation complete!")
    print("="*70)


if __name__ == '__main__':
    main()