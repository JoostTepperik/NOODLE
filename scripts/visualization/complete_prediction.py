"""
CORRECT prediction script for BINNED model.

Your model uses binned prediction (n_bins=36), not direct regression.
The outputs are logits that need to be converted to angles.
"""

import jax
import jax.numpy as jnp
import numpy as np
from pathlib import Path
import flax.linen as nn
from flax.training import train_state
import orbax.checkpoint as ocp
import sys

from nerf_reconstruction import ProteinBackboneReconstructor


# ============================================================================
# BINNED MODEL ARCHITECTURE
# ============================================================================

class BinnedPredictionHead(nn.Module):
    """Binned prediction head - outputs logits for bins"""
    n_bins: int = 72
    
    @nn.compact
    def __call__(self, x):
        """
        Args:
            x: (batch, seq_len, hidden_dim)
        Returns:
            logits_phi: (batch, seq_len, n_bins)
            logits_psi: (batch, seq_len, n_bins)
        """
        logits_phi = nn.Dense(self.n_bins)(x)  # (batch, seq_len, n_bins)
        logits_psi = nn.Dense(self.n_bins)(x)  # (batch, seq_len, n_bins)
        
        return logits_phi, logits_psi


class TorsionPredictor(nn.Module):
    """Full model matching your training script"""
    max_context: int = 7
    embed_dim: int = 64
    hidden_dim: int = 768
    n_layers: int = 3
    dropout_rate: float = 0.1
    n_bins: int = 72
    n_amino_acids: int = 21  # 20 AA + padding token
    
    @nn.compact
    def __call__(self, residues, masks, training=False):
        """
        Args:
            residues: (batch, seq_len) - amino acid indices
            masks: (batch, seq_len) - attention masks
            training: bool
            
        Returns:
            (logits_phi, logits_psi) each (batch, seq_len, n_bins)
        """
        # Embedding
        x = nn.Embed(
            num_embeddings=self.n_amino_acids,
            features=self.embed_dim
        )(residues)
        
        # MLP layers
        for _ in range(self.n_layers):
            x = nn.Dense(self.hidden_dim)(x)
            x = nn.relu(x)
            if training:
                x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not training)
        
        # Binned prediction head
        logits_phi, logits_psi = BinnedPredictionHead(n_bins=self.n_bins)(x)
        
        return logits_phi, logits_psi


# ============================================================================
# ANGLE CONVERSION
# ============================================================================

def bins_to_angles(logits_phi, logits_psi, n_bins=36):
    """
    Convert logits to angles.
    
    Args:
        logits_phi: (seq_len, n_bins) - logits for phi bins
        logits_psi: (seq_len, n_bins) - logits for psi bins
        n_bins: number of bins (default 36)
        
    Returns:
        phi_degrees, psi_degrees (each shape: seq_len)
    """
    # Softmax to get probabilities
    probs_phi = jax.nn.softmax(logits_phi, axis=-1)
    probs_psi = jax.nn.softmax(logits_psi, axis=-1)
    
    # Get most likely bin (argmax)
    bin_phi = jnp.argmax(probs_phi, axis=-1)  # (seq_len,)
    bin_psi = jnp.argmax(probs_psi, axis=-1)  # (seq_len,)
    
    # Convert bin index to angle
    # Bins cover -180° to 180°, evenly spaced
    bin_width = 360.0 / n_bins  # 10° for n_bins=36
    
    # Bin center = -180° + (bin_index + 0.5) * bin_width
    phi_degrees = -180.0 + (bin_phi + 0.5) * bin_width
    psi_degrees = -180.0 + (bin_psi + 0.5) * bin_width
    
    return np.array(phi_degrees), np.array(psi_degrees)


# ============================================================================
# CONFIGURATION
# ============================================================================

CHECKPOINT_DIR = Path("/home/jtepperik/thesis/energy_model/scripts/training/outputs/feedforward_binned_19448143/checkpoints")
CHECKPOINT_PREFIX = "best_"

# Model config (must match training)
MODEL_CONFIG = {
    'max_context': 7,
    'embed_dim': 64,
    'hidden_dim': 768,
    'n_layers': 3,
    'dropout_rate': 0.1,
    'n_bins': 72,
    'n_amino_acids': 21  # 20 amino acids + 1 padding token (idx 20)
}

AA_TO_IDX = {
    'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, 'I': 7, 'K': 8,
    'L': 9, 'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14, 'S': 15, 'T': 16,
    'V': 17, 'W': 18, 'Y': 19, 'PAD': 20  # Padding token for out-of-bounds context
}


# ============================================================================
# TRAIN STATE (matching training script)
# ============================================================================

class TrainState(train_state.TrainState):
    """Custom train state matching training script"""
    dropout_rng: jax.random.PRNGKey


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_model_and_params(checkpoint_dir: Path, prefix: str = "best_"):
    """Load model and parameters from checkpoint"""
    # In your diagnostic/prediction script, replace the loading with:
    import orbax.checkpoint as ocp
    import sys
    sys.path.append('/home/jtepperik/thesis/energy_model/scripts')

    from models.full_model import TorsionPredictor  # ← USE THE REAL MODEL

    # Recreate model with correct config
    model = TorsionPredictor(
        max_context=7,
        embed_dim=64,
        hidden_dim=768,
        n_layers=3,
        dropout_rate=0.1,
        prediction_type='binned',
        n_bins=72
    )

    # Initialize to get structure
    dummy_residues = jnp.ones((1, 7), dtype=jnp.int32)
    dummy_mask = jnp.ones((1, 7), dtype=bool)
    variables = model.init(
        {'params': jax.random.PRNGKey(0), 'dropout': jax.random.PRNGKey(0)},
        dummy_residues, dummy_mask, training=False
    )

    # Load checkpoint - restore full state, extract just params
    checkpointer = ocp.StandardCheckpointer()
    ckpt_path = '/home/jtepperik/thesis/energy_model/scripts/training/outputs/feedforward_binned_19448143/checkpoints/best_10'
    restored = checkpointer.restore(ckpt_path)

    params = restored['params']

    # Verify
    print("Checkpoint param keys:", list(params.keys()))
    # Should show: encoder, prediction_head

    # Verify loading worked - trained weights should have varied statsWGQ
    import numpy as np
    leaves = jax.tree_util.tree_leaves(params)
    stds = [float(np.std(l)) for l in leaves if len(l.shape) > 0]
    print(f"✓ Checkpoint loaded successfully (weight std range: {min(stds):.4f} - {max(stds):.4f})")

    return model, params


def encode_sequence(sequence: str) -> np.ndarray:
    """Encode amino acid sequence to integer indices."""
    return np.array([AA_TO_IDX[aa.upper()] for aa in sequence])


def predict_angles(model, params, sequence: str, n_bins: int = 36) -> tuple:
    """
    Predict phi and psi angles from sequence using sliding window.
    
    Each residue is predicted with ±3 context (total window of 7).
    This ensures edge residues see their true local context.
    
    Returns:
        phi_degrees, psi_degrees (in degrees)
    """
    # Encode sequence
    encoded = encode_sequence(sequence)
    seq_len = len(encoded)
    max_context = MODEL_CONFIG['max_context']
    context_radius = max_context // 2  # 3 for max_context=7
    
    if seq_len > max_context:
        print(f"   Sequence length {seq_len} > max_context {max_context}")
        print(f"   Using sliding window (±{context_radius} context per residue)...")
    
    all_phi_logits = []
    all_psi_logits = []
    
    # Predict each position with its local context window
    for i in range(seq_len):
        # Get ±context_radius around position i
        # For max_context=7 and context_radius=3:
        # Position 0: [-3, -2, -1, 0, 1, 2, 3] → need left padding
        # Position 3: [0, 1, 2, 3, 4, 5, 6] → full context
        # Position 6: [3, 4, 5, 6, 7, 8, 9] → need right padding
        
        start = i - context_radius
        end = i + context_radius + 1
        
        # Extract window, handling boundaries with padding
        window = []
        for pos in range(start, end):
            if pos < 0 or pos >= seq_len:
                window.append(20)  # Padding token (matches training)
            else:
                window.append(encoded[pos])
        
        window = np.array(window)
        
        # Position of target residue in window (should be center = context_radius)
        target_pos = context_radius
        
        # Predict
        batch_residues = jnp.array(window)[None, :]  # (1, max_context)
        batch_mask = jnp.ones((1, max_context), dtype=bool)
        
        logits_phi, logits_psi = model.apply(
            {'params': params},
            batch_residues,
            batch_mask,
            training=False,
            rngs={'dropout': jax.random.PRNGKey(0)}
        )
        
        # Get prediction - real model outputs (1, n_bins) for center position only
        all_phi_logits.append(logits_phi[0, :])
        all_psi_logits.append(logits_psi[0, :])
    
    # Stack all predictions
    logits_phi = jnp.stack(all_phi_logits, axis=0)  # (seq_len, n_bins)
    logits_psi = jnp.stack(all_psi_logits, axis=0)  # (seq_len, n_bins)
    
    # Convert to angles
    phi_degrees, psi_degrees = bins_to_angles(logits_phi, logits_psi, n_bins)
    
    return logits_phi, logits_psi #changed for frame based loop modeling`




def predict_structure(
    sequence: str,
    model,
    params,
    output_pdb: str = "predicted_structure.pdb",
    n_bins: int = 36
):
    """Complete pipeline: sequence → angles → 3D structure → PDB."""
    print(f"\n{'='*60}")
    print(f"PREDICTING 3D STRUCTURE (BINNED MODEL)")
    print(f"{'='*60}")
    print(f"Sequence: {sequence}")
    print(f"Length: {len(sequence)} residues")
    print(f"Output: {output_pdb}")
    print(f"Bins: {n_bins} ({360/n_bins:.1f}° per bin)")
    print(f"{'='*60}\n")
    
    # Step 1: Predict angles
    print("Step 1: Predicting torsion angles from bins...")
    phi_angles, psi_angles = predict_angles(model, params, sequence, n_bins)
    
    print(f"\nPredicted angles:")
    print(f"  Phi range: [{phi_angles.min():.1f}°, {phi_angles.max():.1f}°]")
    print(f"  Psi range: [{psi_angles.min():.1f}°, {psi_angles.max():.1f}°]")
    print(f"  Mean phi: {phi_angles.mean():.1f}° ± {phi_angles.std():.1f}°")
    print(f"  Mean psi: {psi_angles.mean():.1f}° ± {psi_angles.std():.1f}°")
    print(f"\n  First 5 phi: {phi_angles[:5]}")
    print(f"  First 5 psi: {psi_angles[:5]}")
    
    # Step 2: Reconstruct 3D structure
    print("\nStep 2: Reconstructing 3D backbone using NeRF algorithm...")
    reconstructor = ProteinBackboneReconstructor()
    
    N, CA, C, O = reconstructor.build_backbone(
        sequence=sequence,
        phi_angles=phi_angles,
        psi_angles=psi_angles
    )
    
    # Step 3: Save PDB
    print("\nStep 3: Saving PDB file...")
    reconstructor.save_pdb(
        filename=output_pdb,
        sequence=sequence,
        N_coords=N,
        CA_coords=CA,
        C_coords=C,
        O_coords=O
    )
    
    print(f"\n{'='*60}")
    print(f"✓ SUCCESS! Structure saved to {output_pdb}")
    print(f"  Total atoms: {len(sequence) * 4}")
    print(f"  Visualize with: pymol {output_pdb}")
    print(f"{'='*60}\n")
    
    return N, CA, C, O, phi_angles, psi_angles


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Check checkpoint exists
    if not CHECKPOINT_DIR.exists():
        print(f"❌ ERROR: Checkpoint directory not found!")
        print(f"   Looking for: {CHECKPOINT_DIR}")
        sys.exit(1)
    
    checkpoints_found = list(CHECKPOINT_DIR.glob(f"{CHECKPOINT_PREFIX}*"))
    if not checkpoints_found:
        print(f"❌ ERROR: No checkpoints found in {CHECKPOINT_DIR}")
        sys.exit(1)
    
    print(f"\nFound {len(checkpoints_found)} checkpoint(s):")
    for ckpt in checkpoints_found:
        print(f"  - {ckpt.name}")
    
    # Load model
    print(f"\n{'='*60}")
    print("LOADING BINNED MODEL")
    print(f"{'='*60}\n")
    
    try:
        model, params = load_model_and_params(CHECKPOINT_DIR, CHECKPOINT_PREFIX)
        print(f"Model configuration:")
        for key, value in MODEL_CONFIG.items():
            print(f"  {key}: {value}")
    except Exception as e:
        print(f"\n❌ ERROR loading checkpoint:")
        print(f"   {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Test sequences (using shorter sequences that fit max_context=7)
    test_sequences = [
        ("ACDEFGH", "test_mixed_short.pdb"),          # 7 residues - fits exactly
        ("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA", "test_alanine.pdb"),              # 7 residues - alpha helix
        ("VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV", "test_valine.pdb"),               # 7 residues - beta strand
        ("GGGGGGG", "test_glycine.pdb"),              # 7 residues - flexible
        ("CASQHGQREKLIGGDTQYF", "test_mixed_long.pdb"),  # 20 residues - sliding window
    ]
    
    print(f"\n{'='*60}")
    print("RUNNING PREDICTIONS")
    print(f"{'='*60}\n")
    
    results = []
    
    for seq, output_file in test_sequences:
        try:
            N, CA, C, O, phi, psi = predict_structure(
                sequence=seq,
                model=model,
                params=params,
                output_pdb=output_file,
                n_bins=MODEL_CONFIG['n_bins']
            )
            results.append((seq, output_file, phi, psi))
            
        except Exception as e:
            print(f"❌ ERROR for sequence {seq[:20]}...")
            print(f"   {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}\n")
    
    if results:
        print(f"Successfully predicted {len(results)}/{len(test_sequences)} structures:\n")
        
        for seq, pdb_file, phi, psi in results:
            print(f"  {pdb_file:20s} - {len(seq):3d} residues")
            print(f"    Phi: {phi.mean():6.1f}° ± {phi.std():5.1f}°  [{phi.min():6.1f}°, {phi.max():6.1f}°]")
            print(f"    Psi: {psi.mean():6.1f}° ± {psi.std():5.1f}°  [{psi.min():6.1f}°, {psi.max():6.1f}°]")
            
            # Classify secondary structure
            helix_like = np.sum((phi > -90) & (phi < -30) & (psi > -75) & (psi < -15))
            beta_like = np.sum((phi > -150) & (phi < -90) & (psi > 90) & (psi < 180))
            print(f"    Secondary structure: {helix_like} helix-like, {beta_like} beta-like")
            print()
    
    print(f"\n{'='*60}")
    print("NEXT STEPS:")
    print(f"{'='*60}")
    print("""
1. Visualize structures:
   pymol test_mixed.pdb test_alanine.pdb
   
2. Validate structures:
   python visualize_structure.py test_mixed.pdb
   
3. Predict your own sequences:
   Edit the test_sequences list or use programmatically
""")