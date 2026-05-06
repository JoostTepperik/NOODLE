"""
models/refinement.py

Stage 2: Refinement using neighbor predictions
"""

import jax
import jax.numpy as jnp
import flax.linen as nn


class RefinementNetwork(nn.Module):
    """
    Refines predictions using neighbor angle predictions
    
    Input:
        - Original context (sequence)
        - Predicted angles from neighbors (±2 positions)
        - Initial confidence (κ from stage 1)
    
    Output:
        - Refined (μ_φ, κ_φ, μ_ψ, κ_ψ)
    """
    
    hidden_dim: int = 768
    
    @nn.compact
    def __call__(self, h_context, neighbor_angles, initial_kappa):
        """
        Args:
            h_context: [batch, hidden_dim] encoded sequence context
            neighbor_angles: [batch, 8] (phi/psi for i-2, i-1, i+1, i+2)
            initial_kappa: [batch, 2] (kappa_phi, kappa_psi from stage 1)
        
        Returns:
            Refined predictions: (mu_phi, kappa_phi, mu_psi, kappa_psi)
        """
        # Encode neighbor angles
        # Convert angles to 2D representation (sin/cos)
        angles_sin = jnp.sin(neighbor_angles)
        angles_cos = jnp.cos(neighbor_angles)
        angle_features = jnp.concatenate([angles_sin, angles_cos], axis=-1)  # [batch, 16]
        
        h_angles = nn.Dense(128)(angle_features)
        h_angles = nn.gelu(h_angles)
        
        # Encode initial confidence
        h_kappa = nn.Dense(64)(initial_kappa)
        h_kappa = nn.gelu(h_kappa)
        
        # Combine all information
        h_combined = jnp.concatenate([h_context, h_angles, h_kappa], axis=-1)
        
        h = nn.Dense(self.hidden_dim)(h_combined)
        h = nn.gelu(h)
        h = nn.Dense(self.hidden_dim // 2)(h)
        h = nn.gelu(h)
        
        # Predict refinements (similar to VonMisesPredictionHead)
        # φ
        h_phi = nn.Dense(128)(h)
        h_phi = nn.gelu(h_phi)
        mu_phi = nn.Dense(1)(h_phi).squeeze(-1)
        mu_phi = jnp.arctan2(jnp.sin(mu_phi), jnp.cos(mu_phi))
        kappa_phi = jnp.exp(nn.Dense(1)(h_phi).squeeze(-1))
        
        # ψ
        h_psi = nn.Dense(128)(h)
        h_psi = nn.gelu(h_psi)
        mu_psi = nn.Dense(1)(h_psi).squeeze(-1)
        mu_psi = jnp.arctan2(jnp.sin(mu_psi), jnp.cos(mu_psi))
        kappa_psi = jnp.exp(nn.Dense(1)(h_psi).squeeze(-1))
        
        return mu_phi, kappa_phi, mu_psi, kappa_psi