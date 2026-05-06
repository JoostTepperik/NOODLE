"""
training/losses.py

Loss functions with κ regularization
"""

import jax
import jax.numpy as jnp
from jax.scipy.special import i0


def von_mises_nll_loss(predictions, targets):
    """
    Negative log-likelihood for von Mises distribution
    
    Args:
        predictions: (mu_phi, kappa_phi, mu_psi, kappa_psi)
        targets: (phi_true, psi_true)
    
    Returns:
        loss: scalar loss value
    """
    mu_phi, kappa_phi, mu_psi, kappa_psi = predictions
    phi_true, psi_true = targets
    
    # NLL = -κ·cos(θ - μ) + log(2π·I₀(κ))
    # Simplified: -κ·cos(θ - μ) + log(I₀(κ))  (drop constants)
    
    nll_phi = -kappa_phi * jnp.cos(phi_true - mu_phi) + jnp.log(i0(kappa_phi) + 1e-8)
    nll_psi = -kappa_psi * jnp.cos(psi_true - mu_psi) + jnp.log(i0(kappa_psi) + 1e-8)
    
    return (nll_phi + nll_psi).mean()


def kappa_regularization_loss(kappa_phi, kappa_psi, target_kappa_phi=18.0, target_kappa_psi=12.0):
    """
    Encourage κ values near target (stronger gradients)
    
    Args:
        kappa_phi: [batch] predicted φ concentration
        kappa_psi: [batch] predicted ψ concentration
        target_kappa_phi: Target κ for φ (default: 18)
        target_kappa_psi: Target κ for ψ (default: 12)
    
    Returns:
        loss: scalar regularization loss
    """
    penalty_phi = ((kappa_phi - target_kappa_phi) ** 2).mean()
    penalty_psi = ((kappa_psi - target_kappa_psi) ** 2).mean()
    
    return penalty_phi + penalty_psi


def total_loss(predictions, targets, lambda_kappa=0.01):
    """
    Combined loss: NLL + κ regularization
    
    Args:
        predictions: (mu_phi, kappa_phi, mu_psi, kappa_psi)
        targets: (phi_true, psi_true)
        lambda_kappa: Weight for κ regularization
    
    Returns:
        loss: scalar total loss
        metrics: dict of individual loss components
    """
    mu_phi, kappa_phi, mu_psi, kappa_psi = predictions
    
    # NLL loss
    nll = von_mises_nll_loss(predictions, targets)
    
    # Kappa regularization
    kappa_reg = kappa_regularization_loss(kappa_phi, kappa_psi)
    
    # Total
    loss = nll + lambda_kappa * kappa_reg
    
    metrics = {
        'loss': loss,
        'nll': nll,
        'kappa_reg': kappa_reg,
        'mean_kappa_phi': kappa_phi.mean(),
        'mean_kappa_psi': kappa_psi.mean(),
    }
    
    return loss, metrics


def compute_mae(predictions, targets):
    """
    Compute mean absolute error in degrees
    
    Args:
        predictions: (mu_phi, kappa_phi, mu_psi, kappa_psi)
        targets: (phi_true, psi_true)
    
    Returns:
        mae_phi: scalar φ MAE in degrees
        mae_psi: scalar ψ MAE in degrees
    """
    mu_phi, _, mu_psi, _ = predictions
    phi_true, psi_true = targets
    
    # Circular distance
    error_phi = jnp.abs(mu_phi - phi_true)
    error_phi = jnp.minimum(error_phi, 2 * jnp.pi - error_phi)
    
    error_psi = jnp.abs(mu_psi - psi_true)
    error_psi = jnp.minimum(error_psi, 2 * jnp.pi - error_psi)
    
    # Convert to degrees
    mae_phi = (error_phi * 180 / jnp.pi).mean()
    mae_psi = (error_psi * 180 / jnp.pi).mean()
    
    return mae_phi, mae_psi