"""
Simplest von Mises torsion predictor
Focus: Smooth energy gradients for flow matching
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class VonMisesTorsionPredictor(nn.Module):
    """
    Predicts von Mises distributions for φ and ψ angles
    
    Output: (μ_φ, κ_φ, μ_ψ, κ_ψ)
    where μ = mean angle, κ = concentration (1/variance)
    """
    
    def __init__(self, hidden_dim=256, n_layers=2, dropout=0.1):
        super().__init__()
        
        # Input: 3 residue types (one-hot) + 2 distances = 62 dims
        input_dim = 3 * 20 + 2
        
        # Simple MLP
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        
        self.encoder = nn.Sequential(*layers)
        
        # Output: 4 values (μ_φ, κ_φ, μ_ψ, κ_ψ)
        self.output_head = nn.Linear(hidden_dim, 4)
        
    def forward(self, res_prev, res_curr, res_next, ca_dist_prev, ca_dist_next):
        """
        Args:
            res_prev, res_curr, res_next: (batch,) int in [0, 19]
            ca_dist_prev, ca_dist_next: (batch,) float (normalized)
        
        Returns:
            mu_phi, kappa_phi, mu_psi, kappa_psi: (batch,) each
        """
        batch_size = res_prev.size(0)
        
        # One-hot encode residues
        res_prev_oh = F.one_hot(res_prev, num_classes=20).float()
        res_curr_oh = F.one_hot(res_curr, num_classes=20).float()
        res_next_oh = F.one_hot(res_next, num_classes=20).float()
        
        # Concatenate all features
        distances = torch.stack([ca_dist_prev, ca_dist_next], dim=-1)
        x = torch.cat([res_prev_oh, res_curr_oh, res_next_oh, distances], dim=-1)
        
        # Encode
        h = self.encoder(x)
        
        # Predict parameters
        out = self.output_head(h)
        
        # Split into mu and kappa for phi and psi
        mu_phi = torch.tanh(out[:, 0]) * np.pi      # μ ∈ [-π, π]
        kappa_phi = F.softplus(out[:, 1]) + 0.1    # κ > 0
        mu_psi = torch.tanh(out[:, 2]) * np.pi
        kappa_psi = F.softplus(out[:, 3]) + 0.1
        
        return mu_phi, kappa_phi, mu_psi, kappa_psi
    
    def compute_nll(self, res_prev, res_curr, res_next, ca_dist_prev, ca_dist_next,
                    phi_true, psi_true):
        """
        Compute negative log-likelihood (energy)
        
        Args:
            phi_true, psi_true: (batch,) true angles in RADIANS
        
        Returns:
            nll: (batch,) negative log-likelihood
        """
        mu_phi, kappa_phi, mu_psi, kappa_psi = self.forward(
            res_prev, res_curr, res_next, ca_dist_prev, ca_dist_next
        )
        
        # von Mises NLL: -log p(x|μ,κ) = -κ·cos(x-μ) + log(2π·I₀(κ))
        # For numerical stability, approximate log I₀
        
        # Cosine term
        nll_phi = -kappa_phi * torch.cos(phi_true - mu_phi)
        nll_psi = -kappa_psi * torch.cos(psi_true - mu_psi)
        
        # Normalization constant: log(2π·I₀(κ))
        # Approximation for κ > 2: log I₀(κ) ≈ κ - 0.5·log(2π·κ)
        # For small κ, use log I₀(κ) ≈ 0
        
        def log_i0_approx(kappa):
            """Approximate log I₀(κ)"""
            # For κ < 2: use series expansion log I₀(κ) ≈ (κ/2)²
            # For κ ≥ 2: use asymptotic log I₀(κ) ≈ κ - log(2πκ)/2
            small = kappa < 2.0
            large = ~small
            
            result = torch.zeros_like(kappa)
            result[small] = torch.log1p((kappa[small] / 2) ** 2)
            result[large] = kappa[large] - 0.5 * torch.log(2 * np.pi * kappa[large])
            
            return result
        
        nll_phi += np.log(2 * np.pi) + log_i0_approx(kappa_phi)
        nll_psi += np.log(2 * np.pi) + log_i0_approx(kappa_psi)
        
        return nll_phi + nll_psi
    
    def predict(self, res_prev, res_curr, res_next, ca_dist_prev, ca_dist_next):
        """
        Predict angles (just return μ)
        
        Returns:
            phi_pred, psi_pred: (batch,) predicted angles in RADIANS
            phi_uncertainty, psi_uncertainty: (batch,) uncertainty (1/κ)
        """
        mu_phi, kappa_phi, mu_psi, kappa_psi = self.forward(
            res_prev, res_curr, res_next, ca_dist_prev, ca_dist_next
        )
        
        # Uncertainty ≈ 1/√κ (circular std dev)
        phi_std = 1.0 / torch.sqrt(kappa_phi)
        psi_std = 1.0 / torch.sqrt(kappa_psi)
        
        return mu_phi, mu_psi, phi_std, psi_std
    
    def compute_energy_with_grad(self, res_prev, res_curr, res_next, 
                                  ca_dist_prev, ca_dist_next,
                                  phi, psi):
        """
        Compute energy and gradients (for flow matching)
        
        Args:
            phi, psi: (batch,) angles in RADIANS (requires_grad=True)
        
        Returns:
            energy: (batch,) scalar energy
            grad_phi, grad_psi: (batch,) gradients ∂E/∂φ, ∂E/∂ψ
        """
        # Enable gradients for angles
        phi = phi.requires_grad_(True)
        psi = psi.requires_grad_(True)
        
        # Compute energy
        energy = self.compute_nll(res_prev, res_curr, res_next, 
                                  ca_dist_prev, ca_dist_next,
                                  phi, psi)
        
        # Compute gradients
        grad_phi = torch.autograd.grad(energy.sum(), phi, create_graph=True)[0]
        grad_psi = torch.autograd.grad(energy.sum(), psi, create_graph=True)[0]
        
        return energy, grad_phi, grad_psi