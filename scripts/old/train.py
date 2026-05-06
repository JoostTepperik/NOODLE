#!/usr/bin/env python3
"""
Simplest von Mises training
"""

import argparse
import torch
import torch.optim as optim
import numpy as np
from pathlib import Path
from tqdm import tqdm

from model import VonMisesTorsionPredictor
from dataset import create_dataloaders


def train_epoch(model, loader, optimizer, device):
    """Train one epoch"""
    model.train()
    
    total_loss = 0
    total_mae_phi = 0
    total_mae_psi = 0
    count = 0
    
    for features, targets in tqdm(loader, desc='Training'):
        # Move to device
        res_prev = features['res_prev'].to(device)
        res_curr = features['res_curr'].to(device)
        res_next = features['res_next'].to(device)
        ca_dist_prev = features['ca_dist_prev'].to(device)
        ca_dist_next = features['ca_dist_next'].to(device)
        
        # Convert bin targets to radians
        phi_bins = targets['phi_bin'].to(device)
        psi_bins = targets['psi_bin'].to(device)
        
        # Use bin centers as targets
        phi_true = (phi_bins.float() / 36.0 * 2 * np.pi) - np.pi
        psi_true = (psi_bins.float() / 36.0 * 2 * np.pi) - np.pi
        
        # Compute loss
        loss = model.compute_nll(res_prev, res_curr, res_next,
                                ca_dist_prev, ca_dist_next,
                                phi_true, psi_true).mean()
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Metrics
        with torch.no_grad():
            mu_phi, mu_psi, _, _ = model.predict(res_prev, res_curr, res_next,
                                                 ca_dist_prev, ca_dist_next)
            
            # Angular error (handle wraparound)
            def angular_error(pred, true):
                diff = pred - true
                diff = torch.atan2(torch.sin(diff), torch.cos(diff))
                return torch.abs(diff)
            
            mae_phi = angular_error(mu_phi, phi_true).mean()
            mae_psi = angular_error(mu_psi, psi_true).mean()
        
        total_loss += loss.item()
        total_mae_phi += mae_phi.item() * 180 / np.pi
        total_mae_psi += mae_psi.item() * 180 / np.pi
        count += 1
    
    return total_loss / count, total_mae_phi / count, total_mae_psi / count


@torch.no_grad()
def evaluate(model, loader, device):
    """Evaluate"""
    model.eval()
    
    total_loss = 0
    total_mae_phi = 0
    total_mae_psi = 0
    count = 0
    
    for features, targets in tqdm(loader, desc='Evaluating'):
        res_prev = features['res_prev'].to(device)
        res_curr = features['res_curr'].to(device)
        res_next = features['res_next'].to(device)
        ca_dist_prev = features['ca_dist_prev'].to(device)
        ca_dist_next = features['ca_dist_next'].to(device)
        
        phi_bins = targets['phi_bin'].to(device)
        psi_bins = targets['psi_bin'].to(device)
        
        phi_true = (phi_bins.float() / 36.0 * 2 * np.pi) - np.pi
        psi_true = (psi_bins.float() / 36.0 * 2 * np.pi) - np.pi
        
        loss = model.compute_nll(res_prev, res_curr, res_next,
                                ca_dist_prev, ca_dist_next,
                                phi_true, psi_true).mean()
        
        mu_phi, mu_psi, _, _ = model.predict(res_prev, res_curr, res_next,
                                             ca_dist_prev, ca_dist_next)
        
        def angular_error(pred, true):
            diff = pred - true
            diff = torch.atan2(torch.sin(diff), torch.cos(diff))
            return torch.abs(diff)
        
        mae_phi = angular_error(mu_phi, phi_true).mean()
        mae_psi = angular_error(mu_psi, psi_true).mean()
        
        total_loss += loss.item()
        total_mae_phi += mae_phi.item() * 180 / np.pi
        total_mae_psi += mae_psi.item() * 180 / np.pi
        count += 1
    
    return total_loss / count, total_mae_phi / count, total_mae_psi / count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--output_dir', default='outputs/vonmises_v1')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Data
    print("Loading data...")
    train_loader, val_loader, test_loader = create_dataloaders(
        args.data, batch_size=args.batch_size, num_workers=4
    )
    
    # Model
    print("Creating model...")
    model = VonMisesTorsionPredictor(
        hidden_dim=args.hidden_dim,
        n_layers=2,
        dropout=0.1
    ).to(device)
    
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Training
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        train_loss, train_mae_phi, train_mae_psi = train_epoch(
            model, train_loader, optimizer, device
        )
        
        val_loss, val_mae_phi, val_mae_psi = evaluate(
            model, val_loader, device
        )
        
        print(f"Train: Loss={train_loss:.3f}, φ={train_mae_phi:.1f}°, ψ={train_mae_psi:.1f}°")
        print(f"Val:   Loss={val_loss:.3f}, φ={val_mae_phi:.1f}°, ψ={val_mae_psi:.1f}°")
        
        scheduler.step(val_loss)
        
        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss,
            }, output_dir / 'best_model.pth')
            print("  ✓ Saved best model")
    
    # Test
    print("\n" + "="*60)
    print("Final Test Evaluation")
    test_loss, test_mae_phi, test_mae_psi = evaluate(
        model, test_loader, device
    )
    print(f"Test: φ MAE={test_mae_phi:.1f}°, ψ MAE={test_mae_psi:.1f}°")


if __name__ == '__main__':
    main()