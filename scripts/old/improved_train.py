#!/usr/bin/env python3
"""
Improved training with all fixes applied
"""

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from model import TripletTorsionMLP
from dataset import create_dataloaders
from utils import (
    AverageMeter, 
    bins_to_angles, 
    compute_angle_error,
    plot_ramachandran,
    save_checkpoint,
    load_checkpoint
)


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance
    Focuses on hard examples
    """
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha  # Class weights (optional)
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss


class Trainer:
    """Training manager with all improvements"""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.writer = SummaryWriter(self.output_dir / 'logs')
        
        # Load data config
        data_dir = Path(args.data).parent
        with open(data_dir / 'data_config.json') as f:
            self.data_config = json.load(f)
        
        # Create dataloaders
        print("Loading data...")
        self.train_loader, self.val_loader, self.test_loader = create_dataloaders(
            args.data,
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )
        
        print(f"  Train batches: {len(self.train_loader)}")
        print(f"  Val batches: {len(self.val_loader)}")
        print(f"  Test batches: {len(self.test_loader)}")
        
        # Create model (BIGGER)
        print(f"\nCreating model...")
        self.model = TripletTorsionMLP(
            n_phi_bins=self.data_config['n_phi_bins'],
            n_psi_bins=self.data_config['n_psi_bins'],
            residue_embed_dim=args.embed_dim,
            hidden_dim=args.hidden_dim,
            n_hidden_layers=args.n_layers,
            dropout=args.dropout
        ).to(self.device)
        
        print(f"  Parameters: {self.model.get_num_parameters():,}")
        print(f"  Device: {self.device}")
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
        
        # Learning rate scheduler with warmup
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=args.lr,
            epochs=args.epochs,
            steps_per_epoch=len(self.train_loader),
            pct_start=0.1,  # 10% warmup
        )
        
        # Compute class weights if requested
        if args.use_class_weights:
            print("\nComputing class weights...")
            phi_weights, psi_weights = self.compute_class_weights()
            
            if args.use_focal_loss:
                self.phi_criterion = FocalLoss(alpha=phi_weights, gamma=2.0)
                self.psi_criterion = FocalLoss(alpha=psi_weights, gamma=2.0)
                print("  Using Focal Loss with class weights")
            else:
                self.phi_criterion = nn.CrossEntropyLoss(weight=phi_weights)
                self.psi_criterion = nn.CrossEntropyLoss(weight=psi_weights)
                print("  Using Weighted Cross-Entropy")
        else:
            self.phi_criterion = nn.CrossEntropyLoss()
            self.psi_criterion = nn.CrossEntropyLoss()
            print("\nUsing standard Cross-Entropy (no weights)")
        
        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
    
    def compute_class_weights(self):
        """Compute inverse frequency weights"""
        from dataset import TorsionAngleDataset
        
        train_dataset = TorsionAngleDataset(self.args.data, split='train')
        
        phi_bins = train_dataset.phi_bins.numpy()
        psi_bins = train_dataset.psi_bins.numpy()
        
        # Count occurrences
        phi_counts = np.bincount(phi_bins, minlength=self.data_config['n_phi_bins'])
        psi_counts = np.bincount(psi_bins, minlength=self.data_config['n_psi_bins'])
        
        # Inverse frequency (with smoothing)
        phi_weights = 1.0 / (phi_counts + 100)  # Add 100 for smoothing
        psi_weights = 1.0 / (psi_counts + 100)
        
        # Normalize
        phi_weights = phi_weights / phi_weights.mean()
        psi_weights = psi_weights / psi_weights.mean()
        
        # Clip to avoid extreme values
        phi_weights = np.clip(phi_weights, 0.5, 2.0)
        psi_weights = np.clip(psi_weights, 0.5, 2.0)
        
        print(f"  Phi weights: [{phi_weights.min():.2f}, {phi_weights.max():.2f}]")
        print(f"  Psi weights: [{psi_weights.min():.2f}, {psi_weights.max():.2f}]")
        
        # Top 5 most weighted classes
        phi_top = np.argsort(phi_weights)[-5:]
        psi_top = np.argsort(psi_weights)[-5:]
        
        print(f"  Most weighted φ bins: {phi_top} (rare classes)")
        print(f"  Most weighted ψ bins: {psi_top} (rare classes)")
        
        return torch.FloatTensor(phi_weights).to(self.device), \
               torch.FloatTensor(psi_weights).to(self.device)
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        
        losses = AverageMeter()
        phi_accs = AverageMeter()
        psi_accs = AverageMeter()
        
        for batch_idx, (features, targets) in enumerate(self.train_loader):
            # Move to device
            res_prev = features['res_prev'].to(self.device)
            res_curr = features['res_curr'].to(self.device)
            res_next = features['res_next'].to(self.device)
            ca_dist_prev = features['ca_dist_prev'].to(self.device)
            ca_dist_next = features['ca_dist_next'].to(self.device)
            
            phi_target = targets['phi_bin'].to(self.device)
            psi_target = targets['psi_bin'].to(self.device)
            
            # Forward
            phi_logits, psi_logits = self.model(
                res_prev, res_curr, res_next,
                ca_dist_prev, ca_dist_next
            )
            
            # Loss
            loss_phi = self.phi_criterion(phi_logits, phi_target)
            loss_psi = self.psi_criterion(psi_logits, psi_target)
            loss = loss_phi + loss_psi
            
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            
            if self.args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.args.grad_clip
                )
            
            self.optimizer.step()
            self.scheduler.step()  # OneCycleLR updates per batch
            
            # Metrics
            phi_pred = phi_logits.argmax(dim=-1)
            psi_pred = psi_logits.argmax(dim=-1)
            
            phi_acc = (phi_pred == phi_target).float().mean()
            psi_acc = (psi_pred == psi_target).float().mean()
            
            batch_size = res_prev.size(0)
            losses.update(loss.item(), batch_size)
            phi_accs.update(phi_acc.item(), batch_size)
            psi_accs.update(psi_acc.item(), batch_size)
            
            # Logging
            if batch_idx % self.args.log_interval == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f"  Batch [{batch_idx}/{len(self.train_loader)}] "
                      f"Loss: {losses.avg:.4f} "
                      f"φ: {phi_accs.avg:.3f} "
                      f"ψ: {psi_accs.avg:.3f} "
                      f"LR: {current_lr:.2e}")
            
            self.global_step += 1
        
        return losses.avg, phi_accs.avg, psi_accs.avg
    
    @torch.no_grad()
    def validate(self, loader, split='val'):
        """Validate"""
        self.model.eval()
        
        losses = AverageMeter()
        phi_accs = AverageMeter()
        psi_accs = AverageMeter()
        
        all_phi_pred = []
        all_psi_pred = []
        all_phi_true = []
        all_psi_true = []
        all_phi_errors = []
        all_psi_errors = []
        
        for features, targets in loader:
            res_prev = features['res_prev'].to(self.device)
            res_curr = features['res_curr'].to(self.device)
            res_next = features['res_next'].to(self.device)
            ca_dist_prev = features['ca_dist_prev'].to(self.device)
            ca_dist_next = features['ca_dist_next'].to(self.device)
            
            phi_target = targets['phi_bin'].to(self.device)
            psi_target = targets['psi_bin'].to(self.device)
            
            phi_logits, psi_logits = self.model(
                res_prev, res_curr, res_next,
                ca_dist_prev, ca_dist_next
            )
            
            loss_phi = self.phi_criterion(phi_logits, phi_target)
            loss_psi = self.psi_criterion(psi_logits, psi_target)
            loss = loss_phi + loss_psi
            
            phi_pred = phi_logits.argmax(dim=-1)
            psi_pred = psi_logits.argmax(dim=-1)
            
            phi_acc = (phi_pred == phi_target).float().mean()
            psi_acc = (psi_pred == psi_target).float().mean()
            
            batch_size = res_prev.size(0)
            losses.update(loss.item(), batch_size)
            phi_accs.update(phi_acc.item(), batch_size)
            psi_accs.update(psi_acc.item(), batch_size)
            
            # Collect for plotting
            all_phi_pred.extend(phi_pred.cpu().numpy())
            all_psi_pred.extend(psi_pred.cpu().numpy())
            all_phi_true.extend(phi_target.cpu().numpy())
            all_psi_true.extend(psi_target.cpu().numpy())
            
            # Angular errors
            phi_errors = compute_angle_error(
                phi_pred.cpu().numpy(),
                phi_target.cpu().numpy(),
                n_bins=self.data_config['n_phi_bins']
            )
            psi_errors = compute_angle_error(
                psi_pred.cpu().numpy(),
                psi_target.cpu().numpy(),
                n_bins=self.data_config['n_psi_bins']
            )
            
            all_phi_errors.extend(phi_errors)
            all_psi_errors.extend(psi_errors)
        
        mae_phi = np.mean(all_phi_errors)
        mae_psi = np.mean(all_psi_errors)
        
        # Subsample for plotting (to avoid memory issues)
        if len(all_phi_pred) > 10000:
            indices = np.random.choice(len(all_phi_pred), 10000, replace=False)
            all_phi_pred = [all_phi_pred[i] for i in indices]
            all_psi_pred = [all_psi_pred[i] for i in indices]
            all_phi_true = [all_phi_true[i] for i in indices]
            all_psi_true = [all_psi_true[i] for i in indices]
        
        # Convert to angles
        phi_pred_angles = bins_to_angles(np.array(all_phi_pred), self.data_config['n_phi_bins'])
        psi_pred_angles = bins_to_angles(np.array(all_psi_pred), self.data_config['n_psi_bins'])
        phi_true_angles = bins_to_angles(np.array(all_phi_true), self.data_config['n_phi_bins'])
        psi_true_angles = bins_to_angles(np.array(all_psi_true), self.data_config['n_psi_bins'])
        
        # Plot
        plot_file = self.output_dir / f'ramachandran_{split}_epoch{self.epoch}.png'
        plot_ramachandran(
            phi_pred_angles, psi_pred_angles,
            phi_true_angles, psi_true_angles,
            plot_file
        )
        
        return {
            'loss': losses.avg,
            'phi_acc': phi_accs.avg,
            'psi_acc': psi_accs.avg,
            'mae_phi': mae_phi,
            'mae_psi': mae_psi,
        }
    
    def train(self):
        """Main training loop"""
        print("\n" + "="*60)
        print("Starting Training")
        print("="*60)
        
        for epoch in range(self.args.epochs):
            self.epoch = epoch
            
            print(f"\nEpoch {epoch+1}/{self.args.epochs}")
            print("-" * 60)
            
            start_time = time.time()
            train_loss, train_phi_acc, train_psi_acc = self.train_epoch()
            epoch_time = time.time() - start_time
            
            print(f"\nTrain: Loss={train_loss:.4f}, φ={train_phi_acc:.3f}, ψ={train_psi_acc:.3f}, Time={epoch_time:.1f}s")
            
            val_metrics = self.validate(self.val_loader, split='val')
            
            print(f"Val: Loss={val_metrics['loss']:.4f}, φ={val_metrics['phi_acc']:.3f}, ψ={val_metrics['psi_acc']:.3f}, φ_MAE={val_metrics['mae_phi']:.1f}°, ψ_MAE={val_metrics['mae_psi']:.1f}°")
            
            # TensorBoard
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Loss/val', val_metrics['loss'], epoch)
            self.writer.add_scalar('Acc/train_phi', train_phi_acc, epoch)
            self.writer.add_scalar('Acc/val_phi', val_metrics['phi_acc'], epoch)
            self.writer.add_scalar('MAE/phi', val_metrics['mae_phi'], epoch)
            self.writer.add_scalar('MAE/psi', val_metrics['mae_psi'], epoch)
            
            # Save checkpoint
            avg_acc = (val_metrics['phi_acc'] + val_metrics['psi_acc']) / 2
            is_best = avg_acc > self.best_val_acc
            
            if is_best:
                self.best_val_acc = avg_acc
                print(f"  ✓ New best! Avg accuracy: {avg_acc:.3f}")
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'val_acc': avg_acc,
                'best_val_acc': self.best_val_acc,
            }
            
            save_checkpoint(checkpoint, self.output_dir, 'checkpoint_latest.pth')
            if is_best:
                save_checkpoint(checkpoint, self.output_dir, 'checkpoint_best.pth')
        
        # Final test
        print("\n" + "="*60)
        print("Final Test Evaluation")
        print("="*60)
        
        test_metrics = self.validate(self.test_loader, split='test')
        print(f"Test: φ={test_metrics['phi_acc']:.3f}, ψ={test_metrics['psi_acc']:.3f}, φ_MAE={test_metrics['mae_phi']:.1f}°, ψ_MAE={test_metrics['mae_psi']:.1f}°")
        
        self.writer.close()


def main():
    parser = argparse.ArgumentParser()
    
    # Data
    parser.add_argument('--data', required=True)
    parser.add_argument('--batch_size', type=int, default=512)  # Smaller for stability
    parser.add_argument('--num_workers', type=int, default=4)
    
    # Model (BIGGER)
    parser.add_argument('--embed_dim', type=int, default=32)      # Was 16
    parser.add_argument('--hidden_dim', type=int, default=512)    # Was 256
    parser.add_argument('--n_layers', type=int, default=4)        # Was 2
    parser.add_argument('--dropout', type=int, default=0.2)       # Was 0.1
    
    # Training
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=3e-4)         # Lower LR
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    
    # Loss
    parser.add_argument('--use_class_weights', action='store_true', default=True)
    parser.add_argument('--use_focal_loss', action='store_true')
    
    # Output
    parser.add_argument('--output_dir', default='outputs/improved_run')
    parser.add_argument('--log_interval', type=int, default=50)
    
    args = parser.parse_args()
    
    trainer = Trainer(args)
    trainer.train()


if __name__ == '__main__':
    main()