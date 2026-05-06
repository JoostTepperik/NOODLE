"""
training/train.py

Training script supporting binned predictions
Uses wandb for experiment tracking
"""

import jax
import jax.numpy as jnp
import optax
import flax
from flax.training import train_state, checkpoints
from pathlib import Path
import time
from tqdm import tqdm
from datetime import datetime
import json
import wandb

import sys
sys.path.append(str(Path(__file__).parent.parent))

from models.dataloader import create_dataloaders
from models.full_model import TorsionPredictor
from training.losses_binned import total_binned_loss


class TrainState(train_state.TrainState):
    """Custom train state"""
    dropout_rng: jax.random.PRNGKey


def create_train_state(rng, model, learning_rate, warmup_steps=1000):
    """Create initial training state"""
    dummy_residues = jnp.ones((1, model.max_context), dtype=jnp.int32)
    dummy_mask = jnp.ones((1, model.max_context), dtype=bool)

    variables = model.init(
        {'params': rng, 'dropout': rng},
        dummy_residues,
        dummy_mask,
        training=False
    )

    schedule = optax.warmup_cosine_decay_schedule(
        init_value=1e-7,
        peak_value=learning_rate,
        warmup_steps=warmup_steps,
        decay_steps=100000,
        end_value=1e-6
    )

    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(schedule)
    )

    return TrainState.create(
        apply_fn=model.apply,
        params=variables['params'],
        tx=optimizer,
        dropout_rng=rng
    )


def make_train_step(lambda_kappa=0.01, n_bins=72):
    """Create JIT-compiled train step function"""

    @jax.jit
    def train_step(state, batch):
        dropout_rng, new_dropout_rng = jax.random.split(state.dropout_rng)

        def loss_fn(params):
            outputs = state.apply_fn(
                {'params': params},
                batch['residues'],
                batch['masks'],
                training=True,
                rngs={'dropout': dropout_rng}
            )

            targets = (batch['phi'], batch['psi'])

            predictions = outputs
            loss, metrics = total_binned_loss(predictions, targets, n_bins)

            return loss, metrics

        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, metrics), grads = grad_fn(state.params)

        state = state.apply_gradients(grads=grads)
        state = state.replace(dropout_rng=new_dropout_rng)

        metrics['grad_norm'] = optax.global_norm(grads)

        return state, metrics

    return train_step


def make_eval_step(lambda_kappa=0.01, n_bins=72):
    """Create JIT-compiled eval step function"""

    @jax.jit
    def eval_step(state, batch):
        outputs = state.apply_fn(
            {'params': state.params},
            batch['residues'],
            batch['masks'],
            training=False,
            rngs={'dropout': jax.random.PRNGKey(0)}
        )

        targets = (batch['phi'], batch['psi'])

        predictions = outputs
        loss, metrics = total_binned_loss(predictions, targets, n_bins)

        return metrics

    return eval_step


def train_epoch(state, train_loader, train_step_fn, epoch):
    """Train for one epoch, logging each step to wandb"""
    epoch_metrics = []
    global_step = (epoch - 1) * len(train_loader)

    pbar = tqdm(train_loader, desc="Training")
    for step, batch in enumerate(pbar):
        state, metrics = train_step_fn(state, batch)
        metrics = {k: float(v) for k, v in metrics.items()}
        epoch_metrics.append(metrics)

        wandb.log({
            'train/loss': metrics['loss'],
            'train/mae_phi': metrics['mae_phi'],
            'train/mae_psi': metrics['mae_psi'],
            'train/grad_norm': metrics['grad_norm'],
        }, step=global_step + step)

        pbar.set_postfix({
            'loss': f"{metrics['loss']:.4f}",
            'mae_φ': f"{metrics['mae_phi']:.1f}°",
            'mae_ψ': f"{metrics['mae_psi']:.1f}°"
        })

    avg_metrics = {
        k: sum(m[k] for m in epoch_metrics) / len(epoch_metrics)
        for k in epoch_metrics[0].keys()
    }

    return state, avg_metrics


def eval_epoch(state, eval_loader, eval_step_fn):
    """Evaluate on validation or test set"""
    epoch_metrics = []

    for batch in tqdm(eval_loader, desc="Evaluating"):
        metrics = eval_step_fn(state, batch)
        metrics = {k: float(v) for k, v in metrics.items()}
        epoch_metrics.append(metrics)

    avg_metrics = {
        k: sum(m[k] for m in epoch_metrics) / len(epoch_metrics)
        for k in epoch_metrics[0].keys()
    }

    return avg_metrics


def train(
    data_path,
    output_dir,
    max_context=3,
    embed_dim=64,
    hidden_dim=768,
    n_layers=3,
    dropout_rate=0.1,
    batch_size=512,
    learning_rate=1e-3,
    lambda_kappa=0.01,
    n_epochs=30,
    warmup_steps=1000,
    eval_every=1,
    save_every=5,
    seed=42,
    n_bins=72,
    wandb_project='torsion-predictor',
    wandb_name=None,
    wandb_tags=None,
):
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = jax.random.PRNGKey(seed)

    config = {
        'max_context': max_context,
        'embed_dim': embed_dim,
        'hidden_dim': hidden_dim,
        'n_layers': n_layers,
        'dropout_rate': dropout_rate,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'lambda_kappa': lambda_kappa,
        'n_epochs': n_epochs,
        'warmup_steps': warmup_steps,
        'seed': seed,
        'n_bins': n_bins,
    }

    wandb.init(
        entity="joosttepperik-radboudumc",
        project="torsion_predictor",
        name=wandb_name,
        tags=wandb_tags,
        config=config,
        dir=str(output_dir),
    )

    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    print("Creating dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        data_path, batch_size=batch_size
    )

    print(f"Training samples: {train_loader.n_samples:,}")
    print(f"Validation samples: {val_loader.n_samples:,}")
    print(f"Test samples: {test_loader.n_samples:,}")

    wandb.summary['n_train'] = train_loader.n_samples
    wandb.summary['n_val'] = val_loader.n_samples
    wandb.summary['n_test'] = test_loader.n_samples

    print("Initializing model...")
    model = TorsionPredictor(
        max_context=max_context,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        dropout_rate=dropout_rate,
        n_bins=n_bins
    )

    rng, init_rng = jax.random.split(rng)
    state = create_train_state(init_rng, model, learning_rate, warmup_steps)

    n_params = sum(p.size for p in jax.tree_util.tree_leaves(state.params))
    print(f"Model parameters: {n_params:,}")
    wandb.summary['n_params'] = n_params

    print("Compiling step functions...")
    train_step_fn = make_train_step(lambda_kappa, n_bins)
    eval_step_fn = make_eval_step(lambda_kappa, n_bins)

    best_val_mae = float('inf')

    for epoch in range(1, n_epochs + 1):
        print(f"\n{'='*70}")
        print(f"Epoch {epoch}/{n_epochs}")
        print(f"{'='*70}")

        epoch_start = time.time()
        state, train_metrics = train_epoch(state, train_loader, train_step_fn, epoch)
        epoch_time = time.time() - epoch_start

        print(f"\nTraining metrics:")
        print(f"  Loss:      {train_metrics['loss']:.4f}")
        print(f"  MAE φ:     {train_metrics['mae_phi']:.2f}°")
        print(f"  MAE ψ:     {train_metrics['mae_psi']:.2f}°")
        print(f"  Grad norm: {train_metrics['grad_norm']:.4f}")
        print(f"  Time:      {epoch_time:.1f}s")

        wandb.log({
            'epoch': epoch,
            'train/epoch_loss': train_metrics['loss'],
            'train/epoch_mae_phi': train_metrics['mae_phi'],
            'train/epoch_mae_psi': train_metrics['mae_psi'],
            'train/epoch_grad_norm': train_metrics['grad_norm'],
            'train/epoch_time': epoch_time,
        }, step=epoch * len(train_loader))

        if epoch % eval_every == 0:
            val_metrics = eval_epoch(state, val_loader, eval_step_fn)

            print(f"\nValidation metrics:")
            print(f"  Loss:  {val_metrics['loss']:.4f}")
            print(f"  MAE φ: {val_metrics['mae_phi']:.2f}°")
            print(f"  MAE ψ: {val_metrics['mae_psi']:.2f}°")

            wandb.log({
                'val/loss': val_metrics['loss'],
                'val/mae_phi': val_metrics['mae_phi'],
                'val/mae_psi': val_metrics['mae_psi'],
            }, step=epoch * len(train_loader))

            val_mae = (val_metrics['mae_phi'] + val_metrics['mae_psi']) / 2
            if val_mae < best_val_mae:
                best_val_mae = val_mae
                print(f"\n✓ New best validation MAE: {val_mae:.2f}°")
                wandb.summary['best_val_mae'] = best_val_mae
                wandb.summary['best_epoch'] = epoch

                checkpoints.save_checkpoint(
                    ckpt_dir=output_dir / 'checkpoints',
                    target=state,
                    step=epoch,
                    prefix='best_',
                    overwrite=True
                )

        if epoch % save_every == 0:
            checkpoints.save_checkpoint(
                ckpt_dir=output_dir / 'checkpoints',
                target=state,
                step=epoch,
                prefix='checkpoint_',
                keep=3
            )

    print(f"\n{'='*70}")
    print("Final Test Evaluation")
    print(f"{'='*70}")

    test_metrics = eval_epoch(state, test_loader, eval_step_fn)

    print(f"\nTest metrics:")
    print(f"  Loss:  {test_metrics['loss']:.4f}")
    print(f"  MAE φ: {test_metrics['mae_phi']:.2f}°")
    print(f"  MAE ψ: {test_metrics['mae_psi']:.2f}°")

    wandb.summary['test_loss'] = test_metrics['loss']
    wandb.summary['test_mae_phi'] = test_metrics['mae_phi']
    wandb.summary['test_mae_psi'] = test_metrics['mae_psi']

    with open(output_dir / 'test_metrics.json', 'w') as f:
        json.dump(test_metrics, f, indent=2)

    wandb.finish()

    print(f"\n✓ Training complete!")
    print(f"Best validation MAE: {best_val_mae:.2f}°")
    print(f"Checkpoints saved to: {output_dir / 'checkpoints'}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train torsion prediction model')

    parser.add_argument('--data', required=True, help='/path/to/training_data.h5')
    parser.add_argument('--output_dir', default='outputs/variable_context_v1.1')

    parser.add_argument('--max_context', type=int, default=21)
    parser.add_argument('--embed_dim', type=int, default=64)
    parser.add_argument('--hidden_dim', type=int, default=768)
    parser.add_argument('--n_layers', type=int, default=3)
    parser.add_argument('--dropout_rate', type=float, default=0.1)

    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--lambda_kappa', type=float, default=0.01)
    parser.add_argument('--n_epochs', type=int, default=30)
    parser.add_argument('--warmup_steps', type=int, default=1000)

    parser.add_argument('--n_bins', type=int, default=36)

    parser.add_argument('--eval_every', type=int, default=1)
    parser.add_argument('--save_every', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--wandb_project', type=str, default='torsion_predictor')
    parser.add_argument('--wandb_name', type=str, default=None)
    parser.add_argument('--wandb_tags', type=str, nargs='*', default=None)

    args = parser.parse_args()

    train(
        data_path=args.data,
        output_dir=args.output_dir,
        max_context=args.max_context,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        dropout_rate=args.dropout_rate,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        lambda_kappa=args.lambda_kappa,
        n_epochs=args.n_epochs,
        warmup_steps=args.warmup_steps,
        eval_every=args.eval_every,
        save_every=args.save_every,
        seed=args.seed,
        n_bins=args.n_bins,
        wandb_project=args.wandb_project,
        wandb_name=args.wandb_name,
        wandb_tags=args.wandb_tags,
    )