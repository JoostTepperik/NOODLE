"""
training/losses_binned.py

Loss functions for joint (φ, ψ) binned predictions.

Primary loss: Soft target cross-entropy with chord-distance-derived targets.
              For each training example the target distribution is:

                  q_k ∝ exp(-d(z_k, z_true) / τ)

              where z = (sin φ, cos φ, sin ψ, cos ψ) ∈ R⁴ and d is the
              Euclidean chord distance. This is a proper scoring rule,
              respects circular geometry without bin-edge artifacts, and
              produces calibrated per-bin probabilities suitable for
              energy = -log P(φ_bin, ψ_bin).

Monitoring:   Expected angle MAE in degrees (φ and ψ separately)
              Sin/cos unit-circle error

JAX note:     All geometry is built in pure NumPy and stored as NumPy arrays.
              NumPy arrays inside jit are compile-time constants — no tracer
              leaks. jnp.asarray() inside loss functions is a zero-cost view.
"""

import numpy as np
import jax
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Module-level geometry — pure NumPy, no JAX, no tracer leaks
# ---------------------------------------------------------------------------

def _build_joint_geometry(n_bins: int):
    """
    Build all fixed geometric quantities for a given n_bins.
    Everything is computed in pure NumPy and returned as NumPy arrays.

    Returns:
        bin_centers:  [n_bins]        bin-centre angles in radians
        z_bins:       [n_bins², 4]    sin/cos embedding of each joint bin
        d_matrix:     [n_bins², 4]    pairwise chord distances in R⁴
    """
    bin_width = 2 * np.pi / n_bins
    bin_centers = -np.pi + (np.arange(n_bins) + 0.5) * bin_width  # [n_bins]

    # sin/cos for each individual bin
    sin_b = np.sin(bin_centers)
    cos_b = np.cos(bin_centers)

    # Joint bin embedding z_{ij} = (sin φ_i, cos φ_i, sin ψ_j, cos ψ_j)
    n_joint = n_bins * n_bins
    z_bins = np.empty((n_joint, 4), dtype=np.float32)
    for i in range(n_bins):
        for j in range(n_bins):
            k = i * n_bins + j
            z_bins[k, 0] = sin_b[i]
            z_bins[k, 1] = cos_b[i]
            z_bins[k, 2] = sin_b[j]
            z_bins[k, 3] = cos_b[j]

    # Pairwise chord distances between joint bins — shape [n_joint, n_joint]
    diff = z_bins[:, None, :] - z_bins[None, :, :]     # [n_joint, n_joint, 4]
    d_matrix = np.sqrt((diff ** 2).sum(-1) + 1e-12).astype(np.float32)

    return (
        bin_centers.astype(np.float32),   # pure NumPy — intentional
        z_bins,                           # pure NumPy — intentional
        d_matrix,                         # pure NumPy — intentional
    )


# Cache keyed by n_bins — populated once, never inside JIT
_GEOMETRY_CACHE: dict = {}


def _get_geometry(n_bins: int):
    """Return cached NumPy geometry arrays, building them on first call."""
    if n_bins not in _GEOMETRY_CACHE:
        _GEOMETRY_CACHE[n_bins] = _build_joint_geometry(n_bins)
    return _GEOMETRY_CACHE[n_bins]


# Pre-warm the default so the first training step isn't slow
_get_geometry(36)


# ---------------------------------------------------------------------------
# Bin ↔ angle helpers
# ---------------------------------------------------------------------------

def angle_to_bin(angles, n_bins: int = 36):
    """
    Convert angles in radians to bin indices.

    Args:
        angles:  [batch]  angles in radians, range [-π, π]
        n_bins:  number of uniform bins

    Returns:
        bin_indices: [batch] integer indices in [0, n_bins - 1]
    """
    bin_width = 2 * jnp.pi / n_bins
    shifted = angles + jnp.pi                                     # [0, 2π)
    indices = jnp.floor(shifted / bin_width).astype(jnp.int32)
    return jnp.clip(indices, 0, n_bins - 1)


def bin_to_angle(bin_indices, n_bins: int = 36):
    """
    Convert bin indices to bin-centre angles in radians.

    Args:
        bin_indices: [batch] integer bin indices
        n_bins:      number of bins

    Returns:
        angles: [batch] bin-centre angles in radians
    """
    bin_width = 2 * jnp.pi / n_bins
    return (bin_indices + 0.5) * bin_width - jnp.pi


# ---------------------------------------------------------------------------
# Primary loss: soft target cross-entropy with chord-distance targets
# ---------------------------------------------------------------------------

def soft_target_cross_entropy(logits_joint, phi_true, psi_true,
                               n_bins: int = 36, tau: float = 0.25):
    """
    Cross-entropy against a soft target distribution derived from chord
    distances in sin/cos space.

    For each example the target is:

        q_k ∝ exp(-d(z_k, z_true) / τ)

    where z_k = (sin φ_k, cos φ_k, sin ψ_k, cos ψ_k) is the embedding of
    joint bin k, and z_true is the embedding of the true (φ, ψ).  This is
    equivalent to a product-of-von-Mises kernel in the chord-distance metric,
    generalised to the 2-torus.

    τ controls smoothing width:
        τ → 0   : approaches hard one-hot NLL
        τ = 0.25: adjacent bins (chord ≈ 0.17) get ~50% weight of true bin
        τ → ∞   : uniform distribution

    Args:
        logits_joint: [batch, n_bins²]  unnormalised logits
        phi_true:     [batch]           true φ in radians
        psi_true:     [batch]           true ψ in radians
        n_bins:       bins per angle
        tau:          temperature for target smoothing

    Returns:
        loss: scalar mean cross-entropy
    """
    # Retrieve NumPy geometry — jnp.asarray is a zero-cost view inside JIT
    _, z_bins_np, _ = _get_geometry(n_bins)
    z_bins = jnp.asarray(z_bins_np)                       # [n_bins², 4]

    # True angle embedding: z_true ∈ R⁴
    z_true = jnp.stack(
        [jnp.sin(phi_true), jnp.cos(phi_true),
         jnp.sin(psi_true), jnp.cos(psi_true)],
        axis=-1
    )                                                      # [batch, 4]

    # Chord distance from each bin to the true angle
    diff = z_bins[None, :, :] - z_true[:, None, :]        # [batch, n_bins², 4]
    d_to_true = jnp.sqrt((diff ** 2).sum(-1) + 1e-12)     # [batch, n_bins²]

    # Soft target distribution — normalised in log space for numerical stability
    log_q = -d_to_true / tau
    log_q = log_q - jax.scipy.special.logsumexp(log_q, axis=-1, keepdims=True)

    # Cross-entropy: -Σ_k q_k · log p_k
    log_p = jax.nn.log_softmax(logits_joint, axis=-1)      # [batch, n_bins²]
    loss = -(jnp.exp(log_q) * log_p).sum(-1)               # [batch]

    return loss.mean()


# ---------------------------------------------------------------------------
# Monitoring metrics
# ---------------------------------------------------------------------------

def _marginal_probs(p_joint, n_bins: int):
    """
    Derive marginal distributions from joint probabilities.

    Args:
        p_joint: [batch, n_bins²]

    Returns:
        p_phi: [batch, n_bins]  marginal over φ (sum over ψ axis)
        p_psi: [batch, n_bins]  marginal over ψ (sum over φ axis)
    """
    p_2d = p_joint.reshape(-1, n_bins, n_bins)  # [batch, phi_bins, psi_bins]
    p_phi = p_2d.sum(axis=2)                    # [batch, n_bins]
    p_psi = p_2d.sum(axis=1)                    # [batch, n_bins]
    return p_phi, p_psi


def compute_expected_angles(p_joint, n_bins: int = 36):
    """
    Circular mean (φ, ψ) from the joint distribution via marginals.

        θ̂ = atan2(Σ_k p_k sin θ_k,  Σ_k p_k cos θ_k)

    Args:
        p_joint: [batch, n_bins²]  probabilities (post-softmax)
        n_bins:  bins per angle

    Returns:
        phi_hat: [batch]  expected φ in radians
        psi_hat: [batch]  expected ψ in radians
    """
    bin_centers_np, _, _ = _get_geometry(n_bins)
    bin_centers = jnp.asarray(bin_centers_np)              # [n_bins]

    p_phi, p_psi = _marginal_probs(p_joint, n_bins)

    phi_hat = jnp.arctan2(
        (p_phi * jnp.sin(bin_centers)).sum(-1),
        (p_phi * jnp.cos(bin_centers)).sum(-1),
    )
    psi_hat = jnp.arctan2(
        (p_psi * jnp.sin(bin_centers)).sum(-1),
        (p_psi * jnp.cos(bin_centers)).sum(-1),
    )
    return phi_hat, psi_hat


def compute_expected_angle_mae(logits_joint, phi_true, psi_true,
                                n_bins: int = 36):
    """
    MAE between circular-mean predictions and true angles, in degrees.

    Args:
        logits_joint: [batch, n_bins²]
        phi_true:     [batch]
        psi_true:     [batch]
        n_bins:       bins per angle

    Returns:
        mae_phi: scalar degrees
        mae_psi: scalar degrees
    """
    p = jax.nn.softmax(logits_joint)
    phi_hat, psi_hat = compute_expected_angles(p, n_bins)

    def _circular_mae_deg(pred, true):
        err = jnp.abs(pred - true)
        err = jnp.minimum(err, 2 * jnp.pi - err)
        return (err * 180.0 / jnp.pi).mean()

    return _circular_mae_deg(phi_hat, phi_true), _circular_mae_deg(psi_hat, psi_true)


def compute_sincos_error(logits_joint, phi_true, psi_true, n_bins: int = 36):
    """
    Mean unit-circle distance between expected and true angle embeddings.

        err = ||E_p[(sin θ, cos θ)] - (sin θ_true, cos θ_true)||₂

    Range [0, 2], where 0 is perfect.

    Args:
        logits_joint: [batch, n_bins²]
        phi_true:     [batch]
        psi_true:     [batch]
        n_bins:       bins per angle

    Returns:
        sincos_err_phi: scalar
        sincos_err_psi: scalar
    """
    p = jax.nn.softmax(logits_joint)
    phi_hat, psi_hat = compute_expected_angles(p, n_bins)

    def _circle_dist(pred, true):
        return jnp.sqrt(
            (jnp.sin(pred) - jnp.sin(true)) ** 2 +
            (jnp.cos(pred) - jnp.cos(true)) ** 2 + 1e-12
        ).mean()

    return _circle_dist(phi_hat, phi_true), _circle_dist(psi_hat, psi_true)


# ---------------------------------------------------------------------------
# Top-level entry point called by train.py
# ---------------------------------------------------------------------------

def total_binned_loss(predictions, targets, n_bins: int = 36, tau: float = 0.25):
    """
    Compute soft-target CE loss and all monitoring metrics.

    Args:
        predictions: logits_joint  [batch, n_bins²]  — single tensor, not tuple
        targets:     (phi_true, psi_true)  each [batch] in radians
        n_bins:      bins per angle
        tau:         temperature for soft target construction

    Returns:
        loss:    scalar soft-target cross-entropy
        metrics: dict — loss, mae_phi, mae_psi, sincos_err_phi, sincos_err_psi
    """
    logits_joint = predictions
    phi_true, psi_true = targets

    loss = soft_target_cross_entropy(logits_joint, phi_true, psi_true, n_bins, tau)

    mae_phi, mae_psi = compute_expected_angle_mae(
        logits_joint, phi_true, psi_true, n_bins
    )
    sincos_phi, sincos_psi = compute_sincos_error(
        logits_joint, phi_true, psi_true, n_bins
    )

    metrics = {
        'loss':           loss,
        'mae_phi':        mae_phi,
        'mae_psi':        mae_psi,
        'sincos_err_phi': sincos_phi,
        'sincos_err_psi': sincos_psi,
    }

    return loss, metrics