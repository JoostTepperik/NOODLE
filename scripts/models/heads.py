# models/heads.py

import jax.numpy as jnp
import flax.linen as nn


class BinnedPredictionHead(nn.Module):
    """
    Joint binned probability distribution over (φ, ψ) angle pairs.

    Outputs a single distribution over n_bins² joint bins, capturing
    correlations between φ and ψ that independent heads cannot express.
    The output can be reshaped to [batch, n_bins, n_bins] to read off the
    joint Ramachandran table directly.
    """

    hidden_dim: int = 768
    n_bins: int = 36  # 36 bins = 10° per bin → 36×36 = 1296 joint bins

    @nn.compact
    def __call__(self, h):
        """
        Args:
            h: [batch, hidden_dim] encoded context

        Returns:
            logits_joint: [batch, n_bins * n_bins] unnormalized log probs
                          over joint (φ, ψ) bins, in row-major order
                          (φ is the outer axis: bin_idx = phi_bin * n_bins + psi_bin)
        """
        # Shared compression layer — same as before
        h_shared = nn.Dense(self.hidden_dim // 2)(h)
        h_shared = nn.gelu(h_shared)

        # Single joint branch — wider than either old branch to retain capacity
        h_joint = nn.Dense(512)(h_shared)
        h_joint = nn.gelu(h_joint)

        # Project to full joint space: n_bins² outputs
        logits_joint = nn.Dense(self.n_bins * self.n_bins, name='logits_joint')(h_joint)

        return logits_joint  # [batch, 1296]