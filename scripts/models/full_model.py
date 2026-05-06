# models/full_model.py

import flax.linen as nn
from .encoder import VariableContextEncoder
from .heads import BinnedPredictionHead


class TorsionPredictor(nn.Module):
    """
    Complete torsion angle prediction model.

    Architecture:
        1. VariableContextEncoder  — feedforward, mask-based variable context
        2. BinnedPredictionHead    — joint (φ, ψ) distribution over n_bins² bins
    """

    max_context: int = 3
    embed_dim: int = 64
    hidden_dim: int = 768
    n_layers: int = 3
    dropout_rate: float = 0.1
    n_bins: int = 36  # 36 bins × 36 bins = 1296 joint bins

    def setup(self):
        self.encoder = VariableContextEncoder(
            max_context=self.max_context,
            embed_dim=self.embed_dim,
            hidden_dim=self.hidden_dim,
            n_layers=self.n_layers,
            dropout_rate=self.dropout_rate,
        )
        self.prediction_head = BinnedPredictionHead(
            hidden_dim=self.hidden_dim,
            n_bins=self.n_bins,
        )

    def __call__(self, residues, mask, training=False):
        """
        Args:
            residues: [batch, max_context] int8 residue indices (20 = PAD)
            mask:     [batch, max_context] bool, True = real residue
            training: bool

        Returns:
            logits_joint: [batch, n_bins * n_bins]
                          Reshape to [batch, n_bins, n_bins] to get the
                          joint (φ, ψ) Ramachandran table per example.
                          bin_idx = phi_bin * n_bins + psi_bin
        """
        h = self.encoder(residues, mask, training=training)
        logits_joint = self.prediction_head(h)
        return logits_joint