"""
models/encoder.py

Fast feedforward encoder - no attention, handles variable context via masking
"""

import jax
import jax.numpy as jnp
import flax.linen as nn


class VariableContextEncoder(nn.Module):
    """
    Feedforward encoder that handles variable-length context via masking
    
    Much faster than attention for short sequences (7-mer)
    Variable context handled by zeroing out padded positions
    """
    
    max_context: int = 3
    embed_dim: int = 64
    hidden_dim: int = 768
    n_layers: int = 3
    dropout_rate: float = 0.1
    
    @nn.compact
    def __call__(self, residues, mask, training=False):
        """
        Args:
            residues: [batch, max_context] int8, residue indices (20=PAD)
            mask: [batch, max_context] bool, True=real, False=padded
            training: bool, whether in training mode
        
        Returns:
            h: [batch, hidden_dim] context representation
        """
        batch_size = residues.shape[0]
        
        # Single shared embedding table for all positions
        embed = nn.Embed(
            num_embeddings=21,  # 20 AA + PAD
            features=self.embed_dim,
            name='residue_embed'
        )
        
        # Embed all positions
        embeds = embed(residues)  # [batch, max_context, embed_dim]
        
        # CRITICAL: Zero out padded positions
        # This is how we handle variable context!
        mask_expanded = mask[:, :, None]  # [batch, max_context, 1]
        embeds = embeds * mask_expanded  # Padded positions become zero vectors
        
        # Flatten: concatenate all position embeddings
        # Shape: [batch, max_context * embed_dim]
        h = embeds.reshape(batch_size, -1)
        
        # Deep feedforward network
        # This learns position-specific weights and interactions
        for i in range(self.n_layers):
            h = nn.Dense(self.hidden_dim, name=f'dense_{i}')(h)
            h = nn.gelu(h)
            h = nn.LayerNorm(name=f'ln_{i}')(h)
            h = nn.Dropout(self.dropout_rate, deterministic=not training)(h)
        
        # Final projection
        h = nn.Dense(self.hidden_dim, name='final_dense')(h)
        
        return h