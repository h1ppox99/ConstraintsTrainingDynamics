"""
Transformer backbone for tabular (vector) data.

Maps a flat input vector x ∈ ℝ^{input_dim} to a prediction y₀ ∈ ℝ^{output_dim}
using the following architecture:

    1. **Tokenisation** – the input vector is linearly projected into a sequence
       of ``n_tokens`` embeddings of size ``d_model``.  A learnable [CLS] token
       is prepended, giving a sequence of length ``n_tokens + 1``.
    2. **Positional encoding** – learnable positional embeddings are added.
    3. **Transformer encoder** – ``n_layers`` standard encoder blocks
       (multi-head self-attention + feed-forward) with pre-LayerNorm.
    4. **Readout** – the [CLS] representation is passed through a small MLP
       head to produce the final output.

This follows the *FT-Transformer* design (Gorishniy et al., 2021) adapted to
our regression setting.

Usage
-----
    from models.transformer_backbone import TransformerBackbone

    bb = TransformerBackbone(input_dim=20, output_dim=50, d_model=64,
                             n_heads=4, n_layers=3)
    y0 = bb(x)   # (batch, output_dim)
"""

import math

import torch
import torch.nn as nn

torch.set_default_dtype(torch.float64)


class TransformerBackbone(nn.Module):
    """
    Transformer encoder for vector (non-sequential) inputs.

    Parameters
    ----------
    input_dim     : int   – dimension of the input x
    output_dim    : int   – dimension of the raw prediction y₀
    d_model       : int   – transformer embedding dimension   (default 64)
    n_heads       : int   – number of attention heads          (default 4)
    n_layers      : int   – number of encoder layers           (default 3)
    dim_feedforward : int – FFN hidden dimension               (default 128)
    n_tokens      : int   – number of tokens to split input into (default 8)
    dropout       : float – dropout probability                (default 0.1)
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 3,
        dim_feedforward: int = 128,
        n_tokens: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.d_model = d_model
        self.n_tokens = n_tokens

        # --- Tokenisation: project input into n_tokens embeddings ---
        self.token_proj = nn.Linear(input_dim, n_tokens * d_model)

        # Learnable [CLS] token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Learnable positional embeddings (CLS + n_tokens)
        self.pos_embed = nn.Parameter(
            torch.randn(1, n_tokens + 1, d_model) * 0.02
        )

        # --- Transformer encoder ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-LN (more stable training)
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
        )
        self.final_norm = nn.LayerNorm(d_model)

        # --- Readout head: [CLS] → output_dim ---
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, output_dim),
        )

        self._init_weights()

    def _init_weights(self):
        """Xavier uniform for linear layers, zeros for biases."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Map input x → raw prediction y₀.

        Shape: (b, input_dim) → (b, output_dim).
        """
        b = x.size(0)

        # Tokenise: (b, input_dim) → (b, n_tokens, d_model)
        tokens = self.token_proj(x).view(b, self.n_tokens, self.d_model)

        # Prepend [CLS]
        cls = self.cls_token.expand(b, -1, -1)  # (b, 1, d_model)
        tokens = torch.cat([cls, tokens], dim=1)  # (b, n_tokens+1, d_model)

        # Add positional embeddings
        tokens = tokens + self.pos_embed

        # Transformer encoder
        tokens = self.encoder(tokens)  # (b, n_tokens+1, d_model)
        tokens = self.final_norm(tokens)

        # Readout from [CLS] token
        cls_out = tokens[:, 0]  # (b, d_model)
        return self.head(cls_out)  # (b, output_dim)
