"""
Shared MLP backbone for all constraint-enforcement techniques.

Every model in this experiment uses the same encoder architecture so that
differences in training dynamics come solely from the constraint-handling
method, not from the network capacity.

Usage
-----
    from backbone import MLPBackbone

    bb = MLPBackbone(input_dim=50, output_dim=100, hidden_dim=200, n_hidden=2)
    y0 = bb(x)   # (batch, output_dim)

The backbone is a plain stack of [Linear → BatchNorm → ReLU → Dropout] blocks
followed by a final Linear layer.  All hyperparameters (width, depth, dropout)
are exposed so they can be swept without touching the model files.
"""

import torch
import torch.nn as nn

torch.set_default_dtype(torch.float64)


class MLPBackbone(nn.Module):
    """
    Simple feed-forward MLP.

    Parameters
    ----------
    input_dim   : int   – dimension of the input x
    output_dim  : int   – dimension of the raw prediction y₀
    hidden_dim  : int   – width of each hidden layer  (default 200)
    n_hidden    : int   – number of hidden layers      (default 2)
    dropout     : float – dropout probability           (default 0.2)
    use_batchnorm : bool – include BatchNorm1d layers   (default True)
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 200,
        n_hidden: int = 2,
        dropout: float = 0.2,
        use_batchnorm: bool = True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        layers: list[nn.Module] = []

        # Hidden layers
        prev_dim = input_dim
        for _ in range(n_hidden):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout))
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))

        # Kaiming init for Linear layers
        for m in layers:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map input x → raw prediction y₀.  Shape: (b, input_dim) → (b, output_dim)."""
        return self.net(x)
