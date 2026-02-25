"""
Soft-penalty model for QCQP optimisation learning.

Architecture:  MLP backbone only – no projection or differentiable solver.
Constraint satisfaction is encouraged purely through a penalty term in the
training loss:

    L(θ) = objective(y_θ)  +  λ · ||constraint_residuals(y_θ)||²

The model itself is agnostic to the penalty weight λ; that is handled by the
trainer / loss function in QCQPProblem.get_soft_penalty_loss().
"""

import torch
import torch.nn as nn

from .backbone import MLPBackbone

torch.set_default_dtype(torch.float64)


class SoftPenaltyNet(nn.Module):
    """
    Plain MLP that predicts y directly from x.

    No feasibility guarantee — training relies on the soft-penalty loss to
    drive constraint violations toward zero.

    Parameters
    ----------
    input_dim   : dimension of x  (= number of equality constraints e)
    output_dim  : dimension of y  (= number of decision variables n)
    hidden_dim  : MLP hidden-layer width
    n_hidden    : number of hidden layers
    dropout     : dropout probability
    use_batchnorm : whether to use BatchNorm1d
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
        self.backbone = MLPBackbone(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            n_hidden=n_hidden,
            dropout=dropout,
            use_batchnorm=use_batchnorm,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (b, e)  problem parameters

        Returns
        -------
        y : (b, n)  predicted (unconstrained) solution
        """
        return self.backbone(x)
