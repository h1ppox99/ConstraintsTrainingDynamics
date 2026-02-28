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

torch.set_default_dtype(torch.float64)

from .backbone_factory import build_backbone


class SoftPenaltyNet(nn.Module):
    """
    Backbone that predicts y directly from x.

    No feasibility guarantee — training relies on the soft-penalty loss to
    drive constraint violations toward zero.

    Parameters
    ----------
    input_dim     : dimension of x  (= number of equality constraints e)
    output_dim    : dimension of y  (= number of decision variables n)
    backbone_type : 'mlp' or 'transformer'
    **backbone_kwargs : extra arguments forwarded to the backbone constructor
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        backbone_type: str = "mlp",
        **backbone_kwargs,
    ):
        super().__init__()
        self.backbone = build_backbone(
            backbone_type,
            input_dim=input_dim,
            output_dim=output_dim,
            **backbone_kwargs,
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
