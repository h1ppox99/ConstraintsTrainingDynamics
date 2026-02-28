"""
Backbone factory – instantiate a backbone by name.

This module lives separately from ``__init__.py`` to avoid circular imports
(the model classes themselves call ``build_backbone``).
"""

from .backbone import MLPBackbone
from .transformer_backbone import TransformerBackbone


def build_backbone(
    backbone_type: str,
    input_dim: int,
    output_dim: int,
    **kwargs,
):
    """Instantiate a backbone by name.

    Parameters
    ----------
    backbone_type : ``'mlp'`` or ``'transformer'``
    input_dim, output_dim : dimensions forwarded to the backbone.
    **kwargs : extra keyword arguments forwarded to the chosen backbone class.
               Only keys recognised by the selected backbone are used; the rest
               are silently ignored so callers can pass a superset.
    """
    if backbone_type == "mlp":
        valid = {"hidden_dim", "n_hidden", "dropout", "use_batchnorm"}
        return MLPBackbone(
            input_dim=input_dim,
            output_dim=output_dim,
            **{k: v for k, v in kwargs.items() if k in valid},
        )
    elif backbone_type == "transformer":
        valid = {"d_model", "n_heads", "n_layers", "dim_feedforward",
                 "n_tokens", "dropout"}
        return TransformerBackbone(
            input_dim=input_dim,
            output_dim=output_dim,
            **{k: v for k, v in kwargs.items() if k in valid},
        )
    else:
        raise ValueError(
            f"Unknown backbone type: {backbone_type!r}. "
            f"Choose 'mlp' or 'transformer'."
        )
