"""
Experiment models – two constraint-enforcement techniques sharing one backbone.
"""

from .backbone import MLPBackbone
from .transformer_backbone import TransformerBackbone
from .backbone_factory import build_backbone
from .soft_penalty import SoftPenaltyNet
from .theseus_layer import TheseusLayerNet
from .cvxpy_layer import CvxpyLayerNet


__all__ = [
    "MLPBackbone",
    "TransformerBackbone",
    "build_backbone",
    "SoftPenaltyNet",
    "TheseusLayerNet",
    "CvxpyLayerNet",
]
