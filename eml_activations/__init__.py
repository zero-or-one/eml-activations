"""eml-activations: EML operator as PyTorch activation functions.

Based on arXiv:2603.21852 — "All elementary functions from a single binary operator"
"""

from .core import (
    eml,
    eml_e,
    eml_exp,
    eml_ln,
    eml_neg,
    eml_inv,
    eml_sub,
    eml_add,
    eml_mul,
    eml_div,
    eml_pow,
    eml_sqrt,
    eml_sigmoid,
    EML,
    EMLExp,
    EMLSigmoid,
)
from .tree import EMLTree, EMLNode, EMLLeaf

__all__ = [
    "eml",
    "eml_e",
    "eml_exp",
    "eml_ln",
    "eml_neg",
    "eml_inv",
    "eml_sub",
    "eml_add",
    "eml_mul",
    "eml_div",
    "eml_pow",
    "eml_sqrt",
    "eml_sigmoid",
    "EML",
    "EMLExp",
    "EMLSigmoid",
    "EMLTree",
    "EMLNode",
    "EMLLeaf",
]
