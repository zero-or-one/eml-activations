"""Core EML (Exp-Minus-Log) operator and derived activation functions.

Based on: "All elementary functions from a single binary operator"
           Andrzej Odrzywołek, arXiv:2603.21852
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

# ---------------------------------------------------------------------------
# Primitive: eml(x, y) = exp(x) - ln(y)
# ---------------------------------------------------------------------------

def eml(x: Tensor, y: Tensor) -> Tensor:
    """EML operator: exp(x) - ln(y).  Works over complex intermediates."""
    return torch.exp(x) - torch.log(y)


# ---------------------------------------------------------------------------
# Constants derived purely from eml and 1
# ---------------------------------------------------------------------------

def eml_e(*, dtype=torch.float32, device=None) -> Tensor:
    """e = eml(1, 1) = exp(1) - ln(1) = e"""
    one = torch.tensor(1.0, dtype=dtype, device=device)
    return eml(one, one)


# ---------------------------------------------------------------------------
# Unary primitives built from eml  (Table 4 / Wikipedia constructions)
# ---------------------------------------------------------------------------

def eml_exp(x: Tensor) -> Tensor:
    """exp(x) = eml(x, 1)"""
    return eml(x, torch.ones_like(x))


def eml_ln(x: Tensor) -> Tensor:
    """ln(x) = eml(1, eml(eml(1, x), 1))"""
    one = torch.ones_like(x)
    return eml(one, eml(eml(one, x), one))


def eml_neg(x: Tensor) -> Tensor:
    """-x = 0 - x = eml(ln(0), exp(x))  via subtraction identity."""
    return -x  # direct; pure-EML form is depth-8


def eml_inv(x: Tensor) -> Tensor:
    """1/x = exp(-ln(x))"""
    return eml_exp(eml_neg(eml_ln(x)))


def eml_sub(x: Tensor, y: Tensor) -> Tensor:
    """x - y = eml(ln(x), exp(y))"""
    return eml(eml_ln(x), eml_exp(y))


def eml_add(x: Tensor, y: Tensor) -> Tensor:
    """x + y = x - (-y)"""
    return eml_sub(x, eml_neg(y))


def eml_mul(x: Tensor, y: Tensor) -> Tensor:
    """x * y = exp(ln(x) + ln(y))"""
    return eml_exp(eml_add(eml_ln(x), eml_ln(y)))


def eml_div(x: Tensor, y: Tensor) -> Tensor:
    """x / y = x * (1/y)"""
    return eml_mul(x, eml_inv(y))


def eml_pow(x: Tensor, y: Tensor) -> Tensor:
    """x^y = exp(y * ln(x))"""
    return eml_exp(eml_mul(y, eml_ln(x)))


def eml_sqrt(x: Tensor) -> Tensor:
    """sqrt(x) = x^(1/2)"""
    half = torch.tensor(0.5, dtype=x.dtype, device=x.device)
    return eml_pow(x, half)


def eml_sigmoid(x: Tensor) -> Tensor:
    """σ(x) = 1 / (1 + exp(-x))"""
    one = torch.ones_like(x)
    return eml_inv(eml_add(one, eml_exp(eml_neg(x))))


# ---------------------------------------------------------------------------
# nn.Module activation wrappers
# ---------------------------------------------------------------------------

class EML(nn.Module):
    """Raw EML(x, y) = exp(x) - ln(y) as a two-input module."""
    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        return eml(x, y)


class EMLExp(nn.Module):
    """exp(x) built from EML."""
    def forward(self, x: Tensor) -> Tensor:
        return eml_exp(x)


class EMLSigmoid(nn.Module):
    """Sigmoid built from EML primitives."""
    def forward(self, x: Tensor) -> Tensor:
        return eml_sigmoid(x)
