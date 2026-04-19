"""EML binary-tree master formula for gradient-based symbolic regression.

Section 4.3 of arXiv:2603.21852: parameterised EML trees whose weights
can be optimised with Adam and snapped to {0,1} to recover exact formulas.

Each internal node computes  eml(left, right)  where each input is a
soft-selected mix of {1, x, child_result} via a Gumbel-Softmax over
three logits (α, β, γ).  Leaf nodes select from {1, x} only.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .core import eml


class EMLLeaf(nn.Module):
    """Leaf: soft-select between constant 1 and input x."""

    def __init__(self) -> None:
        super().__init__()
        self.logits = nn.Parameter(torch.randn(2))  # [α_1, β_x]

    def forward(self, x: Tensor, *, tau: float = 1.0) -> Tensor:
        w = F.gumbel_softmax(self.logits.expand(x.shape[0], -1), tau=tau, hard=False)
        return w[:, 0] * 1.0 + w[:, 1] * x


class EMLNode(nn.Module):
    """Internal node: eml(left_input, right_input).

    Each input is a soft-select over {1, x, child_output} (3 logits).
    """

    def __init__(self, left: nn.Module, right: nn.Module) -> None:
        super().__init__()
        self.left_child = left
        self.right_child = right
        self.left_logits = nn.Parameter(torch.randn(3))   # [1, x, child]
        self.right_logits = nn.Parameter(torch.randn(3))

    def _select(self, logits: nn.Parameter, x: Tensor, child: Tensor, tau: float) -> Tensor:
        w = F.gumbel_softmax(logits.expand(x.shape[0], -1), tau=tau, hard=False)
        return w[:, 0] * 1.0 + w[:, 1] * x + w[:, 2] * child

    def forward(self, x: Tensor, *, tau: float = 1.0) -> Tensor:
        lc = self.left_child(x, tau=tau)
        rc = self.right_child(x, tau=tau)
        left_in = self._select(self.left_logits, x, lc, tau)
        right_in = self._select(self.right_logits, x, rc, tau)
        # clamp to avoid exp overflow / log-of-negative
        left_in = torch.clamp(left_in, -20.0, 20.0)
        right_in = torch.clamp(right_in, 1e-12, None)
        return eml(left_in, right_in)


class EMLTree(nn.Module):
    """Full binary EML tree of a given depth for symbolic regression.

    Parameters = 5·2^depth − 6  (matches paper formula).

    Usage::

        tree = EMLTree(depth=3)
        x = torch.linspace(0.1, 3.0, 256)
        pred = tree(x, tau=1.0)
        loss = F.mse_loss(pred, target)
    """

    def __init__(self, depth: int = 3) -> None:
        super().__init__()
        self.root = self._build(depth)

    def _build(self, d: int) -> nn.Module:
        if d == 0:
            return EMLLeaf()
        return EMLNode(self._build(d - 1), self._build(d - 1))

    def forward(self, x: Tensor, *, tau: float = 1.0) -> Tensor:
        if x.dim() == 0:
            x = x.unsqueeze(0)
        return self.root(x, tau=tau)

    def snap_weights(self) -> None:
        """Snap all logits to one-hot (hard selection of 1, x, or child)."""
        with torch.no_grad():
            for p in self.parameters():
                idx = p.argmax()
                p.zero_()
                p[idx] = 1.0
