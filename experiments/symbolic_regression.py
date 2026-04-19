"""Symbolic regression with EMLTree.

Discovers closed-form expressions from data using a binary tree of EML nodes.
Each target function is fit by gradient descent, then weights are snapped to
recover the exact formula.

Run:  python experiments/symbolic_regression.py
"""

import torch
import torch.nn.functional as F
from eml_activations import EMLTree


def train_tree(target_fn, name, depth=3, steps=5000, domain=(0.1, 3.0)):
    best_tree, best_loss = None, float("inf")

    for trial in range(5):
        torch.manual_seed(trial)
        x = torch.linspace(*domain, 512)
        target = target_fn(x)

        tree = EMLTree(depth=depth)
        opt = torch.optim.Adam(tree.parameters(), lr=0.005)

        for step in range(steps):
            tau = max(0.01, 1.0 - step / (steps * 0.6))
            pred = tree(x, tau=tau, gumbel=False)
            loss = F.mse_loss(pred, target)
            opt.zero_grad()
            loss.backward()
            opt.step()

        with torch.no_grad():
            loss_val = F.mse_loss(tree(x, tau=0.01, gumbel=False), target).item()
            if loss_val < best_loss:
                best_loss = loss_val
                best_tree = tree

    # snap and evaluate
    with torch.no_grad():
        x = torch.linspace(*domain, 512)
        target = target_fn(x)
        soft_mse = best_loss
        best_tree.snap_weights()
        hard_mse = F.mse_loss(best_tree(x, tau=0.01, gumbel=False), target).item()

    print(f"\n{name}")
    print(f"  soft MSE:    {soft_mse:.6f}")
    print(f"  snapped MSE: {hard_mse:.6f}")
    return best_tree


def main():
    targets = [
        (torch.exp,  "exp(x)",  2),
        (torch.log,  "ln(x)",   3),
        (torch.sqrt, "sqrt(x)", 4),
        (lambda x: x ** 2, "x²", 5),
    ]

    print("=== EML Symbolic Regression ===")
    for fn, name, depth in targets:
        train_tree(fn, name, depth=depth)


if __name__ == "__main__":
    main()
