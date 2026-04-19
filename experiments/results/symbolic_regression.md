# Experiment: Symbolic Regression with EMLTree

Using gradient-based optimisation to discover closed-form mathematical expressions from data, via a binary tree of EML nodes.

## How it works

An `EMLTree` of depth `d` is a full binary tree where every internal node computes `eml(left, right) = exp(left) − ln(right)`. Each input to a node is a soft-selected mix of `{1, x, child_output}` controlled by learnable logits. Temperature annealing drives the soft selection toward a hard one-hot choice, and `snap_weights()` locks it in to recover an exact symbolic formula.

This is the "master formula" from Section 4.3 of the paper — the continuous-math analogue of a programmable logic array built from NAND gates.

## Setup

| | Detail |
|---|---|
| Optimiser | Adam, lr = 0.005 |
| Steps | 5000 per trial |
| Trials | 5 (best kept) |
| Annealing | τ: 1.0 → 0.01 (deterministic softmax, no Gumbel noise) |
| Domain | x ∈ [0.1, 3.0], 512 points |

## Results

| Target | Tree depth | Soft MSE | Snapped MSE | Notes |
|--------|----------:|----------:|------------:|-------|
| `exp(x)` | 2 | 0.000000 | 0.000000 | Perfect — `eml(x, 1)` is depth 1 |
| `ln(x)` | 3 | 0.001174 | 0.001174 | Near-perfect, snapping preserves quality |
| `sqrt(x)` | 4 | 0.000036 | 0.438648 | Good soft fit, snapping introduces error |
| `x²` | 5 | 0.163246 | 12.416070 | Hardest — requires `exp(2·ln(x))`, deep composition |

## Discussion

Functions that are shallow in the EML grammar (`exp`, `ln`) are recovered almost exactly. Deeper compositions (`sqrt = exp(0.5·ln(x))`, `x² = exp(2·ln(x))`) require larger trees and are harder to optimise — the soft→hard snapping gap grows with depth.

This matches the paper's prediction: the EML tree depth directly corresponds to the complexity of the target expression in the `S → 1 | eml(S, S)` grammar.

## Reproduce

```bash
pip install -e .
python experiments/symbolic_regression.py
```
