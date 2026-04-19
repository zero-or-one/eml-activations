# eml-activations

PyTorch implementation of the **EML (Exp-Minus-Log) operator** from:

> *"All elementary functions from a single binary operator"*
> Andrzej Odrzywołek — [arXiv:2603.21852](https://arxiv.org/abs/2603.21852)

A single binary operator `eml(x, y) = exp(x) − ln(y)`, paired with the constant `1`, generates every elementary function on a scientific calculator — addition, multiplication, logarithms, trig, and more. This is the continuous-math analogue of the NAND gate.

## Installation

```bash
pip install -e .
```

Requires Python ≥ 3.9 and PyTorch ≥ 2.0.

## Quick start

```python
import torch
from eml_activations import eml, eml_exp, eml_ln, eml_sigmoid

x = torch.tensor([0.5, 1.0, 2.0])

eml(x, torch.ones_like(x))   # exp(x)
eml_ln(x)                     # ln(x) built purely from eml
eml_sigmoid(x)                # σ(x) = 1/(1+exp(-x))
```

## What's included

### Functional API (`eml_activations.core`)

| Function | Definition |
|---|---|
| `eml(x, y)` | `exp(x) − ln(y)` — the primitive |
| `eml_exp(x)` | `eml(x, 1)` |
| `eml_ln(x)` | `eml(1, eml(eml(1, x), 1))` |
| `eml_neg(x)` | `−x` |
| `eml_inv(x)` | `exp(−ln(x))` |
| `eml_add(x, y)` | `x − (−y)` |
| `eml_sub(x, y)` | `eml(ln(x), exp(y))` |
| `eml_mul(x, y)` | `exp(ln(x) + ln(y))` |
| `eml_div(x, y)` | `x · (1/y)` |
| `eml_pow(x, y)` | `exp(y · ln(x))` |
| `eml_sqrt(x)` | `x^0.5` |
| `eml_sigmoid(x)` | `1 / (1 + exp(−x))` |

### nn.Module wrappers

`EML`, `EMLExp`, `EMLSigmoid` — drop-in activation layers.

### Symbolic regression tree (`eml_activations.tree`)

The paper's "master formula" (Section 4.3): a parameterised binary tree of EML nodes where each input is soft-selected from `{1, x, child}` via Gumbel-Softmax. Train with Adam, then call `snap_weights()` to recover exact closed-form expressions.

```python
from eml_activations import EMLTree
import torch, torch.nn.functional as F

tree = EMLTree(depth=3)
x = torch.linspace(0.1, 3.0, 256)
target = torch.log(x)

opt = torch.optim.Adam(tree.parameters(), lr=0.01)
for step in range(2000):
    pred = tree(x, tau=max(0.1, 1.0 - step / 1500))
    loss = F.mse_loss(pred, target)
    opt.zero_grad(); loss.backward(); opt.step()

tree.snap_weights()  # snap logits → one-hot → exact formula
```

## How it works

The grammar is `S → 1 | eml(S, S)`. Every elementary expression becomes a binary tree of identical nodes — like a digital circuit built from NAND gates, but for continuous math.

```
       eml
      /   \
    eml     1       ← exp(x) = eml(x, 1)
   /   \
  1     x           ← ln(x) = eml(1, eml(eml(1,x), 1))
```

## Experiments

- [MNIST with LearnableEML](experiments/results/mnist_leml.md) — a two-layer MLP where each hidden layer learns its own activation shape from the `exp`/`log` family. Includes comparison against ReLU.
- [Symbolic Regression](experiments/results/symbolic_regression.md) — discovering closed-form expressions (`exp`, `ln`, `sqrt`, `x²`) from data using EMLTree. Demonstrates the "NAND gate for continuous math" idea.

## References

- Odrzywołek, A. (2026). *All elementary functions from a single binary operator.* [arXiv:2603.21852](https://arxiv.org/abs/2603.21852)
- [Wikipedia: EML (mathematical operator)](https://en.wikipedia.org/wiki/EML_(mathematical_operator))

## License

MIT
