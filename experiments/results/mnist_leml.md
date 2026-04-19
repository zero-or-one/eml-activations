# Experiment: MNIST with LearnableEML activations

A two-hidden-layer MLP on MNIST comparing **LearnableEML** (parametric `exp(ax+b) − ln(cx+d)`) against a standard **ReLU** baseline.

## Setup

| | Detail |
|---|---|
| Architecture | `Linear(784→256) → BN → Act → Linear(256→128) → BN → Act → Linear(128→10)` |
| Optimiser | Adam, lr = 1e-3 |
| Batch size | 256 |
| Epochs | 10 |
| LearnableEML init | `a=0.5, b=0, c=0, d=1` (starts as `exp(0.5x)`) |

## Results

### LearnableEML

| Epoch | Loss | Accuracy |
|------:|-----:|---------:|
| 1 | 0.3074 | 96.52% |
| 2 | 0.1141 | 97.14% |
| 3 | 0.0661 | 97.53% |
| 4 | 0.0477 | 97.72% |
| 5 | 0.0407 | 96.29% |
| 6 | 0.1627 | 96.59% |
| 7 | 0.0773 | 97.44% |
| 8 | 0.0721 | 97.17% |
| 9 | 0.0662 | 95.77% |
| 10 | 0.1027 | 97.01% |

### ReLU baseline

| Epoch | Loss | Accuracy |
|------:|-----:|---------:|
| 1 | 0.2785 | 96.65% |
| 2 | 0.0829 | 97.40% |
| 3 | 0.0514 | 97.69% |
| 4 | 0.0346 | 97.83% |
| 5 | 0.0253 | 97.83% |
| 6 | 0.0203 | 97.62% |
| 7 | 0.0165 | 98.06% |
| 8 | 0.0083 | 98.09% |
| 9 | 0.0083 | 98.05% |
| 10 | 0.0084 | 98.11% |

### Summary

| Activation | Best accuracy | Final accuracy |
|---|---:|---:|
| LearnableEML | 97.72% (epoch 4) | 97.01% |
| ReLU | 98.11% (epoch 10) | 98.11% |

## Learned parameters

After training, each LearnableEML layer converged to a different activation shape:

| Layer | a | b | c | d | Interpretation |
|---|---|---|---|---|---|
| Hidden 1 | 0.60 | −0.18 | 0.16 | 0.81 | Moderate exp with small log correction |
| Hidden 2 | 0.58 | 0.19 | −0.32 | 0.79 | Similar exp strength; negative `c` adds to output via `−ln(small)` |

Both layers evolved away from the initial `exp(0.5x)` toward asymmetric shapes that mix exponential and logarithmic behaviour — something fixed activations like ReLU cannot express.

## Discussion

ReLU converges more smoothly and reaches ~1% higher accuracy on this simple task. This is expected — MNIST is well-solved by ReLU MLPs, and LearnableEML introduces 4 extra parameters per layer plus an `exp` in the forward pass that can cause occasional loss spikes.

The value of LearnableEML is not in beating ReLU on toy benchmarks but in:

1. **Activation discovery** — inspecting the learned `(a, b, c, d)` reveals what nonlinearity the task actually needs.
2. **Expressiveness** — on tasks where the optimal activation is not ReLU-shaped, the network can adapt.
3. **Theoretical interest** — every activation is built from a single binary operator (the EML "NAND gate" for continuous math).

## Reproduce

```bash
pip install -e .
pip install torchvision
python experiments/mnist_leml.py
```
