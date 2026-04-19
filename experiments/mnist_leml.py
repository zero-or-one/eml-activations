"""MNIST classifier with LearnableEML activations.

Each hidden layer learns its own (a, b, c, d) activation shape.
Run:  python examples/mnist_leml.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from eml_activations import LearnableEML


class EMLNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(784, 256),
            nn.BatchNorm1d(256),
            LearnableEML(a=0.5, b=0.0, c=0.0, d=1.0),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            LearnableEML(a=0.5, b=0.0, c=0.0, d=1.0),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.layers(x.flatten(1))


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tf = transforms.ToTensor()
    train_ds = datasets.MNIST("data", train=True, download=True, transform=tf)
    test_ds = datasets.MNIST("data", train=False, transform=tf)
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=256, shuffle=True)
    test_dl = torch.utils.data.DataLoader(test_ds, batch_size=1000)

    model = EMLNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(1, 11):
        model.train()
        total_loss = 0.0
        for imgs, labels in train_dl:
            imgs, labels = imgs.to(device), labels.to(device)
            loss = F.cross_entropy(model(imgs), labels)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * imgs.size(0)

        # evaluate
        model.eval()
        correct = 0
        with torch.no_grad():
            for imgs, labels in test_dl:
                imgs, labels = imgs.to(device), labels.to(device)
                correct += (model(imgs).argmax(1) == labels).sum().item()

        acc = correct / len(test_ds) * 100
        avg_loss = total_loss / len(train_ds)
        print(f"Epoch {epoch}  loss={avg_loss:.4f}  acc={acc:.2f}%")

    # show what each activation learned
    print("\nLearned activation parameters:")
    for i, m in enumerate(model.layers):
        if isinstance(m, LearnableEML):
            print(f"  Layer {i}: a={m.a.item():.4f}  b={m.b.item():.4f}"
                  f"  c={m.c.item():.4f}  d={m.d.item():.4f}")


if __name__ == "__main__":
    main()
