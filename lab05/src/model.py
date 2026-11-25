"""Equivalent scalar and vectorized 2 -> 8 -> 1 models."""

import numpy as np
import torch
from torch import nn

from autodiff import Value


def initial_arrays(seed=42):
    rng = np.random.default_rng(seed)
    return {
        "w1": rng.normal(0, np.sqrt(2 / 2), (8, 2)),
        "b1": np.zeros(8),
        "w2": rng.normal(0, np.sqrt(1 / 8), (1, 8)),
        "b2": np.zeros(1),
    }


class ScalarMLP:
    def __init__(self, arrays):
        self.w1 = [[Value(x) for x in row] for row in arrays["w1"]]
        self.b1 = [Value(x) for x in arrays["b1"]]
        self.w2 = [Value(x) for x in arrays["w2"][0]]
        self.b2 = Value(arrays["b2"][0])

    def parameters(self):
        return [p for row in self.w1 for p in row] + self.b1 + self.w2 + [self.b2]

    def logit(self, row):
        hidden = [(w[0] * row[0] + w[1] * row[1] + b).relu()
                  for w, b in zip(self.w1, self.b1)]
        return sum((w * h for w, h in zip(self.w2, hidden)), self.b2)

    def loss(self, x, y):
        # BCE(sigmoid(z), y) == softplus(z) - y*z, without log(0).
        losses = []
        for row, target in zip(x, y):
            z = self.logit(row)
            losses.append(z.softplus() - float(target) * z)
        return sum(losses) / len(losses)

    def probabilities(self, x):
        return np.array([self.logit(row).sigmoid().data for row in x])

    def arrays(self, gradient=False):
        attr = "grad" if gradient else "data"
        return {
            "w1": np.array([[getattr(p, attr) for p in row] for row in self.w1]),
            "b1": np.array([getattr(p, attr) for p in self.b1]),
            "w2": np.array([[getattr(p, attr) for p in self.w2]]),
            "b2": np.array([getattr(self.b2, attr)]),
        }


class TorchMLP(nn.Module):
    def __init__(self, arrays):
        super().__init__()
        self.hidden = nn.Linear(2, 8, dtype=torch.float64)
        self.output = nn.Linear(8, 1, dtype=torch.float64)
        with torch.no_grad():
            for key, parameter in self.parameter_map().items():
                parameter.copy_(torch.from_numpy(arrays[key]))

    def forward(self, x):
        return self.output(torch.relu(self.hidden(x))).squeeze(-1)

    def probabilities(self, x):
        return torch.sigmoid(self(x))

    def parameter_map(self):
        return {"w1": self.hidden.weight, "b1": self.hidden.bias,
                "w2": self.output.weight, "b2": self.output.bias}
