"""Lab 3: run the six assigned steps with python lab03/main.py."""
import copy
import csv
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import random
import time

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/lab03-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision import datasets, transforms
from torchinfo import summary
from thop import profile

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
SEED = 42


class MLP(nn.Sequential):
    def __init__(self):
        super().__init__(nn.Flatten(), nn.Linear(784, 256), nn.ReLU(),
                         nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 10))


class CNN(nn.Sequential):
    def __init__(self):
        super().__init__(nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                         nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                         nn.Flatten(), nn.Linear(32 * 7 * 7, 10))


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(49, 64)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(64, 4, 128, batch_first=True)
            for _ in range(2)
        ])
        self.classifier = nn.Linear(64, 10)

    def forward(self, x):
        # Raster order: four rows of four non-overlapping 7 x 7 patches.
        x = x.unfold(2, 7, 7).unfold(3, 7, 7)
        x = x.permute(0, 2, 3, 1, 4, 5).reshape(x.shape[0], 16, 49)
        x = self.projection(x)
        for layer in self.layers:
            x = layer(x)
        return self.classifier(x.mean(dim=1))


def seed():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def materialize(dataset):
    # Apply the specified torchvision transforms once, then reuse their output.
    batches = list(DataLoader(dataset, batch_size=1024, num_workers=0))
    return TensorDataset(torch.cat([x for x, _ in batches]),
                         torch.cat([y for _, y in batches]))


def count_attention(module, inputs, output):
    batch, seq, d = inputs[0].shape
    # MACs: QKV + QK^T + AV + output projection; no biases/softmax.
    module.total_ops += batch * (4 * seq * d * d + 2 * seq * seq * d)


def costs(name, model):
    model = copy.deepcopy(model).cpu().eval()
    x = torch.zeros(1, 1, 28, 28)
    info = summary(model, input_data=x, verbose=0, depth=5)
    (RESULTS / f"{name.lower()}_summary.txt").write_text(str(info) + "\n")
    params = sum(p.numel() for p in model.parameters())
    assert info.total_params == params
    default_macs, _ = profile(copy.deepcopy(model), inputs=(x,), verbose=False)
    # Restrict profiling to the same matrix/conv arithmetic as the hand count.
    macs, _ = profile(copy.deepcopy(model), inputs=(x,), verbose=False,
                      custom_ops={nn.MultiheadAttention: count_attention,
                                  nn.LayerNorm: lambda m, i, o: None})
    hand = {
        "MLP": 2 * (784 * 256 + 256 * 128 + 128 * 10),
        "CNN": 2 * (9 * 1 * 16 * 28 * 28 + 9 * 16 * 32 * 14 * 14 + 1568 * 10),
        "Attention": 2 * (16 * 49 * 64 + 2 * (
            4 * 16 * 64 * 64 + 2 * 16 * 16 * 64 + 2 * 16 * 64 * 128) + 64 * 10),
    }[name]
    assert int(2 * macs) == hand, (name, macs, hand)
    return {"parameters": params, "hand_flops": hand,
            "thop_default_ops": int(default_macs), "thop_core_macs": int(macs),
            "thop_core_flops": int(2 * macs)}


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    for x, y in loader:
        correct += (model(x.to(device)).argmax(1) == y.to(device)).sum().item()
    return correct / len(loader.dataset)


@torch.inference_mode()
def latency(model, inputs, device, name):
    model.eval()
    rows = []
    medians = {}
    for batch, runs in [(1, 100), (128, 20)]:
        x = inputs[:batch].to(device)
        for _ in range(10):
            model(x)
        sync(device)
        samples = []
        for run in range(1, runs + 1):
            sync(device)
            start = time.perf_counter_ns()
            model(x)
            sync(device)
            ms = (time.perf_counter_ns() - start) / 1e6
            samples.append(ms)
            rows.append({"model": name, "batch_size": batch, "run": run, "latency_ms": ms})
        medians[f"latency_b{batch}_ms"] = float(np.median(samples))
    return medians, rows


def plots(rows):
    fig, ax = plt.subplots(figsize=(6.5, 4.2), layout="constrained")
    for row in rows:
        ax.scatter(row["parameters"], row["test_accuracy"] * 100, s=65)
        ax.annotate(row["model"], (row["parameters"], row["test_accuracy"] * 100),
                    xytext=(6, 6), textcoords="offset points")
    ax.set(xlabel="Parameters", ylabel="Test accuracy (%)", title="Accuracy against parameters")
    ax.margins(x=.18, y=.25)
    ax.grid(alpha=.25)
    fig.savefig(RESULTS / "accuracy_parameters.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), layout="constrained")
    for ax, batch in zip(axes, [1, 128]):
        for row in rows:
            ax.scatter(row[f"latency_b{batch}_ms"], row["test_accuracy"] * 100, s=65)
            ax.annotate(row["model"], (row[f"latency_b{batch}_ms"], row["test_accuracy"] * 100),
                        xytext=(5, 6), textcoords="offset points")
        ax.set(xlabel="Median batch latency (ms)", ylabel="Test accuracy (%)", title=f"Batch {batch}")
        ax.margins(x=.3, y=.25)
        ax.grid(alpha=.25)
    fig.savefig(RESULTS / "accuracy_latency.png", dpi=180)
    plt.close(fig)


def main():
    RESULTS.mkdir(exist_ok=True)
    seed()
    torch.set_num_threads(4)
    # Keep functional MHA visible to hooks; use the same setting for all timings.
    torch.backends.mha.set_fastpath_enabled(False)
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")
    print("PyTorch:", torch.__version__, "Device:", device, flush=True)
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.2860,), (0.3530,))])
    full = materialize(datasets.FashionMNIST(ROOT.parent / "data", train=True,
                                           download=True, transform=transform))
    test = materialize(datasets.FashionMNIST(ROOT.parent / "data", train=False,
                                           download=True, transform=transform))
    train, val = random_split(full, [55000, 5000], generator=torch.Generator().manual_seed(SEED))
    val_loader = DataLoader(val, batch_size=128)
    test_loader = DataLoader(test, batch_size=128)
    environment = {"python": platform.python_version(), "platform": platform.platform(),
                   "processor": platform.processor(), "torch": torch.__version__,
                   "device": str(device), "threads": torch.get_num_threads(), "seed": SEED,
                   "train": len(train), "validation": len(val), "test": len(test),
                   "mha_fastpath": False, "epochs": 5, "batch_size": 128, "lr": .001}
    environment["packages"] = {name: version(name) for name in
                               ["numpy", "torchvision", "torchinfo", "thop", "matplotlib"]}
    (RESULTS / "environment.json").write_text(json.dumps(environment, indent=2) + "\n")
    rows, history, timings = [], [], []
    for name, cls in [("MLP", MLP), ("CNN", CNN), ("Attention", Attention)]:
        seed()
        model = cls()
        row = {"model": name, **costs(name, model)}
        print(name, row, flush=True)
        model.to(device)
        loader = DataLoader(train, batch_size=128, shuffle=True,
                            generator=torch.Generator().manual_seed(SEED))
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        for epoch in range(1, 6):
            model.train()
            total_loss, correct = 0., 0
            start = time.perf_counter()
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(x)
                loss = nn.functional.cross_entropy(logits, y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(y)
                correct += (logits.argmax(1) == y).sum().item()
            entry = {"model": name, "epoch": epoch, "train_loss": total_loss / len(train),
                     "train_accuracy": correct / len(train),
                     "validation_accuracy": evaluate(model, val_loader, device),
                     "epoch_seconds": time.perf_counter() - start}
            history.append(entry)
            write_csv(RESULTS / "training_log.csv", history)
            print(entry, flush=True)
        row["test_accuracy"] = evaluate(model, test_loader, device)
        medians, raw = latency(model, test.tensors[0], device, name)
        row.update(medians)
        rows.append(row)
        timings.extend(raw)
        write_csv(RESULTS / "summary.csv", rows)
        write_csv(RESULTS / "timing_runs.csv", timings)
        print("Result:", row, flush=True)
    plots(rows)


if __name__ == "__main__":
    main()
