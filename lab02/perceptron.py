"""Lab 02: two-hidden-layer neural network, implemented with NumPy only.

Run from any directory: python path/to/lab02/perceptron.py
scikit-learn supplies only the dataset and stratified split, not the model.
"""

from pathlib import Path
from time import perf_counter_ns
import csv
import json
import platform

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


SEED = 42
WIDTHS = (16, 64)
LEARNING_RATES = (0.001, 0.01, 0.1)
BATCH_SIZE = 32
EPOCHS = 50
RESULTS_DIR = Path(__file__).resolve().parent / "results"


def prepare_data():
    """Return a reproducible 70/30 stratified split with one-hot labels."""
    X, labels = load_digits(return_X_y=True)
    X = X.astype(np.float64) / 16.0
    y = np.eye(10, dtype=np.float64)[labels]
    return train_test_split(
        X, y, test_size=0.3, random_state=SEED, stratify=labels
    )


def softmax(logits):
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(shifted)
    return exp_logits / exp_logits.sum(axis=1, keepdims=True)


def cross_entropy_from_logits(logits, y_true):
    """Mean cross-entropy using log-softmax, without log(0) or clipping."""
    shifted = logits - logits.max(axis=1, keepdims=True)
    log_probs = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    return float(-np.mean(np.sum(y_true * log_probs, axis=1)))


def centered_cross_entropy(logits, y_true):
    """Compute CE - ln(classes) accurately near the uniform prediction.

    This differs from CE by a constant, so it has exactly the same gradients.
    Used ONLY for gradient checking at initialization: expm1/log1p avoid
    subtracting two losses near 2.3026 to measure gradients as small as 1e-6.
    Training/evaluation use the general log-softmax loss above.
    """
    centered = logits - logits.mean(axis=1, keepdims=True)
    log_mean_exp = np.log1p(np.expm1(centered).mean(axis=1))
    return float(np.mean(log_mean_exp - np.sum(y_true * centered, axis=1)))


class NeuralNetwork:
    def __init__(self, width, seed=SEED):
        rng = np.random.default_rng(seed)
        self.width = width
        self.params = {
            "W1": rng.normal(0, 0.01, size=(64, width)),
            "b1": np.zeros(width),
            "W2": rng.normal(0, 0.01, size=(width, width)),
            "b2": np.zeros(width),
            "W3": rng.normal(0, 0.01, size=(width, 10)),
            "b3": np.zeros(10),
        }

    def forward(self, X):
        p = self.params
        z1 = X @ p["W1"] + p["b1"]
        a1 = np.maximum(0, z1)
        z2 = a1 @ p["W2"] + p["b2"]
        a2 = np.maximum(0, z2)
        z3 = a2 @ p["W3"] + p["b3"]
        probs = softmax(z3)
        cache = {"X": X, "z1": z1, "a1": a1, "z2": z2,
                 "a2": a2, "z3": z3, "probs": probs}
        return probs, cache

    def backward(self, cache, y_true):
        """Chain rule for mean CE. No parameter updates inside this method."""
        p = self.params
        dz3 = (cache["probs"] - y_true) / len(y_true)
        dW3 = cache["a2"].T @ dz3
        db3 = dz3.sum(axis=0)

        da2 = dz3 @ p["W3"].T
        dz2 = da2 * (cache["z2"] > 0)
        dW2 = cache["a1"].T @ dz2
        db2 = dz2.sum(axis=0)

        da1 = dz2 @ p["W2"].T
        dz1 = da1 * (cache["z1"] > 0)
        dW1 = cache["X"].T @ dz1
        db1 = dz1.sum(axis=0)
        return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2,
                "W3": dW3, "b3": db3}

    def step(self, gradients, learning_rate):
        # All gradients must be computed before any weight is changed.
        for name in self.params:
            self.params[name] -= learning_rate * gradients[name]

    def predict(self, X):
        probabilities, _ = self.forward(X)
        return probabilities.argmax(axis=1)


def sanity_checks(model, X, y):
    probabilities, cache = model.forward(X)
    assert probabilities.shape == (len(X), 10)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1, rtol=0, atol=1e-6)
    loss = cross_entropy_from_logits(cache["z3"], y)
    assert abs(loss - np.log(10)) < 0.01, loss
    gradients = model.backward(cache, y)
    for name, parameter in model.params.items():
        assert gradients[name].shape == parameter.shape
        assert np.isfinite(gradients[name]).all()
    # Verify that the numerically centered objective is the same loss + constant.
    np.testing.assert_allclose(
        centered_cross_entropy(cache["z3"], y) + np.log(10), loss,
        rtol=0, atol=1e-14,
    )
    print(f"W={model.width}: shape={probabilities.shape}, row sums OK, "
          f"initial CE={loss:.8f} (ln(10)={np.log(10):.8f})")
    return loss


def gradient_check(model, X, y, epsilon=1e-5, seed=SEED):
    """Check 10 distinct random weights at +/-epsilon, covering every layer."""
    _, cache = model.forward(X)
    analytical = model.backward(cache, y)
    rng = np.random.default_rng(seed)
    rows = []
    # Stratified random selection prevents checking only one layer by chance.
    for name, count in (("W1", 4), ("W2", 3), ("W3", 3)):
        weight = model.params[name]
        for flat_index in rng.choice(weight.size, size=count, replace=False):
            index = np.unravel_index(flat_index, weight.shape)
            original = weight[index].item()
            try:
                weight[index] = original + epsilon
                _, plus_cache = model.forward(X)
                plus = centered_cross_entropy(plus_cache["z3"], y)
                weight[index] = original - epsilon
                _, minus_cache = model.forward(X)
                minus = centered_cross_entropy(minus_cache["z3"], y)
            finally:
                weight[index] = original
            numerical = (plus - minus) / (2 * epsilon)
            analytic = float(analytical[name][index])
            # No loose absolute tolerance: zero/zero alone is defined as zero.
            denominator = abs(analytic) + abs(numerical)
            error = abs(analytic - numerical) / denominator if denominator else 0.0
            row = {"width": model.width, "parameter": name,
                   "row": int(index[0]), "column": int(index[1]),
                   "analytical": analytic, "numerical": numerical,
                   "relative_error": error, "passed": error < 1e-6}
            rows.append(row)
            print(f"  {name}{index}: analytical={analytic:.8e}, "
                  f"numerical={numerical:.8e}, rel.error={error:.2e}, "
                  f"{'PASS' if row['passed'] else 'FAIL'}")
    passed = sum(row["passed"] for row in rows)
    print(f"W={model.width}: gradient check {passed}/10")
    assert passed >= 9, f"W={model.width}: only {passed}/10 gradients passed"
    return rows


def evaluate(model, X, y):
    probabilities, cache = model.forward(X)
    loss = cross_entropy_from_logits(cache["z3"], y)
    accuracy = float(np.mean(probabilities.argmax(axis=1) == y.argmax(axis=1)))
    return loss, accuracy


def train(model, X_train, y_train, learning_rate, epochs=EPOCHS, batch_size=BATCH_SIZE):
    # Reset the shuffle seed for each experiment for a fair LR comparison.
    rng = np.random.default_rng(SEED)
    history = []
    for epoch in range(1, epochs + 1):
        indices = rng.permutation(len(X_train))
        for start in range(0, len(indices), batch_size):
            batch = indices[start:start + batch_size]
            X_batch, y_batch = X_train[batch], y_train[batch]
            _, cache = model.forward(X_batch)
            gradients = model.backward(cache, y_batch)
            model.step(gradients, learning_rate)
        loss, accuracy = evaluate(model, X_train, y_train)
        assert np.isfinite(loss), "Non-finite training loss"
        row = {"width": model.width, "learning_rate": learning_rate,
               "epoch": epoch, "train_loss": loss, "train_accuracy": accuracy}
        history.append(row)
        print(f"W={model.width:2d} lr={learning_rate:g} "
              f"epoch={epoch:02d}/{epochs} loss={loss:.8f} "
              f"train_accuracy={accuracy:.4f}")
    return history


def time_passes(model, X, y, runs=100):
    """Median batch times after 10 warmups; backward excludes forward/SGD."""
    _, cache = model.forward(X)
    for _ in range(10):
        model.forward(X)
        model.backward(cache, y)
    forward_ns, backward_ns = [], []
    for _ in range(runs):
        start = perf_counter_ns()
        model.forward(X)
        forward_ns.append(perf_counter_ns() - start)
        start = perf_counter_ns()
        model.backward(cache, y)
        backward_ns.append(perf_counter_ns() - start)
    return {"forward_median_us": float(np.median(forward_ns) / 1000),
            "backward_median_us": float(np.median(backward_ns) / 1000),
            "runs": runs, "batch_size": len(X)}, forward_ns, backward_ns


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_losses(histories, path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for axis, width in zip(axes, WIDTHS):
        for learning_rate in LEARNING_RATES:
            rows = histories[(width, learning_rate)]
            axis.plot([0] + [r["epoch"] for r in rows],
                      [rows[0]["initial_loss"]] + [r["train_loss"] for r in rows],
                      label=f"learning rate = {learning_rate:g}")
        axis.axhline(np.log(10), color="gray", linestyle=":", linewidth=1,
                     label="ln(10), uniform prediction")
        axis.set(title=f"Hidden width W = {width}", xlabel="Epoch")
        axis.set_xlim(0, EPOCHS)
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    axes[0].set_ylabel("Mean training cross-entropy")
    fig.suptitle("Digits: 64 → W → ReLU → W → ReLU → 10 → softmax")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_report(summary, checks, initial_losses, environment):
    lines = [
        "# Lab 02 — A neural network from scratch", "",
        "## 1. Goal and method", "",
        "The network uses NumPy for forward propagation, cross-entropy, "
        "backpropagation and mini-batch SGD. scikit-learn is used only for "
        "`load_digits` and the stratified split; no neural-network framework or "
        "automatic differentiation is used.", "",
        "The 1,797 images have 64 pixel values each. Inputs are divided by 16 "
        "and labels are one-hot encoded. The stratified 70/30 split with "
        "`random_state=42` gives 1,257 training images and 540 test images. "
        "Initialization and shuffling also use seed 42. Each experiment starts "
        "with fresh weights from N(0, 0.01²), zero biases, and the same batch "
        "order. Training uses batches of 32 (the final batch has 9 images), "
        "50 epochs and plain SGD. Test accuracy is measured after training; "
        "the test set does not control updates or stopping.", "",
        "The architecture and chain-rule derivation are in [derivation.md](derivation.md).",
        "", "## 2. Sanity checks and gradient checking", "",
        "| Width | Initial batch loss | Passed random weights | Max relative error |",
        "|---:|---:|---:|---:|",
    ]
    for width in WIDTHS:
        rows = [r for r in checks if r["width"] == width]
        lines.append(f"| {width} | {initial_losses[width]:.8f} | "
                     f"{sum(r['passed'] for r in rows)}/10 | "
                     f"{max(r['relative_error'] for r in rows):.3e} |")
    lines += ["", "Both initial losses are close to ln(10) = 2.302585. "
              "Assertions check the output shape (32, 10), row sums with "
              "absolute tolerance 1e-6 and zero relative tolerance, and all "
              "gradient shapes.", "",
              "For each width, 10 distinct random weights across all three "
              "layers are perturbed by ±1e-5. Relative error is "
              "`abs(analytic - numeric) / (abs(analytic) + abs(numeric))`; "
              "zero/zero is defined as zero. At least 9/10 must be below 1e-6 "
              "or execution fails. No failing weights are discarded.", "",
              "With the required small initialization, some slopes are tiny. "
              "Subtracting two ordinary float64 losses near 2.3026 loses "
              "precision. The numerical check therefore evaluates CE − ln(10) "
              "using `expm1` and `log1p` before central differencing. Subtracting "
              "this constant leaves the mathematical gradient unchanged. "
              "Training uses stable log-softmax cross-entropy. Full analytical "
              "and numerical values are in `results/gradient_checks.csv`.", "",
              "## 3. Training results", "",
              "| Width | Learning rate | Final train loss | Train accuracy | Test accuracy |",
              "|---:|---:|---:|---:|---:|"]
    for row in summary:
        lines.append(f"| {row['width']} | {row['learning_rate']:g} | "
                     f"{row['train_loss']:.6f} | {row['train_accuracy']:.2%} | "
                     f"{row['test_accuracy']:.2%} |")
    best = max(summary, key=lambda r: r["test_accuracy"])
    lines += ["", "![Training loss curves](results/loss_curves.png)", "",
              f"The highest observed test accuracy is {best['test_accuracy']:.2%} "
              f"at W={best['width']} and learning rate {best['learning_rate']:g}. "
              "This is a descriptive comparison of the six assigned experiments, "
              "not an independent validation-based model selection.", "",
              "Small learning rates can remain near the uniform-prediction loss "
              "within 50 epochs because small weights attenuate the signals "
              "through both hidden layers. A wider network has more parameters "
              "and more matrix multiplication work. Width alone does not "
              "determine accuracy; the learning rate and fixed training budget "
              "also matter.", "", "## 4. Forward and backward timing", "",
              "Each trained model is timed on the same batch of 32 images. "
              "After 10 warm-up iterations, the median of 100 runs is recorded "
              "with `perf_counter_ns`. Backward uses a precomputed cache, so its "
              "measurement excludes forward propagation, loss evaluation and "
              "SGD updates. Forward includes softmax and cache construction. "
              "Raw timings are in `results/timing_runs.csv`.", "",
              "| Width | Learning rate | Forward (µs) | Backward (µs) | Backward/forward |",
              "|---:|---:|---:|---:|---:|"]
    for row in summary:
        lines.append(f"| {row['width']} | {row['learning_rate']:g} | "
                     f"{row['forward_median_us']:.3f} | "
                     f"{row['backward_median_us']:.3f} | "
                     f"{row['backward_median_us']/row['forward_median_us']:.2f} |")
    backward_wins = sum(r["backward_median_us"] > r["forward_median_us"] for r in summary)
    lines += ["", f"Backward is slower in {backward_wins} of the 6 measured configurations. "
              "Forward uses three matrix products: XW1, A1W2 and A2W3. "
              "Backward uses five: three weight-gradient products and two "
              "products propagating the gradient into hidden layers. "
              "The input gradient is not needed. Ignoring elementwise operations "
              "and counting multiply/add as two FLOPs, forward costs "
              "2B(64W + W² + 10W), while backward costs "
              "2B(64W + 2W² + 20W). Their arithmetic ratio is about 1.29 for "
              "W=16 and 1.54 for W=64. Small matrices also incur Python, "
              "allocation, softmax and reduction overhead, so wall-clock ratios "
              "need not equal these estimates and can favor either pass. "
              "These timings describe this machine and run, not a universal "
              "performance guarantee.", "", "## 5. Reproducibility", "",
              "Run from the repository root:", "", "```bash",
              ".venv/bin/python lab02/perceptron.py", "",
              "# Additional checks, including bias gradients and extreme logits:",
              ".venv/bin/python -m unittest discover -s lab02 -p 'test_*.py' -v", "```", "",
              "The script writes all results relative to its own location, "
              "so it also works when launched from PyCharm with a different "
              "working directory. `results/training_log.csv` contains all "
              "300 epoch records; `results/summary.csv` contains final test "
              "accuracy and timing for all six runs. `results/environment.json` "
              "records the runtime versions.", "",
              f"Measured environment: Python {environment['python']}, "
              f"NumPy {environment['numpy']}, scikit-learn {environment['sklearn']}, "
              f"Matplotlib {environment['matplotlib']}; {environment['platform']}.", ""]
    (RESULTS_DIR.parent / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    X_train, X_test, y_train, y_test = prepare_data()
    print(f"Data: train={X_train.shape}, test={X_test.shape}, one-hot={y_train.shape}")
    checks, initial_losses = [], {}
    for width in WIDTHS:
        model = NeuralNetwork(width)
        initial_losses[width] = sanity_checks(model, X_train[:32], y_train[:32])
        checks.extend(gradient_check(model, X_train[:32], y_train[:32]))
    write_csv(RESULTS_DIR / "gradient_checks.csv", checks)

    histories, summary, all_epochs, timing_runs = {}, [], [], []
    for width in WIDTHS:
        for learning_rate in LEARNING_RATES:
            model = NeuralNetwork(width)
            initial_loss, _ = evaluate(model, X_train, y_train)
            history = train(model, X_train, y_train, learning_rate)
            for row in history:
                row["initial_loss"] = initial_loss
            histories[(width, learning_rate)] = history
            all_epochs.extend(history)
            test_loss, test_accuracy = evaluate(model, X_test, y_test)
            timings, forward_ns, backward_ns = time_passes(
                model, X_train[:32], y_train[:32]
            )
            for run, (fwd, bwd) in enumerate(zip(forward_ns, backward_ns), start=1):
                timing_runs.append({"width": width, "learning_rate": learning_rate,
                                    "run": run, "forward_ns": fwd, "backward_ns": bwd})
            result = {"width": width, "learning_rate": learning_rate,
                      "train_loss": history[-1]["train_loss"],
                      "train_accuracy": history[-1]["train_accuracy"],
                      "test_loss": test_loss, "test_accuracy": test_accuracy, **timings}
            summary.append(result)
            print(f"FINAL W={width} lr={learning_rate:g}: test_accuracy={test_accuracy:.4%}, "
                  f"forward={timings['forward_median_us']:.3f} µs, "
                  f"backward={timings['backward_median_us']:.3f} µs")

    write_csv(RESULTS_DIR / "training_log.csv", all_epochs)
    write_csv(RESULTS_DIR / "summary.csv", summary)
    write_csv(RESULTS_DIR / "timing_runs.csv", timing_runs)
    plot_losses(histories, RESULTS_DIR / "loss_curves.png")
    environment = {"python": platform.python_version(), "numpy": np.__version__,
                   "sklearn": sklearn.__version__, "matplotlib": matplotlib.__version__,
                   "platform": platform.platform(), "seed": SEED,
                   "epochs": EPOCHS, "batch_size": BATCH_SIZE}
    (RESULTS_DIR / "environment.json").write_text(
        json.dumps(environment, indent=2) + "\n", encoding="utf-8"
    )
    write_report(summary, checks, initial_losses, environment)
    print(f"Results saved to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
