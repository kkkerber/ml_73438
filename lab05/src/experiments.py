"""Measured evidence for Lab 5; run from the repository root."""

import argparse
import csv
import gc
import json
import os
import platform
from pathlib import Path
import statistics
import subprocess
import sys
import time

import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import torch
from torch.nn import functional as F

from autodiff import Value
from model import ScalarMLP, TorchMLP, initial_arrays


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
REPEATS = 5
EPOCHS = 300
LR = 0.2


def save_json(name, obj):
    (RESULTS / name).write_text(json.dumps(obj, indent=2) + "\n")


def save_csv(name, rows):
    with (RESULTS / name).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def validate_gradients():
    rows = []
    rng = np.random.default_rng(42)
    # Four-element sums test scalar-engine reduction against a tensor reduction.
    for case in range(25):
        inputs = rng.uniform(-2, 2, (3, 4))
        nodes = [[Value(x) for x in group] for group in inputs]
        outputs = [((a*b+c)**2).relu() for a, b, c in zip(*nodes)]
        loss = sum(outputs)
        loss.backward()
        tensors = [torch.tensor(v, dtype=torch.float64, requires_grad=True) for v in inputs]
        ref = ((tensors[0]*tensors[1]+tensors[2])**2).relu().sum()
        ref.backward()

        def numeric_function(values):
            return np.maximum((values[0]*values[1]+values[2])**2, 0).sum()

        for group, (name, tensor) in enumerate(zip("abc", tensors)):
            for element in range(4):
                plus, minus = inputs.copy(), inputs.copy()
                plus[group, element] += 1e-5
                minus[group, element] -= 1e-5
                numeric = (numeric_function(plus)-numeric_function(minus))/2e-5
                actual = nodes[group][element].grad
                expected = tensor.grad[element].item()
                rows.append(dict(case=case, variable=name, element=element,
                                 custom=actual, pytorch=expected, numeric=numeric,
                                 error_pytorch=abs(actual-expected),
                                 error_numeric=abs(actual-numeric)))
    summary = dict(cases=25, derivatives=len(rows), epsilon=1e-5, dtype="float64",
                   max_error_pytorch=max(r["error_pytorch"] for r in rows),
                   max_error_numeric=max(r["error_numeric"] for r in rows))
    assert summary["max_error_pytorch"] < 1e-6
    assert summary["max_error_numeric"] < 1e-6
    save_csv("gradients.csv", rows)
    save_json("gradient_summary.json", summary)
    # A small, reproducible graph that can be rendered without Graphviz.
    a, b, c = Value(2, label="a"), Value(3, label="b"), Value(1, label="c")
    loss = ((a*b+c)**2).relu().sum()
    loss.backward()
    ordered = loss.topological()
    ids = {node: i for i, node in enumerate(ordered)}
    lines = ["digraph autodiff {", "rankdir=LR;"]
    for node in ordered:
        label = f"{node.label or node.op} | data={node.data:g} | grad={node.grad:g}"
        lines.append(f'v{ids[node]} [label="{label}", shape=box];')
        for parent in node.parents:
            lines.append(f"v{ids[parent]} -> v{ids[node]};")
    lines.append("}")
    (RESULTS / "scalar_graph.dot").write_text("\n".join(lines) + "\n")
    return summary


def train_scalar(arrays, x, y):
    model = ScalarMLP(arrays)
    params, history = model.parameters(), []
    start = time.perf_counter()
    for epoch in range(EPOCHS):
        for p in params:
            p.grad = 0
        loss = model.loss(x, y)
        loss.backward()
        for p in params:
            p.data -= LR * p.grad
        history.append(dict(epoch=epoch+1, loss=loss.data))
        del loss
    elapsed = time.perf_counter() - start
    return model, history, elapsed


def train_torch(arrays, x, y):
    model = TorchMLP(arrays)
    x, y = torch.from_numpy(x), torch.from_numpy(y)
    history = []
    start = time.perf_counter()
    for epoch in range(EPOCHS):
        model.zero_grad(set_to_none=True)
        loss = F.binary_cross_entropy_with_logits(model(x), y)
        loss.backward()
        with torch.no_grad():
            for p in model.parameters():
                p -= LR * p.grad
        history.append(dict(epoch=epoch+1, loss=loss.item()))
    elapsed = time.perf_counter() - start
    return model, history, elapsed


def training_study():
    x, y = make_moons(n_samples=200, noise=0.15, random_state=42)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y.astype(np.float64), test_size=0.25, random_state=42, stratify=y)
    np.savez(RESULTS / "moons_split.npz", x_train=x_train, x_test=x_test,
             y_train=y_train, y_test=y_test)
    arrays = initial_arrays()
    # A small untimed warm-up for both paths; initialization is never timed.
    ScalarMLP(arrays).loss(x_train[:4], y_train[:4]).backward()
    warm = TorchMLP(arrays)
    F.binary_cross_entropy_with_logits(warm(torch.from_numpy(x_train[:4])),
                                       torch.from_numpy(y_train[:4])).backward()
    runs, histories, final_models = [], {}, {}
    for repeat in range(REPEATS):
        # Alternate order to reduce fixed ordering bias.
        names = ["scalar", "pytorch"] if repeat % 2 == 0 else ["pytorch", "scalar"]
        for name in names:
            gc.collect()
            runner = train_scalar if name == "scalar" else train_torch
            model, history, elapsed = runner(arrays, x_train, y_train)
            if name == "scalar":
                probabilities = model.probabilities(x_test)
            else:
                with torch.no_grad():
                    probabilities = model.probabilities(torch.from_numpy(x_test)).numpy()
            accuracy = float(np.mean((probabilities >= 0.5) == y_test))
            runs.append(dict(repeat=repeat+1, implementation=name, seconds=elapsed,
                             final_train_loss=history[-1]["loss"], test_accuracy=accuracy))
            histories[name], final_models[name] = history, model
            print(f"training {repeat+1}/5 {name}: {elapsed:.3f}s, accuracy={accuracy:.3f}", flush=True)
    save_csv("training_runs.csv", runs)
    save_csv("learning_curves.csv", [dict(implementation=name, **entry)
                                   for name, history in histories.items() for entry in history])
    scalar_arrays = final_models["scalar"].arrays()
    torch_arrays = {key: p.detach().numpy() for key, p in final_models["pytorch"].parameter_map().items()}
    parameter_error = max(float(np.max(np.abs(scalar_arrays[k]-torch_arrays[k]))) for k in scalar_arrays)
    curve_error = max(abs(a["loss"]-b["loss"]) for a, b in zip(histories["scalar"], histories["pytorch"]))
    assert parameter_error < 1e-8, parameter_error
    assert curve_error < 1e-8, curve_error
    np.savez(RESULTS / "model_weights.npz", **scalar_arrays)
    medians = {name: statistics.median(r["seconds"] for r in runs if r["implementation"] == name)
               for name in histories}
    summary = dict(n_samples=200, n_train=150, n_test=50, seed=42, noise=0.15,
                   architecture="2 -> 8 ReLU -> 1 sigmoid", parameter_count=33,
                   dtype="float64", device="cpu", epochs=EPOCHS, learning_rate=LR,
                   batch_size=150, repeats=REPEATS, median_seconds=medians,
                   pytorch_speedup=medians["scalar"]/medians["pytorch"],
                   max_final_parameter_difference=parameter_error,
                   max_learning_curve_difference=curve_error,
                   test_accuracy={r["implementation"]: r["test_accuracy"] for r in runs},
                   final_train_loss={r["implementation"]: r["final_train_loss"] for r in runs})
    save_json("training_summary.json", summary)
    return summary, histories, (x_train, y_train, x_test, y_test), final_models["pytorch"]


def dump_autograd_graph(loss):
    root = loss.grad_fn
    nodes, stack, records = {root: 0}, [root], []
    while stack:
        node = stack.pop()
        parents = []
        for parent, slot in node.next_functions:
            if parent is None:
                continue
            if parent not in nodes:
                nodes[parent] = len(nodes)
                stack.append(parent)
            parents.append(dict(node=nodes[parent], output_slot=slot))
        record = dict(id=nodes[node], type=type(node).__name__, next_functions=parents)
        if hasattr(node, "variable"):
            record["leaf_shape"] = list(node.variable.shape)
        records.append(record)
    save_json("pytorch_autograd_graph.json", sorted(records, key=lambda r: r["id"]))


def compile_probe(path):
    """Isolated process: compilation can fail or take too long on this host."""
    torch.set_num_threads(1)
    torch.manual_seed(42)
    model = TorchMLP(initial_arrays()).eval()
    x = torch.randn(256, 2, dtype=torch.float64)
    start = time.perf_counter()
    compiled = torch.compile(model, backend="inductor", fullgraph=True)
    with torch.inference_mode():
        actual = compiled(x)
        torch.testing.assert_close(actual, model(x), rtol=1e-10, atol=1e-10)
    torch.save(compiled.state_dict(), path)
    print(json.dumps(dict(success=True, setup_seconds=time.perf_counter()-start)), flush=True)


def time_calls(call, iterations, warmup=30):
    for _ in range(warmup):
        call()
    start = time.perf_counter()
    for _ in range(iterations):
        call()
    return (time.perf_counter()-start) / iterations


def execution_study():
    torch.manual_seed(42)
    model = TorchMLP(initial_arrays()).eval()
    probe_path = RESULTS / "compile_probe.pt"
    command = [sys.executable, str(Path(__file__).resolve()), "--compile-probe", str(probe_path)]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=60)
        (RESULTS / "compile_probe.log").write_text(result.stdout + result.stderr)
        compiled_supported = result.returncode == 0
        reason = "Inductor probe succeeded" if compiled_supported else "Inductor probe failed; see compile_probe.log"
    except subprocess.TimeoutExpired as exc:
        output = (exc.stdout or b"") + (exc.stderr or b"")
        (RESULTS / "compile_probe.log").write_text(output.decode(errors="replace") + "\nTimed out after 60 seconds.\n")
        compiled_supported, reason = False, "Inductor probe exceeded 60-second budget"
    if probe_path.exists():
        probe_path.unlink()
    example = torch.randn(256, 2, dtype=torch.float64)
    start = time.perf_counter()
    with torch.inference_mode():
        if compiled_supported:
            optimized = torch.compile(model, backend="inductor", fullgraph=True)
            mode = "compile_inductor"
        else:
            optimized = torch.jit.freeze(torch.jit.trace(model, example))
            mode = "jit_trace_frozen"
        optimized(example)  # first execution included in setup, not steady-state
    setup = time.perf_counter()-start
    if not compiled_supported:
        torch.jit.save(optimized, str(RESULTS / "traced_model.pt"))
    def tensor_info(t):
        return dict(shape=list(t.shape), dtype=str(t.dtype), stride=list(t.stride()),
                    contiguous=t.is_contiguous(), requires_grad=t.requires_grad)
    save_json("model_tensor_metadata.json", dict(
        input=tensor_info(example), output=tensor_info(model(example)),
        parameters={name: tensor_info(p) for name, p in model.parameter_map().items()}))
    rows = []
    shape_setup = {}
    with torch.inference_mode():
        for batch in [1, 256, 4096]:
            x = torch.randn(batch, 2, dtype=torch.float64)
            start = time.perf_counter()
            actual = optimized(x)
            shape_setup[batch] = time.perf_counter()-start
            torch.testing.assert_close(actual, model(x), rtol=1e-10, atol=1e-10)
            for repeat in range(REPEATS):
                names = ["eager", mode] if repeat % 2 == 0 else [mode, "eager"]
                for name in names:
                    target = model if name == "eager" else optimized
                    seconds = time_calls(lambda: target(x), iterations=1000)
                    rows.append(dict(batch=batch, repeat=repeat+1, mode=name, seconds_per_call=seconds))
    save_csv("execution_runs.csv", rows)
    summary = dict(mode=mode, reason=reason, setup_seconds=setup, repeats=REPEATS,
                   iterations_per_run=1000, warmup_calls=30, inference_only=True, batches=[])
    for batch in [1, 256, 4096]:
        times = {name: statistics.median(r["seconds_per_call"] for r in rows
                                        if r["batch"] == batch and r["mode"] == name)
                 for name in ["eager", mode]}
        saved = times["eager"]-times[mode]
        summary["batches"].append(dict(batch=batch, median_seconds=times,
                                       speedup=times["eager"]/times[mode],
                                       extra_first_shape_call_seconds=shape_setup[batch],
                                       setup_break_even_calls=(setup+shape_setup[batch])/saved if saved > 0 else None))
    if not compiled_supported:
        (RESULTS / "pytorch_execution_graph.txt").write_text(str(optimized.inlined_graph) + "\n")
    else:
        explanation = torch._dynamo.explain(model)(example)
        graphs = "\n\n".join(str(g.graph) + "\n" + g.code for g in explanation.graphs)
        (RESULTS / "pytorch_execution_graph.txt").write_text(graphs + "\n\n" + str(explanation) + "\n")
    # Autograd graph must be inspected with gradient tracking enabled.
    loss = F.binary_cross_entropy_with_logits(model(example), torch.zeros(256, dtype=torch.float64))
    dump_autograd_graph(loss)
    save_json("execution_summary.json", summary)
    print(f"execution: {mode}; {reason}", flush=True)
    return summary


def layout_study():
    torch.manual_seed(42)
    backing = torch.randn(1024, 2048, dtype=torch.float64)
    view = backing[:, ::2]
    contiguous = view.contiguous()
    torch.testing.assert_close(view, contiguous, rtol=0, atol=0)
    tensors = {"strided_view": view, "contiguous_copy": contiguous}
    metadata = {name: dict(shape=list(t.shape), dtype=str(t.dtype), stride=list(t.stride()),
                           contiguous=t.is_contiguous(), storage_offset=t.storage_offset(),
                           element_size=t.element_size()) for name, t in tensors.items()}
    metadata["strided_view"]["shares_storage_with_backing"] = view.untyped_storage().data_ptr() == backing.untyped_storage().data_ptr()
    # Same values, logical shape, dtype, operation and preallocated output.
    out = torch.empty_like(contiguous)
    rows = []
    with torch.inference_mode():
        for repeat in range(REPEATS):
            names = list(tensors) if repeat % 2 == 0 else list(reversed(tensors))
            for name in names:
                x = tensors[name]
                seconds = time_calls(lambda: torch.add(x, 1, out=out), iterations=200)
                rows.append(dict(repeat=repeat+1, operation="add_preallocated", layout=name,
                                 seconds_per_call=seconds))
            seconds = time_calls(lambda: view.contiguous(), iterations=100)
            rows.append(dict(repeat=repeat+1, operation="materialize_contiguous", layout="strided_view",
                             seconds_per_call=seconds))
    save_csv("layout_runs.csv", rows)
    times = {name: statistics.median(r["seconds_per_call"] for r in rows
                                    if r["operation"] == "add_preallocated" and r["layout"] == name)
             for name in tensors}
    copy = statistics.median(r["seconds_per_call"] for r in rows if r["operation"] == "materialize_contiguous")
    difference = times["strided_view"]-times["contiguous_copy"]
    summary = dict(tensors=metadata, repeats=REPEATS, iterations_per_run=200,
                   warmup_calls=30, operation="torch.add(x, 1, out=out)",
                   median_seconds=times, noncontiguous_slowdown=times["strided_view"]/times["contiguous_copy"],
                   contiguous_materialization_seconds=copy,
                   copy_break_even_operations=copy/difference if difference > 0 else None)
    save_json("layout_summary.json", summary)
    return summary


def figures(histories, split, model):
    os.environ.setdefault("MPLCONFIGDIR", str(RESULTS / ".mplconfig"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, history in histories.items():
        ax.plot([r["epoch"] for r in history], [r["loss"] for r in history],
                label=name, linestyle="--" if name == "pytorch" else "-")
    ax.set(xlabel="Epoch", ylabel="Mean training BCE", title="Identical initialization and full-batch SGD")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS / "learning_curves.png", dpi=150)
    plt.close(fig)
    x_train, y_train, x_test, y_test = split
    grid_x, grid_y = np.meshgrid(np.linspace(-1.6, 2.6, 180), np.linspace(-1.1, 1.6, 140))
    with torch.no_grad():
        values = model.probabilities(torch.tensor(np.c_[grid_x.ravel(), grid_y.ravel()], dtype=torch.float64)).numpy()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.contourf(grid_x, grid_y, values.reshape(grid_x.shape), levels=np.linspace(0, 1, 11), cmap="coolwarm", alpha=0.4)
    ax.contour(grid_x, grid_y, values.reshape(grid_x.shape), levels=[0.5], colors="black", linewidths=1)
    ax.scatter(x_train[:, 0], x_train[:, 1], c=y_train, cmap="coolwarm", s=20, label="train")
    ax.scatter(x_test[:, 0], x_test[:, 1], c=y_test, cmap="coolwarm", marker="s", edgecolors="black", s=35, label="test")
    ax.set(xlabel="x1", ylabel="x2", title="2 → 8 ReLU → 1 sigmoid; held-out test squares")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS / "decision_boundary.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--compile-probe", type=Path)
    parser.add_argument("--only", choices=["gradients", "training", "execution", "layout"],
                        help="Re-run one study without repeating the others")
    args = parser.parse_args()
    if args.compile_probe:
        compile_probe(args.compile_probe)
        return
    RESULTS.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if args.only:
        if args.only == "gradients":
            validate_gradients()
        elif args.only == "training":
            _, histories, split, model = training_study()
            figures(histories, split, model)
        elif args.only == "execution":
            execution_study()
        else:
            layout_study()
        return
    save_json("environment.json", dict(python=sys.version, platform=platform.platform(),
                                      numpy=np.__version__, torch=torch.__version__,
                                      sklearn=__import__("sklearn").__version__,
                                      torch_num_threads=torch.get_num_threads(),
                                      torch_num_interop_threads=torch.get_num_interop_threads(),
                                      device="cpu", timer="time.perf_counter", seed=42))
    print("Checking gradients", flush=True)
    validate_gradients()
    training, histories, split, model = training_study()
    print("Checking compilation and inference timing", flush=True)
    execution_study()
    print("Checking tensor layout timing", flush=True)
    layout_study()
    figures(histories, split, model)
    print("All experiments completed", flush=True)


if __name__ == "__main__":
    main()
