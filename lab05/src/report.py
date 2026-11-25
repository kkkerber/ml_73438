"""Generate the lab report from actual, completed experiment outputs."""

import json
from pathlib import Path


RESULTS = Path(__file__).resolve().parents[1] / "results"


def read(name):
    return json.loads((RESULTS / name).read_text())


def main():
    gradients = read("gradient_summary.json")
    training = read("training_summary.json")
    execution = read("execution_summary.json")
    layout = read("layout_summary.json")
    environment = read("environment.json")
    graph = read("pytorch_autograd_graph.json")
    execution_rows = []
    for row in execution["batches"]:
        times = row["median_seconds"]
        break_even = row["setup_break_even_calls"]
        execution_rows.append(
            f'| {row["batch"]} | {times["eager"]*1e6:.3f} | '
            f'{times[execution["mode"]]*1e6:.3f} | {row["speedup"]:.3f} | '
            f'{break_even:.0f} |' if break_even is not None else
            f'| {row["batch"]} | {times["eager"]*1e6:.3f} | '
            f'{times[execution["mode"]]*1e6:.3f} | {row["speedup"]:.3f} | never |')
    layout_rows = []
    for name, meta in layout["tensors"].items():
        layout_rows.append(f'| {name} | {meta["shape"]} | {meta["dtype"]} | '
                           f'{meta["stride"]} | {meta["contiguous"]} | '
                           f'{layout["median_seconds"][name]*1e6:.3f} |')
    copy_break_even = layout["copy_break_even_operations"]
    copy_note = (f'about {copy_break_even:.1f} repeated additions' if copy_break_even is not None
                 else 'no measured break-even point for this operation')
    modes_note = (
        "The Inductor probe succeeded. The main-process setup can reuse compiler caches populated by "
        "the probe; it is not a cold installation/compiler-cache benchmark. First calls for each "
        "batch shape are also recorded separately."
        if execution["mode"] == "compile_inductor" else
        "The fallback is TorchScript tracing followed by freezing, not an Inductor compilation result. "
        "The network has no data-dependent Python branches, so tracing is appropriate. Outputs were "
        "checked against eager at every measured batch size (absolute and relative tolerance 1e-10)."
    )
    faster_batches = [str(r["batch"]) for r in execution["batches"] if r["speedup"] > 1]
    measured_conclusion = (
        "The optimized path was faster at batch sizes " + ", ".join(faster_batches) + "."
        if faster_batches else
        "The optimized path was slower at every measured batch size, even after warm-up. "
        "For this tiny CPU float64 network, eager execution is the better measured choice; "
        "setup cannot be amortized because no steady-state time was saved."
    )
    report = f"""# Lab 05 - Frameworks and automatic differentiation

Student: Valeriia Khapilina, 73438.

## 1. Scalar graph engine

Implemented `Value` in `../src/autodiff.py`. Every node stores a floating-point
value, gradient, ordered parent references, operation label and local backward
rule. Operators construct new nodes during the forward pass. Parent references
retain repeated operands: `x*x` has two contributions even though `x` is visited
only once in the graph traversal.

The engine implements reverse-mode automatic differentiation. It applies local
derivatives to the operations actually executed, rather than symbolically
rewriting a formula or estimating derivatives by perturbing inputs. This follows
the graph-based mechanism described in [PyTorch autograd mechanics](https://docs.pytorch.org/docs/main/notes/autograd.html).

## 2. Operations and local derivatives

For upstream gradient g = dL/dz:

| Operation | Contribution to input gradients |
|---|---|
| z = a + b | a.grad += g; b.grad += g |
| z = a * b | a.grad += b*g; b.grad += a*g |
| z = a**p, constant p | a.grad += p*a**(p-1)*g |
| z = ReLU(a) | a.grad += (a > 0)*g |

ReLU's chosen derivative at zero is 0. Extra operations include subtraction,
division, exp, log, stable sigmoid and softplus. Their gradients are implemented
in the scalar engine; NumPy and PyTorch do not calculate its derivatives.

## 3. Topological backward pass

An iterative depth-first traversal produces a parent-before-child order without
Python recursion depth limits. The output gradient is set to 1, and nodes run
their local backward rules in reverse topological order. Gradients use `+=`
because shared nodes receive contributions along multiple paths. Internal
gradients are reset on a repeated backward call; leaf gradients accumulate.
Parameter gradients are explicitly zeroed before every training step.

Manual example: a=2, b=3, c=1; u=ab=6, v=u+c=7, L=ReLU(v²)=49.
The chain rule gives dL/dv=14, dL/du=14, dL/da=42, dL/db=28,
and dL/dc=14. `scalar_graph.dot` records these computed values and gradients.

Shared-node example: u=x*x, L=u*u+u+x. Then dL/dx=4x³+2x+1=115
at x=3. The test catches both repeated-operand and shared-branch errors.

## 4. Gradient verification

Checked `((a*b+c)**2).relu().sum()` on {gradients['cases']} seeded cases.
Each a, b and c contains four independent elements: the scalar engine reduces
four scalar graphs with Python `sum`; PyTorch uses its tensor `.sum()`.
For a single `Value`, `.sum()` is an identity. The ReLU after the square is
redundant for values; separate tests cover negative, zero and positive ReLU inputs.

| Check | Result |
|---|---:|
| Individual derivatives checked | {gradients['derivatives']} |
| Max absolute difference from PyTorch | {gradients['max_error_pytorch']:.3e} |
| Max absolute difference from numerical estimates | {gradients['max_error_numeric']:.3e} |
| Required tolerance | < 1e-6 |

Both criteria passed. Numerical derivatives use central differences,
`[f(x+h)-f(x-h)]/(2h)`, with h={gradients['epsilon']}, float64, perturbing one
coordinate at a time. Full comparisons are in `gradients.csv`.

Unit tests additionally check graph order, a 3,000-operation chain, repeated
backward calls, reverse arithmetic, constant zero powers and stable BCE at logits
±1000. A complete model's loss, every parameter gradient and one SGD update are
checked against PyTorch at absolute tolerance 1e-12.

## 5. Tiny model on make_moons

Generated {training['n_samples']} points with noise={training['noise']} and seed=42.
A fixed stratified split contains {training['n_train']} training and
{training['n_test']} test points. No test data are used for gradient updates or
parameter selection. Architecture: {training['architecture']}, with
{training['parameter_count']} trainable parameters.

For hidden features h=ReLU(W1*x+b1), the output logit is z=W2*h+b2 and
the predicted probability is sigmoid(z). The loss is mean binary cross-entropy.
Training computes it as `softplus(z)-y*z`, mathematically equal to
`-y*log(sigmoid(z))-(1-y)*log(1-sigmoid(z))`, with better numerical stability.
The scalar engine uses a stable custom softplus derivative; PyTorch uses
`binary_cross_entropy_with_logits`. Sigmoid is applied for reported predictions.

Both implementations use identical initial arrays, float64, CPU, full batches of
150, {training['epochs']} epochs and SGD with learning rate {training['learning_rate']}.
Updates use the same explicit SGD formula without momentum or regularization.
No hyperparameter search was performed.

| Implementation | Median training time, s | Final training BCE | Held-out accuracy |
|---|---:|---:|---:|
| Scalar engine | {training['median_seconds']['scalar']:.6f} | {training['final_train_loss']['scalar']:.6f} | {training['test_accuracy']['scalar']:.2%} |
| PyTorch | {training['median_seconds']['pytorch']:.6f} | {training['final_train_loss']['pytorch']:.6f} | {training['test_accuracy']['pytorch']:.2%} |

PyTorch was {training['pytorch_speedup']:.1f} times faster on this workload.
Maximum difference between final corresponding parameters:
{training['max_final_parameter_difference']:.3e}. Maximum difference between
the two training loss curves: {training['max_learning_curve_difference']:.3e}.
Both must remain below 1e-8; the experiment asserts this before saving its summary.

Times are medians of five complete training runs from reset initial weights.
The timed region includes forward, backward, zeroing, updates and loss-history
collection; it excludes model construction, data conversion, warm-up, evaluation
and file writing. Scalar garbage collection stays enabled, so Python graph
allocation/collection overhead is included. Run order alternates. The comparison
includes vectorization and Python overhead; it does not isolate autograd alone.
Five identical-seed runs quantify timing variation, not model accuracy uncertainty.

![Training loss curves](learning_curves.png)

![Learned decision boundary](decision_boundary.png)

## 6. Framework graphs, execution modes and tensor layout

### Graph inspection

`pytorch_autograd_graph.json` records {len(graph)} actual backward nodes and their
`next_functions` edges. Node types include
{', '.join(sorted(set(n['type'] for n in graph)))}.
The autograd graph represents derivative dependencies; the execution graph in
`pytorch_execution_graph.txt` represents forward tensor operations. They are
related but serve different purposes. `model_tensor_metadata.json` records
model input/output and parameter shape, dtype, strides and contiguity.

### Eager versus optimized execution

Selected path: `{execution['mode']}`. Reason: {execution['reason']}.
The original probe output is preserved in `compile_probe.log`.
{modes_note}

The [PyTorch 2.3 tracing documentation](https://docs.pytorch.org/docs/2.3/generated/torch.jit.trace.html)
explains that tracing records the tensor operations executed on example inputs;
it does not capture arbitrary data-dependent Python control flow.

Setup in the main process took {execution['setup_seconds']:.6f} s, including first
execution. Below are medians of five runs, each 1,000 forward calls after 30 warm-up
calls, CPU, float64, inference mode, one thread. Model weights are fixed initial
weights for this independent performance study. The timed output is the logit;
the common output sigmoid is excluded from both paths. Input allocation, setup,
warm-up and correctness checks are excluded. Run order alternates.

| Batch | Eager, µs/call | Optimized, µs/call | Eager/optimized | Approx. setup break-even calls |
|---|---:|---:|---:|---:|
{chr(10).join(execution_rows)}

{measured_conclusion}

Break-even divides measured setup plus the first call for that shape by steady-state
time saved per call. It excludes the failed/successful probe process and, if used,
any compiler-cache cold start. It is a local estimate, not a deployment guarantee.

Graph optimization can help when repeatedly running the same supported workload:
less Python dispatch and opportunities to combine operations can reduce execution
cost. Initial setup must be amortized. Very short runs, unsupported operations,
shape specialization or memory-bound workloads may reduce or remove the benefit.
These measurements cover forward inference only; they do not establish training
speedups or make a general claim about all compilation backends.

### Non-contiguous tensors

Allocated a float64 backing tensor of shape (1024, 2048). `backing[:, ::2]`
is a non-contiguous view of shape (1024, 1024), sharing its backing storage.
Its contiguous copy contains exactly the same values, verified with zero tolerance.
Shape describes logical dimensions; dtype describes each element; strides give
steps in memory in units of elements. The contiguous layout has stride (1024, 1),
while the strided view skips every second stored element.

| Layout | Shape | dtype | Strides | Contiguous | Median add, µs |
|---|---|---|---|---|---:|
{chr(10).join(layout_rows)}

Measured operation: `torch.add(x, 1, out=out)` with a shared preallocated contiguous
output. Five runs, 200 operations per run, 30 warm-up calls. Allocations are excluded
from the addition timing. Measured non-contiguous/contiguous ratio:
{layout['noncontiguous_slowdown']:.3f}. A separate contiguous materialization takes
{layout['contiguous_materialization_seconds']*1e6:.3f} µs; amortization would require
{copy_note}, assuming comparable conditions.

The view touches a wider address range and has different cache behavior; this is
part of the layout comparison. Results are specific to this elementwise operation,
size and host. Non-contiguous views do not always slow every PyTorch operation,
and making a copy has its own time and memory cost. This is a warmed-cache CPU
microbenchmark, not a measurement of DRAM bandwidth or GPU kernels.

## Reproducibility and limits

Environment: {environment['platform']}; Python {environment['python'].split()[0]};
PyTorch {environment['torch']}; NumPy {environment['numpy']};
scikit-learn {environment['sklearn']}. CPU, one intra-op and one inter-op thread.
Timing uses `time.perf_counter`; no accelerator synchronization is needed.

Commands and artifact descriptions: [../README.md](../README.md).
Raw repeats remain in CSV, summaries in JSON, weights and split in NPZ. The report
is generated from those files by `../src/report.py`, without invented measurements.
There is one host and one data split; timing noise, cache effects and thermal state
limit generalization. The engine supports first-order scalar differentiation only.
"""
    (RESULTS / "report.md").write_text(report)
    print(RESULTS / "report.md")


if __name__ == "__main__":
    main()
