# Lab 05 - Frameworks and automatic differentiation

Student: Valeriia Khapilina, 73438.

From the repository root, using the existing virtual environment:

```bash
.venv/bin/python -m unittest discover -s lab05/src -p 'test_*.py' -v
.venv/bin/python lab05/src/experiments.py
.venv/bin/python lab05/src/report.py
```

To repeat only inference timing, use `experiments.py --only execution` (the same
option accepts `gradients`, `training` or `layout`), then regenerate the report.

Dependencies are already pinned in the repository's `requirements.txt`: NumPy,
scikit-learn, PyTorch and Matplotlib. No extra packages or dataset downloads are
needed. CPU execution uses one PyTorch intra-op and one inter-op thread.

The experiments take several minutes. They overwrite only generated evidence in
`lab05/results/`. Five complete training runs use the same initialization and
split to measure timing variation, not five independent statistical trials.
The scalar engine is intentionally Python-based; PyTorch uses batched tensors.

`torch.compile(..., backend="inductor")` is first tested in an isolated process
with a 60-second limit. If unavailable or over budget, the runner uses frozen
`torch.jit.trace`, as permitted by the assignment. The failure/timeout is recorded,
not silently treated as successful compilation. Warm-up is excluded from timing.
See the generated report for the chosen path and its limitations.

## Files

- `src/autodiff.py`: scalar `Value`, operations and topological reverse pass.
- `src/model.py`: equivalent scalar and PyTorch networks; stable BCE.
- `src/test_autodiff.py`: correctness and edge-case checks.
- `src/experiments.py`: all six assignment tasks and raw measurements.
- `src/report.py`: regenerates `results/report.md` from saved measurements.
- `results/gradients.csv`: 300 derivatives compared with two references.
- `results/training_runs.csv`, `execution_runs.csv`, `layout_runs.csv`: all repeats.
- `results/*_summary.json`: medians, error maxima and experiment configuration.
- `results/pytorch_autograd_graph.json`: actual backward nodes and dependencies.
- `results/pytorch_execution_graph.txt`: traced graph or Dynamo explanation.
- `results/scalar_graph.dot`: scalar graph with values and gradients.
- `results/model_tensor_metadata.json`: model shapes, dtypes, strides, contiguity.
- `results/model_weights.npz`, `moons_split.npz`: trained scalar weights and data.
- `results/traced_model.pt`: frozen inference model if tracing was selected;
  this graph-inspection model uses initial weights, not the trained weights.
- `results/learning_curves.png`, `decision_boundary.png`: experiment figures.

`Value` supports first derivatives of scalar outputs. Constant numeric powers
are supported on their real differentiable domain. ReLU uses derivative zero at
zero. `.sum()` on one scalar is an identity; Python `sum` reduces multiple
`Value` objects. Repeated `backward()` calls accumulate leaf gradients; reset
parameter gradients before each optimization step. Do not mutate node data
between forward and backward. This educational engine does not implement
tensor broadcasting, higher-order gradients, devices or PyTorch's mutation checks.

For predictions from saved trained weights:

```python
import sys
import numpy as np
import torch
sys.path.insert(0, 'lab05/src')
from model import TorchMLP
with np.load('lab05/results/model_weights.npz') as weights:
    model = TorchMLP(dict(weights)).eval()
with torch.no_grad():
    probabilities = model.probabilities(torch.tensor([[0.0, 0.5]], dtype=torch.float64))
print(probabilities)
```
