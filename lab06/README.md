# Lab 6 - Training pipelines and project proposal

Student: Valeriia Khapilina, 73438.

## Run all evidence

From the repository root, using the existing Python 3.11 environment:

```bash
.venv/bin/python -m pip install -r lab06/requirements.txt
.venv/bin/python lab06/src/studies.py
.venv/bin/python lab06/src/report.py
.venv/bin/python -m unittest discover -s lab06/src -p 'test_*.py' -v
```

The runner executes studies serially in fresh processes and skips completed runs with `summary.json`. To repeat a completed case, move its corresponding `lab06/results/runs/NAME` directory out of the results folder first, then use `--only NAME`. A new local MLflow run will be created. Do not compare throughput while running other heavy jobs. The data already exist in shared `data/FashionMNIST`; training intentionally uses `download=False`. For a fresh checkout, download the official data once:

```bash
.venv/bin/python -c "from torchvision.datasets import FashionMNIST; FashionMNIST('data', train=True, download=True); FashionMNIST('data', train=False, download=True)"
```

Each run receives a single resolved YAML config in `configs/`. `config.yaml` is the default base. CPU is explicit: this host has no usable CUDA/MPS backend. Running the studies requires normal local process/shared-memory access for OpenMP and DataLoader workers. A restricted sandbox can prevent this; no remote tracking server or credentials are required.

## Single run and recovery

```bash
.venv/bin/python lab06/src/train.py --config lab06/config.yaml --name custom
.venv/bin/python lab06/src/train.py --config lab06/config.yaml --name recovery_demo --stop-after 3
.venv/bin/python lab06/src/train.py --config lab06/config.yaml --name recovery_demo --resume lab06/results/runs/recovery_demo/last.pt
```

The total epoch budget stays 5 for both recovery calls, so scheduler state and epoch count are consistent. Recovery is at epoch boundaries and exact recovery is verified for num_workers=0. Persistent worker-process RNG/iterator state is not restored; do not assume exact continuation for worker>0 or add random transforms without extending checkpoint handling. Only load trusted local checkpoints: PyTorch checkpoints include pickled RNG state.

## MLflow

```bash
.venv/bin/mlflow ui --backend-store-uri "file:$(pwd)/lab06/results/mlruns" --host 127.0.0.1 --port 5000
```

Open http://127.0.0.1:5000 and select experiment `lab06`. Params, per-epoch metrics, final test results and artifact copies are stored locally under `results/mlruns`. Run IDs are in `results/study_summary.csv`. File-store artifact URIs refer to this checkout's absolute location. After moving or cloning the project to another location, stop the MLflow UI/server and run `.venv/bin/python lab06/src/relocate_mlflow.py` before opening old artifacts or training new runs. The command updates only local experiment/run artifact-location metadata; `--dry-run` previews the changes. CSV/JSON/checkpoints remain usable independently of MLflow.

## Deliverables

- `results/report.md`: all eight tasks, measured tables, figures, and limitations.
- `results/profile.csv`, `workers.csv`, `batches.csv`, `optimizers_schedulers.csv`, `accumulation.csv`.
- `results/runs/*`: resolved YAML, environment, histories, final checkpoint, summary; profiler operator breakdown in `runs/profile/`.
- `results/recovery_comparison.json` and `accumulation_comparison.json`: numeric checks.
- `results/mlruns/`: real local MLflow runs and their logged artifacts.
- `results/proposal.pdf`: one-page proposal; `proposal.md`: editable source. Student approved the plan in chat; instructor approval remains pending.
- `results/preparation.md`: concept notes with reading links.

`src/render_proposal.py` requires `reportlab` and `pypdf` (available in the Codex bundled artifact Python). It generates the PDF from the Markdown source and asserts exactly one A4 page. The generated PDF was rendered and visually checked. These PDF-authoring packages are optional for training.

The lab reuses Lab 3's exact best architecture and trains from scratch because no trained Lab 3 weights were saved. Study decisions use validation accuracy. Test results for each fixed recipe are reported transparently, not used to tune hyperparameters. Measurements are from a single seed and host; sampled CPU RSS is not an exact GPU memory peak. See the report for timing denominators and exclusions.
