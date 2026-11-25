# Machine Learning Systems - laboratory work

All labs use the same layout:

```text
labXX/
├── src/       # Python code and tests
└── results/   # Reports, manual calculations, models, tables and figures
```

README files, dependency lists and input configuration can remain in each lab's root. Shared downloaded datasets are stored in repository `data/` and are ignored by Git.

| Lab | Code | Report | Manual calculations |
|---|---|---|---|
| 01 - ML Systems Baseline | [src](lab01/src/) | [Report](lab01/results/report.md) | Included in report where applicable |
| 02 - Neural network from scratch | [src](lab02/src/) | [Report](lab02/results/report.md) | [Derivation](lab02/results/derivation.md) |
| 03 - Architectures and costs | [src](lab03/src/) | [Report](lab03/results/report.md) | [Handwritten pages](lab03/results/hand_calculations/) |
| 04 - Data pipeline and quality | [src](lab04/src/) | [Report](lab04/results/report.md) | [50-row dictionary review](lab04/results/manual_review.md) |
| 06 - Training pipelines | [src](lab06/src/) | [Report](lab06/results/report.md) | [Project proposal](lab06/results/proposal.pdf) |

Run from the repository root with the existing virtual environment:

```bash
.venv/bin/python lab01/src/baseline.py
.venv/bin/python lab02/src/perceptron.py
.venv/bin/python lab03/src/main.py
.venv/bin/python lab04/src/main.py
```

Lab 04 setup and artifact descriptions are in [lab04/README.md](lab04/README.md).

Run the existing checks:

```bash
.venv/bin/python -m unittest discover -s lab02/src -p 'test_*.py' -v
.venv/bin/python -m pytest lab04/src/tests -q
```

Lab 06 configuration, experiment commands and MLflow UI instructions are in [lab06/README.md](lab06/README.md).
