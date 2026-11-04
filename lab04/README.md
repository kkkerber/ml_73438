# Lab 04 - Data pipeline and quality

## Setup and run

From the repository root, using the existing virtual environment:

```bash
.venv/bin/python -m pip install -r lab04/requirements.txt
.venv/bin/python lab04/src/main.py
```

The second command executes the entire experiment. Paths are resolved relative to the script, so it also works from PyCharm. Initial loading needs internet. A pre-existing `~/scikit_learn_data` cache is reused; otherwise the cache is stored under repository `data/openml` (ignored by Git). No command is run automatically to install dependencies.

## Project structure

- `src/main.py`: orchestrates the full run and checks raw/test data remain unchanged.
- `src/ingestion.py`: loads OpenML Adult version 2.
- `src/schema.py`: declared types, ranges and categories, including `?` as a missing marker.
- `src/validation.py`: read-only quality checks with machine-readable reports.
- `src/preprocessing.py`: stratified split, training deduplication, imputation, clipping, scaling and encoding.
- `src/features.py`: the two deterministic derived features.
- `src/evaluation.py`: the two fixed LogisticRegression experiments.
- `src/label_quality.py`: conflicting-label audit and seeded 50-row sample.
- `src/reporting.py`: hashes, run configuration, fitted parameters and report.
- `tests/`: behavior checks for leakage and serialization.
- `config.yaml`: reproducible experiment settings.
- `problem_definition.md`, `data_card.md`, `report.md`: written evidence.
- `results/`: generated reports, models, CSVs and logs.

## Order of operations

Load raw data -> validate -> split -> remove exact train duplicates -> fit imputers on train -> impute -> learn train clipping thresholds -> clip -> add derived features (second variant only) -> fit train scaling/encoding -> fit LogisticRegression -> evaluate unchanged test membership.

Test receives only transforms, never fitting or row deletion. No parameters are chosen from the model comparison. Full-dataset validation and label auditing are descriptive and do not supply fitted values to the models. To avoid data-driven decisions, the features/formulas and dictionary rules are fixed in advance by the assignment.

`results/preprocessor.joblib` is the fitted transformer for the variant with derived features. The baseline transformer and both models are saved separately. Transform raw 14-column feature DataFrames in the original column order. Custom transformer classes live in `src`, so include that directory on `sys.path` when loading from another script.

## Validation and tests

```bash
.venv/bin/python -m pytest lab04/tests -q
```

The validation report includes all required types, ranges, NaN/`?` counts, allowed categories and duplicate counts. Structural violations stop the run after the report is saved. Expected missing feature values and duplicates are recorded and processed as specified, rather than treated as fatal schema errors.

The label sample CSV includes a column for reviewer notes. The automatic dictionary scan is distinct from the documented manual/assistant review; neither can independently verify a person's actual income. No 24-hour relabeling delay or Cohen's kappa is required for this Adult experiment.

The hash covers the CSV bytes, including source IDs, split and income, in the recorded row/column order. To verify it independently:

```bash
shasum -a 256 lab04/results/processed_dataset.csv
```

Run the pipeline twice with the same configuration and package versions to compare `processed_dataset.sha256` and `metrics.csv`. Model execution times are intentionally not included in reproducibility artifacts.
