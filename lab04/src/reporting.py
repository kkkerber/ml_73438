"""Stable artifacts, hashes, fitted parameters and an evidence-based report."""
import hashlib
import json
from importlib.metadata import version
import platform
import joblib
import yaml


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def persist_run(results, config, validation, labels, split, metrics, processed):
    processed_path = results / "processed_dataset.csv"
    processed.to_csv(processed_path, index=False, float_format="%.17g", lineterminator="\n")
    digest = hashlib.sha256(processed_path.read_bytes()).hexdigest()
    (results / "processed_dataset.sha256").write_text(f"{digest}  processed_dataset.csv\n")
    manifest = {
        "config": config,
        "processed_dataset": {"file": processed_path.name, "sha256": digest,
                              "rows": len(processed), "columns": len(processed.columns),
                              "representation": "Train then test; cleaned and clipped, with two derived features, before scaling/encoding; row_id, split and income included."},
        "split": split,
        "environment": {"python": platform.python_version(), "packages": {
            name: version(name) for name in ["numpy", "pandas", "scikit-learn", "scipy", "joblib", "PyYAML"]}},
    }
    (results / "config.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    cleaner = joblib.load(results / "preprocessor.joblib").named_steps["clean"]
    write_json(results / "fitted_cleaner.json", {
        "numeric_imputation": dict(zip(cleaner.numeric_columns_, cleaner.numeric_imputer_.statistics_.tolist())),
        "categorical_imputation": dict(zip(cleaner.categorical_columns_, cleaner.categorical_imputer_.statistics_.tolist())),
        "lower_clip_bounds": cleaner.lower_bounds_.to_dict(),
        "upper_clip_bounds": cleaner.upper_bounds_.to_dict(),
    })
    write_json(results / "validation_report.json", validation)
    write_json(results / "label_quality_report.json", labels)
    baseline, derived = metrics.iloc[0], metrics.iloc[1]
    lines = [
        "# Lab 04 - Data pipeline and quality", "",
        "## 1. Problem", "", "See [problem_definition.md](problem_definition.md).", "",
        "## 2. Ingestion and validation", "",
        f"Loaded {validation['rows']} rows and {validation['columns']} columns (14 inputs plus income). Raw data were not cleaned in ingestion.",
        "The OpenML target `class` is exposed as `income` to match the lab schema.",
        "All declared types, categorical values and four assigned ranges were checked. See validation_report.json for counts and violating row IDs.",
        f"Schema/range/category violations: {validation['violation_count']}. No invalid rows were silently dropped.",
        f"Missing values: {validation['missing']['nan_counts']}. Literal '?': {validation['missing']['question_mark_counts']}.",
        f"Exact duplicate excess rows in the full raw dataset: {validation['duplicates']['exact_duplicate_excess_rows']}.", "",
        "## 3. Leakage prevention and cleaning", "",
        f"The fixed stratified 70/30 split produced {split['train_before_dedup']} training and {split['test_rows']} test rows.",
        f"Removed {split['train_duplicates_removed']} exact training duplicates, including income in the duplicate key; {split['train_after_dedup']} training rows remain.",
        "The test membership and raw values were preserved. Test rows receive the fitted transformations for prediction, but none are dropped or used to fit preprocessing.",
        "Numeric medians, categorical modes, 1st/99th percentile clipping thresholds, scaling and one-hot categories were fitted on training data only.",
        "Imputation precedes clipping. Derived features are calculated from the cleaned/clipped inputs. Unseen test categories are ignored by the encoder.",
        "Stored fitted parameters: fitted_cleaner.json. Final fitted transformer: preprocessor.joblib.",
        f"Exact feature vectors present across train/test: {split['cross_split_feature_overlap_rows']}. The assignment's row-wise split permits these natural duplicates; this can make performance optimistic. No test-driven grouping or deletion was performed.", "",
        "## 4. Feature comparison", "",
        "Added capital_net = capital-gain - capital-loss and hours_per_age = hours-per-week / (age - 16).", "",
        "| Variant | Accuracy | F1 (>50K) | Encoded features |",
        "|---|---:|---:|---:|",
    ]
    for row in metrics.to_dict("records"):
        lines.append(f"| {row['variant']} | {row['accuracy']:.6f} | {row['f1']:.6f} | {row['encoded_features']} |")
    lines += ["", f"Derived-minus-baseline differences: accuracy {derived['accuracy'] - baseline['accuracy']:+.6f}; F1 {derived['f1'] - baseline['f1']:+.6f}.",
              "Both models use the same rows, preprocessing rules and fixed LogisticRegression parameters. F1 treats >50K as positive. No hyperparameters or feature selection were tuned on test results.",
              "The comparison is descriptive; this single split does not establish a statistically reliable improvement.", "",
              "## 5. Label quality", "",
              f"Found {labels['conflicting_feature_groups']} feature groups with conflicting income labels, containing {labels['rows_in_conflicting_groups']} rows.",
              "Conflicts are recorded in label_conflicts.csv and retained; there is no independent evidence to choose a correct label.",
              f"Drew {labels['sample_size']} raw rows with seed {labels['sample_seed']}; {labels['sample_rows_with_automatic_issues']} rows failed the automatic dictionary checks.",
              "The sample and notes are in label_review_sample.csv and manual_review.md. Dictionary plausibility does not verify actual income labels.", "",
              "## 6. Reproducibility", "",
              "Run `.venv/bin/python lab04/src/main.py` from the repository root. See [README.md](../README.md) for setup and tests.",
              f"Processed CSV SHA256: `{digest}`.",
              "The exact representation is documented in config.yaml. The effective configuration and package versions are stored alongside the hash.",
              "See data_card.md for provenance, license, missingness, biases and privacy limitations.", ""]
    (results / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return digest
