"""Run Lab 04 from any working directory: python /path/to/lab04/src/main.py."""
import argparse
import logging
from pathlib import Path
import sys

import pandas as pd
import yaml

from ingestion import load_raw_data
from validation import validate_data, has_schema_violations
from preprocessing import split_data, drop_train_duplicates
from label_quality import audit_labels
from evaluation import compare_models
from reporting import persist_run, write_json

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    results = ROOT / "results"
    results.mkdir(exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=[
        logging.StreamHandler(sys.stdout), logging.FileHandler(results / "run.log", mode="w")], force=True)
    # Read an existing sklearn cache without changing it; put first-time downloads inside the repo.
    legacy_cache = Path.home() / "scikit_learn_data"
    cache = legacy_cache if legacy_cache.exists() else ROOT.parent / "data" / "openml"
    raw = load_raw_data(config["dataset"]["name"], config["dataset"]["version"], data_home=cache)
    raw_snapshot = raw.copy(deep=True)
    logging.info("Ingestion: %s rows, %s columns", *raw.shape)
    validation = validate_data(raw)
    validation["violation_count"] = (
        len(validation["missing_columns"]) + len(validation["unexpected_columns"])
        + sum(not item["matches"] for item in validation["types"].values())
        + sum(item["violations"] for item in validation["ranges"].values())
        + sum(item["violations"] for item in validation["categories"].values()))
    write_json(results / "validation_report.json", validation)
    expected = config["dataset"]
    if raw.shape != (expected["expected_rows"], expected["expected_columns"]):
        raise ValueError("Unexpected dataset size; see validation_report.json")
    if has_schema_violations(validation):
        raise ValueError("Schema violations detected; see validation_report.json. No rows were dropped.")
    if raw["income"].isna().any() or raw["income"].eq("?").any():
        raise ValueError("Missing income labels; refusing to silently remove rows")
    logging.info("Validation: %s violations; %s duplicate excess rows",
                 validation["violation_count"], validation["duplicates"]["exact_duplicate_excess_rows"])
    X_train, X_test, y_train, y_test = split_data(raw, **config["split"])
    X_test_snapshot, y_test_snapshot = X_test.copy(deep=True), y_test.copy(deep=True)
    train_before = len(X_train)
    X_train, y_train, removed = drop_train_duplicates(X_train, y_train)
    # Record natural feature overlap without using it to change the assigned split.
    unique_train = X_train.astype(object).drop_duplicates()
    overlap = X_test.astype(object).merge(unique_train, how="inner", on=list(X_train.columns))
    split = {"train_before_dedup": train_before, "train_duplicates_removed": removed,
             "train_after_dedup": len(X_train), "test_rows": len(X_test),
             "cross_split_feature_overlap_rows": len(overlap),
             "train_class_counts": {str(k): int(v) for k, v in y_train.value_counts().items()},
             "test_class_counts": {str(k): int(v) for k, v in y_test.value_counts().items()}}
    pd.DataFrame({"row_id": X_train.index, "split": "train"}).to_csv(results / "train_indices.csv", index=False)
    pd.DataFrame({"row_id": X_test.index, "split": "test"}).to_csv(results / "test_indices.csv", index=False)
    logging.info("Split: train %s -> %s after removing %s duplicates; test %s",
                 train_before, len(X_train), removed, len(X_test))
    labels = audit_labels(raw, results, **config["label_quality"])
    logging.info("Label audit: %s conflicting feature groups; %s rows",
                 labels["conflicting_feature_groups"], labels["rows_in_conflicting_groups"])
    metrics, processed = compare_models(X_train, X_test, y_train, y_test, config, results)
    pd.testing.assert_frame_equal(raw, raw_snapshot)
    pd.testing.assert_frame_equal(X_test, X_test_snapshot)
    pd.testing.assert_series_equal(y_test, y_test_snapshot)
    digest = persist_run(results, config, validation, labels, split, metrics, processed)
    logging.info("%s", metrics.to_string(index=False))
    logging.info("Processed SHA256: %s", digest)
    logging.info("Results: %s", results)


if __name__ == "__main__":
    main()
