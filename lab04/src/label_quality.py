"""Conflicting-label audit and a reproducible 50-row dictionary review sample."""
import pandas as pd
from schema import ALLOWED_CATEGORIES, RANGES


def dictionary_issues(row):
    issues = []
    for column, (lower, upper) in RANGES.items():
        value = row[column]
        if pd.notna(value) and not lower <= value <= upper:
            issues.append(f"{column} outside [{lower}, {upper}]")
    for column, allowed in ALLOWED_CATEGORIES.items():
        value = row[column]
        if pd.notna(value) and value not in allowed:
            issues.append(f"unknown {column}: {value}")
    if pd.notna(row["fnlwgt"]) and row["fnlwgt"] <= 1:
        issues.append("fnlwgt <= 1 contradicts UCI extraction criteria")
    if pd.notna(row["education-num"]) and not 1 <= row["education-num"] <= 16:
        issues.append("education-num outside [1, 16]")
    return "; ".join(issues)


def audit_labels(data, results, sample_size=50, random_state=42):
    features = data.drop(columns="income").columns.tolist()
    # Convert category columns to object to include missing-value groups reliably.
    audit = data.astype(object)
    groups = audit.groupby(features, dropna=False, sort=False, observed=True)["income"]
    conflicting = groups.transform("nunique") > 1
    conflict_rows = data.loc[conflicting].copy()
    conflict_rows.to_csv(results / "label_conflicts.csv", index=True, index_label="row_id")
    conflict_groups = len(conflict_rows[features].drop_duplicates())
    sample = data.sample(n=sample_size, random_state=random_state).copy()
    sample["automatic_dictionary_issues"] = sample.apply(dictionary_issues, axis=1)
    sample["reviewer_notes"] = ""
    sample.to_csv(results / "label_review_sample.csv", index=True, index_label="row_id")
    return {
        "conflicting_feature_groups": conflict_groups,
        "rows_in_conflicting_groups": int(conflicting.sum()),
        "conflicting_row_ids": data.index[conflicting].tolist(),
        "sample_size": sample_size, "sample_seed": random_state,
        "sample_rows_with_automatic_issues": int(sample["automatic_dictionary_issues"].ne("").sum()),
        "review_sample": "label_review_sample.csv",
        "manual_review": "See manual_review.md; automatic checks do not establish label truth.",
    }
