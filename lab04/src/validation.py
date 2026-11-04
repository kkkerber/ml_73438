"""Read-only validation: report problems without silently removing rows."""
from schema import ALLOWED_CATEGORIES, EXPECTED_TYPES, RANGES


def validate_ranges(data):
    report = {}
    for column, (lower, upper) in RANGES.items():
        if column not in data:
            continue
        values = data[column]
        # Non-numeric types are reported by validate_types instead of crashing here.
        import pandas as pd
        values = pd.to_numeric(values, errors="coerce")
        invalid = (values < lower) | (values > upper)
        report[column] = {"lower": lower, "upper": upper,
                          "violations": int(invalid.sum()),
                          "row_ids": data.index[invalid].tolist()}
    return report


def validate_missing_data(data):
    return {
        "policy": "NaN and literal '?' mean missing; '?' is allowed in categorical fields. Missing labels stop training.",
        "nan_counts": {c: int(n) for c, n in data.isna().sum().items()},
        "question_mark_counts": {c: int(n) for c, n in data.eq("?").sum().items()},
    }


def validate_duplicates(data):
    return {"exact_duplicate_excess_rows": int(data.duplicated().sum()),
            "rows_in_duplicate_groups": int(data.duplicated(keep=False).sum())}


def validate_types(data):
    return {
        column: {"expected": expected,
                 "actual": str(data[column].dtype) if column in data else None,
                 "matches": column in data and str(data[column].dtype) == expected}
        for column, expected in EXPECTED_TYPES.items()
    }


def validate_categories(data):
    report = {}
    for column, allowed in ALLOWED_CATEGORIES.items():
        if column not in data:
            continue
        invalid = data[column].notna() & ~data[column].isin(allowed)
        report[column] = {"violations": int(invalid.sum()),
                          "unexpected_values": sorted(data.loc[invalid, column].astype(str).unique().tolist())}
    return report


def validate_data(data):
    return {
        "rows": len(data), "columns": len(data.columns),
        "missing_columns": sorted(set(EXPECTED_TYPES) - set(data.columns)),
        "unexpected_columns": sorted(set(data.columns) - set(EXPECTED_TYPES)),
        "types": validate_types(data), "ranges": validate_ranges(data),
        "categories": validate_categories(data),
        "missing": validate_missing_data(data), "duplicates": validate_duplicates(data),
    }


def has_schema_violations(report):
    return bool(report["missing_columns"] or report["unexpected_columns"]
                or any(not v["matches"] for v in report["types"].values())
                or any(v["violations"] for v in report["ranges"].values())
                or any(v["violations"] for v in report["categories"].values()))
