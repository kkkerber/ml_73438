"""Behavior checks for leakage, missing markers, duplicates and feature formulas."""
import joblib
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from features import DerivedFeatures
from label_quality import audit_labels
from preprocessing import TrainOnlyCleaner, drop_train_duplicates, build_preprocessor
from validation import validate_data, has_schema_violations
from schema import EXPECTED_TYPES


def frame():
    return pd.DataFrame({
        "age": [20, 30, 40, 50], "fnlwgt": [100, 200, 300, 400],
        "education-num": [9, 10, 11, 12], "capital-gain": [0, 10, 20, 30],
        "capital-loss": [0, 1, 2, 3], "hours-per-week": [20, 30, 40, 50],
        "workclass": pd.Categorical(["Private", None, "?", "Local-gov"]),
    })


def test_imputation_clipping_and_no_test_fit():
    train = frame()
    cleaner = TrainOnlyCleaner().fit(train)
    before = cleaner.lower_bounds_.copy(), cleaner.upper_bounds_.copy()
    test = train.iloc[[0, 1]].copy()
    test.loc[0, "age"] = 10000
    test.loc[1, "age"] = np.nan
    snapshot = test.copy(deep=True)
    transformed = cleaner.transform(test)
    assert transformed.loc[0, "age"] == before[1]["age"]
    assert transformed.loc[1, "age"] == 35  # train median, not test median
    assert transformed.loc[1, "workclass"] == "Local-gov"  # train mode tie-break
    pd.testing.assert_series_equal(cleaner.lower_bounds_, before[0])
    pd.testing.assert_series_equal(cleaner.upper_bounds_, before[1])
    assert_frame_equal(test, snapshot)
    assert not cleaner.transform(train).isna().any().any()
    assert not cleaner.transform(train).eq("?").any().any()


def test_training_duplicates_include_labels_and_do_not_mutate():
    X = pd.DataFrame({"age": [20, 20, 20, 30]})
    y = pd.Series(["<=50K", "<=50K", ">50K", ">50K"], name="income")
    snapshot = X.copy()
    cleaned, labels, removed = drop_train_duplicates(X, y)
    assert removed == 1
    assert cleaned.index.tolist() == [0, 2, 3]
    assert labels.index.equals(cleaned.index)
    assert_frame_equal(X, snapshot)


def test_features_follow_formulas_without_mutation():
    X = frame()
    snapshot = X.copy()
    result = DerivedFeatures().fit_transform(X)
    np.testing.assert_allclose(result["capital_net"], [0, 9, 18, 27])
    np.testing.assert_allclose(result["hours_per_age"], X["hours-per-week"] / (X["age"] - 16))
    assert_frame_equal(X, snapshot)


def test_saved_transformer_handles_unseen_categories(tmp_path):
    X = frame()
    preprocessor = build_preprocessor(with_features=True).fit(X)
    test = X.iloc[[0]].copy()
    test["workclass"] = "Unseen-category"
    expected = preprocessor.transform(test)
    path = tmp_path / "preprocessor.joblib"
    joblib.dump(preprocessor, path)
    actual = joblib.load(path).transform(test)
    np.testing.assert_allclose(actual, expected)
    assert np.isfinite(actual).all()


def test_conflict_audit_includes_missing_feature_values(tmp_path):
    data = pd.DataFrame({c: [20, 20, 30] if t == "int64" else ["Private"] * 3
                         for c, t in EXPECTED_TYPES.items()})
    data["workclass"] = pd.Categorical([None, None, "Private"])
    data["income"] = pd.Categorical(["<=50K", ">50K", ">50K"])
    report = audit_labels(data, tmp_path, sample_size=3)
    assert report["conflicting_feature_groups"] == 1
    assert report["rows_in_conflicting_groups"] == 2


def test_validation_reports_missing_column_without_crash():
    report = validate_data(pd.DataFrame({"age": [16, 20, 91]}))
    assert has_schema_violations(report)
    assert "income" in report["missing_columns"]
    assert report["ranges"]["age"]["violations"] == 2
