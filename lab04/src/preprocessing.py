"""Split before cleaning; fit all learned preprocessing on training rows only."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted

from features import DerivedFeatures


def split_data(data, test_size=0.3, random_state=42):
    X = data.drop(columns=["income"])
    y = data["income"]
    return train_test_split(X, y, test_size=test_size,
                            random_state=random_state, stratify=y)


def drop_train_duplicates(X_train, y_train):
    # Include the label: conflicting labels must not be silently deduplicated.
    duplicate = X_train.assign(income=y_train).duplicated()
    return X_train.loc[~duplicate].copy(), y_train.loc[~duplicate].copy(), int(duplicate.sum())


class TrainOnlyCleaner(TransformerMixin, BaseEstimator):
    """Impute, then clip to train percentiles while preserving names and index."""
    def __init__(self, lower_quantile=0.01, upper_quantile=0.99,
                 numeric_imputation="median", categorical_imputation="most_frequent"):
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile
        self.numeric_imputation = numeric_imputation
        self.categorical_imputation = categorical_imputation

    def _prepare(self, X):
        result = X.copy()
        # Object conversion allows replacing '?' even in pandas category columns.
        for column in self.categorical_columns_:
            result[column] = result[column].astype(object).replace("?", np.nan)
        return result

    def fit(self, X, y=None):
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        self.numeric_columns_ = X.select_dtypes(include="number").columns.tolist()
        self.categorical_columns_ = [c for c in X if c not in self.numeric_columns_]
        prepared = self._prepare(X)
        self.numeric_imputer_ = SimpleImputer(strategy=self.numeric_imputation, keep_empty_features=True)
        self.categorical_imputer_ = SimpleImputer(strategy=self.categorical_imputation, keep_empty_features=True)
        numeric = self.numeric_imputer_.fit_transform(prepared[self.numeric_columns_])
        self.categorical_imputer_.fit(prepared[self.categorical_columns_])
        numeric = pd.DataFrame(numeric, columns=self.numeric_columns_, index=X.index)
        self.lower_bounds_ = numeric.quantile(self.lower_quantile)
        self.upper_bounds_ = numeric.quantile(self.upper_quantile)
        return self

    def transform(self, X):
        check_is_fitted(self, "lower_bounds_")
        if list(X.columns) != list(self.feature_names_in_):
            raise ValueError("Input columns/order differ from fitted training schema")
        prepared = self._prepare(X)
        numeric = pd.DataFrame(self.numeric_imputer_.transform(prepared[self.numeric_columns_]),
                               columns=self.numeric_columns_, index=X.index)
        numeric = numeric.clip(lower=self.lower_bounds_, upper=self.upper_bounds_, axis=1)
        categorical = pd.DataFrame(
            self.categorical_imputer_.transform(prepared[self.categorical_columns_]),
            columns=self.categorical_columns_, index=X.index)
        return pd.concat([numeric, categorical], axis=1).loc[:, self.feature_names_in_]


def build_preprocessor(X_train=None, with_features=False, **cleaner_settings):
    # Selectors are evaluated on fit, after derived features have been added.
    encoder = ColumnTransformer([
        ("numeric", StandardScaler(), make_column_selector(dtype_include=np.number)),
        ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=True),
         make_column_selector(dtype_include=object)),
    ])
    return Pipeline([
        ("clean", TrainOnlyCleaner(**cleaner_settings)),
        ("features", DerivedFeatures(enabled=with_features)),
        ("encode", encoder),
    ])
