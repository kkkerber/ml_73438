"""Deterministic derived features; no information is learned from test data."""
from sklearn.base import BaseEstimator, TransformerMixin


class DerivedFeatures(TransformerMixin, BaseEstimator):
    def __init__(self, enabled=True):
        self.enabled = enabled

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        result = X.copy()
        if self.enabled:
            denominator = result["age"] - 16
            if (denominator <= 0).any():
                raise ValueError("hours_per_age requires age > 16")
            result["capital_net"] = result["capital-gain"] - result["capital-loss"]
            result["hours_per_age"] = result["hours-per-week"] / denominator
        return result
