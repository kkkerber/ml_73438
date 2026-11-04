"""Load the requested OpenML version; do not clean or remove raw rows."""
from sklearn.datasets import fetch_openml


def load_raw_data(name="adult", version=2, data_home=None):
    adult = fetch_openml(name=name, version=version, as_frame=True,
                         data_home=data_home, parser="auto")
    raw_data = adult.data.copy()
    raw_data["income"] = adult.target
    return raw_data
