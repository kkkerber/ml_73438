"""One pre-specified comparison on a fixed split, without test-set tuning."""
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from preprocessing import build_preprocessor


def compare_models(X_train, X_test, y_train, y_test, config, results):
    rows = []
    processed = None
    for with_features in (False, True):
        name = "derived_features" if with_features else "baseline"
        preprocessor = build_preprocessor(with_features=with_features,
                                           **config["preprocessing"])
        train = preprocessor.fit_transform(X_train)
        test = preprocessor.transform(X_test)
        model = LogisticRegression(**config["model"])
        model.fit(train, y_train)
        if model.n_iter_.max() >= config["model"]["max_iter"]:
            raise RuntimeError(f"{name}: LogisticRegression did not converge")
        prediction = model.predict(test)
        rows.append({"variant": name, "train_rows": len(X_train), "test_rows": len(X_test),
                     "input_features": X_train.shape[1] + 2 * int(with_features),
                     "encoded_features": train.shape[1],
                     "accuracy": accuracy_score(y_test, prediction),
                     "f1": f1_score(y_test, prediction, pos_label=">50K"),
                     "iterations": int(model.n_iter_.max())})
        pd.DataFrame({"row_id": X_test.index, "income": y_test.to_numpy(),
                      "prediction": prediction}).to_csv(results / f"{name}_predictions.csv", index=False)
        joblib.dump(preprocessor, results / ("preprocessor.joblib" if with_features else "baseline_preprocessor.joblib"))
        joblib.dump(model, results / f"{name}_model.joblib")
        if with_features:
            # Hash/save the readable cleaned dataset before one-hot encoding/scaling.
            clean_features = preprocessor[:2]
            parts = []
            for split, X, y in [("train", X_train, y_train), ("test", X_test, y_test)]:
                part = clean_features.transform(X)
                part.insert(0, "row_id", X.index)
                part.insert(1, "split", split)
                part["income"] = y
                parts.append(part)
            processed = pd.concat(parts, ignore_index=True)
    metrics = pd.DataFrame(rows)
    metrics.to_csv(results / "metrics.csv", index=False, float_format="%.12g")
    return metrics, processed
