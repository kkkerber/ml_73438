
import csv
import os
import time
import threading
from pathlib import Path
from statistics import median

import joblib
import numpy as np
import psutil

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier


# 1. Reproducibility
np.random.seed(42)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# 2. Load dataset
X, y = load_breast_cancer(return_X_y=True)

print("Dataset:", X.shape, y.shape)


# 3. Split data: 70% train, 30% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

print("Train:", X_train.shape, y_train.shape)
print("Test:", X_test.shape, y_test.shape)


# 4. Scale data for Logistic Regression
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# 5. Create models
lr_model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)


# 6. Measure process memory
process = psutil.Process(os.getpid())


def measure_peak_memory(operation):
    """
    Measure approximate peak RSS while an operation runs.
    Returns the peak memory in MB.
    """
    peak_rss = process.memory_info().rss
    stop_event = threading.Event()

    def sample_memory():
        nonlocal peak_rss

        while not stop_event.wait(0.001):
            rss = process.memory_info().rss
            peak_rss = max(peak_rss, rss)

    thread = threading.Thread(
        target=sample_memory,
        daemon=True
    )

    thread.start()

    try:
        operation()
    finally:
        peak_rss = max(
            peak_rss,
            process.memory_info().rss
        )
        stop_event.set()
        thread.join()

    return peak_rss / (1024 ** 2)


# 7. Training time: 1 warm-up + 5 runs
def measure_training(model, X_data, y_data):
    model.fit(X_data, y_data)  # warm-up

    training_times = []

    for i in range(5):
        start = time.perf_counter()
        model.fit(X_data, y_data)
        end = time.perf_counter()

        training_times.append(end - start)

    return median(training_times) * 1000


lr_training_time = measure_training(
    lr_model, X_train_scaled, y_train
)

rf_training_time = measure_training(
    rf_model, X_train, y_train
)


# 8. Accuracy
lr_pred = lr_model.predict(X_test_scaled)
rf_pred = rf_model.predict(X_test)

lr_accuracy = accuracy_score(y_test, lr_pred)
rf_accuracy = accuracy_score(y_test, rf_pred)

print(f"Logistic Regression accuracy: {lr_accuracy:.4f}")
print(f"Random Forest accuracy: {rf_accuracy:.4f}")


# 9. Inference latency: 100 repetitions
def measure_inference(model, sample):
    model.predict(sample)  # warm-up

    inference_times = []

    for i in range(100):
        start = time.perf_counter()
        model.predict(sample)
        end = time.perf_counter()

        inference_times.append(end - start)

    return median(inference_times) * 1000


lr_sample = X_test_scaled[:1]
rf_sample = X_test[:1]

lr_latency = measure_inference(lr_model, lr_sample)
rf_latency = measure_inference(rf_model, rf_sample)


# 10. Peak memory during training
lr_train_memory = measure_peak_memory(
    lambda: lr_model.fit(X_train_scaled, y_train)
)

rf_train_memory = measure_peak_memory(
    lambda: rf_model.fit(X_train, y_train)
)


# 11. Peak memory during inference
lr_infer_memory = measure_peak_memory(
    lambda: [
        lr_model.predict(lr_sample)
        for _ in range(100)
    ]
)

rf_infer_memory = measure_peak_memory(
    lambda: [
        rf_model.predict(rf_sample)
        for _ in range(100)
    ]
)


# 12. Save models
lr_path = RESULTS_DIR / "model.joblib"
rf_path = RESULTS_DIR / "random_forest_model.joblib"

# Include scaler so LR can process raw input later
joblib.dump(
    {"scaler": scaler, "model": lr_model},
    lr_path
)

joblib.dump(rf_model, rf_path)

lr_size_bytes = os.path.getsize(lr_path)
rf_size_bytes = os.path.getsize(rf_path)

lr_size_kb = lr_size_bytes / 1024
rf_size_kb = rf_size_bytes / 1024


# 13. Save accuracy to CSV
with open(
    RESULTS_DIR / "baseline_accuracy.csv",
    "w",
    newline=""
) as file:
    writer = csv.writer(file)

    writer.writerow(["model", "accuracy"])
    writer.writerow([
        "Logistic Regression",
        f"{lr_accuracy:.4f}"
    ])
    writer.writerow([
        "Random Forest",
        f"{rf_accuracy:.4f}"
    ])


# 14. Save all measurements to CSV
with open(
    RESULTS_DIR / "system_measurements.csv",
    "w",
    newline=""
) as file:
    writer = csv.writer(file)

    writer.writerow([
        "model",
        "accuracy",
        "training_ms",
        "inference_ms",
        "model_size_bytes",
        "model_size_kb",
        "training_peak_memory_mb",
        "inference_peak_memory_mb"
    ])

    writer.writerow([
        "Logistic Regression",
        f"{lr_accuracy:.4f}",
        f"{lr_training_time:.6f}",
        f"{lr_latency:.6f}",
        lr_size_bytes,
        f"{lr_size_kb:.2f}",
        f"{lr_train_memory:.2f}",
        f"{lr_infer_memory:.2f}"
    ])

    writer.writerow([
        "Random Forest",
        f"{rf_accuracy:.4f}",
        f"{rf_training_time:.6f}",
        f"{rf_latency:.6f}",
        rf_size_bytes,
        f"{rf_size_kb:.2f}",
        f"{rf_train_memory:.2f}",
        f"{rf_infer_memory:.2f}"
    ])


# 15. Print results
print("\n--- FINAL RESULTS ---")

print("\nLogistic Regression:")
print(f"Accuracy: {lr_accuracy:.4f}")
print(f"Training: {lr_training_time:.4f} ms")
print(f"Inference: {lr_latency:.4f} ms")
print(f"Model size: {lr_size_kb:.2f} KB")
print(f"Training peak memory: {lr_train_memory:.2f} MB")
print(f"Inference peak memory: {lr_infer_memory:.2f} MB")

print("\nRandom Forest:")
print(f"Accuracy: {rf_accuracy:.4f}")
print(f"Training: {rf_training_time:.4f} ms")
print(f"Inference: {rf_latency:.4f} ms")
print(f"Model size: {rf_size_kb:.2f} KB")
print(f"Training peak memory: {rf_train_memory:.2f} MB")
print(f"Inference peak memory: {rf_infer_memory:.2f} MB")

print("\nResults saved to:", RESULTS_DIR)
