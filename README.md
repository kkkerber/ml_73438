# Lab 01 — ML Systems Baseline

## 1. Goal

The goal of this laboratory work was to compare two machine learning models not only by their prediction accuracy, but also by their system performance.

I used Logistic Regression and Random Forest to investigate how training time, inference latency, memory usage, and model size affect the choice of a model for different deployment environments.

## 2. Methodology

I used the Breast Cancer Wisconsin dataset from scikit-learn. The dataset contains 569 samples and 30 features.

The dataset was split into 70% training data (398 samples) and 30% testing data (171 samples). I used `random_state=42` and stratified splitting to preserve the class distribution.

Two models were trained:

- Logistic Regression with `max_iter=1000` and `random_state=42`.
- Random Forest with `n_estimators=100` and `random_state=42`.

StandardScaler was fitted only on the training data and used to transform the training and testing data for Logistic Regression. Random Forest was trained on the original features.

I evaluated both models using accuracy on the test set. Training time was measured using five runs after one warm-up, and the median was reported. Inference latency was measured on one sample using 100 repetitions.

Model sizes were measured using saved joblib files. Memory usage was estimated using the peak resident set size (RSS) of the Python process during training and inference.

All experiments were performed locally.

## 3. Results

| Metric | Logistic Regression | Random Forest |
|---|---:|---:|
| Accuracy | 0.9883 | 0.9357 |
| Training time (ms) | 1.1847 | 94.8463 |
| Inference latency (ms) | 0.0266 | 1.4566 |
| Model size (KB) | 2.17 | 284.09 |
| Training peak RSS (MB) | 130.91 | 130.97 |
| Inference peak RSS (MB) | 131.08 | 131.09 |

Logistic Regression achieved higher accuracy while requiring less training time and storage space.

Random Forest was substantially slower and produced a larger model file.

The memory measurements were very similar because RSS includes the entire Python process, not only the model. The sampling method also provides an approximate peak rather than an exact measurement.

## 4. Deployment Budget Analysis

The laboratory defines four deployment environments: Cloud, Edge, Mobile, and TinyML.

| Environment | Logistic Regression | Random Forest |
|---|---|---|
| Cloud | Fits measured budgets | Fits measured budgets |
| Edge | Fits measured budgets | Fits measured budgets |
| Mobile | Fits measured budgets | Fits measured budgets |
| TinyML | Not confirmed | Does not fit model-size budget |

Both models satisfy the Cloud, Edge, and Mobile budgets based on the measured latency, model size, and process memory.

For TinyML, Logistic Regression satisfies the model-size and latency limits, but its memory requirements on an actual microcontroller have not been verified. Random Forest exceeds the TinyML model-size limit of 100 KB.

## 5. Conclusions

1. **Accuracy is not the only important metric.** Logistic Regression achieved 98.83% accuracy, while Random Forest achieved 93.57%. In this experiment, the simpler model performed better.

2. **Model complexity affects system performance.** Random Forest required much more time for training and inference, and its saved model was significantly larger. This makes Logistic Regression more efficient for this dataset.

3. **Deployment requirements influence model selection.** Both models are suitable for the measured Cloud, Edge, and Mobile budgets, but TinyML has much stricter limitations. Logistic Regression is the more promising candidate, although deployment on actual TinyML hardware would require further testing.

## 6. Reproducibility

The experiment used `random_state=42` for the dataset split and model configuration.

Package versions are recorded in `results/versions.txt`.

Detailed measurements are saved in `results/system_measurements.csv`, and model accuracy is saved in `results/baseline_accuracy.csv`.

The trained models are stored as joblib files in the results directory.