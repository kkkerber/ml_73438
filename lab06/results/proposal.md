# Individual Project Proposal

**Student:** Valeriia Khapilina | **Student ID:** 73438  
**Title:** Compact SMS Spam Detection for Local CPU Inference  
**Status:** Topic selected by student; instructor approval pending.

## Problem and intended use

Build a local classifier that labels an English SMS as spam or legitimate (ham). The aim is to compare a simple baseline with a compact neural classifier and measure whether INT8 quantization reduces storage while preserving classification quality. The prototype will accept text through a command-line interface; automatic blocking is outside the project scope.

## Dataset and evaluation design

Use the UCI SMS Spam Collection (approximately 5.6k labelled messages, CC BY 4.0). Inspect the downloaded row count, label balance and missing values. Remove exact duplicates and group normalized duplicates before making a fixed, stratified 70/15/15 train/validation/test split (seed 42), so repeated messages cannot leak across splits. Fit the vocabulary and TF-IDF statistics on training data only; select model settings and the decision threshold on validation data. Keep the test set untouched until final comparison.

## Models and planned optimization

Start with TF-IDF and logistic regression. Train a PyTorch MLP using at most 4,000 TF-IDF features, one 64-unit ReLU hidden layer and two output logits. This gives approximately 256k parameters, or 1.03 MB of raw FP32 weights before serialization. Apply post-training dynamic INT8 quantization to the Linear layers using a supported CPU backend. Compare the same trained FP32 and INT8 model with identical preprocessing. If the local PyTorch backend is unsuitable, evaluate ONNX Runtime dynamic quantization and document the change.

## Success metrics and deployment constraints

Targets: spam-class F1 >= 0.90 and precision >= 0.95 on held-out data, with an absolute F1 reduction of no more than 0.01 after quantization. Report recall, macro-F1 and a confusion matrix as well. These are goals to test, not measured results. Require p95 latency <= 50 ms for batch size 1 on the local laptop CPU, including text preprocessing and classification, and total deployable model plus vocabulary/IDF/config files <= 5,000,000 bytes (runtime excluded). Record CPU, thread count and software versions; warm up with 100 requests, then measure at least 1,000 serial requests. Report median and p95 latency, artifact bytes and FP32-to-INT8 size ratio.

## Work plan, risks and evidence

1. Audit data and freeze the split; establish the baseline.
2. Train the MLP with YAML configuration, seeded runs and MLflow tracking.
3. Quantize, benchmark, and report the accuracy-size-latency trade-off.

Deliver code, configuration, data audit, MLflow runs, FP32/INT8 artifacts, benchmark samples and a short report. Risks include class imbalance, duplicate leakage and quantization accuracy loss. Use class-aware metrics and validation-based tuning. This historical English SMS dataset does not establish performance on modern scams or other languages.

**Dataset:** Almeida & Hidalgo (2011), UCI SMS Spam Collection, https://doi.org/10.24432/C5CC84
