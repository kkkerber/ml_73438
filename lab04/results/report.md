# Lab 04 - Data pipeline and quality

## 1. Problem

See [problem_definition.md](problem_definition.md).

## 2. Ingestion and validation

Loaded 48842 rows and 15 columns (14 inputs plus income). Raw data were not cleaned in ingestion.
The OpenML target `class` is exposed as `income` to match the lab schema.
All declared types, categorical values and four assigned ranges were checked. See validation_report.json for counts and violating row IDs.
Schema/range/category violations: 0. No invalid rows were silently dropped.
Missing values: {'age': 0, 'workclass': 2799, 'fnlwgt': 0, 'education': 0, 'education-num': 0, 'marital-status': 0, 'occupation': 2809, 'relationship': 0, 'race': 0, 'sex': 0, 'capital-gain': 0, 'capital-loss': 0, 'hours-per-week': 0, 'native-country': 857, 'income': 0}. Literal '?': {'age': 0, 'workclass': 0, 'fnlwgt': 0, 'education': 0, 'education-num': 0, 'marital-status': 0, 'occupation': 0, 'relationship': 0, 'race': 0, 'sex': 0, 'capital-gain': 0, 'capital-loss': 0, 'hours-per-week': 0, 'native-country': 0, 'income': 0}.
Exact duplicate excess rows in the full raw dataset: 52.

## 3. Leakage prevention and cleaning

The fixed stratified 70/30 split produced 34189 training and 14653 test rows.
Removed 24 exact training duplicates, including income in the duplicate key; 34165 training rows remain.
The test membership and raw values were preserved. Test rows receive the fitted transformations for prediction, but none are dropped or used to fit preprocessing.
Numeric medians, categorical modes, 1st/99th percentile clipping thresholds, scaling and one-hot categories were fitted on training data only.
Imputation precedes clipping. Derived features are calculated from the cleaned/clipped inputs. Unseen test categories are ignored by the encoder.
Stored fitted parameters: fitted_cleaner.json. Final fitted transformer: preprocessor.joblib.
Exact feature vectors present across train/test: 27. The assignment's row-wise split permits these natural duplicates; this can make performance optimistic. No test-driven grouping or deletion was performed.

## 4. Feature comparison

Added capital_net = capital-gain - capital-loss and hours_per_age = hours-per-week / (age - 16).

| Variant | Accuracy | F1 (>50K) | Encoded features |
|---|---:|---:|---:|
| baseline | 0.853955 | 0.660102 | 105 |
| derived_features | 0.857777 | 0.669626 | 107 |

Derived-minus-baseline differences: accuracy +0.003822; F1 +0.009524.
Both models use the same rows, preprocessing rules and fixed LogisticRegression parameters. F1 treats >50K as positive. No hyperparameters or feature selection were tuned on test results.
The comparison is descriptive; this single split does not establish a statistically reliable improvement.

## 5. Label quality

Found 5 feature groups with conflicting income labels, containing 10 rows.
Conflicts are recorded in label_conflicts.csv and retained; there is no independent evidence to choose a correct label.
Drew 50 raw rows with seed 42; 0 rows failed the automatic dictionary checks.
The sample and notes are in label_review_sample.csv and manual_review.md. Dictionary plausibility does not verify actual income labels.

## 6. Reproducibility

Run `.venv/bin/python lab04/src/main.py` from the repository root. See [README.md](../README.md) for setup and tests.
Processed CSV SHA256: `69e0cbdaede0f4313cb13f28d4bf375346cabb30e384e502df2c4a5b50c30da5`.
The exact representation is documented in config.yaml. The effective configuration and package versions are stored alongside the hash.
See data_card.md for provenance, license, missingness, biases and privacy limitations.
