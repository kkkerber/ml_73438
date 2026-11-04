# Data card - Adult

## Source and attribution

Dataset: Adult / Census Income, Barry Becker and Ronny Kohavi (1996), UCI Machine Learning Repository, DOI [10.24432/C5XW20](https://doi.org/10.24432/C5XW20). Retrieved through OpenML `adult`, version 2. The [UCI dataset page and variable dictionary](https://archive.ics.uci.edu/dataset/2/adult) provide the source description and category vocabulary. UCI identifies the underlying records as an extraction from the 1994 Census database.

## License

UCI lists [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/). Retain attribution to the dataset creators and identify transformations when sharing derived data. This pipeline imputes missing values, clips numeric inputs, removes exact training duplicates and adds two derived features. It preserves the raw test rows and their labels.

## Size and fields

The requested OpenML version contains 48,842 rows, 14 input features (six numeric and eight categorical) and a binary income label. The positive label is `>50K`; the other label is `<=50K`. OpenML calls the target `class`; the lab calls it `income`. Feature names and allowed categorical values are defined in `../src/schema.py`.

The fixed stratified split allocates 34,189 rows to train and 14,653 to test before training deduplication. Actual retained counts and class frequencies are in `config.yaml`; metrics are in `metrics.csv`.

## Missingness and quality

OpenML represents missing source markers as NaN. The policy also recognizes literal `?` as missing and allows it in categorical validation. Missing counts in this version are 2,799 for workclass, 2,809 for occupation and 857 for native-country. Other fields have no missing values. The full raw dataset contains 52 excess exact duplicates. No records are discarded during validation.

Schema, category and lab-specified range checks are in `validation_report.json`. Training-only imputation uses numeric medians and categorical modes; training-only 1st/99th percentile thresholds clip numeric features. The final transformer scales numeric inputs and one-hot encodes categorical inputs. Derived features are calculated after clipping. Exact training duplicates include the label in their key. Feature duplicates with conflicting labels are audited separately and retained.

## Known biases and limitations

Historical sampling and the relationship between demographics and income can encode social inequalities. Frequencies differ across demographic groups, and overall accuracy/F1 do not establish equal performance across groups. Sex and race are included for fidelity to the assignment; this report does not provide a fairness evaluation. Other features may also act as proxies. Mode filling can further concentrate common categories. Clipping removes extreme numeric magnitudes, including potentially informative capital gains. The dataset includes `fnlwgt`, but this experiment treats it as an input rather than a sample weight.

The fixed row-wise random split can place identical feature records in both train and test; the overlap is disclosed in the run manifest/report. Test duplicates are retained as the assignment specifies, so this limitation may make measured performance optimistic. Conflicting income labels cannot be resolved using plausibility checks. Two feature variants on one split provide a descriptive comparison, not a guarantee of generalization. No model selection or hyperparameter tuning uses the test results.

## Privacy and intended use

Records include income and demographic attributes. Even without direct names, combinations of attributes may reveal sensitive information or enable linkage with other sources. Do not attempt re-identification or attach additional personal records. Use here is limited to an educational data-pipeline experiment. No claim is made that the model is suitable for consequential decisions or for current populations.

## Reproducibility and storage

`../config.yaml` defines the experiment. `config.yaml` saves its effective values, package versions, row counts and SHA256 of the exact `processed_dataset.csv` bytes. That CSV contains train then test, source row IDs, split names, cleaned/clipped values, derived features and income, before scaling/one-hot encoding. The transformer is `preprocessor.joblib`; only load trusted joblib files. Initial OpenML retrieval needs internet; subsequent runs use its local cache. `label_review_sample.csv` stores the seeded 50-row review sample; review scope is documented in `manual_review.md`.
