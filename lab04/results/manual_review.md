# Dictionary review of 50 sampled raw records

Reviewer: Codex assistant (AI-assisted record-by-record inspection, not a human sign-off).
Review date: 2026-11-04. Sample: pandas `sample(n=50, random_state=42)` on the raw OpenML Adult version 2 DataFrame, before splitting or cleaning. Source indices are retained as `row_id` in `label_review_sample.csv`.

Reference: [UCI Adult variable dictionary](https://archive.ics.uci.edu/dataset/2/adult) and the four numeric ranges in the lab. The CSV's `automatic_dictionary_issues` column is an independent automatic scan, not evidence of human review. I also inspected the 50 displayed raw records individually, including numeric bounds, category spellings, missingness, and education/education-num plausibility. No independent census source records were available, so this cannot establish that a recorded income label is true.

## Findings

No impossible values were identified in the sampled records under the stated rules. All observed income labels are one of `<=50K` or `>50K`. Ages are 18-67 and working hours are 4-84 per week; both are within the assigned bounds. All weights are positive and exceed the UCI extraction cutoff. Education values and codes in the sample are internally consistent: 5th-6th/3, 10th/6, 11th/7, HS-grad/9, Some-college/10, Assoc-voc/11, Bachelors/13, Masters/14, Doctorate/16. Missing values are not classified as impossible values and are retained.

| Raw row ID | Inspection note |
|---:|---|
| 7762 | No dictionary/range inconsistency identified. |
| 23881 | Married with relationship Own-child is unusual but not forbidden; no correction inferred. |
| 30507 | Capital gain 14344 is within the specified range. |
| 28911 | No dictionary/range inconsistency identified. |
| 19484 | No dictionary/range inconsistency identified. |
| 43031 | Germany is a valid native-country category. |
| 28188 | No dictionary/range inconsistency identified. |
| 12761 | Capital loss 1887 is within the specified range. |
| 40834 | Dominican-Republic is a valid category; 50 hours is in range. |
| 27875 | No dictionary/range inconsistency identified. |
| 1276 | 70 hours is high but allowed. |
| 22608 | No dictionary/range inconsistency identified. |
| 36230 | Germany is a valid category. |
| 13398 | Education 11th with code 7 is consistent. |
| 43536 | No dictionary/range inconsistency identified. |
| 18627 | Assoc-voc with code 11 is consistent; 60 hours is allowed. |
| 38424 | No dictionary/range inconsistency identified. |
| 35505 | 5th-6th with code 3 is consistent; Mexico is valid. |
| 2372 | Age 66 and work are allowed; no dictionary contradiction. |
| 3375 | 4 hours per week is low but inside [1, 99]. |
| 28331 | Capital loss 1408 is allowed. |
| 46820 | Doctorate/code 16 and capital loss 3004 are allowed. |
| 147 | No dictionary/range inconsistency identified. |
| 21063 | workclass and occupation missing; retained for later mode imputation. Gain 9386 is allowed. |
| 20139 | native-country missing; retained for later mode imputation. |
| 23288 | Married-spouse-absent and Own-child are valid categories; do not infer an error from the combination. |
| 18957 | workclass and occupation missing; retained. Age 62/24 hours are allowed. |
| 31256 | Haiti is a valid native-country category. |
| 29642 | No dictionary/range inconsistency identified. |
| 23638 | 10th with code 6 is consistent. |
| 18067 | Masters with code 14 is consistent. |
| 879 | No dictionary/range inconsistency identified. |
| 7155 | Own-child at age 40 is not prohibited by the dictionary. |
| 43769 | 5th-6th/code 3 is consistent; 75 hours is allowed. |
| 17697 | 52 hours is allowed. |
| 21167 | No dictionary/range inconsistency identified. |
| 3136 | 84 hours is high but below the permitted maximum of 99. |
| 34254 | No dictionary/range inconsistency identified. |
| 17657 | 55 hours is allowed. |
| 37254 | No dictionary/range inconsistency identified. |
| 1422 | No dictionary/range inconsistency identified. |
| 32507 | Age 18, 11th/code 7, 15 hours are plausible and allowed. |
| 30150 | No dictionary/range inconsistency identified. |
| 14172 | Own-child at age 31 is not prohibited. |
| 32273 | Race Other is a declared category. |
| 48262 | No dictionary/range inconsistency identified. |
| 10007 | 10th/code 6 is consistent. |
| 47649 | Ecuador and Other-relative are valid; income cannot be independently verified. |
| 43058 | Assoc-voc/code 11 and Widowed are valid. |
| 708 | No dictionary/range inconsistency identified. |

## Scope

This review detects visible dictionary/range problems. It does not justify relabeling records based on stereotypes or expected income. The separate full-dataset conflict audit found 5 identical-feature groups with opposing income labels (10 records); all are preserved and listed in `label_conflicts.csv`. No 24-hour relabeling or kappa is used for Adult. If a human review is required by the instructor, the student must inspect the CSV and add their own notes; this assistant review does not impersonate that review.

Sample CSV SHA256 at review: `912b73e3b1890b33896439777d2a410e7abd40781a1a74eb37eefe6c47717cf3`.
