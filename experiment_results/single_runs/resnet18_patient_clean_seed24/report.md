# SMC-LUD ResNet-18 patient-disjoint baseline

- Model: randomly initialized ResNet-18, two classes
- Data: Clean images, new patient-disjoint split, seed 24
- Selection: minimum validation loss; test evaluated after selection
- Positive class: HCC (ImageFolder index 0), threshold 0.5
- Patient prediction: arithmetic mean of image-level HCC probabilities
- This split differs from the published image allocation; results are not a direct replication of Table 3.
- Hem_87 lacks a patient-table row; the main test retains its folder label and sensitivity excludes it.

## Run

- Epochs completed: 13
- Best epoch: 6
- Actual batch size: 32

## Test results

| Metric | Image level | Patient level |
|---|---:|---:|
| accuracy | 0.9173 | 0.9459 |
| balanced_accuracy | 0.9159 | 0.9487 |
| hcc_precision | 0.9143 | 0.9750 |
| hcc_sensitivity_recall | 0.9343 | 0.9286 |
| hemangioma_specificity | 0.8974 | 0.9688 |
| hcc_f1 | 0.9242 | 0.9512 |
| macro_f1 | 0.9166 | 0.9453 |
| roc_auc_hcc | 0.9738 | 0.9762 |
| brier_hcc | 0.0604 | 0.0482 |

## Hem_87 sensitivity analysis

Excluding 5 Clean images from one test patient:

- Image accuracy: 0.9157
- Patient accuracy: 0.9452

## Patient bootstrap intervals

Patients resampled with replacement; 1000 replicates requested.

- accuracy: [0.8919, 0.9865] (1000 valid resamples)
- hcc_precision: [0.9189, 1.0000] (1000 valid resamples)
- hcc_sensitivity_recall: [0.8409, 1.0000] (1000 valid resamples)
- hemangioma_specificity: [0.8929, 1.0000] (1000 valid resamples)
- hcc_f1: [0.8928, 0.9892] (1000 valid resamples)
- roc_auc_hcc: [0.9394, 1.0000] (1000 valid resamples)

See run_config.json, history.csv, predictions CSVs, metrics JSONs, and figures for details.
