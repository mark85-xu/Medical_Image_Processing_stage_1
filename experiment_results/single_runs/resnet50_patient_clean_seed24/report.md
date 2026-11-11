# SMC-LUD ResNet-50 patient-disjoint baseline

- Model: randomly initialized ResNet-50, two classes
- Data: Clean images, new patient-disjoint split, seed 24
- Selection: minimum validation loss; test evaluated after selection
- Positive class: HCC (ImageFolder index 0), threshold 0.5
- Patient prediction: arithmetic mean of image-level HCC probabilities
- This split differs from the published image allocation; results are not a direct replication of Table 3.
- Hem_87 lacks a patient-table row; the main test retains its folder label and sensitivity excludes it.

## Run

- Epochs completed: 41
- Best epoch: 34
- Actual batch size: 32

## Test results

| Metric | Image level | Patient level |
|---|---:|---:|
| accuracy | 0.9606 | 0.9730 |
| balanced_accuracy | 0.9591 | 0.9725 |
| hcc_precision | 0.9504 | 0.9762 |
| hcc_sensitivity_recall | 0.9781 | 0.9762 |
| hemangioma_specificity | 0.9402 | 0.9688 |
| hcc_f1 | 0.9640 | 0.9762 |
| macro_f1 | 0.9603 | 0.9725 |
| roc_auc_hcc | 0.9893 | 0.9985 |
| brier_hcc | 0.0329 | 0.0200 |

## Hem_87 sensitivity analysis

Excluding 5 Clean images from one test patient:

- Image accuracy: 0.9598
- Patient accuracy: 0.9726

## Patient bootstrap intervals

Patients resampled with replacement; 1000 replicates requested.

- accuracy: [0.9324, 1.0000] (1000 valid resamples)
- hcc_precision: [0.9230, 1.0000] (1000 valid resamples)
- hcc_sensitivity_recall: [0.9189, 1.0000] (1000 valid resamples)
- hemangioma_specificity: [0.8929, 1.0000] (1000 valid resamples)
- hcc_f1: [0.9367, 1.0000] (1000 valid resamples)
- roc_auc_hcc: [0.9930, 1.0000] (1000 valid resamples)

See run_config.json, history.csv, predictions CSVs, metrics JSONs, and figures for details.
