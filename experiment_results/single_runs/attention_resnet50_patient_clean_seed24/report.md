# SMC-LUD attention-enhanced ResNet-50, patient-disjoint Clean data

- Model: repository model.py CustomModel, ImageNet-pretrained ResNet-50 backbone, dual attention branches, spatial attention, two classes
- The ordinary ResNet-50 baseline was randomly initialized, so differences are not a pure attention ablation.
- Data: Clean images, new patient-disjoint split, seed 24
- Selection: minimum validation loss; test evaluated after selection
- Positive class: HCC (ImageFolder index 0), threshold 0.5
- Patient prediction: arithmetic mean of image-level HCC probabilities
- This split differs from the published image allocation; results are not a direct replication of Table 3.
- Hem_87 lacks a patient-table row; the main test retains its folder label and sensitivity excludes it.

## Run

- Epochs completed: 23
- Best epoch: 16
- Actual batch size: 32

## Test results

| Metric | Image level | Patient level |
|---|---:|---:|
| accuracy | 0.9843 | 0.9865 |
| balanced_accuracy | 0.9829 | 0.9844 |
| hcc_precision | 0.9716 | 0.9767 |
| hcc_sensitivity_recall | 1.0000 | 1.0000 |
| hemangioma_specificity | 0.9658 | 0.9688 |
| hcc_f1 | 0.9856 | 0.9882 |
| macro_f1 | 0.9841 | 0.9862 |
| roc_auc_hcc | 0.9996 | 1.0000 |
| brier_hcc | 0.0138 | 0.0118 |

## Hem_87 sensitivity analysis

Excluding 5 Clean images from one test patient:

- Image accuracy: 0.9839
- Patient accuracy: 0.9863

## Patient bootstrap intervals

Patients resampled with replacement; 1000 replicates requested.

- accuracy: [0.9595, 1.0000] (1000 valid resamples)
- hcc_precision: [0.9231, 1.0000] (1000 valid resamples)
- hcc_sensitivity_recall: [1.0000, 1.0000] (1000 valid resamples)
- hemangioma_specificity: [0.9000, 1.0000] (1000 valid resamples)
- hcc_f1: [0.9600, 1.0000] (1000 valid resamples)
- roc_auc_hcc: [1.0000, 1.0000] (1000 valid resamples)

See run_config.json, history.csv, predictions CSVs, metrics JSONs, and figures for details.
